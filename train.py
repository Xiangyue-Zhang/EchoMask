import os
import sys
import time
import csv
import signal
import warnings
import random
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt

from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from timm.utils import ModelEma


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = torch.device(f"cuda:{args.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/"

        if self.rank == 0:
            if self.args.stat == "ts":
                self.writer = SummaryWriter(log_dir=args.out_path + "custom/" + args.name + args.notes + "/")
            else:
                # 如需离线可设置环境变量 WANDB_MODE=offline
                wandb.init(project=args.project, dir=args.out_path, name=args.name[12:] + args.notes)
                wandb.config.update(args)
                self.writer = None

        if args.train_rvq:
            self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")
        else:
            self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).LMDBNPZDataset(args, "train")
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=args.batch_size,
            shuffle=False if args.ddp else True,
            num_workers=0,
            drop_last=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if args.ddp else None,
        )
        self.train_length = len(self.train_loader)
        logger.info("Init train dataloader success")

        if self.rank == 0:
            if args.train_rvq:
                self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
            else:
                self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).PickleDataset(args, "test")
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=1,  
                shuffle=False,  
                num_workers=args.loader_workers,
                drop_last=False,
            )
            logger.info(f"Init test dataloader success")

        model_module = __import__(f"models.{args.model}", fromlist=["something"])

        if args.ddp:
            model_raw = getattr(model_module, args.g_name)(args).to(self.device)
            model_raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_raw)
            self.model = DDP(
                model_raw,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            self.model_ema = ModelEma(model_raw, decay=0.9999)  # 用原始模型创建 EMA
        else:
            model_raw = getattr(model_module, args.g_name)(args)
            self.model = torch.nn.DataParallel(model_raw, args.gpus).cuda()
            self.model_ema = ModelEma(model_raw, decay=0.9999)

        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {args.g_name} success")
            if args.stat == "wandb":
                wandb.watch(self.model)

        if args.d_name is not None:
            if args.ddp:
                self.d_model = getattr(model_module, args.d_name)(args).to(self.device)
                self.d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.d_model)
                self.d_model = DDP(
                    self.d_model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=True,
                )
            else:
                self.d_model = torch.nn.DataParallel(getattr(model_module, args.d_name)(args), args.gpus).cuda()
            if self.rank == 0:
                logger.info(self.d_model)
                logger.info(f"init {args.d_name} success")
                if args.stat == "wandb":
                    wandb.watch(self.d_model)
            self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)
            self.opt_d_s = create_scheduler(args, self.opt_d)

        if args.e_name is not None:
            eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
            if self.args.ddp:
                self.eval_model = getattr(eval_model_module, args.e_name)(args).to(self.device)
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.device)
            else:
                self.eval_model = getattr(eval_model_module, args.e_name)(args)
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.device)

            other_tools.load_checkpoints(self.eval_copy, args.data_path + args.e_path, args.e_name)
            other_tools.load_checkpoints(self.eval_model, args.data_path + args.e_path, args.e_name)

            if self.args.ddp:
                self.eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.eval_model)
                self.eval_model = DDP(
                    self.eval_model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=True,
                )
            self.eval_model.eval()
            self.eval_copy.eval()
            if self.rank == 0:
                logger.info(self.eval_model)
                logger.info(f"init {args.e_name} success")
                if args.stat == "wandb":
                    wandb.watch(self.eval_model)

        self.opt = create_optimizer(args, self.model)
        self.opt_s = create_scheduler(args, self.opt)

        self.smplx = smplx.create(
            self.args.data_path_1 + "smplx_models/",
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False,
        ).to(self.device).eval()

        self.alignmenter = metric.alignment(
            0.3,
            7,
            self.train_data.avg_vel,
            upper_body=[3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
                        ]) if self.rank == 0 else None
        self.align_mask = 60
        self.l1_calculator = metric.L1div() if self.rank == 0 else None

    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).to(self.device)
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 165), device=self.device)
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 165), device=self.device)
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def inverse_selection_tensor_6d(self, filtered_t, selection_array, n):
        new_selected_array = np.zeros((330))
        new_selected_array[::2] = selection_array
        new_selected_array[1::2] = selection_array
        selection_array = new_selected_array
        selection_array = torch.from_numpy(selection_array).to(self.device)
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 330), device=self.device)
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 330), device=self.device)
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=None):
        pstr = "[%03d][%03d/%03d]  " % (epoch, its, self.train_length)
        for name, states in self.tracker.loss_meters.items():
            mtr = states['train']
            if mtr.count > 0:
                pstr += "{}: {:.3f}\t".format(name, mtr.avg)
                if self.args.stat == "ts":
                    self.writer.add_scalar(f"train/{name}", mtr.avg, epoch * self.train_length + its)
                else:
                    wandb.log({name: mtr.avg}, step=epoch * self.train_length + its)
        pstr += "glr: {:.1e}\t".format(lr_g)
        if self.args.stat == "ts":
            self.writer.add_scalar("lr/glr", lr_g, epoch * self.train_length + its)
        else:
            wandb.log({'glr': lr_g}, step=epoch * self.train_length + its)
        if lr_d is not None:
            pstr += "dlr: {:.1e}\t".format(lr_d)
            if self.args.stat == "ts":
                self.writer.add_scalar("lr/dlr", lr_d, epoch * self.train_length + its)
            else:
                wandb.log({'dlr': lr_d}, step=epoch * self.train_length + its)
        pstr += "dtime: %04d\t" % (t_data * 1000)
        pstr += "ntime: %04d\t" % (t_train * 1000)
        pstr += "mem: {:.2f} ".format(mem_cost * len(self.args.gpus))
        logger.info(pstr)

    def test_recording(self, dict_name, value, epoch):
        self.tracker.update_meter(dict_name, "test", value)
        _ = self.tracker.update_values(dict_name, 'test', epoch)


@logger.catch
def main_worker(args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # 读取 torchrun 注入的环境变量；未使用 torchrun 时会有默认值
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    logger.info(f"[Rank {rank}] main_worker, RANK {rank}, LOCAL_RANK {local_rank}, WORLD_SIZE {world_size}")
    
    args.local_rank = local_rank  # 传给 Trainer
    # args.ddp = (world_size > 1)  # 只以实际 world_size 判断
    # 如未用 torchrun 启动但 args.ddp=True，会造成阻塞，这里降级为单卡
    if args.ddp and world_size <= 1:
        logger.warning("WORLD_SIZE<=1，未检测到 torchrun 多进程，自动将 args.ddp 设为 False（单进程训练）。")
        args.ddp = False
    torch.cuda.set_device(local_rank)

    if args.ddp:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=10),
        )

    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)

    print(f"[Rank {rank}] Before Trainer Init")
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) \
        if args.trainer != "base" else BaseTrainer(args)
    print(f"[Rank {rank}] After Trainer Init")

    
    if args.inference:
        if rank == 0:
            if load_ckpt := args.load_ckpt:
                if os.path.exists(load_ckpt):
                    logger.info(f"Loading checkpoint from {load_ckpt}")
                    other_tools.load_checkpoints2(trainer.model, load_ckpt, "echomask_inference")
                    logger.info("Checkpoint loaded successfully")
                    trainer.model.eval()
                else:
                    logger.warning(f"Checkpoint {load_ckpt} does not exist. Starting training from scratch.")
                    raise FileNotFoundError(f"Checkpoint {load_ckpt} does not exist.")
            trainer.inference(args.audio_infer_path)
        return
    
    if args.test_state:
        if rank == 0:
            if load_ckpt := args.load_ckpt:
                if os.path.exists(load_ckpt):
                    logger.info(f"Loading checkpoint from {load_ckpt}")
                    other_tools.load_checkpoints2(trainer.model, load_ckpt, "echomask_test")
                    logger.info("Checkpoint loaded successfully")
                    trainer.model.eval()
                    fid = trainer.test(42)
                else:
                    logger.warning(f"Checkpoint {load_ckpt} does not exist. Starting training from scratch.")
                    raise FileNotFoundError(f"Checkpoint {load_ckpt} does not exist.")
        return

    logger.info("Training from scratch ...")
    for epoch in range(args.epochs + 1):
        if args.ddp and hasattr(trainer.train_loader, "sampler") and trainer.train_loader.sampler is not None:
            trainer.train_loader.sampler.set_epoch(epoch)
        if rank == 0:
            trainer.tracker.reset()

        trainer.train(epoch)

        if epoch == 0 or (epoch > 200) or (epoch <= 200 and epoch % args.test_period == 0):
            if rank == 0:
                fid = trainer.test(epoch)
                is_best = (fid is not None) and (fid < getattr(trainer, "best_fid", float("inf")))
                if is_best:
                    trainer.best_fid = fid
                    other_tools.save_checkpoints(
                        os.path.join(trainer.checkpoint_path, f"last_model_{epoch}.bin"),
                        trainer.model, opt=None, epoch=None, lrs=None
                    )

    if args.ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    args = config.parse_args()
    main_worker(args)
