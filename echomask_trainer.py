import train
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger
from utils import rotation_conversions as rc
import smplx
from utils import config, logger_tools, other_tools, metric, data_transfer
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
import librosa
import pickle
from torch.cuda.amp import GradScaler, autocast
from timm.utils import ModelEma
import torch.nn.parallel as tnp
import torch.distributed as dist
scaler = GradScaler()
from contextlib import nullcontext
def is_effective_ddp(model) -> bool:
    """只有当 (a) 已 init 进程组, (b) world_size>1, (c) 模型确实是 DDP 时才算有效 DDP。"""
    return (
        isinstance(model, tnp.DistributedDataParallel)
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
class CustomTrainer(train.BaseTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.now_epoch = 0
        self.best_fid = 1e9
        self.joints = self.train_data.joints
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]

        self.joint_mask_face = np.zeros(
            len(list(self.ori_joint_list.keys())) * 3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] -
                                 self.ori_joint_list[joint_name][0]:self.
                                 ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(
            len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] -
                                  self.ori_joint_list[joint_name][0]:self.
                                  ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(
            len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] -
                                  self.ori_joint_list[joint_name][0]:self.
                                  ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(
            len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] -
                                  self.ori_joint_list[joint_name][0]:self.
                                  ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools.EpochTracker([
            "c_h",
            "c_f",
            "af",
            "ah",
            "au",
            "al",
            "fid",
            "l1div",
            "bc",
            "rec",
            "trans",
            "vel",
            "transv",
            'dis',
            'gen',
            'acc',
            'transa',
            'exp',
            'lvd',
            'mse',
            "cls",
            "rec_face",
            "lat",
            "c_full",
            "c_self",
            "c_word",
            "lat_word",
            "lat_self",
            "moau",
            "moau_self",
            "moau_word",
        ], [
            False, True, True, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False
        ])
        ##### vq_model #####
        vq_model_module = __import__(f"models.motion_representation",
                                     fromlist=["something"])
        rvq_model_module = __import__(f"models.rvq", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106  # face
        # self.vq_model_face = getattr(vq_model_module, "VQVAEConvZero_VAE")(self.args).to(self.rank)
        self.vq_model_face = getattr(rvq_model_module,
                                     "RVQVAE")(self.args).to(self.rank)
        # other_tools.load_checkpoints(self.vq_model_face, self.args.data_path_1 +  "pretrained_vq/rvq_face_300_all.bin", args.e_name)

        other_tools.load_checkpoints(
            self.vq_model_face,
            self.args.data_path_1 + "pretrained_vq/rvq_face_600.bin",
            args.e_name)
        # other_tools.load_checkpoints(self.vq_model_face, self.args.data_path_1 +  "pretrained_vq/crf_8_256.bin", args.e_name)

        # other_tools.load_checkpoints(self.vq_model_face, self.args.data_path_1 +  "pretrained_vq/face_1200.bin", args.e_name)

        self.args.vae_test_dim = 78  # upper body
        # self.vq_model_upper = getattr(vq_model_module, "VQVAEConvZero_VAE")(self.args).to(self.rank)
        self.vq_model_upper = getattr(rvq_model_module,
                                      "RVQVAE")(self.args).to(self.rank)
        # other_tools.load_checkpoints(self.vq_model_upper, self.args.data_path_1 +  "pretrained_vq/rvq_upper_300_all.bin", args.e_name)

        other_tools.load_checkpoints(
            self.vq_model_upper,
            self.args.data_path_1 + "pretrained_vq/rvq_upper_500.bin",
            args.e_name)
        # other_tools.load_checkpoints(self.vq_model_upper, self.args.data_path_1 +  "pretrained_vq/cru_8_256.bin", args.e_name)

        # other_tools.load_checkpoints(self.vq_model_upper, self.args.data_path_1 +  "pretrained_vq/upper_1100.bin", args.e_name)

        self.args.vae_test_dim = 180  # hands
        # self.vq_model_hands = getattr(vq_model_module, "VQVAEConvZero_VAE")(self.args).to(self.rank)
        self.vq_model_hands = getattr(rvq_model_module,
                                      "RVQVAE")(self.args).to(self.rank)
        # other_tools.load_checkpoints(self.vq_model_hands, self.args.data_path_1 +  "pretrained_vq/rvq_hands_600_all.bin", args.e_name)
        other_tools.load_checkpoints(
            self.vq_model_hands,
            self.args.data_path_1 + "pretrained_vq/rvq_hands_500.bin",
            args.e_name)
        # other_tools.load_checkpoints(self.vq_model_hands, self.args.data_path_1 +  "pretrained_vq/crh_8_256.bin", args.e_name)

        # other_tools.load_checkpoints(self.vq_model_hands, self.args.data_path_1 +  "pretrained_vq/hands_1000.bin", args.e_name)

        self.args.vae_test_dim = 61  # lower body
        self.args.vae_layer = 4
        # self.vq_model_lower = getattr(vq_model_module, "VQVAEConvZero_VAE")(self.args).to(self.rank)
        self.vq_model_lower = getattr(rvq_model_module,
                                      "RVQVAE")(self.args).to(self.rank)
        # other_tools.load_checkpoints(self.vq_model_lower, self.args.data_path_1 +  "pretrained_vq/rvq_lower_1100_all.bin", args.e_name)
        other_tools.load_checkpoints(
            self.vq_model_lower,
            self.args.data_path_1 + "pretrained_vq/rvq_lower_600.bin",
            args.e_name)
        # other_tools.load_checkpoints(self.vq_model_lower, self.args.data_path_1 +  "pretrained_vq/crl_8_256.bin", args.e_name)

        # other_tools.load_checkpoints(self.vq_model_lower, self.args.data_path_1 +  "pretrained_vq/lower_1100.bin", args.e_name)

        self.args.vae_test_dim = 61  #global motion
        self.args.vae_layer = 4
        self.global_motion = getattr(vq_model_module,
                                     "VAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(
            self.global_motion,
            self.args.data_path_1 + "pretrained_vq/last_1700_foot.bin",
            args.e_name)
        # other_tools.load_checkpoints(self.global_motion, self.args.data_path_1 +  "pretrained_vq/foot_1600.bin", args.e_name)

        self.args.vae_test_dim = 330
        self.args.vae_layer = 4
        self.args.vae_length = 240
        rvq_model_module = __import__(f"models.rvq", fromlist=["something"])
        # self.rvq_model = getattr(rvq_model_module, "RVQVAE")(self.args).to(self.rank)
        # other_tools.load_checkpoints(self.rvq_model, self.args.data_path_1 +  "pretrained_vq/rvq_1000.bin", 'RVQVAE')

        self.vq_model_face.eval()
        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()
        self.global_motion.eval()
        # self.rvq_model.eval()
        self.cls_loss = nn.NLLLoss().to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank)
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)

    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).to(self.rank)
        original_shape_t = torch.zeros((n, 165)).to(self.rank)
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def _load_data(self, dict_data):
        return dict_data

    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data[
            "tar_pose"].shape[1], self.joints

        # ------ full generatation task ------ #
        mask_val = torch.ones(bs, n,
                              337).float().to(self.rank)  # 和latent_all的维度一样
        mask_val[:, :self.args.pre_frames, :] = 0.0  # 文章5.2.1 需要前四帧作为seed pose
        face_zq = loaded_data['zq_face'].to(self.rank)
        hands_zq = loaded_data['zq_hands'].to(self.rank)
        upper_zq = loaded_data['zq_upper'].to(self.rank)
        lower_zq = loaded_data['zq_lower'].to(self.rank)
        # with self.model.no_sync(
        # ) if self.args.ddp and mode == "train" else torch.cuda.amp.autocast():
        net_out_val = self.model(
            hubert=loaded_data['hubert'].to(self.rank),
            hubert_hid=loaded_data['hubert_hid'].to(self.rank),
            face_zq=face_zq,
            hands_zq=hands_zq,
            upper_zq=upper_zq,
            lower_zq=lower_zq,
            mask=mask_val,
            hard_mask=False,
            in_id=loaded_data['tar_id'].to(self.rank),
            in_motion=loaded_data['latent_all'].to(self.rank),
            use_attentions=True,
            use_word=True,
        )

        g_loss_final = torch.tensor(0.).to(self.rank)
        loss_moau = net_out_val["loss_moau"]
        g_loss_final += loss_moau
        if self.rank == 0:
            self.tracker.update_meter("moau", "train", loss_moau.item())
        loss_latent_face = self.reclatent_loss(net_out_val["rec_face"],
                                               face_zq)

        loss_latent_lower = self.reclatent_loss(net_out_val["rec_lower"],
                                                lower_zq)
        loss_latent_hands = self.reclatent_loss(net_out_val["rec_hands"],
                                                hands_zq)
        loss_latent_upper = self.reclatent_loss(net_out_val["rec_upper"],
                                                upper_zq)
        loss_latent = self.args.lf * loss_latent_face + self.args.ll * loss_latent_lower + self.args.lh * loss_latent_hands + self.args.lu * loss_latent_upper
        if self.rank == 0:
            self.tracker.update_meter("lat", "train", loss_latent.item())
        g_loss_final += loss_latent / 6

        


        self.now_epoch += 1
        ##### cons loss #####
        g_loss_final += net_out_val["hubert_cons_loss"] + net_out_val[
            "beat_cons_loss"]
        if self.rank == 0:
            self.tracker.update_meter("c_f", "train",
                                      net_out_val["hubert_cons_loss"].item())
            self.tracker.update_meter("c_h", "train",
                                      net_out_val["beat_cons_loss"].item())

            
        loss_cls = 0
        tar_index_value_face_top = loaded_data[
            "tar_index_value_face_top"].reshape(-1, 6).to(self.rank)
        tar_index_value_upper_top = loaded_data[
            "tar_index_value_upper_top"].reshape(-1, 6).to(self.rank)
        tar_index_value_lower_top = loaded_data[
            "tar_index_value_lower_top"].reshape(-1, 6).to(self.rank)
        tar_index_value_hands_top = loaded_data[
            "tar_index_value_hands_top"].reshape(-1, 6).to(self.rank)
        for i in range(6):
            rec_index_face_val = self.log_softmax(
                net_out_val["cls_face"][:, :, :, i]).reshape(
                    -1, self.args.vae_codebook_size)
            rec_index_upper_val = self.log_softmax(
                net_out_val["cls_upper"][:, :, :, i]).reshape(
                    -1, self.args.vae_codebook_size)
            rec_index_lower_val = self.log_softmax(
                net_out_val["cls_lower"][:, :, :, i]).reshape(
                    -1, self.args.vae_codebook_size)
            rec_index_hands_val = self.log_softmax(
                net_out_val["cls_hands"][:, :, :, i]).reshape(
                    -1, self.args.vae_codebook_size)

            loss_cls_i = self.args.cf*self.cls_loss(rec_index_face_val, tar_index_value_face_top[:,i])\
                + self.args.cu*self.cls_loss(rec_index_upper_val, tar_index_value_upper_top[:,i])\
                + self.args.cl*self.cls_loss(rec_index_lower_val, tar_index_value_lower_top[:,i])\
                + self.args.ch*self.cls_loss(rec_index_hands_val, tar_index_value_hands_top[:,i])
            loss_cls = loss_cls + loss_cls_i / (i + 1)

        if self.rank == 0:
            self.tracker.update_meter("c_full", "train", loss_cls.item())
        g_loss_final += loss_cls

        if mode == 'train':
            #     # ------ masked gesture moderling------ #

            mask_ratio = 0.375
            hard_ratio = (epoch / 400) * 0.2
            soft_ratio = 0.2 - (epoch / 400) * 0.2

            # mask = torch.rand(bs, n, self.args.pose_dims + 3 + 4) < mask_ratio
            # mask = mask.float().to(self.rank)
            # ------ masked self ------ #
            with torch.no_grad():
                mask_frame = self.model_ema.ema.gen_mask_frame(
                    hubert=loaded_data['hubert'].to(self.rank),
                    hubert_hid=loaded_data['hubert_hid'].to(self.rank),
                    face_zq=face_zq,
                    hands_zq=hands_zq,
                    upper_zq=upper_zq,
                    lower_zq=lower_zq,
                    mask_ratio=mask_ratio,
                    hard_ratio=hard_ratio,
                    soft_ratio=soft_ratio,
                    # mask=mask,
                    in_id=loaded_data['tar_id'].to(self.rank),
                    in_motion=loaded_data['latent_all'].to(self.rank),
                    use_attentions=True,
                    use_word=True,
                    hard_mask=True,
                    train_mode=True,
                )
            net_out_self = self.model(
                hubert=loaded_data['hubert'].to(self.rank),
                hubert_hid=loaded_data['hubert_hid'].to(self.rank),
                face_zq=face_zq,
                hands_zq=hands_zq,
                upper_zq=upper_zq,
                lower_zq=lower_zq,
                mask_ratio=mask_ratio,
                hard_ratio=hard_ratio,
                soft_ratio=soft_ratio,
                mask_frame=mask_frame,
                in_id=loaded_data['tar_id'].to(self.rank),
                in_motion=loaded_data['latent_all'].to(self.rank),
                use_attentions=True,
                use_word=False,
                hard_mask=True,
                train_mode=True,
            )
            # for name, param in self.model.module.mask_generator.shared_transformer_decoder.named_parameters():
            #     print(name, param.requires_grad)
            # exit()
            loss_moau_self = net_out_self["loss_moau"]
            g_loss_final += loss_moau_self
            if self.rank == 0:
                self.tracker.update_meter("moau_self", "train",
                                          loss_moau_self.item())
            loss_latent_face_self = self.reclatent_loss(
                net_out_self["rec_face"], face_zq)
            loss_latent_lower_self = self.reclatent_loss(
                net_out_self["rec_lower"], lower_zq)
            loss_latent_hands_self = self.reclatent_loss(
                net_out_self["rec_hands"], hands_zq)
            loss_latent_upper_self = self.reclatent_loss(
                net_out_self["rec_upper"], upper_zq)
            loss_latent_self = self.args.lf * loss_latent_face_self + self.args.ll * loss_latent_lower_self + self.args.lh * loss_latent_hands_self + self.args.lu * loss_latent_upper_self
            if self.rank == 0:
                self.tracker.update_meter("lat_self", "train",
                                          loss_latent_self.item())
            g_loss_final += loss_latent_self / 6
            index_loss_top_self = 0
            for i in range(6):
                rec_index_face_self = self.log_softmax(
                    net_out_self["cls_face"][:, :, :, i]).reshape(
                        -1, self.args.vae_codebook_size)
                rec_index_upper_self = self.log_softmax(
                    net_out_self["cls_upper"][:, :, :, i]).reshape(
                        -1, self.args.vae_codebook_size)
                rec_index_lower_self = self.log_softmax(
                    net_out_self["cls_lower"][:, :, :, i]).reshape(
                        -1, self.args.vae_codebook_size)
                rec_index_hands_self = self.log_softmax(
                    net_out_self["cls_hands"][:, :, :, i]).reshape(
                        -1, self.args.vae_codebook_size)
                index_loss_top_self_i = self.cls_loss(
                    rec_index_face_self,
                    tar_index_value_face_top[:, i]) + self.cls_loss(
                        rec_index_upper_self,
                        tar_index_value_upper_top[:, i]) + self.cls_loss(
                            rec_index_lower_self,
                            tar_index_value_lower_top[:, i]) + self.cls_loss(
                                rec_index_hands_self,
                                tar_index_value_hands_top[:, i])
                index_loss_top_self = index_loss_top_self + index_loss_top_self_i / (
                    i + 1)
            if self.rank == 0:
                self.tracker.update_meter("c_self", "train",
                                          index_loss_top_self.item())
            g_loss_final += index_loss_top_self

            # ------ masked word ------ #
            net_out_word = self.model(
                hubert=loaded_data['hubert'].to(self.rank),
                hubert_hid=loaded_data['hubert_hid'].to(self.rank),
                face_zq=face_zq,
                hands_zq=hands_zq,
                upper_zq=upper_zq,
                lower_zq=lower_zq,
                mask_ratio=mask_ratio,
                hard_ratio=hard_ratio,
                soft_ratio=soft_ratio,
                mask_frame=mask_frame,
                in_id=loaded_data['tar_id'].to(self.rank),
                in_motion=loaded_data['latent_all'].to(self.rank),
                use_attentions=True,
                use_word=True,
                hard_mask=True,
                train_mode=True,
            )
            loss_moau_word = net_out_word["loss_moau"]
            g_loss_final += loss_moau_word
            if self.rank == 0:
                self.tracker.update_meter("moau_word", "train",
                                          loss_moau_word.item())
            loss_latent_face_word = self.reclatent_loss(
                net_out_word["rec_face"], face_zq)
            loss_latent_lower_word = self.reclatent_loss(
                net_out_word["rec_lower"], lower_zq)
            loss_latent_hands_word = self.reclatent_loss(
                net_out_word["rec_hands"], hands_zq)
            loss_latent_upper_word = self.reclatent_loss(
                net_out_word["rec_upper"], upper_zq)
            loss_latent_word = self.args.lf * loss_latent_face_word + self.args.ll * loss_latent_lower_word + self.args.lh * loss_latent_hands_word + self.args.lu * loss_latent_upper_word
            if self.rank == 0:
                self.tracker.update_meter("lat_word", "train",
                                          loss_latent_word.item())
            g_loss_final += loss_latent_word / 6
            index_loss_top_word = 0
            for i in range(6):
                rec_index_face_word = self.log_softmax(
                    net_out_word["cls_face"][:, :, :, i]).reshape(
                        -1, self.args.vae_codebook_size)
                rec_index_upper_word = self.log_softmax(
                    net_out_word["cls_upper"][:, :, :, i]).reshape(
                        -1, self.args.vae_codebook_size)
                rec_index_lower_word = self.log_softmax(
                    net_out_word["cls_lower"][:, :, :, i]).reshape(
                        -1, self.args.vae_codebook_size)
                rec_index_hands_word = self.log_softmax(
                    net_out_word["cls_hands"][:, :, :, i]).reshape(
                        -1, self.args.vae_codebook_size)
                index_loss_top_word_i = self.cls_loss(
                    rec_index_face_word,
                    tar_index_value_face_top[:, i]) + self.cls_loss(
                        rec_index_upper_word,
                        tar_index_value_upper_top[:, i]) + self.cls_loss(
                            rec_index_lower_word,
                            tar_index_value_lower_top[:, i]) + self.cls_loss(
                                rec_index_hands_word,
                                tar_index_value_hands_top[:, i])
                index_loss_top_word = index_loss_top_word + index_loss_top_word_i / (
                    i + 1)
            if self.rank == 0:
                self.tracker.update_meter("c_word", "train",
                                          index_loss_top_word.item())
            g_loss_final += index_loss_top_word

        return g_loss_final

    def _g_test(self, loaded_data):
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data[
            "tar_pose"].shape[1], self.joints
        # print('test data n:', n)
        tar_pose = loaded_data["tar_pose"].to(self.rank)
        tar_beta = loaded_data["tar_beta"].to(self.rank)
        in_word = loaded_data["in_word"].to(self.rank)
        tar_exps = loaded_data["tar_exps"].to(self.rank)
        tar_contact = loaded_data["tar_contact"].to(self.rank)
        in_audio = loaded_data["in_audio"].to(self.rank)
        # in_beat = loaded_data["in_beat"]
        tar_trans = loaded_data["tar_trans"].to(self.rank)
        hubert = loaded_data["hubert"].to(self.rank)
        hubert_hid = loaded_data["hubert_hid"].to(self.rank)
        beat = loaded_data["beat"].to(self.rank)
        remain = n % 8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            # in_beat = in_beat[:, :-remain, :]

            n = n - remain

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(
            tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(
            bs, n, 1 * 6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25 * 3:55 * 3]
        tar_pose_hands = rc.axis_angle_to_matrix(
            tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(
            bs, n, 30 * 6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(
            tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(
            bs, n, 13 * 6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(
            tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(
            bs, n, 9 * 6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact],
                                   dim=2)

        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(
            bs, n, 55 * 6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)

        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []
        # rec_index_all_face_bot = []
        # rec_index_all_upper_bot = []
        # rec_index_all_lower_bot = []
        # rec_index_all_hands_bot = []

        roundt = (n - self.args.pre_frames) // (self.args.pose_length -
                                                self.args.pre_frames)
        remain = (n - self.args.pre_frames) % (self.args.pose_length -
                                               self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames

        # pad latent_all_9 to the same length with latent_all
        # if n - latent_all_9.shape[1] >= 0:
        #     latent_all = torch.cat([latent_all_9, torch.zeros(bs, n - latent_all_9.shape[1], latent_all_9.shape[2]).to(self.rank)], dim=1)
        # else:
        #     latent_all = latent_all_9[:, :n, :]

        for i in range(0, roundt):
            in_word_tmp = in_word[:, i * (round_l):(i + 1) * (round_l) +
                                  self.args.pre_frames]
            # audio fps is 16000 and pose fps is 30
            in_audio_tmp = in_audio[:, i * (16000 // 30 * round_l):(i + 1) *
                                    (16000 // 30 * round_l) +
                                    16000 // 30 * self.args.pre_frames]
            in_beat_tmp = beat[:, i * (round_l):(i + 1) * (round_l) +
                               self.args.pre_frames]
            in_id_tmp = loaded_data['tar_id'][:, i * (round_l):(i + 1) *
                                              (round_l) + self.args.pre_frames]
            hubert_tmp = hubert[:, i * (round_l):(i + 1) * (round_l) +
                                self.args.pre_frames]
            hubert_hid_tmp = hubert_hid[:, i * (round_l):(i + 1) * (round_l) +
                                        self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length,
                                  self.args.pose_dims + 3 + 4).float().to(
                                      self.rank)
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                latent_all_tmp = latent_all[:,
                                            i * (round_l):(i + 1) * (round_l) +
                                            self.args.pre_frames, :]
            else:
                latent_all_tmp = latent_all[:,
                                            i * (round_l):(i + 1) * (round_l) +
                                            self.args.pre_frames, :]
                # print(latent_all_tmp.shape, latent_last.shape)
                latent_all_tmp[:, :self.args.
                               pre_frames, :] = latent_last[:, -self.args.
                                                            pre_frames:, :]

            net_out_val = self.model(
                mask=mask_val,
                in_motion=latent_all_tmp,
                in_id=in_id_tmp,
                hubert=hubert_tmp,
                hubert_hid=hubert_hid_tmp,
                use_attentions=True,
                train_mode=False,
            )

            # net_out_val = self.model(
            #     in_audio = in_audio_tmp,
            #     in_word=in_word_tmp,
            #     mask=mask_val,
            #     in_motion = latent_all_tmp,
            #     in_id = in_id_tmp,
            #     use_attentions=True,)

            rec_index_upper = self.log_softmax(
                net_out_val["cls_upper"]).reshape(-1,
                                                  self.args.vae_codebook_size,
                                                  6)
            # print('tar_pose:',tar_pose.shape)
            # print('rec_index_upper:', rec_index_upper.shape)
            # exit()
            _, rec_index_upper = torch.max(rec_index_upper.reshape(
                -1, 16, self.args.vae_codebook_size, 6),
                                           dim=2)

            rec_index_lower = self.log_softmax(
                net_out_val["cls_lower"]).reshape(-1,
                                                  self.args.vae_codebook_size,
                                                  6)
            _, rec_index_lower = torch.max(rec_index_lower.reshape(
                -1, 16, self.args.vae_codebook_size, 6),
                                           dim=2)

            rec_index_hands = self.log_softmax(
                net_out_val["cls_hands"]).reshape(-1,
                                                  self.args.vae_codebook_size,
                                                  6)
            _, rec_index_hands = torch.max(rec_index_hands.reshape(
                -1, 16, self.args.vae_codebook_size, 6),
                                           dim=2)

            rec_index_face = self.log_softmax(net_out_val["cls_face"]).reshape(
                -1, self.args.vae_codebook_size, 6)
            _, rec_index_face = torch.max(rec_index_face.reshape(
                -1, 16, self.args.vae_codebook_size, 6),
                                          dim=2)

            if i == 0:
                rec_index_all_face.append(rec_index_face)
                rec_index_all_upper.append(rec_index_upper)
                rec_index_all_lower.append(rec_index_lower)
                rec_index_all_hands.append(rec_index_hands)
            else:
                rec_index_all_face.append(rec_index_face[:, 1:])
                rec_index_all_upper.append(rec_index_upper[:, 1:])
                rec_index_all_lower.append(rec_index_lower[:, 1:])
                rec_index_all_hands.append(rec_index_hands[:, 1:])

            # print('rec_index_upper:', rec_index_upper.shape)
            # exit()
            rec_upper_last = self.vq_model_upper.decode(rec_index_upper)

            rec_lower_last = self.vq_model_lower.decode(rec_index_lower)

            rec_hands_last = self.vq_model_hands.decode(rec_index_hands)

            rec_pose_legs = rec_lower_last[:, :, :54]
            bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
            rec_pose_upper = rec_upper_last.reshape(bs, n, 13, 6)
            rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)  #
            rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(
                bs * n, 13 * 3)
            rec_pose_upper_recover = self.inverse_selection_tensor(
                rec_pose_upper, self.joint_mask_upper, bs * n)
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(
                bs * n, 9 * 3)
            rec_pose_lower_recover = self.inverse_selection_tensor(
                rec_pose_lower, self.joint_mask_lower, bs * n)
            rec_pose_hands = rec_hands_last.reshape(bs, n, 30, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(
                bs * n, 30 * 3)
            rec_pose_hands_recover = self.inverse_selection_tensor(
                rec_pose_hands, self.joint_mask_hands, bs * n)
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
            rec_trans_v_s = rec_lower_last[:, :, 54:57]
            rec_x_trans = other_tools.velocity2position(
                rec_trans_v_s[:, :, 0:1], 1 / self.args.pose_fps,
                tar_trans[:, 0, 0:1])
            rec_z_trans = other_tools.velocity2position(
                rec_trans_v_s[:, :, 2:3], 1 / self.args.pose_fps,
                tar_trans[:, 0, 2:3])
            rec_y_trans = rec_trans_v_s[:, :, 1:2]
            rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans],
                                  dim=-1)
            latent_last = torch.cat(
                [rec_pose, rec_trans, rec_lower_last[:, :, 57:61]], dim=-1)

        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)

        rec_upper = self.vq_model_upper.decode(rec_index_upper)

        rec_lower = self.vq_model_lower.decode(rec_index_lower)

        rec_hands = self.vq_model_hands.decode(rec_index_hands)

        rec_face = self.vq_model_face.decode(rec_index_face)

        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)  #
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(
            bs * n, 13 * 3)
        rec_pose_upper_recover = self.inverse_selection_tensor(
            rec_pose_upper, self.joint_mask_upper, bs * n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(
            rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(
            bs * n, 9 * 3)
        rec_pose_lower_recover = self.inverse_selection_tensor(
            rec_pose_lower, self.joint_mask_lower, bs * n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(
            bs * n, 30 * 3)
        rec_pose_hands_recover = self.inverse_selection_tensor(
            rec_pose_hands, self.joint_mask_hands, bs * n)
        rec_pose_jaw = rec_pose_jaw.reshape(bs * n, 6)
        rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
        rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(
            bs * n, 1 * 3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = self.global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1],
                                                    1 / self.args.pose_fps,
                                                    tar_trans[:, 0, 0:1])
        rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3],
                                                    1 / self.args.pose_fps,
                                                    tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:, :, 1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs * n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j * 6)

        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }

    def train(self, epoch):
        use_adv = bool(epoch >= self.args.no_adv_epoch)
        self.model.train()

        t_start = time.time()
        if self.rank == 0:
            self.tracker.reset()
            print(f"[Rank {self.rank}] Start epoch {epoch}")

        accum_steps = getattr(self.args, "accum_steps", 1)
        assert isinstance(accum_steps, int) and accum_steps >= 1
        use_amp = True  # 如需可切到 self.args.amp

        for its, batch_data in enumerate(self.train_loader):
            t_data = time.time() - t_start
            loaded_data = self._load_data(batch_data)

            do_step = ((its + 1) % accum_steps == 0)

            # 仅在“非更新步”屏蔽梯度同步
            need_no_sync = (is_effective_ddp(self.model) and not do_step)
            no_sync_ctx = (self.model.no_sync() if need_no_sync else nullcontext())

            with no_sync_ctx:
                if use_amp:
                    with torch.cuda.amp.autocast():
                        g_loss_final = self._g_training(loaded_data, use_adv, 'train', epoch)
                        if not torch.is_tensor(g_loss_final):
                            g_loss_final = torch.tensor(g_loss_final, device=self.device, dtype=torch.float32)
                        loss = g_loss_final / accum_steps
                    scaler.scale(loss).backward()
                else:
                    g_loss_final = self._g_training(loaded_data, use_adv, 'train', epoch)
                    if not torch.is_tensor(g_loss_final):
                        g_loss_final = torch.tensor(g_loss_final, device=self.device, dtype=torch.float32)
                    loss = g_loss_final / accum_steps
                    loss.backward()

            if do_step:
                # 梯度裁剪前，若用 AMP 需要先 unscale
                if self.args.grad_norm != 0:
                    if use_amp:
                        scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)

                if use_amp:
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    self.opt.step()
                self.opt.zero_grad(set_to_none=True)

                # 仅在真实 step 后再做 EMA.update（你原来已这么做，保留即可）
                if hasattr(self, "model_ema"):
                    raw_model = (self.model.module
                                if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
                                else self.model)
                    with torch.no_grad():
                        self.model_ema.update(raw_model)

            # 计时与日志（与你原来一致）
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                dev_idx = torch.cuda.current_device()
                mem_cost = torch.cuda.memory_allocated(dev_idx) / 1e9
            else:
                mem_cost = 0.0

            lr_g = self.opt.param_groups[0]['lr']
            t_train = time.time() - t_start - t_data
            t_start = time.time()

            if its % self.args.log_period == 0 and self.rank == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)

            if self.args.debug and its >= 1:
                break

        self.opt_s.step(epoch)

    def test(self, epoch):

        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path):
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0
        latent_out = []
        latent_ori = []
        l2_all = 0
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                # print(rec_pose.shape, tar_pose.shape)
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30 / self.args.pose_fps) != 1:
                    assert 30 % self.args.pose_fps == 0
                    n *= int(30 / self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(
                        tar_pose.permute(0, 2, 1),
                        scale_factor=30 / self.args.pose_fps,
                        mode='linear').permute(0, 2, 1)
                    rec_pose = torch.nn.functional.interpolate(
                        rec_pose.permute(0, 2, 1),
                        scale_factor=30 / self.args.pose_fps,
                        mode='linear').permute(0, 2, 1)

                # print(rec_pose.shape, tar_pose.shape)
                rec_pose = rc.rotation_6d_to_matrix(
                    rec_pose.reshape(bs * n, j, 6))
                rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(
                    bs, n, j * 6)
                tar_pose = rc.rotation_6d_to_matrix(
                    tar_pose.reshape(bs * n, j, 6))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(
                    bs, n, j * 6)
                remain = n % self.args.vae_test_len
                latent_out.append(
                    self.eval_copy.map2latent(
                        rec_pose[:, :n - remain]).reshape(
                            -1, self.args.vae_length).detach().cpu().numpy()
                )  # bs * n/8, 240
                latent_ori.append(
                    self.eval_copy.map2latent(
                        tar_pose[:, :n - remain]).reshape(
                            -1, self.args.vae_length).detach().cpu().numpy())

                rec_pose = rc.rotation_6d_to_matrix(
                    rec_pose.reshape(bs * n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(
                    bs * n, j * 3)
                tar_pose = rc.rotation_6d_to_matrix(
                    tar_pose.reshape(bs * n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(
                    bs * n, j * 3)

                vertices_rec = self.smplx(
                    betas=tar_beta.reshape(bs * n, 300),
                    transl=rec_trans.reshape(bs * n, 3) -
                    rec_trans.reshape(bs * n, 3),
                    expression=tar_exps.reshape(bs * n, 100) -
                    tar_exps.reshape(bs * n, 100),
                    jaw_pose=rec_pose[:, 66:69],
                    global_orient=rec_pose[:, :3],
                    body_pose=rec_pose[:, 3:21 * 3 + 3],
                    left_hand_pose=rec_pose[:, 25 * 3:40 * 3],
                    right_hand_pose=rec_pose[:, 40 * 3:55 * 3],
                    return_joints=True,
                    leye_pose=rec_pose[:, 69:72],
                    reye_pose=rec_pose[:, 72:75],
                )

                vertices_rec_face = self.smplx(
                    betas=tar_beta.reshape(bs * n, 300),
                    transl=rec_trans.reshape(bs * n, 3) -
                    rec_trans.reshape(bs * n, 3),
                    expression=rec_exps.reshape(bs * n, 100),
                    jaw_pose=rec_pose[:, 66:69],
                    global_orient=rec_pose[:, :3] - rec_pose[:, :3],
                    body_pose=rec_pose[:, 3:21 * 3 + 3] -
                    rec_pose[:, 3:21 * 3 + 3],
                    left_hand_pose=rec_pose[:, 25 * 3:40 * 3] -
                    rec_pose[:, 25 * 3:40 * 3],
                    right_hand_pose=rec_pose[:, 40 * 3:55 * 3] -
                    rec_pose[:, 40 * 3:55 * 3],
                    return_verts=True,
                    return_joints=True,
                    leye_pose=rec_pose[:, 69:72] - rec_pose[:, 69:72],
                    reye_pose=rec_pose[:, 72:75] - rec_pose[:, 72:75],
                )
                vertices_tar_face = self.smplx(
                    betas=tar_beta.reshape(bs * n, 300),
                    transl=tar_trans.reshape(bs * n, 3) -
                    tar_trans.reshape(bs * n, 3),
                    expression=tar_exps.reshape(bs * n, 100),
                    jaw_pose=tar_pose[:, 66:69],
                    global_orient=tar_pose[:, :3] - tar_pose[:, :3],
                    body_pose=tar_pose[:, 3:21 * 3 + 3] -
                    tar_pose[:, 3:21 * 3 + 3],
                    left_hand_pose=tar_pose[:, 25 * 3:40 * 3] -
                    tar_pose[:, 25 * 3:40 * 3],
                    right_hand_pose=tar_pose[:, 40 * 3:55 * 3] -
                    tar_pose[:, 40 * 3:55 * 3],
                    return_verts=True,
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72] - tar_pose[:, 69:72],
                    reye_pose=tar_pose[:, 72:75] - tar_pose[:, 72:75],
                )

                joints_rec = vertices_rec["joints"].detach().cpu().numpy(
                ).reshape(1, n, 127 * 3)[0, :n, :55 * 3]
                # print('vertices_rec["joints"].shape:', vertices_rec["joints"].shape)
                # if its == 7:
                #     with open('/mnt/disk2T/mm_data/zxy/PantoMatrix-main/scripts/EMAGE_2024/charm_8.pkl', 'wb') as f:  # 'ab'表示以二进制追加模式打开文件
                #         pickle.dump(joints_rec, f)

                # joints_tar = vertices_tar["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
                facial_rec = vertices_rec_face['vertices'].reshape(1, n,
                                                                   -1)[0, :n]
                facial_tar = vertices_tar_face['vertices'].reshape(1, n,
                                                                   -1)[0, :n]
                face_vel_loss = self.vel_loss(
                    facial_rec[1:, :] - facial_tar[:-1, :],
                    facial_tar[1:, :] - facial_tar[:-1, :])
                # print("facial_rec:",facial_rec.shape)
                # print("facial_tar:",facial_tar.shape)
                # exit()
                l2 = self.reclatent_loss(facial_rec, facial_tar)
                l2_all += l2.item() * n
                lvel += face_vel_loss.item() * n

                _ = self.l1_calculator.run(joints_rec)
                if self.alignmenter is not None:
                    in_audio_eval, sr = librosa.load(
                        self.args.data_path + "wave16k/" +
                        test_seq_list.iloc[its]['id'] + ".wav")
                    in_audio_eval = librosa.resample(
                        in_audio_eval,
                        orig_sr=sr,
                        target_sr=self.args.audio_sr)
                    a_offset = int(self.align_mask *
                                   (self.args.audio_sr / self.args.pose_fps))
                    onset_bt = self.alignmenter.load_audio(
                        in_audio_eval[:int(self.args.audio_sr /
                                           self.args.pose_fps * n)], a_offset,
                        len(in_audio_eval) - a_offset, True)
                    beat_vel = self.alignmenter.load_pose(
                        joints_rec, self.align_mask, n - self.align_mask, 30,
                        True)
                    # print(beat_vel)
                    align += (self.alignmenter.calculate_align(
                        onset_bt, beat_vel, 30) * (n - 2 * self.align_mask))

                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(
                    bs * n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(
                    bs * n, 100)
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(
                    bs * n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(
                    bs * n, 3)
                gt_npz = np.load(self.args.data_path + self.args.pose_rep +
                                 "/" + test_seq_list.iloc[its]['id'] + ".npz",
                                 allow_pickle=True)
                np.savez(
                    results_save_path + "gt_" + test_seq_list.iloc[its]['id'] +
                    '.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate=30,
                )
                np.savez(
                    results_save_path + "res_" +
                    test_seq_list.iloc[its]['id'] + '.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate=30,
                )
                # total_length += n
                np.savez(
                    results_save_path + "sem_face_" +
                    test_seq_list.iloc[its]['id'] + '.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np - rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np - rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate=30,
                )
                total_length += n
        logger.info(f"l2 loss: {l2_all/total_length}")
        logger.info(f"lvel loss: {lvel/total_length}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        fid = data_tools.FIDCalculator.frechet_distance(
            latent_out_all, latent_ori_all)
        logger.info(f"fid score: {fid}")
        self.test_recording("fid", fid, epoch)

        align_avg = align / (total_length -
                             2 * len(self.test_loader) * self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.test_recording("bc", align_avg, epoch)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.test_recording("l1div", l1div, epoch)

        # data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion"
        )
        return fid

    
    def inference(self, audio_path):
        mode = 'inference'
        # test_seq_list = self.test_data.selected_file
        from utils import audio_to_frame_tokens


        hubert, hubert_hid  = audio_to_frame_tokens.get_hubert(audio_path, self.args)
        hubert = hubert.to(self.rank)
        hubert_hid = hubert_hid.to(self.rank)
        print("frame:",hubert.shape, hubert_hid.shape, hubert_hid.shape)
        frames = hubert.shape[1]
        bs, j = 1, self.joints
        
        # 从 dataloader 中取出第一个 batch
        batch_data = next(iter(self.test_loader))
        loaded_data = self._load_data(batch_data)
        tar_pose = loaded_data["tar_pose"].cuda()
        tar_beta = loaded_data["tar_beta"].cuda()
        tar_exps = loaded_data["tar_exps"].cuda()
        tar_contact = loaded_data["tar_contact"].cuda()
        tar_trans = loaded_data["tar_trans"].cuda()
        
        
        pos_len = tar_pose.shape[1]
        remain = pos_len%8
        if remain != 0:
            tar_pose = tar_pose[:, :pos_len-remain, :]
            tar_beta = tar_beta[:, :pos_len-remain, :]
            tar_trans = tar_trans[:, :pos_len-remain, :]
            tar_exps = tar_exps[:, :pos_len-remain, :]
            tar_contact = tar_contact[:, :pos_len-remain, :]
            pos_len = pos_len - remain
        n = pos_len
        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []
        sem_score = []
        
        roundt = (frames - self.args.pre_frames) // (self.args.pose_length -
                                                self.args.pre_frames)
        
        round_l = self.args.pose_length - self.args.pre_frames
        
        for i in range(0, roundt):

            in_id_tmp = loaded_data['tar_id'][:, : round_l+self.args.pre_frames]
            hubert_tmp = hubert[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            hubert_hid_tmp = hubert_hid[:, i * (round_l):(i + 1) * (round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
            else:
                latent_all_tmp = torch.zeros_like(latent_last[:, :round_l+self.args.pre_frames, :])
                latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]
            
            net_out_val = self.model(
                mask=mask_val,
                in_motion=latent_all_tmp,
                in_id=in_id_tmp,
                hubert=hubert_tmp,
                hubert_hid=hubert_hid_tmp,
                use_attentions=True,
                train_mode=False,
            )

            rec_index_upper = self.log_softmax(
                net_out_val["cls_upper"]).reshape(-1,
                                                  self.args.vae_codebook_size,
                                                  6)

            _, rec_index_upper = torch.max(rec_index_upper.reshape(
                -1, 16, self.args.vae_codebook_size, 6),
                                           dim=2)

            rec_index_lower = self.log_softmax(
                net_out_val["cls_lower"]).reshape(-1,
                                                  self.args.vae_codebook_size,
                                                  6)
            _, rec_index_lower = torch.max(rec_index_lower.reshape(
                -1, 16, self.args.vae_codebook_size, 6),
                                           dim=2)

            rec_index_hands = self.log_softmax(
                net_out_val["cls_hands"]).reshape(-1,
                                                  self.args.vae_codebook_size,
                                                  6)
            _, rec_index_hands = torch.max(rec_index_hands.reshape(
                -1, 16, self.args.vae_codebook_size, 6),
                                           dim=2)

            rec_index_face = self.log_softmax(net_out_val["cls_face"]).reshape(
                -1, self.args.vae_codebook_size, 6)
            _, rec_index_face = torch.max(rec_index_face.reshape(
                -1, 16, self.args.vae_codebook_size, 6),
                                          dim=2)

            if i == 0:
                rec_index_all_face.append(rec_index_face)
                rec_index_all_upper.append(rec_index_upper)
                rec_index_all_lower.append(rec_index_lower)
                rec_index_all_hands.append(rec_index_hands)
            else:
                rec_index_all_face.append(rec_index_face[:, 1:])
                rec_index_all_upper.append(rec_index_upper[:, 1:])
                rec_index_all_lower.append(rec_index_lower[:, 1:])
                rec_index_all_hands.append(rec_index_hands[:, 1:])

            # print('rec_index_upper:', rec_index_upper.shape)
            # exit()
            rec_upper_last = self.vq_model_upper.decode(rec_index_upper)

            rec_lower_last = self.vq_model_lower.decode(rec_index_lower)

            rec_hands_last = self.vq_model_hands.decode(rec_index_hands)

            rec_pose_legs = rec_lower_last[:, :, :54]
            bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
            rec_pose_upper = rec_upper_last.reshape(bs, n, 13, 6)
            rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)  #
            rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(
                bs * n, 13 * 3)
            rec_pose_upper_recover = self.inverse_selection_tensor(
                rec_pose_upper, self.joint_mask_upper, bs * n)
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(
                bs * n, 9 * 3)
            rec_pose_lower_recover = self.inverse_selection_tensor(
                rec_pose_lower, self.joint_mask_lower, bs * n)
            rec_pose_hands = rec_hands_last.reshape(bs, n, 30, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(
                bs * n, 30 * 3)
            rec_pose_hands_recover = self.inverse_selection_tensor(
                rec_pose_hands, self.joint_mask_hands, bs * n)
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
            rec_trans_v_s = rec_lower_last[:, :, 54:57]
            rec_x_trans = other_tools.velocity2position(
                rec_trans_v_s[:, :, 0:1], 1 / self.args.pose_fps,
                tar_trans[:, 0, 0:1])
            rec_z_trans = other_tools.velocity2position(
                rec_trans_v_s[:, :, 2:3], 1 / self.args.pose_fps,
                tar_trans[:, 0, 2:3])
            rec_y_trans = rec_trans_v_s[:, :, 1:2]
            rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans],
                                  dim=-1)
            latent_last = torch.cat(
                [rec_pose, rec_trans, rec_lower_last[:, :, 57:61]], dim=-1)

        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)

        rec_upper = self.vq_model_upper.decode(rec_index_upper)

        rec_lower = self.vq_model_lower.decode(rec_index_lower)

        rec_hands = self.vq_model_hands.decode(rec_index_hands)

        rec_face = self.vq_model_face.decode(rec_index_face)

        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)  #
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(
            bs * n, 13 * 3)
        rec_pose_upper_recover = self.inverse_selection_tensor(
            rec_pose_upper, self.joint_mask_upper, bs * n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(
            rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(
            bs * n, 9 * 3)
        rec_pose_lower_recover = self.inverse_selection_tensor(
            rec_pose_lower, self.joint_mask_lower, bs * n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(
            bs * n, 30 * 3)
        rec_pose_hands_recover = self.inverse_selection_tensor(
            rec_pose_hands, self.joint_mask_hands, bs * n)
        rec_pose_jaw = rec_pose_jaw.reshape(bs * n, 6)
        rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
        rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(
            bs * n, 1 * 3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = self.global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1],
                                                    1 / self.args.pose_fps,
                                                    tar_trans[:, 0, 0:1])
        rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3],
                                                    1 / self.args.pose_fps,
                                                    tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:, :, 1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]
      
        rec_pose_np = rec_pose.detach().cpu().numpy()
        rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
        rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
        base = os.path.basename(audio_path)
        filename = os.path.splitext(base)[0]+".npz"
        gt_npz = np.load("demo/2_scott_0_1_1.npz", allow_pickle=True)
        save_path = os.path.join("./demo", filename)
        np.savez(   save_path,
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
        print("result saved to ", save_path)
    
