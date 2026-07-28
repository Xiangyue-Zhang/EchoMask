#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

SPEAKERS=(1 2 3 4 5 6 7 9 10 11 12 13 15 16 17 18 20 21 22 23 24 25 27 28 30)
DATA_ROOT=${ECHOMASK_DATA_ROOT:-./BEAT2/beat_english_v2.0.0}
DATASET_ROOT=${ECHOMASK_ALL_SPEAKER_DATASET_ROOT:-./datasets/all_speakers}
WEIGHT_ROOT=${ECHOMASK_ALL_SPEAKER_WEIGHT_ROOT:-./weights/all_speakers}
OUTPUT_ROOT=${ECHOMASK_ALL_SPEAKER_OUTPUT_ROOT:-./outputs/all_speakers}
HUBERT_ROOT=${ECHOMASK_HUBERT_ROOT:-./facebook/hubert-large-ls960-ft}
ROP_PATH=${ECHOMASK_ROP_PATH:-./weights/rop_1024_256.pt}
PYTHON=${PYTHON:-python}
NPROC=${ECHOMASK_NPROC_PER_NODE:-1}
BATCH_SIZE=${ECHOMASK_BATCH_SIZE:-64}
SEED=${ECHOMASK_SEED:-43}
EVAL_EPOCH=${ECHOMASK_EVAL_EPOCH:-380}

GPUS=()
for ((gpu = 0; gpu < NPROC; gpu++)); do
  GPUS+=("$gpu")
done

usage() {
  printf '%s\n' \
    "Usage: scripts/all_speakers/run.sh COMMAND" \
    "" \
    "Commands:" \
    "  prepare-train   Build the 25-speaker training LMDB" \
    "  prepare-test    Build the 25-speaker test pickle" \
    "  train           Train EchoMask on all 25 speakers" \
    "  test            Evaluate the released all-speaker checkpoint"
}

command=${1:-}
case "$command" in
  prepare-train)
    mkdir -p "$DATASET_ROOT"
    "$PYTHON" dataloaders/save_train_dataset.py \
      --training_speakers "${SPEAKERS[@]}" \
      --data_path "$DATA_ROOT/" \
      --cache_path "$DATASET_ROOT/cache" \
      --hubert_path "$HUBERT_ROOT" \
      --pretrained_vq_dir "$WEIGHT_ROOT/pretrained_vq" \
      --rop_path "$ROP_PATH" \
      --dst_lmdb "$DATASET_ROOT/train"
    ;;
  prepare-test)
    mkdir -p "$DATASET_ROOT"
    "$PYTHON" dataloaders/save_test_dataset.py \
      --training_speakers "${SPEAKERS[@]}" \
      --data_path "$DATA_ROOT/" \
      --cache_path "$DATASET_ROOT/cache" \
      --hubert_path "$HUBERT_ROOT" \
      --pretrained_vq_dir "$WEIGHT_ROOT/pretrained_vq" \
      --rop_path "$ROP_PATH" \
      --dst_pkl "$DATASET_ROOT/test.pkl"
    ;;
  train)
    mkdir -p "$OUTPUT_ROOT/seed-$SEED"
    "$PYTHON" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$NPROC" \
      train.py \
      --config configs/echomask_all_speakers.yaml \
      --ddp true \
      --gpus "${GPUS[@]}" \
      --training_speakers "${SPEAKERS[@]}" \
      --data_path "$DATA_ROOT/" \
      --data_path_1 ./weights/ \
      --pretrained_vq_dir "$WEIGHT_ROOT/pretrained_vq" \
      --train_path "$DATASET_ROOT/train" \
      --test_path "$DATASET_ROOT/test.pkl" \
      --out_path "$OUTPUT_ROOT/seed-$SEED/" \
      --notes "_all_speakers_seed_$SEED" \
      --random_seed "$SEED" \
      --batch_size "$BATCH_SIZE" \
      --eval_during_train false \
      --checkpoint_period 10 \
      --checkpoint_training_state true \
      --tf32 true
    ;;
  test)
    mkdir -p "$OUTPUT_ROOT/test"
    "$PYTHON" train.py \
      --config configs/echomask_all_speakers.yaml \
      --ddp false \
      --gpus 0 \
      --training_speakers "${SPEAKERS[@]}" \
      --data_path "$DATA_ROOT/" \
      --data_path_1 ./weights/ \
      --pretrained_vq_dir "$WEIGHT_ROOT/pretrained_vq" \
      --train_path "$DATASET_ROOT/train" \
      --test_path "$DATASET_ROOT/test.pkl" \
      --load_ckpt "$WEIGHT_ROOT/best_echomask.bin" \
      --eval_epoch "$EVAL_EPOCH" \
      --out_path "$OUTPUT_ROOT/test/" \
      --notes "_all_speakers" \
      --test_state \
      --save_eval_npz true
    ;;
  *)
    usage
    exit 2
    ;;
esac
