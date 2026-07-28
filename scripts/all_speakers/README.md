# All-Speaker Training

The original EchoMask release and `configs/echomask.yaml` remain the
Speaker 2 paper protocol. This directory adds a separate 25-speaker workflow
without overwriting the original data, weights, or outputs.

The All-Speaker protocol uses the available English BEAT2 speakers:

```text
1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18,
20, 21, 22, 23, 24, 25, 27, 28, 30
```

## Released layout

Install the separate All-Speaker archive from the repository root:

```shell
unzip EchoMask_all_speakers_weights_25spk.zip
cp -R EchoMask_all_speakers_weights/weights/all_speakers ./weights/
```

The resulting files are organized as:

```text
weights/all_speakers/
├── pretrained_vq/
│   ├── rvq_face_600.bin
│   ├── rvq_hands_500.bin
│   ├── rvq_upper_500.bin
│   ├── rvq_lower_600.bin
│   └── last_1700_foot.bin
└── best_echomask.bin
```

The five motion-representation checkpoints are the same All-Speaker
RVQ/VAE suite released for SemTalk. EchoMask reuses those exact checkpoints;
they do not need to be trained again.

The All-Speaker archive complements the original EchoMask asset archive.
Keep the original evaluator, projection matrix, and SMPL-X assets in
`weights/`; the All-Speaker files are installed in their own subdirectory.

## Dataset

Prepare the 25-speaker EchoMask caches with:

```shell
scripts/all_speakers/run.sh prepare-train
scripts/all_speakers/run.sh prepare-test
```

By default, this writes to `datasets/all_speakers`. The preparation step
requires BEAT2, HuBERT, `weights/rop_1024_256.pt`, and the downloaded
All-Speaker representation checkpoints.

The following environment variables override the default paths:

```text
ECHOMASK_DATA_ROOT
ECHOMASK_HUBERT_ROOT
ECHOMASK_ALL_SPEAKER_DATASET_ROOT
ECHOMASK_ALL_SPEAKER_WEIGHT_ROOT
ECHOMASK_ALL_SPEAKER_OUTPUT_ROOT
ECHOMASK_EVAL_EPOCH
```

## Training

Train one seed with:

```shell
ECHOMASK_SEED=43 \
ECHOMASK_NPROC_PER_NODE=8 \
ECHOMASK_BATCH_SIZE=256 \
scripts/all_speakers/run.sh train
```

The released search trained seeds 42, 43, and 44 independently. Each run used
400 epochs and full-state checkpoints every 10 epochs. All coarse candidates
were evaluated, then training resumed from the checkpoint immediately before
each seed's lowest-FGD coarse epoch. Every epoch in that local window was
evaluated, and the global lowest-FGD checkpoint across all three seeds was
released.

`ECHOMASK_BATCH_SIZE=256` is the released per-GPU H200 setting. Reduce it for
GPUs with less memory.

## Evaluation

Evaluate the released All-Speaker checkpoint and write the paired NPZ files:

```shell
scripts/all_speakers/run.sh test
```

The All-Speaker archive and the original Speaker 2 archive use separate
directories. Select the checkpoint suite that matches the protocol being
evaluated.
