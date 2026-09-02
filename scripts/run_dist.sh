#!/usr/bin/env bash
# tmux 夜间串行跑:
#   cd /home/data1/ShiqiangShu/best/HRMNetv3_1
#   bash scripts/run_dist.sh 2>&1 | tee log/run_dist.log

cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True

GPUS=0,1,2,3
NPROC=4

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$NPROC --master_port=29501 distributed.py \
  --backbone convnextv2-base --optim_preset step_adam_simple \
  --pretrain_batch 56 --finetune_batch 8 --mfusion LSF \
  --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task ISOD

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$NPROC --master_port=29502 distributed.py \
  --backbone convnextv2-base --optim_preset step_adam_simple \
  --pretrain_batch 56 --finetune_batch 8 --mfusion LSF \
  --log_path ./log/ --pretrain_size 384 --finetune_size 384 --task ISOD

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$NPROC --master_port=29503 distributed.py \
  --backbone convnextv2-base --optim_preset step_adam_simple \
  --pretrain_batch 56 --finetune_batch 8 --mfusion LSF \
  --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task COD

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$NPROC --master_port=29504 distributed.py \
  --backbone convnextv2-base --optim_preset step_adam_simple \
  --pretrain_batch 56 --finetune_batch 8 --mfusion LSF \
  --log_path ./log/ --pretrain_size 384 --finetune_size 384 --task COD
