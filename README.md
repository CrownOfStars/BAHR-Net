# DAHRNet

Joint salient / camouflaged object detection with VGSformer.

## Setup

```bash
pip install -r requirements.txt
```

Place datasets under `./dataset` (or symlink) and backbone weights under `./pretrained/ckpts`. YAML configs in `pretrained/configs/` are already included.

## Train

```bash
bash scripts/run_joint.sh
```

Or launch distributed training directly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 joint_train.py \
  --backbone convnextv2-base --optim_preset step_adam_simple \
  --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
  --log_path ./log/ --pretrain_size 384 --finetune_size 576
```

## Test

```bash
python joint_test.py
```

## License

MIT
