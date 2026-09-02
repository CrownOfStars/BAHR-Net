# CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4 joint_train.py \
#    --backbone convnextv2-base --optim_preset step_adam_simple \
#    --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
#    --log_path ./log/ --pretrain_size 384 --finetune_size 576


CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4 joint_train.py \
   --backbone convnextv2-base --optim_preset step_adam_simple \
   --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
   --log_path ./log/ --pretrain_size 384 --finetune_size 576