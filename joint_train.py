
import os
import sys
from tqdm import tqdm
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

#from loader.image.CamoObjDataset import get_camo_loader
from loader.image.JointObjDataset import get_joint_train_loader, get_joint_val_loaders

#from networks.VGSformer import VGSformer
from networks.VGSformer import VGSformer_Joint

from load_config import parse_train_option

from utils.recoder import Recorder
from utils.joint_loss import JointLoss
from utils.optim_presets import get_optimizer_scheduler
from utils.metric import MAE_metric, IoU_metric
from utils.model_utils import defreeze_all, freeze_module, get_allreduce_avg, clip_gradient
from utils.ema import ModelEMA


scaler = GradScaler()
cudnn.benchmark = True
args, config = parse_train_option()
args.nprocs = torch.cuda.device_count()

criterion_joint = JointLoss(
    weight_bce=1.0, weight_iou=1.0, weight_co=0.2,
    weight_iou_sod=0.4, weight_iou_cod=0.6
).cuda()
recoder = Recorder(args, config)



def step_lr_scheduler(scheduler, loss_all):
    """ReduceLROnPlateau 需传入监控指标；StepLR/Cosine 等应无参 step()。"""
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(loss_all)
    else:
        scheduler.step()


def train_one_epoch(train_loader, model, optimizer, scheduler, epoch, local_rank, args, ema=None):
    model.train()

    loss_all = 0
    total_step = len(train_loader)

    try:
        for iter_step, (images, gts, task_ids) in enumerate(train_loader, start=1):
            with autocast():
                optimizer.zero_grad()
                images = images.cuda(local_rank, non_blocking=True)
                gts = gts.cuda(local_rank, non_blocking=True)
                #task_ids = task_ids.cuda(local_rank, non_blocking=True)

                preds= model(images)
                total_loss,_ = criterion_joint(preds[0], gts)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(model, config.TRAIN.CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model.module)

            loss_all = loss_all + total_loss.item()
            if iter_step % 20 == 0 or iter_step == total_step or iter_step == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch+1, args.max_epoch, iter_step, total_step, total_loss.item()))
                recoder.log('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch+1, args.max_epoch, iter_step, total_step, total_loss.item()))


        loss_all /= total_step
        step_lr_scheduler(scheduler, loss_all)
        recoder.log('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch+1, args.max_epoch, loss_all))
        recoder.update_metrics({"epoch": epoch, "loss": loss_all})
        torch.cuda.empty_cache()
    except (KeyboardInterrupt, Exception):
        print('Interrupt: save model and exit.')
        save_model = ema.ema if ema is not None else model
        recoder.save_ckpt(save_model, 'Interrupt.pth')
        print('save checkpoints successfully!')
        raise


# val function
@torch.no_grad()
def val(val_loaders, model, epoch, local_rank, args, ema=None):
    eval_model = ema.ema if ema is not None else model
    eval_model.eval()
    try:
        metric = {}
        for task_name, val_loader in val_loaders.items():
            sum_IoU = 0.0
            sum_mae = 0.0
            local_count = 0
            for _, (images, gts, _) in enumerate(val_loader, start=1):
                gts = gts.cuda(local_rank, non_blocking=True)
                images = images.cuda(local_rank, non_blocking=True)

                res = eval_model(images)
                if isinstance(res, (tuple, list)):
                    res = res[0]
                res = torch.sigmoid(res)

                sum_mae += MAE_metric(res, gts)
                sum_IoU += IoU_metric(res, gts)
                local_count += gts.size(0)

            metric.update(get_allreduce_avg(local_count, {f'{task_name}-miou': sum_IoU, f'{task_name}-mae': sum_mae}))

            print(f"Task:" ,task_name)
            print("MIoU:", metric[f'{task_name}-miou'])
            print("MAE:", metric[f'{task_name}-mae'])
            print("lr:", optimizer.param_groups[0]['lr'])

        metric.update({"epoch": epoch})
        save_model = ema.ema if ema is not None else model
        recoder.update_metrics(metric)


        #print("Best MAE", recoder.best_mae)
        #print("Best Epoch", recoder.best_epoch)
        torch.cuda.empty_cache()
    except (KeyboardInterrupt, Exception):
        print('Interrupt: save model and exit.')
        save_model = ema.ema if ema is not None else model
        recoder.save_ckpt(save_model, 'Interrupt.pth')
        print('save checkpoints successfully!')
        raise


if __name__ == '__main__':
    f = open(os.devnull, "w")
    if args.local_rank != 0:
        sys.stdout = f
        sys.stderr = f

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method='env://')
    model = VGSformer_Joint(config)
    # Barrier to ensure all processes reach cuda operation together
    dist.barrier()

    model = model.cuda(args.local_rank)
    torch.cuda.synchronize(args.local_rank)
    dist.barrier()  # Barrier before DDP

    model = torch.nn.parallel.DistributedDataParallel(model,
                    device_ids=[args.local_rank], find_unused_parameters=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    freeze_module(model, ["encoderR"])
    # 第一阶段不使用 EMA，避免早期大步更新时 EMA 严重滞后
    ema = None
    optimizer, scheduler = get_optimizer_scheduler(model, args, stage=1)
    
    
    print(config.DATA)
    train_loader = get_joint_train_loader(config.DATA, args.pretrain_batch,True)
    val_loaders = get_joint_val_loaders(config.DATA, args.pretrain_batch,True)
    
    for epoch in tqdm(range(args.warmup_epoch)):
        print(train_loader.batch_size)
        train_loader.sampler.set_epoch(epoch)
        
        for val_loader in val_loaders.values():
            val_loader.sampler.set_epoch(epoch)
        
        # train for one epoch
        train_one_epoch(train_loader, model, optimizer, scheduler, epoch, args.local_rank, args, ema)
        val(val_loaders, model, epoch, args.local_rank, args, ema)

    defreeze_all(model)

    # 第二阶段再启用 EMA（可通过 --ema_decay 设为 0 关闭）
    if args.ema_decay > 0:
        ema = ModelEMA(model.module, decay=args.ema_decay)

    
    train_loader = get_joint_train_loader(config.DATA, args.finetune_batch,True)

    optimizer, scheduler = get_optimizer_scheduler(model, args, stage=2, optimizer=optimizer)


    for epoch in tqdm(range(args.warmup_epoch, args.max_epoch)):
        train_loader.sampler.set_epoch(epoch - args.warmup_epoch)
        train_one_epoch(train_loader, model, optimizer, scheduler, epoch, args.local_rank, args, ema)
        val(val_loaders, model, epoch, args.local_rank, args, ema)
    
    log_path = recoder.get_log_path()
    if args.test_now and log_path:
        os.popen("python test.py --test_model {} --gpu_id {}".format(log_path, args.gpu_id))