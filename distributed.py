
import os
import sys
from tqdm import tqdm
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from loader.image.CamoObjDataset import get_camo_loader

from networks.VGSformer import VGSformer
from load_config import parse_train_option

from utils.recoder import Recorder
from utils.loss_presets import get_loss_fn
from utils.optim_presets import get_optimizer_scheduler
from utils.metric import MAE_metric, IoU_metric
from utils.model_utils import defreeze_all, freeze_module, get_allreduce_avg, clip_gradient
from utils.ema import ModelEMA


scaler = GradScaler()
cudnn.benchmark = True
args, config = parse_train_option()
if args.local_rank < 0:
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
args.nprocs = torch.cuda.device_count()

loss_fn = get_loss_fn(args)
recoder = Recorder(args, config)


def train_one_epoch(train_loader, model, optimizer, scheduler, epoch, local_rank, args, ema=None):
    model.train()

    loss_all = 0
    total_step = len(train_loader)

    try:
        for iter_step, (images, gts, bounds) in enumerate(train_loader, start=1):
            with autocast():
                optimizer.zero_grad()
                images = images.cuda(local_rank, non_blocking=True)
                gts = gts.cuda(local_rank, non_blocking=True)
                bounds = bounds.cuda(local_rank, non_blocking=True)

                s1, s2, s3, s4, edge1, edge2, _ = model(images)
                loss = loss_fn(s1, s2, s3, s4, gts, edge1, edge2, bounds)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(model, config.TRAIN.CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model.module)

            loss_all = loss_all + loss.item()
            if iter_step % 20 == 0 or iter_step == total_step or iter_step == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch+1, args.max_epoch, iter_step, total_step, loss.item()))
                recoder.log('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch+1, args.max_epoch, iter_step, total_step, loss.item()))

        scheduler.step()
        loss_all /= total_step
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
def val(val_loader, model, optimizer, epoch, local_rank, args, ema=None):
    eval_model = ema.ema if ema is not None else model
    eval_model.eval()
    try:
        sum_IoU = 0.0
        sum_mae = 0.0
        local_count = 0

        for _, (images, gts) in enumerate(val_loader, start=1):
            gts = gts.cuda(local_rank, non_blocking=True)
            images = images.cuda(local_rank, non_blocking=True)

            res = eval_model(images)
            res = torch.sigmoid(res[0])

            sum_mae += MAE_metric(res, gts)
            sum_IoU += IoU_metric(res, gts)
            local_count += gts.size(0)

        metric = get_allreduce_avg(local_count, {'miou': sum_IoU, 'mae': sum_mae})

        print("MIoU:", metric['miou'])
        print("MAE:", metric['mae'])
        print("lr:", optimizer.param_groups[0]['lr'])

        metric.update({"epoch": epoch})
        save_model = ema.ema if ema is not None else model
        if recoder.update_metrics(metric):
            print(f"update best epoch to epoch {epoch}")
            recoder.save_ckpt(save_model, "Best_mae_test.pth")

        print("Best MAE", recoder.best_mae)
        print("Best Epoch", recoder.best_epoch)
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
    model = VGSformer(config)
    # Barrier to ensure all processes reach cuda operation together
    dist.barrier()

    model = model.cuda(args.local_rank)
    torch.cuda.synchronize(args.local_rank)
    dist.barrier()  # Barrier before DDP

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model,
                    device_ids=[args.local_rank], find_unused_parameters=True)
    
    freeze_module(model, ["encoderR"])
    # 第一阶段不使用 EMA，避免早期大步更新时 EMA 严重滞后
    ema = None
    optimizer, scheduler = get_optimizer_scheduler(model, args, stage=1)
    train_loader = get_camo_loader(config.DATA, args.pretrain_batch,True,"pretrain",args.texture)
    val_loader = get_camo_loader(config.DATA, args.pretrain_batch,True,"val",args.texture)
    print(config)
    for epoch in tqdm(range(args.warmup_epoch)):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        
        # train for one epoch
        train_one_epoch(train_loader, model, optimizer, scheduler, epoch, args.local_rank, args, ema)
        val(val_loader, model, optimizer, epoch, args.local_rank, args, ema)
    
        
    defreeze_all(model)

    # 第二阶段再启用 EMA（可通过 --ema_decay 设为 0 关闭）
    if args.ema_decay > 0:
        ema = ModelEMA(model.module, decay=args.ema_decay)

    
    train_loader = get_camo_loader(config.DATA, args.finetune_batch,True,"finetune",args.texture)

    optimizer, scheduler = get_optimizer_scheduler(model, args, stage=2, optimizer=optimizer)


    for epoch in tqdm(range(args.warmup_epoch, args.max_epoch)):
        train_loader.sampler.set_epoch(epoch - args.warmup_epoch)
        train_one_epoch(train_loader, model, optimizer, scheduler, epoch, args.local_rank, args, ema)
        val(val_loader, model, optimizer, epoch, args.local_rank, args, ema)
    
    log_path = recoder.get_log_path()
    if args.test_now and log_path:
        os.popen("python test.py --test_model {} --gpu_id {}".format(log_path, args.gpu_id))