"""S1-only 联合训练：Stage1+2 从零开始，后训练对齐 v3post 的整网微调。

相对 v3post（joint_train_s1.py）只改两处：
- 从 Best_mae_test.pth 重参起步（不是 Stage2 Last）
- 损失改为 CE + Dice

整网可训，学习率与 v3post 相同（stage2 lr × POST_LR_SCALE，默认 1.5e-5 + StepLR）。
后训练不覆盖 Best_mae_test.pth，分类最好存 Best_cls_miou.pth。

不修改 joint_train.py / joint_train_s1.py / joint_train_s1_cedice.py。
"""

import os
import subprocess
import sys
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from load_config import parse_train_option
from loader.image.JointObjDataset import get_joint_train_loader, get_joint_val_loaders
from networks.VGSformer_joint_s1 import VGSformer_Joint_S1, load_joint_s1_checkpoint
from utils.ema import ModelEMA
from utils.joint_loss import JointLoss
from utils.joint_semantic_gt import joint_twochannel_to_semantic_cls, logits_to_joint_probs
from utils.metric import IoU_metric, MAE_metric
from utils.model_utils import clip_gradient, defreeze_all, freeze_module, get_allreduce_avg
from utils.optim_presets import get_optimizer_scheduler
from utils.recoder import Recorder


scaler = GradScaler()
cudnn.benchmark = True
args, config = parse_train_option()
if args.local_rank < 0:
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
args.nprocs = torch.cuda.device_count()
config.DATA.DATA_ROOT = "./dataset"

criterion_joint = JointLoss(
    weight_bce=1.0,
    weight_iou=1.0,
    weight_co=0.2,
    weight_iou_sod=0.4,
    weight_iou_cod=0.6,
).cuda()
recoder = Recorder(args, config)
best_cls_miou = -1.0


def step_lr_scheduler(scheduler, loss_all):
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(loss_all)
    else:
        scheduler.step()


def _main_logits(preds):
    if isinstance(preds, (tuple, list)):
        return preds[0]
    return preds


def _ckpt_dir():
    if args.tag:
        return os.path.join(args.log_path, args.tag, "ckpt")
    if recoder.is_main:
        return os.path.join(recoder.get_log_path(), "ckpt")
    raise RuntimeError("all ranks need --tag to share Best_mae_test.pth path")


def multiclass_dice_loss(logits, target, eps=1e-6):
    """logits (B,C,H,W), target (B,H,W) long -> 1 - mean Dice over classes."""
    num_classes = logits.shape[1]
    prob = F.softmax(logits.float(), dim=1)
    onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dim=dims)
    card = prob.sum(dim=dims) + onehot.sum(dim=dims)
    dice = (2.0 * inter + eps) / (card + eps)
    return 1.0 - dice.mean()


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
                preds = model(images)
                total_loss, _ = criterion_joint(_main_logits(preds), gts)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(model, config.TRAIN.CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model.module)

            loss_all = loss_all + total_loss.item()
            if iter_step % 20 == 0 or iter_step == total_step or iter_step == 1:
                print(
                    "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}".format(
                        datetime.now(), epoch + 1, args.max_epoch, iter_step, total_step, total_loss.item()
                    )
                )
                recoder.log(
                    "#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}".format(
                        epoch + 1, args.max_epoch, iter_step, total_step, total_loss.item()
                    )
                )

        loss_all /= total_step
        step_lr_scheduler(scheduler, loss_all)
        recoder.log("#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}".format(epoch + 1, args.max_epoch, loss_all))
        recoder.update_metrics({"epoch": epoch, "loss": loss_all})
        torch.cuda.empty_cache()
    except (KeyboardInterrupt, Exception):
        print("Interrupt: save model and exit.")
        save_model = ema.ema if ema is not None else model
        recoder.save_ckpt(save_model, "Interrupt.pth")
        print("save checkpoints successfully!")
        raise


@torch.no_grad()
def val(val_loaders, model, optimizer, epoch, local_rank, args, ema=None):
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
                res = _main_logits(eval_model(images))
                res = torch.sigmoid(res)
                sum_mae += MAE_metric(res, gts)
                sum_IoU += IoU_metric(res, gts)
                local_count += gts.size(0)

            metric.update(
                get_allreduce_avg(local_count, {f"{task_name}-miou": sum_IoU, f"{task_name}-mae": sum_mae})
            )
            print(f"Task: {task_name}")
            print("MIoU:", metric[f"{task_name}-miou"])
            print("MAE:", metric[f"{task_name}-mae"])
            print("lr:", optimizer.param_groups[0]["lr"])

        metric.update({"epoch": epoch})
        mae_keys = [k for k in metric if k.endswith("-mae")]
        if mae_keys:
            metric["mae"] = sum(metric[k] for k in mae_keys) / len(mae_keys)
        save_model = ema.ema if ema is not None else model
        recoder.save_ckpt(save_model, f"Epoch_{epoch:03d}.pth")
        recoder.save_ckpt(save_model, "Last.pth")
        if recoder.update_metrics(metric):
            recoder.save_ckpt(save_model, "Best_mae_test.pth")
        torch.cuda.empty_cache()
    except (KeyboardInterrupt, Exception):
        print("Interrupt: save model and exit.")
        save_model = ema.ema if ema is not None else model
        recoder.save_ckpt(save_model, "Interrupt.pth")
        print("save checkpoints successfully!")
        raise


def train_one_epoch_cls(train_loader, model, optimizer, scheduler, epoch, local_rank, args, ema=None):
    """第三阶段：整网 CE + Dice。"""
    model.train()
    ce_w = float(os.environ.get("CE_WEIGHT", "1.0"))
    dice_w = float(os.environ.get("DICE_WEIGHT", "1.0"))
    loss_all = 0
    ce_all = 0
    dice_all = 0
    total_step = len(train_loader)
    try:
        for iter_step, (images, gts, _) in enumerate(train_loader, start=1):
            with autocast():
                optimizer.zero_grad()
                images = images.cuda(local_rank, non_blocking=True)
                gts = gts.cuda(local_rank, non_blocking=True)
                logits = _main_logits(model(images))
                cls_target = joint_twochannel_to_semantic_cls(gts)
                if logits.shape[-2:] != cls_target.shape[-2:]:
                    logits = F.interpolate(logits, cls_target.shape[-2:], mode="bilinear", align_corners=False)
                loss_ce = F.cross_entropy(logits, cls_target)
                loss_dice = multiclass_dice_loss(logits, cls_target)
                total_loss = ce_w * loss_ce + dice_w * loss_dice

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(model, config.TRAIN.CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model.module)

            loss_all += total_loss.item()
            ce_all += loss_ce.item()
            dice_all += loss_dice.item()
            if iter_step % 20 == 0 or iter_step == total_step or iter_step == 1:
                print(
                    "{} POST Epoch [{:03d}], Step [{:04d}/{:04d}], CE: {:.4f} Dice: {:.4f} Total: {:.4f}".format(
                        datetime.now(),
                        epoch + 1,
                        iter_step,
                        total_step,
                        loss_ce.item(),
                        loss_dice.item(),
                        total_loss.item(),
                    )
                )
                recoder.log(
                    "#POST#:Epoch [{:03d}], Step [{:04d}/{:04d}], CE: {:.4f} Dice: {:.4f} Total: {:.4f}".format(
                        epoch + 1, iter_step, total_step, loss_ce.item(), loss_dice.item(), total_loss.item()
                    )
                )

        loss_all /= total_step
        ce_all /= total_step
        dice_all /= total_step
        step_lr_scheduler(scheduler, loss_all)
        recoder.log(
            "#POST#:Epoch [{:03d}], CE_AVG: {:.4f} Dice_AVG: {:.4f} Total_AVG: {:.4f} lr={:.2e}".format(
                epoch + 1, ce_all, dice_all, loss_all, optimizer.param_groups[0]["lr"]
            )
        )
        recoder.update_metrics({"epoch": epoch, "loss": loss_all, "ce": ce_all, "dice": dice_all})
        torch.cuda.empty_cache()
    except (KeyboardInterrupt, Exception):
        print("Interrupt: save model and exit.")
        save_model = ema.ema if ema is not None else model
        recoder.save_ckpt(save_model, "Interrupt.pth")
        print("save checkpoints successfully!")
        raise


@torch.no_grad()
def val_cls(val_loaders, model, optimizer, epoch, local_rank, args, ema=None):
    """后训练验证：ClsAcc/ClsMIoU；不覆盖 Stage2 的 Best_mae_test.pth。"""
    global best_cls_miou
    eval_model = ema.ema if ema is not None else model
    eval_model.eval()
    try:
        metric = {}
        for task_name, val_loader in val_loaders.items():
            sum_IoU = 0.0
            sum_mae = 0.0
            local_count = 0
            correct = torch.zeros((), device=f"cuda:{local_rank}")
            pixels = torch.zeros((), device=f"cuda:{local_rank}")
            inter = torch.zeros(3, device=f"cuda:{local_rank}")
            union = torch.zeros(3, device=f"cuda:{local_rank}")
            for _, (images, gts, _) in enumerate(val_loader, start=1):
                gts = gts.cuda(local_rank, non_blocking=True)
                images = images.cuda(local_rank, non_blocking=True)
                logits = _main_logits(eval_model(images))
                cls_target = joint_twochannel_to_semantic_cls(gts)
                if logits.shape[-2:] != cls_target.shape[-2:]:
                    logits = F.interpolate(logits, cls_target.shape[-2:], mode="bilinear", align_corners=False)
                pred = logits.argmax(dim=1)
                correct += (pred == cls_target).sum()
                pixels += cls_target.numel()
                for c in range(3):
                    p = pred == c
                    t = cls_target == c
                    inter[c] += (p & t).sum()
                    union[c] += (p | t).sum()

                probs = logits_to_joint_probs(logits, gts.shape[-2:])
                sum_mae += MAE_metric(probs, gts)
                sum_IoU += IoU_metric(probs, gts)
                local_count += gts.size(0)

            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(pixels, op=dist.ReduceOp.SUM)
            dist.all_reduce(inter, op=dist.ReduceOp.SUM)
            dist.all_reduce(union, op=dist.ReduceOp.SUM)
            cls_acc = (correct / (pixels + 1e-6)).item()
            cls_iou = (inter / (union + 1e-6)).cpu()
            cls_miou = cls_iou.mean().item()

            metric.update(
                get_allreduce_avg(local_count, {f"{task_name}-miou": sum_IoU, f"{task_name}-mae": sum_mae})
            )
            metric[f"{task_name}-cls_acc"] = cls_acc
            metric[f"{task_name}-cls_miou"] = cls_miou
            print(f"Task: {task_name}")
            print("MIoU:", metric[f"{task_name}-miou"], "MAE:", metric[f"{task_name}-mae"])
            print("ClsAcc:", cls_acc, "ClsMIoU:", cls_miou, "per-class IoU:", cls_iou.tolist())
            print("lr:", optimizer.param_groups[0]["lr"])

        metric.update({"epoch": epoch})
        mae_keys = [k for k in metric if k.endswith("-mae") and not k.endswith("cls_mae")]
        if mae_keys:
            metric["mae"] = sum(metric[k] for k in mae_keys) / len(mae_keys)
        acc_keys = [k for k in metric if k.endswith("-cls_acc")]
        if acc_keys:
            metric["cls_acc"] = sum(metric[k] for k in acc_keys) / len(acc_keys)
        miou_keys = [k for k in metric if k.endswith("-cls_miou")]
        if miou_keys:
            metric["cls_miou"] = sum(metric[k] for k in miou_keys) / len(miou_keys)
        save_model = ema.ema if ema is not None else model
        recoder.save_ckpt(save_model, f"Epoch_{epoch:03d}.pth")
        recoder.save_ckpt(save_model, "Last.pth")
        recoder.update_metrics(metric)
        if recoder.is_main and metric.get("cls_miou", -1.0) > best_cls_miou:
            best_cls_miou = metric["cls_miou"]
            recoder.save_ckpt(save_model, "Best_cls_miou.pth")
            recoder.log(
                "#POST#:Epoch:{} cls_miou={} best_cls_miou={} (saved Best_cls_miou.pth)".format(
                    epoch, metric["cls_miou"], best_cls_miou
                )
            )
        torch.cuda.empty_cache()
    except (KeyboardInterrupt, Exception):
        print("Interrupt: save model and exit.")
        save_model = ema.ema if ema is not None else model
        recoder.save_ckpt(save_model, "Interrupt.pth")
        print("save checkpoints successfully!")
        raise


def wrap_ddp(raw_model, local_rank):
    raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
    return torch.nn.parallel.DistributedDataParallel(
        raw_model, device_ids=[local_rank], find_unused_parameters=True
    )


def setup_post_from_best_mae(model, local_rank):
    """加载 Stage2 Best_mae_test.pth，重参三分类头，整网微调（与 v3post 相同）。"""
    dist.barrier()
    best_path = os.path.join(_ckpt_dir(), "Best_mae_test.pth")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"Best_mae_test.pth not found: {best_path}")
    sd = torch.load(best_path, map_location=f"cuda:{local_rank}")
    raw = model.module
    msg = load_joint_s1_checkpoint(raw, sd)
    raw.reparam_to_softmax_head()
    defreeze_all(raw)
    model = wrap_ddp(raw, local_rank)

    optimizer, scheduler = get_optimizer_scheduler(model, args, stage=2)
    post_lr_scale = float(os.environ.get("POST_LR_SCALE", "0.5"))
    for g in optimizer.param_groups:
        g["lr"] *= post_lr_scale
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    info = (
        f"post from {best_path}; full-net CE+Dice; lr_scale={post_lr_scale}, "
        f"lr={optimizer.param_groups[0]['lr']}; "
        f"trainable {n_train / 1e6:.2f}M / {n_all / 1e6:.2f}M; load={msg}"
    )
    print(info)
    recoder.log("#POST#: " + info)
    return model, optimizer, scheduler


if __name__ == "__main__":
    f = open(os.devnull, "w")
    if args.local_rank != 0:
        sys.stdout = f
        sys.stderr = f

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    model = VGSformer_Joint_S1(config)
    dist.barrier()

    model = model.cuda(args.local_rank)
    torch.cuda.synchronize(args.local_rank)
    dist.barrier()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], find_unused_parameters=True
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    freeze_module(model, ["encoderR"])
    ema = None
    optimizer, scheduler = get_optimizer_scheduler(model, args, stage=1)

    print(config.DATA)
    train_loader = get_joint_train_loader(config.DATA, args.pretrain_batch, True)
    val_loaders = get_joint_val_loaders(config.DATA, args.pretrain_batch, True)

    for epoch in tqdm(range(args.warmup_epoch)):
        print(train_loader.batch_size)
        train_loader.sampler.set_epoch(epoch)
        for val_loader in val_loaders.values():
            val_loader.sampler.set_epoch(epoch)
        train_one_epoch(train_loader, model, optimizer, scheduler, epoch, args.local_rank, args, ema)
        val(val_loaders, model, optimizer, epoch, args.local_rank, args, ema)

    defreeze_all(model)
    if args.ema_decay > 0:
        ema = ModelEMA(model.module, decay=args.ema_decay)

    train_loader = get_joint_train_loader(config.DATA, args.finetune_batch, True)
    optimizer, scheduler = get_optimizer_scheduler(model, args, stage=2, optimizer=optimizer)

    for epoch in tqdm(range(args.warmup_epoch, args.max_epoch)):
        train_loader.sampler.set_epoch(epoch - args.warmup_epoch)
        train_one_epoch(train_loader, model, optimizer, scheduler, epoch, args.local_rank, args, ema)
        val(val_loaders, model, optimizer, epoch, args.local_rank, args, ema)

    save_model = ema.ema if ema is not None else model
    recoder.save_ckpt(save_model, "Stage2_Last.pth")
    recoder.save_ckpt(save_model, "Last.pth")
    print("stage1+2 done, saved Stage2_Last.pth and Best_mae_test.pth")

    post_epochs = int(os.environ.get("POST_EPOCHS", "40"))
    if post_epochs <= 0:
        print("POST_EPOCHS=0, skip softmax post-training")
    else:
        model, optimizer, scheduler = setup_post_from_best_mae(model, args.local_rank)
        ema = None
        for epoch in tqdm(range(args.max_epoch, args.max_epoch + post_epochs)):
            train_loader.sampler.set_epoch(epoch)
            train_one_epoch_cls(train_loader, model, optimizer, scheduler, epoch, args.local_rank, args, ema)
            val_cls(val_loaders, model, optimizer, epoch, args.local_rank, args, ema)
        recoder.save_ckpt(model, "Last.pth")
        print("post-training done, saved Last.pth / Best_cls_miou.pth (3-class head)")

    dist.barrier()
    log_path = recoder.get_log_path() if recoder.is_main else None
    del model, optimizer, scheduler, train_loader, val_loaders
    if ema is not None:
        del ema
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()

    skip_test = os.environ.get("SKIP_TEST", "0").lower() in ("1", "true", "yes")
    if recoder.is_main and log_path and not skip_test:
        test_gpu = os.environ.get("TEST_GPU")
        if not test_gpu:
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            test_gpu = vis.split(",")[0].strip() or "0"
        cls_ckpt = os.path.join(log_path, "ckpt", "Best_cls_miou.pth")
        test_ckpt = "Best_cls_miou.pth" if os.path.isfile(cls_ckpt) else "Last.pth"
        cmd = [
            sys.executable,
            "joint_ppc_test_s1.py",
            "--test_model",
            log_path,
            "--ckpt",
            test_ckpt,
            "--gpu_id",
            "0",
            "--no_metrics",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = test_gpu
        print("run test:", " ".join(cmd), "CUDA_VISIBLE_DEVICES=" + test_gpu)
        recoder.log("run test: " + " ".join(cmd))
        ret = subprocess.run(cmd, cwd=os.getcwd(), env=env)
        if ret.returncode != 0:
            raise SystemExit(f"joint_ppc_test_s1.py failed with code {ret.returncode}")
        print("test done, maps saved to", os.path.join(log_path, "save"))
    elif recoder.is_main and skip_test:
        print("SKIP_TEST=1, skip joint_ppc_test_s1.py")
