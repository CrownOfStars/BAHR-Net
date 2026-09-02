"""
优化器 + 学习率调度预设，类似 loss_presets。
用法: optimizer, scheduler = get_optimizer_scheduler(model, args, stage=1)
"""
import torch.optim as optim
from torch.optim import lr_scheduler

# StepLR + Adam 推荐配置（细化在 preset 内）
# encoder_early_lr_ratio: 前两 stage（纹理/边缘）学习率比例
# encoder_late_lr_ratio: 后两 stage（语义）学习率比例，相对更大
STEP_ADAM_CONFIG = {
    "stage1": {
        "lr": 3e-4,
        "decay_epoch": 10,
        "encoder_early_lr_ratio": 0.5,
        "encoder_late_lr_ratio": 1.0,
    },
    "stage2": {
        "lr": 3e-5,
        "decay_epoch": 10,
        "encoder_early_lr_ratio": 0.5,
        "encoder_late_lr_ratio": 1.0,
    },
    "gamma": 0.5,
}

# StepLR + Adam 简单版（无分层学习率，统一 lr）
STEP_ADAM_SIMPLE_CONFIG = {
    "stage1": {"lr": 3e-4, "decay_epoch": 10},
    "stage2": {"lr": 3e-5, "decay_epoch": 10},
    "gamma": 0.5,
}

PLATEAU_ADAM_SIMPLE_CONFIG = {
    "gamma": 0.6,      # 每次衰减的比例 (new_lr = old_lr * gamma)
    "mode": "min",      # 监控 Loss 下降
    "stage1": {"lr": 3e-4, "decay_epoch": 10},
    "stage2": {
        "lr": 3e-5,
        "patience": 10
    }
}

# Cosine + AdamW 推荐配置（细化在 preset 内，无需通过 argparse 单独调）
COSINE_ADAMW_CONFIG = {
    "stage1": {
        "lr": 5e-4,
        "head_wd": 0.01,
        "backbone_lr_ratio": 1.0,
        "backbone_wd": 0.01,
    },
    "stage2": {
        "lr": 2e-4,
        "head_wd": 0.01,
        "backbone_wd": 0.005,
        "backbone_lr_ratio": 0.1,
    },
    "cosine_eta_min": 1e-6,
    "no_decay_norm_bias": True,
}


def build_param_groups(model, base_lr, head_wd, backbone_lr_ratio=1.0, backbone_wd=None, no_decay_norm_bias=True):
    """分层参数组：backbone / head，decay / no_decay（norm、bias）。"""
    if backbone_wd is None:
        backbone_wd = head_wd

    head_decay, head_no_decay = [], []
    backbone_decay, backbone_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "encoderR" in name
        no_decay = no_decay_norm_bias and (param.ndim == 1 or name.endswith(".bias"))

        if is_backbone:
            if no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        else:
            if no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    groups = []
    if head_decay:
        groups.append({"params": head_decay, "lr": base_lr, "weight_decay": head_wd})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0})
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": base_lr * backbone_lr_ratio, "weight_decay": backbone_wd})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": base_lr * backbone_lr_ratio, "weight_decay": 0.0})

    return groups


def _update_optimizer_lr(optimizer, lr, head_wd=None, backbone_lr_ratio=1.0, backbone_wd=None):
    """在已有优化器上更新 param_groups 的 lr 和 weight_decay，保留 Adam/AdamW 的动量状态。"""
    if backbone_wd is None:
        backbone_wd = head_wd
    n = len(optimizer.param_groups)
    if n == 1:
        optimizer.param_groups[0]["lr"] = lr
        if head_wd is not None:
            optimizer.param_groups[0]["weight_decay"] = head_wd
    else:
        # 约定: [head_decay, head_no_decay, backbone_decay, backbone_no_decay]
        for i, g in enumerate(optimizer.param_groups):
            if i == 0:
                g["lr"] = lr
                g["weight_decay"] = head_wd if head_wd is not None else g.get("weight_decay", 0)
            elif i == 1:
                g["lr"] = lr
                g["weight_decay"] = 0.0
            elif i == 2:
                g["lr"] = lr * backbone_lr_ratio
                g["weight_decay"] = backbone_wd if backbone_wd is not None else g.get("weight_decay", 0)
            else:
                g["lr"] = lr * backbone_lr_ratio
                g["weight_decay"] = 0.0


def get_optimizer_scheduler(model, args, stage, optimizer=None):
    """
    根据 args.optim_preset 返回 (optimizer, scheduler)。
    stage: 1=pretrain, 2=finetune
    optimizer: 若 stage==2 且传入已有优化器，则仅更新 lr 并返回新 scheduler，不重新实例化（保留 Adam 动量状态）。
    """
    preset = getattr(args, "optim_preset", "step_adam")

    if preset == "step_adam":
        cfg = STEP_ADAM_CONFIG
        gamma = cfg["gamma"]
        if stage == 1:
            s1 = cfg["stage1"]
            optimizer = optim.Adam(model.parameters(), lr=s1["lr"])
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=s1["decay_epoch"], gamma=gamma
            )
        else:
            s2 = cfg["stage2"]
            if optimizer is not None:
                _update_optimizer_lr(optimizer, s2["lr"])
                scheduler = lr_scheduler.StepLR(
                    optimizer, step_size=s2["decay_epoch"], gamma=gamma
                )
            else:
                optimizer = optim.Adam(model.parameters(), lr=s2["lr"])
                scheduler = lr_scheduler.StepLR(
                    optimizer, step_size=s2["decay_epoch"], gamma=gamma
                )

    elif preset == "step_adam_simple":
        cfg = STEP_ADAM_SIMPLE_CONFIG
        gamma = cfg["gamma"]
        if stage == 1:
            s1 = cfg["stage1"]
            optimizer = optim.Adam(model.parameters(), lr=s1["lr"])
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=s1["decay_epoch"], gamma=gamma
            )
        else:
            s2 = cfg["stage2"]
            if optimizer is not None:
                _update_optimizer_lr(optimizer, s2["lr"])
                scheduler = lr_scheduler.StepLR(
                    optimizer, step_size=s2["decay_epoch"], gamma=gamma
                )
            else:
                optimizer = optim.Adam(model.parameters(), lr=s2["lr"])
                scheduler = lr_scheduler.StepLR(
                    optimizer, step_size=s2["decay_epoch"], gamma=gamma
                )
                
    elif preset == "plateau_adam_simple":
        # 假设你的配置文件名为 PLATEAU_ADAM_SIMPLE_CONFIG
        cfg = PLATEAU_ADAM_SIMPLE_CONFIG
        
        # 获取 ReduceLROnPlateau 的全局参数，若没有则设置默认值
        gamma = cfg.get("gamma", 0.6)  # 等同于之前的 gamma
        mode = cfg.get("mode", "min")    # 监控指标模式，通常 'min' 用于监控 Loss
        
        if stage == 1:
            s1 = cfg["stage1"]
            optimizer = optim.Adam(model.parameters(), lr=s1["lr"])
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=s1["decay_epoch"], gamma=gamma
            )
        else:
            s2 = cfg["stage2"]
            if optimizer is not None:
                _update_optimizer_lr(optimizer, s2["lr"])
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode=mode, 
                    factor=gamma, 
                    patience=s2["patience"]
                )
            else:
                optimizer = optim.Adam(model.parameters(), lr=s2["lr"])
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode=mode, 
                    factor=gamma, 
                    patience=s2["patience"]
                )

    elif preset == "cosine_adamw":
        cfg = COSINE_ADAMW_CONFIG
        eta_min = cfg["cosine_eta_min"]
        no_decay = cfg["no_decay_norm_bias"]
        if stage == 1:
            s1 = cfg["stage1"]
            groups = build_param_groups(
                model=model,
                base_lr=s1["lr"],
                head_wd=s1["head_wd"],
                backbone_lr_ratio=s1["backbone_lr_ratio"],
                backbone_wd=s1["backbone_wd"],
                no_decay_norm_bias=no_decay,
            )
            optimizer = optim.AdamW(params=groups, lr=s1["lr"], weight_decay=0.0)
            T_max = args.warmup_epoch
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        else:
            s2 = cfg["stage2"]
            if optimizer is not None:
                _update_optimizer_lr(
                    optimizer,
                    s2["lr"],
                    head_wd=s2["head_wd"],
                    backbone_lr_ratio=s2["backbone_lr_ratio"],
                    backbone_wd=s2["backbone_wd"],
                )
                T_max = args.max_epoch - args.warmup_epoch
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T_max, eta_min=eta_min
                )
            else:
                groups = build_param_groups(
                    model=model,
                    base_lr=s2["lr"],
                    head_wd=s2["head_wd"],
                    backbone_lr_ratio=s2["backbone_lr_ratio"],
                    backbone_wd=s2["backbone_wd"],
                    no_decay_norm_bias=no_decay,
                )
                optimizer = optim.AdamW(params=groups, lr=s2["lr"], weight_decay=0.0)
                T_max = args.max_epoch - args.warmup_epoch
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T_max, eta_min=eta_min
                )

    else:
        raise ValueError(f"Unknown optim_preset: {preset}")

    return optimizer, scheduler
