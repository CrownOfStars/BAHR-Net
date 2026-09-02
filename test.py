import os
import time
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
# 自动禁用非终端环境下的 tqdm
disable_tqdm = not sys.stdout.isatty()

import torch
import torch.nn.functional as F

from networks.VGSformer import VGSformer
from load_config import parse_test_option, save_eval_result_csv
from loader.image.ISalObjDataset import get_isod_loader
from utils.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm


def compute_model_metrics(model, input_size=(30, 3, 384, 384), warmup=50, repeat=200):
    """计算 Params、FLOPS、FPS。

    Args:
        model: 已加载到 GPU 的模型。
        input_size: 输入尺寸 (B, C, H, W)。
        warmup: FPS 预热迭代次数。
        repeat: FPS 计时迭代次数。

    Returns:
        dict: {"params_M": float, "flops_G": float or None, "fps": float}
    """
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(*input_size, device=device)

    # Params (M)
    params = sum(p.numel() for p in model.parameters())
    params_M = params / 1e6

    # FPS（先于 FLOPS 执行，避免 thop 注册的 hooks 在失败后残留在模型上导致后续 model() 崩溃）
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    fps = repeat*input_size[0] / (t1 - t0)

    # FLOPS (G)：thop 与部分模块（如 Hiera 的 MaxPool2d）不兼容时会失败，跳过即可
    flops_G = None
    try:
        import thop
        with torch.no_grad():
            flops, _ = thop.profile(model, inputs=(dummy,), verbose=False)
        flops_G = flops / 1e9
    except Exception:
        pass
    finally:
        # thop 注册的 forward hooks 会残留在模型上，必须清除，否则后续 model() 会触发并崩溃
        for m in model.modules():
            m._forward_hooks.clear()
            m._forward_pre_hooks.clear()

    return {"params_M": params_M, "flops_G": flops_G, "fps": fps}


def print_metrics(metrics):
    """打印并返回指标字符串。"""
    lines = [
        "=" * 50,
        "Model Metrics",
        "=" * 50,
        f"Params: {metrics['params_M']:.2f} M",
        f"FPS:    {metrics['fps']:.1f}",
    ]
    if metrics["flops_G"] is not None:
        lines.append(f"FLOPS:  {metrics['flops_G']:.2f} G")
    else:
        lines.append("FLOPS:  N/A (install thop: pip install thop)")
    lines.append("=" * 50)
    text = "\n".join(lines)
    print(text)
    return text


args,config = parse_test_option()

# 不覆盖启动脚本已设置的 CUDA_VISIBLE_DEVICES。
# 若在 import torch 之后改成 --gpu_id 0，所有进程都会落到物理 GPU 0。
visible = os.environ.get("CUDA_VISIBLE_DEVICES", args.gpu_id)
print("USE GPU (CUDA_VISIBLE_DEVICES):", visible)


model = VGSformer(config, is_pre=False)

#print('NOW USING BackBone:',args.backbone)

ckpt_path = os.path.join(args.test_model, "ckpt", "Best_mae_test.pth")
state_dict = torch.load(ckpt_path, map_location="cpu")

model_dict = {}

for k,v in state_dict.items():
    if k.startswith('module'):
        model_dict[k[7:]] = v
    else:
        model_dict[k] = v

missing, unexpected = model.load_state_dict(model_dict, strict=False)
print(f"Loaded checkpoint (strict=False), missing={len(missing)}, unexpected={len(unexpected)}")

if getattr(args, "prune", False):
    model.prune()
    print("Pruned model: edge detection and deep supervision removed.")

save_path = os.path.join(args.test_model, 'save')

if not os.path.exists(save_path):
    os.mkdir(save_path)


model.cuda()
model.eval()

# 计算 FPS / Params / FLOPS
metrics_text = None
if not getattr(args, "no_metrics", False):
    img_size = args.img_size
    print(img_size)
    metrics = compute_model_metrics(model, input_size=(8, 3, img_size, img_size),warmup=10,repeat=100)
    metrics_text = print_metrics(metrics)
    if getattr(args, "metrics_only", False):
        os.makedirs(args.test_model, exist_ok=True)
        metrics_path = os.path.join(args.test_model, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(metrics_text)
        print(f"Metrics saved to {metrics_path}")
        sys.exit(0)

eval_result = {"MAE":[],
               "maxF":[],
                "avgF":[],
                "wfm": [],
                "sm": [],
                "em": []}


datasets = config.DATA.TESTSET#["PASCAL-S", "ECSSD", "HKU-IS", "DUT-O", "DUTS-TE"]
#["DUT", "DES",  "LFSD",  "NJU2K",  "NLPR",  "SIP",  "SSD",  "STERE", "COME-E", "COME-H", "ReDWeb-S"]
if len(datasets) != 0:
    config.DATA.TESTSET = datasets


test_loaders = get_isod_loader(config.DATA, 1, ds_type='test')


for dataset_name,test_loader in zip(config.DATA.TESTSET,test_loaders):

    dataset_save_path = os.path.join(save_path, dataset_name)
    if not os.path.exists(dataset_save_path):
        os.mkdir(dataset_save_path)

    mae,fm,sm,em,wfm= cal_mae(),cal_fm(len(test_loader)),cal_sm(),cal_em(),cal_wfm()
    with torch.no_grad():

        for image, gt, sz, name in tqdm(test_loader,disable=disable_tqdm):

            gt = gt.squeeze()/255.0

            image = image.unsqueeze(0).cuda()

            res = model(image)

            fname = os.path.join(dataset_save_path,name)
            
            res = F.interpolate(res[0], size=sz, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            cv2.imwrite(fname, (res*255).astype(np.uint8))


            mae.update(res,gt)
            sm.update(res,np.where(gt>0.5,1.0,0.0))
            fm.update(res, np.where(gt>0.5,1.0,0.0))
            em.update(res,np.where(gt>0.5,1.0,0.0))
            wfm.update(res,np.where(gt>0.5,1.0,0.0))

        MAE = mae.show()

        maxf,meanf,_,_ = fm.show()
        sm = sm.show()
        em = em.show()
        wfm = wfm.show()
        print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(dataset_name, MAE, maxf,meanf,wfm,sm,em))
        eval_result['MAE'].append(MAE)
        eval_result['maxF'].append(maxf)
        eval_result['avgF'].append(meanf)
        eval_result['wfm'].append(wfm)
        eval_result['sm'].append(sm)
        eval_result['em'].append(em)


save_eval_result_csv(args.test_model, eval_result, datasets)
print('Test Done!')


