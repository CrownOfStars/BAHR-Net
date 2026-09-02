"""
将 JointObjDataset 的双通道 GT (SOD/COD mask) 转为三分类语义标签，配合 CrossEntropyLoss。

类别定义：
  0：背景（两通道均无前景）
  1：SOD 前景
  2：COD 前景

若像素上 SOD 与 COD 同时为 1，优先标为类别 1（SOD）。
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def joint_twochannel_to_semantic_cls(gts: torch.Tensor) -> torch.Tensor:
    """
    :param gts: Float (B, 2, H, W)，通常为 0/1
    :return: Long (B, H, W)，取值 {0,1,2}
    """
    sod = gts[:, 0] > 0.5
    cod = gts[:, 1] > 0.5
    out = torch.zeros(gts.shape[0], *gts.shape[2:], device=gts.device, dtype=torch.long)
    # COD 单列（避免与 sod 同时为 1 时重复计价）
    out[cod & ~sod] = 2
    out[sod] = 1
    return out


def logits_to_joint_probs(logits_main: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """(B,C,H,W) 三分类 logits -> 对齐 target_hw -> (B,2,H,W) [SOD, COD] 概率."""
    lt = logits_main
    if lt.shape[-2:] != target_hw:
        lt = F.interpolate(lt, target_hw, mode="bilinear", align_corners=False)
    prob = torch.softmax(lt.float(), dim=1)
    prob_sod = prob[:, 1:2, :, :]
    prob_cod = prob[:, 2:3, :, :]
    return torch.cat([prob_sod, prob_cod], dim=1)


def extract_main_logits(model_out: Union[torch.Tensor, tuple, list]) -> torch.Tensor:
    """深监督 tuple 取主输出 logits1；剪枝后可能为单元素 tuple 或 tensor."""
    if isinstance(model_out, (tuple, list)):
        return model_out[0]
    return model_out


def joint_probs_to_eval_map(
    probs: torch.Tensor, task_id: Union[int, torch.Tensor]
) -> torch.Tensor:
    """
    从 (B,2,H,W) 联合概率图选取与 task 对应的单通道评估图（供 MAE/F-measure 等）。
    task_id: 0=SOD, 1=COD, 2=USOD（取 SOD/COD 前景概率较大者，即 union）。
    """
    tid = int(task_id.item() if isinstance(task_id, torch.Tensor) else task_id)
    if tid == 0:
        return probs[:, 0]
    if tid == 1:
        return probs[:, 1]
    return torch.maximum(probs[:, 0], probs[:, 1])


def joint_probs_to_eval_map_np(probs_2hw: np.ndarray, task_id: int) -> np.ndarray:
    """numpy 版，probs_2hw shape (2, H, W)."""
    if task_id == 0:
        return probs_2hw[0]
    if task_id == 1:
        return probs_2hw[1]
    return np.maximum(probs_2hw[0], probs_2hw[1])
