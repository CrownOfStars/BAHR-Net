"""
评估指标函数，用于验证/测试阶段计算 IoU、MAE 等。
"""
import torch


def IoU_metric(pred, target, eps=1e-6):
    """单张图 IoU 求和，用于 batch 内累加。"""
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = inter / (union + eps)
    return iou.sum()


def MAE_metric(pred, target):
    """单张图 MAE 求和，用于 batch 内累加。"""
    mae = torch.abs(pred - target).mean(dim=(1, 2, 3))
    return mae.sum()


def IoU(pred, target):
    """多尺度 pred/target 列表的 IoU 总和。"""
    return sum([
        torch.sum(t * p) / (torch.sum(t) + torch.sum(p) - torch.sum(t * p))
        for p, t in zip(pred, target)
    ])


def IoUs(pred, target):
    """多尺度 pred/target 列表的 IoU，返回 stack 后的张量。"""
    return torch.stack([
        torch.sum(t * p) / (torch.sum(t) + torch.sum(p) - torch.sum(t * p))
        for p, t in zip(pred, target)
    ])
