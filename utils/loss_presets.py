"""
Loss 消融预设：lambda 风格，bounds 可配置 callable。
用法: loss_fn = get_loss_fn(args); loss = loss_fn(s1,s2,s3,s4,gts,edge1,edge2,bounds)
"""
import torch.nn.functional as F

from utils.loss import (
    GTSupervision_cls,
    AblationGTSupervision,
    Ablation2GTSupervision,
    EdgeSupervision,
    BlurSupervision,
)


def _blur_bounds(b, k=3):
    """bounds 模糊预处理，k 为 kernel 边长。"""
    pad = k // 2
    return F.avg_pool2d(
        F.max_pool2d(b, kernel_size=k, stride=1, padding=pad),
        kernel_size=k, stride=1, padding=pad
    )


# bounds 预处理：可配置的 callable
BOUNDS_FN = {
    'raw': lambda b: b,
    'blur3': lambda b: _blur_bounds(b, 3),
    'blur5': lambda b: _blur_bounds(b, 5),
}


def make_loss_fn(gt_loss, edge_loss, bounds_fn=None, edge_weight=1.0, use_edge2=False):
    """
    工厂：生成可配置的 loss 计算函数。
    bounds_fn: callable(bounds) -> 处理后的 target，默认 identity。
    """
    bounds_fn = bounds_fn or BOUNDS_FN['raw']
    blur_sup = BlurSupervision() if use_edge2 else None

    def compute(s1, s2, s3, s4, gt, e1, e2, b):
        L = gt_loss(s1, s2, s3, s4, gt)
        L = L + edge_weight * edge_loss(e1, bounds_fn(b))
        if blur_sup is not None:
            L = L + edge_weight * blur_sup(e2, bounds_fn(b))
        return L

    return compute


def get_loss_fn(args):
    """
    根据 args.loss 和 args.bounds_fn 返回 loss 计算函数。
    """
    bounds_fn = BOUNDS_FN.get(getattr(args, 'bounds_fn', 'raw'), BOUNDS_FN['raw'])
    edge_weight = getattr(args, 'edge_weight', 1.0)
    use_edge2 = getattr(args, 'use_edge2', False)

    preset = getattr(args, 'loss', 'ablation2')

    if preset == 'ablation1':
        return make_loss_fn(
            AblationGTSupervision, EdgeSupervision,
            bounds_fn=bounds_fn, edge_weight=edge_weight, use_edge2=use_edge2
        )
    elif preset == 'ablation2':
        return make_loss_fn(
            Ablation2GTSupervision, EdgeSupervision,
            bounds_fn=bounds_fn, edge_weight=edge_weight, use_edge2=use_edge2
        )
    elif preset == 'full':
        alpha1 = getattr(args, 'alpha1', 2.0)
        alpha2 = getattr(args, 'alpha2', 0.3)
        beta = getattr(args, 'beta', 1.0)
        gt_loss = GTSupervision_cls(alpha1=alpha1, alpha2=alpha2, beta=beta)
        return make_loss_fn(
            gt_loss, EdgeSupervision,
            bounds_fn=bounds_fn, edge_weight=edge_weight, use_edge2=use_edge2
        )
    elif preset == 'gt_only':
        def compute(s1, s2, s3, s4, gt, e1, e2, b):
            return Ablation2GTSupervision(s1, s2, s3, s4, gt)
        return compute
    elif preset == 'edge_only':
        def compute(s1, s2, s3, s4, gt, e1, e2, b):
            return edge_weight * EdgeSupervision(e1, bounds_fn(b))
        return compute
    else:
        raise ValueError(f"Unknown loss preset: {preset}")
