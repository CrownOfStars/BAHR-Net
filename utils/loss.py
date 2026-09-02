"""
损失函数模块，提供可配置的 callable 对象，便于消融实验与网格搜索。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 辅助函数
# =============================================================================

def avg_max_pool(bounds, thickness):
    return F.avg_pool2d(
        F.max_pool2d(bounds, kernel_size=2 * thickness + 1, stride=1, padding=thickness),
        kernel_size=2 * thickness + 1, stride=1, padding=thickness
    )


@torch.no_grad()
def _calcu_weight(pred: torch.Tensor, mask: torch.Tensor, alpha1: float, alpha2: float, beta: float, *kss):
    w_mask, w_pred = torch.zeros_like(mask), torch.zeros_like(mask)
    for ks in kss:
        pad = ks // 2
        w_mask += torch.abs(F.avg_pool2d(mask, ks, stride=1, padding=pad) - mask)
        w_pred += torch.abs(F.avg_pool2d(pred, ks, stride=1, padding=pad) - pred)
    return alpha1 * w_mask + alpha2 * w_pred + beta


def _iou_loss_core(pred, target, eps=1e-16):
    """IoU loss 核心计算，pred 需已 sigmoid。"""
    intersection = (pred * target).float().sum((2, 3))
    union = (pred + target).float().sum((2, 3)) / 2
    iou = (intersection + eps) / (union + eps)
    return 1 - iou.mean()


# =============================================================================
# 基础 Loss 类（可配置）
# =============================================================================

class BCELoss(nn.Module):
    """可配置的 BCE Loss。"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction=self.reduction)


class BCEWithLogitsLoss(nn.Module):
    """可配置的 BCEWithLogits Loss。"""
    def __init__(self, reduction='mean', pos_weight=None):
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(
            pred, target, reduction=self.reduction, pos_weight=self.pos_weight
        )


class IOULoss(nn.Module):
    """可配置的 IoU Loss。"""
    def __init__(self, eps=1e-16, apply_sigmoid=True):
        super().__init__()
        self.eps = eps
        self.apply_sigmoid = apply_sigmoid

    def forward(self, pred, target):
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)
        return _iou_loss_core(pred, target, self.eps)


# =============================================================================
# 结构 Loss 类（可配置，便于消融与网格搜索）
# =============================================================================

class StructureLoss(nn.Module):
    """
    加权 BCE + wIoU 结构损失。
    参数可调，便于消融实验与网格搜索。
    """
    def __init__(self, alpha1=2.0, alpha2=0.3, beta=1.0, ks=(31, 15), wiou_eps=1e-6):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.ks = ks if isinstance(ks, (list, tuple)) else (ks,)
        self.wiou_eps = wiou_eps

    def forward(self, pred, mask):
        sig_pred = torch.sigmoid(pred)
        weit = _calcu_weight(sig_pred, mask, self.alpha1, self.alpha2, self.beta, *self.ks)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((sig_pred * mask) * weit).sum(dim=(2, 3))
        union = ((sig_pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1 + self.wiou_eps)
        return (wbce + wiou).mean()


class Ablation1Loss(nn.Module):
    """消融1：等权结构 loss（weit=1）。"""
    def __init__(self, wiou_eps=1e-6):
        super().__init__()
        self.wiou_eps = wiou_eps

    def forward(self, pred, mask):
        sig_pred = torch.sigmoid(pred)
        weit = torch.ones_like(pred, dtype=torch.float32)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((sig_pred * mask) * weit).sum(dim=(2, 3))
        union = ((sig_pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1 + self.wiou_eps)
        return (wbce + wiou).mean()


class Ablation2Loss(nn.Module):
    """消融2：仅 mask 权重的结构 loss（固定 kernel_size=31）。"""
    def __init__(self, kernel_size=31, wiou_eps=1e-6, mask_weight=5.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.wiou_eps = wiou_eps
        self.mask_weight = mask_weight

    def forward(self, pred, mask):
        pad = self.kernel_size // 2
        weit = 1 + self.mask_weight * torch.abs(
            F.avg_pool2d(mask, self.kernel_size, stride=1, padding=pad) - mask
        )
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1 + self.wiou_eps)
        return (wbce + wiou).mean()


# =============================================================================
# 多尺度监督 Loss 类（可配置）
# =============================================================================

class GTSupervision(nn.Module):
    """
    多尺度 GT 监督，使用 StructureLoss。
    参数可调，便于消融与网格搜索。
    """
    def __init__(
        self,
        alpha1=2.0,
        alpha2=0.3,
        beta=1.0,
        ks_list=None,
        scale_factor=0.5,
        wiou_eps=1e-6,
    ):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.ks_list = ks_list or [(31, 15), (15, 15), (15, 7), (7, 7)]
        self.scale_factor = scale_factor
        self.wiou_eps = wiou_eps

    def forward(self, s1, s2, s3, s4, gt):
        loss = 0.0
        for s, ks in zip([s1, s2, s3, s4], self.ks_list):
            sl = StructureLoss(
                alpha1=self.alpha1, alpha2=self.alpha2, beta=self.beta,
                ks=ks, wiou_eps=self.wiou_eps
            )
            loss += sl(s, gt)
            gt = F.interpolate(gt, scale_factor=self.scale_factor, recompute_scale_factor=True)
        return loss


class AblationGTSupervision(nn.Module):
    """消融1 多尺度监督。"""
    def __init__(self, ablation1_loss=None, ks_list=None, scale_factor=0.5):
        super().__init__()
        self.ablation1_loss = ablation1_loss or Ablation1Loss()
        self.ks_list = ks_list or [(31, 15), (15, 15), (15, 7), (7, 7)]
        self.scale_factor = scale_factor

    def forward(self, s1, s2, s3, s4, gt):
        loss = 0.0
        for s in [s1, s2, s3, s4]:
            loss += self.ablation1_loss(s, gt)
            gt = F.interpolate(gt, scale_factor=self.scale_factor, recompute_scale_factor=True)
        return loss


class Ablation2GTSupervision(nn.Module):
    """消融2 多尺度监督。"""
    def __init__(self, ablation2_loss=None, ks_list=None, scale_factor=0.5):
        super().__init__()
        self.ablation2_loss = ablation2_loss or Ablation2Loss()
        self.ks_list = ks_list or [(31, 15), (15, 15), (15, 7), (7, 7)]
        self.scale_factor = scale_factor

    def forward(self, s1, s2, s3, s4, gt):
        loss = 0.0
        for s in [s1, s2, s3, s4]:
            loss += self.ablation2_loss(s, gt)
            gt = F.interpolate(gt, scale_factor=self.scale_factor, recompute_scale_factor=True)
        return loss


# =============================================================================
# 边缘 / 模糊 / 运动监督
# =============================================================================

class EdgeSupervision(nn.Module):
    """边缘监督，IoU loss。"""
    def __init__(self, eps=1e-16):
        super().__init__()
        self.eps = eps

    def forward(self, edge1, bounds):
        return _iou_loss_core(torch.sigmoid(edge1), bounds, self.eps)


class BlurSupervision(nn.Module):
    """模糊边界监督，BCE + IoU。"""
    def __init__(self, kernel_size=3, bce_weight=1.0, iou_weight=1.0, eps=1e-16):
        super().__init__()
        self.kernel_size = kernel_size
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.eps = eps
        self.bce = BCEWithLogitsLoss()

    def forward(self, edge2, bounds):
        bounds = F.avg_pool2d(
            F.max_pool2d(bounds, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2),
            kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2
        )
        return (
            self.bce_weight * self.bce(edge2, bounds)
            + self.iou_weight * _iou_loss_core(torch.sigmoid(edge2), bounds, self.eps)
        )



# =============================================================================
# 知识蒸馏 DIST
# =============================================================================

def _cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def _pearson_correlation(x, y, eps=1e-8):
    return _cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def _inter_class_relation(y_s, y_t):
    return 1 - _pearson_correlation(y_s, y_t).mean()


def _intra_class_relation(y_s, y_t):
    return _inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    """可配置的 DIST 知识蒸馏损失。"""
    def __init__(self, beta=1.0, gamma=1.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        y_s = y_s.softmax(dim=1)
        y_t = y_t.softmax(dim=1)
        inter_loss = _inter_class_relation(y_s, y_t)
        intra_loss = _intra_class_relation(y_s, y_t)
        return self.beta * inter_loss + self.gamma * intra_loss


# =============================================================================
# 向后兼容：默认实例（保持原有调用方式，如 GTSupervision(s1,s2,s3,s4,gt)）
# 定制参数时使用 _cls 类：GTSupervision_cls(alpha1=3.0)(s1,s2,s3,s4,gt)
# =============================================================================

CELoss = BCELoss(reduction='mean')
CEWLoss = BCEWithLogitsLoss()
IOULoss_default = IOULoss()

# 保存类引用供定制（在覆盖前）
GTSupervision_cls = GTSupervision
EdgeSupervision_cls = EdgeSupervision
AblationGTSupervision_cls = AblationGTSupervision
Ablation2GTSupervision_cls = Ablation2GTSupervision

# 默认实例，保持与 distributed.py 等处的调用兼容
GTSupervision = GTSupervision()
EdgeSupervision = EdgeSupervision()
AblationGTSupervision = AblationGTSupervision()
Ablation2GTSupervision = Ablation2GTSupervision()


if __name__ == "__main__":
    KD_loss = DIST()
    print(KD_loss(torch.randn(4, 128, 56, 56), torch.randn(4, 128, 56, 56)))
