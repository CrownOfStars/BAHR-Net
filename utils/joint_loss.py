import torch
import torch.nn as nn
import torch.nn.functional as F

class JointLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_iou=1.0, weight_co=0.5, weight_iou_sod=0.5, weight_iou_cod=0.5):
        """
        联合损失函数 (支持 SOD/COD 双通道输出)
        :param weight_bce: BCE Loss 的权重
        :param weight_iou: IoU Loss 的权重
        :param weight_co: 类共现惩罚项 (互斥损失) 的权重
        :param weight_iou_sod: SOD IoU Loss 的权重 (与 weight_iou_cod 共同控制任务难度比例)
        :param weight_iou_cod: COD IoU Loss 的权重
        """
        super(JointLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_iou = weight_iou
        self.weight_co = weight_co
        self.weight_iou_sod = weight_iou_sod
        self.weight_iou_cod = weight_iou_cod

    def _iou_loss(self, pred, mask):
        """
        计算单个通道的 IoU Loss
        """
        # pred 和 mask 的 shape 均为 (B, 1, H, W)
        inter = (pred * mask).sum(dim=(2, 3))
        union = (pred + mask).sum(dim=(2, 3)) - inter
        
        # 加 1e-6 防止除以 0
        iou = (inter + 1e-6) / (union + 1e-6)
        
        # iou 越大越好，loss 越小越好
        iou_loss = 1.0 - iou
        return iou_loss.mean()

    def forward(self, preds, targets):
        """
        :param preds: 模型的双通道输出 Logits, shape: (B, 2, H, W)
        :param targets: 组装好的双通道 GT, shape: (B, 2, H, W)
        """
        # 1. 计算基础 BCE Loss (自带 Sigmoid，数值更稳定)
        loss_bce = F.binary_cross_entropy_with_logits(preds, targets)

        # 2. 将 Logits 转换为概率 (0~1)，用于计算 IoU 和共现惩罚
        preds_sigmoid = torch.sigmoid(preds)

        # 分离出 SOD 和 COD 的预测与标签
        pred_sod = preds_sigmoid[:, 0:1, :, :]
        pred_cod = preds_sigmoid[:, 1:2, :, :]
        
        mask_sod = targets[:, 0:1, :, :]
        mask_cod = targets[:, 1:2, :, :]

        # 3. 计算 IoU Loss (SOD和COD分别计算后按权重加权求和)
        loss_iou_sod = self._iou_loss(pred_sod, mask_sod)
        loss_iou_cod = self._iou_loss(pred_cod, mask_cod)
        loss_iou = self.weight_iou_sod * loss_iou_sod + self.weight_iou_cod * loss_iou_cod

        # 4. 核心：计算类共现惩罚项 (Class Co-occurrence Penalty)
        # 逐像素相乘：如果同一个像素点 pred_sod 和 pred_cod 都很大，惩罚就大
        co_occurrence_map = pred_sod * pred_cod
        
        # 求空间维度和 Batch 维度的均值
        loss_co = co_occurrence_map.mean()

        # 5. 总损失加权求和
        total_loss = (self.weight_bce * loss_bce) + \
                     (self.weight_iou * loss_iou) + \
                     (self.weight_co * loss_co) 
        loss_dict = {
            'loss_bce': loss_bce.detach().cpu(),
            'loss_iou': loss_iou.detach().cpu(),
            'loss_co': loss_co.detach().cpu(),
        }

        return total_loss, loss_dict