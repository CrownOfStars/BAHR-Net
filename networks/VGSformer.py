import sys
sys.path.append('../')
sys.path.append('./')
import numpy as np
from networks.PWNet import *
from networks.ASPPFormer import ASPPFormer
from networks.EAFormer import EAFormer

from networks.models_config import parse_option

from networks.MFusionToolBox import *
from networks.EdgeAwareToolBox import *

from networks.backbone.segswin import build_segswin
from networks.backbone.convnextv2 import build_convnextv2
from networks.backbone.pvt_v2_eff import build_pvtv2
from networks.backbone.build_hiera import build_hiera

#from networks.backbone.intern.build_intern import build_internimage_from_ade20k
#from networks.backbone.vmamba.build_mamba import build_vmamba_from_pretrained

"""
TORCH_DISTRIBUTED_DEBUG
"""


def get_parameter_num(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    return ('Trainable Parameters: %.3fM' % parameters)


class Interpolate(nn.Module):
    def __init__(self, size, mode = 'nearest'):
        super(Interpolate, self).__init__()
        self.interpolate = nn.functional.interpolate
        self.size = size
        self.mode = mode
    def forward(self, x):
        x = self.interpolate(x, size=self.size, mode=self.mode)
        return x



class ChannelCompression(nn.Module):
    def norm_layer(channel, norm_name='gn'):
        if norm_name == 'bn':
            return nn.BatchNorm2d(channel)
        elif norm_name == 'gn':
            return nn.GroupNorm(min(32, channel // 4), channel)
    def __init__(self, in_c, out_c=64):
        super(ChannelCompression, self).__init__()
        intermediate_c = in_c // 4 if in_c >= 256 else 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, intermediate_c, 1, bias=False),
            ChannelCompression.norm_layer(intermediate_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_c, intermediate_c, 3, 1, 1, bias=False),
            ChannelCompression.norm_layer(intermediate_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_c, out_c, 1, bias=False),
            ChannelCompression.norm_layer(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class VGSformer(nn.Module):
    def __init__(self, config, is_pre=True):
        super(VGSformer, self).__init__()

        bb_type = str(getattr(config.BACKBONE, "TYPE", "") or "")
        bb_name = str(getattr(config.BACKBONE, "NAME", "") or "")
        if bb_type == "segswin" or bb_name == "segswin-base":
            self.encoderR, embed_dims = build_segswin(config.BACKBONE,config.DATA.PRETRAIN_SIZE,config.TRAIN.PRE_ROOT if is_pre else None)
        elif bb_type == "convnextv2" or "convnextv2" in bb_name:
            self.encoderR, embed_dims = build_convnextv2(config, config.DATA.PRETRAIN_SIZE, config.TRAIN.PRE_ROOT if is_pre else None)
        elif bb_type == "hiera" or bb_name == "hiera-base":
            self.encoderR, embed_dims = build_hiera(config, config.DATA.PRETRAIN_SIZE, config.TRAIN.PRE_ROOT if is_pre else None)
        elif bb_name in ("pvtv2_b4", "pvtv2_b5"):
            self.encoderR, embed_dims = build_pvtv2(config, config.DATA.PRETRAIN_SIZE, config.TRAIN.PRE_ROOT if is_pre else None)
        else:
            raise ValueError(f"Unsupported backbone: type={bb_type}, name={bb_name}")


        embed_dims = self.add_adapter(embed_dims)

        fused_dims = [embed_dims[0]//2,embed_dims[0]//2]+embed_dims
        self.fused_dims = fused_dims

        self.S4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[3], 1, kernel_size=1, bias=False),
        )
        self.S3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[2], 1, kernel_size=1, bias=False),
        )
        self.S2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[1], 1, kernel_size=1, bias=False),
        )
        self.S1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[0], 1, kernel_size=1, bias=False),
        )

        """
        图片分类预训练模型可以用作backbone,但是分类任务对边界并不敏感,加入边界提取可以提高模型迁移到显著性检测的效果
        """
        self.sod_edge_aware = EAFormer(fused_dims[:4],out_scale=2.0)

        """
        (1024, 1024, 1024, 3, 2)
        (1536, 1024, 512, 3, 2)
        (768, 512, 256, 5, 3)
        (384, 256, 128, 5, 3)
        """

        # different scale feature fusion
        self.aspf_dec4 = ASPPFormer(fused_dims[5], fused_dims[4], fused_dims[3], 6, 3)#PATM_BAB(fused_dims[3], fused_dims[3], fused_dims[3], 3, 2)#TODO reduce params
        self.aspf_dec3 = ASPPFormer(fused_dims[4], fused_dims[4], fused_dims[2], 12, 6)#PATM_BAB(fused_dims[2]+fused_dims[3], fused_dims[3], fused_dims[2], 3, 2)
        self.aspf_dec2 = ASPPFormer(fused_dims[3], fused_dims[3], fused_dims[1], 12, 6)#PATM_BAB(fused_dims[1]+fused_dims[2], fused_dims[2], fused_dims[1], 5, 3)
        self.aspf_dec1 = ASPPFormer(fused_dims[2], fused_dims[2], fused_dims[0], 24, 12)#PATM_BAB(fused_dims[0]+fused_dims[1], fused_dims[1], fused_dims[0], 5, 3)
        
        
        
        
    def prune(self):
        """移除边缘检测(sod_edge_aware)和深监督(S2,S3,S4)，仅保留主输出 S1，以最大化 FPS、最小化 FLOPS 和 Params。"""
        if getattr(self, "_pruned", False):
            return
        self._pruned = True
        del self.sod_edge_aware
        del self.S2
        del self.S3
        del self.S4
        
        
    def add_adapter(self, embed_dims):
        self.ada0 = nn.Identity()
        if embed_dims[1] != embed_dims[0]*2:
            self.ada1 = nn.Conv2d(embed_dims[1], embed_dims[0]*2, kernel_size=1, bias=False)
        else:
            self.ada1 = nn.Identity()
        if embed_dims[2] != embed_dims[0]*4:
            self.ada2 = nn.Conv2d(embed_dims[2], embed_dims[0]*4, kernel_size=1, bias=False)
        else:
            self.ada2 = nn.Identity()
        if embed_dims[3] != embed_dims[0]*8:
            self.ada3 = nn.Conv2d(embed_dims[3], embed_dims[0]*8, kernel_size=1, bias=False)
        else:
            self.ada3 = nn.Identity()
        return [embed_dims[0], embed_dims[0]*2, embed_dims[0]*4, embed_dims[0]*8]
            


    def forward(self, rgb, ref=None):
        x0, x1, x2, x3 = self.encoderR(rgb)
        x2_ACCoM = self.ada0(x0)
        x3_ACCoM = self.ada1(x1)
        x4_ACCoM = self.ada2(x2)
        x5_ACCoM = self.ada3(x3)
        mer_cros4 = self.aspf_dec4(x5_ACCoM)
        mer_cros3 = self.aspf_dec3(x4_ACCoM, mer_cros4)
        mer_cros2 = self.aspf_dec2(x3_ACCoM, mer_cros3)
        mer_cros1 = self.aspf_dec1(x2_ACCoM, mer_cros2)
        s1 = self.S1(mer_cros1)
        if getattr(self, "_pruned", False):
            return (s1,)  # 兼容 test.py 的 res[0]
        edge_sod = self.sod_edge_aware(mer_cros1, mer_cros2, mer_cros3, mer_cros4)
        s2 = self.S2(mer_cros2)
        s3 = self.S3(mer_cros3)
        s4 = self.S4(mer_cros4)
        return s1, s2, s3, s4, edge_sod, 0.0, 0.0

    def get_module_params(self):
        for name, item in self.named_children():
            print(name,get_parameter_num(item))



class VGSformer_Joint(nn.Module):
    def __init__(self, config, is_pre=True):
        super(VGSformer_Joint, self).__init__()

        # --- Backbone 保持不变 ---
        if config.BACKBONE.NAME == 'segswin-base':
            self.encoderR, embed_dims = build_segswin(config.BACKBONE,config.DATA.PRETRAIN_SIZE,config.TRAIN.PRE_ROOT if is_pre else None)
        elif config.BACKBONE.NAME == 'convnextv2-base':
            self.encoderR, embed_dims = build_convnextv2(config, config.DATA.PRETRAIN_SIZE, config.TRAIN.PRE_ROOT if is_pre else None)
        elif config.BACKBONE.NAME == 'hiera-base':
            self.encoderR, embed_dims = build_hiera(config, config.DATA.PRETRAIN_SIZE, config.TRAIN.PRE_ROOT if is_pre else None)
        else:
            raise ValueError(f"Unsupported backbone: {config.BACKBONE.TYPE}")

        fused_dims = [embed_dims[0]//2,embed_dims[0]//2]+embed_dims
        self.fused_dims = fused_dims

        # ================= 核心修改 1：双通道输出 =================
        # 将所有深监督头 S1~S4 的输出通道数从 1 改为 2
        # Channel 0: SOD, Channel 1: COD
        self.S4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[3], 2, kernel_size=1, bias=False),
        )
        self.S3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[2], 2, kernel_size=1, bias=False),
        )
        self.S2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[1], 2, kernel_size=1, bias=False),
        )
        self.S1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[0], 2, kernel_size=1, bias=False),
        )

        # ================= 核心修改 2：边缘特征解耦 =================
        # 方案A：如果 EAFormer 支持传入 out_channels，只需：
        # self.edge_aware = EAFormer(fused_dims[:4], out_scale=2.0, out_channels=2)
        
        # 方案B：为了最稳健的特征隔离，直接实例化两个（假设 EAFormer 很轻量）
        self.sod_edge_aware = EAFormer(fused_dims[:4], out_scale=2.0)
        self.cod_edge_aware = EAFormer(fused_dims[:4], out_scale=2.0)

        # different scale feature fusion (保持不变)
        self.aspf_dec4 = ASPPFormer(fused_dims[5], fused_dims[4], fused_dims[3], 6, 3)
        self.aspf_dec3 = ASPPFormer(fused_dims[4], fused_dims[4], fused_dims[2], 12, 6)
        self.aspf_dec2 = ASPPFormer(fused_dims[3], fused_dims[3], fused_dims[1], 12, 6)
        self.aspf_dec1 = ASPPFormer(fused_dims[2], fused_dims[2], fused_dims[0], 24, 12)
        
    def prune(self):
        """剪枝时不仅要移除深监督，还要移除两个边缘模块"""
        if getattr(self, "_pruned", False):
            return
        self._pruned = True
        del self.sod_edge_aware
        del self.cod_edge_aware
        del self.S2
        del self.S3
        del self.S4

    def forward(self, rgb, ref=None):
        x0, x1, x2, x3 = self.encoderR(rgb)
        x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM = x0, x1, x2, x3
        
        mer_cros4 = self.aspf_dec4(x5_ACCoM)
        mer_cros3 = self.aspf_dec3(x4_ACCoM, mer_cros4)
        mer_cros2 = self.aspf_dec2(x3_ACCoM, mer_cros3)
        mer_cros1 = self.aspf_dec1(x2_ACCoM, mer_cros2)
        
        # 此时 s1 的 shape 为 (Batch, 2, H, W)
        s1 = self.S1(mer_cros1)
        
        if getattr(self, "_pruned", False):
            return (s1,)  # 测试阶段，直接返回双通道预测图

        # 训练阶段：分别提取两类任务的边缘
        edge_sod = self.sod_edge_aware(mer_cros1, mer_cros2, mer_cros3, mer_cros4)
        edge_cod = self.cod_edge_aware(mer_cros1, mer_cros2, mer_cros3, mer_cros4)
        
        # 提取深监督预测图，shape 均为 (Batch, 2, H, W)
        s2 = self.S2(mer_cros2)
        s3 = self.S3(mer_cros3)
        s4 = self.S4(mer_cros4)
        
        # 返回主掩码预测 (s1-s4) 和边缘预测
        return s1, s2, s3, s4, edge_sod, edge_cod


class VGSformer_softmax(nn.Module):
    """
    三分支语义分割形式：logits shape (B, 3, H, W)。
    cls0=背景, cls1=SOD 前景, cls2=COD 前景。无 EAFormer 边缘支路，结构与 Joint 解码相同，仅替换为 3 通道头。
    训练建议使用 CrossEntropyLoss；深监督在多尺度上对同一语义标签加权求和。
    """

    num_classes = 3

    def __init__(self, config, is_pre=True):
        super(VGSformer_softmax, self).__init__()

        if config.BACKBONE.NAME == 'segswin-base':
            self.encoderR, embed_dims = build_segswin(
                config.BACKBONE, config.DATA.PRETRAIN_SIZE, config.TRAIN.PRE_ROOT if is_pre else None
            )
        elif config.BACKBONE.NAME == 'convnextv2-base':
            self.encoderR, embed_dims = build_convnextv2(
                config, config.DATA.PRETRAIN_SIZE, config.TRAIN.PRE_ROOT if is_pre else None
            )
        elif config.BACKBONE.NAME == 'hiera-base':
            self.encoderR, embed_dims = build_hiera(
                config, config.DATA.PRETRAIN_SIZE, config.TRAIN.PRE_ROOT if is_pre else None
            )
        else:
            raise ValueError(f"Unsupported backbone: {config.BACKBONE.TYPE}")

        fused_dims = [embed_dims[0] // 2, embed_dims[0] // 2] + embed_dims
        self.fused_dims = fused_dims

        c_out = self.num_classes
        self.S4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[3], c_out, kernel_size=1, bias=False),
        )
        self.S3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[2], c_out, kernel_size=1, bias=False),
        )
        self.S2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[1], c_out, kernel_size=1, bias=False),
        )
        self.S1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fused_dims[0], c_out, kernel_size=1, bias=False),
        )

        self.aspf_dec4 = ASPPFormer(fused_dims[5], fused_dims[4], fused_dims[3], 6, 3)
        self.aspf_dec3 = ASPPFormer(fused_dims[4], fused_dims[4], fused_dims[2], 12, 6)
        self.aspf_dec2 = ASPPFormer(fused_dims[3], fused_dims[3], fused_dims[1], 12, 6)
        self.aspf_dec1 = ASPPFormer(fused_dims[2], fused_dims[2], fused_dims[0], 24, 12)

    def prune(self):
        if getattr(self, "_pruned", False):
            return
        self._pruned = True
        del self.S2
        del self.S3
        del self.S4

    def forward(self, rgb, ref=None):
        x0, x1, x2, x3 = self.encoderR(rgb)
        x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM = x0, x1, x2, x3

        mer_cros4 = self.aspf_dec4(x5_ACCoM)
        mer_cros3 = self.aspf_dec3(x4_ACCoM, mer_cros4)
        mer_cros2 = self.aspf_dec2(x3_ACCoM, mer_cros3)
        mer_cros1 = self.aspf_dec1(x2_ACCoM, mer_cros2)

        logits1 = self.S1(mer_cros1)

        if getattr(self, "_pruned", False):
            return (logits1,)

        logits2 = self.S2(mer_cros2)
        logits3 = self.S3(mer_cros3)
        logits4 = self.S4(mer_cros4)
        return logits1, logits2, logits3, logits4


if __name__ == "__main__":
    
    args,config = parse_option()

    model = VGSformer(config).cuda()
    """
    python train.py --backbone swin-large --texture /namlab40/ --mfusion HAIM --train_batch 20 --gpu_id 6
    """
    rgb = torch.randn(4,3,384,384).cuda()
    x = torch.randn(4,3,384,384).cuda()

    pred = model(rgb,x)
    print(pred[0].shape,pred[1].shape)


