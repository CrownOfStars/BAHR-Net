import os
import torch
import torch.nn as nn
# 根据 MzeroMiko/VMamba 的实际路径导入 VSSM
from VMamba.classification.models.vmamba import VSSM 

class VMambaPyramid(nn.Module):
    """
    鲁棒版 VMamba 特征金字塔提取器
    完全不依赖源码的内部属性 (如 .dim)，防报错！
    """
    def __init__(self, backbone, embed_dims):
        super().__init__()
        self.backbone = backbone
        self.embed_dims = embed_dims # 把外部算好的维度传进来

    def forward(self, x):
        features = []
        x = self.backbone.patch_embed(x)
        if hasattr(self.backbone, 'pos_drop'):
            x = self.backbone.pos_drop(x)
            
        for i, layer in enumerate(self.backbone.layers):
            # A. 正常过 SS2D Blocks
            for blk in layer.blocks:
                x = blk(x)
                
            # B. 核心修复：直接用我们的 embed_dims 去判断需不需要翻转维度！
            # VMamba 算子输出经常是 channels-last: (B, H, W, C)
            if x.dim() == 4 and x.shape[-1] in self.embed_dims:
                feat = x.permute(0, 3, 1, 2).contiguous()
            else:
                feat = x.contiguous()
                
            features.append(feat)
            
            # C. 下采样
            if hasattr(layer, 'downsample') and layer.downsample is not None:
                x = layer.downsample(x)
                
        return features

def build_vmamba_from_pretrained(model_name='vmamba_t', pretrained=True, pt_root='./weights'):
    configs = {
        'vmamba_t': {'depths': [2, 2, 9, 2], 'dims': 96},
        'vmamba_s': {'depths': [2, 2, 27, 2], 'dims': 96},
        'vmamba_b': {'depths': [2, 2, 27, 2], 'dims': 128},
    }
    
    cfg = configs[model_name]
    embed_dims = [cfg['dims'] * (2 ** i) for i in range(4)]
    
    # 强制后端设为 "torch" (如果你刚才没改源码的话，也可以在这里传)
    # 保证一定能跑通
    backbone = VSSM(
        depths=cfg['depths'],
        dims=cfg['dims'],
    )
    
    # 这里我们也可以加上刚才教你的动态修改后端的强硬手段
    # 防止你在别的地方初始化又报错
    if hasattr(backbone, 'selective_scan_backend'):
         backbone.selective_scan_backend = "torch"
    
    if pretrained:
        weight_path = os.path.join(pt_root, f"{model_name}.pth")
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint) 
            clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
            msg = backbone.load_state_dict(clean_state_dict, strict=False)
            print(msg)
        else:
            print(f"⚠️ Warning: Weight not found at {weight_path}")

    # 【重要修改】：把 embed_dims 传进 Wrapper
    model = VMambaPyramid(backbone, embed_dims)
    
    return model, embed_dims

# ================= 测试代码 =================
if __name__ == "__main__":
    # 测试输入 (例如 512x512 高分辨率)
    dummy_input = torch.randn(2, 3, 512, 512).cuda()
    
    print("Initializing VMamba Backbone...")
    model, embed_dims = build_vmamba_from_pretrained('vmamba_s', pretrained=True)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        features = model(dummy_input)
    
    print(f"VMamba Embed Dimensions: {embed_dims}")
    for i, feat in enumerate(features):
        print(f"Stage {i} feature shape: {feat.shape}")