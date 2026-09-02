import os
import torch
import torch.nn as nn
from InternImage.classification.models.intern_image import InternImage

class InternImagePyramid(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        features = []
        # 1. 经过 Stem (Patch Embedding)
        x = self.backbone.patch_embed(x)
        x = self.backbone.pos_drop(x)
        
        # 2. 逐层提取金字塔特征
        for level in self.backbone.levels:
            # A. 手动让数据过当前 Stage 的所有 DCNv3 block
            # 兼容 nn.Sequential 或 nn.ModuleList 的写法
            if isinstance(level.blocks, nn.Sequential):
                x = level.blocks(x)
            else:
                for blk in level.blocks:
                    x = blk(x)
                
            # B. 关键！此时的 x 是【下采样前】的特征，正是我们需要的多尺度金字塔
            feat = x.permute(0, 3, 1, 2).contiguous()
            features.append(feat)
            
            # C. 手动执行 downsample，把缩小的特征图喂给下一个 Stage (最后一层为 None)
            if level.downsample is not None:
                x = level.downsample(x)
                
        return features

def build_internimage_from_ade20k(model_name='internimage_t', pretrained=True, pt_root='./weights'):
    """
    专门适配 MMSegmentation ADE20K 分割权重的 Backbone 构建函数
    """
    configs = {
        'internimage_t': {'channels': 64,  'depths': [4, 4, 18, 4], 'groups': [4, 8, 16, 32]},
        'internimage_s': {'channels': 80,  'depths': [4, 4, 21, 4], 'groups': [5, 10, 20, 40]},
        'internimage_b': {'channels': 112, 'depths': [4, 4, 21, 4], 'groups': [7, 14, 28, 56]},
    }
    
    if model_name not in configs:
        raise ValueError(f"Unsupported model: {model_name}")
        
    cfg = configs[model_name]
    
    # 1. 实例化原生 Backbone
    backbone = InternImage(
        core_op='DCNv3',
        channels=cfg['channels'],
        depths=cfg['depths'],
        groups=cfg['groups']
    )
    
    # 2. 核心：加载并清洗 ADE20K 权重
    if pretrained:
        # 文件名与下载链接对应
        weight_name = f"upernet_{model_name}_512_160k_ade20k.pth"
        weight_path = os.path.join(pt_root, weight_name)
        
        if os.path.exists(weight_path):
            print(f"Loading weights from {weight_path}...")
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            # MMSegmentation 的权重通常存在 'state_dict' 键中
            raw_state_dict = checkpoint.get('state_dict', checkpoint)
            
            # 权重清洗：只保留 backbone 部分，并去除 'backbone.' 前缀
            clean_state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith('backbone.'):
                    # 把 'backbone.levels.0...' 变成 'levels.0...'
                    new_key = k.replace('backbone.', '', 1) 
                    clean_state_dict[new_key] = v
            
            # 加载清洗后的权重 (strict=False 防止一些无关紧要的 buffer key 报错)
            missing_keys, unexpected_keys = backbone.load_state_dict(clean_state_dict, strict=False)
            
            print("✅ Successfully injected ADE20K backbone weights!")
            if unexpected_keys:
                print(f"⚠️ Ignored unexpected keys: {len(unexpected_keys)} keys (Normal for mmseg checkpoints)")
        else:
            raise FileNotFoundError(f"❌ Weight file not found at {weight_path}. Please download it first.")

    # 3. 套上金字塔 Wrapper
    model = InternImagePyramid(backbone)
    embed_dims = [cfg['channels'] * (2 ** i) for i in range(4)]
    
    return model, embed_dims

# 测试运行
if __name__ == "__main__":
    # 请确保你在运行前已经用 wget 下载了权重放入 ./weights
    model, embed_dims = build_internimage_from_ade20k('internimage_b', pretrained=True)
    model = model.cuda()
    
    # 测试输入
    x = torch.randn(2, 3, 768, 768).cuda()
    feats = model(x)
    print("Feature dimensions extracted from ADE20K initialized Backbone:")
    for i, f in enumerate(feats):
        print(f"Stage {i}: {f.shape}")