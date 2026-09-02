import os

import torch
import torch.nn.functional as F

from networks.backbone.hiera.pvt_v2 import pvt_v2_b5
from networks.backbone.hiera.swin_transformer import swin_transformer_B
from networks.backbone.hiera.hieradet import Hiera
from utils.model_utils import safe_load_model


def _build_pvt_v2_b5():
    return pvt_v2_b5()


def _build_swin_b():
    return swin_transformer_B()


def _build_hiera_l():
    return Hiera(
        embed_dim=144,
        num_heads=2,
        stages=[2, 6, 36, 4],
        global_att_blocks=[23, 33, 43],
        window_pos_embed_bkg_spatial_size=[7, 7],
        window_spec=[8, 4, 16, 8],
    )


def _build_hiera_star_l():
    return Hiera(
        embed_dim=144,
        num_heads=2,
        stages=[2, 6, 36],
        global_att_blocks=[23, 33],
        window_pos_embed_bkg_spatial_size=[7, 7],
        window_spec=[8, 4, 16],
        q_pool=2,
    )


def _build_hiera_b():
    return Hiera(
        embed_dim=112,
        num_heads=2,
    )


def _build_hiera_star_b():
    return Hiera(
        embed_dim=112,
        num_heads=2,
        stages=[2, 3, 16],
        global_att_blocks=[12, 16],
        window_pos_embed_bkg_spatial_size=[14, 14],
        window_spec=[8, 4, 11],
        q_pool=2,
    )


# 各 variant 对应的 embed_dims [C1, C2, C3, C4] 或 [C1, C2, C3]
_HIERA_EMBED_DIMS = {
    "pvt_v2_b5": [64, 128, 320, 512],
    "swin_b": [128, 256, 512, 1024],
    "hiera_l": [144, 288, 576, 1152],
    "hiera*_l": [144, 288, 576],
    "hiera_b": [112, 224, 448, 896],
    "hiera*_b": [112, 224, 448],
}

_BUILDERS = {
    "pvt_v2_b5": _build_pvt_v2_b5,
    "swin_b": _build_swin_b,
    "hiera_l": _build_hiera_l,
    "hiera*_l": _build_hiera_star_l,
    "hiera_b": _build_hiera_b,
    "hiera*_b": _build_hiera_star_b,
}


def _load_hiera_weights(model, state_dict):
    """Hiera 预训练权重需移除 image_encoder.trunk. 前缀"""
    model_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("image_encoder.trunk.")
        model_state_dict[new_key] = value
    model.load_state_dict(model_state_dict, strict=False)

def _load_swin_weights(model, state_dict):
    """Swin 预训练权重需处理 attn_mask、absolute_pos_embed、relative_position_bias_table"""
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    for k in list(state_dict.keys()):
        if ("attn.relative_position_index" in k) or ("attn_mask" in k):
            state_dict.pop(k)
    if state_dict.get("absolute_pos_embed") is not None:
        absolute_pos_embed = state_dict["absolute_pos_embed"]
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 == N2 and C1 == C2 and L == H * W:
            state_dict["absolute_pos_embed"] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)
    for table_key in [k for k in state_dict.keys() if "relative_position_bias_table" in k]:
        table_pretrained = state_dict[table_key]
        if table_key not in model.state_dict():
            continue
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 == nH2 and L1 != L2:
            S1 = int(L1**0.5)
            S2 = int(L2**0.5)
            table_pretrained_resized = F.interpolate(
                table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                size=(S2, S2),
                mode="bicubic",
            )
            state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
    model.load_state_dict(state_dict, strict=False)


def build_hiera(config, img_size, pt_root):
    """构建 Hiera 系列骨干网络，支持 pvt_v2_b5、swin_b、hiera_l、hiera*_l、hiera_b、hiera*_b。

    Args:
        config: 配置对象，需包含 config.BACKBONE.TYPE（variant）及 config.BACKBONE.CKPT_NAME。
        img_size: 输入图像尺寸（部分 backbone 可能使用）。
        pt_root: 预训练权重根目录，为 None 时不加载。

    Returns:
        model: 骨干网络模型。
        embed_dims: 各 stage 的 embed_dim 列表 [C1, C2, C3, C4] 或 [C1, C2, C3]。
    """
    variant = config.BACKBONE.TYPE
    if variant not in _BUILDERS:
        raise ValueError(f"Unknown backbone variant: {variant}. Supported: {list(_BUILDERS.keys())}")

    model = _BUILDERS[variant]()
    embed_dims = _HIERA_EMBED_DIMS[variant]

    if pt_root and hasattr(config.BACKBONE, "CKPT_NAME") and config.BACKBONE.CKPT_NAME:
        ckpt_name = config.BACKBONE.CKPT_NAME
        if hasattr(ckpt_name, "keys"):  # OmegaConf 有时解析为 dict
            ckpt_name = list(ckpt_name.keys())[0] if ckpt_name else None
        if ckpt_name:
            ckpt_path = os.path.join(pt_root, ckpt_name)
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            if "hiera" in variant:
                _load_hiera_weights(model, state_dict)
            elif "swin" in variant:
                _load_swin_weights(model, state_dict)
            elif "pvt" in variant:
                state_dict.pop("head.weight", None)
                state_dict.pop("head.bias", None)
                safe_load_model(model, state_dict, ckpt_path)
            else:
                safe_load_model(model, state_dict, ckpt_path)

    return model, embed_dims


# 兼容旧接口：build_backbone(backbone_name, pretrained=True) -> backbone only
def build_backbone(backbone_name, pretrained=True, weights_map=None):
    """兼容旧接口，仅返回 backbone，不返回 embed_dims。建议使用 build_hiera(config, img_size, pt_root)。

    Args:
        backbone_name: 如 pvt_v2_b5, swin_b, hiera_l 等。
        pretrained: 是否加载预训练权重。
        weights_map: 可选，{backbone_name: ckpt_path} 的映射；若为 None 则尝试从 config 获取。
    """
    if backbone_name not in _BUILDERS:
        raise ValueError(f"Unknown backbone: {backbone_name}. Supported: {list(_BUILDERS.keys())}")
    model = _BUILDERS[backbone_name]()
    if pretrained:
        ckpt_path = None
        if weights_map and backbone_name in weights_map:
            ckpt_path = weights_map[backbone_name]
        else:
            try:
                from config import Config
                cfg = Config()
                ckpt_path = getattr(cfg, "weights", {}).get(backbone_name)
            except Exception:
                pass
        if ckpt_path and os.path.isfile(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "hiera" in backbone_name:
                sd = state_dict.get("model", state_dict) if isinstance(state_dict, dict) else state_dict
                _load_hiera_weights(model, sd)
            elif "swin" in backbone_name:
                sd = state_dict.get("model", state_dict) if isinstance(state_dict, dict) else state_dict
                _load_swin_weights(model, sd)
            elif "pvt" in backbone_name:
                sd = state_dict.get("model", state_dict.get("state_dict", state_dict))
                sd = sd if isinstance(sd, dict) else state_dict
                if isinstance(sd, dict):
                    sd.pop("head.weight", None)
                    sd.pop("head.bias", None)
                    safe_load_model(model, sd, ckpt_path)
    return model
