# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from networks.PoolNets import PoolFormer
from networks.ResNets import ResNet,Bottleneck
from networks.swin_transformer import SwinTransformer
from networks.segswin import SegSwinTransformer
from networks.swin_mlp import SwinMLP
from networks.swin_transformer_v2 import SwinTransformerV2
from networks.mobilenetv2 import mobilenet_v2
from networks.models_config import cfg_dict
from networks.wavemlp import *
from networks.ConvNeXt import ConvNeXt
import timm

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'poolformer_s': _cfg(crop_pct=0.9),
    'poolformer_m': _cfg(crop_pct=0.95),
}


def safe_load_model(model,state_dict,ckpt_path):
    
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(ckpt_path, msg))


def build_Rbackbone(config):
    model_type = config.RGBSTREAM.TYPE

    pretrained = config.RGBSTREAM.PRETRAINED

    embed_dims = []
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.RGBSTREAM.SWIN.PATCH_SIZE,
                                in_chans=config.RGBSTREAM.SWIN.IN_CHANS,
                                num_classes=config.RGBSTREAM.NUM_CLASSES,
                                embed_dim=config.RGBSTREAM.SWIN.EMBED_DIM,
                                depths=config.RGBSTREAM.SWIN.DEPTHS,
                                num_heads=config.RGBSTREAM.SWIN.NUM_HEADS,
                                window_size=config.RGBSTREAM.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.RGBSTREAM.SWIN.MLP_RATIO,
                                qkv_bias=config.RGBSTREAM.SWIN.QKV_BIAS,
                                qk_scale=config.RGBSTREAM.SWIN.QK_SCALE,
                                drop_rate=config.RGBSTREAM.DROP_RATE,
                                drop_path_rate=config.RGBSTREAM.DROP_PATH_RATE,
                                ape=config.RGBSTREAM.SWIN.APE,
                                norm_layer=nn.LayerNorm,
                                patch_norm=config.RGBSTREAM.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
        if pretrained:
            ckpt_path = cfg_dict[config.RGBSTREAM.NICKNAME][0]
            state_dict = torch.load(ckpt_path,map_location="cpu")
            safe_load_model(model,state_dict['model'],ckpt_path)
        embed_dims = [config.RGBSTREAM.SWIN.EMBED_DIM*2**i for i in range(4)]

    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.RGBSTREAM.SWINV2.PATCH_SIZE,
                                  in_chans=config.RGBSTREAM.SWINV2.IN_CHANS,
                                  num_classes=config.RGBSTREAM.NUM_CLASSES,
                                  embed_dim=config.RGBSTREAM.SWINV2.EMBED_DIM,
                                  depths=config.RGBSTREAM.SWINV2.DEPTHS,
                                  num_heads=config.RGBSTREAM.SWINV2.NUM_HEADS,
                                  window_size=config.RGBSTREAM.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.RGBSTREAM.SWINV2.MLP_RATIO,
                                  qkv_bias=config.RGBSTREAM.SWINV2.QKV_BIAS,
                                  drop_rate=config.RGBSTREAM.DROP_RATE,
                                  drop_path_rate=config.RGBSTREAM.DROP_PATH_RATE,
                                  ape=config.RGBSTREAM.SWINV2.APE,
                                  patch_norm=config.RGBSTREAM.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.RGBSTREAM.SWINV2.PRETRAINED_WINDOW_SIZES)
        if pretrained:
            ckpt_path = cfg_dict[config.RGBSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt['model'],strict=False)
        embed_dims = [config.RGBSTREAM.SWINV2.EMBED_DIM*2**i for i in range(4)]
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.RGBSTREAM.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.RGBSTREAM.SWIN_MLP.IN_CHANS,
                        num_classes=config.RGBSTREAM.NUM_CLASSES,
                        embed_dim=config.RGBSTREAM.SWIN_MLP.EMBED_DIM,
                        depths=config.RGBSTREAM.SWIN_MLP.DEPTHS,
                        num_heads=config.RGBSTREAM.SWIN_MLP.NUM_HEADS,
                        window_size=config.RGBSTREAM.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.RGBSTREAM.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.RGBSTREAM.DROP_RATE,
                        drop_path_rate=config.RGBSTREAM.DROP_PATH_RATE,
                        ape=config.RGBSTREAM.SWIN_MLP.APE,
                        patch_norm=config.RGBSTREAM.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        if pretrained:
            ckpt_path = cfg_dict[config.RGBSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt['model'],strict=False)
        embed_dims = [config.RGBSTREAM.SWIN_MLP.EMBED_DIM*2**i for i in range(4)]
    elif model_type == "pool":
        model = PoolFormer(config.RGBSTREAM.POOL.LAYERS, embed_dims=config.RGBSTREAM.POOL.EMBED_DIMS, 
            mlp_ratios=config.RGBSTREAM.POOL.MLP_RATIOS, downsamples=config.RGBSTREAM.POOL.DOWNSAMPLES, 
            layer_scale_init_value=1e-6, 
            fork_feat=True)
        model.default_cfg = default_cfgs[config.RGBSTREAM.POOL.TYPE]
        if pretrained:
            ckpt_path = cfg_dict[config.RGBSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt['model'],strict=False)
        embed_dims = config.RGBSTREAM.POOL.EMBED_DIMS
    elif model_type == "resnet":
        model_type = config.RGBSTREAM.TYPE
        pretrained = config.RGBSTREAM.PRETRAINED
        model = ResNet(Bottleneck, config.RGBSTREAM.RESNET.LAYERS)
        if pretrained:
            ckpt_path = cfg_dict[config.RGBSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt,strict=False)

        embed_dims = config.RGBSTREAM.RESNET.EMBED_DIMS
    elif model_type == "mobilenet":
        model_type = config.RGBSTREAM.TYPE
        pretrained = config.RGBSTREAM.PRETRAINED
        model = mobilenet_v2(False)
        if pretrained:
            ckpt_path = cfg_dict[config.RGBSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt,strict=False)

        embed_dims = config.RGBSTREAM.MOBILE.EMBED_DIMS
    elif model_type == "wavemlp":
        model_type = config.RGBSTREAM.TYPE
        pretrained = config.RGBSTREAM.PRETRAINED
        model = WaveMLP_M(False)
        if pretrained:
            ckpt_path = cfg_dict[config.RGBSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt,strict=False)

        embed_dims = config.RGBSTREAM.WAVEMLP.EMBED_DIMS
    elif model_type == "convnext":
        model_type = config.RGBSTREAM.TYPE
        pretrained = config.RGBSTREAM.PRETRAINED
        embed_dims = config.RGBSTREAM.CONVNEXT.EMBED_DIMS
        model = ConvNeXt(num_classes=21841,depths=config.RGBSTREAM.CONVNEXT.DEPTHS, dims=embed_dims)
        if pretrained:
            ckpt_path = cfg_dict[config.RGBSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt["model"])
    elif model_type == 'segswin':
        model = SegSwinTransformer(pretrain_img_size=config.DATA.IMG_SIZE,
                                patch_size=config.RGBSTREAM.SWIN.PATCH_SIZE,
                                in_chans=config.RGBSTREAM.SWIN.IN_CHANS,
                                embed_dim=config.RGBSTREAM.SWIN.EMBED_DIM,
                                depths=config.RGBSTREAM.SWIN.DEPTHS,
                                num_heads=config.RGBSTREAM.SWIN.NUM_HEADS,
                                window_size=config.RGBSTREAM.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.RGBSTREAM.SWIN.MLP_RATIO,
                                qkv_bias=config.RGBSTREAM.SWIN.QKV_BIAS,
                                qk_scale=config.RGBSTREAM.SWIN.QK_SCALE,
                                attn_drop_rate=config.RGBSTREAM.DROP_RATE,
                                drop_path_rate=config.RGBSTREAM.DROP_PATH_RATE,
                                norm_layer=nn.LayerNorm,
                                ape=config.RGBSTREAM.SWIN.APE,
                                patch_norm=config.RGBSTREAM.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        )

        if pretrained:
            ckpt_path = cfg_dict[config.RGBSTREAM.NICKNAME][0]
            state_dict = torch.load(ckpt_path,map_location="cpu")
            safe_load_model(model,state_dict['model'],ckpt_path)
        embed_dims = [config.RGBSTREAM.SWIN.EMBED_DIM*2**i for i in range(4)]
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    
    return model,embed_dims

def build_RGB_backbone(config):

    arch_name = config.RGBSTREAM.ARCH
    pretrained = config.RGBSTREAM.PRETRAINED
    archs = {
    "swin": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "swinv2": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    "pool": "poolformer_m48.sail_in1k",
    "convnext": "convnext_base.fb_in22k_ft_in1k_384",
    "resnet50": "resnet50.a1_in1k",
    "maxvit": "maxvit_base_tf_384.in21k_ft_in1k"
    }

    model = timm.create_model(
    archs[arch_name],
    pretrained=pretrained,
    features_only=True,
    )
    return model




def build_Xbackbone(config):
    model_type = config.XSTREAM.TYPE
    
    pretrained = config.XSTREAM.PRETRAINED
    
    embed_dims = []
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.XSTREAM.SWIN.PATCH_SIZE,
                                in_chans=config.XSTREAM.SWIN.IN_CHANS,
                                num_classes=config.XSTREAM.NUM_CLASSES,
                                embed_dim=config.XSTREAM.SWIN.EMBED_DIM,
                                depths=config.XSTREAM.SWIN.DEPTHS,
                                num_heads=config.XSTREAM.SWIN.NUM_HEADS,
                                window_size=config.XSTREAM.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.XSTREAM.SWIN.MLP_RATIO,
                                qkv_bias=config.XSTREAM.SWIN.QKV_BIAS,
                                qk_scale=config.XSTREAM.SWIN.QK_SCALE,
                                drop_rate=config.XSTREAM.DROP_RATE,
                                drop_path_rate=config.XSTREAM.DROP_PATH_RATE,
                                ape=config.XSTREAM.SWIN.APE,
                                norm_layer=nn.LayerNorm,
                                patch_norm=config.XSTREAM.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
        if pretrained:
            ckpt_path = cfg_dict[config.XSTREAM.NICKNAME][0]
            state_dict = torch.load(ckpt_path,map_location="cpu")
            safe_load_model(model,state_dict['model'],ckpt_path)
        embed_dims = [config.XSTREAM.SWIN.EMBED_DIM*2**i for i in range(4)]

    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.XSTREAM.SWINV2.PATCH_SIZE,
                                  in_chans=config.XSTREAM.SWINV2.IN_CHANS,
                                  num_classes=config.XSTREAM.NUM_CLASSES,
                                  embed_dim=config.XSTREAM.SWINV2.EMBED_DIM,
                                  depths=config.XSTREAM.SWINV2.DEPTHS,
                                  num_heads=config.XSTREAM.SWINV2.NUM_HEADS,
                                  window_size=config.XSTREAM.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.XSTREAM.SWINV2.MLP_RATIO,
                                  qkv_bias=config.XSTREAM.SWINV2.QKV_BIAS,
                                  drop_rate=config.XSTREAM.DROP_RATE,
                                  drop_path_rate=config.XSTREAM.DROP_PATH_RATE,
                                  ape=config.XSTREAM.SWINV2.APE,
                                  patch_norm=config.XSTREAM.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.XSTREAM.SWINV2.PRETRAINED_WINDOW_SIZES)
        if pretrained:
            ckpt_path = cfg_dict[config.XSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt['model'],strict=False)
        embed_dims = [config.XSTREAM.SWINV2.EMBED_DIM*2**i for i in range(4)]
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.XSTREAM.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.XSTREAM.SWIN_MLP.IN_CHANS,
                        num_classes=config.XSTREAM.NUM_CLASSES,
                        embed_dim=config.XSTREAM.SWIN_MLP.EMBED_DIM,
                        depths=config.XSTREAM.SWIN_MLP.DEPTHS,
                        num_heads=config.XSTREAM.SWIN_MLP.NUM_HEADS,
                        window_size=config.XSTREAM.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.XSTREAM.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.XSTREAM.DROP_RATE,
                        drop_path_rate=config.XSTREAM.DROP_PATH_RATE,
                        ape=config.XSTREAM.SWIN_MLP.APE,
                        patch_norm=config.XSTREAM.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        if pretrained:
            ckpt_path = cfg_dict[config.XSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt['model'],strict=False)
        embed_dims = [config.XSTREAM.SWIN_MLP.EMBED_DIM*2**i for i in range(4)]
    elif model_type == "pool":
        model = PoolFormer(config.XSTREAM.POOL.LAYERS, embed_dims=config.XSTREAM.POOL.EMBED_DIMS, 
            mlp_ratios=config.XSTREAM.POOL.MLP_RATIOS, downsamples=config.XSTREAM.POOL.DOWNSAMPLES, 
            layer_scale_init_value=1e-6, 
            fork_feat=True)
        model.default_cfg = default_cfgs[config.XSTREAM.POOL.TYPE]
        if pretrained:
            ckpt_path = cfg_dict[config.XSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt['model'],strict=False)
        embed_dims = config.XSTREAM.POOL.EMBED_DIMS
    elif model_type == "resnet":
        model_type = config.XSTREAM.TYPE
        pretrained = config.XSTREAM.PRETRAINED
        model = ResNet(Bottleneck, config.XSTREAM.RESNET.LAYERS)
        if pretrained:
            ckpt_path = cfg_dict[config.XSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt,strict=False)

        embed_dims = config.XSTREAM.RESNET.EMBED_DIMS
    elif model_type == "mobilenet":
        model_type = config.XSTREAM.TYPE
        pretrained = config.XSTREAM.PRETRAINED
        model = mobilenet_v2(False)
        if pretrained:
            ckpt_path = cfg_dict[config.XSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt,strict=False)

        embed_dims = config.XSTREAM.MOBILE.EMBED_DIMS
    elif model_type == "wavemlp":
        model_type = config.XSTREAM.TYPE
        pretrained = config.XSTREAM.PRETRAINED
        model = WaveMLP_M(False)
        if pretrained:
            ckpt_path = cfg_dict[config.XSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt,strict=False)

        embed_dims = config.XSTREAM.WAVEMLP.EMBED_DIMS
    elif model_type == 'segswin':
        model_type = config.XSTREAM.TYPE

        model = SegSwinTransformer(pretrain_img_size=config.DATA.IMG_SIZE,
                                patch_size=config.XSTREAM.SWIN.PATCH_SIZE,
                                in_chans=config.XSTREAM.SWIN.IN_CHANS,
                                embed_dim=config.XSTREAM.SWIN.EMBED_DIM,
                                depths=config.XSTREAM.SWIN.DEPTHS,
                                num_heads=config.XSTREAM.SWIN.NUM_HEADS,
                                window_size=config.XSTREAM.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.XSTREAM.SWIN.MLP_RATIO,
                                qkv_bias=config.XSTREAM.SWIN.QKV_BIAS,
                                qk_scale=config.XSTREAM.SWIN.QK_SCALE,
                                attn_drop_rate=config.XSTREAM.DROP_RATE,
                                drop_path_rate=config.XSTREAM.DROP_PATH_RATE,
                                norm_layer=nn.LayerNorm,
                                ape=config.XSTREAM.SWIN.APE,
                                patch_norm=config.XSTREAM.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        )
        if pretrained:
            ckpt_path = cfg_dict[config.XSTREAM.NICKNAME][0]
            state_dict = torch.load(ckpt_path,map_location="cpu")
            safe_load_model(model,state_dict['model'],ckpt_path)
        embed_dims = [config.XSTREAM.SWIN.EMBED_DIM*2**i for i in range(4)]
    elif model_type == "convnext":
        model_type = config.RGBSTREAM.TYPE
        pretrained = config.RGBSTREAM.PRETRAINED
        embed_dims = config.RGBSTREAM.CONVNEXT.EMBED_DIMS
        model = ConvNeXt(num_classes=21841,depths=config.RGBSTREAM.CONVNEXT.DEPTHS, dims=embed_dims)
        if pretrained:
            ckpt_path = cfg_dict[config.RGBSTREAM.NICKNAME][0]
            ckpt = torch.load(ckpt_path,map_location="cpu")
            model.load_state_dict(ckpt["model"])
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    
    return model,embed_dims


if __name__ == "__main__":
    pass