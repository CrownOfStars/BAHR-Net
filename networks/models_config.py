import os
import yaml
from yacs.config import CfgNode as CN
import argparse
import json

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

_C.DATA.TEXTURE = None
# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.BACKBONES = 'segswin-base'
_C.MODEL.MFUSION = 'HAIM'
_C.MODEL.SFUSION = 'PATM-BAB'


_C.RGBSTREAM = CN()

# Model type
_C.RGBSTREAM.TYPE = 'swin'
# Model nickname
_C.RGBSTREAM.NICKNAME = 'swin-base'
# Model name
_C.RGBSTREAM.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.RGBSTREAM.PRETRAINED = True
# Checkpoint to resume, could be overwritten by command line argument
_C.RGBSTREAM.RESUME = ''
# Number of classes, overwritten in data preparation
_C.RGBSTREAM.NUM_CLASSES = 21841
# Dropout rate
_C.RGBSTREAM.DROP_RATE = 0.0
# Drop path rate
_C.RGBSTREAM.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.RGBSTREAM.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.RGBSTREAM.SWIN = CN()
_C.RGBSTREAM.SWIN.PATCH_SIZE = 4
_C.RGBSTREAM.SWIN.IN_CHANS = 3
_C.RGBSTREAM.SWIN.EMBED_DIM = 96
_C.RGBSTREAM.SWIN.DEPTHS = [2, 2, 6, 2]
_C.RGBSTREAM.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.RGBSTREAM.SWIN.WINDOW_SIZE = 7
_C.RGBSTREAM.SWIN.MLP_RATIO = 4.
_C.RGBSTREAM.SWIN.QKV_BIAS = True
_C.RGBSTREAM.SWIN.QK_SCALE = None
_C.RGBSTREAM.SWIN.APE = False
_C.RGBSTREAM.SWIN.PATCH_NORM = True



# Swin Transformer V2 parameters
_C.RGBSTREAM.SWINV2 = CN()
_C.RGBSTREAM.SWINV2.PATCH_SIZE = 4
_C.RGBSTREAM.SWINV2.IN_CHANS = 3
_C.RGBSTREAM.SWINV2.EMBED_DIM = 96
_C.RGBSTREAM.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.RGBSTREAM.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.RGBSTREAM.SWINV2.WINDOW_SIZE = 7
_C.RGBSTREAM.SWINV2.MLP_RATIO = 4.
_C.RGBSTREAM.SWINV2.QKV_BIAS = True
_C.RGBSTREAM.SWINV2.APE = False
_C.RGBSTREAM.SWINV2.PATCH_NORM = True
_C.RGBSTREAM.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Swin Transformer MoE parameters
_C.RGBSTREAM.SWIN_MOE = CN()
_C.RGBSTREAM.SWIN_MOE.PATCH_SIZE = 4
_C.RGBSTREAM.SWIN_MOE.IN_CHANS = 3
_C.RGBSTREAM.SWIN_MOE.EMBED_DIM = 96
_C.RGBSTREAM.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.RGBSTREAM.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.RGBSTREAM.SWIN_MOE.WINDOW_SIZE = 7
_C.RGBSTREAM.SWIN_MOE.MLP_RATIO = 4.
_C.RGBSTREAM.SWIN_MOE.QKV_BIAS = True
_C.RGBSTREAM.SWIN_MOE.QK_SCALE = None
_C.RGBSTREAM.SWIN_MOE.APE = False
_C.RGBSTREAM.SWIN_MOE.PATCH_NORM = True
_C.RGBSTREAM.SWIN_MOE.MLP_FC2_BIAS = True
_C.RGBSTREAM.SWIN_MOE.INIT_STD = 0.02
_C.RGBSTREAM.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.RGBSTREAM.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.RGBSTREAM.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.RGBSTREAM.SWIN_MOE.TOP_VALUE = 1
_C.RGBSTREAM.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.RGBSTREAM.SWIN_MOE.COSINE_ROUTER = False
_C.RGBSTREAM.SWIN_MOE.NORMALIZE_GATE = False
_C.RGBSTREAM.SWIN_MOE.USE_BPR = True
_C.RGBSTREAM.SWIN_MOE.IS_GSHARD_LOSS = False
_C.RGBSTREAM.SWIN_MOE.GATE_NOISE = 1.0
_C.RGBSTREAM.SWIN_MOE.COSINE_ROUTER_DIM = 256
_C.RGBSTREAM.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
_C.RGBSTREAM.SWIN_MOE.MOE_DROP = 0.0
_C.RGBSTREAM.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01

# Swin MLP parameters
_C.RGBSTREAM.SWIN_MLP = CN()
_C.RGBSTREAM.SWIN_MLP.PATCH_SIZE = 4
_C.RGBSTREAM.SWIN_MLP.IN_CHANS = 3
_C.RGBSTREAM.SWIN_MLP.EMBED_DIM = 96
_C.RGBSTREAM.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.RGBSTREAM.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.RGBSTREAM.SWIN_MLP.WINDOW_SIZE = 7
_C.RGBSTREAM.SWIN_MLP.MLP_RATIO = 4.
_C.RGBSTREAM.SWIN_MLP.APE = False
_C.RGBSTREAM.SWIN_MLP.PATCH_NORM = True

# Pool parameters
_C.RGBSTREAM.POOL = CN()
_C.RGBSTREAM.POOL.PATCH_SIZE = 4
_C.RGBSTREAM.POOL.TYPE = "poolformer_s"
_C.RGBSTREAM.POOL.IN_CHANS = 3
_C.RGBSTREAM.POOL.EMBED_DIMS = [64, 128, 320, 512]
_C.RGBSTREAM.POOL.LAYERS = [4, 4, 12, 4]
_C.RGBSTREAM.POOL.MLP_RATIOS = [4, 4, 4, 4]
_C.RGBSTREAM.POOL.DEPTHS = [2, 2, 6, 2]
_C.RGBSTREAM.POOL.NUM_HEADS = [3, 6, 12, 24]
_C.RGBSTREAM.POOL.WINDOW_SIZE = 7
_C.RGBSTREAM.POOL.DOWNSAMPLES = [True, True, True, True]
_C.RGBSTREAM.POOL.APE = False
_C.RGBSTREAM.POOL.PATCH_NORM = True


# [SimMIM] Norm target during training
_C.RGBSTREAM.SIMMIM = CN()
_C.RGBSTREAM.SIMMIM.NORM_TARGET = CN()
_C.RGBSTREAM.SIMMIM.NORM_TARGET.ENABLE = False
_C.RGBSTREAM.SIMMIM.NORM_TARGET.PATCH_SIZE = 47


# [ResNet]

_C.RGBSTREAM.RESNET = CN()
_C.RGBSTREAM.RESNET.LAYERS = [3, 8, 36, 3]
_C.RGBSTREAM.RESNET.TYPE = "resnet-152"
_C.RGBSTREAM.RESNET.EMBED_DIMS = [256, 512, 1024, 2048]

# [MOBILENet]

_C.RGBSTREAM.MOBILE = CN()
_C.RGBSTREAM.MOBILE.TYPE = "MobileNetV2"
_C.RGBSTREAM.MOBILE.EMBED_DIMS = [24, 32, 96, 320]

_C.RGBSTREAM.CONVNEXT = CN()
_C.RGBSTREAM.CONVNEXT.TYPE = "ConvNeXT"
_C.RGBSTREAM.CONVNEXT.DEPTHS = [3, 3, 27, 3]
_C.RGBSTREAM.CONVNEXT.EMBED_DIMS = [128, 256, 512, 1024]

_C.RGBSTREAM.SEGSWIN = CN()
_C.RGBSTREAM.SEGSWIN.TYPE = "SegSwin"
_C.RGBSTREAM.SEGSWIN.EMBED_DIMS = [128, 256, 512, 1024]

_C.RGBSTREAM.SEGRESNET = CN()
_C.RGBSTREAM.SEGRESNET.TYPE = "SegResNet"
_C.RGBSTREAM.SEGRESNET.EMBED_DIMS = [256, 512, 1024, 2048]
_C.RGBSTREAM.SEGRESNET.REPLACE = [False, False, True]

_C.XSTREAM = CN()

# Model type
_C.XSTREAM.TYPE = 'swin'
# Model nickname
_C.XSTREAM.NICKNAME = 'swin-base-224'
# Model name
_C.XSTREAM.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.XSTREAM.PRETRAINED = True
# Checkpoint to resume, could be overwritten by command line argument
_C.XSTREAM.RESUME = ''
# Number of classes, overwritten in data preparation
_C.XSTREAM.NUM_CLASSES = 21841
# Dropout rate
_C.XSTREAM.DROP_RATE = 0.0
# Drop path rate
_C.XSTREAM.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.XSTREAM.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.XSTREAM.SWIN = CN()
_C.XSTREAM.SWIN.PATCH_SIZE = 4
_C.XSTREAM.SWIN.IN_CHANS = 3
_C.XSTREAM.SWIN.EMBED_DIM = 96
_C.XSTREAM.SWIN.DEPTHS = [2, 2, 6, 2]
_C.XSTREAM.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.XSTREAM.SWIN.WINDOW_SIZE = 7
_C.XSTREAM.SWIN.MLP_RATIO = 4.
_C.XSTREAM.SWIN.QKV_BIAS = True
_C.XSTREAM.SWIN.QK_SCALE = None
_C.XSTREAM.SWIN.APE = False
_C.XSTREAM.SWIN.PATCH_NORM = True


# Swin Transformer V2 parameters
_C.XSTREAM.SWINV2 = CN()
_C.XSTREAM.SWINV2.PATCH_SIZE = 4
_C.XSTREAM.SWINV2.IN_CHANS = 3
_C.XSTREAM.SWINV2.EMBED_DIM = 96
_C.XSTREAM.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.XSTREAM.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.XSTREAM.SWINV2.WINDOW_SIZE = 7
_C.XSTREAM.SWINV2.MLP_RATIO = 4.
_C.XSTREAM.SWINV2.QKV_BIAS = True
_C.XSTREAM.SWINV2.APE = False
_C.XSTREAM.SWINV2.PATCH_NORM = True
_C.XSTREAM.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Swin Transformer MoE parameters
_C.XSTREAM.SWIN_MOE = CN()
_C.XSTREAM.SWIN_MOE.PATCH_SIZE = 4
_C.XSTREAM.SWIN_MOE.IN_CHANS = 3
_C.XSTREAM.SWIN_MOE.EMBED_DIM = 96
_C.XSTREAM.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.XSTREAM.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.XSTREAM.SWIN_MOE.WINDOW_SIZE = 7
_C.XSTREAM.SWIN_MOE.MLP_RATIO = 4.
_C.XSTREAM.SWIN_MOE.QKV_BIAS = True
_C.XSTREAM.SWIN_MOE.QK_SCALE = None
_C.XSTREAM.SWIN_MOE.APE = False
_C.XSTREAM.SWIN_MOE.PATCH_NORM = True
_C.XSTREAM.SWIN_MOE.MLP_FC2_BIAS = True
_C.XSTREAM.SWIN_MOE.INIT_STD = 0.02
_C.XSTREAM.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.XSTREAM.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.XSTREAM.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.XSTREAM.SWIN_MOE.TOP_VALUE = 1
_C.XSTREAM.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.XSTREAM.SWIN_MOE.COSINE_ROUTER = False
_C.XSTREAM.SWIN_MOE.NORMALIZE_GATE = False
_C.XSTREAM.SWIN_MOE.USE_BPR = True
_C.XSTREAM.SWIN_MOE.IS_GSHARD_LOSS = False
_C.XSTREAM.SWIN_MOE.GATE_NOISE = 1.0
_C.XSTREAM.SWIN_MOE.COSINE_ROUTER_DIM = 256
_C.XSTREAM.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
_C.XSTREAM.SWIN_MOE.MOE_DROP = 0.0
_C.XSTREAM.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01

# Swin MLP parameters
_C.XSTREAM.SWIN_MLP = CN()
_C.XSTREAM.SWIN_MLP.PATCH_SIZE = 4
_C.XSTREAM.SWIN_MLP.IN_CHANS = 3
_C.XSTREAM.SWIN_MLP.EMBED_DIM = 96
_C.XSTREAM.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.XSTREAM.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.XSTREAM.SWIN_MLP.WINDOW_SIZE = 7
_C.XSTREAM.SWIN_MLP.MLP_RATIO = 4.
_C.XSTREAM.SWIN_MLP.APE = False
_C.XSTREAM.SWIN_MLP.PATCH_NORM = True

# Pool parameters
_C.XSTREAM.POOL = CN()
_C.XSTREAM.POOL.PATCH_SIZE = 4
_C.XSTREAM.POOL.TYPE = "poolformer_s"
_C.XSTREAM.POOL.IN_CHANS = 3
_C.XSTREAM.POOL.EMBED_DIMS = [64, 128, 320, 512]
_C.XSTREAM.POOL.LAYERS = [4, 4, 12, 4]
_C.XSTREAM.POOL.MLP_RATIOS = [4, 4, 4, 4]
_C.XSTREAM.POOL.DEPTHS = [2, 2, 6, 2]
_C.XSTREAM.POOL.NUM_HEADS = [3, 6, 12, 24]
_C.XSTREAM.POOL.WINDOW_SIZE = 7
_C.XSTREAM.POOL.DOWNSAMPLES = [True, True, True, True]
_C.XSTREAM.POOL.APE = False
_C.XSTREAM.POOL.PATCH_NORM = True


# [SimMIM] Norm target during training
_C.XSTREAM.SIMMIM = CN()
_C.XSTREAM.SIMMIM.NORM_TARGET = CN()
_C.XSTREAM.SIMMIM.NORM_TARGET.ENABLE = False
_C.XSTREAM.SIMMIM.NORM_TARGET.PATCH_SIZE = 47


# [ResNet]

_C.XSTREAM.RESNET = CN()
_C.XSTREAM.RESNET.LAYERS = [3, 8, 36, 3]
_C.XSTREAM.RESNET.TYPE = "resnet-152"
_C.XSTREAM.RESNET.EMBED_DIMS = [256, 512, 1024, 2048]

# [MOBILENet]

_C.XSTREAM.MOBILE = CN()
_C.XSTREAM.MOBILE.TYPE = "MobileNetV2"
_C.XSTREAM.MOBILE.EMBED_DIMS = [24, 32, 96, 320]

_C.XSTREAM.CONVNEXT = CN()
_C.XSTREAM.CONVNEXT.TYPE = "ConvNeXT"
_C.XSTREAM.CONVNEXT.DEPTHS = [3, 3, 27, 3]
_C.XSTREAM.CONVNEXT.EMBED_DIMS = [128, 256, 512, 1024]

_C.XSTREAM.SEGSWIN = CN()
_C.XSTREAM.SEGSWIN.TYPE = "SegSwin"
_C.XSTREAM.SEGSWIN.EMBED_DIMS = [128, 256, 512, 1024]

_C.XSTREAM.SEGRESNET = CN()
_C.XSTREAM.SEGRESNET.TYPE = "SegResNet"
_C.XSTREAM.SEGRESNET.EMBED_DIMS = [256, 512, 1024, 2048]
_C.XSTREAM.SEGRESNET.REPLACE = [False, False, True]


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False


cfg_dict = {
    "wavemlp-s":["./pretrained/WaveMLP_S.pth.tar","./pretrained/configs/wavemlp/wavemlp_m_224.yaml"],
    "wavemlp-m":["./pretrained/WaveMLP_M.pth.tar","./pretrained/configs/wavemlp/wavemlp_s_224.yaml"],
    "resnet-50":["./pretrained/resnet50-19c8e357.pth","./pretrained/configs/resnet/resnet50.yaml"],
    "resnet-101":["./pretrained/resnet101-5d3b4d8f.pth","./pretrained/configs/resnet/resnet101.yaml"],
    "resnet-152":["./pretrained/resnet152-b121ed2d.pth","./pretrained/configs/resnet/resnet152.yaml"],
    "swin-tiny":["./pretrained/swin_tiny_patch4_window7_224_22k.pth","./pretrained/configs/swin/swin_small_patch4_window7_224_22k.yaml"],
    "swin-small":["./pretrained/swin_small_patch4_window7_224_22k.pth","./pretrained/configs/swin/swin_small_patch4_window7_224_22k.yaml"],
    "swin-base":["./pretrained/swin_base_patch4_window7_224_22k.pth","./pretrained/configs/swin/swin_base_patch4_window7_224_22k.yaml"],#batch_size 24
    "swin-large":["./pretrained/swin_large_patch4_window7_224_22k.pth","./pretrained/configs/swin/swin_large_patch4_window7_224_22k.yaml"],
    "swin-base-384":["./pretrained/swin_base_patch4_window12_384_22k.pth","./pretrained/configs/swin/swin_base_patch4_window12_384_22kto1k_finetune.yaml"],
    "swinv2-base":["./pretrained/swinv2_base_patch4_window12_192_22k.pth","./pretrained/configs/swinv2/swinv2_base_patch4_window12_192_22k.yaml"],
    "swinmlp-base":["./pretrained/swin_mlp_base_patch4_window7_224.pth","./pretrained/configs/swinmlp/swin_mlp_base_patch4_window7_224.yaml"],
    "pool-s24":["./pretrained/ckpts/pool/poolformer_s24.pth","./pretrained/configs/pool/pool_s24_patch4_224.yaml"],
    "pool-s36":["./pretrained/poolformer_s36.pth","./pretrained/configs/pool/pool_s36_patch4_224.yaml"],
    "pool-m36":["./pretrained/poolformer_m36.pth","./pretrained/configs/pool/pool_m36_patch4_224.yaml"],
    "pool-m48":["./pretrained/poolformer_m48.pth","./pretrained/configs/pool/pool_m48_patch4_224.yaml"],
    "mobilenetv2":["./pretrained/mobilenet_v2-b0353104.pth","./pretrained/configs/mobilenet/mobilenetv2.yaml"],
    "convnext-base":["./pretrained/convnext_base_22k_224.pth","./pretrained/configs/convnext/convnext.yaml"],
    "segswin-base":["./pretrained/ckpts/swin/swin_base_patch4_window12_384_22k.pth","./pretrained/configs/segswin/swin_base_patch4_window12_384_22kto1k_finetune.yaml"],
    "segswin-small":["./pretrained/ckpts/swin/swin_small_patch4_window7_224_22k.pth","./pretrained/configs/segswin/swin_small_patch4_window7_224_22k.yaml"],
    "segresnet-101":["./pretrained/resnet101.pth","./pretrained/configs/segresnet/segresnet-101.yaml"]
}

"""
    model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    }
"""

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    new_cfg = yaml_cfg
    

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    
    print('=> merge config from {}'.format(cfg_file))
    
    #config.merge_from_other_cfg(cfg_other)
    
    #config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    if len(args.backbone) == 1:
        assert args.backbone[0] in cfg_dict.keys(),cfg_dict.keys()
        cfg_file = cfg_dict[args.backbone[0]][1]
        print('=> merge config from {}'.format(cfg_file))
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        new_cfg = {}

        for k,v in yaml_cfg.items():
            if k == 'MODEL':
                new_cfg['XSTREAM'] = v
                new_cfg['RGBSTREAM'] = v
            else:
                new_cfg[k] = v

        config.merge_from_other_cfg(CN(new_cfg))
    elif len(args.backbone) == 2:
        assert args.backbone[0] in cfg_dict.keys(),cfg_dict.keys()
        cfg_file = cfg_dict[args.backbone[1]][1]
        print('=> merge config from {}'.format(cfg_file))
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        new_cfg = {}
        for k,v in yaml_cfg.items():
            if k == 'MODEL':
                new_cfg['XSTREAM'] = v
            else:
                new_cfg[k] = v
                
        assert args.backbone[1] in cfg_dict.keys(),cfg_dict.keys()
        cfg_file = cfg_dict[args.backbone[0]][1]
        print('=> merge config from {}'.format(cfg_file))
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        for k,v in yaml_cfg.items():
            if k == 'MODEL':
                new_cfg['RGBSTREAM'] = v
            else:
                new_cfg[k] = v
        
        config.merge_from_other_cfg(CN(new_cfg))
    else:
        raise NotImplementedError("only support two stream")
    
    config.defrost()
    if args.opts:
        
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True
    if _check_args('mfusion'):
        config.MODEL.MFUSION = args.mfusion
    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    #config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    return config


def parse_option(mode = "train"):
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--backbone', type=str,nargs='+', help='path to config file')
    parser.add_argument('--mfusion', type=str, default= 'HAIM' )    
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--warmup_epoch', type=int, default=30)
    parser.add_argument('--max_epoch', type=int, default=80, help='max epoch number')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--clip', type=float, default=0.8, help='gradient clipping margin')
    parser.add_argument('--train_batch', type=int, default=16, help='training batch size')
    parser.add_argument('--decay_epoch', type=int, default=10, help='every n epochs decay learning rate')
    parser.add_argument('--gamma',type=float,default=0.5)
    parser.add_argument('--gpu_id', type=str, default='1', help='select gpu id')
    parser.add_argument('--train_root', type=str, default='./dataset/ISOD_dataset/train/', help='the train images root')
    parser.add_argument('--val_root', type=str, default='./dataset/ISOD_dataset/val/', help='the val images root')


    parser.add_argument('--log_path', type=str, default='./log/', help='the path to save models and logs')

    parser.add_argument('--img_size', type=int,default=0)

    parser.add_argument('--texture',choices=[None,'/namlab20/','/namlab25/','/namlab30/','/namlab40/','/namlab50/','/namlab60/','/bound/','/teed/','/cats/','/bound/'])
    parser.add_argument('--test_model',type=str,required=False)
    parser.add_argument('--save_path', type=str, default='./save/', help='the path to save images')
    
    parser.add_argument('--test_batch',type=int,default=16)
    parser.add_argument('--test_path',type=str,default='./dataset/ISOD_dataset/test/',help='test dataset path')
    parser.add_argument('--save_result',type=bool,default=True)
    
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')

    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local-rank", type=int, default= -1, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')


    args = parser.parse_args()
    gpu_id = args.gpu_id
    test_path = args.test_path
    if args.test_model and mode == "test":
        test_model = args.test_model
        args_dict = vars(args)
        with open(test_model+"/args.json", mode="r") as f:
            args_dict.update(json.load(f))
            args.test_model = test_model
            config = get_config(args)
            
            _update_config_from_file(config, args.test_model+"/config.yaml")
    elif args.test_model and mode == "train":
        raise NotImplementedError()
    else:
        config = get_config(args)
        print(type(config))
        config.dump(stream=open(os.path.join("./config","default", "config.yaml"), "w"))

    args.gpu_id = gpu_id
    args.test_path = test_path
    config.defrost()
    config.MODEL.DROP_PATH_RATE = 0.5
    config.RGBSTREAM.NICKNAME = args.backbone[0]
    config.XSTREAM.NICKNAME = args.backbone[len(args.backbone)>1]
    config.DATA.TEXTURE = args.texture
    config.MODEL.MFUSION = args.mfusion
    if args.img_size:
        config.DATA.IMG_SIZE = args.img_size
    config.freeze()

    return args, config
