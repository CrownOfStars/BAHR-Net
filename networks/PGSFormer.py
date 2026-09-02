import sys
sys.path.append('../')
sys.path.append('./')
from networks.PWNet import *
from networks.SwinNets import build_Rbackbone,build_Xbackbone,safe_load_model
from networks.models_config import parse_option
import numpy as np
from networks.MFusionToolBox import *
from networks.EdgeAwareToolBox import *
from networks.segswin import SegSwinTransformer
from networks.seg_swin_transformer import swin_tiny_patch4_window7_224,load_pretrained_weights
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.swin_transformer import *
from timm.models.helpers import build_model_with_cfg
import timm

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

class GSformer(nn.Module):
    def __init__(self,config):
        super(GSformer, self).__init__()


        self.encoderR, embed_dims = build_Rbackbone(config)

        self.encoderD, fused_dims = build_Xbackbone(config)

        input_size = config.DATA.IMG_SIZE

        self.FFT1, self.FFT2, self.FFT3, self.FFT4 = build_modilty_fusion(config.MODEL.MFUSION,embed_dims,fused_dims)

        self.S4 = nn.ConvTranspose2d(fused_dims[3], 1, 2, stride=2)
        self.S3 = nn.ConvTranspose2d(fused_dims[2], 1, 2, stride=2)
        self.S2 = nn.ConvTranspose2d(fused_dims[1], 1, 2, stride=2)
        self.S1 = nn.ConvTranspose2d(fused_dims[0], 1, 2, stride=2)
        

        self.up_loss = Interpolate(size=input_size//4)
        """
        图片分类预训练模型可以用作backbone,但是分类任务对边界并不敏感,加入边界提取可以提高模型迁移到显著性检测的效果
        """
        self.sod_edge_aware = Edge_Aware(fused_dims,input_size)
        self.rgb_edge_aware = Edge_Aware(embed_dims,input_size)
        self.depth_edge_aware = Edge_Aware(fused_dims,input_size)


        """
        (1024, 1024, 1024, 3, 2)
        (1536, 1024, 512, 3, 2)
        (768, 512, 256, 5, 3)
        (384, 256, 128, 5, 3)
        """

        # different scale feature fusion
        self.PATM4 = PATM_BAB(fused_dims[3], fused_dims[3], fused_dims[3], 3, 2)#TODO reduce params
        self.PATM3 = PATM_BAB(fused_dims[2]+fused_dims[3], fused_dims[3], fused_dims[2], 3, 2)
        self.PATM2 = PATM_BAB(fused_dims[1]+fused_dims[2], fused_dims[2], fused_dims[1], 5, 3)
        self.PATM1 = PATM_BAB(fused_dims[0]+fused_dims[1], fused_dims[1], fused_dims[0], 5, 3)
        

    def forward(self, rgb, x):
        x0,x1,x2,x3 = self.encoderR(rgb)        
        y0,y1,y2,y3 = self.encoderD(x)

        """
        [16, 128, 96, 96]
        [16, 256, 48, 48]
        [16, 512, 24, 24]
        [16, 1024, 12, 12 ]
        """

        edge_rgb = self.rgb_edge_aware(x0,x1,x2,x3)
        edge_depth = self.depth_edge_aware(y0,y1,y2,y3)
        
        #x0,x1,x2,x3 = self.ccpr0(x0),self.ccpr1(x1),self.ccpr2(x2),self.ccpr3(x3)

        x2_ACCoM = self.FFT1(x0, y0)
        x3_ACCoM = self.FFT2(x1, y1)
        x4_ACCoM = self.FFT3(x2, y2)
        x5_ACCoM = self.FFT4(x3, y3)
        
    
        edge_sod = self.sod_edge_aware(x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM)

        mer_cros4 = self.PATM4(x5_ACCoM)

        m4 = torch.cat((mer_cros4,x4_ACCoM),dim=1)
        mer_cros3 = self.PATM3(m4)
        m3 = torch.cat((mer_cros3, x3_ACCoM), dim=1)
        mer_cros2 = self.PATM2(m3)
        m2 = torch.cat((mer_cros2, x2_ACCoM), dim=1)
        mer_cros1 = self.PATM1(m2)

        """
        torch.Size([16, 128, 112, 112]) torch.Size([16, 1024, 7, 7])
        torch.Size([16, 256, 56, 56]) torch.Size([16, 1536, 14, 14])
        torch.Size([16, 512, 28, 28]) torch.Size([16, 768, 28, 28])
        torch.Size([16, 1024, 14, 14]) torch.Size([16, 384, 56, 56])
        """

        s1 = self.S1(mer_cros1)
        s2 = self.S2(mer_cros2)
        s3 = self.S3(mer_cros3)
        s4 = self.S4(mer_cros4)

        return s1,s2,s3,s4,edge_sod,edge_rgb,edge_depth

    def get_module_params(self):
        for name, item in self.named_children():
            print(name,get_parameter_num(item))


class CrossWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_q, x_kv):
        """
        x_q: (B*nW, N, C)  # query features
        x_kv: (B*nW, N, C) # key/value features
        """
        B_, N, C = x_q.shape

        q = self.q_proj(x_q).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(x_kv).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        """
        LayerNorm over channel axis (per spatial location), input shape: (B, C, H, W)
        """
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # mean/var over (C, H, W) for each sample independently
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            x = x * self.weight + self.bias
        return x

class CrossSwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm2d):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossWindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x_q, x_kv):
        """
        x_q, x_kv: (B, C, H, W)
        """
        H, W = self.input_resolution


        shortcut = x_q
        x_q = self.norm1_q(x_q)
        x_kv = self.norm1_kv(x_kv)

        if self.shift_size > 0:
            x_q = torch.roll(x_q, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_kv = torch.roll(x_kv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_q_windows = window_partition(x_q, self.window_size).view(-1, self.window_size * self.window_size, C)
        x_kv_windows = window_partition(x_kv, self.window_size).view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_q_windows, x_kv_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PGSformer(nn.Module):
    def __init__(self,config):
        super(PGSformer, self).__init__()


        self.encoderR, embed_dims = build_Rbackbone(config)
        self.encoderD, fused_dims = build_Xbackbone(config)
  
        input_size = config.DATA.IMG_SIZE

        self.S4 = nn.ConvTranspose2d(embed_dims[3], 1, 2, stride=2)
        self.S3 = nn.ConvTranspose2d(embed_dims[2], 1, 2, stride=2)
        self.S2 = nn.ConvTranspose2d(embed_dims[1], 1, 2, stride=2)
        self.S1 = nn.ConvTranspose2d(embed_dims[0], 1, 2, stride=2)
        

        self.up_loss = Interpolate(size=input_size//4)
        """
        图片分类预训练模型可以用作backbone,但是分类任务对边界并不敏感,加入边界提取可以提高模型迁移到显著性检测的效果
        """
        self.sod_edge_aware = Edge_Aware(embed_dims,input_size)
        self.rgb_edge_aware = Edge_Aware(embed_dims,input_size)


        """
        (1024, 1024, 1024, 3, 2)
        (1536, 1024, 512, 3, 2)
        (768, 512, 256, 5, 3)
        (384, 256, 128, 5, 3)
        """

        # different scale feature fusion
        self.PATM4 = PATM_BAB(embed_dims[3], embed_dims[3], embed_dims[2], 3, 2)#TODO reduce params
        self.PATM3 = PATM_BAB(embed_dims[2], embed_dims[2], embed_dims[1], 3, 2)
        self.PATM2 = PATM_BAB(embed_dims[1], embed_dims[1], embed_dims[0], 5, 3)
        self.PATM1 = PATM_BAB(embed_dims[0], embed_dims[1], embed_dims[0], 5, 3)
        
        self.down1 = nn.Sequential(
                        nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
                    )
        
        self.cwa1 = CrossSwinTransformerBlock(dim=128,input_resolution=96,num_heads=4,window_size=6,shift_size=0)
        self.cswa1 = CrossSwinTransformerBlock(dim=128,input_resolution=96,num_heads=4,window_size=6,shift_size=3)

        self.down2 = nn.Sequential(
                        nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )
 
        self.cwa2 = CrossSwinTransformerBlock(dim=256,input_resolution=48,num_heads=8,window_size=6,shift_size=0)
        self.cswa2 = CrossSwinTransformerBlock(dim=256,input_resolution=48,num_heads=8,window_size=6,shift_size=3)
        
        self.down3 = nn.Sequential(
                        nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True)
                    )
        
        self.cwa3 = CrossSwinTransformerBlock(dim=512,input_resolution=24,num_heads=16,window_size=6,shift_size=0)
        self.cswa3 = CrossSwinTransformerBlock(dim=512,input_resolution=24,num_heads=16,window_size=6,shift_size=3)

        #self.cwa4 = CrossSwinTransformerBlock(dim=1024,input_resolution=12,num_heads=32,window_size=6,shift_size=0)
        #self.cwa4 = CrossSwinTransformerBlock(dim=1024,input_resolution=12,num_heads=32,window_size=6,shift_size=3)

    def forward(self, rgb, x):
        x0,x1,x2,x3 = self.encoderR(rgb)        

        """
        [16, 128, 96, 96]
        [16, 256, 48, 48]
        [16, 512, 24, 24]
        [16, 1024, 12, 12 ]
        """

        edge_rgb = self.rgb_edge_aware(x0,x1,x2,x3)
        
        mer_cros4 = self.PATM4(x3)
        
        _,p1,p2,p3 = self.encoderD(x) 

        m4 = self.down3(torch.cat((mer_cros4,x2),dim=1))#512
        m4 = self.cwa3(p3,m4)
        m4 = self.cswa3(p3,m4)
        mer_cros3 = self.PATM3(m4)
        m3 = self.down2(torch.cat((mer_cros3, x1), dim=1))#256
        m3 = self.cwa2(p2,m3)
        m3 = self.cswa2(p2,m3)
        mer_cros2 = self.PATM2(m3)
        m2 = self.down1(torch.cat((mer_cros2, x0), dim=1))#128
        m2 = self.cwa2(p1,m2)
        m2 = self.cswa2(p1,m2)
        mer_cros1 = self.PATM1(m2)
        edge_sod = self.sod_edge_aware(mer_cros1, mer_cros2, mer_cros3, mer_cros4)
        """
        torch.Size([16, 128, 112, 112]) torch.Size([16, 1024, 7, 7])
        torch.Size([16, 256, 56, 56]) torch.Size([16, 1536, 14, 14])
        torch.Size([16, 512, 28, 28]) torch.Size([16, 768, 28, 28])
        torch.Size([16, 1024, 14, 14]) torch.Size([16, 384, 56, 56])
        """

        s1 = self.S1(mer_cros1)
        s2 = self.S2(mer_cros2)
        s3 = self.S3(mer_cros3)
        s4 = self.S4(mer_cros4)

        return s1,s2,s3,s4,edge_sod,edge_rgb,edge_rgb

    def get_module_params(self):
        for name, item in self.named_children():
            print(name,get_parameter_num(item))




if __name__ == "__main__":
    
    args,config = parse_option()

    model = GSformer(config).cuda()
    """
    python train.py --backbone swin-large --texture /namlab40/ --mfusion HAIM --train_batch 20 --gpu_id 6
    """
    rgb = torch.randn(4,3,384,384).cuda()
    x = torch.randn(4,3,384,384).cuda()

    pred = model(rgb,x)
    print(pred[0].shape,pred[1].shape)


