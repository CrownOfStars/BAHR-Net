import torch
import torch.nn as nn
from timm.layers import DropPath
import torch.nn.functional as F
from timm.layers import DropPath


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        return self.relu(self.bn(self.conv(x)))

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class BasicASConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = AtrousSeparableConvolution(in_planes, out_planes, kernel_size=kernel_size,stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        return self.relu(self.bn(self.conv(x)))        

class ConvMlp(nn.Module):
    def __init__(self, in_features=64, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features*4
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=1),
            act_layer(),
            nn.Conv2d(hidden_features, out_features, 1),
            DropPath(0.1),
        )

    def forward(self, x):
        return self.conv_mlp(x)


class LayerNorm2d(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return x - self.pool(x)

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels,kernel_size=1, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class EAFormer(nn.Module):
    def __init__(self,in_chans,out_scale):
        super(EAFormer, self).__init__()

        self.in_chans = in_chans
        self.proj4 = nn.Sequential(
            BasicASConv2d(in_chans[3], in_chans[0], padding=3,dilation=3),
            nn.Upsample(scale_factor=8.0),
        )
        self.proj3 = nn.Sequential(
            BasicASConv2d(in_chans[2], in_chans[0], padding=6,dilation=6),
            nn.Upsample(scale_factor=4.0),
        )
        self.proj2 = nn.Sequential(
            BasicASConv2d(in_chans[1], in_chans[0], padding=12,dilation=12),
            nn.Upsample(scale_factor=2.0),
        )

        self.edge_query = nn.AdaptiveMaxPool2d(1)
        
        self.query_mlp = ConvMlp(in_features=in_chans[0], out_features=in_chans[0]*3)
        
        self.token_mixer = SepConv(in_chans[0])
        
        self.norm1 = LayerNorm2d(in_chans[0])
        self.norm2 = LayerNorm2d(in_chans[0])
        self.drop_path = DropPath(0.1)
        self.mlp = ConvMlp(in_chans[0])
        
        self.conv_out = nn.Sequential(
            nn.Upsample(scale_factor=out_scale, mode='bilinear', align_corners=False),
            nn.Conv2d(in_chans[0], 1, kernel_size=3,stride=1,padding=1)
        )
    
    def forward(self, x, y, z, v):

        x_query = self.edge_query(x)
        
        q_y,q_z,q_v = torch.chunk(self.query_mlp(x_query), chunks=3, dim=1)

        x_y = torch.mul(q_y, self.proj2(y))
        x_z = torch.mul(q_z, self.proj3(z))
        x_v = torch.mul(q_v, self.proj4(v))

        x_mix = x + self.drop_path(self.token_mixer(self.norm1(x_y + x_z + x_v)))
    
        x_mix = x_mix + self.drop_path(self.mlp(self.norm2(x_mix)))
        
        return self.conv_out(x_mix)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

#-----------
