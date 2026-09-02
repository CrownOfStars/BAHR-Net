import torch
import torch.nn as nn
from networks.BConv2d import *
import torch.nn.functional as F
from timm.layers import DropPath

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Edge_Module(nn.Module):
    def __init__(self, in_fea=[128, 256, 512], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv3 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[3], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_3 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.classifer = nn.Conv2d(mid_fea * 4, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 4)
        self.edge_feature = conv3x3_bn_relu(1, 32)
        self.up_edge = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2),
            conv3x3(32, 1)
        )


    def forward(self, x2, x3, x4, x5):
        _, _, h, w = x2.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        
        edge3_fea = self.relu(self.conv3(x3))
        edge3 = self.relu(self.conv5_3(edge3_fea))        
        
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        
        edge5_fea = self.relu(self.conv5(x5))
        edge5 = self.relu(self.conv5_5(edge5_fea))

        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)
        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)
        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge2,edge3, edge4, edge5], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        
        up_edge = self.up_edge(self.edge_feature(edge))
        return up_edge


class Mlp(nn.Module):
    def __init__(self, in_features=64, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print('x',x.shape)
        x = self.fc1(x)
        # print('fc',x.shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Edge_Aware(nn.Module):
    def __init__(self,in_chans,img_size):
        super(Edge_Aware, self).__init__()

        self.in_chans = in_chans
        self.conv1 = TransBasicConv2d(in_chans[3], in_chans[0],kernel_size=4,stride=8,padding=0,dilation=2,output_padding=1)
        self.conv2 = TransBasicConv2d(in_chans[2], in_chans[0],kernel_size=2,stride=4,padding=0,dilation=2,output_padding=1)
        self.conv3 = TransBasicConv2d(in_chans[1], in_chans[0],kernel_size=2,stride=2,padding=1,dilation=2,output_padding=1)
        self.pos_embed = BasicConv2d(in_chans[0], in_chans[0] )
        self.pos_embed3 = BasicConv2d(in_chans[0], in_chans[0])
        self.conv31 = nn.Conv2d(in_chans[0],1, kernel_size=1)
        self.conv512_64 = TransBasicConv2d(in_chans[3],in_chans[0])
        self.conv320_64 = TransBasicConv2d(in_chans[2], in_chans[0])
        self.conv128_64 = TransBasicConv2d(in_chans[1], in_chans[0])
        self.up = nn.Upsample(img_size//4)
        self.up2 = nn.Upsample(img_size)
        self.norm1 = nn.LayerNorm(in_chans[0])
        self.norm2 = nn.BatchNorm2d(in_chans[0])
        self.drop_path = DropPath(0.3)
        self.maxpool =nn.AdaptiveMaxPool2d(1)
        # self.qkv = nn.Linear(64, 64 * 3, bias=False)
        self.num_heads = 8
        self.mlp1 = Mlp(in_features=in_chans[0], out_features=in_chans[0])
        self.mlp2 = Mlp(in_features=in_chans[0], out_features=in_chans[0])
        self.mlp3 = Mlp(in_features=in_chans[0], out_features=in_chans[0])
    
    def forward(self, x, y, z, v):

        v = self.up(self.conv512_64(v))
        z = self.up(self.conv320_64(z))
        y = self.up(self.conv128_64(y))
        x = self.up(x)

        x_max = self.maxpool(x)
        # print('x_max',x_max.shape)
        b,_,_,_ = x_max.shape
        x_max = x_max.reshape(b, -1)
        x_y = self.mlp1(x_max)
        x_z = self.mlp2(x_max)
        x_v = self.mlp3(x_max)

        x_y = x_y.reshape(b,self.in_chans[0],1,1)
        x_z = x_z.reshape(b, self.in_chans[0], 1, 1)
        x_v = x_v.reshape(b, self.in_chans[0], 1, 1)
        x_y = torch.mul(x_y, y)
        x_z = torch.mul(x_z, z)
        x_v = torch.mul(x_v, v)


        # x_mix_1 = torch.cat((x_y,x_z,x_v),dim=1)
        x_mix_1 = x_y+ x_z+ x_v
        # print('sd',x_mix_1.shape)
        x_mix_1 =  self.norm2(x_mix_1)
        # print('x_mix_1',x_mix_1.shape)
        x_mix_1= self.pos_embed3(x_mix_1)
        x_mix = self.drop_path(x_mix_1)
        x_mix = x_mix_1 + self.pos_embed3(x_mix)
        x_mix = self.up2(self.conv31(x_mix))
        return x_mix
    
#-----------


def build_edge_aware(config,embed_dims,fused_dims):
    if config.MODEL.EDGEAWARE == "EA":
        edge_aware = Edge_Aware(embed_dims if config.DATA.TEXTURE != '/bound/' else fused_dims,config.DATA.IMG_SIZE)
        return edge_aware
    elif config.MODEL.EDGEAWARE == "EM":
        edge_aware = Edge_Module(embed_dims if config.DATA.TEXTURE != '/bound/' else fused_dims)
        return edge_aware
    else:
        raise NotImplementedError("no such edge-aware module")