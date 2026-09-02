import torch
import torch.nn as nn
import sys
sys.path.append('../')

from networks.wavemlp import WaveMLP_S

from torch.nn.functional import kl_div

import torch.nn.functional as F

from networks.BConv2d import *



class PATM_BAB(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256, dilation_1=3, dilation_2=2):
        super().__init__()
        self.conv1 = BasicConv2d(channel_1, channel_2, 3, padding=1)
        self.conv1_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_2, dilation=dilation_2)
        self.conv3 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv_fuse = BasicConv2d(channel_2 *2, channel_3, 3, padding=1)
        self.drop = nn.Dropout(0.5)
        self.conv_last=TransBasicConv2d(channel_3, channel_3,  kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
    def forward(self, x):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)

        x2 = self.conv2(x1)
        x2_dila = self.conv2_Dila(x2)

        x3 = self.conv3(x2)
        x1_dila = torch.cat([x1_dila * torch.cos(x1_dila), x1_dila * torch.sin(x1_dila)], dim=1)
        x2_dila = torch.cat([x2_dila * torch.cos(x2_dila), x2_dila * torch.sin(x2_dila)], dim=1)
        x3 = torch.cat([x3 * torch.cos(x3), x3 * torch.sin(x3)], dim=1)
        # print('x1_dila + x2_dila+x3',x1_dila.shape)
        x_fuse = self.conv_fuse(x1_dila + x2_dila +x3)
        # x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila, x3), 1))
        # print('x_f',x_fuse.shape)
        x_fuse= self.drop(x_fuse)
        x_fuse = self.conv_last(x_fuse)
        return x_fuse






"""
TODO: replace decoder
"""


"""
TODO: replace namlab supervision with gt/gt_bound supervision
"""

"""
TODO: add depth namlab supervision
"""

"""
TODO: depth Calibration
"""



class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels=64, channels=64, latent_size=6):
        super(Mutual_info_reg, self).__init__()
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, rgb_feat, depth_feat):

        rgb_feat = self.soft(rgb_feat)
        depth_feat = self.soft(depth_feat)

        return  kl_div(rgb_feat.log(), depth_feat, reduction='batchmean')



if __name__ == "__main__":
    #model = PWNet()
    #ff = FFusion(1024,1024)
    rgb = torch.randn(4,1024,7,7)
    x = torch.randn(4,1024,7,7)
    #print(ff(rgb,x).shape)
    # up2 = nn.UpsamplingBilinear2d(scale_factor = 2)
    #ff_swin = fuse_enhance(1024)
    #print(up2(ff_swin(rgb,x)).shape)

    #fusion = CrossAttentionFusionPool(1024, 2, 5)
    
    #haim = HAIM(1024)
    
    #out = haim(rgb,x)
    
    t = torch.randn(16,512,14,14)
    

    f1 = torch.randn(16, 128, 56, 56)
    f2 = torch.randn(16, 256, 28, 28)
    f3 = torch.randn(16, 512, 14, 14)
    f4 = torch.randn(16, 1024, 7, 7 )
    
    #sod_edge = Edge_Module(in_fea=[128,256,512,1024],mid_fea=32)
    
    #sod_edge1 = Edge_Aware(in_chans=[128,256,512,1024],img_size=224)
    
    #x = sod_edge(f1,f2,f3,f4)
    #y = sod_edge1(f1,f2,f3,f4)
    
    
    
    #print(x.shape,rgb.shape)
    #x = fusion(rgb,x)
    
    #tbc = TransBasicConv2d(1024,512)
    
    #out = tbc(f4)
    
    