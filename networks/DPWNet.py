import sys
sys.path.append('../')
from networks.PWNet import *

class NPATM_BAB(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256,channel_4 = 1, dilation_1=3, dilation_2=2):
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
        self.seg_head = nn.Conv2d(channel_3, 1, 3, stride=1, padding=1)
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
        # print()
        x_fuse = self.conv_last(x_fuse)
        return x_fuse,self.seg_head(x_fuse)


class DPWNet(nn.Module):
    def __init__(self, num_class=2):
        super(DPWNet, self).__init__()
        self.encoderR = poolformer_s36_feat(pretrained=True)
        # Lateral layers
        self.lateral_conv0 = BasicConv2d(64, 64, 3, stride=1, padding=1)
        self.lateral_conv1 = BasicConv2d(128, 64, 3, stride=1, padding=1)
        self.lateral_conv2 = BasicConv2d(320, 128, 3, stride=1, padding=1)
        self.lateral_conv3 = BasicConv2d(512, 320, 3, stride=1, padding=1)

        self.FFT1 = FFusion(64,64)
        self.FFT2 = FFusion(128,128)
        self.FFT3 = FFusion(320,320)
        self.FFT4 = FFusion(512,512)


        self.conv512_64 = BasicConv2d(512, 64)
        self.conv320_64 = BasicConv2d(320, 64)
        self.conv128_64 = BasicConv2d(128, 64)
        self.sigmoid = nn.Sigmoid()
        
        self.up1 = nn.Upsample(224)
        self.up2 = nn.Upsample(224)
        self.up3 = nn.Upsample(224)
        self.up_loss = nn.Upsample(92)
        
        # Mutual_info_reg1
        self.mi_level1 = Mutual_info_reg(64, 64, 6)
        self.mi_level2 = Mutual_info_reg(64, 64, 6)
        self.mi_level3 = Mutual_info_reg(64, 64, 6)
        self.mi_level4 = Mutual_info_reg(64, 64, 6)

        self.edge = Edge_Aware()
        
        self.PATM4 = NPATM_BAB(512, 512, 512, num_class, 3, 2)
        self.PATM3 = NPATM_BAB(832, 512, 320, num_class, 3, 2)
        self.PATM2 = NPATM_BAB(448, 256, 128, num_class, 5, 3)
        self.PATM1 = NPATM_BAB(192, 128, 64, num_class, 5, 3)

    def forward(self, x_rgb,x_thermal):
        x0,x1,x2,x3 = self.encoderR(x_rgb)        
        y0,y1,y2,y3 = self.encoderR(x_thermal)

        x2_ACCoM = self.FFT1(x0, y0)
        
        x3_ACCoM = self.FFT2(x1, y1)
        
        x4_ACCoM = self.FFT3(x2, y2)
        
        x5_ACCoM = self.FFT4(x3, y3)
        
        edge = self.edge(x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM)

        mer_cros4,seg4 = self.PATM4(x5_ACCoM)

        m4 = torch.cat((mer_cros4,x4_ACCoM),dim=1)
        mer_cros3,seg3 = self.PATM3(m4)
       
        m3 = torch.cat((mer_cros3, x3_ACCoM), dim=1)
        mer_cros2,seg2 = self.PATM2(m3)

        m2 = torch.cat((mer_cros2, x2_ACCoM), dim=1)
        mer_cros1,seg1 = self.PATM1(m2)

        s1 = self.up1(seg1)
        s2 = self.up2(seg2)
        s3 = self.up3(seg3)
        s4 = self.up3(seg4)

        x_loss0 = x0
        y_loss0 = y0
        x_loss1 = self.up_loss(self.conv128_64(x1))
        y_loss1 = self.up_loss(self.conv128_64(y1))
        x_loss2 = self.up_loss(self.conv320_64(x2))
        y_loss2 = self.up_loss(self.conv320_64(y2))
        x_loss3 = self.up_loss(self.conv512_64(x3))
        y_loss3 = self.up_loss(self.conv512_64(y3))

        lat_loss0 = self.mi_level1(x_loss0, y_loss0)
        lat_loss1 = self.mi_level2(x_loss1, y_loss1)
        lat_loss2 = self.mi_level3(x_loss2, y_loss2)
        lat_loss3 = self.mi_level4(x_loss3, y_loss3)
        lat_loss = lat_loss0 + lat_loss1 + lat_loss2 + lat_loss3
        return s1, s2, s3, s4, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(s4),edge,lat_loss