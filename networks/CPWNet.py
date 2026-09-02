import sys
sys.path.append('../')
from networks.PWNet import *

class CPWNet(nn.Module):
    def __init__(self, channel=32):
        super(CPWNet, self).__init__()
        self.encoderR = poolformer_s36_feat(pretrained=True)
        self.encoderX = poolformer_s12_feat(pretrained=True)
        # Lateral layers
        self.lateral_conv0 = BasicConv2d(64, 64, 3, stride=1, padding=1)
        self.lateral_conv1 = BasicConv2d(128, 64, 3, stride=1, padding=1)
        self.lateral_conv2 = BasicConv2d(320, 128, 3, stride=1, padding=1)
        self.lateral_conv3 = BasicConv2d(512, 320, 3, stride=1, padding=1)

        self.FFT1 = UPFFT(64,64,56)
        self.FFT2 = UPFFT(128,128,28)
        self.FFT3 = UPFFT(320,320,14)
        self.FFT4 = UPFFT(512,512,7)


        self.conv512_64 = BasicConv2d(512, 64)
        self.conv320_64 = BasicConv2d(320, 64)
        self.conv128_64 = BasicConv2d(128, 64)
        self.sigmoid = nn.Sigmoid()
        self.S4 = nn.Conv2d(512, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(320, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(128, 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        
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
        self.PATM4 = PATM_BAB(512, 512, 512, 3, 2)
        self.PATM3 = PATM_BAB(832, 512, 320, 3, 2)
        self.PATM2 = PATM_BAB(448, 256, 128, 5, 3)
        self.PATM1 = PATM_BAB(192, 128, 64, 5, 3)

    def forward(self, x_rgb,x_thermal):
        x0,x1,x2,x3 = self.encoderR(x_rgb)
        y0,y1,y2,y3 = self.encoderX(x_thermal)

        x2_ACCoM = self.FFT1(x0, y0)
        
        x3_ACCoM = self.FFT2(x1, y1)
        
        x4_ACCoM = self.FFT3(x2, y2)
        
        x5_ACCoM = self.FFT4(x3, y3)
        
        edge = self.edge(x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM)

        mer_cros4 = self.PATM4(x5_ACCoM)

        m4 = torch.cat((mer_cros4,x4_ACCoM),dim=1)
        mer_cros3 = self.PATM3(m4)
        m3 = torch.cat((mer_cros3, x3_ACCoM), dim=1)
        mer_cros2 = self.PATM2(m3)
        m2 = torch.cat((mer_cros2, x2_ACCoM), dim=1)
        mer_cros1 = self.PATM1(m2)



        s1 = self.up1(self.S1(mer_cros1))
        s2 = self.up2(self.S2(mer_cros2))
        s3 = self.up3(self.S3(mer_cros3))
        s4 = self.up3(self.S4(mer_cros4))


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
