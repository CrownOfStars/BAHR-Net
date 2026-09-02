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

from timm.models.swin_transformer import *
from timm.models.helpers import build_model_with_cfg
import timm

"""
TORCH_DISTRIBUTED_DEBUG
"""

def get_graydepth_model():
    model_kwargs = dict(
        in_chans=2,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),features_only=True,
    )

    model = build_model_with_cfg(
            SwinTransformer, 'swin_small_chan2_patch4_window7_384', False,
            img_size=384,num_classes=0,feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),**model_kwargs)
    
    mdict = torch.load('./pretrained/epoch_299.pth')

    model_dict = {}
    for k,v in mdict.items():
        if k.startswith('module'):
            model_dict[k[7:]] = v
        else:
            model_dict[k] = v

    gd_state_dict = {}

    for k,v in model_dict.items():
        print(k)
        if k.startswith('xgray_encoder'):
            gd_state_dict[k[14:]] = v

    # print(model_dict.keys())
    # print("**********************")
    # print(gd_model.state_dict().keys())

    msg = model.load_state_dict(gd_state_dict)
    print(msg)

    return model



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

    def fork_feat(self,rgb,x):
        x0,x1,x2,x3 = self.encoderR(rgb)        
        y0,y1,y2,y3 = self.encoderD(x)



        edge_rgb = self.rgb_edge_aware(x0,x1,x2,x3)
        edge_depth = self.depth_edge_aware(y0,y1,y2,y3)
        

        x0.retain_grad()
        x1.retain_grad()
        x2.retain_grad()
        x3.retain_grad()

        y0.retain_grad()
        y1.retain_grad()
        y2.retain_grad()
        y3.retain_grad()

        x2_ACCoM = self.FFT1(x0, y0)
        x3_ACCoM = self.FFT2(x1, y1)
        x4_ACCoM = self.FFT3(x2, y2)
        x5_ACCoM = self.FFT4(x3, y3)
        
        x2_ACCoM.retain_grad()
        x3_ACCoM.retain_grad()
        x4_ACCoM.retain_grad()
        x5_ACCoM.retain_grad()

        edge_sod = self.sod_edge_aware(x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM)

        mer_cros4 = self.PATM4(x5_ACCoM)

        m4 = torch.cat((mer_cros4,x4_ACCoM),dim=1)
        mer_cros3 = self.PATM3(m4)
        m3 = torch.cat((mer_cros3, x3_ACCoM), dim=1)
        mer_cros2 = self.PATM2(m3)
        m2 = torch.cat((mer_cros2, x2_ACCoM), dim=1)
        mer_cros1 = self.PATM1(m2)

        s1 = self.S1(mer_cros1)
        s2 = self.S2(mer_cros2)
        s3 = self.S3(mer_cros3)
        s4 = self.S4(mer_cros4)

        return (x0,x1,x2,x3),(y0,y1,y2,y3),(x2_ACCoM,x3_ACCoM,x4_ACCoM,x5_ACCoM),s1,s2,s3,s4,edge_sod,edge_rgb,edge_depth        

    def get_module_params(self):
        for name, item in self.named_children():
            print(name,get_parameter_num(item))

class GDGSformer(nn.Module):
    def __init__(self,args):
        super(GDGSformer, self).__init__()


        self.encoderR = timm.create_model(
            "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
            pretrained=True,
            features_only=True
        )

        self.encoderD = get_graydepth_model()
        
        embed_dims = [128*(2**i) for i in range(4)]
        fused_dims = [64*(2**i) for i in range(4)]

 
        input_size = args.img_size

        self.FFT1, self.FFT2, self.FFT3, self.FFT4 = build_modilty_fusion(config,fused_dims)

        self.ccpr3 = ChannelCompression(embed_dims[3], fused_dims[3])
        self.ccpr2 = ChannelCompression(embed_dims[2], fused_dims[2])
        self.ccpr1 = ChannelCompression(embed_dims[1], fused_dims[1])
        self.ccpr0 = ChannelCompression(embed_dims[0], fused_dims[0])
        
        self.S4 = nn.ConvTranspose2d(fused_dims[3], 1, 2, stride=2)
        self.S3 = nn.ConvTranspose2d(fused_dims[2], 1, 2, stride=2)
        self.S2 = nn.ConvTranspose2d(fused_dims[1], 1, 2, stride=2)
        self.S1 = nn.ConvTranspose2d(fused_dims[0], 1, 2, stride=2)
        
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
        [16, 128, 56, 56]
        [16, 256, 28, 28]
        [16, 512, 14, 14]
        [16, 1024, 7, 7 ]
        """

        x0 = x0.permute(0,3,1,2)
        x1 = x1.permute(0,3,1,2)
        x2 = x2.permute(0,3,1,2)
        x3 = x3.permute(0,3,1,2)

        y0 = y0.permute(0,3,1,2)
        y1 = y1.permute(0,3,1,2)
        y2 = y2.permute(0,3,1,2)
        y3 = y3.permute(0,3,1,2)

        edge_rgb = self.rgb_edge_aware(x0,x1,x2,x3)
        edge_depth = self.depth_edge_aware(y0,y1,y2,y3)
        
        x0,x1,x2,x3 = self.ccpr0(x0),self.ccpr1(x1),self.ccpr2(x2),self.ccpr3(x3)

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

class ViTGSformer(nn.Module):
    def __init__(self,args):
        super(ViTGSformer, self).__init__()


        self.encoderR = timm.create_model(
            "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
            pretrained=True,
            features_only=True
        )

        self.encoderX = timm.create_model(
            'maxvit_tiny_tf_384.in1k',
            pretrained=True,
            features_only=True,
        )

        
        embed_dims = [128*(2**i) for i in range(4)]
        fused_dims = [64*(2**i) for i in range(4)]

 
        input_size = args.img_size

        self.FFT1, self.FFT2, self.FFT3, self.FFT4 = build_modilty_fusion(args,embed_dims)

        self.ccpr3 = ChannelCompression(embed_dims[3], fused_dims[3])
        self.ccpr2 = ChannelCompression(embed_dims[2], fused_dims[2])
        self.ccpr1 = ChannelCompression(embed_dims[1], fused_dims[1])
        self.ccpr0 = ChannelCompression(embed_dims[0], fused_dims[0])
        
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
        self.x_edge_aware = Edge_Aware(fused_dims,input_size)


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


    def forward_interactive(self,rgb,x):
        #stage 0
        r0 = self.encoderR.patch_embed(rgb)
        x0 = self.encoderX.patch_embed(x)

        r1 = self.encoderR.patch_embed(r0)
        x1 = self.encoderX.patch_embed(x0)
        
        f1,r1,x1 = self.FFT1(r1,x1)

        r2 = self.encoderR.patch_embed(r1)
        x2 = self.encoderX.patch_embed(x1)

        f2,r1,x1 = self.FFT1(r2,x2)

        r3 = self.encoderR.patch_embed(r2)
        x3 = self.encoderX.patch_embed(x2)

        f3,r1,x1 = self.FFT1(r3,x3)

        r4 = self.encoderR.patch_embed(r3)
        x4 = self.encoderX.patch_embed(x3)

        f4,r1,x1 = self.FFT1(r4,x4)
        return f1,f2,f3,f4

    def forward(self, rgb, x):
        x0,x1,x2,x3 = self.encoderR(rgb)        
        y0,y1,y2,y3 = self.encoderX(x)

        """
        [16, 128, 56, 56]
        [16, 256, 28, 28]
        [16, 512, 14, 14]
        [16, 1024, 7, 7 ]
        """

        x0 = x0.permute(0,3,1,2)
        x1 = x1.permute(0,3,1,2)
        x2 = x2.permute(0,3,1,2)
        x3 = x3.permute(0,3,1,2)

        y0 = y0.permute(0,3,1,2)
        y1 = y1.permute(0,3,1,2)
        y2 = y2.permute(0,3,1,2)
        y3 = y3.permute(0,3,1,2)

        edge_rgb = self.rgb_edge_aware(x0,x1,x2,x3)
        edge_depth = self.x_edge_aware(y0,y1,y2,y3)
        
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

class NGSCFFormer(nn.Module):
    def __init__(self,encoderC,encoderX,fusionMM,decoderMS):
        super(NGSCFFormer, self).__init__()
        self.encoderC = encoderC
        self.encoderX = encoderX

        self.fusions = fusionMM

        self.decoder = decoderMS

    def forward(self,rgb,x):
        r0 = self.encoderC.patch_embed(rgb)
        x0 = self.encoderX.patch_embed(x)

        r1 = self.encoderC.layers_0(r0)
        x1 = self.encoderX.layers_0(x0)

        f1,r1,x1 = self.fusions[0](r1,x1)

        r2 = self.encoderC.layers_1(r1)
        x2 = self.encoderX.layers_1(x1)

        f2,r2,x2 = self.fusions[1](r2,x2)

        r3 = self.encoderC.layers_2(r2)
        x3 = self.encoderX.layers_2(x2)

        f3,r3,x3 = self.fusions[2](r3,x3)

        r4 = self.encoderC.layers_3(r3)
        x4 = self.encoderX.layers_3(x3)

        f4,r4,x4 = self.fusions[3](r4,x4)

        return self.decoder(f1,f2,f3,f4)


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


