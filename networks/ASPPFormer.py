import torch
import torch.nn as nn
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
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
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



class SoftMoE_ConvMlp(nn.Module):
    def __init__(self, in_features=64, hidden_features=None, out_features=None, act_layer=nn.GELU, num_experts=3, temperature=0.25):
        super().__init__()
        self.num_experts = num_experts
        # 训练初期可用较小 temperature（路由更尖），训练中通过 set_temperature 线性升温使 softmax 更平滑、梯度更稳
        self.temperature = float(temperature)
        self.balance_loss = 0
        
        self.experts = nn.ModuleList([
            ConvMlp(in_features, hidden_features, out_features, act_layer) 
            for _ in range(num_experts)
        ])
        
        # 【修改】：去掉自带的 Softmax，我们要手动操作 Logits
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(in_features, num_experts, kernel_size=1)
        )

    def set_temperature(self, t: float):
        """运行时更新路由温度（用于训练过程中缓慢升温）。"""
        self.temperature = max(float(t), 1e-4)

    def forward(self, x):
        # 1. 计算原始路由得分 (Logits)
        logits = self.router(x)
        
        # ===================================================
        # 绝招2：训练时注入路由噪声 (Noisy Routing)
        # ===================================================
        if self.training:
            # 引入极小的高斯噪声，打破确定性，促使专家分化
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
            
        # ===================================================
        # 绝招1：使用温度缩放进行 Softmax
        # temperature < 1 会逼迫权重趋向于 One-Hot (例如从 [0.4, 0.3, 0.3] 变成 [0.8, 0.1, 0.1])
        # ===================================================
        temp = max(self.temperature, 1e-4)
        weights = F.softmax(logits / temp, dim=1)
        
        # 计算负载均衡 Loss (保持不变)
        mean_weights = weights.mean(dim=0).squeeze() 
        balance_loss = self.num_experts * torch.sum(mean_weights ** 2)
        self.balance_loss = balance_loss
        # 2. 专家融合
        out = 0
        for i in range(self.num_experts):
            out += weights[:, i:i+1, :, :] * self.experts[i](x)
            
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class LayerNorm2d(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class ASPPFormer(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256, dilation_1=3, dilation_2=2, channel_up=None):
        """channel_up: x_up 的通道数，用于 conv3 投影。若 None 且 channel_2==channel_1 则用 channel_1//2"""
        super().__init__()
        self.conv1 = BasicConv2d(channel_1, channel_1//2, 3, padding=1)
        self.conv1_Dila = BasicASConv2d(channel_1//2, channel_3, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_3, 3, padding=1)
        self.conv2_Dila = BasicASConv2d(channel_3, channel_3, 3, padding=dilation_2, dilation=dilation_2)
        
        self.conv_mlp = ConvMlp(channel_3)
        self.drop = DropPath(0.1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(channel_3, channel_3, kernel_size=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        if channel_up is not None:
            self.conv3 = BasicConv2d(channel_up, channel_3)
        elif channel_2 == channel_1:
            self.conv3 = BasicConv2d(channel_1 // 2, channel_3)
        else:
            self.conv3 = nn.Identity()
        
        self._init_weights()

    def forward(self, x, x_up=None):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)
    
        x2 = self.conv2(torch.cat((x1,x_up),dim=1) if x_up is not None else x1)
        x2_dila = self.conv2_Dila(x2) 

        x_fuse = self.conv3(x_up) + x1_dila + self.drop(self.conv_mlp(x2_dila)) if x_up is not None else x1_dila + self.drop(self.conv_mlp(x2_dila))
        return self.conv_last(x_fuse)
    
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


class SoftMoE_ASPPFormer(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256, dilation_1=3, dilation_2=2, channel_up=None, num_experts=2):
        super().__init__()
        self.conv1 = BasicConv2d(channel_1, channel_1//2, 3, padding=1)
        self.conv1_Dila = BasicASConv2d(channel_1//2, channel_3, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_3, 3, padding=1)
        self.conv2_Dila = BasicASConv2d(channel_3, channel_3, 3, padding=dilation_2, dilation=dilation_2)
        
        self.conv_mlp = SoftMoE_ConvMlp(channel_3, num_experts=num_experts)
        self.drop = DropPath(0.1)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(channel_3, channel_3, kernel_size=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        if channel_up is not None:
            self.conv3 = BasicConv2d(channel_up, channel_3)
        elif channel_2 == channel_1:
            self.conv3 = BasicConv2d(channel_1 // 2, channel_3)
        else:
            self.conv3 = nn.Identity()

    def set_moe_temperature(self, t: float):
        self.conv_mlp.set_temperature(t)

    def forward(self, x, x_up=None):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)
    
        x2 = self.conv2(torch.cat((x1,x_up),dim=1) if x_up is not None else x1)
        x2_dila = self.conv2_Dila(x2) 

        if x_up is not None:
            x_fuse = self.conv3(x_up) + x1_dila + self.drop(self.conv_mlp(x2_dila))
        
        else:
            x_fuse = x1_dila + self.drop(self.conv_mlp(x2_dila))
        return self.conv_last(x_fuse)

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