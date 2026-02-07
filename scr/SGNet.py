import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.layers import trunc_normal_
from thop import profile

from .MPS import MPS4, MPS5
from .TDSA import TDSAttention
from .SKG import SAMGuidedFeatureFusion
from .Conv import *


def adjust_backbone_strides(backbone):
    """独立函数，用于调整 backbone 的 stride"""
    backbone.conv1.stride = (1, 1)   # 修改conv1的stride为1（取消第一次下采样）
    backbone.maxpool = nn.Identity()  # 完全移除maxpool
    if hasattr(backbone, "layer3"):  # 修改layer3第一个block的stride为2（50x50 -> 25x25）
        backbone.layer3[0].conv1.stride = (2, 2)
        if hasattr(backbone.layer3[0], "downsample"):
            backbone.layer3[0].downsample[0].stride = (2, 2)
    if hasattr(backbone, "layer4"):   # 修改layer4第一个block的stride为2（保持25x25）
        backbone.layer4[0].conv1.stride = (2, 2)
        if hasattr(backbone.layer4[0], "downsample"):
            backbone.layer4[0].downsample[0].stride = (2, 2)
    return backbone  # 可选：返回调整后的backbone


class TransformerDecoderBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, mlp_ratio=4., dropout=0.5):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.self_attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout)
        self.Conv1_1 = Conv(in_channels, in_channels // 2, kernel_size=1)

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mlp_hidden_dim, 1, groups=4),  # 分组卷积
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_hidden_dim, in_channels, 1, groups=4),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape

        # Self-Attention
        residual = x
        x = self.norm1(x)
        x = x.flatten(2).permute(2, 0, 1)  # [B, C, H, W] -> [H*W, B, C]
        x, _ = self.self_attn(x, x, x)  # 自注意力
        x = x.permute(1, 2, 0).view(B, C, H, W)  # 恢复形状 [B, C, H, W]
        x = residual + self.dropout(x)

        # MLP
        residual = x
        x = self.mlp(x)
        x = residual + self.dropout(x)
        x = self.Conv1_1(x)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, dim, height, width):
        super().__init__()
        self.height = height
        self.width = width
        # 创建可学习的位置嵌入 (H*W, 1, C)
        self.pos_embed = nn.Parameter(torch.zeros(height * width, 1, dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        """
        x: 输入序列 [H*W, B, C]
        """
        # 添加位置嵌入 (广播到批量维度)
        return x + self.pos_embed


class WF(nn.Module):
    def __init__(self, in_channels, ou_channels, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, ou_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(ou_channels, ou_channels, kernel_size=3)

    def forward(self, x, res):

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), num_classes=2, dropout=0.5, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.context_proj = nn.Conv2d(encoder_channels[3], encoder_channels[3], kernel_size=1)
        self.pos_embed = PositionEmbedding(encoder_channels[3], 25, 25)  # Adjust based on mps5 size

        self.Conv4_5 = Conv(encoder_channels[0]+encoder_channels[3], encoder_channels[3], kernel_size=1)

        # Transformer blocks
        self.block4 = nn.Sequential(TransformerDecoderBlock(encoder_channels[3], num_heads, mlp_ratio),
                                    nn.BatchNorm2d(encoder_channels[2]),
                                    nn.Dropout2d(dropout))
        self.p3 = WF(encoder_channels[0], encoder_channels[2])

        self.block3 = nn.Sequential(TransformerDecoderBlock(encoder_channels[2], num_heads, mlp_ratio),
                                    nn.BatchNorm2d(encoder_channels[1]),
                                    nn.Dropout2d(dropout))
        self.p2 = WF(encoder_channels[0], encoder_channels[1])

        self.block2 = nn.Sequential(TransformerDecoderBlock(encoder_channels[1], num_heads, mlp_ratio),
                                    nn.BatchNorm2d(encoder_channels[0]),
                                    nn.Dropout2d(dropout))
        self.p1 = WF(encoder_channels[0], encoder_channels[0])

        self.block1 = nn.Sequential(ConvBNReLU(encoder_channels[0], encoder_channels[0]),
                                    nn.BatchNorm2d(encoder_channels[0]),
                                    nn.Dropout2d(p=dropout, inplace=True),
                                    Conv(encoder_channels[0], num_classes, kernel_size=1))

    def forward(self, skg1, skg2, skg3, skg4, mps5):
        # 处理mps5的特征，并加入位置信息
        x = self.context_proj(mps5)
        B, C, H, W = x.shape
        x_f = x.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        # x_f = self.pos_embed(x_f)   # 嵌入位置信息
        context_mps5 = x_f.permute(1, 2, 0).view(B, -1, H, W)
        skg4_mps5 = torch.cat((skg4, context_mps5),dim=1)
        skg4_mps5 = self.Conv4_5(skg4_mps5)  # B 512 25 25
        x4 = self.block4(skg4_mps5)
        w3 = self.p3(x4, skg3)
        x3 = self.block3(w3)
        w2 = self.p2(x3, skg2)
        x2 = self.block2(w2)
        w1 = self.p1(x2, skg1)
        out = self.block1(w1)
        return out


class SGNet(nn.Module):
    def __init__(self,
                 in_channels,
                 prior_channels,
                 out_channels,
                 backbone_name='resnet18'):
        super().__init__()

        self.in_channels = in_channels
        self.prior_channels = prior_channels
        self.out_channels = out_channels

        # 创建 backbone（features_only=True 返回多尺度特征）
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            output_stride=8,  # 调整下采样倍数,最终输出为200/8=25
            out_indices=(1, 2, 3, 4),  # 对应 layer1, layer2, layer3, layer4
            pretrained=False  # 手动加载权重
        )

        self.backbone = adjust_backbone_strides(self.backbone)  # 局部调整resnet结构

        model_path = '/home/ubuntu/data0/ygt/CNN/SAMP_Net/scr/resnet18-5c106cde.pth'
        state_dict = torch.load(model_path, map_location="cpu")
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in self.backbone.state_dict()}
        self.backbone.load_state_dict(filtered_state_dict, strict=False)  # 非严格模式加载
        encoder_channels = self.backbone.feature_info.channels()  # 编码器端通道(64, 128, 256, 512)
        # print(encoder_channels)

        self.Conv1_2 = Conv(encoder_channels[1], encoder_channels[0], kernel_size=1)
        self.Conv1_3 = Conv(encoder_channels[2], encoder_channels[0], kernel_size=1)
        self.Conv1_4 = Conv(encoder_channels[3], encoder_channels[0], kernel_size=1)

        self.tda = TDSAttention(in_channels=encoder_channels[0])
        # self.mps4 = MPS4(in_channels=encoder_channels[0]*4, out_channels=encoder_channels[0]*4)
        self.mps5 = MPS5(in_channels=encoder_channels[0]*5, out_channels=encoder_channels[3])

        self.skg1 = SAMGuidedFeatureFusion(encoder_channels[0], encoder_channels[3], 3)
        self.skg2 = SAMGuidedFeatureFusion(encoder_channels[0], encoder_channels[3], 3)
        self.skg3 = SAMGuidedFeatureFusion(encoder_channels[0], encoder_channels[3], 3)
        self.skg4 = SAMGuidedFeatureFusion(encoder_channels[0], encoder_channels[3], 3)
        self.Decoder = Decoder(encoder_channels, out_channels)

    def forward(self, x, x_seg):
        res1, res2, res3, res4 = self.backbone(x)
        res2_1 = self.Conv1_2(res2)
        res3_1 = self.Conv1_3(res3)
        res4_1 = self.Conv1_4(res4)
        # return res1, res2_1, res3_1, res4_1  # FLOPs: 22.243 G, Params: 11.234 M

        tda1 = self.tda(res1)
        tda2 = self.tda(res2_1)
        tda3 = self.tda(res3_1)
        tda4 = self.tda(res4_1)
        # return tda1, tad2, tad3, tad4  # FLOPs: 22.705 G, Params: 11.243 M

        # mps4 = self.mps4(res1, res2_1, res3_1, res4_1)
        mps5 = self.mps5(tda1, tda2, tda3, tda4, res4_1)
        # return tda1, tda2, tda3, tda4, mps5  # FLOPs: 29.709 G, Params: 14.709 M

        skg1 = self.skg1(tda1, mps5, x_seg)
        skg2 = self.skg2(tda2, mps5, x_seg)
        skg3 = self.skg3(tda3, mps5, x_seg)
        skg4 = self.skg4(tda4, mps5, x_seg)
        # return skg4,skg3,skg2,skg1,mps5  # FLOPs: 35.683 G, Params: 15.159 M

        out_put = self.Decoder(skg1, skg2, skg3, skg4, mps5)
        return out_put


if __name__ == '__main__':

    model = SGNet(in_channels=3, prior_channels=3, out_channels=2)
    input = torch.rand(1, 3, 200, 200)
    seg_tensor = torch.rand(1, 3, 200, 200)
    out = model(input, seg_tensor)
    print(model)
    # print(out.shape)

    flops, params = profile(model, inputs=(input, seg_tensor), verbose=False)
    print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")
