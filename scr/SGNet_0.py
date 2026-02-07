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
from .Decoder import *


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


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')
    return src


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

        self.tda1 = TDSAttention(in_channels=encoder_channels[0])
        self.tda2 = TDSAttention(in_channels=encoder_channels[1])
        self.tda3 = TDSAttention(in_channels=encoder_channels[2])
        self.tda4 = TDSAttention(in_channels=encoder_channels[3])

        self.Conv2_2 = Conv(encoder_channels[1], encoder_channels[0], kernel_size=1)
        self.Conv3_3 = Conv(encoder_channels[2], encoder_channels[0], kernel_size=1)
        self.Conv4_4 = Conv(encoder_channels[3], encoder_channels[0], kernel_size=1)

        self.Conv12345 = Conv(encoder_channels[0]*23, encoder_channels[3], kernel_size=1)

        self.mps5 = MPS5(in_channels=encoder_channels[0]*12, out_channels=encoder_channels[3])

        self.skg1 = SAMGuidedFeatureFusion(encoder_channels[0], encoder_channels[3], 3)
        self.skg2 = SAMGuidedFeatureFusion(encoder_channels[0]*2, encoder_channels[3], 3)
        self.skg3 = SAMGuidedFeatureFusion(encoder_channels[0]*4, encoder_channels[3], 3)
        self.skg4 = SAMGuidedFeatureFusion(encoder_channels[0]*8, encoder_channels[3], 3)
        self.mps5_1 = RSU4F(512, 256, 512)
        # Start Decoder
        self.decoder_4 = RSU4(1024, 256, 256)
        self.decoder_3 = RSU5(256, 128, 128)
        self.decoder_2 = RSU6(128, 64, 64)
        self.decoder_1 = RSU7(64, 32, 32)

        self.outconv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x, x_seg):
        res1, res2, res3, res4 = self.backbone(x)
        # res2_1 = self.Conv1_2(res2)
        # res3_1 = self.Conv1_3(res3)
        # res4_1 = self.Conv1_4(res4)
        # return res1, res2_1, res3_1, res4_1  # FLOPs: 22.243 G, Params: 11.234 M
        tda1 = self.tda1(res1)
        tda2 = self.tda2(res2)
        tda3 = self.tda3(res3)
        tda4 = self.tda4(res4)
        # return tda1, tad2, tad3, tad4  # FLOPs: 22.705 G, Params: 11.243 M
        tda2_2 = self.Conv2_2(res2)
        tda3_3 = self.Conv3_3(res3)
        tda4_4 = self.Conv4_4(res4)
        mps5 = self.mps5(tda1, tda2_2, tda3_3, tda4_4, res4)
        # return tda1, tda2, tda3, tda4, mps5  # FLOPs: 29.709 G, Params: 14.709 M
        skg1 = self.skg1(tda1, mps5, x_seg)
        skg2 = self.skg2(tda2, mps5, x_seg)
        skg3 = self.skg3(tda3, mps5, x_seg)
        skg4 = self.skg4(tda4, mps5, x_seg)
        # return skg4,skg3,skg2,skg1,mps5  # FLOPs: 35.683 G, Params: 15.159 M
        mps5_1 = self.mps5_1(mps5)
        # Start Decoder
        decoder_4 = self.decoder_4(torch.cat((mps5_1, skg4), dim=1))
        decoder_4_up = _upsample_like(decoder_4, skg3)
        decoder_3 = self.decoder_3(decoder_4_up)
        decoder_3_up = _upsample_like(decoder_3, skg2)
        decoder_2 = self.decoder_2(decoder_3_up)
        decoder_2_up = _upsample_like(decoder_2, skg1)
        decoder_1 = self.decoder_1(decoder_2_up)
        Out = self.outconv(decoder_1)
        return Out





if __name__ == '__main__':

    model = SGNet(in_channels=3, prior_channels=3, out_channels=2)
    input = torch.rand(1, 3, 200, 200)
    seg_tensor = torch.rand(1, 3, 200, 200)
    out = model(input, seg_tensor)
    # print(model)
    print(out.shape)

    flops, params = profile(model, inputs=(input, seg_tensor), verbose=False)
    print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")
