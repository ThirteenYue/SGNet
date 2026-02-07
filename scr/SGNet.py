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


# UpSample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')
    return src


class SGNet(nn.Module):
    def __init__(self,in_channels, prior_channels, out_channels):
        super().__init__()
        self.prior_channels = prior_channels
        encoder_channels = [64, 128, 256, 512]
        self.encoder_1 = RSU7(in_channels, 16, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.encoder_2 = RSU6(64, 32, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.encoder_3 = RSU5(128, 64, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.encoder_4 = RSU4(256, 128, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.encoder_5 = RSU4F(960, 256, 512)

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
        self.decoder_3 = RSU5(512, 128, 128)
        self.decoder_2 = RSU6(256, 64, 64)
        self.decoder_1 = RSU7(128, 32, 32)

        self.deeploss_5 = nn.Conv2d(512, out_channels, 3, padding=1)
        self.deeploss_4 = nn.Conv2d(256, out_channels, 3, padding=1)
        self.deeploss_3 = nn.Conv2d(128, out_channels, 3, padding=1)
        self.deeploss_2 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.deeploss_1 = nn.Conv2d(32, out_channels, 3, padding=1)

        self.outconv = nn.Conv2d(out_channels*5, out_channels, 1)

    def forward(self, x, x_seg):
        encoder_1 = self.encoder_1(x)
        encoder_1_down = self.pool1(encoder_1)
        encoder_2 = self.encoder_2(encoder_1_down)
        encoder_2_down = self.pool2(encoder_2)
        encoder_3 = self.encoder_3(encoder_2_down)
        encoder_3_down = self.pool3(encoder_3)
        encoder_4 = self.encoder_4(encoder_3_down)
        encoder_4_down = self.pool4(encoder_4)

        encoder_1_5 = _upsample_like(encoder_1, encoder_4_down)
        encoder_2_5 = _upsample_like(encoder_2, encoder_4_down)
        encoder_3_5 = _upsample_like(encoder_3, encoder_4_down)
        encoder_5 = self.encoder_5(torch.cat((encoder_1_5, encoder_2_5, encoder_3_5, encoder_4_down), 1))

        tda1 = self.tda1(encoder_1_down)
        tda2 = self.tda2(encoder_2_down)
        tda3 = self.tda3(encoder_3_down)
        tda4 = self.tda4(encoder_4_down)

        tda2_2 = self.Conv2_2(encoder_2_down)
        tda3_3 = self.Conv3_3(encoder_3_down)
        tda4_4 = self.Conv4_4(encoder_4_down)
        mps5 = self.mps5(tda1, tda2_2, tda3_3, tda4_4, encoder_5)

        skg1 = self.skg1(tda1, mps5, x_seg)
        skg2 = self.skg2(tda2, mps5, x_seg)
        skg3 = self.skg3(tda3, mps5, x_seg)
        skg4 = self.skg4(tda4, mps5, x_seg)

        mps5_1 = self.mps5_1(mps5)

        # Start Decoder
        decoder_4 = self.decoder_4(torch.cat((mps5_1, skg4), dim=1))
        decoder_4_up = _upsample_like(decoder_4, skg3)

        decoder_3 = self.decoder_3(torch.cat((decoder_4_up, skg3), dim=1))
        decoder_3_up = _upsample_like(decoder_3, skg2)

        decoder_2 = self.decoder_2(torch.cat((decoder_3_up, skg2), dim=1))
        decoder_2_up = _upsample_like(decoder_2, skg1)
        decoder_1 = self.decoder_1(torch.cat((decoder_2_up, skg1), dim=1))

        # deep supervision
        deep_feature_5 = self.deeploss_5(mps5_1)
        D_5 = _upsample_like(deep_feature_5, x)
        deep_feature_4 = self.deeploss_4(decoder_4)
        D_4 = _upsample_like(deep_feature_4, x)
        deep_feature_3 = self.deeploss_3(decoder_3)
        D_3 = _upsample_like(deep_feature_3, x)
        deep_feature_2 = self.deeploss_2(decoder_2)
        D_2 = _upsample_like(deep_feature_2, x)
        deep_feature_1 = self.deeploss_1(decoder_1)
        D_1 = _upsample_like(deep_feature_1, x)

        # return D_5, D_4, D_3, D_2, D_1
        output = self.outconv(torch.cat((D_5, D_4, D_3, D_2, D_1), dim=1))
        return output


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
