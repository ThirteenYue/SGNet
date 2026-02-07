import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class SAMGuidedFeatureFusion(nn.Module):  # SAM引导的特征融合模块
    """
        in_channels_A: 编码器特征A的通道数
        in_channels_B: 解码器特征B的通道数
        in_channels_C: SAM特征C的通道数 (默认为3)
    """
    def __init__(self, in_channels_A, in_channels_B, in_channels_C):
        super(SAMGuidedFeatureFusion, self).__init__()

        # print(in_channels_A)
        # print(in_channels_B)
        # print(in_channels_C)
        self.Conv1_1 = nn.Conv2d(in_channels_B, in_channels_A, kernel_size=1)

        self.concat_ab = nn.Sequential(
            nn.Conv2d(in_channels_A*2, in_channels_A, 3, padding=1),
            nn.BatchNorm2d(in_channels_A),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

        # SAM特征处理器 (保留空间信息)
        self.sam_processor = nn.Sequential(
            nn.Conv2d(in_channels_C, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),  # 输出单通道注意力图
            nn.Dropout2d(0.5)
        )

        # 输出融合层
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels_A * 2, in_channels_A, 3, padding=1),
            nn.BatchNorm2d(in_channels_A),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

        self.delta = nn.Parameter(torch.zeros(1))
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, A, B, C):
        h, w = A.size()[-2:]
        """
        输入:
            A: 编码器特征 [B, C_A, H, W]
            B: 解码器特征 [B, C_B, H, W]
            C: SAM特征 [B, 3, H, W]
        输出: 融合后的特征 [B, C_A, H, W]
        """

        B = F.interpolate(B, size=(h, w), mode='bilinear', align_corners=False)
        B = self.Conv1_1(B)

        # 拼接A和B
        AB = torch.cat([A, B], dim=1)  # [B, C_A*2, H, W]
        AB = self.concat_ab(AB)  # [B, C_A, H, W]

        # 处理SAM特征生成空间注意力图
        sam_attn = self.sam_processor(C)  # [B, 1, H, W]
        sam_attn = torch.sigmoid(sam_attn)  # 归一化到0-1
        c_sam = F.interpolate(sam_attn, size=(h, w), mode='bilinear', align_corners=False)

        # 应用SAM空间注意力
        AB_attn = AB * (c_sam*self.delta)  # 缺陷区域特征增强
        # 与原始A拼接
        fused = torch.cat([A, AB_attn], dim=1)  # [B, C_A*2, H, W]
        output = self.final_fusion(fused)*self.epsilon + A  # 残差连接
        return output


if __name__ == '__main__':
    model = SAMGuidedFeatureFusion(in_channels_A=64, in_channels_B=512, in_channels_C=3)
    x1 = torch.randn((2, 64, 200, 200))
    x2 = torch.randn((2, 512, 25, 25))
    x3 = torch.randn((2, 3, 200, 200))
    out = model(x1, x2, x3)
    print(out.shape)

    flops, params = profile(model, inputs=(x1, x2, x3), verbose=False)
    print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")
    # FLOPs: 14.500G
    # Params: 0.181M
