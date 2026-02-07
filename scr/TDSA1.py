import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class TDSAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.Height = min(reduction_ratio, in_channels // reduction_ratio)  # 缩减通道维数(变量)
        self.Width = min(reduction_ratio, in_channels // reduction_ratio)   # 缩减通道维数（变量）
        self.Channels = min(reduction_ratio, in_channels // reduction_ratio)  # 缩减通道维数（变量）

        self.out_channels = in_channels   # 恢复通道维数

        self.f_Height = nn.Conv2d(in_channels, self.Height, kernel_size=1)
        self.f_Width = nn.Conv2d(in_channels, self.Width, kernel_size=1)
        self.f_Channels = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.o_channels = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

        self.alpha = nn.Parameter(torch.zeros(1))  # 初始化H维度注意力权重为零
        self.beta = nn.Parameter(torch.zeros(1))   # 初始化W维度注意力权重为零
        self.gamma = nn.Parameter(torch.zeros(1))  # 初始化C维度注意力权重为零

    def forward(self, x):
        B, C, H, W, = x.size()
        device = x.device
        # 计算重塑矩阵
        f_hei = self.f_Height(x)  # B, C, H, W
        f_h = f_hei.permute(0, 3, 2, 1).contiguous().view(B*W, -1, H).permute(0,2,1)  # 输出 B*W, H, C
        f_w = f_hei.permute(0, 2, 1, 3).contiguous().view(B*H, -1, W).permute(0,2,1)  # 输出 B*H, W, C
        # 计算转置矩阵
        f_wid = self.f_Width(x)  # B, C, H, W
        ff_h = f_wid.permute(0, 3, 1, 2).contiguous().view(B*W, -1, H)  # 输出 B*W, C, H
        ff_w = f_wid.permute(0, 2, 1, 3).contiguous().view(B*H, -1, W)  # 输出 B*H, C, W
        # 计算通道注意力
        f_ch = self.f_Channels(x)  # B, C, H, W
        f_c = f_ch.contiguous().view(B, -1, H*W)  # 输出 B, C, H*W
        ff_c = f_ch.permute(0, 2, 3, 1).contiguous().view(B, H*W, -1)  # 输出 B, H*W, C
        Affinity_c = F.softmax(torch.matmul(f_c, ff_c), dim=2)  # 输出 B, C, C  亲和矩阵/关系矩阵
        out_C = torch.matmul(ff_c, Affinity_c).view(B, -1, H, W)

        # 计算高度和宽度注意力分数
        E_h = torch.bmm(f_h, ff_h)  # B*W, H, H
        mask = torch.eye(H, device=device).unsqueeze(0).repeat(B*W, 1, 1) * -1e9  # 对角线设为-1e9，相当于softmax后接近0）
        E_h = (E_h + mask).view(B, W, H, H).permute(0, 2, 1, 3)  # B, H, W, H
        E_w = torch.bmm(f_w, ff_w).view(B, H, W, W)  # B, H, W, W

        # 合并注意力
        concate = F.softmax(torch.cat((E_h, E_w), dim=3), dim=3)  # B, H, W, H+W
        h_att = concate[:, :, :, 0:H].permute(0, 2, 1, 3).contiguous().view(B*W, H, H)  # B*W, H, H  纵向稀疏注意力
        w_att = concate[:, :, :, H:H+W].contiguous().view(B*H, W, W)  # B*H, W, W  横向稀疏注意力

        # 计算最终输出
        f_cha = self.o_channels(x)  # B, C, H, W
        P_H = f_cha.permute(0, 3, 1, 2).contiguous().view(B*W, -1, H)  # 输出 B*W, C, H
        P_W = f_cha.permute(0, 2, 1, 3).contiguous().view(B*H, -1, W)  # 输出 B*H, C, W
        out_H = torch.bmm(P_H, h_att.permute(0, 2, 1)).view(B, W, -1, H).permute(0, 2, 3, 1)  # B, C, H, W
        out_W = torch.bmm(P_W, w_att.permute(0, 2, 1)).view(B, H, -1, W).permute(0, 2, 1,3)  # B, C, H, W

        return x+self.alpha*out_H+self.beta*out_W+self.gamma*out_C


if __name__ == '__main__':
    model = TDSAttention(in_channels=64, reduction_ratio=16)
    x = torch.randn((1, 64, 200, 200))
    out = model(x)
    print(out.shape)

    flops, params = profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")

