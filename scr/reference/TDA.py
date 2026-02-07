import torch
import torch.nn as nn
import torch.nn.functional as F


class TDSAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape

class EfficientMDA(nn.Module):
    def __init__(self, in_ch, reduction_ratio=16):
        super(EfficientMDA, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.reduced_ch = max(1, in_ch // reduction_ratio)

        # 通道注意力组件
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, self.reduced_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduced_ch, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力组件
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 维度交互组件
        self.dim_interact = nn.Sequential(
            nn.Conv2d(in_ch, self.reduced_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduced_ch, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

        # 参数化融合权重
        self.alpha = nn.Parameter(torch.tensor([0.3]))  # 通道权重
        self.beta = nn.Parameter(torch.tensor([0.3]))  # 空间权重
        self.gamma = nn.Parameter(torch.tensor([0.4]))  # 维度交互权重

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_fc(x)
        channel_out = x * channel_att * self.alpha

        # 空间注意力
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_cat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_att = self.spatial_conv(spatial_cat)
        spatial_out = x * spatial_att * self.beta

        # 维度交互注意力
        dim_att = self.dim_interact(x)
        dim_out = x * dim_att * self.gamma

        # 加权融合
        out = channel_out + spatial_out + dim_out
        return out + x  # 残差连接


# class Efficient3DAttention(nn.Module):
#     def __init__(self, in_ch, reduction_ratio=16):
#         super().__init__()
#         self.reduction_ratio = reduction_ratio
#         self.reduced_ch = max(1, in_ch // reduction_ratio)
#
#         # 通道压缩层
#         self.channel_reduce = nn.Sequential(
#             nn.Conv2d(in_ch, self.reduced_ch, 1),
#             nn.BatchNorm2d(self.reduced_ch),
#             nn.ReLU(inplace=True)
#
#         # 维度注意力分支
#         self.height_fc = nn.Linear(self.reduced_ch, H)  # 需根据输入动态调整
#         self.width_fc = nn.Linear(self.reduced_ch, W)
#         self.channel_fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(self.reduced_ch, in_ch, 1),
#             nn.Sigmoid())
#
#         # 参数化融合
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#         self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, H, W = x.shape

        # 动态调整全连接层
        if hasattr(self, 'height_fc') and self.height_fc.out_features != H:
            self.height_fc = nn.Linear(self.reduced_ch, H).to(x.device)
        if hasattr(self, 'width_fc') and self.width_fc.out_features != W:
            self.width_fc = nn.Linear(self.reduced_ch, W).to(x.device)

        # 通道压缩
        compressed = self.channel_reduce(x)  # [B, C/r, H, W]

        # 高度注意力
        h_pool = F.adaptive_avg_pool2d(compressed, (H, 1))  # [B, C/r, H, 1]
        h_vec = h_pool.squeeze(-1).permute(0, 2, 1)  # [B, H, C/r]
        h_att = torch.sigmoid(self.height_fc(h_vec)).permute(0, 2, 1)  # [B, H, H]

        # 宽度注意力
        w_pool = F.adaptive_avg_pool2d(compressed, (1, W))  # [B, C/r, 1, W]
        w_vec = w_pool.squeeze(-2).permute(0, 2, 1)  # [B, W, C/r]
        w_att = torch.sigmoid(self.width_fc(w_vec))  # [B, W, W]

        # 通道注意力
        c_att = self.channel_fc(compressed)  # [B, C, 1, 1]

        # 空间注意力融合
        spatial_att = torch.einsum('bhi,bwj->bhwij', h_att, w_att)  # [B, H, W, H, W]
        spatial_att = spatial_att.view(B, H * W, H * W)
        spatial_att = F.softmax(spatial_att, dim=-1)

        # 应用空间注意力
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        spatial_out = torch.bmm(spatial_att, x_flat).permute(0, 2, 1).view(B, C, H, W)

        # 最终融合
        out = self.alpha * spatial_out + self.beta * (x * c_att)
        return out


class CoordinateSparseAttention(nn.Module):
    def __init__(self, in_channels, k_neighbors=8):
        super().__init__()
        self.k = k_neighbors
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.pos_enc = nn.Parameter(torch.randn(1, in_channels, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成坐标网格
        h_grid = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        w_grid = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        coords = torch.cat([h_grid, w_grid], dim=1)  # [B, 2, H, W]

        # 计算K近邻
        coords_flat = coords.view(B, 2, -1).permute(0, 2, 1)  # [B, N, 2]
        dist = torch.cdist(coords_flat, coords_flat)  # [B, N, N]
        _, topk_idx = torch.topk(dist, self.k, dim=2, largest=False)  # [B, N, k]

        # 生成QKV
        qkv = self.to_qkv(x).view(B, 3, C, H * W).permute(1, 0, 3, 2)  # [3, B, N, C]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, C]

        # 稀疏注意力计算
        k_selected = torch.gather(k.unsqueeze(2).expand(-1, -1, self.k, -1),
                                  2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, C))
        v_selected = torch.gather(v.unsqueeze(2).expand(-1, -1, self.k, -1),
                                  2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, C))

        # 注意力分数
        attn = torch.einsum('b n c, b n k c -> b n k', q, k_selected) / (C ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        # 聚合特征
        output = torch.einsum('b n k, b n k c -> b n c', attn, v_selected)
        return output.view(B, C, H, W) + self.pos_enc


