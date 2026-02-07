import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

# 设置Matplotlib使用Agg后端（无GUI）
matplotlib.use('Agg')  # 这行必须在导入pyplot之前
import matplotlib.pyplot as plt
import numpy as np
import os


class TDSAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, return_attention=False):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.return_attention = return_attention  # 新增：是否返回注意力权重

        # 确保每个维度至少为1
        self.Height = max(1, min(reduction_ratio, in_channels // reduction_ratio))
        self.Width = max(1, min(reduction_ratio, in_channels // reduction_ratio))
        self.Channels = max(1, min(reduction_ratio, in_channels // reduction_ratio))

        self.out_channels = in_channels  # 恢复通道维数

        self.f_Height = nn.Conv2d(in_channels, self.Height, kernel_size=1)
        self.f_Width = nn.Conv2d(in_channels, self.Width, kernel_size=1)
        self.f_Channels = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.o_channels = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

        self.alpha = nn.Parameter(torch.zeros(1))  # 初始化H维度注意力权重为零
        self.beta = nn.Parameter(torch.zeros(1))  # 初始化W维度注意力权重为零
        self.gamma = nn.Parameter(torch.zeros(1))  # 初始化C维度注意力权重为零

    def forward(self, x):
        B, C, H, W, = x.size()
        device = x.device

        # 计算重塑矩阵
        f_hei = self.f_Height(x)  # B, C, H, W
        f_h = f_hei.permute(0, 3, 2, 1).contiguous().view(B * W, -1, H).permute(0, 2, 1)  # 输出 B*W, H, C
        f_w = f_hei.permute(0, 2, 1, 3).contiguous().view(B * H, -1, W).permute(0, 2, 1)  # 输出 B*H, W, C

        # 计算转置矩阵
        f_wid = self.f_Width(x)  # B, C, H, W
        ff_h = f_wid.permute(0, 3, 1, 2).contiguous().view(B * W, -1, H)  # 输出 B*W, C, H
        ff_w = f_wid.permute(0, 2, 1, 3).contiguous().view(B * H, -1, W)  # 输出 B*H, C, W

        # 计算通道注意力
        f_ch = self.f_Channels(x)  # B, C, H, W
        f_c = f_ch.contiguous().view(B, -1, H * W)  # 输出 B, C, H*W
        ff_c = f_ch.permute(0, 2, 3, 1).contiguous().view(B, H * W, -1)  # 输出 B, H*W, C
        Affinity_c = F.softmax(torch.matmul(f_c, ff_c), dim=2)  # 输出 B, C, C  亲和矩阵/关系矩阵
        out_C = torch.matmul(ff_c, Affinity_c).view(B, -1, H, W)

        # 计算高度和宽度注意力分数
        E_h = torch.bmm(f_h, ff_h)  # B*W, H, H
        mask = torch.eye(H, device=device).unsqueeze(0).repeat(B * W, 1, 1) * -1e9  # 对角线设为-1e9，相当于softmax后接近0）
        E_h = (E_h + mask).view(B, W, H, H).permute(0, 2, 1, 3)  # B, H, W, H
        E_w = torch.bmm(f_w, ff_w).view(B, H, W, W)  # B, H, W, W

        # 合并注意力
        concate = F.softmax(torch.cat((E_h, E_w), dim=3), dim=3)  # B, H, W, H+W
        h_att = concate[:, :, :, 0:H].permute(0, 2, 1, 3).contiguous().view(B * W, H, H)  # B*W, H, H  纵向稀疏注意力
        w_att = concate[:, :, :, H:H + W].contiguous().view(B * H, W, W)  # B*H, W, W  横向稀疏注意力

        # 计算最终输出
        f_cha = self.o_channels(x)  # B, C, H, W
        P_H = f_cha.permute(0, 3, 1, 2).contiguous().view(B * W, -1, H)  # 输出 B*W, C, H
        P_W = f_cha.permute(0, 2, 1, 3).contiguous().view(B * H, -1, W)  # 输出 B*H, C, W
        out_H = torch.bmm(P_H, h_att.permute(0, 2, 1)).view(B, W, -1, H).permute(0, 2, 3, 1)  # B, C, H, W
        out_W = torch.bmm(P_W, w_att.permute(0, 2, 1)).view(B, H, -1, W).permute(0, 2, 1, 3)  # B, C, H, W

        # 如果设置了返回注意力，则返回额外信息
        if self.return_attention:
            # 返回注意力权重和输出
            attention_info = {
                'alpha': self.alpha.detach().cpu().item(),
                'beta': self.beta.detach().cpu().item(),
                'gamma': self.gamma.detach().cpu().item(),
                'height_attention': h_att.detach().cpu(),  # 高度注意力矩阵
                'width_attention': w_att.detach().cpu(),  # 宽度注意力矩阵
                'channel_attention': Affinity_c.detach().cpu()  # 通道注意力矩阵
            }
            return x + self.alpha * out_H + self.beta * out_W + self.gamma * out_C, attention_info
        else:
            return x + self.alpha * out_H + self.beta * out_W + self.gamma * out_C


class AttentionVisualizer:
    def __init__(self, save_dir='attention_visualizations'):
        self.colors = plt.cm.get_cmap('tab10')
        self.save_dir = save_dir

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

    def _save_or_show(self, fig, filename, show=False):
        """
        保存或显示图像
        """
        save_path = os.path.join(self.save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)  # 关闭图形以释放内存

    def visualize_attention_weights(self, attention_info, batch_idx=0, filename='attention_weights.png', show=False):
        """
        可视化注意力权重
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 显示标量权重
        weights = [attention_info['alpha'], attention_info['beta'], attention_info['gamma']]
        labels = ['Alpha (Height)', 'Beta (Width)', 'Gamma (Channel)']
        colors = ['red', 'green', 'blue']

        axes[0, 0].bar(labels, weights, color=colors)
        axes[0, 0].set_title('Scalar Attention Weights')
        axes[0, 0].set_ylabel('Weight Value')
        axes[0, 0].grid(True, alpha=0.3)

        # 添加数值标签
        for i, (label, weight) in enumerate(zip(labels, weights)):
            axes[0, 0].text(i, weight + 0.01, f'{weight:.4f}',
                            ha='center', va='bottom', fontweight='bold')

        # 2. 高度注意力矩阵热力图（取第一个样本的第一个宽度位置）
        h_att = attention_info['height_attention']
        if len(h_att.shape) > 2:
            # 取第一个样本的第一个宽度位置的注意力
            sample_h_att = h_att[batch_idx * h_att.shape[0] // 1]  # 简化处理
            im1 = axes[0, 1].imshow(sample_h_att[:20, :20], cmap='Reds', aspect='auto')
            axes[0, 1].set_title('Height Attention Matrix (20x20)')
            axes[0, 1].set_xlabel('Target Height')
            axes[0, 1].set_ylabel('Source Height')
            plt.colorbar(im1, ax=axes[0, 1])

        # 3. 宽度注意力矩阵热力图（取第一个样本的第一个高度位置）
        w_att = attention_info['width_attention']
        if len(w_att.shape) > 2:
            sample_w_att = w_att[batch_idx * w_att.shape[0] // 1]
            im2 = axes[0, 2].imshow(sample_w_att[:20, :20], cmap='Greens', aspect='auto')
            axes[0, 2].set_title('Width Attention Matrix (20x20)')
            axes[0, 2].set_xlabel('Target Width')
            axes[0, 2].set_ylabel('Source Width')
            plt.colorbar(im2, ax=axes[0, 2])

        # 4. 通道注意力矩阵热力图
        c_att = attention_info['channel_attention']
        if len(c_att.shape) >= 2:
            # 取第一个样本的通道注意力
            sample_c_att = c_att[0]
            # 只显示前20个通道
            im3 = axes[1, 0].imshow(sample_c_att[:20, :20], cmap='Blues', aspect='auto')
            axes[1, 0].set_title('Channel Attention Matrix (20x20)')
            axes[1, 0].set_xlabel('Target Channel')
            axes[1, 0].set_ylabel('Source Channel')
            plt.colorbar(im3, ax=axes[1, 0])

        # 5. 三维注意力权重分布
        x = np.arange(3)
        y = weights
        axes[1, 1].bar(x, y, color=colors, alpha=0.7)
        axes[1, 1].set_title('3D Attention Weights Distribution')
        axes[1, 1].set_xlabel('Attention Type')
        axes[1, 1].set_ylabel('Weight Value')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(labels)
        axes[1, 1].grid(True, alpha=0.3)

        # 6. 注意力权重统计信息
        axes[1, 2].axis('off')
        stats_text = (
            f"Alpha (Height): {weights[0]:.6f}\n"
            f"Beta (Width): {weights[1]:.6f}\n"
            f"Gamma (Channel): {weights[2]:.6f}\n\n"
            f"Max Height Attention: {h_att.max().item():.4f}\n"
            f"Min Height Attention: {h_att.min().item():.4f}\n"
            f"Mean Height Attention: {h_att.mean().item():.4f}\n\n"
            f"Max Width Attention: {w_att.max().item():.4f}\n"
            f"Min Width Attention: {w_att.min().item():.4f}\n"
            f"Mean Width Attention: {w_att.mean().item():.4f}"
        )
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        self._save_or_show(fig, filename, show)

    def visualize_multi_sample_attention(self, attention_list, num_samples=5, filename='multi_sample_attention.png',
                                         show=False):
        """
        可视化多个样本的注意力权重
        """
        fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))

        for i in range(num_samples):
            if i < len(attention_list):
                att_info = attention_list[i]

                # 第一行：高度注意力
                h_att = att_info['height_attention']
                if len(h_att.shape) > 2:
                    sample_h_att = h_att[0]
                    axes[0, i].imshow(sample_h_att[:10, :10], cmap='Reds', aspect='auto')
                axes[0, i].set_title(f'Sample {i + 1}: Height')
                axes[0, i].set_xlabel('Target')
                axes[0, i].set_ylabel('Source')

                # 第二行：宽度注意力
                w_att = att_info['width_attention']
                if len(w_att.shape) > 2:
                    sample_w_att = w_att[0]
                    axes[1, i].imshow(sample_w_att[:10, :10], cmap='Greens', aspect='auto')
                axes[1, i].set_title(f'Sample {i + 1}: Width')
                axes[1, i].set_xlabel('Target')
                axes[1, i].set_ylabel('Source')

                # 第三行：标量权重
                weights = [att_info['alpha'], att_info['beta'], att_info['gamma']]
                axes[2, i].bar(['α', 'β', 'γ'], weights, color=['red', 'green', 'blue'])
                axes[2, i].set_title(f'Sample {i + 1}: Weights')
                axes[2, i].set_ylabel('Value')
                axes[2, i].grid(True, alpha=0.3)
            else:
                # 如果没有足够的样本，隐藏多余的子图
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                axes[2, i].axis('off')

        plt.tight_layout()
        self._save_or_show(fig, filename, show)

    def plot_attention_evolution(self, attention_history, filename='attention_evolution.png', show=False):
        """
        绘制注意力权重随时间的变化
        attention_history: 列表，每个元素是包含alpha, beta, gamma的字典
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 提取历史数据
        steps = range(len(attention_history))
        alphas = [h['alpha'] for h in attention_history]
        betas = [h['beta'] for h in attention_history]
        gammas = [h['gamma'] for h in attention_history]

        # 绘制折线图
        axes[0].plot(steps, alphas, 'r-', label='Alpha (Height)', linewidth=2)
        axes[0].plot(steps, betas, 'g-', label='Beta (Width)', linewidth=2)
        axes[0].plot(steps, gammas, 'b-', label='Gamma (Channel)', linewidth=2)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Weight Value')
        axes[0].set_title('Attention Weights Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 绘制堆叠面积图
        axes[1].stackplot(steps, alphas, betas, gammas,
                          labels=['Alpha', 'Beta', 'Gamma'],
                          colors=['red', 'green', 'blue'], alpha=0.7)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Weight Value')
        axes[1].set_title('Attention Weights Distribution Over Time')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_or_show(fig, filename, show)


if __name__ == '__main__':
    from thop import profile

    print("开始测试TDSAttention模块...")

    # 1. 创建模型，设置return_attention=True
    model = TDSAttention(in_channels=64, reduction_ratio=16, return_attention=True)

    # 2. 创建输入数据
    x = torch.randn((1, 64, 200, 200))  # 批大小为1

    # 3. 前向传播，获取注意力和输出
    output, attention_info = model(x)
    print(f"Output shape: {output.shape}")

    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")

    # 4. 创建可视化工具
    visualizer = AttentionVisualizer(save_dir='attention_results')

    # 5. 可视化单个样本的注意力
    print("\n=== 单个样本注意力可视化 ===")
    visualizer.visualize_attention_weights(
        attention_info,
        batch_idx=0,
        filename='single_sample_attention.png',
        show=False  # 设置为True则显示图像
    )

    # 6. 收集多个样本的注意力
    print("\n=== 收集多个样本的注意力 ===")
    attention_list = []

    for i in range(3):  # 模拟处理3个不同输入
        test_input = torch.randn((1, 64, 200, 200))
        _, att_info = model(test_input)
        attention_list.append(att_info)

    visualizer.visualize_multi_sample_attention(
        attention_list,
        num_samples=3,
        filename='multi_sample_attention.png',
        show=False
    )

    # 7. 模拟训练过程中的注意力权重变化
    print("\n=== 注意力权重演化过程 ===")
    attention_history = []

    # 重新初始化模型参数
    model.alpha.data = torch.zeros(1)
    model.beta.data = torch.zeros(1)
    model.gamma.data = torch.zeros(1)

    # 模拟梯度下降更新
    optimizer = torch.optim.SGD([model.alpha, model.beta, model.gamma], lr=0.01)

    for step in range(50):
        # 模拟损失计算和反向传播
        optimizer.zero_grad()

        # 假设损失函数鼓励alpha增加，beta减少，gamma保持不变
        loss = -model.alpha + model.beta  # 简单的示例损失

        loss.backward()
        optimizer.step()

        # 记录当前权重
        attention_history.append({
            'alpha': model.alpha.item(),
            'beta': model.beta.item(),
            'gamma': model.gamma.item()
        })

    # 可视化演化过程
    visualizer.plot_attention_evolution(
        attention_history,
        filename='attention_evolution.png',
        show=False
    )

    print("\n=== 测试完成 ===")
    print(f"所有可视化结果已保存到: {os.path.abspath('attention_results')}")

    # 列出保存的文件
    if os.path.exists('attention_results'):
        print("\n保存的文件:")
        for file in os.listdir('attention_results'):
            if file.endswith('.png'):
                print(f"  - {file}")