import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss0(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, predicted, target):
        c_losses = []
        for c in range(predicted.shape[1]):
            cross_entropy = F.binary_cross_entropy(predicted[:, c, :, :], target[:, c, :, :])
            focal_loss = cross_entropy * (1 - torch.exp(-cross_entropy)) ** self.gamma
            c_losses.append(focal_loss)

        c_losses = torch.stack(c_losses, dim=0)
        weighted_losses = 1 * c_losses
        loss = weighted_losses.sum()
        return loss


class WeightedBCE(torch.nn.Module):
    def __init__(self, weights_per_class):
        super(WeightedBCE, self).__init__()
        self.weights_per_class = weights_per_class

    def forward(self, predicted, target):
        # put different weights on different classes
        weights = torch.tensor(self.weights_per_class).to(predicted.device)
        c_losses = []
        for c in range(predicted.shape[1]):
            c_losses.append(F.binary_cross_entropy(predicted[:, c, :, :], target[:, c, :, :]))

        c_losses = torch.stack(c_losses, dim=0)
        weighted_losses = weights * c_losses
        loss = weighted_losses.sum()
        return loss


class ClassBalanceLoss(torch.nn.Module):
    def __init__(self, pixel_per_classes_distr, beta=0.9999, gamma=2.0):
        super(ClassBalanceLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.pixel_per_classes_distr = pixel_per_classes_distr
        self.loss_type = "cross_entropy"

    def forward(self, predicted, target):
        effective_num = 1.0 - np.power(self.beta, self.pixel_per_classes_distr)
        weights = (1.0 - self.beta) / effective_num
        weights = torch.tensor(weights / np.sum(weights)).float().to(predicted.device)
        c_losses = []
        for c in range(predicted.shape[1]):

            if self.loss_type == "cross_entropy":
                lossona = F.binary_cross_entropy(predicted[:, c, :, :], target[:, c, :, :])

            elif self.loss_type == "focal":
                cross_entropy = F.binary_cross_entropy(predicted[:, c, :, :], target[:, c, :, :])
                lossona = cross_entropy * (1 - torch.exp(-cross_entropy)) ** self.gamma

            c_losses.append(lossona)

        c_losses = torch.stack(c_losses, dim=0)
        weighted_losses = weights * c_losses
        loss = weighted_losses.sum()
        return loss


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        # Flatten the predictions and targets
        predicted_flat = predicted.flatten()
        target_flat = target.flatten()

        # Intersection and Union
        intersection = torch.sum(predicted_flat * target_flat)
        union = torch.sum(predicted_flat) + torch.sum(target_flat)

        # Dice Coefficient
        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss
        dice_loss = 1.0 - dice_coefficient

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=(0.1, 0.9), gamma=3,ignore_index=255, reduction='mean'):
        """
        Focal Loss for imbalanced segmentation tasks.

        参数:
            alpha (float or tuple): 平衡因子，可以是标量（全局权重）或元组（每个类别的权重）。
            gamma (float): 聚焦参数，gamma越大，难样本的权重越高。
            size_average (bool): 已弃用，改用reduction参数。
            ignore_index (int): 忽略的标签值（如255）。
            reduction (str): 损失聚合方式，可选 'none', 'mean', 'sum'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction  # 替代 size_average

    def forward(self, inputs, targets):
        """
        输入:
            inputs: 模型输出，形状为 (B, C, H, W)，未经过softmax。
            targets: 真实标签，形状为 (B, H, W)，值为 [0, C-1] 或 ignore_index。
        """
        # 确保 inputs 是 log probabilities
        log_probs = F.log_softmax(inputs, dim=1)

        # 获取有效像素的索引（忽略 ignore_index）
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device)  # 如果没有有效像素，返回0

        # 提取有效像素的 log_probs 和 targets
        log_probs = log_probs.permute(0, 2, 3, 1).reshape(-1, inputs.size(1))  # (B*H*W, C)
        targets = targets.masked_select(valid_mask)  # (N,)

        # 计算交叉熵损失（仅对有效像素）
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')  # (N,)

        # 计算 pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # 动态调整 alpha（如果 alpha 是元组，则按类别选择）
        if isinstance(self.alpha, (list, tuple)):
            alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
        else:
            alpha_t = self.alpha

        # 计算 Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss