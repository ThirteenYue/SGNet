import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
import random
import torch
from torchvision import transforms
import sys


class Dataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(Dataset, self).__init__()
        self.flag = "train" if train else "val"
        data_root = os.path.join(root, "ESDI", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transform = transforms
        # 读取数据集
        img_names = sorted([i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".png")])
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]

        prior_names = sorted([i for i in os.listdir(os.path.join(data_root, "seg_priors")) if i.endswith(".png")])
        self.prior_list = [os.path.join(data_root, "seg_priors", i) for i in prior_names]

        mask_names = sorted([i for i in os.listdir(os.path.join(data_root, "masks")) if i.endswith(".png")])
        self.mask_list = [os.path.join(data_root, "masks", i) for i in mask_names]

    def __getitem__(self, idx):
        # 获取当前样本的路径（用于调试）
        # img_path = self.img_list[idx]
        # prior_path = self.prior_list[idx]
        # mask_path = self.mask_list[idx]
        # 调试信息：打印当前读取的文件路径
        # print(f"路径: {img_path}, {prior_path}, {mask_path}")
        # prior = cv2.imread(prior_path)
        # if prior is None:
        #     raise FileNotFoundError(f"Failed to load prior: {prior_path}")

        image = cv2.imread(self.img_list[idx])  # cv2.imread默认以BGR格式读取
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prior = cv2.imread(self.prior_list[idx])  # 先验类别知识
        pri = cv2.cvtColor(prior, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
        mask = mask // 255

        seed = 42

        if self.transform is not None:
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)

            aug = self.transform(image=img, prior=pri, mask=mask)
            img = aug["image"]
            pri = aug["prior"]
            mask = aug["mask"]

        img = transforms.ToTensor()(img)
        pri = transforms.ToTensor()(pri)
        mask = np.int64(mask)

        return img, pri, mask

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':

    dataset = Dataset(root="../data", train=True)

    # # 测试第一个样本
    img, pri, label = dataset[0]
    print(img.shape, pri.shape, label.shape)

