import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class MPS4(nn.Module):
    def __init__(self, in_channels, out_channels):  # in:256 out:256
        super(MPS4, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)  #可选卷积和大小

        self.pcr1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),  # 尺寸减半
            nn.ReLU(inplace=True))
        self.pcr2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),  # 尺寸减半
            nn.ReLU(inplace=True))
        self.pcr3 = nn.Sequential(
            nn.Conv2d(in_channels=(in_channels//4)*3, out_channels=in_channels//4, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),  # 尺寸减半
            nn.ReLU(inplace=True))

        self.pc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels//4, out_channels=out_channels//4, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
        )
        self.pc2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 4, out_channels=out_channels // 4, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),)

    def forward(self, x1, x2, x3, x4):
        r1 = self.pcr1(x1)     # res1路径
        r12 = self.pc1(r1)    # res1路径
        r123 = self.pc2(r12)  # res1路径

        r1_2 = r1+x2
        r2 = self.pcr2(r1_2)

        r1_2_3 = torch.cat((r12, r2, x3), dim=1)
        r3 = self.pcr3(r1_2_3)

        r1_2_3_4 = torch.cat((r3, r123, x4), dim=1)
        r4 = self.relu(r1_2_3_4)
        o4 = self.conv(torch.cat((r4, x4), dim=1))  # C:256, H:25, W:25
        return o4


class MPS5(nn.Module):
    def __init__(self, in_channels, out_channels):  # in:512 out:512
        super(MPS5, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.pcr1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels//12, out_channels=in_channels//12, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=in_channels//12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2))  # 在ReLU之前使用最大池化可以来节省一些计算
        self.pcr2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels//6, out_channels=in_channels//12, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=in_channels // 12),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2))  # 尺寸减半
        self.pcr3 = nn.Sequential(
            nn.Conv2d(in_channels=(in_channels//12)*3 , out_channels=in_channels // 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=in_channels // 12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2))  # 尺寸减半

        self.pc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels//12, out_channels=in_channels//12, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=in_channels // 12),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
        )
        self.pc2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 12, out_channels=in_channels // 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=in_channels // 12),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),)

    def forward(self, x1, x2, x3, x4, x5):
        r1 = self.pcr1(x1)     # res1路径
        r12 = self.pc1(r1)    # res1路径
        r123 = self.pc2(r12)  # res1路径
        r1_2 = torch.cat((r1, x2),dim=1)
        # r1_2 = r1+x2
        r2 = self.pcr2(r1_2)
        r1_2_3 = torch.cat((r12, r2, x3), dim=1)
        r3 = self.pcr3(r1_2_3)
        r1_2_3_4 = torch.cat((r3, r123, x4), dim=1)
        r5 = self.relu(r1_2_3_4)
        o5 = self.conv(torch.cat((r5, x4, x5), dim=1))
        return o5


if __name__ == '__main__':
    model4 = MPS4(in_channels=256, out_channels=256)
    model5 = MPS5(in_channels=768, out_channels=512)
    x1 = torch.randn((1, 64, 100, 100))
    x2 = torch.randn((1, 64, 50, 50))
    x3 = torch.randn((1, 64, 25, 25))
    x4 = torch.randn((1, 64, 13, 13))
    x5 = torch.randn((1, 512, 13, 13))
    # out = model4(x1, x2, x3, x4)
    out = model5(x1, x2, x3, x4, x5)  # C:2048, H:25, W:25
    print(out.shape)

    flops, params = profile(model5, inputs=(x1, x2, x3, x4, x5), verbose=False)
    print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")

