import os, cv2
import numpy as np
import torch
import time
import datetime
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from my_dataset import Dataset
from MLtrainer import SegmentationMetric
from loss import FocalLoss
import random
import albumentations as albu


# from scr.SGNet import SGNet  # transformer decoder
# from scr.SGNet_0 import SGNet  # U2Net decoder
# from scr.SGNet_1 import SGNet  # U2Net backbone
# from scr.SGNet_2 import SGNet  # U2Net backbone
from scr.SGNet_3 import SGNet  # U2Net backbone
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True


def SG_Net():
    return SGNet(in_channels=3, prior_channels=3, out_channels=2)


def train(data_loader, model, optimizer, criterion, device, cosine_schedule):
    model.train()
    running_loss = 0.0
    running_OA = 0.0
    running_mIoU = 0.0
    lr = optimizer.param_groups[0]["lr"]  # 初始化学习率
    count = 0
    for batch_idx, (inputs, prior, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        prior = prior.to(device)  # SAM引导
        labels = labels.to(device)
        # 前向传播
        outputs = model(inputs, prior)   # 加入SAM
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)  # 梯度裁剪
        optimizer.step()
        # 更新学习率
        # cosine_schedule.step()  # 每个step更新学习率
        lr = optimizer.param_groups[0]["lr"]
        # 无梯度下计算指标
        with torch.no_grad():
            # 计算预测结果
            pred = torch.argmax(outputs, dim=1)
            metric = SegmentationMetric(numClass=2)  # 分割的类
            metric.addBatch(pred.cpu().numpy(), labels.cpu().numpy())

            OA = metric.OverallAccuracy()
            MIoU = metric.MeanIntersectionOverUnion()
            running_loss += loss.item() * inputs.size(0)
            running_OA += OA * inputs.size(0)
            running_mIoU += MIoU * inputs.size(0)  # 累积 MIoU
            count += inputs.size(0)

        if batch_idx % args.log_interval == 0:
            print(f'[{batch_idx * len(inputs)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\t'
                  f'Loss:{loss.item():.4f}  \tOA:{OA:.4f}  \tLr: {lr:.6f}  \tT_iou: {MIoU:.4f}')

    # 计算 epoch 平均值
    avg_loss = running_loss / count
    avg_OA = running_OA / count
    avg_mIoU = running_mIoU / count  # 平均 mIoU
    return avg_loss, avg_OA, lr, avg_mIoU


def validation(data_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_OA = 0.0
    running_mIoU = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (inputs, prior, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            prior = prior.to(device)
            labels = labels.to(device)
            # 前向传播预测
            outputs = model(inputs, prior)
            loss = criterion(outputs, labels)
            # 计算预测结果
            pred = torch.argmax(outputs, dim=1)
            metric = SegmentationMetric(numClass=2)  # 分割的类
            metric.addBatch(pred.cpu().numpy(), labels.cpu().numpy())

            OA = metric.OverallAccuracy()
            MIoU = metric.MeanIntersectionOverUnion()

            running_loss += loss.item() * inputs.size(0)
            running_OA += OA * inputs.size(0)
            running_mIoU += MIoU * inputs.size(0)  # 累积 MIoU
            count += inputs.size(0)
    return running_loss / count, running_OA / count,  running_mIoU / count



def create_lr_scheduler(optimizer, num_step, epochs, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0
    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_mat = "/home/ubuntu/data0/ygt/CNN/SAMP_Net/train_txt/ESDI/{}-L0.0001_B8_E2000_CL_25_1.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # seed_everything(42)

    data_transforms = {
        'train': albu.Compose([
            # albu.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),  # 平移，缩放，和旋转
            albu.GaussianBlur(),  # 随机高斯模糊
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),  # 随机亮度对比度变化
            albu.Normalize(),
        ], additional_targets={'prior': 'image', 'mask': 'mask'}),

        'valid': albu.Compose([
            # albu.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
            # albu.RandomCrop(256, 256),
            albu.Normalize(),
        ], additional_targets={'prior': 'image', 'mask': 'mask'})}

    train_dataset = Dataset(args.data_folder, train=True, transforms=data_transforms['train'])
    val_dataset = Dataset(args.data_folder, train=False, transforms=data_transforms['valid'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True)
    print(f"Using {len(train_dataset)} images for training, {len(val_dataset)} images for validation.")   # 给出训练和验证数据的数量

    model = SG_Net().to(device)

    # 更新策略
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # num_steps_per_epoch = len(train_loader)
    # cosine_schedule = create_lr_scheduler(optimizer, num_step=num_steps_per_epoch, epochs=args.epochs, warmup=True)  # 热身训练
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000, eta_min=0.00000001)         # 余弦退火
    # cosine_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5,
    #                                                        verbose=False, threshold=1e-4, threshold_mode='rel',      # 自适应衰减
    #                                                        cooldown=0, min_lr=0, eps=1e-8)
    # cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)  # 带warmup的余弦退火

    criterion = nn.CrossEntropyLoss()  # 正常计算loss
    # criterion = FocalLoss(ignore_index=255)   # 前景和背景不均衡

    # 绘制指标
    x = []
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    lr_list = []
    viou = []
    tiou = []
    best_loss = float('inf')
    best_iou = float('-inf')

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"================== Epoch: {epoch} ==================")

        tra_loss, tra_acc, lr, tra_IoU = train(train_loader, model, optimizer, criterion, device, cosine_schedule)
        val_loss, val_acc, val_IoU = validation(val_loader, model, criterion, device)

        # cosine_schedule.step(val_IoU)  # 自适应 每个epoch更新
        cosine_schedule.step()  # 余弦  每个epoch更新(之前一直没注意到^^)

        with open(results_mat, "a") as f:
            f.write("t_acc:{:.4f}  v_acc:{:.4f}  t_loss:{:.4f}  v_loss:{:.4f}  t_IoU:{:.4f}  v_IoU:{:.4f}  lr:{:.7f} \n"
                    .format(tra_acc,  val_acc, tra_loss, val_loss, tra_IoU, val_IoU, lr))

        print(f'[End of Epoch {epoch}]\tTra Loss: {tra_loss:.5f}\tTra Acc: {tra_acc:.5f}\ttra_Iou: {tra_IoU:.5f}')
        print(f'[End of Epoch {epoch}]\tVal Loss: {val_loss:.5f}\tVal Acc: {val_acc:.5f}\tval_IoU: {val_IoU:.5f}')

        # matplotlib
        x.append(epoch)
        train_loss_list.append(tra_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(tra_acc)
        val_acc_list.append(val_acc)
        lr_list.append(lr)
        viou.append(val_IoU)
        tiou.append(tra_IoU)

        plt.clf()
        plt.plot(x, tiou, 'r', lw=1)
        plt.plot(x, viou, 'b', lw=1)  # lw为曲线宽度
        plt.title("MIoU")
        plt.xlabel("epoch")
        plt.ylabel("MIou")
        plt.legend(["Tra_MIou", "Val_MIou"])
        plt.savefig('/home/ubuntu/data0/ygt/CNN/SAMP_Net/cure/ESDI/MIoU_L0.0001_B8_E2000_CL_25_1.png')

        plt.clf()
        plt.plot(x, train_loss_list, 'r', lw=1)  # lw为曲线宽度
        plt.plot(x, val_loss_list, 'b', lw=1)
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["train_loss", "val_loss"])
        plt.savefig('/home/ubuntu/data0/ygt/CNN/SAMP_Net/cure/ESDI/loss_L0.0001_B8_E2000_CL_25_1.png')

        plt.clf()
        plt.plot(x, train_acc_list, 'r', lw=1)  # lw为曲线宽度
        plt.plot(x, val_acc_list, 'b', lw=1)
        plt.title("acc")
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.legend(["train_acc", "val_acc"])
        plt.savefig('/home/ubuntu/data0/ygt/CNN/SAMP_Net/cure/ESDI/acc_L0.0001_B8_E2000_CL_25_1.png')

        plt.clf()
        plt.plot(x, lr_list, 'r', lw=1)  # lw为曲线宽度
        plt.title("lr")
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.legend(["train_lr"])
        plt.savefig('/home/ubuntu/data0/ygt/CNN/SAMP_Net/cure/ESDI/lr_L0.0001_B8_E2000_CL_25_1.png')

        if val_IoU > best_iou:
            best_iou = val_IoU
            # torch.save(model.state_dict(), os.path.join(args.model_dir, f'1000_best_epoch_{epoch}.pt'))  #保存多个模型权重
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'L0.0001_B8_E2000_CL_25.pt'))

        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     torch.save(model.state_dict(), os.path.join(args.model_dir, f'best_epoch_{epoch}.pt'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UNet Training')
    parser.add_argument('--data-folder', type=str, default='/home/ubuntu/data0/ygt/CNN/data', help='path to dataset')
    parser.add_argument('--model-dir', type=str, default='/home/ubuntu/data0/ygt/CNN/SAMP_Net/weight/ESDI', help='model directory')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--betas', default=(0.9, 0.999), type=float, help='betas for Adam optimizer')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log-interval', type=int, default=10, help='logging interval')
    # parser.add_argument('--log-dir', type=str, default='./logs', help='log directory')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    main()