from glob import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
# from scr.SGNet import SGNet
from scr.SGNet_3 import SGNet  # U2Net backbone

import os
from MLtrainer import SegmentationMetric
import re
import random
from torchvision import transforms
import albumentations as albu
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'


# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True

# Load pre-trained model and weights
def load_model(device, pre_trained_path):
    model = SGNet(in_channels=3,prior_channels=3,out_channels=2).to(device)
    state_dict = torch.load(pre_trained_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Calculate segmentation metrics
def get_IOU(pred, mask, num_classes=2):
    metric = SegmentationMetric(num_classes)
    metric.addBatch(pred, mask)
    OA = metric.OverallAccuracy()  # Overall Accuracy
    MP = metric.MeanPrecision()  # Mean Pixel Accuracy
    MR = metric.MeanRecall()  # Mean Recall
    MIou = metric.MeanIntersectionOverUnion()  # Mean IoU
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    MDice = 2 * MP * MR / (MP + MR)  # Dice Coefficient      这里MDice就是MF1
    return OA, MP, MR, MIou, FWIoU, MDice  # 这里的M指的是图片中有几个类别的平均


# Convert segmentation mask to RGB image
def uavid2rgb(mask):
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    color_map = {
        0: [0, 0, 0],
        # 1: [255, 150, 50],  # 前景类 - ESDI-荧光橙
        1: [50, 220, 255],  # 前景类 - SD00-荧光蓝
        # 1: [50, 255, 50]    # 荧光绿


        # 0: [155, 38, 182],
        # 1: [14, 135, 204],
        # 2: [124, 252, 0],
        # 3: [255, 20, 147],
        # 4: [169, 169, 169],

        # 0: [128, 0, 0],
        # 1: [128, 64, 128],
        # 2: [0, 128, 0],
        # 3: [128, 128, 0],
        # 4: [64, 0, 128],
        # 5: [192, 0, 192],
        # 6: [64, 64, 0],
        # 7: [0, 0, 0]
    }
    for class_idx, color in color_map.items():
        mask_rgb[mask == class_idx] = color
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb



if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    # seed_everything(42)

    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--data-folder', type=str, default=
    '/home/ubuntu/data0/ygt/CNN/data/SD900/test/images', help='Path to data folder')
    parser.add_argument('--seg-folder', type=str, default=
    '/home/ubuntu/data0/ygt/CNN/data/SD900/test/seg_priors', )
    parser.add_argument('--pre-trained', type=str, default=
    "/home/ubuntu/data0/ygt/CNN/SAMP_Net/weight/SD900/L0.0001_B8_E1000_CL.pt", help='Path to pre-trained weights')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the model on')
    args = parser.parse_args()
    args.device = torch.device("cuda")

    image_files = sorted(glob('{}/*.pn*g'.format(args.data_folder)))
    seg_files = sorted(glob('{}/*.pn*g'.format(args.seg_folder)))

    model = load_model(args.device, args.pre_trained)
    transform = albu.Compose([
        # transforms.RandomCrop((1024, 512)),
        # transforms.Resize((1024, 1024)),
        albu.Normalize(),
    ], additional_targets={"seg_rgb": "image"})

    print('Model loaded')
    print(len(image_files), ' files in folder ', args.data_folder)

    results_dir = '/home/ubuntu/data0/ygt/CNN/SAMP_Net/output/test'
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, "00000_L0.0001_B8_E1000_CL_1.txt")

    Test_OA = []
    Test_MP = []
    Test_MR = []
    Test_MIoU = []
    Test_FWIoU = []
    Test_MDice = []

    with open(metrics_file, "w") as f:
        for i, image_file in enumerate(image_files):
            img_id = os.path.splitext(os.path.basename(image_file))[0]  # 获取ID
            image = cv2.imread(image_file)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            seg_file = re.sub("images", "seg_priors", image_file)
            seg = cv2.imread(seg_file)
            seg_rgb = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)

            mask_file = re.sub("images", "masks", image_file)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

            # pil_img = Image.fromarray(image_rgb)
            aug = transform(image=image_rgb, seg_rgb=seg_rgb, mask=mask)
            torch_img = aug["image"]
            torch_seg = aug["seg_rgb"]
            torch_mask = aug["mask"]

            torch_img = transforms.ToTensor()(torch_img)
            torch_img = torch_img.unsqueeze(0).to(args.device)
            torch_seg = transforms.ToTensor()(torch_seg)
            torch_seg = torch_seg.unsqueeze(0).to(args.device)

            mask = np.array(torch_mask)

            with torch.no_grad():
                logits = model(torch_img, torch_seg)

                probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()   # 计算PR曲线
                pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

            mask = mask//255

            OA, MP, MR, MIoU, FWIoU, MDice = get_IOU(pred, mask)
            # pred = uavid2rgb(pred)  # 输出变色

            probs = (probs * 255).astype(np.uint8)  # 计算PR曲线
            # pred = (pred * 255).astype(np.uint8)  # 预测结果映射到0和255
            output_path = os.path.join(results_dir, os.path.basename(image_file))
            cv2.imwrite(output_path, probs)
            # cv2.imwrite(output_path, pred)

            print(f"OA: {OA:.4f}  MP: {MP:.4f}  MR: {MR:.4f}  MIoU: {MIoU:.4f}  FWIoU: {FWIoU:.4f}  MDice: {MDice:.4f}  ID: [{img_id}]")
            f.write(f"OA: {OA:.4f}  MP: {MP:.4f}  MR: {MR:.4f}  MIoU: {MIoU:.4f}  FWIoU: {FWIoU:.4f}  MDice: {MDice:.4f}  ID: {img_id}\n")

            Test_OA.append(OA)
            Test_MP.append(MP)
            Test_MR.append(MR)
            Test_MIoU.append(MIoU)
            Test_FWIoU.append(FWIoU)
            Test_MDice.append(MDice)

        mean_Test_OA = np.mean(Test_OA)
        mean_Test_MP = np.mean(Test_MP)
        mean_Test_MR = np.mean(Test_MR)
        mean_Test_MIoU = np.mean(Test_MIoU)
        mean_Test_FWIoU = np.mean(Test_FWIoU)
        mean_Test_MDice = np.mean(Test_MDice)
        #
        f.write(f"\nmean_Test_OA: {mean_Test_OA:.4f}")
        f.write(f"\nmean_Test_MP: {mean_Test_MP:.4f}")
        f.write(f"\nmean_Test_MR: {mean_Test_MR:.4f}")
        f.write(f"\nmean_Test_MIoU: {mean_Test_MIoU:.4f}")
        f.write(f"\nmean_Test_FWIoU: {mean_Test_FWIoU:.4f}")
        f.write(f"\nmean_Test_MDice: {mean_Test_MDice:.4f}")

    print("Metrics saved to", metrics_file)
