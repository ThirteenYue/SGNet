"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：本代码横着代表预测值，竖着代表真实值
二分类示例：
       \ 真实   | 正例   反例
    预测 \------|------------
    正例        | TP      FP
    反例        | FN      TN
axis=0代表行
axis=1代表列
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)
        self.eps = 1e-10

    # def get_tp_fp_tn_fn(self):
    #     tp = np.diag(self.confusionMatrix)
    #     fp = self.confusionMatrix.sum(axis=0) - np.diag(self.confusionMatrix)
    #     fn = self.confusionMatrix.sum(axis=1) - np.diag(self.confusionMatrix)
    #     tn = np.diag(self.confusionMatrix).sum() - (tp+fp+fn)
    #     return tp, fp, tn, fn

    def OverallAccuracy(self):  # Overall Accuracy
        #  OA = (TP + TN) / (TP + FP + FN + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def ClassPrecision(self):  # 各类别精确率
        # Precision = (TP) / TP + FP
        ClassPre = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + self.eps)
        return ClassPre  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测精确率

    def MeanPrecision(self):  # 平均精确率
        classAcc = self.ClassPrecision()
        Mean_Precision = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return Mean_Precision  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def ClassRecall(self):
        # Recall = TP/(TP+FP)
        Class_Recall = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + self.eps)
        return Class_Recall  # 返回的是一个列表值，如：[0.48, 0.98, 0.76]，表示类别1 2 3各类别的预测召回率

    def MeanRecall(self):
        classRecall = self.ClassRecall()
        Mean_Recall = np.nanmean(classRecall)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return Mean_Recall  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def ClassIntersectionOverUnion(self):
        # IoU = TP / (TP + FP + FN)
        tp = np.diag(self.confusionMatrix)
        fp = self.confusionMatrix.sum(axis=0) - tp  # 预测为某类但实际不是
        fn = self.confusionMatrix.sum(axis=1) - tp  # 实际是某类但预测不是
        union = tp + fp + fn
        IoU = tp / (union + self.eps)
        return IoU # 返回单个值

    def MeanIntersectionOverUnion(self):
        IoU = self.ClassIntersectionOverUnion()
        MIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # 频权交并比是根据每一类出现的频率设置权重，权重乘以每一类的IoU并进行求和
        # FWIOU =  [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)+self.eps
        iou = self.ClassIntersectionOverUnion()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def ClassF1Score(self):
        precision = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + self.eps)
        recall = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + self.eps)
        Class_f1score = 2 * precision * recall / (precision + recall + self.eps)
        return Class_f1score

    def MeanF1Score(self):
        F1 = self.ClassF1Score()
        MF1 = np.nanmean(F1)
        return MF1

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype('int') + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    imgPredict = np.array([0, 1, 1, 2, 2, 0])  # 可直接换成预测图片
    imgLabel = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成标注图片
    metric = SegmentationMetric(3)  # 3表示有3个分类，有几个分类就填几
    metric.addBatch(imgPredict, imgLabel)

    OA = metric.OverallAccuracy()
    CP = metric.ClassPrecision()
    MP = metric.MeanPrecision()
    CR = metric.ClassRecall()
    MR = metric.MeanRecall()
    CIoU = metric.ClassIntersectionOverUnion()
    MIoU = metric.MeanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    CF1 = metric.ClassF1Score()
    MF1 = metric.MeanF1Score()

    print('Overall Accuracy is : %f' % OA)
    print('Class Precision is :', CP)  # List of class Precision
    print('Mean Precision is : %f' % MP)
    print('Recall is :', CR)  # List of class Recall
    print('mre is : %f' % MR)
    print('Class IoU is :', CIoU)  # List of class IoU
    print('Mean IoU is : %f' % MIoU)
    print('FWIoU is : %f' % FWIoU)
    print('Class F1 Score is :', CF1)  # List of class FI
    print('Mean F1 Score is :%f' % MF1)

    # print(dir(SegmentationMetric))

