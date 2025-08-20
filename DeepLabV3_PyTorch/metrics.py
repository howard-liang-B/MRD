import numpy as np
from sklearn import metrics
import torch

# Source: https://github.com/sacmehta/ESPNet/blob/master/train/IOUEval.py

class IOUEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
        self.mIOU = 0
        self.batchCount = 1

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses)
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        predict = predict.cpu().numpy().flatten()
        gth = gth.cpu().numpy().flatten()

        epsilon = 0.00000001
        hist = self.compute_hist(predict, gth)
        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
        mIou = np.nanmean(per_class_iu)

        self.overall_acc +=overall_acc
        self.per_class_acc += per_class_acc
        self.per_class_iu += per_class_iu
        self.mIOU += mIou
        self.batchCount += 1

    def getMetric(self):
        overall_acc = self.overall_acc/self.batchCount
        per_class_acc = self.per_class_acc / self.batchCount
        per_class_iu = self.per_class_iu / self.batchCount
        mIOU = self.mIOU / self.batchCount

        return overall_acc, per_class_acc, per_class_iu, mIOU

def get_roc_auc(gt_mask, outputs):
  # 處理 gt_mask 和 outputs
  gt_mask = (gt_mask > 0).astype(int)  # 大於 0 的值轉為 1，其餘為 0

  max_outs = torch.max(outputs, dim=1)[0]  # 取得每個像素位置的最大值，形狀為 (1, 512, 512)
  sigmoid_outs = torch.sigmoid(max_outs)  # 對最大值應用 Sigmoid，範圍在 [0, 1]
  sigmoid_outs = sigmoid_outs.cpu().detach().numpy().squeeze()  # 轉換為 NumPy，並移除多餘的維度，形狀為 (512, 512)

  # 展平數組以便於 roc_curve 使用
  y_true = gt_mask.ravel()  # 二進制標籤，形狀為 (512*512,)
  y_score = sigmoid_outs.ravel()  # 預測分數，形狀為 (512*512,)

  # 打印 shape 和 裡面的value 和 unique
  # print(gt_mask.shape, sigmoid_outs.shape)
  # print(y_true, y_score)
  # print(f'## y_true: {np.unique(y_true)} , ## y_score: {np.unique(y_score)}')

  # roc_curve 需要 y_true, y_score 大小相同
  fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
  auc = metrics.auc(fpr, tpr)
  # print(f'@ fpr: {fpr}, @ tpr: {tpr}, @ auc: {auc}')

  return fpr, tpr, auc

 