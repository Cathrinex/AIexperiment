import torch
import torchvision
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.patches as patches
import matplotlib.pyplot as plt
import glob

from model import base_net, get_blk, blk_forward
from prediction import cls_predictor, bbox_predictor, flatten_pred, concat_preds
from loss import calc_loss, cls_eval, bbox_eval
from nms import multibox_detection, nms, offset_inverse
from anchor_frame import multibox_target
from data import load_data
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
train_iter = load_data(batch_size)
# 定义锚框与图像的比例
# 底层大尺度特征图采用小比例锚框，用于检测小特征
# 底层大尺度特征图采用大比例锚框，用于检测大特征
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
# 定义锚框长宽比
ratios = [[1, 2, 0.5]] * 5
# 定义每个像素点的锚框数量
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

# 定义TinySSD算法类
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

class Accumulator:
    """
    在‘n’个变量上累加
    """
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def _getitem_(self,idx):
        return self.data[idx]

net = TinySSD(num_classes=1)
net = net.to('cuda')

trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

num_epochs = 200 #20
for epoch in range(num_epochs):
    print('epoch: ', epoch)
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = Accumulator(4)
    net.train()
    for features, target in train_iter:
        trainer.zero_grad()

        X, Y = features.to('cuda'), target.to('cuda')
        # X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                    bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                    bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric.data[0] / metric.data[1], metric.data[2] / metric.data[3]
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    # 保存模型参数
    if epoch % 10 == 0:
        torch.save(net.state_dict(), 'net_' + str(epoch) + '.pkl')