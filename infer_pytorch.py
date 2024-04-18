#! /usr/bin/env python
# coding:utf-8
"""
Author: zhiying
Date: 24-4-15 下午12:47
Description: CatsCls prediction

"""
import torchvision
from torch import nn
import torch
from Dataset import ToTensor
from util import transforms_test

from PIL import Image   # 图像读取
import os
# 进行预测和提交
# 首先拿到预测文件的路径列表

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

device=torch.device("cpu")
test_path = []
listdir('./imgs/cat_12_test', test_path)
#加载训练好的模型
net = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
in_features = net.fc.in_features
net.fc = nn.Linear(in_features, 12)
net.load_state_dict(torch.load('train_models/acc0.9040178656578064.pth'))
net.eval()

pre_classes = []
# 生成预测结果
for path in test_path:
    image_path = path

    image = Image.open(image_path)  # H, W, C
    image = ToTensor(image)

    if len(image.shape) == 2:
        image = torch.unsqueeze(image, axis=0)  # H, W, C
        # image = ToPILImage(image)
        image = image.repeat(3, 1, 1)

    # print(self.train_img[self.train_locs[idx]])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # H, W, C
    # image = Image.open(os.path.join("./imgs/", self.train_img[self.train_locs[idx]]))

    if image.shape[0] != 3:
        image = torch.sum(image, dim=0, keepdim=True)  # HxWx1
        image = image.repeat(3, 1, 1)
    # image = ToPILImage(image)
    image = transforms_test(image)

    x = torch.as_tensor(image, dtype=torch.float)
    x = torch.unsqueeze(x, axis=0)
    x = x.to(device)
    pre = net(x)
    pre = torch.argmax(pre)

    # print(pre.item())

    print("图片：", path, "预测结果：", pre.item())
    pre_classes.append(pre.item())

print(pre_classes)

# 导入csv模块
import csv

# 1、创建文件对象
with open('submit.csv', 'w', encoding='gbk', newline="") as f:
    # 2、基于文件对象构建csv写入对象
    csv_writer = csv.writer(f)
    for i in range(240):
        csv_writer.writerow([test_path[i].split('/')[-1], pre_classes[i]])
    print('写入数据完成')