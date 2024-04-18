#! /usr/bin/env python
# coding:utf-8
"""
Author: zhiying
Date: 24-2-28 下午9:11
Description: CatsCls prediction

"""
import paddle
import cv2
import paddle.vision.transforms as T       # 数据增强
from paddle.io import Dataset, DataLoader  # 定义数据集

import numpy as np
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


test_path = []
listdir('./imgs/cat_12_test', test_path)
#加载训练好的模型
pre_model = paddle.vision.models.resnet18(pretrained=True, num_classes=12)
pre_model.set_state_dict(paddle.load('train_models/acc0.8598130841121495_epoch17.model'))
pre_model.eval()

pre_classes = []
normalize = T.Normalize(mean=0, std=1)
# 生成预测结果
for path in test_path:
    image_path = path

    image = np.array(Image.open(image_path))  # H, W, C
    try:
        image = image.transpose([2, 0, 1])[:3]  # C, H, W
    except:
        image = np.array([image, image, image])  # C, H, W

    # 图像变换
    features = cv2.resize(image.transpose([1, 2, 0]), (256, 256)).transpose([2, 0, 1]).astype(np.float32)
    features = normalize(features)

    features = paddle.to_tensor([features])
    pre = list(np.array(pre_model(features)[0]))
    # print(pre)
    max_item = max(pre)
    pre = pre.index(max_item)
    print("图片：", path, "预测结果：", pre)
    pre_classes.append(pre)

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