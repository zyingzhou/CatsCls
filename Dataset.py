# /usr/bin/env python
# 自定义数据集
# 按比例随机切割数据集
import random
import os
import numpy
from torchvision import transforms
import torch

import cv2
# 深度学习包
import paddle
import paddle.vision.transforms as T       # 数据增强
from paddle.io import Dataset, DataLoader  # 定义数据集

import numpy as np
from PIL import Image              # 图像读取
from util import PILToTensor
from util import ToPILImage
from util import ToTensor

# 错误记录
def write_log(msg):
    with open('logs/log.txt', 'a') as f:
        f.write(msg+'\n')


train_ratio = 0.9  # 训练集占0.9，验证集占0.1

train_paths, train_labels = [], []
valid_paths, valid_labels = [], []
with open('imgs/train_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if random.uniform(0, 1) < train_ratio:
            train_paths.append(line.split('	')[0])
            label = line.split('	')[1]
            train_labels.append(int(line.split('	')[1]))
        else:
            valid_paths.append(line.split('	')[0])
            valid_labels.append(int(line.split('	')[1]))


# 定义训练数据集
class TrainData(Dataset):
    def __init__(self):
        super().__init__()
        self.color_jitter = T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        self.normalize = T.Normalize(mean=0, std=1)
        self.random_crop = T.RandomCrop(224, pad_if_needed=True)

    def __getitem__(self, index):
        # 读取图片
        image_path = 'imgs/'+ train_paths[index]

        image = np.array(Image.open(image_path))  # H, W, C
        try:
            image = image.transpose([2, 0, 1])[:3]  # C, H, W
        except:
            image = np.array([image, image, image])  # C, H, W

        # 图像增广
        features = self.color_jitter(image.transpose([1, 2, 0]))
        features = self.random_crop(features)
        features = self.normalize(features.transpose([2, 0, 1])).astype(np.float32)

        # 读取标签
        labels = train_labels[index]

        return features, labels

    def __len__(self):
        return len(train_paths)


# 定义验证数据集
class ValidData(Dataset):
    def __init__(self):
        super().__init__()
        self.normalize = T.Normalize(mean=0, std=1)

    def __getitem__(self, index):
        # 读取图片
        image_path = 'imgs/'+ valid_paths[index]

        image = np.array(Image.open(image_path))  # H, W, C
        try:
            image = image.transpose([2, 0, 1])[:3]  # C, H, W
        except:
            image = np.array([image, image, image])  # C, H, W

        # 图像变换
        features = cv2.resize(image.transpose([1, 2, 0]), (256, 256)).transpose([2, 0, 1]).astype(np.float32)
        features = self.normalize(features)

        # 读取标签
        labels = valid_labels[index]

        return features, labels

    def __len__(self):
        return len(valid_paths)

class Cats_Dataset(Dataset):
    '''
    树叶数据集的训练集 自定义Dataset
    '''

    def __init__(self, train, transform=None, test=False):
        '''
        train_path : 传入记录图像路径及其标号的csv文件
        transform : 对图像进行的变换
        '''
        super().__init__()
        # 标签位置
        self.train_txt = './imgs/train_list.txt'
        self.train_locs = train
        # 训练图片位置
        self.train_img = []
        # 训练标签
        self.train_label = []
        with open(self.train_txt, 'r') as f:
            for line in f.readlines():
                self.train_img.append(line.split('\t')[0])
                label = line.split('\t')[-1]
                self.train_label.append(int(label.strip()))
        self.test = test
        self.transform = transform

    def __getitem__(self, idx):
        '''
        idx : 所需要获取的图像的索引
        return : image， label
        '''
        # image = read_image(os.path.join("./imgs/", self.train_img[self.train_locs[idx]]))
        # print(self.train_img[self.train_locs[idx]])
        # # image = cv2.imread(os.path.join("./imgs/", self.train_img[self.train_locs[idx]])) # H W C
        # print(image.shape)
        img_path = os.path.join("./imgs/", self.train_img[self.train_locs[idx]]);
        # image = np.array(Image.open(img_path))  # H, W, C
        image = Image.open(img_path)  # H, W, C
        image = ToTensor(image)

        if len(image.shape) == 2:
            image = torch.unsqueeze(image, axis=0) # H, W, C
            # image = ToPILImage(image)
            image = image.repeat(3, 1, 1)

        # print(self.train_img[self.train_locs[idx]])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # H, W, C
        # image = Image.open(os.path.join("./imgs/", self.train_img[self.train_locs[idx]]))
        if (self.transform != None):
            if image.shape[0] != 3:
                image = torch.sum(image, dim=0, keepdim=True) # HxWx1
                image = image.repeat(3, 1, 1)
            # image = ToPILImage(image)
            image = self.transform(image)
        # print(img_path)
        # print(image['image'].shape)
        if not self.test:
            label = self.train_label[self.train_locs[idx]]
            #return PILToTensor(image), label
            return image, label

    def __len__(self):
        return len(self.train_locs)

class Cats_DatasetV2(Dataset):
        '''
        树叶数据集的训练集 自定义Dataset
        '''

        def __init__(self, train, transform=None, test=False):
            '''
            train_path : 传入记录图像路径及其标号的csv文件
            transform : 对图像进行的变换
            '''
            super().__init__()
            self.color_jitter = T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
            self.normalize = T.Normalize(mean=0, std=1)
            self.random_crop = T.RandomCrop(224, pad_if_needed=True)
            # 标签位置
            self.train_txt = './imgs/train_list.txt'
            self.train_locs = train
            # 训练图片位置
            self.train_img = []
            # 训练标签
            self.train_label = []
            with open(self.train_txt, 'r') as f:
                for line in f.readlines():
                    self.train_img.append(line.split('\t')[0])
                    label = line.split('\t')[-1]
                    self.train_label.append(int(label.strip()))
            self.test = test
            self.transform = transform

        def __getitem__(self, idx):
            '''
            idx : 所需要获取的图像的索引
            return : image， label
            '''
            # image = read_image(os.path.join("./imgs/", self.train_img[self.train_locs[idx]]))
            # print(self.train_img[self.train_locs[idx]])
            # # image = cv2.imread(os.path.join("./imgs/", self.train_img[self.train_locs[idx]])) # H W C
            # print(image.shape)
            img_path = os.path.join("./imgs/", self.train_img[self.train_locs[idx]]);
            image = np.array(Image.open(img_path))  # H, W, C
            try:
                image = image.transpose([2, 0, 1])[:3]  # C, H, W
            except:
                image = np.array([image, image, image])  # C, H, W

            # 图像增广
            features = self.color_jitter(image.transpose([1, 2, 0]))
            features = self.random_crop(features)
            features = self.normalize(features.transpose([2, 0, 1])).astype(np.float32)

            # print(img_path)
            # print(image['image'].shape)
            if not self.test:
                label = self.train_label[self.train_locs[idx]]
                return features, label

        def __len__(self):
            return len(self.train_locs)


