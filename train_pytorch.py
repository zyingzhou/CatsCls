"""
Author: zhiying
Date: 24-3-31 下午20:23
Description: CatsCls training
"""
# https://www.kaggle.com/code/wjfearth/8th-classify-leaves-with-tpu-5hrs-0-989
# 参考https://www.kaggle.com/code/zachary666/classify-leaf
import albumentations
import copy
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from Dataset import Cats_Dataset
from util import transforms_train


def train_model(train_loader, valid_loader, device=torch.device("cpu")):
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, 12)
    net = net.to(device)
    epoch = 30
    best_epoch = 0
    best_score = 0.0
    best_model_state = None
    early_stopping_round = 3
    losses = []
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
    loss = nn.CrossEntropyLoss(reduction='mean')
#     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = 1e-6)
    scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)
    for i in range(epoch):
        acc = 0
        loss_sum = 0
        net.train()
        for x, y in tqdm(train_loader):
            # x = x['image']
            x = torch.as_tensor(x, dtype=torch.float)
            x = x.to(device)
            y = torch.as_tensor(y)
            y = y.to(device)
            y_hat = net(x)
            loss_temp = loss(y_hat, y)
            loss_sum += loss_temp
            optimizer.zero_grad()
            loss_temp.backward()
            optimizer.step()
#             scheduler.step()
            acc += torch.sum(y_hat.argmax(dim=1).type(y.dtype) == y)
        scheduler.step()
        losses.append(loss_sum.cpu().detach().numpy() / len(train_loader))
        print("epoch: ", i, "loss=", loss_sum.item(), "训练集准确度=", (acc/(len(train_loader)*train_loader.batch_size)).item(), end="")

        test_acc = 0
        net.eval()
        for x, y in tqdm(valid_loader):
            x = x.to(device)
            x = torch.as_tensor(x, dtype=torch.float)
            y = y.to(device)
            y_hat = net(x)
            test_acc += torch.sum(y_hat.argmax(dim=1).type(y.dtype) == y)
        print("验证集准确度", (test_acc / (len(valid_loader) * valid_loader.batch_size)).item())
        if test_acc > best_score:
            best_model_state = copy.deepcopy(net.state_dict())
            best_score = test_acc
            best_epoch = i
            print('best epoch save!')
            torch.save(net.state_dict(), f'./train_models/{test_acc}.pth')
        if i - best_epoch >= early_stopping_round:
            break
    # net.load_state_dict(best_model_state)

label_path = './imgs/train_list.txt'
train = []
labels = []
with open(label_path, 'r') as f:
    for line in f.readlines():
        img_path = line.split('\t')[0]
        train.append(img_path)
        label = line.split('\t')[-1]
        labels.append(int(label.strip()))


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
prediction_df = pd.DataFrame()
for fold_n, (trn_idx, val_idx) in enumerate(skf.split(train, labels)):
    print(f'fold {fold_n} training...')
    trainset = Cats_Dataset(trn_idx, transforms_train)
    evalset = Cats_Dataset(val_idx, transforms_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, drop_last=False)
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=64, shuffle=False, drop_last=False)
    train_model(train_loader, eval_loader)
    # prediction_df[f'fold_{fold_n}'] = predictions

