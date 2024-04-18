# 模型训练
import numpy as np
from paddle import nn
from paddle.io import DataLoader  # 定义数据集

from Dataset import TrainData, ValidData

import os
import random

import numpy as np
# 加载飞桨相关库
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F

# 从 visualdl 库中引入 LogWriter 类
from visualdl import LogWriter
# 创建 LogWriter 对象，指定 logdir 参数，如果指定路径不存在将会创建一个文件夹
logwriter = LogWriter(logdir='./logs/catscls_experiment')

# 调用resnet50模型
paddle.vision.set_image_backend('cv2')
model = paddle.vision.models.resnet18(pretrained=True, num_classes=12)

train_data = TrainData()
valid_data = ValidData()

batch_size = 64
# 定义数据迭代器
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)

# 定义优化器
opt = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters(), weight_decay=paddle.regularizer.L2Decay(1e-4))

# 定义损失函数
loss_fn = paddle.nn.CrossEntropyLoss()

# 设置gpu环境
# paddle.set_device('gpu:0')

# 整体训练流程
for epoch_id in range(30):
    model.train()
    for batch_id, data in enumerate(train_dataloader()):
        # 读取数据
        features, labels = data
        features = paddle.to_tensor(features)
        labels = paddle.to_tensor(labels)

        # 前向传播
        predicts = model(features)

        # 损失计算
        loss = loss_fn(predicts, labels)

        avg_loss = paddle.mean(loss)

        # 记录当前训练 Loss 到 VisualDL
        logwriter.add_scalar("train_avg_loss", value=avg_loss.numpy(),
                             step=batch_id + epoch_id * (batch_size))

        # 记录网络中最后一个 fc 层的参数到 VisualDL
        logwriter.add_histogram("fc_weight", values=model.fc.weight.numpy(),
                                step=batch_id + epoch_id * (batch_size))
        # 反向传播
        avg_loss.backward()

        # 更新
        opt.step()

        # 清零梯度
        opt.clear_grad()

        # 打印损失
        if batch_id % 2 == 0:
            print('epoch_id:{}, batch_id:{}, loss:{}'.format(epoch_id, batch_id, avg_loss.numpy()))
    model.eval()
    print('开始评估')
    i = 0
    acc = 0
    for image, label in valid_data:
        image = paddle.to_tensor([image])

        pre = list(np.array(model(image)[0]))
        max_item = max(pre)
        pre = pre.index(max_item)

        i += 1
        if pre == label:
            acc += 1
        if i % 10 == 0:
            print('精度：', acc / i)

    paddle.save(model.state_dict(), 'train_models/acc{}_epoch{}.model'.format(acc / i, epoch_id))