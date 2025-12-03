# -*- coding: utf-8 -*-
# author:你也叫小艾
# time: 2025/12/2 15:00
import torch.optim
from torch.utils.data import DataLoader
from torch import nn
from network.VGG16 import VGG16
from utils import MDateset

train_ants_data = MDateset.dateset('data/train','ants')
train_bees_data = MDateset.dateset('data/train','bees')
train_data = train_ants_data + train_bees_data

test_ants_data = MDateset.dateset('data/val','ants')
test_bees_data = MDateset.dateset('data/val','bees')
test_data = test_ants_data + test_bees_data

train_dataloader = DataLoader(dataset=train_data,batch_size=8,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=8,shuffle=True)


vgg = VGG16()
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.SGD(vgg.parameters(),lr=learning_rate)

for i in range(10):
    print("++++++++第{}轮训练开始++++++++++".format(i))
    for idx, data in enumerate(train_dataloader):
        img, lable = data
        output = vgg(img)
        loss = loss_fn(output, lable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)







