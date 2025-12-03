# -*- coding: utf-8 -*-
# author:你也叫小艾
# time: 2025/12/2 15:00
from utils import MDateset
train_data = MDateset.dateset('data/train','ants')
test_data = MDateset.dateset('data/val','ants').__getitem__(1)
img,lab = test_data
print(type(img))

