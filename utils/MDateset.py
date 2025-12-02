# -*- coding: utf-8 -*-
# author:你也叫小艾
# time: 2025/12/2 18:35
import os
from torch.utils.data import Dataset

class dateset(Dataset):
    def __init__(self, root_dir,label):
        self.root_dir = root_dir
        self.label = label
        self.path = os.path.join(root_dir,self.label)
        self.all_imgs = os.listdir(self.path)
    def __getitem__(self, idx):
        img = os.path.join(self.path, self.all_imgs[idx])
        label = self.label
        return img,label
    def __len__(self):
        size = len(self.all_imgs)
        return size
