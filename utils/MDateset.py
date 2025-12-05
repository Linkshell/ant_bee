# -*- coding: utf-8 -*-
# author:你也叫小艾
# time: 2025/12/2 18:35
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch
m_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class dateset(Dataset):
    def __init__(self, root_dir,label):
        self.root_dir = root_dir
        self.label = label
        self.path = os.path.join(root_dir,self.label)
        self.all_imgs = os.listdir(self.path)
    def __getitem__(self, idx):
        img = os.path.join(self.path, self.all_imgs[idx])
        img = Image.open(img)
        img = m_transform(img)
        lable = torch.tensor(0) if self.label == 'ants' else torch.tensor(1)
        return img,lable
    def __len__(self):
        size = len(self.all_imgs)
        return size
