from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import json


# TODO
# 1. implement gender
# 2. how to do data augmentation


class ReadDataSource(Dataset):
    def __init__(self, x_dir, y_dir):
        self.x_dir = x_dir  # folder '/joints_val_img', '/joints_test_img' or '/joints_train_img'

        with open(y_dir, 'rb') as f:
            self.y_dict = json.loads(f.read())

        self.x_list = self.y_dict['annotations']

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, item):
        x_ID = self.x_list[item]['image_id']  # hand number
        y_area = self.x_list[item]['area']
        bbox = self.x_list[item]['bbox']
        y_category = self.x_list[item]['category_id']

        x_path = self.x_dir + x_ID + '.png'

        x = torch.zeros((3, 1080, 1920))  # channel * height * width
        img = Image.open(x_path)
        img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])(img)
        x[:, :, :] = img

        y = self.y_list[item]  # bone age
        return x, y


if __name__ == "__main__":

    x_dir = '/data1/jiapan/Dataset/BA/joints_val_img'
    y_file = '/data1/jiapan/Dataset/BA/val_set.csv'
    tr = ReadDataSource(x_dir, y_file)

    trld = DataLoader(dataset=tr, batch_size=4, shuffle=False)
    for idx, (x, y) in enumerate(trld):
        print('batch_', idx)
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)
