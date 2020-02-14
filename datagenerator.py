from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import json


# TODO
# 1. implement gender
# 2. how to do data augmentation


class ReadDataSource(Dataset):
    def __init__(self, x_dir, y_dir):
        self.x_dir = x_dir  # .json dir

        with open(y_dir, 'rb') as f:
            self.y_dict = json.loads(f.read())

        self.x_list = self.y_dict['annotations']

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, item):
        x_ID = str(self.x_list[item]['image_id'])
        y_area = self.x_list[item]['area']
        y_bbox = self.x_list[item]['bbox']
        y_category = self.x_list[item]['category_id']

        x_path = self.x_dir + x_ID + '.png'

        x = torch.zeros((3, 1080, 1920))  # channel * height * width
        img = Image.open(x_path)
        img = transforms.Compose([
            # transforms.Resize((3, 256, 256)),
            transforms.ToTensor()
        ])(img)
        x[:, :, :] = img

        y = y_category  #
        # todo: shape of y is not yet defined
        return x, y


if __name__ == "__main__":

    x_dir = '../seperate_waste_piece_train/images_withoutrect/'
    y_file = './train.json'
    tr = ReadDataSource(x_dir, y_file)

    trld = DataLoader(dataset=tr, batch_size=4, shuffle=True, drop_last=True)
    for idx, (x, y) in enumerate(trld):
        print('batch_', idx)
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)
