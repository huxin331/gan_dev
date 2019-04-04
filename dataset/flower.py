from dataset.config import DatasetConfig, BaseDataset
from torch.utils.data import Dataset
from skimage import transform, io
import numpy as np
import glob
import os


@BaseDataset.register('Flower')
class FlowerPhoto(Dataset, BaseDataset):
    def __init__(self, suffix=('_center', '_1', '_-1'), train=True, size=128):
        img_path = '/home/pc2842_columbia_edu/data-disk/flowers_focused/*_center.png'
        path = '/home/pc2842_columbia_edu/data-disk/flowers_focused/'
        img_list = glob.glob(img_path)
        keys = [p.split('/')[-1][:-11] for p in img_list]
        self.suffix = suffix
        self.img_list = [
            {s[1:]: os.path.join(path, k + s + '.png') for s in suffix} for k in keys
        ]
        if train:
            self.img_list = self.img_list[:int(len(self.img_list) * 0.9)]
        else:
            self.img_list = self.img_list[int(len(self.img_list) * 0.9):]
        self.size = size

    def __getitem__(self, index):
        data = {key: transform.resize(io.imread(val), (self.size, self.size)) for key, val in self.img_list[index].items()}
        for key in data:
            data[key] -= 0.5
            data[key] *= 2
        return data

    def __len__(self):
        return len(self.img_list)
