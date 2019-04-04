from dataset.config import DatasetConfig, BaseDataset
from torch.utils.data import Dataset
from skimage import transform, io
import numpy as np
import glob


@BaseDataset.register('CelebA')
class CelebA(Dataset, BaseDataset):
    def __init__(self, dataset_len, image_size, preload_len):
        # get full attrs of this dataset
        data = []
        self.image_size = image_size
        self.imgs_paths = glob.glob('/home/bourgan/BourGAN_Refined/datasets/resized_celebA_64/*.jpg')[:dataset_len]
        self.preload_len = preload_len
        preload_paths = self.imgs_paths[:preload_len]
        for path in preload_paths:
            img = io.imread(path)
            img = transform.resize(img, (image_size, image_size))
            data.append(img[:, :, :3])
        self.data = np.array(data)
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        self.data = (self.data - 0.5) * 2
        print('CelebA data: ', self.data.shape)

    def __getitem__(self, item):
        if item < self.preload_len:
            return self.data[item]
        path = self.imgs_paths[item]
        img = io.imread(path)[:, :, :3]
        img = transform.resize(img, (self.image_size, self.image_size))
        img = np.transpose(img, (2, 0, 1))
        img = (img - 0.5) * 2
        return img

    def __len__(self):
        return len(self.imgs_paths)
