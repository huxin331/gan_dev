from dataset.config import DatasetConfig, BaseDataset
from torch.utils.data import Dataset
from skimage import transform
import numpy as np
import numpy.random as nr


@BaseDataset.register('MNIST')
class MNIST(Dataset, BaseDataset):
    def __init__(self, train=True, stack=False, along_width=False, digits=None, size=32):
        if digits is not None:
            self.digits = digits
        self.stack = stack
        self.along_width = along_width
        self.size = size
        if train:
            self.reader("/home/bourgan/gan_dev/data/mnist/raw/train-images-idx3-ubyte",
                        "/home/bourgan/gan_dev/data/mnist/raw/train-labels-idx1-ubyte", 60000)
        else:
            self.reader("/home/bourgan/gan_dev/data/mnist/raw/t10k-images-idx3-ubyte",
                        "/home/bourgan/gan_dev/data/mnist/raw/t10k-labels-idx1-ubyte", 10000)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

    def reader(self, imagename, labelname, n):
        f = open(imagename, 'rb')
        l = open(labelname, 'rb')
        f.read(16)
        l.read(8)
        images = []
        labels = []
        for _ in range(n):
            labels.append(ord(l.read(1)))
            image = []
            for j in range(28 * 28):
                image.append(ord(f.read(1)))
            image = np.reshape(np.array(image), (28, 28)).astype('uint8')
            image = transform.resize(image, (self.size, self.size))
            images.append(np.expand_dims(image, axis=0))

        label2 = {}

        for i in range(10):
            # if self.digits is not None and i not in self.digits:
            #     continue
            idxs = np.where(np.array(labels) == i)[0]
            nr.shuffle(idxs)
            label2[i] = idxs

        f.close()
        l.close()

        self.data = []
        self.label = []

        if self.stack or self.along_width:
            if self.stack:
                size_each = 20
                for i in range(10):
                    for j in range(10):
                        for k in range(10):
                            idxs1 = nr.choice(label2[i], size=size_each)
                            idxs2 = nr.choice(label2[j], size=size_each)
                            idxs3 = nr.choice(label2[k], size=size_each)
                            for (idx1, idx2, idx3) in zip(idxs1, idxs2, idxs3):
                                img1, img2, img3 = images[idx1], images[idx2], images[idx3]
                                newimg = np.zeros((self.size, self.size, 3))
                                newimg[:, :, 0] = img1
                                newimg[:, :, 1] = img2
                                newimg[:, :, 2] = img3
                                self.data.append(newimg)
                                self.label.append(i * 100 + j * 10 + k)
            else:
                size_each = 20
                for i in range(10):
                    for j in range(10):
                        for k in range(10):
                            idxs1 = nr.choice(label2[i], size=size_each)
                            idxs2 = nr.choice(label2[j], size=size_each)
                            idxs3 = nr.choice(label2[k], size=size_each)
                            for (idx1, idx2, idx3) in zip(idxs1, idxs2, idxs3):
                                img1, img2, img3 = images[idx1], images[idx2], images[idx3]
                                newimg = np.zeros((self.size, self.size * 3, 1))
                                newimg[:, 0: self.size, 0] = img1
                                newimg[:, self.size: 2 * self.size, 0] = img2
                                newimg[:, 2 * self.size:, 0] = img3
                                self.data.append(newimg)
                                self.label.append(i * 100 + j * 10 + k)

        else:  # Plain MNIST
            size_each = 2000
            for i in range(10):
                idxs1 = nr.choice(label2[i], size=size_each)
                for idx1 in idxs1:
                    img1 = images[idx1]
                    newimg = np.zeros((self.size, self.size, 1))
                    newimg[:, :, 0] = img1
                    self.data.append(newimg)
                    self.label.append(i)

        self.data = np.array(self.data)
        self.label = np.array(self.label)
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        self.data = (self.data - 0.5) * 2
        print("MNIST dataset genereated, data.shape={}, label.shape={}".format(
            self.data.shape, self.label.shape
        ))