import unittest
from utils.loss import *
from utils.optim import *
from utils.sampler import *
from utils.dataloader import *
from model.zoo import *
from dataset.config import *
from utils.util_model import *


class TestUtilityModel(unittest.TestCase):
    def test_naive_classifier(self):
        train_data_cfg = DatasetConfig('MNIST', stack=False, train=True, along_width=False)
        test_data_cfg = DatasetConfig('MNIST', stack=False, train=False, along_width=False)

        train_loader = LoaderConfig('naive', batch_size=128, shuffle=True)
        test_loader = LoaderConfig('naive', batch_size=128, shuffle=True)

        cfg = UtilityModelConfig('NaiveClassifier', True, '/home/bourgan/gan_dev/checkpoints/mnist_naive.pth.tar',
                                 15, train_loader, test_loader, train_data_cfg, test_data_cfg, 32, 'cuda')
        model = cfg.get()
