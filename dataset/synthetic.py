from utils.class_helper import BaseHelper
from dataset.config import DatasetConfig
from marshmallow import Schema, fields, post_load
import numpy as np
from torch.utils.data import Dataset
from dataset.config import DatasetConfig, BaseDataset


# class SyntheticConfig(DatasetConfig):
#     def __init__(self, name, **kwargs):
#         super(SyntheticConfig, self).__init__(name, **kwargs)
#
#     def get(self):
#         acc_args = BaseSynthetic.get_arguments(self.name)
#         args = {key: self.__dict__[key] for key in acc_args if key in self.__dict__}
#         args['name'] = self.name
#         return BaseSynthetic.create(**args)
#
#
# class BaseSynthetic(BaseHelper):
#     subclasses = {}
#
#
# class SyntheticSchema(Schema):
#     name = fields.Str()
#     mode = fields.Int()
#     sig = fields.Float()
#     num_per_mode = fields.Int()
#     max_val = fields.Int()
#     __model__ = SyntheticConfig
#
#     @post_load
#     def make_object(self, data):
#         return self.__model__(
#             name=data['name'],
#             mode=data['mode'],
#             sig=data['sig'],
#             num_per_mode=data['num_per_mode'],
#             max_val=data['max_val']
#         )


@BaseDataset.register('Spiral')
class Spiral(Dataset, BaseDataset):
    def __init__(self, mode, sig, num_per_mode, hard=True, return_idx=False):
        # x = cos(t)
        # y = sin(t)
        # r = t
        self.mode = mode
        self.num_per_mode = num_per_mode
        self.sig = sig
        self.length = num_per_mode * mode
        self.return_idx = return_idx
        if hard:
            self.modes = np.array([[np.cos(i / 3) * i * i, np.sin(i / 3) * i * i] for i in range(self.mode)])
        else:
            self.modes = np.array([[np.cos(i) * i, np.sin(i) * i] for i in range(self.mode)])
        data = None
        for loc in range(self.mode):
            if data is None:
                data = np.random.multivariate_normal((self.modes[loc, 0], self.modes[loc, 1]),
                                                     cov=[[self.sig, 0], [0, self.sig]], size=num_per_mode)
            else:
                data = np.concatenate(
                    [
                        data,
                        np.random.multivariate_normal((self.modes[loc, 0], self.modes[loc, 1]),
                                                      cov=[[self.sig, 0], [0, self.sig]], size=num_per_mode)
                    ]
                )
        self.data = data

    def __getitem__(self, item):
        if self.return_idx:
            onehot = np.zeros(self.length, dtype=np.int)
            onehot[item] = 1
            return onehot, self.data[item]
        else:
            return self.data[item]

    def __len__(self):
        return self.length


@BaseDataset.register('GMM')
class GMM(Dataset, BaseDataset):
    """Data set from AdaGAN GMM."""

    def __init__(self, mode=3, num_per_mode=64000, max_val=15, return_idx=False):
        # np.random.seed(851)
        self.return_idx = return_idx
        opts = {
            'gmm_modes_num': mode,  # 3
            'gmm_max_val': max_val,  # 15.
            'toy_dataset_dim': 2,
            'toy_dataset_size': num_per_mode * mode  # 64 * 1000
        }

        modes_num = opts["gmm_modes_num"]
        # np.random.seed(opts["random_seed"])
        max_val = opts['gmm_max_val']
        # mixture_means = np.random.uniform(
        #     low=-max_val, high=max_val,
        #     size=(modes_num, opts['toy_dataset_dim']))

        mixture_means = np.array(
            [[14.75207179, -9.5863695],
             [0.80064377, 3.41224097],
             [5.37076641, 11.76694952],
             [8.48660686, 11.73943841],
             [12.41315706, 2.82228677],
             [14.59626141, -2.52886563],
             [-7.8012091, 13.23184103],
             [-5.23725599, 6.27326752],
             [-6.87097889, 11.95825351],
             [10.79436725, -11.47316948]]
        )

        def variance_factor(num, dim):
            if num == 1: return 3 ** (2. / dim)
            if num == 2: return 3 ** (2. / dim)
            if num == 3: return 8 ** (2. / dim)
            if num == 4: return 20 ** (2. / dim)
            if num == 5: return 10 ** (2. / dim)
            return num ** 2.0 * 3

        mixture_variance = max_val / variance_factor(modes_num, opts['toy_dataset_dim'])

        # Now we sample points, for that we unseed
        # np.random.seed()
        num = opts['toy_dataset_size']
        X = np.zeros((num, opts['toy_dataset_dim'], 1, 1))
        for idx in range(num):
            comp_id = np.random.randint(modes_num)
            mean = mixture_means[comp_id]
            cov = mixture_variance * np.identity(opts["toy_dataset_dim"])
            X[idx, :, 0, 0] = np.random.multivariate_normal(mean, cov, 1)

        # self.data_shape = (opts['toy_dataset_dim'], 1, 1)
        # self.data = Data(opts, X)
        # self.num_points = len(X)
        self.data = X[:, :, 0, 0]
        self.modes = mixture_means

    def __getitem__(self, item):
        if self.return_idx:
            onehot = np.zeros(len(self.data), dtype=np.int)
            onehot[item] = 1
            return onehot, self.data[item]
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)


