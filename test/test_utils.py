import unittest
from utils.loss import *
from utils.optim import *
from utils.sampler import *
from utils.dataloader import *
from model.zoo import *
from dataset.config import *


class TestSerialization(unittest.TestCase):
    def test_loss(self):
        names = BaseLoss.subclasses.keys()
        for name in names:
            cfg = LossConfig(name=name)
            obj = cfg.get()
            schema = LossSchema()
            s_cfg = schema.dump(cfg)
            print(s_cfg)
            new_cfg = schema.load(s_cfg)
            self.assertEqual(new_cfg, cfg)

    def test_optim(self):
        model_cfg = ModelConfig('MLG', input_size=32, hidden_size=32, output_size=32)
        model = model_cfg.get()
        names = BaseOptim.subclasses.keys()
        for name in names:
            cfg = OptimConfig(name=name, lr=0.01, beta=(0.5, 0.5))
            obj = cfg.get(model.parameters())
            schema = OptimSchema()
            s_cfg = schema.dump(cfg)
            print(s_cfg)
            new_cfg = schema.load(s_cfg)
            self.assertEqual(new_cfg, cfg)

    def test_zoo(self):
        names = BaseModel.subclasses.keys()
        for name in names:
            cfg = ModelConfig(name, input_size=32, hidden_size=32, output_size=32)
            obj = cfg.get()
            schema = ModelSchema()
            s_cfg = schema.dump(cfg)
            print(s_cfg)
            new_cfg = schema.load(s_cfg)
            self.assertEqual(new_cfg, cfg)

    def test_dataloader(self):
        names = BaseLoader.subclasses.keys()
        dataset_cfg = DatasetConfig(name='Spiral', mode=10, sig=0.01, num_per_mode=200, max_val=15)
        dataset = dataset_cfg.get()
        for name in names:
            cfg = LoaderConfig(name, 32, True)
            cfg.get(dataset)
            schema = LoaderSchema()
            s_cfg = schema.dump(cfg)
            print(s_cfg)
            new_cfg = schema.load(s_cfg)
            self.assertEqual(new_cfg, cfg)

    def test_sampler(self):
        names = BaseSampler.subclasses.keys()
        for name in names:
            cfg = SamplerConfig(
                name, out_shape=(32, 1, 1),
                data_num=300, epoch=10,
                alpha=1, batch_size=32,
                loss_cfg=LossConfig('MSE'),
                optim_cfg=OptimConfig('Adam', lr=1)
            )
            cfg.get()
            schema = SamplerSchema()
            s_cfg = schema.dump(cfg)
            print(s_cfg)
            new_cfg = schema.load(s_cfg)
            self.assertEqual(new_cfg, cfg)

    def test_dataset(self):
        names = BaseDataset.subclasses.keys()
        for name in names:
            cfg = DatasetConfig(name=name, mode=10, sig=0.01, num_per_mode=200, max_val=15)
            cfg.get()
            schema = DatasetSchema()
            s_cfg = schema.dump(cfg)
            print(s_cfg)
            new_cfg = schema.load(s_cfg)
            self.assertEqual(new_cfg, cfg)
