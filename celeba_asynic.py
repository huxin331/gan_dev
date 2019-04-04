import matplotlib
matplotlib.use('agg')

from utils.logger import Logger
from utils.sampler import SamplerConfig
from utils.loss import LossConfig
from utils.optim import OptimConfig
from utils.dataloader import LoaderConfig
from model.gan import GanConfig, GanSchema, BaseGan
from model.zoo import ModelConfig
from dataset.config import DatasetConfig
from matplotlib import pyplot as plt
from utils.util_model import UtilityModelConfig, UtilityModelSchema
import os
from marshmallow import Schema, fields, post_load
from torchvision.utils import make_grid, save_image
import numpy as np
import torch as th
from sklearn.manifold import TSNE
from utils.func import js_div_uniform
from model.zoo import SampleMatrix


logger = Logger(cat='celeba64')


@BaseGan.register('CelebAGAN')
class CelebAGAN(BaseGan):
    def __init__(self, gen_cfg, dis_cfg, gen_step,
                 dis_step, gan_epoch, loader_cfg,
                 dataset_cfg, gloss_cfg, dloss_cfg,
                 goptim_cfg, doptim_cfg,
                 sampler_cfg, label_smooth, dist_loss,
                 device):
        super(CelebAGAN, self).__init__(
            gen_cfg, dis_cfg, gen_step,
            dis_step, gan_epoch, loader_cfg,
            dataset_cfg, gloss_cfg, dloss_cfg,
            goptim_cfg, doptim_cfg,
            sampler_cfg, label_smooth, dist_loss,
            device
        )

    @BaseGan.add_pre_train_hook
    def pre_train(self):
        embedding_optim_cfg = OptimConfig('Adam', lr=1e-3)
        self.embedding_optim = embedding_optim_cfg.get(self.gen.embedding.parameters())
        embedding_loss_cfg = LossConfig('MSE')
        self.embedding_loss = embedding_loss_cfg.get()
        self.gen_optim = self.goptim_cfg.get(filter(lambda x: type(x) != SampleMatrix, self.gen.parameters()))

    @BaseGan.add_post_iterate_hook
    def post_iterate(self, iter_num):
        if (iter_num + 1) % 1000 == 0:
            self.vis_gen(postfix=iter_num)

        if iter_num < 10000:
            return

        if iter_num == 10000:
            for _ in range(1000):
                self.embedding_iter()
            return

        for _ in range(1):
            self.embedding_iter()

    def embedding_iter(self):
        self.gen.zero_grad()
        latent_samples, target = self.sampler.sampling_with_dataset(128, self.dataset.data)
        g_gen_input = latent_samples.to(dtype=th.float32, device=self.device)
        g_real_data = target.to(dtype=th.float32, device=self.device)
        g_fake_data = self.gen(g_gen_input)
        g_loss = self.embedding_loss(g_fake_data, g_real_data)

        g_loss.backward()
        self.embedding_optim.step()

    @BaseGan.add_post_train_hook
    def post_train(self):
        self.save(logger.exp_dir, 'final')

    def vis_gen(self, postfix):
        self.eval_mode()
        x_generated = self.sampling(128, 1)
        img = make_grid(th.Tensor(x_generated), nrow=16)
        save_image(img, os.path.join(logger.img_dir, 'gen_{0:05d}.jpg'.format(postfix)))


if __name__ == '__main__':
    MODE = 10
    LATENT_DIM = 100
    IMAGE_SIZE = 64
    DATASET_LEN = 20000
    PRELOAD_LEN = 20000
    EPOCHS = 50000
    device = 'cuda'
    gloss_cfg = LossConfig('BCE')
    dloss_cfg = LossConfig('BCE')

    sampler_cfg = SamplerConfig(
        name='Onehot',
        out_shape=PRELOAD_LEN,
        latent_dim=LATENT_DIM,
        random_sampling=False
    )
    goptim_cfg = OptimConfig('Adam', lr=0.0002)
    doptim_cfg = OptimConfig('Adam', lr=0.0002)

    dataset_cfg = DatasetConfig('CelebA', dataset_len=DATASET_LEN, image_size=IMAGE_SIZE, preload_len=PRELOAD_LEN)
    loader_cfg = LoaderConfig('naive', batch_size=128, shuffle=True)

    gen_cfg = ModelConfig(
        'EDCG', input_size=LATENT_DIM,
        hidden_size=128, output_size=3,
        data_num=PRELOAD_LEN, out_64=IMAGE_SIZE == 64
    )
    dis_cfg = ModelConfig('DCD', input_size=3, hidden_size=64, output_size=1, out_64=IMAGE_SIZE == 64)

    gan_cfg = GanConfig(
        name='CelebAGAN', gen_cfg=gen_cfg, dis_cfg=dis_cfg,
        gen_step=1, dis_step=1, gan_epoch=EPOCHS, loader_cfg=loader_cfg,
        dataset_cfg=dataset_cfg, gloss_cfg=gloss_cfg, dloss_cfg=dloss_cfg,
        goptim_cfg=goptim_cfg, doptim_cfg=doptim_cfg, label_smooth=False,
        sampler_cfg=sampler_cfg, dist_loss=False, device=device
    )
    gan_schema = GanSchema()
    gan_desc = gan_schema.dump(gan_cfg)
    logger.save_cfg(gan_desc)

    gan = gan_cfg.get()
    gan.train(use_tqdm=True)
    gan.save('checkpoints', 'CelebA64')
