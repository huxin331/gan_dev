import matplotlib
matplotlib.use('agg')

from utils.sampler import SamplerConfig
from utils.loss import LossConfig
from utils.optim import OptimConfig
from utils.dataloader import LoaderConfig
from model.gan import GanConfig, BaseGan
from model.zoo import ModelConfig
from dataset.config import DatasetConfig
from matplotlib import pyplot as plt
from utils.util_model import UtilityModelConfig, UtilityModelSchema
import os
from torchvision.utils import make_grid, save_image
import numpy as np
import torch as th
from torch import nn
from tensorboardX import SummaryWriter

logger = SummaryWriter('./log/p2p')
# logger = Logger(cat='p2p')


@BaseGan.register('Pix2Pix')
class Pix2Pix(BaseGan):
    def __init__(self, gen_cfg, dis_cfg, gen_step,
                 dis_step, gan_epoch, loader_cfg,
                 dataset_cfg, gloss_cfg, dloss_cfg,
                 goptim_cfg, doptim_cfg,
                 device):
        self.device = device
        self.gen = gen_cfg.get().to(self.device)
        self.dis = dis_cfg.get().to(self.device)

        self.gen_loss = gloss_cfg.get()
        self.dis_loss = dloss_cfg.get()
        self.goptim_cfg = goptim_cfg
        self.doptim_cfg = doptim_cfg

        self.gen_step = gen_step
        self.dis_step = dis_step
        self.gan_epoch = gan_epoch
        self.dataset = dataset_cfg.get()
        self.dataloader = loader_cfg.get(self.dataset)
        self.batch_size = loader_cfg.batch_size

        self.l1_loss = nn.L1Loss()
        self.gen_optim = self.goptim_cfg.get(self.gen.parameters())
        self.dis_optim = self.doptim_cfg.get(self.dis.parameters())

        self.lambda_L1 = 100

    def iterate(self, gen_step=None, dis_step=None, **kwargs):
        self.train_mode()
        gen_step = gen_step if gen_step is not None else self.gen_step
        dis_step = dis_step if dis_step is not None else self.dis_step
        data = next(iter(self.dataloader))
        self.real_a, self.real_b = data['center'], data['1']

        self.real_a = self.real_a.permute((0, 3, 1, 2)).to(self.device).float()
        self.real_b = self.real_b.permute((0, 3, 1, 2)).to(self.device).float()
        for f in self._pre_iterate_hooks_:
            f(self, **kwargs)

        for _ in range(dis_step):
            self.dis_iter()

        for f in self._mid_iterate_hooks_:
            f(self, **kwargs)

        for _ in range(gen_step):
            self.gen_iter()

        for f in self._post_iterate_hooks_:
            f(self, **kwargs)

    def gen_iter(self):
        self.gen.zero_grad()

        self.fake_b = self.gen(self.real_a)
        fake_ab = th.cat((self.real_a, self.fake_b), 1)
        pred_fake = self.dis(fake_ab)
        g_fake_labels = th.ones(pred_fake.shape, dtype=th.float32, device=self.device)
        g_fake_loss = self.gen_loss(pred_fake, g_fake_labels)
        self.g_loss = g_fake_loss + self.lambda_L1 * self.l1_loss(self.fake_b, self.real_b)
        self.g_loss.backward()
        self.gen_optim.step()

    def dis_iter(self):
        self.dis.zero_grad()
        self.fake_b = self.gen(self.real_a)
        fake_ab = th.cat((self.real_a, self.fake_b), 1)
        pred_fake = self.dis(fake_ab)

        real_ab = th.cat((self.real_a, self.real_b), 1)
        pred_real = self.dis(real_ab)

        d_fake_labels = th.zeros(pred_fake.shape, dtype=th.float32, device=self.device)
        d_real_labels = th.ones(pred_fake.shape, dtype=th.float32, device=self.device)

        d_real_loss = self.dis_loss(pred_real, d_real_labels)
        d_fake_loss = self.dis_loss(pred_fake, d_fake_labels)
        self.d_loss = d_real_loss + d_fake_loss
        self.d_loss.backward()
        self.dis_optim.step()

    @BaseGan.add_post_iterate_hook
    def post_iterate(self, iter_num):
        if iter_num % 50 == 0:
            logger.add_scalars('scalar/loss', {
                'g_loss': self.g_loss,
                'd_loss': self.d_loss
            }, iter_num)

            ra = make_grid(self.real_a, normalize=True, nrow=4)
            rb = make_grid(self.real_b, normalize=True, nrow=4)
            fb = make_grid(self.fake_b, normalize=True, nrow=4)

            logger.add_image('image/real_a', ra, iter_num)
            logger.add_image('image/real_b', rb, iter_num)
            logger.add_image('image/fake_b', fb, iter_num)


if __name__ == '__main__':
    device = 'cuda'
    GAN_EPOCHS = 10000
    IMG_SIZE = 256
    BATCH_SIZE = 2
    # gloss_cfg = LossConfig('BCEWithLogits')
    # dloss_cfg = LossConfig('BCEWithLogits')
    gloss_cfg = LossConfig('MSE')
    dloss_cfg = LossConfig('MSE')

    goptim_cfg = OptimConfig('Adam', lr=1e-3)
    doptim_cfg = OptimConfig('Adam', lr=1e-3)

    dataset_cfg = DatasetConfig('Flower', suffix=('_center', '_1'), size=IMG_SIZE)
    loader_cfg = LoaderConfig('naive', batch_size=BATCH_SIZE, shuffle=True)

    # gen_cfg = ModelConfig('ResNetGen', input_nc=3, output_nc=3, ngf=64, n_blocks=6)
    gen_cfg = ModelConfig('UNetGen', input_nc=3, output_nc=3, num_downs=8, ngf=64)
    dis_cfg = ModelConfig('PatchDis', input_nc=6)

    gan_cfg = GanConfig(
        name='Pix2Pix', gen_cfg=gen_cfg, dis_cfg=dis_cfg,
        gen_step=1, dis_step=1, gan_epoch=GAN_EPOCHS, loader_cfg=loader_cfg,
        dataset_cfg=dataset_cfg, gloss_cfg=gloss_cfg, dloss_cfg=dloss_cfg,
        goptim_cfg=goptim_cfg, doptim_cfg=doptim_cfg, device=device
    )
    gan = gan_cfg.get()
    gan.train(use_tqdm=True)
    gan.save('p2p', 'unet')
