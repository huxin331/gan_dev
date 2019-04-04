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
import os
import tqdm
import numpy as np


logger = Logger(cat='learning_synthetic')


def radius_coverage(fake_points, modes, sigma, eps=0.1):
    result = np.zeros(modes.shape[0] + 1)
    for pt in fake_points:
        dist = np.linalg.norm(modes - pt, axis=1)
        mode_num = np.argmin(dist)
        if 3 * sigma > dist[mode_num]:
            result[mode_num] += 1
        else:
            result[-1] += 1
    c = 0
    # dataset_len = len(fake_points)
    dataset_len = np.sum(result[:-1])
    if dataset_len == 0:
        return 0, result[-1]/len(fake_points), result
    for key, val in enumerate(result):
        if key == len(result) - 1:
            continue
        if val / dataset_len > 1 / len(modes) * eps:
            c += 1
    return c, result[-1]/len(fake_points), result


@BaseGan.register('LearningPrior')
class LearningPriorGan(BaseGan):
    @BaseGan.add_pre_train_hook
    def pre_train(self):
        logger.create_scalar('coverage')
        logger.create_scalar('unwanted_ratio')

    @BaseGan.add_post_iterate_hook
    def post_iterate(self, iter_num):
        if iter_num > 1000 and iter_num % 50 == 0:
            self.sampler.update(self)
        if (iter_num + 1) % 500 == 0:
            self.vis_gen(postfix=iter_num)
            self.vis_sampler(postfix=iter_num)
            self.coverage()

    def coverage(self):
        self.eval_mode()
        x_generated = self.sampling(100, 100)
        modes = self.dataset.modes
        coverage, unwanted_ratio, _ = radius_coverage(x_generated, modes, sigma=0.2, eps=0.01)
        logger.append_scalar('coverage', coverage)
        logger.append_scalar('unwanted_ratio', unwanted_ratio)

    @BaseGan.add_post_train_hook
    def post_train(self):
        fig = plt.figure()
        plt.plot(logger.scalar['coverage'])
        plt.title('coverage')
        plt.savefig(os.path.join(logger.img_dir, 'coverage.jpg'))
        plt.close()

        fig = plt.figure()
        plt.plot(logger.scalar['unwanted_ratio'])
        plt.title('unwanted_ratio')
        plt.savefig(os.path.join(logger.img_dir, 'unwanted.jpg'))
        plt.close()

    def vis_gen(self, postfix):
        self.eval_mode()
        x_generated = self.sampling(100, 100)
        xrange = ((self.dataset.data[:, 0].min() - 1, self.dataset.data[:, 0].max() + 1),
                  (self.dataset.data[:, 1].min() - 1, self.dataset.data[:, 1].max() + 1))
        fig = plt.figure()
        plt.hist2d(x_generated[:, 0], x_generated[:, 1], bins=100, range=xrange)
        plt.savefig(os.path.join(logger.img_dir, 'hmap_gen_{0:05d}.jpg'.format(postfix)))
        plt.close()

        fig = plt.figure()
        plt.scatter(x_generated[:, 0], x_generated[:, 1], s=1)
        plt.scatter(self.dataset.data[:, 0], self.dataset.data[:, 1], s=1, alpha=0.3)
        plt.savefig(os.path.join(logger.img_dir, 'scatter_gen_{0:05d}.jpg'.format(postfix)))
        plt.close()

    def vis_sampler(self, postfix):
        sampler_modes = self.sampler.sampler_modes.clone().detach().cpu().numpy()

        fig = plt.figure()
        plt.scatter(sampler_modes[:, 0], sampler_modes[:, 1], s=1)
        plt.savefig(os.path.join(logger.img_dir, 'sampler_modes_{0:05d}.jpg'.format(postfix)))
        plt.close()


if __name__ == '__main__':
    dataset_len = 2000
    mode = 10
    latent_dim = 10

    device = 'cuda'
    gloss_cfg = LossConfig('BCE')
    dloss_cfg = LossConfig('BCE')

    sampler_loss_cfg = LossConfig('MSE')
    sampler_optim_cfg = OptimConfig('Adam', lr=0.001)
    sampler_cfg = SamplerConfig(
        name='Learning',
        out_shape=latent_dim,
        data_num=dataset_len,
        batch_size=2000,
        epoch=50,
        alpha=0,
        loss_cfg=sampler_loss_cfg,
        optim_cfg=sampler_optim_cfg,
    )

    goptim_cfg = OptimConfig('Adam', lr=1e-3)
    doptim_cfg = OptimConfig('Adam', lr=1e-3)

    # dataset_cfg = DatasetConfig('Spiral', mode=mode, sig=1, num_per_mode=dataset_len // mode)
    dataset_cfg = DatasetConfig('GMM', mode=mode, sig=1, num_per_mode=dataset_len // mode)
    loader_cfg = LoaderConfig('naive', batch_size=128, shuffle=True)

    gen_cfg = ModelConfig('MLG', input_size=latent_dim, hidden_size=16, output_size=2)
    dis_cfg = ModelConfig('MLD', input_size=2, hidden_size=16, output_size=1)

    gan_cfg = GanConfig(
        name='LearningPrior', gen_cfg=gen_cfg, dis_cfg=dis_cfg,
        gen_step=1, dis_step=1, gan_epoch=2000, loader_cfg=loader_cfg,
        dataset_cfg=dataset_cfg, gloss_cfg=gloss_cfg, dloss_cfg=dloss_cfg,
        goptim_cfg=goptim_cfg, doptim_cfg=doptim_cfg, label_smooth=False,
        sampler_cfg=sampler_cfg, dist_loss=False, device=device
    )
    gan_schema = GanSchema()
    gan_desc = gan_schema.dump(gan_cfg)
    logger.save_cfg(gan_desc)

    gan = gan_cfg.get()
    gan.train(use_tqdm=True)
