import matplotlib
matplotlib.use('agg')

from utils.logger import Logger
from utils.sampler import SamplerConfig
from utils.loss import LossConfig
from utils.optim import OptimConfig
from utils.dataloader import LoaderConfig
from model.gan import GanConfig, BaseGan
from model.zoo import ModelConfig
from dataset.config import DatasetConfig
from matplotlib import pyplot as plt
import os
import tqdm
import numpy as np
import torch as th
from torch.nn import functional as F
from model.zoo import SampleMatrix
from sklearn import manifold


logger = Logger(cat='embedding')


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


@BaseGan.register('Embedding')
class LearningPriorGan(BaseGan):
    def gen_iter(self):
        self.gen.zero_grad()
        sampler_idx = self.sampler.sampling(self.batch_size)
        g_gen_input = sampler_idx.to(dtype=th.float32, device=self.device)
        g_fake_data = self.gen(g_gen_input)
        g_fake_deision = self.dis(g_fake_data)
        g_fake_labels = th.ones(g_fake_deision.shape, dtype=th.float32, device=self.device)

        g_loss = self.gen_loss(g_fake_deision, g_fake_labels)
        g_loss.backward()
        self.gen_optim.step()

    def dis_iter(self):
        self.dis.zero_grad()
        real_idx, real_samples = next(iter(self.dataloader))
        d_real_data = real_samples.to(dtype=th.float32, device=self.device)
        d_real_decision = self.dis(d_real_data)
        d_real_labels = th.ones(d_real_decision.shape, dtype=th.float32, device=self.device)
        d_real_loss = self.dis_loss(d_real_decision, d_real_labels)

        d_gen_input = real_idx.to(dtype=th.float32, device=self.device)
        d_fake_data = self.gen(d_gen_input)
        d_fake_decision = self.dis(d_fake_data)
        d_fake_labels = th.zeros(d_fake_decision.shape, dtype=th.float32, device=self.device)
        d_fake_loss = self.dis_loss(d_fake_decision, d_fake_labels)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.dis_optim.step()

    def sampling(self, num_batch, batch_size):
        generated_pts = []
        for i in range(num_batch):
            samples = self.sampler.sampling(batch_size)
            samples = samples.to(self.device).float()
            self.eval_mode()
            res = self.gen(samples)
            generated_pts.append(res.cpu().detach().numpy())
        generated_pts = np.concatenate(generated_pts)
        return generated_pts

    @BaseGan.add_pre_train_hook
    def pre_train(self):
        logger.create_scalar('coverage')
        logger.create_scalar('unwanted_ratio')
        self.TSNE = manifold.TSNE(verbose=1)

    @BaseGan.add_post_iterate_hook
    def post_iterate(self, iter_num):
        if (iter_num + 1) % 1000 == 0:
            self.vis_gen(postfix=iter_num)
            self.coverage()
        if (iter_num + 1) % 5000 == 0:
            self.vis_embedding(postfix=iter_num)

    def coverage(self):
        self.eval_mode()
        x_generated = self.sampling(100, 100)
        modes = self.dataset.modes
        coverage, unwanted_ratio, _ = radius_coverage(x_generated, modes, sigma=1, eps=0.01)
        print(coverage, unwanted_ratio)
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

    def vis_embedding(self, postfix):
        weights = list(self.gen.embedding.parameters())[0].detach().cpu().numpy().T
        embed = self.TSNE.fit_transform(weights)

        fig = plt.figure()
        plt.scatter(embed[:, 0], embed[:, 1], s=1)
        plt.savefig(os.path.join(logger.img_dir, 'embed_{0:05d}.jpg'.format(postfix)))
        plt.close()


if __name__ == '__main__':
    DATASET_LEN = 2000
    MODE = 20
    LATENT_DIM = 10
    GAN_EPOCHS = 40000

    device = 'cuda'
    gloss_cfg = LossConfig('BCE')
    dloss_cfg = LossConfig('BCE')

    sampler_cfg = SamplerConfig(
        name='Onehot',
        out_shape=DATASET_LEN,
    )

    goptim_cfg = OptimConfig('Adam', lr=1e-3)
    doptim_cfg = OptimConfig('Adam', lr=1e-3)

    dataset_cfg = DatasetConfig('Spiral', mode=MODE, sig=1, num_per_mode=DATASET_LEN // MODE, return_idx=True)
    # dataset_cfg = DatasetConfig('GMM', mode=mode, sig=1, num_per_mode=dataset_len // mode, return_idx=True)
    loader_cfg = LoaderConfig('naive', batch_size=128, shuffle=True)

    gen_cfg = ModelConfig('EMLG', input_size=LATENT_DIM, hidden_size=32, output_size=2, data_num=DATASET_LEN)
    dis_cfg = ModelConfig('MLD', input_size=2, hidden_size=32, output_size=1)

    gan_cfg = GanConfig(
        name='Embedding', gen_cfg=gen_cfg, dis_cfg=dis_cfg,
        gen_step=1, dis_step=1, gan_epoch=GAN_EPOCHS, loader_cfg=loader_cfg,
        dataset_cfg=dataset_cfg, gloss_cfg=gloss_cfg, dloss_cfg=dloss_cfg,
        goptim_cfg=goptim_cfg, doptim_cfg=doptim_cfg, label_smooth=False,
        sampler_cfg=sampler_cfg, dist_loss=False, device=device
    )
    gan_schema = GanSchema()
    gan_desc = gan_schema.dump(gan_cfg)
    logger.save_cfg(gan_desc)

    gan = gan_cfg.get()
    gan.train(use_tqdm=True)
    gan.save('embedding', 'trial1')
