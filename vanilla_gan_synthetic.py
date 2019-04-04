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


logger = Logger(cat='test')


@BaseGan.register('Vanilla')
class VanillaGan(BaseGan):
    @BaseGan.add_post_iterate_hook
    def vis_gen(self, iter_num):
        if (iter_num + 1) % 1000 != 0:
            return
        self.eval_mode()
        x_generated = self.sampling(100, 100)
        xrange = ((self.dataset.data[:, 0].min() - 1, self.dataset.data[:, 0].max() + 1),
                  (self.dataset.data[:, 1].min() - 1, self.dataset.data[:, 1].max() + 1))
        fig = plt.figure()
        plt.hist2d(x_generated[:, 0], x_generated[:, 1], bins=100, range=xrange)
        plt.savefig(os.path.join(logger.img_dir, 'hmap_gen_{}.jpg'.format(iter_num)))
        plt.close()

        fig = plt.figure()
        plt.scatter(x_generated[:, 0], x_generated[:, 1], s=1)
        plt.scatter(self.dataset.data[:, 0], self.dataset.data[:, 1], s=1, alpha=0.3)
        plt.savefig(os.path.join(logger.img_dir, 'scatter_gen_{}.jpg'.format(iter_num)))
        plt.close()


if __name__ == '__main__':
    dataset_len = 2000
    mode = 20
    latent_dim = 10

    device = 'cuda'
    gloss_cfg = LossConfig('BCE')
    dloss_cfg = LossConfig('BCE')

    sampler_cfg = SamplerConfig(
        name='Gaussian',
        out_shape=latent_dim,
    )

    goptim_cfg = OptimConfig('Adam', lr=1e-3)
    doptim_cfg = OptimConfig('Adam', lr=1e-3)

    dataset_cfg = DatasetConfig('Spiral', mode=mode, sig=1, num_per_mode=dataset_len // mode)
    # dataset_cfg = DatasetConfig('GMM', mode=mode, sig=1, num_per_mode=dataset_len // mode)
    loader_cfg = LoaderConfig('naive', batch_size=128, shuffle=True)

    gen_cfg = ModelConfig('MLG', input_size=latent_dim, hidden_size=32, output_size=2)
    dis_cfg = ModelConfig('MLD', input_size=2, hidden_size=32, output_size=1)

    gan_cfg = GanConfig(
        name='Vanilla', gen_cfg=gen_cfg, dis_cfg=dis_cfg,
        gen_step=1, dis_step=1, gan_epoch=15000, loader_cfg=loader_cfg,
        dataset_cfg=dataset_cfg, gloss_cfg=gloss_cfg, dloss_cfg=dloss_cfg,
        goptim_cfg=goptim_cfg, doptim_cfg=doptim_cfg, label_smooth=False,
        sampler_cfg=sampler_cfg, device=device
    )
    gan_schema = GanSchema()
    gan_desc = gan_schema.dump(gan_cfg)
    logger.save_cfg(gan_desc)

    gan = gan_cfg.get()
    gan.train(use_tqdm=True)


