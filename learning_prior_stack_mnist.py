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

logger = Logger(cat='stack_mnist')


class StackMNISTGANSchema(GanSchema):
    util_cfg = fields.Nested(UtilityModelSchema)


@BaseGan.register('StackMNISTLPG')
class StackMNISTGAN(BaseGan):
    def __init__(self, gen_cfg, dis_cfg, gen_step,
                 dis_step, gan_epoch, loader_cfg,
                 dataset_cfg, gloss_cfg, dloss_cfg,
                 goptim_cfg, doptim_cfg,
                 sampler_cfg, label_smooth, dist_loss,
                 device, util_cfg):
        super(StackMNISTGAN, self).__init__(
            gen_cfg, dis_cfg, gen_step,
            dis_step, gan_epoch, loader_cfg,
            dataset_cfg, gloss_cfg, dloss_cfg,
            goptim_cfg, doptim_cfg,
            sampler_cfg, label_smooth, dist_loss,
            device
        )
        self.util_model = util_cfg.get().to(device)

    @BaseGan.add_pre_train_hook
    def pre_train(self):
        logger.create_scalar('js')
        logger.create_scalar('coverage')
        logger.create_scalar('coverage_HQ')
        logger.create_scalar('confidence')
        logger.create_scalar('ratio')

    @BaseGan.add_post_iterate_hook
    def post_iterate(self, iter_num):
        if (iter_num + 1) % 1000 == 0:
            self.vis_gen(postfix=iter_num)
            JS, C, C_actual, conf, conf_ratio = self.coverage()
            logger.append_scalar('js', JS)
            logger.append_scalar('coverage', C)
            logger.append_scalar('coverage_HQ', C_actual)
            logger.append_scalar('confidence', conf)
            logger.append_scalar('ratio', conf_ratio)

    @BaseGan.add_post_train_hook
    def post_train(self):
        self.save(logger.exp_dir, 'final')

        fig = plt.figure()
        plt.plot(logger.scalar['coverage'])
        plt.title('coverage')
        plt.savefig(os.path.join(logger.img_dir, 'coverage.jpg'))
        plt.close()

        fig = plt.figure()
        plt.plot(logger.scalar['js'])
        plt.title('js')
        plt.savefig(os.path.join(logger.img_dir, 'js.jpg'))
        plt.close()

        fig = plt.figure()
        plt.plot(logger.scalar['coverage_HQ'])
        plt.title('coverage_HQ')
        plt.savefig(os.path.join(logger.img_dir, 'coverage_HQ.jpg'))
        plt.close()

        fig = plt.figure()
        plt.plot(logger.scalar['confidence'])
        plt.title('confidence')
        plt.savefig(os.path.join(logger.img_dir, 'confidence.jpg'))
        plt.close()

        fig = plt.figure()
        plt.plot(logger.scalar['ratio'])
        plt.title('ratio')
        plt.savefig(os.path.join(logger.img_dir, 'ratio.jpg'))
        plt.close()

    def vis_gen(self, postfix):
        self.eval_mode()
        x_generated = self.sampling(1, 100)
        r = x_generated[:, 0, :, :]
        g = x_generated[:, 1, :, :]
        b = x_generated[:, 2, :, :]
        x = np.concatenate([r[:, None], g[:, None], b[:, None]], axis=3)
        img = make_grid(th.Tensor(x))
        save_image(img, os.path.join(logger.img_dir, 'gen_{0:05d}.jpg'.format(postfix)))

    def coverage(self):
        self.util_model.eval()
        self.eval_mode()
        bs = 100
        along_width = False
        classifier_threshold = 0.99

        result = []
        result_probs = []
        result_is_confident = []

        for b_num in range(100):
            x_generated = self.sampling(10, 100)
            for idx in range(x_generated.shape[0] // bs):
                inp1, inp2, inp3 = np.split(
                    x_generated[idx * bs: idx * bs + bs],
                    3,
                    axis=3 if along_width else 1
                )

                res1 = self.util_model(th.Tensor(inp1).to(self.device)).cpu().detach().numpy()
                res2 = self.util_model(th.Tensor(inp2).to(self.device)).cpu().detach().numpy()
                res3 = self.util_model(th.Tensor(inp3).to(self.device)).cpu().detach().numpy()

                prob1 = np.exp(res1)
                prob2 = np.exp(res2)
                prob3 = np.exp(res3)

                prob1 = np.max(prob1, axis=1)
                prob2 = np.max(prob2, axis=1)
                prob3 = np.max(prob3, axis=1)

                res1 = np.argmax(res1, axis=1)
                res2 = np.argmax(res2, axis=1)
                res3 = np.argmax(res3, axis=1)

                res = res1 * 100 + res2 * 10 + res3

                result.append(res)
                result_is_confident.append(
                    (prob1 > classifier_threshold) *
                    (prob2 > classifier_threshold) *
                    (prob3 > classifier_threshold)
                )
                result_probs.append(np.column_stack((prob1, prob2, prob3)))

        result = np.hstack(result)
        result_probs = np.vstack(result_probs)
        result_is_confident = np.hstack(result_is_confident)

        digits = result.astype(int)
        print('Ratio of confident predictions: %.4f' % np.mean(result_is_confident))

        conf_ratio = np.mean(result_is_confident)

        conf = np.mean(result_probs)
        if np.sum(result_is_confident) == 0:
            C_actual = 0.
            C = 0.
            JS = 2.
        else:
            # Compute the actual coverage
            C_actual = len(np.unique(digits[result_is_confident])) / 1000.
            # Compute the JS with uniform
            JS = js_div_uniform(digits)
            # Compute Pdata(Pmodel > t) where Pmodel( Pmodel > t ) = 0.95
            # np.percentaile(a, 10) returns t s.t. np.mean( a <= t ) = 0.1
            phat = np.bincount(digits[result_is_confident], minlength=1000)
            phat = (phat + 0.) / np.sum(phat)
            threshold = np.percentile(phat, 5)
            ratio_not_covered = np.mean(phat <= threshold)
            C = 1. - ratio_not_covered

        print('Evaluating: JS=%.3f, C=%.3f, C_actual=%.3f, Confidence=%.4f\n' % (JS, C, C_actual, conf))
        return JS, C, C_actual, conf, conf_ratio


if __name__ == '__main__':
    MODE = 1000
    LATENT_DIM = 100
    IMAGE_SIZE = 32
    DATASET_LEN = 1000 * 20
    EPOCHS = 20000
    device = 'cuda'
    gloss_cfg = LossConfig('BCE')
    dloss_cfg = LossConfig('BCE')

    sampler_cfg = SamplerConfig(
        name='Onehot',
        out_shape=DATASET_LEN,
        latent_dim=LATENT_DIM
    )
    goptim_cfg = OptimConfig('Adam', lr=1e-3)
    doptim_cfg = OptimConfig('Adam', lr=1e-3)

    dataset_cfg = DatasetConfig('MNIST', train=True, stack=True, along_width=False, size=32)
    loader_cfg = LoaderConfig('naive', batch_size=128, shuffle=True)

    gen_cfg = ModelConfig('EDCG', input_size=LATENT_DIM, hidden_size=128, output_size=3, data_num=DATASET_LEN)
    dis_cfg = ModelConfig('DCD', input_size=3, hidden_size=64, output_size=1)

    train_data_cfg = DatasetConfig('MNIST', stack=False, train=True, along_width=False)
    test_data_cfg = DatasetConfig('MNIST', stack=False, train=False, along_width=False)

    train_loader = LoaderConfig('naive', batch_size=128, shuffle=True)
    test_loader = LoaderConfig('naive', batch_size=128, shuffle=True)

    util_cfg = UtilityModelConfig('NaiveClassifier', False, '/home/bourgan/gan_dev/checkpoints/mnist_naive.pth.tar',
                                  15, train_loader, test_loader, train_data_cfg, test_data_cfg, 32, 'cuda')
    gan_cfg = GanConfig(
        name='StackMNISTLPG', gen_cfg=gen_cfg, dis_cfg=dis_cfg,
        gen_step=1, dis_step=1, gan_epoch=EPOCHS, loader_cfg=loader_cfg,
        dataset_cfg=dataset_cfg, gloss_cfg=gloss_cfg, dloss_cfg=dloss_cfg,
        goptim_cfg=goptim_cfg, doptim_cfg=doptim_cfg, label_smooth=False,
        sampler_cfg=sampler_cfg, dist_loss=False, device=device, util_cfg=util_cfg
    )
    gan_schema = GanSchema()
    gan_desc = gan_schema.dump(gan_cfg)
    logger.save_cfg(gan_desc)

    gan = gan_cfg.get()
    gan.train(use_tqdm=True)
    gan.save('checkpoints', 'stack_mnist')
