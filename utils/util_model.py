import torch as th
import abc
import numpy as np
from utils.class_helper import ConfigHelper, BaseHelper
from marshmallow import Schema, fields, post_load
from utils.optim import OptimSchema
from utils.loss import LossSchema
from utils.dataloader import LoaderSchema
from dataset.config import DatasetSchema
from torch import nn
import logging
import tqdm
from torch.nn import functional as F


class UtilityModelConfig(ConfigHelper):
    def __init__(self, name, scratch, weight_pth, epochs,
                 trainloader_cfg, testloader_cfg,
                 train_data_cfg, test_data_cfg, img_size, device):
        super(UtilityModelConfig, self).__init__()
        self.name = name
        self.scratch = scratch
        self.weight_pth = weight_pth
        self.train_data_cfg = train_data_cfg
        self.test_data_cfg = test_data_cfg
        self.dataloader_cfg = trainloader_cfg
        self.testloader_cfg = testloader_cfg
        self.device = device
        self.epochs = epochs
        self.img_sz = img_size

    def get(self):
        model = BaseUtilityModel.create(name=self.name, img_sz=self.img_sz)
        if self.scratch:
            print('Training a NaiveClassifier from scratch.')
            dataset = self.train_data_cfg.get()
            loader = self.dataloader_cfg.get(dataset)
            model = NaiveClassifier(self.img_sz)
            optim = th.optim.Adam(model.parameters())
            cri = th.nn.NLLLoss()
            model.train()
            model.to(self.device)
            for epoch in tqdm.tqdm(range(self.epochs)):
                totloss = 0.0
                for batch_idx, (data, target) in enumerate(loader):
                    data, target = data.to(self.device).float(), target.to(self.device)
                    optim.zero_grad()
                    output = model(data)
                    loss = cri(output, target)
                    loss.backward()
                    optim.step()
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, totloss))
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, totloss))
            state_dict = model.state_dict()
            th.save(state_dict, self.weight_pth)
            print('Weight saved to {}'.format(self.weight_pth))

            test_data = self.test_data_cfg.get()
            test_loader = self.testloader_cfg.get(test_data)
            model.eval()
            test_loss = 0
            correct = 0
            with th.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device).float(), target.to(self.device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target).item()  # sum up batch loss
                    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

        else:
            if self.scratch:
                logging.warning('Weight file exists.')
                print('Weight file exists.')
            state_dict = th.load(self.weight_pth)
            model.load_state_dict(state_dict)
            print('Loading model from {}'.format(self.weight_pth))
        return model


class UtilityModelSchema(Schema):
    name = fields.Str()
    scratch = fields.Str()
    weight_pth = fields.Str()
    train_data_cfg = fields.Nested(DatasetSchema)


class BaseUtilityModel(BaseHelper):
    subclasses = {}


@BaseUtilityModel.register('NaiveClassifier')
class NaiveClassifier(BaseUtilityModel, nn.Module):
    def __init__(self, img_sz):
        super(NaiveClassifier, self).__init__()
        self.img_sz = img_sz
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        if img_sz == 32:
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
        else:
            self.fc1 = nn.Linear(16 * 4 * 4, 120)

    def forward(self, x):
        # embed()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.img_sz == 32:
            x = x.view(-1, 16 * 5 * 5)
        else:
            x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
