import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import pytz
import skimage.io
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import torchfcn
from typing import Any

n = 15

def loss_func(input, target, weight=None, size_average=True):
    input2 = input[:,:,:,3:27]
    input2 = torch.reshape(input2, ((1,100,4,6)))
    input = input[:,:,:,0:3]
    a = 0.7
    mse_loss = torch.nn.MSELoss()
    bce_loss = nn.BCELoss(reduction = 'sum')
    target1 = target[:,:,:,6:12]
    if n == 15:
        loss2 = bce_loss(input2[:,:,:,0:6], torch.round(target[:,:,:,6:12]).float()) # torch.abs(input2[:,:,:,0:6]-torch.round(target[:,:,:,6:12])).sum()

    # mask1 = (input[:,:,:,6:12])
    # mask1 = ((mask1[:,:,:,0:3] > torch.zeros_like(mask1)[:,:,:,0:3]) & (mask1[:,:,:,3:6] <= torch.zeros_like(mask1)[:,:,:,3:6]))
    mask2 = (target[:,:,:,6:12])
    mask2 = ((mask2[:,:,:,0:3] > torch.zeros_like(mask2)[:,:,:,0:3]) & (mask2[:,:,:,3:6] <= torch.zeros_like(mask2)[:,:,:,3:6]))
    # input[:,:,:,0:3] = input[:,:,:,0:3] * mask1
    target[:,:,:,0:3] = target[:,:,:,0:3] * mask2
    mask3 = torch.sum(mask2, dim=2)
    mask3[mask3 == 0] = 1
    target[:,:,0:1,0:3] = torch.reshape(torch.sum(target[:,:,:,0:3]-target[:,:,:,3:6], dim=2)/mask3, (1, 100, 1, 3))
    # target[torch.isinf(target)] = 0
    target2 = target[:,:,:,0:3]
    if n == 3:
        loss1 = mse_loss(input[:,:,:,0:3], target[:,:,:,0:3])
    else:
        loss1 = mse_loss(input[:,:,:,0:3], target[:,:,0:1,0:3])
    if n == 15:
        # loss2 = bce_loss(input[:,:,:,6:15], torch.round(target[:,:,:,6:15]).float())
        loss = a*loss1 + (1-a)*loss2
    else:
        loss = loss1
    return loss


class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(self.val_loader), total=len(self.val_loader),
                                                    desc='Valid iteration=%d' % self.iteration,
                                                    leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if np.shape(data[0])[0] < 100 or np.shape(target[0])[0] < 100:
                continue

            target = target[0:1,0:100,0:4,0:n]
            data = data[0:1,0:100,0:4,0:n]

            # data1 = torch.reshape(data, (100,4,n))
            with torch.no_grad():
                score = self.model(data)
                # score = torch.reshape(score, ((1,100,1,27)))
            loss = loss_func(score, target,
                                   size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)
            label_trues.append(target.cpu().numpy()) # n, 1, 100, 4, 15
            label_preds.append(score.cpu().numpy())
        metrics = torchfcn.utils.label_accuracy_score(label_trues, label_preds, n_class)
        val_loss /= len(self.val_loader) if len(self.val_loader) != 0 else 1
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)
        i = 0
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if np.shape(data[0])[0] < 100 or np.shape(target[0])[0] < 100:
                continue

            target = target[0:1,0:100,0:4,0:n]
            data = data[0:1,0:100,0:4,0:n]
            i=i+1
            self.optim.zero_grad()
            # data1 = torch.reshape(data, (100,4,n))
            score = self.model(data)
            # score = torch.reshape(score, ((1,100,1,27)))

            loss = loss_func(score, target,
                                   size_average=self.size_average)
            loss /= len(data)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            # if i > 50:
            loss.backward()
            self.optim.step()
                # self.optim.zero_grad()
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print(name, param.data)

            metrics = []
            lbl_pred = score.data.cpu().numpy()[:, :, :] #max(1)[1].
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = \
                torchfcn.utils.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
        torch.save(self.model.state_dict(),"./model/fcn8s3_1.pth")
        torch.save(self.model,"./model/fcn8s3_2.pth")
