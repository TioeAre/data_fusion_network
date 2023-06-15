#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

import torch
import torch.utils
from torch.utils.data import DataLoader
import yaml

import torchfcn

from examples.voc.train_fcn32s import get_parameters
from examples.voc.train_fcn32s import git_hash


here = osp.dirname(osp.abspath(__file__))


def main():

    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(
        torchfcn.datasets.fusion(split='train'),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = DataLoader(
        torchfcn.datasets.fusion(split='seg11valid'),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN8s(n_class=1)
    start_epoch = 0
    start_iteration = 0
    # if args.resume:
    #     checkpoint = torch.load(args.resume)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     start_iteration = checkpoint['iteration']
    # else:
    #     fcn16s = torchfcn.models.FCN8s()
    #     state_dict = torch.load(args.pretrained_model)
    #     try:
    #         fcn16s.load_state_dict(state_dict)
    #     except RuntimeError:
    #         fcn16s.load_state_dict(state_dict['model_state_dict'])
    #     model.copy_params_from_fcn16s(fcn16s)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': args.lr * 2, 'weight_decay': 0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    # if args.resume:
        # optim.load_state_dict(checkpoint['optim_state_dict'])



if __name__ == '__main__':
    main()
