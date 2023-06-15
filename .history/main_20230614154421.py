#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

import torch
import torch.utils
from torch.utils.data import DataLoader
import yaml
import numpy as np
import torchfcn

from examples.voc.train_fcn32s import get_parameters
from examples.voc.train_fcn32s import git_hash


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        required=False, help='gpu id')
    # parser.add_argument('--resume', help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--max-iteration', type=int, default=1208*1, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    # parser.add_argument(
    #     '--pretrained-model',
    #     default="/home/tioeare/project/FASTLAB/network/data_fusion_fcn/model/fcn8s-heavy-pascal.pth",
    #     help='pretrained model of FCN16s',
    # )
    args = parser.parse_args()

    args.model = 'FCN8s'
    # args.git_hash = git_hash()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
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
        torchfcn.datasets.fusion(split='val'),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN8s(n_class=100)
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
            {'params': get_parameters(model, bias=True), 'lr': args.lr * 2, 'weight_decay': 0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    # if args.resume:
        # optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=4000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
