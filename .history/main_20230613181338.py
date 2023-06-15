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
    model = torchfcn.models.FCN8s()


if __name__ == '__main__':
    main()
