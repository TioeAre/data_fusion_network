#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data

class fusuion(data.Dataset):
    def __init__(self) -> None:
        super().__init__()