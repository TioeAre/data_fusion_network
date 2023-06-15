#!/usr/bin/env python
import os
import json
import collections
import os.path as osp

import numpy as np
import torch
from torch.utils import data

class fusion(data.Dataset):
    class_names = np.zeros(100)
    mean_bgr = np.random.rand(100, 4, 15)
    a = np.array([300, 100, 100])
    mean_bgr[:,:,0:3] = a[:]

    def __init__(self, split='train'):
        self.split = split

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join('/home/tioeare/project/FASTLAB/network/make_pose_dataset/src/make_pose_dataset/scripts/data')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            train_filenames = [f for f in os.listdir(f'{dataset_dir}/{split}/data') if os.path.isfile(os.path.join(f'{dataset_dir}/{split}/data', f))]
            label_filenames = [f for f in os.listdir(f'{dataset_dir}/{split}/label') if os.path.isfile(os.path.join(f'{dataset_dir}/{split}/label', f))]
            for i in range(len(train_filenames)):
                with open(f'{dataset_dir}/{split}/data/{train_filenames[i]}', encoding='utf-8') as f:
                    data = json.load(f)
                with open(f'{dataset_dir}/{split}/label/{label_filenames[i]}', encoding='utf-8') as f:
                    label = json.load(f)
                data = json.loads(data)
                label = json.loads(label)
                # json data
                self.files[split].append({'data':data,
                                          'label':label})

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        data = data_file['data']
        img = np.array(data, dtype=np.float32)
        # load label
        lbl_file = data_file['label']
        lbl = np.array(lbl_file, dtype=np.float32)
        return img, lbl

    def untransform(self, img, lbl):
        return img, lbl

    def transform(self, img, lbl):
        return img, lbl