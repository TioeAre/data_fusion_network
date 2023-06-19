#!/usr/bin/env python
import argparse
import os
import os.path as osp
from torch.utils.data import DataLoader
import fcn
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import torchfcn
import tqdm
from torchfcn.trainer import loss_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', default="/root/data_fusion_network/model/fcn8s2_1.pth", help='Model path')

    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    val_loader = DataLoader(
        torchfcn.datasets.fusion(split='val'),
        batch_size=1, shuffle=False, **kwargs)

    n_class = len(val_loader.dataset.class_names)

    model = torchfcn.models.FCN8s()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    visualizations = []
    label_trues, label_preds = [], []
    val_loss = 0
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        torch.no_grad()
        target = target[0:1,0:100,0:4,0:15]
        data = data[0:1,0:100,0:4,0:15]
        data1 = torch.reshape(data, (1,100,4,15))
        score = model(data1)
        # score = torch.reshape(score, ((1,100,1,27)))

        imgs = data.data.cpu()
        loss = loss_func(score, target)
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while validating')
        val_loss += loss_data / len(data)
        label_trues.append(target.cpu().numpy()) # n, 1, 100, 4, 15
        label_preds.append(score.cpu().detach().numpy())
    metrics = torchfcn.utils.label_accuracy_score(label_trues, label_preds, n_class)
    val_loss /= len(val_loader) if len(val_loader) != 0 else 1
    print(val_loss)



if __name__ == '__main__':
    main()
