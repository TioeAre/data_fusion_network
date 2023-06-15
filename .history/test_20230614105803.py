import torch

import numpy as np
import skimage.data
from torch.utils.data import DataLoader
import torchfcn
import os
import torch.utils
import torch
from torchfcn.models.fcn8s import get_upsampling_weight


def test_get_upsampling_weight():
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    val_loader = DataLoader(
        torchfcn.datasets.fusion(split='val'),
        batch_size=1, shuffle=False, **kwargs)
    print(np.shape(val_loader.dataset))
    src = np.array(val_loader.dataset[0])
    x = src.transpose(2, 0, 1)
    x = x[np.newaxis, :, :, :]
    x = torch.from_numpy(x).float()
    x = torch.autograd.Variable(x)

    in_channels = 100
    out_channels = 100
    kernel_size = 4

    m = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=2, bias=False)
    m.weight.data = get_upsampling_weight(
        in_channels, out_channels, kernel_size)

    y = m(x)

    y = y.data.numpy()
    y = y[0]
    y = y.transpose(1, 2, 0)
    dst = y.astype(np.uint8)

    assert abs(src.shape[0] * 2 - dst.shape[0]) <= 2
    assert abs(src.shape[1] * 2 - dst.shape[1]) <= 2

    return src, dst


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    src, dst = test_get_upsampling_weight()
    plt.subplot(121)
    plt.imshow(src)
    plt.title('x1: {}'.format(src.shape))
    plt.subplot(122)
    plt.imshow(dst)
    plt.title('x2: {}'.format(dst.shape))
    plt.show()
