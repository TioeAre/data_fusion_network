import os.path as osp
import torch
import numpy as np
import fcn
import torch.nn as nn
import torchfcn.trainer

n = torchfcn.trainer.n


def get_upsampling_weight(in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                        dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight).float()

class FCN8s(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=1ieXWyoG68xqoHJPWdyrDyaIaaWlfmUxI',  # NOQA
            path=cls.pretrained_model,
            md5='de93e540ec79512f8770033849c8ae89',
        )

    def __init__(self, n_class=100):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(100, 100, 7, padding=2)
        self.conv5_1 = nn.Conv2d(100, 100, 4, padding=2)
        self.relu1_1 = nn.PReLU() # inplace=True)
        self.conv1_2 = nn.Conv2d(100, 100, 4, padding=1)
        self.conv5_2 = nn.Conv2d(100, 100, 4, padding=1)
        self.relu1_2 = nn.PReLU() # iinplace=True)
        # self.pool1 = nn.MaxPool1d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(100, 100, 3, padding=1)
        self.conv6_1 = nn.Conv2d(100, 100, 3, padding=1)
        self.relu2_1 = nn.PReLU() # iinplace=True)
        self.conv2_2 = nn.Conv2d(100, 100, 3, padding=1)
        self.conv6_2 = nn.Conv2d(100, 100, 3, padding=1)
        self.relu2_2 = nn.PReLU() # iinplace=True)
        # self.pool2 = nn.MaxPool1d(2, stride=2, ceil_mode=True)  # 1/4

        self.conv3_3 = nn.Conv2d(100, 100, 3, padding=1)
        self.conv7_3 = nn.Conv2d(100, 100, 3, padding=1)
        self.relu3_3 = nn.PReLU() # iinplace=True)
        self.relu7_3 = nn.PReLU() # iinplace=True)
        # self.pool3 = nn.MaxPool1d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(n_class, n_class, 3, padding=1)

        self.relu4_1 = nn.PReLU() # iinplace=True)
        self.relu8_1 = nn.PReLU() # iinplace=True)
        # self.conv4_2 = nn.Conv2d(4096, 1024, 3, padding=1)
        self.score_fr = nn.Conv2d(100, n_class, 2, padding=1) # 4096
        self.conv8_2 = nn.Conv2d(100, 100, 3, padding=1)
        self.conv4_3 = nn.Conv2d(100, 100, 3, padding=1)
        self.conv8_3 = nn.Conv2d(100, 100, 3, padding=1)

        # self.score_fr = nn.Conv2d(32, n_class, 2, padding=1) # 4096

        self.score_pool3 = nn.Conv2d(n_class, n_class, 1)
        # self.score_pool4 = nn.Conv1d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore3 = nn.ConvTranspose2d(
            1, 4, 4, 1, 1, bias=False)

        # self.upscore8 = nn.ConvTranspose1d(
        #     n_class, n_class, 16, stride=8, bias=False)
        # self.upscore_pool4 = nn.ConvTranspose1d(
        #     n_class, n_class, 4, stride=2, bias=False)
        # self.pool6 = nn.AdaptiveAvgPool1d((1,15))
        if n == 15:
            self.liner1 = nn.Linear(12, 3, device=0)
            self.liner = nn.Linear(60, 60, device=0)
        elif n == 6 :
            self.liner = nn.Linear(64, 24, device=0)
        else:
            self.liner = nn.Linear(40, 12, device=0)
        # self.liner3 = nn.Linear(168, 12, device=0)
        self.liner3 = nn.Linear(96, 12, device=0)
        self.liner4 = nn.Linear(36, 24, device=0)
        # self.softmax = torch.nn.functional.softmax(dim=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, nn.PReLU):
            #     nn.init.normal_(m.weight, mean=1.0, std=0.01)

    def forward(self, x):
        # x = x[0:1,0:100,0:4,0:15]

        # h = x
        h = torch.cat((x[:,:,:,0:6], x[:,:,:,12:15]), dim = 3)

        if n == 15:
            h1 = x[:,:,:,6:15]
            h2 = torch.cat((x[:,:,:,0:3], x[:,:,:,6:12]), dim = 3) # result 100*4*9
            # h2[:,:,:,0:3] = torch.nn.functional.normalize(h2[:,:,:,0:3], dim=2)
            h2 = self.conv5_1(h2) # 100*4*15
            h2 = self.conv5_2(h2)
            h2 = self.conv6_1(h2)
            h2 = self.conv6_2(h2)
            h2 = self.relu7_3(self.conv7_3(h2))
            h2 = self.conv8_3(h2)
            h2 = torch.reshape(h2, (100,-1))
            h2 = self.liner4(h2)
            h2 = torch.reshape(h2, (1, 100,4,6))
            # h2 = torch.cat(((h2[:,:,:,0:3] > torch.zeros_like(h2)[:,:,:,0:3]),(h2[:,:,:,3:6] <= torch.zeros_like(h2)[:,:,:,3:6])), dim=3)
            h2 = torch.nn.functional.sigmoid(h2) # torch.nn.functional.softmax(h2, dim=2)
            # h2 = torch.reshape(h2,((100,4,6)))
            mask1 = (h2[:,:,:,:])
            mask2 = torch.full(mask1.shape, 0.5).cuda() # ((mask1[:,:,:,0:3] > torch.zeros_like(mask1)[:,:,:,0:3]) & (mask1[:,:,:,3:6] <= torch.zeros_like(mask1)[:,:,:,3:6])).cuda() # to(torch.device("cuda"))
            mask1 = ((mask1[:,:,:,0:3] > mask2[:,:,:,0:3]) & (mask1[:,:,:,3:6] <= mask2[:,:,:,3:6]))
            # h2 = torch.sigmoid(h2)
            # mask3 = (h2[:,:,:,0:3] > torch.zeros_like(h2)[:,:,:,0:3]) & (h2[:,:,:,3:6] <= torch.zeros_like(h2)[:,:,:,3:6])
            h[:,:,:,0:3] = h[:,:,:,0:3] * mask1 # mask3


        # print(h.shape)
        h = self.conv1_1(h)
        h = self.conv1_2(h)
        # h = self.pool1(h)
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        # h = self.pool2(h)
        h = self.relu3_3(self.conv3_3(h))
        h = self.score_fr(h)
        h = self.score_pool3(h)
        h = self.upscore2(h)
        # h = self.conv4_1(h)

        h = torch.reshape(h, (h.size(1),h.size(2),h.size(3)))
        h = torch.reshape(h, (100,-1))
        h = self.liner3(h)
        h = torch.reshape(h, (1, 100, 4, 3))

        if n == 15:

            h = h.contiguous()
            h = torch.reshape(h, (100,-1))
            h = self.liner1(h)
            h = torch.reshape(h, (1,100,1,3))
            h2 = torch.reshape(h2,((1,100,1,24)))
            h = torch.cat((h, h2), dim = 3)
            # h = h.view(h.size(0), -1)
            # h = self.liner(h)
        else:
            h = h.contiguous()
            h = h.view(h.size(0), -1)
            h = self.liner(h)
        if n == 15:
            h = torch.reshape(h, (1,100,1,27))
            # h[:,:,6:15] = torch.sigmoid(h[:,:,6:15])
            # h[:,:,6:15] = torch.where(h[:,:,6:15] > 0.5, torch.tensor(1), torch.tensor(0)).float()
            # h[:,:,6:15] = torch.round(h[:,:,6:15])
        h = h.contiguous()
        return h

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)

    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                        dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight).float()


class FCN8sAtOnce(FCN8s):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s-atonce_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vblE1VUIxV1o2d2M',
            path=cls.pretrained_model,
            md5='bfed4437e941fef58932891217fe6464',
        )

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                if l1.bias is not None and l2.bias is not None:
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data.copy_(l1.weight.data)
                    l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
