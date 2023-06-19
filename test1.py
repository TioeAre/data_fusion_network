import torch
from torch import nn
In = nn.InstanceNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)
x = torch.rand(10, 3, 5, 5)*10000
official_In = In(x)   # 官方代码

x1 = x.reshape(30, -1)  # 对（H,W）计算均值方差
mean = x1.mean(dim=1).reshape(10, 3, 1, 1)
std = x1.std(dim=1, unbiased=False).reshape(10, 3, 1, 1)
my_In = (x - mean)/std
print(official_In)
print(my_In)