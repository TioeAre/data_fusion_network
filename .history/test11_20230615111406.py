import torch
import numpy as np


a = np.array([[[[0, 0, 0,
                0, 0, 0,
                -1, 1, -1,
                -1, 0.6, -0.5,
                1, 1, 1]]]])

input = torch.from_numpy(a)

mask = (input[:, :, :, 6:12])
mask = ((mask[:, :, :, 0:3] > torch.zeros_like(mask)[:, :, :, 0:3]) & (
    mask[:, :, :, 3:6] < torch.zeros_like(mask)[:, :, :, 3:6]))

b = torch.from_numpy(np.array([[[[5, 5, 5,
                                  0, 0, 0,
                                  1, 1, 1,
                                  1, 0, 0,
                                  1, 1, 1]]]]))
b[:,:,:,0:3] = b[:,:,:,0:3] * mask

print(mask)
print(b)
