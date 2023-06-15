import torch
import numpy as np


a = np.array([[[[0, 0, 0,
                0, 0, 0,
                1, 1, 1,
                0, 0, 0,
                1, 1, 1]]]])

input = torch.from_numpy(a)

mask = (input[:,:,:,6:12]>0)
print(mask)
print(mask[:,:,:,0:3])
mask2 = mask[:,:,:,0:3]
mask1 = torch.zeros_like(mask)[:,:,:,0:3]
print(mask1)
print("mask")
print(mask2 != mask1)
print(mask[:,:,:,3:6] == torch.zeros_like(mask)[:,:,:,3:6])
mask = (mask[:,:,:,0:3] != torch.zeros_like(mask)[:,:,:,0:3] and mask[:,:,:,3:6] == torch.zeros_like(mask)[:,:,:,3:6])

print(mask)