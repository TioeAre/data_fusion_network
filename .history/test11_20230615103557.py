import torch
import numpy as np


a = np.array([[[0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0]]])

input = torch.from_numpy(a)

mask = (input[:,:,:,6:12]>0)
mask = (mask[:,:,:,6:9] > torch.zeros_like(mask)[:,:,:,6:9] and mask[:,:,:,9:12] == torch.zeros_like(mask)[:,:,:,9:12])