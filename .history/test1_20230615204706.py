import torch

original_tensor = torch.tensor([[[1, 1, 1], [1, 1, 1]]])

fixed_value = 0.5
new_tensor = torch.full(original_tensor.shape, fixed_value)
mask1 = ((original_tensor[:,:,0:3] > new_tensor[:,:,0:3]) & (original_tensor[:,:,0:3] <= new_tensor[:,:,0:3]))
print(original_tensor)
print(new_tensor)
print(mask1)