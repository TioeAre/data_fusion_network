import torch

original_tensor = torch.tensor([1, 1, 1], [0, 0, 0])

fixed_value = 0.5
new_tensor = torch.full(original_tensor.shape, fixed_value)

print(new_tensor)