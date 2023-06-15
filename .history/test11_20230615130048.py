    mask1 = (input[:,:,:,6:12])
    mask1 = ((mask1[:,:,:,0:3] > torch.zeros_like(mask1)[:,:,:,0:3]) & (mask1[:,:,:,3:6] <= torch.zeros_like(mask1)[:,:,:,3:6]))
    input[:,:,:,0:3] = input[:,:,:,0:3] * mask1