import numpy as np

mean_bgr = np.random.rand(100, 4, 15)
a = np.array([300, 100, 100])
mean_bgr[:,:,:3] = a[:]
mean_bgr[:,:,:3]
print(np.shape(mean_bgr))
print(mean_bgr[:,:,:3])
print(mean_bgr[0:1,:,0:3])