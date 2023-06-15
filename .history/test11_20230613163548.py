import numpy as np

mean_bgr = np.random.rand(100, 4, 15)
a = np.array([300, 100, 100])
mean_bgr[:,:,:3] = a[:]
print(np.shape(mean_bgr))
# print(mean_bgr[:,:,:3])
print(mean_bgr[99:100,0:1])