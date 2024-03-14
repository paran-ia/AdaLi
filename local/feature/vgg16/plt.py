import numpy as np
import matplotlib.pyplot as plt
# 加载6个Numpy文件
data1 = np.load('before_Spike_Conv2d0.npy')
data2 = np.load('after_IF1.npy')
data3 = np.load('after_Spike_Conv2d2.npy')
data4 = np.load('after_Spike_Conv2d3.npy')
data5 = np.load('after_Spike_Conv2d4.npy')
data6 = np.load('after_Spike_Conv2d5.npy')
# 创建2x3的图像布局
plt.figure(figsize=(10, 6))
cmap='viridis'
# 第一个子图
plt.subplot(2, 3, 1)
plt.imshow(data1, cmap=cmap)

# 第二个子图
plt.subplot(2, 3, 2)
plt.imshow(data2, cmap=cmap)

plt.subplot(2, 3, 3)
plt.imshow(data3, cmap=cmap)
plt.subplot(2, 3, 4)
plt.imshow(data4, cmap=cmap)
plt.subplot(2, 3, 5)
plt.imshow(data5, cmap=cmap)
plt.subplot(2, 3, 6)
plt.imshow(data6, cmap=cmap)
plt.show()
