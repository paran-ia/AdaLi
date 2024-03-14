import matplotlib.pyplot as plt
import numpy as np
vanilla_03='acc_plot/CIFAR10_resnet18_LIF_T1_vanilla_wl0.3_wr0.3_b128_e500.npy'
vanilla_15='acc_plot/CIFAR10_resnet18_LIF_T1_vanilla_wl1.5_wr1.5_b128_e500.npy'
log_15='acc_plot/CIFAR10_resnet18_LIF_T1_log_wl1.5_wr1.5_b128_e500.npy'
linear_15='acc_plot/CIFAR10_resnet18_LIF_T1_linear_wl1.5_wr1.5_b128_e500.npy'
#93.83(linear)
#94.66(log)
#94.22(sqrt)
#91.91(0.25)
#90.97(1.5)
# Data for plotting
s1 = np.load(vanilla_03)
s2 = np.load(vanilla_15)
s3 = np.load(log_15)
s4 = np.load(linear_15)
t = np.arange(0,500, 1)
print(s1[-1])
print(s2[-1])
print(s3[-1])
print(s4[-1])
plt.figure(figsize=(8, 6))

# 画第一个折线图
plt.plot(t , s1, label='vanilla,bound=0.3')
plt.plot(t , s2, label='vanilla,bound=1.5')
plt.plot(t , s3, label='whith log adaptive function')
plt.plot(t , s4, label='whith linear adaptive function')


# 添加标题和标签
plt.xlabel('epoch',size='18')
plt.ylabel('accuracy%',size='18')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# 添加图例
plt.legend(prop={'size': 16})

# 显示图形
plt.show()
