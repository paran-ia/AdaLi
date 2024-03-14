import numpy as np
import matplotlib.pyplot as plt
loaded_data = np.load('Mp_spike/e_1_layer5.2.npz')

# 访问保存的数组
loaded_array1 = loaded_data['u'].flatten()
loaded_array2 = loaded_data['out']
# 计算1的频率
num_ones = np.sum(loaded_array2  == 1)

# 计算总元素个数
total_elements = loaded_array2.size

# 计算1的频率
frequency_of_ones = num_ones / total_elements
# 打印加载的数组
# print("Loaded Array 1:", loaded_array1)
# print("Loaded Array 2:", loaded_array2)
q1 = np.percentile(loaded_array1, 25)
q3 = np.percentile(loaded_array1, 75)
print(q1)
print(q3)
print(frequency_of_ones)
print(loaded_array1.max())
print(loaded_array1.min())
print(loaded_array1.mean())
plt.boxplot(loaded_array1)
plt.show()