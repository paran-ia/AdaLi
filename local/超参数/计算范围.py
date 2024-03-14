import numpy as np

# 读取.npy文件
data = np.load('../Mp_spike/e_199_linear.npy')

# 定义范围
lower_bound = 1-1.02
upper_bound = 1+1.02

# 计算在范围内的数所占比例
count_in_range = np.sum((data >= lower_bound) & (data <= upper_bound))
total_count = data.size
percentage = count_in_range / total_count * 100

print(f"The percentage of numbers in the range [{lower_bound}, {upper_bound}] is: {percentage:.2f}%")
