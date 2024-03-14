import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = [1,100,200,300,400,500]
c1 =[70.21,22.81,20.05,20.14,19.82,20.05]#log
c2=[68.77,60.93,56.27,46.07,34.83,19.79]
# x = [1,100,200,300,400,500]
# c1 =[70.21,22.81,20.05,20.14,19.82,20.05]#log
# a1=[28.46,87.73,89.42,91.19,93.03,94.24]
# c2=[68.77,60.93,56.27,46.07,34.83,19.79]
# a2=[27.76,83.35,87.33,90.13,92.12,93.38]
# 绘制折线图
plt.plot(x, c1, label='with log adaptive function',linestyle='-', markeredgewidth=1, linewidth=4,color='#66B2FF', marker='^', markersize=12, markerfacecolor='white')
plt.plot(x, c2, label='with linear adaptive function',linestyle='-', markeredgewidth=1, linewidth=4,color='#FFB266', marker='o', markersize=10, markerfacecolor='white')

# 设置x轴和y轴标签
plt.xlabel('epoch', fontsize=18)
plt.ylabel('computational density%', fontsize=18)
plt.xticks([1,100,200,300,400,500], fontsize=14)  # 设置x轴刻度
plt.yticks(np.arange(0, 102, 10), fontsize=14)  # 设置y轴刻度范围
# 显示网格

plt.ylim((15,75))
plt.grid(True)
# # 调整布局
plt.tight_layout()
plt.legend(prop={'size': 14})
# 显示图形
plt.show()
