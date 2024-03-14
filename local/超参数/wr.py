import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0.5, 2.5, 5)
y = [94.04,93.8,94.65,93.93,94.05]

# 绘制折线图
plt.plot(x, y, linestyle='-', markeredgewidth=1, linewidth=4,color='#66B2FF', marker='^', markersize=14, markerfacecolor='orange')

# 设置x轴和y轴标签
plt.xlabel('hyperparameter $V^1_{br}$', fontsize=18)
plt.ylabel('accuracy%', fontsize=18)
plt.xticks(np.arange(0.5, 2.6, 0.5), fontsize=14)  # 设置x轴刻度
plt.yticks(np.arange(0, 102, 5), fontsize=14)  # 设置y轴刻度范围
# 显示网格

plt.ylim((90,100))
plt.grid(True)
# # 调整布局
plt.tight_layout()

# 显示图形
plt.show()
