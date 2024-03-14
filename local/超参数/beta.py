import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0.3, 0.7, 5)
y = [93.02,93.52,94.65,94.05,94.13]

# 绘制折线图
plt.plot(x, y, linestyle='-', markeredgewidth=1, linewidth=4,color='#66B2FF', marker='^', markersize=14, markerfacecolor='orange')

# 设置x轴和y轴标签
plt.xlabel('hyperparameter $\\beta$', fontsize=18)
plt.ylabel('accuracy%', fontsize=18)
plt.xticks(np.arange(0.3, 0.8, 0.1), fontsize=14)  # 设置x轴刻度
plt.yticks(np.arange(0, 102, 5), fontsize=14)  # 设置y轴刻度范围
# 显示网格

plt.ylim((90,100))
plt.grid(True)
# # 调整布局
plt.tight_layout()

# 显示图形
plt.show()
