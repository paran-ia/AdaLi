import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
e_1_layer4_1_relu2 = np.load('Mp_spike/old/e_0_layer4.1.relu2.npz')
e_1_layer1_1_relu2 = np.load('Mp_spike/old/e_0_layer1.1.relu2.npz')
e_100_layer4_1_relu2 = np.load('Mp_spike/old/e_99_layer4.1.relu2.npz')
e_100_layer1_1_relu2 = np.load('Mp_spike/old/e_99_layer1.1.relu2.npz')

e_1_layer1_1_relu2_u=e_1_layer1_1_relu2['u'].flatten().tolist()
e_1_layer4_1_relu2_u=e_1_layer4_1_relu2['u'].flatten().tolist()
e_100_layer1_1_relu2_u=e_100_layer1_1_relu2['u'].flatten().tolist()
e_100_layer4_1_relu2_u=e_100_layer4_1_relu2['u'].flatten().tolist()


u_list = e_1_layer1_1_relu2_u+e_1_layer4_1_relu2_u+e_100_layer1_1_relu2_u+e_100_layer4_1_relu2_u
len1 = int((len(u_list)/2))
len2 = int(len1/2)


data = {
    'Time Point': ['epoch 1'] * len1 + ['epoch 100'] * len1,
    'Object': ['block 1'] *len2 + ['block 4'] * len2 + ['block 1'] * len2 + ['block 4'] * len2,
    'Observation': u_list
}

# 转换为 DataFrame
df = pd.DataFrame(data)
# 使用 Seaborn 绘制分组箱线图
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
palette = sns.color_palette("pastel")

# 绘制分组箱线图
sns.boxplot(x='Time Point', y='Observation', hue='Object', data=df, width=0.5, palette=palette)

# 添加标题和标签
#plt.title('两个时间点的分组箱线图', fontsize=16)
plt.xlabel('epoch', fontsize=27)
plt.ylabel('membrane potential', fontsize=27)

# 显示图例
plt.legend(title='snn neurons', title_fontsize='27', fontsize='27')#,loc='upper right'

# 设置坐标轴标签字体大小
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

plt.show()
