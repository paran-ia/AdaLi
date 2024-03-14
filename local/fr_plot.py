import matplotlib.pyplot as plt
import numpy as np

e_1_layer1_1_relu2 = np.load('Mp_spike/old/e_0_layer1.1.relu2.npz')
e_1_layer2_1_relu2 = np.load('Mp_spike/old/e_0_layer2.1.relu2.npz')
e_1_layer3_1_relu2 = np.load('Mp_spike/old/e_0_layer3.1.relu2.npz')
e_1_layer4_1_relu2 = np.load('Mp_spike/old/e_0_layer4.1.relu2.npz')

e_100_layer1_1_relu2 = np.load('Mp_spike/old/e_99_layer1.1.relu2.npz')
e_100_layer2_1_relu2 = np.load('Mp_spike/old/e_99_layer2.1.relu2.npz')
e_100_layer3_1_relu2 = np.load('Mp_spike/old/e_99_layer3.1.relu2.npz')
e_100_layer4_1_relu2 = np.load('Mp_spike/old/e_99_layer4.1.relu2.npz')

e_1_layer1_s=e_1_layer1_1_relu2['out'].flatten()
e_1_layer2_s=e_1_layer2_1_relu2['out'].flatten()
e_1_layer3_s=e_1_layer3_1_relu2['out'].flatten()
e_1_layer4_s=e_1_layer4_1_relu2['out'].flatten()

e_100_layer1_s=e_100_layer1_1_relu2['out'].flatten()
e_100_layer2_s=e_100_layer2_1_relu2['out'].flatten()
e_100_layer3_s=e_100_layer3_1_relu2['out'].flatten()
e_100_layer4_s=e_100_layer4_1_relu2['out'].flatten()

spikes=[]
total_num=[]
spikes.append(np.sum(e_1_layer1_s  == 1))
total_num.append(e_1_layer1_s.size)

spikes.append(np.sum(e_1_layer2_s  == 1))
total_num.append(e_1_layer2_s.size)

spikes.append(np.sum(e_1_layer3_s  == 1))
total_num.append(e_1_layer3_s.size)

spikes.append(np.sum(e_1_layer4_s  == 1))
total_num.append(e_1_layer4_s.size)

spikes.append(np.sum(e_100_layer1_s  == 1))
total_num.append(e_100_layer1_s.size)

spikes.append(np.sum(e_100_layer2_s  == 1))
total_num.append(e_100_layer2_s.size)

spikes.append(np.sum(e_100_layer3_s  == 1))
total_num.append(e_100_layer3_s.size)

spikes.append(np.sum(e_100_layer4_s  == 1))
total_num.append(e_100_layer4_s.size)

fr = np.divide(spikes, total_num)

fr = np.round(fr,decimals=2)
species = ("block 1", "block 2", "block 3", "block 4")
penguin_means = {
    'epoch 1': fr[0:4],
    'epoch 100': fr[4:],

}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('firing rate',size='18')
ax.set_xlabel('blocks',size='18')
#ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, species)
ax.legend(loc='upper right', ncols=1,fontsize='16')
ax.set_ylim(0, 1)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
plt.show()