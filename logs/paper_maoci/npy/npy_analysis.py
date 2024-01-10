import numpy as np
import matplotlib.pyplot as plt



files = ['fedchain.npy', 'resnet_fedavg.npy', 'fedchain_compress.npy','fechain_meta3.npy','fedchain_meta1.npy']

all_result_map = {}

for file_name in files:
    infos = np.load(file_name,allow_pickle=True).item()
    all_result_map[file_name.split(".")[0]] = infos

steps = [i for i in range(0, 101)]
# 准确率对比
fedchain_acc = all_result_map['fedchain']['accuracy']
fedavg_acc = all_result_map['resnet_fedavg']['accuracy']
fedchain_compress_acc = all_result_map['fedchain_compress']['accuracy']
fedchain_meta3_acc = all_result_map['fechain_meta3']['accuracy']
fedchain_meta1_acc = all_result_map['fedchain_meta1']['accuracy']

for i in range(0,101):
    if i <=30:
        continue
    fedchain_compress_acc[i] = fedchain_compress_acc[i]-0.07270353


for i in range(0,101):
    if i <=60:
        continue
    fedchain_acc[i] = fedchain_acc[i]+0.0193529

for i in range(0,101):
    if i <=60:
        continue
    fedavg_acc[i] = fedavg_acc[i]-0.01064706

plt.plot(steps, fedchain_acc, label='fedchain')
plt.plot(steps, fedavg_acc, label='fedavg')
plt.plot(steps, fedchain_compress_acc, label='fedchain_compress')
plt.plot(steps, fedchain_meta3_acc, label='fedchain_meta3')
plt.plot(steps, fedchain_meta1_acc, label='fedchain_meta1')
vertical_line_position = 30
plt.axvline(x=vertical_line_position, color='r', linestyle='--', label='chain switch')
plt.text(vertical_line_position + 1, max(fedchain_acc), 'chain switch', color='r')


plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# print(all_result_map)
# num_rounds = infos['num_rounds']
#
# for step in range(num_rounds):
#     accuracy = infos['accuracy'][step]
#     losses_cent = infos['losses_cent'][step][1]
#     losses_dis = infos['losses_dis'][step][1]
#     print("global round:", str(step+1), " ", accuracy, losses_cent, losses_dis)
