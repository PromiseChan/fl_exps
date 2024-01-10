import numpy as np
import matplotlib.pyplot as plt



files = ['fedchain.npy', 'resnet_fedavg.npy', 'fedchain_compress.npy','fechain_meta3.npy','fedchain_meta1.npy']

all_result_map = {}

for file_name in files:
    infos = np.load(file_name,allow_pickle=True).item()
    all_result_map[file_name.split(".")[0]] = infos

steps = [i for i in range(0, 101)]

def get_loss_array(loss_dict):
    loss_array = []
    for i in range(0,101):
        loss_array.append(loss_dict[i][1])
    return loss_array

# 准确率对比
fedchain_loss = get_loss_array(all_result_map['fedchain']['losses_cent'])
fedavg_loss = get_loss_array(all_result_map['resnet_fedavg']['losses_cent'])
fedchain_compress_loss = get_loss_array(all_result_map['fedchain_compress']['losses_cent'])
fechain_meta3_loss = get_loss_array(all_result_map['fechain_meta3']['losses_cent'])
fechain_meta1_loss = get_loss_array(all_result_map['fedchain_meta1']['losses_cent'])


window_size = 2
fedavg_loss = np.convolve(fedavg_loss, np.ones(window_size)/window_size, mode='valid')


plt.plot(steps, fedchain_loss, label='fedchain')
plt.plot(steps[0:100], fedavg_loss, label='fedavg')
plt.plot(steps, fedchain_compress_loss, label='fedchain_compress')
plt.plot(steps, fechain_meta3_loss, label='fechain_meta3')
plt.plot(steps, fechain_meta1_loss, label='fechain_meta1')
vertical_line_position = 30
plt.axvline(x=vertical_line_position, color='r', linestyle='--', label='chain switch')
plt.text(vertical_line_position + 1, max(fedchain_compress_loss), 'chain switch', color='r')


plt.xlabel('Steps')
plt.ylabel('loss')
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
