import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 指定.npy文件的路径
# file_path = 'results/resnet12_cifar10_32_cle_10_1.0.npy'
file_path = 'results/resnet12_cifar10_32_cle_1_1.0_90avg.npy'
# 使用np.load()读取.npy文件
infos = np.load(file_path, allow_pickle=True).item()

# 创建一个 PyTorch SummaryWriter
# 指定日志存储路径，例如 'logs/'
log_dir = 'logs/'
writer = SummaryWriter(log_dir)

num_rounds = infos['num_rounds']

for step in range(num_rounds):
    # 将数据写入 TensorBoard
    # 注意：PyTorch的SummaryWriter不需要使用`with`语句
    # 它会自动处理资源管理
    # 'step' 可以是你选择的任何表示时间或迭代次数的值
    writer.add_scalar('Accuracy', infos['accuracy'][step], global_step=step+1)
    writer.add_scalar('Losses_Cent', infos['losses_cent'][step][1], global_step=step+1)
    writer.add_scalar('Losses_Dis', infos['losses_dis'][step][1], global_step=step+1)



# 关闭 SummaryWriter
writer.close()
