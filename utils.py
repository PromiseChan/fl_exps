import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable, Optional, Tuple, List
import numpy as np
from collections import OrderedDict
from prettytable import PrettyTable
import brevitas.nn as qnn
from args import args


def pile_str(line,item):
    return "_".join([line,item])

def get_tensor_parameters(model):
    from flwr.common.parameter import ndarrays_to_parameters
    parameters = []
    model.global_layer_names.sort()
    for i in range(len(model.global_layer_names)):
        name = model.global_layer_names[i]
        parameters.append(model.state_dict()[name].cpu().numpy())
    return ndarrays_to_parameters(parameters)

# 本地训练完成后，获取最新的参数，发送给服务器
def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    parameters = []
    model.global_layer_names.sort()
    for i in range(len(model.global_layer_names)):
        name = model.global_layer_names[i]
        parameters.append(model.state_dict()[name].cpu().numpy())
    return parameters

# 接收来自服务器的参数
def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    model.global_layer_names.sort()
    params_dict = zip(model.global_layer_names, params)
    linear_layer = nn.Linear(10 * args.feature_maps, 10)


    parameter = []
    for name, param in linear_layer.state_dict().items():
        parameter.append(param.cpu().numpy())

    local_dict = zip(model.local_layer_names,parameter)

    tmp_map = {}
    for k, v in local_dict:
        tmp_map[k] = torch.from_numpy(np.copy(v))
    for k, v in params_dict:
        tmp_map[k] = torch.from_numpy(np.copy(v))


    state_dict = OrderedDict(tmp_map)

    model.load_state_dict(state_dict, strict=True)


def tell_history(hist,file_name : str, infos = None,path : str = "",head=None):

    _, accuracy = zip(*hist.metrics_centralized["accuracy"])
    losses_cent = hist.losses_centralized
    losses_dis = hist.losses_distributed
    accuracy = np.asarray(accuracy)

    infos["accuracy"] = accuracy
    infos["losses_cent"] = losses_cent
    infos["losses_dis"] = losses_dis
    chain_epochs = args.chain_epochs
    with open(path+file_name+"_"+str(chain_epochs)+"avg"+".npy","wb") as f:
        np.save(f,infos)

# borrowed from Pytorch quickstart example
def train(net, trainloader, epochs, device: str,
          lr : float = 0.01, momentum : float = 0.9):

    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    optimizer_local = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()

    # todo fedrecon 恢复 私有层参数，只训练 local_layer层
    for name,val in net.state_dict().items():
        if name not in net.local_layer_names:
            val.requires_grad = False

    for _ in range(3):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer_local.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer_local.step()

    # 训练全部参数
    for name, val in net.state_dict().items():
        val.requires_grad = True
    # 开始训练
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if args.bnn:
                net.binarization()
            loss = criterion(net(images), labels)
            loss.backward()
            
            if args.bnn:
                net.restore()

            optimizer.step()

            if args.bnn:
                net.clip()            

# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        if args.bnn:
            net.binarization()
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    loss = loss / len(testloader)
    return loss, accuracy

def total_sum_params(iter):
    return sum(
        param.numel() for param in iter.parameters()
    )

def get_model_size(model,wbits):

    total_params = total_sum_params(model)
    quant_params = 0

    for layer in model.modules() :
        if(isinstance(layer,qnn.QuantConv2d) or isinstance(layer,qnn.QuantLinear)):
            quant_params += total_sum_params(layer)

    fp_params = total_params-quant_params

    quant_size = quant_params*wbits
    model_size = quant_size + fp_params*32
    
    # no format (kiB or MiB)
    return model_size,total_params,quant_size,quant_params 

def pretty_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def prune_threshold(params):

    pruning_rate = args.prate
    sorted = torch.cat([torch.from_numpy(i).flatten().abs() for i in params]).sort()[0]
    threshold = sorted[int(len(sorted)*pruning_rate)]

    for i,p in enumerate(params):
        params[i][np.abs(p)<threshold.item()] = 0

    return params

# 计算每一层的稀疏度
# 该函数的作用是计算神经网络每一层参数的稀疏性，返回一个列表，
# 其中每个元素表示对应层参数的非零元素比例
def layer_sparsity(params):
    num_zeros = list(map(np.count_nonzero,params))
    total_per_layer = list(map(np.size,params))
    return list(map(np.divide,num_zeros,total_per_layer))
