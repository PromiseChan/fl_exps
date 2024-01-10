import random
from pathlib import Path

import flwr as fl
import torch

from dataset_utils import get_cifar_10, do_fl_partitioning,get_cifar_100, get_dataloader,dict_tranforms
from utils import set_params,test,tell_history,pile_str,get_tensor_parameters
import multiprocess as mp
mp.set_start_method('spawn',force=True)
from args import args
from datetime import datetime
import time
from binaryconnect import BC

#torch.backends.cudnn.benchmark = True

if(args.model == "resnet18"):
    from models.resnets import ResNet18
    resnet_model = ResNet18
elif(args.model == "resnet20"):
    from models.resnets import ResNet20
    resnet_model = ResNet20
elif(args.model == "resnet12"):
    from models.resnet12 import ResNet12
    resnet_model = ResNet12
elif(args.model == "resnet8"):
    from models.resnets import ResNet8
    resnet_model = ResNet8
elif(args.model == "qresnet12"):
    from models.resnet12_brev import QResNet12
    resnet_model = QResNet12
elif(args.model == "qresnet8"):
    from models.qresnets import QResNet8
    resnet_model = QResNet8

def fit_config(server_round):
    """Return a configuration with static batch size and (local) epochs."""
    print("全局round: " + str(server_round))
    print("chain epoch: " + str(args.chain_epochs))
    if int(server_round) <= args.chain_epochs:
        config = {
            "epochs": 3,
            "batch_size": args.cl_bs,
            "cl_lr": args.cl_lr,
            "cl_momentum": args.cl_momentum,
            "fedrecon_epoch": 1
        }
    elif int(server_round) <= args.chain_epochs + 13:
        config = {
            "epochs": 1,
            "batch_size": args.cl_bs,
            "cl_lr": args.cl_lr,
            "cl_momentum": args.cl_momentum,
            "fedrecon_epoch": 0
        }
    else:
        config = {
            "epochs": args.cl_epochs,  # number of local epochs
            "batch_size": args.cl_bs,
            "cl_lr": args.cl_lr,
            "cl_momentum": args.cl_momentum,
            "fedrecon_epoch": 0
        }
    return config

def get_evaluate_fn( testset,dataset_info) :
    """Return an evaluation function for centralized evaluation."""
    def evaluate(server_round, parameters, config) :
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        if(args.only_cpu):
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = resnet_model(args.feature_maps, dataset_info["input_shape"], dataset_info["num_classes"],batchn=False)
        if(args.bnn):
            model = BC(model)

        #model = Net()
        train_path = Path('./demos/CIFAR10') / "cifar-10-batches-py"/"training.pt"

        set_params(model, parameters)
        model.to(device)

        fed_dir = do_fl_partitioning(
            train_path, pool_size=args.num_clients, alpha=1, num_classes=10, val_ratio=args.val_ratio
        )

        cid = str(random.randint(0, 4))
        trainloader = get_dataloader(
            fed_dir,
            cid,
            is_train=True,
            batch_size=args.cl_bs,
            workers=2,
            transform = dict_tranforms['cifar10']
        )

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)

        # todo fedrecon 恢复 私有层参数，只训练 local_layer层
        optimizer_local = torch.optim.SGD(model.parameters(), lr=args.cl_lr, momentum=args.cl_momentum)
        criterion = torch.nn.CrossEntropyLoss()

        for name, val in model.state_dict().items():
            if name not in model.local_layer_names:
                val.requires_grad = False

        for _ in range(1):
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer_local.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer_local.step()
        # 训练全部参数
        for name, val in model.state_dict().items():
            val.requires_grad = True

        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate

def start_server(srv_addr,strategy,num_rounds,server_queue):
    """Start the server."""

    server_queue.put(
        fl.server.start_server(
                                server_address=srv_addr ,
                                config=fl.server.ServerConfig(num_rounds=num_rounds),
                                strategy=strategy))

def start_client(model, dataset_info, saddr,cid,fed_dir,features_maps,only_cpu):
    from client import FlowerClient

    client = FlowerClient(model,dataset_info,saddr,cid,fed_dir,features_maps,only_cpu)

    client.start_client()


# Start simulation (a _default server_ will be created)
if __name__ == "__main__":

    saddr = "127.0.0.1:8080"
    processes = []
    pool_size = args.num_clients  # number of dataset partions (= number of total clients)
    file_name = args.model
    file_name=pile_str(file_name,args.dataset)
    file_name=pile_str(file_name,str(args.wbits))
    file_name=pile_str(file_name,"cle_"+str(args.cl_epochs))
    if(args.prune):
        file_name=pile_str(file_name,"prune")
        file_name=pile_str(file_name,str(args.prate))

    infos_dict = vars(args)

    start = time.time()
    log = open("log.txt","a")
    now = datetime.now().strftime("%H:%M")
    log.write(f"Starting Exp : {file_name} at {now}\n")
    log.flush()

    # Download dataset
    if(args.dataset == "cifar10"):
        train_path, testset,num_classes,input_shape = get_cifar_10()
    elif(args.dataset == "cifar100"):
        train_path, testset,num_classes,input_shape = get_cifar_100()
    else:
        print("Wrong dataset name")
        exit(-1)

    if(args.alpha_inf):
        alpha = float('inf')
        file_name=pile_str(file_name,"uniform")
    else:
        alpha = args.alpha
        file_name=pile_str(file_name,str(alpha))

    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=alpha, num_classes=num_classes, val_ratio=args.val_ratio
    )

    model = resnet_model(args.feature_maps,input_shape,num_classes)
    if(args.bnn):
        model = BC(model)
        model.binarization()

    initial_weights = model
    dataset_info = {"name" : args.dataset,"input_shape" : input_shape, "num_classes":num_classes}
    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.samp_rate,
        fraction_evaluate=args.samp_rate,
        min_fit_clients=int(pool_size*args.samp_rate),
        min_evaluate_clients=int(pool_size*args.samp_rate),
        min_available_clients=pool_size,  # All clients should be available
        initial_parameters =get_tensor_parameters(initial_weights),
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset,dataset_info),  # centralised evaluation of global model
    )
    # Start the server
    server_queue = mp.Queue()
    server_process = mp.Process(
            target=start_server,
            args=(saddr,strategy,args.num_rounds,server_queue)
        )

    server_process.start()

    time.sleep(2)

    processes.append(server_process)

    for cid in range(pool_size):
        client_process = mp.Process(target=start_client,
                                    args=(resnet_model,
                                          dataset_info,
                                          saddr,
                                          str(cid),
                                          fed_dir,
                                          args.feature_maps,
                                          args.only_cpu))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()

    hist = server_queue.get()

    tell_history(hist,file_name,infos=infos_dict,path='results/')

    end = time.time()
    now = datetime.now().strftime("%H:%M")
    log.write(f"Finishing Exp at {now} - Elapsed time {(end-start)/60:.2f} mins\n")
    log.write("\n")
    log.close()
