from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil
from PIL import Image
from torchvision.datasets import VisionDataset
from typing import Callable, Optional, Tuple, Any
from common import create_lda_partitions
import os
from args import args


dict_tranforms = {  
    "cifar10"           : transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
    "emnist"            : transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))]), 
    "cifar100"          : transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]), }

dict_tranforms_test = {
    "cifar10"           : transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
    "emnist"            : transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))]),
    "cifar100"          : transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]), }

def get_dataset(path_to_data, cid, partition, transform ):
    # generate path to cid's data
    path_to_data = path_to_data / cid / (partition + ".pt")

    return TorchVision_FL(path_to_data, transform=transform)

def get_dataloader(
    path_to_data, cid, is_train, batch_size, workers,transform
):
    """Generates trainset/valset object and returns appropiate dataloader."""

    partition = "train" if is_train else "val"
    dataset = get_dataset(Path(path_to_data), cid, partition,transform)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


def get_random_id_splits(total, val_ratio, shuffle = True):
    """splits a list of length `total` into two following a
    (1-val_ratio):val_ratio partitioning.

    By default the indices are shuffled before creating the split and
    returning.
    """

    if isinstance(total, int):
        indices = list(range(total))
    else:
        indices = total

    split = int(np.floor(val_ratio * len(indices)))
    # print(f"Users left out for validation (ratio={val_ratio}) = {split} ")
    if shuffle:
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def do_fl_partitioning(path_to_dataset, pool_size, alpha, num_classes, val_ratio=0.0):

    images, labels = torch.load(path_to_dataset,encoding='latin1')
    idx = np.array(range(len(images)))
    dataset = [idx, labels]
    partitions, _ = create_lda_partitions(
        dataset, num_partitions=pool_size, concentration=alpha, accept_imbalanced=True
    )

    # Show label distribution for first partition (purely informative)
    partition_zero = partitions[0][1]
    hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
    print(
        f"Class histogram for 0-th partition (alpha={alpha}, {num_classes} classes): {hist}"
    )

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    splits_dir = path_to_dataset.parent / "federated"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    Path.mkdir(splits_dir, parents=True)

    for p in range(pool_size):

        labels = partitions[p][1]
        image_idx = partitions[p][0]
        imgs = images[image_idx]

        # create dir
        Path.mkdir(splits_dir / str(p))

        if val_ratio > 0.0:
            # split data according to val_ratio
            train_idx, val_idx = get_random_id_splits(len(labels), val_ratio)
            val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            with open(splits_dir / str(p) / "val.pt", "wb") as f:
                torch.save([val_imgs, val_labels], f)

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / str(p) / "train.pt", "wb") as f:
            torch.save([imgs, labels], f)

    return splits_dir

class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data=None,
        data=None,
        targets=None,
        transform= None,
    ):
        path = path_to_data.parent if path_to_data else None
        super(TorchVision_FL, self).__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()

            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_cifar_10(path_to_data="./demos/CIFAR10"):
    """Downloads CIFAR10 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    data_loc = Path(path_to_data) / "cifar-10-batches-py"
    training_data = data_loc / "training.pt"
    if(args.skip_gen_training):
        input_shape = [1,3,32,32]
        print("Re-using previous generated CIFAR-10 dataset")
    else:
        # download dataset and load train set
        train_set = datasets.CIFAR10(root=path_to_data, train=True, download=True)

        input_shape = torch.stack(list(map(transforms.ToTensor(), train_set.data))).shape

        print("Generating unified CIFAR-10 dataset")
        # fuse all data splits into a single "training.pt"
        torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = datasets.CIFAR10(
        root=path_to_data, train=False, transform=dict_tranforms_test["cifar10"]
    )

    # returns path where training data is and testset
    return training_data, test_set, max(test_set.targets) + 1,input_shape[1:]

def get_cifar_100(path_to_data="./data"):
    """Downloads CIFAR100 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    data_loc = Path(path_to_data) / "cifar-100-python"
    training_data = data_loc / "training.pt"
    if(args.skip_gen_training):
        input_shape = [1,3,32,32]
        print("Re-using previous generated CIFAR-100 dataset")
    else:
        # fuse all data splits into a single "training.pt"
        print("Generating unified CIFAR-100 dataset")
        train_set = datasets.CIFAR100(root=path_to_data, train=True, download=True)
        torch.save([train_set.data, np.array(train_set.targets)], training_data)
        input_shape = torch.stack(list(map(transforms.ToTensor(), train_set.data))).shape
    test_set = datasets.CIFAR100(root=path_to_data, train=False, download=True, transform=dict_tranforms_test["cifar100"])

    # returns path where training data is and testset
    return training_data, test_set, max(test_set.targets) + 1,input_shape[1:]

def get_EMNIST(path_to_data="./data"):
    """Downloads EMNIST dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = datasets.EMNIST(root=path_to_data, split='byclass',train=True, download=True)

    input_shape = train_set.data.unsqueeze(1).shape[1:]

    mnist_path = Path(path_to_data) / "EMNIST/processed"

    if(not os.path.exists(mnist_path)):
        os.mkdir(mnist_path)

    # fuse all data splits into a single "training.pt"
    training_data = Path(path_to_data) / "EMNIST/processed/training.pt"
    print("Generating unified EMNIST dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = datasets.EMNIST(root=path_to_data, split='byclass',train=False,transform=dict_tranforms["emnist"])

    # returns path where training data is and testset
    return training_data, test_set, max(train_set.targets) + 1,input_shape
