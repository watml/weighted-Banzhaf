import numpy as np
import torch
from sklearn.datasets import fetch_openml
from torchvision import datasets, transforms
from utils.funcs import *
from collections import defaultdict

dataset2id = {
    "2dplanes": 727,
    "bank-marketing": 1461,
    "bioresponse": 4134,
    "covertype": 150,
    "cpu": 761,
    "credit": 31,
    "default": 42477,
    "diabetes": 37,
    "fraud": 42175,
    "gas": 1476,
    "har": 1478,
    "iris": 61,
    "letter": 6,
    "optdigits": 28,
    "pendigits": 32,
    "phoneme": 1489,
    "pol": 722,
    "satimage": 182,
    "segment": 36,
    "spambase": 44,
    "texture": 40499,
    "wind": 847,
}


def load_dataset(dataset, *, n_valued=1, n_val=1, dataset_seed=2023, path="dataset"):
    if dataset in ["MNIST", "FMNIST"]:
        return load_MNIST(dataset, n_valued, n_val, dataset_seed, path)
    else:
        return load_OpenML(dataset, n_valued, n_val, dataset_seed, path)


def load_OpenML(dataset, n_valued, n_val, dataset_seed, path, percent_train=0.8):
    global dataset2id
    data, target = fetch_openml(data_id=dataset2id[dataset], data_home=path, return_X_y=True, as_frame=False)
    target_unique = np.unique(target)
    num_class = len(target_unique)
    dict_transform = dict(zip(target_unique, range(num_class)))
    target = np.array([dict_transform[key] for key in target])

    num_total = len(target)
    num_train = int(np.round(num_total * percent_train))
    with set_numpy_seed(dataset_seed):
        pi = np.random.permutation(num_total)
    data_train, label_train = data[pi[:num_train]], target[pi[:num_train]]
    data_mean = np.mean(data_train, axis=0)
    data_std = np.std(data_train, axis=0)
    idx = data_std > 0  # to deal with optdigits dataset
    data_train = np.divide(data_train[:, idx] - data_mean[None, idx],  data_std[None, idx])

    label2pos = defaultdict()
    num_cut = np.inf
    for i in range(num_class):
        pos = np.where(label_train == i)[0]
        num = len(pos)
        label2pos[i] = pos
        if num < num_cut:
            num_cut = num
    assert n_valued + n_val <= num_cut * num_class

    pos_all = np.zeros(num_class, dtype=np.int32)
    pos_valued = np.empty(n_valued, dtype=np.int32)
    pos_val = np.empty(n_val, dtype=np.int32)
    for pos_cur in range(n_valued + n_val):
        pos_class = pos_cur % num_class
        if pos_cur < n_valued:
            pos_valued[pos_cur] = label2pos[pos_class][pos_all[pos_class]]
        else:
            pos_val[pos_cur - n_valued] = label2pos[pos_class][pos_all[pos_class]]
        pos_all[pos_class] += 1
    with set_numpy_seed(dataset_seed):
        np.random.shuffle(pos_valued)
        np.random.shuffle(pos_val)
    X_valued, y_valued = data_train[pos_valued], label_train[pos_valued]
    X_val, y_val = data_train[pos_val], label_train[pos_val]

    return (torch.tensor(X_valued, dtype=torch.float64), torch.tensor(y_valued, dtype=torch.int64)), \
           (torch.tensor(X_val, dtype=torch.float64), torch.tensor(y_val, dtype=torch.int64)), num_class


def load_MNIST(dataset, n_valued, n_val, dataset_seed, path):
    assert n_valued + n_val <= 60000
    if dataset == "FMNIST":
        download_func = datasets.FashionMNIST
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(0.2860, 0.3530)])
    elif dataset == "MNIST":
        download_func = datasets.MNIST
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(0.1307, 0.3081)])
    else:
        raise NotImplementedError(f"Check {dataset}")

    trainset = download_func(path, download=True, train=True, transform=transform)
    with set_torch_seed(dataset_seed):
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=n_valued + n_val, shuffle=True)
        for X, y in trainloader:
            break
    X_valued, X_val = X[:n_valued][:, None, :, :, :], X[n_valued:n_valued + n_val]
    y_valued, y_val = y[:n_valued][:, None], y[n_valued:n_valued + n_val]

    X_valued, X_val = X_valued.type(torch.float64), X_val.type(torch.float64)
    return (X_valued, y_valued), (X_val, y_val), 10


def flip_label(label, num_class, flip_percent=0.2, flip_seed=2023):
    label_numpy = label.numpy()
    label_flipped = label_numpy.copy()
    num_flipped = int(np.ceil(len(label) * flip_percent))
    with set_numpy_seed(flip_seed):
        pos_flipped = np.random.choice(range(len(label)), size=num_flipped, replace=False)
        offset = np.random.choice(range(1, num_class), size=num_flipped, replace=True)
    for i in range(num_flipped):
        y = label_flipped[pos_flipped[i]]
        label_flipped[pos_flipped[i]] = (y + offset[i]) % num_class
    return torch.tensor(np.stack((label_numpy, label_flipped)), dtype=torch.int64)

