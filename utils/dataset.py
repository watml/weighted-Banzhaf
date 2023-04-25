import numpy as np
from dotmap import DotMap
from sklearn.datasets import fetch_covtype
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import fetch_openml
from torchvision import datasets, transforms
from utils.aids import *

def load_dataset(dataset, *, n_valued=200, n_val=200, n_test=1000, dataset_seed=2023, flip_percent=None, flip_seed=2023,
                 dataset_path="dataset", display=True):
    np.random.seed(dataset_seed)
    if dataset == "covertype":
        """
        number of each type:
        1:211840, 2:283301, 3:35754, 4:2747, 5:9493, 6:17367, 7:20510
        num of features: 54
        acc: 31.07, 53.49
        """
        data, target = fetch_covtype(data_home=dataset_path, return_X_y=True, shuffle=False)
        target -= 1
    elif dataset == "cpu":
        """
        sum(target=="P") = 2477
        sum(target=="N") = 5715
        num_feature = 21
        acc: 86.13, 88.21
        """
        data, target = fetch_openml(data_id=761, data_home=dataset_path, return_X_y=True, as_frame=False)
        target = np.int32(target=="P")
    elif dataset == "2dplanes":
        """
        sum(target=="P") = 20348
        sum(target=="N") = 20420
        num_feature = 10
        acc: 72.80, 81.28
        """
        data, target = fetch_openml(data_id=727, data_home=dataset_path, return_X_y=True, as_frame=False)
        target = np.int32(target=="P")
    elif dataset == "pol":
        """
        sum(target=="P") = 9959
        sum(target=="N") = 5041
        num_feature = 48
        acc: 69.60, 82.64
        acc: 72.13, 78.93, lr=3.5, r=0.0
        """
        data, target = fetch_openml(data_id=722, data_home=dataset_path, return_X_y=True, as_frame=False)
        target = np.int32(target=="P")
    elif dataset == "wind":
        """
        sum(target=="P") = 3501
        sum(target=="N") = 3073
        num_feature = 14
        acc: 87.33, 81.83
        """
        data, target = fetch_openml(data_id=847, data_home=dataset_path, return_X_y=True, as_frame=False)
        target = np.int32(target=="P")
    elif dataset == "phoneme":
        """
        sum(target=="1") = 3818
        sum(target=="2") = 1586
        num_feature = 5
        acc: 67.33, 74.48
        """
        data, target = fetch_openml(data_id=1489, data_home=dataset_path, return_X_y=True, as_frame=False)
        target = np.int32(target=="2")
    # elif dataset == "click":
    #     """
    #     sum(target=="0") = 1664406
    #     sum(target=="1") = 333004
    #     num_feature = 11
    #     acc = ?, 0.5959
    #     """
    #     data, target = fetch_openml(data_id=1218, data_home=dataset_path, return_X_y=True, as_frame=False)
    #     target = np.int32(target=="1")
    # elif dataset == "vehicle":
    #     """
    #     sum(target=="1") = 49264
    #     sum(target=="-1") = 49264
    #     num_feature = 100
    #     acc: 54,27, 79.36
    #     """
    #     data, target = fetch_openml(data_id=357, data_home=dataset_path, return_X_y=True, as_frame=False)
    #     target = np.int32(target=="1")
    #     data = data.toarray()
    elif dataset == "fraud":
        """
        sum(target==0) = 284315
        sum(target==1) = 492
        num_feature = 30
        acc: 74.13, 88.63
        """
        data, target = fetch_openml(data_id=42175, data_home=dataset_path, return_X_y=True, as_frame=False)
    # elif dataset == "creditcard":
    #     """
    #     sum(target=="1") = 6636
    #     sum(target=="0") = 23364
    #     num_feature = 23
    #     acc: 49.73, 62.24
    #     """
    #     data, target = fetch_openml(data_id=42477, data_home=dataset_path, return_X_y=True, as_frame=False)
    #     target = np.int32(target=="1")
    elif dataset == "fmnist":
        trainset = datasets.FashionMNIST(dataset_path, download=True, train=True)
        testset = datasets.FashionMNIST(dataset_path, download=True, train=False)
    else:
        raise NotImplementedError(f"Check {dataset}")

    if dataset == "fmnist":
        # parameters of Normalize are got using trainset
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(0.2860, 0.1246)
                                        ])

    else:
        id_extract = balance_dataset(target)
        (X, y), (X_val, y_val), (X_test, y_test) = split_dataset(data, target, id_extract, n_valued, n_val, n_test)
        np.random.seed(None)


        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    if flip_percent is not None:
        labels = np.unique(y)
        num_label = len(labels)
        assert 0 < flip_percent and flip_percent < 1
        num_flip = np.int32(np.ceil(n_valued * flip_percent))
        np.random.seed(flip_seed)
        idx_list = np.random.choice(range(n_valued), size=num_flip, replace=False)
        y_flipped = y.copy()
        for idx in idx_list:
            label = y_flipped[idx]
            offset = np.random.choice(range(1, num_label), size=1)[0]
            y_flipped[idx] = (label + offset) % num_label
        np.random.seed(None)
        y = np.concatenate((y[np.newaxis,:], y_flipped[np.newaxis,:]), axis=0)



    if display:
        info = DotMap()
        if flip_percent is not None:
            unique, counts = np.unique(y[0], return_counts=True)
        else:
            unique, counts = np.unique(y, return_counts=True)
        info.train = dict(zip(unique, counts))
        unique, counts = np.unique(y_val, return_counts=True)
        info.validation = dict(zip(unique, counts))
        unique, counts = np.unique(y_test, return_counts=True)
        info.test = dict(zip(unique, counts))
        print(info)


    return (torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long)), \
           (torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.long)), \
           (torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))