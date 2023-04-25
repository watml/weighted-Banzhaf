import scipy
import numpy as np
import itertools
from scipy import special
from scipy import stats
import torch
from utils import universe

def spearman_rank_corr(x,y):
    x_rank = stats.rankdata(-x,method='min')
    y_rank = stats.rankdata(-y,method='min')
    res = scipy.stats.spearmanr(x_rank,y_rank)
    return res.correlation

def average_spearman_rank_corr(X):
    record_corr = np.zeros(np.int32(special.binom(len(X), 2)))
    j = 0
    for comb in itertools.combinations(X, 2):
        record_corr[j] = spearman_rank_corr(comb[0], comb[1])
        j += 1
    return record_corr


def load_torch_device(use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        universe.device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        universe.device = torch.device("cpu")
        print('Using CPU')
    torch.backends.deterministic = True
    torch.backends.cudnn.benchmark = False

def balance_dataset(target):
    labels, counts = np.unique(target, return_counts=True)
    min_num = np.min(counts)
    num_label = len(labels)

    id_extract = np.zeros((num_label, min_num), dtype=np.int32)
    label_order = np.random.permutation(num_label)
    for i, label in enumerate(label_order):
        idx = np.argwhere(target==label).squeeze()
        select = np.random.choice(len(idx), min_num, replace=False)
        id_extract[i] = idx[select].copy()
    return id_extract


    # labels, counts = np.unique(target, return_counts=True)
    # min_num = np.min(counts)
    # num_label = len(labels)
    # num_feature = data.shape[1]
    #
    #
    # data_extract = np.zeros((num_label, min_num, num_feature))
    # target_extract = np.zeros((num_label, min_num), dtype=np.int32)
    # for i, label in enumerate(np.random.permutation(num_label)):
    #     idx = target == label
    #     data_tmp = data[idx]
    #     target_tmp = target[idx]
    #     idx = np.random.choice(len(target_tmp), min_num, replace=False)
    #     data_extract[i] = data_tmp[idx].copy()
    #     target_extract[i] = target_tmp[idx].copy()
    # del data, target
    # return data_extract, target_extract

def split_dataset(data, target, id_extract, n_valued, n_val, n_test):
    num_class = id_extract.shape[0]
    num_instance = id_extract.shape[1]
    num_feature = data.shape[1]
    assert n_valued + n_val + n_test <= num_class * num_instance
    pointer = np.zeros(num_class, dtype=np.int32)
    X = np.zeros((n_valued, num_feature))
    y = np.zeros(n_valued, dtype=np.int32)

    data_extract = np.zeros((num_class, num_instance, num_feature))
    target_extract = np.zeros((num_class, num_instance), dtype=np.int32)
    for i in range(num_class):
        data_extract[i] = data[id_extract[i]].copy()
        target_extract[i] = target[id_extract[i]].copy()
    del data, target

    for i in range(n_valued):
        cur = i % num_class
        X[i] = data_extract[cur, pointer[cur]].copy()
        y[i] = target_extract[cur, pointer[cur]].copy()
        pointer[cur] += 1
    X_val = np.zeros((n_val, num_feature))
    y_val = np.zeros(n_val, dtype=np.int32)
    for i in range(n_val):
        cur = (i + n_valued) % num_class
        X_val[i] = data_extract[cur, pointer[cur]].copy()
        y_val[i] = target_extract[cur, pointer[cur]].copy()
        pointer[cur] += 1
    X_test = np.zeros((n_test, num_feature))
    y_test = np.zeros(n_test, dtype=np.int32)
    for i in range(n_test):
        cur = (i + n_valued + n_val) % num_class
        X_test[i] = data_extract[cur, pointer[cur]].copy()
        y_test[i] = target_extract[cur, pointer[cur]].copy()
        pointer[cur] += 1

    del data_extract, target_extract
    return (X, y), (X_val, y_val), (X_test, y_test)



