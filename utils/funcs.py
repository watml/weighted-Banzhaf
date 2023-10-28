import numpy as np
import torch
import itertools
from dotmap import DotMap
import os

class set_numpy_seed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self.state)


class set_torch_seed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = torch.get_rng_state()
        torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_rng_state(self.state)


def generate_root(args_all, name_top, name_sub):
    assert len(name_top) > 0 and len(name_sub) > 0
    for args in args_all:
        path_top = ""
        path_sub = ""
        for key in name_top:
            path_top += key + "=" + str(args[key]) + ";"
        for key in name_sub:
            path_sub += key + "=" + str(args[key]) + ";"
        path_top = path_top[:-1]
        path_sub = path_sub[:-1]
        args.update(root=os.path.join(path_top, path_sub))


def args_product(args_group):
    args_all = []
    for args_each in args_group:
        for key, value in args_each.items():
            if isinstance(value, np.ndarray):
                args_each[key] = value.tolist()
            elif not isinstance(value, list):
                args_each[key] = [value]

        keys = args_each.keys()
        vals = args_each.values()
        for instance in itertools.product(*vals):
            args_all.append(DotMap(dict(zip(keys, instance))))
    return args_all