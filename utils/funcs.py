import numpy as np
import torch
import itertools
from dotmap import DotMap
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_curves(x, ys, fig_saved, labels=None, x_label=None, y_label=None, title=None, axis=0, plot_std=True):
    # sns.set_theme()
    fig, ax = plt.subplots(figsize=(32, 24))
    plt.grid()
    clrs = sns.color_palette(n_colors=len(ys))
    if labels is None:
        enum = zip(ys)
    else:
        enum = zip(ys, labels)
    for i, take in enumerate(enum):
        if len(take) == 2:
            y, label = take
        else:
            y = take[0]
            label = None
        mean = np.mean(y, axis=axis)
        std = np.std(y, axis=axis)
        num_remain = len(x) - len(mean)
        if num_remain > 0:
            mean = np.pad(mean, (0, num_remain), constant_values=np.nan)
            std = np.pad(std, (0, num_remain), constant_values=np.nan)
        ax.plot(x, mean, label=label, c=clrs[i], linewidth=10)
        if plot_std:
            ax.fill_between(x, mean-std, mean+std, alpha=0.3, facecolor=clrs[i])

    ax.tick_params(axis='x', labelsize=70)
    ax.tick_params(axis='y', labelsize=70)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=80)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=80)
    if title is not None:
        ax.set_title(title, fontsize=80)
    plt.legend(fontsize=80)
    plt.savefig(fig_saved, bbox_inches='tight')
    plt.close(fig)
