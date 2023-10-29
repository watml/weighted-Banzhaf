import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils.args import *
from dotmap import DotMap

parser = argparse.ArgumentParser()
parser.add_argument("-dir", type=str, default="tmp")
args_input = parser.parse_args()


os.makedirs("fig", exist_ok=True)
path_exp = os.path.join("exp", args_input.dir)
fig_format = "noisy;dataset={}.png"


for path_top in next(os.walk(path_exp))[1]:
    dataset = path_top.split("=")[1]
    path_top = os.path.join(path_exp, path_top)
    f1_collect = dict(weighted_banzhaf=defaultdict(list), beta_shapley=defaultdict(list), shapley=defaultdict(list))
    for path_sub in next(os.walk(path_top))[1]:
        path_sub = os.path.join(path_top, path_sub)
        file_saved = os.path.join(path_sub, "f1_score.npz")
        if not os.path.exists(file_saved):
            continue
        with open(os.path.join(path_sub, "args.txt"), 'r') as file:
            args = eval(file.readline())
        f1_each = f1_collect[args.value]
        with np.load(file_saved) as data:
            f1_each[args.param].append(data["f1_scores"])

    x_label = []
    y_data = []
    for key, value in f1_collect["weighted_banzhaf"].items():
        if len(value):
            x_label.append('{:.2f}'.format(round(key, 2)))
            y_data.append(np.array(value))
    for key, value in f1_collect["beta_shapley"].items():
        if len(value):
            x_label.append(key)
            y_data.append(np.array(value))

    for key, value in f1_collect["shapley"].items():
        if len(value):
            x_label.append("Shapley")
            y_data.append(np.array(value))

    fig, ax = plt.subplots(figsize=(32, 28))
    plt.grid()

    sns.stripplot(data=y_data, ax=ax, linewidth=.3, size=10, edgecolor="black")
    sns.barplot(data=y_data, ax=ax, errorbar='sd', capsize=.5)

    ax.set_xticklabels(x_label)
    ticks = ax.get_xticklabels()
    for i in range(len(x_label)):
        ticks[i].set_rotation(-90)
    ax.tick_params(axis='x', labelsize=70)
    ax.tick_params(axis='y', labelsize=70)

    plt.xlabel("semi-value", fontsize=80, labelpad=-120)
    plt.ylabel("F1-score", fontsize=80)
    ax.set_title(f"lr={lr_auto[args.n_valued][args.dataset]}", fontsize=80)

    plt.savefig(os.path.join("fig", fig_format.format(dataset)), bbox_inches='tight')
    plt.close(fig)

