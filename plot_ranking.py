import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from dotmap import DotMap
import argparse
# from utils.vd import *
from scipy import stats
from scipy import special
import itertools
from utils.args import *

parser = argparse.ArgumentParser()
parser.add_argument("-dir", type=str, default="tmp")
args_input = parser.parse_args()


os.makedirs("fig", exist_ok=True)
path_exp = os.path.join("exp", args_input.dir)
fig_format = "ranking;dataset={}.png"

def average_spearman_rank_corr(X):
    record_corr = np.zeros(int(special.binom(len(X), 2)), dtype=np.float64)
    j = 0
    for pair in itertools.combinations(X, 2):
        res = stats.spearmanr(pair[0], pair[1])
        record_corr[j] = res.correlation
        j += 1
    return record_corr

for path_top in next(os.walk(path_exp))[1]:
    dataset = path_top.split("=")[1]
    path_top = os.path.join(path_exp, path_top)
    values_collect = dict(weighted_banzhaf=defaultdict(list), beta_shapley=defaultdict(list), shapley=defaultdict(list))
    for path_sub in next(os.walk(path_top))[1]:
        path_sub = os.path.join(path_top, path_sub)
        file_saved = os.path.join(path_sub, "values.npz")
        if not os.path.exists(file_saved):
            continue
        with open(os.path.join(path_sub, "args.txt"), 'r') as file:
            args = eval(file.readline())
        values_each = values_collect[args.value]
        with np.load(file_saved) as data:
            values_each[args.param].append(data["values_traj"][-1])

    x_label = []
    y_data = []
    for key, value in values_collect["weighted_banzhaf"].items():
        if len(value):
            x_label.append('{:.2f}'.format(round(key, 2)))
            value = np.stack(value)
            corr = average_spearman_rank_corr(value)
            corr = corr[~np.isnan(corr)]
            y_data.append(corr)
    for key, value in values_collect["beta_shapley"].items():
        if len(value):
            x_label.append(key)
            value = np.stack(value)
            corr = average_spearman_rank_corr(value)
            y_data.append(corr)

    for key, value in values_collect["shapley"].items():
        if len(value):
            x_label.append("Shapley")
            value = np.stack(value)
            corr = average_spearman_rank_corr(value)
            y_data.append(corr)

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
        plt.ylabel("Spearman's rank correlation coefficient", fontsize=80)
        ax.set_title(f"lr={lr_auto[args.n_valued][args.dataset]}", fontsize=80)

        plt.savefig(os.path.join("fig", fig_format.format(dataset)), bbox_inches='tight')
        plt.close(fig)


