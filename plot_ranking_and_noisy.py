import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils.args import *
from dotmap import DotMap
from scipy import stats
from scipy import special
import itertools

def binplot(collect, _type):
    global args

    x_label = []
    y_data = []
    for key, value in collect.items():
        if key is not None:
            x_label.append(key)
        else:
            x_label.append("Shapley")

        if _type == "noisy":
            y_data.append(np.array(value))
        elif _type == "ranking":
            value = np.stack(value)
            corr = np.zeros(int(special.binom(len(value), 2)), dtype=np.float64)
            for j, pair in enumerate(itertools.combinations(value, 2)):
                res = stats.spearmanr(pair[0], pair[1])
                corr[j] = res.correlation
            corr = corr[~np.isnan(corr)]
            y_data.append(corr)
        else:
            raise NotImplementedError


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

    ylabels = dict(noisy="F1-score", ranking="Spearman's rank correlation coefficient")
    plt.xlabel("semi-value", fontsize=80, labelpad=-120)
    plt.ylabel(ylabels[_type], fontsize=80)
    ax.set_title(f"lr={lr_auto[args.n_valued][args.dataset]}", fontsize=80)

    plt.savefig(os.path.join("fig", f"{_type};dataset={args.dataset};n_valued={args.n_valued}"), bbox_inches='tight')
    plt.close(fig)

    return x_label, y_data


if __name__ == "__main__":
    os.makedirs("fig", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", type=str, default="tmp")
    args_input = parser.parse_args()
    path_exp = os.path.join("exp", args_input.dir)

    for path_top in next(os.walk(path_exp))[1]:
        dataset = path_top.split("=")[1]
        path_top = os.path.join(path_exp, path_top)

        f1_collect = defaultdict(list)
        values_collect = defaultdict(list)
        for path_sub in next(os.walk(path_top))[1]:
            path_sub = os.path.join(path_top, path_sub)
            with open(os.path.join(path_sub, "args.txt"), 'r') as file:
                args = eval(file.readline())

            with np.load(os.path.join(path_sub, "values.npz")) as data:
                values_collect[args.param].append(data["values_traj"][-1])
            with np.load(os.path.join(path_sub, "f1_score.npz")) as data:
                f1_collect[args.param].append(data["f1_scores"])

        labels, f1s = binplot(f1_collect, "noisy")
        _, coefs = binplot(values_collect, "ranking")
        print("{:<14}| noisy label detection | ranking".format(dataset))
        msg = "{:<14}| mean {:.3f}, std {:.3f} | mean {:.3f}, std {:.3f}"
        for label, f1, coef in zip(labels, f1s, coefs):
            print(msg.format(label, np.mean(f1), np.std(f1), np.mean(coef), np.std(coef)))

