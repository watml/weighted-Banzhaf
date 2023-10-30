import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from dotmap import DotMap
from scipy import stats
from scipy import special
import itertools
import argparse

def binplot(x, y, _type):
    global dataset
    fig, ax = plt.subplots(figsize=(32, 28))
    plt.grid()

    if _type == "ranking":
        sns.stripplot(data=y, ax=ax, linewidth=.3, size=10, edgecolor="black")
    sns.barplot(data=y, ax=ax, errorbar='sd', capsize=.5)
    ax.set_xticklabels(x)
    ticks = ax.get_xticklabels()
    for i in range(len(x_label)):
        ticks[i].set_rotation(-90)
    ax.tick_params(axis='x', labelsize=70)
    ax.tick_params(axis='y', labelsize=70)

    plt.xlabel("semi-value", fontsize=80, labelpad=-120)
    ylabels = dict(ranking="Spearman's rank correlation coefficient", variance="variance")
    plt.ylabel(ylabels[_type], fontsize=80)

    plt.savefig(os.path.join("fig", f"authentic;{_type};dataset={dataset}"), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    os.makedirs("fig", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", type=str, default="authentic")
    args_input = parser.parse_args()

    path_exp = os.path.join("exp", args_input.dir)
    for path_top in next(os.walk(path_exp))[1]:
        dataset = path_top.split("=")[1]
        path_top = os.path.join(path_exp, path_top)
        values_collect = defaultdict(list)
        for path_sub in next(os.walk(path_top))[1]:
            path_sub = os.path.join(path_top, path_sub)
            with open(os.path.join(path_sub, "args.txt"), 'r') as file:
                args = eval(file.readline())
            if "param=-1" in args.root:
                args.param = "robust"
            with np.load(os.path.join(path_sub, "values.npz")) as data:
                values_collect[args.param].append(data["values_traj"][-1])

        x_label = []
        y_coef = []
        y_var = []
        for key, value in values_collect.items():
            x_label.append(key)
            value = np.stack(value)
            corr = np.zeros(int(special.binom(len(value), 2)), dtype=np.float64)
            for j, pair in enumerate(itertools.combinations(value, 2)):
                res = stats.spearmanr(pair[0], pair[1])
                corr[j] = res.correlation
            y_coef.append(corr)
            y_var.append([np.sum(np.var(value, axis=0))])

        binplot(x_label, y_coef, "ranking")
        binplot(x_label, y_var, "variance")

        print("{:<14}| ranking               | variance".format(dataset))
        msg = "{:<14}| mean {:.3f}, std {:.3f} | {:.5f}"
        for label, coef, _var in zip(x_label, y_coef, y_var):
            print(msg.format(label, np.mean(coef), np.std(coef), _var[0]))



