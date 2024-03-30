import os
NUM_THREAD = 1
os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}" # openmp, export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}" # openblas, export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}" # mkl, export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}" # accelerate, export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}" # numexpr, export NUMEXPR_NUM_THREADS=1
import numpy as np
import torch
torch.backends.deterministic = True
torch.backends.cudnn.benchmark = False
from utils.load_datasets import *
from utils.utilityFuncs import *
from utils.estimators import *
import argparse
from scipy import stats

def generate_noise_vector(s11, s12, s22, seed_noise):
    global n_valued
    noise_cov = [[s11, s12], [s12, s22]]
    with set_numpy_seed(seed_noise):
        noise_vec = np.random.multivariate_normal([0, 0], noise_cov, size=n_valued)
    return noise_vec

def get_spearmanr(vec, vecs):
    num = len(vecs)
    result = np.empty(num, dtype=np.float64)
    for i in range(num):
        vecs_each = vecs[i]
        res = stats.spearmanr(vec, vecs_each)
        result[i] = res.correlation
    return result

if __name__ == "__main__":
    os.makedirs("fig", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="spambase")
    parser.add_argument("--n_process", type=int, default=1)
    args = parser.parse_args()
    dataset = args.dataset
    n_process = args.n_process

    n_valued = 10
    n_val = 200
    lr = 0.05
    range_sigma12 = np.round(np.arange(-0.48, 0.50, 0.02), 2)
    root = os.path.join("exp", "synthetic_noises")
    os.makedirs(root, exist_ok=True)
    num_seed = 10
    batch_size_avg = 50

    path_exp = os.path.join(root, f"{dataset}")
    os.makedirs(path_exp, exist_ok=True)
    diff = np.empty((7, num_seed, len(range_sigma12)), dtype=np.float64)

    (X_valued, y_valued), (X_val, y_val), num_class = load_dataset(dataset=dataset, n_valued=n_valued, n_val=n_val)
    game_args = dict(X_valued=X_valued, y_valued=y_valued, X_val=X_val, y_val=y_val, num_class=num_class,
                     arch="logistic", lr=lr)
    file_saved = os.path.join(path_exp, "values_exact.npz")
    if not os.path.exists(file_saved):
        values_exact = np.empty((6, n_valued), dtype=np.float64)
        runner = runEstimator(estimator="exact_value", n_process=n_process, value="shapley", param=None,
                              game_func=gameTraining, game_args=game_args, num_player=n_valued,
                              batch_size_avg=batch_size_avg, num_sample_avg=None, interval_track_avg=None)
        values_exact[0], _ = runner.run()
        runner.value = "weighted_banzhaf"
        runner.param = 0.5
        values_exact[1], _ = runner.run()
        runner.value = "beta_shapley"
        for i, param in enumerate([(16, 1), (4, 1), (1, 4), (1, 16)]):
            runner.param = param
            values_exact[2 + i], _ = runner.run()
        np.savez_compressed(file_saved, values_exact=values_exact)
    else:
        with np.load(file_saved) as data:
            values_exact = data["values_exact"]

    game_args_perturbed = game_args.copy()
    game_args_perturbed.update(game_func="gameTraining", noise_vector=None)
    for (sigma11, sigma22) in [(1.5, 0.5), (0.5, 1.5)]:
        for i, sigma12 in enumerate(range_sigma12):
            file_saved = os.path.join(path_exp, f"sigma11={sigma11};sigma12={sigma12};sigma22={sigma22}.npz")
            if not os.path.exists(file_saved):
                param_robust = (sigma11 - sigma12) / (sigma11 + sigma22 - 2 * sigma12)
                runner = runEstimator(estimator="exact_value", n_process=n_process, value="weighted_banzhaf",
                                      param=param_robust, game_func=gameTraining, game_args=game_args,
                                      num_player=n_valued, batch_size_avg=batch_size_avg, num_sample_avg=None,
                                      interval_track_avg=None)
                values_robust, _ = runner.run()

                values_noisy = np.empty((num_seed, 7, n_valued), dtype=np.float64)
                for seed in range(num_seed):
                    noise_vector = generate_noise_vector(sigma11, sigma12, sigma22, seed)
                    game_args_perturbed["noise_vector"] = noise_vector
                    runner = runEstimator(estimator="exact_value", n_process=n_process, value="shapley", param=None,
                                          game_func=gamePerturbed, game_args=game_args_perturbed, num_player=n_valued,
                                          batch_size_avg=batch_size_avg, num_sample_avg=None, interval_track_avg=None)
                    values_noisy[seed, 0], _ = runner.run()
                    runner.value = "weighted_banzhaf"
                    runner.param = 0.5
                    values_noisy[seed, 1], _ = runner.run()
                    runner.value = "beta_shapley"
                    for i, param in enumerate([(16, 1), (4, 1), (1, 4), (1, 16)]):
                        runner.param = param
                        values_noisy[seed, 2 + i], _ = runner.run()
                    runner.value = "weighted_banzhaf"
                    runner.param = param_robust
                    values_noisy[seed, -1], _ = runner.run()
                np.savez_compressed(file_saved, values_robust=values_robust, values_noisy=values_noisy)
            else:
                with np.load(file_saved) as data:
                    values_robust = data["values_robust"]
                    values_noisy = data["values_noisy"]

            diff[-1, :, i] = get_spearmanr(values_robust, values_noisy[:, -1, :])
            for j in range(6):
                diff[j, :, i] = get_spearmanr(values_exact[j], values_noisy[:, j, :])

        labels = ["Shapley", "Banzhaf", f"Beta{(16, 1)}", f"Beta{(4, 1)}", f"Beta{(1, 4)}", f"Beta{(1, 16)}",
                  "robust semi-value"]
        fig_saved = os.path.join("fig", f"synthetic_noises;dataset={dataset};sigma11={sigma11};sigma22={sigma22}.png")
        plot_curves(range_sigma12, diff, fig_saved, labels=labels, x_label=r"$\sigma_{12}$",
                    y_label="Spearman's rank correlation coefficient", plot_std=False)




