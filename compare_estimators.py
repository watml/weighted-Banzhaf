import os
NUM_THREAD = 1
os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}" # openmp, export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}" # openblas, export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}" # mkl, export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}" # accelerate, export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}" # numexpr, export NUMEXPR_NUM_THREADS=1
import numpy as np
from utils.load_datasets import *
import torch
torch.backends.deterministic = True
torch.backends.cudnn.benchmark = False
from utils.utilityFuncs import *
from utils.estimators import *
from utils.args import *
from utils.funcs import *
import argparse
from dotmap import DotMap

# python compare_estimators.py -dataset 2dplanes -n_process 40 -dir estimator -n_valued 16 -value weighted_banzhaf -param 0.8 -lr 1.0 -estimator_seed 0 1 2 3 4 5 -num_sample_avg 1500 -batch_size_avg 50 -interval_track_avg 30
if __name__ == "__main__":
    os.makedirs("fig", exist_ok=True)
    args = DotMap(get_args())
    assert len(args.dataset) == 1
    assert len(args.param) == 1
    dataset = args.dataset[0]
    n_valued = args.n_valued
    n_val = args.n_val
    lr = args.lr
    n_process = args.n_process
    value = args.value
    param = args.param[0]
    batch_size_avg = args.batch_size_avg
    num_sample_avg = args.num_sample_avg
    interval_track_avg = args.interval_track_avg

    path_exp = os.path.join(args.dir, f"{args.dataset}")
    os.makedirs(path_exp, exist_ok=True)

    (X_valued, y_valued), (X_val, y_val), num_class = load_dataset(dataset=dataset, n_valued=n_valued,
                                                                   n_val=n_val)
    game_args = dict(X_valued=X_valued, y_valued=y_valued, X_val=X_val, y_val=y_val, num_class=num_class, arch="logistic",
                     lr=lr)

    data_exact = os.path.join(path_exp, "values_exact.npz")
    if not os.path.exists(data_exact):
        runner = runEstimator(estimator="exact_value", n_process=n_process, value=value, param=param,
                              game_func=gameTraining, game_args=game_args, num_player=n_valued,
                              batch_size_avg=batch_size_avg, num_sample_avg=num_sample_avg,
                              interval_track_avg=interval_track_avg)
        values_exact, _ = runner.run()
        np.savez_compressed(data_exact, values_exact=values_exact)
    else:
        with np.load(data_exact) as data:
            values_exact = data["values_exact"]

    data_SL = os.path.join(path_exp, "values_SL.npz")
    if not os.path.exists(data_SL):
        values_SL = []
        for seed in args.estimator_seed:
            runner = runEstimator(estimator="sampling_lift", n_process=n_process, value=value, param=param,
                                  game_func=gameTraining, game_args=game_args, num_player=n_valued,
                                  batch_size_avg=batch_size_avg, num_sample_avg=num_sample_avg,
                                  interval_track_avg=interval_track_avg, estimator_seed=seed)
            _, values_traj = runner.run()
            values_SL.append(values_traj)
        np.savez_compressed(data_SL, values_SL=np.stack(values_SL))
    else:
        with np.load(data_SL) as data:
            values_SL = data["values_SL"]

    data_WSL = os.path.join(path_exp, "values_WSL.npz")
    if not os.path.exists(data_WSL):
        values_WSL = []
        for seed in args.estimator_seed:
            runner = runEstimator(estimator="weighted_sampling_lift", n_process=n_process, value=value, param=param,
                                  game_func=gameTraining, game_args=game_args, num_player=n_valued,
                                  batch_size_avg=batch_size_avg, num_sample_avg=num_sample_avg,
                                  interval_track_avg=interval_track_avg, estimator_seed=seed)
            _, values_traj = runner.run()
            values_WSL.append(values_traj)
        np.savez_compressed(data_WSL, values_WSL=np.stack(values_WSL))
    else:
        with np.load(data_WSL) as data:
            values_WSL = data["values_WSL"]

    data_MSR = os.path.join(path_exp, "values_MSR.npz")
    if not os.path.exists(data_MSR):
        values_MSR = []
        for seed in args.estimator_seed:
            runner = runEstimator(estimator="maximum_sample_reuse", n_process=n_process, value=value, param=param,
                                  game_func=gameTraining, game_args=game_args, num_player=n_valued,
                                  batch_size_avg=batch_size_avg, num_sample_avg=num_sample_avg,
                                  interval_track_avg=interval_track_avg, estimator_seed=seed)
            _, values_traj = runner.run()
            values_MSR.append(values_traj)
        np.savez_compressed(data_MSR, values_MSR=np.stack(values_MSR))
    else:
        with np.load(data_MSR) as data:
            values_MSR = data["values_MSR"]

    norm_exact = np.linalg.norm(values_exact)
    diff_SL = np.linalg.norm(values_SL - values_exact[None, None, :], axis=2) / norm_exact
    diff_WSL = np.linalg.norm(values_WSL - values_exact[None, None, :], axis=2) / norm_exact
    diff_MSR = np.linalg.norm(values_MSR - values_exact[None, None, :], axis=2) / norm_exact

    x = np.arange(interval_track_avg, num_sample_avg+1, interval_track_avg)
    ys = [diff_SL, diff_WSL, diff_MSR]
    labels = ["sampling lift", "weighted sampling lift", "maximum sample reuse"]
    fig_saved = os.path.join("fig", f"{dataset};sampling.png")
    plot_curves(x, ys, fig_saved, labels=labels, x_label="number of utility evaluations per datum",
                y_label="relative difference")

    print("The figure is plotted.")






