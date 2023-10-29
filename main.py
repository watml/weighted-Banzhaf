import os
NUM_THREAD=1
# used to set the number of threads used by each process
# the number of total threads #threads is NUM_THREAD * n_process, make sure that #threads <= #available cpus
# increase NUM_THREAD may speed up each process
os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}"
from utils.args import *
from utils.funcs import *
from utils.load_datasets import *
from utils.estimators import *
from utils.utilityFuncs import *
import multiprocessing as mp
import torch
import numpy as np
torch.backends.deterministic = True
torch.backends.cudnn.benchmark = False

args_input = get_args()
root = os.path.join("exp", args_input.pop("dir"))
n_process = args_input.pop("n_process")
name_top = ["dataset"]
name_sub = ["value", "param", "game_seed", "estimator"]


def extreme_case(param):
    global game_func, game_args, args
    vs_traj = np.empty((1, args.n_valued), dtype=np.float64)
    game = game_func(**game_args)
    if param == 0:
        subset = np.zeros(args.n_valued, dtype=bool)
        value_empty = game.evaluate(subset)
        for player in range(args.n_valued):
            subset.fill(0)
            subset[player] = 1
            vs_traj[0, player] = game.evaluate(subset) - value_empty
    elif param == 1:
        subset = np.ones(args.n_valued, dtype=bool)
        value_all = game.evaluate(subset)
        for player in range(args.n_valued):
            subset.fill(1)
            subset[player] = 0
            vs_traj[0, player] = value_all - game.evaluate(subset)
    return vs_traj


args_all = args_product([args_input])
fill_auto(args_all)
generate_root(args_all, name_top, name_sub)
for args in args_all:
    path_exp = os.path.join(root, args.root)
    os.makedirs(path_exp, exist_ok=True)
    data_saved = os.path.join(path_exp, "values.npz")
    if os.path.exists(data_saved):
        continue

    print(f"running on {args}")

    with open(os.path.join(path_exp, "args.txt"), "w") as file:
        file.write(str(args))

    (X_valued, y_valued), (X_val, y_val), num_class = load_dataset(dataset=args.dataset, n_valued=args.n_valued,
                                                                   n_val=args.n_val, dataset_seed=args.dataset_seed)
    if args.flip_percent:
        y_clean, y_valued = flip_label(y_valued, num_class, flip_percent=args.flip_percent, flip_seed=args.flip_seed)
    game_args = dict(X_valued=X_valued, y_valued=y_valued, X_val=X_val, y_val=y_val, num_class=num_class,
                     arch=args.arch, lr=args.lr, game_seed=args.game_seed)
    game_func = getattr(sys.modules[__name__], args.game_func)

    if args.value == "weighted_banzhaf" and (args.param == 0 or args.param == 1):
        with mp.Pool(1) as pool:
            process = pool.imap(extreme_case, [args.param])
            for values_traj in process:
                pass
    else:
        runner = runEstimator(root=path_exp, estimator=args.estimator, n_process=n_process,
                              value=args.value, param=args.param, game_func=game_func, game_args=game_args,
                              num_player=args.n_valued, batch_size_avg=args.batch_size_avg,
                              num_sample_avg=args.num_sample_avg, interval_track_avg=args.interval_track_avg,
                              estimator_seed=args.estimator_seed)
        _, values_traj = runner.run()
    np.savez_compressed(data_saved, values_traj=values_traj)

    if args.flip_percent:
        noisy_label_idx = np.squeeze(y_clean != y_valued)
        values = values_traj[-1]
        num_flipped = noisy_label_idx.sum()
        noisy_detected_idx = np.zeros(args.n_valued, dtype=bool)
        noisy_detected_idx[values.argsort()[:num_flipped]] = 1
        num_overlap = sum(noisy_label_idx & noisy_detected_idx)
        f1_score = 2 * num_overlap / (sum(noisy_label_idx) + sum(noisy_detected_idx))
        np.savez_compressed(os.path.join(path_exp, "f1_score.npz"), f1_scores=f1_score)







