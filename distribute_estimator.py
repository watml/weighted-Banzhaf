import os
# below is the complete list of environmental variables and the package
# that uses that variable to control the number of threads it spawns.
os.environ["OMP_NUM_THREADS"] = "1" # openmp, export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # openblas, export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # mkl, export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # accelerate, export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # numexpr, export NUMEXPR_NUM_THREADS=1
import fcntl
from utils.default import *
from utils import vd
import numpy as np
import traceback
from utils.aids import *
from utils.utilityFuncs import *
from utils.estimator_parallel import *
import contextlib
import time
import argparse
import platform
import multiprocessing as mp
from utils import universe
########################################################
use_gpu = 0
exp_name = "weighted"
err_file = "err"
out_file = "out"
prog_file = "prog"
args_list = [
    dict(
        estimator = dict(
            value="weighted_banzhaf",
            param=np.round(np.arange(0.05, 1.0, 0.05), 2),
            #param=None,
            #param=[(16,1),(4,1),(1,4),(1,16),(2,2)],
            method="maximum_sample_reuse",
            num_eval_per_player=4000
        ),
        game = dict(
            dataset=["covertype","wind","cpu","2dplanes","pol","phoneme","fraud"],
            metric=["accuracy"],
            game_seed=np.arange(20)
        ),
        dataset = dict(
            n_valued=200,
            n_val=200
        )
    ),
    dict(
        estimator = dict(
            value="beta_shapley",
            #param=np.round(np.arange(0.05, 1.0, 0.05), 2),
            #param=None,
            param=[(16,1),(4,1),(1,4),(1,16),(2,2)],
            method="reweighted_sampling_lift",
            num_eval_per_player=4000
        ),
        game = dict(
            dataset=["covertype","wind","cpu","2dplanes","pol","phoneme","fraud"],
            metric=["accuracy"],
            game_seed=np.arange(20)
        ),
        dataset = dict(
            n_valued=200,
            n_val=200
        )
    ),
    dict(
        estimator = dict(
            value="shapley",
            #param=np.round(np.arange(0.05, 1.0, 0.05), 2),
            param=None,
            #param=[(16,1),(4,1),(1,4),(1,16),(2,2)],
            method="permutation",
            num_eval_per_player=4000
        ),
        game = dict(
            dataset=["covertype","wind","cpu","2dplanes","pol","phoneme","fraud"],
            metric=["accuracy"],
            game_seed=np.arange(20)
        ),
        dataset = dict(
            n_valued=200,
            n_val=200
        )
    )
]
n_process = 16

path_variable = ["value", "param", "method", "metric", "game_seed"]
# with n_valued=n_val=200, n_test=2000, n_process=20 will explode, n_process=15 is fine
# with n_valued=n_val=n_test=15, set n_process=30
root = "/home/wangy1g/wd/datavaluation/exp"
########################################################
def get_acc(args_game, args_dataset):
    game = game_from_data(**args_game, **args_dataset)
    return game.apply_seed()

parser = argparse.ArgumentParser()
parser.add_argument("--n_process", type=int, default=n_process)
parser.add_argument("--use_gpu", type=int, default=use_gpu)
arg_in = parser.parse_args()
n_process = arg_in.n_process
use_gpu = arg_in.use_gpu

err_file = os.path.join(root, f"{exp_name}_{err_file}.txt")
out_file = os.path.join(root, f"{exp_name}_{out_file}.txt")
prog_file = os.path.join(root, f"{exp_name}_{prog_file}.txt")

load_torch_device(use_gpu=use_gpu)
universe.pool = mp.Pool(n_process)

parser = args_parser()
args_all = parser.product_args_list(path_variable=path_variable, args_list=args_list)
try:
    for args_dict in args_all:
        save_path = os.path.join(root, args_dict.save_path)

        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass

        if os.path.exists(os.path.join(save_path, ".completed")):
            continue

        lock_process = vd.os_lock(save_path)
        if not lock_process.acquire():
            continue

        tic = time.time()
        print('\033[94m' + "running on the argument\033[0m:", args_dict)
        lock_writer = fcntl_lock(root)
        while True:
            is_lock = lock_writer.acquire()
            if is_lock:
                break
        with open(out_file, "a+") as f_out:
            f_out.write(f"{platform.node()} is running on {args_dict}\n")
        lock_writer.realse()
        del lock_writer

        with mp.Pool(1) as p:
            acc = p.starmap(get_acc, [(args_dict.game, args_dict.dataset)])[0]


        estimator = value_estimator_parallel(args_game=args_dict.game, args_dataset=args_dict.dataset, use_gpu=use_gpu,
                                    n_process=n_process, prog_file=prog_file)
        values = estimator.run(**args_dict.estimator)

        save_file = os.path.join(save_path, "values.npz")
        np.savez_compressed(save_file, values=values, acc=acc)

        toc = time.time()
        print(f"time elapsed {toc - tic} seconds {datetime.now()}")
        with open(os.path.join(save_path, "log.txt"), "w+") as f:
            f.write(f"{platform.node()} time elapsed {toc - tic} seconds {datetime.now()}")

        f = open(os.path.join(save_path, ".completed"), "w+")
        f.close()
        lock_process.realse()
        del lock_process
    universe.pool.terminate()
except:
    if "lock_process" in locals():
        lock_process.realse()
    if "lock_writer" in locals():
        lock_writer.realse()
    with open(err_file, "a+") as f_stderr:
        f_stderr.write("\n")
        traceback.print_exc(file=f_stderr)
    traceback.print_exc()
    universe.pool.terminate()














