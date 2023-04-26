import numpy as np
import itertools
from scipy import special
from tqdm import tqdm
import platform
import functools
from utils.dataset import *
from utils import models, universe
import torch
import os
from utils.utilityFuncs import *
from utils.aids import *
from utils.vd import *


tqdm_display_step = 100


class value_estimator_parallel():
    def __init__(self, *, args_game, args_dataset, use_gpu, n_process, prog_file, probability_target):
        self.args_game = args_game
        self.args_dataset = args_dataset
        self.use_gpu = use_gpu
        self.n_process = n_process
        self.prog_file = prog_file
        self.probability_target = probability_target


    def run(self, *, value="shapley", param=None, method="exact", num_eval_per_player=4000, estimator_seed=2023,
            track_interval_per_player=250):
        """
        track_interval=50 means the approximate value will be tracked at 50,100,150,200,..., sample points.

        For different methods, the number of samples correspond to different numbers of model evalutions:
        "sampling_lift" and "reweighted_sampling_lift": it is 2 * #samples * num_player
        "permutation": #samples * num_player
        "maximum_sample_reuse": #samples
        """

        if method == "weighted_banzhaf":
            assert 0<=param and param<=1, f"Check {param}"
        elif method == "beta_shapley":
            assert param[0]>0 and param[1]>0, f"Check {param}"
        if track_interval_per_player is not None and method != "exact":
            assert  num_eval_per_player % track_interval_per_player == 0


        # pool = mp.Pool(self.n_process)

        num_player = self.args_dataset.n_valued
        calculated_value = np.zeros(num_player)
        traj = []

        args_dict = dict(method=method)
        np.random.seed(estimator_seed)
        if method == "exact":
            weight_all = np.zeros(num_player)
            for size_subset in range(num_player):
                weight_all[size_subset] = self.calculate_weight(value, param, size_subset, num_player - 1)
            args_dict.update(weight_all=weight_all)

            do_job = functools.partial(self.estimate_parallel, args_dict=args_dict)
            jobs = list(itertools.product([1, 0], repeat=num_player-1))
            chunksize = -(len(jobs) // -self.n_process)
            job_list = self.divide_chunks(jobs, chunksize)
            process = universe.pool.imap(do_job, job_list)
            for r in process:
                calculated_value += r

        elif method == "permutation":
            assert value == "shapley"
            num_sample = num_eval_per_player

            do_job = functools.partial(self.estimate_parallel, args_dict=args_dict)
            jobs = [None] * num_sample
            for i in range(num_sample):
                jobs[i] = np.random.permutation(num_player)

            if track_interval_per_player is not None:
                assert num_sample % self.n_process == 0
                chunksize = num_sample // self.n_process
                assert track_interval_per_player % chunksize == 0
                cpu_per_track_point = track_interval_per_player // chunksize
            else:
                chunksize = -(num_sample // -self.n_process)

            job_list = list(self.divide_chunks(jobs, chunksize))
            process = universe.pool.imap(do_job, job_list)
            for count, r in enumerate(process):
                calculated_value += r
                if track_interval_per_player is not None:
                    if (count+1) % cpu_per_track_point == 0:
                        traj.append(np.divide(calculated_value, (count+1)*chunksize))
            calculated_value /= num_sample

        elif method == "sampling_lift":
            prob = self.calculate_probability(value, param, num_player)
            assert num_eval_per_player % 2 == 0
            num_sample = num_eval_per_player // 2
            jobs = [None] * num_sample
            for i in range(num_sample):
                size_subset = np.random.choice(np.arange(num_player), p=prob)
                jobs[i] = np.random.choice(np.arange(num_player - 1), size_subset, replace=False)

            if track_interval_per_player is not None:
                assert num_sample % self.n_process == 0
                chunksize = num_sample // self.n_process
                assert track_interval_per_player % 2 == 0
                assert (track_interval_per_player // 2) % chunksize == 0
                cpu_per_track_point = (track_interval_per_player // 2) // chunksize
            else:
                chunksize = -(num_sample // -self.n_process)


            job_list = list(self.divide_chunks(jobs, chunksize))
            do_job = functools.partial(self.estimate_parallel, args_dict=args_dict)
            process = universe.pool.imap(do_job, job_list)
            for count, r in enumerate(process):
                calculated_value += r
                if track_interval_per_player is not None:
                    if (count+1) % cpu_per_track_point == 0:
                        traj.append(np.divide(calculated_value, (count+1)*chunksize))
            calculated_value /= num_sample

        elif method == "reweighted_sampling_lift":
            assert num_eval_per_player % 2 == 0
            num_sample = num_eval_per_player // 2
            jobs = [None] * num_sample
            for i in range(num_sample):
                size_subset = np.random.choice(np.arange(num_player))
                jobs[i] = np.random.choice(np.arange(num_player - 1), size_subset, replace=False)

            if track_interval_per_player is not None:
                assert num_sample % self.n_process == 0
                chunksize = num_sample // self.n_process
                assert track_interval_per_player % 2 == 0
                assert (track_interval_per_player // 2) % chunksize == 0
                cpu_per_track_point = (track_interval_per_player // 2) // chunksize
            else:
                chunksize = -(num_sample // -self.n_process)


            job_list = list(self.divide_chunks(jobs, chunksize))
            weight_all = num_player * self.calculate_probability(value, param, num_player)
            args_dict.update(weight_all=weight_all)
            do_job = functools.partial(self.estimate_parallel, args_dict=args_dict)
            process = universe.pool.imap(do_job, job_list)
            for count, r in enumerate(process):
                calculated_value += r
                if track_interval_per_player is not None:
                    if (count+1) % cpu_per_track_point == 0:
                        traj.append(np.divide(calculated_value, (count+1)*chunksize))
            calculated_value /= num_sample

        elif method == "maximum_sample_reuse":
            assert value == "weighted_banzhaf"
            assert param>0 and param<1
            num_sample = num_eval_per_player * num_player
            jobs = np.random.binomial(1, param, size=(num_sample, num_player))
            if track_interval_per_player is not None:
                assert num_sample % self.n_process == 0
                chunksize = num_sample // self.n_process
                assert (track_interval_per_player * num_player) % chunksize == 0
                cpu_per_track_point = (track_interval_per_player * num_player) // chunksize
            else:
                chunksize = -(num_sample // -self.n_process)

            job_list = list(self.divide_chunks(jobs, chunksize))
            do_job = functools.partial(self.estimate_parallel, args_dict=args_dict)
            process = universe.pool.imap(do_job, job_list)

            count_left = np.zeros(num_player)
            count_right = np.zeros(num_player)
            sum_left = np.zeros(num_player)
            sum_right = np.zeros(num_player)
            for count, r in enumerate(process):
                count_left_chip, count_right_chip, sum_left_chip, sum_right_chip = r
                count_left += count_left_chip
                count_right += count_right_chip
                sum_left += sum_left_chip
                sum_right += sum_right_chip
                if track_interval_per_player is not None:
                    if (count+1) % cpu_per_track_point == 0:
                        count_left[count_right == 0] = -1
                        count_right[count_right == 0] = -1
                        traj.append(np.divide(sum_left, count_left) + np.divide(sum_right, count_right))
                        count_left[count_right == -1] = 0
                        count_right[count_right == -1] = 0
            count_left[count_right==0] = -1
            count_right[count_right==0] = -1
            calculated_value = np.divide(sum_left,count_left) + np.divide(sum_right,count_right)

        else:
            raise NotImplementedError(f"Check {method}")

        np.random.seed(None)
        if len(traj):
            return traj
        else:
            return calculated_value

    def estimate_parallel(self, request, args_dict):
        game = game_for_parallel(**self.args_game, **self.args_dataset, probability_target=self.probability_target)
        method = args_dict["method"]
        num_player = self.args_dataset.n_valued

        id = request[0]
        if id == self.n_process-2 or (self.n_process == 1 and id == 0):
            display_tqdm = True
        else:
            display_tqdm = False

        jobs = request[1]
        num_job = len(jobs)
        miniters = num_job // tqdm_display_step

        value_chip = np.zeros(num_player)
        if method == "exact":
            right_index = np.zeros(num_player, dtype=bool)
            left_index = np.ones_like(right_index)
            weight_all = args_dict["weight_all"]
            for job in vd_tqdm(jobs, file_to_write=self.prog_file,  miniters=miniters, maxinterval=float('inf'),
                               disable=not display_tqdm):
                weight = weight_all[np.sum(job)]
                right_index[:num_player - 1] = job
                left_index[:num_player - 1] = job
                value_chip[-1] += weight * (game.evaluate(left_index) - game.evaluate(right_index))
                for player in range(num_player - 1):
                    right_index[-1], right_index[player] = right_index[player], right_index[-1]
                    left_index[-1], left_index[player] = left_index[player], left_index[-1]
                    value_chip[player] += weight * (game.evaluate(left_index) - game.evaluate(right_index))
                    right_index[-1], right_index[player] = right_index[player], right_index[-1]
                    left_index[-1], left_index[player] = left_index[player], left_index[-1]
        elif method == "permutation":
            subset_index = np.zeros(num_player, dtype=bool)
            empty_value = game.evaluate(subset_index)
            for job in vd_tqdm(jobs, file_to_write=self.prog_file,  miniters=miniters, maxinterval=float('inf'),
                               disable=not display_tqdm):
                pre_value = empty_value
                for i in range(num_player):
                    player = job[i]
                    value_chip[player] -= pre_value
                    subset_index[player] = 1
                    cur_value = game.evaluate(subset_index)
                    value_chip[player] += cur_value
                    pre_value = cur_value
                subset_index.fill(0)

        elif method == "sampling_lift":
            subset_index = np.zeros(num_player, dtype=bool)
            for job in vd_tqdm(jobs, file_to_write=self.prog_file,  miniters=miniters, maxinterval=float('inf'),
                               disable=not display_tqdm):
                subset_index[job] = 1
                value_chip[-1] -= game.evaluate(subset_index)
                subset_index[-1] = 1
                value_chip[-1] += game.evaluate(subset_index)
                for player in range(num_player - 1):
                    subset_index[-1], subset_index[player] = subset_index[player], subset_index[-1]
                    value_chip[player] += game.evaluate(subset_index)
                    subset_index[player] = 0
                    value_chip[player] -= game.evaluate(subset_index)
                    subset_index[player] = 1
                    subset_index[-1], subset_index[player] = subset_index[player], subset_index[-1]
                subset_index.fill(0)

        elif method == "reweighted_sampling_lift":
            subset_index = np.zeros(num_player, dtype=bool)
            weight_all = args_dict["weight_all"]
            for job in vd_tqdm(jobs, file_to_write=self.prog_file,  miniters=miniters, maxinterval=float('inf'),
                               disable=not display_tqdm):
                weight = weight_all[len(job)]
                subset_index[job] = 1
                value_right = game.evaluate(subset_index)
                subset_index[-1] = 1
                value_left = game.evaluate(subset_index)
                value_chip[-1] += weight * (value_left - value_right)
                for player in range(num_player - 1):
                    subset_index[-1], subset_index[player] = subset_index[player], subset_index[-1]
                    value_left = game.evaluate(subset_index)
                    subset_index[player] = 0
                    value_right = game.evaluate(subset_index)
                    value_chip[player] += weight * (value_left - value_right)
                    subset_index[player] = 1
                    subset_index[-1], subset_index[player] = subset_index[player], subset_index[-1]
                subset_index.fill(0)
        elif method == "maximum_sample_reuse":
            count_left = np.zeros(num_player)
            count_right = np.zeros(num_player)
            sum_left = np.zeros(num_player)
            sum_right = np.zeros(num_player)
            for job in vd_tqdm(jobs, file_to_write=self.prog_file,  miniters=miniters, maxinterval=float('inf'),
                               disable=not display_tqdm):
                value_subset = game.evaluate(job)
                index_left = job == 1
                count_left[index_left] += 1
                count_right[~index_left] += 1
                sum_left[index_left] += value_subset
                sum_right[~index_left] -= value_subset
            return (count_left, count_right, sum_left, sum_right)

        else:
            raise NotImplementedError

        return value_chip

    @staticmethod
    def divide_chunks(l, n):
        for i, j in enumerate(range(0, len(l), n)):
            yield (i, l[j: j + n])


    @staticmethod
    def calculate_weight(value, param, size_subset, num_active_player):
        if value == "shapley":
            weight = special.beta(num_active_player+1-size_subset, size_subset+1)
        elif value == "weighted_banzhaf":
            weight = (param ** size_subset) * ((1 - param) ** (num_active_player-size_subset))
        elif value == "beta_shapley":
            weight = 1.
            alpha, beta = param
            for k in range(1,size_subset+1):
                weight *= (beta+k-1)/(alpha+beta+k-1)
            for k in range(size_subset+1, num_active_player+1):
                weight *= (alpha+k-size_subset-1)/(alpha+beta+k-1)
        else:
            raise NotImplementedError(f"Check {value}")
        return weight



    def calculate_probability(self, value, param, num_player):
        prob = np.ones(num_player)
        if value == "shapley":
            prob *= 1./num_player
        elif value == "weighted_banzhaf":
            for i in range(num_player):
                prob[i] = self.calculate_weight(value, param, i, num_player-1) * special.binom(num_player-1, i)
            # binom will explode when num_player >= 2000
        elif value == "beta_shapley":
            alpha, beta = param
            for s in range(num_player):
                weight = 1.
                for k in range(1,num_player):
                    weight *= k/(alpha+beta+k-1)
                for k in range(1, s+1):
                    weight *= (beta+k-1)/k
                for k in range(1, num_player-s):
                    weight *= (alpha+k-1)/k
                prob[s] = weight
        else:
            raise NotImplementedError(f"Check {value}")
        return prob













