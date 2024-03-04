import numpy as np
import itertools
from scipy import special
import sys
import multiprocessing as mp
from tqdm import tqdm

class runEstimator:
    def __init__(self, *, value, param, estimator, n_process, **kwargs):
        self.value = value
        self.param = param
        self.estimator = estimator
        self.n_process = n_process
        self.estimator_args = kwargs


    def run(self):
        estimator = getattr(sys.modules[__name__], self.estimator)(value=self.value, param=self.param,
                                                                   **self.estimator_args)
        if self.n_process > 1:
            with mp.Pool(self.n_process) as pool:
                process = pool.imap(estimator.run, estimator.sampling())
                for chunk in tqdm(process, total=-(-estimator.num_sample//estimator.batch_size),
                                  miniters=self.n_process):
                    estimator.aggregate(chunk)
        else:
            for samples in tqdm(estimator.sampling(), total=-(-estimator.num_sample//estimator.batch_size)):
                estimator.aggregate(estimator.run(samples))
        return estimator.finalize()


class estimatorBasic:
    def __init__(self, *, value, param, game_func, game_args, num_player, num_sample_avg, batch_size_avg,
                 interval_track_avg, estimator_seed=2023):
        self.value = value
        self.param = param
        self.game_func = game_func
        self.game_args = game_args
        self.num_player = num_player
        self.num_sample_avg = num_sample_avg
        self.batch_size_avg = batch_size_avg
        self.interval_track_avg = interval_track_avg
        self.estimator_seed = estimator_seed

    def _sampling(self):
        pass

    def sampling(self):
        np.random.seed(self.estimator_seed)
        return self._sampling()


class exact_value(estimatorBasic):
    def __init__(self, **kwargs):
        super(exact_value, self).__init__(**kwargs)
        self.values = np.zeros(self.num_player, dtype=np.float64)
        self.num_sample = 2 ** (self.num_player - 1)
        self.batch_size = -(-self.batch_size_avg // 2)

    def sampling(self):
        count = 0
        samples = np.empty((self.batch_size, self.num_player-1), dtype=bool)
        for subset in itertools.product([True, False], repeat=self.num_player-1):
            samples[count] = subset
            count += 1
            if count == self.batch_size:
                yield samples.copy()
                count = 0
        if count:
            yield samples[:count]

    def run(self, samples):
        weights = np.empty(self.num_player, dtype=np.float64)
        for i in range(self.num_player):
            if self.value == "shapley":
                weights[i] = special.beta(self.num_player - i, i + 1)
            elif self.value == "weighted_banzhaf":
                weights[i] = (self.param ** i) * ((1 - self.param) ** (self.num_player - 1 - i))
            elif self.value == "beta_shapley":
                weights[i] = 1
                alpha, beta = self.param
                for k in range(1, i+1):
                    weights[i] *= (beta+k-1) / (alpha+beta+k-1)
                for k in range(i+1, self.num_player):
                    weights[i] *= (alpha+k-i-1) / (alpha+beta+k-1)
            else:
                raise NotImplementedError(f"Check {self.value}")

        game = self.game_func(**self.game_args)
        fragment = np.zeros(self.num_player)
        right_index = np.zeros(self.num_player, dtype=bool)
        left_index = np.ones_like(right_index)
        for sample in samples:
            weight = weights[sample.sum()]
            right_index[:self.num_player - 1] = sample
            left_index[:self.num_player - 1] = sample
            fragment[-1] += weight * (game.evaluate(left_index) - game.evaluate(right_index))
            for player in range(self.num_player - 1):
                right_index[-1], right_index[player] = right_index[player], right_index[-1]
                left_index[-1], left_index[player] = left_index[player], left_index[-1]
                fragment[player] += weight * (game.evaluate(left_index) - game.evaluate(right_index))
                right_index[-1], right_index[player] = right_index[player], right_index[-1]
                left_index[-1], left_index[player] = left_index[player], left_index[-1]
        return fragment

    def aggregate(self, fragment):
        self.values += fragment

    def finalize(self):
        return self.values, self.values[None, :]


class sampling_lift(estimatorBasic):
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        assert self.interval_track_avg % 2 == 0
        self.interval_track = self.interval_track_avg // 2
        assert self.num_sample_avg % 2 == 0
        self.num_sample = self.num_sample_avg // 2
        self.values_aggregate = np.zeros(self.num_player, dtype=np.float64)
        self.count_aggregate = 0
        self.batch_size = -(-self.batch_size_avg // 2)

        num_traj = self.num_sample_avg // self.interval_track_avg
        self.values_traj = np.empty((num_traj, self.num_player), dtype=np.float64)
        self.pos_traj = 0

        len_buffer = self.interval_track + self.batch_size - 1
        self.buffer = np.empty((len_buffer, self.num_player), dtype=np.float64)
        self.pos_buffer = 0

    def _sampling(self):
        count = 0
        samples = np.empty((self.batch_size, self.num_player-1), dtype=np.int32)
        for _ in range(self.num_sample):
            if self.value == "weighted_banzhaf":
                t = self.param
            elif self.value == "shapley":
                t = np.random.rand()
            elif self.value == "beta_shapley":
                t = np.random.beta(self.param[1], self.param[0])
            else:
                raise NotImplementedError
            samples[count] = np.random.binomial(1, t, size=self.num_player-1)
            count += 1
            if count == self.batch_size:
                yield samples.copy()
                count = 0
        if count:
            yield samples[:count]

    def run(self, samples):
        game = self.game_func(**self.game_args)
        values_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        for i, sample in enumerate(samples):
            values = values_collect[i]
            subset[:self.num_player-1] = sample
            values[-1] -= game.evaluate(subset)
            subset[-1] = 1
            values[-1] += game.evaluate(subset)
            for player in range(self.num_player - 1):
                subset[-1], subset[player] = subset[player], subset[-1]
                values[player] += game.evaluate(subset)
                subset[player] = 0
                values[player] -= game.evaluate(subset)
                subset[player] = 1
                subset[-1], subset[player] = subset[player], subset[-1]
            subset[-1] = 0
        return values_collect

    def aggregate(self, values_collect):
        self.buffer[self.pos_buffer:self.pos_buffer+len(values_collect)] = values_collect
        self.pos_buffer += len(values_collect)
        num_collect = self.pos_buffer // self.interval_track
        if num_collect:
            for i in range(num_collect):
                buffer_take = self.buffer[i*self.interval_track:(i+1)*self.interval_track]
                self.values_aggregate += buffer_take.mean(axis=0)
                self.count_aggregate += 1
                self.values_traj[self.pos_traj] = self.values_aggregate / self.count_aggregate
                self.pos_traj += 1
            num_left = self.pos_buffer-(i+1)*self.interval_track
            self.buffer[:num_left] = self.buffer[(i+1)*self.interval_track:self.pos_buffer]
            self.pos_buffer = num_left

    def finalize(self):
        if self.pos_buffer:
            count_total = self.count_aggregate * self.interval_track + self.pos_buffer
            values_final = (self.values_aggregate * self.interval_track + self.buffer[:self.pos_buffer].sum(axis=0)) \
                           / count_total
        else:
            values_final = self.values_traj[-1]
        return values_final, self.values_traj


class weighted_sampling_lift(sampling_lift):
    def __init__(self, **kwargs):
        super(weighted_sampling_lift, self).__init__(**kwargs)
        assert self.value != "shapley"

    def _sampling(self):
        count = 0
        samples = np.empty((self.batch_size, self.num_player - 1), dtype=np.int32)
        for _ in range(self.num_sample):
            t = np.random.rand()
            samples[count] = np.random.binomial(1, t, size=self.num_player - 1)
            count += 1
            if count == self.batch_size:
                yield samples.copy()
                count = 0
        if count:
            yield samples[:count]

    def run(self, samples):
        if self.value == "weighted_banzhaf":
            weights = np.ones(self.num_player, dtype=np.float64)
            for k in range(self.num_player):
                for i in range(k):
                    weights[k] *= (self.num_player - 1 - i) / (i + 1) * self.param
                weights[k] *= (1 - self.param) ** (self.num_player - 1 - k)
        elif self.value == "beta_shapley":
            alpha, beta = self.param
            weights = np.ones(self.num_player, dtype=np.float64)
            tmp_range = np.arange(1, self.num_player)
            weights *= np.divide(tmp_range, tmp_range + (alpha + beta - 1)).prod()
            for s in range(self.num_player):
                r_cur = weights[s]
                tmp_range = np.arange(1, s + 1)
                r_cur *= np.divide(tmp_range + (beta - 1), tmp_range).prod()
                tmp_range = np.arange(1, self.num_player - s)
                r_cur *= np.divide((alpha - 1) + tmp_range, tmp_range).prod()
                weights[s] = r_cur
        else:
            raise NotImplementedError(f"Check {self.value}")
        weights *= self.num_player

        game = self.game_func(**self.game_args)
        values_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        for i, sample in enumerate(samples):
            values = values_collect[i]
            weight = weights[sample.sum()]
            subset[:self.num_player-1] = sample
            value_right = game.evaluate(subset)
            subset[-1] = 1
            value_left = game.evaluate(subset)
            values[-1] += weight * (value_left - value_right)
            for player in range(self.num_player - 1):
                subset[-1], subset[player] = subset[player], subset[-1]
                value_left = game.evaluate(subset)
                subset[player] = 0
                value_right = game.evaluate(subset)
                values[player] += weight * (value_left - value_right)
                subset[player] = 1
                subset[-1], subset[player] = subset[player], subset[-1]
            subset[-1] = 0
        return values_collect


class permutation(sampling_lift):
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        assert self.value == "shapley"
        self.num_sample = self.num_sample_avg
        self.interval_track = self.interval_track_avg
        self.values_aggregate = np.zeros(self.num_player, dtype=np.float64)
        self.count_aggregate = 0
        self.batch_size = self.batch_size_avg

        num_traj = self.num_sample_avg // self.interval_track_avg
        self.values_traj = np.empty((num_traj, self.num_player), dtype=np.float64)
        self.pos_traj = 0

        len_buffer = self.interval_track + self.batch_size - 1
        self.buffer = np.empty((len_buffer, self.num_player), dtype=np.float64)
        self.pos_buffer = 0

    def _sampling(self):
        count = 0
        samples = np.empty((self.batch_size, self.num_player), dtype=np.int32)
        for _ in range(self.num_sample):
            samples[count] = np.random.permutation(self.num_player)
            count += 1
            if count == self.batch_size:
                yield samples.copy()
                count = 0
        if count:
            yield samples[:count]

    def run(self, samples):
        game = self.game_func(**self.game_args)
        values_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        empty_value = game.evaluate(subset)
        for i, sample in enumerate(samples):
            values = values_collect[i]
            pre_value = empty_value
            for j in range(self.num_player):
                player = sample[j]
                values[player] -= pre_value
                subset[player] = 1
                cur_value = game.evaluate(subset)
                values[player] += cur_value
                pre_value = cur_value
            subset.fill(0)
        return values_collect


class maximum_sample_reuse(estimatorBasic):
    def __init__(self, **kwargs):
        super(maximum_sample_reuse, self).__init__(**kwargs)
        assert self.value == "weighted_banzhaf"
        assert 0 < self.param and self.param < 1
        self.num_sample = self.num_sample_avg * self.num_player
        self.interval_track = self.interval_track_avg * self.num_player
        self.results_aggregate = np.zeros((4, self.num_player), dtype=np.float64)
        self.batch_size = self.batch_size_avg * self.num_player

        num_traj = self.num_sample_avg // self.interval_track_avg
        self.values_traj = np.empty((num_traj, self.num_player), dtype=np.float64)
        self.pos_traj = 0

        len_buffer = self.interval_track + self.batch_size - 1
        self.buffer = np.empty((len_buffer, self.num_player+1), dtype=np.float64)
        self.pos_buffer = 0

    def _sampling(self):
        for i in range(self.batch_size, self.num_sample, self.batch_size):
            collect = np.random.binomial(1, self.param, size=(self.batch_size, self.num_player)).astype(bool)
            yield collect
        num_rest = self.num_sample - i
        collect = np.random.binomial(1, self.param, size=(num_rest, self.num_player)).astype(bool)
        yield collect

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), self.num_player+1), dtype=np.float64)
        for i, sample in enumerate(samples):
            results_collect[i, :self.num_player] = sample
            results_collect[i, -1] = game.evaluate(sample)
        return results_collect

    def aggregate(self, results_collect):
        self.buffer[self.pos_buffer:self.pos_buffer + len(results_collect)] = results_collect
        self.pos_buffer += len(results_collect)
        num_collect = self.pos_buffer // self.interval_track
        if num_collect:
            for i in range(num_collect):
                buffer_take = self.buffer[i*self.interval_track:(i+1)*self.interval_track]
                for take in buffer_take:
                    subset = take[:self.num_player] == 1
                    value = take[-1]
                    self.results_aggregate[0, subset] += value
                    self.results_aggregate[1, subset] += 1
                    self.results_aggregate[2, ~subset] += value
                    self.results_aggregate[3, ~subset] += 1
                self.values_traj[self.pos_traj] = self.generate_value()
                self.pos_traj += 1
            num_left = self.pos_buffer - (i + 1) * self.interval_track
            self.buffer[:num_left] = self.buffer[(i + 1) * self.interval_track:self.pos_buffer]
            self.pos_buffer = num_left

    def finalize(self):
        if self.pos_buffer:
            for take in self.buffer[:self.pos_buffer]:
                subset = take[:self.num_player] == 1
                value = take[-1]
                self.results_aggregate[0, subset] += value
                self.results_aggregate[1, subset] += 1
                self.results_aggregate[2, ~subset] += value
                self.results_aggregate[3, ~subset] += 1
            values_final = self.generate_value()
        else:
            values_final = self.values_traj[-1]
        return values_final, self.values_traj

    def generate_value(self):
        left = np.divide(self.results_aggregate[0], self.results_aggregate[1],
                         out=np.zeros(self.num_player, dtype=np.float64),
                         where=self.results_aggregate[1] != 0)
        right = np.divide(self.results_aggregate[2], self.results_aggregate[3],
                          out=np.zeros(self.num_player, dtype=np.float64),
                          where=self.results_aggregate[3] != 0)
        return left - right


