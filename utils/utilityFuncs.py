import numpy as np
import torch
from utils import models
from utils.funcs import *
import sys


class gameTraining:
    def __init__(self, *, X_valued, y_valued, X_val, y_val, num_class,
                 arch, lr, game_seed=2023):
        self.X_valued, self.y_valued = X_valued, y_valued
        self.X_val, self.y_val = X_val, y_val
        self.arch = arch

        self.num_player = len(y_valued)

        # initialize everything related to each specified game_seed
        self._game_seed = self.X_train = self.y_train = self.perm = None
        self.game_seed = game_seed

        # load model and optimizer
        if arch == "logistic":
            self.model = models.LogisticRegression(self.X_val.shape[1], num_class)
        elif arch == "LeNet":
            self.model = models.LeNet()
        else:
            raise NotImplementedError(f"Check {arch}")
        self.model.double() # float64 is used for more consistent reproducibility across platforms

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    @property
    def game_seed(self):
        return self._game_seed

    @game_seed.setter
    def game_seed(self, value):
        self._game_seed = value
        with set_numpy_seed(value):
            self.perm = np.random.permutation(self.num_player)
        self.X_train = self.X_valued[self.perm]
        self.y_train = self.y_valued[self.perm]


    def train_model(self, X, y):
        with set_numpy_seed((X > 0).sum().item() + (y == 0).sum().item()):
            pi = np.random.permutation(len(y))
        X, y = X[pi], y[pi]
        with set_torch_seed(self.game_seed):
            for layer in self.model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if len(y):
            for datum, label in zip(X, y):
                logit = self.model(datum)
                loss = self.criterion(logit, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        subset = subset[self.perm]
        self.train_model(self.X_train[subset], self.y_train[subset])
        self.model.eval()
        with torch.no_grad():
            logit = self.model(self.X_val)
        self.model.train()
        assert ~torch.isnan(logit.sum())
        predict = np.argmax(logit.numpy(), 1)
        label = self.y_val.numpy()
        score = np.sum(predict == label) / len(label)
        return score

class gamePerturbed:
    def __init__(self, *, game_func, noise_vector, **kwargs):
        game_func = getattr(sys.modules[__name__], game_func)
        self.num_player = len(kwargs["y_valued"])
        self.game = game_func(**kwargs)
        self.noise_vector = noise_vector

    def evaluate(self, subset):
        assert isinstance(subset[0], np.bool_)
        score = self.game.evaluate(subset)
        return score + np.prod(self.noise_vector[np.arange(self.num_player), np.int_(subset)])


















