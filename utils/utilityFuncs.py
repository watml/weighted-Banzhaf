import numpy as np
from utils.aids import *
import torch
from utils import universe
from utils.dataset import *
from utils import models
import torch.nn.functional as F
import os



class easyGame():
    # for weights, a player is dummy if the corresponding value is with value 0
    def __init__(self, weight, func):
        self.num_player = len(weight)
        self.weight = np.array(weight)
        self.func = func

    def evaluate(self, subset):
        return self.func(np.dot(self.weight, subset))




class perturbGame():
    def __init__(self, game, *, noise=[1,0,1], seed=2023):
        self.seed = seed
        self.var11 = noise[0]
        self.var12 = noise[1]
        self.var22 = noise[2]
        self.check_noise_structure()

        self.num_player = game.num_player
        self.game = game
        self.noise_vec = self.generate_noise_vec()

    def check_noise_structure(self):
        assert self.var11 * self.var22 - self.var12 ** 2 > 0
        assert self.var11 > 0 and self.var22 > 0
        assert self.var11 > self.var12 and self.var22 > self.var12

    def generate_noise_vec(self):
        self.check_noise_structure()
        noise_cov = [[self.var11, self.var12], [self.var12, self.var22]]
        mean = [0,0]
        np.random.seed(self.seed)
        noise_vec = np.random.multivariate_normal(mean, noise_cov, size=self.num_player)
        np.random.seed(None)
        return noise_vec

    def refresh_noise_vec(self):
        self.noise_vec = self.generate_noise_vec()


    def evaluate(self, subset):
        clean_output = self.game.evaluate(subset)
        associated_noise = np.prod(self.noise_vec[np.arange(self.num_player),np.int64(subset)])
        return clean_output + associated_noise

class game_from_data():
    def __init__(self, dataset, *, model_type="logistic", metric="accuracy", game_seed=2023,
                 lr=0.2, r=1.0,
                 display=True,
                 **kwargs):
        self.game_seed = game_seed
        self.metric = metric
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_type = model_type
        self.r = r
        self.is_set = False
        self.display = display

        (self.X, self.y), (self.X_val, self.y_val), _ = load_dataset(dataset, display=display, **kwargs)
        if len(self.y) == 2:
            self.y = self.y[1]
        self.num_player = len(self.y)
        num_feature = self.X.shape[1]
        num_label = len(np.unique(self.y))

        if model_type == "logistic":
            self.model = models.LogisticRegression(num_feature, num_label).to(universe.device)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Check {model_type}")


        self.X_val = self.X_val.to(universe.device)
        self.y_val = self.y_val.to(universe.device)
        self.perm, self.X_train, self.y_train, self.target_prob = None, None, None, None
        # apply_seed() will initialize perm, X_train, y_train and log_prob under the given seed




    def evaluate(self, subset):
        idx = subset[self.perm]==1
        self.train_model(self.X_train[idx], self.y_train[idx])
        return self.output_score()

    def train_model(self, X, y):
        torch.manual_seed(self.game_seed)
        # use model.children(), it will omit some trainable layers deeper in components such as nn.Sequential
        # model.modules() can solve this issue, but it looks very likely it will bring additional running time
        # so choose to pay attentino on the structure of model so as to use children()
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        torch.manual_seed(int.from_bytes(os.urandom(8), byteorder="big"))
        if y.numel():
            for input, label in zip(X, y):
                self.optimizer.zero_grad()
                logit = self.model(input)
                loss = self.criterion(logit, label)
                if self.model_type == "logistic":
                    loss += torch.sum(torch.square(self.model.linear.weight)) * self.r / 2
                loss.backward()
                self.optimizer.step()


    def output_score(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_val)
        self.model.train()
        if self.metric == "accuracy":
            predict = np.argmax(logits.cpu().numpy(), 1)
            label = self.y_val.cpu().numpy()
            score = np.sum(predict==label) / len(label)
        elif self.metric == "cross_entropy":
            score = self.criterion(logits, self.y_val).cpu().numpy()
        elif self.metric == "KL":
            log_prob = F.log_softmax(logits, dim=1)
            score = -F.kl_div(log_prob, self.target_prob, reduction="batchmean").cpu().numpy()
        else:
            raise NotImplementedError(f"Check {self.metric}")
        return score


    def apply_seed(self):
        # after setting self.seed=seed, call self.apply_seed to initialize
        np.random.seed(self.game_seed)
        self.perm = np.random.permutation(self.num_player)
        np.random.seed(None)
        self.X_train = self.X[self.perm].to(universe.device)
        self.y_train = self.y[self.perm].to(universe.device)

        self.train_model(self.X_train, self.y_train)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_val)
        self.model.train()

        self.target_prob = F.softmax(logits, dim=1)
        predict = np.argmax(logits.cpu().numpy(), 1)
        label = self.y_val.cpu().numpy()
        acc = np.sum(predict == label) / len(label)
        self.is_set = True

        if self.display:
            print(f"the accuracy trained on full dataset is {acc} given seed {self.game_seed}")
        return acc, self.target_prob





# note multiple instances will be created in parallel running for one experiment setting.
# the differences are: 1) it won't print anything;
# 2) it assumes game_seed is given once and for all, i.e, self.apply_seed() is removed;
# 3) remove metric == "entropy_loss".
class game_for_parallel():
    def __init__(self, *, dataset, model_type="logistic", metric="accuracy", game_seed=2023, lr=0.2, r=1.0,
                 probability_target=None, **kwargs):
        self.game_seed = game_seed
        self.metric = metric
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_type = model_type
        self.r = r
        self.probability_target = probability_target
        self.acc = None

        (self.X, self.y), (self.X_val, self.y_val), _ = load_dataset(dataset, display=False, **kwargs)
        if len(self.y) == 2:
            self.y = self.y[1]
        self.num_player = len(self.y)
        num_feature = self.X.shape[1]
        num_label = len(np.unique(self.y))

        if model_type == "logistic":
            self.model = models.LogisticRegression(num_feature, num_label).to(universe.device)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Check {model_type}")


        self.X_val = self.X_val.to(universe.device)
        self.y_val = self.y_val.to(universe.device)
        np.random.seed(self.game_seed)
        self.perm = np.random.permutation(self.num_player)
        np.random.seed(None)
        self.X_train = self.X[self.perm].to(universe.device)
        self.y_train = self.y[self.perm].to(universe.device)

        if self.probability_target is None:
            self.train_model(self.X_train, self.y_train)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(self.X_val)
            self.model.train()

            self.probability_target = F.softmax(logits, dim=1)
            predict = np.argmax(logits.cpu().numpy(), 1)
            label = self.y_val.cpu().numpy()
            self.acc = np.sum(predict == label) / len(label)



    def evaluate(self, subset):
        idx = subset[self.perm]==1
        self.train_model(self.X_train[idx], self.y_train[idx])
        return self.output_score()

    def train_model(self, X, y):
        torch.manual_seed(self.game_seed)
        # use model.children(), it will omit some trainable layers deeper in components such as nn.Sequential
        # model.modules() can solve this issue, but it looks very likely it will bring additional running time
        # so choose to pay attentino on the structure of model so as to use children()
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        torch.manual_seed(int.from_bytes(os.urandom(8), byteorder="big"))
        if y.numel():
            for datum, label in zip(X, y):
                self.optimizer.zero_grad()
                logit = self.model(datum)
                loss = self.criterion(logit, label)
                if self.model_type == "logistic":
                    loss += torch.sum(torch.square(self.model.linear.weight)) * self.r / 2
                loss.backward()
                self.optimizer.step()


    def output_score(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_val)
        self.model.train()
        if self.metric == "accuracy":
            predict = np.argmax(logits.cpu().numpy(), 1)
            label = self.y_val.cpu().numpy()
            score = np.sum(predict==label) / len(label)
        elif self.metric == "cross_entropy":
            score = self.criterion(logits, self.y_val).cpu().numpy()
        elif self.metric == "KL":
            log_prob = F.log_softmax(logits, dim=1)
            score = -F.kl_div(log_prob, self.probability_target, reduction="batchmean").cpu().numpy()
        else:
            raise NotImplementedError(f"Check {self.metric}")
        return score

















