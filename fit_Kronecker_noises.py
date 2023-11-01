import numpy as np
import torch
torch.backends.deterministic = True
torch.backends.cudnn.benchmark = False
from utils.load_datasets import *
from utils.utilityFuncs import *
from tqdm import tqdm
from utils.funcs import *

def get_optimal_value(accs):
    global root, num_seed_ord, num_seed_game, dataset

    epochs = 100000
    interval_print = 100
    tol = 1e-8
    lr_opt = 2e-6

    def compute_loss():
        cov_appr = torch.zeros(n_valued+1, n_valued+1, dtype=torch.float64)
        for i in range(n_valued+1):
            for j in range(i, n_valued+1):
                cov_appr[i, j] = var_c ** i * var_b ** (j - i) * var_a ** (n_valued - j)
                if j > i:
                    cov_appr[j, i] = cov_appr[i, j]
        _q = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n_valued+1), cov_appr)
        return torch.distributions.kl.kl_divergence(_p, _q)

    traj_saved = os.path.join(root, f"traj_opt;dataset={dataset}.npz")
    if not os.path.exists(traj_saved):
        msg = "epoch {} | loss {:<.8f}, sigma11 {:<.5f}, sigma12 {:<.5f}, sigma22 {:<.5f}, weight {}"
        for i in range(num_seed_ord):
            index = np.arange(i * num_seed_game, (i + 1) * num_seed_game)
            takeout = np.mean(accs[:, index], axis=1, keepdims=True)
            accs[:, index] -= takeout
        mt_cov = torch.tensor(np.cov(accs,  bias=True), dtype=torch.float64)

        _p = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n_valued + 1), mt_cov)
        var_a = torch.tensor(1, requires_grad=True, dtype=torch.float64)
        var_c = torch.tensor(1, requires_grad=True, dtype=torch.float64)
        var_b = torch.tensor(0, requires_grad=True, dtype=torch.float64)
        optimizer = torch.optim.SGD((var_a, var_b, var_c), lr=lr_opt)

        sigma_list = np.empty((epochs, 3), dtype=np.float64)
        loss_list = np.empty(epochs, dtype=np.float64)

        loss = compute_loss()

        flag_up = 0
        flag_converge = 0
        loss_cur = loss.detach().numpy()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = compute_loss()
            loss_list[epoch] = loss.detach().numpy()
            sigma_list[epoch] = [var_a.detach().item(), var_b.detach().item(), var_c.detach().item()]
            if (epoch + 1) % interval_print == 0:
                sigma_cur = sigma_list[epoch]
                weight_cur = (sigma_cur[0] - sigma_cur[1]) / (sigma_cur[0] + sigma_cur[2] - 2 * sigma_cur[1])
                print(msg.format(epoch + 1, loss_list[epoch], sigma_cur[0], sigma_cur[1], sigma_cur[2], weight_cur))


            loss_pre = loss_cur
            loss_cur = loss.detach().numpy()
            diff = loss_pre - loss_cur
            if diff < 0:
                flag_up = 1
                print(f"the procedure ended because the loss just increased.")
                break

            diff_relative = diff / loss_pre
            if diff_relative < tol:
                flag_converge = 1
                break

        if not flag_up and not flag_converge:
            print(f"The procedure ended without meeting the specified tolerance {tol}.")

        sigma_list = sigma_list[:epoch + 1]
        loss_list = loss_list[:epoch + 1]

        index = np.argmin(loss_list)
        sigma_optimal = sigma_list[index]
        loss_optimal = loss_list[index]
        np.savez_compressed(traj_saved, sigma_optimal=sigma_optimal, loss_optimal=loss_optimal)
    else:
        with np.load(traj_saved) as data:
            sigma_optimal = data["sigma_optimal"]
            loss_optimal = data["loss_optimal"]

    weight_optimal = (sigma_optimal[0] - sigma_optimal[1]) / (sigma_optimal[0] + sigma_optimal[2] - 2 * sigma_optimal[1])
    msg_optimal = "optimal result | loss {:<.8f}, sigma11 {:<.5f}, sigma12 {:<.5f}, sigma22 {:<.5f}, weight {}"
    print(msg_optimal.format(loss_optimal, sigma_optimal[0], sigma_optimal[1], sigma_optimal[2], weight_optimal))


if __name__ == "__main__":
    num_seed_ord = 128
    num_seed_game = 128
    datasets = ["iris", "phoneme"]
    n_valued = 10
    n_val = 10
    lr = 1.0
    root = os.path.join("exp", "fit_Kronecker_noises")
    os.makedirs(root, exist_ok=True)

    list_seed_ord = np.arange(num_seed_ord)
    list_seed_game = np.arange(num_seed_game)
    subset = np.zeros(n_valued, dtype=bool)
    for dataset in datasets:
        print(f"working on dataset {dataset}...")
        print("estimating the emprical covariance matrix...")

        var_saved = os.path.join(root, f"covariance;dataset={dataset}.npz")
        if not os.path.exists(var_saved):
            (X_valued, y_valued), (X_val, y_val), num_class = load_dataset(dataset=dataset, n_valued=n_valued,
                                                                           n_val=n_val)
            game = gameTraining(X_valued=X_valued, y_valued=y_valued, X_val=X_val, y_val=y_val, num_class=num_class,
                                arch="logistic", lr=lr)
            acc_record = np.empty((n_valued + 1, num_seed_ord * num_seed_game), dtype=np.float64)
            for seed_ord in tqdm(list_seed_ord):
                with set_numpy_seed(seed_ord):
                    pi = np.random.permutation(n_valued)
                for game_seed in list_seed_game:
                    game.game_seed = game_seed
                    subset.fill(0)
                    acc_record[0, seed_ord * num_seed_game + game_seed] = game.evaluate(subset)
                    for i in range(n_valued):
                        subset[pi[i]] = 1
                        acc_record[i + 1, seed_ord * num_seed_game + game_seed] = game.evaluate(subset)
            np.savez_compressed(var_saved, acc_record=acc_record)
        else:
            with np.load(var_saved) as data:
                acc_record = data["acc_record"]

        print("optimizing the KL divergence for fitting a Kronecker noise to the estimated covariance matrix...")
        get_optimal_value(acc_record)



