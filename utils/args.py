import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset-list", nargs='+', type=str, default="2dplanes")
    # "2dplanes", "bank-marketing", "bioresponse", "covertype", "cpu", "credit", "default", "diabetes", "fraud", "gas",
    # "har", "letter", "optdigits", "pendigits", "phoneme", "pol", "satimage", "segment", "spambase", "texture", "wind",
    # "MNIST", "FMNIST"
    parser.add_argument("-n_valued", type=int, default=200,
                        help="the number of data being valuated, data that constitutes D_{tr}")
    parser.add_argument("-n_val", type=int, default=200,
                        help="the number of data used for reporting the performance of trained models")
    parser.add_argument("-flip_percent", type=float, default=0.2, help="the percent of data, i.e., D_{tr}, to be flipped")
    parser.add_argument("-game_func", type=str, default="gameTraining",
                        help="the class that defines a type of utility function")
    # "gameTraining" or "gamePerturbed"
    parser.add_argument("-arch", type=str, default="logistic",
                        help="the architecture of trainable models for defining utility functions")
    # "logistic" or "LeNet"
    parser.add_argument("-lr", type=float, default=0, help="the learning rate used by utility functions")
    # 0 to use the fine-tuned one
    parser.add_argument("-value", type=str, default="shapley", help="which semi-value to employ")
    # "beta_shapley" or "shapley" or "weighted_banzhaf"
    parser.add_argument("-param", "--param-list", nargs='+', type=eval, default=None,
                        help="the parameter that specifies a semi-value")
    # param in [0, 1] if value=="weighted_banzhaf"
    # param = (alpha, beta), where alpha, beta >= 1 and are integers, if value="beta_shapley"
    # param could be anything if value="shapley"
    # if value="weighted_banzhaf", param="auto" is to use the predicted paramters from the fitted Kronecker noises
    parser.add_argument("-estimator", type=str, default="permutation", help="the estimator use for approximating values")
    # "maximum_sample_reuse" if value="weighted_banzhaf"
    # "permutation" if value="shapley"
    # "sampling lift" or "weighted_sampling_lift" for any value
    parser.add_argument("-num_sample_avg", type=int, default=100,
                        help="the number of averaged utility evaluations for approximation")
    parser.add_argument("-batch_size_avg", type=int, default=2,
                        help="the number of averaged utility evaluations each process will run at a time")
    parser.add_argument("-interval_track_avg", type=int, default=20,
                        help="the number of averaged utility evaluations to record while approximating")
    parser.add_argument("-dataset_seed",  type=int, default=2023, help="the randomness used to split datasets")
    parser.add_argument("-flip_seed", type=int, default=2023, help="the randomness used to flip labels")
    parser.add_argument("-game_seed", "--game_seed-list", nargs='+', type=int, default=2023,
                        help="the randomness used by utility functions")
    parser.add_argument("-estimator_seed", "--estimator_seed-list", nargs='+', type=int, default=2023,
                        help="the randomness used by approximation")


    parser.add_argument("-n_process", type=int, default=1, help="the number of processes for parallel computing")
    parser.add_argument("-dir", type=str, default="tmp", help="directory to store results")

    args = vars(parser.parse_args())
    args["dataset"] = args.pop("dataset_list")
    args["param"] = args.pop("param_list")
    args["game_seed"] = args.pop("game_seed_list")
    args["estimator_seed"] = args.pop("estimator_seed_list")
    return args


def fill_auto(args_all):
    global lr_auto, param_robust
    for args in args_all:
        if args["lr"] == 0:
            args["lr"] = lr_auto[args["n_valued"]][args["dataset"]]
        if args["param"] == "auto":
            assert args["value"] == "weighted_banzhaf"
            args["param"] = param_robust[args["dataset"]]

# fine-tuned learning rate for each dataset
# 1000 means the one for n_valued=1000 while it is always that n_val=200, game_seed=dataset_seed=2023, flip_percent=0
lr_auto = {
    200: {
        "2dplanes": 0.06,
        "bank-marketing": 0.01,
        "bioresponse": 0.5,
        "covertype": 0.02,
        "cpu": 0.06,
        "credit": 0.02,
        "default": 0.53,
        "diabetes": 0.02,
        "fraud": 0.05,
        "gas": 0.02,
        "har": 0.01,
        "letter": 0.17,
        "optdigits": 0.05,
        "pendigits": 0.11,
        "phoneme": 0.02,
        "pol": 0.11,
        "satimage": 0.04,
        "segment": 0.1,
        "spambase": 0.02,
        "texture": 0.08,
        "wind": 0.04,
    },
    1000: {
        "2dplanes": 0.01,
        "bank-marketing": 0.01,
        "bioresponse": 0.85,
        "covertype": 0.01,
        "cpu": 0.02,
        "default": 0.02,
        "gas": 0.02,
        "har": 0.01,
        "letter": 0.09,
        "optdigits": 0.03,
        "pendigits": 0.14,
        "phoneme": 0.01,
        "pol": 0.03,
        "satimage": 0.01,
        "segment": 0.07,
        "spambase": 0.03,
        "texture": 0.07,
        "wind": 0.02,
    },
    2000: {
        "MNIST": 0.01,
        "FMNIST": 0.02,
    }
}

# the predicted param for weighted Banzhaf values using fitted Kronecker noises.
param_robust = {
    "phoneme": 0.5341478928613749,
    "iris": 0.525456490955982
}




