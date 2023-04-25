from dotmap import DotMap
import itertools
import os
import numpy as np

class args_parser:
    def __init__(self):
        self.args_default = DotMap(
            dict(
                estimator = dict(
                    value = "weighted_banzhaf",
                    param = 0.5,
                    method = "maximum_sample_reuse",
                    num_eval_per_player=4000,
                    track_interval_per_player=250,
                    estimator_seed=2023
                ),
                game = dict(
                    dataset = "cpu",
                    metric = "accuracy",
                    game_seed = 2023,
                    model_type = "auto",
                    lr = "auto",
                    r = "auto"
                ),
                dataset = dict(
                    n_valued = 200,
                    n_val = 200,
                    n_test = 1,
                    dataset_seed = 2023,
                    flip_percent = None,
                    flip_seed = 2023
                )
            )
        )
        self.args_auto = {
            "covertype" : dict(
                model_type = "logistic",
                lr = 0.14,
                r = 0.03
            ),
            "cpu" : dict(
                model_type = "logistic",
                lr = 0.67,
                r = 0.0
            ),
            "2dplanes" : dict(
                model_type = "logistic",
                lr = 0.46,
                r = 0.0
            ),
            "pol" : dict(
                model_type = "logistic",
                lr = 0.39,
                r = 0.0
            ),
            "wind" : dict(
                model_type = "logistic",
                lr = 1.0,
                r = 0.0
            ),
            "phoneme" : dict(
                model_type = "logistic",
                lr = 0.62,
                r = 0.018
            ),
            "vehicle" : dict(
                model_type = "logistic",
                lr = 0.11,
                r = 0.0
            ),
            "fraud": dict(
                model_type="logistic",
                lr=2.2,
                r=0.001
            )
        }


    def product_args_list(self, *, path_variable, args_list):
        args_list = self.fill_args_list(args_list)
        args_list = self.standardize_args_list(args_list)

        args_all = []
        for d in args_list:
            main_path = ""
            d_flatten = dict()
            marker = []
            for div in d.values():
                for key, value in div.items():
                    if len(value) > 1 or key in path_variable:
                        marker.append(key)
                    else:
                        main_path += key + "=" + str(value[0]) + ";"

                d_flatten.update(div)
            args_flatten = self.dict_product(d_flatten)
            main_path = main_path[:-1]

            for args in args_flatten:
                sub_path = ""
                for key, value in args.items():
                    if key in marker:
                        sub_path += key + "=" + str(value) + ";"
                sub_path = sub_path[:-1]
                args.update(save_path=os.path.join(main_path, sub_path))
            args_all += args_flatten

        args_all = self.format_args(args_all)
        args_all = self.auto_complete(args_all)
        return args_all


    def fill_args_list(self, args_list):
        args_list_filled = []
        for args_dict in args_list:
            args_cur = self.args_default.copy()
            for key_sub, value_sub in args_dict.items():
                d_cur = args_cur[key_sub]
                for key, value in value_sub.items():
                    d_cur[key] = value
            args_list_filled.append(args_cur)

        del args_list
        return args_list_filled




    def standardize_args_list(self, args_list):
        for d in args_list:
            for div in d.values():
                for key, value in div.items():
                    if isinstance(value, np.ndarray):
                        div[key] = value.tolist()
                    elif not isinstance(value, list):
                        div[key] = [value]
        return args_list

    @staticmethod
    def dict_product(d):
        keys = d.keys()
        vals = d.values()
        args_all = []
        for instance in itertools.product(*vals):
            args_all.append(dict(zip(keys, instance)))
        return args_all

    def format_args(self, args_all):
        args_all_formatted = []
        for args in args_all:
            args_cur = self.args_default.copy()
            for d in args_cur.values():
                for key in d.keys():
                    d[key] = args[key]
            args_cur.update(save_path=args["save_path"])
            args_all_formatted.append(args_cur)

        del args_all
        return args_all_formatted



    def auto_complete(self, args_all):
        for args in args_all:
            game = args.game
            for key, value in game.items():
                if value == "auto":
                    game[key] = self.args_auto[game.dataset][key]
        return args_all
