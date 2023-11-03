# Robust Data Valuation with Weighted Banzhaf Values

This repository is to provide an implementation to replicate all the results reported in the paper *Robust Data Valuation with Weighted Banzhaf Values* accepted to NeurIPS 2023: 
> @inproceedings{LiYu23,
> 
>   title       = {Robust Data Valuation with Weighted {B}anzhaf Values},
> 
>   author      = {W. Li and Y. Yu},
> 
>   booktitle   = {Advances in Neural Information Processing Systems {(NeurIPS)}},
> 
>   year        = {2023},
> 
>   url         = {https://openreview.net/forum?id=u359tNBpxF},
> 
> }

## Quick Start
All user-specified arguments are contained in args.py. In our paper, using n different random seeds means the set of random seeds we used is {0, 1, 2, ..., n-1}. To replicate the additional results on size-200 datasets reported in the Appendix, take the Shapley value on the two datasets 2dplanes and gas for example, one can run the following command,

`
python main.py -n_process 50 -dir 200data -value shapley -estimator permutation -dataset 2dplanes gas -game_seed 0 1 2 3 4 5 6 7 8 9 -n_valued 200 -n_val 200 -flip_percent 0.1 -num_sample_avg 1000 -batch_size_avg 20
`

Specifically, n_process is the number of cpus used for parallel computing (we have set the number of threads for each process to be 1), the default of which is 1, and thus one has to make sure that the number of available cpus is at least n_process; the number of processes that will be run in total is -(-num_sample_avg // batch_size_avg) if estimator != "exact_value"; in other words, n_process should be no more than -(-num_sample_avg // batch_size_avg); 
the total number of utility evaluations is the product of num_sample_avg and n_valued if estimator != "exact_value";
dir is to specify a directory (which will be created in the directory exp) that will store results; game_seed refers to the random seed used to decide the randomness contained in utility functions. n_valued is the number of data being valuated, while n_val is the number of data used for reported the performance of trained models.

To approximate weighted Banzhaf values, the command is, e.g.,

`
python main.py -n_process 50 -dir 200data -value weighted_banzhaf -param 0.1 0.2 0.3 -estimator maximum_sample_reuse -dataset 2dplanes gas -game_seed 0 1 2 3 4 5 6 7 8 9 -n_valued 200 -n_val 200 -flip_percent 0.1 -num_sample_avg 1000 -batch_size_avg 20
`

It will estimate the corresponding 0.1-, 0.2- and 0.3-weighted Banzhaf values. For Beta Shapley values, it is 

`
python main.py -n_process 50 -dir 200data -value beta_shapley -param (16,1) (4,1) (1,4) (1,16) -estimator weighted_sampling_lift -dataset 2dplanes gas -game_seed 0 1 2 3 4 5 6 7 8 9 -n_valued 200 -n_val 200 -flip_percent 0.1 -num_sample_avg 1000 -batch_size_avg 20
`

After all the estimated values are generated, to report the results of ranking consistency and noisy label detection, just run

`
python plot_ranking_and_noisy.py -dir 200data
`

The produced figures are all saved in the directory fig, which will be created automatically.
To replicate all the reported results in the experiments of ranking consistency as well as noisy label detection, just plug in the choices of arguments reported in our paper. Specifically, num_sample_avg = 200 if n_valued = 2000, and it is 400 if n_valued = 1000.

## Compare Estimators
To have Figure 2 reported in the paper, one can run

`
python compare_estimators.py -dataset 2dplanes -dir compare_estimators -n_valued 16 -value weighted_banzhaf -param 0.8 -lr 1.0 -estimator_seed 0 1 2 3 4 5 6 7 8 9 -num_sample_avg 1500 -batch_size_avg 50 -interval_track_avg 30
`

Note that one can further specify n_process for parallel computing, the maximum of which will not be idle in this case is 30.

## Synthetic Noises
The following command will produce the first column of Figure 3 in the paper

`
python plot_synthetic_noises.py -dataset spambase
`

To speed up, n_process can be specified (the maximum is 21 in this case). Note the calculation for the number of processes that will be run is different if estimator=="exact_value".

## Authentic Noises
To generate the parameters for the predicted robust weighted Bazhaf values, one can run

`
python fit_Kronecker_noises.py 
`

Note that the final results have been included in args.py so that they will be automatically filled in if param==-1. Since the default of param is -1, to replicate the reported results for the one tagged as robust, one can run

`
python main.py -dir authentic_noises -dataset iris phoneme -n_valued 10 -n_val 10 -flip_percent 0 -lr 1.0 -value weighted_banzhaf -estimator exact_value game_seed 0 1 2 3 4 5 6 7 8 9
` 

After all the exact values are calculated, the reported figures can be obtained by running

`
python plot_authentic_noises.py -dir authentic_noises
` 

