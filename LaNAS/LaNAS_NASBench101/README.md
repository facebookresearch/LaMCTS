## LaNAS on NASBench-101

This folder has everything you need to test LaNAS on NASBench-101. Before you start, please download a preprocessed NASBench-101 from <a href="https://github.com/linnanwang/AlphaX-NASBench101">AlphaX</a> (see section <b>Download the dataset</b>).
```
place nasbench_dataset in LaNAS/LaNAS_NASBench101
python MCTS.py
```
The program will stop once it finds the global optimum. The search usually takes a few hours to a day. Once it finishes, The search results will be written into the last row in results.txt. Here is an example to interpret the result.

>[[0.9313568472862244, 1], <b>[0.9326255321502686, 47]</b>, [0.9332265059153239, 51], [0.9342948794364929, 72], [0.9343950351079305, 76], [0.93873530626297, 81], [0.9388020833333334, 224], [0.9388688604036967, 472], [0.9391693472862244, 639], [0.9407051205635071, 740], [0.9420072237650553, 831], [0.9423410892486572, 1545], [0.943175752957662, 3259]]

This means before the 47th sample, the best validation accuracy is 0.9326255321502686; and in this case LaNAS finds the best network using 3259 samples. The results of a new experiment will be appended as a new row in results.txt.

We also provided results of our past runs in <b>our_past_results.txt</b>, you can use that for comparisions; but feel free to reproduce the results with this release.

## About NASBench-101
Please check <a href="https://github.com/linnanwang/AlphaX-NASBench101">AlphaX</a> to see our encoding of NASBench.

## About Predictor based Search Methods

<b>The simplest way to verify "why predictor not working" is to try it on the 10 dimensional continuous Ackley function (in functions.py in LA-MCTS). In practice, the search space has 10^30 architectures, you CANNOT predict every one; and whatever predictor you use will fail.</b>

<b>Why predictor works well on NASBench?</b>The main issue of predictor based methods is that these methods need to predict every architecture in the search space to perform well, and misses an acquisition (e.g. in Bayesian Optimization) to make the trade-off between exploration and exploitation. NASBench only has 4.2*10^5 networks, which can be predicted in a second. We show a simple MLP can perform well (< 1000 samples) if it predicts on all the architectures in NASBench. Besides, the following figure visualizes the distribution of network-accuracy on NASBench-101, y in log scale. So it is not surprising to see even using random search can find a reasonable result, since most networks are pretty good.

<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LaNAS/nasbench_distribution.png?raw=true' width="400">
</p>

In fact, the purpose of neural predictors, e.g. Graph Neural Network-based predictors, are very similar to the surrogate model (e.g. Gaussian Process) used in Bayesian Optimizations. The original NASBench-101 paper chose a set of very good baselines for comparisons.


<b>Why predictor works in NASNet or EfficientNet search space?</b> These search space are constructed under very strong prior experience; and the final network accuracy can be hack with bag of tricks listed <a href="https://github.com/facebookresearch/LaMCTS/tree/master/LaNAS/LaNet">here</a>.

In this implementation, we used MLP to predict samples to assign an architecture to a partition. This is an engineering simplification and can be replaced by a hit-and-run sampler, i.e. sampling from a convex polytope. However, we do replace this with a sampling method in one-shot LaNAS, i.e. Fig.6(c) in LaNAS; and also see LA-MCTS. Thank you.
