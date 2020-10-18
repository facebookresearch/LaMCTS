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
Recent works show very excellent results using a predictor such as Graph Neural Network. These approaches are the same as the surrogate model used in Bayesian Optimization, except for using different predictors. The main issue of predictor based methods is that these methods need to predict every architecture in the search space to perform well, and misses an acquisition (e.g. in Bayesian Optimization) to make the trade-off between exploration and exploitation. See this <a href="https://github.com/linnanwang/MLP-NASBench-101">repository</a>, a simple MLP can perform well (< 1000 samples) if it predicts on all the architectures in NASBench (4.2x10^5 architectures). However, the MLP performs the worst on a supernet that renders a search space of 3.5x10^21 architectures, because it is impossible to predict every architectures. See Fig.6 in <a href="https://linnanwang.github.io/latent-actions.pdf">LaNAS</a>.

In LaNAS, we used MLP to predict samples to assign an architecture to a partition. This is an engineering simplification and can be replaced by a hit-and-run sampler, i.e. sampling from a convex polytope. However, we do replace this with a sampling method in one-shot LaNAS.
