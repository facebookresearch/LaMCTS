<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LA-MCTS/logo.png?raw=true' width="600">
</p>

# Latent Action Monte Carlo Tree Search (LA-MCTS)

LA-MCTS is a new MCTS based derivative-free meta-solver. Compared to Bayesian optimization, and evolutionary algorithm, it learns to partition the search space, thereby 🌟 finding a better solution with fewer samples.

# Contributors
Linnan Wang (First Author), Yuandong Tian (Principal Investigator), Yiyang Zhao, Saining Xie, Teng Li and Rodrigo Fonesca.

\* **Linnan Wang is looking for a full-time position**. 

# What's in this release?

This release contains our implementation of LA-MCTS and its application to Neural Architecture Search (LaNAS), but it can also be applied to large-scale hyper-parameter optimization, reinforcement learning, scheduling, optimizing computer systems, and many others.

## Neural Architecture Search (NAS) 
- <a href="./LaNAS/LaNAS_NASBench101">**Evaluation on NASBench-101** </a>: Evaluating LaNAS on NASBench-101 on your laptop without training models. 

- <a href="./LaNAS/LaNet">**Our Searched Models, LaNet**</a>: SoTA results: • 99.03% on CIFAR-10 • 77.7% @ 240MFLOPS on ImageNet.

- <a href="./LaNAS/one-shot_LaNAS">**One/Few-shot LaNAS**</a>: Using a supernet to evaluate the model, obtaining results within a few GPU days.

- <a href="./LaNAS/Distributed_LaNAS">**Distributed LaNAS**</a>: Distributed framework for LaNAS, usable with hundreds of GPUs.

- <a href="./LaNAS/LaNet">**Training heuristics used**</a>: We list all tricks used in ImageNet training to reach SoTA. 

## Black-box optimization 
- <a href="./LA-MCTS">**Performance with baselines**</a>: 1 minute evaluations of LA-MCTS v.s. Bayesian Optimization and Evolutionary Search.

- <a href="./LA-MCTS">**Mujoco Experiments**</a>: LA-MCTS on Mujoco environment. 


#  Project Logs
## Building the MCTS based NAS agent

>Inspired by AlphaGo, we build the very first NAS search algorithm based on Monte Carlo Tree Search (MCTS) in 2017, namely AlphaX. The action space is fixed (layer-by-layer construction) and MCTS is used to steer towards promising search regions. We showed the Convolutional Neural Network designed by AlphaX improve the downstream applications such as detection, style transfer, image captioning, and many others.

<a href="https://arxiv.org/pdf/1805.07440.pdf">Neural Architecture Search using Deep Neural Networks and Monte Carlo Tree Search</a> </br>
AAAI-2020, [<a href="https://github.com/linnanwang/AlphaX-NASBench101">code</a>]</br>
Linnan Wang (Brown), Yiyang Zhao(WPI), Yuu Jinnai(Brown), Yuandong Tian(FAIR), Rodrigo Fonseca(Brown)</br>

## From AlphaX to LaNAS
>On AlphaX, we find that different action space used in MCTS significantly affects the search efficiency, which motivates the idea of learning action space for MCTS on the fly during training.
This leads to LaNAS. 
LaNAS uses a linear classifier at each decision node of MCTS to learn good versus bad actions, and evaluates each leaf node, which now represents a subregion of the search space rather than a single architecture, by a uniform random sampling one architecture and evalute. 
The first version of LaNAS implemented a distributed system to perform NAS by training every such samples from scratch using 500 GPUs. 
The second version of LaNAS, called one-shot LaNAS, uses a single one-shot subnetwork to evaluate the quality of samples, trading evaluation efficiency with accuracy. 
One-shot LaNAS finds a reasonable solution in a few GPU days.  

<a href="https://linnanwang.github.io/latent-actions.pdf">Sample-Efficient Neural Architecture Search by Learning Action Space for Monte Carlo Tree Search</a> </br>
Linnan Wang (Brown), Saining Xie (FAIR), Teng Li(FAIR), Rodrigo Fonesca (Brown), Yuandong Tian (FAIR)</br>

## From LaNAS to a generic solver LA-MCTS
> Since LaNAS works very well on NAS datasets, e.g. NASBench-101, and the core of the algorithm can be easily generalized to other problems, we extend it to be a generic solver for black-box function optimization. 
LA-MCTS further improves by using a nonlinear classifier at each decision node in MCTS and use a surrogate (e.g., a function approximator) to evaluate each sample in the leaf node. 
The surrogate can come from any existing Black-box optimizer (e.g., Bayesian Optimization). 
The details of LA-MCTS can be found in the following paper.  

<a href="https://arxiv.org/abs/2007.00708">Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search</a> </br>
NeurIPS 2020 </br>
Linnan Wang (Brown University), Rodrigo Fonesca (Brown University), Yuandong Tian (Facebook AI Research) </br>

## From one-shot NAS to few-shot NAS
> To overcome issues of one-shot NAS, we propose few-shot NAS that uses multiple supernets, each covering different regions of the search space specified by the intermediate of the search tree. Extensive experiments show that few-shot NAS significantly improves upon one-shot methods. See the paper below for details.

<a href="https://arxiv.org/abs/2006.06863">Few-shot Neural Architecture Search</a> </br> [<a href="https://github.com/aoiang/few-shot-NAS">code</a>] </br>
Yiyang Zhao (WPI), Linnan Wang (Brown), Yuandong Tian (FAIR), Rodrigo Fonseca (Brown), Tian Guo (WPI)


## License
LA-MCTS is under [CC-BY-NC 4.0 license](./LICENSE).
