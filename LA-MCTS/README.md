# Latent Action Monte Carlo Tree Search (LA-MCTS)
LA-MCTS is a meta-algortihm that partitions the search space for black-box optimizations. LA-MCTS progressively learns to partition and explores promising regions in the search space, so that solvers such as Bayesian Optimizations (BO) can focus on promising subregions, mitigating the over-exploring issue in high-dimensional problems. 

<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LA-MCTS/meta_algorithms.png?raw=true' width="300">
</p>

Please reference the following publication when using this package. ArXiv <a href="https://arxiv.org/abs/2007.00708">link</a>.

```
@article{wang2020learning,
  title={Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search},
  author={Wang, Linnan and Fonseca, Rodrigo and Tian, Yuandong},
  journal={NeurIPS},
  year={2020}
}
```

## Run LaMCTS and Baselines in test functions (1 minute tutorial)
For 1 minute evaluation of Bayesian Optimizations (BO) and Evolutionary Algorithm (EA) v.s. LA-MCTS boosted BO, please follow the procedures below. 

Here we test on <a href="https://www.sfu.ca/~ssurjano/ackley.html">Ackley</a> or <a href="https://www.sfu.ca/~ssurjano/levy.html">Levy</a> in 10 dimensions; Please run multiple times to compare the average performance.

- ***Evaluate LA-MCTS boosted Bayesian Optimization***: using GP surrogate, EI acuqusition, plus LA-MCTS.
```
cd LA-MCTS
python run.py --func ackley --dims 10 --iterations 100
```

- ***Evaluate Bayesian Optimization***: using GP surrogate, EI acuqusition.
```
pip install scikit-optimize
cd LA-MCTS-baselines/Bayesian-Optimization
python run.py --func ackley --dims 10 --iterations 100
```

- ***Evaluate Evolutionary Algorithm***: using NGOpt from Nevergrad.
```
pip install nevergrad
cd LA-MCTS-baselines/Nevergrad
python run.py --func ackley --dims 10 --iterations 100
```


## How to use LA-MCTS to optimize your own function?
Please wrap your function into a class defined as follows; functions/functions.py provides a few examples.

```
class myFunc:
    def __init__(self, dims=1):
        self.dims    = dims                   #problem dimensions
        self.lb      =  np.ones(dims)         #lower bound for each dimensions 
        self.ub      =  np.ones(dims)         #upper bound for each dimensions 
        self.tracker = tracker('myFunc')      #defined in functions.py

    def __call__(self, x):
        # some sanity check of x        
        f(x) = myFunc(x)
        self.tracker.track( f(x), x )        
        return f(x)
```

After defining your function, e.g. f = func(), minimizing f(x) is as easy as passing f into MCTS.
```
f = myFunc()
agent = MCTS(lb = f.lb,     # the lower bound of each problem dimensions
             ub = f.ub,     # the upper bound of each problem dimensions
             dims = f.dims, # the problem dimensions
             ninits = 40,   # the number of random samples used in initializations 
             func = f       # function object to be optimized
             )
agent.search(iterations = 100)
```
Please check `run.py`. 


## What it can and cannot do?
In this release, the codes only support optimizing continuous black box functions.

## Tuning LA-MCTS

### **Cp**: controling the amount of exploration, MCTS.py line 27.
> <b>We set Cp = 0.1 * max of f(x) </b>. 

> For example, if f(x) measures the test accuracy of a neural network x, Cp = 0.1. But the best performance should be tuned in specific cases. A large Cp encourages LA-MCTS to visit bad regions more often (exploration), and a small Cp otherwise. LA-MCTS degenreates to random search if Cp = 0, while LA-MCTS degenerates to a pure greedy based policy, e.g. regression tree, at Cp = 0. Both are undesired. 

### **Leaf Size**: the splitting threshold (θ), MCTS.py line 38.
> <b> We set θ ∈ [20, 100] </b> in our experiments.

> the splitting threshold controls the speed of tree growth. Given the same \#samples, smaller θ leads to a deeper tree. If Ω is very large, more splits enable LA-MCTS to quickly focus on a small promising region, and yields good results. However, if θ is too small, the performance and the boundary estimation of the region become more unreliable. 

### **SVM kernel**: the type of kernels used by SVM, Classifier.py line 35.
> <b> kernel can be 'linear', 'poly', 'rbf'</b>

> From our experiments, linear kernel is the fastest, but rbf or poly are generally producing better results. If you want to draw > 1000 samples, we suggest using linear kernel, and rbf and poly otherwise.

## Mujoco tasks and Gym Games.
<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LA-MCTS/lunar_landing.gif?raw=true' width="400">
</p>

- **Run Lunarlanding**:
1. Install gym and start running.
```
pip install gym
python run.py --func lunar --samples 500
```
Copy the final "current best x" from the output to visualize your policy.

2. Visualize your policy
```
cd functions
Replace our policy value to your learned policy from the previous step, i.e. policy = np.array([xx]).
python visualize_policy.py
```
- **Run MuJoCo**:
<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LA-MCTS/swimmer.gif?raw=true' width="400"> &nbsp; &nbsp; &nbsp;
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LA-MCTS/hopper.gif?raw=true' width="400">
</p>

1. Setup mujoco-py, see <a href="https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key">here</a>.

2. Download TuRBO from <a href="https://github.com/uber-research/TuRBO">here</a>. Extract and find the turbo folder.

3. Move revision.patch to the turbo folder, 
```
cp revision.patch TuRBO-master/turbo
cd TuRBO-master/turbo
patch -p1 < revision.patch
cd ../..
mv TuRBO-master/turbo ./turbo_1
```
4. Open file Classifier.py, uncomment line 23, 343->368. Open file MCTS.py, change line 47, solver_type from bo to turbo.

5. Now it is ready to run,
```
python run.py --func swimmer --iterations 1000
```

<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LA-MCTS/mujoco_performance.png?raw=true' width="1000">
</p>

## Possible Extensions
In MCTS.py line 229, the select returns a path and a leaf to bound the sampling space. We used ```get_sample_ratio_in_region``` in Classifier.py to acquire samples in the selected partition. Other sampler can also be used. 

LAMCTS can be used together with any value function / evaluator / cost models.
