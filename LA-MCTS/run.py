# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from functions.functions import *
from functions.mujoco_functions import *
from lamcts import MCTS
import argparse




parser = argparse.ArgumentParser(description='Process inputs')
parser.add_argument('--func', help='specify the test function')
parser.add_argument('--dims', type=int, help='specify the problem dimensions')
parser.add_argument('--iterations', type=int, help='specify the iterations to collect in the search')


args = parser.parse_args()

f = None
iteration = 0
if args.func == 'ackley':
    assert args.dims > 0
    f = Ackley(dims =args.dims)
elif args.func == 'levy':
    assert args.dims > 0
    f = Levy(dims = args.dims)
elif args.func == 'lunar': 
    f = Lunarlanding()
elif args.func == 'swimmer':
    f = Swimmer()
elif args.func == 'hopper':
    f = Hopper()
else:
    print('function not defined')
    os._exit(1)

assert f is not None
assert args.iterations > 0


# f = Ackley(dims = 10)
# f = Levy(dims = 10)
# f = Swimmer()
# f = Hopper()
# f = Lunarlanding()

agent = MCTS(
             lb = f.lb,              # the lower bound of each problem dimensions
             ub = f.ub,              # the upper bound of each problem dimensions
             dims = f.dims,          # the problem dimensions
             ninits = f.ninits,      # the number of random samples used in initializations 
             func = f,               # function object to be optimized
             Cp = f.Cp,              # Cp for MCTS
             leaf_size = f.leaf_size, # tree leaf size
             kernel_type = f.kernel_type, #SVM configruation
             gamma_type = f.gamma_type    #SVM configruation
             )

agent.search(iterations = args.iterations)

"""
FAQ:

1. How to retrieve every f(x) during the search?

During the optimization, the function will create a folder to store the f(x) trace; and
the name of the folder is in the format of function name + function dimensions, e.g. Ackley10.

Every 100 samples, the function will write a row to a file named results + total samples, e.g. result100 
mean f(x) trace in the first 100 samples.

Each last row of result file contains the f(x) trace starting from 1th sample -> the current sample.
Results of previous rows are from previous experiments, as we always append the results from a new experiment
to the last row.

Here is an example to interpret a row of f(x) trace.
[5, 3.2, 2.1, ..., 1.1]
The first sampled f(x) is 5, the second sampled f(x) is 3.2, and the last sampled f(x) is 1.1 

2. How to improve the performance?
Tune Cp, leaf_size, and improve BO sampler with others.

"""
