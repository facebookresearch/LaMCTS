# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from functions import *
import argparse
import os
from skopt import gp_minimize
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
    f = Levy(dims = args.dims)
else:
    print('function not defined')
    os._exit(1)

assert args.dims > 0
assert f is not None
assert args.iterations > 0

lower = f.lb
upper = f.ub

bounds = []
for idx in range(0, len(f.lb) ):
    bounds.append( ( float(f.lb[idx]), float(f.ub[idx])) )

res = gp_minimize(f,                          # the function to minimize
                  bounds,                     # the bounds on each dimension of x
                  acq_func="EI",              # the acquisition function
                  n_calls=args.iterations,
                  acq_optimizer = "sampling", # using sampling to be consisent with our BO implementation
                  n_initial_points=40
                  )