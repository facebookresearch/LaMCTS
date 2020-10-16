# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from functions import *
import argparse
import os
import nevergrad as ng

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


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return np.ravel(xx)
    
init = from_unit_cube( np.random.rand(f.dims).reshape(1,-1), f.lb, f.ub)

param = ng.p.Array(init=init ).set_bounds(f.lb, f.ub)

optimizer = ng.optimizers.NGOpt(parametrization=param, budget=args.iterations)

recommendation = optimizer.minimize(f)