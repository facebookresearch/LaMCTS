# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
import gym
import json
import os


class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.curt_best = float("inf")
        self.curt_best_x = None
        self.foldername = foldername
        try:
            os.mkdir(foldername)
        except OSError:
            print ("Creation of the directory %s failed" % foldername)
        else:
            print ("Successfully created the directory %s " % foldername)
        
    def dump_trace(self):
        trace_path = self.foldername + '/result' + str(len( self.results) )
        final_results_str = json.dumps(self.results)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')
            
    def track(self, result, x = None):
        if result < self.curt_best:
            self.curt_best = result
            self.curt_best_x = x
        print("")
        print("="*10)
        print("iteration:", self.counter, " total samples:", len(self.results))
        print("="*10)
        print("current best f(x):", self.curt_best)
        print("current best x:", np.around(self.curt_best_x, decimals=1))
        self.results.append(self.curt_best)
        self.counter += 1
        if len(self.results) % 100 == 0:
            self.dump_trace()            
    
class Levy:
    def __init__(self, dims=1):
        self.dims    = dims
        self.lb      = -10 * np.ones(dims)
        self.ub      =  10 * np.ones(dims)
        self.counter = 0
        print("####dims:", dims)
        self.tracker = tracker('Levy'+str(dims))

    def __call__(self, x):
        x = np.array(x)
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = []
        for idx in range(0, len(x)):
            w.append( 1 + (x[idx] - 1) / 4 )
        w = np.array(w)
        
        term1 = ( np.sin( np.pi*w[0] ) )**2;
        
        term3 = ( w[-1] - 1 )**2 * ( 1 + ( np.sin( 2 * np.pi * w[-1] ) )**2 );
        
        
        term2 = 0;
        for idx in range(1, len(w) ):
            wi  = w[idx]
            new = (wi-1)**2 * ( 1 + 10 * ( np.sin( np.pi* wi + 1 ) )**2)
            term2 = term2 + new
        
        result = term1 + term2 + term3

        self.tracker.track( result, x )

        return result
        
class Ackley:
    def __init__(self, dims=3):
        self.dims   = dims
        self.lb    = -5 * np.ones(dims)
        self.ub    =  10 * np.ones(dims)
        self.counter = 0
        self.tracker = tracker('Ackley'+str(dims))
        

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        
        self.tracker.track( result, x )
                
        return result   