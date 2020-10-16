# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
import gym
import json
import os

class Swimmer:
    
    def __init__(self):
        self.policy_shape = (2, 8)
        self.mean         = 0
        self.std          = 1
        self.dims         = 16
        self.lb           = -1 * np.ones(self.dims)
        self.ub           =  1 * np.ones(self.dims)
        self.counter      = 0
        self.env          = gym.make('Swimmer-v2')
        self.num_rollouts = 3
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp           = 20
        self.leaf_size    = 10
        self.kernel_type  = "poly"
        self.gamma_type   = "scale"
        self.ninits       = 40
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape )
        
        self.render = False
        
    
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        M = x.reshape(self.policy_shape)
        
        returns = []
        observations = []
        actions = []
        
        
        for i in range(self.num_rollouts):
            obs    = self.env.reset()
            done   = False
            totalr = 0.
            steps  = 0
            
            while not done:
                
                action = np.dot(M, (obs - self.mean)/self.std)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1                
                if self.render:
                    self.env.render()            
            returns.append(totalr)
            
        
        return np.mean(returns)*-1
        

class Hopper:
    
    def __init__(self):
        self.mean    = np.array([1.41599384, -0.05478602, -0.25522216, -0.25404721, 
                                 0.27525085, 2.60889529,  -0.0085352, 0.0068375, 
                                 -0.07123674, -0.05044839, -0.45569644])
        self.std     = np.array([0.19805723, 0.07824488,  0.17120271, 0.32000514, 
                                 0.62401884, 0.82814161,  1.51915814, 1.17378372, 
                                 1.87761249, 3.63482761, 5.7164752 ])
        self.dims    = 33
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('Hopper-v2')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (3, 11)
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp           = 10
        self.leaf_size    = 100
        self.kernel_type  = "poly"
        self.gamma_type   = "auto"
        self.ninits       = 150
        
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape )
            
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        M = x.reshape(self.policy_shape)
        
        returns = []
        observations = []
        actions = []
        
        for i in range(self.num_rollouts):
            obs    = self.env.reset()
            done   = False
            totalr = 0.
            steps  = 0
            while not done:
                # M      = self.policy
                inputs = (obs - self.mean)/self.std
                action = np.dot(M, inputs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps  += 1
                if self.render:
                    self.env.render()
            returns.append(totalr)
            
        return np.mean(returns)*-1

# ######################################## #
# Visualize the learned policy for Swimmer #
# ######################################## #
# f = Swimmer()
# x = np.array([-0.5343142,-0.46203456, -0.70218485,
#               -0.00929887, 0.4072553, 0.04604763,
#               0.67289615, -0.5894774, 0.79874759,
#               0.84010238, 0.54327755, 0.25715409,
#               0.89032131, -0.56112252, -0.0960243,
#               0.13397496])
# f.render = True
# result = f(x)
# print( result )

# f = Hopper()
# x = np.random.rand(f.dims)
# result = f(x)
# print( result )



