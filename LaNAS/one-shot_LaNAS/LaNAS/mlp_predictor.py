# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
# import matplotlib.pyplot as plt
import json
from scipy.stats import norm
from scipy.optimize import minimize
import random
import time
import os
import itertools
import operator
import torch
import torch.nn as nn
from torch.autograd import Variable
import json
from torch import optim
import numpy as np
import random 


class LinearModel(nn.Module):

    # def __init__(self, input_dim, output_dim):
    #     super(LinearModel, self).__init__()
    #     self.fc1 = nn.Linear(input_dim, output_dim)
    #     torch.nn.init.xavier_uniform_( self.fc1.weight )

    # def forward(self, x):
    #     y = self.fc1(x)
    #     return y

    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

        torch.nn.init.xavier_uniform_( self.fc1.weight )
        torch.nn.init.xavier_uniform_( self.fc2.weight )

    def weights_init(self):
        torch.nn.init.xavier_uniform_( self.fc1.weight )
        torch.nn.init.xavier_uniform_( self.fc2.weight )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = torch.relu(x1)
        y  = self.fc2(x2)
        y  = torch.sigmoid(y)
        return y
        
    def train(self, samples):
        optimiser  = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        
        X_sample = None
        Y_sample = None
        for sample in samples:
            if X_sample is None or Y_sample is None:
                X_sample = np.array( json.loads(sample) )
                Y_sample = np.array( samples[ sample ]  )
            else:
                X_sample = np.vstack([X_sample, json.loads(sample) ] )
                Y_sample = np.vstack([Y_sample, samples[ sample ] ] )
        batch_size = 100
        print("dataset:", len(samples) )
        chunks = int( X_sample.shape[0] / batch_size )
        if  X_sample.shape[0] % batch_size > 0:
            chunks += 1
        for epoch in range(0, 150):
            X_sample_split = np.array_split(X_sample, chunks)
            Y_sample_split = np.array_split(Y_sample, chunks)
            #print("epoch=", epoch)
            for i in range(0, chunks):
                optimiser.zero_grad()
                inputs = torch.from_numpy( np.asarray(X_sample_split[i], dtype=np.float32).reshape(X_sample_split[i].shape[0], X_sample_split[i].shape[1]) )
                outputs = self.forward( inputs )
                loss = nn.MSELoss()(outputs, torch.from_numpy( np.asarray(Y_sample_split[i], dtype=np.float32) ).reshape(-1, 1)  )
                loss.backward()# back props
                nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimiser.step()# update the parameters
    
    def propose_networks( self, search_space ):
        ''' search space to predict by a meta-DNN for points selection  '''
        networks = []
        for network in search_space.keys():
            networks.append( json.loads( network ) )
        X    = np.array( networks )
        X    = torch.from_numpy( np.asarray(X, dtype=np.float32).reshape(X.shape[0], X.shape[1]) )
        Y    = self.forward( X )
        Y    = Y.data.numpy()
        Y    = Y.reshape( len(networks) )
        X    = X.data.numpy( )
        proposed_networks = []
        n    = 10
        if Y.shape[0] < n:
            n = Y.shape[0]
        indices = np.argsort(Y)[-n:]
        print("indices:", indices.shape)
        proposed_networks = X[indices]
        return proposed_networks.tolist()
    
        
        

# ####preprocess data####
# dataset = []
# with open('nasbench_dataset', 'r') as infile:
#     dataset = json.loads( infile.read() )
#
# samples = {}
# for data in dataset:
#     samples[json.dumps(data["feature"])] = data["acc"]
#
# BEST_ACC   = 0
# BEST_ARCH  = None
# CURT_BEST  = 0
# BEST_TRACE = {}
# for i in dataset:
#     arch = i['feature']
#     acc  = i['acc']
#     if acc > BEST_ACC:
#         BEST_ACC  = acc
#         BEST_ARCH = json.dumps( arch )
# print("##target acc:", BEST_ACC)
# #######################
#
# # bounds = np.array([[-1.0, 2.0]])
# noise = 0.2
# #
# #
# # def f(X, noise=noise):
# #     return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)
# #
# # X_init = np.array([[-0.9], [1.1]])
# # Y_init = f(X_init)
# #
# # X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
# # Y = f(X,0)
# #
#
#
#
# # Gaussian process with Matern kernel as surrogate model
#
# init_samples = random.sample(samples.keys(), 100)
#
#
# # Initialize samples
# #
# # Number of iterations
# n_iter = 1000000000000
# #
# # plt.figure(figsize=(12, n_iter * 3))
# # plt.subplots_adjust(hspace=0.4)
# #
# predictor  = LinearModel(49, 1)
#
# window_size = 100
# sample_counter = 0
#
# #     # Obtain next sampling point from the acquisition function (expected_improvement)
#     X_next = propose_location(predictor, X_sample, Y_sample, samples)
# #     # Obtain next noisy sample from the objective function
#     for network in X_next:
#         X_sample = np.vstack([X_sample, network] )
#     for network in X_next:
#         sample_counter += 1
#         acc = samples[ json.dumps( network.tolist() ) ]
#         if acc > CURT_BEST:
#             BEST_TRACE[json.dumps( network.tolist() ) ] = [acc, sample_counter]
#             CURT_BEST = acc
#         if acc == BEST_ACC:
#             sorted_best_traces = sorted(BEST_TRACE.items(), key=operator.itemgetter(1))
#             for item in sorted_best_traces:
#                 print(item[0],"==>", item[1])
#             final_results = []
#             for item in sorted_best_traces:
#                 final_results.append( item[1] )
#             final_results_str = json.dumps(final_results)
#             with open("result.txt", "a") as f:
#                 f.write(final_results_str + '\n')
#             print("$$$$$$$$$$$$$$$$$$$CONGRATUGLATIONS$$$$$$$$$$$$$$$$$$$")
#             os._exit(1)
#
#         print(network, acc)
#         del samples[ json.dumps( network.tolist() ) ]
#         Y_sample = np.vstack([Y_sample, acc] )








