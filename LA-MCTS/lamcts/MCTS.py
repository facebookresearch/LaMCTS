# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import json
import collections
import copy as cp
import math
from collections import OrderedDict
import os.path
import numpy as np
import time
import operator
import sys
import pickle
import os
import random
from datetime import datetime
from .Node import Node
from .utils import latin_hypercube, from_unit_cube
from torch.quasirandom import SobolEngine
import torch

class MCTS:
    #############################################

    def __init__(self, lb, ub, dims, ninits, func, Cp = 1, leaf_size = 20, kernel_type = "rbf", gamma_type = "auto"):
        self.dims                    =  dims
        self.samples                 =  []
        self.nodes                   =  []
        self.Cp                      =  Cp
        self.lb                      =  lb
        self.ub                      =  ub
        self.ninits                  =  ninits
        self.func                    =  func
        self.curt_best_value         =  float("-inf")
        self.curt_best_sample        =  None
        self.best_value_trace        =  []
        self.sample_counter          =  0
        self.visualization           =  False
        
        self.LEAF_SAMPLE_SIZE        =  leaf_size
        self.kernel_type             =  kernel_type
        self.gamma_type              =  gamma_type
        
        self.solver_type             = 'bo' #solver can be 'bo' or 'turbo'
        
        print("gamma_type:", gamma_type)
        
        #we start the most basic form of the tree, 3 nodes and height = 1
        root = Node( parent = None, dims = self.dims, reset_id = True, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
        self.nodes.append( root )
        
        self.ROOT = root
        self.CURT = self.ROOT
        self.init_train()
        
    def populate_training_data(self):
        #only keep root
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        new_root  = Node(parent = None,   dims = self.dims, reset_id = True, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
        self.nodes.append( new_root )
        
        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_bag( self.samples )
    
    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            if node.is_leaf() == True and len(node.bag) > self.LEAF_SAMPLE_SIZE and node.is_svm_splittable == True:
                status.append( True  )
            else:
                status.append( False )
        return np.array( status )
        
    def get_split_idx(self):
        split_by_samples = np.argwhere( self.get_leaf_status() == True ).reshape(-1)
        return split_by_samples
    
    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False
        
    def dynamic_treeify(self):
        # we bifurcate a node once it contains over 20 samples
        # the node will bifurcate into a good and a bad kid
        self.populate_training_data()
        assert len(self.ROOT.bag) == len(self.samples)
        assert len(self.nodes)    == 1
                
        while self.is_splitable():
            to_split = self.get_split_idx()
            #print("==>to split:", to_split, " total:", len(self.nodes) )
            for nidx in to_split:
                parent = self.nodes[nidx] # parent check if the boundary is splittable by svm
                assert len(parent.bag) >= self.LEAF_SAMPLE_SIZE
                assert parent.is_svm_splittable == True
                # print("spliting node:", parent.get_name(), len(parent.bag))
                good_kid_data, bad_kid_data = parent.train_and_split()
                #creat two kids, assign the data, and push into lists
                # children's lb and ub will be decided by its parent
                assert len(good_kid_data) + len(bad_kid_data) == len(parent.bag)
                assert len(good_kid_data) > 0
                assert len(bad_kid_data)  > 0
                good_kid = Node(parent = parent, dims = self.dims, reset_id = False, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
                bad_kid  = Node(parent = parent, dims = self.dims, reset_id = False, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
                good_kid.update_bag( good_kid_data )
                bad_kid.update_bag(  bad_kid_data  )
            
                parent.update_kids( good_kid = good_kid, bad_kid = bad_kid )
            
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)
                
            #print("continue split:", self.is_splitable())
        
        self.print_tree()
        
    def collect_samples(self, sample, value = None):
        #TODO: to perform some checks here
        if value == None:
            value = self.func(sample)*-1
            
        if value > self.curt_best_value:
            self.curt_best_value  = value
            self.curt_best_sample = sample 
            self.best_value_trace.append( (value, self.sample_counter) )
        self.sample_counter += 1
        self.samples.append( (sample, value) )
        return value
        
    def init_train(self):
        
        # here we use latin hyper space to generate init samples in the search space
        init_points = latin_hypercube(self.ninits, self.dims)
        init_points = from_unit_cube(init_points, self.lb, self.ub)
        
        for point in init_points:
            self.collect_samples(point)
        
        print("="*10 + 'collect '+ str(len(self.samples) ) +' points for initializing MCTS'+"="*10)
        print("lb:", self.lb)
        print("ub:", self.ub)
        print("Cp:", self.Cp)
        print("inits:", self.ninits)
        print("dims:", self.dims)
        print("="*58)
        
    def print_tree(self):
        print('-'*100)
        for node in self.nodes:
            print(node)
        print('-'*100)

    def reset_to_root(self):
        self.CURT = self.ROOT
    
    def load_agent(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                print("=====>loads:", len(self.samples)," samples" )

    def dump_agent(self):
        node_path = 'mcts_agent'
        print("dumping the agent.....")
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)
            
    def dump_samples(self):
        sample_path = 'samples_'+str(self.sample_counter)
        with open(sample_path, "wb") as outfile:
            pickle.dump(self.samples, outfile)
    
    def dump_trace(self):
        trace_path = 'best_values_trace'
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')

    def greedy_select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        if self.visualization == True:
            curt_node.plot_samples_and_boundary(self.func)
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_xbar() )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            if curt_node.is_leaf() == False and self.visualization == True:
                curt_node.plot_samples_and_boundary(self.func)
            print("=>", curt_node.get_name(), end=' ' )
        print("")
        return curt_node, path

    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_uct(self.Cp) )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            print("=>", curt_node.get_name(), end=' ' )
        print("")
        return curt_node, path
    
    def backpropogate(self, leaf, acc):
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
            curt_node.n    += 1
            curt_node       = curt_node.parent

    def search(self, iterations):
        for idx in range(self.sample_counter, iterations):
            print("")
            print("="*10)
            print("iteration:", idx)
            print("="*10)
            self.dynamic_treeify()
            leaf, path = self.select()
            for i in range(0, 1):
                if self.solver_type == 'bo':
                    samples = leaf.propose_samples_bo( 1, path, self.lb, self.ub, self.samples )
                elif self.solver_type == 'turbo':
                    samples, values = leaf.propose_samples_turbo( 10000, path, self.func )
                else:
                    raise Exception("solver not implemented")
                for idx in range(0, len(samples)):
                    if self.solver_type == 'bo':
                        value = self.collect_samples( samples[idx])
                    elif self.solver_type == 'turbo':
                        value = self.collect_samples( samples[idx], values[idx] )
                    else:
                        raise Exception("solver not implemented")
                    
                    self.backpropogate( leaf, value )
            print("total samples:", len(self.samples) )
            print("current best f(x):", np.absolute(self.curt_best_value) )
            # print("current best x:", np.around(self.curt_best_sample, decimals=1) )
            print("current best x:", self.curt_best_sample )



