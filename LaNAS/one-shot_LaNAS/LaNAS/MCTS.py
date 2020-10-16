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

class MCTS:
    #############################################

    def __init__(self, search_space, trainer, tree_height ):        
        self.ARCH_CODE_LEN           =  int( len( search_space["b"] ) / 2 )
        self.SEARCH_COUNTER          =  0
        self.samples                 =  {}
        self.nodes                   =  []
        # search space is a tuple, 
        # 0: left side of the constraint, i.e. A
        # 1: right side of the constraint, i.e. b
        self.search_space            =  search_space
        self.Cp                      =  10
        self.trainer                 =  trainer
        # pre-defined for generating masks for supernet
        
        print("architecture code length:", self.ARCH_CODE_LEN )
        # set random seed
        np.random.seed(seed=int(time.time() ) )
        random.seed(datetime.now() )
        
        #initialize the a full tree
        total_nodes = 2**tree_height - 1
        for i in range(1, total_nodes + 1):
            is_good_kid = False
            if (i-1)  > 0 and (i-1) % 2 == 0:
                is_good_kid = False
            elif (i -1) > 0:
                is_good_kid = True
            parent_id = i // 2  - 1
            if parent_id == -1:
                self.nodes.append( Node( None, is_good_kid, self.ARCH_CODE_LEN, True ) )
            else:
                self.nodes.append( Node(self.nodes[parent_id], is_good_kid, self.ARCH_CODE_LEN, False) )
        
        self.ROOT = self.nodes[0]
        self.CURT = self.ROOT
        
        print('='*10 + 'search space start' + '='*10)
        print("total architectures: 2^", len(search_space) )
        print('='*10 + 'search space end  ' + '='*10)
        
    def dump_all_states(self):
        node_path = 'mcts_agent'
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)
        
    def collect_samples(self, results):
        for arch in results:
            if arch not in self.samples:
                self.samples[arch] = results[arch]

    def train_nodes(self):
        for i in self.nodes:
            i.train()

    def predict_nodes(self):
        for i in self.nodes:
            i.predict()

    def reset_node_data(self):
        for i in self.nodes:
            i.clear_data()

    def populate_training_data(self):
        self.reset_node_data()
        for k, v in self.samples.items():
            self.ROOT.put_in_bag( json.loads(k), v )

    def populate_prediction_data(self):
        self.reset_node_data()
        for k in self.search_space:
            self.ROOT.put_in_bag( k, 0.0 )
    
    def init_train(self):
        for i in range(0, 300):
            net = random.choice(self.search_space)
            self.search_space.remove(net)
            net_str = json.dumps( net )
            acc  = self.net_trainer.train_net( net )
            self.samples[net_str] = acc
        print("="*10 + 'collect '+ str(len(self.samples) ) +' nets for initializing MCTS')
        
    def print_tree(self):
        print('-'*100)
        for i in self.nodes:
            print(i)
        print('-'*100)

    def reset_to_root(self):
        self.CURT = self.ROOT

    def dump_agent(self):
        node_path = 'mcts_agent'
        print("dumping the agent.....")
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)

    def load_agent(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                print("=====>loads:", len(self.samples)," samples" )
                print("=====>loads:", self.SEARCH_COUNTER," counter" )
                
    def select(self):
        self.reset_to_root()
        boundaries = []
        curt_node = self.ROOT
        
        curt_node.print_bag()
        starting_point = curt_node.get_rand_sample_from_bag()
        
        while curt_node.is_leaf == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_uct(self.Cp) )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            w, b, x_bar    = curt_node.get_boundary( )
            boundaries.append( (w, b, x_bar, choice) )
            curt_node = curt_node.kids[choice]
            if curt_node.get_rand_sample_from_bag() is not None:
                starting_point = curt_node.get_rand_sample_from_bag()
        return curt_node, boundaries, starting_point

    def backpropogate(self, leaf, acc):
        curt_node = leaf
        while curt_node is not None:
            if acc > 0:
                if curt_node.n > 0:
                    curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
                else:
                    curt_node.x_bar = acc
            curt_node.n    += 1
            curt_node = curt_node.parent

    def check_leaf_bags(self):
        counter = 0
        for i in self.nodes:
            if i.is_leaf is True:
                counter += len( i.bag )
        assert counter == len( self.search_space )

        return counter
    
    def prepare_boundaries(self, boundaries):
        #2*self.ARCH_CODE_LEN+
        W = []
        B = []
        for boundary in boundaries:
            w, b, x_bar, choice = boundary
            righthand = x_bar - b[0]
            lefthand  = w
            if righthand == float("inf"):
                continue
            if choice == 0:
                #transform to W*x <= b
                lefthand  = -1*lefthand
                righthand = -1*righthand
            lefthand = lefthand.reshape(np.prod(lefthand.shape))
            W.append(lefthand)
            B.append(righthand)
        W = np.array(W)
        B = np.array(B)
        return W, B
        
    def search_samples_under_constraints(self, W, b):
        r_counter = 0
        sample_counter = 0
        while True:
            sample_counter += 1
            rand_arch = self.trainer.propose_nasnet_mask()
            for r_counter in range(0, len(b) ):
                left = rand_arch * W[r_counter]
                if np.sum(left) <  b[r_counter]:
                    print("total sampled:", sample_counter )
                    return rand_arch
                # print("left:", np.sum(left), np.sum(left) <  b[r_counter] )
                # print("right:", b[r_counter] )
                
    def dump_results(self):
        sorted_samples = sorted(self.samples.items(), key=operator.itemgetter(1))
        final_results_str = json.dumps(sorted_samples )
        with open("result.txt", "w") as f:
            f.write(final_results_str + '\n')

    def search(self):
        search_counter = 0
        while True:
            #assemble the training data:
            self.populate_training_data()
            print("-"*20,"iteration:", self.SEARCH_COUNTER )
            print("populate training data")
            self.print_tree()

            #training the tree
            self.train_nodes()
            print("finishing training")
            self.print_tree()
            
            #select
            target_bin, boundaries, starting_point = self.select()
            W, b = self.prepare_boundaries( boundaries )
            print( W.shape, b.shape )
            for i in range(0, len(b)):
                print( W[i], b[i] )
            print("starting point:", starting_point )
            
            sampled_arch   = self.search_samples_under_constraints(W, b)
            sampled_result = self.trainer.infer_masks(sampled_arch)
            
            self.collect_samples( sampled_result )
            
            #back-progagate
            print("sampled architecture:", sampled_arch, sampled_result[ json.dumps(sampled_arch.tolist() ) ])
            self.backpropogate( target_bin, sampled_result[ json.dumps(sampled_arch.tolist() ) ] )
            
            if self.SEARCH_COUNTER % 1  == 0:
                self.dump_agent()
            
            if self.SEARCH_COUNTER % 1 == 0:
                self.dump_results()

            self.SEARCH_COUNTER += 1
