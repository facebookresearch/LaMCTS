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
from Node import Node
from net_training import Net_Trainer

class MCTS:
    #############################################

    def __init__(self, search_space, tree_height, arch_code_len ):
        assert type(search_space) == type([])
        assert len(search_space)  >= 1
        
        assert type(search_space)    == type([])
        assert type(search_space[0]) == type([])
        self.ARCH_CODE_LEN           =  arch_code_len
        self.SEARCH_COUNTER          =  0
        self.samples                 =  {}
        self.nodes                   =  []
        self.search_space            =  search_space
        self.Cp                      =  0.1
        # 49 is the length of architectuer encoding, 1 is for predicted accuracy
        #self.metaDNN                 = LinearModel(49, 1) 
        #querying the accuracy from nasbench
        self.net_trainer             = Net_Trainer( )
        
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
        
        # self.loads_all_states()
        self.ROOT = self.nodes[0]
        self.CURT = self.ROOT
        
        print('='*10 + 'search space start' + '='*10)
        print("total architectures:", len(search_space) )
        print('='*10 + 'search space end  ' + '='*10)
        self.init_train()

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
        for i in range(0, 10):
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

    def dump_all_states(self):
        
        node_path = 'mcts_agent'
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)


    def loads_all_states(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                print("=====>loads:", len(self.samples)," samples" )
                print("=====>loads:", self.SEARCH_COUNTER," counter" )
                print("=====>loads:", len(self.nodes)," counter" )

                return True
        return False
                
    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        while curt_node.is_leaf == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_uct(self.Cp) )
            curt_node = curt_node.kids[np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]]
            print("going:", curt_node.get_name(), end=' ' )
        return curt_node

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
    

    def search(self):
        while len(self.search_space) > 0:
            #self.dump_all_states()
            #assemble the training data:
            self.populate_training_data()
            print("-"*20,"iteration:", self.SEARCH_COUNTER)
            print("populate training data")
            #self.print_tree()

            #training the tree
            self.train_nodes()
            print("finishing training")
            #self.print_tree()
            
            #clear the data in nodes
            print("reset training data")
            self.reset_node_data()
            #self.print_tree()
            
            print("populate prediction data")
            self.populate_prediction_data()
            #self.print_tree()
            
            print("predict:", len(self.samples) )
            self.predict_nodes()
            self.check_leaf_bags()
            
            #print("training meta-dnn toward #samples:", len( self.samples ) )
            #self.metaDNN.train( self.samples )
            self.print_tree()
            
            for i in range(0, 20):
                #select
                target_bin        = self.select()
                sampled_arch      = target_bin.sample_arch()
                if sampled_arch is not None:
                #TODO: back-propogate an architecture
                    sampled_acc  = self.net_trainer.train_net(sampled_arch)
                    self.samples[ json.dumps(sampled_arch) ] = sampled_acc
                    print("sampled architecture:", sampled_arch, sampled_acc)
                    self.backpropogate( target_bin, sampled_acc)
                    self.search_space.remove(sampled_arch)
                else:
                    for n in self.nodes:
                        if n.is_leaf == True:
                            sampled_arch = n.sample_arch()
                            if sampled_arch is not None:
                                print(sampled_arch)
                                sampled_acc  = self.net_trainer.train_net(sampled_arch)
                                self.samples[ json.dumps(sampled_arch) ] = sampled_acc
                                self.backpropogate( n, sampled_acc)
                                self.search_space.remove(sampled_arch)
                                break
                            else:
                                continue
        
            self.print_tree()
            self.SEARCH_COUNTER += 1
            #sample a network from the target bin

data = {}
with open('nasbench_dataset', 'r') as infile:
    data = json.loads( infile.read() )

search_space  = []
for d in data:
    search_space.append(d['feature'] )
arch_code_len = len( search_space[0] )
print("the length of architecture codes:", arch_code_len)
print("total architectures:", len(search_space) )

node_path = 'mcts_agent'
if os.path.isfile(node_path) == True:
    with open(node_path, 'rb') as json_data:
        agent = pickle.load(json_data)
    print("=====>loads:", len(agent.samples)," samples" )
    print("=====>loads:", agent.SEARCH_COUNTER," counter" )
    print("=====>loads:", len(agent.nodes)," counter" )
    agent.search()
else:
    agent = MCTS(search_space, 5, arch_code_len)
    agent.search()
