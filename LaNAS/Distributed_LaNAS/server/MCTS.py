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
import jsonpickle
import pickle
import os
import random
from datetime import datetime
from Node import Node
from multiprocessing.connection import Listener



class MCTS:
    

    def __init__(self, search_space, tree_height, arch_code_len ):
        assert type(search_space)    == type([])
        assert len(search_space)     >= 1
        assert type(search_space)    == type([])
        assert type(search_space[0]) == type([])
        #############################################
        self.ARCH_CODE_LEN = arch_code_len
        self.ROOT           = None
        self.Cp             = 0.2
        self.search_space   = None
        self.ARCH_CODE_LEN  = arch_code_len
        self.nodes          = []
        self.samples        = {}
        self.TASK_QUEUE     = []
        self.DISPATCHED_JOB = {}
        self.JOB_COUNTER    = 0
        self.TOTAL_SEND     = 0
        self.TOTAL_RECV     = 0
        self.ITERATION      = 0
        self.MAX_ACC        = 0
        #############################################

        
        
        # set random seed
        np.random.seed(seed=int(time.time() ) )
        random.seed(datetime.now() )
        
        self.search_space = search_space
        
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

    def print_task_queue(self):
        print("task queue", "#"*10)
        for net in self.TASK_QUEUE:
            print(net)
        print("#"*10)

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
        for i in range(0, 2000):
            net     = random.choice(self.search_space)
            self.search_space.remove(net)
            net_str = json.dumps( net )
            self.TASK_QUEUE.append( net )

        print("="*10 + 'collect '+ str(len(self.TASK_QUEUE) ) +' nets for initializing MCTS')
        
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
        node_path = 'nodes'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'r') as json_data:
                self.nodes = jsonpickle.decode(json.load(json_data, object_pairs_hook=OrderedDict))
        print("=>LOAD", len(self.nodes), " MCTS nodes")

    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        #print("select:", "-"*10)
        while curt_node.is_leaf == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_uct() )
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

    def dispatch_and_retrieve_jobs(self, server = None):
        while len(self.TASK_QUEUE) > 50:
            is_send_successful = False
            job  = self.TASK_QUEUE.pop()
            try:
                print("########################")
                print( "job counter:", self.JOB_COUNTER )
                print( "get job from QUEUE==>", job ) 
                print("listening......")
                conn = server.accept()
                conn.send( [job] )
                conn.close()
                is_send_successful = True
                self.TOTAL_SEND += 1
                print("==>dispatch job:", " total_send:", self.TOTAL_SEND, " total_recv:", self.TOTAL_RECV )
                self.JOB_COUNTER += 1
            
                for i in range(0, 5):
                    conn = server.accept()
                    if conn.poll(0.5):
                        receive_signal               = conn.recv()
                        client_name                  = receive_signal[0]
                        job_str                      = receive_signal[1]
                        acc                          = receive_signal[2]
                        self.DISPATCHED_JOB[job_str] = acc
                        self.samples[job_str]        = acc
                        if acc > self.MAX_ACC:
                            self.MAX_ACC = acc
                        received = True
                        self.TOTAL_RECV += 1
                        print("retrieve job, curt max acc:", self.MAX_ACC)
                        print("="*10, " total_send:", self.TOTAL_SEND, " total_recv:", self.TOTAL_RECV)
                        if received:
                            self.JOB_COUNTER -= 1
                            print(client_name, "==> net:", job_str, acc, " total samples:", len(self.samples)," job counter:", self.JOB_COUNTER )

            except Exception as error:
                if not is_send_successful:
                    self.TASK_QUEUE.append(job)
                print("send or recv timeout, curt queue len:", len(self.TASK_QUEUE) )


    def search(self):
        address = ('XXX.XX.XX.XXX', 8000)
        server = Listener(address, authkey=b'nasnet')

        while len(self.search_space) > 0:
            self.dump_all_states()
            print("-"*20,"iteration:", self.ITERATION )

            #dispatch & retrieve jobs:
            self.dispatch_and_retrieve_jobs(server)
            
            #assemble the training data:
            self.populate_training_data()
            print("populate training data###", "total samples:", len(self.samples)," trained:", len(self.DISPATCHED_JOB)," task queue:", len(self.TASK_QUEUE) )
            self.print_tree()

            #training the tree
            self.train_nodes()
            print("finishing training###", "total samples:", len(self.samples)," trained:", len(self.DISPATCHED_JOB)," task queue:", len(self.TASK_QUEUE) )
            self.print_tree()
            
            #clear the data in nodes
            print("reset training data###", "total samples:", len(self.samples)," trained:", len(self.DISPATCHED_JOB)," task queue:", len(self.TASK_QUEUE) )
            self.reset_node_data()
            self.print_tree()
            
            print("populate prediction data###", "total samples:", len(self.samples)," trained:", len(self.DISPATCHED_JOB)," task queue:", len(self.TASK_QUEUE) )
            self.populate_prediction_data()
            #self.print_tree()
            
            print("predict###", "total samples:", len(self.samples)," trained:", len(self.DISPATCHED_JOB)," task queue:", len(self.TASK_QUEUE) )
            self.predict_nodes()
            self.check_leaf_bags()
            self.print_tree()
            
            for i in range(0, 50):
                #select
                target_bin   = self.select()
                sampled_arch = target_bin.sample_arch()
                sampled_arch = None
                #NOTED: the sampled arch can be None
                if sampled_arch is not None:
                #TODO: back-propogate an architecture
                #push the arch into task queue
                    if json.dumps(sampled_arch) not in self.DISPATCHED_JOB:
                        self.TASK_QUEUE.append( sampled_arch )
                        self.DISPATCHED_JOB[json.dumps(sampled_arch)] = 0
                        self.search_space.remove(sampled_arch)
                else:
                    #trail 1: pick a network from the best leaf
                    for n in self.nodes:
                        if n.is_leaf == True:
                            sampled_arch = n.sample_arch()
                            if sampled_arch is not None:
                                if json.dumps(sampled_arch) not in self.DISPATCHED_JOB:
                                    self.TASK_QUEUE.append( sampled_arch )
                                    self.DISPATCHED_JOB[json.dumps( sampled_arch )] = 0
                                    self.search_space.remove(sampled_arch)
                                    break
                            else:
                                continue
            self.print_task_queue()
            self.print_tree()
            self.ITERATION  += 1

############MAIN############
data = {}
with open('search_space', 'r') as infile:
    data = json.loads( infile.read() )

total_max     = 0
search_space  = []
for d in data:
    if max(d) > total_max:
        total_max = max(d)
    search_space.append( d )
arch_code_len = len( search_space[0] )
print("the length of architecture codes:", arch_code_len)
print("total architectures:", len(search_space) )
print("largest single element:", total_max)

node_path = 'mcts_agent'
if os.path.isfile(node_path) == True:
    with open(node_path, 'rb') as json_data:
        agent = pickle.load(json_data)
    print("=====>loads:", len(agent.samples)," samples" )
    print("=====>loads:", agent.ITERATION," counter" )
    print("=====>loads:", len(agent.nodes)," nodes" )
    print("=====>loads:", len(agent.DISPATCHED_JOB)," dispatched jobs")
    print("=====>loads:", len(agent.TASK_QUEUE)," task_queue jobs")
    print("=====>send&recv:", agent.TOTAL_SEND, agent.TOTAL_RECV)

    agent.search()
else:
    agent = MCTS(search_space, 8, arch_code_len)
    agent.search()
