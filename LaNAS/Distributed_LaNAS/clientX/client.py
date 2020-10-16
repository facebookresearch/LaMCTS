# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import time
import json
import random
import sys
import os
import pickle
import socket
import signal
import numpy as np
import traceback
import train_client
from nasnet_set import *
from array import array
from multiprocessing.connection import Client


class Client_t:
    
    def __init__(self):
        self.addr           = ('100.97.66.131', 8000)
        self.client_name    = "client"
        self.total_send     = 0
        self.total_recv     = 0
        self.accuracy_trace = {}
        self.load_acc_trace()
        signal.signal(signal.SIGUSR1, self.sig_handler)
        signal.signal(signal.SIGTERM, self.term_handler)
        #below two is to signal the client status
        self.received       = False
        self.network        = []
        self.acc            = 0
    
    def print_client_status(self):
        print("client->receive status: ", client.received  )
        print("client->network: ",        client.network )
        print("client->acc: ",            client.acc )
        print("client->trace_len:",       len(client.accuracy_trace) )

        
    def sig_handler(self, signum, frame):
        print("caught signal", signum," about to exit, dump client")
        self.dump_client()
        if os.path.isfile('client.inst'):
            print("dump successful")

    def term_handler(self, signum, frame):
        self.dump_client()
        if os.path.isfile('client.inst'):
             print("dump successful")
        print("terminated caught", flush=True)

    def dump_client(self):
        client_path = 'client.inst'
        with open(client_path,"wb") as outfile:
            pickle.dump(self, outfile)
    
    def dump_acc_trace(self):
        with open('acc_trace.json', 'w') as fp:
            json.dump(self.accuracy_trace, fp)

    def load_acc_trace(self):
        if os.path.isfile('acc_trace.json'):
            with open('acc_trace.json') as fp:
                self.accuracy_trace = json.load(fp)
            print("loading #", len(self.accuracy_trace )," prev trained networks")

    def train(self):
        while True:
            while not self.received:
                try:
                    send_address = ('100.97.66.131', 8000)
                    conn = Client(send_address, authkey=b'nasnet')
                    if conn.poll(2):
                        [ self.network ] = conn.recv()
                        self.total_recv += 1
                        conn.close()
                        self.received = True
                        self.dump_client()
                        print("RECEIEVE:=>", self.network)
                        print("RECEIEVE:=>", " total_send:", self.total_send, " total_recv:", self.total_recv)
                        self.print_client_status()
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print("client recv error")

            if self.received:
                print("prepare training the network:", self.network)
                network = np.array( self.network, dtype = 'int' )
                network = network.tolist()
                net     = gen_code_from_list( network, node_num=7 ) #TODO: change it to 7
                net_str = json.dumps( network )
                if net_str in self.accuracy_trace:
                    self.acc = self.accuracy_trace[net_str]
                else:
                    genotype_net = translator([net, net], max_node=7) #TODO: change it to 7
                    print("--"*15)
                    print(genotype_net)
                    print("training the above network")
                    print("--"*15)
                    self.acc = train_client.run(genotype_net, epochs=600, batch_size=200)
                    self.accuracy_trace[net_str] = self.acc
                    self.dump_acc_trace()

            #TODO: train the actual network
            #time.sleep(random.randint(2, 5) )
            while self.received:
                try:
                    recv_address = ('100.97.66.131', 8000)
                    conn = Client(recv_address, authkey=b'nasnet')
                    network_str = json.dumps( np.array(network).tolist() )
                    conn.send([self.client_name, network_str, self.acc])
                    self.total_send += 1
                    print("SEND:=>", self.network, self.acc)
                    self.network  = []
                    self.acc      = 0
                    self.received = False
                    self.dump_client()
                    print("SEND:=>", " total_send:", self.total_send, " total_recv:", self.total_recv)
                    conn.close()
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print("client send error, reconnecting")

inst_path = 'client.inst'
if os.path.isfile( inst_path ) == True:
    with open(inst_path, 'rb') as client_data:
        client = pickle.load( client_data )
        client.print_client_status()
    client.train()
else:
    client = Client_t()
    client.train()
