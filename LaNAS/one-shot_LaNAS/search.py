# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from supernet.generator import encode_supernet, define_search_space
from LaNAS.MCTS import MCTS
from supernet.supernet_train import Trainer
import numpy as np
import os
import pickle

############## first step, training supernet 

# trainer.run(300)

############## second step, searching over the supernet
search_space = define_search_space()
print( search_space["init_point"], len( search_space["init_point"] ) )
# sample = mcts.zero_supernet_generator()
# print(sample)
#
#
trainer = Trainer( batch_size=40, init_channels= 48  )
trainer.load_model("./NASNet_Supernet.pt")

#
node_path = "mcts_agent"
if os.path.isfile(node_path) == True:
    with open(node_path, 'rb') as json_data:
        agent = pickle.load(json_data)
        print("=====>loads:", len(agent.samples)," samples" )
        print("=====>loads:", agent.SEARCH_COUNTER," counter" )
        agent.search( )
else:
    mcts   = MCTS(search_space, trainer, 5)
    sample = mcts.trainer.propose_nasnet_mask()
    result = trainer.infer_masks( sample )
    mcts.collect_samples( result )
    mcts.search( )
