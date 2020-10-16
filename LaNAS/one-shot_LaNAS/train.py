# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from supernet.generator import encode_supernet, define_search_space
from LaNAS.MCTS import MCTS
from supernet.supernet_train import Trainer


trainer = Trainer( batch_size=80, init_channels= 48, epochs = 300 )
trainer.run()

