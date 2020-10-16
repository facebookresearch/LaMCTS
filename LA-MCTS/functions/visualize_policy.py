# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from functions import Lunarlanding
from mujoco_functions import *
import numpy as np




# policy = np.array([ 0.6829919, 0.4348611, 1.93635682, 1.53007997,
#                1.69574236, 0.66056938, 0.28328839, 1.12798157,
#                0.06496076, 1.71219888, 0.23686494, 0.20135697 ] )
# f = Lunarlanding()
# f.render = True
# result = f(policy)
# print( result )

policy = np.array(
[ 0.40721659,  0.64248771,   0.31267019, -0.69240676, -0.00208609, -0.86336196,
 -0.54423801,  0.28333422,  -0.68388651, -0.26167397, -0.58448575,  0.11981415,
 -0.90660989,  0.55700556,  -0.22651554,  0.42790948,  0.15368999,  0.7514032,
 -0.42978046, -0.60632853,  -0.88724493, -0.01787839,  0.74753749, -0.8137155,
  0.41300612,  0.08062934,  -0.25451053, -0.77197475, -0.09003459, -0.76673666,
 -0.30785222,  0.41125726,  -0.11475573]
)
f = Hopper()
f.render = True
result = f(policy)
print(result)
