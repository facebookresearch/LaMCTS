# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import json
import csv



with open("our_past_results.txt", "r") as f:
    l = f.readlines()


list_net = []
for i in range(len(l)):
    l[i] = l[i].rstrip('\n')
    list_net.append(json.loads(l[i]))
    #print(json.loads(l[i]))
    print(str(json.loads(l[i])[-1][1]), end =", "),
print("")

