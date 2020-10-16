# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import json
import os
import operator

total_trace  = {}
for i in range(1, 800):
    path = 'client' + str(i) + '/' + 'acc_trace.json'
    if os.path.exists(path):
        with open(path, 'r') as json_data:
            data = json.load(json_data)
        for k, v in data.items():
            total_trace[k] = v

with open('total_trace.json', 'w') as outfile:
    json.dump(total_trace, outfile)
print("total element:", len(total_trace) )



