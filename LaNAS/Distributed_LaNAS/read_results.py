# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import json
import os
import operator

data  = {}

with open('total_trace.json') as json_file:
    data = json.load(json_file)


sorted_trace = {}
sorted_trace = sorted(data.items(), key=operator.itemgetter(1))
for k,v in sorted_trace:
    print(k, v)

