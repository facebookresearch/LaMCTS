# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import json

with open('result.txt') as json_data:
    data = json.load(json_data)

counter = 0
for elem in data:
    print(elem[0], elem[1], counter)
    counter += 1
