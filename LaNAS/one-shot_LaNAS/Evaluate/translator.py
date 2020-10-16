# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
import random

node = 5
layer_type = [
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5'
]


def get_rand_vector(layer_type):
    vec_length = len(layer_type)
    masked_vec = []
    for i in range(0, vec_length):
        if random.random() > 0.5:
            masked_vec.append(1)
        else:
            masked_vec.append(0)
    return masked_vec


def mask_rand_generator():
    supernet = [[] for v in range(node)]
    for i in range(node):
        for j in range(node + 2):
            if j < i + 2:
                supernet[i].append(get_rand_vector(layer_type))
            else:
                supernet[i].append(0)
    return supernet


def supernet_generator(node, layer_type):
    vec_length = len(layer_type)
    masked_vec = np.ones((1, vec_length))[0].tolist()
    supernet = [[] for v in range(node)]
    for i in range(node):
        for j in range(node + 2):
            if j < i + 2:
                supernet[i].append(masked_vec.copy())
            else:
                supernet[i].append(0)
    return supernet


def mask_specific_value(supernet, node_id, input_id, operation_id):
    supernet[node_id][input_id][operation_id] = 0.0
    return supernet


def selected_specific_value(supernet, node_id, input_id, operation_id):
    for i in range(len(supernet[node_id][input_id])):
        if i != operation_id:
            supernet[node_id][input_id][i] = 0.0
    return supernet


def encoding_to_masks(encoding):
    encoding = np.array(encoding).reshape(-1, 4)
    supernet_normal = supernet_generator(node, layer_type)
    supernet_reduce = supernet_generator(node, layer_type)
    supernet = [supernet_normal, supernet_reduce]
    mask = []
    counter = 0
    for cell in supernet:
        mask_cell = []
        for row in cell:
            mask_row = []
            for col in row:
                if type(col) == type([]):
                    mask_row.append(encoding[counter].tolist())
                    counter += 1
                else:
                    mask_row.append(0)
            mask_cell.append(mask_row)
        mask.append(mask_cell)

    normal_mask = mask[0]
    reduce_mask = mask[1]

    return normal_mask, reduce_mask


def supernet_mask():
    supernet_normal = supernet_generator(node, layer_type)
    supernet_reduce = supernet_generator(node, layer_type)
    return supernet_normal, supernet_reduce


def encode_supernet():
    supernet_normal = supernet_generator(node, layer_type)
    supernet_reduce = supernet_generator(node, layer_type)
    supernet = [supernet_normal, supernet_reduce]

    layer_types_count = len(layer_type)
    count = 0
    assert type(supernet) == type([])
    for cell in supernet:
        for row in cell:
            for col in row:
                if type(col) == type([]):
                    count += layer_types_count
    return np.ones((count)).tolist()


def define_search_space():
    # hit-and-run default
    # A x <= b
    search_space = encode_supernet()
    A = []
    b = []
    init_point = []
    param_pos = 0
    for i in range(0, len(search_space)):
        tmp = np.zeros(len(search_space))
        tmp[i] = 1
        A.append(np.copy(tmp))
        b.append(1.0000001)
        # we need relax a little bit here for the precision issue
        # A*x <= 1, we use 1.000+epsilon
        # A*x >= 0, we use 0-epslon
        A.append(-1 * np.copy(tmp))
        b.append(0.0000001)
    for i in range(0, len(search_space)):
        if random.random() >= 0.5:
            init_point.append(0.0)
        else:
            init_point.append(1.0)
    return {"A": np.array(A), "b": np.array(b), "init_point": np.array(init_point)}


c = [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1]


def expend_to_supernet_code(old_supernet):
    for i in range(len(old_supernet)):
        for j in range(len(old_supernet[i])):
            if old_supernet[i][j] == 0:
                old_supernet[i][j] = [0] * len(old_supernet[0][0])
    for i in range(len(old_supernet)):
        for j in range(len(old_supernet[i])):
            for n in range(len(old_supernet[i][j])):
                old_supernet[i][j][n] = float(old_supernet[i][j][n])

    return old_supernet



normal, reduce = encoding_to_masks(c)

normal = expend_to_supernet_code(normal)
reduce = expend_to_supernet_code(reduce)

print(normal)
print(reduce)


# print(supernet_reduce)