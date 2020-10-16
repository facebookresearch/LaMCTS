# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
import itertools
import random
import copy

node = 4
layer_type = [
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5'
]


def supernet_generator(node, layer_type):
    vec_length = len(layer_type)
    masked_vec = np.ones((1, vec_length))[0].tolist()
    disconnected_vec = np.zeros((1, vec_length))[0].tolist()
    supernet = [[] for v in range(node)]
    for i in range(node):
        for j in range(node + 2):
            if j < i + 2:
                supernet[i].append(masked_vec.copy())
            else:
                supernet[i].append(disconnected_vec.copy())
    return supernet

def mask_specific_value(supernet, node_id, input_id, operation_id):
    supernet[node_id][input_id][operation_id] = 0.0
    return supernet


def selected_specific_value(supernet, node_id, input_id, operation_id):
    for i in range(len(supernet[node_id][input_id])):
        if i != operation_id:
            supernet[node_id][input_id][i] = 0.0
    return supernet



def name_compression_encoder(uncompressed_supernet, layer_type):
    supernet = copy.deepcopy(uncompressed_supernet)

    connectivity_domain = [0.0, 1.0]

    mix_operator = [p for p in itertools.product(connectivity_domain, repeat=len(layer_type))]
    for i in range(len(mix_operator)):
        mix_operator[i] = list(mix_operator[i])

    for i in range(len(supernet)):
        for j in range(len(supernet[i])):
            if type(supernet[i][j]) is list:
                for p in range(len(mix_operator)):
                    if supernet[i][j] == mix_operator[p]:
                        supernet[i][j] = p
    return supernet

def name_compression_decoder(compressed_supernet, layer_type):
    supernet = copy.deepcopy(compressed_supernet)

    connectivity_domain = [0.0, 1.0]

    mix_operator = [p for p in itertools.product(connectivity_domain, repeat=len(layer_type))]
    for i in range(len(mix_operator)):
        mix_operator[i] = list(mix_operator[i])

    for i in range(len(supernet)):
        for j in range(len(supernet[i])):
            supernet[i][j] = mix_operator[supernet[i][j]]

    return supernet


def layer_type_encoder(layer_type):
    encoded_type = []
    for i in range(len(layer_type)):
        if layer_type[i] == 'skip_connect':
            encoded_type.append(0)
        if layer_type[i] == 'max_pool_3x3':
            encoded_type.append(1)
        if layer_type[i] == 'sep_conv_3x3':
            encoded_type.append(2)
        if layer_type[i] == 'sep_conv_5x5':
            encoded_type.append(3)

    return sorted(encoded_type)


def random_net_generator(supernet, numbers):

    avail_node = []
    for i in range(len(supernet)):
        for j in range(len(supernet[i])):
            if supernet[i][j] != 0:
                avail_node.append([i, j])

    net_list = []
    i = 0
    while True:
        new_net = copy.deepcopy(supernet)
        changed_time = random.randint(1, 10)

        for j in range(changed_time):
            changed_node = random.choice(avail_node)
            changed_value = random.randint(0, 15)
            new_net[changed_node[0]][changed_node[1]] = changed_value

        if new_net not in net_list:
            n = copy.deepcopy(new_net)
            net_list.append(n)
            i += 1

        if i == numbers:
            break
    # print(net_list)
    return net_list


def resume_net_from_file(path):
    with open(path, 'r') as f:
        network = eval(f.read())
    return network



supernet_normal = supernet_generator(node, layer_type)
supernet_normal = mask_specific_value(supernet_normal, 0, 0, 1)

supernet_reduce = supernet_generator(node, layer_type)
supernet_reduce = mask_specific_value(supernet_reduce, 0, 0, 1)

