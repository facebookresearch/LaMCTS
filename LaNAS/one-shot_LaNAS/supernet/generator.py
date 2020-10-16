# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
import random
import copy

node = 5
layer_type = [
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5'
]

def check_avail(net, node=5):
    is_avail = True
    base = 0
    for i in range(node):
        if sum(net[base : base + (i+2) * 4]) != 2:
            is_avail = False
        base += (i+2) * 4
    return is_avail

def create_index_list(nums, shift=0):
    index_list = []
    for i in range(nums):
        index_list.append(i+shift)
    return index_list

def zero_supernet_generator(node, layer_type, is_int=False):
    vec_length = len(layer_type)
    masked_vec = np.zeros((1, vec_length))[0].tolist()
    disconnected_vec = np.zeros((1, vec_length))[0].tolist()
    supernet = [[] for v in range(node)]
    for i in range(node):
        for j in range(node + 2):
            if j < i + 2:
                supernet[i].append(masked_vec.copy())
            else:
                supernet[i].append(disconnected_vec.copy())
    if is_int:
        for i in range(len(supernet)):
            for j in range(len(supernet[i])):
                for n in range(len(supernet[i][j])):
                    supernet[i][j][n] = int(supernet[i][j][n])
    return supernet
    
def sampled_nets_generator(based_net, nums=1000):
    nets_dict = {}
    nets_list = []
    for n in range(nums):
        normal = random_supernet_generator(base_supernet=based_net)
        reduce = random_supernet_generator(base_supernet=based_net)
        sample = flatten_to_1D_vector(normal, reduce)
        assert check_avail(sample)
        if str(sample) not in nets_dict:
            nets_dict[str(sample)] = 1
            nets_list.append(sample)
    return nets_list
    
def flatten_to_1D_vector(normal, reduce):
    oneD_normal = []
    oneD_reduce =[]
    for i in range(len(normal)):
        for j in range(i+2):
            for n in range(len(normal[i][j])):
                oneD_normal.append(normal[i][j][n])
                oneD_reduce.append(reduce[i][j][n])
    assert len(oneD_normal) == (sum(create_index_list(len(normal), shift=2)) * 4)
    assert len(oneD_reduce) == (sum(create_index_list(len(normal), shift=2)) * 4)
    oneD_normal.extend(oneD_reduce)
    assert len(oneD_normal) == (sum(create_index_list(len(normal), shift=2)) * 4 * 2)

    return oneD_normal

def random_supernet_generator(base_supernet):
    index_list = []
    for i in range(len(base_supernet)):
        index_list.append(create_index_list((i+2) * 4))

    sample_net = copy.deepcopy(base_supernet)
    for i in range(len(base_supernet)):
        selected_index = random.sample(index_list[i], 2)
        assert selected_index[0] < (i + 2) * 4 and selected_index[1] < (i + 2) * 4
        sample_net[i][selected_index[0] // 4][selected_index[0] % 4] = 1.0
        sample_net[i][selected_index[1] // 4][selected_index[1] % 4] = 1.0

    return sample_net

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
                supernet[i].append(get_rand_vector(layer_type) )
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
    encoding = np.array(encoding).reshape( -1, 4 )
    supernet_normal = supernet_generator(node, layer_type)
    supernet_reduce = supernet_generator(node, layer_type)
    supernet        = [supernet_normal, supernet_reduce]
    mask            = []
    counter         = 0
    for cell in supernet:
        mask_cell = []
        for row in cell:
            mask_row = []
            for col in row:
                if type(col) == type([]):
                    mask_row.append( encoding[counter].tolist() )
                    counter += 1
                else:
                    mask_row.append( 0 )
            mask_cell.append( mask_row )
        mask.append( mask_cell )
    
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
    supernet        = [supernet_normal, supernet_reduce]
    
    layer_types_count = len( layer_type ) 
    count = 0
    assert type(supernet) == type([])
    for cell in supernet:
        for row in cell:
            for col in row:
                if type(col) == type([]):
                    count += layer_types_count 
    return np.ones( (count) ).tolist()

def define_search_space():
    # hit-and-run default
    # A x <= b
    search_space = encode_supernet()
    A = []
    b = []
    init_point = []
    param_pos = 0
    for i in range(0, len(search_space) ):
        tmp = np.zeros( len(search_space) )
        tmp[i] = 1
        A.append( np.copy(tmp) )
        b.append( 1.0000001 )
        #we need relax a little bit here for the precision issue
        # A*x <= 1, we use 1.000+epsilon
        # A*x >= 0, we use 0-epslon
        A.append( -1*np.copy(tmp) )
        b.append( 0.0000001 )
    for i in range(0, len(search_space) ):
        if random.random() >= 0.5:
            init_point.append(0.0)
        else:
            init_point.append(1.0)
    return {"A":np.array(A), "b":np.array(b), "init_point":np.array(init_point) }

supernet_normal = supernet_generator(node, layer_type)
supernet_reduce = supernet_generator(node, layer_type)
# print(supernet_normal)
# print(supernet_reduce)

# based_net = zero_supernet_generator(node, layer_type, is_int=True)
# print(based_net)
# net_list = sampled_nets_generator(based_net, nums=10)
# print(len(net_list))
# print(net_list[0])
