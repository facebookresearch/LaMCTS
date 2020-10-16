# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import random
from collections import namedtuple
import math
import json
import os
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

Max_node = 6

root = [[0 for col in range(4)] for row in range(Max_node)]


def check_valid(sample, node_num=6):
    valid_sample = []
    invalid_sample = []
    assert len(sample[0]) == node_num * 4
    for i in range(len(sample)):
        flag = True
        for j in range(len(sample[i])):
            if j <= (node_num * 2 - 1):
                if not (0 <= sample[i][j] <= 3):
                    invalid_sample.append(sample[i])
                    flag = False
                    break
            else:
                # print(j, (int((j - 20) / 2) + 1))
                if not (0 <= sample[i][j] <= (int((j - node_num * 2) / 2) + 1)):
                    # print(j, (int((j - 20) / 2) + 1), sample[i][j])
                    invalid_sample.append(sample[i])
                    flag = False
                    break
        if flag:
            valid_sample.append(sample[i])

    print(len(valid_sample), len(invalid_sample))

    return valid_sample, invalid_sample



def get_dist(sample):
    dist = []
    for i in range(len(sample)):
        dist.append(sum(sample[i]))
    return dist



def get_dist2(sample):
    dist = []
    for i in range(len(sample)):
        s = 0
        for j in range(len(sample[i])):
            s += sample[i][j] ** 2
        s = s ** 0.5
        dist.append(s)
    return dist

def gen_code_from_list(sample, node_num=6):
    node = [[-1 for col in range(4)] for row in range(node_num)]
    for i in range(node_num):
        for j in range(4):
            if j <= 1:
                node[i][j] = sample[i * 2 + j]
            else:
                node[i][j] = sample[i * 2 + j + (node_num - 1) * 2]
    return node



operations = ['sep_conv_3x3',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_5x5']

def gen_all_nets(max_node=Max_node):
    block = []
    all_connections = []
    for i in range(max_node):
        connections = []
        for j in range(i+2):
            connections.append(j)
        all_connections.append(connections)
    b = [[] for i in range(max_node)]

    for i in range(max_node):
        for left_type in range(len(operations)):
            for right_type in range(len(operations)):
                for left_input in all_connections[i]:
                    for right_input in all_connections[i]:
                        if [right_type, left_type, left_input, right_input] not in b[i]:
                            b[i].append([left_type, right_type, left_input, right_input])

    sample_nums = 1
    for i in range(max_node):
        sample_nums *= len(b[i])

    print(sample_nums)
    for o in b[0]:
        for t in b[1]:
            for th in b[2]:
                for f in b[3]:
                    if edit_distance([o, t, th, f], root) == 6:
                        block.append([o, t, th, f])

    print(len(block))

    return block



def edit_distance(net_1, net_2):
    distance = 0
    for i in range(len(net_1)):
        if [net_1[i][2], net_1[i][3]] == [net_2[i][2], net_2[i][3]]:
            if net_1[i][0] != net_2[i][0]:
                distance += 1
            if net_1[i][1] != net_2[i][1]:
                distance += 1
        elif [net_1[i][2], net_1[i][3]] == [net_2[i][3], net_2[i][2]]:
            if net_1[i][0] != net_2[i][1]:
                distance += 1
            if net_1[i][1] != net_2[i][0]:
                distance += 1
        elif [net_1[i][0], net_1[i][2]] == [net_2[i][0], net_2[i][2]]:
            if net_1[i][1] != net_2[i][1]:
                distance += 1
            if net_1[i][3] != net_2[i][3]:
                distance += 1
        elif [net_1[i][0], net_1[i][2]] == [net_2[i][1], net_2[i][3]]:
            if net_1[i][1] != net_2[i][0]:
                distance += 1
            if net_1[i][3] != net_2[i][2]:
                distance += 1
        elif [net_1[i][1], net_1[i][3]] == [net_2[i][1], net_2[i][3]]:
            if net_1[i][0] != net_2[i][0]:
                distance += 1
            if net_1[i][2] != net_2[i][2]:
                distance += 1
        elif [net_1[i][1], net_1[i][3]] == [net_2[i][0], net_2[i][2]]:
            if net_1[i][0] != net_2[i][1]:
                distance += 1
            if net_1[i][2] != net_2[i][3]:
                distance += 1
        else:
            if net_1[i][0] in net_2[i][:2]:
                if net_1[i][0] == net_2[i][0]:
                    if net_1[i][2] != net_2[i][2]:
                        distance += 1
                    if net_1[i][1] != net_2[i][1]:
                        distance += 1
                    if net_1[i][3] != net_2[i][3]:
                        distance += 1
                else:
                    if net_1[i][2] != net_2[i][3]:
                        distance += 1
                    if net_1[i][1] != net_2[i][0]:
                        distance += 1
                    if net_1[i][3] != net_2[i][2]:
                        distance += 1
            elif net_1[i][1] in net_2[i][:2]:
                if net_1[i][1] == net_2[i][1]:
                    if net_1[i][2] != net_2[i][2]:
                        distance += 1
                    if net_1[i][0] != net_2[i][0]:
                        distance += 1
                    if net_1[i][3] != net_2[i][3]:
                        distance += 1
                else:
                    if net_1[i][2] != net_2[i][3]:
                        distance += 1
                    if net_1[i][0] != net_2[i][1]:
                        distance += 1
                    if net_1[i][3] != net_2[i][2]:
                        distance += 1
            else:
                distance += 2
                if net_1[i][2] in net_2[i][-2:]:
                    if net_1[i][2] == net_2[i][2]:
                        if net_1[i][3] != net_2[i][3]:
                            distance += 1
                    elif net_1[i][2] == net_2[i][3]:
                        if net_1[i][3] != net_2[i][2]:
                            distance += 1
                elif net_1[i][3] in net_2[i][-2:]:
                    if net_1[i][3] == net_2[i][3]:
                        if net_1[i][2] != net_2[i][3]:
                            distance += 1
                    elif net_1[i][3] == net_2[i][2]:
                        if net_1[i][2] != net_2[i][3]:
                            distance += 1
                else:
                    distance += 2
    return distance


def gen_code(max_node=Max_node):
    normal_node = [[-1 for col in range(4)] for row in range(max_node)]
    reduction_node = [[-1 for col in range(4)] for row in range(max_node)]

    for i in range(max_node):
        for j in range(4):
            if j <= 1:
                normal_node[i][j] = random.randint(0, len(operations) - 1)
                reduction_node[i][j] = random.randint(0, len(operations) - 1)
            else:
                normal_node[i][j] = random.randint(0, i+1)
                reduction_node[i][j] = random.randint(0, i+1)


    concat_code = [normal_node, reduction_node]
    return concat_code


def get_node_depth(cur_node, node_code):
    ## get the depth for current node
    assert cur_node[-1] != -1
    assert cur_node[-2] != -1
    depth_left = 1
    depth_right = 1
    if cur_node[-1] == 0 or cur_node[-1] == 1:
        if cur_node[-2] == 0 or cur_node[-2] == 1:
            return 1
    if not (cur_node[-2] == 0 or cur_node[-2] == 1):
        depth_left += get_node_depth(cur_node=node_code[cur_node[-2] - 2], node_code=node_code)
    if not (cur_node[-1] == 0 or cur_node[-1] == 1):
        depth_right += get_node_depth(cur_node=node_code[cur_node[-1] - 2], node_code=node_code)
    return max(depth_left, depth_right)


def split_and_output(net_code, splitted_num=500, to_file=False):

    sample = int(len(net_code) / splitted_num)
    gpu_nodes = [[] for i in range(splitted_num)]
    for i in range(splitted_num):
        gpu_nodes[i] = net_code[i * sample:(i + 1) * sample]
    remainer = net_code[splitted_num * sample:]
    for i in range(len(remainer)):
        gpu_nodes[i].append(remainer[i])


    for i in range(splitted_num):
        path = './gpu_files/' + 'gpu' + str(i)
        if not os.path.exists(path):
            os.makedirs(path)

    if to_file:
        for i in range(splitted_num):
            outfile = './gpu_files/' + 'gpu' + str(i) + '/splitted_lstm_dataset'
            with open(outfile, 'w') as json_data:
                json.dump(gpu_nodes[i], json_data)

def load_from_file(id=0):
    outfile = 'gpu' + str(id) + '.txt'
    net_list = []
    with open(outfile, 'w') as f:
        while True:
            line = f.readline().strip('\n')
            if line is None:
                break
            net_list.append([json.loads(line), json.loads(line)])
    return net_list



def translator(code, max_node=6):
    # input: code type
    # output: geno type
    n = 0
    normal = []
    normal_concat = []
    reduce_concat = []
    for i in range(max_node+2):
        normal_concat.append(i)
        reduce_concat.append(i)
    reduce = []

    for cell in range(len(code)):
        if cell == 0: # for normal cell
            for block in range(len(code[cell])):
                normal.append((operations[code[cell][block][0]], code[cell][block][2]))
                normal.append((operations[code[cell][block][1]], code[cell][block][3]))
                if code[cell][block][2] in normal_concat:
                    normal_concat.remove(code[cell][block][2])
                if code[cell][block][3] in normal_concat:
                    normal_concat.remove(code[cell][block][3])

        else: # for reduction cell
            for block in range(len(code[cell])):
                reduce.append((operations[code[cell][block][0]], code[cell][block][2]))
                reduce.append((operations[code[cell][block][1]], code[cell][block][3]))
                if code[cell][block][2] in reduce_concat:
                    reduce_concat.remove(code[cell][block][2])
                if code[cell][block][3] in reduce_concat:
                    reduce_concat.remove(code[cell][block][3])

    if 0 in reduce_concat:
        reduce_concat.remove(0)
    if 1 in reduce_concat:
        reduce_concat.remove(1)

    gen_type = Genotype(normal=normal, normal_concat=normal_concat, reduce=reduce, reduce_concat=reduce_concat)

    # print(normal, normal_concat, reduce, reduce_concat)
    return gen_type