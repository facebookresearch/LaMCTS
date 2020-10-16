# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *
from torch.autograd import Variable



class MixedOp(nn.Module):
    def __init__(self, C, stride, layer_type):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for layer in layer_type:
            op = OPS[layer](C, stride, False)
            if 'pool' in layer:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        # print(len(weights), len(self._ops))
        return sum(w * op(x) for w, op in zip(weights, self._ops))  # use sum instead concat


class Cell(nn.Module):
    def __init__(self, layer_type, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, layer_type)
                self._ops.append(op)

    def forward(self, s0, s1, supernet_matrix):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, supernet_matrix[i][j]) for j, h in enumerate(states))

            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, supernet_normal, supernet_reduce, layer_type, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self.supernet_normal = supernet_normal
        self.supernet_reduce = supernet_reduce
        self.layer_type = layer_type
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                cell = Cell(self.layer_type, len(self.supernet_reduce), multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            else:
                reduction = False
                cell = Cell(self.layer_type, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def change_masks(self, normal_mask, reduce_mask):            
        self.supernet_normal = normal_mask
        self.supernet_reduce = reduce_mask

    def forward(self, input):       
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if not cell.reduction:
                s0, s1 = s1, cell(s0, s1, self.supernet_normal)
            else:
                s0, s1 = s1, cell(s0, s1, self.supernet_reduce)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

