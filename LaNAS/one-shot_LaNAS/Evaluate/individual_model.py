# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
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

    def forward(self, x):
        return sum(op(x) for op in self._ops)
        # return sum(w * op(x) for w, op in zip(weights, self._ops))  # use sum instead concat


class Cell(nn.Module):
    def __init__(self, layer_type, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, supernet_matrix):
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
                op_list = []
                for type in range(len(supernet_matrix[i][j])):
                    if supernet_matrix[i][j][type] == 1.0:
                        op_list.append(layer_type[type])
                stride = 2 if reduction and j < 2 else 1

                op = MixedOp(C, stride, layer_type=op_list)
                self._ops.append(op)


    def forward(self, s0, s1, supernet_matrix, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            H = []
            op = []
            for j, h in enumerate(states):
                if len(self._ops[offset + j]._ops) != 0:
                    H.append(self._ops[offset + j](h))
                    op.append(self._ops[offset + j])

            if self.training and drop_prob > 0.:
                for hn_index in range(len(H)):
                    if len(op[hn_index]._ops) != 0:
                        if not isinstance(op[hn_index]._ops[0], Identity):
                        # print(len(op[hn_index]._ops))
                        # print(op[hn_index]._ops[0])
                        # print(op[hn_index])
                            H[hn_index] = drop_path(H[hn_index], drop_prob)

            s = sum(hn for hn in H)

            offset += len(states)
            states.append(s)
        return torch.cat(states[-len(supernet_matrix):], dim=1)



def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob

        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        try:
            x.mul_(mask)
        except:
            mask = torch.cuda.HalfTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            x.mul_(mask)
    return x


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Network(nn.Module):
    def __init__(self, supernet_normal, supernet_reduce, layer_type, C, num_classes, layers, auxiliary, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self.supernet_normal = supernet_normal
        self.supernet_reduce = supernet_reduce
        self.layer_type = layer_type
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._auxiliary = auxiliary
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
                cell = Cell(self.layer_type, len(self.supernet_reduce), multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.supernet_reduce)
            else:
                reduction = False
                cell = Cell(self.layer_type, len(self.supernet_normal), multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.supernet_normal)

            reduction_prev = reduction
            self.cells += [cell]

            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)



    def forward(self, input):

        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if not cell.reduction:
                s0, s1 = s1, cell(s0, s1, self.supernet_normal, self.drop_path_prob)
            else:
                s0, s1 = s1, cell(s0, s1, self.supernet_reduce, self.drop_path_prob)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
