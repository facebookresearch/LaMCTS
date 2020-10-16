# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import os
import sys
import time
import glob
import numpy as np
import torch
import hashlib
import timeit

import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from .utils import _data_transforms_cifar10
from .utils import count_parameters_in_MB
from .utils import create_exp_dir
from .utils import AvgrageMeter
from .utils import accuracy

from .generator import *
from .supernet_model import Network
from .generator import mask_rand_generator
import json
from .generator import sampled_nets_generator

class Trainer:
    
    def __init__( self, 
        data='../data', batch_size=64, 
        learning_rate=0.025, learning_rate_min=0.001, 
        momentum=0.9, weight_decay=3e-4, 
        report_freq=50, gpu=0, 
        epochs=50, 
        init_channels=16, layers=8, 
        cutout=False, cutout_length=16, 
        drop_path_prob=0.3, seed=2, 
        grad_clip=5, save_prefix='EXP',
        init_masks = []):
        
        #assert len(init_masks) > 0
        
        if not torch.cuda.is_available():
          print('no gpu device available')
          sys.exit(1)
        
        #device level hyerparameters
        np.random.seed( seed )
        torch.cuda.manual_seed( seed )
        torch.cuda.set_device(gpu)
        cudnn.enabled=True
        cudnn.benchmark = True
        print('gpu device = %d' % gpu)
        print('data=', data, 'batch_size=', batch_size, 'learning_rate=', learning_rate, 'learning_rate_min=',
              learning_rate_min, 'momentum=', momentum, 'weight_decay=', weight_decay, 'report_freq=', report_freq, 
              'gpu=', gpu, 'epochs=', epochs, 'init_channels=', init_channels, 'layers=', layers, 'cutout=', cutout,
              'cutout_length=', cutout_length, 'drop_path_prob=', drop_path_prob, 'seed=', seed, 'grad_clip=', grad_clip )
        
        savedirs = "supernet-logs"

        continue_train   = False
        if os.path.exists(savedirs + '/model.pt'):
          continue_train = True
         
        #prepare logging
        if not continue_train:
          create_exp_dir(savedirs, scripts_to_save=glob.glob('*.py'))
        
        #training hyperparameters
        self.data              = data
        self.batch_size        = batch_size
        self.learning_rate     = learning_rate
        self.learning_rate_min = learning_rate_min
        self.momentum          = momentum
        self.weight_decay      = weight_decay
        self.report_freq       = report_freq
        self.epochs            = epochs
        self.init_channels     = init_channels
        self.layers            = layers
        self.cutout            = cutout
        self.cutout_length     = 16
        self.drop_path_prob    = drop_path_prob
        self.grad_clip         = grad_clip
        self.save_prefix       = savedirs
        self.start_epoch       = 0
        self.mask_to_train     = init_masks #masks drive the iterations
        CIFAR_CLASSES          = 10
          
        #setup network
        self.criterion  = nn.CrossEntropyLoss()
        self.criterion  = self.criterion.cuda()
        self.supernet   = Network(supernet_normal, supernet_reduce, layer_type, init_channels, CIFAR_CLASSES, layers, self.criterion, steps=len(supernet_normal))
        self.supernet   = self.supernet.cuda()
        self.optimizer  = torch.optim.SGD( self.supernet.parameters(), self.learning_rate, momentum = self.momentum, weight_decay = weight_decay)
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, float(epochs), eta_min = learning_rate_min )
        
        #setup training&test data
        train_transform, valid_transform = _data_transforms_cifar10(cutout, cutout_length)
        train_data = dset.CIFAR10(root=data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=data, train=False, download=True, transform=valid_transform)
        self.train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
        self.valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
        
        print("length of training & valid queue:", len(self.train_queue), len(self.valid_queue) )
        print("param size = %fMB"% count_parameters_in_MB(self.supernet) )
        
        if continue_train:
            print('continue train from checkpoint')
            checkpoint       = torch.load(self.save + '/model.pt')
            self.supernet.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler   = checkpoint['scheduler']
        
        self.curt_epoch = self.start_epoch
        self.curt_step  = 0
        
        self.base_net   = zero_supernet_generator(node, layer_type )
        
        
    def propose_nasnet_mask(self, nums=1):
        net_mask = np.array( sampled_nets_generator(self.base_net, nums=1 ) )
        return net_mask

    def zero_supernet_generator(self):
        vec_length = len(self.layer_type)
        masked_vec = np.zeros((1, vec_length))[0].tolist()
        disconnected_vec = np.zeros((1, vec_length))[0].tolist()
        supernet = [[] for v in range(self.arch_node)]
        for i in range(self.arch_node):
            for j in range(self.arch_node + 2):
                if j < i + 2:
                    supernet[i].append(masked_vec.copy())
                else:
                    supernet[i].append(disconnected_vec.copy())
        for i in range(len(supernet)):
            for j in range(len(supernet[i])):
                for n in range(len(supernet[i][j])):
                    supernet[i][j][n] = int(supernet[i][j][n])
        return supernet
            
    def train( self ):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        
        #evenly divde curt epochs to all masks
        for step, (input, target) in enumerate(self.train_queue):
            mask = self.propose_nasnet_mask()
            curt_normal_mask, curt_reduce_mask = encoding_to_masks( mask )            
            self.supernet.change_masks(curt_normal_mask, curt_reduce_mask)
            
            self.supernet.train()
            n = input.size(0)
            
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.supernet(input)
            loss   = self.criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm_(self.supernet.parameters(), self.grad_clip )
            self.optimizer.step()

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            if step % self.report_freq == 0:
                print('train steps: %03d loss: %e top-1: %f top-5: %f'% (step, objs.avg, top1.avg, top5.avg) )
        return top1.avg, objs.avg
        
    def load_model(self, path):
        print('continue train from checkpoint')
        checkpoint       = torch.load(path)
        self.supernet.load_state_dict(checkpoint['model_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler   = checkpoint['scheduler']
        
    def infer_masks( self, mask ):
        result = {}
        normal_mask, reduce_mask   = encoding_to_masks( mask )
        print("normal mask:", normal_mask)
        print("reduce mask:", reduce_mask)
        valid_acc, valid_obj       = self.infer( [normal_mask, reduce_mask] )
        result[ json.dumps(mask.tolist() ) ] = valid_acc
        # result[ json.dumps(mask.tolist() ) ] = random.random()
        return result
        
    def infer( self, mask ):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        
        self.supernet.change_masks( mask[0], mask[1] )

        self.supernet.eval()

        for step, (input, target) in enumerate(self.valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = self.supernet(input)
            loss   = self.criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.report_freq == 0:
                print('valid step: %03d loss: %e top-1: %f top-5: %f'% ( step, objs.avg, top1.avg, top5.avg ) )

        return top1.avg, objs.avg            

    def run(self):
        best_acc = 0.0
        
        mask = self.propose_nasnet_mask()
        curt_normal_mask, curt_reduce_mask = encoding_to_masks( mask )
        print("proposed norm mask:",  curt_normal_mask)   
        print("proposed reduce mask:", curt_reduce_mask)            
                 
        valid_acc, valid_obj = self.infer( [curt_normal_mask, curt_reduce_mask] )
        
        
        #for epoch in range(self.start_epoch, self.epochs):
        for epoch in range(0, self.epochs):
          # training
          train_acc, train_obj = self.train( )

          self.scheduler.step()
          print('train_acc %f'% train_acc)

          # validation on the entire supernet
          curt_normal_mask, curt_reduce_mask = encoding_to_masks( mask )
          print("proposed norm mask:",  curt_normal_mask)   
          print("proposed reduce mask:", curt_reduce_mask)                   
          valid_acc, valid_obj = self.infer( [curt_normal_mask, curt_reduce_mask] )
          print('valid_acc %f'% valid_acc)
          print('saving the latest model')
          torch.save({'epoch': epoch + 1, 'model_state_dict': self.supernet.state_dict(), 'scheduler': self.scheduler,
                        'optimizer_state_dict': self.optimizer.state_dict()}, os.path.join(self.save_prefix, 'latest_model.pt'))