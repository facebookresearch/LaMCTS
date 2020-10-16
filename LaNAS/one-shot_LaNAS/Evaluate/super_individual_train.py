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
import utils
import logging

import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from individual_model import Network
# from generator import *
from generator import name_compression_encoder
from generator import layer_type_encoder
import hashlib
import argparse
import translator



parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_ch', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--exp_path', type=str, default='exp/cifar10', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--supernet_normal', type=str, default='[[[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]'
, help='experiment name')
parser.add_argument('--supernet_reduce', type=str, default='[[[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]'
, help='experiment name')

parser.add_argument('--masked_code', type=str, default=None, help='masked_code')

args = parser.parse_args()




def run():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  cur_epoch = 0

  layers_type = [
      'max_pool_3x3',
      'skip_connect',
      'sep_conv_3x3',
      'sep_conv_5x5'
  ]
  # supernet_normal = eval(args.supernet_normal)
  # supernet_reduce = eval(args.supernet_reduce)


  supernet_normal, supernet_reduce = translator.encoding_to_masks(eval(args.masked_code))
  supernet_normal = translator.expend_to_supernet_code(supernet_normal)
  supernet_reduce = translator.expend_to_supernet_code(supernet_reduce)

  if not continue_train:
    print('train from scratch!')

    model = Network(supernet_normal, supernet_reduce, layers_type, args.init_ch, CIFAR_CLASSES, args.layers,
                    args.auxiliary,
                    steps=len(supernet_normal), multiplier=len(supernet_normal))
    # model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))



  else:
    print('train from checkpoints')
    # model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)

    model = Network(supernet_normal, supernet_reduce, layers_type, args.init_ch, CIFAR_CLASSES, args.layers,
                    args.auxiliary,
                    steps=len(supernet_normal))
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
      model.parameters(),
      args.lr,
      momentum=args.momentum,
      weight_decay=args.wd
    )

    checkpoint = torch.load(args.save + '/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    cur_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = checkpoint['scheduler']





  train_transform, valid_transform = utils._data_transforms_cifar10(args, args.cutout_length)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)



  best_acc = 0.0


  for epoch in range(cur_epoch, args.epochs):
    print("=====> current epoch:", epoch)
    logging.info('=====> current epoch: %d', epoch)

    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    # train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    train_acc, train_obj = train(train_queue, model, criterion, optimizer, args.grad_clip, args.report_freq)

    scheduler.step()
    logging.info('train_acc %f', train_acc)

    # valid_acc, valid_obj = infer(valid_queue, model, criterion)
    valid_acc, valid_obj = infer(valid_queue, model, criterion, args.report_freq)
    logging.info('valid_acc %f', valid_acc)

    if valid_acc > best_acc:
      best_acc = valid_acc
      print('this model is the best')
      logging.info('this model is the best')
      torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'scheduler': scheduler,
                  'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save, 'top_1.pt'))

    torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'scheduler': scheduler,
                'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save, 'model.pt'))

    logging.info('best_acc: %f', best_acc)
    print('current best acc is', best_acc)



def train(train_queue, model, criterion, optimizer, grad_clip, report_freq):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = input.cuda()
    target = target.cuda(non_blocking=True)


    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, report_freq):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

supernet_normal, supernet_reduce = translator.encoding_to_masks(eval(args.masked_code))
supernet_normal = translator.expend_to_supernet_code(supernet_normal)
supernet_reduce = translator.expend_to_supernet_code(supernet_reduce)

layers = [
      'max_pool_3x3',
      'skip_connect',
      'sep_conv_3x3',
      'sep_conv_5x5'
  ]
s = name_compression_encoder(supernet_normal, layers)
s.append(layer_type_encoder(layers))

# save = hashlib.md5(str(s).encode('utf-8')).hexdigest()
args.save = str(s)



continue_train = False
if os.path.exists(args.save + '/model.pt'):
  continue_train = True

if not continue_train:
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
CIFAR_CLASSES = 10
run()
