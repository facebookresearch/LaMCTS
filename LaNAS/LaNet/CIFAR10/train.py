# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import  os
import  sys
import  time
import  glob
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
import  torch.utils
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn
from nasnet_set import *
from collections import namedtuple
import hashlib
from datetime import datetime
from model import NetworkCIFAR as Network
from utils import ModelEma
from utils import *
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from sklearn.model_selection._split import StratifiedShuffleSplit
from torch.utils.data.dataset import Subset




parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
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





parser.add_argument('--model_ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model_ema_force_cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

parser.add_argument('--track_ema', action='store_true', default=False, help='track ema')
parser.add_argument('--auto_augment', action='store_true', default=False)





args = parser.parse_args()
args.save = 'checkpoints/auto_aug-1500-{}-{}-{}-{}-{}-{}-{}'.format(args.arch, args.init_ch, args.model_ema,
                                                         args.model_ema_decay, args.drop_path_prob,
                                                               args.layers, args.cutout_length)


print("save path:", args.save)


continue_train = False
if os.path.exists(args.save+'/model.pt'):
    continue_train = True

if not continue_train:
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def flatten_params(model):
    return torch.sum( torch.cat([param.data.view(-1) for param in model.parameters()], 0))



def main():


    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    cur_epoch = 0

    net = eval(args.arch)
    print(net)
    code = gen_code_from_list(net, node_num=int((len(net) / 4)))
    genotype = translator([code, code], max_node=int((len(net) / 4)))
    print(genotype)

    model_ema = None

    if not continue_train:

        print('train from the scratch')
        model = Network(args.init_ch, 10, args.layers, args.auxiliary, genotype).cuda()
        print("model init params values:", flatten_params(model))

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


        criterion = CutMixCrossEntropyLoss(True).cuda()


        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd
        )

        if args.model_ema:
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '')


    else:
        print('continue train from checkpoint')

        model = Network(args.init_ch, 10, args.layers, args.auxiliary, genotype).cuda()

        criterion = CutMixCrossEntropyLoss(True).cuda()

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd
        )

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        checkpoint = torch.load(args.save+'/model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if args.model_ema:

            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume=args.save + '/model.pt')


    train_transform, valid_transform = utils._auto_data_transforms_cifar10(args)

    ds_train = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)


    args.cv = -1
    if args.cv >= 0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        sss = sss.split(list(range(len(ds_train))), ds_train.targets)
        for _ in range(args.cv + 1):
            train_idx, valid_idx = next(sss)
        ds_valid = Subset(ds_train, valid_idx)
        ds_train = Subset(ds_train, train_idx)
    else:
        ds_valid = Subset(ds_train, [])

    train_queue = torch.utils.data.DataLoader(
        CutMix(ds_train, 10,
               beta=1.0, prob=0.5, num_mix=2),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        dset.CIFAR10(root=args.data, train=False, transform=valid_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)



    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    best_acc = 0.0

    if continue_train:
        for i in range(cur_epoch+1):
            scheduler.step()

    for epoch in range(cur_epoch, args.epochs):
        print('cur_epoch is', epoch)
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        if model_ema is not None:
            model_ema.ema.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch, model_ema)
        logging.info('train_acc: %f', train_acc)

        if model_ema is not None and not args.model_ema_force_cpu:
            valid_acc_ema, valid_obj_ema = infer(valid_queue, model_ema.ema, criterion, ema=True)
            logging.info('valid_acc_ema %f', valid_acc_ema)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc: %f', valid_acc)


        if valid_acc > best_acc:
            best_acc = valid_acc
            print('this model is the best')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save, 'top1.pt'))
        print('current best acc is', best_acc)
        logging.info('best_acc: %f', best_acc)

        if model_ema is not None:
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'state_dict_ema': get_state_dict(model_ema)},
                os.path.join(args.save, 'model.pt'))

        else:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save, 'model.pt'))

        print('saved to: trained.pt')


def train(train_queue, model, criterion, optimizer, epoch, model_ema=None):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()

    for step, (x, target) in enumerate(train_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(x)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        losses.update(loss.item(), x.size(0))

        if len(target.size()) == 1:
            # print(target.size())
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()


        if model_ema is not None:
            model_ema.update(model)

        if step % args.report_freq == 0:
            logging.info('train %03d', step)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, ema=False):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (x, target) in enumerate(valid_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits, _ = model(x)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)


        if step % args.report_freq == 0:
            if not ema:
                logging.info('>>Validation: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            else:
                logging.info('>>Validation_ema: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)


    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
