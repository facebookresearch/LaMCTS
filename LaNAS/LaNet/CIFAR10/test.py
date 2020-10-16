# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import sys
import utils
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from collections import namedtuple
from model import NetworkCIFAR as Network
from utils import *
from torch.utils.data.dataset import Subset
import logging
from nasnet_set import *



parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--layers', type=int, default=24, help='total number of layers')
parser.add_argument('--init_ch', type=int, default=36, help='num of init channels')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='', help='which architecture to use')
parser.add_argument('--checkpoint', type=str, default='', help='load from checkpoint')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')





args = parser.parse_args()


net = eval(args.arch)
print(net)
code = gen_code_from_list(net, node_num=int((len(net) / 4)))
genotype = translator([code, code], max_node=int((len(net) / 4)))
print(genotype)





log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.checkpoint, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main():

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)


    model = Network(args.init_ch, 10, args.layers, True, genotype).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    checkpoint = torch.load(args.checkpoint + '/top1.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.CrossEntropyLoss().cuda()


    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])




    valid_queue = torch.utils.data.DataLoader(
            dset.CIFAR10(root=args.data, train=False, transform=valid_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)


    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc: %f', valid_acc)



def infer(valid_queue, model, criterion):

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
            logging.info('>>Validation: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)



    return top1.avg, objs.avg



if __name__ == '__main__':
    main()