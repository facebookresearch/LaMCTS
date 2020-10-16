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
import json
import hashlib
import  apex


from model import NetworkCIFAR as Network



def run(net, init_ch=32, layers=20, auxiliary=True, lr=0.025, momentum=0.9, wd=3e-4, cutout=True, cutout_length=16, data='../data', batch_size=96, epochs=600, drop_path_prob=0.2, auxiliary_weight=0.4):
    save = '/checkpoint/linnanwang/nasnet/' + hashlib.md5(json.dumps(net).encode()).hexdigest()
    utils.create_exp_dir(save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


    np.random.seed(0)
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(0)
    logging.info('gpu device = %d' % 0)
    # logging.info("args = %s", args)


    genotype = net
    model = Network(init_ch, 10, layers, auxiliary, genotype).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=momentum,
        weight_decay=wd
    )
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O3")



    train_transform, valid_transform = utils._data_transforms_cifar10(cutout, cutout_length)
    train_data = dset.CIFAR10(root=data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

    best_acc = 0.0

    for epoch in range(epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = drop_path_prob * epoch / epochs


        train_acc, train_obj = train(train_queue, model, criterion, optimizer, auxiliary=auxiliary, auxiliary_weight=auxiliary_weight)
        logging.info('train_acc: %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc: %f', valid_acc)

        if valid_acc > best_acc and epoch >= 50:
            print('this model is the best')
            torch.save(model.state_dict(), os.path.join(save, 'model.pt'))
        if valid_acc > best_acc:
            best_acc = valid_acc
        print('current best acc is', best_acc)

        if epoch == 100:
            break

        # utils.save(model, os.path.join(args.save, 'trained.pt'))
        print('saved to: model.pt')

    return best_acc


def train(train_queue, model, criterion, optimizer, auxiliary=True, auxiliary_weight=0.4, grad_clip=float(5), report_freq=float(50)):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (x, target) in enumerate(train_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(x)
        loss = criterion(logits, target)
        if auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = x.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, report_freq=float(50)):

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

        if step % report_freq == 0:
            logging.info('>>Validation: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg



