import os
import logging
import numpy as np
import time
import torch
import argparse
import math
import shutil
from collections import OrderedDict
import pdb

import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch.distributed as dist

from models.resnet import resnet_18, resnet_34, resnet_50
from models.mobilenetv2 import mobilenet_v2

import imagenet
import utils

parser = argparse.ArgumentParser("ImageNet Compression")

# directories for data , saving (ckpt, config, log), pretrained
parser.add_argument('--data_dir', type=str, default='../data/ImageNet/',
                    help='path to dataset')
parser.add_argument('--job_dir', type=str, default='./test_dir',
                    help='path for saving trained models')
parser.add_argument('--use_dali',
                    action='store_true', help='whether use dali module to load data')
parser.add_argument('--use_pretrain', action='store_true',
                    help='whether use pretrain model')
parser.add_argument('--pretrain_dir', type=str, default='',
                    help='pretrain model path')

# training setting
parser.add_argument('--arch', type=str, default='resnet_18',
                    help='architecture')
parser.add_argument('--epochs', type=int, default=120,
                    help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='init learning rate')
parser.add_argument('--lr_decay_step', default='30,60,90', type=str,
                    help='learning rate decay step')
parser.add_argument('--lr_type', default='step', type=str,
                    help='learning rate decay schedule')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--label_smooth', type=float, default=0.01,
                    help='label smoothing')
parser.add_argument('--resume', action='store_true',
                    help='whether continue training from the same directory')

# pruning and quantization setting
parser.add_argument('--bit', default=5, type=int,
                    help='the bit-width of the quantized network')

# test
parser.add_argument('--test_only', action='store_true',
                    help='whether it is test mode')
parser.add_argument('--test_model_dir', type=str, default='',
                    help='test model path')

# device
parser.add_argument('--gpu', type=str, default='0,1,2,3',
                    help='Select gpu to use')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

CLASSES = 1000
print_freq = 128000 // args.batch_size

# save the ckpt, log, config to the sub-directory based on the architecture and bit
if str(args.bit) == '32':
    args.learning_rate = 0.1
    args.sub_job_dir = os.path.join(args.job_dir,
                                    args.arch + "_" + str(args.bit) + "bit",
                                    "fp"
                                    )
else:
    args.sub_job_dir = os.path.join(args.job_dir,
                                    args.arch + "_" + str(args.bit) + "bit",
                                    )

if args.test_model_dir == '':
    args.test_model_dir = os.path.join(args.sub_job_dir,
                                       "model_best.pth.tar")

if not os.path.isdir(args.sub_job_dir):
    os.makedirs(args.sub_job_dir)

if not args.test_only:
    if args.resume:
        filemode = 'a+'
    else:
        filemode = 'w'
    utils.record_config(args)
    utils.setup_logging(log_file=os.path.join(args.sub_job_dir, 'train_logger' + '.log'),
                        filemode=filemode)
else:
    utils.setup_logging(log_file=os.path.join(args.sub_job_dir, 'test_logger' + '.log'),
                        filemode="w")


def main():
    cudnn.benchmark = True
    cudnn.enabled = True

    start_t = time.time()
    # load model
    logging.info("args = %s", args)
    logging.info('model bit: {}'.format(args.bit))
    logging.info('==> Building model..')
    model = eval(args.arch)(bit=args.bit)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = utils.CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    # load training data
    print('==> Preparing data..')
    data_tmp = imagenet.Data(args)
    train_loader = data_tmp.train_loader
    val_loader = data_tmp.test_loader

    if args.test_only:
        if os.path.isfile(args.test_model_dir):
            logging.info('loading checkpoint {} ..........'.format(args.test_model_dir))
            checkpoint = torch.load(args.test_model_dir)
            if 'state_dict' in checkpoint:
                tmp_ckpt = checkpoint['state_dict']
            else:
                tmp_ckpt = checkpoint

            new_state_dict = OrderedDict()
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)
            model.cuda()

            valid_obj, valid_top1_acc, valid_top5_acc = validate(0, val_loader, model, criterion, args)
        else:
            logging.info('please specify a checkpoint file')

        return

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    start_epoch = 0
    best_top1_acc = 0
    best_top5_acc = 0

    # load the checkpoint if it exists
    checkpoint_dir = os.path.join(args.sub_job_dir, 'checkpoint.pth.tar')
    if args.resume:
        logging.info('loading checkpoint {} ..........'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        start_epoch = checkpoint['epoch'] + 1
        best_top1_acc = checkpoint['best_top1_acc']
        if 'best_top5_acc' in checkpoint:
            best_top5_acc = checkpoint['best_top5_acc']

        # deal with the single-multi GPU problem
        new_state_dict = OrderedDict()
        tmp_ckpt = checkpoint['state_dict']
        if len(args.gpu) > 1:
            for k, v in tmp_ckpt.items():
                new_state_dict['module.' + k.replace('module.', '')] = v
        else:
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v

        model.load_state_dict(new_state_dict)
        logging.info("loaded checkpoint {} epoch = {}".format(checkpoint_dir, checkpoint['epoch']))
    else:
        if args.use_pretrain:
            if "resnet" in args.arch or "mobilenet_v2" in args.arch:
                if args.bit == 5:
                    # use the pretrained weight in the torchvision.
                    pass
                elif args.bit < 5:
                    ckpt = torch.load(os.path.join(args.job_dir,
                                                   args.arch + "_" + str(args.bit + 1) + "bit",
                                                   "model_best.pth.tar"
                                                   ))
                    model.load_state_dict(ckpt['state_dict'])

        else:
            logging.info('training from scratch')

    params_count = sum([p.numel() for p in model.module.parameters()])
    logging.info('After Load Weight, Params: %.2f (M)' % (params_count / (10 ** 6)))

    for epoch in range(start_epoch):
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()

    epoch = start_epoch
    while epoch < args.epochs:
        if epoch % 5 == 0:
            model.module.show_params() if len(args.gpu) > 1 else model.show_params()
        train_obj, train_top1_acc, train_top5_acc = train(epoch, train_loader, model, criterion_smooth, optimizer)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)
        scheduler.step()

        if args.use_dali:
            train_loader.reset()
            val_loader.reset()

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            best_top5_acc = valid_top5_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'best_top5_acc': best_top5_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.sub_job_dir)

        epoch += 1
        logging.info("{} bit W/A ==>Best accuracy Top1: {:.3f}, Top5: {:.3f}".format(
            args.bit, best_top1_acc, best_top5_acc))

    training_time = (time.time() - start_t) / 36000
    logging.info('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    if args.use_dali:
        num_iter = train_loader._size // args.batch_size
    else:
        num_iter = len(train_loader)

    print_freq = num_iter // 10

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        # compute output
        logits = model(images)
        loss = criterion(logits, targets)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logging.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    if args.use_dali:
        num_iter = val_loader._size // args.batch_size
    else:
        num_iter = len(val_loader)

    model.eval()
    with torch.no_grad():
        end = time.time()
        if args.use_dali:
            for batch_idx, batch_data in enumerate(val_loader):
                images = batch_data[0]['data'].cuda()
                targets = batch_data[0]['label'].squeeze().long().cuda()

                # compute output
                logits = model(images)
                loss = criterion(logits, targets)

                # measure accuracy and record loss
                pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        else:
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.cuda()
                targets = targets.cuda()

                # compute output
                logits = model(images)
                loss = criterion(logits, targets)

                # measure accuracy and record loss
                pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))

                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        logging.info(' Model {arch}: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                     .format(arch=args.arch, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
