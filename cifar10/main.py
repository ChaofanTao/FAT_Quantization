import os
import numpy as np
import time, datetime
import argparse
import copy
import logging
# from thop import profile
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from models.vgg import vgg_7_bn, vgg_16_bn
from models.resnet import resnet_20, resnet_32, resnet_56, resnet_110
import data_preprocess
import utils

import pdb

parser = argparse.ArgumentParser("Cifar-10 Compression")

# directories for data , saving (ckpt, config, log), pretrained
parser.add_argument('--data_dir', type=str, default='./dataset/cifar10',
                    help='path to dataset')
parser.add_argument('--job_dir', type=str, default='./test_dir',
                    help='path for saving trained models')
parser.add_argument('--use_pretrain', action='store_true',
                    help='whether use pretrain model')
parser.add_argument('--pretrain_dir', type=str, default='',
                    help='pretrain model path')

# training setting
parser.add_argument('--arch', type=str, default='resnet_56',
                    help='architecture')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=400,
                    help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.04,
                    help='init learning rate')
parser.add_argument('--lr_decay_step', default='150,225', type=str,
                    help='learning decay step')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--resume', action='store_true',
                    help='whether continue training from the same directory')

# quantization setting
parser.add_argument('--bit', default=5, type=int,
                    help='the bit-width of the quantized network')

# test
parser.add_argument('--test_only', action='store_true',
                    help='whether it is test mode')
parser.add_argument('--test_model_dir', type=str, default='',
                    help='test model path')

# device
parser.add_argument('--gpu', type=str, default='0',
                    help='Select gpu or gpus, e.g. 0,1,2,3')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
CLASSES = 10
print_freq = (256 * 50) // args.batch_size

# save the ckpt, log, config to the sub-directory based on the architecture and bit
if str(args.bit) == '32':
    args.learning_rate = 0.1
    args.sub_job_dir = os.path.join(args.job_dir,
                                    args.arch + "_" + str(args.bit) + "bit",
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

if len(args.gpu) > 1:
    name_base = 'module.'
else:
    name_base = ''


def main():
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info("args = %s", args)

    # load model
    logging.info('model bit: {}'.format(args.bit))
    logging.info('==> Building model..')
    model = eval(args.arch)(bit=args.bit).cuda()

    # load training data
    train_loader, val_loader = data_preprocess.load_data(args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    if args.test_only:
        # import pdb;pdb.set_trace()
        if os.path.isfile(args.test_model_dir):
            logging.info('loading checkpoint {} ..........'.format(args.test_model_dir))
            checkpoint = torch.load(args.test_model_dir)
            new_state_dict = {}
            if len(args.gpu) == 1:
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k.replace('module.', '')] = v
            else:
                new_state_dict = checkpoint['state_dict']
            model.load_state_dict(new_state_dict)
            valid_obj, valid_top1_acc, valid_top5_acc = validate(0, val_loader, model, criterion, args)
        else:
            logging.info('please specify a checkpoint file')
        return

    logging.info(model)

    model_params = []
    all_params = model.module.named_parameters() if len(args.gpu) > 1 else model.named_parameters()
    params_count = 0
    for name, params in all_params:
        params_count += params.numel()
        if 'act_alpha' in name:
            model_params += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
        elif 'wgt_alpha' in name:
            model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]
        elif 'transform' in name:
            model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]
        elif 'threshold' in name:
            model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]
        else:
            model_params += [{'params': [params]}]
    del all_params
    # optimizer = torch.optim.Adam(model_params, lr=args.learning_rate,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model_params, lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    start_epoch = 0
    best_top1_acc = 0

    # load the checkpoint if it exists
    checkpoint_dir = os.path.join(args.sub_job_dir, 'checkpoint.pth.tar')
    if args.resume:
        logging.info('loading checkpoint {} ..........'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        start_epoch = checkpoint['epoch'] + 1
        best_top1_acc = checkpoint['best_top1_acc']

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
            logging.info('resuming from pretrain model')
            if args.bit == 5:
                init_bit = 32
                ckpt = torch.load(os.path.join(args.job_dir,
                                               args.arch + "_" + str(init_bit) + "bit",
                                               "model_best.pth.tar"
                                               ),
                                  map_location='cuda:0')
            else:
                ckpt = torch.load(os.path.join(args.job_dir,
                                               args.arch + "_" + str(args.bit + 1) + "bit",
                                               "model_best.pth.tar"
                                               ),
                                  map_location='cuda:0')

            model.load_state_dict(ckpt['state_dict'], strict=False)

        else:
            logging.info('training from scratch')

    if len(args.gpu) > 1:
        params_count = sum([p.numel() for p in model.module.parameters()])
    else:
        params_count = sum([p.numel() for p in model.parameters()])
    logging.info('After Load Weight, Params: %.2f (M)' % (params_count / (10 ** 6)))
    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        if epoch % 10 == 0:
            model.module.show_params() if len(args.gpu) > 1 else model.show_params()
        train_obj, train_top1_acc, train_top5_acc = train(epoch, train_loader, model, criterion, optimizer, scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.sub_job_dir)

        epoch += 1
        logging.info("{} bit W/A ==>Best accuracy {:.3f}".format(args.bit, best_top1_acc))  #


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    # for p in model.parameters():
    #     p.retain_grad()

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
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

        if i % print_freq == 0:
            logging.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, i, num_iter, loss=losses,
                    top1=top1, top5=top5))

    scheduler.step()

    return losses.avg, top1.avg, top5.avg


def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logging.info(' Model {arch}: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                     .format(arch=args.arch, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
