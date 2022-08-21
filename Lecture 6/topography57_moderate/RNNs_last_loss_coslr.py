from __future__ import print_function

import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
from dataloader import Dataset

from utils import Logger, AverageMeter, accuracy, mkdir_p

import math
import random
import numpy as np

import argparse
import os

import models.rnn as models

# hyper parameters for training
parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--epoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--seq_length', default=1000, type=int, metavar='N',
                    help='length of training sequence')
parser.add_argument('--iters', default=20, type=int, metavar='N',
                    help='number of iters each epoch to run')
parser.add_argument('--train-batch', default=50, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[15, 25],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# checkpoints setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/simpleRNN', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# # Architecture
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

# save config
with open(args.checkpoint + "/Config.txt", 'w') as f:
    for (k, v) in args._get_kwargs():
        f.write(k + ' : ' + str(v) + '\n')

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

def main():
    # model
    model = models.SimpleRNN(input_size = 2, hidden_size= 50, output_size = 1)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    with open(args.checkpoint + "/Config.txt", 'w') as f:
        f.write('Total params: %.2f' % (sum(p.numel() for p in model.parameters())) + '\n')
    print('    Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # logger
    logger = Logger(os.path.join(args.checkpoint,'log.txt'), title = 'log')
    logger.set_names(['Learning Rate.', 'Train Loss.'])

    # data loader
    training_set = Dataset('data/', seq_length=args.seq_length)
    sampler = torch.utils.data.RandomSampler(training_set, replacement=True, num_samples=args.train_batch * args.iters)
    trainloader = data.DataLoader(training_set, sampler=sampler, batch_size=args.train_batch, num_workers=8)

    for epoch in range(args.epoch):
        # adjust_learning_rate(optimizer, epoch)
        lr = cosine_lr(optimizer, args.lr, epoch,args.epoch)
        print(optimizer.param_groups[0]['lr'])
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, state['epoch'], lr))
        train_loss = train(trainloader, model, criterion, optimizer, use_cuda = True)

        # append logger file
        logger.append([state['lr'], train_loss])
        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, checkpoint=state['checkpoint'])
    logger.close()

def train(trainloader, model, criterion, optimizer, use_cuda):

    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    loss_item = 0
    for idx, (x_t, y_t, y_tt) in enumerate(trainloader):
        # x_t of size batch size * seq_length
        inputs = torch.cat((x_t.unsqueeze(2), y_t.unsqueeze(2)), 2).transpose(0,1)   # seq_length * batch size * 2
        targets = y_tt.transpose(0,1).unsqueeze(2)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs.float()), torch.autograd.Variable(targets.float())

        # compute output
        outputs = model(inputs)
        # loss = criterion(outputs, targets)

        # print(loss)
        # losses.update(loss.item(), inputs.size(0))
        # loss = criterion(outputs, targets)
        loss_unreduced = torch.mean(torch.nn.functional.mse_loss(outputs, targets, reduce=False), dim=1)[-1, 0]
        losses.update(loss_unreduced.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_unreduced.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        LR = optimizer.param_groups[0]['lr']
        suffix = 'Train_Loss::{loss:.4f} lr::{lr:.8f}'.format(loss = losses.avg, lr = LR)
        print(suffix)

    return losses.avg

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in state['schedule']:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

def save_checkpoint(state, is_best = False, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    # if is_best:
    #     torch.save(state, filepath)

if __name__ == '__main__':
    main()