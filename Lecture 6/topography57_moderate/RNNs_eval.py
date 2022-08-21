from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
from dataloader import Dataset
from scipy.io import loadmat
from utils import Logger, AverageMeter, accuracy, mkdir_p
import math
import random
import numpy as np
import argparse
import os

import models.rnn as models

# hyper parameters for training
parser = argparse.ArgumentParser(description='hyper parameters')

parser.add_argument('--seq_length_test', default=20, type=int, metavar='N',
                    help='length of training sequence')

# checkpoints setting
parser.add_argument('-c', '--checkpoint', default='simpleRNN', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# # Architecture
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

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

    # load the checkpoint
    checkpoint = torch.load(os.path.join('checkpoint', args.checkpoint, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    # get the residual
    log = np.loadtxt(os.path.join('checkpoint', args.checkpoint, 'log.txt'), skiprows=1)
    var = log[-1, 1]

    # inference
    test(model, var)

def test(model, var):

    model.eval()
    # some fixed number
    dt = 0.05
    damp = 0.1
    Ueq = -0.5
    sigma = 0.447213595499958
    # file_num = 0
    u_true = np.load('data/u.npy')
    F_true = np.load('data/f.npy')

    trajectory_true = np.column_stack((u_true.reshape(-1,1), F_true.reshape(-1,1)))
    trajectory_true = trajectory_true.reshape(trajectory_true.shape[0], 1, trajectory_true.shape[1])

    trajectory_pred = np.zeros((trajectory_true.shape[0], 1, trajectory_true.shape[2]))
    trajectory_pred[:trajectory_true.shape[0]//2, 0:1, :] = trajectory_true[:trajectory_true.shape[0]//2]

    for start in range(trajectory_true.shape[0]//2-10, trajectory_true.shape[0]):
        print(start, '/', trajectory_true.shape[0])
        seq = trajectory_pred[start-args.seq_length_test: start]
        inputs = torch.FloatTensor(seq)
        outputs = model(inputs)
        outputs_np = outputs[-1]
        # print(var)
        trajectory_pred[start, 0, 1] = outputs_np.squeeze(1).cpu().detach().numpy() + np.sqrt(var) * np.random.randn(1)
        trajectory_pred[start, 0, 0] = trajectory_pred[start-1, 0, 0] +  \
                                       dt * (trajectory_pred[start-1, 0, 1] - damp * (trajectory_pred[start-1, 0, 0]-Ueq)) + \
                                       np.sqrt(dt) * sigma * np.random.randn(1)
        # print(trajectory_pred[start], trajectory_true[start])
    np.save(os.path.join('checkpoint', args.checkpoint, args.checkpoint + '_test_' + str(args.seq_length_test)), trajectory_pred)
    Tcorr = 1000

    trajectory_pred_x = np.load(os.path.join('checkpoint', args.checkpoint, args.checkpoint + '_test_' + str(args.seq_length_test) + '.npy'))
    # print(trajectory_pred_x.shape)
    trajectory_pred_x = trajectory_pred_x[:, :, 0]
    seq_length, batch_size= trajectory_pred_x.shape
    trajectory_pred_x = trajectory_pred_x - np.mean(trajectory_pred_x, axis=0, keepdims=True)

    ACF_pred_x = np.zeros((Tcorr, batch_size))

    for s in range(Tcorr):
        print(s)
        ACF_pred_x[s:s + 1] = np.mean(
            trajectory_pred_x[:seq_length - Tcorr] * trajectory_pred_x[s:seq_length - Tcorr + s], axis=0, keepdims=True)
    np.save(os.path.join('checkpoint', args.checkpoint, args.checkpoint + '_test_' + str(args.seq_length_test) + '_ACF') , ACF_pred_x)

    return

if __name__ == '__main__':
    main()