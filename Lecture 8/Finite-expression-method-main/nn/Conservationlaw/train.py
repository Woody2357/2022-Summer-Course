from __future__ import print_function

import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import math
import numpy as np
from torch import sin, cos, exp
import torch.nn.functional as F
from utils import Logger, AverageMeter, mkdir_p
from numpy.polynomial.legendre import leggauss
# torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser(description='PyTorch Density Function Training')
# Datasets
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--iters', default=30000, type=int, metavar='N', help='number of total iterations to run')
parser.add_argument('--dim', default=2, type=int)
parser.add_argument('--trainbs', default=10, type=int, metavar='N', help='train batchsize')
# parser.add_argument('--bdbs', default=10, type=int, metavar='N', help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--function', default='resnet', type=str, help='function to approximate')
parser.add_argument('--optim', default='adam', type=str, help='function to approximate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--bdbs', default=1000, type=int)
parser.add_argument('--weight', default=100, type=float, help='weight')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
# parser.add_argument('--exp', type=int, default='0', help='use exp last layer')

#Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# region
parser.add_argument('--left', default=-3, type=float, help='left boundary of square region')
parser.add_argument('--right', default=3, type=float, help='right boundary of square region')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

print(state)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def get_boundary(num_pts, dim):
    bd_pts = (torch.rand(num_pts, dim).cuda()) * (args.right - args.left) + args.left
    bd_pts[:, 0] = 0
    return bd_pts

scale = math.pi/4#math.pi

def LHS_pde(u, x, dim_set):

    v = torch.ones(u.shape).cuda()
    ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
    coefficient = torch.ones(1, dim_set).cuda()
    coefficient = -coefficient
    coefficient[:,0] = (dim_set-1)*scale
    ux_mul_coef = ux*coefficient
    LHS = torch.sum(ux_mul_coef, 1, keepdim=True)
    return LHS

def RHS_pde(x):
    bs = x.size(0)
    return torch.zeros(bs, 1).cuda()

def true_solution(x):
    dim = x.size(1)
    coefficient = torch.ones(1, dim).cuda()*scale
    coefficient[:, 0] = 1
    return torch.sin(torch.sum(x*coefficient, dim=1, keepdim=True))


# the input dimension is modified to 2
class ResNet(nn.Module):
    def __init__(self, m):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(args.dim, m)
        self.fc2 = nn.Linear(m, m)

        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)

        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)

        self.outlayer = nn.Linear(m, 1, bias=True)

        # # initialize the network bias
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                m.weight.data.normal_(0.0, 0.1)
        #         if args.use_bias != 0 and m.bias.data.shape[0] == 1:
        #             m.bias.data.fill_(args.use_bias)
        #         # print(m.weight.data.shape, m.bias.data.shape[0], m.bias.data)

    def forward(self, x):

        x1 = (x -args.left)*2/(args.right -args.left) + (-1)
        s = torch.nn.functional.pad(x1, (0, m - args.dim))

        y = self.fc1(x1)
        y = F.relu(y)#F.relu(y)#F.tanh(y)#F.relu(y ** 2) # a RELU(X) + b RELU(X)^2
        y = self.fc2(y)
        y = F.relu(y)#F.relu(y)#F.tanh(y)#F.relu(y ** 2)#
        y = y + s

        s = y
        y = self.fc3(y)
        y = F.relu(y)#F.relu(y)#F.tanh(y)#F.relu(y ** 2)
        y = self.fc4(y)
        y = F.relu(y)#F.relu(y)#F.tanh(y)#F.relu(y ** 2)
        y = y + s

        s = y
        y = self.fc5(y)
        y = F.relu(y)#F.relu(y)#F.tanh(y)#F.relu(y ** 2)
        y = self.fc6(y)
        y = F.relu(y)#F.relu(y)#F.tanh(y)#F.relu(y ** 2)
        y = y + s

        output = self.outlayer(y)
        # output = torch.exp(output)
        return output

'''
HyperParams Setting for Network
'''
m = 50 # number of hidden size

# for ResNet
Ix = torch.zeros([1,m]).cuda()
Ix[0,0] = 1


def main():

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    model = ResNet(m)

    model = model.cuda()
    cudnn.benchmark = True

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    with open(args.checkpoint + "/Config.txt", 'w+') as f:
        for (k, v) in args._get_kwargs():
            f.write(k + ' : ' + str(v) + '\n')

    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    """
    Define Residual Methods and Optimizer
    """
    criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    # Resume
    title = ''

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Losses', 'pdeloss', 'bdloss'])

    # Train and val
    for iter in range(0, args.iters):

        lr = cosine_lr(optimizer, args.lr, iter, args.iters)

        losses, pdeloss, intloss = train(model, criterion, optimizer, use_cuda, iter, lr)
        logger.append([lr, losses, pdeloss, intloss])

    # save model
    save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint=args.checkpoint)

    numerators = []
    denominators = []

    for i in range(1000):
        print(i)
        t = torch.rand(100000, 1).cuda()
        x1 = (torch.rand(100000, args.dim - 1).cuda()) * (args.right - args.left) + args.left
        x = torch.cat((t, x1), 1)
        # print(true_solution(x).size(), model(x).size())
        sq_de = torch.mean((true_solution(x)) ** 2)
        sq_nu = torch.mean((true_solution(x) - model(x)) ** 2)
        numerators.append(sq_nu.item())
        denominators.append(sq_de.item())

    relative_l2 = math.sqrt(sum(numerators)) / math.sqrt(sum(denominators))
    print('relative l2 error: ', relative_l2)
    logger.append(['relative_l2', relative_l2, 0, 0])

    # logger.append([0, 0, relative_l2, 0, 0])

    logger.close()


def train(model, criterion, optimizer, use_cuda, iter, lr):

    # switch to train mode
    model.train()
    end = time.time()
    '''
    points sampling
    '''
    # the range is [0,1] --> [left, right]
    t = torch.rand(args.trainbs, 1).cuda()
    x1 = (torch.rand(args.trainbs, args.dim-1).cuda())*(args.right-args.left)+args.left
    x = torch.cat((t,x1), 1)
    # print(x)
    x.requires_grad = True

    bd_pts = get_boundary(args.bdbs, args.dim)
    bc_true = true_solution(bd_pts)
    bd_nn = model(bd_pts)
    bd_error = torch.nn.functional.mse_loss(bc_true, bd_nn)
    function_error = torch.nn.functional.mse_loss(LHS_pde(model(x), x, args.dim), RHS_pde(x))
    loss = function_error + args.weight * bd_error

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time = time.time() - end
    suffix = '{iter:.1f} {lr:.8f}| Batch: {bt:.3f}s | Loss: {loss:.8f} | pdeloss: {pdeloss:.8f} | bd loss {bd: .8f} |'.format(
        bt=batch_time, loss=loss.item(), iter=iter, lr=lr, pdeloss=function_error.item(), bd=bd_error.item())
    print(suffix)
    return loss.item(), function_error.item(), bd_error.item()

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()
