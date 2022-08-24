"""
@author: Yong Zheng Ong
implements IAENet
"""
import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers.iae_blocks import TokenDenseBlock
from ...layers.fno_blocks import FNOBlock

class Model(nn.Module):
    def __init__(self, modes, width, size, input_channel=2, output_channel=1, num_blocks=4, num_d=1, **_):
        super(Model, self).__init__()
        """
        The overall network. It contains 4 IAE Blocks.
        """

        # set some constants
        self.modes = modes
        self.width = width
        self.size = size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.num_blocks = num_blocks
        self.num_d = num_d # 1 for 1D problems, 2 for 2D problems

        # number of modes for fno - 1D uses 16, 2D uses 12
        if self.num_d == 1:
            self.fno_modes = 16
        elif self.num_d == 2:
            self.fno_modes = 12

        # for pre and post processing
        self.fc0 = nn.Linear(self.input_channel, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_channel)
        self.out = nn.Sequential(
            FNOBlock(self.width, self.width, self.fno_modes, num_d=self.num_d, activation=F.relu),
            FNOBlock(self.width, self.width, self.fno_modes, num_d=self.num_d)
        )

        self.blocks = nn.ModuleList([])
        # build the blocks
        for i in range(self.num_blocks):
            self.blocks.append(TokenDenseBlock(self.width, self.width, self.modes, self.width, num_d=self.num_d, activation=F.relu, layernorm=True, block_number=i+1))

        # reducer
        self.reducer = nn.Conv1d(self.width*(self.num_blocks+1), self.width, 1)

    def forward(self, x):
        # get constants
        batchsize = x.shape[0]
        size_x = x.shape[1]

        # perform pre processing
        x = self.fc0(x)

        if self.num_d == 1:
            x = x.permute(0, 2, 1)
        elif self.num_d == 2:
            x = x.permute(0, 3, 1, 2) # raise dim if 2d

        # build list for skip connections
        x = [x]

        # run through IAE blocks
        for block in self.blocks:
            x = block(x)

        # merge all values
        x = torch.cat(x, dim=1)
        x = self.reducer(x.reshape(batchsize, self.width*(self.num_blocks+1), -1))
        if self.num_d == 1:
            x = x.reshape(batchsize, self.width, size_x)
        elif self.num_d == 2:
            x = x.reshape(batchsize, self.width, size_x, size_x)

        # perform post processing
        x = self.out(x)

        if self.num_d == 1:
            x = x.permute(0, 2, 1)
        elif self.num_d == 2:
            x = x.permute(0, 2, 3, 1) # raise dim if 2d

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
