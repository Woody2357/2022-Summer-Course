"""
contains the layers used in fno

ref: https://github.com/zongyi-li/fourier_neural_operator
"""
from functools import partial

import torch
import torch.nn as nn

# Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

# Fourier layers
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, num_d=1):
        super(SpectralConv, self).__init__()
        """
        Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.num_d = num_d

        self.scale = (1 / (in_channels*out_channels))

        if self.num_d == 1:
            self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, 2))
        elif self.num_d == 2:
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, 2))
            self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        x_size = x.size(-1)
        x_fourier = x.size(-1)//2 + 1

        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.rfft(x, self.num_d, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        if self.num_d == 1:
            out_ft = torch.zeros(batchsize, self.out_channels, x_fourier, 2, device=x.device)
            out_ft[:, :, :self.modes] = compl_mul1d(x[:, :, :self.modes], self.weights)
            del x

            signal_sizes = (x_size,)

        elif self.num_d == 2:
            out_ft = torch.zeros(batchsize, self.out_channels,  x_size, x_fourier, 2, device=x.device)
            out_ft[:, :, :self.modes, :self.modes] = \
                compl_mul2d(x[:, :, :self.modes, :self.modes], self.weights1)
            out_ft[:, :, -self.modes:, :self.modes] = \
                compl_mul2d(x[:, :, -self.modes:, :self.modes], self.weights2)
            del x

            signal_sizes = (x_size, x_size)

        #Return to physical space
        x = torch.irfft(out_ft, self.num_d, normalized=True, onesided=True, signal_sizes=signal_sizes)
        del out_ft

        return x

class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes, num_d=1, activation=None):
        super(FNOBlock, self).__init__()
        """
        Implements a single FNO block. Includes 1 layer of SpectralConv1d and 1 layer of skip connection
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.num_d = num_d
        self.activation = activation

        self.conv = SpectralConv(self.in_channels, self.out_channels, self.modes, num_d=self.num_d)
        self.w = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, x):
        # (-1, w, s)
        batchsize = x.shape[0]
        size_x = x.shape[2]

        x1 = self.conv(x)
        x2 = self.w(x.reshape(batchsize, self.in_channels, -1))
        if self.num_d == 1:
            x2 = x2.reshape(batchsize, self.out_channels, size_x)
        elif self.num_d == 2:
            x2 = x2.reshape(batchsize, self.out_channels, size_x, size_x)
        x = x1 + x2
        del x1, x2

        if self.activation is not None:
            return self.activation(x)

        return x
