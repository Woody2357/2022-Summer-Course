"""
@author: Yong Zheng Ong
implements IAE blocks
"""
import numpy as np

import torch
import torch.nn as nn

# implementation of IAE components
class Phi1(nn.Module):
    def __init__(self, modes, width):
        super(Phi1, self).__init__()
        """
        implementing the integral kernels for the encoder
        input shape: (-1, s, w+1)
        output shape: (-1, s, m) kernel matrix

        params:
        modes -> number of fixed size inputs
        width -> channel width of inputs
        """

        # save parameters
        self.modes = modes # m
        self.width = width # w

        self.model = nn.Sequential(
                nn.Linear(self.width + 1, self.modes * 1),
                nn.Dropout(p=0.1),
                nn.LayerNorm(self.modes * 1),
                nn.ReLU(),
                nn.Linear(self.modes * 1, self.modes),
                nn.Dropout(p=0.1),
            )

    def forward(self, x):
        """
        takes as input (a(x),x), outputs kernel values (phi(a(x),x,z_1), ..., phi(a(x),x,z_m))
        """
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, modes, width, num_d=1):
        super(Encoder, self).__init__()
        """
        The encoding block for the network.
        input shape: (-1, w, s)
        output shape: (-1, w, m)
        """

        # save parameters
        self.num_d = num_d # number of dimensions in problem
        self.modes = modes # m
        self.width = width # w

        if self.num_d == 1: # for 1d problems, use a 1-layer IAE
            self.phi = Phi1(self.modes, self.width)
        elif self.num_d == 2: # for 2d problems, use a 2-layer IAE
            self.phi_x = Phi1(self.modes, self.width)
            self.phi_y = Phi1(self.modes, self.width)

        # pointwise MLP to process outputs of nonlinear integral transform
        self.ff_trunk = nn.Sequential(
            nn.Linear(self.width, 2 * self.width),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(2 * self.width, self.width),
            nn.Dropout(p=0.1),
        )
        self.ln = nn.LayerNorm(self.width)
        self.ln1 = nn.LayerNorm(self.width)

    def forward(self, x):
        # get some constants
        batchsize = x.shape[0] # batch size of input
        if self.num_d == 1:
            size_x = x.shape[2] # s size
        elif self.num_d == 2:
            size_x1 = x.shape[2] # s size
            size_x2 = x.shape[3] # s size

        if self.num_d == 1:
            # assume uniform grid
            # since s_z is fixed constant, we do not need to explicitly give it as input to phi1
            x_uni = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).to(x.device).reshape(1, size_x, 1).repeat(batchsize, 1, 1)
            in_phi = torch.cat([x.permute(0, 2, 1), x_uni], 2)

            # get kernel matrix
            kernel = self.phi(in_phi).reshape(batchsize, size_x, self.modes)
            del x_uni, in_phi

            # perform matrix multiplication
            x = x.reshape(batchsize, self.width, size_x) # (-1, w, s)
            x = self.ln1(torch.matmul(x, kernel).permute(0, 2, 1)).permute(0, 2, 1)
            del kernel

            # additional post processing
            x = self.ln(x.permute(0, 2, 1) + self.ff_trunk(x.permute(0, 2, 1))).permute(0, 2, 1).reshape(batchsize, self.width, self.modes)

        if self.num_d == 2:
            # perform 2 layer IAE
            ### layer 1 ###
            # assume uniform grid
            # since s_z is fixed constant, we do not need to explicitly give it as input to phi1
            x_uni = torch.tensor(np.linspace(0, 1, size_x2), dtype=torch.float).to(x.device).reshape(1, 1, size_x2, 1).repeat(batchsize, size_x1, 1, 1)
            in_phi = torch.cat([x.permute(0, 2, 3, 1), x_uni], 3)

            # get kernel matrix
            kernel = self.phi_x(in_phi).reshape(batchsize, size_x1, size_x2, self.modes)
            del x_uni, in_phi

            # perform matrix multiplication
            x = x.reshape(batchsize, self.width, size_x1, size_x2).permute(0, 2, 1, 3) # (-1, s, w, s)
            x = torch.matmul(x, kernel).permute(0, 2, 3, 1) # (-1, w, m, s)
            del kernel

            ### layer 2 ###
            # assume uniform grid
            # since s_z is fixed constant, we do not need to explicitly give it as input to phi1
            x_uni = torch.tensor(np.linspace(0, 1, size_x1), dtype=torch.float).to(x.device).reshape(1, 1, size_x1, 1).repeat(batchsize, self.modes, 1, 1)
            in_phi2 = torch.cat([x.permute(0, 2, 3, 1), x_uni], 3) # (-1, m, s, w+1)

            # get kernel matrix
            kernel = self.phi_y(in_phi2).reshape(batchsize, self.modes, size_x1, self.modes)
            del x_uni, in_phi2

            # perform matrix multiplication
            x = x.permute(0, 2, 1, 3) # (-1, m, w, s)
            x = self.ln1(torch.matmul(x, kernel).permute(0, 3, 1, 2)).permute(0, 3, 1, 2).reshape(batchsize, self.width, self.modes * self.modes)
            del kernel

            # additional post processing
            x = self.ln(x.permute(0, 2, 1) + self.ff_trunk(x.permute(0, 2, 1))).permute(0, 2, 1).reshape(batchsize, self.width, self.modes, self.modes)

        return x

class Phi2(nn.Module):
    def __init__(self, modes, width):
        super(Phi2, self).__init__()
        """
        implementing the integral kernels for the decoder
        input shape: (-1, s, w+1)
        output shape: (-1, s, m) kernel matrix

        params:
        modes -> number of fixed size inputs
        width -> channel width of inputs
        """

        # save parameters
        self.modes = modes # m
        self.width = width # w

        self.model = nn.Sequential(
            nn.Linear(self.width + 1, self.modes * 1),
            nn.Dropout(p=0.1),
            nn.LayerNorm(self.modes * 1),
            nn.ReLU(),
            nn.Linear(self.modes * 1, self.modes),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, modes, width, num_d=1):
        super(Decoder, self).__init__()
        """
        The decoding block for the network.
        input shape: (batchsize, w, m)
        output shape: (batchsize, w, s)
        """

        # save parameters
        self.num_d = num_d # number of dimensions in problem
        self.modes = modes # m
        self.width = width # w

        if self.num_d == 1: # for 1d problems, use a 1-layer IAE
            self.phi = Phi2(self.modes, self.width)

            # prepare auxiliary information for decoder
            self.compressor = nn.Sequential(
                nn.Linear(self.modes, 1),
            )
        elif self.num_d == 2: # for 1d problems, use a 2-layer IAE
            self.phi_x = Phi2(self.modes, self.width)
            self.phi_y = Phi2(self.modes, self.width)

            # prepare auxiliary information for decoder
            self.compressor_x = nn.Sequential(
                nn.Linear(self.modes, 1),
            )
            self.compressor_y = nn.Sequential(
                nn.Linear(self.modes, 1),
            )

        # pointwise MLP to process outputs of nonlinear integral transform
        self.ff_trunk = nn.Sequential(
            nn.Linear(self.width, 2 * self.width),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(2 * self.width, self.width),
            nn.Dropout(p=0.1),
        )
        self.ln = nn.LayerNorm(self.width)
        self.ln1 = nn.LayerNorm(self.width)

    def forward(self, x, size_x, size_x2=None):
        # get some constants
        batchsize = x.shape[0] # batch size of input
        if self.num_d == 2:
            size_x1 = size_x # s size
            size_x2 = size_x2 # s size

        if self.num_d == 1:
            # assume uniform grid
            # since s_z is fixed constant, we do not need to explicitly give it as input to phi1
            x_uni = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).to(x.device).reshape(1, size_x, 1).repeat(batchsize, 1, 1)
            in_phi = torch.cat([self.compressor(x).permute(0, 2, 1).reshape(batchsize, 1, self.width).repeat(1, size_x, 1), x_uni], 2)

            # get kernel matrix
            kernel = self.phi(in_phi).reshape(batchsize, size_x, self.modes).permute(0, 2, 1)
            del x_uni, in_phi

            # perform matrix multiplication
            x = x.reshape(batchsize, self.width, self.modes)
            x = self.ln1(torch.matmul(x, kernel).permute(0, 2, 1)).permute(0, 2, 1)
            del kernel

            # additional post processing
            x = self.ln(x.permute(0, 2, 1) + self.ff_trunk(x.permute(0, 2, 1))).permute(0, 2, 1).reshape(batchsize, self.width, size_x)

        if self.num_d == 2:
            # perform 2 layer IAE
            ### layer 1 ###
            # assume uniform grid
            # since s_z is fixed constant, we do not need to explicitly give it as input to phi1
            x_uni = torch.tensor(np.linspace(0, 1, size_x1), dtype=torch.float).to(x.device).reshape(1, size_x1, 1, 1).repeat(batchsize, 1, self.modes, 1)
            in_phi2 = torch.cat([self.compressor_x(x).permute(0, 2, 3, 1).reshape(batchsize, 1, self.modes, self.width).repeat(1, size_x1, 1, 1), x_uni], 3)

            # get kernel matrix
            kernel = self.phi_x(in_phi2).reshape(batchsize, size_x1, self.modes, self.modes).permute(0, 2, 3, 1)
            del x_uni, in_phi2

            # perform matrix multiplication
            x = x.reshape(batchsize, self.width, self.modes, self.modes).permute(0, 2, 1, 3)
            x = torch.matmul(x, kernel).permute(0, 2, 3, 1) # (-1, w, s, m)
            del kernel

            ### layer 2 ###
            # assume uniform grid
            # since s_z is fixed constant, we do not need to explicitly give it as input to phi1
            x_uni = torch.tensor(np.linspace(0, 1, size_x2), dtype=torch.float).to(x.device).reshape(1, size_x2, 1, 1).repeat(batchsize, 1, size_x1, 1)
            in_phi = torch.cat([self.compressor_y(x).permute(0, 2, 3, 1).reshape(batchsize, 1, size_x1, self.width).repeat(1, size_x2, 1, 1), x_uni], 3)

            # get kernel matrix
            kernel = self.phi_y(in_phi).reshape(batchsize, size_x2, size_x1, self.modes).permute(0, 2, 3, 1)
            del x_uni, in_phi

            # perform matrix multiplication
            x = x.permute(0, 2, 1, 3)
            x = self.ln1(torch.matmul(x, kernel).permute(0, 1, 3, 2)).permute(0, 3, 1, 2).reshape(batchsize, self.width, size_x1 * size_x2)
            del kernel

            # additional post processing
            x = self.ln(x.permute(0, 2, 1) + self.ff_trunk(x.permute(0, 2, 1))).permute(0, 2, 1).reshape(batchsize, self.width, size_x1, size_x2)

        return x

# implements a single IAE block
class TokenConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, num_d=1):
        super(TokenConv, self).__init__()
        """
        Implements the normal branch
        input shape: (-1, in_channels, s)
        output shape: (-1, out_channels, s)
        """

        # save parameters
        self.in_channels = in_channels # number of input channels
        self.out_channels = out_channels # number of output channels
        self.modes = modes # m
        self.num_d = num_d # number of dimensions of problem

        if self.num_d == 1:
            # for 1d case
            self.phi_0 = nn.Sequential(
                nn.Linear(self.modes, 2*self.modes),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(2*self.modes, self.modes),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(self.modes, self.modes),
            )
        if self.num_d == 2:
            # for 2d case
            self.phi_0 = nn.Sequential(
                nn.Linear(self.modes*self.modes, self.modes*self.modes),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(self.modes*self.modes, self.modes*self.modes),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(self.modes*self.modes, self.modes*self.modes),
            )

        self.encoder = Encoder(self.modes, self.in_channels, self.num_d)
        self.decoder = Decoder(self.modes, self.out_channels, self.num_d)

    def forward(self, x):
        batchsize = x.shape[0]
        target_size = x.shape[2]

        # pass through encoder
        # x = self.encoder(x) # (-1, in_channels, m)
        x = self.encoder(x) # (-1, in_channels, m)

        # pass through phi_0
        if self.num_d == 1:
            x = self.phi_0(x).reshape(batchsize, self.out_channels, self.modes)
        if self.num_d == 2:
            x = self.phi_0(x.reshape(batchsize, self.in_channels, self.modes*self.modes)).reshape(batchsize, self.out_channels, self.modes, self.modes)

        # pass through decoder
        x = self.decoder(x, target_size, target_size)

        return x

class TokenFourier(nn.Module):
    def __init__(self, in_channels, out_channels, modes, num_d=1):
        super(TokenFourier, self).__init__()
        """
        Implements the fourier branch
        input shape: (-1, in_channels, s)
        output shape: (-1, out_channels, s)
        """

        # save parameters
        self.in_channels = in_channels # number of input channels
        self.out_channels = out_channels # number of output channels
        self.modes = modes # m
        self.num_d = num_d # number of dimensions of problem

        if self.num_d == 1:
            # for 1d case
            self.phi_0 = nn.Sequential(
                nn.Linear(self.modes, 2*self.modes),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(2*self.modes, self.modes),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(self.modes, self.modes),
            )
        if self.num_d == 2:
            # for 2d case
            self.phi_0 = nn.Sequential(
                nn.Linear(self.modes*self.modes, self.modes*self.modes),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(self.modes*self.modes, self.modes*self.modes),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(self.modes*self.modes, self.modes*self.modes),
            )

        self.encoder = Encoder(self.modes, 2*self.in_channels, self.num_d)
        self.decoder = Decoder(self.modes, 2*self.out_channels, self.num_d)

    def forward(self, x):
        batchsize = x.shape[0]
        target_size = x.shape[2]

        # get fourier transformed inputs
        x = torch.rfft(x, self.num_d, normalized=True, onesided=True)

        # handle reshaping
        if self.num_d == 1:
            x = x.permute(0, 1, 3, 2).reshape(batchsize, self.in_channels*2, target_size//2+1)
        elif self.num_d == 2:
            x = x.permute(0, 1, 4, 2, 3).reshape(batchsize, self.in_channels*2, target_size, target_size//2+1)

        # pass through encoder
        x = self.encoder(x) # (-1, 2*in_channels, m)

        # pass through phi_0
        if self.num_d == 1:
            x = self.phi_0(x).reshape(batchsize, 2*self.out_channels, self.modes)
        if self.num_d == 2:
            x = self.phi_0(x.reshape(batchsize, 2*self.in_channels, self.modes*self.modes)).reshape(batchsize, 2*self.out_channels, self.modes, self.modes)

        # pass through decoder
        if self.num_d == 1:
            x = self.decoder(x, target_size//2+1)
        elif self.num_d == 2:
            x = self.decoder(x, target_size, target_size//2+1)

        # handle reshaping
        if self.num_d == 1:
            x = x.reshape(batchsize, self.out_channels, 2, target_size//2+1).permute(0, 1, 3, 2)
            signal_sizes = (target_size,)
        elif self.num_d == 2:
            x = x.reshape(batchsize, self.out_channels, 2, target_size, target_size//2+1).permute(0, 1, 3, 4, 2)
            signal_sizes = (target_size, target_size)

        # get inverse fourier transformed outputs
        x = torch.irfft(x, self.num_d, normalized=True, onesided=True, signal_sizes=signal_sizes)

        return x

class TokenDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes, width, num_d=1, activation=None, layernorm=True, block_number=1):
        super(TokenDenseBlock, self).__init__()
        """
        Implements a single IAE block. Includes 2 parallel IAE and the merging MLP
        """

        # save parameters
        self.in_channels = in_channels # number of input channels
        self.out_channels = out_channels # number of output channels
        self.modes = modes # m
        self.width = width # w
        self.num_d = num_d # number of dimensions of problem
        self.activation = activation # activation function
        self.layernorm = layernorm # use layernorm or not
        self.block_number = block_number # block number in sequence for merging

        # merge dense blocks
        self.reducer = nn.Conv1d(self.in_channels*self.block_number, self.in_channels, 1)

        # standard path
        self.sconv = TokenConv(self.in_channels, self.out_channels, self.modes, num_d=self.num_d)
        # fourier path
        self.fconv = TokenFourier(self.in_channels, self.out_channels, self.modes, num_d=self.num_d)

        # skip connection
        self.w = nn.Conv1d(self.in_channels, self.out_channels, 1)

        # channel MLP
        self.in_ln = nn.LayerNorm(3*self.in_channels)
        self.ff_ln = nn.LayerNorm(self.out_channels)
        self.reducer2 = nn.Linear(3*self.out_channels, self.out_channels)
        self.ff = nn.Sequential(
            nn.Linear(3*self.out_channels, 2*self.out_channels),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(2*self.out_channels, self.out_channels),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        batchsize = x[0].shape[0]
        size_x = x[0].shape[2]

        # in the dense version, input is a tuple of length = block_number, each input of size (-1, w, s)
        # concatenate the inputs along width
        input = torch.cat(x, dim=1) # (-1, w*block_number, s)

        # reduce the dimension
        input = self.reducer(input.reshape(batchsize, self.width*self.block_number, -1))
        if self.num_d == 1:
            input = input.reshape(batchsize, self.width, size_x)
        elif self.num_d == 2:
            input = input.reshape(batchsize, self.width, size_x, size_x)

        # perform IAE
        # standard path output
        x1 = self.sconv(input)
        # fourier path output
        x2 = self.fconv(input)

        # split connection output
        x3 = self.w(input.reshape(batchsize, self.width, -1))
        if self.num_d == 1:
            x3 = x3.reshape(batchsize, self.width, size_x)
        elif self.num_d == 2:
            x3 = x3.reshape(batchsize, self.width, size_x, size_x)

        # merge outputs
        input = torch.cat([x3,x1,x2], dim=1)
        del x1, x2, x3

        # normalize
        if self.num_d == 1:
            if self.layernorm:
                input = self.in_ln(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.num_d == 2:
            if self.layernorm:
                input = self.in_ln(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # channel mlp
        if self.num_d == 1:
            input = self.reducer2(input.permute(0, 2, 1)) + self.ff(input.permute(0, 2, 1))
            if self.layernorm:
                input = self.ff_ln(input)
            input = input.permute(0, 2, 1)
        elif self.num_d == 2:
            input = self.reducer2(input.permute(0, 2, 3, 1)) + self.ff(input.permute(0, 2, 3, 1))
            if self.layernorm:
                input = self.ff_ln(input)
            input = input.permute(0, 3, 1, 2)

        # apply activation
        if self.activation is not None:
            input = self.activation(input)

        # append value to next dense block
        x.append(input)
        del input

        return x
