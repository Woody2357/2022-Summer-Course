import torch.nn as nn
import torch

__all__ = ['SimpleRNN']

class small_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """"Constructor of the class"""
        super(small_cell, self).__init__()
        self.seq = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.ReLU(inplace=True),
                      nn.Linear(hidden_size, 10 * hidden_size),
                      nn.ReLU(inplace=True),
                      nn.Linear(10 * hidden_size, 4 * hidden_size))
    def forward(self,x):
        return self.seq(x)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout = 0.0):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            if i==0:
                ih.append(small_cell(input_size, hidden_size))
                hh.append(small_cell(hidden_size, hidden_size))
            else:
                ih.append(nn.Linear(hidden_size, hidden_size))
                hh.append(nn.Linear(hidden_size, hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate)
            ncx = (f_gate * cx) + (i_gate * c_gate)
            nhx = o_gate * torch.tanh(ncx)
            cy.append(ncx)
            hy.append(nhx)
            input = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)  # number of layer * batch * hidden
        return hy, cy

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstmcell = LSTMCell(input_size, hidden_size, 1, dropout = 0.0)
        self.linear = nn.Sequential(nn.Linear(hidden_size, hidden_size//2),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(hidden_size//2, output_size))
    def forward(self, inputs):
        hidden = []
        for i in range(inputs.size(0)):
            if i == 0:
                ht = torch.zeros(1, inputs.size(1), self.hidden_size).cuda()  # 1 mean number of layers
                ct = torch.zeros(1, inputs.size(1), self.hidden_size).cuda()
                ht, ct = self.lstmcell(inputs[i], (ht, ct))
                hidden.append(self.linear(ht[-1]))
            else:
                ht, ct = self.lstmcell(inputs[i], (ht, ct))
                hidden.append(self.linear(ht[-1]))
        return torch.stack(hidden, 0)

if __name__ == '__main__':
    lstm = SimpleRNN(2, 50, 1).cuda()
    seq = torch.randn(1000, 10, 2).cuda()
    output_seq = lstm(seq)
    print(output_seq.size())