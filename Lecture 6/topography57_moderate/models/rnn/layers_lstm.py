import torch.nn as nn
import torch

__all__ = ['LSTM']

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = torch.nn.Linear(hidden_size, output_size)
    def forward(self, inputs):
        hidden_output, (h_n, c_n) = self.lstm(inputs)
        final_out = []
        for i in range(hidden_output.size(0)):
            final_out.append(self.linear(hidden_output[i]))
        return torch.stack(final_out, 0)

if __name__ == '__main__':
    lstm = SimpleRNN(2, 50, 1).cuda()
    seq = torch.randn(1000, 10, 2).cuda()
    output_seq = lstm(seq)
    print(output_seq.size())