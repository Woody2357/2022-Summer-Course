from libs import *

class Net(nn.Module):
    # NL: the number of hidden layers
    # NN: the number of vertices in each layer
    def __init__(self, NL, NN):
        super(Net, self).__init__()

        self.input_layer = nn.Linear(3, NN)

        self.hidden_layers = nn.ModuleList([nn.Linear(NN, NN) for i in range(NL)])

        self.output_layer = nn.Linear(NN, 1)
        
    def forward(self, x):
        o = self.act(self.input_layer(x))

        for i, li in enumerate(self.hidden_layers):
            o = self.act(li(o))
        
        out = self.output_layer(o)
        
        return out

    def act(self, x):
        return x * torch.sigmoid(x)