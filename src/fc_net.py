import torch
from torch import nn

class FC_net(nn.Module):
    """ Simple fully connected net with not activation after the last layer. """
    def __init__(self, in_dims, out_dims, nr_hidden=1, hidden_dims=256, activation=nn.ReLU()):
        super.__init__()

        if nr_hidden < 1:
            print("Must have at leas one hidden layer")
            return None

        layers = [nn.Sequential(nn.Linear(in_dims, hidden_dims), activation)]

        for i in range(nr_hidden - 1):
            new_layer = nn.Sequential(nn.Linear(hidden_dims, hidden_dims), activation)
            layers.append(new_layer)
 
        layers.append(nn.Linear(hidden_dims, out_dims))
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        return self.layers(x)
