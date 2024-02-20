import torch
import torch.nn as nn
import numpy as np
from math import floor

class FullyConnectedNet(nn.Module):
    def __init__(self, layer_sizes, batch_size):
        nn.Module.__init__(self)
        
        def init_weights(m):
            # print(m)
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                # print(m.weight)

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+2 < len(layer_sizes):
                self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(0))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())


        self.MLP.apply(init_weights)

    def forward(self, x):
        x = self.MLP(x)
        return x

class SelfCollNet(nn.Module):
    def __init__(self, fc_layer_sizes, batch_size, device):
        
        nn.Module.__init__(self)


        self.fc = FullyConnectedNet(fc_layer_sizes,batch_size)

    def forward(self, x_q):
        """
        x_q: input normailzed_q(7) or nerf_q(21)
        """
        y = self.fc(x_q)
        
        return y
        

