import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim 

class CNNToy(nn.Module):
    def __init__(self):
        super(CNNToy, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,1,3,padding=1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(1,1,3,padding=1),
        )
        self.fc = nn.Linear(9,2)
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        # self.layers[0].weight.data = torch.FloatTensor(np.random.randn(self.in_dim,50)*0.1).T
        self.layers[0].weight.data = None
        self.layers[0].bias.data = None
        self.layers[2].weight.data = None
        self.layers[2].bias.data = None

    def get_activation(self, x):
        return self.layers(x)
    
    def forward(self, x):
        out = self.layers(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out 
