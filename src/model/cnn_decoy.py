import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim 

class CNNDecoyMNIST(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(CNNDecoyMNIST, self).__init__()
    if type(in_dim) == int:
      in_dim = int(in_dim**0.5)
      in_dim = (in_dim,in_dim)
    self.in_dim = in_dim 
    self.out_dim = out_dim

    self.conv = nn.Sequential(
        nn.Conv2d(1,32,3,1,1),
        nn.ReLU(),
        nn.Conv2d(32,64,3,1,1),
        nn.ReLU()
    )
    self.maxpool = nn.MaxPool2d(2)
    self.fc = nn.Sequential(
        nn.Linear(64*((in_dim[0])//2)**2, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Conv2d):
      module.weight.data.normal_(mean=0., std=1.)
      if module.bias is not None:
        module.bias.data.zero_()
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0., std=1.)
      if module.bias is not None:
        module.bias.data.zero_()

  def forward(self, x, return_h=False):
    if x.shape[1] != self.in_dim[0]:
      x = x.reshape(-1, 1, self.in_dim[0], self.in_dim[1])
    h = self.conv(x)
    out = self.maxpool(h)
    out = torch.flatten(out,1)
    out = self.fc(out)
    if return_h:
      return out, h
    return out


if __name__=="__main__":
  pass