import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim 

class MLP(nn.Module):
  # adam: lr: 0.001, weight_decay: 0.0001
  # batch size: 256
  def __init__(self, in_dim, out_dim, dims=(50,30)):
    super(MLP, self).__init__()
    if type(in_dim) != int:
      in_dim = in_dim[0] * in_dim[1]
    self.in_dim = in_dim 
    self.out_dim = out_dim
    
    layers = [nn.Linear(in_dim, dims[0]), nn.ReLU()]
    for i in range(len(dims)-1):
      layers.append(nn.Linear(dims[i], dims[i+1]))
      layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-1], out_dim))
    self.layers = nn.Sequential(*layers)
    self.apply(self._init_weights)

  def _init_weights(self, module):
      np.random.seed(0)
      self.layers[0].weight.data = torch.FloatTensor(np.random.randn(self.in_dim,50)*0.1).T
      self.layers[0].bias.data = torch.FloatTensor(np.random.randn(50)*0.1)
      self.layers[2].weight.data = torch.FloatTensor(np.random.randn(50,30)*0.1).T
      self.layers[2].bias.data = torch.FloatTensor(np.random.randn(30)*0.1)
      self.layers[4].weight.data = torch.FloatTensor(np.random.randn(30,self.out_dim)*0.1).T
      self.layers[4].bias.data = torch.FloatTensor(np.random.randn(self.out_dim)*0.1)
      # if isinstance(module, nn.Linear):
      #     module.weight.data.normal_(mean=0.0, std=0.1)
      #     if module.bias is not None:
      #         module.bias.data.normal_(mean=0.0, std=0.1)

  def forward(self, x):
    if len(x.shape) > 2:
      batch_size = len(x)
      x = x.reshape(batch_size, -1)

    out = self.layers[0](x)
    out = self.layers[1](out)
    return self.layers(x)


if __name__=="__main__":
  pass