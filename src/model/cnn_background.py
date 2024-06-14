# Code from https://github.com/MadryLab/backgrounds_challenge/blob/master/imagenet_models/resnet.py
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50Background(nn.Module):
    def __init__(self):
        super(ResNet50Background, self).__init__()
        self.model = models.resnet50() #weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 9)

    def forward(self, images, return_h=False, return_logits=True, neuron_mask=None, detach_h=False, return_pooled_h=False):
        cnn = nn.Sequential(*list(self.model.children())[:-2])
        avgpool = self.model.avgpool
        fc = self.model.fc
        h = cnn(images)
        if neuron_mask is not None:
            h = h * neuron_mask
        pooled_h = avgpool(h)
        logits = torch.flatten(pooled_h, 1)
        if detach_h:
            logits = logits.detach()
        logits = fc(logits)
        
        if return_h and return_logits: 
            return logits, h 
        elif return_pooled_h and return_logits:
            return pooled_h, logits 
        elif return_pooled_h:
            return pooled_h
        elif return_logits: 
            return logits 
        elif return_h:
            return h

        return

