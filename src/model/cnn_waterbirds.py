# source: https://github.com/akrishna77/bias-discovery/blob/0b2c7bc3ba710008ac6976b794d74d9df138b229/models/celeba_classifier.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1
    
def grad_reverse(x):
    return GradReverse.apply(x)

class ResNet50Waterbirds(nn.Module):
    def __init__(self):
        """
        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        """
        super(ResNet50Waterbirds, self).__init__()

        self.model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 2)

    def forward(self, images, return_h=False, return_logits=True, neuron_mask=None, detach_h=False, return_pooled_h=False):
        """
        Take a batch of images and run them through the model to
        produce a score for each class.
        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width
        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        """
        cnn = nn.Sequential(*list(self.model_ft.children())[:-2])
        avgpool = self.model_ft.avgpool
        fc = self.model_ft.fc

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
        elif return_logits: 
            return logits 
        elif return_h:
            return h

        return
        # scores = None
        # scores = self.model_ft(images)