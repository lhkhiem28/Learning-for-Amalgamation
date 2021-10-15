import torch
import torch.nn as nn

class Sigmoid(nn.Module):
    def forward(self, input):
        return torch.sigmoid(input)

class Swish(nn.Module):
    def forward(self, input):
        return input*torch.sigmoid(input)
