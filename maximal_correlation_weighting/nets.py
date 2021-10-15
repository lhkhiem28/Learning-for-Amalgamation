"""
LeNet architectures for different datasets
"""

import torch.nn as nn
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self, dataset):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        if dataset == "cifar":
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
        elif dataset == "dogs":
            self.pool = nn.MaxPool2d(4, 4)
            self.fc1 = nn.Linear(16 * 7 * 7, 120)
        elif dataset == "tiny_imagenet":
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 13 * 13, 120)
        else:
             raise Exception('Invalid dataset type')
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc2 = nn.Linear(120, 84)
        self.dataset = dataset

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.dataset == "cifar":
            x = x.view(-1, 16 * 5 * 5)
        elif self.dataset == "dogs":
            x = x.view(-1, 16 * 7 * 7)
        elif self.dataset == "tiny_imagenet":
            x = x.view(-1, 16 * 13 * 13)
        else:
             raise Exception('Invalid dataset type')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Net2(nn.Module):
    def __init__(self,in_size,num_classes):
        super(Net2, self).__init__()
        self.fc3 = nn.Linear(in_size, num_classes)

    def forward(self, x):
        x = self.fc3(x)
        return x

def generate_net(dataset):
    if dataset == "cifar":
        num_classes = 2
    elif dataset == "dogs":
        num_classes = 5
    elif dataset == "tiny_imagenet":
        num_classes = 5
    else:
         raise Exception('Invalid dataset type')
    return nn.Sequential(Net1(dataset),Net2(84,num_classes))


