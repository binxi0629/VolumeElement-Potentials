import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubNet(nn.Module):
    def __init__(self, d_representation, diff=64, dropout=0.1):
        super().__init__()
        self.FC1 = nn.Linear(d_representation, diff)
        self.FC2 = nn.Linear(diff, 16)
        self.FC3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.FC1(x)
        x = self.nn.LeakyRelu(x)
        x = self.dropout(x)
        x = self.FC2(x)
        x = self.nn.LeakyRelu(x)
        x = self.FC3(x)
        return x


def get_clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Network(nn.Module):
    def __init__(self, d_representation, diff, dropout,N):
        super(Network, self).__init__()
        self.num_subnets = N
        self.Layer = get_clone(SubNet(d_representation, diff, dropout), self.num_subnets)

    def forward(self, area_elements: list):
        y = torch.tensor([self.Layer[i](area_elements[i]) for i in range(self.num_subnets)])
        return torch.sum(y)