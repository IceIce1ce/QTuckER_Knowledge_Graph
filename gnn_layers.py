import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

#QGNN encoder
def make_quaternion_mul(kernel):
    dim = kernel.size(1) // 4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0) 
    i2 = torch.cat([i, r, -k, j], dim=0)  
    j2 = torch.cat([j, k, r, -i], dim=0)  
    k2 = torch.cat([k, -j, i, r], dim=0) 
    hamilton = torch.cat([r2, i2, j2, k2], dim=1) 
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

class QGNN_layer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(QGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0/(self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        support = torch.mm(input, hamilton)  
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return self.act(output)

#OGNN encoder
def make_octonion_mul(kernel):  
    dim = kernel.size(1)//8
    e0, e1, e2, e3, e4, e5, e6, e7 = torch.split(kernel, [dim, dim, dim, dim, dim, dim, dim, dim], dim=1)
    e_0 = torch.cat([e0, -e1, -e2, -e3, -e4, -e5, -e6, -e7], dim=0)  
    e_1 = torch.cat([e1, e0, -e3, e2, -e5, e4, e7, -e6], dim=0)  
    e_2 = torch.cat([e2, e3, e0, -e1, -e6, -e7, e4, e5], dim=0)  
    e_3 = torch.cat([e3, -e2, e1, e0, -e7, e6, -e5, e4], dim=0)  
    e_4 = torch.cat([e4, e5, e6, e7, e0, -e1, -e2, -e3], dim=0) 
    e_5 = torch.cat([e5, -e4, e7, -e6, e1, e0, e3, -e2], dim=0) 
    e_6 = torch.cat([e6, -e7, -e4, e5, e2, -e3, e0, e1], dim=0) 
    e_7 = torch.cat([e7, e6, -e5, -e4, e3, e2, -e1, e0], dim=0) 
    halminton = torch.cat([e_0, e_1, e_2, e_3, e_4, e_5, e_6, e_7], dim=1)
    assert kernel.size(1) == halminton.size(1)
    return halminton

class OGNN_layer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(OGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_features//8, self.out_features))
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0/(self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_octonion_mul(self.weight)
        support = torch.mm(input, hamilton)  
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return self.act(output)