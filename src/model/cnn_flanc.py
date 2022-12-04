
import torchvision.models as models
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model import common
import torch
import torch.nn.functional as F
import copy

def make_model(args, parent=False):
    
    return CNN_FLANC(args)


class conv_basis(nn.Module):
    def __init__(self, filter_bank, in_channels, basis_size, n_basis, kernel_size, stride=1, bias=True):
        super(conv_basis, self).__init__()
        self.in_channels = in_channels
        self.n_basis = n_basis
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.stride = stride
        self.group = in_channels // basis_size
        self.weight = filter_bank
        self.bias = nn.Parameter(torch.zeros(n_basis)) if bias else None
        #print(stride)
    def forward(self, x):
        if self.group == 1:
            x = F.conv2d(input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2)
        else:
            #print(self.weight.shape)
            x = torch.cat([F.conv2d(input=xi, weight=self.weight, bias=self.bias, stride=self.stride,
                                    padding=self.kernel_size//2)
                           for xi in torch.split(x, self.basis_size, dim=1)], dim=1)
        return x

    def __repr__(self):
        s = 'Conv_basis(in_channels={}, basis_size={}, group={}, n_basis={}, kernel_size={}, out_channel={})'.format(
            self.in_channels, self.basis_size, self.group, self.n_basis, self.kernel_size, self.group * self.n_basis)
        return s


class DecomBlock(nn.Module):
    def __init__(self, filter_bank, in_channels, out_channels, n_basis, basis_size, kernel_size,
                 stride=1, bias=False, conv=common.default_conv, norm=common.default_norm, act=common.default_act):
        super(DecomBlock, self).__init__()
        group = in_channels // basis_size
        modules = [conv_basis(filter_bank,in_channels, basis_size, n_basis, kernel_size, stride, bias)]
        modules.append(conv(group * n_basis, out_channels, kernel_size=1, stride=1, bias=bias))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


class CNN_FLANC(nn.Module):

    """ Simple network"""

    def __init__(self,args):
        super().__init__()
        basis_fract = args.basis_fraction
        net_fract= args.net_fraction
        n_basis = args.n_basis
        self.head = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        m1 = round(128*n_basis)
        n1 = round(64*basis_fract)
        self.filter_bank_1 = nn.Parameter(torch.empty(m1, n1, 3, 3))
        
        m2 = round(128*n_basis)
        n2 = round(128*basis_fract)
        self.filter_bank_2 = nn.Parameter(torch.empty(m2, n2, 3, 3))
        
        X1 = torch.empty(m1, n1, 3, 3)
        torch.nn.init.orthogonal(X1)
        self.filter_bank_1.data = copy.deepcopy(X1)
        X2 = torch.empty(m2, n2, 3, 3)
        torch.nn.init.orthogonal(X2)
        self.filter_bank_2.data = copy.deepcopy(X2)
        out_1 = round(128*net_fract)
        self.conv1 = DecomBlock(self.filter_bank_1, 64, out_1, m1, n1, kernel_size=3, bias=False) # 28
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 14
        out_2 = round(128*net_fract)
        self.conv2 = DecomBlock(self.filter_bank_2, out_1, out_2, m2, n2, kernel_size=3, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 7
        
        self.classifier = nn.Linear(out_2 * 7 * 7, 10)
        

    def forward(self, x):
        #print(x.shape)
        x = self.head(x)
        x = self.conv1(x)
        x = self.pool1(self.relu1(x))

        x = self.conv2(x)
        x = self.pool2(self.relu2(x))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def loss_type(loss_para_type):
    if loss_para_type == 'L1':
        loss_fun = nn.L1Loss()
    elif loss_para_type == 'L2':
        loss_fun = nn.MSELoss()
    else:
        raise NotImplementedError
    return loss_fun

def orth_loss(model, args, para_loss_type='L2'):

    loss_fun = loss_type(para_loss_type)

    loss = 0
    for l_id in range(1,3): 
        filter_bank = getattr(model,"filter_bank_"+str(l_id))
        
        #filter_bank_2 = getattr(block,"filter_bank_2")
        all_bank = filter_bank
        num_all_bank = filter_bank.shape[0]
        B = all_bank.view(num_all_bank, -1)
        D = torch.mm(B,torch.t(B))
        D = loss_fun(D, torch.eye(num_all_bank, num_all_bank).cuda())
        loss = loss + D
    return loss
