"""
Author: Yawei Li
Date: 21/08/2019
Try to decompose resnet18 for ImageNet classification
"""

import torchvision.models as models
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model import common
import torch
import torch.nn.functional as F
import copy
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def make_model(args, parent=False):
    return ResNet18_FLANC(args)

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
        #self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(n_basis, basis_size, kernel_size, kernel_size)))
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
        #if norm is not None: modules.append(norm(group * n_basis))
        modules.append(conv(group * n_basis, out_channels, kernel_size=1, stride=1, bias=bias))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, n_basis, basis_size, block_expansion=0, stride=1, downsample=False, conv3x3=common.default_conv, conv1x1=common.default_conv):
        super(BasicBlock, self).__init__()
        self.filter_bank_1 = nn.Parameter(torch.empty(n_basis, basis_size, 3, 3))
        self.filter_bank_2 = nn.Parameter(torch.empty(n_basis, basis_size, 3, 3))
        
        X = torch.empty(n_basis*2, basis_size, 3, 3)
        torch.nn.init.orthogonal(X)
        self.filter_bank_1.data = copy.deepcopy(X[:n_basis,:])
        self.filter_bank_2.data = copy.deepcopy(X[n_basis:,:])

        self.conv1 = DecomBlock(self.filter_bank_1, inplanes, planes, n_basis, basis_size, kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = DecomBlock(self.filter_bank_2, planes, planes, n_basis, basis_size, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * block_expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_expansion),
            )
        else:
            self.downsample = None
        self.stride = stride
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print("1",out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        #print("2",out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DefaultBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,  n_basis, basis_size, block_expansion=0, stride=1, downsample=False, conv3x3=common.default_conv, conv1x1=common.default_conv):
        super(DefaultBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * block_expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_expansion),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, n_basis=1, basis_fract=1, net_fract=1, num_classes=10, conv3x3=common.default_conv, conv1x1=common.default_conv):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.basis_fract = basis_fract
        self.net_fract = net_fract
        self.n_basis = n_basis 
        self.conv3x3 = conv3x3
        self.conv1x1 = conv1x1
        self.conv1 = conv3x3(3, 64, kernel_size=3, stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        m = round(64*self.n_basis) #for n_basis
        n = round(64*self.basis_fract) #for basis_size
        cfg = [(m,n),(m,n)]
        self.layer1 = self._make_layer(block, 64, layers[0], cfg, stride=1) #64,64
        
        m = round(128*self.n_basis)
        n = round(64*self.basis_fract)
        n2 = round(128*self.basis_fract)
        cfg = [(m,n),(m,n2)]
        self.layer2 = self._make_layer(block, 128, layers[1], cfg, stride=2)
        
        m = round(256*self.n_basis)
        n = round(128*self.basis_fract)
        n2 = round(256*self.basis_fract)
        cfg = [(m,n),(m,n2)]
        self.layer3 = self._make_layer(block, 256, layers[2], cfg, stride=2)

        m = round(512*self.n_basis)
        n = round(256*self.basis_fract)
        n2 = round(512*self.basis_fract)
        cfg = [(m,n),(m,n2)]
        self.layer4 = self._make_layer(block, 512, layers[3], cfg, stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(round(512 * block.expansion * self.net_fract), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):

        planes = round(planes*self.net_fract) #if planes!=64 else planes
        downsample = stride != 1 or self.inplanes != planes * block.expansion
        layers = []
        
        
        layers.append(block(self.inplanes, planes, *cfg[0], block.expansion, stride, downsample, conv3x3=self.conv3x3, conv1x1=self.conv1x1))
        
        self.inplanes = planes * block.expansion

        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, *cfg[i], conv3x3=self.conv3x3))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        #print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet18_FLANC(ResNet):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=common.default_conv):
        """Constructs a ResNet-18 model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        num_classes = 100 if "cifar100" in args.data_train else 10
        super(ResNet18_FLANC, self).__init__(BasicBlock, [2, 2, 2, 2], n_basis=args.n_basis, num_classes=num_classes, basis_fract=args.basis_fraction, net_fract=args.net_fraction, conv3x3=conv3x3, conv1x1=conv1x1)
        pretrained = args.pretrained == 'True'
        if conv3x3 == common.default_conv:
            if pretrained:
                self.load(args, strict=True)

    def load(self, args, strict):
        self.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=strict)

def loss_type(loss_para_type):
    if loss_para_type == 'L1':
        loss_fun = nn.L1Loss()
    elif loss_para_type == 'L2':
        loss_fun = nn.MSELoss()
    else:
        raise NotImplementedError
    return loss_fun

def orth_loss(model, args, para_loss_type='L2'):
    #from IPython import embed; embed(); exit()

    #current_state = list(model.named_parameters())
    loss_fun = loss_type(para_loss_type)

    loss = 0
    for l_id in range(1,5):
        layer = getattr(model,"layer"+str(l_id))
        for b_id in range(2): 
            block = getattr(layer,str(b_id))
            for f_id in range(1,3):
                filter_bank = getattr(block,"filter_bank_"+str(f_id))
                #filter_bank_2 = getattr(block,"filter_bank_2")
                all_bank = filter_bank
                num_all_bank = filter_bank.shape[0]
                B = all_bank.view(num_all_bank, -1)
                D = torch.mm(B,torch.t(B))
                D = loss_fun(D, torch.eye(num_all_bank, num_all_bank).cuda())
                loss = loss + D
    return loss
