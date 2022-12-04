import math
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
count_data = 0


def add_feature_map_storage_handle(modules, save_dir, store_input=False, store_output=False, layer_select=-1):
    # modules = find_conv(net, criterion)
    for l, m in enumerate(modules):
        if layer_select == -1 or layer_select == l:
            m.__count_layer__ = l
            m.__save_dir__ = os.path.join(save_dir, '{}_{}'.format(m.__class__.__name__, m.__count_layer__))
            if not os.path.exists(m.__save_dir__):
                os.makedirs(m.__save_dir__)
            # print(m.__save_dir__, m.__count_layer__)
            if store_output and store_input:
                handle = m.register_forward_hook(feature_map_storage_hook)
                m.__storage_handle__ = handle
            elif store_output and not store_input:
                handle = m.register_forward_hook(output_feature_map_storage_hook)
                m.__storage_handle__ = handle
            elif not store_output and store_input:
                handle = m.register_forward_hook(input_feature_map_storage_hook)
                m.__storage_handle__ = handle
            else:
                raise NotImplementedError('Feature maps need to be stored for calling the function ' + add_feature_map_storage_handle.__name__)


def remove_feature_map_storage_handle(modules, store_input=False, store_output=False):
    # modules = find_conv(net, criterion)
    for m in modules:
        if store_output or store_input:
            m.__storage_handle__.remove()
            del m.__storage_handle__
            del m.__save_dir__
            del m.__count_layer__


def reset_feature_map_storage_handle(modules, save_dir, store_input=False, store_output=False, layer_select=-1):
    remove_feature_map_storage_handle(modules, store_input, store_output)
    add_feature_map_storage_handle(modules, save_dir, store_input, store_output, layer_select=layer_select)


def output_feature_map_storage_hook(module, input ,output):
    global count_data
    features = {'output': output}
    torch.save(features, os.path.join(module.__save_dir__, 'Batch{}_Device{}.pt'.format(count_data, torch.cuda.current_device())))
    print('{} {}, Data Batch {}, Device {}'.format(module.__class__.__name__, module.__count_layer__, count_data, torch.cuda.current_device()))


def input_feature_map_storage_hook(module, input, output):
    global count_data
    features = {'input': input[0]}
    torch.save(features, os.path.join(module.__save_dir__, 'Batch{}_Device{}.pt'.format(count_data, torch.cuda.current_device())))
    print('{} {}, Data Batch {}, Device {}'.format(module.__class__.__name__, module.__count_layer__, count_data, torch.cuda.current_device()))


def feature_map_storage_hook(module, input, output):
    global count_data
    features = {'input': input[0], 'output': output}
    torch.save(features, os.path.join(module.__save_dir__, 'Batch{}_Device{}.pt'.format(count_data, torch.cuda.current_device())))
    print('{} {}, Data Batch {}, Device {}'.format(module.__class__.__name__, module.__count_layer__, count_data, torch.cuda.current_device()))
    # from IPython import embed; embed()
    # print(torch.cuda.current_device())


def activate_dconv2d_feature_map_storage(modules, save_dir, store_input=False, store_middle=False, store_output=False,
                                         layer_select=-1):
    for l, m in enumerate(modules):
        if layer_select == -1 or layer_select == l:
            m.__save_dir__ = os.path.join(save_dir, '{}_{}'.format(m.__class__.__name__, l))
            if not os.path.exists(m.__save_dir__):
                os.makedirs(m.__save_dir__)
            m.__count_layer__ = l
            m.__store_input__ = store_input
            m.__store_output__ = store_output
            m.__store_middle__ = store_middle
            print(m.__store_input__, m.__store_output__, m.__store_middle__)


def deactivate_dconv2d_feature_map_storage(modules, layer_select=-1):
    for l, m in enumerate(modules):
        if layer_select == -1 or layer_select == l:
            del m.__save_dir__, m.__count_layer__
            del m.__store_input__
            del m.__store_output__
            del m.__store_middle__


class DConv2d(nn.Module):

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=1, bias=None):
        super(DConv2d, self).__init__()
        self.stride = stride

    def __repr__(self):
        s = '{}(weight_size={}, '.format(self.__class__.__name__, self.weight.shape)
        bias_shape = None if self.bias is None else self.bias.shape
        if hasattr(self, 'projection'):
            s += 'projection_size={}, bias_size={}, '.format(self.projection.shape,bias_shape)
        else:
            s += 'bias_size={}, '.format(bias_shape)
        if hasattr(self, 'projection2'):
            s += 'projection2_size={}, '.format(self.projection2.shape)
        return s
        # if hasattr(self, 'projection2'):
        #     return '{}(weight_size={}, projection_size={}, projection2_size={})'\
        #         .format(self.__class__.__name__, self.weight.shape, self.projection.shape, self.projection2.shape)
        # elif hasattr(self, 'projection'):
        #     return '{}(weight_size={}, projection_size={})'.\
        #         format(self.__class__.__name__, self.weight.shape, self.projection.shape)
        # else:
        #     return '{}(weight_size={}'.\
        #         format(self.__class__.__name__, self.weight.shape)

    def set_params(self, input):
        self.weight = input['weight']
        self.bias = input['bias']
        if 'projection' in input:
            self.projection = input['projection']
        if 'projection2' in input:
            self.projection2 = input['projection2']
        self.padding = self.weight.shape[-1] // 2

    def forward(self, x):
        # print(x.shape, self.weight.shape)
        if hasattr(self, 'projection'):
            bias_shape = None if self.bias is None else self.bias.shape[0]
            # if self.weight.shape[0] == bias_shape:
            #     m = F.conv2d(x, weight=self.weight, bias=self.bias, padding=self.padding, stride=self.stride)
            #     y = F.conv2d(m, weight=self.projection)
            # else:
            m = F.conv2d(x, weight=self.weight, padding=self.padding, stride=self.stride)
            y = F.conv2d(m, weight=self.projection, bias=self.bias)

            if hasattr(self, '__store_input__'):
                self.feature_map_storage(x, m, y)
                # print(self.__class__.__name__, self.__store_input__, self.__store_output__, self.__store_middle__, id(self), torch.cuda.current_device(), self.__count_layer__)
        else:
            y = F.conv2d(x, weight=self.weight, bias=self.bias, padding=self.padding, stride=self.stride)

        if hasattr(self, 'projection2'):
            y = F.conv2d(x, weight=self.projection2)

        #     basis_size = self.weight.shape[1]
        #     if basis_size == x.shape[1]:
        #         x = F.conv2d(x, weight=self.weight, bias=None, padding=self.padding)
        #         x = F.conv2d(x, weight=self.projection, bias=self.bias)
        #     else:
        #         x = torch.cat([F.conv2d(xi, weight=self.weight, bias=None, padding=self.padding)
        #                        for xi in torch.split(x, basis_size, dim=1)], dim=1)
        #         x = F.conv2d(x, weight=self.projection, bias=self.bias)
        # else:
        #     x = F.conv2d(x, weight=self.weight, bias=self.bias, padding=self.padding)
        return y

    def feature_map_storage(self, x, m, y):
        features = {}
        if self.__store_input__:
            features['input'] = x
        if self.__store_middle__:
            features['middle'] = m.norm(dim=(2, 3)).mean(dim=0, keepdim=True)
        if self.__store_output__:
            features['output'] = y
        torch.save(features, os.path.join(self.__save_dir__, 'Batch{}_Device{}.pt'.format(count_data, torch.cuda.current_device())))
        print('{} {}, Data Batch {}, Device {}'.format(self.__class__.__name__, self.__count_layer__, count_data, torch.cuda.current_device()))

    def feature_map_inter_norm(self, m):
        feat = m.detach().cpu().norm(dim=(2, 3)).mean(dim=0, keepdim=True)
        if hasattr(self, '__feature_map_norm__'):
            self.__feature_map_norm__ = torch.cat((self.__feature_map_norm__, feat), dim=0)
        else:
            self.__feature_map_norm__ = feat
        print(self.__class__.__name__, self.__store_input__, self.__store_output__, self.__store_middle__, id(self))

    # def feature_map_inter_norm_2(self, m):
    #     dv = torch.cuda.current_device()
    #     tn = '__feature_map_norm_device{}__'.format(dv)
    #     feat = getattr(self, tn)
    #     feat_add = m.detach().norm(dim=(2, 3))
    #     if hasattr(self, tn):
    #         setattr(self, tn, torch.cat((feat, feat_add), dim=0))
    #     else:
    #         setattr(self, tn, feat_add)

def default_conv(
        in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):

    m = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias, groups=groups
    )
    return m

def nopad_conv(
        in_channels, out_channels, kernel_size, stride=1, bias=True):

    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=0, stride=stride, bias=bias)

def default_linear(in_channels, out_channels, bias=True):
    return nn.Linear(in_channels, out_channels, bias=bias)

def default_norm(in_channels):
    return nn.BatchNorm2d(in_channels)

def default_act():
    return nn.ReLU(False)

def init_vgg(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            c_out, _, kh, kw = m.weight.size()
            n = kh * kw * c_out
            m.weight.data.normal_(0, math.sqrt(2 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            _, c_in, kh, kw = m.weight.size()
            n = kh * kw * c_in
            m.weight.data.normal_(0, math.sqrt(2 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            _, c_in = m.weight.size()
            m.weight.data.normal_(0, math.sqrt(1 / c_in))
            m.bias.data.zero_()

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=True,
        conv3x3=default_conv, norm=default_norm, act=default_act):
        modules = []
        modules.append(conv3x3(
            in_channels, out_channels, kernel_size, stride=stride, bias=bias
        ))
        if norm is not None: modules.append(norm(out_channels))
        if act is not None: modules.append(act())

        super(BasicBlock, self).__init__(*modules)
