import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from model import common
import torchvision.models as models
def make_model(args, parent=False):
    print(args)
    return VGG(args[0])

# reference: torchvision
class VGG(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=None):
        super(VGG, self).__init__()
        # args = args[0]
        # we use batch noramlization for VGG
        norm = common.default_norm
        self.norm = norm
        bias = not args.no_bias

        configs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '16': [64, 64, 'M', 128, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 512, 'M'],
            '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            'ef': [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
        }

        body_list = []
        in_channels = args.n_colors
        if args.data_train.find('CIFAR') >= 0:

            for i, v in enumerate(configs[args.vgg_type]):
                if v == 'M':
                    body_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    conv2d = conv3x3(in_channels, v, kernel_size=3)
                    if norm is not None:
                        body_list += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        body_list += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v


        # assert(args.data_train.find('CIFAR') >= 0)
        self.features = nn.Sequential(*body_list)
        if args.data_train.find('CIFAR') >= 0:
            n_classes = int(args.data_train[5:])
            self.classifier = nn.Linear(in_channels, n_classes)
        elif args.data_train == 'ImageNet':
            n_classes = 1000
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, n_classes),
            )
            # self.classifier = nn.Sequential(
            #     nn.Linear(512 * 7 * 7, n_classes)
            # )
        print(conv3x3, conv3x3 == common.default_conv or conv3x3 == nn.Conv2d)

        if conv3x3 == common.default_conv or conv3x3 == nn.Conv2d:
            self.load(args, strict=True)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        return x

    def load(self, args, strict=True):
        model_dir = os.path.join('..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        if args.data_train.find('CIFAR') >= 0:
            if args.pretrained == 'download' or args.extend == 'download':
                url = (
                    'https://cv.snu.ac.kr/'
                    'research/clustering_kernels/models/vgg16-89711a85.pt'
                )

                state = model_zoo.load_url(url, model_dir=model_dir)
            elif args.extend:
                state = torch.load(args.extend)
            else:
                common.init_vgg(self)
                return
        elif args.data_train == 'ImageNet':
            if args.pretrained == 'download':
                #print('pretrained download')
                if self.norm is not None:
                    url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
                else:
                    url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
                state = model_zoo.load_url(url, model_dir=model_dir)
            else:
                common.init_vgg(self)
                return
        else:
            raise NotImplementedError('Unavailable dataset {}'.format(args.data_train))
        #print(state['features.0.bias'])
        self.load_state_dict(state, strict=strict)
