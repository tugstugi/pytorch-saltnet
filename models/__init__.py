import torch
import torch.nn as nn

from .unet import UNet, BASENET_CHOICES, deep_supervised_criterion


def create(model, basenet, pretrained):
    if model == 'unet':
        net = UNet(basenet=basenet, pretrained=pretrained)
    else:
        raise NotImplementedError(model)

    net.args = dict(model=model, basenet=basenet, pretrained=pretrained)

    return net


def save(net, filename):
    if isinstance(net, nn.DataParallel):
        net = net.module

    data = dict(args=net.args,
                state_dict=net.state_dict())
    torch.save(data, filename)


def load(filename, use_gpu=True):
    print('load {}'.format(filename))
    data = torch.load(filename, map_location=None if use_gpu else 'cpu')
    net = create(**data['args'])
    net.load_state_dict(data['state_dict'])
    return net
