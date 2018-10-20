"""Encoders for the UNet."""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from .inplace_abn import ActivatedBatchNorm

BASENET_CHOICES = ('vgg11', 'vgg13', 'vgg16', 'vgg19',
                   'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                   'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                   'resnext101_32x4d', 'resnext101_64x4d',
                   'se_resnet50', 'se_resnet101', 'se_resnet152',
                   'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154',
                   'darknet')

MODEL_ZOO_URL = 'https://drontheimerstr.synology.me/model_zoo/'

MODEL_URLS = {
    'resnet50': {
        # 'voc':  MODEL_ZOO_URL + 'SSDretina_resnet50_c21-fb6036d1.pth',  # SSDretina_resnet50_c81-a584ead7.pth pretrained
        'voc': MODEL_ZOO_URL + 'SSDretina_resnet50_c21-1c85a349.pth',  # SSDretina_resnet50_c501-06095077.pth pretrained
        'coco': MODEL_ZOO_URL + 'SSDretina_resnet50_c81-a584ead7.pth',
        'oid': MODEL_ZOO_URL + 'SSDretina_resnet50_c501-06095077.pth'},
    'resnext101_32x4d': {'coco': MODEL_ZOO_URL + 'SSDretina_resnext101_32x4d_c81-fdb37546.pth'}
}


def conv(*args, **kwargs):
    return lambda last_layer: nn.Conv2d(last_layer.out_channels, *args, **kwargs)


def get_out_channels(layers):
    """access out_channels from last layer of nn.Sequential/list"""
    if hasattr(layers, 'out_channels'):
        return layers.out_channels
    elif isinstance(layers, int):
        return layers
    else:
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            if hasattr(layer, 'out_channels'):
                return layer.out_channels
            elif isinstance(layer, nn.Sequential):
                return get_out_channels(layer)
    raise RuntimeError("cant get_out_channels from {}".format(layers))


def Sequential(*args):
    f = nn.Sequential(*args)
    f.out_channels = get_out_channels(args)
    return f


def sequential(*args):
    def create_sequential(last_layer):
        layers = []
        for a in args:
            layers.append(a(last_layer))
            last_layer = layers[-1]
        return Sequential(*layers)

    return create_sequential


def ConvBnRelu(*args, **kwargs):
    """drop in block for nn.Conv2d with BatchNorm and ReLU"""
    c = nn.Conv2d(*args, **kwargs)
    return Sequential(c,
                      nn.BatchNorm2d(c.out_channels),
                      nn.ReLU(inplace=True))


def conv_bn_relu(*args, **kwargs):
    return lambda last_layer: ConvBnRelu(get_out_channels(last_layer), *args, **kwargs)


def ConvRelu(*args, **kwargs):
    return Sequential(nn.Conv2d(*args, **kwargs),
                      nn.ReLU(inplace=True))


def conv_relu(*args, **kwargs):
    return lambda last_layer: ConvRelu(get_out_channels(last_layer), *args, **kwargs)


def ReluConv(*args, **kwargs):
    return Sequential(nn.ReLU(inplace=True),
                      nn.Conv2d(*args, **kwargs))


def relu_conv(*args, **kwargs):
    return lambda last_layer: ReluConv(get_out_channels(last_layer), *args, **kwargs)


def BnReluConv(*args, **kwargs):
    """drop in block for nn.Conv2d with BatchNorm and ReLU"""
    c = nn.Conv2d(*args, **kwargs)
    return Sequential(nn.BatchNorm2d(c.in_channels),
                      nn.ReLU(inplace=True),
                      c)


def bn_relu_conv(*args, **kwargs):
    return lambda last_layer: BnReluConv(get_out_channels(last_layer), *args, **kwargs)


def vgg_base_extra(bn):
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    block = ConvBnRelu if bn else ConvRelu
    conv6 = block(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = block(1024, 1024, kernel_size=1)
    return [pool5, conv6, conv7]


def vgg(name, pretrained):
    if name == 'vgg11':
        net_class = torchvision.models.vgg11
    elif name == 'vgg13':
        net_class = torchvision.models.vgg13
    elif name == 'vgg16':
        net_class = torchvision.models.vgg16
    elif name == 'vgg19':
        net_class = torchvision.models.vgg19
    elif name == 'vgg11_bn':
        net_class = torchvision.models.vgg11_bn
    elif name == 'vgg13_bn':
        net_class = torchvision.models.vgg13_bn
    elif name == 'vgg16_bn':
        net_class = torchvision.models.vgg16_bn
    elif name == 'vgg19_bn':
        net_class = torchvision.models.vgg19_bn
    else:
        raise RuntimeError("unknown model {}".format(name))

    imagenet_pretrained = pretrained == 'imagenet'
    vgg = net_class(pretrained=imagenet_pretrained)

    # for have exact same layout as original paper
    if name == 'vgg16':
        vgg.features[16].ceil_mode = True

    bn = name.endswith('bn')
    layers = []
    l = []
    for i in range(len(vgg.features) - 1):
        if isinstance(vgg.features[i], nn.MaxPool2d):
            layers.append(l)
            l = []
        l.append(vgg.features[i])
    l += vgg_base_extra(bn=bn)
    layers.append(l)

    # layers of feature scaling 2**5
    block = ConvBnRelu if bn else ConvRelu
    layer5 = [block(1024, 256, 1, 1, 0),
              block(256, 512, 3, 2, 1)]
    layers.append(layer5)

    layers = [Sequential(*l) for l in layers]
    n_pretrained = 4 if imagenet_pretrained else 0
    return layers, bn, n_pretrained


def resnet(name, pretrained):
    if name == 'resnet18':
        net_class = torchvision.models.resnet18
    elif name == 'resnet34':
        net_class = torchvision.models.resnet34
    elif name == 'resnet50':
        net_class = torchvision.models.resnet50
    elif name == 'resnet101':
        net_class = torchvision.models.resnet101
    elif name == 'resnet152':
        net_class = torchvision.models.resnet152

    imagenet_pretrained = pretrained == 'imagenet'
    resnet = net_class(pretrained=imagenet_pretrained)
    layer0 = Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

    layer0[-1].out_channels = resnet.bn1.num_features

    def get_out_channels_from_resnet_block(layer):
        block = layer[-1]
        if isinstance(block, torchvision.models.resnet.BasicBlock):
            return block.conv2.out_channels
        elif isinstance(block, torchvision.models.resnet.Bottleneck):
            return block.conv3.out_channels
        raise RuntimeError("unknown resnet block: {}".format(block))

    resnet.layer1.out_channels = resnet.layer1[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer1)
    resnet.layer2.out_channels = resnet.layer2[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer2)
    resnet.layer3.out_channels = resnet.layer3[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer3)
    resnet.layer4.out_channels = resnet.layer4[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer4)
    n_pretrained = 5 if imagenet_pretrained else 0
    return [layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4], True, n_pretrained


def resnext(name, pretrained):
    import pretrainedmodels
    if name in ['resnext101_32x4d', 'resnext101_64x4d']:
        imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
        resnext = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=imagenet_pretrained)
    else:
        return NotImplemented

    resnext_features = resnext.features
    layer0 = [resnext_features[i] for i in range(4)]
    layer0 = nn.Sequential(*layer0)
    layer0.out_channels = layer0[-1].out_channels = 64

    layer1 = resnext_features[4]
    layer1.out_channels = layer1[-1].out_channels = 256

    layer2 = resnext_features[5]
    layer2.out_channels = layer2[-1].out_channels = 512

    layer3 = resnext_features[6]
    layer3.out_channels = layer3[-1].out_channels = 1024

    layer4 = resnext_features[7]
    layer4.out_channels = layer4[-1].out_channels = 2048
    n_pretrained = 5 if imagenet_pretrained else 0
    return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained


def replace_bn(bn, act=None):
    slop = 0.01
    if isinstance(act, nn.ReLU):
        activation = 'leaky_relu'  # approximate relu
    elif isinstance(act, nn.LeakyReLU):
        activation = 'leaky_relu'
        slope = act.negative_slope
    elif isinstance(act, nn.ELU):
        activation = 'elu'
    else:
        activation = 'none'
    abn = ActivatedBatchNorm(num_features=bn.num_features,
                             eps=bn.eps,
                             momentum=bn.momentum,
                             affine=bn.affine,
                             track_running_stats=bn.track_running_stats,
                             activation=activation,
                             slope=slop)
    abn.load_state_dict(bn.state_dict())
    return abn


def replace_bn_in_sequential(layer0, block=None):
    layer0_modules = []
    last_bn = None
    for n, m in layer0.named_children():
        if isinstance(m, nn.BatchNorm2d):
            last_bn = (n, m)
        else:
            activation = 'none'
            if last_bn:
                abn = replace_bn(last_bn[1], m)
                activation = abn.activation
                layer0_modules.append((last_bn[0], abn))
                last_bn = None
            if activation == 'none':
                if block and isinstance(m, block):
                    m = replace_bn_in_block(m)
                elif isinstance(m, nn.Sequential):
                    m = replace_bn_in_sequential(m, block)
                layer0_modules.append((n, m))
    if last_bn:
        abn = replace_bn(last_bn[1])
        layer0_modules.append((last_bn[0], abn))
    return nn.Sequential(OrderedDict(layer0_modules))


class DummyModule(nn.Module):
    def forward(self, x):
        return x


def replace_bn_in_block(block):
    block.bn1 = replace_bn(block.bn1, block.relu)
    block.bn2 = replace_bn(block.bn2, block.relu)
    block.bn3 = replace_bn(block.bn3)
    block.relu = DummyModule()
    if block.downsample:
        block.downsample = replace_bn_in_sequential(block.downsample)
    return nn.Sequential(block,
                         nn.ReLU(inplace=True))


def se_net(name, pretrained):
    import pretrainedmodels
    if name in ['se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154']:
        imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
        senet = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=imagenet_pretrained)
    else:
        return NotImplemented

    layer0 = replace_bn_in_sequential(senet.layer0)

    block = senet.layer1[0].__class__
    layer1 = replace_bn_in_sequential(senet.layer1, block=block)
    layer1.out_channels = layer1[-1].out_channels = senet.layer1[-1].conv3.out_channels
    layer0.out_channels = layer0[-1].out_channels = senet.layer1[0].conv1.in_channels

    layer2 = replace_bn_in_sequential(senet.layer2, block=block)
    layer2.out_channels = layer2[-1].out_channels = senet.layer2[-1].conv3.out_channels

    layer3 = replace_bn_in_sequential(senet.layer3, block=block)
    layer3.out_channels = layer3[-1].out_channels = senet.layer3[-1].conv3.out_channels

    layer4 = replace_bn_in_sequential(senet.layer4, block=block)
    layer4.out_channels = layer4[-1].out_channels = senet.layer4[-1].conv3.out_channels

    n_pretrained = 5 if imagenet_pretrained else 0
    return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained


def darknet(pretrained):
    from .darknet import KitModel as DarkNet
    net = DarkNet()
    if pretrained:
        state_dict = torch.load("/media/data/model_zoo/coco/pytorch_yolov3.pth")
        net.load_state_dict(state_dict)
    n_pretrained = 3 if pretrained else 0
    return [net.model0, net.model1, net.model2], True, n_pretrained


class MockModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.backbone = nn.ModuleList(layers)


def load_pretrained_weights(layers, name, dataset_name):
    state_dict = model_zoo.load_url(MODEL_URLS[name][dataset_name])
    mock_module = MockModule(layers)
    mock_module.load_state_dict(state_dict, strict=False)


def create_basenet(name, pretrained):
    """
    Parameters
    ----------
    name: model name
    pretrained: dataset name

    Returns
    -------
    list of modules, is_batchnorm, num_of_pretrained_module
    """
    if name.startswith('vgg'):
        layers, bn, n_pretrained = vgg(name, pretrained)
    elif name.startswith('resnet'):
        layers, bn, n_pretrained = resnet(name, pretrained)
    elif name.startswith('resnext'):
        layers, bn, n_pretrained = resnext(name, pretrained)
    elif name.startswith('se'):
        layers, bn, n_pretrained = se_net(name, pretrained)
    elif name == 'darknet':
        layers, bn, n_pretrained = darknet(pretrained)
    else:
        raise NotImplemented(name)

    if pretrained in ('coco', 'oid'):
        load_pretrained_weights(layers, name, pretrained)
        n_pretrained = len(layers)

    return layers, bn, n_pretrained


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="show network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', nargs=1, choices=BASENET_CHOICES, help='load saved model')
    parser.add_argument('--pretrained', default=False, type=str, choices=('imagenet', 'voc', 'coco', 'oid'),
                        help='pretrained dataset')

    args = parser.parse_args()
    model = create_basenet(args.model[0], args.pretrained)
    print(model)
