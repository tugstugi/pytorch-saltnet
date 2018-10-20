"""UNet supporting different encoders."""
import torch
import torch.nn as nn

from losses.lovasz_losses import lovasz_loss_ignore_empty, lovasz_hinge
from .basenet import create_basenet, BASENET_CHOICES
from .oc_net import BaseOC
from .inplace_abn import ActivatedBatchNorm
import torch.nn.functional as F


def upsample(size=None, scale_factor=None):
    return nn.Upsample(size=size, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    # return nn.Upsample(size=size, scale_factor=scale_factor, mode='nearest')


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            # nn.Dropout2d(p=0.1, inplace=True),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            ActivatedBatchNorm(middle_channels),
            # DANetHead(middle_channels, middle_channels),
            BaseOC(in_channels=middle_channels, out_channels=middle_channels,
                   key_channels=middle_channels // 2,
                   value_channels=middle_channels // 2,
                   dropout=0.2),
            # Parameters were chosen to avoid artifacts, suggested by https://distill.pub/2016/deconv-checkerboard/
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            # upsample(scale_factor=2)
        )

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class ConcatPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.ap = nn.AvgPool2d(kernel_size, stride)
        self.mp = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=(1, 1)):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


# https://github.com/ternaus/TernausNet/blob/master/unet_models.py
class UNet(nn.Module):
    def __init__(self, basenet='vgg11', num_filters=16, pretrained='imagenet'):
        super().__init__()
        net, bn, n_pretrained = create_basenet(basenet, pretrained)

        if basenet.startswith('vgg'):
            self.encoder1 = net[0]  # 64
        else:
            # add upsample
            self.encoder1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                net[0])
            self.encoder1.out_channels = net[0].out_channels

        self.encoder2 = net[1]  # 64
        self.encoder3 = net[2]  # 128
        self.encoder4 = net[3]  # 256

        context_channels = num_filters * 8 * 4
        self.encoder5 = nn.Sequential(
            net[4],
            nn.Conv2d(net[4].out_channels, context_channels, kernel_size=3, stride=1, padding=1),
            ActivatedBatchNorm(context_channels, activation='none'),
            BaseOC(in_channels=context_channels, out_channels=context_channels,
                   key_channels=context_channels // 2,
                   value_channels=context_channels // 2,
                   dropout=0.05)
        )
        self.encoder5.out_channels = context_channels

        self.fuse_image = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True)
        )
        self.logit_image = nn.Sequential(
            nn.Linear(32, 1)
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.center = Decoder(self.encoder5.out_channels, num_filters * 8 * 2, num_filters * 8)

        self.decoder5 = Decoder(self.encoder5.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.decoder4 = Decoder(self.encoder4.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 4)
        self.decoder3 = Decoder(self.encoder3.out_channels + num_filters * 4, num_filters * 4 * 2, num_filters * 2)

        if basenet.startswith('vgg'):
            self.decoder2 = Decoder(self.encoder2.out_channels + num_filters * 2, num_filters * 2 * 2, num_filters)
            self.decoder1 = nn.Sequential(
                nn.Conv2d(self.encoder1.out_channels + num_filters, num_filters, kernel_size=3, padding=1),
                nn.ReLU(inplace=True))
        else:
            self.decoder2 = nn.Sequential(
                nn.Conv2d(self.encoder2.out_channels + num_filters * 2, num_filters * 2 * 2, kernel_size=3, padding=1),
                ActivatedBatchNorm(num_filters * 2 * 2),
                nn.Conv2d(num_filters * 2 * 2, num_filters, kernel_size=3, padding=1),
                ActivatedBatchNorm(num_filters))
            self.decoder1 = Decoder(self.encoder1.out_channels + num_filters, num_filters * 2, num_filters)

        self.logit = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(96, num_filters, kernel_size=3, padding=1),
            ActivatedBatchNorm(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )

        self.fuse_pixel = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(num_filters * (8 + 4 + 2 + 1 + 1), 64, kernel_size=1, padding=0)
        )
        self.logit_pixel5 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(num_filters * 8, num_filters, kernel_size=3, padding=1),
            ActivatedBatchNorm(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )
        self.logit_pixel4 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(num_filters * 4, num_filters, kernel_size=3, padding=1),
            ActivatedBatchNorm(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )
        self.logit_pixel3 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, padding=1),
            ActivatedBatchNorm(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )
        self.logit_pixel2 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            ActivatedBatchNorm(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )
        self.logit_pixel1 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            ActivatedBatchNorm(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] -= mean[0]
        x[:, 0, :, :] /= std[0]
        x[:, 1, :, :] -= mean[1]
        x[:, 1, :, :] /= std[1]
        x[:, 2, :, :] -= mean[2]
        x[:, 2, :, :] /= std[2]

        # depth encoding / CoordConv
        # coord_scale = 1 / std[0]
        # coord_x = (torch.abs(torch.linspace(-1, 1, steps=x.size(3))) - 0.5) * coord_scale
        # x[:, 1] = coord_x.unsqueeze(0).expand_as(x[:, 1])
        # coord_y = (torch.linspace(-1, 1, steps=x.size(2))) * coord_scale
        # x[:, 2] = coord_y.unsqueeze(-1).expand_as(x[:, 2])

        e1 = self.encoder1(x)  # ; print('e1', e1.size())
        e2 = self.encoder2(e1)  # ; print('e2', e2.size())
        e3 = self.encoder3(e2)  # ; print('e3', e3.size())
        e4 = self.encoder4(e3)  # ; print('e4', e4.size())
        e5 = self.encoder5(e4)  # ; print('e5', e5.size())

        c = self.center(self.pool(e5))  # ; print('c', c.size())

        d5 = self.decoder5(c, e5)  # ; print('d5', d5.size())
        d4 = self.decoder4(d5, e4)  # ; print('d4', d4.size())
        d3 = self.decoder3(d4, e3)  # ; print('d3', d3.size())
        d2 = self.decoder2(torch.cat((d3, e2), 1))  # ; print('d2', d2.size())
        d1 = self.decoder1(d2, e1)  # ; print('d1', d1.size())

        d1_size = d1.size()[2:]
        upsampler = upsample(size=d1_size)
        u5 = upsampler(d5)
        u4 = upsampler(d4)
        u3 = upsampler(d3)
        u2 = upsampler(d2)

        d = torch.cat((d1, u2, u3, u4, u5), 1)
        # logit = self.logit(d)#;print(logit.size())

        fuse_pixel = self.fuse_pixel(d)

        logit_pixel = (
            self.logit_pixel1(d1), self.logit_pixel2(u2), self.logit_pixel3(u3), self.logit_pixel4(u4),
            self.logit_pixel5(u5),
        )

        e = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)  # image pool
        e = F.dropout(e, p=0.50, training=self.training)
        fuse_image = self.fuse_image(e)
        logit_image = self.logit_image(fuse_image).view(-1)

        # print(fuse_pixel.size())
        # print(fuse_image.size())
        logit = self.logit(torch.cat([  # fuse
            fuse_pixel,
            F.upsample(fuse_image.view(batch_size, -1, 1, 1, ), scale_factor=128, mode='nearest')
        ], 1))

        return logit, logit_pixel, logit_image


def symmetric_lovasz(outputs, targets):
    return (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1 - targets)) / 2


def symmetric_lovasz_ignore_empty(outputs, targets, truth_image):
    return (lovasz_loss_ignore_empty(outputs, targets, truth_image) +
            lovasz_loss_ignore_empty(-outputs, 1 - targets, truth_image)) / 2


def deep_supervised_criterion(logit, logit_pixel, logit_image, truth_pixel, truth_image, is_average=True):
    loss_image = F.binary_cross_entropy_with_logits(logit_image, truth_image, reduce=is_average)
    loss_pixel = 0
    for l in logit_pixel:
        loss_pixel += symmetric_lovasz_ignore_empty(l.squeeze(1), truth_pixel, truth_image)
    loss = symmetric_lovasz(logit.squeeze(1), truth_pixel)
    return 0.05 * loss_image + 0.1 * loss_pixel + 1 * loss


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--basenet", choices=BASENET_CHOICES, default='vgg11', help='model of basenet')
    parser.add_argument("--num-filters", type=int, default=16, help='num filters for decoder')

    args = parser.parse_args()

    net = UNet(**vars(args))
    # print(net)
    parameters = [p for p in net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in parameters)
    print('N of parameters {} ({} tensors)'.format(n_params, len(parameters)))
    encoder_parameters = [p for name, p in net.named_parameters() if p.requires_grad and name.startswith('encoder')]
    n_encoder_params = sum(p.numel() for p in encoder_parameters)
    print('N of encoder parameters {} ({} tensors)'.format(n_encoder_params, len(encoder_parameters)))
    print('N of decoder parameters {} ({} tensors)'.format(n_params - n_encoder_params,
                                                           len(parameters) - len(encoder_parameters)))

    x = torch.empty((1, 3, 128, 128))
    y = net(x)
    print(x.size(), '-->', y.size())
