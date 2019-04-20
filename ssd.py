import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os


# TODO: 1. validate the SSD512 network and the first is the 'config'
# TODO: 2. implement the vertical offset of the bounding boxes
# TODO: 3. modify the number in mbox['512']
# TODO: 4. modify the size of loc and conf
# TODO: 5. modify the kernel size in the multibox layer and calculate the padding size
# TODO: 6. validate whether vgg_source=[21, -2] or not


class TBPP(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(TBPP, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = {
            'num_classes': 2,
            'lr_steps': (80000, 100000, 120000),
            'max_iter': 120000,
            'feature_maps': [64, 32, 16, 8, 4, 2, 1],
            'min_dim': 512,
            'steps': [8, 16, 32, 64, 128, 256, 512],
            'min_sizes': [20, 51, 133, 215, 296, 378, 460],
            'max_sizes': [51, 133, 215, 296, 378, 460, 542],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'MINE'}
        self.priorbox = PriorBox(self.cfg)  # calculate the size of prior boxes, i.e. defaults boxes
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # TBPP network
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc predictions
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),  # conf predictions
                self.priors.type(type(x.data)))  # prior boxes, i.e. default boxes
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors)

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict ...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Loaded!')
        else:
            print('Sorry, only .pth and .pkl files are supported.')


def vgg(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, in_channels):
    layers = []
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):  # the index start from 2
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128]}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [4, 6, 6, 6, 6, 4, 4]}


def build_tbpp(phase, size=512, num_classes=2):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 512:
        print("ERROR: You specified size " + repr(size) + ". Only size=512 are supported!")
        return
    base_net_list, extras_net_list, head = multibox(
        vgg=vgg(cfg=base[str(size)], in_channels=3, batch_norm=True),
        extra_layers=add_extras(cfg=extras[str(size)], in_channels=1024),
        cfg=mbox[str(size)],
        num_classes=num_classes)
    tbpp = TBPP(phase, size, base_net_list, extras_net_list, head, num_classes)
    return tbpp
