#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
import torch.nn.functional as F
from torch.autograd import Variable
import os
from layers import *
from data.config import cfg
import numpy as np


class conv_bn(nn.Module):
    """docstring for conv"""

    def __init__(self,
                 in_plane,
                 out_plane,
                 kernel_size,
                 stride,
                 padding):
        super(conv_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, out_plane,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        x = self.conv1(x)
        return self.bn1(x)


class CPM(nn.Module):
    """docstring for CPM"""

    def __init__(self, in_plane):
        super(CPM, self).__init__()
        self.branch1 = conv_bn(in_plane, 1024, 1, 1, 0)
        self.branch2a = conv_bn(in_plane, 256, 1, 1, 0)
        self.branch2b = conv_bn(256, 256, 3, 1, 1)
        self.branch2c = conv_bn(256, 1024, 1, 1, 0)

        self.ssh_1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.ssh_dimred = nn.Conv2d(
            1024, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_3a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out_residual = self.branch1(x)
        x = F.relu(self.branch2a(x), inplace=True)
        x = F.relu(self.branch2b(x), inplace=True)
        x = self.branch2c(x)

        rescomb = F.relu(x + out_residual, inplace=True)
        ssh1 = self.ssh_1(rescomb)
        ssh_dimred = F.relu(self.ssh_dimred(rescomb), inplace=True)
        ssh_2 = self.ssh_2(ssh_dimred)
        ssh_3a = F.relu(self.ssh_3a(ssh_dimred), inplace=True)
        ssh_3b = self.ssh_3b(ssh_3a)

        ssh_out = torch.cat([ssh1, ssh_2, ssh_3b], dim=1)
        ssh_out = F.relu(ssh_out, inplace=True)
        return ssh_out


class PyramidBox(nn.Module):
    """docstring for PyramidBox"""

    def __init__(self,
                 phase,
                 base,
                 extras,
                 lfpn_cpm,
                 head,
                 num_classes):
        super(PyramidBox, self).__init__()
        #self.use_transposed_conv2d = use_transposed_conv2d

        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)
        
        self.lfpn_topdown = nn.ModuleList(lfpn_cpm[0])
        self.lfpn_later = nn.ModuleList(lfpn_cpm[1])
        self.cpm = nn.ModuleList(lfpn_cpm[2])

        self.loc_layers = nn.ModuleList(head[0])
        self.conf_layers = nn.ModuleList(head[1])

        

        self.is_infer = False
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)
            self.is_infer = True

    def _upsample_prod(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') * y

    def forward(self, x):
        size = x.size()[2:]

        # apply vgg up to conv3_3 relu
        for k in range(16):
            x = self.vgg[k](x)
        conv3_3 = x
        # apply vgg up to conv4_3
        for k in range(16, 23):
            x = self.vgg[k](x)
        conv4_3 = x

        for k in range(23, 30):
            x = self.vgg[k](x)
        conv5_3 = x

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        convfc_7 = x

        # apply extra layers and cache source layer outputs
        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)
        conv6_2 = x
        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)
        conv7_2 = x

        x = F.relu(self.lfpn_topdown[0](convfc_7), inplace=True)
        lfpn2_on_conv5 = F.relu(self._upsample_prod(
            x, self.lfpn_later[0](conv5_3)), inplace=True)

        x = F.relu(self.lfpn_topdown[1](lfpn2_on_conv5), inplace=True)
        lfpn1_on_conv4 = F.relu(self._upsample_prod(
            x, self.lfpn_later[1](conv4_3)), inplace=True)

        x = F.relu(self.lfpn_topdown[2](lfpn1_on_conv4), inplace=True)
        lfpn0_on_conv3 = F.relu(self._upsample_prod(
            x, self.lfpn_later[2](conv3_3)), inplace=True)

        ssh_conv3_norm = self.cpm[0](self.L2Norm3_3(lfpn0_on_conv3))
        ssh_conv4_norm = self.cpm[1](self.L2Norm4_3(lfpn1_on_conv4))
        ssh_conv5_norm = self.cpm[2](self.L2Norm5_3(lfpn2_on_conv5))
        ssh_convfc7 = self.cpm[3](convfc_7)
        ssh_conv6 = self.cpm[4](conv6_2)
        ssh_conv7 = self.cpm[5](conv7_2)

        face_locs, face_confs = [], []
        head_locs, head_confs = [], []

        N = ssh_conv3_norm.size(0)

        mbox_loc = self.loc_layers[0](ssh_conv3_norm)
        face_loc, head_loc = torch.chunk(mbox_loc, 2, dim=1)

        face_loc = face_loc.permute(
            0, 2, 3, 1).contiguous().view(N, -1, 4)
        if not self.is_infer:
            head_loc = head_loc.permute(
                0, 2, 3, 1).contiguous().view(N, -1, 4)

        mbox_conf = self.conf_layers[0](ssh_conv3_norm)
        face_conf1 = mbox_conf[:, 3:4, :, :]
        face_conf3_maxin, _ = torch.max(
            mbox_conf[:, 0:3, :, :], dim=1, keepdim=True)

        face_conf = torch.cat((face_conf3_maxin, face_conf1), dim=1)

        face_conf = face_conf.permute(
            0, 2, 3, 1).contiguous().view(N, -1, 2)

        if not self.is_infer:
            head_conf3_maxin, _ = torch.max(
                mbox_conf[:, 4:7, :, :], dim=1, keepdim=True)
            head_conf1 = mbox_conf[:, 7:, :, :]
            head_conf = torch.cat((head_conf3_maxin, head_conf1), dim=1)
            head_conf = head_conf.permute(
                0, 2, 3, 1).contiguous().view(N, -1, 2)

        face_locs.append(face_loc)
        face_confs.append(face_conf)
        if not self.is_infer:
            head_locs.append(head_loc)
            head_confs.append(head_conf)

        inputs = [ssh_conv4_norm, ssh_conv5_norm,
                  ssh_convfc7, ssh_conv6, ssh_conv7]

        feature_maps = []
        feat_size = ssh_conv3_norm.size()[2:]
        feature_maps.append([feat_size[0], feat_size[1]])

        for i, feat in enumerate(inputs):
            feat_size = feat.size()[2:]
            feature_maps.append([feat_size[0], feat_size[1]])
            mbox_loc = self.loc_layers[i + 1](feat)
            face_loc, head_loc = torch.chunk(mbox_loc, 2, dim=1)
            face_loc = face_loc.permute(
                0, 2, 3, 1).contiguous().view(N, -1, 4)
            if not self.is_infer:
                head_loc = head_loc.permute(
                    0, 2, 3, 1).contiguous().view(N, -1, 4)

            mbox_conf = self.conf_layers[i + 1](feat)
            face_conf1 = mbox_conf[:, 0:1, :, :]
            face_conf3_maxin, _ = torch.max(
                mbox_conf[:, 1:4, :, :], dim=1, keepdim=True)
            face_conf = torch.cat((face_conf1, face_conf3_maxin), dim=1)

            face_conf = face_conf.permute(
                0, 2, 3, 1).contiguous().view(N, -1, 2)

            if not self.is_infer:
                head_conf = mbox_conf[:, 4:, :, :].permute(
                    0, 2, 3, 1).contiguous().view(N, -1, 2)

            face_locs.append(face_loc)
            face_confs.append(face_conf)

            if not self.is_infer:
                head_locs.append(head_loc)
                head_confs.append(head_conf)

        face_mbox_loc = torch.cat(face_locs, dim=1)
        face_mbox_conf = torch.cat(face_confs, dim=1)

        if not self.is_infer:
            head_mbox_loc = torch.cat(head_locs, dim=1)
            head_mbox_conf = torch.cat(head_confs, dim=1)

        priors_boxes = PriorBox(size, feature_maps, cfg)
        priors = Variable(priors_boxes.forward(), volatile=True)

        if not self.is_infer:
            output = (face_mbox_loc, face_mbox_conf,
                      head_mbox_loc, head_mbox_conf, priors)
        else:
            output = self.detect(face_mbox_loc,
                                 self.softmax(face_mbox_conf),
                                 priors)

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()


vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
           512, 512, 512, 'M']

extras_cfg = [256, 'S', 512, 128, 'S', 256]

lfpn_cpm_cfg = [256, 512, 512, 1024, 512, 256]

multibox_cfg = [512, 512, 512, 512, 512, 512]


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
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
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def add_lfpn_cpm(cfg):
    lfpn_topdown_layers = []
    lfpn_latlayer = []
    cpm_layers = []

    for k, v in enumerate(cfg):
        cpm_layers.append(CPM(v))

    fpn_list = cfg[::-1][2:]
    for k, v in enumerate(fpn_list[:-1]):
        lfpn_latlayer.append(nn.Conv2d(
            fpn_list[k + 1], fpn_list[k + 1], kernel_size=1, stride=1, padding=0))
        lfpn_topdown_layers.append(nn.Conv2d(
            v, fpn_list[k + 1], kernel_size=1, stride=1, padding=0))

    return (lfpn_topdown_layers, lfpn_latlayer, cpm_layers)


def multibox(vgg, extra_layers):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, 28, -2]
    i = 0
    loc_layers += [nn.Conv2d(multibox_cfg[i],
                             8, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(multibox_cfg[i],
                              8, kernel_size=3, padding=1)]
    i += 1
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(multibox_cfg[i],
                                 8, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(multibox_cfg[i],
                                  6, kernel_size=3, padding=1)]
        i += 1
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(multibox_cfg[i],
                                 8, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(multibox_cfg[i],
                                  6, kernel_size=3, padding=1)]
        i += 1
    return vgg, extra_layers, (loc_layers, conf_layers)


def build_net(phase, num_classes=2):
    base_, extras_, head_ = multibox(
        vgg(vgg_cfg, 3), add_extras((extras_cfg), 1024))
    lfpn_cpm = add_lfpn_cpm(lfpn_cpm_cfg)
    return PyramidBox(phase, base_, extras_, lfpn_cpm, head_, num_classes)


if __name__ == '__main__':
    inputs = Variable(torch.randn(1, 3, 640, 640))
    net = build_net('train', num_classes=2)
    print(net)
    out = net(inputs)
