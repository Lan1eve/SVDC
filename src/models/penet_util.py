from __future__ import print_function
import torch
import torch.nn as nn
import math
import numpy as np


class AddCoordsNp():
    """Add coords to a tensor"""

    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def call(self):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        # batch_size_tensor = np.shape(input_tensor)[0]

        xx_ones = np.ones([self.x_dim], dtype=np.int32)
        xx_ones = np.expand_dims(xx_ones, 1)

        xx_range = np.expand_dims(np.arange(self.y_dim), 0)

        xx_channel = np.matmul(xx_ones, xx_range)
        xx_channel = np.expand_dims(xx_channel, -1)

        yy_ones = np.ones([self.y_dim], dtype=np.int32)
        yy_ones = np.expand_dims(yy_ones, 0)

        yy_range = np.expand_dims(np.arange(self.x_dim), 1)

        yy_channel = np.matmul(yy_range, yy_ones)
        yy_channel = np.expand_dims(yy_channel, -1)

        xx_channel = xx_channel.astype('float32') / (self.y_dim - 1)
        yy_channel = yy_channel.astype('float32') / (self.x_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        ret = np.concatenate([xx_channel, yy_channel], axis=-1)

        if self.with_r:
            rr = np.sqrt(np.square(xx_channel - 0.5) + np.square(yy_channel - 0.5))
            ret = np.concatenate([ret, rr], axis=-1)

        return ret

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def convbnrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias)


class SparseDownSampleClose(nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600

    def forward(self, d, mask):
        encode_d = - (1 - mask) * self.large_number - d

        d = - self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1 - mask_result) * self.large_number

        return d_result, mask_result

class GeometryFeature(nn.Module):
    def __init__(self):
        super(GeometryFeature, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z * (0.5 * h * (vnorm + 1) - ch) / fh
        y = z * (0.5 * w * (unorm + 1) - cw) / fw
        return torch.cat((x, y, z), 1)


class BasicBlockGeo(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, geoplanes=3):
        super(BasicBlockGeo, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = encoding.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes + geoplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes + geoplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2, out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LightEncoder(nn.Module):
    def __init__(self):
        super(LightEncoder, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.adjust = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            self.lrelu,
            nn.Conv2d(16, 16, 3, 2, 1),
            self.lrelu,)  # [1,16,240,320]
        self.conv0 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            self.lrelu,)  # [1,32,120,160]
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            self.lrelu,)  # [1,64,60,80]
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            self.lrelu,)  # [1,64,30,40]

    def forward(self, x):
        features = [self.adjust(x)]
        features.append(self.conv0(features[-1]))
        features.append(self.conv1(features[-1]))
        features.append(self.conv2(features[-1]))

        return features[1:]


class ENet(nn.Module):
    def __init__(self, ):
        super(ENet, self).__init__()
        self.convolutional_layer_encoding = "uv"
        self.network_model = "e"
        self.geofeature = None
        self.geoplanes = 3
        if self.convolutional_layer_encoding == "xyz":
            self.geofeature = GeometryFeature()
        elif self.convolutional_layer_encoding == "std":
            self.geoplanes = 0
        elif self.convolutional_layer_encoding == "uv":
            self.geoplanes = 2
        elif self.convolutional_layer_encoding == "z":
            self.geoplanes = 1

        self.rgb_conv_init = convbnrelu(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=2)

        self.rgb_encoder_layer1 = BasicBlockGeo(inplanes=16, planes=32, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer2 = BasicBlockGeo(inplanes=32, planes=32, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer3 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)

        self.rgb_encoder_layer4 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        # self.rgb_encoder_layer5 = BasicBlockGeo(inplanes=64+32+32, planes=128, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer5 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer6 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        # self.rgb_encoder_layer7 = BasicBlockGeo(inplanes=128+64+64, planes=256, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer7 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer8 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        # self.rgb_encoder_layer9 = BasicBlockGeo(inplanes=256+128+128, planes=512, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer9 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer10 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

        weights_init(self)

    def forward(self, rgb, d, position, depth_features):

        depth_feat1, depth_feat2, depth_feat3 = depth_features
        unorm = position[:, 0:1, :, :]
        vnorm = position[:, 1:2, :, :]

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        valid_mask = torch.where(d > 0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None

        if self.convolutional_layer_encoding == "xyz":
            a = 1
        elif self.convolutional_layer_encoding == "uv":
            geo_s1 = torch.cat((vnorm, unorm), dim=1)
            geo_s2 = torch.cat((vnorm_s2, unorm_s2), dim=1)
            geo_s3 = torch.cat((vnorm_s3, unorm_s3), dim=1)
            geo_s4 = torch.cat((vnorm_s4, unorm_s4), dim=1)
            geo_s5 = torch.cat((vnorm_s5, unorm_s5), dim=1)
            geo_s6 = torch.cat((vnorm_s6, unorm_s6), dim=1)
        elif self.convolutional_layer_encoding == "z":
            geo_s1 = d
            geo_s2 = d_s2
            geo_s3 = d_s3
            geo_s4 = d_s4
            geo_s5 = d_s5
            geo_s6 = d_s6

        rgb_feature = self.rgb_conv_init(torch.cat((rgb, d), dim=1))  # [1, 16, 480, 640]

        rgb_feature1 = self.rgb_encoder_layer1(rgb_feature, geo_s1, geo_s2)  # [1, 32, 240, 320]

        rgb_feature2 = self.rgb_encoder_layer2(rgb_feature1, geo_s2, geo_s2)  # [1, 32, 240, 320]

        rgb_feature3 = self.rgb_encoder_layer3(rgb_feature2, geo_s2, geo_s3)  # [1, 64, 120, 160]

        rgb_feature4 = self.rgb_encoder_layer4(rgb_feature3, geo_s3, geo_s3)  # [1, 64, 120, 160]

        rgb_feature5 = self.rgb_encoder_layer5(rgb_feature4, geo_s3, geo_s4)  # [1, 128, 60, 80]
        rgb_feature6 = self.rgb_encoder_layer6(rgb_feature5, geo_s4, geo_s4)  # [1, 128, 60, 80]

        rgb_feature7 = self.rgb_encoder_layer7(rgb_feature6, geo_s4, geo_s5)  # [1, 256, 30, 40]
        rgb_feature8 = self.rgb_encoder_layer8(rgb_feature7, geo_s5, geo_s5)  # [1, 256, 30, 40]

        rgb_feature9 = self.rgb_encoder_layer9(rgb_feature8, geo_s5, geo_s6)  # [1, 512, 15, 20]
        rgb_feature10 = self.rgb_encoder_layer10(rgb_feature9, geo_s6, geo_s6)  # [1, 512, 15, 20]

        feature = []

        feature.append(rgb_feature2)
        feature.append(rgb_feature4)
        feature.append(rgb_feature6)
        feature.append(rgb_feature8)
        feature.append(rgb_feature10)
        return feature
