import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthRegression(nn.Module):
    def __init__(self, in_channels, dim_out=256, embedding_dim=128, norm='linear'):
        super(DepthRegression, self).__init__()
        self.norm = norm

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        range_attention_maps = self.conv3x3(x)
        regression_head = self.conv1x1(x)
        regression_head = regression_head.mean([2,3])

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps


class DepthRegression_light(nn.Module):
    def __init__(self, in_channels, dim_out=256, embedding_dim=128, norm='linear'):
        super(DepthRegression_light, self).__init__()
        self.norm = norm

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        range_attention_maps = self.conv3x3(x)
        regression_head = self.conv1x1(x)
        regression_head = regression_head.mean([2,3])

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps



class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        if concat_with is None:
            up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            f = up_x
        else:
            up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
            f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)



