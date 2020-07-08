import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import torch.nn.init as nn_init


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        if self.training:
            noise = Variable(input.data.new(input.size()).normal_(std=self.sigma))
            return input + noise
        else:
            return input


class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, input):
        return self.func(input)


class WN_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Linear, self).__init__(in_features, out_features, bias=bias)
        if train_scale:
            self.weight_scale = Parameter(torch.ones(self.out_features))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_features))

        self.train_scale = train_scale
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(0, std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)

        # normalize weight matrix and linear projection
        norm_weight = self.weight * (
        weight_scale / torch.sqrt((self.weight ** 2).sum(1) + 1e-8)).unsqueeze(1)
        activation = F.linear(input, norm_weight)

        if self.init_mode == True:
            mean_act = activation.mean(0).squeeze(0)
            activation = activation - mean_act.unsqueeze(0)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(0) + 1e-8).squeeze(0)
            activation = activation * inv_stdv.unsqueeze(0)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias.expand_as(activation)

        return activation


class WN_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 train_scale=False, init_stdv=1.0):
        super(WN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))

        self.train_scale = train_scale
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [out x in x h x w]
        # for each output dimension, normalize through (in, h, w) = (1, 2, 3) dims
        norm_weight = self.weight * (weight_scale / torch.sqrt((self.weight ** 2).sum(3).sum(2).sum(1) + 1e-8))\
            .unsqueeze(1).unsqueeze(2).unsqueeze(3)
        activation = F.conv2d(input, norm_weight, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0) + 1e-8).squeeze()
            activation = activation * inv_stdv.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None, :, None, None].expand_as(activation)

        return activation


class WN_Linear_Mean_Only_BN(nn.Linear):
    """Weight norm combined with mean-only batch norm for linear layer"""
    def __init__(self, in_features, out_features, bias=True, train_scale=False, init_stdv=1.0, bn_momentum=0.001):
        super(WN_Linear_Mean_Only_BN, self).__init__(in_features, out_features, bias=bias)
        if train_scale:
            self.weight_scale = Parameter(torch.ones(self.out_features))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_features))

        self.train_scale = train_scale
        self.init_mode = False
        self.init_stdv = init_stdv

        # mean-only batch norm params
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.bn_momentum = bn_momentum

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(0, std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)
        self.running_mean.zero_()

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)

        # normalize weight matrix and linear projection
        norm_weight = self.weight * (
        weight_scale / torch.sqrt((self.weight ** 2).sum(1) + 1e-8)).unsqueeze(1)
        activation = F.linear(input, norm_weight)

        if self.init_mode == True:
            mean_act = activation.mean(0).squeeze(0)
            activation = activation - mean_act.unsqueeze(0)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(0) + 1e-8).squeeze(0)
            activation = activation * inv_stdv.unsqueeze(0)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data
        else:
            training_mean = activation.mean(0).squeeze(0)
            if self.training:
                mean = training_mean
                self.running_mean = self.running_mean * (1 - self.bn_momentum) + training_mean.data * self.bn_momentum
            else:
                mean = Variable(self.running_mean)

            activation = activation - mean.unsqueeze(0)

            if self.bias is not None:
                activation = activation + self.bias.expand_as(activation)

        return activation


class WN_Conv2d_Mean_Only_BN(nn.Conv2d):
    """Weight norm combined with mean-only batch norm for 2d ConvNet"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 train_scale=False, init_stdv=1.0, bn_momentum=0.001):
        super(WN_Conv2d_Mean_Only_BN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))

        self.train_scale = train_scale
        self.init_mode = False
        self.init_stdv = init_stdv

        # mean-only batch norm params
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.bn_momentum = bn_momentum

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)
        self.running_mean.zero_()

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [out x in x h x w]
        # for each output dimension, normalize through (in, h, w) = (1, 2, 3) dims
        norm_weight = self.weight * (weight_scale / torch.sqrt((self.weight ** 2).sum(3).sum(2).sum(1) + 1e-8))\
            .unsqueeze(1).unsqueeze(2).unsqueeze(3)
        activation = F.conv2d(input, norm_weight, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0) + 1e-8).squeeze()
            activation = activation * inv_stdv.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            training_mean = activation.mean(3).mean(2).mean(0).squeeze()
            if self.training:
                mean = training_mean
                self.running_mean = self.running_mean * (1 - self.bn_momentum) + training_mean.data * self.bn_momentum
            else:
                mean = Variable(self.running_mean)

            activation = activation - mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            if self.bias is not None:
                activation = activation + self.bias[None, :, None, None].expand_as(activation)

        return activation

