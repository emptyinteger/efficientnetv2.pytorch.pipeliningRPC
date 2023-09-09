import os
import threading
import time
import math
from functools import wraps

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

from torchvision.models.resnet import Bottleneck
from effnetv2part import *

num_classes = 1000
        
        
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)
        
class EffNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        
        self.cfgs = effnetv2_s()
        
        # building first layer
        self._lock = threading.Lock()
        input_channel = _make_divisible(24 * width_mult, 8)
        layers1 = [conv_3x3_bn(3, input_channel, 2)]
        layers2 = []
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                if c<140 :
                    layers1.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                else:
                    layers2.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features1 = nn.Sequential(*layers1)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792


        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        layers2.append(self.conv)
        layers2.append(self.avgpool)
        
        self.features2 = nn.Sequential(*layers2)
        
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
    

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]



class EffNetV2Shard1(EffNetV2):
    def __init__(self, device, *args, **kwargs):
        super(EffNetV2Shard1, self).__init__(
            *args, **kwargs)
        self.device = device
        self.seq = nn.Sequential(
            self.features1
        ).to(self.device)
        
    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out =  self.seq(x)
        return out.cpu()


class EffNetV2Shard2(EffNetV2):
    def __init__(self, device, *args, **kwargs):
        super(EffNetV2Shard2, self).__init__(
            *args, **kwargs)
        self.device = device
        self.seq = nn.Sequential(
            self.features2
        ).to(self.device)
        self.classifier = self.classifier.to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            preout =  self.seq(x)
            preout = preout.to(self.device)
            out = self.classifier(preout.view(preout.size(0), -1))
        return out.cpu()
    
class DistEffNetV2(nn.Module):
    def __init__(self, num_split, workers, *args, **kwargs):
        super(DistEffNetV2, self).__init__()

        self.num_split = num_split

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            EffNetV2Shard1,
            args = ("cuda:0",) + args,
            kwargs = kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            EffNetV2Shard2,
            args = ("cuda:1",) + args,
            kwargs = kwargs
        )

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.num_split, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params
    

def effnetv2_s():
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return cfgs


def effnetv2_m():
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ]
    return cfgs


def effnetv2_l():
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return cfgs


def effnetv2_xl():
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return cfgs
