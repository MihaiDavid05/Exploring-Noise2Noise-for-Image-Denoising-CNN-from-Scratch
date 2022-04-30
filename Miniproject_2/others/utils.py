import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold


class TransposeConv2d:
    pass


class NearestUpsampling:
    pass


class ReLU:
    pass


class Sigmoid:
    pass


class MSE:
    pass


class SGD:
    pass


class Sequential:
    pass


class Conv2D(object):
    torch.set_grad_enabled(False)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
