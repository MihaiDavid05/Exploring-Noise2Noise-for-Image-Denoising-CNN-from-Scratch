import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold


class Conv2D(object):
    torch.set_grad_enabled(False)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
