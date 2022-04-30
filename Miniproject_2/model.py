import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
torch.set_grad_enabled(False)
from .others.utils import *


class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        pass

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model (pickle format)
        pass

    def train(self, train_input, train_target) -> None:
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images ,
        # which only differs from the input by their noise .
        pass

    def predict (self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)
        pass

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

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
