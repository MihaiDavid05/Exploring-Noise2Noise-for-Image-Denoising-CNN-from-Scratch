import torch
import math
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
from pathlib import Path
from collections import OrderedDict
torch.set_grad_enabled(False)


class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Sequential(Conv2D(3, 16, 3, stride=2), ReLU(),
                              Conv2D(16, 16, 3, stride=2), ReLU(),
                              NearestUpsampling(), ReLU(),
                              NearestUpsampling(), Sigmoid())
        self.optimizer = SGD()
        self.criterion = MSE()
        self.bestmodel_path = Path(__file__).parent / "bestmodel.pth"

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


class Module:
    def __init__(self):
        self._modules = OrderedDict()

    def __call__(self, *input):
        self.forward(*input)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def add_module(self, name, module):
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{name} is not a Module subclass!")
        elif not isinstance(name, str):
            raise TypeError("module name should be a string")
        self._modules[name] = module

    def forward(self, x):
        for module in self:
            x = module(x)
        return x

    def backward(self, *gradwrtoutput):
        raise NotImplementedError


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self._in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class TransposeConv2d(Module):
    def __init__(self):
        super(TransposeConv2d, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class NearestUpsampling(Module):
    def __init__(self):
        super(NearestUpsampling, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, *input):
        raise NotImplementedError
        # ReLU function
        # input[input < 0] = 0
        # return input

    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        # Derivative of ReLU
        # x[x <= 0] = 0
        # x[x > 0] = 1
        # return x

    def param(self):
        return []


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class MSE(object):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class SGD(object):
    def __init__(self):
        super(SGD, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
