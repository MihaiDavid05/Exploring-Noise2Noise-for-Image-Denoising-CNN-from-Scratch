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
                              NearestUpsampling(16, 16, 3), ReLU(),
                              NearestUpsampling(16, 3, 3), Sigmoid())
        self.optimizer = SGD()
        self.criterion = MSELoss()
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

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
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
            raise TypeError("Module name should be a string")
        self._modules[name] = module

    def forward(self, input):
        for module in self._modules:
            output = module(input)
            input = output
        return output

    def backward(self, gradwrtoutput):
        for module in self._modules:
            gradwrtinput = module.backward(gradwrtoutput)
            gradwrtoutput = gradwrtinput
        return gradwrtinput


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', device=None):
        super(Conv2d, self).__init__()
        # Store parameters of the Module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if not isinstance(kernel_size, tuple):
            self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        if not isinstance(stride, tuple):
            self.stride = (stride, stride)
        self.padding = padding
        if not isinstance(padding, tuple):
            self.padding = (padding, padding)
        self.dilation = dilation
        if not isinstance(dilation, tuple):
            self.dilation = (dilation, dilation)
        self.groups = groups # FIXME: Never change groups
        #self.padding_mode = padding_mode
        #self.device = device
        # Initialize the weights and biases
        k = self.groups / (in_channels * kernel_size[0] * kernel_size[1])
        k = k ** .5
        self.weight = empty(self.out_channels, self.in_channels // self.groups,
                             self.kernel_size[0], self.kernel_size[1]
                            ).uniform_(-k, k)
        if bias:
            self.bias = empty(self.out_channels).uniform_(-k, k)
        else:
            self.bias = None

    def forward(self, input):
        # Get input sizes
        batch_size, in_channels, height_input, width_input = input.shape
        self.input_size = (height_input, width_input)
        if in_channels != self.in_channels:
            raise TypeError(f"Invalid input channels {in_channels} should be {self.in_channels}")

        # Unfold the input matrix such that each field of view is a column
        self.unfolded = unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

        # Apply the convolutional filter to each field of view
        wxb = self.weight.view(self.out_channels, -1).matmul(self.unfolded)
        if self.bias:
            wxb += self.bias.view(1, -1, 1)

        # Compute height and width of the output
        self.output_height = (height_input + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        self.output_width = (width_input + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

        # Reshape the output
        actual = wxb.view(batch_size, self.out_channels, self.output_height, self.output_width)
        self.batch_size = batch_size
        return actual

    def backward(self, gradwrtoutput):
        # Write backpropagation for convolutional layer
        weight_reshaped = self.weight.view(self.out_channels, -1)
        gradwrtoutput_reshaped = gradwrtoutput.transpose(0, 1).transpose(1, 2).transpose(2, 3).reshape(self.out_channels, -1)

        self.gradwrtbias = gradwrtoutput.sum(dim=(0, 2, 3)).reshape(self.out_channels, -1)
        self.gradwrtweight = gradwrtoutput_reshaped.matmul(self.unfolded.view(-1, gradwrtoutput_reshaped.shape[1]).T).reshape(self.weight.shape)

        weight_reshaped.T.matmul(gradwrtoutput_reshaped)

        gradwrtinput = []
        for batch_gradwrtoutput in gradwrtoutput:
            batch_gradwrtoutput_reshaped = batch_gradwrtoutput.view(self.out_channels, -1)
            batch_gradwrtinput_reshaped = weight_reshaped.T.matmul(batch_gradwrtoutput_reshaped)
            batch_gradwrtinput = fold(batch_gradwrtinput_reshaped, self.input_size, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            gradwrtinput.append(batch_gradwrtinput.unsqueeze(0)) # https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868/2
        gradwrtinput = cat(gradwrtinput)
        return gradwrtinput

        # FIXME: Just for one
        # gradwrtoutput_reshaped = gradwrtoutput.transpose(0, 1).transpose(1, 2).transpose(2, 3).reshape(self.out_channels, -1)
        # dX_col = weight_reshaped.T.matmul(gradwrtoutput_reshaped)
        # dx = fold(dX_col, self.input_size, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        # return dx


    def param(self):
        return []


class TransposeConv2d(Module):
    def __init__(self):
        super(TransposeConv2d, self).__init__()

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class NearestUpsampling(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(NearestUpsampling, self).__init__()

        self._in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.zero = empty(1).zero_()

    def forward(self, input):
        aux = (input > 0)
        self.gradwrtinput = aux
        #output = input.maximum(self.zero) # time
        output = input * aux
        return output

    def backward(self, gradwrtoutput):
        grad = gradwrtoutput * self.gradwrtinput
        return grad

    def param(self):
        return []


class Sigmoid(Module):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def sigmoid(self, x):
        return 1 / (1 + (-x).exp())

    def forward(self, input):
        aux = self.sigmoid(input)
        self.gradwrtinput = aux * (1 - aux)
        output = aux
        return output

    def backward(self, gradwrtoutput):
        grad = gradwrtoutput * self.gradwrtinput
        return grad

    def param(self):
        return []


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input):
        pred, target = input
        aux = (pred - target)
        self.gradwrtinput = 2.0 * aux / pred.size()[0]
        output = aux.pow(2.0).mean()
        return output

    def backward(self, gradwrtoutput):
        grad = gradwrtoutput * self.gradwrtinput
        return grad

    def param(self):
        return []


class SGD(Module):
    def __init__(self):
        super(SGD, self).__init__()

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
