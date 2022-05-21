import torch
import math
from torch import empty, cat, arange
from torch.autograd import grad
from torch.nn.functional import fold, unfold
from pathlib import Path
from collections import OrderedDict
import random
import pickle

from typing import Optional, List, Tuple, Union

torch.set_grad_enabled(False)

class Model:
    def __init__(self, lr=0.1, momentum=0.9, dampening=0, weight_decay=0,
                nesterov=True) -> None:
        self.net = Sequential(
            Conv2d(in_channels=3, out_channels=8, kernel_size=2, stride=1, bias=True),
            ReLU(),
            Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1, bias=True),
            ReLU(),
            ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=1, bias=True),
            ReLU(),
            ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=2, stride=1, bias=True),
        )
        self.optimizer = SGD(params=self.net.param(), lr=lr, momentum=momentum,
            dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        self.criterion = MSELoss()
        self.bestmodel_path = Path(__file__).parent / "bestmodel.pth"

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model (pickle format)
        with open(self.bestmodel_path, 'rb') as handle:
            best_params = pickle.load(handle)
            # The best parameters are stored in a dictionary that maps layer id
            # to a dictionary of parameters.
            for layer_id, layer_module in enumerate(list(self.net._modules.values())):
                layer_params = best_params[layer_id]
                # The parameters of interest are weight and bias
                if "weight" in layer_params:
                    layer_module.weight.copy_(layer_params["weight"])
                if "bias" in layer_params:
                    layer_module.bias.copy_(layer_params["bias"])

    def train(self, train_input, train_target, num_epochs=1) -> None:
        #: train_input: tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target: tensor of size (N, C, H, W) containing another noisy version of the same images,
        # which only differs from the input by their noise.
        #: num_epochs: the number of epochs for which the model will be trained
        mini_batch_size = 100
        nb_batches = train_input.shape[0] // mini_batch_size

        permutation = list(range(train_input.shape[0]))
        random.shuffle(permutation)
        print("Training started!")
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_start in range(0, train_input.shape[0], mini_batch_size):
                batch_end = batch_start + mini_batch_size
                batch_input = train_input[permutation[batch_start:batch_end]]
                batch_target = train_target[permutation[batch_start:batch_end]]
                
                # Forward pass
                self.optimizer.zero_grad()
                preds = self.net(batch_input)

                # Compute loss
                loss = self.criterion(preds, batch_target)

                # Perform backward pass
                gradwrtoutput = self.criterion.backward()
                self.net.backward(gradwrtoutput)

                # Perform SGD
                self.optimizer.step()

                epoch_loss += loss
            epoch_loss = epoch_loss / nb_batches
            print(f'Epoch {epoch + 1} -> train loss: {loss}\n')
        print("Training ended!")

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)
        return self.net(test_input)

    def save_model(self, path):
        # Iterate the list of the current sequential module
        param_iter = 0
        model_state = {}
        for module_id, module in enumerate(self.net._modules.values()):
            # Build a dictionary of parameters for each module
            module_params = {}
            if hasattr(module, 'weight'):
                module_params['weight'] = module.weight
            if hasattr(module, 'bias'):
                module_params['bias'] = module.bias
            model_state[module_id] = module_params
        with open(Path(__file__).parent / path, 'wb+') as handle:
            pickle.dump(model_state, handle)


class Module(object):
    """
    Base class to implement modules and losses.

    Attributes:
        _modules (OrderedDict): TODO.
    """

    def __init__(self) -> None:
        """
        Constructs all the module's attributes.
        """
        self._modules = OrderedDict()

    def __call__(self, *input: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Calls the forward step.

        Args:
            input (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): the input for
                the forward pass.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: the output of the
                module.
        """
        return self.forward(*input)

    def forward(self, *input: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Computes the forward step.

        Args:
            input (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): the input for
                the forward pass.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: the output of the
                module.

        Raises:
            NotImplementedError: if the module dosen't implement this function.
        """
        raise NotImplementedError

    def backward(self, *gradwrtoutput: Union[torch.Tensor, \
                                             Tuple[torch.Tensor, ...]]) \
        -> torch.Tensor:
        """
        Computes the backward step.

        Also, accumulates the gradient with respect to the parameters.

        Args:
            gradwrtoutput (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): the
                gradient of the loss with respect to the module's output.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: the gradient of the
                loss with respect to the module's input.

        Raises:
            NotImplementedError: if the module dosen't implement this function.
        """
        raise NotImplementedError

    def param(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Creates the list of parameteres of this module. This list should be
        empty for parameterless modules.

        Returns:
            List[Tuple(torch.Tensor, torch.Tensor)]: a list of pairs composed of
                a parameter tensor and a gradient tensor of the same size.
        """
        return []
    
    def zero_grad(self) -> None:
        """
        Set the gradients to zero.
        """
        pass


# TODO: refactor methods for gradients wrt parameters.
class Conv2d(Module):
    """
    Class to implement two dimensional convolution.

    Attributes:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (Tuple[int, int]): size of the kernel.
        stride (Tuple[int, int]): size of the stride.
        padding (Tuple[int, int]): padding on all sides.
        dilation (Tuple[int, int]): kernel elements spacing.
        bias (torch.Tensor): bias.
        gradwrtbias (torch.Tensor): gradient with respect to bias.
        weight (torch.Tensor): weight.
        gradwrtweight (torch.Tensor): gradient with respect to weight.
        input_unfolded (torch.Tensor): unfolded input.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]]=1,
                 padding: Union[int, Tuple[int, int]]=0,
                 dilation: Union[int, Tuple[int, int]]=1,
                 groups: int=1, bias: bool=True) -> None:
        """
        Constructs all the module's attributes.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): size of the kernel.
            stride (Union[int, Tuple[int, int]]): size of the stride.
                Default: 1.
            padding (Union[int, Tuple[int, int]]): padding on all sides.
                Default: 0.
            dilation (Union[int, Tuple[int, int]]): kernel elements spacing.
                Default: 1.
            groups (Union[int, Tuple[int, int]]): TODO: REMOVE. Default: 1.
            bias (bool): whether to add a bias or not. Default: True.
        """
        super(Conv2d, self).__init__()
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
        # Initialize the weight and the bias along with the respective gradients
        k = 1.0 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if bias:
            self.bias = empty(self.out_channels).uniform_(-(k ** .5), k ** .5)
            self.gradwrtbias = torch.empty(self.bias.size()).zero_()
        self.weight = empty(self.out_channels, self.in_channels,
                            self.kernel_size[0], self.kernel_size[1]
                           ).uniform_(-(k ** .5), k ** .5)
        self.gradwrtweight = torch.empty(self.weight.size()).zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward step.

        Args:
            input (torch.Tensor): the input for the forward pass.

        Returns:
            torch.Tensor: the output of the module.

        Raises:
            TypeError: if the number of input channels is invalid.
        """
        # Get input sizes
        batch_size, in_channels, in_height, in_width = input.shape
        if in_channels != self.in_channels:
            raise TypeError(f"Invalid input channels {in_channels} should be {self.in_channels}")
        # Compute height and width of the output
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] *
                      (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] *
                     (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        # Unfold the input matrix such that each field of view is a column
        self.input_unfolded = unfold(input, kernel_size=self.kernel_size,
                                     dilation=self.dilation,
                                     padding=self.padding, stride=self.stride)
        # self.input_unfolded.shape == batch_size,
        #                              self.in_channels * self.kernel_size[0] * self.kernel_size[1],
        #                              L
        # Apply the convolutional filter to each field of view
        weight_reshaped = self.weight.view(self.out_channels, -1)
        # weight_reshaped.shape == self.out_channels,
        #                          self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        output_reshaped = weight_reshaped.matmul(self.input_unfolded)
        # output_reshaped.shape == batch_size,
        #                          self.out_channels,
        #                          L
        if not self.bias is None:
            bias_reshaped = self.bias.view(1, -1, 1)
            # bias_reshaped.shape == 1, self.out_channels, 1
            output_reshaped += bias_reshaped
        output = output_reshaped.view(batch_size, self.out_channels, out_height,
                                      out_width)
        return output

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Computes the backward step.

        Also, accumulates the gradient with respect to the parameters.

        Args:
            gradwrtoutput (torch.Tensor): the gradient of the loss with respect
                to the module's output.

        Returns:
            torch.Tensor: the gradient of the loss with respect to the module's
                input.

        Raises:
            TypeError: if the number of output channels is invalid.
        """
        # Get output sizes
        batch_size, out_channels, out_height, out_width = gradwrtoutput.shape
        if out_channels != self.out_channels:
            raise TypeError(f"Invalid output channels {out_channels} should be {self.out_channels}")
        # Compute height and width of the input
        in_height = (out_height - 1) * self.stride[0] - 2 * self.padding[0] + \
                    self.dilation[0] * (self.kernel_size[0] - 1) + 1
        in_width = (out_width - 1) * self.stride[1] - 2 * self.padding[1] + \
                   self.dilation[1] * (self.kernel_size[1] - 1) + 1
        input_size = (in_height, in_width)
        # Gradient with respect to bias
        if not self.bias is None:
            gradwrtbias = gradwrtoutput.sum(dim=(0, 2, 3))
            self.gradwrtbias.copy_(gradwrtbias)
        # Gradient with respect to weight
        gradwrtoutput_transposed = gradwrtoutput.transpose(0, 1) \
                                                .transpose(1, 2) \
                                                .transpose(2, 3) # Batch last
        gradwrtoutput_reshaped = gradwrtoutput_transposed \
                                 .reshape(self.out_channels, -1)
        # gradwrtoutput_reshaped.shape == self.out_channels,
        #                                 out_height * out_width * batch_size
        input_unfolded_transposed = self.input_unfolded.transpose(2, 1) \
                                                       .transpose(1, 0)
        # input_unfolded_transposed.shape == L,
        #                                    batch_size,
        #                                    self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        input_unfolded_reshaped = input_unfolded_transposed \
                                  .reshape(gradwrtoutput_reshaped.shape[1], -1)
        # input_unfolded_reshaped.shape == out_height * out_width * batch_size
        #                                  self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        gradwrtweight_reshaped = gradwrtoutput_reshaped \
                                 .matmul(input_unfolded_reshaped)
        # gradwrtweight_reshaped.shape == self.out_channels,
        #                                 self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        gradwrtweight = gradwrtweight_reshaped.reshape(self.weight.shape)
        self.gradwrtweight.copy_(gradwrtweight)
        # Gradient with respect to input
        weight_reshaped_transposed = self.weight.view(self.out_channels, -1).T
        # weight_reshaped_transposed.shape == self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        #                                     self.out_channels
        gradwrtinput_unfolded_reshaped = weight_reshaped_transposed \
                                         .matmul(gradwrtoutput_reshaped)
        # gradwrtinput_unfolded_reshaped.shape == self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        #                                         out_height * out_width * batch_size
        gradwrtinput_unfolded_transposed = gradwrtinput_unfolded_reshaped \
                                           .reshape(self.in_channels *
                                                    self.kernel_size[0] *
                                                    self.kernel_size[1],
                                                    out_height *
                                                    out_width,
                                                    batch_size)
        gradwrtinput_unfolded = gradwrtinput_unfolded_transposed \
                                .transpose(2, 1).transpose(1, 0)
        # gradwrtinput_unfolded.shape == batch_size,
        #                                out_height * out_width
        #                                self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        gradwrtinput = fold(gradwrtinput_unfolded, input_size,
                            kernel_size=self.kernel_size,
                            dilation=self.dilation, padding=self.padding,
                            stride=self.stride)
        return gradwrtinput

    def param(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Creates the list of parameteres of this module. This list should be
        empty for parameterless modules.

        Returns:
            List[Tuple(torch.Tensor, torch.Tensor)]: a list of pairs composed of
                a parameter tensor and a gradient tensor of the same size.
        """
        if not self.bias is None:
            return [(self.weight, self.gradwrtweight),
                    (self.bias, self.gradwrtbias)]
        return [(self.weight, self.gradwrtweight)]

    def zero_grad(self) -> None:
        """
        Set the gradients to zero.
        """
        if not self.bias is None:
            self.gradwrtbias.zero_()
        self.gradwrtweight.zero_()


# TODO: refactor methods for gradients wrt parameters.
class ConvTranspose2d(Module):
    """
    Class to implement two dimensional transposed convolution.

    The forward pass is the backward pass of convolution.
    The backward pass is the forward pass of convolution.

    Attributes:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (Tuple[int, int]): size of the kernel.
        stride (Tuple[int, int]): size of the stride.
        padding (Tuple[int, int]): padding on all sides.
        dilation (Tuple[int, int]): kernel elements spacing.
        bias (torch.Tensor): bias.
        gradwrtbias (torch.Tensor): gradient with respect to bias.
        weight (torch.Tensor): weight.
        gradwrtweight (torch.Tensor): gradient with respect to weight.
        input_reshaped (torch.Tensor): reshaped input.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]]=1,
                 padding: Union[int, Tuple[int, int]]=0,
                 output_padding: int=0, groups: int=1,
                 bias: bool=True,
                 dilation: Union[int, Tuple[int, int]]=1) -> None:
        """
        Constructs all the module's attributes.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): size of the kernel.
            stride (Union[int, Tuple[int, int]]): size of the stride.
                Default: 1.
            padding (Union[int, Tuple[int, int]]): padding on all sides.
                Default: 0.
            output_padding (int): TODO: REMOVE. Default: 0
            groups (Union[int, Tuple[int, int]]): TODO: REMOVE. Default: 1.
            bias (bool): whether to add a bias or not. Default: True.
            dilation (Union[int, Tuple[int, int]]): kernel elements spacing.
                Default: 1.
        """
        super(ConvTranspose2d, self).__init__()
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
        # Initialize the weight and the bias along with the respective gradients
        k = 1.0 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if bias:
            self.bias = empty(self.out_channels).uniform_(-(k ** .5), k ** .5)
            self.gradwrtbias = torch.empty(self.bias.size()).zero_()
        self.weight = empty(self.in_channels, self.out_channels,
                             self.kernel_size[0], self.kernel_size[1]
                            ).uniform_(-(k ** .5), k ** .5)
        self.gradwrtweight = torch.empty(self.weight.size()).zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward step.

        Args:
            input (torch.Tensor): the input for the forward pass.

        Returns:
            torch.Tensor: the output of the module.

        Raises:
            TypeError: if the number of input channels is invalid.
        """
        # Get output sizes
        batch_size, in_channels, in_height, in_width = input.shape
        if in_channels != self.in_channels:
            raise TypeError(f"Invalid input channels {in_channels} should be {self.in_channels}")
        # Compute height and width of the output
        out_height = (in_height - 1) * self.stride[0] - 2 * self.padding[0] + \
                     self.dilation[0] * (self.kernel_size[0] - 1) + 1
        out_width = (in_width - 1) * self.stride[1] - 2 * self.padding[1] + \
                    self.dilation[1] * (self.kernel_size[1] - 1) + 1
        output_size = (out_height, out_width)

        input_transposed = input.transpose(0, 1).transpose(1, 2) \
                                                .transpose(2, 3) # Batch last
        self.input_reshaped = input_transposed.reshape(self.in_channels, -1)
        # self.input_reshaped.shape == self.in_channels,
        #                              in_height * in_width * batch_size
        # Gradient with respect to input
        weight_reshaped_transposed = self.weight.view(self.in_channels, -1).T
        # weight_reshaped_transposed.shape == self.out_channels * self.kernel_size[0] * self.kernel_size[1],
        #                                     self.in_channels
        output_unfolded_reshaped = weight_reshaped_transposed \
                                   .matmul(self.input_reshaped)
        # output_unfolded_reshaped.shape == self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        #                                   in_height * in_width * batch_size
        output_unfolded_transposed = output_unfolded_reshaped \
                                     .reshape(self.out_channels *
                                              self.kernel_size[0] *
                                              self.kernel_size[1],
                                              in_height * in_width, batch_size)
        output_unfolded = output_unfolded_transposed.transpose(2, 1) \
                                                    .transpose(1, 0)
        # output_unfolded.shape == batch_size,
        #                          in_height * in_width
        #                          self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        output = fold(output_unfolded, output_size,
                      kernel_size=self.kernel_size, dilation=self.dilation,
                      padding=self.padding, stride=self.stride)
        if not self.bias is None:
            bias_reshaped = self.bias.view(1, -1, 1, 1)
            # bias_reshaped.shape == 1, self.out_channels, 1, 1
            output += bias_reshaped
        return output

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Computes the backward step.

        Also, accumulates the gradient with respect to the parameters.

        Args:
            gradwrtoutput (torch.Tensor): the gradient of the loss with respect
                to the module's output.

        Returns:
            torch.Tensor: the gradient of the loss with respect to the module's
                input.

        Raises:
            TypeError: if the number of output channels is invalid.
        """
        # Get output sizes
        batch_size, out_channels, out_height, out_width = gradwrtoutput.shape
        if out_channels != self.out_channels:
            raise TypeError(f"Invalid output channels {out_channels} should be {self.out_channels}")
        # Compute height and width of the input
        in_height = (out_height + 2 * self.padding[0] - self.dilation[0] *
                     (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        in_width = (out_width + 2 * self.padding[1] - self.dilation[1] *
                    (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        # Gradient with respect to bias
        if not self.bias is None:
            gradwrtbias = gradwrtoutput.sum(dim=(0, 2, 3))
            self.gradwrtbias.copy_(gradwrtbias)
        # Gradient with respect to weight
        # TODO: Unfold the gradwrtoutput matrix such that each field of view is a column
        gradwrtoutput_unfolded = unfold(gradwrtoutput,
                                        kernel_size=self.kernel_size,
                                        dilation=self.dilation,
                                        padding=self.padding,
                                        stride=self.stride)
        # gradwrtoutput_unfolded.shape == batch_size,
        #                                 self.out_channels * self.kernel_size[0] * self.kernel_size[1],
        #                                 L
        gradwrtoutput_unfolded_transposed = gradwrtoutput_unfolded \
                                            .transpose(2, 1).transpose(1, 0)
        # gradwrtoutput_unfolded_transposed.shape == L,
        #                                            batch_size,
        #                                            self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        gradwrtoutput_unfolded_reshaped = gradwrtoutput_unfolded_transposed \
                                          .reshape(self.input_reshaped.shape[1],
                                                   -1)
        # gradwrtoutput_unfolded_reshaped.shape == in_height * in_width * batch_size
        #                                          self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        gradwrtweight_reshaped = self.input_reshaped \
                                 .matmul(gradwrtoutput_unfolded_reshaped)
        # gradwrtweight_reshaped.shape == self.in_channels
        #                                 self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        gradwrtweight = gradwrtweight_reshaped.reshape(self.weight.shape)
        self.gradwrtweight.copy_(gradwrtweight)
        # Gradient with respect to input
        # TODO: Apply the convolutional filter to each field of view
        weight_reshaped = self.weight.view(self.in_channels, -1)
        # weight_reshaped.shape == self.in_channels,
        #                          self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        gradwrtinput_reshaped = weight_reshaped.matmul(gradwrtoutput_unfolded)
        # gradwrtinput_reshaped.shape == batch_size,
        #                                self.in_channels,
        #                                L
        gradwrtinput = gradwrtinput_reshaped.view(batch_size, self.in_channels,
                                                  in_height, in_width)
        return gradwrtinput

    def param(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Creates the list of parameteres of this module. This list should be
        empty for parameterless modules.

        Returns:
            List[Tuple(torch.Tensor, torch.Tensor)]: a list of pairs composed of
                a parameter tensor and a gradient tensor of the same size.
        """
        if not self.bias is None:
            return [(self.weight, self.gradwrtweight),
                    (self.bias, self.gradwrtbias)]
        return [(self.weight, self.gradwrtweight)]

    def zero_grad(self) -> None:
        """
        Set the gradients to zero.
        """
        if not self.bias is None:
            self.gradwrtbias.zero_()
        self.gradwrtweight.zero_()


# TODO: figure out what to do with the attirubtes and how to use ConvTranspose2d
class Upsample(Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None, recompute_scale_factor=None):
        pass


class ReLU(Module):
    """
    Class to implement ReLU activation function.

    Attributes:
        aux (torch.Tensor): TODO: describe and rename the attribute.
    """

    def __init__(self) -> None:
        """
        Constructs all the module's attributes.
        """
        super(ReLU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward step.

        Args:
            input (torch.Tensor): the input for the forward pass.

        Returns:
            torch.Tensor: the output of the module.
        """
        self.aux = (input > 0.0)
        output = input * self.aux
        return output

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Computes the backward step.

        Args:
            gradwrtoutput (torch.Tensor): the gradient of the loss with respect
                to the module's output.

        Returns:
            torch.Tensor: the gradient of the loss with respect to the module's
                input.
        """
        gradwrtinput = gradwrtoutput * self.aux
        return gradwrtinput


class Sigmoid(Module):
    """
    Class to implement sigmoid activation function.

    Attributes:
        aux (torch.Tensor): TODO: describe and rename the attribute.
    """

    def __init__(self) -> None:
        """
        Constructs all the module's attributes.
        """
        super(Sigmoid, self).__init__()

    def sigmoid(self, input: torch.Tensor) -> torch.Tensor:
        """
        Utility method to compute sigmoid.

        Args:
            input (torch.Tensor): the value to compute the sigmoid for.

        Returns:
            torch.Tensor: the result of the sigmoid function.
        """
        output = 1.0 / (1.0 + (-input).exp())
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward step.

        Args:
            input (torch.Tensor): the input for the forward pass.

        Returns:
            torch.Tensor: the output of the module.
        """
        output = self.sigmoid(input)
        self.aux = output * (1 - output)
        return output

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Computes the backward step.

        Args:
            gradwrtoutput (torch.Tensor): the gradient of the loss with respect
                to the module's output.

        Returns:
            torch.Tensor: the gradient of the loss with respect to the module's
                input.
        """
        gradwrtinput = gradwrtoutput * self.aux
        return gradwrtinput


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
        for module in self._modules.values():
            output = module(input)
            input = output
        return output

    def backward(self, gradwrtoutput):
        for module in reversed(list(self._modules.values())):
            gradwrtinput = module.backward(gradwrtoutput)
            gradwrtoutput = gradwrtinput
        return gradwrtinput

    def layers(self, l):
        return list(self._modules.values())[l]
    
    def param(self):
        # Create a list of the parameters of the modules and their gradients
        params = []
        for module in self._modules.values():
            module_params = module.param()
            for module_param in module_params:
                params.append(module_param)
        return params

    def zero_grad(self):
        # Call zero_grad on all of the modules
        for module in self._modules.values():
            module.zero_grad()


class MSELoss(Module):
    """
    Class to implement mean squared error loss.

    Attributes:
        reduction (string): "none" | "mean" | "sum" reduction type.
        aux (torch.Tensor): TODO: describe and rename the attribute.
    """

    def __init__(self, reduction: str="mean") -> None:
        """
        Constructs all the module's attributes.

        Args:
            reduction (string): "none" | "mean" | "sum" reduction type.
                Default: "mean".
        """
        super(MSELoss, self).__init__()
        if not reduction in ["none", "mean", "sum"]:
            pass # TODO: do something, throw? or set to default
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) \
        -> torch.Tensor:
        """
        Computes the forward step.

        Args:
            input (torch.Tensor): the input for the forward pass.

        Returns:
            torch.Tensor: the output of the module.
        """
        difference = (input - target)
        self.aux = 2.0 * difference
        output = difference.pow(2.0)
        if self.reduction == "mean":
            self.aux = self.aux / difference.numel()
            output = output.mean()
        elif self.reduction == "sum":
            output = output.sum()
        return output

    def backward(self, gradwrtoutput: Optional[torch.Tensor]=None) \
        -> torch.Tensor:
        """
        Computes the backward step.

        Args:
            gradwrtoutput (Optional[torch.Tensor]): the gradient of the loss
                with respect to the module's output. Default: None.
        Returns:
            torch.Tensor: the gradient of the loss with respect to the module's
                input.
        """
        if gradwrtoutput is None and self.reduction != "none":
            gradwrtinput = self.aux
        else:
            gradwrtinput = gradwrtoutput * self.aux
        return gradwrtinput


class SGD(Module):
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0,
                nesterov=False, maximize=False):
        super(SGD, self).__init__()
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        # Store past values of the gradients to use momentum
        self.momentum_buffer = []

    def step(self):
        # This implementation is based on the pseudocode found below
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        for param_idx, param in enumerate(self.params):
            # The first element is the value of the param
            # the second one is its gradient
            p, g = param
            if self.weight_decay != 0:
                # g = g + self.weight_decay * p
                g.add_(self.weight_decay * p)
            if self.momentum != 0:
                if len(self.momentum_buffer) == len(self.params):
                    # We found a momentum entry for the current parameter
                    self.momentum_buffer[param_idx].mul_(self.momentum)
                    self.momentum_buffer[param_idx].add_((1 - self.dampening) * g)
                else:
                    # Store value of the first available gradient
                    self.momentum_buffer.append(g.clone())
                if self.nesterov:
                    # g = g + self.momentum * self.momentum_buffer[param_idx]
                    g.add_(self.momentum * self.momentum_buffer[param_idx])
                else:
                    # g = self.momentum_buffer[param_idx]
                    g.copy_(self.momentum_buffer[param_idx])

            if self.maximize:
                # In case the objective is to maximize the function
                # p = p + self.lr * g
                p.add_(self.lr * g)
            else:
                #p = p - self.lr * g
                p.sub_(self.lr * g)

    def param(self):
        return []
    
    def zero_grad(self):
        for param_idx, param in enumerate(self.params):
            p, g = param
            g.zero_()

