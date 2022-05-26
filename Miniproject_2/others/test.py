import unittest
import random
import torch
from Miniproject_2 import model


class ModelTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ModelTests, self).__init__(*args, **kwargs)
        self.rtol = 1e-03
        self.atol = 1e-06

    def _test_module(self, expected_m, actual_m, input, target=None):
        # Expected values
        with torch.set_grad_enabled(True):
            params = []
            for parameter in expected_m.parameters():
                param = parameter.clone().detach()
                params.append(param)
                parameter.requires_grad = True
            # Forward
            expected_input = input.clone().detach().requires_grad_(True)
            if not target is None:
                expected_target = target.clone().detach().requires_grad_(True)
                expected_output = expected_m(expected_input, expected_target)
            else:
                expected_output = expected_m(expected_input)
            expected_output.retain_grad()
            aggregate = expected_output.mean()
            # Backward
            aggregate.backward()
            output_grad = expected_output.grad.clone().detach()
            expected_grad = expected_input.grad.clone().detach()
            expected_param_grads = []
            for parameter in expected_m.parameters():
                expected_param_grad = parameter.grad.clone().detach()
                expected_param_grads.append(expected_param_grad)
        # Actual values
        with torch.set_grad_enabled(False):
            for parameter, (param, param_grad) in zip(params, actual_m.param()):
                param.copy_(parameter)
            # Forward
            actual_input = input.clone().detach().requires_grad_(True)
            if not target is None:
                actual_target = target.clone().detach().requires_grad_(True)
                actual_output = actual_m(actual_input, actual_target)
            else:
                actual_output = actual_m(actual_input)
            # Backward
            actual_grad = actual_m.backward(output_grad)
            actual_param_grads = []
            for param, param_grad in actual_m.param():
                actual_param_grad = param_grad
                actual_param_grads.append(actual_param_grad)
        # Compare
        self.assertTrue(torch.allclose(actual_input, expected_input,
                                       rtol=self.rtol, atol=self.atol))
        self.assertTrue(torch.allclose(actual_output, expected_output,
                                       rtol=self.rtol, atol=self.atol))
        self.assertTrue(torch.allclose(actual_grad, expected_grad,
                                       rtol=self.rtol, atol=self.atol))
        for actual_param_grad, expected_param_grad in zip(actual_param_grads,
                                                          expected_param_grads):
            self.assertTrue(torch.allclose(actual_param_grad,
                                           expected_param_grad,
                                           rtol=self.rtol, atol=self.atol))
        return True

    def test_module(self):
        expected_m = torch.nn.Module()
        actual_m = model.Module()

    def test_conv2d(self):
        # Prepare parameters
        in_channels = 4
        out_channels = 3
        kernel_size = (2, 2)
        stride = 2
        padding = 2
        dilation = 3
        hasBias = True
        # Prepare input
        input = torch.randn((10, in_channels, 32, 32))
        # Expected model
        expected_m = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation,
                                     bias=hasBias)
        # Actual model
        actual_m = model.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation,
                                bias=hasBias)
        self._test_module(expected_m, actual_m, input)

    def test_convtranspose2d(self):
        # Prepare parameters
        in_channels = 3
        out_channels = 3
        kernel_size = (2, 2)
        stride = 2
        padding = 5
        output_padding = 1
        dilation = 3
        hasBias = True
        # Prepare input
        input = torch.randn((10, in_channels, 32, 32))
        # Expected model
        expected_m = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride, padding=padding,
                                              output_padding=output_padding,
                                              bias=hasBias, dilation=dilation)
        # Actual model
        actual_m = model.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding,
                                         output_padding=output_padding,
                                         bias=hasBias, dilation=dilation)
        self._test_module(expected_m, actual_m, input)

    def test_upsample(self):
        # Prepare parameters
        in_channels = 3
        out_channels = 3
        kernel_size = (2, 2)
        stride = 2
        padding = 5
        output_padding = 1
        dilation = 3
        hasBias = True
        # Prepare input
        input = torch.randn((10, in_channels, 32, 32))
        # Expected model
        expected_m = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride, padding=padding,
                                              output_padding=output_padding,
                                              bias=hasBias, dilation=dilation)
        # Actual model
        actual_m = model.Upsample(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding,
                                  output_padding=output_padding, bias=hasBias,
                                  dilation=dilation)
        self._test_module(expected_m, actual_m, input)

    def test_relu(self):
        # Prepare parameters
        in_channels = 3
        # Prepare input
        input = torch.randn((10, in_channels, 32, 32))
        # Expected model
        expected_m = torch.nn.ReLU()
        # Actual model
        actual_m = model.ReLU()
        self._test_module(expected_m, actual_m, input)

    def test_sigmoid(self):
        # Prepare parameters
        in_channels = 3
        # Prepare input
        input = torch.randn((10, in_channels, 32, 32))
        # Expected model
        expected_m = torch.nn.Sigmoid()
        # Actual model
        actual_m = model.Sigmoid()
        self._test_module(expected_m, actual_m, input)

    def test_sequential(self):
        # Prepare parameters
        in_channels = 3
        out_channels = 4
        kernel_size = (2, 2)
        stride = 2
        padding = 2
        output_padding = 1
        dilation = 1
        hasBias = True
        # Prepare input
        input = torch.randn((10, in_channels, 32, 32))
        # Expected model
        expected_m = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation,
                                        bias=hasBias),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=out_channels,
                                        out_channels=16,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation,
                                        bias=hasBias),
                        torch.nn.ReLU(),
                        torch.nn.ConvTranspose2d(in_channels=16,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride, padding=padding,
                                                 output_padding=output_padding,
                                                 bias=hasBias,
                                                 dilation=dilation),
                        torch.nn.ReLU(),
                        torch.nn.ConvTranspose2d(in_channels=out_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride, padding=padding,
                                                 output_padding=output_padding,
                                                 bias=hasBias,
                                                 dilation=dilation),
                        torch.nn.Sigmoid())
        # Actual model
        actual_m = model.Sequential(
                        model.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation,
                                     bias=hasBias),
                        model.ReLU(),
                        model.Conv2d(in_channels=out_channels, out_channels=16,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation,
                                     bias=hasBias),
                        model.ReLU(),
                        model.Upsample(in_channels=16,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,  stride=stride,
                                       padding=padding,
                                       output_padding=output_padding,
                                       bias=hasBias, dilation=dilation),
                        model.ReLU(),
                        model.Upsample(in_channels=out_channels,
                                       out_channels=in_channels,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding,
                                       output_padding=output_padding,
                                       bias=hasBias, dilation=dilation),
                        model.Sigmoid())
        self._test_module(expected_m, actual_m, input)

    def test_mseloss(self):
        # Prepare parameters
        in_channels = 3
        reduction = "sum"
        # Prepare input
        input = torch.randn((10, in_channels, 32, 32))
        target = torch.randn((10, in_channels, 32, 32))
        # Expected model
        expected_m = torch.nn.MSELoss(reduction=reduction)
        # Actual model
        actual_m = model.MSELoss(reduction=reduction)
        self._test_module(expected_m, actual_m, input, target)

    def _test_sgd(self, expected_m, expected_o, expected_l, actual_m, actual_o,
                  actual_l, input, target, batch_size=2, num_epochs=2):
        num_batches = input.shape[0] // batch_size
        permutation = list(range(input.shape[0]))
        random.Random(0).shuffle(permutation)
        for epoch in range(num_epochs):
            expected_epoch_loss = 0
            actual_epoch_loss = 0
            for batch_start in range(0, input.shape[0], batch_size):
                batch_end = batch_start + batch_size
                batch_input = input[permutation[batch_start:batch_end]]
                batch_target = target[permutation[batch_start:batch_end]]
                # Expected values
                with torch.set_grad_enabled(True):
                    if epoch == 0:
                        params = []
                        for parameter in expected_m.parameters():
                            param = parameter.clone().detach()
                            params.append(param)
                            parameter.requires_grad = True
                    # Forward
                    expected_input = batch_input.clone().detach() \
                                                .requires_grad_(True)
                    expected_target = batch_target.clone().detach() \
                                                .requires_grad_(True)
                    expected_o.zero_grad()
                    expected_output = expected_m(expected_input)
                    expected_output.retain_grad()
                    expected_loss = expected_l(expected_output, expected_target)
                    # Backward
                    expected_loss.backward()
                    expected_o.step()
                    expected_epoch_loss += expected_loss
                    expected_grad = expected_input.grad.clone().detach()
                    expected_params = []
                    expected_param_grads = []
                    for parameter in expected_m.parameters():
                        expected_param = parameter.clone().detach()
                        expected_params.append(expected_param)
                        expected_param_grad = parameter.grad.clone().detach()
                        expected_param_grads.append(expected_param_grad)
                # Actual values
                with torch.set_grad_enabled(False):
                    if epoch == 0:
                        for parameter, (param, param_grad) in \
                            zip(params, actual_m.param()):
                            param.copy_(parameter)
                    # Forward
                    actual_input = batch_input.clone().detach() \
                                            .requires_grad_(True)
                    actual_target = batch_target.clone().detach() \
                                                .requires_grad_(True)
                    actual_o.zero_grad()
                    actual_output = actual_m(actual_input)
                    actual_loss = actual_l(actual_output, actual_target)
                    # Backward
                    actual_grad = actual_m.backward(actual_l.backward())
                    actual_o.step()
                    actual_epoch_loss += actual_loss
                    actual_params = []
                    actual_param_grads = []
                    for param, param_grad in actual_m.param():
                        actual_param = param
                        actual_params.append(actual_param)
                        actual_param_grad = param_grad
                        actual_param_grads.append(actual_param_grad)
                # Compare
                self.assertTrue(torch.allclose(actual_input, expected_input,
                                               rtol=self.rtol, atol=self.atol))
                self.assertTrue(torch.allclose(actual_output, expected_output,
                                               rtol=self.rtol, atol=self.atol))
                self.assertTrue(torch.allclose(actual_grad, expected_grad,
                                               rtol=self.rtol, atol=self.atol))
                for actual_param_grad, expected_param_grad in \
                    zip(actual_param_grads, expected_param_grads):
                    self.assertTrue(torch.allclose(actual_param_grad,
                                                   expected_param_grad,
                                                   rtol=self.rtol,
                                                   atol=self.atol))
                for actual_param, expected_param in \
                    zip(actual_params, expected_params):
                    self.assertTrue(torch.allclose(actual_param, expected_param,
                                                   rtol=self.rtol, atol=self.atol))
            expected_epoch_loss /= num_batches
            actual_epoch_loss /= num_batches
            self.assertTrue(torch.allclose(actual_epoch_loss,
                                           expected_epoch_loss,
                                           rtol=self.rtol, atol=self.atol))
        return True

    def test_sgd(self):
        # Prepare parameters
        in_channels = 3
        out_channels = 48
        kernel_size = (3, 3)
        stride = 2
        padding = 1
        output_padding = 1
        dilation = 1
        hasBias = True
        # Prepare input
        input = torch.randn((10, in_channels, 32, 32))
        target = torch.randn((10, in_channels, 32, 32))
        # Expected model
        # Expected model
        expected_m = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation,
                                        bias=hasBias),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=out_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation,
                                        bias=hasBias),
                        torch.nn.ReLU(),
                        torch.nn.ConvTranspose2d(in_channels=out_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride, padding=padding,
                                                 output_padding=output_padding,
                                                 bias=hasBias,
                                                 dilation=dilation),
                        torch.nn.ReLU(),
                        torch.nn.ConvTranspose2d(in_channels=out_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride, padding=padding,
                                                 output_padding=output_padding,
                                                 bias=hasBias,
                                                 dilation=dilation),
                        torch.nn.Sigmoid())
        expected_o = torch.optim.SGD(expected_m.parameters(), lr=0.001)
        expected_l = torch.nn.MSELoss()
        # Actual model
        actual_m = model.Sequential(
                        model.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation,
                                     bias=hasBias),
                        model.ReLU(),
                        model.Conv2d(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation,
                                     bias=hasBias),
                        model.ReLU(),
                        model.Upsample(in_channels=out_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,  stride=stride,
                                       padding=padding,
                                       output_padding=output_padding,
                                       bias=hasBias, dilation=dilation),
                        model.ReLU(),
                        model.Upsample(in_channels=out_channels,
                                       out_channels=in_channels,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding,
                                       output_padding=output_padding,
                                       bias=hasBias, dilation=dilation),
                        model.Sigmoid())
        actual_o = model.SGD(actual_m.param(), lr=0.001)
        actual_l = model.MSELoss()
        self._test_sgd(expected_m, expected_o, expected_l, actual_m, actual_o,
                       actual_l, input, target)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=0, exit=False)
