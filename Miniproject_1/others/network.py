import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear, dropout, cut_last_convblock):
        super(UNet).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.dropout = dropout
        self.cut_last_convblock = cut_last_convblock
        # TODO: make model

    def forward(self, x):
        pass


def build_network(config):
    """
    Build netowrk according to type specified in config.
    Args:
        config: Config dictionary

    Returns: Network

    """
    if config["model"] == 'unet':
        net = UNet(n_channels=None, bilinear=None, dropout=None, cut_last_convblock=None)
    else:
        raise KeyError("Model specified not implemented")
    return net
