import torch
import torch.nn as nn

# REF: https://github.com/joeylitalien/noise2noise-pytorch
# REF: Noise2Noise paper: https://arxiv.org/pdf/1803.04189.pdf


class UNetSmallUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, cut_last_convblock):
        super(UNetSmallUpsample, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.in_channelsx2 = in_channels * 2
        self.cut_last_convblock = cut_last_convblock

        self._block1 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self._block3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self._block4 = nn.Sequential(
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self._block5 = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self._block6 = nn.Sequential(
            nn.Conv2d(self.in_channelsx2 + out_channels, 64, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, (3, 3), padding=1),
        )

    def forward(self, x):
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2_1(pool2)

        upsample3 = self._block3(pool3)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block4(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        return self._block6(concat1)


class UNetSmall(nn.Module):
    def __init__(self, in_channels, out_channels, cut_last_convblock):
        super(UNetSmall, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.in_channelsx2 = in_channels * 2
        self.cut_last_convblock = cut_last_convblock

        self._block1 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self._block3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.in_channels, self.in_channels, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        )

        self._block4 = nn.Sequential(
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.in_channelsx2, self.in_channelsx2, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        )

        self._block5 = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.in_channelsx2, self.in_channelsx2, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        )

        self._block6 = nn.Sequential(
            nn.Conv2d(self.in_channelsx2 + out_channels, 64, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, (3, 3), padding=1),
        )

    def forward(self, x):
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2_1(pool2)

        upsample3 = self._block3(pool3)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block4(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        return self._block6(concat1)


class UNetSmallNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels, cut_last_convblock):
        super(UNetSmallNoSkip, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.in_channelsx2 = in_channels * 2
        self.cut_last_convblock = cut_last_convblock

        self._block1 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2_1 = nn.Sequential(
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self._block3 = nn.Sequential(
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.in_channelsx2, self.in_channelsx2, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        )

        self._block4 = nn.Sequential(
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.in_channelsx2, self.in_channelsx2, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        )

        self._block5 = nn.Sequential(
            nn.Conv2d(self.in_channelsx2, self.in_channelsx2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channelsx2, self.in_channels, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.in_channels, self.in_channels, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        )

        self._block6 = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, (3, 3), padding=1),
        )

    def forward(self, x):
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2_1(pool2)

        upsample3 = self._block3(pool3)
        upsample2 = self._block4(upsample3)
        upsample1 = self._block5(upsample2)

        return self._block6(upsample1)


def build_network(config):
    """
    Build netowrk according to type specified in config.
    Args:
        config: Config dictionary

    Returns: Network

    """
    if config["model"] == 'unet_small':
        net = UNetSmall(in_channels=48, out_channels=3, cut_last_convblock=False)
    elif config["model"] == 'unet_small_noskip':
        net = UNetSmallNoSkip(in_channels=48, out_channels=3, cut_last_convblock=False)
    elif config["model"] == 'unet_small_upsample':
        net = UNetSmallUpsample(in_channels=48, out_channels=3, cut_last_convblock=False)
    else:
        raise KeyError("Model specified not implemented")
    return net
