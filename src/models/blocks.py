"""
Shared building blocks for super-resolution models.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..utils import get_logger

logger = get_logger()


def get_activation(activation: str, num_parameters: int = 1) -> nn.Module:
    """
    Get an activation layer by its name.

    Args:
        activation: Name of the activation function ('relu', 'prelu', 'leaky_relu', 'none').
        num_parameters: Number of parameters for PReLU (1 for shared, >1 for channel-wise).

    Returns:
        An nn.Module activation layer.
    """
    if activation.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif activation.lower() == 'prelu':
        return nn.PReLU(num_parameters=num_parameters)
    elif activation.lower() == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation.lower() == 'none':
        return nn.Identity()
    else:
        logger.warning(f"Unknown activation '{activation}', using ReLU.")
        return nn.ReLU(inplace=True)


def initialize_weights(modules):
    """
    Initialize network weights using Kaiming normal distribution.

    Args:
        modules: An iterable of nn.Module objects.
    """
    for module in modules:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.
    Used in SRGAN generator architecture.
    """

    def __init__(
            self,
            channels: int = 64,
            kernel_size: int = 3,
            use_batch_norm: bool = True,
            activation: str = 'prelu'
    ):
        super(ResidualBlock, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=not use_batch_norm)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(channels)
        self.activation = get_activation(activation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=not use_batch_norm)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(channels)

        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        return identity + out


class PixelShuffleBlock(nn.Module):
    """Pixel shuffle upsampling block for super-resolution."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            upscale_factor: int = 2,
            kernel_size: int = 3,
            activation: str = 'prelu'
    ):
        super(PixelShuffleBlock, self).__init__()
        intermediate_channels = out_channels * (upscale_factor ** 2)
        self.conv = nn.Conv2d(in_channels, intermediate_channels, kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.activation = get_activation(activation)
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.activation(out)
        return out


class ConvBNReLUBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            use_batch_norm: bool = True,
            activation: str = 'relu',
            bias: Optional[bool] = None
    ):
        super(ConvBNReLUBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        if padding is None:
            padding = kernel_size // 2
        if bias is None:
            bias = not use_batch_norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_batch_norm:
            out = self.bn(out)
        out = self.activation(out)
        return out
