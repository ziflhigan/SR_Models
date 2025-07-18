"""
SRGAN Discriminator implementation.
"""

from typing import Dict, Any

import torch
import torch.nn as nn

from .blocks import ConvBNReLUBlock
from ..config import Config
from ..utils import get_logger

logger = get_logger()


class SRGANDiscriminator(nn.Module):
    """
    SRGAN Discriminator network based on the VGG architecture.

    The discriminator's job is to distinguish real high-resolution images
    from generated (super-resolved) ones. It uses a series of
    convolutional blocks to downsample the image and extract features,
    followed by a classifier.
    """

    def __init__(
            self,
            num_channels: int = 3,
            num_features: int = 64,
    ):
        """
        Initialize SRGAN Discriminator.

        Args:
            num_channels: Number of input image channels (e.g., 3 for RGB)
            num_features: Number of features in the first convolutional layer
        """
        super(SRGANDiscriminator, self).__init__()

        self.num_channels = num_channels
        self.num_features = num_features

        logger.info("Initializing SRGAN Discriminator:")
        logger.info(f"  - Input channels: {num_channels}")
        logger.info(f"  - Initial features: {num_features}")
        logger.info("  - Using AdaptiveAvgPool2d for input size independence.")

        # Feature extractor: a sequence of convolutional blocks
        self.features = nn.Sequential(
            # Input: (batch, 3, 96, 96)
            ConvBNReLUBlock(num_channels, num_features, kernel_size=3, stride=1, use_batch_norm=False,
                            activation='leaky_relu'),
            # -> (batch, 64, 96, 96)
            ConvBNReLUBlock(num_features, num_features, kernel_size=3, stride=2, use_batch_norm=True,
                            activation='leaky_relu'),
            # -> (batch, 64, 48, 48)

            ConvBNReLUBlock(num_features, num_features * 2, kernel_size=3, stride=1, use_batch_norm=True,
                            activation='leaky_relu'),
            # -> (batch, 128, 48, 48)
            ConvBNReLUBlock(num_features * 2, num_features * 2, kernel_size=3, stride=2, use_batch_norm=True,
                            activation='leaky_relu'),
            # -> (batch, 128, 24, 24)

            ConvBNReLUBlock(num_features * 2, num_features * 4, kernel_size=3, stride=1, use_batch_norm=True,
                            activation='leaky_relu'),
            # -> (batch, 256, 24, 24)
            ConvBNReLUBlock(num_features * 4, num_features * 4, kernel_size=3, stride=2, use_batch_norm=True,
                            activation='leaky_relu'),
            # -> (batch, 256, 12, 12)

            ConvBNReLUBlock(num_features * 4, num_features * 8, kernel_size=3, stride=1, use_batch_norm=True,
                            activation='leaky_relu'),
            # -> (batch, 512, 12, 12)
            ConvBNReLUBlock(num_features * 8, num_features * 8, kernel_size=3, stride=2, use_batch_norm=True,
                            activation='leaky_relu'),
            # -> (batch, 512, 6, 6)
        )

        # Use Adaptive Average Pooling to handle any input size.
        # The output of self.features is (batch, 512, H/16, W/16).
        # AdaptiveAvgPool2d(1) will reduce this to (batch, 512, 1, 1).
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

        # Initialize weights
        self._initialize_weights()

        # Log model information
        total_params = self.count_parameters()
        trainable_params = self.count_trainable_parameters()
        logger.info(f"SRGAN Discriminator created with {total_params:,} total parameters "
                    f"({trainable_params:,} trainable)")

        # Log detailed layer information
        self._log_layer_info()

    def _initialize_weights(self):
        """Initialize network weights."""
        logger.debug("Initializing SRGAN Discriminator weights...")
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        logger.debug("Weight initialization completed")

    def _log_layer_info(self):
        """Log detailed information about each layer."""
        logger.debug("SRGAN Discriminator Layer Information:")
        feature_params = sum(p.numel() for p in self.features.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        logger.debug(f"  Feature extractor params: {feature_params:,}")
        logger.debug(f"  Classifier params: {classifier_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Discriminator.

        Args:
            x: Input image tensor (real HR or fake SR) of shape
               (batch, channels, height, width).

        Returns:
            A single logit score for each image in the batch.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")

        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': 'SRGAN_Discriminator',
            'num_channels': self.num_channels,
            'num_features': self.num_features,
            'hr_crop_size': self.hr_crop_size,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': self.count_trainable_parameters(),
        }


def create_srgan_discriminator_from_config(config: Config) -> SRGANDiscriminator:
    """
    Create SRGAN Discriminator from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        SRGANDiscriminator model instance
    """
    dataset_config = config.get('dataset', {})
    srgan_config = config.get('srgan', {})
    discriminator_config = srgan_config.get('discriminator', {})

    logger.info("Creating SRGAN Discriminator from configuration...")

    # Extract parameters
    num_channels = discriminator_config.get('num_channels', 3)
    num_features = discriminator_config.get('num_features', 64)

    # Create model
    model = SRGANDiscriminator(
        num_channels=num_channels,
        num_features=num_features
    )

    logger.info("SRGAN Discriminator created successfully from configuration")

    return model
