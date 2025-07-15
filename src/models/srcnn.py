"""
SRCNN (Super-Resolution Convolutional Neural Network) implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, cast

from ..config import Config
from ..utils import get_logger

logger = get_logger()


class SRCNN(nn.Module):
    """
    SRCNN: Super-Resolution Convolutional Neural Network.

    A lightweight 3-layer CNN for single image super-resolution.
    The network performs patch extraction, non-linear mapping, and reconstruction.
    """

    def __init__(
            self,
            num_channels: int = 3,
            f1: int = 64,  # Number of filters in first layer
            f2: int = 32,  # Number of filters in second layer
            kernel_sizes: Optional[Tuple[int, int, int]] = None,
            scale_factor: int = 4,
            init_type: str = "kaiming",
            padding_mode: str = "same"
    ):
        """
        Initialize SRCNN model.

        Args:
            num_channels: Number of input/output channels (3 for RGB, 1 for grayscale)
            f1: Number of filters in first layer (patch extraction)
            f2: Number of filters in second layer (non-linear mapping)
            kernel_sizes: Tuple of kernel sizes for (layer1, layer2, layer3)
            scale_factor: Super-resolution scale factor (for logging purposes)
        """
        super(SRCNN, self).__init__()

        self.num_channels = num_channels
        self.f1 = f1
        self.f2 = f2
        self.scale_factor = scale_factor
        self.init_type = init_type

        self.padding_mode = padding_mode
        self.border_crop_size = 0

        # Default kernel sizes: (9, 1, 5) as in original paper
        # Alternative: (9, 5, 5) for better performance
        if kernel_sizes is None:
            kernel_sizes = (9, 1, 5)

        self.kernel_sizes = kernel_sizes

        logger.info(f"Initializing SRCNN model:")
        logger.info(f"  - Input/Output channels: {num_channels}")
        logger.info(f"  - Architecture: {num_channels}-{f1}-{f2}-{num_channels}")
        logger.info(f"  - Kernel sizes: {kernel_sizes}")
        logger.info(f"  - Scale factor: {scale_factor}x")
        logger.info(f"  - Padding mode: {padding_mode}")

        # Define padding based on the mode
        k1, k2, k3 = kernel_sizes
        if padding_mode == 'same':
            p1, p2, p3 = k1 // 2, k2 // 2, k3 // 2
        elif padding_mode == 'valid':
            p1, p2, p3 = 0, 0, 0
            # Calculate the total border pixels removed by all conv layers
            total_border = (k1 - 1) + (k2 - 1) + (k3 - 1)
            # We need to crop half of this from each side (top, bottom, left, right)
            self.border_crop_size = total_border // 2
            logger.info(
                f"  - Valid padding enabled. Labels will be cropped by {self.border_crop_size} pixels on each side.")
        else:
            raise ValueError(f"Invalid padding_mode: {padding_mode}. Choose 'same' or 'valid'.")

        # Layer 1: Patch extraction and representation
        # Extracts overlapping patches and represents them as feature vectors
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=f1,
            kernel_size=kernel_sizes[0],
            padding=p1
        )

        # Layer 2: Non-linear mapping
        # Maps feature vectors to another high-dimensional space
        self.conv2 = nn.Conv2d(
            in_channels=f1,
            out_channels=f2,
            kernel_size=kernel_sizes[1],
            padding=p2
        )

        # Layer 3: Reconstruction
        # Aggregates features to produce the final super-resolved image
        self.conv3 = nn.Conv2d(
            in_channels=f2,
            out_channels=num_channels,
            kernel_size=kernel_sizes[2],
            padding=p3
        )

        # Activation function
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        self._initialize_weights()

        # Log model information
        total_params = self.count_parameters()
        trainable_params = self.count_trainable_parameters()
        logger.info(f"SRCNN model created with {total_params:,} total parameters "
                    f"({trainable_params:,} trainable)")

        # Log layer information
        self._log_layer_info()

    def _initialize_weights(self):
        """Initialize network weights based on the specified init_type."""
        logger.debug(f"Initializing SRCNN weights using '{self.init_type}' method...")

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if self.init_type == 'kaiming':
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif self.init_type == 'normal':
                    # As per original paper: N(0, 0.001)
                    nn.init.normal_(module.weight, mean=0.0, std=0.001)
                else:
                    raise ValueError(f"Unknown init_type: {self.init_type}")

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _log_layer_info(self):
        """Log detailed information about each layer."""
        logger.debug("SRCNN Layer Information:")

        # Layer 1
        layer1_params = sum(p.numel() for p in self.conv1.parameters())
        logger.debug(f"  Layer 1 (Patch Extraction): {self.num_channels}->{self.f1}, "
                     f"kernel={self.kernel_sizes[0]}, params={layer1_params:,}")

        # Layer 2
        layer2_params = sum(p.numel() for p in self.conv2.parameters())
        logger.debug(f"  Layer 2 (Non-linear Mapping): {self.f1}->{self.f2}, "
                     f"kernel={self.kernel_sizes[1]}, params={layer2_params:,}")

        # Layer 3
        layer3_params = sum(p.numel() for p in self.conv3.parameters())
        logger.debug(f"  Layer 3 (Reconstruction): {self.f2}->{self.num_channels}, "
                     f"kernel={self.kernel_sizes[2]}, params={layer3_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SRCNN.

        Note: Input x should be the low-resolution image already upscaled
        to target resolution using bicubic interpolation.

        Args:
        x: Input tensor of shape (batch, channels, height, width). This
           should be the low-resolution image already upscaled to the
           target resolution using an interpolator like bicubic.

        Returns:
        Super-resolved image tensor of same shape as input.

        Note on Padding:
        This implementation uses 'same' padding to ensure the output
        dimensions match the input dimensions. The original SRCNN paper
        did not use padding, resulting in a smaller output that was
        compared against a centrally cropped ground-truth label. This
        design choice simplifies the pipeline at the cost of a minor
        deviation from the original paper.
        """
        # Validate input
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")

        if x.size(1) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} input channels, "
                             f"got {x.size(1)}")

        # Stage 1: Patch extraction and representation
        # Extract overlapping patches and represent as 64-dimensional features
        x1 = self.relu(self.conv1(x))

        # Stage 2: Non-linear mapping
        # Map 64-dimensional features to 32-dimensional features
        x2 = self.relu(self.conv2(x1))

        # Stage 3: Reconstruction
        # Aggregate features to produce the final image (no activation)
        out = self.conv3(x2)

        return out

    def count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': 'SRCNN',
            'num_channels': self.num_channels,
            'architecture': f"{self.num_channels}-{self.f1}-{self.f2}-{self.num_channels}",
            'kernel_sizes': self.kernel_sizes,
            'scale_factor': self.scale_factor,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': self.count_trainable_parameters(),
            'layer_info': {
                'conv1': {
                    'in_channels': self.num_channels,
                    'out_channels': self.f1,
                    'kernel_size': self.kernel_sizes[0],
                    'parameters': sum(p.numel() for p in self.conv1.parameters())
                },
                'conv2': {
                    'in_channels': self.f1,
                    'out_channels': self.f2,
                    'kernel_size': self.kernel_sizes[1],
                    'parameters': sum(p.numel() for p in self.conv2.parameters())
                },
                'conv3': {
                    'in_channels': self.f2,
                    'out_channels': self.num_channels,
                    'kernel_size': self.kernel_sizes[2],
                    'parameters': sum(p.numel() for p in self.conv3.parameters())
                }
            }
        }

    def estimate_memory_usage(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Estimate memory usage for given input shape.

        Args:
            input_shape: (batch_size, channels, height, width)

        Returns:
            Dictionary with memory estimates in MB
        """
        batch_size, channels, height, width = input_shape

        # Bytes per element (float32)
        bytes_per_element = 4

        # Input memory
        input_memory = batch_size * channels * height * width * bytes_per_element

        # Feature map memory (after conv1)
        conv1_memory = batch_size * self.f1 * height * width * bytes_per_element

        # Feature map memory (after conv2)
        conv2_memory = batch_size * self.f2 * height * width * bytes_per_element

        # Output memory
        output_memory = batch_size * self.num_channels * height * width * bytes_per_element

        # Parameter memory
        param_memory = self.count_parameters() * bytes_per_element

        total_memory = input_memory + conv1_memory + conv2_memory + output_memory + param_memory

        return {
            'input_mb': input_memory / (1024 ** 2),
            'conv1_features_mb': conv1_memory / (1024 ** 2),
            'conv2_features_mb': conv2_memory / (1024 ** 2),
            'output_mb': output_memory / (1024 ** 2),
            'parameters_mb': param_memory / (1024 ** 2),
            'total_mb': total_memory / (1024 ** 2)
        }

    def freeze_layers(self, layer_names: list):
        """
        Freeze specific layers for transfer learning.

        Args:
            layer_names: List of layer names to freeze ('conv1', 'conv2', 'conv3')
        """
        logger.info(f"Freezing SRCNN layers: {layer_names}")

        for layer_name in layer_names:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                for param in layer.parameters():
                    param.requires_grad = False
                logger.debug(f"Frozen layer: {layer_name}")
            else:
                logger.warning(f"Layer {layer_name} not found in model")

    def unfreeze_layers(self, layer_names: list):
        """
        Unfreeze specific layers.

        Args:
            layer_names: List of layer names to unfreeze ('conv1', 'conv2', 'conv3')
        """
        logger.info(f"Unfreezing SRCNN layers: {layer_names}")

        for layer_name in layer_names:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
                logger.debug(f"Unfrozen layer: {layer_name}")
            else:
                logger.warning(f"Layer {layer_name} not found in model")


def create_srcnn_from_config(config: Config) -> SRCNN:
    """
    Create SRCNN model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        SRCNN model instance
    """
    config.get('model', {})
    dataset_config = config.get('dataset', {})
    srcnn_config = config.get('srcnn', {})
    init_type = srcnn_config.get('init_type', 'kaiming')
    padding_mode = srcnn_config.get('padding_mode', 'same')

    logger.info("Creating SRCNN model from configuration...")

    # Extract parameters
    num_channels = srcnn_config.get('num_channels', 3)
    f1 = srcnn_config.get('f1', 64)
    f2 = srcnn_config.get('f2', 32)
    raw = srcnn_config.get("kernel_sizes", [9, 1, 5])
    kernel_sizes = cast(Tuple[int, int, int], tuple(raw))
    scale_factor = dataset_config.get('scale_factor', 4)

    # Create model
    model = SRCNN(
        num_channels=num_channels,
        f1=f1,
        f2=f2,
        kernel_sizes=kernel_sizes,
        scale_factor=scale_factor,
        init_type=init_type,
        padding_mode=padding_mode
    )

    logger.info("SRCNN model created successfully from configuration")

    return model


def srcnn_inference(model: SRCNN, lr_image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Perform inference with SRCNN model.

    Args:
        model: SRCNN model
        lr_image: Low-resolution image tensor
        device: Device to run inference on

    Returns:
        Super-resolved image tensor
    """
    model.eval()
    model = model.to(device)
    lr_image = lr_image.to(device)

    with torch.no_grad():
        # Note: For SRCNN, the LR image should be pre-upscale using bicubic interpolation
        # to match the target resolution before feeding to the network
        sr_image = model(lr_image)

    return sr_image
