"""
SRGAN Generator implementation.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple, Optional
from .blocks import ResidualBlock, PixelShuffleBlock, get_activation, initialize_weights
from ..config import Config
from ..utils import get_logger

logger = get_logger()


class SRGANGenerator(nn.Module):
    """
    SRGAN Generator network with residual blocks and sub-pixel upsampling.

    Architecture:
    1. Initial feature extraction layer
    2. Multiple residual blocks
    3. Global skip connection
    4. Upsampling layers (PixelShuffle)
    5. Final reconstruction layer
    """

    def __init__(
            self,
            num_channels: int = 3,
            num_features: int = 64,
            num_blocks: int = 16,
            scale_factor: int = 4,
            use_batch_norm: bool = True,
            activation: str = 'prelu'
    ):
        """
        Initialize SRGAN Generator.

        Args:
            num_channels: Number of input/output channels
            num_features: Number of feature channels throughout the network
            num_blocks: Number of residual blocks
            scale_factor: Super-resolution scale factor (2, 4, 8)
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('prelu', 'relu', 'leaky_relu')
        """
        super(SRGANGenerator, self).__init__()

        self.num_channels = num_channels
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.scale_factor = scale_factor
        self.use_batch_norm = use_batch_norm
        self.activation_type = activation

        logger.info(f"Initializing SRGAN Generator:")
        logger.info(f"  - Input/Output channels: {num_channels}")
        logger.info(f"  - Feature channels: {num_features}")
        logger.info(f"  - Residual blocks: {num_blocks}")
        logger.info(f"  - Scale factor: {scale_factor}x")
        logger.info(f"  - Batch normalization: {use_batch_norm}")
        logger.info(f"  - Activation: {activation}")

        # Validate scale factor
        if scale_factor not in [2, 4, 8]:
            logger.warning(f"Scale factor {scale_factor} may not work properly. "
                           "Recommended values: 2, 4, 8")

        # Calculate number of upsampling blocks needed
        self.num_upsample_blocks = int(math.log2(scale_factor))
        logger.debug(f"Number of upsampling blocks: {self.num_upsample_blocks}")

        # Initial feature extraction layer
        self.initial_conv = nn.Conv2d(
            num_channels, num_features,
            kernel_size=9, padding=4
        )
        self.initial_activation = get_activation(activation)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                channels=num_features,
                use_batch_norm=use_batch_norm,
                activation=activation
            ) for _ in range(num_blocks)
        ])

        # Post-residual convolution (before global skip connection)
        self.post_residual_conv = nn.Conv2d(
            num_features, num_features,
            kernel_size=3, padding=1
        )

        if use_batch_norm:
            self.post_residual_bn = nn.BatchNorm2d(num_features)

        # Upsampling blocks
        self.upsampling_blocks = nn.ModuleList([
            PixelShuffleBlock(
                in_channels=num_features,
                out_channels=num_features,
                upscale_factor=2,
                activation=activation
            ) for _ in range(self.num_upsample_blocks)
        ])

        # Final reconstruction layer
        self.final_conv = nn.Conv2d(
            num_features, num_channels,
            kernel_size=9, padding=4
        )

        # Initialize weights
        self._initialize_weights()

        # Log model information
        total_params = self.count_parameters()
        trainable_params = self.count_trainable_parameters()
        logger.info(f"SRGAN Generator created with {total_params:,} total parameters "
                    f"({trainable_params:,} trainable)")

        # Log detailed layer information
        self._log_layer_info()

    def _initialize_weights(self):
        """
        Initialize weights for layers defined directly in the generator.
        Submodules (ResidualBlock, PixelShuffleBlock) are self-initializing.
        """
        logger.debug("Initializing SRGAN Generator's direct layers...")

        # Initialize the direct conv and bn layers
        direct_modules = [self.initial_conv, self.post_residual_conv]
        if self.use_batch_norm:
            direct_modules.append(self.post_residual_bn)

        initialize_weights(direct_modules)

        # Special, smaller initialization for the final layer for training stability
        nn.init.xavier_normal_(self.final_conv.weight, gain=0.1)
        if self.final_conv.bias is not None:
            nn.init.constant_(self.final_conv.bias, 0)

        logger.debug("Generator direct layer weight initialization completed.")

    def _log_layer_info(self):
        """Log detailed information about each layer."""
        logger.debug("SRGAN Generator Layer Information:")

        # Initial conv
        initial_params = sum(p.numel() for p in self.initial_conv.parameters())
        logger.debug(f"  Initial conv: {self.num_channels}->{self.num_features}, "
                     f"kernel=9, params={initial_params:,}")

        # Residual blocks
        if self.residual_blocks:
            block_params = sum(p.numel() for p in self.residual_blocks[0].parameters())
            total_residual_params = sum(p.numel() for p in self.residual_blocks.parameters())
            logger.debug(f"  Residual blocks: {self.num_blocks} blocks, "
                         f"{block_params:,} params each, "
                         f"{total_residual_params:,} total")

        # Post-residual conv
        post_params = sum(p.numel() for p in self.post_residual_conv.parameters())
        if hasattr(self, 'post_residual_bn'):
            post_params += sum(p.numel() for p in self.post_residual_bn.parameters())
        logger.debug(f"  Post-residual conv: params={post_params:,}")

        # Upsampling blocks
        if self.upsampling_blocks:
            upsample_params = sum(p.numel() for p in self.upsampling_blocks.parameters())
            logger.debug(f"  Upsampling blocks: {self.num_upsample_blocks} blocks, "
                         f"{upsample_params:,} total params")

        # Final conv
        final_params = sum(p.numel() for p in self.final_conv.parameters())
        logger.debug(f"  Final conv: {self.num_features}->{self.num_channels}, "
                     f"kernel=9, params={final_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SRGAN Generator.

        Args:
            x: Input low-resolution image tensor (batch, channels, height, width)

        Returns:
            Super-resolved image tensor (batch, channels, height*scale, width*scale)
        """
        # Validate input
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")

        if x.size(1) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} input channels, "
                             f"got {x.size(1)}")

        # Initial feature extraction
        initial_features = self.initial_activation(self.initial_conv(x))

        # Pass through residual blocks
        residual_output = initial_features
        for i, residual_block in enumerate(self.residual_blocks):
            residual_output = residual_block(residual_output)

        # Post-residual convolution
        post_residual = self.post_residual_conv(residual_output)
        if self.use_batch_norm:
            post_residual = self.post_residual_bn(post_residual)

        # Global skip connection
        # Add the initial features to the output of all residual blocks
        skip_connection_output = post_residual + initial_features

        # Upsampling
        upsampled = skip_connection_output
        for i, upsample_block in enumerate(self.upsampling_blocks):
            upsampled = upsample_block(upsampled)
            logger.debug(f"After upsampling block {i + 1}: {upsampled.shape}")

        # Final reconstruction
        output = self.final_conv(upsampled)

        # Validate output size
        expected_height = x.size(2) * self.scale_factor
        expected_width = x.size(3) * self.scale_factor

        if output.size(2) != expected_height or output.size(3) != expected_width:
            logger.warning(f"Output size mismatch: expected ({expected_height}, {expected_width}), "
                           f"got ({output.size(2)}, {output.size(3)})")

        return output

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': 'SRGAN_Generator',
            'num_channels': self.num_channels,
            'num_features': self.num_features,
            'num_blocks': self.num_blocks,
            'scale_factor': self.scale_factor,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation_type,
            'num_upsample_blocks': self.num_upsample_blocks,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': self.count_trainable_parameters(),
            'layer_counts': {
                'residual_blocks': len(self.residual_blocks),
                'upsampling_blocks': len(self.upsampling_blocks)
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
        bytes_per_element = 4  # float32

        # Input memory
        input_memory = batch_size * channels * height * width * bytes_per_element

        # Feature memory (after initial conv)
        feature_memory = batch_size * self.num_features * height * width * bytes_per_element

        # Memory for each upsampling stage
        current_height, current_width = height, width
        upsample_memory = 0
        for _ in range(self.num_upsample_blocks):
            current_height *= 2
            current_width *= 2
            upsample_memory += batch_size * self.num_features * current_height * current_width * bytes_per_element

        # Output memory
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_memory = batch_size * self.num_channels * output_height * output_width * bytes_per_element

        # Parameter memory
        param_memory = self.count_parameters() * bytes_per_element

        total_memory = input_memory + feature_memory + upsample_memory + output_memory + param_memory

        return {
            'input_mb': input_memory / (1024 ** 2),
            'features_mb': feature_memory / (1024 ** 2),
            'upsampling_mb': upsample_memory / (1024 ** 2),
            'output_mb': output_memory / (1024 ** 2),
            'parameters_mb': param_memory / (1024 ** 2),
            'total_mb': total_memory / (1024 ** 2)
        }

    def freeze_feature_extractor(self):
        """Freeze the initial feature extraction layer."""
        logger.info("Freezing SRGAN Generator feature extractor")
        for param in self.initial_conv.parameters():
            param.requires_grad = False

    def freeze_residual_blocks(self, block_indices: Optional[list] = None):
        """
        Freeze specific residual blocks or all blocks.

        Args:
            block_indices: List of block indices to freeze (None for all blocks)
        """
        if block_indices is None:
            block_indices = list(range(len(self.residual_blocks)))

        logger.info(f"Freezing SRGAN Generator residual blocks: {block_indices}")

        for idx in block_indices:
            if 0 <= idx < len(self.residual_blocks):
                for param in self.residual_blocks[idx].parameters():
                    param.requires_grad = False
            else:
                logger.warning(f"Invalid block index: {idx}")

    def unfreeze_all(self):
        """Unfreeze all parameters in the generator."""
        logger.info("Unfreezing all SRGAN Generator parameters")
        for param in self.parameters():
            param.requires_grad = True


def create_srgan_generator_from_config(config: Config) -> SRGANGenerator:
    """
    Create SRGAN Generator from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        SRGANGenerator model instance
    """
    dataset_config = config.get('dataset', {})
    srgan_config = config.get('srgan', {})
    generator_config = srgan_config.get('generator', {})

    logger.info("Creating SRGAN Generator from configuration...")

    # Extract parameters
    num_channels = generator_config.get('num_channels', 3)
    num_features = generator_config.get('num_features', 64)
    num_blocks = generator_config.get('num_blocks', 16)
    scale_factor = dataset_config.get('scale_factor', 4)
    use_batch_norm = generator_config.get('use_batch_norm', True)

    # Create model
    model = SRGANGenerator(
        num_channels=num_channels,
        num_features=num_features,
        num_blocks=num_blocks,
        scale_factor=scale_factor,
        use_batch_norm=use_batch_norm
    )

    logger.info("SRGAN Generator created successfully from configuration")

    return model


def srgan_generator_inference(
        model: SRGANGenerator,
        lr_image: torch.Tensor,
        device: torch.device
) -> torch.Tensor:
    """
    Perform inference with SRGAN Generator.

    Args:
        model: SRGAN Generator model
        lr_image: Low-resolution image tensor
        device: Device to run inference on

    Returns:
        Super-resolved image tensor
    """
    model.eval()
    model = model.to(device)
    lr_image = lr_image.to(device)

    with torch.no_grad():
        sr_image = model(lr_image)

    return sr_image
