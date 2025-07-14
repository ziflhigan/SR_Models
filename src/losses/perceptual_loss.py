"""
Perceptual loss functions using VGG features.
"""

from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torchvision import models

from ..utils import get_logger

logger = get_logger()


class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor for perceptual loss.
    """

    def __init__(
            self,
            layer_name: str = "relu5_4",
            requires_grad: bool = False
    ):
        """
        Initialize VGG feature extractor.

        Args:
            layer_name: VGG layer to extract features from
            requires_grad: Whether to compute gradients for VGG parameters
        """
        super(VGGFeatureExtractor, self).__init__()

        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True)
        self.requires_grad = requires_grad

        # Layer mapping
        layer_mapping = {
            'relu1_1': 2, 'relu1_2': 4,
            'relu2_1': 7, 'relu2_2': 9,
            'relu3_1': 12, 'relu3_2': 14, 'relu3_3': 16, 'relu3_4': 18,
            'relu4_1': 21, 'relu4_2': 23, 'relu4_3': 25, 'relu4_4': 27,
            'relu5_1': 30, 'relu5_2': 32, 'relu5_3': 34, 'relu5_4': 36
        }

        if layer_name not in layer_mapping:
            raise ValueError(f"Invalid layer name: {layer_name}. "
                             f"Available: {list(layer_mapping.keys())}")

        # Extract features up to the specified layer
        layer_idx = layer_mapping[layer_name]
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:layer_idx + 1])

        # Freeze VGG parameters if not requiring gradients
        if not requires_grad:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.layer_name = layer_name

        # VGG normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        logger.debug(f"VGGFeatureExtractor initialized: layer={layer_name}, "
                     f"requires_grad={requires_grad}")

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input for VGG (ImageNet normalization).

        Args:
            x: Input tensor in range [0, 1]

        Returns:
            Normalized tensor
        """
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract VGG features.

        Args:
            x: Input tensor (batch, 3, height, width) in range [0, 1]

        Returns:
            VGG features
        """
        # Ensure input is in [0, 1] range
        if x.min() < 0 or x.max() > 1:
            x = torch.clamp(x, 0, 1)

        # Normalize for VGG
        x = self.normalize_input(x)

        # Extract features
        features = self.feature_extractor(x)
        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    """

    def __init__(
            self,
            layer_name: str = "relu5_4",
            criterion: str = "mse",
            weight: float = 1.0,
            reduction: str = "mean"
    ):
        """
        Initialize perceptual loss.

        Args:
            layer_name: VGG layer for feature extraction
            criterion: Loss criterion ('mse' or 'l1')
            weight: Loss weight
            reduction: Reduction method
        """
        super(PerceptualLoss, self).__init__()

        self.feature_extractor = VGGFeatureExtractor(
            layer_name=layer_name,
            requires_grad=False
        )

        if criterion.lower() == "mse":
            self.criterion = nn.MSELoss(reduction=reduction)
        elif criterion.lower() == "l1":
            self.criterion = nn.L1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")

        self.weight = weight
        self.layer_name = layer_name

        logger.debug(f"PerceptualLoss initialized: layer={layer_name}, "
                     f"criterion={criterion}, weight={weight}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted images (batch, 3, height, width)
            target: Target images (batch, 3, height, width)

        Returns:
            Perceptual loss
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        if pred.size(1) != 3:
            raise ValueError(f"Expected 3 channels, got {pred.size(1)}")

        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        # Compute loss
        loss = self.criterion(pred_features, target_features)
        return self.weight * loss


class VGGLoss(PerceptualLoss):
    """Alias for PerceptualLoss for backward compatibility."""
    pass


class MultiLayerPerceptualLoss(nn.Module):
    """
    Perceptual loss using multiple VGG layers.
    """

    def __init__(
            self,
            layer_names=None,
            layer_weights: Optional[List[float]] = None,
            criterion: str = "mse",
            weight: float = 1.0
    ):
        """
        Initialize multi-layer perceptual loss.

        Args:
            layer_names: List of VGG layer names
            layer_weights: Weights for each layer (None for equal weights)
            criterion: Loss criterion
            weight: Overall loss weight
        """
        super(MultiLayerPerceptualLoss, self).__init__()

        if layer_names is None:
            layer_names = ["relu2_2", "relu3_4", "relu5_4"]
        self.layer_names = layer_names

        if layer_weights is None:
            self.layer_weights = [1.0 / len(layer_names)] * len(layer_names)
        else:
            if len(layer_weights) != len(layer_names):
                raise ValueError("Number of weights must match number of layers")
            self.layer_weights = layer_weights

        # Create feature extractors for each layer
        self.extractors = nn.ModuleDict({
            layer_name: VGGFeatureExtractor(layer_name, requires_grad=False)
            for layer_name in layer_names
        })

        if criterion.lower() == "mse":
            self.criterion = nn.MSELoss()
        elif criterion.lower() == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")

        self.weight = weight

        logger.debug(f"MultiLayerPerceptualLoss initialized: layers={layer_names}, "
                     f"weights={self.layer_weights}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute multi-layer perceptual loss.

        Args:
            pred: Predicted images
            target: Target images

        Returns:
            Multi-layer perceptual loss
        """
        total_loss = 0.0

        for layer_name, layer_weight in zip(self.layer_names, self.layer_weights):
            extractor = self.extractors[layer_name]

            pred_features = extractor(pred)
            target_features = extractor(target)

            layer_loss = self.criterion(pred_features, target_features)
            total_loss += layer_weight * layer_loss

        return self.weight * total_loss


def create_perceptual_loss_from_config(config: Dict[str, Any]) -> Optional[PerceptualLoss]:
    """
    Create perceptual loss from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        PerceptualLoss instance or None if not using perceptual loss
    """
    if config.get('model', {}).get('name') != 'srgan':
        return None

    srgan_config = config.get('srgan', {})
    loss_config = srgan_config.get('loss', {})

    content_loss = loss_config.get('content_loss', 'mse')

    if content_loss == 'vgg':
        vgg_layer = loss_config.get('vgg_layer', 'relu5_4')
        content_weight = loss_config.get('content_weight', 1.0)

        return PerceptualLoss(
            layer_name=vgg_layer,
            criterion='mse',
            weight=content_weight
        )

    return None
