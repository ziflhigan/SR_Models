"""
Pixel-wise loss functions for super-resolution.
"""

from typing import Dict, Any

import torch
import torch.nn as nn

from ..utils import get_logger

logger = get_logger()


class PixelLoss(nn.Module):
    """
    Pixel-wise loss with support for MSE, L1, and weighted combinations.
    """

    def __init__(
            self,
            loss_type: str = "mse",
            reduction: str = "mean",
            weight: float = 1.0
    ):
        """
        Initialize pixel loss.

        Args:
            loss_type: Type of loss ('mse', 'l1', or 'combined')
            reduction: Reduction method ('mean', 'sum', 'none')
            weight: Weight for the loss
        """
        super(PixelLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.weight = weight

        if self.loss_type == "mse":
            self.criterion = nn.MSELoss(reduction=reduction)
        elif self.loss_type == "l1":
            self.criterion = nn.L1Loss(reduction=reduction)
        elif self.loss_type == "combined":
            self.mse_criterion = nn.MSELoss(reduction=reduction)
            self.l1_criterion = nn.L1Loss(reduction=reduction)
            self.mse_weight = 0.7
            self.l1_weight = 0.3
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        logger.debug(f"PixelLoss initialized: type={loss_type}, weight={weight}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel-wise loss.

        Args:
            pred: Predicted images (batch, channels, height, width)
            target: Target images (batch, channels, height, width)

        Returns:
            Computed loss
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        if self.loss_type == "combined":
            mse_loss = self.mse_criterion(pred, target)
            l1_loss = self.l1_criterion(pred, target)
            loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
        else:
            loss = self.criterion(pred, target)

        return self.weight * loss


class MSELoss(PixelLoss):
    """Mean Squared Error loss."""

    def __init__(self, reduction: str = "mean", weight: float = 1.0):
        super(MSELoss, self).__init__(
            loss_type="mse",
            reduction=reduction,
            weight=weight
        )


class L1Loss(PixelLoss):
    """L1 (Mean Absolute Error) loss."""

    def __init__(self, reduction: str = "mean", weight: float = 1.0):
        super(L1Loss, self).__init__(
            loss_type="l1",
            reduction=reduction,
            weight=weight
        )


def create_pixel_loss_from_config(config: Dict[str, Any]) -> PixelLoss:
    """
    Create pixel loss from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        PixelLoss instance
    """
    # For SRCNN
    if config.get('model', {}).get('name') == 'srcnn':
        loss_type = config.get('srcnn', {}).get('loss_type', 'mse')
        return PixelLoss(loss_type=loss_type)

    # For SRGAN content loss component
    elif config.get('model', {}).get('name') == 'srgan':
        content_loss = config.get('srgan', {}).get('loss', {}).get('content_loss', 'mse')
        if content_loss in ['mse', 'l1']:
            return PixelLoss(loss_type=content_loss)

    # Default
    return MSELoss()
