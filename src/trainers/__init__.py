"""
Training module for SR_Models.
"""

from .base_trainer import BaseTrainer
from .srcnn_trainer import SRCNNTrainer
from .srgan_trainer import SRGANTrainer

__all__ = ['BaseTrainer', 'SRCNNTrainer', 'SRGANTrainer']


def create_trainer(config, train_loader, valid_loader=None):
    """
    Factory function to create appropriate trainer based on config.

    Args:
        config: Configuration object
        train_loader: Training data loader
        valid_loader: Validation data loader (optional)

    Returns:
        Trainer instance
    """
    model_name = config.get('model.name')

    if model_name == 'srcnn':
        return SRCNNTrainer(config, train_loader, valid_loader)
    elif model_name == 'srgan':
        return SRGANTrainer(config, train_loader, valid_loader)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
