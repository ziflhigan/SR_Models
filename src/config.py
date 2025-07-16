"""
Configuration parser and validator for SR_Models.
Handles loading, validation, and access to configuration parameters.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from src.utils import get_logger

logger = get_logger()


class Config:
    """Configuration class for SR models."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self._config = self._load_config()
        self._validate_config()
        self._setup_directories()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate model name
        model_name = self.get('model.name')
        if model_name not in ['srcnn', 'srgan']:
            raise ValueError(f"Invalid model name: {model_name}. Must be 'srcnn' or 'srgan'")

        # Validate device
        device = self.get('model.device')
        if device not in ['cuda', 'cpu']:
            raise ValueError(f"Invalid device: {device}. Must be 'cuda' or 'cpu'")

        # Validate scale factor
        scale_factor = self.get('dataset.scale_factor')
        if scale_factor not in [2, 3, 4, 8]:
            raise ValueError(f"Invalid scale factor: {scale_factor}. Must be 2, 3, 4 or 8")

        # Validate dataset paths
        dataset_paths = [
            self.get('dataset.train_hr_dir'),
            self.get('dataset.train_lr_dir'),
            self.get('dataset.valid_hr_dir'),
            self.get('dataset.valid_lr_dir')
        ]

        for path in dataset_paths:
            if not os.path.exists(path):
                raise ValueError(f"Dataset path does not exist: {path}")

        # Validate normalization type
        norm_type = self.get('dataset.norm_type')
        if norm_type not in ['zero_one', 'minus_one_one']:
            raise ValueError(f"Invalid normalization type: {norm_type}")

        # Validate scheduler type
        scheduler_type = self.get('training.scheduler.type')
        if scheduler_type not in ['step', 'cosine', 'plateau', 'two_stage']:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}")

        # Model-specific validation
        if model_name == 'srcnn':
            self._validate_srcnn_config()
        else:
            self._validate_srgan_config()

    def _validate_srcnn_config(self) -> None:
        """Validate SRCNN-specific configuration."""
        loss_type = self.get('srcnn.loss_type')
        if loss_type not in ['mse', 'l1']:
            raise ValueError(f"Invalid SRCNN loss type: {loss_type}")

        kernel_sizes = self.get('srcnn.kernel_sizes')
        if len(kernel_sizes) != 3:
            raise ValueError("SRCNN must have exactly 3 kernel sizes")

        # Add a warning if the second kernel size is not a common value
        if kernel_sizes[1] not in [1, 5]:
            logger.warning(
                f"SRCNN's second kernel size is {kernel_sizes[1]}. "
                f"The original paper uses 1 or 5. Results may differ."
            )

    def _validate_srgan_config(self) -> None:
        """Validate SRGAN-specific configuration."""
        content_loss = self.get('srgan.loss.content_loss')
        if content_loss not in ['mse', 'l1', 'vgg']:
            raise ValueError(f"Invalid content loss type: {content_loss}")

        if content_loss == 'vgg':
            vgg_layer = self.get('srgan.loss.vgg_layer')
            valid_layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']
            if vgg_layer not in valid_layers:
                raise ValueError(f"Invalid VGG layer: {vgg_layer}")

    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        dirs = [
            self.get('output.checkpoint_dir'),
            self.get('output.log_dir'),
            os.path.join(self.get('output.log_dir'), 'info'),
            os.path.join(self.get('output.log_dir'), 'error'),
            self.get('output.result_dir'),
            os.path.join(self.get('output.result_dir'), 'images'),
            os.path.join(self.get('output.result_dir'), 'metrics'),
        ]

        # Add model-specific checkpoint directories
        model_name = self.get('model.name')
        dirs.append(os.path.join(self.get('output.checkpoint_dir'), model_name))

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'model.name')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output path (uses original path if not specified)
        """
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting."""
        self.set(key, value)

    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self._config, default_flow_style=False, sort_keys=False)

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.get('model.name')

    @property
    def device(self) -> str:
        """Get device."""
        return self.get('model.device')

    @property
    def scale_factor(self) -> int:
        """Get scale factor."""
        return self.get('dataset.scale_factor')

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.get('training.batch_size')

    @property
    def epochs(self) -> int:
        """Get number of epochs."""
        return self.get('training.epochs')

    @property
    def checkpoint_dir(self) -> str:
        """Get checkpoint directory for current model."""
        return os.path.join(self.get('output.checkpoint_dir'), self.model_name)

    @property
    def dict(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary."""
        return self._config


# Convenience function for quick config loading
def load_config(config_path: str = "config.yaml") -> Config:
    """Load and return configuration object."""
    return Config(config_path)
