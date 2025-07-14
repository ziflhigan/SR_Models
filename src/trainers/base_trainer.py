"""
Abstract base trainer class for super-resolution models.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..utils.checkpoint import save_checkpoint
from ..utils.logger import get_logger
from ..utils.metrics import MetricTracker

logger = get_logger()


class EarlyStopping:
    """Early stopping utility class."""

    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = 'max'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like PSNR, 'min' for losses
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        logger.debug(f"EarlyStopping initialized: patience={patience}, min_delta={min_delta}, mode={mode}")

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")

        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


class BaseTrainer(ABC):
    """Abstract base trainer class."""

    def __init__(
            self,
            config: Dict[str, Any],
            train_loader: DataLoader,
            valid_loader: Optional[DataLoader] = None
    ):
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Training parameters
        self.epochs = config.get('training.epochs', 100)
        self.device = torch.device(config.get('model.device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.use_amp = config.get('training.use_amp', False) and self.device.type == 'cuda'

        # Init tracking and components
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.train_metrics = MetricTracker()
        self.valid_metrics = MetricTracker()
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(self.model)
        self.loss_function = self.build_loss_function()
        self.scheduler = self.build_scheduler()

        # Move model to device
        self.model.to(self.device)

        # Checkpoint and validation config
        self.checkpoint_dir = Path(config.get('output.checkpoint_dir'))
        self.valid_interval = config.get('validation.interval', 1)
        self.save_interval = config.get('training.checkpoint.save_interval', 10)
        self.metric_name = config.get('training.checkpoint.metric', 'psnr')

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using Automatic Mixed Precision (AMP).")

        logger.info(f"BaseTrainer initialized for model '{self.config.get('model.name')}' on device '{self.device}'.")

    @abstractmethod
    def build_model(self) -> nn.Module:
        pass

    @abstractmethod
    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def build_loss_function(self):
        pass

    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        pass

    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info(f"Starting training for {self.config.get('model.name').upper()} model...")

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {total_params:,}")

        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_metrics = self.train_epoch()
            valid_metrics = self.validate_epoch() if self.valid_loader and epoch % self.valid_interval == 0 else {}

            self.update_scheduler(valid_metrics)

            epoch_time = time.time() - epoch_start_time
            self.log_epoch_results(epoch, train_metrics, valid_metrics, epoch_time)

            self.handle_checkpoints(epoch, valid_metrics)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 3600:.2f} hours.")
        logger.info("=" * 80)

    def build_scheduler(self) -> None | StepLR | CosineAnnealingLR | ReduceLROnPlateau:
        """Build learning rate scheduler."""
        scheduler_config = self.config.get('training.scheduler', {})
        if not scheduler_config.get('enabled', True):
            return None

        scheduler_type = scheduler_config.get('type', 'step')

        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
            logger.debug(f"StepLR scheduler: step_size={step_size}, gamma={gamma}")

        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', self.epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
            logger.debug(f"CosineAnnealingLR scheduler: T_max={T_max}")

        elif scheduler_type == 'plateau':
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=patience, factor=factor
            )
            logger.debug(f"ReduceLROnPlateau scheduler: patience={patience}, factor={factor}")

        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}")
            return None

        return scheduler

    def update_scheduler(self, valid_metrics: Dict[str, float]):
        if not self.scheduler:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            metric_val = valid_metrics.get(self.metric_name, 0)
            self.scheduler.step(metric_val)
        else:
            self.scheduler.step()

    def log_epoch_results(self, epoch, train_metrics, valid_metrics, epoch_time):
        train_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        logger.info(f"Epoch {epoch}/{self.epochs} | Train: [{train_str}]")
        if valid_metrics:
            valid_str = ", ".join([f"{k}: {v:.4f}" for k, v in valid_metrics.items()])
            logger.info(f"               | Valid: [{valid_str}] | Time: {epoch_time:.2f}s")

    def handle_checkpoints(self, epoch: int, valid_metrics: Dict[str, float]):
        if valid_metrics:
            current_metric = valid_metrics.get(self.metric_name, float('-inf'))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                logger.info(f"---> New best {self.metric_name}: {self.best_metric:.4f} at epoch {epoch}")
                # Save best model checkpoint
                self._save_checkpoint(f"best_model.pth")

        # Save periodic checkpoint
        if epoch % self.save_interval == 0:
            self._save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

    def _save_checkpoint(self, filename: str):
        # Default implementation for simple models like SRCNN
        model_dir = self.checkpoint_dir / self.config.get('model.name')
        save_checkpoint(
            path=model_dir / filename,
            epoch=self.current_epoch,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
        )
