"""
Abstract base trainer class for super-resolution models.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..config import Config
from ..utils.checkpoint import save_checkpoint
from ..utils.logger import get_logger
from ..utils.metrics import MetricTracker
from ..utils.visualization import plot_training_metrics, save_sample_images

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
            config: Config,
            train_loader: DataLoader,
            valid_loader: Optional[DataLoader] = None
    ):
        self.schedulers = None
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Training parameters
        self.epochs = config.get('training.epochs', 100)
        self.device = torch.device(config.get('model.device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.use_amp = config.get('training.use_amp', False) and self.device.type == 'cuda'

        # Visualization settings from config
        self.vis_config = self.config.get('visualization', {})
        self.vis_enabled = self.vis_config.get('enabled', True)

        # Init tracking and components
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.train_metrics = MetricTracker()
        self.valid_metrics = MetricTracker()

        # Metric history tracking
        self.history = []
        self.sample_validation_batch = None

        self.model = self.build_model()
        self.optimizer = self.build_optimizer(self.model)
        self.loss_function = self.build_loss_function()
        self.scheduler = self.build_scheduler()

        # Move model to device
        self.model.to(self.device)

        # Checkpoint and validation config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.result_dir = Path(config.get('output.result_dir')) / config.model_name
        self.valid_interval = config.get('validation.interval', 1)
        self.save_interval = config.get('training.checkpoint.save_interval', 10)
        self.metric_name = config.get('training.checkpoint.metric', 'psnr')

        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
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

    @abstractmethod
    def get_sr_image(self, lr_batch: torch.Tensor) -> torch.Tensor:
        """Abstract method for specific models to generate SR images."""
        pass

    @abstractmethod
    def resume_from_checkpoint(self, path: Path):
        """Load state from a checkpoint to resume training."""
        pass

    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info(f"Starting training for {self.config.get('model.name').upper()} model...")

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {total_params:,}")

        # Get a fixed sample batch for consistent visualization
        if self.valid_loader and self.vis_enabled:
            try:
                self.sample_validation_batch = next(iter(self.valid_loader))
                logger.info("Saved a sample validation batch for visualization.")
            except StopIteration:
                logger.warning("Validation loader is empty, cannot save sample images.")
                self.sample_validation_batch = None

        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Dictionary to hold all metrics for the current epoch
            epoch_log = {"epoch": epoch}

            # Run the training epoch and add its metrics to the log
            train_metrics = self.train_epoch()
            for key, value in train_metrics.items():
                epoch_log[f'train_{key}'] = value

            # Run validation (if it's time) and add its metrics to the log
            valid_metrics = {}
            if self.valid_loader and epoch % self.valid_interval == 0:
                valid_metrics = self.validate_epoch()
                for key, value in valid_metrics.items():
                    epoch_log[f'valid_{key}'] = value

            # Append the single log dictionary for this epoch to the history list
            self.history.append(epoch_log)

            self.update_scheduler()

            epoch_time = time.time() - epoch_start_time
            self.log_epoch_results(epoch, train_metrics, valid_metrics, epoch_time)

            self.handle_checkpoints_and_visuals(epoch, valid_metrics)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 3600:.2f} hours.")

        # Save final metrics plot
        if self.vis_enabled:
            plot_path = self.result_dir / 'plots' / f"{self.config.model_name}_final_training_metrics"
            plot_training_metrics(
                self.history,
                self.config.model_name,
                str(plot_path),
                self.vis_config.get('style', 'seaborn-v0_8-darkgrid'),
                self.vis_config.get('image_format', 'png')
            )
            logger.info(
                f"Final training metrics plot saved to {plot_path}.{self.vis_config.get('image_format', 'png')}")

        logger.info("=" * 80)

    def build_scheduler(self) -> None | StepLR | CosineAnnealingLR | ReduceLROnPlateau:
        """Build learning rate scheduler."""
        scheduler_config = self.config.get('training.scheduler', {})
        if not scheduler_config.get('enabled', True):
            return None

        scheduler_type = scheduler_config.get('type', 'step')

        # Handle optimizers for both SRCNN (single) and SRGAN (multiple)
        optimizers = []
        if hasattr(self, 'optimizer'):
            optimizers.append(self.optimizer)
        if hasattr(self, 'optimizer_g'):
            optimizers.append(self.optimizer_g)
        if hasattr(self, 'optimizer_d'):
            optimizers.append(self.optimizer_d)

        # Create a list of schedulers, one for each optimizer
        self.schedulers = []

        for optimizer in optimizers:
            if scheduler_type == 'step':
                step_size = scheduler_config.get('step_size', 30)
                gamma = scheduler_config.get('gamma', 0.5)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                logger.debug(f"StepLR scheduler: step_size={step_size}, gamma={gamma}")

            elif scheduler_type == 'two_stage':
                milestone = scheduler_config.get('milestone', self.epochs // 2)
                gamma = scheduler_config.get('decay_factor', 0.1)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone], gamma=gamma)
                logger.debug(f"MultiStepLR scheduler: milestone={milestone}, gamma={gamma}")

            elif scheduler_type == 'cosine':
                t_max = scheduler_config.get('T_max', self.epochs)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=t_max
                )
                logger.debug(f"CosineAnnealingLR scheduler: T_max={t_max}")

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

            self.schedulers.append(scheduler)

        return self.schedulers[0]  # Return the main one for the base class if needed

    def update_scheduler(self):
        """
        Update learning rate schedulers.
        For ReduceLROnPlateau, it uses the validation metric.
        For others, it just steps.
        """
        if not hasattr(self, 'schedulers') or not self.schedulers:
            return

        # Check if any scheduler depends on validation metrics
        is_plateau = any(isinstance(s, ReduceLROnPlateau) for s in self.schedulers)

        # If we have a plateau scheduler, we must step it after validation
        if is_plateau:
            # Find the most recent validation metric from the history list
            validation_metric_value = 0.0
            for epoch_log in reversed(self.history):
                if f'valid_{self.metric_name}' in epoch_log:
                    validation_metric_value = epoch_log[f'valid_{self.metric_name}']
                    break  # Found the latest one, no need to look further

            if validation_metric_value > 0:  # Ensure we have a valid metric
                for scheduler in self.schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(validation_metric_value)
            return  # Stop here to not step other schedulers twice

        # For all other schedulers (StepLR, MultiStepLR), step once per epoch after training
        for scheduler in self.schedulers:
            scheduler.step()

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

    def handle_checkpoints_and_visuals(self, epoch: int, valid_metrics: Dict[str, float]):
        """Consolidated handling of checkpoints and visualizations."""
        if valid_metrics:
            current_metric = valid_metrics.get(self.metric_name, float('-inf'))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                logger.info(f"---> New best {self.metric_name}: {self.best_metric:.4f} at epoch {epoch}")
                # Save best model checkpoint
                self._save_checkpoint(f"best_model.pth")

                # NEW: Save sample images and interim plots for the new best model
                if self.vis_enabled:
                    if self.sample_validation_batch:
                        self.save_visual_results(epoch)
                    if self.vis_config.get('save_interim_plots', True):
                        plot_path = self.result_dir / 'plots' / f"{self.config.model_name}_metrics_epoch_{epoch}"
                        plot_training_metrics(
                            self.history,
                            self.config.model_name,
                            str(plot_path),
                            self.vis_config.get('style', 'seaborn-v0_8-darkgrid'),
                            self.vis_config.get('image_format', 'png')
                        )

        # Save periodic checkpoint
        if epoch % self.save_interval == 0:
            self._save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

    def save_visual_results(self, epoch: int):
        """Generates and saves visual results for the sample batch."""
        logger.debug(f"Generating visual results for epoch {epoch}...")
        self.model.eval()
        with torch.no_grad():
            lr_imgs, hr_imgs = self.sample_validation_batch
            lr_imgs = lr_imgs.to(self.device)

            # Use the model-specific method to get SR image
            sr_imgs = self.get_sr_image(lr_imgs)

            save_sample_images(
                lr_images=lr_imgs.cpu(),
                sr_images=sr_imgs.cpu(),
                hr_images=hr_imgs.cpu(),
                epoch=epoch,
                output_dir=str(self.result_dir),
                model_name=self.config.model_name,
                num_samples=self.config.get('validation.num_samples', 5)
            )

    def _save_checkpoint(self, filename: str):
        save_checkpoint(
            path=self.checkpoint_dir / filename,
            epoch=self.current_epoch,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
        )
