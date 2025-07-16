"""
SRGAN trainer implementation with distinct pre-training and adversarial stages.
"""
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..config import Config
from ..losses.adversarial_loss import create_adversarial_losses_from_config
from ..models.srgan_discriminator import create_srgan_discriminator_from_config
from ..models.srgan_generator import create_srgan_generator_from_config
from ..utils.checkpoint import save_checkpoint
from ..utils.logger import get_logger
from ..utils.metrics import MetricTracker, calculate_psnr, calculate_ssim

logger = get_logger()


class SRGANTrainer(BaseTrainer):
    """Trainer class for SRGAN model."""

    def __init__(self, config: Config, train_loader, valid_loader=None):
        self.optimizer_d = None
        self.optimizer_g = None
        self.discriminator = None
        self.generator = None
        self.is_pretraining = False

        self.original_checkpoint_dir = None
        self.original_result_dir = None

        ratio_config = config.get('srgan.training_ratio', {})
        self.ratio_enabled = ratio_config.get('enabled', False)
        self.g_steps = ratio_config.get('generator_steps', 1)
        self.d_steps = ratio_config.get('discriminator_steps', 1)
        self.clip_grad_config = config.get('training.gradient_clipping', {})
        if self.ratio_enabled:
            logger.info(
                f"Using training ratio: G updates every {self.g_steps} steps, D updates every {self.d_steps} steps.")

        super().__init__(config, train_loader, valid_loader)

        self.mse_loss = nn.MSELoss()
        self.g_loss_fn, self.d_loss_fn = create_adversarial_losses_from_config(self.config)
        self.g_loss_fn.to(self.device)
        self.d_loss_fn.to(self.device)

        self.pretrain_config = self.config.get('srgan.pretrain', {})

    def build_model(self) -> torch.nn.Module:
        self.generator = create_srgan_generator_from_config(self.config)
        self.discriminator = create_srgan_discriminator_from_config(self.config)
        self.discriminator.to(self.device)
        return self.generator  # Base class tracks the generator

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        lr_g = self.config.get('training.learning_rate.generator', 1e-4)
        lr_d = self.config.get('training.learning_rate.discriminator', 1e-4)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))
        return self.optimizer_g

    def build_loss_function(self):
        _, self.d_loss_fn = create_adversarial_losses_from_config(self.config)
        return self.d_loss_fn

    def train(self):
        """Orchestrate pre-training and adversarial training stages."""
        # Stage 1: Pre-training
        if self.pretrain_config.get('enabled', False):
            self.setup_pretrain_stage()
            super().train()  # Run the main training loop from BaseTrainer

        # Stage 2: Adversarial Training
        self.setup_adversarial_stage()
        super().train()

    def setup_pretrain_stage(self):
        """Set up the trainer for the pre-training phase."""
        logger.info("=" * 80)
        logger.info("--- Configuring for Generator Pre-training Stage ---")
        self.is_pretraining = True

        # Store original paths if not already stored
        if self.original_checkpoint_dir is None:
            self.original_checkpoint_dir = self.checkpoint_dir
            self.original_result_dir = self.result_dir

        # Set paths to pre-train subdirectories
        self.checkpoint_dir = self.original_checkpoint_dir / 'pre-train'
        self.result_dir = self.original_result_dir / 'pre-train'

        # Use pre-training epochs
        self.epochs = self.pretrain_config.get('epochs', 10)
        self.valid_interval = self.pretrain_config.get('validation_interval', 1)
        self.save_interval = self.pretrain_config.get('checkpoint_interval', 5)

    def setup_adversarial_stage(self):
        """Set up the trainer for the adversarial training phase."""
        logger.info("=" * 80)
        logger.info("--- Configuring for Adversarial Training Stage ---")
        self.is_pretraining = False

        # Revert to original paths
        if self.original_checkpoint_dir:
            self.checkpoint_dir = self.original_checkpoint_dir
            self.result_dir = self.original_result_dir

        # Use main training epochs and intervals
        self.epochs = self.config.get('training.epochs', 100)
        self.valid_interval = self.config.get('validation.interval', 1)
        self.save_interval = self.config.get('training.checkpoint.save_interval', 10)
        self.history.clear()  # Clear history to start plots fresh for this stage

    def train_epoch(self) -> Dict[str, float]:
        if self.is_pretraining:
            return self._pretrain_epoch()
        else:
            return self._adversarial_epoch()

    def _pretrain_epoch(self) -> Dict[str, float]:
        """Run one epoch of generator pre-training."""
        self.generator.train()
        metrics = MetricTracker()
        pbar = tqdm(self.train_loader, desc=f"Pre-train Epoch {self.current_epoch}/{self.epochs}", leave=False)

        for lr, hr in pbar:
            lr, hr = lr.to(self.device), hr.to(self.device)

            self.optimizer_g.zero_grad()
            sr = self.generator(lr)
            loss = self.mse_loss(sr, hr)
            loss.backward()

            # Conditionally clip gradients
            if self.clip_grad_config.get('enabled', False):
                max_norm = self.clip_grad_config.get('max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=max_norm)

            self.optimizer_g.step()

            psnr = calculate_psnr(sr, hr)
            ssim = calculate_ssim(sr, hr)
            metrics.update(loss=loss.item(), psnr=psnr, ssim=ssim)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'PSNR': f"{psnr:.2f}"})

        return {'loss': metrics.get_average('loss'), 'psnr': metrics.get_average('psnr'),
                'ssim': metrics.get_average('ssim')}

    def _adversarial_epoch(self) -> Dict[str, float]:
        """Run one epoch of adversarial training."""
        self.generator.train()
        self.discriminator.train()
        g_metrics, d_metrics = MetricTracker(), MetricTracker()
        pbar = tqdm(self.train_loader, desc=f"Adversarial Epoch {self.current_epoch}/{self.epochs}", leave=False)

        for i, (lr, hr) in enumerate(pbar):
            lr, hr = lr.to(self.device), hr.to(self.device)

            # Discriminator training
            # Wrap discriminator training in the ratio condition
            if not self.ratio_enabled or (i % self.d_steps == 0):
                self.optimizer_d.zero_grad()
                with torch.no_grad():
                    sr = self.generator(lr)
                real_pred = self.discriminator(hr)
                fake_pred = self.discriminator(sr.detach())
                d_loss, d_loss_dict = self.d_loss_fn(real_pred, fake_pred)
                d_loss.backward()
                self.optimizer_d.step()
                d_metrics.update(**d_loss_dict)

            # Generator training
            # Wrap generator training in the ratio condition
            if not self.ratio_enabled or (i % self.g_steps == 0):
                self.optimizer_g.zero_grad()
                sr_for_g = self.generator(lr)
                fake_pred_for_g = self.discriminator(sr_for_g)
                g_loss, g_loss_dict = self.g_loss_fn(sr_for_g, hr, fake_pred_for_g)
                g_loss.backward()

                # Conditionally clip gradients for the generator
                if self.clip_grad_config.get('enabled', False):
                    max_norm = self.clip_grad_config.get('max_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=max_norm)

                self.optimizer_g.step()
                g_metrics.update(**g_loss_dict)

            pbar.set_postfix({'G_Loss': f"{g_metrics.get_current('g_total_loss'):.4f}",
                              'D_Loss': f"{d_metrics.get_current('d_total_loss'):.4f}"})

        # Combine metrics for logging
        avg_metrics = {**g_metrics.metrics, **d_metrics.metrics}
        return {key: sum(val) / len(val) for key, val in avg_metrics.items() if val}

    def get_sr_image(self, lr_batch: torch.Tensor) -> torch.Tensor:
        """Generate SR image for a given LR batch for SRGAN."""
        return self.generator(lr_batch)

    def validate_epoch(self) -> Dict[str, float]:
        self.generator.eval()
        metrics = MetricTracker()
        with torch.no_grad():
            for lr, hr in self.valid_loader:
                lr, hr = lr.to(self.device), hr.to(self.device)
                sr = self.generator(lr)
                metrics.update(psnr=calculate_psnr(sr, hr), ssim=calculate_ssim(sr, hr))
        return {'psnr': metrics.get_average('psnr'), 'ssim': metrics.get_average('ssim')}

    def _save_checkpoint(self, filename: str):
        """Save checkpoint based on the current training stage."""
        if self.is_pretraining:
            # During pre-training, only save the generator
            save_checkpoint(
                path=self.checkpoint_dir / filename,
                epoch=self.current_epoch,
                model_state_dict=self.generator.state_dict(),
                optimizer_state_dict=self.optimizer_g.state_dict(),
                best_metric=self.best_metric
            )
        else:
            # During adversarial training, save both models and optimizers
            save_checkpoint(
                path=self.checkpoint_dir / filename,
                epoch=self.current_epoch,
                generator_state_dict=self.generator.state_dict(),
                discriminator_state_dict=self.discriminator.state_dict(),
                optimizer_g_state_dict=self.optimizer_g.state_dict(),
                optimizer_d_state_dict=self.optimizer_d.state_dict(),
                best_metric=self.best_metric
            )

    def resume_from_checkpoint(self, path: Path):
        """Resume training from a checkpoint."""
        logger.info(f"Resuming SRGAN training from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        # Check if it's an adversarial checkpoint by looking for discriminator state
        is_adversarial_ckpt = 'discriminator_state_dict' in checkpoint

        self.generator.load_state_dict(checkpoint.get('model_state_dict') or checkpoint['generator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint.get('optimizer_state_dict') or checkpoint['optimizer_g_state_dict'])

        if is_adversarial_ckpt:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            logger.info("Resuming from a full adversarial checkpoint.")
        else:
            logger.info("Resuming from a generator-only (pre-trained) checkpoint.")

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        logger.info(f"Successfully resumed from epoch {self.current_epoch}.")

    def update_scheduler(self):
        """
        Overrides the base method to handle stage-specific scheduler updates
        """
        if not self.schedulers:
            return

        # During adversarial training, use the logic from the base trainer
        if not self.is_pretraining:
            super().update_scheduler()
            return

        # During pre-training, handle schedulers specifically
        for scheduler in self.schedulers:
            # Check if the scheduler is ReduceLROnPlateau
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # It needs the validation metric to step
                validation_metric_value = 0.0
                # Find the most recent validation metric from the history list
                for epoch_log in reversed(self.history):
                    metric_key = f'valid_{self.metric_name}'
                    if metric_key in epoch_log:
                        validation_metric_value = epoch_log[metric_key]
                        break  # Found the latest one

                if validation_metric_value > 0:
                    scheduler.step(validation_metric_value)
                else:
                    logger.warning(
                        f"ReduceLROnPlateau scheduler did not step. "
                        f"No validation metric '{self.metric_name}' found in history."
                    )
            else:
                # For other schedulers like StepLR, just step normally
                scheduler.step()
