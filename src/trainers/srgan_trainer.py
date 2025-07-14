"""
SRGAN trainer implementation.
"""
from typing import Dict

import torch
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..losses.adversarial_loss import create_adversarial_losses_from_config
from ..models.srgan_discriminator import create_srgan_discriminator_from_config
from ..models.srgan_generator import create_srgan_generator_from_config
from ..utils.checkpoint import save_checkpoint
from ..utils.logger import get_logger
from ..utils.metrics import MetricTracker, calculate_psnr, calculate_ssim
from ..utils.visualization import save_sample_images

logger = get_logger()


class SRGANTrainer(BaseTrainer):
    """Trainer class for SRGAN model."""

    def __init__(self, config, train_loader, valid_loader=None):
        self.mse_loss = None
        self.d_loss_fn = None
        self.g_loss_fn = None
        self.optimizer_d = None
        self.optimizer_g = None
        self.discriminator = None
        self.generator = None
        self.pretrain_epochs = config.get('srgan.pretrain_epochs', 10)
        super().__init__(config, train_loader, valid_loader)

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

        return self.optimizer_g  # Return G's optimizer to base class

    def build_loss_function(self):
        self.g_loss_fn, self.d_loss_fn = create_adversarial_losses_from_config(self.config)
        self.mse_loss = torch.nn.MSELoss()  # For pre-training
        return self.g_loss_fn

    def train(self):
        # Optional: Pre-train generator with MSE
        if self.config.get('srgan.pretrain_generator', True):
            self.pretrain_generator()
        super().train()  # Call base class train loop for adversarial training

    def pretrain_generator(self):
        logger.info("--- Starting Generator Pre-training ---")
        for epoch in range(1, self.pretrain_epochs + 1):
            self.generator.train()
            pbar = tqdm(self.train_loader, desc=f"Pre-train {epoch}/{self.pretrain_epochs}", leave=False)
            for lr_images, hr_images in pbar:
                lr, hr = lr_images.to(self.device), hr_images.to(self.device)

                self.optimizer_g.zero_grad()
                sr = self.generator(lr)
                loss = self.mse_loss(sr, hr)
                loss.backward()
                self.optimizer_g.step()
                pbar.set_postfix({'MSE Loss': f"{loss.item():.4f}"})
        logger.info("--- Generator Pre-training Complete ---")

    def train_epoch(self) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        g_metrics = MetricTracker()
        d_metrics = MetricTracker()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Adversarial]", leave=False)
        for lr, hr in pbar:
            lr, hr = lr.to(self.device), hr.to(self.device)

            # Train Discriminator
            self.optimizer_d.zero_grad()
            sr = self.generator(lr).detach()
            real_pred = self.discriminator(hr)
            fake_pred = self.discriminator(sr)
            d_loss, d_loss_dict = self.d_loss_fn(real_pred, fake_pred)
            d_loss.backward()
            self.optimizer_d.step()
            d_metrics.update(**d_loss_dict)

            # Train Generator
            self.optimizer_g.zero_grad()
            sr_for_g = self.generator(lr)
            fake_pred_for_g = self.discriminator(sr_for_g)
            g_loss, g_loss_dict = self.g_loss_fn(sr_for_g, hr, fake_pred_for_g)
            g_loss.backward()
            self.optimizer_g.step()
            g_metrics.update(**g_loss_dict)

            pbar.set_postfix({
                'G_Loss': f"{g_loss.item():.4f}",
                'D_Loss': f"{d_loss.item():.4f}",
                'D_Acc': f"R: {d_loss_dict['d_real_accuracy']:.2f} F: {d_loss_dict['d_fake_accuracy']:.2f}"
            })

        return {**g_metrics.metrics, **d_metrics.metrics}

    def validate_epoch(self) -> Dict[str, float]:
        self.generator.eval()
        metrics = MetricTracker()
        with torch.no_grad():
            for lr, hr in self.valid_loader:
                lr, hr = lr.to(self.device), hr.to(self.device)
                sr = self.generator(lr)
                metrics.update(psnr=calculate_psnr(sr, hr), ssim=calculate_ssim(sr, hr))

        # Save sample images
        lr, hr = next(iter(self.valid_loader))
        sr = self.generator(lr.to(self.device))
        save_sample_images(lr, sr, hr, self.current_epoch, self.config.get('output.result_dir'), 'srgan')

        return {'psnr': metrics.get_average('psnr'), 'ssim': metrics.get_average('ssim')}

    def _save_checkpoint(self, filename: str):
        # Override base method to save both models and optimizers
        model_dir = self.checkpoint_dir / self.config.get('model.name')
        save_checkpoint(
            path=model_dir / filename,
            epoch=self.current_epoch,
            generator_state_dict=self.generator.state_dict(),
            discriminator_state_dict=self.discriminator.state_dict(),
            optimizer_g_state_dict=self.optimizer_g.state_dict(),
            optimizer_d_state_dict=self.optimizer_d.state_dict(),
        )
