"""
SRCNN trainer implementation.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..losses.pixel_loss import create_pixel_loss_from_config
from ..models.srcnn import create_srcnn_from_config
from ..utils.logger import get_logger
from ..utils.metrics import calculate_psnr, calculate_ssim
from ..utils.visualization import save_sample_images

logger = get_logger()


class SRCNNTrainer(BaseTrainer):
    """Trainer class for SRCNN model."""

    def __init__(self, config, train_loader, valid_loader=None):
        super().__init__(config, train_loader, valid_loader)
        self.scale_factor = config.get('dataset.scale_factor', 4)

    def build_model(self) -> nn.Module:
        return create_srcnn_from_config(self.config)

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        lr = self.config.get('training.learning_rate.generator', 1e-4)
        return torch.optim.Adam(model.parameters(), lr=lr)

    def build_loss_function(self):
        return create_pixel_loss_from_config(self.config)

    def _run_epoch(self, data_loader, is_train: bool) -> Dict[str, float]:
        self.model.train() if is_train else self.model.eval()
        metrics = self.train_metrics if is_train else self.valid_metrics
        metrics.reset()

        pbar_desc = f"Epoch {self.current_epoch}/{self.epochs} [{'Train' if is_train else 'Valid'}]"

        with (torch.enable_grad() if is_train else torch.no_grad()):
            for lr_images, hr_images in tqdm(data_loader, desc=pbar_desc, leave=False):
                lr_images, hr_images = lr_images.to(self.device), hr_images.to(self.device)

                lr_upsampled = functional.interpolate(lr_images, scale_factor=self.scale_factor, mode='bicubic')

                if is_train:
                    self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    sr_images = self.model(lr_upsampled)
                    loss = self.loss_function(sr_images, hr_images)

                if is_train:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                psnr = calculate_psnr(sr_images, hr_images)
                ssim = calculate_ssim(sr_images.cpu(), hr_images.cpu())
                metrics.update(loss=loss.item(), psnr=psnr, ssim=ssim)

        return {
            'loss': metrics.get_average('loss'),
            'psnr': metrics.get_average('psnr'),
            'ssim': metrics.get_average('ssim'),
        }

    def train_epoch(self) -> Dict[str, float]:
        return self._run_epoch(self.train_loader, is_train=True)

    def validate_epoch(self) -> Dict[str, float]:
        results = self._run_epoch(self.valid_loader, is_train=False)
        # Add sample image saving during validation
        lr, hr = next(iter(self.valid_loader))
        sr = self.model(functional.interpolate(lr.to(self.device), scale_factor=self.scale_factor, mode='bicubic'))
        save_sample_images(lr, sr, hr, self.current_epoch, self.config.get('output.result_dir'), 'srcnn')
        return results
