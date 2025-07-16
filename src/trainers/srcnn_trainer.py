"""
SRCNN trainer implementation.
"""
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..config import Config
from ..losses.pixel_loss import create_pixel_loss_from_config
from ..models.srcnn import create_srcnn_from_config
from ..utils.logger import get_logger
from ..utils.metrics import calculate_psnr, calculate_ssim

logger = get_logger()


class SRCNNTrainer(BaseTrainer):
    """Trainer class for SRCNN model."""

    def __init__(self, config: Config, train_loader, valid_loader=None):
        super().__init__(config, train_loader, valid_loader)
        self.scale_factor = config.get('dataset.scale_factor', 4)
        self.padding_mode = self.config.get('srcnn.padding_mode', 'same')

    def build_model(self) -> nn.Module:
        return create_srcnn_from_config(self.config)

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        lr = self.config.get('training.learning_rate.generator', 1e-4)
        return torch.optim.Adam(model.parameters(), lr=lr)

    def build_loss_function(self):
        return create_pixel_loss_from_config(self.config)

    def get_sr_image(self, lr_batch: torch.Tensor) -> torch.Tensor:
        """Generate SR image for a given LR batch for SRCNN."""
        lr_upsampled = functional.interpolate(lr_batch, scale_factor=self.scale_factor, mode='bicubic')
        return self.model(lr_upsampled)

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

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    sr_images = self.model(lr_upsampled)
                    # If using 'valid' padding, crop the HR target to match the SR output size
                    if self.padding_mode == 'valid':
                        crop_size = self.model.border_crop_size
                        if crop_size > 0:
                            hr_images = hr_images[..., crop_size:-crop_size, crop_size:-crop_size]

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
        return self._run_epoch(self.valid_loader, is_train=False)

    def resume_from_checkpoint(self, path: Path):
        logger.info(f"Resuming SRCNN training from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        logger.info(f"Successfully resumed from epoch {self.current_epoch}.")
