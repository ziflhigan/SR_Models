"""
Visualization utilities for saving sample images.
"""

from pathlib import Path
import torch
import torchvision.transforms as transforms
from ..utils.logger import get_logger

logger = get_logger()


def save_sample_images(
        lr_images: torch.Tensor,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        epoch: int,
        output_dir: str,
        model_name: str,
        num_samples: int = 5
):
    """
    Save sample images for visualization.

    Args:
        lr_images: Low-resolution input images.
        sr_images: Super-resolved output images.
        hr_images: High-resolution target images.
        epoch: Current epoch number.
        output_dir: Base directory to save results.
        model_name: Name of the model ('srcnn' or 'srgan').
        num_samples: Number of samples to save from the batch.
    """
    try:
        results_dir = Path(output_dir) / 'images'
        epoch_dir = results_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        to_pil = transforms.ToPILImage()
        num_to_save = min(num_samples, lr_images.size(0))

        for i in range(num_to_save):
            # Clamp values to valid [0, 1] range before converting
            lr_pil = to_pil(torch.clamp(lr_images[i].cpu(), 0, 1))
            sr_pil = to_pil(torch.clamp(sr_images[i].cpu(), 0, 1))
            hr_pil = to_pil(torch.clamp(hr_images[i].cpu(), 0, 1))

            lr_pil.save(epoch_dir / f"sample_{i}_lr.png")
            sr_pil.save(epoch_dir / f"sample_{i}_sr_{model_name}.png")
            hr_pil.save(epoch_dir / f"sample_{i}_hr.png")

        logger.debug(f"Saved {num_to_save} {model_name.upper()} sample images to {epoch_dir}")

    except Exception as e:
        logger.warning(f"Failed to save sample images: {str(e)}")
