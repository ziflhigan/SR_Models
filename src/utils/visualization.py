from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image

from ..utils.logger import get_logger
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import pandas as pd

logger = get_logger()
sns.set_theme()


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
    Save sample images for visualization when a new best model is found.
    Images are saved in a dedicated folder for that epoch.

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
        # Save best samples in a folder named after the epoch
        epoch_dir = results_dir / f"best_epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        to_pil = transforms.ToPILImage()
        num_to_save = min(num_samples, lr_images.size(0))

        for i in range(num_to_save):
            # Clamp values to valid [0, 1] range before converting
            lr_pil = to_pil(torch.clamp(lr_images[i].cpu(), 0, 1))
            sr_pil = to_pil(torch.clamp(sr_images[i].cpu(), 0, 1))
            hr_pil = to_pil(torch.clamp(hr_images[i].cpu(), 0, 1))

            # Combine LR, SR, and HR images side-by-side for easy comparison
            total_width = lr_pil.width + sr_pil.width + hr_pil.width
            max_height = max(lr_pil.height, sr_pil.height, hr_pil.height)
            combined_img = Image.new('RGB', (total_width, max_height))
            combined_img.paste(lr_pil, (0, 0))
            combined_img.paste(sr_pil, (lr_pil.width, 0))
            combined_img.paste(hr_pil, (lr_pil.width + sr_pil.width, 0))

            combined_img.save(epoch_dir / f"sample_{i}_comparison.png")

        logger.info(f"Saved {num_to_save} comparison images to {epoch_dir}")

    except Exception as e:
        logger.warning(f"Failed to save sample images: {str(e)}")


def plot_training_metrics(
        history: List[Dict[str, Any]],
        model_name: str,
        output_path: str,
        style: str = "seaborn-v0_8-darkgrid",
        img_format: str = "png"
):
    """
    Generate and save beautiful, informative plots of training metrics.

    Args:
        history: A dictionary containing lists of metric values for each epoch.
                 e.g., {'train_loss': [...], 'valid_psnr': [...]}
        model_name: The name of the model for the plot title.
        output_path: Path to save the plot image.
        style: The matplotlib style to use for the plot.
        img_format: The file format for the saved plot.
    """
    try:
        plt.style.use(style)
        df = pd.DataFrame(history)

        plot_configs = []

        # Determine which plots to create based on available metrics
        if 'train_loss' in df.columns or 'valid_loss' in df.columns:
            plot_configs.append({'title': 'Model Loss', 'metrics': ['train_loss', 'valid_loss']})
        if 'train_g_total_loss' in df.columns:  # SRGAN
            plot_configs.append(
                {'title': 'Generator & Discriminator Loss', 'metrics': ['train_g_total_loss', 'train_d_total_loss']})
        if 'train_psnr' in df.columns or 'valid_psnr' in df.columns:
            plot_configs.append({'title': 'Model PSNR (dB)', 'metrics': ['train_psnr', 'valid_psnr']})
        if 'train_ssim' in df.columns or 'valid_ssim' in df.columns:
            plot_configs.append({'title': 'Model SSIM', 'metrics': ['train_ssim', 'valid_ssim']})

        if not plot_configs:
            logger.warning("No plottable metrics found in history. Skipping plot generation.")
            return

        fig, axes = plt.subplots(len(plot_configs), 1, figsize=(12, 6 * len(plot_configs)), sharex=True)
        if len(plot_configs) == 1:
            axes = [axes]

        for ax, p_config in zip(axes, plot_configs):
            for metric in p_config['metrics']:
                if metric in df.columns:
                    sns.lineplot(x='epoch', y=metric, data=df.dropna(subset=[metric]), ax=ax,
                                 label=metric.replace('_', ' ').title())
            ax.set_title(f'{model_name.upper()}: {p_config["title"]}', fontsize=16)
            ax.set_ylabel(p_config['title'].split(' ')[-1])
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel('Epoch', fontsize=12)
        fig.tight_layout(pad=3.0)

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = f"{output_path}.{img_format}"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved training plot to {save_path}")

    except Exception as e:
        logger.error(f"Failed to generate or save training plot: {e}", exc_info=True)
