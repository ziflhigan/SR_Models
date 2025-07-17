"""
Evaluation script for trained SR_Models.

This script evaluates a trained model on the validation dataset. It generates:
1.  Average performance metrics (PSNR, SSIM).
2.  Detailed per-image metrics saved to a CSV file.
3.  Distribution plots for PSNR and SSIM scores.
4.  Side-by-side comparison images (LR, SR, HR) for visual inspection.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import Config
from src.div2k import DIV2KDataset
from src.models.srcnn import create_srcnn_from_config
from src.models.srgan_generator import create_srgan_generator_from_config
from src.utils import get_logger, setup_logger
from src.utils.metrics import calculate_psnr, calculate_ssim


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Super-Resolution models based on a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration file'
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, model_type: str, config: Config, device: torch.device) -> torch.nn.Module:
    """Load a trained model from a checkpoint file."""
    logger = get_logger()
    logger.info(f"Loading {model_type.upper()} model...")

    # Create a new model instance
    if model_type == 'srcnn':
        model = create_srcnn_from_config(config)
    elif model_type == 'srgan':
        model = create_srgan_generator_from_config(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats (from pre-training or adversarial training)
        state_dict_key = None
        if 'generator_state_dict' in checkpoint:
            state_dict_key = 'generator_state_dict'
        elif 'model_state_dict' in checkpoint:
            state_dict_key = 'model_state_dict'

        if state_dict_key:
            model.load_state_dict(checkpoint[state_dict_key])
            epoch = checkpoint.get('epoch', 'N/A')
            logger.info(f"Loaded model weights from checkpoint (Epoch: {epoch})")
        else:
            # Fallback for checkpoints that only contain the state dict
            model.load_state_dict(checkpoint)
            logger.info("Loaded model weights from a raw state_dict checkpoint.")

    except Exception as e:
        logger.error(f"Failed to load checkpoint from '{checkpoint_path}': {e}", exc_info=True)
        raise

    model.to(device)
    model.eval()
    return model


def save_comparison_image(lr_img: Image.Image, sr_img: Image.Image, hr_img: Image.Image, path: Path, psnr: float,
                          ssim: float):
    """Saves a side-by-side comparison image of LR, SR, and HR."""
    lr_w, lr_h = lr_img.size
    sr_w, sr_h = sr_img.size

    # Create a new image with a white background
    grid = Image.new('RGB', (lr_w + sr_w + sr_w, sr_h + 40), 'white')

    # Paste the upscaled LR, SR, and HR images
    bicubic_lr = lr_img.resize(sr_img.size, Image.Resampling.BICUBIC)
    grid.paste(bicubic_lr, (0, 40))
    grid.paste(sr_img, (sr_w, 40))
    grid.paste(hr_img, (sr_w * 2, 40))

    # Add titles and metrics
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid)
    draw.text((10, 10), "Bicubic", font=font, fill="black")
    draw.text((sr_w + 10, 10), f"SR Output (PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f})", font=font, fill="black")
    draw.text((sr_w * 2 + 10, 10), "Ground Truth", font=font, fill="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def generate_plots(df: pd.DataFrame, output_dir: Path, model_name: str):
    """Generates and saves plots for performance metrics."""
    logger = get_logger()
    logger.info("Generating performance plots...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'{model_name.upper()} Performance Distribution on Validation Set', fontsize=18)

    # PSNR Distribution Plot
    sns.histplot(df['psnr'], kde=True, ax=axes[0], color='skyblue', bins=20)
    axes[0].set_title('PSNR (dB) Distribution', fontsize=14)
    axes[0].set_xlabel('PSNR (dB)', fontsize=12)
    axes[0].axvline(df['psnr'].mean(), color='r', linestyle='--', label=f"Mean: {df['psnr'].mean():.2f}")
    axes[0].legend()

    # SSIM Distribution Plot
    sns.histplot(df['ssim'], kde=True, ax=axes[1], color='salmon', bins=20)
    axes[1].set_title('SSIM Distribution', fontsize=14)
    axes[1].set_xlabel('SSIM', fontsize=12)
    axes[1].axvline(df['ssim'].mean(), color='r', linestyle='--', label=f"Mean: {df['ssim'].mean():.4f}")
    axes[1].legend()

    plot_path = output_dir / 'performance_plots.png'
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved performance plots to: {plot_path}")


def main():
    """Main evaluation function."""
    args = parse_arguments()
    config = Config(args.config)
    logger = setup_logger(config.dict)

    # Get evaluation settings from config
    eval_config = config.get('evaluation', {})
    model_type = config.get('model.name')
    checkpoint_path = eval_config.get('checkpoint_path')
    output_dir = Path(eval_config.get('output_dir', 'evaluation'))
    num_comparison = eval_config.get('num_comparison_images', 5)

    if not checkpoint_path:
        logger.error("'evaluation.checkpoint_path' not found in config. Please specify the model to evaluate.")
        sys.exit(1)

    device = torch.device(config.get('model.device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Starting evaluation for {model_type.upper()} on device '{device}'")

    # Create output directories
    comparison_dir = output_dir / 'comparison_images'
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Load the trained model
    model = load_model(checkpoint_path, model_type, config, device)

    dataset = DIV2KDataset(
        hr_dir=config.get('dataset.valid_hr_dir'),
        lr_dir=config.get('dataset.valid_lr_dir'),
        scale_factor=config.get('dataset.scale_factor'),
        hr_crop_size=None,
        augmentation={},
        normalization_config=config.get('dataset.normalization', {}),
        mode='val'
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=eval_config.get('batch_size', 1))

    # Evaluation loop
    results = []
    total_time = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating on validation set")

    to_pil = transforms.ToPILImage()

    with torch.no_grad():
        for i, (lr_t, hr_t) in pbar:
            lr_t, hr_t = lr_t.to(device), hr_t.to(device)

            # For SRCNN, the input must be bicubic-upscaled first
            input_t = lr_t
            if model_type == 'srcnn':
                input_t = F.interpolate(lr_t, scale_factor=config.get('dataset.scale_factor'), mode='bicubic')

            # Inference
            start_time = time.time()
            sr_t = model(input_t)
            total_time += time.time() - start_time

            # Denormalize HR and SR tensors for metric calculation and saving
            # Assuming hr_norm_type is 'minus_one_one' or 'zero_one' and output range is similar
            norm_type_hr = config.get('dataset.normalization.hr_norm_type')

            # Calculate metrics
            psnr = calculate_psnr(sr_t, hr_t, norm_type=norm_type_hr)
            ssim = calculate_ssim(sr_t, hr_t, norm_type=norm_type_hr)

            results.append({'image_index': i, 'psnr': psnr, 'ssim': ssim})
            pbar.set_postfix({'PSNR': f"{psnr:.2f}", 'SSIM': f"{ssim:.4f}"})

            # Save comparison grid for the first N images
            if i < num_comparison:
                # Convert all tensors to PIL images for saving
                # We assume LR is [0,1] and HR/SR are [-1,1] or [0,1] based on config
                lr_img = to_pil(lr_t.squeeze(0).cpu())  # LR is always [0,1]

                # De-normalize SR and HR before converting to PIL
                sr_t_denorm = (sr_t.clamp(-1, 1) + 1) / 2 if norm_type_hr == 'minus_one_one' else sr_t.clamp(0, 1)
                hr_t_denorm = (hr_t.clamp(-1, 1) + 1) / 2 if norm_type_hr == 'minus_one_one' else hr_t.clamp(0, 1)

                sr_img = to_pil(sr_t_denorm.squeeze(0).cpu())
                hr_img = to_pil(hr_t_denorm.squeeze(0).cpu())

                save_comparison_image(
                    lr_img, sr_img, hr_img,
                    path=comparison_dir / f'comparison_{i:04d}.png',
                    psnr=psnr, ssim=ssim
                )

    df = pd.DataFrame(results)

    # Save detailed metrics to CSV
    csv_path = output_dir / 'evaluation_metrics_detailed.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved detailed metrics to: {csv_path}")

    generate_plots(df, output_dir, model_type)

    avg_psnr = df['psnr'].mean()
    std_psnr = df['psnr'].std()
    avg_ssim = df['ssim'].mean()
    std_ssim = df['ssim'].std()
    avg_time = (total_time / len(dataloader)) * 1000

    summary = (
        f"\n{'=' * 50}\n"
        f"Evaluation Summary for {model_type.upper()}\n"
        f"Checkpoint: {checkpoint_path}\n"
        f"{'=' * 50}\n"
        f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB\n"
        f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n"
        f"Average Inference Time: {avg_time:.2f} ms per image\n"
        f"Images Evaluated: {len(df)}\n"
        f"{'=' * 50}\n"
    )
    logger.info(summary)

    summary_path = output_dir / 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Saved summary to: {summary_path}")


if __name__ == '__main__':
    main()
