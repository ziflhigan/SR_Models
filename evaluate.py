"""
Evaluation script for trained SR_Models.
Supports inference on single images, directories, or validation datasets.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.config import Config
from src.div2k import DIV2KDataset
from src.models.srcnn import create_srcnn_from_config
from src.models.srgan_generator import create_srgan_generator_from_config
from src.utils import get_logger, setup_logger
from src.utils.metrics import calculate_psnr, calculate_ssim


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Super-Resolution models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint file'
    )

    # Input options (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to single input image'
    )
    input_group.add_argument(
        '--dir', '-d',
        type=str,
        help='Path to directory containing images'
    )
    input_group.add_argument(
        '--valid',
        action='store_true',
        help='Evaluate on validation dataset'
    )

    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        choices=['srcnn', 'srgan'],
        required=True,
        help='Model type'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--scale-factor',
        type=int,
        choices=[2, 3, 4, 8],
        help='Super-resolution scale factor (overrides config)'
    )

    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/evaluation',
        help='Directory to save output images'
    )

    parser.add_argument(
        '--save-lr',
        action='store_true',
        help='Also save LR input images for comparison'
    )

    parser.add_argument(
        '--save-grid',
        action='store_true',
        help='Save comparison grid (LR, SR, HR if available)'
    )

    # Processing options
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for processing'
    )

    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Skip metric calculation (for images without HR reference)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )

    parser.add_argument(
        '--suffix',
        type=str,
        default='_sr',
        help='Suffix for output filenames'
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, model_type: str, config: Config, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    logger = get_logger()

    # Create model
    if model_type == 'srcnn':
        model = create_srcnn_from_config(config.dict)
    elif model_type == 'srgan':
        model = create_srgan_generator_from_config(config.dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)

        epoch = checkpoint.get('epoch', 'unknown')
        logger.info(f"Loaded {model_type.upper()} model from epoch {epoch}")

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    model.to(device)
    model.eval()

    return model


def process_single_image(
        image_path: str,
        model: torch.nn.Module,
        scale_factor: int,
        device: torch.device,
        model_type: str
) -> Tuple[Image.Image, Image.Image, float]:
    """Process a single image and return LR, SR images and inference time."""

    # Load the image provided by the user, assume it's the LR input
    lr_image = Image.open(image_path).convert('RGB')

    if model_type == 'srcnn':
        # SRCNN expects an input that is already upscaled to the target size
        target_size = (lr_image.width * scale_factor, lr_image.height * scale_factor)
        input_image = lr_image.resize(target_size, Image.Resampling.BICUBIC)
    else:  # srgan
        # SRGAN takes the LR image directly
        input_image = lr_image

    # Convert to tensor
    transform = transforms.ToTensor()
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        if model_type == 'srcnn':
            sr_tensor = model(input_tensor)
        else:  # srgan
            sr_tensor = model(input_tensor)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    inference_time = time.time() - start_time

    # Convert back to PIL image
    sr_tensor = torch.clamp(sr_tensor, 0, 1)
    to_pil = transforms.ToPILImage()
    sr_image = to_pil(sr_tensor.squeeze(0).cpu())

    return lr_image, sr_image, inference_time


def evaluate_on_dataset(
        model: torch.nn.Module,
        config: Config,
        device: torch.device,
        model_type: str,
        output_dir: Path,
        save_images: bool = True
) -> Dict[str, float]:
    """Evaluate model on validation dataset."""
    logger = get_logger()

    # Create validation dataset
    dataset = DIV2KDataset(
        hr_dir=config.get('dataset.valid_hr_dir'),
        lr_dir=config.get('dataset.valid_lr_dir'),
        scale_factor=config.get('dataset.scale_factor', 4),
        hr_crop_size=None,  # Use full images for validation
        augmentation={},  # No augmentation for validation
        normalize=config.get('dataset.normalize', True),
        norm_type=config.get('dataset.norm_type', 'zero_one'),
        mode='val'
    )

    # Metrics
    psnr_list = []
    ssim_list = []
    total_time = 0

    # Process each image
    logger.info(f"Evaluating on {len(dataset)} validation images...")

    with tqdm(total=len(dataset), desc="Evaluating") as pbar:
        for idx in range(len(dataset)):
            lr_tensor, hr_tensor = dataset[idx]
            lr_tensor = lr_tensor.unsqueeze(0).to(device)
            hr_tensor = hr_tensor.unsqueeze(0).to(device)

            # SRCNN expects upscaled input
            if model_type == 'srcnn':
                scale = config.get('dataset.scale_factor', 4)
                lr_tensor = functional.interpolate(lr_tensor, scale_factor=scale, mode='bicubic', align_corners=False)

            # Run inference
            start_time = time.time()
            with torch.no_grad():
                sr_tensor = model(lr_tensor)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_time = time.time() - start_time
            total_time += inference_time

            # Calculate metrics
            psnr = calculate_psnr(sr_tensor, hr_tensor)
            ssim = calculate_ssim(sr_tensor, hr_tensor)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            # Save sample images
            if save_images and idx < 10:  # Save first 10 images
                save_path = output_dir / f"val_{idx:04d}.png"
                to_pil = transforms.ToPILImage()
                sr_image = to_pil(torch.clamp(sr_tensor.squeeze(0).cpu(), 0, 1))
                sr_image.save(save_path)

            pbar.update(1)
            pbar.set_postfix({
                'PSNR': f"{psnr:.2f}",
                'SSIM': f"{ssim:.4f}",
                'Time': f"{inference_time * 1000:.1f}ms"
            })

    # Calculate average metrics
    avg_metrics = {
        'avg_psnr': np.mean(psnr_list),
        'std_psnr': np.std(psnr_list),
        'avg_ssim': np.mean(ssim_list),
        'std_ssim': np.std(ssim_list),
        'avg_time_ms': (total_time / len(dataset)) * 1000,
        'total_images': len(dataset)
    }

    return avg_metrics


def process_directory(
        input_dir: Path,
        model: torch.nn.Module,
        scale_factor: int,
        device: torch.device,
        model_type: str,
        output_dir: Path,
        suffix: str,
        save_lr: bool
) -> Dict[str, Any]:
    """Process all images in a directory."""
    logger = get_logger()

    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    image_files = [f for f in input_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return {}

    logger.info(f"Processing {len(image_files)} images from {input_dir}")

    total_time = 0
    results = []

    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Process image
            lr_image, sr_image, inference_time = process_single_image(
                str(image_path), model, scale_factor, device, model_type
            )
            total_time += inference_time

            # Save SR image
            sr_filename = image_path.stem + suffix + image_path.suffix
            sr_path = output_dir / sr_filename
            sr_image.save(sr_path)

            # Save LR image if requested
            if save_lr:
                lr_filename = image_path.stem + '_lr' + image_path.suffix
                lr_path = output_dir / lr_filename
                lr_image.save(lr_path)

            results.append({
                'filename': image_path.name,
                'inference_time': inference_time,
                'sr_path': str(sr_path)
            })

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            continue

    return {
        'processed_images': len(results),
        'total_time': total_time,
        'avg_time_ms': (total_time / len(results) * 1000) if results else 0,
        'results': results
    }


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Apply overrides
    if args.scale_factor:
        config.set('dataset.scale_factor', args.scale_factor)

    # Setup logger
    logger = setup_logger(config.dict)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.get('model.device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    try:
        # Load model
        logger.info(f"Loading {args.model.upper()} model from {args.checkpoint}")
        model = load_model(args.checkpoint, args.model, config, device)

        # Get scale factor
        scale_factor = config.get('dataset.scale_factor', 4)
        logger.info(f"Scale factor: {scale_factor}x")

        # Process based on input type
        if args.image:
            # Single image processing
            logger.info(f"Processing single image: {args.image}")

            if not Path(args.image).exists():
                logger.error(f"Image file not found: {args.image}")
                sys.exit(1)

            lr_image, sr_image, inference_time = process_single_image(
                args.image, model, scale_factor, device, args.model
            )

            # Save results
            input_path = Path(args.image)
            sr_filename = input_path.stem + args.suffix + input_path.suffix
            sr_path = output_dir / sr_filename
            sr_image.save(sr_path)

            if args.save_lr:
                lr_filename = input_path.stem + '_lr' + input_path.suffix
                lr_path = output_dir / lr_filename
                lr_image.save(lr_path)

            logger.info(f"Saved super-resolved image to: {sr_path}")
            logger.info(f"Inference time: {inference_time * 1000:.2f} ms")

        elif args.dir:
            # Directory processing
            input_dir = Path(args.dir)
            if not input_dir.exists():
                logger.error(f"Directory not found: {args.dir}")
                sys.exit(1)

            results = process_directory(
                input_dir, model, scale_factor, device, args.model,
                output_dir, args.suffix, args.save_lr
            )

            logger.info(f"\nProcessed {results['processed_images']} images")
            logger.info(f"Average inference time: {results['avg_time_ms']:.2f} ms")
            logger.info(f"Total processing time: {results['total_time']:.2f} seconds")

        elif args.valid:
            # Validation dataset evaluation
            metrics = evaluate_on_dataset(
                model, config, device, args.model, output_dir,
                save_images=not args.no_metrics
            )

            # Print results
            logger.info("\n" + "=" * 50)
            logger.info("Validation Results:")
            logger.info("=" * 50)
            logger.info(f"Average PSNR: {metrics['avg_psnr']:.2f} ± {metrics['std_psnr']:.2f} dB")
            logger.info(f"Average SSIM: {metrics['avg_ssim']:.4f} ± {metrics['std_ssim']:.4f}")
            logger.info(f"Average inference time: {metrics['avg_time_ms']:.2f} ms")
            logger.info(f"Total images evaluated: {metrics['total_images']}")
            logger.info("=" * 50)

            # Save metrics to file
            metrics_file = output_dir / 'validation_metrics.txt'
            with open(metrics_file, 'w') as f:
                f.write(f"Model: {args.model.upper()}\n")
                f.write(f"Checkpoint: {args.checkpoint}\n")
                f.write(f"Scale Factor: {scale_factor}x\n")
                f.write(f"Device: {device}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Average PSNR: {metrics['avg_psnr']:.2f} ± {metrics['std_psnr']:.2f} dB\n")
                f.write(f"Average SSIM: {metrics['avg_ssim']:.4f} ± {metrics['std_ssim']:.4f}\n")
                f.write(f"Average inference time: {metrics['avg_time_ms']:.2f} ms\n")
                f.write(f"Total images: {metrics['total_images']}\n")

            logger.info(f"Metrics saved to: {metrics_file}")

        logger.info("\nEvaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.close()


if __name__ == '__main__':
    main()
