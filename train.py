"""
Main training entry point for SR_Models.
Supports training both SRCNN and SRGAN models with configurable parameters.
"""

import argparse
import os
import sys
import torch
from pathlib import Path

from src.config import Config
from src.div2k import create_data_loaders
from src.trainers import SRCNNTrainer, SRGANTrainer
from src.utils import setup_logger, get_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Super-Resolution models (SRCNN or SRGAN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    # Optional overrides
    parser.add_argument(
        '--model',
        type=str,
        choices=['srcnn', 'srgan'],
        help='Model type to train (overrides config)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size (overrides config)'
    )

    parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        help='Learning rate (overrides config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use for training (overrides config)'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory to save checkpoints (overrides config)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )

    parser.add_argument(
        '--scale-factor',
        type=int,
        choices=[2, 3, 4, 8],
        help='Super-resolution scale factor (overrides config)'
    )

    parser.add_argument(
        '--crop-size',
        type=int,
        help='HR crop size for training patches (overrides config)'
    )

    parser.add_argument(
        '--use-amp',
        action='store_true',
        help='Enable automatic mixed precision training'
    )

    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable automatic mixed precision training'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        help='Number of data loading workers (overrides config)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (overrides config)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # SRGAN specific arguments
    parser.add_argument(
        '--content-loss',
        type=str,
        choices=['mse', 'l1', 'vgg'],
        help='Content loss type for SRGAN (overrides config)'
    )

    parser.add_argument(
        '--lambda-adv',
        type=float,
        help='Adversarial loss weight for SRGAN (overrides config)'
    )

    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        help='Number of MSE pre-training epochs for SRGAN (overrides config)'
    )

    return parser.parse_args()


def apply_overrides(config, args):
    """Apply command line overrides to configuration."""
    logger = get_logger()

    # Model selection
    if args.model:
        config.set('model.name', args.model)
        logger.info(f"Override: model = {args.model}")

    # Training parameters
    if args.epochs:
        config.set('training.epochs', args.epochs)
        logger.info(f"Override: epochs = {args.epochs}")

    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
        logger.info(f"Override: batch_size = {args.batch_size}")

    if args.learning_rate:
        config.set('training.learning_rate.generator', args.learning_rate)
        config.set('training.learning_rate.discriminator', args.learning_rate)
        logger.info(f"Override: learning_rate = {args.learning_rate}")

    # Device
    if args.device:
        config.set('model.device', args.device)
        logger.info(f"Override: device = {args.device}")

    # Dataset parameters
    if args.scale_factor:
        config.set('dataset.scale_factor', args.scale_factor)
        logger.info(f"Override: scale_factor = {args.scale_factor}")

    if args.crop_size:
        config.set('dataset.hr_crop_size', args.crop_size)
        logger.info(f"Override: hr_crop_size = {args.crop_size}")

    # Output directories
    if args.checkpoint_dir:
        config.set('output.checkpoint_dir', args.checkpoint_dir)
        logger.info(f"Override: checkpoint_dir = {args.checkpoint_dir}")

    # Training options
    if args.use_amp:
        config.set('training.use_amp', True)
        logger.info("Override: use_amp = True")
    elif args.no_amp:
        config.set('training.use_amp', False)
        logger.info("Override: use_amp = False")

    if args.num_workers:
        config.set('training.num_workers', args.num_workers)
        logger.info(f"Override: num_workers = {args.num_workers}")

    # Logging
    if args.log_level:
        config.set('logging.level', args.log_level)
        logger.info(f"Override: log_level = {args.log_level}")

    # SRGAN specific
    if args.content_loss and config.get('model.name') == 'srgan':
        config.set('srgan.loss.content_loss', args.content_loss)
        logger.info(f"Override: content_loss = {args.content_loss}")

    if args.lambda_adv is not None and config.get('model.name') == 'srgan':
        config.set('srgan.loss.adversarial_weight', args.lambda_adv)
        logger.info(f"Override: lambda_adv = {args.lambda_adv}")

    if args.pretrain_epochs is not None and config.get('model.name') == 'srgan':
        config.set('srgan.pretrain_epochs', args.pretrain_epochs)
        logger.info(f"Override: pretrain_epochs = {args.pretrain_epochs}")


def set_random_seed(seed: int, config: Config):
    """Set random seed and configure cuDNN for reproducibility or performance."""
    import random
    import numpy as np
    logger = get_logger()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Read the reproducibility setting from the config
    is_reproducible = config.get('training.reproducible', False)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if is_reproducible:
            # Settings for full reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.warning(
                "Reproducible training enabled. cuDNN benchmark is disabled, which may slow down training."
            )
        else:
            # Default settings for performance
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            logger.info(
                "Performance-optimized training enabled. cuDNN benchmark is active."
            )


def main():
    """Main training function."""
    args = parse_arguments()

    try:
        config = Config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    apply_overrides(config, args)

    logger = setup_logger(config.dict)

    set_random_seed(args.seed, config)
    logger.info(f"Random seed set to {args.seed}")

    try:
        # Log training start
        logger.log_training_start(config.model_name, {
            'Model': config.model_name,
            'Device': config.device,
            'Epochs': config.epochs,
            'Batch Size': config.batch_size,
            'Scale Factor': config.scale_factor,
            'Learning Rate': config.get('training.learning_rate.generator'),
            'Use AMP': config.get('training.use_amp', False),
            'Checkpoint Dir': config.checkpoint_dir
        })

        # Validate dataset paths
        dataset_config = config.get('dataset', {})
        train_hr_dir = dataset_config.get('train_hr_dir')
        train_lr_dir = dataset_config.get('train_lr_dir')
        valid_hr_dir = dataset_config.get('valid_hr_dir')
        valid_lr_dir = dataset_config.get('valid_lr_dir')

        # Check if dataset directories exist
        for dir_path, dir_name in [
            (train_hr_dir, 'train_hr_dir'),
            (train_lr_dir, 'train_lr_dir'),
            (valid_hr_dir, 'valid_hr_dir'),
            (valid_lr_dir, 'valid_lr_dir')
        ]:
            if not os.path.exists(dir_path):
                logger.error(f"Dataset directory not found: {dir_path}")
                logger.error(f"Please check your config or download the DIV2K dataset")
                sys.exit(1)

        # Create data loaders
        logger.info("Creating data loaders...")
        with logger.catch_errors("Failed to create data loaders"):
            train_loader, valid_loader = create_data_loaders(
                config.dict,
                train_hr_dir=train_hr_dir,
                train_lr_dir=train_lr_dir,
                valid_hr_dir=valid_hr_dir,
                valid_lr_dir=valid_lr_dir
            )

        # Select and create trainer
        model_name = config.model_name.lower()
        logger.info(f"Creating {model_name.upper()} trainer...")

        if model_name == 'srcnn':
            trainer = SRCNNTrainer(config, train_loader, valid_loader)
        elif model_name == 'srgan':
            trainer = SRGANTrainer(config, train_loader, valid_loader)
        else:
            logger.error(f"Unknown model type: {model_name}")
            sys.exit(1)

        # Handle checkpoint resuming
        if args.resume:
            checkpoint_path = Path(args.resume)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint file not found: {args.resume}")
                sys.exit(1)
            try:
                trainer.resume_from_checkpoint(checkpoint_path)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
                sys.exit(1)

        # Start training
        logger.info("Starting training...")
        trainer.train()

        logger.info("Training completed successfully!")

        # Save final configuration for reference
        final_config_path = Path(config.checkpoint_dir) / config.model_name / 'final_config.yaml'
        config.save(str(final_config_path))
        logger.info(f"Final configuration saved to: {final_config_path}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.close()


if __name__ == '__main__':
    main()
