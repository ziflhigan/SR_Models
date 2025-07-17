"""
DIV2K dataset implementation for super-resolution training.
"""

import random
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger()


class DIV2KDataset(Dataset):
    """
    DIV2K dataset for super-resolution training.

    Supports loading HR/LR image pairs with data augmentation,
    random cropping, and various normalization options.
    """

    def __init__(
            self,
            hr_dir: str,
            lr_dir: str,
            scale_factor: int = 4,
            hr_crop_size: Optional[int] = None,
            augmentation: Optional[Dict[str, bool]] = None,
            normalization_config: Optional[Dict[str, any]] = None,
            mode: str = "train"
    ):
        """
        Initialize DIV2K dataset.

        Args:
            hr_dir: Path to high-resolution images directory
            lr_dir: Path to low-resolution images directory
            scale_factor: Up-scaling factor (2, 3, 4, or 8)
            hr_crop_size: Size of HR patches to crop (None for full images)
            augmentation: Dictionary of augmentation options
            normalization_config: Normalization configurations for both HR and LR images
            mode: Dataset mode ("train", "val", or "test")
        """
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.scale_factor = scale_factor
        self.hr_crop_size = hr_crop_size
        self.mode = mode
        self.norm_config = normalization_config or {'enabled': False}

        # Log dataset initialization
        logger.info(f"Initializing DIV2KDataset for '{self.mode}' mode.")
        logger.debug(
            f"Params: HR Dir='{hr_dir}', LR Dir='{lr_dir}', Scale=x{scale_factor}, "
            f"Crop Size={hr_crop_size}, Mode='{mode}'"
        )

        # Default augmentation settings
        self.augmentation = augmentation or {
            "horizontal_flip": False,
            "vertical_flip": False,
            "rotation": False
        }

        # Validate directories and log errors
        if not self.hr_dir.exists():
            error_msg = f"HR directory not found: {self.hr_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        if not self.lr_dir.exists():
            error_msg = f"LR directory not found: {self.lr_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.image_files = self._get_image_files()

        # Log number of files found or raise an error
        if not self.image_files:
            error_msg = f"No valid image pairs found in {hr_dir} and {lr_dir} for scale x{self.scale_factor}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Found {len(self.image_files)} matching image pairs for '{self.mode}' mode.")

        self._setup_transforms()

    def _get_image_files(self) -> list:
        """Get list of image file names that exist in both HR and LR directories."""
        hr_files = set()
        lr_files = set()
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

        for file in self.hr_dir.iterdir():
            if file.suffix.lower() in extensions:
                hr_files.add(file.stem)

        for file in self.lr_dir.iterdir():
            if file.suffix.lower() in extensions:
                stem = file.stem
                if f'x{self.scale_factor}' in stem:
                    stem = stem.replace(f'x{self.scale_factor}', '')
                lr_files.add(stem)

        logger.debug(f"Found {len(hr_files)} HR files and {len(lr_files)} LR files before matching.")
        common_files = hr_files.intersection(lr_files)
        return sorted(list(common_files))

    def _setup_transforms(self):
        """Set up image transforms based on the normalization config."""
        self.to_tensor = transforms.ToTensor()

        def get_transform(norm_type):
            if norm_type == "minus_one_one":
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            # Default to [0, 1] range
            return transforms.ToTensor()

        if self.norm_config.get('enabled', False):
            lr_norm_type = self.norm_config.get('lr_norm_type', 'zero_one')
            hr_norm_type = self.norm_config.get('hr_norm_type', 'minus_one_one')
            self.lr_transform = get_transform(lr_norm_type)
            self.hr_transform = get_transform(hr_norm_type)
            logger.debug(f"Transforms set up: LR norm='{lr_norm_type}', HR norm='{hr_norm_type}'.")
        else:
            self.lr_transform = self.to_tensor
            self.hr_transform = self.to_tensor
            logger.debug("Normalization is disabled.")

    def _load_image_pair(self, file_stem: str) -> Tuple[Image.Image, Image.Image]:
        """Load HR and LR image pair."""
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        hr_image, lr_image = None, None

        for ext in extensions:
            hr_path = self.hr_dir / f"{file_stem}{ext}"
            if hr_path.exists():
                hr_image = Image.open(hr_path).convert('RGB')
                break

        for ext in extensions:
            lr_path = self.lr_dir / f"{file_stem}x{self.scale_factor}{ext}"
            if lr_path.exists():
                lr_image = Image.open(lr_path).convert('RGB')
                break

        if hr_image is None:
            msg = f"HR image not found for stem '{file_stem}' in {self.hr_dir}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        if lr_image is None:
            msg = f"LR image not found for stem '{file_stem}x{self.scale_factor}' in {self.lr_dir}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        return hr_image, lr_image

    def _random_crop(self, hr_image: Image.Image, lr_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Perform random cropping on HR and corresponding LR regions."""
        hr_width, hr_height = hr_image.size
        lr_crop_size = self.hr_crop_size // self.scale_factor

        if self.hr_crop_size > hr_width or self.hr_crop_size > hr_height:
            raise ValueError(
                f"hr_crop_size ({self.hr_crop_size}) is larger than the image dimensions "
                f"({hr_width}x{hr_height}). Please use a smaller crop size."
            )

        max_x = hr_width - self.hr_crop_size
        max_y = hr_height - self.hr_crop_size
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0

        hr_cropped = hr_image.crop((x, y, x + self.hr_crop_size, y + self.hr_crop_size))
        lr_x = x // self.scale_factor
        lr_y = y // self.scale_factor
        lr_cropped = lr_image.crop((lr_x, lr_y, lr_x + lr_crop_size, lr_y + lr_crop_size))

        return hr_cropped, lr_cropped

    def _apply_augmentation(self, hr_image: Image.Image, lr_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply data augmentation to image pair."""
        if self.mode != "train" or not any(self.augmentation.values()):
            return hr_image, lr_image

        def apply_transpose(operation: Image.Transpose):
            nonlocal hr_image, lr_image
            hr_image = hr_image.transpose(operation)
            lr_image = lr_image.transpose(operation)

        if self.augmentation.get("horizontal_flip", False) and random.random() > 0.5:
            apply_transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.augmentation.get("vertical_flip", False) and random.random() > 0.5:
            apply_transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if self.augmentation.get("rotation", False) and random.random() > 0.5:
            rotation_op = random.choice([
                Image.Transpose.ROTATE_90,
                Image.Transpose.ROTATE_180,
                Image.Transpose.ROTATE_270
            ])
            apply_transpose(rotation_op)

        return hr_image, lr_image

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset."""
        file_stem = self.image_files[idx]
        hr_image, lr_image = self._load_image_pair(file_stem)

        if self.hr_crop_size is not None and self.mode == "train":
            hr_image, lr_image = self._random_crop(hr_image, lr_image)

        hr_image, lr_image = self._apply_augmentation(hr_image, lr_image)
        hr_tensor = self.hr_transform(hr_image)
        lr_tensor = self.lr_transform(lr_image)

        return lr_tensor, hr_tensor

    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """Get information about an image without loading it."""
        file_stem = self.image_files[idx]
        hr_image, lr_image = self._load_image_pair(file_stem)
        return {
            'file_stem': file_stem,
            'hr_size': hr_image.size,
            'lr_size': lr_image.size,
            'hr_mode': hr_image.mode,
            'lr_mode': lr_image.mode
        }


def create_data_loaders(
        config: Config,
        train_hr_dir: str,
        train_lr_dir: str,
        valid_hr_dir: Optional[str] = None,
        valid_lr_dir: Optional[str] = None
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """Create data loaders for training and optionally validation."""
    logger.info("Creating data loaders...")
    dataset_config = config.get('dataset', {})
    training_config = config.get('training', {})

    # Extract the entire normalization dictionary from the config
    normalization_config = dataset_config.get('normalization', {})

    batch_size = training_config.get('batch_size', 16)
    num_workers = training_config.get('num_workers', 4)

    train_dataset = DIV2KDataset(
        hr_dir=train_hr_dir,
        lr_dir=train_lr_dir,
        scale_factor=dataset_config.get('scale_factor', 4),
        hr_crop_size=dataset_config.get('hr_crop_size', 96),
        augmentation=dataset_config.get('augmentation', {}),
        normalization_config=normalization_config,
        mode='train'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    logger.info(f"Training DataLoader created with batch_size={batch_size}, num_workers={num_workers}.")

    if valid_hr_dir and valid_lr_dir:
        valid_dataset = DIV2KDataset(
            hr_dir=valid_hr_dir,
            lr_dir=valid_lr_dir,
            scale_factor=dataset_config.get('scale_factor', 4),
            hr_crop_size=None,
            augmentation={},
            normalization_config=normalization_config,
            mode='val'
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,  # Typically 1 for validation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        logger.info(f"Validation DataLoader created with batch_size={batch_size}, num_workers={num_workers}.")
        return train_loader, valid_loader

    return train_loader


def get_dataset_stats(dataset: DIV2KDataset, num_samples: int = 100) -> Dict[str, Any]:
    """Calculate dataset statistics for a sample of images."""
    logger.info(f"Calculating dataset stats for '{dataset.mode}' mode with {num_samples} samples.")
    num_samples = min(num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)

    hr_sizes, lr_sizes = [], []
    for idx in sample_indices:
        info = dataset.get_image_info(idx)
        hr_sizes.append(info['hr_size'])
        lr_sizes.append(info['lr_size'])

    hr_widths = [size[0] for size in hr_sizes]
    hr_heights = [size[1] for size in hr_sizes]
    lr_widths = [size[0] for size in lr_sizes]
    lr_heights = [size[1] for size in lr_sizes]

    stats = {
        'num_images': len(dataset),
        'sample_size': len(sample_indices),
        'hr_resolution': {
            'width': {'min': min(hr_widths), 'max': max(hr_widths), 'mean': sum(hr_widths) / len(hr_widths)},
            'height': {'min': min(hr_heights), 'max': max(hr_heights), 'mean': sum(hr_heights) / len(hr_heights)}
        },
        'lr_resolution': {
            'width': {'min': min(lr_widths), 'max': max(lr_widths), 'mean': sum(lr_widths) / len(lr_widths)},
            'height': {'min': min(lr_heights), 'max': max(lr_heights), 'mean': sum(lr_heights) / len(lr_heights)}
        },
        'scale_factor': dataset.scale_factor,
        'crop_size': dataset.hr_crop_size,
        'mode': dataset.mode
    }
    logger.info(f"Dataset stats calculated. Total images: {stats['num_images']}.")
    return stats
