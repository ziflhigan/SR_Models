"""
Metric calculation and tracking utilities.
"""

import torch
from skimage.metrics import structural_similarity as ssim_skimage
import numpy as np


class MetricTracker:
    """Utility class for tracking training metrics."""

    def __init__(self):
        self.metrics = {}
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(float(value))

    def get_average(self, key: str) -> float:
        """Get average value for a metric."""
        if key in self.metrics and self.metrics[key]:
            return sum(self.metrics[key]) / len(self.metrics[key])
        return 0.0

    def get_current(self, key: str) -> float:
        """Get last recorded value for a metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return 0.0


def calculate_psnr(pred: torch.Tensor,
                   target: torch.Tensor,
                   max_val: float = 255.0,
                   norm_type: str = "minus_one_one") -> float:
    """
    Calculate Peak Signal-to-Noise Ratio with configurable de-normalization.
    """
    # De-normalize tensors to the [0, 255] range before calculation
    pred = _de_normalize(pred, norm_type).clamp(0, max_val)
    target = _de_normalize(target, norm_type).clamp(0, max_val)

    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(pred: torch.Tensor,
                   target: torch.Tensor,
                   max_val: float = 255.0,
                   norm_type: str = "minus_one_one") -> float:
    """
    Calculate Structural Similarity Index (SSIM) for a batch of images.
    Uses scikit-image for a robust implementation.
    """
    # De-normalize tensors to the [0, 255] range and move to CPU
    pred = _de_normalize(pred.cpu(), norm_type).clamp(0, max_val)
    target = _de_normalize(target.cpu(), norm_type).clamp(0, max_val)

    # Handle batches by iterating and averaging
    if pred.dim() == 4:
        ssim_scores = []
        for i in range(pred.shape[0]):
            pred_img = pred[i]
            target_img = target[i]

            # Permute from (C, H, W) to (H, W, C) for scikit-image
            pred_np = pred_img.permute(1, 2, 0).detach().numpy()
            target_np = target_img.permute(1, 2, 0).detach().numpy()

            multichannel = pred_np.shape[2] > 1

            ssim_val = ssim_skimage(
                target_np,
                pred_np,
                data_range=max_val,
                channel_axis=2 if multichannel else -1
            )
            ssim_scores.append(ssim_val)

        return float(np.mean(ssim_scores))

    # Handle a single image (3D tensor)
    elif pred.dim() == 3:
        pred_np = pred.permute(1, 2, 0).detach().numpy()
        target_np = target.permute(1, 2, 0).detach().numpy()

        multichannel = pred_np.shape[2] > 1

        return float(ssim_skimage(
            target_np,
            pred_np,
            data_range=max_val,
            channel_axis=2 if multichannel else -1
        ))
    else:
        raise ValueError(f"Unsupported tensor dimension for SSIM: {pred.dim()}. Expected 3 or 4.")


def _de_normalize(tensor: torch.Tensor, norm_type: str) -> torch.Tensor:
    """De-normalizes a tensor from a given range to [0, 255]."""
    if norm_type == "minus_one_one":
        # De-normalize from [-1, 1] to [0, 255]
        return tensor.mul(127.5).add(127.5)
    elif norm_type == "zero_one":
        # De-normalize from [0, 1] to [0, 255]
        return tensor.mul(255.0)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")
