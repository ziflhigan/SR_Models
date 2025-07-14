"""
Checkpoint management utilities.
"""
from pathlib import Path
import torch

from ..utils.logger import get_logger

logger = get_logger()


def save_checkpoint(path: Path, **kwargs):
    """
    Save checkpoint to a file.

    Args:
        path: Path to save the checkpoint file.
        **kwargs: Dictionary of states to save (e.g., epoch, model_state_dict).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kwargs, path)
    logger.debug(f"Checkpoint saved: {path}")


def load_checkpoint(path: Path, device: torch.device, **kwargs):
    """
    Load states from a checkpoint file.

    Args:
        path: Path to the checkpoint file.
        device: Device to load the checkpoint onto.
        **kwargs: References to objects to load state into (e.g., model, optimizer).
    """
    if not path.exists():
        logger.warning(f"Checkpoint file not found: {path}")
        return 0

    logger.info(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)

    for key, obj in kwargs.items():
        if key in checkpoint and hasattr(obj, 'load_state_dict'):
            try:
                obj.load_state_dict(checkpoint[key])
                logger.debug(f"Loaded state for: {key}")
            except Exception as e:
                logger.error(f"Failed to load state for {key}: {e}")
        else:
            logger.warning(f"State for '{key}' not found in checkpoint or object is not loadable.")

    return checkpoint.get('epoch', 0)
