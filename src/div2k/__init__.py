"""
Dataset implementations for super-resolution models.
"""

from .div2k import DIV2KDataset, create_data_loaders

__all__ = ['DIV2KDataset', 'create_data_loaders']