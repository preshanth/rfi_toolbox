"""
Dataset utilities for RFI detection.

Provides batched streaming datasets for memory-efficient training.
"""

from .batched_dataset import TorchDataset, BatchWriter
from .rfi_mask_dataset import RFIMaskDataset

__all__ = ["TorchDataset", "BatchWriter", "RFIMaskDataset"]
