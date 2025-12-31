"""
Dataset utilities for RFI detection.

Provides batched streaming datasets for memory-efficient training.
"""

from .batched_dataset import TorchDataset, BatchWriter

__all__ = ["TorchDataset", "BatchWriter"]
