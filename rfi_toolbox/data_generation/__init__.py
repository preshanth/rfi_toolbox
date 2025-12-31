"""
Data generation for RFI detection.

Provides synthetic RFI generation with physics-based models.
"""

from .synthetic_generator import SyntheticDataGenerator, RawPatchDataset

__all__ = ["SyntheticDataGenerator", "RawPatchDataset"]
