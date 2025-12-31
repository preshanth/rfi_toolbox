"""
Preprocessing utilities for RFI detection.

Handles patchification, augmentation, normalization, and feature extraction.
"""

from .preprocessor import Preprocessor, GPUPreprocessor

__all__ = ["Preprocessor", "GPUPreprocessor"]
