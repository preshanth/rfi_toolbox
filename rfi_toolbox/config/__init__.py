"""
Configuration loading and validation for RFI toolbox.

Handles YAML config files for data generation, training, and evaluation.
"""

from .loader import DataConfig, TrainingConfig

__all__ = ["DataConfig", "TrainingConfig"]
