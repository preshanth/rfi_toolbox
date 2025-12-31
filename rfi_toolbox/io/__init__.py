"""
IO utilities for RFI toolbox.

Handles CASA measurement set I/O and synthetic data injection.
"""

from .ms_loader import MSLoader
from .ms_injection import inject_synthetic_data

__all__ = ["MSLoader", "inject_synthetic_data"]
