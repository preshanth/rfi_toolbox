"""
IO utilities for RFI toolbox.

Handles CASA measurement set I/O and synthetic data injection.
"""

try:
    from .ms_loader import MSLoader
    _HAS_CASA = True
except ImportError:
    _HAS_CASA = False
    MSLoader = None

from .ms_injection import inject_synthetic_data

__all__ = ["MSLoader", "inject_synthetic_data"]
