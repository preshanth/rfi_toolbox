"""
IO utilities for RFI toolbox.

Handles CASA measurement set I/O and synthetic data injection.
"""

print("[DEBUG io/__init__] Starting io module import")

print("[DEBUG io/__init__] Attempting to import MSLoader")
try:
    from .ms_loader import MSLoader

    print("[DEBUG io/__init__] MSLoader imported successfully")
    _HAS_CASA = True
except ImportError as e:
    print(f"[DEBUG io/__init__] MSLoader import failed: {e}")
    _HAS_CASA = False
    MSLoader = None

print("[DEBUG io/__init__] Attempting to import inject_synthetic_data")
try:
    from .ms_injection import inject_synthetic_data

    print("[DEBUG io/__init__] inject_synthetic_data imported successfully")
except ImportError as e:
    print(f"[DEBUG io/__init__] inject_synthetic_data import failed: {e}")
    inject_synthetic_data = None

print("[DEBUG io/__init__] io module import complete")

__all__ = ["MSLoader", "inject_synthetic_data"]
