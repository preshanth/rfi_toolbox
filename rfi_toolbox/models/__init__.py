"""
Model architectures for RFI detection.

Provides reference UNet implementation and model definitions.

Example:
    from rfi_toolbox.models import UNet

    model = UNet(in_channels=8, out_channels=1)
"""

from .unet import UNet

__all__ = ["UNet"]