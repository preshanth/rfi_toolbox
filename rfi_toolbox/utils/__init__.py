"""
Utility functions and custom exceptions for RFI Toolbox.

Provides:
- Custom exception classes
- Error handling utilities
- Common helper functions

Example:
    from rfi_toolbox.utils import ConfigValidationError

    raise ConfigValidationError("Invalid configuration")
"""

from .errors import ConfigValidationError, DataShapeError, RFIToolboxError

__all__ = ["ConfigValidationError", "DataShapeError", "RFIToolboxError"]