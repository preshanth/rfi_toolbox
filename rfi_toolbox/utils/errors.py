"""
Custom exceptions for rfi_toolbox.
"""


class RFIToolboxError(Exception):
    """Base exception for rfi_toolbox."""
    pass


class ConfigValidationError(RFIToolboxError):
    """
    Raised when configuration validation fails.
    
    Used to catch invalid configuration parameters early
    before expensive operations like training or data generation.
    """
    pass


class DataShapeError(RFIToolboxError):
    """
    Raised when data has unexpected shape.
    
    Example: When loading MS data with incompatible dimensions
    or when preprocessing produces wrong-sized patches.
    """
    pass
