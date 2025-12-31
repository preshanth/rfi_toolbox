"""
Core RFI simulation functionality.

Provides physics-based RFI simulators for generating realistic training data.

Example:
    from rfi_toolbox.core import RFISimulator

    simulator = RFISimulator()
    data, mask = simulator.generate()
"""

from .simulator import RFISimulator

__all__ = ["RFISimulator"]