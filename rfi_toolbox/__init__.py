"""
rfi_toolbox - Universal foundation for ML-based RFI detection in radio astronomy.

Core modules (always available):
- io: Measurement Set I/O
- evaluation: Segmentation metrics and statistics
- preprocessing: Patchification and normalization
- datasets: PyTorch dataset loaders
- data_generation: Synthetic RFI generation
- config: Configuration loading
- utils: Error handling and utilities

Optional modules (require optional dependencies):
- models: UNet architecture (requires [training])
- visualization: Interactive Bokeh viewers (requires [viz])
- core: RFI simulators (requires [training])
- scripts: CLI entry points (various dependencies)

Example:
    from rfi_toolbox.io import MSLoader
    from rfi_toolbox.evaluation import evaluate_segmentation
"""

__version__ = "0.2.0"
__author__ = "Preshanth Jagannathan"
__email__ = "pjaganna@nrao.edu"

# Eager imports (required for multiprocessing pickle compatibility)
from . import datasets, preprocessing

# Lazy imports for other modules to avoid circular dependencies
def __getattr__(name):
    """Lazy import for optional modules."""
    import importlib
    print(f"[DEBUG __getattr__] Lazy loading: {name}")

    # List of valid lazy-loaded modules
    valid_modules = {
        "utils", "evaluation", "config", "data_generation", "io",
        "models", "visualization", "core", "scripts"
    }

    if name in valid_modules:
        print(f"[DEBUG __getattr__] Calling importlib.import_module for: {name}")
        # Use importlib to avoid triggering __getattr__ recursion
        mod = importlib.import_module(f".{name}", __name__)
        print(f"[DEBUG __getattr__] Successfully imported: {name}")
        # Cache in globals to avoid repeated imports
        globals()[name] = mod
        return mod

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")