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

# Core modules (minimal dependencies - numpy, scipy, tqdm, pyyaml only)
from . import utils
from . import evaluation
from . import config
from . import data_generation

# Optional modules (lazy import to avoid requiring optional dependencies)
# These will raise ImportError with helpful message if dependencies missing
def __getattr__(name):
    """Lazy import for optional modules."""
    if name == "io":
        from . import io
        return io
    elif name == "preprocessing":
        from . import preprocessing
        return preprocessing
    elif name == "datasets":
        from . import datasets
        return datasets
    elif name == "models":
        from . import models
        return models
    elif name == "visualization":
        from . import visualization
        return visualization
    elif name == "core":
        from . import core
        return core
    elif name == "scripts":
        from . import scripts
        return scripts
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")