"""
Import smoke tests for rfi_toolbox.

These tests verify that all core modules can be imported without circular dependencies.
Should expose any import issues that manifest in clean environments (like CI).
"""

import pytest


class TestCoreImports:
    """Test that core rfi_toolbox modules import cleanly."""

    def test_package_import(self):
        """Test that rfi_toolbox package imports."""
        try:
            import rfi_toolbox

            assert rfi_toolbox.__version__ is not None
        except ImportError as e:
            pytest.fail(f"Failed to import rfi_toolbox: {e}")

    def test_utils_import(self):
        """Test that rfi_toolbox.utils imports."""
        try:
            from rfi_toolbox.utils import ConfigValidationError, DataShapeError

            assert ConfigValidationError is not None
            assert DataShapeError is not None
        except ImportError as e:
            pytest.fail(
                f"Failed to import rfi_toolbox.utils: {e}\n"
                "This is the first eager import in __init__.py - circular import likely here"
            )

    def test_evaluation_import(self):
        """Test that rfi_toolbox.evaluation imports."""
        try:
            from rfi_toolbox.evaluation import compute_iou, evaluate_segmentation

            assert compute_iou is not None
            assert evaluate_segmentation is not None
        except ImportError as e:
            pytest.fail(f"Failed to import rfi_toolbox.evaluation: {e}")

    def test_config_import(self):
        """Test that rfi_toolbox.config imports."""
        try:
            from rfi_toolbox.config import DataConfig, TrainingConfig

            assert DataConfig is not None
            assert TrainingConfig is not None
        except ImportError as e:
            pytest.fail(f"Failed to import rfi_toolbox.config: {e}")

    def test_data_generation_import(self):
        """Test that rfi_toolbox.data_generation imports."""
        try:
            from rfi_toolbox.data_generation import (
                RawPatchDataset,
                SyntheticDataGenerator,
            )

            assert SyntheticDataGenerator is not None
            assert RawPatchDataset is not None
        except ImportError as e:
            pytest.fail(f"Failed to import rfi_toolbox.data_generation: {e}")

    def test_preprocessing_import(self):
        """Test that rfi_toolbox.preprocessing imports."""
        try:
            from rfi_toolbox.preprocessing import GPUPreprocessor, Preprocessor

            assert Preprocessor is not None
            assert GPUPreprocessor is not None
        except ImportError as e:
            pytest.fail(f"Failed to import rfi_toolbox.preprocessing: {e}")

    def test_datasets_import(self):
        """Test that rfi_toolbox.datasets imports."""
        try:
            from rfi_toolbox.datasets import BatchWriter, RFIMaskDataset, TorchDataset

            assert BatchWriter is not None
            assert TorchDataset is not None
            assert RFIMaskDataset is not None
        except ImportError as e:
            pytest.fail(
                f"Failed to import rfi_toolbox.datasets: {e}\n"
                "Check if sklearn dependency was removed properly"
            )


class TestDirectSubmoduleImports:
    """Test importing submodules directly (bypassing __init__.py)."""

    def test_direct_utils_import(self):
        """Import utils directly from submodule path."""
        try:
            from rfi_toolbox.utils.errors import ConfigValidationError

            assert ConfigValidationError is not None
        except ImportError as e:
            pytest.fail(f"Direct import of utils.errors failed: {e}")

    def test_direct_evaluation_import(self):
        """Import evaluation directly from submodule path."""
        try:
            from rfi_toolbox.evaluation.metrics import compute_iou

            assert compute_iou is not None
        except ImportError as e:
            pytest.fail(f"Direct import of evaluation.metrics failed: {e}")

    def test_direct_preprocessing_import(self):
        """Import preprocessing directly from submodule path."""
        try:
            from rfi_toolbox.preprocessing.preprocessor import Preprocessor

            assert Preprocessor is not None
        except ImportError as e:
            pytest.fail(f"Direct import of preprocessing.preprocessor failed: {e}")

    def test_direct_datasets_import(self):
        """Import datasets directly from submodule path."""
        try:
            from rfi_toolbox.datasets.batched_dataset import BatchWriter, TorchDataset

            assert BatchWriter is not None
            assert TorchDataset is not None
        except ImportError as e:
            pytest.fail(f"Direct import of datasets.batched_dataset failed: {e}")


class TestImportIsolation:
    """Test that imports work in isolation."""

    def test_fresh_import(self):
        """Test import in clean state (removes cached modules)."""
        import sys

        # Remove all rfi_toolbox modules from cache
        to_remove = [key for key in sys.modules if key.startswith("rfi_toolbox")]
        for mod in to_remove:
            del sys.modules[mod]

        # Now import fresh
        try:
            import rfi_toolbox  # noqa: F401

            assert "rfi_toolbox" in sys.modules
        except ImportError as e:
            pytest.fail(f"Fresh import failed: {e}")

    def test_submodule_import_without_package_init(self):
        """Test if submodules can import without triggering package __init__."""
        import sys

        # Remove rfi_toolbox from cache
        to_remove = [key for key in sys.modules if key.startswith("rfi_toolbox")]
        for mod in to_remove:
            del sys.modules[mod]

        # Import a submodule directly
        try:
            from rfi_toolbox.utils.errors import ConfigValidationError  # noqa: F401

            # Check if main package was initialized
            # If __init__ is eager, this will be True
            # If __init__ is lazy, this might be False
            package_initialized = "rfi_toolbox" in sys.modules

            print(
                f"Package initialized after submodule import: {package_initialized}"
            )  # noqa: T201
        except ImportError as e:
            pytest.fail(f"Submodule import failed: {e}")


class TestCircularImportDetection:
    """Specifically test for circular import scenarios."""

    def test_utils_does_not_import_from_package(self):
        """Verify utils module doesn't import back from rfi_toolbox."""
        import sys

        # Clear cache
        to_remove = [key for key in sys.modules if key.startswith("rfi_toolbox")]
        for mod in to_remove:
            del sys.modules[mod]

        try:
            # This should work without triggering full package init
            from rfi_toolbox.utils.errors import RFIToolboxError  # noqa: F401

            # If we get here, check what got imported
            imported = [key for key in sys.modules if key.startswith("rfi_toolbox")]
            print(f"Modules loaded after utils import: {imported}")  # noqa: T201

            # The main package should ideally NOT be initialized yet
            # but it will be if __init__.py has eager imports
        except ImportError as e:
            pytest.fail(f"Utils import triggered circular dependency: {e}")
