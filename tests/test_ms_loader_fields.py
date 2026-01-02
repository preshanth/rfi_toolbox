"""
Unit tests for MSLoader field-based selection functionality.

Verifies that field_id parameter and get_available_fields() method are properly
integrated into MSLoader without requiring actual MS data.
"""

import pytest
import inspect

# Direct import to test actual implementation
try:
    from rfi_toolbox.io.ms_loader import MSLoader
    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CASA_AVAILABLE,
    reason="CASA not available - skipping MSLoader tests"
)


class TestMSLoaderFieldFunctionality:
    """Verify field-related API exists and is correct."""

    def test_field_id_in_init(self):
        """__init__ should accept optional field_id parameter."""
        sig = inspect.signature(MSLoader.__init__)
        assert 'field_id' in sig.parameters
        assert sig.parameters['field_id'].default is None

    def test_field_id_in_load(self):
        """load() should accept optional field_id parameter."""
        sig = inspect.signature(MSLoader.load)
        assert 'field_id' in sig.parameters
        assert sig.parameters['field_id'].default is None

    def test_field_id_in_load_single_baseline(self):
        """load_single_baseline() should accept optional field_id parameter."""
        sig = inspect.signature(MSLoader.load_single_baseline)
        assert 'field_id' in sig.parameters
        assert sig.parameters['field_id'].default is None

    def test_get_available_fields_exists(self):
        """get_available_fields() method should exist."""
        assert hasattr(MSLoader, 'get_available_fields')
        assert callable(MSLoader.get_available_fields)

    def test_docstrings_updated(self):
        """Docstrings should mention field functionality."""
        assert MSLoader.__doc__ is not None
        assert 'field' in MSLoader.__doc__.lower()

        assert MSLoader.__init__.__doc__ is not None
        assert 'field_id' in MSLoader.__init__.__doc__.lower()

        assert MSLoader.load.__doc__ is not None
        assert 'field_id' in MSLoader.load.__doc__.lower()

        assert MSLoader.get_available_fields.__doc__ is not None
        assert 'field' in MSLoader.get_available_fields.__doc__.lower()
