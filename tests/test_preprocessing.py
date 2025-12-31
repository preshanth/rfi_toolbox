"""
Unit tests for preprocessing module, specifically the torch-based patchify implementation.
"""

import numpy as np
import pytest

from rfi_toolbox.preprocessing.preprocessor import patchify


class TestPatchify:
    """Test suite for torch-based patchify function."""

    def test_basic_patchify_shape(self):
        """Test that patchify produces correct output shape."""
        arr = np.arange(16).reshape(4, 4)
        patches = patchify(arr, (2, 2), step=2)

        assert patches.shape == (2, 2, 2, 2), \
            f"Expected shape (2, 2, 2, 2), got {patches.shape}"

    def test_patch_content(self):
        """Test that patches contain correct values."""
        arr = np.arange(16).reshape(4, 4)
        patches = patchify(arr, (2, 2), step=2)

        # First patch should be top-left 2x2
        expected_first = np.array([[0, 1], [4, 5]])
        np.testing.assert_array_equal(patches[0, 0], expected_first)

        # Last patch should be bottom-right 2x2
        expected_last = np.array([[10, 11], [14, 15]])
        np.testing.assert_array_equal(patches[1, 1], expected_last)

    def test_large_array_patchify(self):
        """Test patchify on realistic large arrays (1024x1024)."""
        large = np.random.rand(1024, 1024)
        patches = patchify(large, (128, 128), step=128)

        assert patches.shape == (8, 8, 128, 128), \
            f"Expected shape (8, 8, 128, 128), got {patches.shape}"

    def test_non_square_patches(self):
        """Test patchify with non-square input."""
        arr = np.arange(24).reshape(6, 4)
        patches = patchify(arr, (2, 2), step=2)

        assert patches.shape == (3, 2, 2, 2), \
            f"Expected shape (3, 2, 2, 2), got {patches.shape}"

    def test_single_patch(self):
        """Test when array is exactly one patch size."""
        arr = np.arange(4).reshape(2, 2)
        patches = patchify(arr, (2, 2), step=2)

        assert patches.shape == (1, 1, 2, 2), \
            f"Expected shape (1, 1, 2, 2), got {patches.shape}"
        np.testing.assert_array_equal(patches[0, 0], arr)

    def test_dtype_preservation(self):
        """Test that patchify preserves array dtype."""
        arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        patches = patchify(arr, (2, 2), step=2)

        assert patches.dtype == np.float32, \
            f"Expected dtype float32, got {patches.dtype}"
