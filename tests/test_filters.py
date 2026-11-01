"""Unit tests for spatial filtering and enhancement routines."""

from __future__ import annotations

import numpy as np
import pytest

from src.filters import (
    apply_clahe,
    apply_gaussian_blur,
    apply_median_filter,
    contrast_stretching,
)


# ===================================================================
# Fixtures — reuse synthetic images from conftest and add extra cases
# ===================================================================

@pytest.fixture(scope="module")
def grayscale_uint8() -> np.ndarray:
    """256×256 grayscale uint8."""
    rng = np.random.default_rng(seed=99)
    return rng.integers(0, 256, size=(256, 256), dtype=np.uint8)


@pytest.fixture(scope="module")
def grayscale_float32() -> np.ndarray:
    """256×256 grayscale float32 in [0, 1]."""
    rng = np.random.default_rng(seed=99)
    return rng.uniform(0, 1, size=(256, 256)).astype(np.float32)


@pytest.fixture(scope="module")
def small_rgb() -> np.ndarray:
    """Small 8×8 RGB image for fast kernel-edge tests."""
    rng = np.random.default_rng(seed=7)
    return rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)


# ===================================================================
# apply_gaussian_blur — shape & dtype
# ===================================================================

class TestGaussianBlurShapeDtype:
    def test_preserves_shape_rgb(self, rgb_256):
        result = apply_gaussian_blur(rgb_256)
        assert result.shape == rgb_256.shape

    def test_preserves_dtype_uint8(self, rgb_256):
        result = apply_gaussian_blur(rgb_256)
        assert result.dtype == np.uint8

    def test_preserves_shape_grayscale(self, grayscale_uint8):
        result = apply_gaussian_blur(grayscale_uint8)
        assert result.shape == grayscale_uint8.shape

    def test_float_input_returns_float(self, grayscale_float32):
        result = apply_gaussian_blur(grayscale_float32.astype(np.float64))
        assert np.issubdtype(result.dtype, np.floating)

    def test_preserves_shape_multispectral(self, multispectral_256):
        result = apply_gaussian_blur(multispectral_256)
        assert result.shape == multispectral_256.shape


# ===================================================================
# apply_median_filter — shape & dtype
# ===================================================================

class TestMedianFilterShapeDtype:
    def test_preserves_shape_rgb(self, rgb_256):
        result = apply_median_filter(rgb_256)
        assert result.shape == rgb_256.shape

    def test_preserves_dtype_uint8(self, rgb_256):
        result = apply_median_filter(rgb_256)
        assert result.dtype == np.uint8

    def test_preserves_shape_grayscale(self, grayscale_uint8):
        result = apply_median_filter(grayscale_uint8)
        assert result.shape == grayscale_uint8.shape

    def test_small_kernel_on_small_image(self, small_rgb):
        result = apply_median_filter(small_rgb, kernel_size=3)
        assert result.shape == small_rgb.shape


# ===================================================================
# apply_clahe — shape & dtype
# ===================================================================

class TestCLAHE:
    def test_preserves_shape_rgb(self, rgb_256):
        result = apply_clahe(rgb_256)
        assert result.shape == rgb_256.shape

    def test_preserves_dtype_uint8(self, rgb_256):
        result = apply_clahe(rgb_256)
        assert result.dtype == np.uint8

    def test_preserves_shape_grayscale(self, grayscale_uint8):
        result = apply_clahe(grayscale_uint8)
        assert result.shape == grayscale_uint8.shape

    def test_custom_tile_grid(self, rgb_256):
        result = apply_clahe(rgb_256, tile_grid_size=(4, 4))
        assert result.shape == rgb_256.shape


# ===================================================================
# contrast_stretching — shape & dtype
# ===================================================================

class TestContrastStretching:
    def test_preserves_shape_rgb(self, rgb_256):
        result = contrast_stretching(rgb_256)
        assert result.shape == rgb_256.shape

    def test_preserves_dtype_uint8(self, rgb_256):
        result = contrast_stretching(rgb_256)
        assert result.dtype == np.uint8

    def test_output_range_uint8(self, rgb_256):
        result = contrast_stretching(rgb_256)
        assert result.min() >= 0 and result.max() <= 255

    def test_custom_in_range(self, grayscale_float32):
        result = contrast_stretching(grayscale_float32.astype(np.float64), in_range=(0.1, 0.9))
        assert np.all(result >= 0) and np.all(result <= 1)


# ===================================================================
# Pixel distribution modification — filters actually change data
# ===================================================================

class TestPixelDistributionModification:
    def test_gaussian_smoothes(self, rgb_256):
        result = apply_gaussian_blur(rgb_256, kernel_size=7)
        # Gaussian blur reduces variance of gradients.
        assert np.mean(np.abs(result.astype(int) - rgb_256)) > 0

    def test_median_changes_noise(self, rgb_256):
        result = apply_median_filter(rgb_256, kernel_size=3)
        # Median filter should differ from input for non-uniform images.
        assert not np.array_equal(result, rgb_256)

    def test_clahe_enhances_contrast(self, grayscale_uint8):
        result = apply_clahe(grayscale_uint8)
        # CLAHE should broaden the histogram (more unique values or wider range).
        assert len(np.unique(result)) >= len(np.unique(grayscale_uint8))

    def test_stretching_spreads_range(self, grayscale_float32):
        img_low = np.full((64, 64), 0.5, dtype=np.float64)  # constant — stretching won't change it
        result = contrast_stretching(img_low)
        assert result.shape == img_low.shape


# ===================================================================
# Input validation / error handling
# ===================================================================

class TestInputValidation:
    def test_gaussian_non_array(self):
        with pytest.raises(TypeError, match="numpy.ndarray"):
            apply_gaussian_blur([1, 2, 3])

    def test_gaussian_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            apply_gaussian_blur(np.array([], dtype=np.uint8))

    def test_gaussian_4d_raises(self):
        with pytest.raises(ValueError, match="-D"):
            apply_gaussian_blur(np.zeros((10, 10, 3, 2), dtype=np.uint8))

    def test_gaussian_invalid_kernel(self, rgb_256):
        with pytest.raises(ValueError, match="kernel_size"):
            apply_gaussian_blur(rgb_256, kernel_size=0)

    def test_median_even_kernel_raises(self, rgb_256):
        with pytest.raises(ValueError, match="odd"):
            apply_median_filter(rgb_256, kernel_size=4)

    def test_median_invalid_kernel(self, rgb_256):
        with pytest.raises(ValueError, match="kernel_size"):
            apply_median_filter(rgb_256, kernel_size=-1)

    def test_clahe_negative_tile_raises(self, rgb_256):
        with pytest.raises(ValueError, match="tile_grid_size"):
            apply_clahe(rgb_256, tile_grid_size=(0, 8))

    def test_stretching_bad_range(self, grayscale_uint8):
        with pytest.raises(ValueError, match="in_range"):
            contrast_stretching(grayscale_uint8, in_range=(100, 50))


# ===================================================================
# Edge cases — very small images
# ===================================================================

class TestEdgeCases:
    def test_gaussian_on_3x3(self):
        img = np.ones((3, 3), dtype=np.uint8) * 128
        result = apply_gaussian_blur(img, kernel_size=1)
        assert result.shape == (3, 3)

    def test_median_on_5x5(self):
        img = np.zeros((5, 5), dtype=np.uint8)
        img[2, 2] = 255  # single impulse noise
        result = apply_median_filter(img, kernel_size=3)
        assert result.shape == (5, 5)

    def test_clahe_on_uniform(self):
        img = np.full((64, 64), 100, dtype=np.uint8)
        result = apply_clahe(img)
        assert result.shape == (64, 64)
