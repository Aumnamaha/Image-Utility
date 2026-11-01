"""Unit tests for color space conversions and range-based masking."""

from __future__ import annotations

import numpy as np
import pytest

from src.color_processing import (
    convert_color_space,
    create_color_mask,
    extract_region,
)


# ===================================================================
# Fixtures — synthetic RGB images in known colour spaces
# ===================================================================

@pytest.fixture(scope="module")
def rgb_solid_red() -> np.ndarray:
    """Solid-red 64×64 image (uint8)."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # R=1.0
    return img


@pytest.fixture(scope="module")
def rgb_solid_blue() -> np.ndarray:
    """Solid-blue 64×64 image (uint8)."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # B=1.0
    return img


@pytest.fixture(scope="module")
def rgb_noise() -> np.ndarray:
    """Random RGB noise 128×128 (uint8)."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def rgb_float() -> np.ndarray:
    """Random RGB in [0, 1] float64."""
    rng = np.random.default_rng(seed=7)
    return rng.uniform(0, 1, size=(64, 64, 3)).astype(np.float64)


@pytest.fixture(scope="module")
def hsv_red() -> np.ndarray:
    """HSV representation of solid-red (H≈0/1, S=0, V=1)."""
    # Red in HSV: H∈[0,1] wraps around; skimage uses H∈[0,1].
    return np.array([[[0.98, 0.0, 1.0]]], dtype=np.float64)


@pytest.fixture(scope="module")
def hsv_blue() -> np.ndarray:
    """HSV representation of solid-blue (H≈2/3)."""
    # Blue in skimage HSV convention is H ≈ 2/3 = 0.666...
    return np.array([[[0.666, 1.0, 1.0]]], dtype=np.float64)


# ===================================================================
# convert_color_space — shape & correctness
# ===================================================================

class TestConvertColorSpace:
    def test_rgb2hsv_preserves_shape(self, rgb_noise):
        result = convert_color_space(rgb_noise, "RGB2HSV")
        assert result.shape == (128, 128, 3)

    def test_rgb2lab_preserves_shape(self, rgb_noise):
        result = convert_color_space(rgb_noise, "RGB2LAB")
        assert result.shape == (128, 128, 3)

    def test_rgb2gray_reduces_dim(self, rgb_noise):
        result = convert_color_space(rgb_noise, "RGB2GRAY")
        assert result.ndim == 2
        assert result.shape == (128, 128)

    def test_rgb2hsv_hue_range(self, rgb_solid_red):
        hsv = convert_color_space(rgb_solid_red, "RGB2HSV")
        # Red hue should be near 0 or 1 in skimage convention.
        h = hsv[:, :, 0]
        assert np.any((h < 0.05) | (h > 0.95))

    def test_rgb2lab_L_range(self, rgb_solid_red):
        lab = convert_color_space(rgb_solid_red, "RGB2LAB")
        L = lab[:, :, 0]
        # Red is bright → L should be high (> 50 on skimage's ~0-100 scale).
        assert np.mean(L) > 50

    def test_rgb2gray_float_input(self, rgb_float):
        result = convert_color_space(rgb_float, "RGB2GRAY")
        assert result.shape == (64, 64)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_invalid_conversion_raises(self, rgb_noise):
        with pytest.raises(ValueError, match="Unknown conversion"):
            convert_color_space(rgb_noise, "HSV2RGB")

    def test_non_rgb_raises(self):
        with pytest.raises(ValueError, match="(H, W, 3)|must be 3-D"):
            convert_color_space(np.zeros((64, 64), dtype=np.uint8), "RGB2GRAY")


# ===================================================================
# create_color_mask — mask generation & boundaries
# ===================================================================

class TestCreateColorMask:
    def test_solid_red_mask_all_white(self):
        hsv = np.full((32, 32, 3), [[0.98, 0.1, 0.9]], dtype=np.float64)
        lower = np.array([0.95, 0.0, 0.5])
        upper = np.array([1.0, 1.0, 1.0])
        mask = create_color_mask(hsv, lower, upper)
        assert mask.shape == (32, 32)
        assert np.all(mask == 255)

    def test_solid_blue_outside_range(self):
        hsv = np.full((32, 32, 3), [[0.666, 1.0, 1.0]], dtype=np.float64)
        lower = np.array([0.90, 0.0, 0.5])
        upper = np.array([1.0, 1.0, 1.0])
        mask = create_color_mask(hsv, lower, upper)
        assert np.all(mask == 0)

    def test_partial_overlap(self):
        hsv = np.zeros((4, 4, 3), dtype=np.float64)
        hsv[0:2, :, 0] = 0.97  # first two rows red-ish hue
        lower = np.array([0.95, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        mask = create_color_mask(hsv, lower, upper)
        assert mask[0:2, :].sum() == 8 * 255   # 2 rows × 4 cols all white
        assert mask[2:4, :].sum() == 0          # black in second half

    def test_lower_upper_mismatch_raises(self):
        hsv = np.zeros((8, 8, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="must match"):
            create_color_mask(hsv, np.array([0.5, 0.5]), np.array([0.8, 0.9, 1.0]))

    def test_bounds_wrong_length_raises(self):
        hsv = np.zeros((8, 8, 3), dtype=np.float64)
        # Different lengths → shape mismatch fires first.
        with pytest.raises(ValueError, match="must match"):
            create_color_mask(hsv, np.array([0.5]), np.array([0.8, 0.9, 1.0]))

    def test_invalid_hsv_shape_raises(self):
        bad = np.zeros((8, 8), dtype=np.float64)
        with pytest.raises(ValueError, match="(H, W, 3)|must be"):
            create_color_mask(bad, [0, 0, 0], [1, 1, 1])


# ===================================================================
# extract_region — bitwise application & shape matching
# ===================================================================

class TestExtractRegion:
    def test_uint8_bitwise(self):
        img = np.ones((16, 16, 3), dtype=np.uint8) * 200
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:, :8] = 255
        result = extract_region(img, mask)
        assert result.shape == img.shape
        # Left half preserved.
        assert np.all(result[:, :8, :] == 200)
        # Right half zeroed.
        assert np.all(result[:, 8:, :] == 0)

    def test_float_multiplication(self):
        img = np.ones((16, 16, 3), dtype=np.float64) * 0.8
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:, :8] = 255
        result = extract_region(img, mask)
        assert np.allclose(result[:, :8], 0.8)
        assert np.all(result[:, 8:] == 0)

    def test_grayscale_mask(self):
        img = np.full((16, 16), 128, dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[8:, :] = 255
        result = extract_region(img, mask)
        assert np.all(result[:8] == 0)
        assert np.all(result[8:] == 128)

    def test_mask_spatial_mismatch_raises(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        with pytest.raises(ValueError, match="does not match"):
            extract_region(img, mask)

    def test_mask_3d_raises(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        bad_mask = np.zeros((16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be 2-D"):
            extract_region(img, bad_mask)


# ===================================================================
# Integration — full pipeline: RGB → HSV → mask → extract
# ===================================================================

class TestPipelineIntegration:
    def test_red_extraction(self):
        # Create a half-red / half-blue image.
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :32, 0] = 255   # left half red
        img[:, 32:, 2] = 255   # right half blue

        hsv = convert_color_space(img, "RGB2HSV")
        # In skimage HSV convention, pure red has H≈0.
        lower = np.array([0.0, 0.2, 0.3])
        upper = np.array([0.1, 1.0, 1.0])
        mask = create_color_mask(hsv, lower, upper)

        extracted = extract_region(img, mask)
        # Left half should have at least one non-zero channel (red preserved).
        assert np.all(np.any(extracted[:, :32] > 0, axis=2))
        assert np.all(extracted[:, 32:] == 0)
