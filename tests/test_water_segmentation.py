"""Unit tests for NDWI-based water body segmentation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.water_segmentation import WaterSegmentationPipeline


# ===================================================================
# Fixtures — synthetic satellite bands
# ===================================================================

@pytest.fixture(scope="module")
def green_band() -> np.ndarray:
    """64×64 green band, float64."""
    rng = np.random.default_rng(seed=10)
    return rng.uniform(0.2, 0.7, size=(64, 64)).astype(np.float64)


@pytest.fixture(scope="module")
def nir_band() -> np.ndarray:
    """64×64 NIR band, float64."""
    rng = np.random.default_rng(seed=11)
    return rng.uniform(0.3, 0.9, size=(64, 64)).astype(np.float64)


@pytest.fixture(scope="module")
def synthetic_satellite() -> tuple[np.ndarray, np.ndarray]:
    """Half-water / half-land satellite scene (128×128)."""
    h = 128
    green = np.zeros((h, h), dtype=np.float64)
    nir = np.zeros((h, h), dtype=np.float64)

    # Top half: water — high green (~0.35), low NIR (~0.15)
    green[: h // 2, :] = 0.35 + np.random.default_rng(20).uniform(-0.05, 0.05, (h // 2, h))
    nir[: h // 2, :] = 0.15 + np.random.default_rng(21).uniform(-0.05, 0.05, (h // 2, h))

    # Bottom half: land — lower green (~0.4), higher NIR (~0.6)
    green[h // 2 :, :] = 0.4 + np.random.default_rng(22).uniform(-0.1, 0.1, (h // 2, h))
    nir[h // 2 :, :] = 0.6 + np.random.default_rng(23).uniform(-0.1, 0.1, (h // 2, h))

    return green, nir


@pytest.fixture(scope="module")
def pipeline() -> WaterSegmentationPipeline:
    """Default pipeline instance."""
    return WaterSegmentationPipeline(kernel_size=(3, 3), epsilon=1e-5)


# ===================================================================
# NDWI computation tests
# ===================================================================

class TestComputeNDWI:
    def test_shape_preserved(self, green_band, nir_band):
        ndwi = WaterSegmentationPipeline.compute_ndwi(green_band, nir_band)
        assert ndwi.shape == green_band.shape

    def test_output_in_range(self, green_band, nir_band):
        ndwi = WaterSegmentationPipeline.compute_ndwi(green_band, nir_band)
        assert np.all(ndwi >= -1.0) and np.all(ndwi <= 1.0)

    def test_water_like_positive(self):
        """High green / low NIR → positive NDWI (water)."""
        g = np.full((32, 32), 0.4, dtype=np.float64)
        n = np.full((32, 32), 0.1, dtype=np.float64)
        ndwi = WaterSegmentationPipeline.compute_ndwi(g, n)
        assert np.mean(ndwi) > 0

    def test_land_like_negative(self):
        """Low green / high NIR → negative NDWI (land)."""
        g = np.full((32, 32), 0.2, dtype=np.float64)
        n = np.full((32, 32), 0.8, dtype=np.float64)
        ndwi = WaterSegmentationPipeline.compute_ndwi(g, n)
        assert np.mean(ndwi) < 0

    def test_equal_bands_yield_zero(self):
        g = np.ones((16, 16), dtype=np.float64) * 0.5
        n = np.ones((16, 16), dtype=np.float64) * 0.5
        ndwi = WaterSegmentationPipeline.compute_ndwi(g, n)
        assert np.allclose(ndwi, 0.0)

    def test_zero_bands_uses_epsilon(self):
        """When both bands are zero, epsilon prevents division by zero."""
        g = np.zeros((16, 16), dtype=np.float64)
        n = np.zeros((16, 16), dtype=np.float64)
        ndwi = WaterSegmentationPipeline.compute_ndwi(g, n, epsilon=1e-5)
        assert not np.any(np.isnan(ndwi))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            WaterSegmentationPipeline.compute_ndwi(
                np.zeros((32, 32)), np.zeros((64, 64))
            )

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            WaterSegmentationPipeline.compute_ndwi(np.array([], dtype=np.float64), np.array([]))


# ===================================================================
# segment_water — structure & Otsu extraction
# ===================================================================

class TestSegmentWater:
    def test_returns_expected_keys(self, pipeline, synthetic_satellite):
        green, nir = synthetic_satellite
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = pipeline.segment_water(ndwi)
        assert set(result.keys()) == {"raw_ndwi", "threshold_val", "binary_mask", "cleaned_mask"}

    def test_raw_ndwi_8bit(self, pipeline, synthetic_satellite):
        green, nir = synthetic_satellite
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = pipeline.segment_water(ndwi)
        assert result["raw_ndwi"].dtype == np.uint8
        assert 0 <= result["raw_ndwi"].min() and result["raw_ndwi"].max() <= 255

    def test_otsu_threshold_is_float(self, pipeline, synthetic_satellite):
        green, nir = synthetic_satellite
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = pipeline.segment_water(ndwi)
        assert isinstance(result["threshold_val"], float)

    def test_binary_mask_uint8(self, pipeline, synthetic_satellite):
        green, nir = synthetic_satellite
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = pipeline.segment_water(ndwi)
        assert result["binary_mask"].dtype == np.uint8

    def test_cleaned_mask_same_shape(self, pipeline, synthetic_satellite):
        green, nir = synthetic_satellite
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = pipeline.segment_water(ndwi)
        assert result["cleaned_mask"].shape == result["binary_mask"].shape

    def test_cleaning_changes_mask(self, pipeline):
        """Morphological operations should alter at least some pixels on a noisy map."""
        rng = np.random.default_rng(99)
        ndwi_noisy = np.zeros((64, 64), dtype=np.float64)
        # Add scattered noise patches.
        ndwi_noisy[10:20, 10:20] = -0.3   # water-like patch
        ndwi_noisy[50:, :] = rng.uniform(-0.8, -0.1, (14, 64))  # more water

        result = pipeline.segment_water(ndwi_noisy)
        raw = result["binary_mask"]
        cleaned = result["cleaned_mask"]
        # Opening + closing should differ from raw for noisy data.
        assert not np.array_equal(raw, cleaned)

    def test_ndwi_not_2d_raises(self):
        pipeline = WaterSegmentationPipeline()
        with pytest.raises(ValueError, match="must be 2-D"):
            pipeline.segment_water(np.zeros((16, 16, 3)))

    def test_empty_ndwi_raises(self):
        pipeline = WaterSegmentationPipeline()
        with pytest.raises(ValueError, match="non-empty"):
            pipeline.segment_water(np.zeros((0, 10), dtype=np.float64))


# ===================================================================
# Integration — synthetic satellite scene
# ===================================================================

class TestIntegration:
    def test_water_detected_in_top_half(self, pipeline, synthetic_satellite):
        """Top half of the synthetic scene is water → cleaned mask should be active there."""
        green, nir = synthetic_satellite
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = pipeline.segment_water(ndwi)
        cleaned = result["cleaned_mask"]

        # Top half: majority of pixels should be classified as water (255).
        top_water_ratio = np.mean(cleaned[: 64, :] == 255) / 1.0  # max possible
        assert top_water_ratio > 0.85   # expect > 85% water coverage

    def test_land_not_detected_as_water(self, pipeline, synthetic_satellite):
        """Bottom half is land → cleaned mask should be mostly zero there."""
        green, nir = synthetic_satellite
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = pipeline.segment_water(ndwi)
        cleaned = result["cleaned_mask"]

        bottom_water_ratio = np.mean(cleaned[64:, :] == 255) / 1.0
        assert bottom_water_ratio < 0.3   # expect < 30% false positives

    def test_custom_kernel_size(self, synthetic_satellite):
        pipeline = WaterSegmentationPipeline(kernel_size=(5, 5))
        green, nir = synthetic_satellite
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = pipeline.segment_water(ndwi, kernel_size=(7, 7))
        # Larger kernel → more aggressive cleanup.
        assert result["cleaned_mask"].shape == (128, 128)
