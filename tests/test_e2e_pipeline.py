"""End-to-end integration tests for the full water segmentation pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path so src modules are importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.water_segmentation import WaterSegmentationPipeline
from src.metrics import evaluate_segmentation


# ===================================================================
# Fixtures — synthetic satellite scene
# ===================================================================

@pytest.fixture(scope="module")
def synthetic_bands() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (green, nir, ground_truth) for a 256×256 half-water / half-land scene."""
    rng = np.random.default_rng(seed=42)
    h, w = 256, 256

    green = np.zeros((h, w), dtype=np.float64)
    nir = np.zeros((h, w), dtype=np.float64)

    # Water (top half): high green, low NIR → positive NDWI.
    green[: h // 2, :] = rng.uniform(0.3, 0.5, (h // 2, w))
    nir[: h // 2, :] = rng.uniform(0.1, 0.25, (h // 2, w))

    # Land (bottom half): lower green, higher NIR → negative NDWI.
    green[h // 2 :, :] = rng.uniform(0.3, 0.45, (h // 2, w))
    nir[h // 2 :, :] = rng.uniform(0.5, 0.75, (h // 2, w))

    ground_truth = np.zeros((h, w), dtype=np.uint8)
    ground_truth[: h // 2, :] = 255

    return green, nir, ground_truth


# ===================================================================
# Full pipeline — NDWI → Otsu → morphology → metrics → visualization
# ===================================================================

class TestFullPipeline:
    def test_ndwi_computation(self, synthetic_bands):
        green, nir, _ = synthetic_bands
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        assert ndwi.shape == (256, 256)
        assert np.all(ndwi >= -1.0) and np.all(ndwi <= 1.0)

    def test_segmentation_returns_dict(self, synthetic_bands):
        green, nir, _ = synthetic_bands
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = WaterSegmentationPipeline().segment_water(ndwi)
        assert set(result.keys()) == {"raw_ndwi", "threshold_val", "binary_mask", "cleaned_mask"}

    def test_otsu_threshold_extracted(self, synthetic_bands):
        green, nir, _ = synthetic_bands
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = WaterSegmentationPipeline().segment_water(ndwi)
        assert isinstance(result["threshold_val"], float)

    def test_morphological_cleanup(self, synthetic_bands):
        green, nir, _ = synthetic_bands
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = WaterSegmentationPipeline().segment_water(ndwi)
        assert result["cleaned_mask"].shape == result["binary_mask"].shape

    def test_metrics_computed(self, synthetic_bands):
        green, nir, gt = synthetic_bands
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        cleaned = WaterSegmentationPipeline().segment_water(ndwi)["cleaned_mask"]
        metrics = evaluate_segmentation(gt, cleaned)
        assert "iou" in metrics and "dice" in metrics
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_visualization_saves(self, synthetic_bands, tmp_path):
        green, nir, gt = synthetic_bands
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        result = WaterSegmentationPipeline().segment_water(ndwi)

        rgb_dummy = np.stack([green * 255] * 3, axis=-1).astype(np.uint8)
        from src.visualizer import plot_segmentation_results
        path = str(tmp_path / "dashboard.png")
        fig = plot_segmentation_results(
            rgb_image=rgb_dummy,
            ndwi_map=ndwi,
            ground_truth=gt,
            thresholded=result["binary_mask"],
            cleaned=result["cleaned_mask"],
            save_path=path,
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000  # reasonable file size

    def test_water_detected_in_top_half(self, synthetic_bands):
        green, nir, _ = synthetic_bands
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        cleaned = WaterSegmentationPipeline().segment_water(ndwi)["cleaned_mask"]
        top_ratio = np.mean(cleaned[: 128, :] == 255) / 1.0
        assert top_ratio > 0.75

    def test_land_not_detected_as_water(self, synthetic_bands):
        green, nir, _ = synthetic_bands
        ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
        cleaned = WaterSegmentationPipeline().segment_water(ndwi)["cleaned_mask"]
        bottom_ratio = np.mean(cleaned[128:, :] == 255) / 1.0
        assert bottom_ratio < 0.3


# ===================================================================
# CLI --demo mode
# ===================================================================

class TestCLIDemo:
    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "main.py"), "--help"],
            capture_output=True, text=True, cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "--green" in result.stdout

    def test_cli_demo_mode(self):
        """Run --demo and verify JSON output contains expected keys."""
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "main.py"), "--demo", "--out-dir", "."],
            capture_output=True, text=True, cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0
        # JSON report should be in stdout.
        assert "threshold_val" in result.stdout

    def test_cli_demo_saves_files(self):
        """--demo with --save-plots should produce output files."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(_PROJECT_ROOT / "main.py"),
                 "--demo", "--out-dir", tmp, "--save-plots"],
                capture_output=True, text=True, cwd=str(_PROJECT_ROOT),
            )
            assert result.returncode == 0
            assert (Path(tmp) / "dashboard.png").exists()
            assert (Path(tmp) / "ndwi.npy").exists()
            assert (Path(tmp) / "cleaned_mask.npy").exists()
