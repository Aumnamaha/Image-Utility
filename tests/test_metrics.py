"""Unit tests for segmentation evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import evaluate_segmentation


# ===================================================================
# Fixtures — synthetic binary masks of known overlap
# ===================================================================

@pytest.fixture(scope="module")
def size_32() -> tuple:
    """Return a (gt, pred) pair with 50 % overlap on a 32×32 grid."""
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[:16, :] = 255  # top half foreground
    pred = np.zeros((32, 32), dtype=np.uint8)
    pred[8:24, :] = 255  # middle strip → overlaps gt in rows 8-15 (8 rows)
    return gt, pred


# ===================================================================
# Perfect overlap — IoU=1, Dice=1, Precision=1, Recall=1
# ===================================================================

class TestPerfectOverlap:
    def test_identical_masks(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[:16, :] = 255
        result = evaluate_segmentation(mask, mask.copy())
        assert result["iou"] == pytest.approx(1.0)
        assert result["dice"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_both_all_zeros(self):
        gt = np.zeros((32, 32), dtype=np.uint8)
        pred = np.zeros((32, 32), dtype=np.uint8)
        result = evaluate_segmentation(gt, pred)
        assert result["iou"] == pytest.approx(1.0)
        assert result["dice"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_both_all_ones(self):
        gt = np.ones((32, 32), dtype=np.uint8) * 255
        pred = np.ones((32, 32), dtype=np.uint8) * 255
        result = evaluate_segmentation(gt, pred)
        assert result["iou"] == pytest.approx(1.0)


# ===================================================================
# Zero overlap — IoU=0, Dice=0, Precision=0, Recall=0
# ===================================================================

class TestZeroOverlap:
    def test_disjoint_masks(self):
        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[:16, :] = 255
        pred = np.zeros((32, 32), dtype=np.uint8)
        pred[16:, :] = 255
        result = evaluate_segmentation(gt, pred)
        assert result["iou"] == pytest.approx(0.0)
        assert result["dice"] == pytest.approx(0.0)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)

    def test_one_empty_other_full(self):
        gt = np.zeros((32, 32), dtype=np.uint8)
        pred = np.ones((32, 32), dtype=np.uint8) * 255
        result = evaluate_segmentation(gt, pred)
        assert result["iou"] == pytest.approx(0.0)
        assert result["dice"] == pytest.approx(0.0)
        assert result["precision"] == pytest.approx(0.0)

    def test_full_gt_one_empty(self):
        gt = np.ones((32, 32), dtype=np.uint8) * 255
        pred = np.zeros((32, 32), dtype=np.uint8)
        result = evaluate_segmentation(gt, pred)
        assert result["iou"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)


# ===================================================================
# Partial overlap — verify exact arithmetic
# ===================================================================

class TestPartialOverlap:
    def test_known_overlap(self):
        """32×32 grid, 50 % GT area, partial pred overlap."""
        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[:16, :] = 255  # 512 pixels

        pred = np.zeros((32, 32), dtype=np.uint8)
        pred[8:24, :] = 255  # 512 pixels; overlap rows 8-15 → 256 pixels

        result = evaluate_segmentation(gt, pred)

        # IoU = intersection / union = 256 / (512 + 512 - 256) = 256/768
        assert result["iou"] == pytest.approx(256 / 768, abs=1e-9)
        # Dice = 2*intersection / (gt_sum + pred_sum) = 512 / 1024
        assert result["dice"] == pytest.approx(512 / 1024, abs=1e-9)
        # Precision = intersection / pred_sum = 256/512
        assert result["precision"] == pytest.approx(256 / 512, abs=1e-9)
        # Recall = intersection / gt_sum = 256/512
        assert result["recall"] == pytest.approx(256 / 512, abs=1e-9)

    def test_metrics_are_floats(self):
        gt = np.ones((8, 8), dtype=np.uint8) * 255
        pred = np.ones((8, 8), dtype=np.uint8) * 255
        result = evaluate_segmentation(gt, pred)
        assert isinstance(result["iou"], float)
        assert isinstance(result["dice"], float)
        assert isinstance(result["precision"], float)
        assert isinstance(result["recall"], float)


# ===================================================================
# Shape validation
# ===================================================================

class TestShapeValidation:
    def test_shape_mismatch_raises(self):
        gt = np.zeros((32, 32), dtype=np.uint8)
        pred = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="Shapes must match"):
            evaluate_segmentation(gt, pred)


# ===================================================================
# Metric bounds — all values ∈ [0.0, 1.0]
# ===================================================================

class TestMetricBounds:
    def test_all_metrics_in_range(self):
        rng = np.random.default_rng(42)
        gt = (rng.integers(0, 2, size=(64, 64)) * 255).astype(np.uint8)
        pred = (rng.integers(0, 2, size=(64, 64)) * 255).astype(np.uint8)
        result = evaluate_segmentation(gt, pred)
        for key in ("iou", "dice", "precision", "recall"):
            assert 0.0 <= result[key] <= 1.0 + 1e-9
