"""Segmentation evaluation metrics and visualization helpers."""

from __future__ import annotations

import numpy as np


def evaluate_segmentation(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> dict:
    """Compute segmentation quality metrics between *ground_truth* and *prediction*.

    Both arrays are interpreted as binary masks (any non-zero value → foreground).

    Parameters
    ----------
    ground_truth : np.ndarray
        Reference binary mask.
    prediction : np.ndarray
        Predicted binary mask; must have the same shape as *ground_truth*.

    Returns
    -------
    dict with keys:

    - ``'iou'`` (float): Intersection over Union / Jaccard Index.
    - ``'dice'`` (float): Dice coefficient (F1-score).
    - ``'precision'`` (float): TP / (TP + FP).
    - ``'recall'`` (float): TP / (TP + FN).

    Notes
    -----
    When both masks are entirely empty (no foreground pixels in either), all
    metrics return ``1.0`` to indicate perfect agreement (vacuously true).
    If exactly one mask is empty, metrics involving the non-empty side yield
    ``0.0`` (complete disagreement).
    """
    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shapes must match; got {gt.shape} vs {pred.shape}"
        )

    intersection = np.logical_and(gt, pred).sum()
    gt_sum = gt.sum()
    pred_sum = pred.sum()

    # --- zero-division handling -----------------------------------------------
    if gt_sum == 0 and pred_sum == 0:
        # Both empty → vacuous perfect agreement.
        return {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0}

    iou = float(intersection) / (float(gt_sum + pred_sum - intersection)) if (gt_sum + pred_sum - intersection) > 0 else 0.0
    dice = float(2 * intersection) / float(gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 0.0
    precision = float(intersection) / float(pred_sum) if pred_sum > 0 else 0.0
    recall = float(intersection) / float(gt_sum) if gt_sum > 0 else 0.0

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }
