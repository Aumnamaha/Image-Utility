"""Visualization helpers for segmentation results."""

from __future__ import annotations

import numpy as np


def _ensure_uint8(val: np.ndarray) -> np.ndarray:
    """Normalise an array to uint8 in [0, 255] for display."""
    if val.dtype == np.uint8:
        return val
    arr = val.astype(np.float64)
    mn, mx = arr.min(), arr.max()
    rng = mx - mn if mx > mn else 1.0
    return np.clip(((arr - mn) / rng) * 255.0, 0, 255).astype(np.uint8)


def plot_segmentation_results(
    rgb_image: np.ndarray,
    ndwi_map: np.ndarray,
    ground_truth: np.ndarray,
    thresholded: np.ndarray,
    cleaned: np.ndarray,
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":  # noqa: F821 — type resolved at runtime
    """Render a 6-panel dashboard of segmentation results.

    Panels (row × col):

    .. code-block:: text

        ┌──────────┬───────────┬──────────────┐
        │ RGB      │ NDWI map  │ Ground Truth │
        ├──────────┼───────────┼──────────────┤
        │ Threshold│ Cleaned   │ Overlay      │
        └──────────┴───────────┴──────────────┘

    Parameters
    ----------
    rgb_image : np.ndarray
        Original RGB image (H×W×3), uint8 or float.
    ndwi_map : np.ndarray
        Normalised NDWI map, 2-D float.
    ground_truth : np.ndarray
        Reference binary mask (H×W).
    thresholded : np.ndarray
        Raw Otsu-thresholded mask.
    cleaned : np.ndarray
        Morphologically cleaned mask.
    save_path : str | None
        If provided, save the figure to this path before returning.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered Figure object (use ``fig.show()`` or ``plt.show()``).
    """
    import matplotlib
    matplotlib.use("Agg")  # headless backend — no GUI required
    import matplotlib.pyplot as plt

    h, w = rgb_image.shape[:2]

    # Normalise inputs for display.
    rgb_disp = _ensure_uint8(rgb_image) if rgb_image.ndim == 3 else np.stack([_ensure_uint8(rgb_image)] * 3, axis=-1)
    ndwi_disp = _ensure_uint8(ndwi_map)
    gt_disp = ground_gt = ground_truth.astype(bool)
    thresh_disp = _ensure_uint8(thresholded)
    clean_disp = _ensure_uint8(cleaned)

    # Overlay: cleaned mask over RGB (green tint for water).
    overlay = rgb_disp.copy()
    overlay[ground_gt, :3] = np.array([0, 255, 0], dtype=np.uint8)  # green channel

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    titles = [
        "RGB Input",
        "NDWI Map",
        "Ground Truth",
        "Thresholded Mask",
        "Cleaned Mask",
        "Overlay (GT highlighted)",
    ]

    for ax, data, title in zip(axes.ravel(), [
        rgb_disp, ndwi_disp, gt_disp.astype(np.uint8) * 255,
        thresh_disp, clean_disp, overlay,
    ], titles):
        ax.imshow(data, cmap="gray" if data.ndim < 3 else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("Segmentation Results Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
