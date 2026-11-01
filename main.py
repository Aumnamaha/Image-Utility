#!/usr/bin/env python3
"""Command-line runner for satellite water body segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.config import ImageConfig
from src.water_segmentation import WaterSegmentationPipeline
from src.metrics import evaluate_segmentation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-utility",
        description="Satellite water body segmentation via NDWI + Otsu thresholding.",
    )
    parser.add_argument("--green", type=str, help="Path to green-band image or .npy array")
    parser.add_argument("--nir", type=str, help="Path to NIR-band image or .npy array")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for masks and plots (default: from config)",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save 6-panel dashboard PNG")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a synthetic end-to-end demo without input files",
    )
    return parser


def _load_band(path: str) -> np.ndarray:
    """Load a spectral band from a .npy file."""
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found — {path}", file=sys.stderr)
        sys.exit(1)
    return np.load(p)


def _run_demo(out_dir: str, save_plots: bool) -> dict:
    """Generate synthetic satellite data and run the full pipeline."""
    rng = np.random.default_rng(seed=42)
    h, w = 256, 256

    # --- synthetic bands -------------------------------------------------------
    green = np.zeros((h, w), dtype=np.float64)
    nir = np.zeros((h, w), dtype=np.float64)

    # Top half: water (high green, low NIR → positive NDWI)
    green[: h // 2, :] = rng.uniform(0.3, 0.5, (h // 2, w))
    nir[: h // 2, :] = rng.uniform(0.1, 0.25, (h // 2, w))

    # Bottom half: land (lower green, higher NIR → negative NDWI)
    green[h // 2 :, :] = rng.uniform(0.3, 0.45, (h // 2, w))
    nir[h // 2 :, :] = rng.uniform(0.5, 0.75, (h // 2, w))

    # Ground truth for evaluation
    ground_truth = np.zeros((h, w), dtype=np.uint8)
    ground_truth[: h // 2, :] = 255

    return _run_pipeline(green, nir, ground_truth, out_dir, save_plots)


def _run_real(green_path: str, nir_path: str, ground_truth_path: str | None,
              out_dir: str, save_plots: bool) -> dict:
    """Run pipeline on real input bands."""
    green = _load_band(green_path)
    nir = _load_band(nir_path)

    # Load ground truth if provided (for evaluation only).
    gt = None
    if ground_truth_path and Path(ground_truth_path).exists():
        gt = _load_band(ground_truth_path)

    return _run_pipeline(green, nir, gt, out_dir, save_plots)


def _run_pipeline(
    green: np.ndarray,
    nir: np.ndarray,
    ground_truth: np.ndarray | None,
    out_dir: str,
    save_plots: bool,
) -> dict:
    """Core pipeline execution — shared by demo and real modes."""
    # Ensure output directory exists.
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # --- NDWI computation ------------------------------------------------------
    ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
    print(f"NDWI range: [{ndwi.min():.4f}, {ndwi.max():.4f}]")

    # --- segmentation ----------------------------------------------------------
    pipeline = WaterSegmentationPipeline(kernel_size=(3, 3))
    result = pipeline.segment_water(ndwi)

    threshold_val = result["threshold_val"]
    cleaned_mask = result["cleaned_mask"]

    print(f"Otsu threshold: {threshold_val:.2f}")
    water_pixels = int(cleaned_mask[cleaned_mask > 0].size)
    total_pixels = cleaned_mask.size
    print(f"Water coverage : {water_pixels / total_pixels * 100:.1f}%")

    # --- evaluation ------------------------------------------------------------
    metrics: dict | None = None
    if ground_truth is not None and ground_truth.shape == ndwi.shape:
        metrics = evaluate_segmentation(ground_truth, cleaned_mask)
        print("\nSegmentation Metrics:")
        for key in ("iou", "dice", "precision", "recall"):
            print(f"  {key:>10s}: {metrics[key]:.4f}")

    # --- visualization ---------------------------------------------------------
    if save_plots:
        from src.visualizer import plot_segmentation_results

        rgb_dummy = np.stack([green * 255] * 3, axis=-1).astype(np.uint8)
        fig = plot_segmentation_results(
            rgb_image=rgb_dummy,
            ndwi_map=ndwi,
            ground_truth=ground_truth if ground_truth is not None else cleaned_mask.copy(),
            thresholded=result["binary_mask"],
            cleaned=cleaned_mask,
            save_path=str(Path(out_dir) / "dashboard.png"),
        )

    # --- export masks ------------------------------------------------------------
    np.save(str(Path(out_dir) / "ndwi.npy"), ndwi.astype(np.float32))
    np.save(str(Path(out_dir) / "cleaned_mask.npy"), cleaned_mask)

    report = {
        "threshold_val": threshold_val,
        "water_coverage_pct": round(water_pixels / total_pixels * 100, 2),
        "metrics": metrics,
    }
    print("\n--- JSON Report ---")
    print(json.dumps(report, indent=2))

    return report


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ImageConfig().output_dir
    save_plots = args.save_plots

    if args.demo or (not args.green and not args.nir):
        print("=== DEMO MODE ===\n")
        _run_demo(str(out_dir), save_plots)
    else:
        gt_path = sys.argv[1] if len(sys.argv) > 3 else None
        _run_real(args.green, args.nir, gt_path, str(out_dir), save_plots)


if __name__ == "__main__":
    main()
