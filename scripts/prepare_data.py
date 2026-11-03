#!/usr/bin/env python3
"""CLI script to convert a satellite band image or KML vector file into a ``.npy`` array.

Example usage::

    # Raster images (.tif / .png / .jpg) — converted to normalised float32 bands:
    python scripts/prepare_data.py --image data/raw/green.tif \\
                                   --out  data/raw/green_band.npy

    # KML vector files (Google Earth polygons) — rasterized to binary uint8 masks:
    python scripts/prepare_data.py --image data/ground_truth/water_mask.kml \\
                                   --out  data/ground_truth/water_mask.npy \\
                                   --shape 512 512

If ``--out`` is omitted the ``.npy`` file is written next to the source image
with an appended ``_band`` suffix (e.g. ``green_band.npy``) for raster inputs,
or ``<stem>_mask.npy`` for KML inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src.dataset_utils`` imports work
# regardless of how the script is invoked.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.dataset_utils import convert_image_to_npy, rasterize_kml_mask


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a satellite band image (.tif/.png/.jpg) or KML "
                    "vector file (.kml) to a .npy array.",
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input file (TIFF, PNG, JPG, or KML).",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Output .npy file path. "
             "Defaults to <source_stem>_band.npy for raster / "
             "<source_stem>_mask.npy for KML.",
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Target (height, width) for KML rasterization. Default: 256 256.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry-point for the CLI script."""
    args = parse_args(argv)

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Derive default output path.
    out_path = args.out
    if out_path is None:
        suffix = "_mask.npy" if image_path.suffix.lower() == ".kml" else "_band.npy"
        out_dir = image_path.parent / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{image_path.stem}{suffix}")

    # --- route to the appropriate converter ----------------------------------
    try:
        if image_path.suffix.lower() == ".kml":
            shape = tuple(args.shape) if args.shape else (256, 256)
            arr = rasterize_kml_mask(str(image_path), shape=shape, output_npy_path=out_path)
        else:
            arr = convert_image_to_npy(str(image_path), output_dir=str(Path(out_path).parent))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Converted {image_path} → {out_path}")
    print(f"  Shape : {arr.shape}")
    print(f"  Dtype : {arr.dtype}")
    if arr.dtype == np.uint8:
        unique_vals = sorted(np.unique(arr))
        print(f"  Values: {unique_vals}")
    else:
        print(f"  Range : [{arr.min():.2f}, {arr.max():.2f}]")


if __name__ == "__main__":
    main()
