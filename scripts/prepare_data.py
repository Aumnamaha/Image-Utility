#!/usr/bin/env python3
"""CLI script to convert a satellite band image file into a ``.npy`` array.

Example usage::

    python scripts/prepare_data.py --image data/raw/green.tif \\
                                   --out  data/raw/green_band.npy

If ``--out`` is omitted the ``.npy`` file is written next to the source image
with an appended ``_band`` suffix (e.g. ``green_band.npy``).
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

from src.dataset_utils import load_and_convert_band


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a satellite band image (.tif/.png/.jpg) to a "
                    "normalised 2-D float32 .npy array.",
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input image file (TIFF, PNG, or JPG).",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Output .npy file path. "
             "Defaults to <source_stem>_band.npy in the same directory.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry-point for the CLI script."""
    args = parse_args(argv)

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Derive default output path.
    out_path = args.out
    if out_path is None:
        out_dir = image_path.parent / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{image_path.stem}_band.npy")

    try:
        arr = load_and_convert_band(str(image_path), output_npy_path=out_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Converted {image_path} → {out_path}")
    print(f"  Shape : {arr.shape}")
    print(f"  Dtype : {arr.dtype}")
    print(f"  Range : [{arr.min():.2f}, {arr.max():.2f}]")


if __name__ == "__main__":
    main()
