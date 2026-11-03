"""Dataset helper module for satellite band image-to-NumPy conversion.

Provides utilities to load single-band or multi-band satellite imagery
(TIFF, PNG, JPG) and convert it into a standardised 2D ``float32`` array
normalised to the ``[0, 255]`` range — ready for downstream NDWI /
segmentation pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_and_convert_band(
    image_path: str,
    output_npy_path: Optional[str] = None,
) -> np.ndarray:
    """Load a satellite band image and convert it to a normalised 2D float32 array.

    Parameters
    ----------
    image_path : str
        Path to the input image file (TIFF, PNG, or JPG supported).
    output_npy_path : str | None, optional
        If provided, save the resulting ``np.ndarray`` as a ``.npy`` binary at
        this path before returning it.

    Returns
    -------
    np.ndarray
        2-D ``float32`` array of shape ``(H, W)`` with values in ``[0, 255]``.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist on disk.
    ValueError
        If the image file is corrupt or cannot be decoded.
    """
    path = Path(image_path)

    # --- existence check ----------------------------------------------------
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # --- read with OpenCV (handles TIFF, PNG, JPG natively) -----------------
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if bgr is None:
        raise ValueError(
            f"Could not decode image file. It may be corrupt or in an "
            f"unsupported format: {image_path}"
        )

    # --- normalise to 2-D float32 [0, 255] ----------------------------------
    if bgr.ndim == 2:
        # Already single-band (grayscale) — keep as-is.
        band = bgr.astype(np.float32)
    elif bgr.ndim == 3:
        channels = bgr.shape[2]
        if channels in (3, 4):
            # Multi-band: convert to grayscale via weighted luminance formula
            # that matches OpenCV's COLOR_BGR2GRAY conversion.
            band = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        elif channels == 1:
            band = bgr[:, :, 0].astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported number of channels ({channels}). "
                f"Expected 1-4 (single-band or RGB/RGBA)."
            )
    else:
        raise ValueError(
            f"Unexpected array dimensions {bgr.shape}. Expected 2-D or 3-D."
        )

    # Normalise to [0, 255] — handles cases where the source is already in a
    # different range (e.g. float64 reflectance values).
    band_min = band.min()
    band_max = band.max()
    if band_max > band_min:
        band = ((band - band_min) / (band_max - band_min)) * 255.0
    # else: constant image — all pixels stay at 0.

    assert band.ndim == 2, f"Expected 2-D array after conversion, got {band.ndim}-D"
    assert band.dtype == np.float32, f"Expected float32 dtype, got {band.dtype}"

    # --- optional export -----------------------------------------------------
    if output_npy_path is not None:
        out = Path(output_npy_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), band)

    return band


def convert_image_to_npy(
    image_path: str,
    output_dir: Optional[str] = None,
) -> str:
    """Convenience wrapper that auto-generates the ``.npy`` output path.

    Parameters
    ----------
    image_path : str
        Path to the input satellite band image.
    output_dir : str | None, optional
        Directory in which to place the output ``.npy`` file. Defaults to the
        parent directory of *image_path*.

    Returns
    -------
    str
        The full path of the saved ``.npy`` file.
    """
    img_name = Path(image_path).stem  # strip original extension
    if output_dir is None:
        out_path = f"{img_name}_band.npy"
    else:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{img_name}_band.npy")

    load_and_convert_band(image_path, out_path)
    return out_path
