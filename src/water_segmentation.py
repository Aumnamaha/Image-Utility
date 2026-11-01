"""Satellite image water body segmentation via NDWI + Otsu thresholding."""

from __future__ import annotations

import numpy as np
import cv2


class WaterSegmentationPipeline:
    """End-to-end satellite-water extraction pipeline.

    Parameters
    ----------
    kernel_size : tuple[int, int]
        Morphological structuring element for opening / closing. Default ``(3, 3)``.
    epsilon : float
        Small constant added to the NDWI denominator to avoid division by zero.
    """

    def __init__(self, kernel_size: tuple[int, int] = (3, 3), epsilon: float = 1e-5):
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # NDWI computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ndwi(
        green_band: np.ndarray,
        nir_band: np.ndarray,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Compute the Normalized Difference Water Index (NDWI).

        NDWI = (Green - NIR) / (Green + NIR + epsilon)

        Water bodies typically yield positive values (> 0); land yields
        negative or near-zero values.

        Parameters
        ----------
        green_band : np.ndarray
            Green spectral band, same shape as *nir_band*. Positive float.
        nir_band : np.ndarray
            Near-infrared band, same shape as *green_band*. Positive float.
        epsilon : float
            Regularisation constant to prevent division by zero. Default ``1e-5``.

        Returns
        -------
        np.ndarray
            NDWI map with values clamped to ``[-1.0, 1.0]``, dtype ``float64``.

        Raises
        ------
        ValueError
            If input arrays differ in shape or are empty.
        """
        green = np.asarray(green_band, dtype=np.float64)
        nir = np.asarray(nir_band, dtype=np.float64)

        if green.shape != nir.shape:
            raise ValueError(
                f"Green and NIR bands must have the same shape; "
                f"got {green.shape} vs {nir.shape}"
            )
        if green.size == 0:
            raise ValueError("Input bands must be non-empty")

        ndwi = (green - nir) / (green + nir + epsilon)
        return np.clip(ndwi, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Water segmentation
    # ------------------------------------------------------------------

    def segment_water(
        self,
        ndwi_map: np.ndarray,
        kernel_size: tuple[int, int] | None = None,
    ) -> dict:
        """Segment water bodies from an NDWI map using Otsu thresholding.

        Parameters
        ----------
        ndwi_map : np.ndarray
            Pre-computed NDWI array with values in ``[-1, 1]``. Any numeric
            dtype is accepted; the map is normalised internally.
        kernel_size : tuple[int, int] | None
            Morphological structuring element ``(rows, cols)``. Falls back to
            ``self.kernel_size`` when ``None``.

        Returns
        -------
        dict with keys:

        - ``'raw_ndwi'`` (np.ndarray): normalised 8-bit NDWI map.
        - ``'threshold_val'`` (float): Otsu threshold on the 0–255 scale.
        - ``'binary_mask'`` (np.ndarray, uint8): raw Otsu binary mask.
        - ``'cleaned_mask'`` (np.ndarray, uint8): morphologically cleaned mask.

        Raises
        ------
        ValueError
            If *ndwi_map* is empty or not 2-D.
        """
        ndwi = np.asarray(ndwi_map, dtype=np.float64)

        if ndwi.ndim != 2:
            raise ValueError(f"NDWI map must be 2-D, got {ndwi.ndim}-D")
        if ndwi.size == 0:
            raise ValueError("NDWI map must be non-empty")

        # ---- normalise to 8-bit -------------------------------------------------
        # NDWI > 0 → water, < 0 → land. Normalise linearly so higher values are brighter.
        norm = (
            (ndwi - ndwi.min()) / (ndwi.max() - ndwi.min() + self.epsilon) * 255.0
        ).astype(np.uint8)

        # ---- Otsu thresholding ---------------------------------------------------
        thresh_val, binary_mask = cv2.threshold(
            norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # ---- morphological cleanup -----------------------------------------------
        ks = kernel_size if kernel_size is not None else self.kernel_size
        ksize = (min(ks), min(ks))
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)

        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, structuring_element)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, structuring_element)

        return {
            "raw_ndwi": norm,
            "threshold_val": float(thresh_val),
            "binary_mask": binary_mask,
            "cleaned_mask": cleaned_mask,
        }
