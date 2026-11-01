"""Spatial-domain image filtering and enhancement routines."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from skimage.filters import gaussian, median
from skimage.exposure import equalize_adapthist, rescale_intensity


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ArrayLike = Union[np.ndarray, int, float]


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate and normalise an image array.

    Accepts 2-D (grayscale) or 3-D (multi-channel) arrays of dtype ``uint8``
    or ``float32`` / ``float64``.  Returns a **contiguous** copy with the
    validated dtype so downstream skimage calls never see non-contiguous data.

    Raises
    ------
    TypeError
        If *image* is not a NumPy array, is empty, or has an unsupported
        number of dimensions / dtype.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Image array must be non-empty")

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Image must be 2-D (grayscale) or 3-D (multi-channel), got {image.ndim}-D"
        )

    # Accept uint8 / float32 / float64; cast others safely.
    if image.dtype == np.uint8:
        out = image.astype(np.float64, copy=True)
        return out
    elif np.issubdtype(image.dtype, np.floating):
        return np.ascontiguousarray(image.astype(np.float64, copy=True))
    else:
        # Try casting; may raise on overflow for integer types > uint8.
        try:
            return image.astype(np.float64, copy=True)
        except (TypeError, OverflowError) as exc:
            raise TypeError(
                f"Unsupported dtype {image.dtype}; expected uint8 or float*"
            ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    """Apply a Gaussian blur to *image*.

    Parameters
    ----------
    image : np.ndarray
        Input image (2-D grayscale or 3-D multi-channel). Must be uint8 or
        floating-point.
    kernel_size : int
        Half-width of the Gaussian kernel. The full kernel will be
        ``2 * kernel_size + 1``.  Default is ``5`` (kernel width 11).
    sigma : float
        Standard deviation of the Gaussian.  If ``0``, it is computed from
        *kernel_size* automatically.

    Returns
    -------
    np.ndarray
        Blurred image, same shape and dtype class as input (uint8 → uint8,
        float → float64).

    Raises
    ------
    ValueError
        If *kernel_size* is non-positive.
    """
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")

    img = _validate_image(image)

    # Process each channel independently for 3-D images.
    # When sigma is not specified, derive it from kernel_size.
    # skimage needs a non-zero sigma; with sigma=0 the kernel width becomes
    # ``int(truncate * sigma)`` which is 0 → identity operation.
    if sigma <= 0:
        sigma = max(kernel_size / 3.0, 0.1)

    truncate_val = kernel_size / sigma if sigma > 0 else kernel_size

    if img.ndim == 3:
        result = np.empty_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            result[:, :, c] = gaussian(
                img[:, :, c], sigma=sigma, truncate=truncate_val,
            )
    else:
        result = gaussian(img, sigma=sigma, truncate=truncate_val)

    # Cast back to match input dtype class.
    if isinstance(image, np.ndarray) and image.dtype == np.uint8:
        return np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply a median filter to *image*.

    Parameters
    ----------
    image : np.ndarray
        Input image (2-D grayscale or 3-D multi-channel). Must be uint8 or
        floating-point.
    kernel_size : int
        Side length of the square neighbourhood. Default is ``3``.

    Returns
    -------
    np.ndarray
        Filtered image, same shape and dtype class as input.

    Raises
    ------
    ValueError
        If *kernel_size* is non-positive or even (median requires odd kernels).
    """
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    img = _validate_image(image)
    footprint = np.ones((kernel_size, kernel_size), dtype=bool)

    # skimage.median doesn't support 3-D arrays with a 2-D footprint;
    # process each channel independently.
    if img.ndim == 3:
        result = np.empty_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            result[:, :, c] = median(img[:, :, c], footprint=footprint)
    else:
        result = median(img, footprint=footprint)

    if image.dtype == np.uint8:
        return np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    For multi-channel images each channel is equalised independently.

    Parameters
    ----------
    image : np.ndarray
        Input image (2-D or 3-D). Must be uint8 or floating-point in [0, 1].
    clip_limit : float
        Threshold for contrast limiting. Higher values = more contrast. Default ``2.0``.
    tile_grid_size : tuple[int, int]
        Number of tiles in the ``(rows, cols)`` direction. Default ``(8, 8)``.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image with the same shape as *image*.

    Raises
    ------
    ValueError
        If tile grid dimensions are not positive.
    """
    if any(d < 1 for d in tile_grid_size):
        raise ValueError("tile_grid_size dimensions must be >= 1")

    img = _validate_image(image)

    # skimage equalize_adapthist expects float64 in [0, 1] or uint8.
    # Normalise to [0, 1] for consistent behaviour across dtypes.
    if img.dtype == np.float64 and img.max() > 1.0:
        img_norm = img / 255.0
    else:
        img_norm = img

    result = equalize_adapthist(
        img_norm,
        clip_limit=clip_limit,
        kernel_size=tile_grid_size,
    )

    # Cast back to match input dtype class.
    if isinstance(image, np.ndarray) and image.dtype == np.uint8:
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
    return result


def contrast_stretching(
    image: np.ndarray,
    in_range: Tuple[float, float] | None = None,
) -> np.ndarray:
    """Linearly stretch the intensity values of *image*.

    Parameters
    ----------
    image : np.ndarray
        Input image (2-D or 3-D). Must be uint8 or floating-point.
    in_range : tuple[float, float] | None
        ``(low, high)`` input percentiles for clipping.  If ``None``, the
        full data range is used. Default ``None``.

    Returns
    -------
    np.ndarray
        Stretched image scaled to [0, 255] (uint8 output) or [0, 1]
        (float output), same shape as input.

    Raises
    ------
    ValueError
        If *in_range* is given but ``low >= high``.
    """
    img = _validate_image(image)

    if in_range is not None:
        low, high = in_range
        if low >= high:
            raise ValueError(f"in_range low ({low}) must be < high ({high})")

    # Use scikit-image rescale_intensity for robust clipping & scaling.
    # Only pass `in_range` when explicitly set; passing None breaks the call.
    if in_range is not None:
        result = rescale_intensity(img, in_range=in_range)
    else:
        result = rescale_intensity(img)

    if image.dtype == np.uint8:
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
    return result
