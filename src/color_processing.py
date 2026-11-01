"""Color space conversions and range-based color segmentation."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray, rgb2hsv, rgb2lab


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ConversionCode = str  # Literal['RGB2HSV', 'RGB2LAB', 'RGB2GRAY']


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    """Validate that *image* is a 3-D RGB array (H×W×3).

    Accepts ``uint8`` in [0, 255] and float32/float64 in [0.0, 1.0].
    Returns a contiguous ``float64`` copy for safe downstream computation.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"RGB image must be 3-D with shape (H, W, 3), got {image.shape}"
        )

    if image.size == 0:
        raise ValueError("Image array must be non-empty")

    # Normalise to float64 in [0, 1].
    if image.dtype == np.uint8:
        return (image.astype(np.float64) / 255.0).astype(np.float64)
    elif np.issubdtype(image.dtype, np.floating):
        return np.ascontiguousarray(
            np.clip(image.astype(np.float64), 0.0, 1.0)
        )
    else:
        raise TypeError(
            f"Unsupported dtype {image.dtype}; expected uint8 or float*"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_color_space(
    image: np.ndarray,
    conversion_code: str,
) -> np.ndarray:
    """Convert an RGB image to another colour space.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image (H×W×3), uint8 or float in [0, 1].
    conversion_code : str
        One of ``'RGB2HSV'``, ``'RGB2LAB'``, ``'RGB2GRAY'``.

    Returns
    -------
    np.ndarray
        Converted image.  For ``RGB2GRAY`` the result is 2-D (H×W); all others
        are 3-D (H×W×C).

    Raises
    ------
    ValueError
        If *conversion_code* is not recognised or *image* is not valid RGB.
    """
    img = _validate_rgb(image)

    if conversion_code == "RGB2HSV":
        result = rgb2hsv(img)
    elif conversion_code == "RGB2LAB":
        result = rgb2lab(img)
    elif conversion_code == "RGB2GRAY":
        result = rgb2gray(img)
    else:
        raise ValueError(
            f"Unknown conversion '{conversion_code}'; "
            f"expected 'RGB2HSV', 'RGB2LAB', or 'RGB2GRAY'"
        )

    return result


def create_color_mask(
    image_hsv: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> np.ndarray:
    """Create a binary mask from HSV colour range bounds.

    Parameters
    ----------
    image_hsv : np.ndarray
        HSV image (H×W×3), float in [0, 1]. Hue is expected in [0, 1]
        (scikit-image convention).
    lower_bound : np.ndarray
        Lower bound array of shape ``(3,)`` — ``[h_min, s_min, v_min]``.
    upper_bound : np.ndarray
        Upper bound array of shape ``(3,)`` — ``[h_max, s_max, v_max]``.

    Returns
    -------
    np.ndarray
        Binary mask (H×W) with dtype ``uint8``, values 0 or 255.

    Raises
    ------
    ValueError
        If *image_hsv* is not 3-D with 3 channels, or bounds arrays have
        mismatched shapes / are not the same length as each other.
    """
    if image_hsv.ndim != 3 or image_hsv.shape[2] != 3:
        raise ValueError(
            f"HSL image must be (H, W, 3), got {image_hsv.shape}"
        )

    lower = np.asarray(lower_bound, dtype=np.float64)
    upper = np.asarray(upper_bound, dtype=np.float64)

    # Check shape compatibility before individual validity.
    if lower.shape != upper.shape:
        raise ValueError(
            f"lower_bound shape {lower.shape} must match upper_bound shape {upper.shape}"
        )
    if lower.ndim != 1 or len(lower) != 3:
        raise ValueError("lower_bound must be a 1-D array of length 3")
    if upper.ndim != 1 or len(upper) != 3:
        raise ValueError("upper_bound must be a 1-D array of length 3")

    mask = np.ones(image_hsv.shape[:2], dtype=np.uint8) * 255
    for c in range(3):
        channel = image_hsv[:, :, c]
        mask[channel < lower[c]] = 0
        mask[channel > upper[c]] = 0

    return mask


def extract_region(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Extract pixels from *image* where *mask* is non-zero.

    Performs a channel-wise bitwise AND so background pixels (where mask=0)
    become black and foreground pixels are preserved.

    Parameters
    ----------
    image : np.ndarray
        Source image — 2-D grayscale or 3-D multi-channel. uint8 or float.
    mask : np.ndarray
        Binary mask of shape ``(H, W)`` with values in {0, 255}.

    Returns
    -------
    np.ndarray
        Extracted region, same shape and dtype as *image*.

    Raises
    ------
    ValueError
        If *mask* spatial dimensions do not match the image.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D (H×W), got {mask.ndim}-D")

    img_h, img_w = image.shape[:2]
    msk_h, msk_w = mask.shape
    if img_h != msk_h or img_w != msk_w:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image spatial dims ({img_h}, {img_w})"
        )

    # Expand mask to match channels for broadcasting.
    if image.ndim == 3:
        expanded_mask = np.stack([mask] * image.shape[2], axis=-1)
    else:
        expanded_mask = mask

    if image.dtype == np.uint8:
        result = (image.astype(np.int32, copy=True) & expanded_mask.astype(np.int32)).astype(np.uint8)
    else:
        # For float images, multiply by the mask normalised to [0, 1].
        result = image * (expanded_mask / 255.0)

    return result
