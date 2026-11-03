"""Dataset helper module for satellite band image-to-NumPy conversion and KML vector parsing.

Provides utilities to:

1. Load single-band or multi-band satellite imagery (TIFF, PNG, JPG) and convert
   it into a standardised 2D ``float32`` array normalised to the ``[0, 255]``
   range — ready for downstream NDWI / segmentation pipelines.
2. Parse Google Earth ``.kml`` vector files, extract coordinate polygons,
   and rasterize them into 2-D binary uint8 masks (0 = background / 255 = foreground)
   matching a target image matrix shape for ground-truth evaluation.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

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
            f"Could not decode image file. It may be corrupt or an "
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


# ---------------------------------------------------------------------------
# KML polygon parsing & rasterization
# ---------------------------------------------------------------------------

_KML_NS = "http://www.opengis.net/kml/2.2"
_KML_TAG = lambda tag: f"{{{_KML_NS}}}{tag}"  # e.g. {http://...}Polygon


def _parse_coordinates(text: str) -> np.ndarray:
    """Parse a KML ``<coordinates>`` text block into an ``(N, 2)`` float64 array.

    Expected format::

        lon1,lat1,z1\\nlon2,lat2,z2\\n...

    Parameters
    ----------
    text : str
        Raw coordinate string from the KML XML element.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` with columns ``[longitude, latitude]``.
    """
    if not text or not text.strip():
        return np.empty((0, 2), dtype=np.float64)

    pairs: List[Tuple[float, float]] = []
    for token in text.strip().split():
        parts = token.strip(",\\n\\r").split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                pairs.append((lon, lat))
            except ValueError:
                continue
    return np.array(pairs, dtype=np.float64)


def parse_kml_polygons(kml_path: str) -> List[np.ndarray]:
    """Parse a KML file and extract coordinate arrays for all polygons.

    Traverses the XML tree looking for ``<Polygon>`` elements (including those
    inside ``<Placemark>``).  Each polygon's outer boundary is taken from its
    innermost ``<LinearRing>`` (or falls back to a direct ``<coordinates>``
    child of the ``<Polygon>``).

    Parameters
    ----------
    kml_path : str
        Path to the input ``.kml`` file.

    Returns
    -------
    list[np.ndarray]
        A list of coordinate arrays, each of shape ``(N, 2)`` with columns
        ``[longitude, latitude]``.  Polygons are returned in document order.

    Raises
    ------
    FileNotFoundError
        If *kml_path* does not exist on disk.
    ValueError
        If the KML file contains no valid polygon elements.
    """
    path = Path(kml_path)
    if not path.is_file():
        raise FileNotFoundError(f"KML file not found: {kml_path}")

    try:
        tree = ET.parse(str(path))
    except ET.ParseError as exc:
        raise ValueError(
            f"Invalid XML in KML file: {kml_path} — {exc}"
        ) from exc

    root = tree.getroot()

    polygons: List[np.ndarray] = []

    # Helper to collect elements with a given tag, deduplicating by id.
    def _find_all(parent, *queries):
        """Collect elements matching any of the given queries, deduped."""
        found: list[ET.Element] = []
        seen: set[int] = set()
        for q in queries:
            for el in parent.findall(q):
                if id(el) not in seen:
                    seen.add(id(el))
                    found.append(el)
        return found

    # Find all Polygon elements (try both namespaced and namespace-less).
    polygon_elements = _find_all(
        root,
        f".//{_KML_TAG('Polygon')}",
        ".//Polygon",  # fallback: no namespace
    )

    for poly_el in polygon_elements:
        # Try LinearRing first (most common structured KML).
        ring_els = _find_all(
            poly_el,
            f".//{_KML_TAG('LinearRing')}",
            ".//LinearRing",
        )

        coords_text = ""
        if ring_els:
            # Use the first (outer) LinearRing.
            for ring_el in ring_els:
                coord_el = _find_all(
                    ring_el,
                    f".//{_KML_TAG('coordinates')}",
                    ".//coordinates",
                )
                if coord_el and isinstance(coord_el[0], ET.Element) and coord_el[0].text:
                    coords_text = coord_el[0].text.strip()
                    break
        else:
            # Fallback: direct <coordinates> child of Polygon.
            coord_el = _find_all(
                poly_el,
                f".{_KML_TAG('coordinates')}",
                "./{_KML_TAG('coordinates')}",
            )
            if coord_el and isinstance(coord_el[0], ET.Element) and coord_el[0].text:
                coords_text = coord_el[0].text.strip()

        if coords_text:
            arr = _parse_coordinates(coords_text)
            # Close the ring: KML spec requires last point == first point,
            # but we enforce it defensively.
            if len(arr) > 0 and not np.allclose(arr[0], arr[-1]):
                arr = np.vstack([arr, arr[0]])
            polygons.append(arr)

    if not polygons:
        raise ValueError(
            f"No valid polygon elements found in {kml_path}. "
            f"Expected <Polygon> or <LinearRing> with <coordinates>."
        )

    return polygons


def rasterize_kml_mask(
    kml_path: str,
    shape: Tuple[int, int] = (256, 256),
    output_npy_path: Optional[str] = None,
) -> np.ndarray:
    """Rasterize KML polygon coordinates onto a 2-D binary mask.

    The function parses the KML file, extracts polygon coordinate arrays in
    ``(lon, lat)`` form, and maps them onto pixel coordinates within ``shape``
    ``(height, width)``.  Polygon interiors are filled with **255** (water /
    foreground); everything else is **0** (background).

    Parameters
    ----------
    kml_path : str
        Path to the input ``.kml`` file.
    shape : tuple[int, int], optional
        Target ``(height, width)`` of the output mask. Default ``(256, 256)``.
    output_npy_path : str | None, optional
        If provided, save the resulting ``uint8`` array as a ``.npy`` binary at
        this path before returning it.

    Returns
    -------
    np.ndarray
        2-D ``uint8`` mask of shape ``(height, width)`` with values in ``{0, 255}``.

    Raises
    ------
    FileNotFoundError
        If *kml_path* does not exist on disk.
    ValueError
        If the KML file contains no valid polygon elements or coordinate data.
    """
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple (height, width)")

    polygons = parse_kml_polygons(kml_path)

    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)

    # Determine bounding box of all polygon coordinates.
    all_coords = np.vstack(polygons) if polygons else np.empty((0, 2))
    if len(all_coords) == 0:
        raise ValueError("No coordinate data to rasterize")

    lon_min, lat_max = float(all_coords[:, 0].min()), float(all_coords[:, 1].max())
    lon_max, lat_min = float(all_coords[:, 0].max()), float(all_coords[:, 1].min())

    # Prevent division by zero if a polygon collapses to a point.
    lon_range = max(lon_max - lon_min, 1e-6)
    lat_range = max(lat_max - lat_min, 1e-6)

    for poly in polygons:
        pts = np.zeros_like(poly, dtype=np.int32)
        pts[:, 0] = ((poly[:, 0] - lon_min) / lon_range * (shape[1] - 1)).astype(np.int32)
        pts[:, 1] = ((poly[:, 1] - lat_min) / lat_range * (shape[0] - 1)).astype(np.int32)
        # Clamp to image bounds.
        pts[:, 0] = np.clip(pts[:, 0], 0, shape[1] - 1).astype(np.int32)
        pts[:, 1] = np.clip(pts[:, 1], 0, shape[0] - 1).astype(np.int32)

        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)

    # --- optional export -----------------------------------------------------
    if output_npy_path is not None:
        out = Path(output_npy_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), mask)

    return mask