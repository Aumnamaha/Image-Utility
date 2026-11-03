"""Unit tests for ``src.dataset_utils`` — satellite band image-to-NumPy conversion."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — write synthetic images to temporary files in supported formats
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_images(tmp_path_factory):
    """Create a small temp directory with synthetic images in TIFF, PNG, JPG."""
    store = {}
    tmpdir = tmp_path_factory.mktemp("test_images")

    # Deterministic 64×64 grayscale uint8 image.
    gray = np.random.default_rng(seed=99).integers(0, 256, size=(64, 64), dtype=np.uint8)

    for ext in ("tif", "png", "jpg"):
        path = tmpdir / f"green.{ext}"
        cv2.imwrite(str(path), gray)
        store[ext] = str(path)

    # Multi-band (RGB) uint8 image.
    rgb = np.random.default_rng(seed=7).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    for ext in ("tif", "png", "jpg"):
        path = tmpdir / f"rgb.{ext}"
        cv2.imwrite(str(path), rgb)

    # Float32 reflectance-style image (values in [0.0, 1.0]).
    refl = np.random.default_rng(seed=42).uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)
    for ext in ("tif", "png"):
        path = tmpdir / f"reflectance.{ext}"
        cv2.imwrite(str(path), (refl * 255).astype(np.uint8))

    return store, tmpdir


# ---------------------------------------------------------------------------
# Tests — load_and_convert_band
# ---------------------------------------------------------------------------

class TestLoadAndConvertBand:
    """Core functionality tests for ``load_and_convert_band``."""

    def test_returns_numpy_array(self, tmp_images):
        """The function should always return a numpy.ndarray."""
        from src.dataset_utils import load_and_convert_band

        path = tmp_images[0]["tif"]
        result = load_and_convert_band(path)
        assert isinstance(result, np.ndarray), "Expected np.ndarray return type"

    def test_grayscale_tiff(self, tmp_images):
        """Single-band TIFF → 2-D float32 [0, 255]."""
        from src.dataset_utils import load_and_convert_band

        arr = load_and_convert_band(tmp_images[0]["tif"])
        assert arr.ndim == 2, f"Expected 2-D array, got {arr.ndim}-D"
        assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
        assert arr.shape == (64, 64)

    def test_grayscale_png(self, tmp_images):
        """Single-band PNG → 2-D float32 [0, 255]."""
        from src.dataset_utils import load_and_convert_band

        arr = load_and_convert_band(tmp_images[0]["png"])
        assert arr.ndim == 2 and arr.dtype == np.float32

    def test_grayscale_jpg(self, tmp_images):
        """Single-band JPG → 2-D float32 [0, 255]."""
        from src.dataset_utils import load_and_convert_band

        arr = load_and_convert_band(tmp_images[0]["jpg"])
        assert arr.ndim == 2 and arr.dtype == np.float32

    def test_multiband_rgb_converted_to_grayscale(self, tmp_images):
        """Multi-band (RGB) image should be converted to a single grayscale band."""
        from src.dataset_utils import load_and_convert_band

        _, tmpdir = tmp_images
        for ext in ("tif", "png", "jpg"):
            arr = load_and_convert_band(str(tmpdir / f"rgb.{ext}"))
            assert arr.ndim == 2, f"RGB {ext} should produce 2-D array"
            assert arr.dtype == np.float32

    def test_normalization_range(self, tmp_images):
        """Output values must lie within [0, 255]."""
        from src.dataset_utils import load_and_convert_band

        store = tmp_images[0]
        for ext in ("tif", "png", "jpg"):
            arr = load_and_convert_band(store[ext])
            assert arr.min() >= 0.0, f"Min value {arr.min()} < 0"
            assert arr.max() <= 255.0 + 1e-6, f"Max value {arr.max()} > 255"

    def test_constant_image_stays_zero(self):
        """A uniform (constant) image should remain all zeros after normalisation."""
        from src.dataset_utils import load_and_convert_band

        zero_img = np.zeros((32, 32), dtype=np.uint8)
        fd, img_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)  # close the OS-level handle so we can delete on Windows.
        try:
            cv2.imwrite(img_path, zero_img)
            arr = load_and_convert_band(img_path)
            assert np.allclose(arr, 0.0), "Constant image should normalise to zeros"
        finally:
            os.unlink(img_path)

    def test_float_reflectance_source(self, tmp_images):
        """Float reflectance source (stored as uint8 after *255) → [0, 255] output."""
        from src.dataset_utils import load_and_convert_band

        _, tmpdir = tmp_images
        for ext in ("tif", "png"):
            path = str(tmpdir / f"reflectance.{ext}")
            arr = load_and_convert_band(path)
            assert arr.ndim == 2 and arr.dtype == np.float32
            # Because we stored uint8(reflectance*255), min should be ~0, max ~255.
            assert 0 <= arr.min() < 10, f"Min unexpectedly high: {arr.min()}"
            assert 245 <= arr.max() <= 255, f"Max unexpectedly low: {arr.max()}"


class TestNpyExport:
    """Tests for .npy file export functionality."""

    def test_npy_export_creates_file(self):
        """Specifying output_npy_path should create a valid .npy file on disk."""
        from src.dataset_utils import load_and_convert_band

        gray = np.random.default_rng(seed=55).integers(0, 256, size=(32, 32), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as img_f:
            cv2.imwrite(img_f.name, gray)
            img_path = img_f.name

        with tempfile.TemporaryDirectory() as tmpdir:
            npy_path = os.path.join(tmpdir, "output_band.npy")

            arr = load_and_convert_band(img_path, output_npy_path=npy_path)

            assert os.path.isfile(npy_path), ".npy file was not created"
            loaded = np.load(npy_path)
            np.testing.assert_array_equal(arr, loaded)
            assert loaded.dtype == np.float32

        os.unlink(img_path)


class TestErrorHandling:
    """Tests for proper error raising on invalid inputs."""

    def test_missing_file_raises_file_not_found(self):
        """A non-existent path should raise FileNotFoundError."""
        from src.dataset_utils import load_and_convert_band

        with pytest.raises(FileNotFoundError, match="not found"):
            load_and_convert_band("/nonexistent/path/to/image.tif")

    def test_corrupt_image_raises_value_error(self):
        """A corrupt / un-decodable file should raise ValueError."""
        from src.dataset_utils import load_and_convert_band

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            f.write(b"this is not a valid image")
            corrupt_path = f.name

        try:
            with pytest.raises(ValueError, match="Could not decode"):
                load_and_convert_band(corrupt_path)
        finally:
            os.unlink(corrupt_path)


class TestConvertImageToNpy:
    """Tests for the convenience wrapper ``convert_image_to_npy``."""

    def test_auto_output_path(self):
        """When output_dir is None, .npy should be placed next to the source."""
        from src.dataset_utils import convert_image_to_npy

        gray = np.random.default_rng(seed=33).integers(0, 256, size=(16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, gray)
            img_path = f.name

        try:
            out = convert_image_to_npy(img_path)
            assert os.path.isfile(out)
            assert out.endswith("_band.npy")
        finally:
            os.unlink(img_path)

    def test_custom_output_dir(self):
        """When output_dir is given, .npy should be placed there."""
        from src.dataset_utils import convert_image_to_npy

        gray = np.random.default_rng(seed=33).integers(0, 256, size=(16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, gray)
            img_path = f.name

        with tempfile.TemporaryDirectory() as tmpdir:
            out = convert_image_to_npy(img_path, output_dir=tmpdir)
            assert os.path.isfile(out)
            assert Path(out).parent == Path(tmpdir)

        os.unlink(img_path)


# ---------------------------------------------------------------------------
# Helpers — create synthetic .kml files in temp directories
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_kml(tmp_path_factory):
    """Create a small temp directory with synthetic KML files."""
    import tempfile as _tempfile

    store: dict = {}
    tmpdir = tmp_path_factory.mktemp("test_kml")

    # --- Simple rectangular polygon (a water body) -------------------------
    kml_rect = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Water Body</name>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              -73.98,40.76,-10
              -73.95,40.76,-10
              -73.95,40.74,-10
              -73.98,40.74,-10
              -73.98,40.76,-10
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>"""
    rect_path = tmpdir / "water_rect.kml"
    rect_path.write_text(kml_rect, encoding="utf-8")
    store["rect"] = str(rect_path)

    # --- Two separate polygons (multi-placemark) ----------------------------
    kml_multi = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Patch A</name>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              -73.98,40.76,-10
              -73.97,40.76,-10
              -73.97,40.75,-10
              -73.98,40.75,-10
              -73.98,40.76,-10
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
    <Placemark>
      <name>Patch B</name>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              -73.96,40.75,-10
              -73.95,40.75,-10
              -73.95,40.74,-10
              -73.96,40.74,-10
              -73.96,40.75,-10
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>"""
    multi_path = tmpdir / "water_multi.kml"
    multi_path.write_text(kml_multi, encoding="utf-8")
    store["multi"] = str(multi_path)

    # --- Malformed KML (no polygons) ----------------------------------------
    kml_empty = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Empty</name>
  </Document>
</kml>"""
    empty_path = tmpdir / "empty.kml"
    empty_path.write_text(kml_empty, encoding="utf-8")
    store["empty"] = str(empty_path)

    # --- Corrupt KML (not valid XML) ----------------------------------------
    fd, corrupt_path = _tempfile.mkstemp(suffix=".kml")
    os.close(fd)
    Path(corrupt_path).write_bytes(b"this is not xml at all <><>")
    store["corrupt"] = corrupt_path

    return store, tmpdir


# ---------------------------------------------------------------------------
# Tests — parse_kml_polygons
# ---------------------------------------------------------------------------

class TestParseKmlPolygons:
    """Tests for the ``parse_kml_polygons`` function."""

    def test_returns_list_of_arrays(self, tmp_kml):
        """Should return a list of numpy arrays."""
        from src.dataset_utils import parse_kml_polygons

        result = parse_kml_polygons(tmp_kml[0]["rect"])
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_single_polygon_returns_one_array(self, tmp_kml):
        """A KML with one polygon should return exactly one coordinate array."""
        from src.dataset_utils import parse_kml_polygons

        polys = parse_kml_polygons(tmp_kml[0]["rect"])
        assert len(polys) == 1, f"Expected 1 polygon, got {len(polys)}"

    def test_multi_polygon_returns_multiple_arrays(self, tmp_kml):
        """A KML with two placemarks should return two coordinate arrays."""
        from src.dataset_utils import parse_kml_polygons

        polys = parse_kml_polygons(tmp_kml[0]["multi"])
        assert len(polys) == 2, f"Expected 2 polygons, got {len(polys)}"

    def test_coordinate_array_shape(self, tmp_kml):
        """Each coordinate array should have shape (N, 2)."""
        from src.dataset_utils import parse_kml_polygons

        polys = parse_kml_polygons(tmp_kml[0]["rect"])
        assert polys[0].ndim == 2
        assert polys[0].shape[1] == 2, "Expected (N, 2) coordinate arrays"

    def test_ring_is_closed(self, tmp_kml):
        """The last coordinate should equal the first (closed ring)."""
        from src.dataset_utils import parse_kml_polygons

        polys = parse_kml_polygons(tmp_kml[0]["rect"])
        assert np.allclose(polys[0][0], polys[0][-1]), "Ring should be closed"

    def test_missing_file_raises(self):
        """A non-existent .kml path should raise FileNotFoundError."""
        from src.dataset_utils import parse_kml_polygons

        with pytest.raises(FileNotFoundError, match="not found"):
            parse_kml_polygons("/nonexistent/path/file.kml")

    def test_no_polygon_raises(self, tmp_kml):
        """A KML with no polygon elements should raise ValueError."""
        from src.dataset_utils import parse_kml_polygons

        with pytest.raises(ValueError, match="No valid polygon"):
            parse_kml_polygons(tmp_kml[0]["empty"])

    def test_corrupt_kml_raises(self, tmp_kml):
        """A corrupt / invalid XML KML should raise ValueError."""
        from src.dataset_utils import parse_kml_polygons

        with pytest.raises(ValueError):
            parse_kml_polygons(tmp_kml[0]["corrupt"])


# ---------------------------------------------------------------------------
# Tests — rasterize_kml_mask
# ---------------------------------------------------------------------------

class TestRasterizeKmlMask:
    """Tests for the ``rasterize_kml_mask`` function."""

    def test_returns_numpy_array(self, tmp_kml):
        """Should return a numpy.ndarray."""
        from src.dataset_utils import rasterize_kml_mask

        result = rasterize_kml_mask(tmp_kml[0]["rect"])
        assert isinstance(result, np.ndarray)

    def test_default_shape(self, tmp_kml):
        """Default shape should be (256, 256)."""
        from src.dataset_utils import rasterize_kml_mask

        mask = rasterize_kml_mask(tmp_kml[0]["rect"])
        assert mask.shape == (256, 256), f"Expected (256, 256), got {mask.shape}"

    def test_custom_shape(self, tmp_kml):
        """Custom shape should be respected."""
        from src.dataset_utils import rasterize_kml_mask

        mask = rasterize_kml_mask(tmp_kml[0]["rect"], shape=(512, 640))
        assert mask.shape == (512, 640), f"Expected (512, 640), got {mask.shape}"

    def test_dtype_is_uint8(self, tmp_kml):
        """Output dtype should be uint8."""
        from src.dataset_utils import rasterize_kml_mask

        mask = rasterize_kml_mask(tmp_kml[0]["rect"])
        assert mask.dtype == np.uint8, f"Expected uint8, got {mask.dtype}"

    def test_values_are_0_or_255(self, tmp_kml):
        """All pixel values should be either 0 or 255."""
        from src.dataset_utils import rasterize_kml_mask

        mask = rasterize_kml_mask(tmp_kml[0]["rect"])
        unique_vals = np.unique(mask)
        for v in unique_vals:
            assert v in (0, 255), f"Unexpected value {v} in mask"

    def test_polygon_interior_filled(self, tmp_kml):
        """The rasterized mask should contain at least one non-zero pixel."""
        from src.dataset_utils import rasterize_kml_mask

        mask = rasterize_kml_mask(tmp_kml[0]["rect"])
        assert (mask == 255).any(), "Polygon interior was not filled"

    def test_multi_polygon_both_filled(self, tmp_kml):
        """Both polygons in a multi-placemark KML should be rasterized."""
        from src.dataset_utils import rasterize_kml_mask

        mask = rasterize_kml_mask(tmp_kml[0]["multi"])
        assert (mask == 255).sum() > 10, "Expected significant filled area from two polygons"

    def test_invalid_shape_raises(self, tmp_kml):
        """A non-2-tuple shape should raise ValueError."""
        from src.dataset_utils import rasterize_kml_mask

        with pytest.raises(ValueError, match="must be a 2-tuple"):
            rasterize_kml_mask(tmp_kml[0]["rect"], shape=(100,))

    def test_npy_export_creates_file(self):
        """Specifying output_npy_path should create a valid .npy file."""
        from src.dataset_utils import rasterize_kml_mask

        # Create a minimal KML on the fly.
        kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              -73.98,40.76,-10
              -73.95,40.76,-10
              -73.95,40.74,-10
              -73.98,40.74,-10
              -73.98,40.76,-10
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>"""
        fd, kml_path = tempfile.mkstemp(suffix=".kml")
        os.close(fd)
        try:
            Path(kml_path).write_text(kml_content, encoding="utf-8")

            with tempfile.TemporaryDirectory() as tmpdir:
                npy_path = os.path.join(tmpdir, "mask.npy")
                mask = rasterize_kml_mask(kml_path, shape=(128, 128), output_npy_path=npy_path)

                assert os.path.isfile(npy_path), ".npy file was not created"
                loaded = np.load(npy_path)
                np.testing.assert_array_equal(mask, loaded)
                assert loaded.dtype == np.uint8
        finally:
            os.unlink(kml_path)

    def test_missing_kml_raises(self):
        """A non-existent KML path should raise FileNotFoundError."""
        from src.dataset_utils import rasterize_kml_mask

        with pytest.raises(FileNotFoundError, match="not found"):
            rasterize_kml_mask("/nonexistent/path/file.kml")

    def test_empty_kml_raises(self, tmp_kml):
        """A KML with no polygons should raise ValueError."""
        from src.dataset_utils import rasterize_kml_mask

        with pytest.raises(ValueError):
            rasterize_kml_mask(tmp_kml[0]["empty"])
