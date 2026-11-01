"""Pytest fixtures for generating synthetic test images."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# RGB fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rgb_256() -> np.ndarray:
    """A deterministic 256×256 uint8 RGB image (shape: 256, 256, 3)."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def rgb_gradient() -> np.ndarray:
    """A smooth linear gradient RGB image for edge-detection tests."""
    x = np.linspace(0, 255, 256, dtype=np.float32)
    grad = np.outer(x, x)  # 256×256
    rgb = np.stack([grad, grad * 0.7, grad * 0.4], axis=-1).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Multi-spectral float32 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def multispectral_256() -> np.ndarray:
    """A deterministic 256×256 multi-spectral (6 bands) float32 image."""
    rng = np.random.default_rng(seed=128)
    return rng.standard_normal((256, 256, 6)).astype(np.float32)


@pytest.fixture(scope="session")
def multispectral_hyperspectral() -> np.ndarray:
    """A small hyperspectral cube: 64×64 pixels × 32 bands, float32."""
    rng = np.random.default_rng(seed=256)
    return rng.uniform(0.0, 1.0, size=(64, 64, 32)).astype(np.float32)


# ---------------------------------------------------------------------------
# Utility fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def empty_image() -> np.ndarray:
    """A zero-filled 256×256 RGB image."""
    return np.zeros((256, 256, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def solid_color() -> np.ndarray:
    """A solid-red 256×256 RGB image for contrast tests."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # red channel full
    return img
