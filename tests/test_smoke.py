"""Smoke tests for conftest fixtures — validates synthetic image shapes/dtypes."""

import numpy as np


# -- RGB fixtures -----------------------------------------------------------

def test_rgb_256_shape(rgb_256):
    assert rgb_256.shape == (256, 256, 3)
    assert rgb_256.dtype == np.uint8


def test_rgb_gradient_shape(rgb_gradient):
    assert rgb_gradient.shape == (256, 256, 3)
    assert rgb_gradient.dtype == np.uint8


# -- Multi-spectral fixtures ------------------------------------------------

def test_multispectral_256_shape(multispectral_256):
    assert multispectral_256.shape == (256, 256, 6)
    assert multispectral_256.dtype == np.float32


def test_hyperspectral_shape(multispectral_hyperspectral):
    assert multispectral_hyperspectral.shape == (64, 64, 32)
    assert multispectral_hyperspectral.dtype == np.float32


# -- Utility fixtures -------------------------------------------------------

def test_empty_image(empty_image):
    assert empty_image.shape == (256, 256, 3)
    assert np.all(empty_image == 0)


def test_solid_color(solid_color):
    assert solid_color.shape == (256, 256, 3)
    assert solid_color[:, :, 0].max() == 255
    assert np.all(solid_color[:, :, 1] == 0)
    assert np.all(solid_color[:, :, 2] == 0)
