"""
Phase 3 validation tests for geopyv_dev.Image.

Strategy (per plan §Phase 3 — Validation):
  The Phase 1 fixture file (geopyv/tests/image/fixtures.py) runs against
  `geopyv` to establish ground-truth golden values.  This file imports
  `geopyv_dev` and asserts the *same* hard-coded golden values.

Image loading: cv2 is used to replicate geopyv's pre-processing exactly
(BGR2GRAY + GaussianBlur), then the resulting numpy array is passed to
`Image(image_gs=arr, border=20)`.  This decouples the B-spline math from
any image-loading discrepancy.

Tolerance tiers:
  Tier A  atol = 1e-12  pure matrix algebra (QCQT blocks)
  Tier B  rtol = 1e-8   FFT-based B-spline coefficient computation
"""

import os

import cv2
import numpy as np
import pytest

from geopyv_dev import Image

# ---------------------------------------------------------------------------
# Shared test image
# ---------------------------------------------------------------------------

REF_JPG = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",   # geopyv-dev/ → claude_playground/
    "geopyv", "tests", "ref.jpg",
)
BORDER = 20


@pytest.fixture(scope="module")
def rust_image():
    """Build a Rust Image from ref.jpg pre-processed by cv2 (matches geopyv)."""
    img = cv2.imread(REF_JPG, cv2.IMREAD_COLOR)
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gs = cv2.GaussianBlur(gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
    image_gs_f64 = gs.astype(np.float64)
    return Image(image_gs=image_gs_f64, border=BORDER)


# ---------------------------------------------------------------------------
# image_gs property
# ---------------------------------------------------------------------------


def test_image_gs_shape(rust_image):
    assert rust_image.image_gs.shape == (1001, 1001)


def test_image_gs_dtype(rust_image):
    assert rust_image.image_gs.dtype == np.float64


def test_image_gs_spot_values(rust_image):
    """Values must exactly match the Python uint8 greyscale intensities."""
    gs = rust_image.image_gs
    assert gs[0, 0] == 0.0
    assert gs[0, 1000] == 66.0
    assert gs[500, 500] == 143.0
    assert gs[100, 200] == 0.0
    assert gs[300, 700] == 24.0


# ---------------------------------------------------------------------------
# border property
# ---------------------------------------------------------------------------


def test_border(rust_image):
    assert rust_image.border == BORDER


# ---------------------------------------------------------------------------
# qcqt property — Tier A, atol = 1e-12
# ---------------------------------------------------------------------------


def test_qcqt_shape(rust_image):
    assert rust_image.qcqt.shape == (1001 * 6, 1001 * 6)


def test_qcqt_dtype(rust_image):
    assert rust_image.qcqt.dtype == np.float64


def _block(qcqt, i, j):
    return qcqt[i * 6 : i * 6 + 6, j * 6 : j * 6 + 6]


def test_qcqt_block_pixel_50_100(rust_image):
    """Tier A: atol = 1e-12."""
    expected = np.array([
        [ 3.70000000e+01,  3.13512169e+00, -2.09785911e-03,
         -1.51161262e-01,  4.48231642e-02, -2.66857304e-02],
        [-1.92522253e+01, -1.89387821e+00, -1.04191507e+00,
          4.85758336e-01,  6.42435533e-01, -2.92713101e-01],
        [ 3.89951113e+00,  1.30853700e+00,  9.10810432e-01,
         -2.57111311e-01, -1.30227043e+00,  5.56555063e-01],
        [-8.67700867e-01,  2.26579782e-01,  1.15347357e+00,
         -2.94902360e-01, -6.21739458e-01,  2.63702601e-01],
        [ 1.84572374e-01, -1.29945806e+00, -9.11452884e-01,
          4.14961036e-01,  1.11179600e+00, -5.00214394e-01],
        [ 3.58426931e-02,  4.89544127e-01,  1.39958851e-01,
         -1.28873724e-01, -3.02398251e-01,  1.42814912e-01],
    ])
    np.testing.assert_allclose(_block(rust_image.qcqt, 50, 100), expected, atol=1e-12)


def test_qcqt_block_pixel_200_400(rust_image):
    """Tier A: atol = 1e-12."""
    expected = np.array([
        [ 1.12000000e+02, -2.71957874e+01, -1.98206839e+00,
          7.02763137e-01,  7.43631885e-01, -2.68539270e-01],
        [-1.28263805e+01,  1.69579698e+00, -7.17958231e-01,
          8.20868246e-01,  8.34921470e-01, -4.69458198e-01],
        [-3.55149520e+00,  1.27801639e+00,  1.88174620e+00,
         -5.17698833e-01, -8.45795552e-01,  3.25614670e-01],
        [ 3.24301248e-01,  1.07369535e+00,  3.42111755e-02,
         -7.25477721e-01, -1.65623485e-01,  1.91895835e-01],
        [ 1.95671695e-02, -6.22739524e-01, -1.08183470e+00,
          5.46995510e-01,  3.55733769e-01, -1.65029954e-01],
        [ 3.40072392e-02,  4.64312111e-02,  3.75785323e-01,
         -8.91380488e-02, -8.22463858e-02,  2.12885135e-02],
    ])
    np.testing.assert_allclose(_block(rust_image.qcqt, 200, 400), expected, atol=1e-12)


def test_qcqt_block_pixel_500_500(rust_image):
    """Tier A: atol = 1e-12."""
    expected = np.array([
        [ 1.14000000e+02,  1.99126012e+01,  7.74082233e-01,
          1.37112642e-01,  3.83866373e-01, -2.07662410e-01],
        [ 9.46325659e+00, -4.43634594e+00, -3.63693537e-01,
         -1.40308007e-01,  1.16213395e-01, -3.45929241e-02],
        [ 4.09012373e+00,  5.26748992e-01,  2.91275195e-01,
         -7.08309066e-01, -9.15220517e-01,  5.25612497e-01],
        [ 2.24124699e-02, -8.74583818e-02, -2.72802118e-01,
          1.43049378e-01,  5.81199022e-02, -2.38315859e-02],
        [-9.14806521e-01, -1.53165659e-01,  2.57346429e-01,
          3.27271969e-01,  3.31239760e-01, -2.37057116e-01],
        [ 3.39013733e-01,  2.88153491e-02, -8.32824579e-02,
         -1.11112497e-01, -1.24018509e-01,  8.55054531e-02],
    ])
    np.testing.assert_allclose(_block(rust_image.qcqt, 500, 500), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr(rust_image):
    r = repr(rust_image)
    assert "Image" in r
    assert "1001" in r
    assert "20" in r
