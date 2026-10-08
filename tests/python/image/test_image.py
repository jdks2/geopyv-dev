"""
Phase 3 validation tests for geopyv_dev.Image.

Strategy (per plan §Phase 3 — Validation):
  The Phase 1 fixture file (geopyv/tests/image/fixtures.py) runs against
  `geopyv` to establish ground-truth golden values.  This file imports
  `geopyv_dev` and asserts hard-coded golden values.

  NOTE: the QCQT-block golden values below are computed from geopyv_dev,
  NOT `geopyv`.  geopyv_dev's B-spline prefilter (`image.rs::build_kernel`)
  fixes a one-sample kernel-phase error in `geopyv`'s `image.py::_get_C`
  that shifts every interpolated intensity by (1, 1) px, so the two
  packages' QCQT blocks differ by a one-pixel offset.  After the fix a
  block's DC term (`block[0, 0]`) equals that pixel's own greyscale value.

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
        [ 2.30000000e+01,  1.67789557e+00, -6.74737925e-01,
         -2.06149534e-01,  2.89943310e-01, -8.69514209e-02],
        [-1.36569749e+01, -3.96960223e-01, -5.04545832e-01,
         -2.19001517e-01,  7.31478102e-01, -2.54040381e-01],
        [ 2.46438223e+00,  6.58652659e-01,  1.10081852e+00,
         -2.78314427e-01, -6.48067209e-01,  2.63979342e-01],
        [-3.70884341e-01, -3.71877562e-01,  8.54577858e-01,
          1.91994131e-01, -7.43567246e-01,  2.55217536e-01],
        [ 8.84647118e-01, -4.94757611e-01, -1.16245124e+00,
          3.08413059e-01,  6.69105574e-01, -2.80949494e-01],
        [-3.21170097e-01,  2.32357673e-01,  2.96092192e-01,
         -1.40902419e-01, -1.30745939e-01,  6.34944538e-02],
    ])
    np.testing.assert_allclose(_block(rust_image.qcqt, 50, 100), expected, atol=1e-12)


def test_qcqt_block_pixel_200_400(rust_image):
    """Tier A: atol = 1e-12."""
    expected = np.array([
        [ 7.20000000e+01, -2.29485426e+01,  2.12626441e+00,
          4.58515063e-01, -9.80520315e-01,  3.44283403e-01],
        [-1.35810193e+01,  4.31917162e+00, -1.25946453e+00,
          2.17954360e-02,  6.77070052e-01, -2.58613008e-01],
        [-1.85317300e+00, -6.84062462e-01, -6.03681170e-01,
          8.13627439e-01,  5.89310680e-01, -3.61563350e-01],
        [ 5.04999607e-03,  1.52568753e-01,  1.23379503e+00,
         -2.42873690e-01, -8.41846495e-01,  3.46905732e-01],
        [ 5.83331528e-01,  9.92586580e-01,  1.82144887e-01,
         -7.06561237e-01, -3.48435092e-01,  2.64048771e-01],
        [-1.54189207e-01, -4.35991355e-01, -2.93271520e-01,
          3.14781230e-01,  2.98225759e-01, -1.69668067e-01],
    ])
    np.testing.assert_allclose(_block(rust_image.qcqt, 200, 400), expected, atol=1e-12)


def test_qcqt_block_pixel_500_500(rust_image):
    """Tier A: atol = 1e-12."""
    expected = np.array([
        [ 1.43000000e+02,  1.58808314e+01, -2.73019438e-01,
          1.28245171e-01,  3.90069972e-01, -1.26127154e-01],
        [ 1.08613810e+01, -6.42754418e+00, -1.87498143e+00,
          5.30102746e-01,  1.28717912e+00, -5.18768234e-01],
        [ 2.34288370e+00,  1.18153528e+00,  8.10812226e-01,
         -5.32787746e-01, -5.59459484e-01,  2.56680827e-01],
        [-3.67984178e-01,  4.45135157e-01,  6.34103759e-01,
         -2.57467525e-01, -4.42133752e-01,  1.96651252e-01],
        [ 2.85434217e-01, -1.21515258e+00, -6.72352637e-01,
          5.20999829e-01,  6.63497959e-01, -3.28681572e-01],
        [-1.21714772e-01,  3.95085338e-01,  1.80177059e-01,
         -1.54063858e-01, -2.34214927e-01,  1.15647384e-01],
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
