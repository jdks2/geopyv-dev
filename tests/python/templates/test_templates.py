"""
Phase 3 validation tests for geopyv_dev Mask (local, circle and square shapes).

Asserts golden values from the Rust implementation using the unified Mask API.
Coordinate convention: coords[:, 0] = x-offset, coords[:, 1] = y-offset.
"""

import numpy as np
import pytest

from geopyv_dev import Mask


# ---------------------------------------------------------------------------
# Tests: circle shape
# ---------------------------------------------------------------------------

def test_circle_shape_attr():
    t = Mask(mask_type="local", shape="circle", size=5)
    assert t.shape == "circle"


def test_circle_dimension_attr():
    t = Mask(mask_type="local", shape="circle", size=5)
    assert t.dimension == "radius"


def test_circle_size():
    t = Mask(mask_type="local", shape="circle", size=5)
    assert t.size == 5


def test_circle_n_px():
    """81 lattice points within radius 5."""
    t = Mask(mask_type="local", shape="circle", size=5)
    assert t.n_px == 81


def test_circle_coords_shape():
    t = Mask(mask_type="local", shape="circle", size=5)
    assert t.coords.shape == (81, 2)


def test_circle_coords_dtype():
    t = Mask(mask_type="local", shape="circle", size=5)
    assert t.coords.dtype == np.float64


def test_circle_first_coord():
    """First coord in row-major scan: row=0, col=5 → [x=0, y=-5]."""
    t = Mask(mask_type="local", shape="circle", size=5)
    np.testing.assert_array_equal(t.coords[0], [0.0, -5.0])


def test_circle_last_coord():
    """Last coord: row=10, col=5 → [x=0, y=5]."""
    t = Mask(mask_type="local", shape="circle", size=5)
    np.testing.assert_array_equal(t.coords[-1], [0.0, 5.0])


def test_circle_coord_range():
    t = Mask(mask_type="local", shape="circle", size=5)
    assert np.min(t.coords) >= -5
    assert np.max(t.coords) <= 5


def test_circle_subset_mask_shape():
    t = Mask(mask_type="local", shape="circle", size=5)
    assert t.subset_mask.shape == (11, 11)


def test_circle_subset_mask_centre():
    t = Mask(mask_type="local", shape="circle", size=5)
    assert t.subset_mask[5, 5] == 1


def test_circle_subset_mask_poles():
    t = Mask(mask_type="local", shape="circle", size=5)
    assert t.subset_mask[0, 5] == 1  # top pole
    assert t.subset_mask[5, 0] == 1  # left pole


def test_circle_subset_mask_corners():
    t = Mask(mask_type="local", shape="circle", size=5)
    assert t.subset_mask[0, 0] == 0
    assert t.subset_mask[0, 10] == 0
    assert t.subset_mask[10, 0] == 0
    assert t.subset_mask[10, 10] == 0


def test_circle_repr():
    t = Mask(mask_type="local", shape="circle", size=10)
    assert "circle" in repr(t)
    assert "10" in repr(t)


def test_circle_default_size():
    t = Mask(mask_type="local", shape="circle")
    assert t.size == 25


# ---------------------------------------------------------------------------
# Tests: square shape
# ---------------------------------------------------------------------------

def test_square_shape_attr():
    t = Mask(mask_type="local", shape="square", size=5)
    assert t.shape == "square"


def test_square_dimension_attr():
    t = Mask(mask_type="local", shape="square", size=5)
    assert t.dimension == "length"


def test_square_size():
    t = Mask(mask_type="local", shape="square", size=5)
    assert t.size == 5


def test_square_n_px():
    t = Mask(mask_type="local", shape="square", size=5)
    assert t.n_px == 121


def test_square_coords_shape():
    t = Mask(mask_type="local", shape="square", size=5)
    assert t.coords.shape == (121, 2)


def test_square_first_coord():
    """First entry: x=-5, y=-5."""
    t = Mask(mask_type="local", shape="square", size=5)
    np.testing.assert_array_equal(t.coords[0], [-5.0, -5.0])


def test_square_last_coord():
    """Last entry: x=5, y=5."""
    t = Mask(mask_type="local", shape="square", size=5)
    np.testing.assert_array_equal(t.coords[-1], [5.0, 5.0])


def test_square_subset_mask_all_ones():
    t = Mask(mask_type="local", shape="square", size=5)
    assert np.all(t.subset_mask == 1)


def test_square_repr():
    t = Mask(mask_type="local", shape="square", size=7)
    assert "square" in repr(t)
    assert "7" in repr(t)


def test_square_default_size():
    t = Mask(mask_type="local", shape="square")
    assert t.size == 25


# ---------------------------------------------------------------------------
# Tests: unknown shape
# ---------------------------------------------------------------------------

def test_unknown_shape_raises():
    with pytest.raises(Exception):
        Mask(mask_type="local", shape="triangle", size=5)


# ---------------------------------------------------------------------------
# Tests: mask() method
# ---------------------------------------------------------------------------

def test_mask_full_ones_square2():
    t = Mask(mask_type="local", shape="square", size=2)
    mask = np.ones((20, 20), dtype=np.uint8)
    t.mask([5.0, 5.0], mask)
    assert t.m_n_px == 25


def test_mask_zeros_square2():
    t = Mask(mask_type="local", shape="square", size=2)
    mask = np.zeros((20, 20), dtype=np.uint8)
    t.mask([5.0, 5.0], mask)
    assert t.m_n_px == 0


def test_mask_updates_coords():
    """After mask(), coords has shape (m_n_px, 2) with x-first convention."""
    t = Mask(mask_type="local", shape="square", size=1)
    mask = np.ones((10, 10), dtype=np.uint8)
    t.mask([3.0, 3.0], mask)
    assert t.coords.shape == (9, 2)
    # First pixel: row=0, col=0 → x_off=-1, y_off=-1
    np.testing.assert_array_equal(t.coords[0], [-1.0, -1.0])
