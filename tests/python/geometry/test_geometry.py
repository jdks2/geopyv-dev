"""
Phase 3 validation tests for geopyv_dev geometry (utilities, region, meshing).

Asserts the same golden values established by the Phase 1 fixtures in
geopyv/tests/geometry/*/fixtures.py against the Rust implementation.
"""

import numpy as np
import pytest

from geopyv_dev import (
    area_to_length,
    poly_area,
    ccw,
    intersect,
    polysect,
    polycentroid,
    CircleRegion,
    PathRegion,
    mask_image,
)


# ===========================================================================
# Utilities
# ===========================================================================

class TestAreaToLength:
    def test_unit_area(self):
        expected = np.sqrt(4.0 / np.sqrt(3))
        np.testing.assert_allclose(area_to_length(1.0), expected, rtol=1e-12)

    def test_negative_area(self):
        np.testing.assert_allclose(area_to_length(-1.0), area_to_length(1.0), rtol=1e-12)

    def test_area_4(self):
        expected = np.sqrt(16.0 / np.sqrt(3))
        np.testing.assert_allclose(area_to_length(4.0), expected, rtol=1e-12)


class TestPolyArea:
    def test_unit_square(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(poly_area(pts), 1.0, rtol=1e-12)

    def test_right_triangle(self):
        pts = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(poly_area(pts), 6.0, rtol=1e-12)

    def test_4x4_square(self):
        pts = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
        np.testing.assert_allclose(poly_area(pts), 16.0, rtol=1e-12)


class TestCcwIntersect:
    def test_ccw_true(self):
        assert ccw([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]) is True

    def test_ccw_false(self):
        assert ccw([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]) is False

    def test_intersect_crossing(self):
        assert intersect([0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]) is True

    def test_intersect_non_crossing(self):
        assert intersect([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]) is False


class TestPolysect:
    def test_convex_hexagon_no_intersect(self):
        n = np.array([
            [np.cos(2 * np.pi * i / 6), np.sin(2 * np.pi * i / 6)]
            for i in range(6)
        ])
        assert polysect(n) is None

    def test_bowtie_detects_intersection(self):
        n = np.array([
            [0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0], [1.0, 3.0], [3.0, 3.0],
        ])
        result = polysect(n)
        assert result is not None
        assert list(result) == [0, 2]

    def test_requires_six_points(self):
        with pytest.raises(Exception):
            polysect(np.zeros((4, 2)))


class TestPolycentroid:
    def test_x_component_square(self):
        coords = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
        c = polycentroid(coords)
        np.testing.assert_allclose(c[0], 2.0, atol=1e-10)

    def test_y_component_buggy_value(self):
        """Python source has a typo; Rust replicates it — golden = 256/96."""
        coords = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
        c = polycentroid(coords)
        np.testing.assert_allclose(c[1], 256.0 / 96.0, rtol=1e-10)


# ===========================================================================
# Region
# ===========================================================================

class TestCircleRegion:
    def test_node_count(self):
        expected = max(6, int(2 * np.pi * 50.0 / 20.0))
        r = CircleRegion([100.0, 100.0], radius=50.0, size=20.0)
        assert r.current_nodes.shape[0] == expected

    def test_nodes_on_circle(self):
        centre = np.array([50.0, 50.0])
        r = CircleRegion(centre.tolist(), radius=30.0, size=10.0)
        dists = np.sqrt(np.sum((r.current_nodes - centre) ** 2, axis=1))
        np.testing.assert_allclose(dists, 30.0, rtol=1e-12)

    def test_initial_counter(self):
        r = CircleRegion([0.0, 0.0])
        assert r.counter == 0

    def test_minimum_six_nodes(self):
        r = CircleRegion([0.0, 0.0], radius=5.0, size=100.0)
        assert r.current_nodes.shape[0] == 6

    def test_option_attr(self):
        r = CircleRegion([0.0, 0.0], option="R")
        assert r.option == "R"

    def test_hard_attr(self):
        r = CircleRegion([0.0, 0.0], hard=False)
        assert r.hard is False

    def test_store_flexible(self):
        r = CircleRegion([0.0, 0.0], radius=10.0, size=5.0)
        n = r.current_nodes.shape[0]
        warp = np.ones((n, 2)) * 2.0
        r.store_flexible(warp)
        assert r.counter == 1

    def test_repr(self):
        r = CircleRegion([0.0, 0.0])
        assert "CircleRegion" in repr(r)


class TestPathRegion:
    def test_auto_centre(self):
        nodes = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
        r = PathRegion(nodes)
        np.testing.assert_allclose(r.current_centre, [2.0, 2.0], atol=1e-12)

    def test_explicit_centre(self):
        nodes = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
        r = PathRegion(nodes, centre=[1.0, 0.5])
        np.testing.assert_allclose(r.current_centre, [1.0, 0.5], atol=1e-12)

    def test_store_flexible_appends(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        r = PathRegion(nodes)
        warp = np.ones((4, 2))
        r.store_flexible(warp)
        assert r.counter == 1

    def test_store_flexible_node_update(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        r = PathRegion(nodes)
        warp = np.full((4, 2), [1.0, 0.5])
        r.store_flexible(warp)
        # current_nodes unchanged until update() is called
        assert r.counter == 1

    def test_update_from_filepath(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        r = PathRegion(nodes)
        warp = np.full((4, 2), [1.0, 1.0])
        r.store_flexible(warp)
        r.update("image_001.jpg")
        np.testing.assert_allclose(r.current_nodes[0], [1.0, 1.0], atol=1e-12)

    def test_repr(self):
        nodes = np.zeros((3, 2))
        r = PathRegion(nodes)
        assert "PathRegion" in repr(r)


# ===========================================================================
# Meshing
# ===========================================================================

class TestMaskImage:
    def test_shape(self):
        bn = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])
        m = mask_image((50, 50), bn)
        assert m.shape == (50, 50)

    def test_dtype(self):
        bn = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])
        m = mask_image((50, 50), bn)
        assert m.dtype == np.uint8

    def test_interior_is_one(self):
        bn = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])
        m = mask_image((50, 50), bn)
        assert m[25, 25] == 1

    def test_exterior_is_zero(self):
        bn = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])
        m = mask_image((50, 50), bn)
        assert m[2, 2] == 0

    def test_soft_boundary_fills_all(self):
        bn = np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]])
        m = mask_image((20, 20), bn, boundary_hard=False)
        assert np.all(m == 1)

    def test_exclusion_zeros_interior(self):
        boundary = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]])
        exclusion = np.array([[15.0, 15.0], [30.0, 15.0], [30.0, 30.0], [15.0, 30.0]])
        m = mask_image((50, 50), boundary, exclusion_nodes=[exclusion], exclusions_hard=[True])
        assert m[22, 22] == 0  # well inside exclusion
        assert m[5, 5] == 1    # outside exclusion


