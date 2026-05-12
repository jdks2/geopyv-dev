"""
Phase 3 validation tests for the mesh module PyO3 wrapper.

Each test mirrors its Phase 1 counterpart in geopyv/tests/mesh/fixtures.py,
asserting the same golden values via geopyv_dev module-level free functions.

Tolerance tiers (per plan):
  Tier A  atol=1e-12  pure matrix algebra
  Tier B  rtol=1e-8   floating-point arithmetic chains
"""

import numpy as np
import pytest

from geopyv_dev import (
    mesh_connectivity,
    mesh_corr,
    mesh_element_area,
    mesh_element_strains,
    mesh_find_seed_node,
    mesh_flow_calc,
    mesh_r_calc,
    mesh_shape_function,
)

try:
    from geopyv_dev import mesh_adaptive_target_areas
    _HAS_ADAPTIVE = True
except ImportError:
    _HAS_ADAPTIVE = False

_skip_adaptive = pytest.mark.skipif(
    not _HAS_ADAPTIVE, reason="mesh_adaptive_target_areas not yet implemented"
)

# ---------------------------------------------------------------------------
# Synthetic meshes (identical to Phase 1 fixtures)
# ---------------------------------------------------------------------------

NODES_O1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS_O1 = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

NODES_O2 = np.array(
    [
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [0.0, 1.0],  # 2
        [1.0, 1.0],  # 3
        [0.5, 0.0],  # 4
        [0.5, 0.5],  # 5
        [0.0, 0.5],  # 6
        [1.0, 0.5],  # 7
        [0.5, 1.0],  # 8
    ]
)
ELEMS_O2 = np.array(
    [
        [0, 1, 2, 4, 5, 6],
        [1, 3, 2, 7, 8, 5],
    ],
    dtype=np.int64,
)

# ---------------------------------------------------------------------------
# Tests: mesh_element_area  [Tier A]
# ---------------------------------------------------------------------------


def test_element_area_unit_right_triangle():
    """Single right triangle with legs 1 — area = 0.5."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2]], dtype=np.int64)
    areas = mesh_element_area(nodes, elements)
    assert areas.shape == (1,)
    np.testing.assert_allclose(areas[0], 0.5, atol=1e-12)


def test_element_area_two_elements():
    areas = mesh_element_area(NODES_O1, ELEMS_O1)
    assert areas.shape == (2,)
    np.testing.assert_allclose(areas, [0.5, 0.5], atol=1e-12)


def test_element_area_sign_positive():
    """CW winding gives negative determinant — magnitude is 0.5."""
    nodes = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    elements = np.array([[0, 1, 2]], dtype=np.int64)
    areas = mesh_element_area(nodes, elements)
    assert abs(areas[0]) == pytest.approx(0.5, abs=1e-12)


def test_element_area_scaled_triangle():
    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    elements = np.array([[0, 1, 2]], dtype=np.int64)
    areas = mesh_element_area(nodes, elements)
    np.testing.assert_allclose(areas[0], 2.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: mesh_shape_function  [Tier A]
# ---------------------------------------------------------------------------


def test_shape_function_order1_N_sums_to_one():
    N, dN, d2N = mesh_shape_function(1)
    assert abs(np.sum(N) - 1.0) < 1e-12


def test_shape_function_order1_d2N_is_none():
    _, _, d2N = mesh_shape_function(1)
    assert d2N is None


def test_shape_function_order1_dN_shape():
    _, dN, _ = mesh_shape_function(1)
    assert dN.shape == (2, 3)


def test_shape_function_order2_N_values():
    N, _, _ = mesh_shape_function(2)
    np.testing.assert_allclose(N, [-1 / 9, -1 / 9, -1 / 9, 4 / 9, 4 / 9, 4 / 9], atol=1e-12)


def test_shape_function_order2_dN_shape():
    _, dN, _ = mesh_shape_function(2)
    assert dN.shape == (2, 6)


def test_shape_function_order2_d2N_shape():
    _, _, d2N = mesh_shape_function(2)
    assert d2N.shape == (3, 6)


# ---------------------------------------------------------------------------
# Tests: mesh_element_strains  [Tier A / B]
# ---------------------------------------------------------------------------


def test_element_strains_pure_translation_order1():
    """Pure translation → all strain components zero."""
    dx, dy = 3.5, -2.1
    disps = np.tile([dx, dy], (4, 1))
    warps = mesh_element_strains(NODES_O1, ELEMS_O1, disps, 1)
    np.testing.assert_allclose(warps[:, 0], dx, atol=1e-12)
    np.testing.assert_allclose(warps[:, 1], dy, atol=1e-12)
    np.testing.assert_allclose(warps[:, 2:6], 0.0, atol=1e-10)


def test_element_strains_pure_x_shear_order1():
    """Node 1 displaced by (1,0) on unit right triangle."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2]], dtype=np.int64)
    disps = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    warps = mesh_element_strains(nodes, elements, disps, 1)
    np.testing.assert_allclose(warps[0, 0], 1 / 3, atol=1e-12)
    np.testing.assert_allclose(warps[0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(warps[0, 2], 1.0, atol=1e-10)   # du/dx
    np.testing.assert_allclose(warps[0, 3], 0.0, atol=1e-10)   # dv/dx
    np.testing.assert_allclose(warps[0, 4], 0.0, atol=1e-10)   # du/dy
    np.testing.assert_allclose(warps[0, 5], 0.0, atol=1e-10)   # dv/dy


def test_element_strains_warps_shape_order1():
    disps = np.zeros((4, 2))
    warps = mesh_element_strains(NODES_O1, ELEMS_O1, disps, 1)
    assert warps.shape == (2, 12)


def test_element_strains_pure_translation_order2():
    dx, dy = 1.0, 2.0
    disps = np.tile([dx, dy], (9, 1))
    warps = mesh_element_strains(NODES_O2, ELEMS_O2, disps, 2)
    np.testing.assert_allclose(warps[:, 0], dx, atol=1e-12)
    np.testing.assert_allclose(warps[:, 1], dy, atol=1e-12)
    np.testing.assert_allclose(warps[:, 2:], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: mesh_connectivity  [exact]
# ---------------------------------------------------------------------------


def test_connectivity_order1_corner_node():
    result = mesh_connectivity(ELEMS_O1, 1, 0, False)
    np.testing.assert_array_equal(sorted(result), [1, 2])


def test_connectivity_order1_shared_node():
    result = mesh_connectivity(ELEMS_O1, 1, 1, False)
    np.testing.assert_array_equal(sorted(result), [0, 2, 3])


def test_connectivity_order1_full_equals_immediate():
    r_imm = sorted(mesh_connectivity(ELEMS_O1, 1, 1, False))
    r_full = sorted(mesh_connectivity(ELEMS_O1, 1, 1, True))
    np.testing.assert_array_equal(r_imm, r_full)


def test_connectivity_order2_corner_returns_adjacent_mids():
    result = mesh_connectivity(ELEMS_O2, 2, 0, False)
    # Corner 0 at pos 0 in elem 0=[0,1,2,4,5,6] → row[3::2] = [4, 6]
    np.testing.assert_array_equal(sorted(result), [4, 6])


def test_connectivity_order2_midpoint_returns_adjacent_corners():
    result = mesh_connectivity(ELEMS_O2, 2, 4, False)
    # Midpoint 4 at pos 3 in elem 0 → row[:2] = [0, 1]
    np.testing.assert_array_equal(sorted(result), [0, 1])


def test_connectivity_order2_full_true_returns_all_element_peers():
    result = mesh_connectivity(ELEMS_O2, 2, 1, True)
    assert 0 in result and 2 in result and 3 in result


# ---------------------------------------------------------------------------
# Tests: mesh_find_seed_node  [exact]
# ---------------------------------------------------------------------------


def test_find_seed_node_exact_match():
    assert mesh_find_seed_node(NODES_O1, [1.0, 0.0]) == 1


def test_find_seed_node_nearest():
    assert mesh_find_seed_node(NODES_O1, [0.4, 0.4]) == 0


def test_find_seed_node_returns_int():
    result = mesh_find_seed_node(NODES_O1, [0.5, 0.5])
    assert isinstance(result, (int, np.integer))


# ---------------------------------------------------------------------------
# Tests: mesh_corr  [exact]
# ---------------------------------------------------------------------------


def test_corr_no_outliers():
    C = np.ones(10) * 0.9
    result = mesh_corr(C)
    assert len(result) == 0


def test_corr_detects_single_outlier():
    C = np.concatenate([np.ones(19) * 0.9, [-10.0]])
    result = mesh_corr(C)
    assert 19 in result


def test_corr_outlier_count():
    rng = np.random.default_rng(42)
    C = rng.uniform(0.7, 1.0, 100)
    C[0] = -5.0
    C_LQ = np.percentile(C, 25)
    C_IQR = np.percentile(C, 75) - C_LQ
    expected = np.argwhere(C < C_LQ - 2.5 * C_IQR).flatten()
    result = np.sort(mesh_corr(C))
    np.testing.assert_array_equal(result, np.sort(expected))


# ---------------------------------------------------------------------------
# Tests: mesh_flow_calc  [Tier B]
# ---------------------------------------------------------------------------


def test_flow_calc_aligned_returns_one():
    disps = np.array([[1.0, 0.0]] * 4)
    result = mesh_flow_calc(0, disps, ELEMS_O1, 1)
    assert result == pytest.approx(1.0, abs=1e-10)


def test_flow_calc_anti_aligned_returns_minus_one():
    disps = np.array([[1.0, 0.0]] * 4)
    disps[0] = [-1.0, 0.0]
    result = mesh_flow_calc(0, disps, ELEMS_O1, 1, displacement=disps[0].tolist())
    assert result == pytest.approx(-1.0, abs=1e-10)


def test_flow_calc_no_neighbours_returns_minus_one():
    disps = np.array([[1.0, 0.0]] * 4)
    result = mesh_flow_calc(0, disps, ELEMS_O1, 1, exclude=[1, 2])
    assert result == -1.0


# ---------------------------------------------------------------------------
# Tests: mesh_r_calc  [Tier A]
# ---------------------------------------------------------------------------


def test_r_calc_unit_displacement():
    assert mesh_r_calc([1.0, 0.0]) == pytest.approx(1.0, abs=1e-12)


def test_r_calc_diagonal():
    assert mesh_r_calc([3.0, 4.0]) == pytest.approx(5.0, abs=1e-12)


def test_r_calc_zero():
    assert mesh_r_calc([0.0, 0.0]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Tests: mesh_adaptive_target_areas  [Tier B]
# ---------------------------------------------------------------------------


@_skip_adaptive
def test_adaptive_target_areas_uniform_shear():
    """Uniform shear × area → D constant → ratio = 1 → target = area.

    alpha = 0.5 follows the Python convention: clip range is [alpha, 1/alpha]
    = [0.5, 2.0], so alpha < 1 is required.
    """
    n_elems = 4
    warps = np.zeros((n_elems, 12))
    warps[:, 3] = 0.1  # dv/dx
    warps[:, 4] = 0.1  # du/dy
    areas = np.ones(n_elems) * 0.5
    alpha = 0.5  # clip range [0.5, 2.0]; D uniform → ratio = 1 → target = area
    target = mesh_adaptive_target_areas(warps, areas, alpha)
    np.testing.assert_allclose(target, areas, rtol=1e-8)


@_skip_adaptive
def test_adaptive_target_areas_zero_shear():
    """Zero shear everywhere → D = 0 → ratio defaults to 1 → target = area."""
    warps = np.zeros((3, 12))
    areas = np.array([0.5, 0.25, 1.0])
    target = mesh_adaptive_target_areas(warps, areas, 0.5)
    np.testing.assert_allclose(target, areas, rtol=1e-8)


@_skip_adaptive
def test_adaptive_target_areas_shape():
    warps = np.zeros((5, 12))
    areas = np.ones(5)
    target = mesh_adaptive_target_areas(warps, areas, 0.5)
    assert target.shape == (5,)
