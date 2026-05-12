"""
Phase 3 validation tests for geopyv_dev.particle.

Each test asserts the same hard-coded golden values as the Phase 1 fixture
(geopyv/tests/particle/fixtures.py), but imports from `geopyv_dev` instead
of replicating method bodies.

Tolerance tiers (per plan):
  Tier A  atol=1e-12  pure matrix algebra
  Tier B  rtol=1e-8   floating-point arithmetic chains
"""

import numpy as np
import pytest

from geopyv_dev import (
    Particle,
    ParticleWrapper,
    particle_local_coordinates,
    particle_shape_function,
    particle_warp_increment,
    particle_element_locator,
    particle_compute_centroids,
    particle_strain_def,
    particle_vol_strains,
)

# ---------------------------------------------------------------------------
# Shared test data — mirrors fixtures.py
# ---------------------------------------------------------------------------

TRI_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

NODES_O1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS_O1 = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

NODES_O2 = np.array([
    [0.0, 0.0],  # 0
    [1.0, 0.0],  # 1
    [0.0, 1.0],  # 2
    [1.0, 1.0],  # 3
    [0.5, 0.0],  # 4 mid(0,1)
    [0.5, 0.5],  # 5 mid(1,2)
    [0.0, 0.5],  # 6 mid(0,2)
    [1.0, 0.5],  # 7 mid(1,3)
    [0.5, 1.0],  # 8 mid(2,3)
])
ELEMS_O2 = np.array([
    [0, 1, 2, 4, 5, 6],
    [1, 3, 2, 7, 8, 5],
], dtype=np.int64)

CENTROIDS_O1 = np.array([
    np.mean(NODES_O1[[0, 1, 2]], axis=0),
    np.mean(NODES_O1[[1, 3, 2]], axis=0),
])


# ===========================================================================
# Tests: particle_local_coordinates
# Tolerance: Tier A (atol=1e-12)
# ===========================================================================

def test_local_coordinates_centroid():
    z, e, t, _ = particle_local_coordinates([1/3, 1/3], TRI_NODES)
    np.testing.assert_allclose([z, e, t], [1/3, 1/3, 1/3], atol=1e-12)


def test_local_coordinates_vertex0():
    z, e, t, _ = particle_local_coordinates([0.0, 0.0], TRI_NODES)
    np.testing.assert_allclose([z, e, t], [1.0, 0.0, 0.0], atol=1e-12)


def test_local_coordinates_vertex1():
    z, e, t, _ = particle_local_coordinates([1.0, 0.0], TRI_NODES)
    np.testing.assert_allclose([z, e, t], [0.0, 1.0, 0.0], atol=1e-12)


def test_local_coordinates_vertex2():
    z, e, t, _ = particle_local_coordinates([0.0, 1.0], TRI_NODES)
    np.testing.assert_allclose([z, e, t], [0.0, 0.0, 1.0], atol=1e-12)


def test_local_coordinates_partition_of_unity():
    z, e, t, _ = particle_local_coordinates([0.3, 0.25], TRI_NODES)
    np.testing.assert_allclose(z + e + t, 1.0, atol=1e-12)


def test_local_coordinates_interior_point():
    z, e, t, _ = particle_local_coordinates([0.25, 0.25], TRI_NODES)
    np.testing.assert_allclose([z, e, t], [0.5, 0.25, 0.25], atol=1e-12)


def test_local_coordinates_reconstruction():
    coord = np.array([0.2, 0.3])
    z, e, t, _ = particle_local_coordinates(coord, TRI_NODES)
    recovered = np.array([z, e, t]) @ TRI_NODES
    np.testing.assert_allclose(recovered, coord, atol=1e-12)


# ===========================================================================
# Tests: particle_shape_function
# Tolerance: Tier A (atol=1e-12)
# ===========================================================================

def test_shape_function_order1_centroid_N():
    N, dN, d2N = particle_shape_function(1, 1/3, 1/3, 1/3)
    np.testing.assert_allclose(N, [1/3, 1/3, 1/3], atol=1e-12)


def test_shape_function_order1_dN():
    _, dN, _ = particle_shape_function(1, 0.5, 0.25, 0.25)
    expected = np.array([[1, 0, -1], [0, 1, -1]])
    np.testing.assert_array_equal(dN, expected)


def test_shape_function_order1_d2N_is_None():
    _, _, d2N = particle_shape_function(1, 1/3, 1/3, 1/3)
    assert d2N is None


def test_shape_function_order1_vertex():
    N, _, _ = particle_shape_function(1, 1.0, 0.0, 0.0)
    np.testing.assert_allclose(N, [1.0, 0.0, 0.0], atol=1e-12)


def test_shape_function_order2_centroid_N():
    N, _, _ = particle_shape_function(2, 1/3, 1/3, 1/3)
    expected = np.array([-1/9, -1/9, -1/9, 4/9, 4/9, 4/9])
    np.testing.assert_allclose(N, expected, atol=1e-12)


def test_shape_function_order2_partition_of_unity():
    N, _, _ = particle_shape_function(2, 0.4, 0.35, 0.25)
    np.testing.assert_allclose(np.sum(N), 1.0, atol=1e-12)


def test_shape_function_order2_vertex():
    N, _, _ = particle_shape_function(2, 1.0, 0.0, 0.0)
    np.testing.assert_allclose(N, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_shape_function_order2_midpoint_node3():
    N, _, _ = particle_shape_function(2, 0.5, 0.5, 0.0)
    np.testing.assert_allclose(N, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-12)


def test_shape_function_order2_d2N():
    _, _, d2N = particle_shape_function(2, 0.3, 0.4, 0.3)
    expected = np.array([
        [4, 0, 4, 0,  0, -8],
        [0, 0, 4, 4, -4, -4],
        [0, 4, 4, 0, -8,  0],
    ])
    np.testing.assert_array_equal(d2N, expected)


# ===========================================================================
# Tests: particle_warp_increment
# Tolerance: Tier A (atol=1e-12)
# ===========================================================================

def test_warp_increment_pure_translation_order1():
    u, v = 3.5, -2.1
    disps = np.tile([u, v], (3, 1))
    w = particle_warp_increment([1/3, 1/3], TRI_NODES, disps, 1)
    np.testing.assert_allclose(w[:2], [u, v], atol=1e-12)
    np.testing.assert_allclose(w[2:], 0.0, atol=1e-12)


def test_warp_increment_pure_x_stretch_order1():
    disps = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    w = particle_warp_increment([1/3, 1/3], TRI_NODES, disps, 1)
    np.testing.assert_allclose(w[0], 1/3, atol=1e-12)
    np.testing.assert_allclose(w[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(w[2], 1.0, atol=1e-12)   # du/dx
    np.testing.assert_allclose(w[3:6], 0.0, atol=1e-12)


def test_warp_increment_pure_y_stretch_order1():
    disps = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    w = particle_warp_increment([1/3, 1/3], TRI_NODES, disps, 1)
    np.testing.assert_allclose(w[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(w[1], 1/3, atol=1e-12)
    np.testing.assert_allclose(w[2:5], 0.0, atol=1e-12)
    np.testing.assert_allclose(w[5], 1.0, atol=1e-12)   # dv/dy


def test_warp_increment_pure_translation_order2():
    u, v = 1.5, 2.5
    disps = np.tile([u, v], (6, 1))
    nodes6 = NODES_O2[:6]
    w = particle_warp_increment([1/3, 1/3], nodes6, disps, 2)
    np.testing.assert_allclose(w[:2], [u, v], atol=1e-11)
    np.testing.assert_allclose(w[2:], 0.0, atol=1e-11)


# ===========================================================================
# Tests: particle_compute_centroids + particle_element_locator
# ===========================================================================

def test_compute_centroids_shape():
    c = particle_compute_centroids(NODES_O1, ELEMS_O1)
    assert c.shape == (2, 2)


def test_compute_centroids_values():
    c = particle_compute_centroids(NODES_O1, ELEMS_O1)
    np.testing.assert_allclose(c[0], np.mean(NODES_O1[[0, 1, 2]], axis=0), atol=1e-12)
    np.testing.assert_allclose(c[1], np.mean(NODES_O1[[1, 3, 2]], axis=0), atol=1e-12)


def test_element_locator_centroid_elem0():
    c = particle_compute_centroids(NODES_O1, ELEMS_O1)
    idx = particle_element_locator(list(c[0]), NODES_O1, ELEMS_O1, c)
    assert idx == 0


def test_element_locator_centroid_elem1():
    c = particle_compute_centroids(NODES_O1, ELEMS_O1)
    idx = particle_element_locator(list(c[1]), NODES_O1, ELEMS_O1, c)
    assert idx == 1


def test_element_locator_interior_point_elem0():
    c = particle_compute_centroids(NODES_O1, ELEMS_O1)
    idx = particle_element_locator([0.2, 0.2], NODES_O1, ELEMS_O1, c)
    assert idx == 0


def test_element_locator_interior_point_elem1():
    c = particle_compute_centroids(NODES_O1, ELEMS_O1)
    idx = particle_element_locator([0.8, 0.8], NODES_O1, ELEMS_O1, c)
    assert idx == 1


def test_element_locator_exterior_fallback():
    c = particle_compute_centroids(NODES_O1, ELEMS_O1)
    idx = particle_element_locator([2.0, 2.0], NODES_O1, ELEMS_O1, c)
    assert idx == 1  # element 1 centroid (2/3, 2/3) is nearer


def test_element_locator_returns_int():
    c = particle_compute_centroids(NODES_O1, ELEMS_O1)
    idx = particle_element_locator(list(c[0]), NODES_O1, ELEMS_O1, c)
    assert isinstance(idx, int)


# ===========================================================================
# Tests: particle_strain_def
# Tolerance: Tier A (atol=1e-12) zero cases; Tier B (rtol=1e-8) otherwise
# ===========================================================================

def test_strain_def_all_zeros():
    warps = np.zeros((3, 6))
    strains, incs = particle_strain_def(warps)
    np.testing.assert_allclose(strains, 0.0, atol=1e-12)
    np.testing.assert_allclose(incs, 0.0, atol=1e-12)


def test_strain_def_eps_xx():
    warps = np.zeros((3, 6))
    warps[:, 2] = [0.0, 0.1, 0.2]
    strains, _ = particle_strain_def(warps, factor=0.0)
    np.testing.assert_allclose(strains[:, 0], [0.0, -0.1, -0.2], rtol=1e-8)


def test_strain_def_eps_yy():
    warps = np.zeros((3, 6))
    warps[:, 5] = [0.0, 0.05, 0.10]
    strains, _ = particle_strain_def(warps, factor=0.0)
    np.testing.assert_allclose(strains[:, 1], [0.0, -0.05, -0.10], rtol=1e-8)


def test_strain_def_shear():
    warps = np.zeros((3, 6))
    warps[:, 3] = [0.0, 0.02, 0.04]
    warps[:, 4] = [0.0, 0.04, 0.08]
    strains, _ = particle_strain_def(warps)
    expected = -(warps[:, 3] + warps[:, 4]) / 2
    np.testing.assert_allclose(strains[:, 5], expected, rtol=1e-8)


def test_strain_def_increment_shape():
    warps = np.zeros((4, 6))
    strains, incs = particle_strain_def(warps)
    assert strains.shape == (4, 6)
    assert incs.shape == (3, 6)


def test_strain_def_factor_zero_no_correction():
    warps = np.zeros((3, 6))
    warps[:, 2] = [0.0, 0.1, 0.2]
    warps[:, 5] = [0.0, 0.05, 0.1]
    _, incs = particle_strain_def(warps, factor=0.0)
    np.testing.assert_allclose(incs[:, 0], [-0.1, -0.1], rtol=1e-8)
    np.testing.assert_allclose(incs[:, 1], [-0.05, -0.05], rtol=1e-8)


def test_strain_def_factor_one_removes_mean():
    warps = np.zeros((3, 6))
    warps[:, 2] = [0.0, 0.1, 0.2]
    warps[:, 5] = [0.0, 0.05, 0.1]
    _, incs = particle_strain_def(warps, factor=1.0)
    np.testing.assert_allclose(incs[:, 0], [-0.025, -0.025], rtol=1e-8)
    np.testing.assert_allclose(incs[:, 1], [ 0.025,  0.025], rtol=1e-8)


def test_strain_def_cumsum_eps_xx_factor_zero():
    warps = np.zeros((4, 6))
    warps[:, 2] = [0.0, 0.05, 0.12, 0.20]
    strains, incs = particle_strain_def(warps, factor=0.0)
    np.testing.assert_allclose(strains[1:, 0], np.cumsum(incs[:, 0]), rtol=1e-8)


# ===========================================================================
# Tests: particle_vol_strains
# ===========================================================================

def test_vol_strains_all_same():
    vols = np.array([1.0, 1.0, 1.0])
    vs = particle_vol_strains(vols)
    np.testing.assert_allclose(vs, [0.0, 0.0, 0.0], atol=1e-12)


def test_vol_strains_doubles():
    vols = np.array([1.0, 2.0, 3.0])
    vs = particle_vol_strains(vols)
    # (v - v0) / v0
    np.testing.assert_allclose(vs, [0.0, 1.0, 2.0], atol=1e-12)


# ===========================================================================
# Tests: Particle class
# ===========================================================================

def test_particle_new_defaults():
    p = Particle([10.0, 20.0], [0.0]*6, 1e9, 3)
    assert p.mesh_order == 1
    assert p.track is True
    assert not p.solved
    assert p.coordinates.shape == (3, 2)
    np.testing.assert_allclose(p.coordinates[0], [10.0, 20.0], atol=1e-12)


def test_particle_new_invalid_volume():
    with pytest.raises(Exception):
        Particle([0.0, 0.0], [0.0]*6, -1.0, 3)


def test_particle_repr():
    p = Particle([0.0, 0.0], [0.0]*6, 1.0, 2)
    r = repr(p)
    assert "Particle" in r


def test_particle_solve_increment_pure_translation():
    """Uniform displacement (u=0.5, v=0.1) → strain increments ≈ 0."""
    nodes = NODES_O1
    elements = ELEMS_O1
    disps = np.tile([0.5, 0.1], (4, 1))

    p = Particle([1/3, 1/3], [0.0]*6, 1e9, 2)
    ok = p.solve_increment(0, nodes, elements, disps, 1)
    assert ok is True

    # Lagrangian → coordinate moves
    np.testing.assert_allclose(
        p.coordinates[1], [1/3 + 0.5, 1/3 + 0.1], atol=1e-10
    )
    # Pure translation → all strain components ≈ 0
    np.testing.assert_allclose(p.incs[1, 2:], 0.0, atol=1e-10)


def test_particle_solve_increment_eulerian():
    """Eulerian (track=False) → coordinate stays at initial position."""
    nodes = NODES_O1
    elements = ELEMS_O1
    disps = np.tile([0.5, 0.1], (4, 1))

    p = Particle([1/3, 1/3], [0.0]*6, 1e9, 2, track=False)
    p.solve_increment(0, nodes, elements, disps, 1)

    np.testing.assert_allclose(
        p.coordinates[1], [1/3, 1/3], atol=1e-10
    )


def test_particle_warp_accumulation():
    """Two identical translation steps with ref_update → warps[2] ≈ 2 × warps[1].

    Without ref_update both increments measure from reference_index=0 and
    warps[2,0] equals warps[1,0] (not double).  With ref_update=True at
    step 1 the reference shifts to frame 1 and warp accumulates.
    """
    nodes = NODES_O1
    elements = ELEMS_O1
    disps = np.tile([0.3, 0.0], (4, 1))

    # Without ref_update: reference stays at 0 for both steps → same warp
    p = Particle([1/3, 1/3], [0.0]*6, 1e9, 3)
    p.solve_increment(0, nodes, elements, disps, 1, ref_update=False)
    p.solve_increment(1, nodes, elements, disps, 1, ref_update=False)
    np.testing.assert_allclose(p.warps[2, 0], p.warps[1, 0], rtol=1e-8)

    # With ref_update=True at step 1: reference shifts to frame 1 → accumulates
    p2 = Particle([1/3, 1/3], [0.0]*6, 1e9, 3)
    p2.solve_increment(0, nodes, elements, disps, 1, ref_update=False)
    p2.solve_increment(1, nodes, elements, disps, 1, ref_update=True)
    np.testing.assert_allclose(p2.warps[2, 0], p2.warps[1, 0] * 2.0, rtol=1e-8)


def test_particle_solve_full_sequence():
    """solve() with 2 identical meshes accumulates correctly."""
    nodes = NODES_O1
    elements = ELEMS_O1
    disps = np.tile([0.1, 0.0], (4, 1))

    p = Particle([1/3, 1/3], [0.0]*6, 1e6, 3)
    sol = p.solve(
        [nodes, nodes],
        [elements, elements],
        [disps, disps],
        [1, 1],
    )

    assert isinstance(sol, ParticleWrapper)
    assert p.solved is True
    assert sol.coordinates.shape == (3, 2)
    assert sol.strains.shape == (3, 6)
    assert sol.strain_incs.shape == (2, 6)


def test_particle_solve_wrong_mesh_count():
    """solve() with wrong number of meshes raises an error."""
    nodes = NODES_O1
    elements = ELEMS_O1
    disps = np.tile([0.1, 0.0], (4, 1))

    # inc_no=3 requires 2 mesh entries; supply only 1
    p = Particle([1/3, 1/3], [0.0]*6, 1e6, 3)
    with pytest.raises(Exception):
        p.solve([nodes], [elements], [disps], [1])


def test_particle_solution_fields():
    """ParticleSolution has all expected array fields."""
    nodes = NODES_O1
    elements = ELEMS_O1
    disps = np.tile([0.05, 0.02], (4, 1))

    p = Particle([1/3, 1/3], [0.0]*6, 1e6, 3)
    sol = p.solve([nodes, nodes], [elements, elements], [disps, disps], [1, 1])

    assert sol.coordinates.shape == (3, 2)
    assert sol.warps.shape == (3, 6)
    assert sol.incs.shape == (3, 6)
    assert sol.volumes.shape == (3,)
    assert sol.strains.shape == (3, 6)
    assert sol.strain_incs.shape == (2, 6)
    assert sol.vol_strains.shape == (3,)
    assert isinstance(sol.reference_update_register, list)


def test_particle_solution_repr():
    nodes = NODES_O1
    elements = ELEMS_O1
    disps = np.zeros((4, 2))
    p = Particle([1/3, 1/3], [0.0]*6, 1.0, 2)
    sol = p.solve([nodes], [elements], [disps], [1])
    assert "ParticleSolution" in repr(sol)
