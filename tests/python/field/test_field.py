"""
Phase 3 validation tests for geopyv_dev.field.

Asserts the same golden values as the Phase 1 fixture
(geopyv/tests/field/fixtures.py) but imports from `geopyv_dev`.

Tolerance tiers:
  Tier A  atol=1e-12  pure matrix algebra
  Tier B  rtol=1e-8   floating-point arithmetic chains
"""

import numpy as np
import pytest

from geopyv_dev import (
    Field,
    FieldSolution,
    ParticleSolution,
    field_distribute_particles,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

NODES_O1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS_O1 = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

NODES_TRI = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
ELEMS_TRI = np.array([[0, 1, 2]], dtype=np.int64)

NODES_GRID = np.array([
    [0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]
])
ELEMS_GRID = np.array([
    [0, 1, 4],
    [1, 3, 4],
    [3, 2, 4],
    [2, 0, 4],
], dtype=np.int64)


def make_disps(u, v, n_nodes=4):
    return np.tile([u, v], (n_nodes, 1))


# ===========================================================================
# Tests: field_distribute_particles — coordinates
# Tolerance: Tier A (atol=1e-12)
# ===========================================================================

def test_distribute_particles_coordinates_unit_triangles():
    coords, _ = field_distribute_particles(NODES_O1, ELEMS_O1)
    np.testing.assert_allclose(coords[0], np.mean(NODES_O1[[0, 1, 2]], axis=0), atol=1e-12)
    np.testing.assert_allclose(coords[1], np.mean(NODES_O1[[1, 3, 2]], axis=0), atol=1e-12)


def test_distribute_particles_coordinates_single_triangle():
    coords, _ = field_distribute_particles(NODES_TRI, ELEMS_TRI)
    np.testing.assert_allclose(coords[0], np.mean(NODES_TRI, axis=0), atol=1e-12)


def test_distribute_particles_coordinates_shape():
    coords, _ = field_distribute_particles(NODES_GRID, ELEMS_GRID)
    assert coords.shape == (4, 2)


def test_distribute_particles_uses_only_corner_nodes():
    elems_o2 = np.array([[0, 1, 2, 99, 99, 99], [1, 3, 2, 99, 99, 99]], dtype=np.int64)
    coords_o2, _ = field_distribute_particles(NODES_O1, elems_o2)
    coords_o1, _ = field_distribute_particles(NODES_O1, ELEMS_O1)
    np.testing.assert_allclose(coords_o2, coords_o1, atol=1e-12)


# ===========================================================================
# Tests: field_distribute_particles — volumes
# Tolerance: Tier A (atol=1e-12)
# ===========================================================================

def test_distribute_particles_volumes_unit_right_triangle():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]], dtype=np.int64)
    _, vols = field_distribute_particles(nodes, elems, depth=1.0)
    np.testing.assert_allclose(vols[0], 0.5, atol=1e-12)


def test_distribute_particles_volumes_unit_square_total():
    _, vols = field_distribute_particles(NODES_O1, ELEMS_O1, depth=1.0)
    np.testing.assert_allclose(np.sum(vols), 1.0, atol=1e-12)


def test_distribute_particles_volumes_depth_scaling():
    _, vols_1 = field_distribute_particles(NODES_O1, ELEMS_O1, depth=1.0)
    _, vols_d = field_distribute_particles(NODES_O1, ELEMS_O1, depth=3.7)
    np.testing.assert_allclose(vols_d, vols_1 * 3.7, atol=1e-12)


def test_distribute_particles_volumes_equilateral():
    _, vols = field_distribute_particles(NODES_TRI, ELEMS_TRI, depth=1.0)
    np.testing.assert_allclose(vols[0], 1.0, atol=1e-12)


def test_distribute_particles_volumes_shape():
    _, vols = field_distribute_particles(NODES_GRID, ELEMS_GRID)
    assert vols.shape == (4,)


def test_distribute_particles_volumes_all_positive():
    _, vols = field_distribute_particles(NODES_GRID, ELEMS_GRID, depth=2.0)
    assert np.all(vols > 0)


def test_distribute_particles_grid_total_area():
    _, vols = field_distribute_particles(NODES_GRID, ELEMS_GRID, depth=1.0)
    np.testing.assert_allclose(np.sum(vols), 4.0, atol=1e-12)


# ===========================================================================
# Tests: Field constructor
# ===========================================================================

def test_field_new_defaults():
    coords, vols = field_distribute_particles(NODES_O1, ELEMS_O1)
    f = Field(coords, vols, inc_no=3)
    assert f.n_particles == 2
    assert f.inc_no == 3
    assert f.track is True
    assert f.depth == 1.0
    assert not f.solved


def test_field_new_lagrangian_false():
    coords, vols = field_distribute_particles(NODES_O1, ELEMS_O1)
    f = Field(coords, vols, inc_no=2, track=False)
    assert f.track is False


def test_field_new_depth():
    coords, vols = field_distribute_particles(NODES_O1, ELEMS_O1)
    f = Field(coords, vols, inc_no=2, depth=5.0)
    assert f.depth == 5.0


def test_field_new_invalid_volume():
    coords = np.array([[0.5, 0.5], [0.5, 0.8]])
    vols = np.array([-1.0, 0.5])
    with pytest.raises(Exception):
        Field(coords, vols, inc_no=2)


def test_field_new_mismatched_volumes():
    coords, vols = field_distribute_particles(NODES_O1, ELEMS_O1)
    with pytest.raises(Exception):
        Field(coords, vols[:1], inc_no=2)


def test_field_new_inc_no_too_small():
    coords, vols = field_distribute_particles(NODES_O1, ELEMS_O1)
    with pytest.raises(Exception):
        Field(coords, vols, inc_no=1)


def test_field_repr():
    coords, vols = field_distribute_particles(NODES_O1, ELEMS_O1)
    f = Field(coords, vols, inc_no=2)
    assert "Field" in repr(f)


# ===========================================================================
# Tests: Field.solve — free functions
# Tolerance: Tier B (rtol=1e-8)
# ===========================================================================

def _make_field(inc_no=2, track=True, depth=1.0):
    coords, vols = field_distribute_particles(NODES_O1, ELEMS_O1)
    return Field(coords, vols, inc_no=inc_no, track=track, depth=depth)


def _solve(field, disps, inc_no=None, **kwargs):
    n = inc_no or (field.inc_no - 1)
    nodes_list = [NODES_O1] * n
    elems_list = [ELEMS_O1] * n
    disps_list = [disps] * n
    orders = [1] * n
    return field.solve(nodes_list, elems_list, disps_list, orders, **kwargs)


def test_field_solve_returns_field_solution():
    f = _make_field()
    disps = make_disps(0.3, 0.0)
    sol = _solve(f, disps)
    assert isinstance(sol, FieldSolution)


def test_field_solve_sets_solved_flag():
    f = _make_field()
    assert not f.solved
    _solve(f, make_disps(0.0, 0.0))
    assert f.solved


def test_field_solve_pure_translation_no_strain():
    f = _make_field()
    sol = _solve(f, make_disps(0.5, 0.1))
    for p in sol.particles:
        # strain components (indices 2–5 of warp) ≈ 0
        np.testing.assert_allclose(p.warps[1, 2:], 0.0, atol=1e-10)


def test_field_solve_particle_count():
    f = _make_field()
    sol = _solve(f, make_disps(0.0, 0.0))
    assert len(sol.particles) == 2  # 2 elements in ELEMS_O1


def test_field_solve_particles_are_particle_solutions():
    f = _make_field()
    sol = _solve(f, make_disps(0.0, 0.0))
    for p in sol.particles:
        assert isinstance(p, ParticleSolution)


def test_field_solve_lagrangian_coordinate_shift():
    u = 0.25
    coords_init, _ = field_distribute_particles(NODES_O1, ELEMS_O1)
    f = _make_field(track=True)
    sol = _solve(f, make_disps(u, 0.0))
    for pi, p in enumerate(sol.particles):
        np.testing.assert_allclose(
            p.coordinates[1, 0], coords_init[pi, 0] + u, rtol=1e-8
        )


def test_field_solve_eulerian_coordinates_fixed():
    coords_init, _ = field_distribute_particles(NODES_O1, ELEMS_O1)
    f = _make_field(track=False)
    sol = _solve(f, make_disps(0.3, -0.1))
    for pi, p in enumerate(sol.particles):
        np.testing.assert_allclose(p.coordinates[1], coords_init[pi], atol=1e-12)


def test_field_solve_vol_totals_shape():
    f = _make_field(inc_no=3)
    sol = _solve(f, make_disps(0.0, 0.0), inc_no=2)
    assert sol.vol_totals.shape == (3,)


def test_field_solve_vol_totals_pure_translation():
    f = _make_field()
    coords, vols = field_distribute_particles(NODES_O1, ELEMS_O1)
    total = float(np.sum(vols))
    f = Field(coords, vols, inc_no=2)
    sol = _solve(f, make_disps(0.2, 0.0))
    # Pure translation: each particle volume unchanged → sum unchanged
    np.testing.assert_allclose(sol.vol_totals[0], total, rtol=1e-8)
    np.testing.assert_allclose(sol.vol_totals[1], total, rtol=1e-8)


def test_field_solve_x_stretch_eps_xx():
    # Node 1 and 3 displaced +0.1 in x → uniform 10% x-stretch on all elements
    disps = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.1, 0.0]])
    f = _make_field()
    sol = _solve(f, disps)
    for p in sol.particles:
        np.testing.assert_allclose(p.incs[1, 2], 0.1, rtol=1e-8)


def test_field_solve_ref_update_register_empty():
    f = _make_field()
    sol = _solve(f, make_disps(0.0, 0.0))
    assert sol.reference_update_register == []


def test_field_solve_ref_update_register_step0():
    f = _make_field(inc_no=3)
    sol = _solve(f, make_disps(0.0, 0.0), inc_no=2,
                 ref_updates=[False, True])
    assert sol.reference_update_register == [1]


def test_field_solve_two_increments_coordinate_warp():
    f = _make_field(inc_no=3)
    sol = _solve(f, make_disps(0.1, 0.0), inc_no=2)
    # Each particle has 3 frames
    for p in sol.particles:
        assert p.coordinates.shape == (3, 2)
        assert p.warps.shape == (3, 6)


def test_field_solve_wrong_mesh_count():
    # inc_no=3 needs 2 meshes; supplying 1 should raise
    f = _make_field(inc_no=3)
    with pytest.raises(Exception):
        f.solve(
            [NODES_O1],
            [ELEMS_O1],
            [make_disps(0.0, 0.0)],
            [1],
        )


def test_field_solution_repr():
    f = _make_field()
    sol = _solve(f, make_disps(0.0, 0.0))
    assert "FieldSolution" in repr(sol)


def test_field_volume_consistency():
    """vol_totals[0] equals sum of initial volumes."""
    coords, vols = field_distribute_particles(NODES_O1, ELEMS_O1)
    f = Field(coords, vols, inc_no=2)
    sol = _solve(f, make_disps(0.0, 0.0))
    # At frame 0 all particles have their initial volume
    np.testing.assert_allclose(
        sol.vol_totals[0], float(np.sum(vols)), rtol=1e-8
    )
