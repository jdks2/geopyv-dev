"""
Phase 3 validation tests for geopyv_dev.sequence.

Mirrors the Phase 1 fixture tests in geopyv/tests/sequence/fixtures.py,
asserting the same golden values via geopyv_dev.

Tolerance tiers (per plan):
  Tier A  atol=1e-12  pure matrix algebra
  Tier B  rtol=1e-8   floating-point arithmetic chains
"""

import os
import math
import numpy as np
import pytest

from geopyv_dev import (
    Sequence,
    SequenceSolution,
    MeshSolution,
    Circle,
    sequence_deformation_preconditioning,
)

# ---------------------------------------------------------------------------
# Synthetic mesh — unit square with two order-1 triangles
# ---------------------------------------------------------------------------

NODES_O1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS_O1 = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)


def make_mesh_solution(u, v, nodes=None, elements=None):
    """Build a synthetic MeshSolution with uniform displacement (u, v)."""
    if nodes is None:
        nodes = NODES_O1.copy()
    if elements is None:
        elements = ELEMS_O1.copy()
    n = nodes.shape[0]
    disps = np.tile([u, v], (n, 1))
    warps = np.zeros((elements.shape[0], 12))
    areas = np.array([0.5] * elements.shape[0])
    c_zncc = np.ones(n)
    p = np.zeros((n, 6))
    return MeshSolution(
        nodes=nodes,
        elements=elements,
        boundary=[0, 1, 2, 3],
        exclusions=[],
        areas=areas,
        warps=warps,
        displacements=disps,
        c_zncc=c_zncc,
        p=p,
        seed_node=0,
        mesh_order=1,
        subset_order=1,
    )


# ---------------------------------------------------------------------------
# Helpers: borders/segments/curves for a unit-square ROI
# ---------------------------------------------------------------------------

def square_roi(size):
    """100×100 square: borders, segments, curves in triangulation format."""
    borders = np.array(
        [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]],
        dtype=np.float64,
    )
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    curves = [[0, 1, 2, 3]]
    return borders, segments, curves


# Test image paths (relative to the repository root).
_HERE = os.path.dirname(__file__)
REF_IMG = os.path.abspath(os.path.join(_HERE, "../../../../geopyv/tests/ref.jpg"))
TAR_IMG = os.path.abspath(os.path.join(_HERE, "../../../../geopyv/tests/tar.jpg"))
IMAGES_AVAILABLE = os.path.isfile(REF_IMG) and os.path.isfile(TAR_IMG)

# ===========================================================================
# MeshSolution construction helper tests
# ===========================================================================


def test_mesh_solution_constructible():
    """MeshSolution can be constructed with synthetic data."""
    sol = make_mesh_solution(0.0, 0.0)
    assert sol is not None


def test_mesh_solution_repr():
    """MeshSolution.__repr__ returns a non-empty string."""
    sol = make_mesh_solution(0.0, 0.0)
    assert len(repr(sol)) > 0


# ===========================================================================
# Tests: Sequence construction validation
# ===========================================================================


def test_sequence_too_few_images_raises():
    """Sequence with only one image path raises an error."""
    borders, segments, curves = square_roi(100.0)
    with pytest.raises(Exception):
        Sequence(
            image_paths=["/tmp/only_one.jpg"],
            borders=borders,
            segments=segments,
            curves=curves,
            size_lower=5.0,
            size_upper=50.0,
            target_nodes=20,
        )


def test_sequence_nonexistent_image_raises():
    """Sequence raises an error when any image path does not exist."""
    borders, segments, curves = square_roi(100.0)
    with pytest.raises(Exception):
        Sequence(
            image_paths=["/no/such/file_a.jpg", "/no/such/file_b.jpg"],
            borders=borders,
            segments=segments,
            curves=curves,
            size_lower=5.0,
            size_upper=50.0,
            target_nodes=20,
        )


def test_sequence_size_lower_ge_upper_raises():
    """size_lower >= size_upper raises an error."""
    borders, segments, curves = square_roi(100.0)
    with pytest.raises(Exception):
        Sequence(
            image_paths=[REF_IMG, TAR_IMG],
            borders=borders,
            segments=segments,
            curves=curves,
            size_lower=100.0,
            size_upper=10.0,
            target_nodes=20,
        )


def test_sequence_zero_size_lower_raises():
    """size_lower = 0 raises an error."""
    borders, segments, curves = square_roi(100.0)
    with pytest.raises(Exception):
        Sequence(
            image_paths=[REF_IMG, TAR_IMG],
            borders=borders,
            segments=segments,
            curves=curves,
            size_lower=0.0,
            size_upper=50.0,
            target_nodes=20,
        )


# ===========================================================================
# Tests: Sequence properties
# ===========================================================================


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_n_pairs_two_images():
    """Two images → n_pairs = 1."""
    borders, segments, curves = square_roi(100.0)
    seq = Sequence(
        image_paths=[REF_IMG, TAR_IMG],
        borders=borders,
        segments=segments,
        curves=curves,
        size_lower=5.0,
        size_upper=50.0,
        target_nodes=10,
    )
    assert seq.n_pairs == 1


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_image_paths_round_trips():
    """image_paths getter returns the same paths passed in."""
    borders, segments, curves = square_roi(100.0)
    seq = Sequence(
        image_paths=[REF_IMG, TAR_IMG],
        borders=borders,
        segments=segments,
        curves=curves,
        size_lower=5.0,
        size_upper=50.0,
        target_nodes=10,
    )
    returned = seq.image_paths
    assert len(returned) == 2
    assert returned[0] == REF_IMG
    assert returned[1] == TAR_IMG


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_repr():
    """Sequence.__repr__ mentions n_images and n_pairs."""
    borders, segments, curves = square_roi(100.0)
    seq = Sequence(
        image_paths=[REF_IMG, TAR_IMG],
        borders=borders,
        segments=segments,
        curves=curves,
        size_lower=5.0,
        size_upper=50.0,
        target_nodes=10,
    )
    r = repr(seq)
    assert "2" in r   # n_images
    assert "1" in r   # n_pairs


# ===========================================================================
# Tests: sequence_deformation_preconditioning
# Mirrors Phase 1 deformation_preconditioning tests (Tier B tolerance)
# ===========================================================================


def test_deformation_preconditioning_pure_translation_displacement():
    """Pure translation u=0.5, v=0 → seed_displacement ≈ [0.5, 0.0].

    Tier B: rtol=1e-8.
    """
    u, v = 0.5, 0.0
    sol = make_mesh_solution(u, v)
    seed_coord = [0.333, 0.333]
    disp, _ = sequence_deformation_preconditioning(sol, seed_coord, 1, 1)
    np.testing.assert_allclose(disp[0], u, rtol=1e-8)
    np.testing.assert_allclose(disp[1], v, atol=1e-12)


def test_deformation_preconditioning_pure_translation_negative():
    """Pure translation u=0.3, v=-0.2 → seed_displacement ≈ [0.3, -0.2].

    Tier B: rtol=1e-8.
    """
    u, v = 0.3, -0.2
    sol = make_mesh_solution(u, v)
    disp, _ = sequence_deformation_preconditioning(sol, [0.333, 0.333], 1, 1)
    np.testing.assert_allclose(disp[0], u, rtol=1e-8)
    np.testing.assert_allclose(disp[1], v, rtol=1e-8)


def test_deformation_preconditioning_pure_translation_no_strain():
    """Pure translation → strain components of seed_warp ≈ 0.

    Tier B: atol=1e-8.
    """
    sol = make_mesh_solution(0.3, -0.2)
    _, seed_warp = sequence_deformation_preconditioning(sol, [0.333, 0.333], 1, 1)
    np.testing.assert_allclose(seed_warp[2:], 0.0, atol=1e-8)


def test_deformation_preconditioning_warp_length_order1():
    """seed_warp has length 6 for subset_order=1.

    Tier A (shape check).
    """
    sol = make_mesh_solution(0.0, 0.0)
    _, seed_warp = sequence_deformation_preconditioning(sol, [0.5, 0.2], 1, 1)
    assert len(seed_warp) == 6


def test_deformation_preconditioning_warp_length_order2():
    """seed_warp has length 12 for subset_order=2.

    Tier A (shape check).
    """
    sol = make_mesh_solution(0.0, 0.0)
    _, seed_warp = sequence_deformation_preconditioning(sol, [0.5, 0.2], 1, 2)
    assert len(seed_warp) == 12


def test_deformation_preconditioning_warp_matches_displacement():
    """seed_warp[:2] equals seed_displacement for pure translation.

    Tier B: rtol=1e-8.
    """
    u, v = 0.25, 0.1
    sol = make_mesh_solution(u, v)
    disp, seed_warp = sequence_deformation_preconditioning(sol, [0.333, 0.333], 1, 1)
    np.testing.assert_allclose(seed_warp[:2], disp, rtol=1e-8)


def test_deformation_preconditioning_zeros_gives_zero_disp():
    """Zero displacements everywhere → zero seed displacement.

    Tier A: atol=1e-12.
    """
    sol = make_mesh_solution(0.0, 0.0)
    disp, _ = sequence_deformation_preconditioning(sol, [0.5, 0.3], 1, 1)
    np.testing.assert_allclose(disp, 0.0, atol=1e-12)


def test_deformation_preconditioning_seed_warp_zeros_for_zero_disp():
    """Zero displacements → zero seed warp.

    Tier A: atol=1e-12.
    """
    sol = make_mesh_solution(0.0, 0.0)
    _, seed_warp = sequence_deformation_preconditioning(sol, [0.5, 0.3], 1, 1)
    np.testing.assert_allclose(seed_warp, 0.0, atol=1e-12)


def test_deformation_preconditioning_order1_mesh_order2_subset():
    """order-1 mesh, order-2 subset: first 6 warp terms match disp, last 6 ≈ 0.

    Tier B: rtol=1e-8.
    """
    u, v = 0.4, -0.1
    sol = make_mesh_solution(u, v)
    disp, seed_warp = sequence_deformation_preconditioning(sol, [0.333, 0.333], 1, 2)
    assert len(seed_warp) == 12
    np.testing.assert_allclose(seed_warp[0], disp[0], rtol=1e-8)
    np.testing.assert_allclose(seed_warp[1], disp[1], rtol=1e-8)
    np.testing.assert_allclose(seed_warp[6:], 0.0, atol=1e-12)


def test_deformation_preconditioning_displacement_shape():
    """seed_displacement is a 1-D array of length 2."""
    sol = make_mesh_solution(0.5, 0.0)
    disp, _ = sequence_deformation_preconditioning(sol, [0.333, 0.333], 1, 1)
    assert disp.shape == (2,)


def test_deformation_preconditioning_returns_numpy():
    """Both outputs are numpy arrays."""
    sol = make_mesh_solution(0.5, 0.0)
    disp, warp = sequence_deformation_preconditioning(sol, [0.333, 0.333], 1, 1)
    assert isinstance(disp, np.ndarray)
    assert isinstance(warp, np.ndarray)


# ===========================================================================
# Integration test: Sequence.solve with real DIC images (one pair)
# ===========================================================================


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_solve_one_pair_returns_solution():
    """Sequence with one image pair returns a valid SequenceSolution.

    Uses a small ROI (200×200 px) centred at (500, 500) with a 10-pixel
    template so the solve completes quickly.
    """
    # 200×200 px ROI centred at (500, 500) — well within the 1001×1001 images.
    cx, cy = 500.0, 500.0
    half = 100.0
    borders = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    curves = [[0, 1, 2, 3]]

    seq = Sequence(
        image_paths=[REF_IMG, TAR_IMG],
        borders=borders,
        segments=segments,
        curves=curves,
        size_lower=10.0,
        size_upper=100.0,
        target_nodes=15,
        mesh_order=1,
    )

    # Template: circle of radius 10 pixels.
    template = Circle(10)
    template_coords = template.coords

    sol = seq.solve(
        template_coords=template_coords,
        seed_coord=[cx, cy],
        seed_warp=[0.0] * 6,
        max_norm=1e-3,
        max_iterations=20,
        subset_order=1,
        tolerance=0.0,    # accept any result for smoke test
        guide=False,      # no preconditioning for a 1-pair sequence
        sync=False,
        border=20,
    )

    assert isinstance(sol, SequenceSolution)


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_solve_one_pair_result_shape():
    """SequenceSolution.mesh_solutions has length 1 for a 2-image sequence."""
    cx, cy = 500.0, 500.0
    half = 100.0
    borders = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    curves = [[0, 1, 2, 3]]

    seq = Sequence(
        image_paths=[REF_IMG, TAR_IMG],
        borders=borders,
        segments=segments,
        curves=curves,
        size_lower=10.0,
        size_upper=100.0,
        target_nodes=15,
    )

    template = Circle(10)
    sol = seq.solve(
        template_coords=template.coords,
        seed_coord=[cx, cy],
        seed_warp=[0.0] * 6,
        max_norm=1e-3,
        max_iterations=20,
        tolerance=0.0,
        guide=False,
        sync=False,
        border=20,
    )

    assert len(sol.mesh_solutions) == 1
    assert isinstance(sol.mesh_solutions[0], MeshSolution)


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_solution_repr():
    """SequenceSolution.__repr__ includes solved/unsolvable info."""
    cx, cy = 500.0, 500.0
    half = 100.0
    borders = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    curves = [[0, 1, 2, 3]]

    seq = Sequence(
        image_paths=[REF_IMG, TAR_IMG],
        borders=borders,
        segments=segments,
        curves=curves,
        size_lower=10.0,
        size_upper=100.0,
        target_nodes=15,
    )

    template = Circle(10)
    sol = seq.solve(
        template_coords=template.coords,
        seed_coord=[cx, cy],
        seed_warp=[0.0] * 6,
        max_norm=1e-3,
        max_iterations=20,
        tolerance=0.0,
        guide=False,
        sync=False,
        border=20,
    )

    r = repr(sol)
    assert "SequenceSolution" in r
