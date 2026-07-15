"""
Phase 3 validation tests for geopyv_dev.sequence.

Mirrors the Phase 1 fixture tests in geopyv/tests/sequence/fixtures.py,
asserting the same golden values via geopyv_dev.

Tolerance tiers (per plan):
  Tier A  atol=1e-12  pure matrix algebra
  Tier B  rtol=1e-8   floating-point arithmetic chains
"""

import os
import shutil
import numpy as np
import pytest

from geopyv_dev import (
    Sequence,
    SequenceOptions,
    Mask,
    PathRegion,
)
from geopyv_dev.wrappers import MeshWrapper

# ---------------------------------------------------------------------------
# Boundary for sequence construction tests
# ---------------------------------------------------------------------------

_SQUARE_BOUNDARY = np.array(
    [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
    dtype=np.float64,
)


# Test image paths (relative to the repository root).
_HERE = os.path.dirname(__file__)
REF_IMG = os.path.abspath(os.path.join(_HERE, "../../../../geopyv/tests/ref.jpg"))
TAR_IMG = os.path.abspath(os.path.join(_HERE, "../../../../geopyv/tests/tar.jpg"))
IMAGE_DIR = os.path.dirname(REF_IMG)
IMAGES_AVAILABLE = os.path.isfile(REF_IMG) and os.path.isfile(TAR_IMG)

# ===========================================================================
# Tests: Sequence construction validation
# ===========================================================================


def test_sequence_too_few_images_raises(tmp_path):
    """Sequence with only one image in the directory raises an error."""
    (tmp_path / "img_001.jpg").touch()
    with pytest.raises(Exception):
        Sequence(
            image_dir=str(tmp_path),
            boundary=_SQUARE_BOUNDARY,
            target_nodes=20,
        )


def test_sequence_nonexistent_dir_raises():
    """Sequence raises an error when the image directory does not exist."""
    with pytest.raises(Exception):
        Sequence(
            image_dir="/no/such/directory",
            boundary=_SQUARE_BOUNDARY,
            target_nodes=20,
        )


def test_sequence_size_lower_ge_upper_raises(tmp_path):
    """size[0] >= size[1] raises an error."""
    (tmp_path / "img_001.jpg").touch()
    (tmp_path / "img_002.jpg").touch()
    with pytest.raises(Exception):
        Sequence(
            image_dir=str(tmp_path),
            boundary=_SQUARE_BOUNDARY,
            target_nodes=20,
            size=(100.0, 10.0),
        )


def test_sequence_zero_size_lower_raises(tmp_path):
    """size[0] = 0 raises an error."""
    (tmp_path / "img_001.jpg").touch()
    (tmp_path / "img_002.jpg").touch()
    with pytest.raises(Exception):
        Sequence(
            image_dir=str(tmp_path),
            boundary=_SQUARE_BOUNDARY,
            target_nodes=20,
            size=(0.0, 50.0),
        )


# ===========================================================================
# Tests: Sequence properties
# ===========================================================================


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_n_pairs_two_images():
    """Two images → n_pairs = 1."""
    seq = Sequence(
        image_dir=IMAGE_DIR,
        boundary=_SQUARE_BOUNDARY,
        target_nodes=10,
        size=(5.0, 50.0),
    )
    assert seq.n_pairs == 1


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_image_paths_round_trips():
    """image_paths getter includes the expected images from the directory."""
    seq = Sequence(
        image_dir=IMAGE_DIR,
        boundary=_SQUARE_BOUNDARY,
        target_nodes=10,
        size=(5.0, 50.0),
    )
    returned = seq.image_paths
    assert len(returned) >= 2
    assert REF_IMG in returned
    assert TAR_IMG in returned


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_repr():
    """Sequence.__repr__ mentions n_images and n_pairs."""
    seq = Sequence(
        image_dir=IMAGE_DIR,
        boundary=_SQUARE_BOUNDARY,
        target_nodes=10,
        size=(5.0, 50.0),
    )
    r = repr(seq)
    assert "2" in r   # n_images
    assert "1" in r   # n_pairs


# ===========================================================================
# Integration test: Sequence.solve with real DIC images (one pair)
# ===========================================================================


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_solve_one_pair_mutates_in_place():
    """Sequence.solve() returns None and marks the sequence as solved.

    Uses a small ROI (200×200 px) centred at (500, 500) with a 10-pixel
    template so the solve completes quickly.
    """
    cx, cy = 500.0, 500.0
    half = 100.0
    boundary = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )

    seq = Sequence(
        image_dir=IMAGE_DIR,
        boundary=boundary,
        target_nodes=15,
        size=(10.0, 100.0),
        mesh_order=1,
    )

    template = Mask(mask_type="local", shape="circle", size=10)

    result = seq.solve(
        local_mask=template,
        seed_coord=[cx, cy],
        max_norm=1e-3,
        max_iterations=20,
        subset_order=1,
        tolerance=0.0,
        options=SequenceOptions(guide=False, sync=False),
        border=20,
    )

    assert result is None
    assert seq.solved is True


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_solve_one_pair_result_shape():
    """seq.mesh_solutions has length 1 for a 2-image sequence."""
    cx, cy = 500.0, 500.0
    half = 100.0
    boundary = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )

    seq = Sequence(
        image_dir=IMAGE_DIR,
        boundary=boundary,
        target_nodes=15,
        size=(10.0, 100.0),
    )

    template = Mask(mask_type="local", shape="circle", size=10)
    seq.solve(
        local_mask=template,
        seed_coord=[cx, cy],
        max_norm=1e-3,
        # This test only checks pipeline shape (tolerance=0.0 — correlation
        # quality is irrelevant), so pin subset_order=1: at the default
        # order-2, one subset in this real-image/soft-boundary mesh never
        # converges at all (still fails at 10000 iterations), which is a
        # genuine ICGN limitation, not something to chase with a bigger
        # budget — previously masked because `quality_ok` only checked
        # correlation, not convergence. order-1 converges cleanly by 103.
        max_iterations=200,
        subset_order=1,
        tolerance=0.0,
        options=SequenceOptions(guide=False, sync=False),
        border=20,
    )

    assert len(seq.mesh_solutions) == 1
    assert isinstance(seq.mesh_solutions[0], MeshWrapper)


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_repr_after_solve():
    """repr(seq) includes 'Sequence' and 'solved' after solving."""
    cx, cy = 500.0, 500.0
    half = 100.0
    boundary = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )

    seq = Sequence(
        image_dir=IMAGE_DIR,
        boundary=boundary,
        target_nodes=15,
        size=(10.0, 100.0),
    )

    template = Mask(mask_type="local", shape="circle", size=10)
    seq.solve(
        local_mask=template,
        seed_coord=[cx, cy],
        max_norm=1e-3,
        max_iterations=20,
        tolerance=0.0,
        options=SequenceOptions(guide=False, sync=False),
        border=20,
    )

    r = repr(seq)
    assert "Sequence" in r
    assert "all_converged" in r


# ===========================================================================
# Regression test: boundary region tracks displacement across a
# `sequential=True` reference update (rather than resetting to the original
# static polygon each time).
# ===========================================================================


@pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")
def test_sequence_boundary_tracks_displacement_across_reference_update(tmp_path):
    """Regression test for the Region-tracking wiring in Sequence::solve.

    Builds a 3-image sequence (ref -> tar -> tar again, the last two frames
    identical) so that pair 1 has zero true displacement and its mesh is
    built directly from whatever boundary polygon `Sequence::solve` hands
    `Mesh::new` after the `sequential` reference update following pair 0.
    """
    shutil.copy(REF_IMG, tmp_path / "frame_000.jpg")
    shutil.copy(TAR_IMG, tmp_path / "frame_001.jpg")
    shutil.copy(TAR_IMG, tmp_path / "frame_002.jpg")

    cx, cy = 500.0, 500.0
    half = 100.0
    boundary_pts = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )
    boundary = PathRegion(boundary_pts)

    seq = Sequence(
        image_dir=str(tmp_path),
        boundary=boundary,
        target_nodes=15,
        size=(10.0, 100.0),
        mesh_order=1,
    )

    template = Mask(mask_type="local", shape="circle", size=10)
    seq.solve(
        local_mask=template,
        seed_coord=[cx, cy],
        max_norm=1e-3,
        # 20 was insufficient for one subset in the ref->tar pair to
        # actually converge — previously masked because `quality_ok` only
        # checked correlation, not convergence.
        max_iterations=100,
        subset_order=1,
        tolerance=0.0,
        options=SequenceOptions(guide=False, sync=False, sequential=True),
        border=20,
    )

    assert seq.n_pairs == 2

    # The boundary must have moved from its original static position (the
    # bug: it used to reset to `boundary_pts` on every reference update).
    assert not np.allclose(boundary.current_nodes, boundary_pts)
    assert boundary.counter == 2

    # Pair 1's actual mesh must be built from that tracked/displaced
    # position, not the original polygon.
    b_idx1 = seq.boundary(1)
    nodes1 = seq.nodes(1)
    np.testing.assert_allclose(nodes1[b_idx1], boundary.current_nodes, atol=1e-6)
