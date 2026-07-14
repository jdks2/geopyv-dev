import matplotlib
matplotlib.use("Agg")

import os
import cv2
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from geopyv_dev import (
    Image, Subset, Mesh, Mask, Sequence, SequenceOptions,
    Field,
    Particle,
    field_distribute_particles,
)
from geopyv_dev.plots import (
    inspect_subset,
    inspect_mesh,
    inspect_particle,
    inspect_field,
    convergence_subset,
    convergence_mesh,
    convergence_sequence,
    contour_mesh,
    contour_sequence,
    contour_field,
    history_particle,
    trace_particle,
    history_field,
    trace_field,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
REF_JPG = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "geopyv", "tests", "ref.jpg"))
TAR_JPG = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "geopyv", "tests", "tar.jpg"))
IMAGE_DIR = os.path.dirname(REF_JPG)
IMAGES_AVAILABLE = os.path.isfile(REF_JPG) and os.path.isfile(TAR_JPG)

pytestmark = pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
COORD = [200.43, 200.76]
RADIUS = 25
BORDER = 20


def _preprocess(fp):
    img = cv2.imread(fp, cv2.IMREAD_COLOR)
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gs, (5, 5), sigmaX=1.1, sigmaY=1.1)


def make_circle_template(radius):
    return Mask(mask_type="local", shape="circle", size=radius)




# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ref_img():
    gs = _preprocess(REF_JPG).astype(np.float64)
    return Image(image_gs=gs, border=BORDER)


@pytest.fixture(scope="module")
def tar_img():
    gs = _preprocess(TAR_JPG).astype(np.float64)
    return Image(image_gs=gs, border=BORDER)


@pytest.fixture(scope="module")
def solved_subset(ref_img, tar_img):
    tmpl = make_circle_template(RADIUS)
    s = Subset(COORD, tmpl, ref_img, tar_img)
    s.solve()
    return s


@pytest.fixture(scope="module")
def solved_mesh(ref_img, tar_img):
    tmpl = make_circle_template(20)
    boundary = np.array(
        [[150.0, 150.0], [250.0, 150.0], [250.0, 250.0], [150.0, 250.0]],
        dtype=np.float64,
    )
    mesh = Mesh(boundary, target_nodes=15, f_img=ref_img, g_img=tar_img,
                size=(20.0, 40.0))
    mesh.solve(tmpl, seed_coord=[200.0, 200.0], max_iterations=20)
    return mesh


@pytest.fixture(scope="module")
def solved_particle(solved_mesh):
    coords, vols = field_distribute_particles(solved_mesh.nodes, solved_mesh.elements)
    p = Particle(source=solved_mesh, coordinate=[float(coords[0, 0]), float(coords[0, 1])])
    p.solve()
    return p


@pytest.fixture(scope="module")
def solved_field(solved_mesh):
    # Field requires a SequenceSolution source — cannot be constructed from a
    # single MeshSolution without multiple image pairs.  Fixture is a no-op.
    pytest.skip("Field fixture requires SequenceSolution (multiple image pairs)")


# ---------------------------------------------------------------------------
# Unsolved-object fixtures — for guard tests below (plotting on data that has
# never had solve() called must raise, not silently plot garbage).
# ---------------------------------------------------------------------------

@pytest.fixture
def unsolved_subset(ref_img, tar_img):
    tmpl = make_circle_template(RADIUS)
    return Subset(COORD, tmpl, ref_img, tar_img)


@pytest.fixture
def unsolved_mesh(ref_img, tar_img):
    tmpl = make_circle_template(20)
    boundary = np.array(
        [[150.0, 150.0], [250.0, 150.0], [250.0, 250.0], [150.0, 250.0]],
        dtype=np.float64,
    )
    return Mesh(boundary, target_nodes=15, f_img=ref_img, g_img=tar_img,
                size=(20.0, 40.0))


@pytest.fixture
def unsolved_sequence():
    boundary = np.array(
        [[150.0, 150.0], [250.0, 150.0], [250.0, 250.0], [150.0, 250.0]],
        dtype=np.float64,
    )
    return Sequence(image_dir=IMAGE_DIR, boundary=boundary, target_nodes=15,
                     size=(20.0, 40.0))


@pytest.fixture
def unsolved_particle(solved_mesh):
    coords, vols = field_distribute_particles(solved_mesh.nodes, solved_mesh.elements)
    return Particle(source=solved_mesh, coordinate=[float(coords[0, 0]), float(coords[0, 1])])


@pytest.fixture
def unsolved_field(solved_mesh):
    return Field(sequence_solution=solved_mesh)


# ===========================================================================
# TestInspectSubset
# ===========================================================================

class TestInspectSubset:
    def test_returns_fig_ax(self, solved_subset):
        fig, ax = inspect_subset(solved_subset, show=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_single_axes(self, solved_subset):
        fig, ax = inspect_subset(solved_subset, show=False)
        assert len(fig.axes) == 1

    def test_imshow_present(self, solved_subset):
        fig, ax = inspect_subset(solved_subset, show=False)
        assert len(ax.images) == 1

    def test_annotation_present(self, solved_subset):
        fig, ax = inspect_subset(solved_subset, show=False)
        texts = [c.get_text() for c in ax.get_children() if hasattr(c, "get_text")]
        assert any("SSSIG" in t for t in texts)

    def test_save_writes_file(self, solved_subset, tmp_path):
        out = str(tmp_path / "subset.png")
        inspect_subset(solved_subset, show=False, save=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_show_false_no_error(self, solved_subset):
        inspect_subset(solved_subset, show=False)

    def test_kwargs_forwarded(self, solved_subset):
        fig, ax = inspect_subset(solved_subset, show=False, cmap="plasma")
        assert ax.images[0].cmap.name == "plasma"


# ===========================================================================
# TestInspectMesh
# ===========================================================================

class TestInspectMesh:
    def test_returns_fig_ax(self, solved_mesh):
        fig, ax = inspect_mesh(solved_mesh, show=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_single_axes(self, solved_mesh):
        fig, ax = inspect_mesh(solved_mesh, show=False)
        assert len(fig.axes) == 1

    def test_imshow_present(self, solved_mesh):
        fig, ax = inspect_mesh(solved_mesh, show=False)
        assert len(ax.images) == 1

    def test_edges_plotted(self, solved_mesh):
        fig, ax = inspect_mesh(solved_mesh, show=False)
        assert len(ax.lines) > 0

    def test_element_annotations(self, solved_mesh):
        fig, ax = inspect_mesh(solved_mesh, show=False)
        texts = [c.get_text() for c in ax.get_children() if hasattr(c, "get_text")]
        assert any(t.isdigit() for t in texts)

    def test_show_areas_adds_patches(self, solved_mesh):
        fig, ax = inspect_mesh(solved_mesh, show=False, show_areas=True)
        assert len(ax.patches) > 0

    def test_save_writes_file(self, solved_mesh, tmp_path):
        out = str(tmp_path / "mesh.png")
        inspect_mesh(solved_mesh, show=False, save=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0


# ===========================================================================
# TestInspectParticle
# ===========================================================================

class TestInspectParticle:
    def test_returns_fig_ax(self, solved_particle):
        fig, ax = inspect_particle(solved_particle, show=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_single_axes(self, solved_particle):
        fig, ax = inspect_particle(solved_particle, show=False)
        assert len(fig.axes) == 1

    def test_imshow_present(self, solved_particle):
        fig, ax = inspect_particle(solved_particle, show=False)
        assert len(ax.images) == 1

    def test_scatter_present(self, solved_particle):
        fig, ax = inspect_particle(solved_particle, show=False)
        assert len(ax.collections) > 0

    def test_save_writes_file(self, solved_particle, tmp_path):
        out = str(tmp_path / "particle.png")
        inspect_particle(solved_particle, show=False, save=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0


# ===========================================================================
# TestInspectField
# ===========================================================================

class TestInspectField:
    def test_returns_fig_ax(self, solved_field):
        fig, ax = inspect_field(solved_field, show=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_all_particles_plotted(self, solved_field):
        fig, ax = inspect_field(solved_field, show=False)
        n_particles = len(solved_field.particles)
        offsets = ax.collections[0].get_offsets()
        assert offsets.shape[0] == n_particles

    def test_particle_idx_highlights(self, solved_field):
        fig, ax = inspect_field(solved_field, particle_idx=0, show=False)
        assert len(ax.collections) >= 2


# ===========================================================================
# TestConvergenceSubset
# ===========================================================================

class TestConvergenceSubset:
    def test_returns_fig_ax_array(self, solved_subset):
        fig, ax = convergence_subset(solved_subset, show=False)
        assert isinstance(ax, np.ndarray)
        assert len(ax) == 2

    def test_two_panels(self, solved_subset):
        fig, ax = convergence_subset(solved_subset, show=False)
        assert len(fig.axes) == 2

    def test_threshold_lines(self, solved_subset):
        fig, ax = convergence_subset(solved_subset, show=False)
        assert len(ax[0].lines) >= 2
        assert len(ax[1].lines) >= 2

    def test_save_writes_file(self, solved_subset, tmp_path):
        out = str(tmp_path / "conv_subset.png")
        convergence_subset(solved_subset, show=False, save=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0


# ===========================================================================
# TestConvergenceMesh
# ===========================================================================

class TestConvergenceMesh:
    def test_returns_fig_ax(self, solved_mesh):
        fig, ax = convergence_mesh(solved_mesh, "C_ZNCC", show=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_single_axes(self, solved_mesh):
        fig, ax = convergence_mesh(solved_mesh, "C_ZNCC", show=False)
        assert len(fig.axes) == 1

    def test_hist_present(self, solved_mesh):
        fig, ax = convergence_mesh(solved_mesh, "C_ZNCC", show=False)
        assert len(ax.patches) > 0

    def test_valid_quantities(self, solved_mesh):
        for q in ("C_ZNCC", "iterations", "norm"):
            convergence_mesh(solved_mesh, q, show=False)

    def test_invalid_quantity_raises(self, solved_mesh):
        with pytest.raises((ValueError, Exception)):
            convergence_mesh(solved_mesh, "bad_quantity", show=False)


# ===========================================================================
# TestContourMesh
# ===========================================================================

class TestContourMesh:
    def test_returns_fig_ax(self, solved_mesh):
        fig, ax = contour_mesh(solved_mesh, "C_ZNCC", show=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_colorbar_present(self, solved_mesh):
        fig, ax = contour_mesh(solved_mesh, "C_ZNCC", show=False)
        assert len(fig.axes) == 2

    def test_colorbar_label(self, solved_mesh):
        fig, ax = contour_mesh(solved_mesh, "C_ZNCC", show=False)
        label = fig.axes[-1].get_ylabel()
        assert len(label) > 0

    def test_image_present(self, solved_mesh):
        fig, ax = contour_mesh(solved_mesh, "C_ZNCC", show=False)
        assert len(ax.images) == 1

    def test_tricontourf_present(self, solved_mesh):
        fig, ax = contour_mesh(solved_mesh, "C_ZNCC", show=False)
        assert len(ax.collections) > 0

    def test_all_valid_quantities(self, solved_mesh):
        for q in ("C_ZNCC", "iterations", "norm", "u", "v", "R"):
            contour_mesh(solved_mesh, q, show=False)

    def test_invalid_quantity_raises(self, solved_mesh):
        with pytest.raises((ValueError, Exception)):
            contour_mesh(solved_mesh, "bad_quantity", show=False)

    def test_kwargs_forwarded(self, solved_mesh):
        fig, ax = contour_mesh(solved_mesh, "C_ZNCC", show=False, alpha=0.3)
        assert len(ax.collections) > 0

    def test_save_writes_file(self, solved_mesh, tmp_path):
        out = str(tmp_path / "contour_mesh.png")
        contour_mesh(solved_mesh, "C_ZNCC", show=False, save=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0


# ===========================================================================
# TestContourField
# ===========================================================================

class TestContourField:
    def test_returns_fig_ax(self, solved_field):
        fig, ax = contour_field(solved_field, "u", show=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_colorbar_present(self, solved_field):
        fig, ax = contour_field(solved_field, "u", show=False)
        assert len(fig.axes) == 2

    def test_colorbar_label_accumulated(self, solved_field):
        fig, ax = contour_field(solved_field, "u", dt=None, show=False)
        label = fig.axes[-1].get_ylabel()
        assert "/s" not in label

    def test_colorbar_label_rate(self, solved_field):
        fig, ax = contour_field(solved_field, "u", dt=0.1, show=False)
        label = fig.axes[-1].get_ylabel()
        assert "/s" in label

    def test_absolute_flag_no_error(self, solved_field):
        contour_field(solved_field, "u", absolute=True, show=False)

    def test_window_subset_no_error(self, solved_field):
        contour_field(solved_field, "u", window=[0, 1], show=False)

    def test_all_valid_quantities(self, solved_field):
        for q in ("u", "v", "R", "ep_xx", "ep_yy", "ep_xy", "ep_vol"):
            contour_field(solved_field, q, show=False)

    def test_save_writes_file(self, solved_field, tmp_path):
        out = str(tmp_path / "contour_field.png")
        contour_field(solved_field, "u", show=False, save=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0


# ===========================================================================
# TestUnsolvedGuards
#
# Plotting solve-dependent data on an object that has never had solve()
# called must raise, not silently plot near-empty/garbage data. In
# particular, history_particle/trace_particle's default quantity="warps"
# used to read unconditionally-available (zero-filled) fields with no
# guard at all.
# ===========================================================================

class TestUnsolvedGuards:
    def test_convergence_subset_raises(self, unsolved_subset):
        with pytest.raises(RuntimeError, match="has not been solved"):
            convergence_subset(unsolved_subset, show=False)

    def test_convergence_mesh_raises(self, unsolved_mesh):
        with pytest.raises(RuntimeError, match="has not been solved"):
            convergence_mesh(unsolved_mesh, "C_ZNCC", show=False)

    def test_contour_mesh_raises(self, unsolved_mesh):
        with pytest.raises(RuntimeError, match="has not been solved"):
            contour_mesh(unsolved_mesh, "C_ZNCC", show=False)

    def test_convergence_sequence_raises(self, unsolved_sequence):
        with pytest.raises(RuntimeError, match="has not been solved"):
            convergence_sequence(unsolved_sequence, show=False)

    def test_contour_sequence_raises(self, unsolved_sequence):
        with pytest.raises(RuntimeError, match="has not been solved"):
            contour_sequence(unsolved_sequence, mesh_idx=0, quantity="C_ZNCC")

    def test_history_particle_default_quantity_raises(self, unsolved_particle):
        # Regression test: this used to silently plot near-zero data instead
        # of raising, because the default quantity="warps" read an
        # unconditionally-available (zero-filled) field.
        with pytest.raises(RuntimeError, match="has not been solved"):
            history_particle(unsolved_particle, show=False)

    def test_history_particle_strains_quantity_raises(self, unsolved_particle):
        with pytest.raises(RuntimeError, match="has not been solved"):
            history_particle(unsolved_particle, quantity="strains", show=False)

    def test_trace_particle_default_quantity_raises(self, unsolved_particle):
        with pytest.raises(RuntimeError, match="has not been solved"):
            trace_particle(unsolved_particle, show=False)

    def test_history_field_raises(self, unsolved_field):
        with pytest.raises(RuntimeError, match="has not been solved"):
            history_field(unsolved_field, particle_index=0, show=False)

    def test_trace_field_raises(self, unsolved_field):
        with pytest.raises(RuntimeError, match="has not been solved"):
            trace_field(unsolved_field, show=False)

    def test_contour_field_raises(self, unsolved_field):
        with pytest.raises(RuntimeError, match="has not been solved"):
            contour_field(unsolved_field, "u", show=False)
