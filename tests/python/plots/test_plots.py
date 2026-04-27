import matplotlib
matplotlib.use("Agg")

import os
import cv2
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from geopyv_dev import (
    Image, Subset, Mesh, Template,
    Field, FieldSolution,
    Particle, ParticleSolution,
    field_distribute_particles,
)
from geopyv_dev.plots import (
    inspect_subset,
    inspect_mesh,
    inspect_particle,
    inspect_field,
    convergence_subset,
    convergence_mesh,
    contour_mesh,
    contour_field,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
REF_JPG = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "geopyv", "tests", "ref.jpg"))
TAR_JPG = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "geopyv", "tests", "tar.jpg"))
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
    return Template("circle", size=radius)


def square_roi(x0, y0, x1, y1):
    borders = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    curves = [[0, 1, 2, 3]]
    return borders, segments, curves


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
    s = Subset(COORD, tmpl, ref_img)
    s.solve_icgn(tar_img, [0.0] * 6)
    return s


@pytest.fixture(scope="module")
def solved_mesh(ref_img, tar_img):
    tmpl = make_circle_template(20)
    borders, segments, curves = square_roi(150.0, 150.0, 250.0, 250.0)
    mesh = Mesh(borders, segments, curves, size_lower=20.0, size_upper=40.0, target_nodes=15)
    sol = mesh.solve(ref_img, tar_img, tmpl, [200.0, 200.0], [0.0] * 6,
                     max_norm=1e-3, max_iterations=20)
    return sol


@pytest.fixture(scope="module")
def solved_particle(solved_mesh):
    coords, vols = field_distribute_particles(solved_mesh.nodes, solved_mesh.elements)
    p = Particle(
        coordinate=[float(coords[0, 0]), float(coords[0, 1])],
        initial_warp=[0.0] * 6,
        initial_volume=float(vols[0]),
        inc_no=2,
    )
    sol = p.solve(
        [solved_mesh.nodes],
        [solved_mesh.elements],
        [solved_mesh.displacements],
        [int(solved_mesh.mesh_order)],
    )
    return sol


@pytest.fixture(scope="module")
def solved_field(solved_mesh):
    coords, vols = field_distribute_particles(solved_mesh.nodes, solved_mesh.elements)
    f = Field(coordinates=coords, volumes=vols, inc_no=2)
    sol = f.solve(
        [solved_mesh.nodes],
        [solved_mesh.elements],
        [solved_mesh.displacements],
        [int(solved_mesh.mesh_order)],
    )
    return sol


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
