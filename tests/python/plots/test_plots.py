import matplotlib
matplotlib.use("Agg")

import os
import warnings
import cv2
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from geopyv_dev import (
    Image, Subset, Mesh, Mask, Sequence, SequenceOptions,
    Field,
    Particle,
    MeshlessParams,
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
        # Element edges are one LineCollection (not per-edge ax.plot lines).
        fig, ax = inspect_mesh(solved_mesh, show=False)
        from matplotlib.collections import LineCollection
        assert any(isinstance(c, LineCollection) for c in ax.collections)

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

    def test_subset_idx_renders_template_crop(self, solved_mesh):
        # With template metadata (schema 0x07+), inspect_mesh(subset_idx=)
        # draws a crop of that node's subset template, not the whole mesh.
        fig, ax = inspect_mesh(solved_mesh, subset_idx=2, show=False)
        assert len(ax.images) == 1
        # crop is far smaller than the full mesh bbox
        x0, x1 = ax.get_xlim()
        assert abs(x1 - x0) <= 2 * 20 + 2
        texts = [c.get_text() for c in ax.get_children() if hasattr(c, "get_text")]
        assert any("Subset 2" in t for t in texts)


# ===========================================================================
# ===========================================================================
# Template metadata (subset_idx crops)
# ===========================================================================

class TestMeshTemplateMetadata:
    def test_template_recorded_after_solve(self, solved_mesh):
        assert solved_mesh.template_shape == "circle"
        assert np.all(np.asarray(solved_mesh.template_sizes) == 20)

    def test_no_zonal_record_any_more(self, solved_mesh):
        assert not hasattr(solved_mesh._inner, "zonal_masking")


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
        fig, ax = contour_field(solved_field, "u", show=False)
        label = fig.axes[-1].get_ylabel()
        assert "/s" not in label

    def test_colorbar_label_rate(self, solved_field):
        fig, ax = contour_field(solved_field, "u", rate=True, show=False)
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
# TestContourFieldGammaMaxGrad
#
# gamma_max_grad (and ep1/ep2/gamma_max/theta_p) are now pure lookups of
# Particle.principal_strains / Particle.gamma_max_grad, computed once at
# Field.solve() time, not recomputed at plot time -- see
# ParticleSolution::gamma_max_grad's doc comment (src/particle.rs) for why.
# Uses its own Field fixtures rather than the shared `solved_field` (which
# is a stale skip -- see that fixture's own comment; unrelated to this
# work, not fixed here).
# ===========================================================================

@pytest.fixture(scope="module")
def solved_field_meshless(solved_mesh):
    field = Field(sequence_solution=solved_mesh)
    field.solve()  # strain_method=None -> the subset average, the default (gradient pass as meshless)
    return field


@pytest.fixture(scope="module")
def solved_field_mesh_forced(solved_mesh):
    field = Field(sequence_solution=solved_mesh)
    field.solve(strain_method=False)  # explicit opt-out -> StrainMethod::Mesh
    return field


class TestContourFieldGammaMaxGrad:
    def test_runs_without_raising_when_meshless(self, solved_field_meshless):
        fig, ax = contour_field(solved_field_meshless, "gamma_max_grad", show=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_raises_clearly_when_strain_method_is_mesh(self, solved_field_mesh_forced):
        with pytest.raises(ValueError, match="gamma_max_grad"):
            contour_field(solved_field_mesh_forced, "gamma_max_grad", show=False)

    def test_principal_strain_quantities_are_pure_lookups(self, solved_field_meshless):
        # Used to call _core.particle_principal_strains on demand; confirm
        # they still work now that they read Particle.principal_strains.
        for q in ("ep1", "ep2", "gamma_max", "theta_p"):
            contour_field(solved_field_meshless, q, show=False)


# ===========================================================================
# TestContourFieldCore
#
# contour_field's values, positions and triangles now come from the core
# (Field.contour_values / contour_coordinates / contour_triangles), shared
# with the GUI. _legacy_contour_values is the numpy reduction contour_field
# used before that move, kept here verbatim as the parity reference.
# ===========================================================================

def _legacy_reduce_series(v, absolute):
    v = np.atleast_1d(v)
    if absolute:
        return float(np.sum(np.abs(np.diff(v)))) if len(v) > 1 else 0.0
    return float(v[-1] - v[0]) if len(v) > 1 else float(v[-1])


def _legacy_contour_values(field, quantity, window, absolute):
    strain_col = {"ep_xx": 0, "ep_yy": 1, "ep_xy": 5}
    principal_col = {"ep1": 0, "ep2": 1, "gamma_max": 2, "theta_p": 3}
    if window is not None:
        if isinstance(window, (list, tuple)) and len(window) == 2:
            w = slice(window[0], window[1])
        else:
            w = window
    else:
        w = slice(None)
    values = []
    for p in field.particles:
        if quantity == "gamma_max_grad":
            grad = np.asarray(p.gamma_max_grad)
            gx = _legacy_reduce_series(grad[w, 0], absolute)
            gy = _legacy_reduce_series(grad[w, 1], absolute)
            values.append(float(np.hypot(gx, gy)))
            continue
        warps = np.asarray(p.warps)
        strains = np.asarray(p.strains)
        if quantity == "u":
            v = warps[w, 0]
        elif quantity == "v":
            v = warps[w, 1]
        elif quantity == "R":
            v = np.sqrt(warps[w, 0]**2 + warps[w, 1]**2)
        elif quantity in strain_col:
            v = strains[w, strain_col[quantity]]
        elif quantity == "ep_vol":
            v = np.asarray(p.vol_strains)[w]
        else:
            v = np.asarray(p.principal_strains)[w, principal_col[quantity]]
        values.append(_legacy_reduce_series(v, absolute))
    return np.array(values)


_ALL_FIELD_QUANTITIES = ("u", "v", "R", "ep_xx", "ep_yy", "ep_xy", "ep_vol",
                         "ep1", "ep2", "gamma_max", "theta_p", "gamma_max_grad")


class TestContourFieldCore:
    @pytest.mark.parametrize("window", [None, [0, 1], [1, 2], [0, 2], [None, None], -1, 0, slice(0, 2)])
    @pytest.mark.parametrize("absolute", [False, True])
    def test_core_values_match_legacy_numpy(self, solved_field_meshless, window, absolute):
        from geopyv_dev.plots import _resolve_window
        field = solved_field_meshless
        w = _resolve_window(window, field.inc_no)
        for q in _ALL_FIELD_QUANTITIES:
            core = np.asarray(field.contour_values(q, window=w, absolute=absolute))
            legacy = _legacy_contour_values(field, q, window, absolute)
            np.testing.assert_allclose(core, legacy, rtol=1e-12, atol=1e-15, err_msg=q)

    def test_rate_without_time_raises(self, solved_field_meshless):
        assert solved_field_meshless.time is None
        with pytest.raises(ValueError, match="time"):
            contour_field(solved_field_meshless, "u", rate=True, show=False)

    def test_region_stored_and_triangles_inside_it(self, solved_field_meshless):
        boundary, exclusions = solved_field_meshless.region
        assert np.asarray(boundary).shape[1] == 2
        assert exclusions == []
        tris = np.asarray(solved_field_meshless.contour_triangles())
        assert tris.ndim == 2 and tris.shape[1] == 3 and len(tris) > 0
        assert tris.max() < solved_field_meshless.n_particles

    def test_deformed_coordinates(self, solved_field_meshless):
        field = solved_field_meshless
        ref = np.asarray(field.contour_coordinates())
        np.testing.assert_array_equal(ref, np.asarray(field.coordinates))
        deformed = np.asarray(field.contour_coordinates(deformed=True))
        last = np.array([np.asarray(p.coordinates)[-1] for p in field.particles])
        np.testing.assert_array_equal(deformed, last)

    def test_deformed_contour_plots(self, solved_field_meshless):
        fig, ax = contour_field(solved_field_meshless, "u", deformed=True, show=False)
        assert len(ax.collections) > 0

    def test_no_region_warning_on_fresh_solve(self, solved_field_meshless):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            contour_field(solved_field_meshless, "u", show=False)

    def test_bad_window_step_raises(self, solved_field_meshless):
        with pytest.raises(ValueError, match="step"):
            contour_field(solved_field_meshless, "u", window=slice(0, 2, 2), show=False)

    def test_region_survives_save_load(self, solved_field_meshless, tmp_path):
        import geopyv_dev as gp
        path = str(tmp_path / "field.pyv")
        solved_field_meshless.save(path)
        loaded = gp.load(path)
        b0, _ = solved_field_meshless.region
        b1, _ = loaded.region
        np.testing.assert_array_equal(np.asarray(b0), np.asarray(b1))
        np.testing.assert_array_equal(np.asarray(loaded.contour_triangles()),
                                      np.asarray(solved_field_meshless.contour_triangles()))


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


# ===========================================================================
# Removed options: zonal masking and method="adaptive" now raise
# ===========================================================================

class TestRemovedZonalOptions:
    def test_method_adaptive_raises(self, solved_mesh):
        with pytest.raises(ValueError, match="adaptive"):
            solved_mesh.solve(make_circle_template(20), seed_coord=[200.0, 200.0],
                              method="adaptive")

    @pytest.mark.parametrize("kw", ["adaptive_k", "adaptive_meshless_params"])
    def test_adaptive_kwargs_raise(self, solved_mesh, kw):
        with pytest.raises(ValueError, match="adaptive"):
            solved_mesh.solve(make_circle_template(20), seed_coord=[200.0, 200.0], **{kw: 1.0})

    def test_masking_zonal_raises(self, solved_mesh):
        with pytest.raises(ValueError, match="zonal masking has been removed"):
            solved_mesh.solve(make_circle_template(20), seed_coord=[200.0, 200.0],
                              solver_options={"masking": "zonal"})

    def test_zonal_subdict_raises(self, solved_mesh):
        with pytest.raises(ValueError, match="zonal masking has been removed"):
            solved_mesh.solve(make_circle_template(20), seed_coord=[200.0, 200.0],
                              solver_options={"zonal": {"k": 3.0}})

    def test_binding_masking_zonal_raises(self, unsolved_mesh):
        with pytest.raises(ValueError, match="removed"):
            unsolved_mesh._inner.solve(make_circle_template(20), [200.0, 200.0], masking="zonal")

    @pytest.mark.parametrize("kw", ["zonal_k", "zone_mask"])
    def test_binding_zonal_kwargs_gone(self, unsolved_mesh, kw):
        with pytest.raises(TypeError, match=kw):
            unsolved_mesh._inner.solve(make_circle_template(20), [200.0, 200.0], **{kw: 1.0})

    def test_meshless_params_zone_aware_gone(self):
        with pytest.raises(TypeError, match="zone_aware"):
            MeshlessParams(radius=15.0, zone_aware=True)

    def test_masking_uniform_still_accepted(self, unsolved_mesh):
        unsolved_mesh.solve(make_circle_template(20), seed_coord=[200.0, 200.0],
                            max_iterations=20, solver_options={"masking": "uniform"})
        assert unsolved_mesh.solved



class TestSolverOptionsSurface:
    """New solver=/solver_options= surface -- solver_options_restructure.md."""

    def test_solver_icgn_no_warning(self, unsolved_mesh):
        tmpl = make_circle_template(20)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning -> test failure
            unsolved_mesh.solve(tmpl, seed_coord=[200.0, 200.0], max_iterations=20,
                                 solver="icgn")
        assert unsolved_mesh.solved

    def test_unknown_solver_warns_and_falls_back_to_icgn(self, unsolved_mesh):
        tmpl = make_circle_template(20)
        with pytest.warns(UserWarning, match="Unknown solver 'fgan'"):
            unsolved_mesh.solve(tmpl, seed_coord=[200.0, 200.0], max_iterations=20,
                                 solver="fgan")
        assert unsolved_mesh.solved

    def test_method_kernel_sense_deprecated(self, unsolved_mesh):
        tmpl = make_circle_template(20)
        with pytest.warns(DeprecationWarning, match="method"):
            unsolved_mesh.solve(tmpl, seed_coord=[200.0, 200.0], max_iterations=20,
                                 method="icgn")
        assert unsolved_mesh.solved

    def test_topology_adaptive_not_implemented(self, solved_mesh):
        tmpl = make_circle_template(20)
        with pytest.raises(NotImplementedError, match="topology"):
            solved_mesh.solve(tmpl, seed_coord=[200.0, 200.0],
                               solver_options={"topology": "adaptive"})

    def test_preconditioning_layer_rg_works(self, unsolved_mesh):
        # layer-RG landed (src/mesh.rs::expand_parallel); it agrees with RG
        # within Tier C -- see the Rust tests in src/mesh.rs.
        tmpl = make_circle_template(20)
        unsolved_mesh.solve(tmpl, seed_coord=[200.0, 200.0], max_iterations=20,
                             solver_options={"preconditioning": "layer-RG",
                                             "layer_rg": {"batch_factor": 4}})
        assert unsolved_mesh.solved

    def test_preconditioning_rg_still_works(self, unsolved_mesh):
        tmpl = make_circle_template(20)
        unsolved_mesh.solve(tmpl, seed_coord=[200.0, 200.0], max_iterations=20,
                             solver_options={"preconditioning": "RG"})
        assert unsolved_mesh.solved

    def test_preconditioning_layer_rg_matches_rg(self, ref_img, tar_img):
        # layer_rg_plan.md §6.8: layer-RG agrees with RG within Tier C.
        tmpl = make_circle_template(20)
        boundary = np.array([[50.0, 50.0], [350.0, 50.0], [350.0, 350.0], [50.0, 350.0]])
        m_rg = Mesh(boundary, target_nodes=40, f_img=ref_img, g_img=tar_img)
        m_lr = Mesh(boundary, target_nodes=40, f_img=ref_img, g_img=tar_img)
        m_rg.solve(tmpl, seed_coord=[200.0, 200.0], max_iterations=30,
                   solver_options={"preconditioning": "RG"})
        m_lr.solve(tmpl, seed_coord=[200.0, 200.0], max_iterations=30,
                   solver_options={"preconditioning": "layer-RG"})
        assert m_rg.solved and m_lr.solved
        d_rg = np.asarray(m_rg.displacements)
        d_lr = np.asarray(m_lr.displacements)
        scale = np.maximum(np.abs(d_rg).max(axis=1, keepdims=True), 1.0)
        assert np.all(np.abs(d_rg - d_lr) <= 1e-5 * scale)
        np.testing.assert_allclose(np.asarray(m_rg.c_zncc), np.asarray(m_lr.c_zncc),
                                   atol=1e-5)

    def test_layer_rg_subdict_tunables_accepted(self, unsolved_mesh):
        tmpl = make_circle_template(20)
        unsolved_mesh.solve(tmpl, seed_coord=[200.0, 200.0], max_iterations=20,
                             solver_options={"preconditioning": "layer-RG",
                                             "layer_rg": {"batch_factor": 2,
                                                          "root_rel_eps": 0.05,
                                                          "max_workers": 2}})
        assert unsolved_mesh.solved

    def test_unknown_top_level_key_raises(self, solved_mesh):
        tmpl = make_circle_template(20)
        with pytest.raises(ValueError, match="unknown"):
            solved_mesh.solve(tmpl, seed_coord=[200.0, 200.0],
                               solver_options={"bogus_axis": "x"})

    def test_unknown_axis_value_raises(self, solved_mesh):
        tmpl = make_circle_template(20)
        with pytest.raises(ValueError, match="masking"):
            solved_mesh.solve(tmpl, seed_coord=[200.0, 200.0],
                               solver_options={"masking": "bogus"})

    def test_subdict_for_inactive_axis_raises(self, solved_mesh):
        tmpl = make_circle_template(20)
        with pytest.raises(ValueError, match="layer_rg"):
            solved_mesh.solve(tmpl, seed_coord=[200.0, 200.0],
                               solver_options={"preconditioning": "RG",
                                               "layer_rg": {"batch_factor": 3}})

    def test_unknown_subdict_key_raises(self, solved_mesh):
        tmpl = make_circle_template(20)
        with pytest.raises(ValueError, match="bogus"):
            solved_mesh.solve(tmpl, seed_coord=[200.0, 200.0],
                               solver_options={"preconditioning": "layer-RG",
                                               "layer_rg": {"bogus": 1}})

    def test_solver_options_not_a_dict_raises(self, solved_mesh):
        tmpl = make_circle_template(20)
        with pytest.raises(TypeError):
            solved_mesh.solve(tmpl, seed_coord=[200.0, 200.0], solver_options="uniform")

    # Sequence-level solver=/solver_options= coverage (needs a real on-disk
    # image directory, not the in-memory ref_img/tar_img fixtures this module
    # otherwise uses) lives in tests/python/sequence/test_sequence.py,
    # alongside that file's existing IMAGES_AVAILABLE-guarded Sequence tests.
