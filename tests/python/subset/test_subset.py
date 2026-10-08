"""
Tests for geopyv_dev.Subset — new API (g_img at construction, solve mutates in place).

Tolerance tiers (per plan):
  Tier B  rtol=1e-8   B-spline intensity interpolation
  Tier C  rtol=1e-5   Solver outputs (ZNCC, warp at convergence)

Golden values are captured from geopyv_dev.  They do NOT match the original
geopyv Python package: geopyv_dev's B-spline prefilter
(`image.rs::build_kernel`) corrects a one-sample kernel-phase error in
geopyv's `image.py::_get_C` that shifts every interpolated intensity by
(1, 1) px.  Reference-subset quantities (f_m, delta_f, sssig, ...) and the
converged translation therefore differ from geopyv by that one-pixel offset.
"""

import os
import tempfile
import warnings

import cv2
import numpy as np
import pytest

from geopyv_dev import Image, Subset, Mask

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "geopyv", "tests"
)
REF_JPG = os.path.join(TESTS_DIR, "ref.jpg")
TAR_JPG = os.path.join(TESTS_DIR, "tar.jpg")
IMAGES_AVAILABLE = os.path.exists(REF_JPG) and os.path.exists(TAR_JPG)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COORD = [200.43, 200.76]
RADIUS = 25
BORDER = 20

GOLDEN_N_PX = 1961
GOLDEN_F_M = 72.4607678591
GOLDEN_DELTA_F = 3166.7538783369
GOLDEN_SSSIG = 430193.4940805415
GOLDEN_SIGMA_INT = 71.5114385032

GOLDEN_ICGN_O1_ZNCC = 0.9999868056
GOLDEN_ICGN_O1_ITERS = 3
GOLDEN_ICGN_O1_P = [
    3.38914761e-02, 3.51348955e-02,
    9.94055325e-05, -5.21745977e-05,
    4.29306692e-05, -5.84241277e-05,
]

GOLDEN_FAGN_O1_ZNCC = 0.9999868042
GOLDEN_FAGN_O1_ITERS = 3
GOLDEN_FAGN_O1_P = [
    3.34467004e-02, 3.44336229e-02,
    9.55317604e-05, -5.02131885e-05,
    4.17281040e-05, -4.96162986e-05,
]

pytestmark = pytest.mark.skipif(
    not IMAGES_AVAILABLE,
    reason="test images not found (geopyv/tests/ref.jpg, tar.jpg)",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _preprocess(filepath):
    """BGR → greyscale → GaussianBlur (matches geopyv Image)."""
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gs, (5, 5), sigmaX=1.1, sigmaY=1.1)


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
def tmpl():
    return Mask(mask_type="local", shape="circle", size=RADIUS)


@pytest.fixture(scope="module")
def ref_img_with_path():
    """Image loaded from file — has a filepath attribute for save/load tests."""
    return Image(filepath=REF_JPG, border=BORDER)


@pytest.fixture(scope="module")
def tar_img_with_path():
    return Image(filepath=TAR_JPG, border=BORDER)


@pytest.fixture(scope="module")
def unsolved_subset(ref_img, tar_img, tmpl):
    """A fresh, unsolved Subset."""
    return Subset(COORD, tmpl, ref_img, tar_img)


# ---------------------------------------------------------------------------
# Construction and reference quantities (Tier B: rtol=1e-8)
# ---------------------------------------------------------------------------


class TestSubsetConstruction:
    def test_n_px(self, unsolved_subset):
        assert unsolved_subset.n_px == GOLDEN_N_PX

    def test_coord(self, unsolved_subset):
        assert unsolved_subset.coord[0] == pytest.approx(COORD[0], rel=1e-12)
        assert unsolved_subset.coord[1] == pytest.approx(COORD[1], rel=1e-12)

    def test_template_shape(self, unsolved_subset):
        assert unsolved_subset.template_shape == "circle"

    def test_template_size(self, unsolved_subset):
        assert unsolved_subset.template_size == RADIUS

    def test_template_n_px(self, unsolved_subset):
        assert unsolved_subset.template_n_px == GOLDEN_N_PX

    def test_order_default(self, unsolved_subset):
        assert unsolved_subset.subset_order == 1

    def test_solved_false_before_solve(self, unsolved_subset):
        assert unsolved_subset.solved is False

    def test_f_coords_shape(self, unsolved_subset):
        assert unsolved_subset.f_coords.shape == (GOLDEN_N_PX, 2)

    def test_f_coords_range(self, unsolved_subset):
        fc = unsolved_subset.f_coords
        assert float(fc[:, 0].min()) == pytest.approx(COORD[0] - RADIUS, rel=1e-12)
        assert float(fc[:, 0].max()) == pytest.approx(COORD[0] + RADIUS, rel=1e-12)

    def test_f_shape(self, unsolved_subset):
        assert unsolved_subset.f.shape == (GOLDEN_N_PX,)

    def test_f_m(self, unsolved_subset):
        assert unsolved_subset.f_m == pytest.approx(GOLDEN_F_M, rel=1e-8)

    def test_delta_f(self, unsolved_subset):
        assert unsolved_subset.delta_f == pytest.approx(GOLDEN_DELTA_F, rel=1e-8)

    def test_grad_f_shape(self, unsolved_subset):
        assert unsolved_subset.grad_f.shape == (GOLDEN_N_PX, 2)

    def test_grad_f_finite(self, unsolved_subset):
        assert np.all(np.isfinite(unsolved_subset.grad_f))

    def test_sssig(self, unsolved_subset):
        assert unsolved_subset.sssig == pytest.approx(GOLDEN_SSSIG, rel=1e-8)

    def test_sigma_intensity(self, unsolved_subset):
        assert unsolved_subset.sigma_intensity == pytest.approx(GOLDEN_SIGMA_INT, rel=1e-8)

    def test_f_img_accessible(self, unsolved_subset):
        assert unsolved_subset.f_img is not None

    def test_g_img_accessible(self, unsolved_subset):
        assert unsolved_subset.g_img is not None

    def test_order2_construction(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img, subset_order=2)
        assert s.subset_order == 2
        assert s.solved is False

    def test_invalid_order_raises(self, ref_img, tar_img, tmpl):
        with pytest.raises(ValueError):
            Subset(COORD, tmpl, ref_img, tar_img, subset_order=3)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_unsolved(self, unsolved_subset):
        r = repr(unsolved_subset)
        assert "Subset" in r
        assert "solved=False" in r

    def test_repr_solved(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve()
        r = repr(s)
        assert "solved=True" in r
        assert "p=" in r

    def test_repr_contains_n_px(self, unsolved_subset):
        assert "n_px" in repr(unsolved_subset)


# ---------------------------------------------------------------------------
# ICGN solver (Tier C: rtol=1e-5)
# ---------------------------------------------------------------------------


class TestSolveICGN:
    @pytest.fixture(scope="class")
    def solved(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve(algorithm="icgn")
        return s

    def test_returns_none(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        rv = s._inner.solve_icgn()
        assert rv is None

    def test_solved_true_after_solve(self, solved):
        assert solved.solved is True

    def test_converges_order1(self, solved):
        assert solved.converged is True

    def test_zncc_order1(self, solved):
        assert solved.c_zncc == pytest.approx(GOLDEN_ICGN_O1_ZNCC, rel=1e-5)

    def test_iterations_order1(self, solved):
        assert solved.iterations == GOLDEN_ICGN_O1_ITERS

    def test_p_order1(self, solved):
        for i, (got, exp) in enumerate(zip(solved.p, GOLDEN_ICGN_O1_P)):
            assert got == pytest.approx(exp, rel=1e-5), f"p[{i}]: {got} != {exp}"

    def test_p_length_order1(self, solved):
        assert len(solved.p) == 6

    def test_znssd_range(self, solved):
        assert 0.0 <= solved.c_znssd <= 4.0

    def test_zncc_from_znssd(self, solved):
        assert solved.c_zncc == pytest.approx(1.0 - solved.c_znssd / 2.0, rel=1e-12)

    def test_history_length(self, solved):
        assert len(solved.history) == solved.iterations

    def test_converges_order2(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img, subset_order=2)
        s.solve(algorithm="icgn")
        assert s.converged
        assert s.c_zncc > 0.999
        assert len(s.p) == 12

    def test_max_iterations_respected(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s._inner.solve_icgn(max_norm=1e-20, max_iterations=1)
        assert s.iterations == 1
        assert not s.converged


# ---------------------------------------------------------------------------
# FAGN solver (Tier C: rtol=1e-5)
# ---------------------------------------------------------------------------


class TestSolveFAGN:
    @pytest.fixture(scope="class")
    def solved(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve(algorithm="fagn")
        return s

    def test_solved_true(self, solved):
        assert solved.solved is True

    def test_converges_order1(self, solved):
        assert solved.converged is True

    def test_zncc_order1(self, solved):
        assert solved.c_zncc == pytest.approx(GOLDEN_FAGN_O1_ZNCC, rel=1e-5)

    def test_iterations_order1(self, solved):
        assert solved.iterations == GOLDEN_FAGN_O1_ITERS

    def test_p_order1(self, solved):
        for i, (got, exp) in enumerate(zip(solved.p, GOLDEN_FAGN_O1_P)):
            assert got == pytest.approx(exp, rel=1e-5), f"p[{i}]: {got} != {exp}"

    def test_p_length_order1(self, solved):
        assert len(solved.p) == 6

    def test_converges_order2(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img, subset_order=2)
        s.solve(algorithm="fagn")
        assert s.converged
        assert s.c_zncc > 0.999

    def test_max_iterations_respected(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s._inner.solve_fagn(max_norm=1e-20, max_iterations=1)
        assert s.iterations == 1
        assert not s.converged


# ---------------------------------------------------------------------------
# Unified solve() dispatcher
# ---------------------------------------------------------------------------


class TestSolveDispatcher:
    def test_icgn_algorithm(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve(algorithm="icgn")
        assert s.solved
        assert s.c_zncc == pytest.approx(GOLDEN_ICGN_O1_ZNCC, rel=1e-5)

    def test_fagn_algorithm(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve(algorithm="fagn")
        assert s.solved
        assert s.c_zncc == pytest.approx(GOLDEN_FAGN_O1_ZNCC, rel=1e-5)

    def test_default_algorithm_is_icgn(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve()
        assert s.c_zncc == pytest.approx(GOLDEN_ICGN_O1_ZNCC, rel=1e-5)

    def test_unknown_algorithm_warns_and_falls_back(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # algorithm= is itself a deprecated alias for solver= (see
            # solver_options_restructure.md §2) -- using it here (rather
            # than solver=) additionally emits a DeprecationWarning, on top
            # of the UserWarning this test actually checks for.
            s.solve(algorithm="unknown_algo")
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)
                         and not issubclass(x.category, DeprecationWarning)]
        assert len(user_warnings) == 1
        assert "unknown_algo" in str(user_warnings[0].message).lower()
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert s.solved

    def test_case_insensitive(self, ref_img, tar_img, tmpl):
        s1 = Subset(COORD, tmpl, ref_img, tar_img)
        s1.solve(algorithm="ICGN")
        s2 = Subset(COORD, tmpl, ref_img, tar_img)
        s2.solve(algorithm="icgn")
        assert s1.c_zncc == pytest.approx(s2.c_zncc, rel=1e-12)


# ---------------------------------------------------------------------------
# Warp-adequacy diagnostic (eta_u/eta_v, residual_map) -- order-1 and order-2.
#
# Real photographed ref.jpg/tar.jpg at this coord/radius is a near-rigid,
# well-converged case (GOLDEN_ICGN_O1_ZNCC ~ 0.999987) -- i.e. the "adequate"
# side of the report's adequate/under-fit split, so both eta and the
# residual map should be small, not merely "some finite number". This
# doesn't exercise the "under-fit" side (that needs a manufactured
# quadratic/localised-displacement image pair, not the fixed real fixture
# images this test module uses) -- see the plan discussion for that gap.
#
# Order-2's omitted modes are cubic, not the report's own (order-1,
# quadratic) case -- see `omitted_mode_diagnostic`'s doc comment in
# `src/subset.rs` for the caveat that this extension isn't itself validated
# by the report.
# ---------------------------------------------------------------------------


class TestWarpAdequacyDiagnostic:
    def test_eta_present_for_order1(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve(algorithm="icgn")
        assert s.eta_u is not None
        assert s.eta_v is not None
        assert s.eta_u >= 0.0
        assert s.eta_v >= 0.0

    def test_eta_present_for_order2(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img, subset_order=2)
        s.solve(algorithm="icgn")
        assert s.eta_u is not None
        assert s.eta_v is not None
        assert s.eta_u >= 0.0
        assert s.eta_v >= 0.0

    def test_eta_none_when_unsolved(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        assert s.eta_u is None
        assert s.eta_v is None

    def test_eta_small_for_well_converged_real_subset(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve(algorithm="icgn")
        assert s.eta_u < 0.5
        assert s.eta_v < 0.5

    def test_eta_small_for_well_converged_real_subset_order2(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img, subset_order=2)
        s.solve(algorithm="icgn")
        assert s.eta_u < 0.5
        assert s.eta_v < 0.5

    def test_eta_fagn_also_populated(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve(algorithm="fagn")
        assert s.eta_u is not None
        assert s.eta_v is not None

    def test_residual_map_shape_matches_n_px(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve(algorithm="icgn")
        residual = s.residual_map()
        assert residual.shape == (s.n_px,)

    def test_residual_map_small_for_well_converged_real_subset(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve(algorithm="icgn")
        residual = s.residual_map()
        assert np.abs(residual).mean() < 0.1

    def test_residual_map_raises_when_unsolved(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        with pytest.raises(Exception):
            s.residual_map()

    def test_inspect_residual_does_not_raise(self, ref_img, tar_img, tmpl):
        import matplotlib.pyplot as plt
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s.solve(algorithm="icgn")
        fig, ax = s.inspect(residual=True, show=False, block=False)
        plt.close(fig)


# ---------------------------------------------------------------------------
# p_0 mismatch warnings (Section G)
# ---------------------------------------------------------------------------


class TestP0Mismatch:
    def test_p0_mismatch_order1_truncates(self, ref_img, tar_img, tmpl):
        """p_0 length 12 with subset_order=1: silently truncated to 6."""
        s = Subset(COORD, tmpl, ref_img, tar_img, subset_order=1)
        p_0_long = [0.0] * 12
        s._inner.solve_icgn(p_0=p_0_long)
        assert len(s.p) == 6

    def test_p0_mismatch_order2_pads(self, ref_img, tar_img, tmpl):
        """p_0 length 6 with subset_order=2: silently padded to 12."""
        s = Subset(COORD, tmpl, ref_img, tar_img, subset_order=2)
        p_0_short = [0.0] * 6
        s._inner.solve_icgn(p_0=p_0_short)
        assert len(s.p) == 12
        # Second-order terms should be near zero (padded from zero start).
        for v in s.p[6:]:
            assert abs(v) < 1e-3

    def test_default_p0_order1(self, ref_img, tar_img, tmpl):
        """p_0=None defaults to zeros (order-1)."""
        s = Subset(COORD, tmpl, ref_img, tar_img)
        s._inner.solve_icgn()
        assert s.solved
        assert len(s.p) == 6

    def test_default_p0_order2(self, ref_img, tar_img, tmpl):
        """p_0=None defaults to zeros (order-2)."""
        s = Subset(COORD, tmpl, ref_img, tar_img, subset_order=2)
        s._inner.solve_icgn()
        assert s.solved
        assert len(s.p) == 12


# ---------------------------------------------------------------------------
# Save / load (Section G)
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_unsolved_raises(self, ref_img, tar_img, tmpl):
        s = Subset(COORD, tmpl, ref_img, tar_img)
        with tempfile.NamedTemporaryFile(suffix=".pyv") as f:
            with pytest.raises(RuntimeError):
                s.save(f.name)

    def test_save_load_roundtrip(self, ref_img_with_path, tar_img_with_path, tmpl):
        """Solve, save, load, assert key fields are preserved."""
        import geopyv_dev as gp
        s = Subset(COORD, tmpl, ref_img_with_path, tar_img_with_path)
        s.solve()

        with tempfile.NamedTemporaryFile(suffix=".pyv", delete=False) as f:
            path = f.name

        try:
            s.save(path)
            loaded = gp.load(path)

            assert loaded.coord[0] == pytest.approx(s.coord[0], rel=1e-12)
            assert loaded.coord[1] == pytest.approx(s.coord[1], rel=1e-12)
            assert loaded.template_size == s.template_size
            assert loaded.c_zncc == pytest.approx(s.c_zncc, rel=1e-12)
            for got, exp in zip(loaded.p, s.p):
                assert got == pytest.approx(exp, rel=1e-12)
            assert loaded.solved is True
            assert loaded.eta_u == pytest.approx(s.eta_u, rel=1e-12)
            assert loaded.eta_v == pytest.approx(s.eta_v, rel=1e-12)
        finally:
            os.unlink(path)

    def test_load_restores_images(self, ref_img_with_path, tar_img_with_path, tmpl):
        """After load, f_img should be non-None with the correct filepath."""
        import geopyv_dev as gp
        s = Subset(COORD, tmpl, ref_img_with_path, tar_img_with_path)
        s.solve()

        with tempfile.NamedTemporaryFile(suffix=".pyv", delete=False) as f:
            path = f.name

        try:
            s.save(path)
            loaded = gp.load(path)
            assert loaded.f_img is not None
            assert loaded.f_img_path is not None
            assert "ref" in loaded.f_img_path.lower()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Cross-solver consistency
# ---------------------------------------------------------------------------


class TestSolverConsistency:
    def test_icgn_fagn_zncc_close(self, ref_img, tar_img, tmpl):
        si = Subset(COORD, tmpl, ref_img, tar_img)
        si.solve(algorithm="icgn")
        sf = Subset(COORD, tmpl, ref_img, tar_img)
        sf.solve(algorithm="fagn")
        assert abs(si.c_zncc - sf.c_zncc) < 1e-4

    def test_icgn_fagn_displacement_close(self, ref_img, tar_img, tmpl):
        si = Subset(COORD, tmpl, ref_img, tar_img)
        si.solve(algorithm="icgn")
        sf = Subset(COORD, tmpl, ref_img, tar_img)
        sf.solve(algorithm="fagn")
        assert abs(si.p[0] - sf.p[0]) < 1e-3
        assert abs(si.p[1] - sf.p[1]) < 1e-3

