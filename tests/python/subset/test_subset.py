"""
Phase 3 validation tests for geopyv_dev.Subset.

Tolerance tiers (per plan):
  Tier B  rtol=1e-8   B-spline intensity interpolation
  Tier C  rtol=1e-5   Solver outputs (ZNCC, warp at convergence)

Golden values were captured by running:

    pytest ../geopyv/tests/subset/fixtures.py -v -s

in the geopyv-dev venv.
"""

import os

import cv2
import numpy as np
import pytest

from geopyv_dev import Image, Subset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "geopyv", "tests")
REF_JPG = os.path.join(TESTS_DIR, "ref.jpg")
TAR_JPG = os.path.join(TESTS_DIR, "tar.jpg")

# ---------------------------------------------------------------------------
# Constants (match fixtures.py exactly)
# ---------------------------------------------------------------------------

COORD = [200.43, 200.76]
RADIUS = 25
BORDER = 20

P_ZERO_1 = [0.0] * 6
P_ZERO_2 = [0.0] * 12
P_SMALL = [2.0, 1.5, 0.0, 0.0, 0.0, 0.0]

# Golden values from fixture run
GOLDEN_N_PX      = 1961
GOLDEN_F_M       = 70.8558139299
GOLDEN_DELTA_F   = 3179.3809800153
GOLDEN_SSSIG     = 420037.8754963874
GOLDEN_SIGMA_INT = 71.7965829255

GOLDEN_ICGN_O1_ZNCC = 0.999987
GOLDEN_ICGN_O1_ITERS = 3
GOLDEN_ICGN_O1_P = [3.41341653e-02, 3.53146414e-02,
                     9.80768226e-05, -9.05461037e-05,
                     2.90342155e-05, -8.11538336e-05]

GOLDEN_FAGN_O1_ZNCC = 0.999987
GOLDEN_FAGN_O1_ITERS = 3
GOLDEN_FAGN_O1_P = [3.36763959e-02, 3.45660975e-02,
                     9.56551027e-05, -8.60159524e-05,
                     2.57900894e-05, -7.25744604e-05]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _preprocess(filepath):
    """BGR → greyscale → GaussianBlur (matches geopyv Image)."""
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gs, (5, 5), sigmaX=1.1, sigmaY=1.1)


def make_circle_coords(radius):
    """Replicate Circle(radius).coords exactly (row-major, np.where order)."""
    size = 2 * radius + 1
    x, y = np.meshgrid(range(size), range(size))
    x, y = x - radius, y - radius
    dist = np.sqrt(x**2 + y**2)
    x_s, y_s = np.where(dist <= radius)
    n_px = x_s.shape[0]
    coords = np.empty((n_px, 2), order="F")
    coords[:, 0] = (x_s - radius).astype(float)
    coords[:, 1] = (y_s - radius).astype(float)
    return coords


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
def tmpl_coords():
    return make_circle_coords(RADIUS)


@pytest.fixture(scope="module")
def ref_subset(ref_img, tmpl_coords):
    return Subset(COORD, tmpl_coords, ref_img.qcqt)


# ---------------------------------------------------------------------------
# Construction and reference quantities (Tier B: rtol=1e-8)
# ---------------------------------------------------------------------------


class TestSubsetConstruction:
    def test_n_px(self, ref_subset):
        assert ref_subset.n_px == GOLDEN_N_PX

    def test_coord(self, ref_subset):
        assert ref_subset.coord[0] == pytest.approx(COORD[0], rel=1e-12)
        assert ref_subset.coord[1] == pytest.approx(COORD[1], rel=1e-12)

    def test_f_coords_shape(self, ref_subset):
        assert ref_subset.f_coords.shape == (GOLDEN_N_PX, 2)

    def test_f_coords_range(self, ref_subset):
        """f_coords[:,0] spans [coord[0]-radius, coord[0]+radius]."""
        fc = ref_subset.f_coords
        assert float(fc[:, 0].min()) == pytest.approx(COORD[0] - RADIUS, rel=1e-12)
        assert float(fc[:, 0].max()) == pytest.approx(COORD[0] + RADIUS, rel=1e-12)

    def test_f_shape(self, ref_subset):
        assert ref_subset.f.shape == (GOLDEN_N_PX,)

    def test_f_m(self, ref_subset):
        """Tier B: f_m matches numpy golden value."""
        assert ref_subset.f_m == pytest.approx(GOLDEN_F_M, rel=1e-8)

    def test_delta_f(self, ref_subset):
        """Tier B: delta_f matches numpy golden value."""
        assert ref_subset.delta_f == pytest.approx(GOLDEN_DELTA_F, rel=1e-8)

    def test_grad_f_shape(self, ref_subset):
        assert ref_subset.grad_f.shape == (GOLDEN_N_PX, 2)

    def test_grad_f_finite(self, ref_subset):
        assert np.all(np.isfinite(ref_subset.grad_f))

    def test_sssig(self, ref_subset):
        """Tier B: SSSIG matches numpy golden value."""
        assert ref_subset.sssig == pytest.approx(GOLDEN_SSSIG, rel=1e-8)

    def test_sigma_intensity(self, ref_subset):
        """Tier B: sigma_intensity matches numpy golden value."""
        assert ref_subset.sigma_intensity == pytest.approx(GOLDEN_SIGMA_INT, rel=1e-8)

    def test_f_m_consistent(self, ref_subset):
        """f_m == mean(f) — internal consistency."""
        assert ref_subset.f_m == pytest.approx(float(ref_subset.f.mean()), rel=1e-12)

    def test_delta_f_consistent(self, ref_subset):
        """delta_f == sqrt(sum((f - f_m)^2)) — internal consistency."""
        f = ref_subset.f
        f_m = ref_subset.f_m
        expected = float(np.sqrt(np.sum((f - f_m) ** 2)))
        assert ref_subset.delta_f == pytest.approx(expected, rel=1e-12)

    def test_sssig_consistent(self, ref_subset):
        """sssig == sum(0.5*(gx^2 + gy^2)) — internal consistency."""
        gf = ref_subset.grad_f
        expected = float(np.sum(0.5 * (gf[:, 0] ** 2 + gf[:, 1] ** 2)))
        assert ref_subset.sssig == pytest.approx(expected, rel=1e-12)

    def test_repr(self, ref_subset):
        r = repr(ref_subset)
        assert "Subset" in r
        assert "n_px" in r

    def test_empty_template_raises(self, ref_img):
        """Empty template_coords → InvalidInput error."""
        empty = np.zeros((0, 2), dtype=np.float64)
        with pytest.raises(Exception):
            Subset(COORD, empty, ref_img.qcqt)


# ---------------------------------------------------------------------------
# ICGN solver (Tier C: rtol=1e-5)
# ---------------------------------------------------------------------------


class TestSolveICGN:
    def test_returns_dict(self, ref_subset, tar_img):
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1)
        assert isinstance(r, dict)
        for key in ("p", "c_zncc", "c_znssd", "iterations", "converged", "history"):
            assert key in r

    def test_converges_order1(self, ref_subset, tar_img):
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1, max_norm=1e-3, max_iterations=50)
        assert r["converged"], f"ICGN order-1 did not converge: {r}"

    def test_zncc_order1(self, ref_subset, tar_img):
        """Tier C: ZNCC ≈ 0.999987."""
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1)
        assert r["c_zncc"] == pytest.approx(GOLDEN_ICGN_O1_ZNCC, rel=1e-5)

    def test_iterations_order1(self, ref_subset, tar_img):
        """ICGN order-1 converges in exactly 3 iterations for this pair."""
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1)
        assert r["iterations"] == GOLDEN_ICGN_O1_ITERS

    def test_p_order1(self, ref_subset, tar_img):
        """Tier C: warp displacement components match golden values."""
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1)
        p = r["p"]
        for i, (got, exp) in enumerate(zip(p, GOLDEN_ICGN_O1_P)):
            assert got == pytest.approx(exp, rel=1e-5), f"p[{i}]: {got} != {exp}"

    def test_znssd_range(self, ref_subset, tar_img):
        """ZNSSD ∈ [0, 4]."""
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1)
        assert 0.0 <= r["c_znssd"] <= 4.0

    def test_zncc_from_znssd(self, ref_subset, tar_img):
        """ZNCC = 1 - ZNSSD/2."""
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1)
        assert r["c_zncc"] == pytest.approx(1.0 - r["c_znssd"] / 2.0, rel=1e-12)

    def test_history_length(self, ref_subset, tar_img):
        """history has one entry per iteration."""
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1)
        assert len(r["history"]) == r["iterations"]

    def test_converges_order2(self, ref_subset, tar_img):
        """ICGN order-2 also converges on this pair."""
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_2)
        assert r["converged"]
        assert r["c_zncc"] > 0.999

    def test_p_length_order1(self, ref_subset, tar_img):
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1)
        assert len(r["p"]) == 6

    def test_p_length_order2(self, ref_subset, tar_img):
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_2)
        assert len(r["p"]) == 12

    def test_max_iterations_respected(self, ref_subset, tar_img):
        """Capping at 1 iteration gives no convergence but returns a result."""
        r = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1, max_norm=1e-20, max_iterations=1)
        assert r["iterations"] == 1
        assert not r["converged"]


# ---------------------------------------------------------------------------
# FAGN solver (Tier C: rtol=1e-5)
# ---------------------------------------------------------------------------


class TestSolveFAGN:
    def test_returns_dict(self, ref_subset, tar_img):
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        assert isinstance(r, dict)
        for key in ("p", "c_zncc", "c_znssd", "iterations", "converged", "history"):
            assert key in r

    def test_converges_order1(self, ref_subset, tar_img):
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        assert r["converged"], f"FAGN order-1 did not converge: {r}"

    def test_zncc_order1(self, ref_subset, tar_img):
        """Tier C: ZNCC ≈ 0.999987."""
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        assert r["c_zncc"] == pytest.approx(GOLDEN_FAGN_O1_ZNCC, rel=1e-5)

    def test_iterations_order1(self, ref_subset, tar_img):
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        assert r["iterations"] == GOLDEN_FAGN_O1_ITERS

    def test_p_order1(self, ref_subset, tar_img):
        """Tier C: warp displacement components match golden values."""
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        p = r["p"]
        for i, (got, exp) in enumerate(zip(p, GOLDEN_FAGN_O1_P)):
            assert got == pytest.approx(exp, rel=1e-5), f"p[{i}]: {got} != {exp}"

    def test_znssd_range(self, ref_subset, tar_img):
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        assert 0.0 <= r["c_znssd"] <= 4.0

    def test_zncc_from_znssd(self, ref_subset, tar_img):
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        assert r["c_zncc"] == pytest.approx(1.0 - r["c_znssd"] / 2.0, rel=1e-12)

    def test_history_length(self, ref_subset, tar_img):
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        assert len(r["history"]) == r["iterations"]

    def test_converges_order2(self, ref_subset, tar_img):
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_2)
        assert r["converged"]
        assert r["c_zncc"] > 0.999

    def test_p_length_order1(self, ref_subset, tar_img):
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        assert len(r["p"]) == 6

    def test_p_length_order2(self, ref_subset, tar_img):
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_2)
        assert len(r["p"]) == 12

    def test_preconditioned_converges(self, ref_subset, tar_img):
        """FAGN converges from a close initial guess."""
        p_0 = [0.01, 0.02, 0.0, 0.0, 0.0, 0.0]
        r = ref_subset.solve_fagn(tar_img.qcqt, p_0)
        assert r["converged"]
        assert r["c_zncc"] > 0.999

    def test_max_iterations_respected(self, ref_subset, tar_img):
        r = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1, max_norm=1e-20, max_iterations=1)
        assert r["iterations"] == 1
        assert not r["converged"]


# ---------------------------------------------------------------------------
# Cross-solver consistency
# ---------------------------------------------------------------------------


class TestSolverConsistency:
    def test_icgn_fagn_zncc_close(self, ref_subset, tar_img):
        """ICGN and FAGN converge to the same ZNCC score (< 1e-4 difference)."""
        ri = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1)
        rf = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        assert abs(ri["c_zncc"] - rf["c_zncc"]) < 1e-4

    def test_icgn_fagn_displacement_close(self, ref_subset, tar_img):
        """ICGN and FAGN displacements agree to within 1e-3 pixels."""
        ri = ref_subset.solve_icgn(tar_img.qcqt, P_ZERO_1)
        rf = ref_subset.solve_fagn(tar_img.qcqt, P_ZERO_1)
        assert abs(ri["p"][0] - rf["p"][0]) < 1e-3, f"u: {ri['p'][0]} vs {rf['p'][0]}"
        assert abs(ri["p"][1] - rf["p"][1]) < 1e-3, f"v: {ri['p'][1]} vs {rf['p'][1]}"
