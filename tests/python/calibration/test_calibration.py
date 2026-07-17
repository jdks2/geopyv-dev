import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

import numpy as np
import pytest

import geopyv_dev as gp
from geopyv_dev import Calibration, CalibrationParams

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
SHEAR_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "images", "shear"))
CALIB_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "..", "geopyv", "images", "calibration")
)
SHEAR_AVAILABLE = os.path.isdir(SHEAR_DIR)
CALIB_AVAILABLE = os.path.isdir(CALIB_DIR) and len(os.listdir(CALIB_DIR)) > 0

BOUNDARY_NODES = np.array(
    [[300.0, 300.0], [300.0, 700.0], [700.0, 700.0], [700.0, 300.0]]
)


# ---------------------------------------------------------------------------
# Calibration-math helpers (no image data needed)
# ---------------------------------------------------------------------------

def _identity_calibration_params():
    """A camera model whose i2o/o2i is exactly the identity map. extmat can't
    literally be the identity matrix (divides by zero at z=0) — the depth
    (extmat[2,3]) and focal length must match instead."""
    intmat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    extmat = np.eye(4)
    extmat[2, 3] = 1.0
    return CalibrationParams(intmat=intmat, extmat=extmat, dist=np.zeros(5))


def _half_scale_calibration_params():
    """Same as identity, but the focal length is doubled: i2o(pt) == 0.5 * pt."""
    intmat = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    extmat = np.eye(4)
    extmat[2, 3] = 1.0
    return CalibrationParams(intmat=intmat, extmat=extmat, dist=np.zeros(5))


class TestCalibrationParams:
    def test_identity_roundtrip(self):
        cal = _identity_calibration_params()
        pts = np.array([[300.0, 200.0], [700.0, 600.0]])
        assert np.allclose(cal.i2o(pts), pts)
        assert np.allclose(cal.o2i(pts), pts)

    def test_half_scale(self):
        cal = _half_scale_calibration_params()
        pts = np.array([[300.0, 200.0], [700.0, 600.0]])
        assert np.allclose(cal.i2o(pts), pts * 0.5)

    def test_modify_does_not_mutate_original(self):
        cal = _half_scale_calibration_params()
        original_extmat = np.array(cal.extmat)
        modified = cal.modify(dangles=[0.0, 0.0, 0.05], centre=[400.0, 300.0])
        assert not np.allclose(modified.extmat, original_extmat)
        assert np.allclose(cal.extmat, original_extmat)

    def test_modify_result_still_roundtrips(self):
        cal = _half_scale_calibration_params()
        modified = cal.modify(dangles=[0.02, -0.01, 0.03], centre=[500.0, 400.0])
        pts = np.array([[300.0, 200.0], [700.0, 600.0]])
        obj = modified.i2o(pts)
        back = modified.o2i(obj)
        assert np.allclose(back, pts, atol=1e-4)


class TestCalibrationModify:
    """Calibration.modify() wiring, without needing a real ChArUco solve."""

    def _bare_calibration(self, params):
        cal = Calibration.__new__(Calibration)
        cal.solved = True
        cal.params = params
        return cal

    def test_modify_replaces_params(self):
        cal = self._bare_calibration(_half_scale_calibration_params())
        original_extmat = np.array(cal.params.extmat)
        cal.modify(dangles=[0.0, 0.0, 0.02], centre=[100.0, 100.0])
        assert not np.allclose(cal.params.extmat, original_extmat)

    def test_modify_before_solve_raises(self):
        cal = Calibration.__new__(Calibration)
        cal.solved = False
        cal.params = None
        with pytest.raises(RuntimeError):
            cal.modify()


@pytest.mark.skipif(not SHEAR_AVAILABLE, reason="images/shear not found")
class TestRegionCalibrationOrdering:
    """Region.calibrate() is only safe *after* it has built a Mesh/Sequence —
    see docs/tutorials/08_calibration.rst 'Calibrating objects'."""

    def _boundary(self):
        return gp.PathRegion(nodes=BOUNDARY_NODES.copy(), hard=False)

    def _images(self):
        f_img = gp.Image(filepath=os.path.join(SHEAR_DIR, "shear_0.jpg"))
        g_img = gp.Image(filepath=os.path.join(SHEAR_DIR, "shear_4.jpg"))
        return f_img, g_img

    def _mesh(self, boundary):
        f_img, g_img = self._images()
        return gp.Mesh(boundary=boundary, target_nodes=30, f_img=f_img, g_img=g_img,
                        size=(15.0, 70.0), mesh_order=1)

    def test_uncalibrated_region_builds_mesh_normally(self):
        mesh = self._mesh(self._boundary())
        assert mesh.nodes.shape[0] > 0

    def test_calibrate_region_after_mesh_build_is_safe(self):
        boundary = self._boundary()
        mesh = self._mesh(boundary)  # consumes boundary's pixel-space nodes here
        original_nodes = np.array(boundary.current_nodes)
        n_nodes_before = mesh.nodes.shape[0]

        cal = _half_scale_calibration_params()
        boundary.current_nodes = cal.i2o(boundary.current_nodes)
        boundary.calibrated = True

        assert np.allclose(boundary.current_nodes, original_nodes * 0.5)
        # The already-built mesh is unaffected by calibrating boundary afterward.
        assert mesh.nodes.shape[0] == n_nodes_before

    def test_calibrated_region_rejected_by_mesh_construction(self):
        boundary = self._boundary()
        cal = _half_scale_calibration_params()
        boundary.current_nodes = cal.i2o(boundary.current_nodes)
        boundary.calibrated = True
        with pytest.raises(TypeError):
            self._mesh(boundary)

    def test_calibrated_region_rejected_by_sequence_construction(self):
        boundary = self._boundary()
        cal = _half_scale_calibration_params()
        boundary.current_nodes = cal.i2o(boundary.current_nodes)
        boundary.calibrated = True
        with pytest.raises(TypeError):
            gp.Sequence(image_dir=SHEAR_DIR, boundary=boundary, target_nodes=30,
                        size=(15.0, 70.0), mesh_order=1)


@pytest.mark.skipif(not SHEAR_AVAILABLE, reason="images/shear not found")
class TestMeshContourCalibration:
    @pytest.fixture(scope="class")
    def solved_mesh(self):
        f_img = gp.Image(filepath=os.path.join(SHEAR_DIR, "shear_0.jpg"))
        g_img = gp.Image(filepath=os.path.join(SHEAR_DIR, "shear_4.jpg"))
        boundary = gp.PathRegion(nodes=BOUNDARY_NODES.copy(), hard=False)
        local_mask = gp.Mask(mask_type="local", shape="circle", size=25)
        mesh = gp.Mesh(boundary=boundary, target_nodes=30, f_img=f_img, g_img=g_img,
                        size=(15.0, 70.0), mesh_order=1)
        mesh.solve(local_mask=local_mask, seed_coord=[500.0, 500.0], seed_warp=[0.0] * 6,
                   subset_order=1, tolerance=0.75, seed_tolerance=0.9, method="icgn")
        return mesh

    def test_calibrated_u_values_match_manual_i2o(self, solved_mesh, monkeypatch):
        cal = _half_scale_calibration_params()
        nodes = np.asarray(solved_mesh.nodes)
        disps = np.asarray(solved_mesh.displacements)
        expected_u = (cal.i2o(nodes + disps) - cal.i2o(nodes))[:, 0]

        captured = {}
        import matplotlib.axes
        original = matplotlib.axes.Axes.tricontourf

        def spy(self_ax, *args, **kwargs):
            captured["values"] = np.asarray(args[-1])
            return original(self_ax, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "tricontourf", spy)
        fig, ax = solved_mesh.contour("u", calibration=cal, show=False)
        plt.close(fig)

        assert np.allclose(captured["values"], expected_u)

    def test_uncalibrated_positions_unchanged(self, solved_mesh, monkeypatch):
        # Node positions used for the triangulation must stay in pixel space
        # even when calibration= is supplied — only the value changes.
        nodes = np.asarray(solved_mesh.nodes)
        captured = {}
        import matplotlib.tri as tri
        original = tri.Triangulation.__init__

        def spy(self_tri, x, y, *args, **kwargs):
            captured["x"] = np.asarray(x)
            captured["y"] = np.asarray(y)
            return original(self_tri, x, y, *args, **kwargs)

        monkeypatch.setattr(tri.Triangulation, "__init__", spy)
        cal = _half_scale_calibration_params()
        fig, ax = solved_mesh.contour("R", calibration=cal, show=False)
        plt.close(fig)

        assert np.allclose(captured["x"], nodes[:, 0])
        assert np.allclose(captured["y"], nodes[:, 1])

    def test_calibration_with_non_spatial_quantity_raises(self, solved_mesh):
        cal = _half_scale_calibration_params()
        with pytest.raises(ValueError):
            solved_mesh.contour("C_ZNCC", calibration=cal, show=False)


@pytest.mark.skipif(
    not CALIB_AVAILABLE,
    reason="geopyv calibration images not found (sibling ../geopyv checkout)",
)
class TestCalibrationSolveAndPlots:
    @pytest.fixture(scope="class")
    def solved_cal(self):
        cal = Calibration(
            calibration_dir=CALIB_DIR + "/",
            board_parameters=(29, 18, 10, 8),
        )
        cal.solve(ext_id=2068, binary_threshold=110)
        return cal

    def test_solved(self, solved_cal):
        assert solved_cal.solved
        assert solved_cal.params is not None
        assert len(solved_cal._accepted_images) > 0

    def test_inspect(self, solved_cal):
        fig, ax = solved_cal.inspect(image_index=0, show=False)
        plt.close(fig)

    def test_visualise(self, solved_cal):
        fig, ax = solved_cal.visualise(show=False)
        plt.close(fig)

    def test_contour(self, solved_cal):
        fig, ax = solved_cal.contour(quantity="R", show=False)
        plt.close(fig)

    def test_error(self, solved_cal):
        fig, ax = solved_cal.error(quantity="R", points=False, show=False)
        plt.close(fig)

    def test_error_invalid_quantity_raises(self, solved_cal):
        with pytest.raises(ValueError):
            solved_cal.error(quantity="bogus", show=False)

    def test_calibrate_region(self, solved_cal):
        boundary = gp.PathRegion(nodes=BOUNDARY_NODES.copy(), hard=False)
        original = np.array(boundary.current_nodes)
        solved_cal.calibrate(boundary)
        assert boundary.calibrated
        assert not np.allclose(boundary.current_nodes, original)

    def test_save_load_round_trip(self, solved_cal, tmp_path):
        path = str(tmp_path / "calibration.pyv")
        solved_cal.save(path)
        loaded = gp.load(path)

        assert isinstance(loaded, Calibration)
        assert loaded.solved
        assert np.allclose(loaded.params.intmat, solved_cal.params.intmat)
        assert np.allclose(loaded.params.extmat, solved_cal.params.extmat)
        assert np.allclose(loaded.params.dist, solved_cal.params.dist)
        assert len(loaded._accepted_images) == len(solved_cal._accepted_images)
        assert len(loaded._reimgpnts) == len(solved_cal._reimgpnts)

        # Plots still work on the reloaded object, without a fresh solve.
        fig, ax = loaded.inspect(image_index=0, show=False)
        plt.close(fig)
        fig, ax = loaded.error(quantity="R", points=False, show=False)
        plt.close(fig)
