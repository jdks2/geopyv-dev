"""
Phase 3 validation tests for the IO module.

Tests that Mesh, FieldSolution, and SequenceSolution round-trip through
geopyv_dev.save / geopyv_dev.load with bit-exact precision (Tier A, atol=1e-12).
"""
import tempfile

import numpy as np
import pytest

from geopyv_dev import Image, Mesh, Mask, save, load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(seed, shape=(200, 200)):
    """Create a synthetic greyscale image with texture suitable for DIC."""
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    img = 128.0 + 50.0 * np.sin(2 * np.pi * xx / 20) * np.cos(2 * np.pi * yy / 20)
    img += 8.0 * rng.standard_normal(shape)
    return np.clip(img, 0.0, 255.0)


def make_solved_mesh():
    """Create and solve a Mesh on synthetic images."""
    f_arr = _make_image(42)
    g_arr = _make_image(42)   # identical pattern → near-zero displacement
    f_img = Image(image_gs=f_arr)
    g_img = Image(image_gs=g_arr)
    tmpl = Mask(mask_type="local", shape="circle", size=20)
    boundary = np.array(
        [[40.0, 40.0], [160.0, 40.0], [160.0, 160.0], [40.0, 160.0]],
        dtype=np.float64,
    )
    mesh = Mesh(boundary, target_nodes=10, f_img=f_img, g_img=g_img, size=(20.0, 60.0))
    mesh.solve(tmpl, seed_coord=[100.0, 100.0], max_iterations=20)
    return mesh


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMeshRoundTrip:
    """Mesh save/load round-trips."""

    def test_nodes_exact(self, tmp_path):
        mesh = make_solved_mesh()
        path = str(tmp_path / "mesh.pyv")
        save(path, mesh)
        loaded = load(path)
        np.testing.assert_array_equal(loaded.nodes, mesh.nodes)

    def test_elements_exact(self, tmp_path):
        mesh = make_solved_mesh()
        path = str(tmp_path / "mesh.pyv")
        save(path, mesh)
        loaded = load(path)
        np.testing.assert_array_equal(loaded.elements, mesh.elements)

    def test_displacements_exact(self, tmp_path):
        mesh = make_solved_mesh()
        path = str(tmp_path / "mesh.pyv")
        save(path, mesh)
        loaded = load(path)
        np.testing.assert_allclose(loaded.displacements, mesh.displacements, atol=1e-12)

    def test_warps_exact(self, tmp_path):
        mesh = make_solved_mesh()
        path = str(tmp_path / "mesh.pyv")
        save(path, mesh)
        loaded = load(path)
        np.testing.assert_allclose(loaded.warps, mesh.warps, atol=1e-12)

    def test_c_zncc_exact(self, tmp_path):
        mesh = make_solved_mesh()
        path = str(tmp_path / "mesh.pyv")
        save(path, mesh)
        loaded = load(path)
        np.testing.assert_allclose(loaded.c_zncc, mesh.c_zncc, atol=1e-12)

    def test_areas_exact(self, tmp_path):
        mesh = make_solved_mesh()
        path = str(tmp_path / "mesh.pyv")
        save(path, mesh)
        loaded = load(path)
        np.testing.assert_allclose(loaded.areas, mesh.areas, atol=1e-12)

    def test_boundary_exact(self, tmp_path):
        mesh = make_solved_mesh()
        path = str(tmp_path / "mesh.pyv")
        save(path, mesh)
        loaded = load(path)
        np.testing.assert_array_equal(loaded.boundary, mesh.boundary)

    def test_metadata_exact(self, tmp_path):
        mesh = make_solved_mesh()
        path = str(tmp_path / "mesh.pyv")
        save(path, mesh)
        loaded = load(path)
        assert loaded.seed_node == mesh.seed_node
        assert loaded.mesh_order == mesh.mesh_order
        assert loaded.subset_order == mesh.subset_order

    def test_loaded_type_is_mesh(self, tmp_path):
        mesh = make_solved_mesh()
        path = str(tmp_path / "mesh.pyv")
        save(path, mesh)
        loaded = load(path)
        assert isinstance(loaded, Mesh)

    def test_save_unsolved_raises(self, tmp_path):
        f_arr = _make_image(1)
        g_arr = _make_image(1)
        f_img = Image(image_gs=f_arr)
        g_img = Image(image_gs=g_arr)
        boundary = np.array(
            [[40.0, 40.0], [160.0, 40.0], [160.0, 160.0], [40.0, 160.0]],
            dtype=np.float64,
        )
        mesh = Mesh(boundary, target_nodes=10, f_img=f_img, g_img=g_img, size=(20.0, 60.0))
        path = str(tmp_path / "unsolved.pyv")
        with pytest.raises(RuntimeError, match="[Ss]olve"):
            save(path, mesh)


class TestErrorCases:
    """Error handling in save/load."""

    def test_load_nonexistent_file(self, tmp_path):
        with pytest.raises(RuntimeError, match="file not found|No such file"):
            load(str(tmp_path / "nonexistent.pyv"))

    def test_load_wrong_magic(self, tmp_path):
        path = str(tmp_path / "bad.pyv")
        with open(path, "wb") as f:
            f.write(b"XXXX\x01" + b"\x00" * 8)
        with pytest.raises(RuntimeError, match="magic"):
            load(path)

    def test_load_wrong_version(self, tmp_path):
        path = str(tmp_path / "badver.pyv")
        # 0x7f is unrecognised (supported: 0x01-0x03, 0x07 and the current 0x08).
        with open(path, "wb") as f:
            f.write(b"GPYV\x7f" + b"\x00" * 8)
        with pytest.raises(RuntimeError, match="version"):
            load(path)

    def test_save_wrong_type(self, tmp_path):
        path = str(tmp_path / "bad.pyv")
        with pytest.raises((RuntimeError, TypeError)):
            save(path, {"not": "a solution"})
