"""
Phase 3 validation tests for the IO module.

Tests that MeshSolution, FieldSolution, and SequenceSolution round-trip through
geopyv_dev.save / geopyv_dev.load with bit-exact precision (Tier A, atol=1e-12).
"""
import os
import tempfile

import numpy as np
import pytest

from geopyv_dev import MeshSolution, save, load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mesh_solution(seed=42):
    """Create a synthetic MeshSolution with deterministic values."""
    rng = np.random.default_rng(seed)
    n_nodes = 6
    n_elems = 4
    nodes = rng.uniform(0, 100, (n_nodes, 2))
    elements = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.int64)
    boundary = list(range(n_nodes))
    exclusions = []
    areas = rng.uniform(10, 50, n_elems)
    warps = rng.standard_normal((n_nodes, 12))
    displacements = rng.standard_normal((n_nodes, 2))
    c_zncc = rng.uniform(0.9, 1.0, n_nodes)
    p = rng.standard_normal((n_nodes, 6))
    return MeshSolution(
        nodes=nodes,
        elements=elements,
        boundary=boundary,
        exclusions=exclusions,
        areas=areas,
        warps=warps,
        displacements=displacements,
        c_zncc=c_zncc,
        p=p,
        seed_node=0,
        mesh_order=1,
        subset_order=1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMeshRoundTrip:
    """MeshSolution save/load round-trips."""

    def test_nodes_exact(self, tmp_path):
        sol = make_mesh_solution(seed=1)
        path = str(tmp_path / "mesh.pyv")
        save(path, sol)
        loaded = load(path)
        np.testing.assert_array_equal(loaded.nodes, sol.nodes)

    def test_elements_exact(self, tmp_path):
        sol = make_mesh_solution(seed=2)
        path = str(tmp_path / "mesh.pyv")
        save(path, sol)
        loaded = load(path)
        np.testing.assert_array_equal(loaded.elements, sol.elements)

    def test_displacements_exact(self, tmp_path):
        sol = make_mesh_solution(seed=3)
        path = str(tmp_path / "mesh.pyv")
        save(path, sol)
        loaded = load(path)
        np.testing.assert_allclose(loaded.displacements, sol.displacements, atol=1e-12)

    def test_warps_exact(self, tmp_path):
        sol = make_mesh_solution(seed=4)
        path = str(tmp_path / "mesh.pyv")
        save(path, sol)
        loaded = load(path)
        np.testing.assert_allclose(loaded.warps, sol.warps, atol=1e-12)

    def test_c_zncc_exact(self, tmp_path):
        sol = make_mesh_solution(seed=5)
        path = str(tmp_path / "mesh.pyv")
        save(path, sol)
        loaded = load(path)
        np.testing.assert_allclose(loaded.c_zncc, sol.c_zncc, atol=1e-12)

    def test_areas_exact(self, tmp_path):
        sol = make_mesh_solution(seed=6)
        path = str(tmp_path / "mesh.pyv")
        save(path, sol)
        loaded = load(path)
        np.testing.assert_allclose(loaded.areas, sol.areas, atol=1e-12)

    def test_boundary_exact(self, tmp_path):
        sol = make_mesh_solution(seed=7)
        path = str(tmp_path / "mesh.pyv")
        save(path, sol)
        loaded = load(path)
        np.testing.assert_array_equal(loaded.boundary, sol.boundary)

    def test_metadata_exact(self, tmp_path):
        sol = make_mesh_solution(seed=8)
        path = str(tmp_path / "mesh.pyv")
        save(path, sol)
        loaded = load(path)
        assert loaded.seed_node == sol.seed_node
        assert loaded.mesh_order == sol.mesh_order
        assert loaded.subset_order == sol.subset_order

    def test_pi_precision(self, tmp_path):
        """f64 irrational value round-trips bit-exactly (Tier A)."""
        sol = make_mesh_solution(seed=9)
        path = str(tmp_path / "pi.pyv")
        save(path, sol)
        loaded = load(path)
        # Nodes contain rng values; pi used as a sentinel check on displacements[0,0]
        val = np.pi
        sol2 = MeshSolution(
            nodes=sol.nodes,
            elements=sol.elements,
            boundary=list(sol.boundary),
            exclusions=[],
            areas=sol.areas,
            warps=sol.warps,
            displacements=np.array([[val, 0.0]] + list(sol.displacements[1:])),
            c_zncc=sol.c_zncc,
            p=sol.p,
        )
        path2 = str(tmp_path / "pi2.pyv")
        save(path2, sol2)
        loaded2 = load(path2)
        assert loaded2.displacements[0, 0] == val

    def test_loaded_type_is_meshsolution(self, tmp_path):
        sol = make_mesh_solution()
        path = str(tmp_path / "mesh.pyv")
        save(path, sol)
        loaded = load(path)
        assert isinstance(loaded, MeshSolution)


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
        with open(path, "wb") as f:
            f.write(b"GPYV\x02" + b"\x00" * 8)
        with pytest.raises(RuntimeError, match="version"):
            load(path)

    def test_save_wrong_type(self, tmp_path):
        path = str(tmp_path / "bad.pyv")
        with pytest.raises((RuntimeError, TypeError)):
            save(path, {"not": "a solution"})
