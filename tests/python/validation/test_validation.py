"""
Phase 3 validation tests for the Validation module — anomaly removal.

These tests assert the same golden values verified in geopyv/tests/validation/fixtures.py,
but call geopyv_dev.anomalies instead of the Python reference implementation.
Tolerance: Tier A (atol=1e-12) — pure array manipulation with no floating-point
accumulation.
"""
import numpy as np
import pytest

from geopyv_dev import anomalies


class TestAnomalies:

    def test_skim_zero_passthrough(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((3, 10, 12))
        o = rng.standard_normal((3, 10, 12))
        ra, ro = anomalies(a, o, 0)
        np.testing.assert_allclose(ra, a, atol=1e-12)
        np.testing.assert_allclose(ro, o, atol=1e-12)

    def test_removes_single_outlier(self):
        a = np.zeros((1, 5, 12))
        o = np.zeros((1, 5, 12))
        a[0, 3, 0] = 10.0
        ra, ro = anomalies(a, o, 1)
        assert ra.shape == (1, 4, 12)
        np.testing.assert_allclose(ra, np.zeros((1, 4, 12)), atol=1e-12)

    def test_removes_top_two(self):
        a = np.zeros((1, 5, 12))
        o = np.zeros((1, 5, 12))
        a[0, 1, 0] = 5.0
        a[0, 4, 1] = 8.0
        ra, ro = anomalies(a, o, 2)
        assert ra.shape == (1, 3, 12)
        np.testing.assert_allclose(ra, np.zeros((1, 3, 12)), atol=1e-12)

    def test_multiframe_independent(self):
        a = np.zeros((2, 4, 12))
        o = np.zeros((2, 4, 12))
        a[0, 2, 0] = 9.0
        a[1, 0, 0] = 9.0
        ra, ro = anomalies(a, o, 1)
        assert ra.shape == (2, 3, 12)
        np.testing.assert_allclose(ra, np.zeros((2, 3, 12)), atol=1e-12)

    def test_values_preserved_exact(self):
        a = np.zeros((1, 3, 12))
        o = np.zeros((1, 3, 12))
        a[0, 0, 0] = 1.23456789012345
        o[0, 0, 1] = 9.87654321098765
        a[0, 2, 0] = 100.0
        ra, ro = anomalies(a, o, 1)
        assert ra.shape == (1, 2, 12)
        np.testing.assert_allclose(ra[0, 0, 0], 1.23456789012345, atol=1e-12)
        np.testing.assert_allclose(ro[0, 0, 1], 9.87654321098765, atol=1e-12)

    def test_output_shape(self):
        a = np.random.randn(5, 20, 12)
        o = np.random.randn(5, 20, 12)
        for skim in [0, 1, 5, 10, 19]:
            ra, ro = anomalies(a, o, skim)
            assert ra.shape == (5, 20 - skim, 12)
            assert ro.shape == (5, 20 - skim, 12)

    def test_golden_values(self):
        """Same golden values as fixtures.py — Rust and Python agree exactly."""
        rng = np.random.default_rng(7)
        n_frames, n_particles = 2, 8
        a = rng.standard_normal((n_frames, n_particles, 12))
        o = rng.standard_normal((n_frames, n_particles, 12))

        ra, ro = anomalies(a, o, 2)

        # Verify against Python reference.
        for frame in range(n_frames):
            errors = np.sqrt(np.sum((a[frame, :, :2] - o[frame, :, :2]) ** 2, axis=1))
            sorted_idx = np.argsort(errors)[::-1]
            removed = set(sorted_idx[:2])
            kept = [j for j in range(n_particles) if j not in removed]
            np.testing.assert_allclose(ra[frame], a[frame, kept], atol=1e-12)
            np.testing.assert_allclose(ro[frame], o[frame, kept], atol=1e-12)

    def test_skim_too_large_raises(self):
        a = np.zeros((1, 3, 12))
        o = np.zeros((1, 3, 12))
        with pytest.raises(RuntimeError):
            anomalies(a, o, 3)

    def test_shape_mismatch_raises(self):
        a = np.zeros((1, 3, 12))
        o = np.zeros((1, 4, 12))
        with pytest.raises(RuntimeError):
            anomalies(a, o, 1)

    def test_contiguous_input(self):
        """Non-contiguous slices (e.g. every-other column) should still work."""
        a = np.zeros((1, 4, 12))
        o = np.zeros((1, 4, 12))
        a[0, 3, 0] = 5.0
        # Pass C-contiguous copies.
        ra, ro = anomalies(np.ascontiguousarray(a), np.ascontiguousarray(o), 1)
        assert ra.shape == (1, 3, 12)
