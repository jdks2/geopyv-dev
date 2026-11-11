"""
Tests for the Validation.solve() pipeline.
"""
import numpy as np
import pytest

import geopyv_dev as gp
import geopyv_dev._geopyv_dev as _core


def _make_speckle(image_no=4, u=1.0):
    comp = [0.0] * 12
    comp[0] = u
    return _core.Speckle(
        image_cfg={"image_dir": "/tmp", "name": "test_vs", "image_size": (200, 200)},
        speckle_cfg={"speckle_size": 1.0, "speckle_number": 10},
        progression="deformation",
        deformation_cfg={"comp": comp},
        noise_cfg=(0.0, 0.0),
        scale_cfg={"scale": "lin", "n": image_no},
    )


class TestValidationPyWrapper:

    def test_validation_wrapper_exists(self):
        assert hasattr(gp, "Validation")

    def test_core_validation_exists(self):
        assert hasattr(_core, "Validation")

    def test_core_validation_solution_exists(self):
        assert hasattr(_core, "ValidationSolution")

    def test_core_validation_field_data_exists(self):
        assert hasattr(_core, "ValidationFieldData")


class TestSpeckleWarp:

    def test_translation_warp(self):
        sp = _make_speckle(image_no=3, u=2.0)
        c = np.array([[10.0, 20.0], [150.0, 30.0]])
        w = sp.warp(2, c)
        assert w.shape == (2, 12)
        np.testing.assert_allclose(w[:, 0], 2.0, atol=1e-12)
        np.testing.assert_allclose(w[:, 1:], 0.0, atol=1e-12)

    def test_reference_image_is_zero(self):
        sp = _make_speckle(image_no=3, u=2.0)
        w = sp.warp(0, np.array([[10.0, 20.0]]))
        np.testing.assert_allclose(w, 0.0, atol=1e-12)

    def test_out_of_range_image(self):
        sp = _make_speckle(image_no=3)
        with pytest.raises(ValueError):
            sp.warp(3, np.array([[10.0, 20.0]]))
