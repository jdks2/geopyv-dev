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
