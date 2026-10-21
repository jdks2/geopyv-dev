"""Adaptive masking (solver_options={"masking": "adaptive"}) on Mesh and Sequence.

Synthetic images through gp.Speckle: a rigid translation over three frames, so
frames 0 -> 1 form a still pair (rigid motion is allowed) and nothing should
need correcting. These are pipeline tests; accuracy against a true field is
checked in the sandbox (geopyv_dev_pub/docs/screening_summary.pdf).
"""

import numpy as np
import pytest

import geopyv_dev as gp
import geopyv_dev._geopyv_dev as _core

_BOUNDARY = np.array([[60.0, 60.0], [240.0, 60.0], [240.0, 240.0], [60.0, 240.0]])
_SEED = [150.0, 150.0]


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    d = tmp_path_factory.mktemp("adaptive")
    comp = [0.0] * 12
    comp[0] = 1.0
    sp = _core.Speckle(
        image_cfg={"image_dir": str(d), "name": "frame", "image_size": (300, 300), "file_format": ".png"},
        speckle_cfg={"speckle_size": 3.0, "speckle_number": 4000},
        progression="deformation",
        deformation_cfg={"comp": comp},
        noise_cfg=(0.0, 0.01),            # intensity noise, fraction of full scale (~2 grey levels)
        scale_cfg={"scale": "lin", "n": 3},
    )
    sp.solve(seed=1)
    paths = sorted(str(p) for p in d.glob("frame_*.png"))
    assert len(paths) == 3
    return d, paths


def _mesh(f, g, noise=None):
    return gp.Mesh(_BOUNDARY, 40, gp.Image(filepath=f), gp.Image(filepath=g), time=(0, 1), noise=noise)


def _solve(m, masking="adaptive", **kw):
    m.solve(gp.Mask(mask_type="local", shape="circle", size=15), _SEED, tolerance=0.5,
            solver_options={"masking": masking}, **kw)


def test_mesh_defaults_to_three_node_elements(frames):
    _, p = frames
    assert np.asarray(_mesh(p[0], p[1]).elements).shape[1] == 3


def test_uniform_solve_has_no_record(frames):
    _, p = frames
    m = _mesh(p[0], p[2])
    _solve(m, "uniform")
    assert m.seams is None


def test_adaptive_with_still_pair(frames):
    _, p = frames
    still = _mesh(p[0], p[1])
    _solve(still, "uniform")
    m = _mesh(p[0], p[2], noise=still)
    _solve(m)
    s = m.seams
    n = len(m.nodes)
    assert s["noise_source"] == "still pair"
    assert np.asarray(m.p).shape == (n, 12)
    assert np.asarray(s["sent"]).shape == (n,) and np.asarray(s["still_mismatch"]).shape == (n,)
    assert np.all(np.isfinite(np.asarray(m.displacements)))


def test_adaptive_forces_subset_order_one(frames):
    _, p = frames
    m = _mesh(p[0], p[2])
    with pytest.warns(UserWarning, match="subset_order=1"):
        _solve(m, subset_order=2)
    assert m.seams is not None


def test_masking_rejects_unknown_values(frames):
    _, p = frames
    m = _mesh(p[0], p[2])
    for bad in ("zonal", "bogus"):
        with pytest.raises(ValueError):
            _solve(m, bad)


def test_seed_patch_noise_without_still_pair(frames):
    _, p = frames
    m = _mesh(p[0], p[2])
    _solve(m)
    src = m.seams["noise_source"]
    assert src.startswith("seed patch")
    sigma = float(src.split("(")[1].split()[0])
    assert 1.0 < sigma < 3.5, src          # images carry ~2 grey levels


def test_subset_adaptive_and_order_two_rebuilt(frames):
    _, p = frames
    f, g = gp.Image(filepath=p[0]), gp.Image(filepath=p[2])
    m = gp.Mask(mask_type="local", shape="circle", size=15)
    s = gp.Subset([150.0, 150.0], m, f, g, time=(0, 1))
    s.solve(max_norm=1e-5, tolerance=0.0)
    assert s.seams is None
    s = gp.Subset([150.0, 150.0], m, f, g, subset_order=2, time=(0, 1))
    with pytest.warns(UserWarning, match="subset_order=1"):
        s.solve(max_norm=1e-5, tolerance=0.0, solver_options={"masking": "adaptive"})
    assert s.seams is not None
    with pytest.raises(ValueError):
        s.solve(solver_options={"preconditioning": "RG"})


def test_sequence_uses_frames_zero_one_for_noise(frames):
    d, _ = frames
    seq = gp.Sequence(str(d), _BOUNDARY, 40, time=1.0)
    seq.solve(local_mask=gp.Mask(mask_type="local", shape="circle", size=15), seed_coord=_SEED,
              tolerance=0.5, solver_options={"masking": "adaptive"})
    ms = seq.mesh_solutions
    assert len(ms) == 2
    assert ms[0].seams is None                      # the still pair: noise only
    assert ms[1].seams["noise_source"] == "still pair"
