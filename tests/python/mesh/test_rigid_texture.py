"""Rigid speckle texture (solver_options={"texture": "rigid", "grain": ...}) on
Subset, Mesh and Sequence.

Synthetic images through gp.Speckle (round speckles that move rigidly with the
material under their centres) under a uniform simple shear. These are pipeline
tests plus one "it helps" check; accuracy against the true field is measured in
the sandbox trial (geopyv_dev_pub/sandbox_break_test/rigid_texture_trial/).
"""

import os

import numpy as np
import pytest

import geopyv_dev as gp
import geopyv_dev._geopyv_dev as _core

_BOUNDARY = np.array([[90.0, 90.0], [210.0, 90.0], [210.0, 210.0], [90.0, 210.0]])
_SEED = [150.0, 150.0]
_SHEAR = 0.05


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    d = tmp_path_factory.mktemp("rigid_texture")
    comp = [0.0] * 12
    comp[4] = _SHEAR
    sp = _core.Speckle(
        image_cfg={"image_dir": str(d), "name": "frame", "image_size": (301, 301), "file_format": ".png",
                   "origin": (150.0, 150.0)},
        speckle_cfg={"speckle_size": 5.0, "speckle_number": 3000},
        progression="deformation",
        deformation_cfg={"comp": comp},
        noise_cfg=(0.0, 0.0),
        scale_cfg={"scale": "lin", "n": 3},
    )
    sp.solve(seed=3)
    paths = sorted(str(p) for p in d.glob("frame_*.png"))
    assert len(paths) == 3
    return d, paths


def _mask(size=15):
    return gp.Mask(mask_type="local", shape="circle", size=size)


def _subset(paths, texture, order=1):
    s = gp.Subset([150.3, 150.6], _mask(), gp.Image(filepath=paths[0]), gp.Image(filepath=paths[-1]),
                  subset_order=order)
    p0 = [0.0] * (6 * order)
    opts = {"texture": "deformable"} if texture == "deformable" else {"texture": "rigid", "grain": "sphere"}
    s.solve(p_0=p0, max_norm=1e-6, max_iterations=100, solver_options=opts)
    return s


def test_subset_rigid_fits_better_and_records(frames, tmp_path):
    _, p = frames
    d, r = _subset(p, "deformable"), _subset(p, "rigid")
    assert d.texture == ("deformable", None) and r.texture == ("rigid", "sphere")
    assert r.converged
    assert r.c_znssd < d.c_znssd
    path = str(tmp_path / "s.pyv")
    r.save(path)
    assert gp.load(path).texture == ("rigid", "sphere")


def test_subset_order0_records_deformable(frames):
    _, p = frames
    s = gp.Subset([150.0, 150.0], _mask(), gp.Image(filepath=p[0]), gp.Image(filepath=p[0]),
                  subset_order=0)
    s.solve(solver_options={"texture": "rigid", "grain": "sphere"})
    assert s.converged and s.texture == ("deformable", None)


def test_subset_rejects_bad_texture_and_unknown_keys(frames):
    _, p = frames
    s = gp.Subset([150.0, 150.0], _mask(), gp.Image(filepath=p[0]), gp.Image(filepath=p[1]))
    with pytest.raises(ValueError, match="texture"):
        s.solve(solver_options={"texture": "painted"})
    with pytest.raises(ValueError, match="grain"):
        s.solve(solver_options={"texture": "rigid", "grain": "cube"})
    with pytest.raises(ValueError, match="grain"):
        s.solve(solver_options={"texture": "deformable", "grain": "shape"})
    with pytest.raises(ValueError, match="accepts only"):
        s.solve(solver_options={"preconditioning": "RG"})


def _mesh(paths):
    return gp.Mesh(_BOUNDARY, 40, gp.Image(filepath=paths[0]), gp.Image(filepath=paths[-1]), time=(0, 1))


def test_mesh_rigid_runs_and_round_trips(frames, tmp_path):
    _, p = frames
    m = _mesh(p)
    m.solve(_mask(), _SEED, tolerance=0.5, solver_options={"texture": "rigid", "grain": "sphere"})
    assert m.texture == ("rigid", "sphere")
    assert np.all(np.isfinite(np.asarray(m.displacements)))
    path = str(tmp_path / "m.pyv")
    m.save(path)
    assert gp.load(path).texture == ("rigid", "sphere")


def test_mesh_default_is_deformable(frames):
    _, p = frames
    m = _mesh(p)
    m.solve(_mask(), _SEED, tolerance=0.5)
    assert m.texture == ("deformable", None)


def test_mesh_rejects_bad_texture(frames):
    _, p = frames
    with pytest.raises(ValueError, match="texture"):
        _mesh(p).solve(_mask(), _SEED, solver_options={"texture": "painted"})


def test_sequence_rigid_reaches_every_mesh(frames):
    d, _ = frames
    seq = gp.Sequence(image_dir=str(d), boundary=_BOUNDARY, target_nodes=30, size=(10.0, 100.0), mesh_order=1)
    seq.solve(local_mask=_mask(), seed_coord=_SEED, subset_order=1, tolerance=0.5,
              solver_options={"texture": "rigid", "grain": "sphere"})
    assert seq.solved
    assert all(m.texture == ("rigid", "sphere") for m in seq.mesh_solutions)


def test_sequence_rejects_bad_texture(frames):
    d, _ = frames
    seq = gp.Sequence(image_dir=str(d), boundary=_BOUNDARY, target_nodes=30, size=(10.0, 100.0), mesh_order=1)
    with pytest.raises(ValueError, match="texture"):
        seq.solve(local_mask=_mask(), seed_coord=_SEED, solver_options={"texture": "painted"})


# --- texture="rigid", grain="shape" (default): grains of any shape ---------------------

@pytest.fixture(scope="module")
def opaque_ellipses(tmp_path_factory):
    d = tmp_path_factory.mktemp("opaque_ellipses")
    comp = [0.0] * 12
    comp[4] = _SHEAR
    sp = _core.Speckle(
        image_cfg={"image_dir": str(d), "name": "frame", "image_size": (301, 301), "file_format": ".png"},
        speckle_cfg={"speckle_size": 10.0, "speckle_number": 700, "speckle_limit": True,
                     "shape": "ellipse", "aspect": 2.0, "overlap": "opaque"},
        progression="deformation", deformation_cfg={"comp": comp, "origin": (150.0, 150.0)},
        noise_cfg=(0.0, 0.0), scale_cfg={"scale": "lin", "n": 2})
    sp.solve(seed=5)
    return sorted(str(p) for p in d.glob("frame_*.png"))


def test_subset_grains_fits_better_and_records(opaque_ellipses, tmp_path):
    p = opaque_ellipses
    out = {}
    for texture in ("deformable", "rigid"):
        s = gp.Subset([150.3, 150.6], _mask(25), gp.Image(filepath=p[0]), gp.Image(filepath=p[1]))
        s.solve(max_norm=1e-6, max_iterations=100, solver_options={"texture": texture})
        assert s.converged
        out[texture] = s
    assert out["rigid"].texture == ("rigid", "shape")      # "shape" is the default
    assert out["rigid"].c_znssd < out["deformable"].c_znssd
    path = str(tmp_path / "g.pyv")
    out["rigid"].save(path)
    assert gp.load(path).texture == ("rigid", "shape")


def test_mesh_grains_runs(opaque_ellipses):
    p = opaque_ellipses
    m = gp.Mesh(_BOUNDARY, 40, gp.Image(filepath=p[0]), gp.Image(filepath=p[-1]), time=(0, 1))
    m.solve(_mask(), _SEED, tolerance=0.5, solver_options={"texture": "rigid", "grain": "shape"})
    assert m.texture == ("rigid", "shape")
    assert np.all(np.isfinite(np.asarray(m.displacements)))


# --- gp.Speckle soil images -------------------------------------------------------------

@pytest.mark.parametrize("soil", ["clay", "sand"])
def test_speckle_soil_images(soil, tmp_path):
    from PIL import Image
    comp = [0.0] * 12
    comp[4] = 0.02
    sp = _core.Speckle(
        image_cfg={"image_dir": str(tmp_path), "name": soil, "image_size": (160, 140), "file_format": ".png"},
        speckle_cfg={"soil": soil, "matrix_brightness": 180, "seed_brightness": 120, "asr": 0.2,
                     "grains": "distribution", "psd": np.array([[200.0, 10.0], [310.0, 50.0], [500.0, 90.0]]), "d_p": 6,
                     "aspect_ratio": 0.74, "aspect_ratio_sd": 0.11},
        progression="deformation", deformation_cfg={"comp": comp}, noise_cfg=(0.0, 0.01),
        scale_cfg={"scale": "lin", "n": 2})
    sp.solve(seed=2)
    a = np.asarray(Image.open(tmp_path / f"{soil}_0.png"), float)
    assert a.shape == (140, 160) and a.std() > 5
    path = str(tmp_path / "s.pyv")
    gp.save(path, sp)
    assert gp.load(path) is not None


@pytest.mark.parametrize("soil", ["clay", "sand"])
def test_speckle_soil_grains(soil, tmp_path):
    """grains(i): table and pixel map agree with the image, move with the material, and survive save/load."""
    from PIL import Image
    comp = [0.0] * 12
    comp[0] = 3.0
    sp = _core.Speckle(
        image_cfg={"image_dir": str(tmp_path), "name": soil, "image_size": (160, 140), "file_format": ".png"},
        speckle_cfg={"soil": soil, "matrix_brightness": 180, "matrix_spread": 0.0, "seed_brightness": 90,
                     "seed_spread": 0.0, "asr": 0.2, "aspect_ratio": 0.5},
        progression="deformation", deformation_cfg={"comp": comp}, noise_cfg=(0.0, 0.0),
        scale_cfg={"scale": "lin", "n": 2})
    with pytest.raises(Exception, match="solve"):
        sp.grains(0)
    sp.solve(seed=2)
    table, owner = sp.grains(0)
    assert owner.shape == (140, 160) and owner.min() >= -1 and owner.max() <= len(table["x"])  # deep sand grains may be hidden
    assert np.allclose(table["aspect_ratio"], 0.5, atol=0.01) and np.allclose(table["size"], 6.0, atol=0.05)
    img = np.asarray(Image.open(tmp_path / f"{soil}_0.png"), float)
    dyed = owner > 0
    dyed[dyed] = table["dyed"][owner[dyed] - 1]
    assert img[dyed].mean() < 120 and img[owner == 0].mean() > 150 if soil == "clay" else img[dyed].mean() < 120
    t1, o1 = sp.grains(1)                                  # moved 3 px in x
    assert np.allclose(t1["x"] - table["x"], 3.0) and np.array_equal(o1[:, 3:], owner[:, :-3])
    path = str(tmp_path / "g.pyv")
    gp.save(path, sp)
    t2, o2 = gp.load(path).grains(0)
    assert np.array_equal(o2, owner) and np.allclose(t2["size"], table["size"])


def test_speckle_soil_rejects_bad_input(tmp_path):
    base = dict(image_cfg={"image_dir": str(tmp_path), "name": "x", "image_size": (64, 64)},
                progression="deformation", deformation_cfg={"comp": [0.0] * 12}, noise_cfg=(0.0, 0.0),
                scale_cfg={"scale": "lin", "n": 2})
    with pytest.raises(ValueError, match="soil"):
        _core.Speckle(speckle_cfg={"soil": "gravel", "matrix_brightness": 1, "seed_brightness": 1, "asr": 0.1}, **base)
    with pytest.raises(ValueError, match="does not apply"):
        _core.Speckle(speckle_cfg={"soil": "clay", "matrix_brightness": 180, "seed_brightness": 120, "asr": 0.2,
                                   "shape": "ellipse"}, **base)
    with pytest.raises(ValueError, match="does not apply"):
        _core.Speckle(speckle_cfg={"soil": "sand", "matrix_brightness": 180, "seed_brightness": 120, "asr": 0.2,
                                   "grain_size": 6}, **base)
    with pytest.raises(ValueError, match="psd"):
        _core.Speckle(speckle_cfg={"soil": "sand", "matrix_brightness": 180, "seed_brightness": 120, "asr": 0.2,
                                   "grains": "distribution", "psd": [[300.0, 50.0], [200.0, 90.0]]}, **base)
    with pytest.raises(ValueError, match="distribution"):
        _core.Speckle(speckle_cfg={"soil": "sand", "matrix_brightness": 180, "seed_brightness": 120, "asr": 0.2,
                                   "psd": [[200.0, 10.0], [300.0, 90.0]]}, **base)
    with pytest.raises(ValueError, match="needs psd"):
        _core.Speckle(speckle_cfg={"soil": "sand", "matrix_brightness": 180, "seed_brightness": 120, "asr": 0.2,
                                   "grains": "distribution"}, **base)
    with pytest.raises(ValueError, match="pixel"):
        _core.Speckle(speckle_cfg={"soil": "sand", "matrix_brightness": 180, "seed_brightness": 120, "asr": 0.2,
                                   "d_p": 0.5}, **base)
    with pytest.raises(ValueError, match="asr"):
        _core.Speckle(speckle_cfg={"soil": "clay", "matrix_brightness": 180, "seed_brightness": 120,
                                   "asr": 0.9}, **base)
    with pytest.raises(ValueError, match="asr"):
        _core.Speckle(speckle_cfg={"soil": "sand", "matrix_brightness": 180, "seed_brightness": 120,
                                   "asr": 1.5}, **base)


# --- Subset soil_properties ----------------------------------------------------------

def test_subset_soil_properties(tmp_path):
    """soil_properties round-trips, takes a Speckle speckle_cfg as it is (image keys ignored), and changes the
    sphere correction through the grain size."""
    comp = [0.0] * 12
    comp[5] = -0.05
    cfg = {"soil": "clay", "matrix_brightness": 180, "seed_brightness": 100, "asr": 0.15, "d_p": 6,
           "aspect_ratio": 0.5, "blur": 1.0, "pattern_size": 8}
    sp = _core.Speckle(image_cfg={"image_dir": str(tmp_path), "name": "c", "image_size": (161, 161),
                                  "file_format": ".png"},
                       speckle_cfg=cfg, progression="deformation", deformation_cfg={"comp": comp},
                       noise_cfg=(0.0, 0.0), scale_cfg={"scale": "lin", "n": 2})
    sp.solve(seed=3)
    f, g = gp.Image(filepath=str(tmp_path / "c_0.png")), gp.Image(filepath=str(tmp_path / "c_1.png"))
    mask = gp.Mask(mask_type="local", shape="circle", size=30)
    truth = np.asarray(sp.warp(1, np.array([[80.0, 80.0]])))[0]

    def solve(props):
        s = gp.Subset([80.0, 80.0], mask, f, g, soil_properties=props)
        s.solve(p_0=[truth[0], truth[1], 0, 0, 0, 0], solver_options={"texture": "rigid", "grain": "sphere"})
        return s

    plain, known = solve(None), solve(cfg)
    assert plain.soil_properties is None
    sp_back = known.soil_properties
    assert sp_back["soil"] == "clay" and sp_back["aspect_ratio"] == 0.5 and "blur" not in sp_back
    big = solve({**cfg, "d_p": 12})
    assert not np.allclose(np.asarray(known.p), np.asarray(big.p))
    with pytest.raises(ValueError, match="soil"):
        gp.Subset([80.0, 80.0], mask, f, g, soil_properties={"matrix_brightness": 1})
    with pytest.raises(ValueError, match="asr"):
        gp.Subset([80.0, 80.0], mask, f, g, soil_properties={**cfg, "asr": 2.0})


def test_edge_texture_needs_soil_properties(tmp_path):
    """grain='edge': needs the subset's soil_properties and reports ('rigid', 'edge')."""
    comp = [0.0] * 12
    comp[5] = -0.05
    cfg = {"soil": "clay", "matrix_brightness": 180, "seed_brightness": 100, "asr": 0.15, "d_p": 6, "aspect_ratio": 0.5}
    sp = _core.Speckle(image_cfg={"image_dir": str(tmp_path), "name": "c", "image_size": (161, 161),
                                  "file_format": ".png"},
                       speckle_cfg=cfg, progression="deformation", deformation_cfg={"comp": comp},
                       noise_cfg=(0.0, 0.0), scale_cfg={"scale": "lin", "n": 2})
    sp.solve(seed=3)
    f, g = gp.Image(filepath=str(tmp_path / "c_0.png")), gp.Image(filepath=str(tmp_path / "c_1.png"))
    mask = gp.Mask(mask_type="local", shape="circle", size=30)
    opts = {"texture": "rigid", "grain": "edge"}
    with pytest.raises(ValueError, match="soil_properties"):
        gp.Subset([80.0, 80.0], mask, f, g).solve(solver_options=opts)
    s = gp.Subset([80.0, 80.0], mask, f, g, soil_properties=cfg)
    truth = np.asarray(sp.warp(1, np.array([[80.0, 80.0]])))[0]
    s.solve(p_0=[truth[0], truth[1], 0, 0, 0, 0], solver_options=opts)
    assert s.texture == ("rigid", "edge") and s.converged


def test_mesh_sequence_soil_properties(tmp_path):
    """Mesh and Sequence take soil_properties as Subset does, hand them to every subset (the sphere correction
    then matches a Subset given the same properties), and accept grain='edge' only with them."""
    comp = [0.0] * 12
    comp[5] = -0.05
    cfg = {"soil": "clay", "matrix_brightness": 180, "seed_brightness": 100, "asr": 0.15, "d_p": 6,
           "aspect_ratio": 0.5, "blur": 1.0}
    sp = _core.Speckle(image_cfg={"image_dir": str(tmp_path), "name": "c", "image_size": (201, 201),
                                  "file_format": ".png"},
                       speckle_cfg=cfg, progression="deformation", deformation_cfg={"comp": comp},
                       noise_cfg=(0.0, 0.0), scale_cfg={"scale": "lin", "n": 2})
    sp.solve(seed=3)
    f, g = gp.Image(filepath=str(tmp_path / "c_0.png")), gp.Image(filepath=str(tmp_path / "c_1.png"))
    boundary = np.array([[50.0, 50.0], [150.0, 50.0], [150.0, 150.0], [50.0, 150.0]])
    mask = gp.Mask(mask_type="local", shape="circle", size=20)
    opts = {"texture": "rigid", "grain": "sphere"}

    def mesh(props):
        m = gp.Mesh(gp.PathRegion(nodes=boundary, hard=False), 30, f, g, time=(0, 1), soil_properties=props)
        m.solve(local_mask=mask, seed_coord=[100.0, 100.0], max_norm=1e-6, tolerance=0.5, solver_options=opts)
        return m

    plain, known = mesh(None), mesh(cfg)
    assert plain.soil_properties is None and known.soil_properties["d_p"] == 6 and "blur" not in known.soil_properties
    assert not np.allclose(np.asarray(plain.displacements), np.asarray(known.displacements))
    # the seed node solved as a Subset with the same properties from the mesh's own result agrees
    nodes = np.asarray(known.nodes)
    k = int(np.argmin(np.hypot(*(nodes - [100.0, 100.0]).T)))
    s = gp.Subset(list(nodes[k]), mask, f, g, time=(0, 1), soil_properties=cfg)
    s.solve(p_0=list(np.asarray(known.warps)[k][:6]), max_norm=1e-6, solver_options=opts)
    assert np.allclose(np.asarray(s.p)[:2], np.asarray(known.displacements)[k], atol=1e-4)
    with pytest.raises(ValueError, match="soil_properties"):
        plain.solve(local_mask=mask, seed_coord=[100.0, 100.0], solver_options={"texture": "rigid", "grain": "edge"})
    edge = gp.Mesh(gp.PathRegion(nodes=boundary, hard=False), 30, f, g, time=(0, 1), soil_properties=cfg)
    edge.solve(local_mask=mask, seed_coord=[100.0, 100.0], tolerance=0.5,
               solver_options={"texture": "rigid", "grain": "edge"})
    assert edge.solved

    seq = gp.Sequence(str(tmp_path), gp.PathRegion(nodes=boundary, hard=False), 30, time=1.0, soil_properties=cfg)
    assert seq.soil_properties["soil"] == "clay"
    seq.solve(local_mask=mask, seed_coord=[100.0, 100.0], max_norm=1e-6, tolerance=0.5, solver_options=opts)
    assert np.allclose(np.asarray(seq.mesh_solutions[0].displacements), np.asarray(known.displacements), atol=1e-6)
    with pytest.raises(ValueError, match="soil_properties"):
        gp.Sequence(str(tmp_path), gp.PathRegion(nodes=boundary, hard=False), 30, time=1.0).solve(
            local_mask=mask, seed_coord=[100.0, 100.0], solver_options={"texture": "rigid", "grain": "edge"})


def _soil_speckle(tmp_path, **cfg):
    return _core.Speckle(
        image_cfg={"image_dir": str(tmp_path), "name": "s", "image_size": (120, 100), "file_format": ".png"},
        speckle_cfg={"soil": "sand", "matrix_brightness": 180, "seed_brightness": 90, "asr": 0.2, **cfg},
        progression="deformation", deformation_cfg={"comp": [0.0] * 12}, noise_cfg=(0.0, 0.0),
        scale_cfg={"scale": "lin", "n": 2})


@pytest.mark.parametrize("key", ["sphericity", "sphericity_sd"])
def test_soil_sphericity_is_refused(key, tmp_path):
    """Sphericity is no longer a soil setting (Speckle or soil_properties)."""
    with pytest.raises(ValueError, match="no longer a soil setting"):
        _soil_speckle(tmp_path, **{key: 0.8})


def test_soil_grain_brightness_settings(tmp_path):
    """Brightness inside grains is on by default (the calibrated values given explicitly give the same
    image); it changes the image against flat grains but not which grain each pixel shows; negative
    values are refused."""
    from PIL import Image
    imgs, owners = {}, {}
    for tag, cfg in [("default", {}),
                     ("explicit", {"grain_tilt": 0.14, "grain_dome": 0.17, "grain_rim": 0.0, "grain_texture": 0.085,
                                   "texture_size": 0.10}),
                     ("flat", {"grain_tilt": 0.0, "grain_dome": 0.0, "grain_rim": 0.0, "grain_texture": 0.0})]:
        folder = tmp_path / tag
        folder.mkdir()
        sp = _soil_speckle(folder, d_p=20, aspect_ratio=0.7, **cfg)
        sp.solve(seed=3)
        imgs[tag] = np.asarray(Image.open(folder / "s_0.png"), dtype=float)
        owners[tag] = sp.grains(0)[1]
    assert np.array_equal(imgs["default"], imgs["explicit"])
    assert np.array_equal(owners["default"], owners["flat"])
    inside = owners["flat"] > 0
    assert np.abs(imgs["flat"] - imgs["default"])[inside].mean() > 2
    with pytest.raises(ValueError, match="grain_texture"):
        _soil_speckle(tmp_path, grain_texture=-0.1)


def test_qicpic_statistics_of_ellipses():
    """Outline statistics from silhouettes: ellipses have almost no irregularity and no dent, and the
    reported aspect ratios are averaged; a figure of eight is left out."""
    from geopyv_dev import _qicpic
    y, x = np.mgrid[-120:121, -120:121]
    masks, aspect = [], []
    for k in range(12):
        q, t = 0.55 + 0.03 * k, 0.4 * k
        u, v = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
        masks.append((u / 90) ** 2 + (v / (90 * q)) ** 2 <= 1)
        aspect.append(q)
    eight = ((x - 45) ** 2 + y ** 2 <= 50 ** 2) | ((x + 45) ** 2 + y ** 2 <= 50 ** 2)
    st = _qicpic.statistics(masks + [eight], np.array(aspect + [0.5]))
    assert st["grains"] == 12
    assert st["irregularity"] < 0.01 and st["dent"][-1] < 0.01
    assert abs(st["aspect_ratio"] - np.mean(aspect)) < 1e-12
    assert len(st["profile"]) == len(st["profile_sd"]) == 22 and len(st["dent"]) == 21


_REPORT = "/home/jdks2/Documents/claude_playground/docs/soil_characterisation/Leighton_Buzzard.docx"


@pytest.mark.skipif(not os.path.exists(_REPORT), reason="QicPic report not available")
def test_speckle_outlines_from_report(tmp_path):
    """outlines=<report>: aspect ratio from the report unless given; the drawn grains have it."""
    sp = _soil_speckle(tmp_path, grains="distribution", psd=np.array([[0.7, 10.0], [1.0, 50.0], [1.4, 90.0]]),
                       d_p=12, outlines=_REPORT)
    sp.solve(seed=3)
    table, _ = sp.grains(0)
    top = table["layer"] == 0
    assert abs(np.mean(table["aspect_ratio"][top]) - 0.70) < 0.03
    sp2 = _soil_speckle(tmp_path / "b", outlines=_REPORT, aspect_ratio=0.5)          # uniform: no spread
    sp2.solve(seed=3)
    assert np.allclose(sp2.grains(0)[0]["aspect_ratio"], 0.5, atol=0.02)
