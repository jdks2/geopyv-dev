"""solver_options={"topology": "tracked"}: the same subsets through every
reference update, each carrying its own history (MeshSolution.tracks)."""
import glob
import os

import numpy as np
import pytest

import geopyv_dev as gp

# u = 3 + 0.04 x' + 0.03 y' about the image centre, growing linearly over the images
COMP = [3.0, 0.0, 0.04, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
BOUNDARY = np.array([[80.0, 80.0], [320.0, 80.0], [320.0, 320.0], [80.0, 320.0]])


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    d = tmp_path_factory.mktemp("tracked")
    sp = gp.Speckle(image_cfg={"image_dir": str(d), "name": "s", "image_size": (401, 401), "file_format": ".png"},
                    speckle_cfg={"speckle_size": 2.5, "speckle_number": 1800}, progression="deformation",
                    deformation_cfg={"comp": COMP, "origin": (200.0, 200.0)}, noise_cfg=(0.0, 0.0),
                    scale_cfg={"scale": "lin", "n": 4})
    sp.solve(seed=3)
    assert len(glob.glob(os.path.join(str(d), "*.png"))) == 4
    return str(d), sp


def solve(image_dir, topology):
    seq = gp.Sequence(image_dir, gp.PathRegion(nodes=BOUNDARY, hard=False), 120)
    seq.solve(local_mask=gp.Mask(mask_type="local", shape="circle", size=20), seed_coord=[300.0, 300.0],
              options=gp.SequenceOptions(sequential=True), solver_options={"topology": topology})
    return seq.mesh_solutions


def test_tracked_subsets_keep_their_identity_and_history(images):
    image_dir, sp = images
    meshes = solve(image_dir, "tracked")
    first = meshes[0].tracks
    for k, m in enumerate(meshes):
        t = m.tracks
        assert t is not None
        # the same material points throughout (nothing leaves this region)
        np.testing.assert_array_equal(t["id"], first["id"])
        np.testing.assert_array_equal(t["origin"], first["origin"])
        # each subset sits where its own history put it
        np.testing.assert_allclose(t["position"], t["origin"] + t["reference_warp"][:, :2], atol=1e-9)
        # its history against the applied deformation of its material point
        truth = np.asarray(sp.warp(k + 1, np.ascontiguousarray(t["origin"])))[:, :6]
        assert np.abs(t["warp"][:, :2] - truth[:, :2]).max() < 0.15
        assert np.abs(t["warp"][:, 2:] - truth[:, 2:]).max() < 0.01
    # sequential updates move the subsets: the second reference is the first target
    np.testing.assert_allclose(meshes[1].tracks["position"],
                               first["position"] + first["increment"][:, :2], atol=1e-9)


def test_uniform_topology_has_no_tracks(images):
    image_dir, _ = images
    assert all(m.tracks is None for m in solve(image_dir, "uniform"))


def test_tracked_topology_needs_mesh_order_1(images):
    image_dir, _ = images
    seq = gp.Sequence(image_dir, gp.PathRegion(nodes=BOUNDARY, hard=False), 120, mesh_order=2)
    with pytest.raises(Exception, match="mesh_order=1"):
        seq.solve(local_mask=gp.Mask(mask_type="local", shape="circle", size=20), seed_coord=[300.0, 300.0],
                  solver_options={"topology": "tracked"})


def test_tracked_topology_is_for_sequences_only(images):
    image_dir, _ = images
    paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    f, g = gp.Image(filepath=paths[0]), gp.Image(filepath=paths[1])
    mesh = gp.Mesh(boundary=gp.PathRegion(nodes=BOUNDARY, hard=False), target_nodes=120, f_img=f, g_img=g)
    with pytest.raises(ValueError, match="Sequence.solve only"):
        mesh.solve(local_mask=gp.Mask(mask_type="local", shape="circle", size=20), seed_coord=[300.0, 300.0],
                   solver_options={"topology": "tracked"})


def test_reused_mesh_keeps_the_hard_region_mask(images):
    """sync=True reuses the first pair's mesh for the next pair of the same
    reference. With a hard boundary the reused mesh must still leave pixels
    outside the region out of its subsets: the second pair must then solve as
    on a freshly built mesh (sync=False)."""
    image_dir, _ = images
    region = lambda: gp.PathRegion(nodes=BOUNDARY, hard=True)
    out = {}
    for sync in (True, False):
        seq = gp.Sequence(image_dir, region(), 120)
        seq.solve(local_mask=gp.Mask(mask_type="local", shape="circle", size=20), seed_coord=[300.0, 300.0],
                  options=gp.SequenceOptions(sync=sync))
        out[sync] = seq.mesh_solutions[1]
    np.testing.assert_allclose(np.asarray(out[True].nodes), np.asarray(out[False].nodes), atol=1e-9)
    np.testing.assert_allclose(np.asarray(out[True].p), np.asarray(out[False].p), atol=1e-6)
