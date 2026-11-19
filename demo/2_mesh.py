"""Demo 2 - Mesh: measuring a whole area between two images.

A Mesh covers the area of interest with triangles; a subset is centred on
every corner (NODE) and tracked as in 1_subset.py. The nodes are solved
outwards from a SEED node in order of match quality - each solved node gives
its neighbours a good first guess, which is fast and robust (this is called
reliability-guided correlation). Pick the seed somewhere the motion is small
and the texture good.

Displacements are known at the nodes and interpolated across each triangle.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images" / "shear"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

f_img = gp.Image(filepath=str(IMAGES / "shear_0.png"))
g_img = gp.Image(filepath=str(IMAGES / "shear_4.png"))
local_mask = gp.Mask(mask_type="local", shape="circle", size=30)

# The area to measure, and a hole in it. hard=False on the boundary: subsets
# near the edge may still use the pixels just outside it. The hole is hard:
# its pixels are never used (a Mesh takes hard/soft for exclusions from
# exclusions_hard, all True by default).
boundary = gp.PathRegion(
    nodes=np.array([[300.0, 300.0], [300.0, 700.0], [700.0, 700.0], [700.0, 300.0]]),
    hard=False,
)
exclusion = gp.CircleRegion(centre=[600.0, 600.0], radius=40.0, size=15.0, option="F")

# ---------------------------------------------------------------------------
# 1. Build the mesh
# ---------------------------------------------------------------------------
print("=== 1. Building a mesh ===")
mesh = gp.Mesh(
    boundary=boundary,
    target_nodes=200,  # About how many nodes to place...
    f_img=f_img,
    g_img=g_img,
    size=(15.0, 200.0),  # ...with triangle sides kept between these lengths, px.
    exclusions=[exclusion],
    mesh_order=1,  # 1: straight-sided triangles; 2: curved (6-node) triangles.
    time=0.4,  # Optional: seconds between the two images, for velocities.
)
print(mesh)
print(f"  {mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} triangles")

# ---------------------------------------------------------------------------
# 2. Solve
# ---------------------------------------------------------------------------
print("\n=== 2. Solving ===")
mesh.solve(
    local_mask=local_mask,
    seed_coord=[500.0, 500.0],  # Start here: the shear is about the centre, so it barely moves.
    seed_warp=[0.0] * 6,  # First guess at the seed: no motion (6 numbers for order 1).
    subset_order=1,
    tolerance=0.75,  # Lowest C_ZNCC accepted at a node.
    seed_tolerance=0.9,  # The seed must match better than this.
)
print(mesh)
print(f"  C_ZNCC: mean {mesh.c_zncc.mean():.4f}, worst {mesh.c_zncc.min():.4f}")
u, v = mesh.displacements[:, 0], mesh.displacements[:, 1]
print(f"  u from {u.min():.2f} to {u.max():.2f} px, v from {v.min():.2f} to {v.max():.2f} px")
print("  (expected about +-2 px at the boundary: 1 % shear over 200 px from the centre)")

mesh.inspect()  # The mesh on the reference image.
mesh.inspect(subset_idx=mesh.seed_node)  # One node: its subset and result.
mesh.convergence(quantity="C_ZNCC")  # Histogram of match quality over all nodes.

print("\nContours of u (horizontal) and v (vertical) displacement:")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
mesh.contour("u", ax=ax1, show=False)
mesh.contour("v", ax=ax2, show=False)
plt.tight_layout()
plt.show()
print("With time=0.4 s, rate=True gives velocity (px/s) instead:")
mesh.contour("R", rate=True)  # R: the size of the displacement.

# ---------------------------------------------------------------------------
# 3. Save and reload
# ---------------------------------------------------------------------------
path = OUTPUT / "mesh.pyv"
mesh.save(str(path))
mesh = gp.load(str(path))
print(f"\nLoaded from output/mesh.pyv: {mesh}")
print("\nNext: 3_sequence.py follows the sand through all five images.")
