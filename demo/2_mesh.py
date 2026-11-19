"""Demo 2 - Mesh: full-field correlation of one image pair."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images" / "shear"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

f_img = gp.Image(filepath=str(IMAGES / "shear_0.jpg"))
g_img = gp.Image(filepath=str(IMAGES / "shear_4.jpg"))
local_mask = gp.Mask(mask_type="local", shape="circle", size=25)

boundary = gp.PathRegion(
    nodes=np.array([[300.0, 300.0], [300.0, 700.0], [700.0, 700.0], [700.0, 300.0]]),
    hard=False,
)
exclusion = gp.CircleRegion(centre=[600.0, 600.0], radius=40.0, size=15.0, option="F", hard=True)

# ---------------------------------------------------------------------------
# Build: triangulate the region of interest
# ---------------------------------------------------------------------------
print("=== Mesh ===")
mesh = gp.Mesh(
    boundary=boundary,  # A region object or an (N, 2) coordinate array.
    target_nodes=200,  # Approximate number of nodes.
    f_img=f_img,  # Reference image.
    g_img=g_img,  # Target image.
    size=(15.0, 70.0),  # (min, max) element edge length, px.
    exclusions=[exclusion],  # Holes in the mesh.
    mesh_order=2,  # 1 = linear, 2 = quadratic elements.
    time=0.4,  # Optional: seconds between the two images.
)
print(mesh)
print(f"  {mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} elements")

# ---------------------------------------------------------------------------
# Solve: reliability-guided propagation from a seed node
# ---------------------------------------------------------------------------
mesh.solve(
    local_mask=local_mask,
    seed_coord=[500.0, 500.0],
    seed_warp=[0.0] * 12,  # Length 6 * subset_order.
    subset_order=2,
    tolerance=0.75,
    seed_tolerance=0.9,
)
print(mesh)
print(f"  solved:      {mesh.solved}")
print(f"  mean C_ZNCC: {mesh.c_zncc.mean():.4f}")
print(f"  seed node:   {mesh.seed_node}")

mesh.inspect()
mesh.inspect(subset_idx=mesh.seed_node)
mesh.convergence(quantity="C_ZNCC")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
mesh.contour("u", ax=ax1, show=False)
mesh.contour("v", ax=ax2, show=False)
plt.tight_layout()
plt.show()
mesh.contour("R", rate=True)  # Speed (px/s), using time=0.4.

# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------
path = OUTPUT / "mesh.pyv"
mesh.save(str(path))
mesh = gp.load(str(path))
print(f"\nLoaded: {mesh}")
print(f"  mean C_ZNCC: {mesh.c_zncc.mean():.4f}")
mesh.contour("u")
