"""Demo 0 - Images, masks and regions.

Loads the demo speckle images, builds local and global masks, defines
boundary/exclusion regions, and generates a small synthetic speckle series.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images" / "shear"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
print("=== Image ===")
f_img = gp.Image(filepath=str(IMAGES / "shear_0.jpg"))
g_img = gp.Image(filepath=str(IMAGES / "shear_4.jpg"))
print(f"  shape:    {f_img.image_gs.shape}")
print(f"  border:   {f_img.border} px")
print(f"  filepath: {f_img.filepath}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
ax1.imshow(f_img.image_gs, cmap="gist_gray")
ax1.set_title("Reference (shear_0)")
ax2.imshow(g_img.image_gs, cmap="gist_gray")
ax2.set_title("Target (shear_4)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Local masks: the subset shape reused at every node
# ---------------------------------------------------------------------------
print("\n=== Local masks ===")
mask_c = gp.Mask(mask_type="local", shape="circle", size=25)
mask_s = gp.Mask(mask_type="local", shape="square", size=25)
print(f"  {mask_c}")
print(f"  {mask_s}")

# ---------------------------------------------------------------------------
# Regions: boundaries and exclusions
# ---------------------------------------------------------------------------
print("\n=== Regions ===")
boundary_nodes = np.array(
    [[250.0, 250.0], [250.0, 750.0], [750.0, 750.0], [750.0, 250.0]]
)
excl_nodes = np.array(
    [[560.0, 560.0], [560.0, 680.0], [680.0, 680.0], [680.0, 560.0]]
)
boundary = gp.PathRegion(nodes=boundary_nodes, hard=True)
exclusion = gp.PathRegion(nodes=excl_nodes, hard=True)
circle = gp.CircleRegion(centre=[400.0, 400.0], radius=60.0, size=15.0, option="F", hard=True)
print(f"  boundary nodes:  {boundary.current_nodes.shape}")
print(f"  circle option:   {circle.option}, centre {circle.current_centre}")

# ---------------------------------------------------------------------------
# Global mask: the active pixels inside the boundary, minus exclusions
# ---------------------------------------------------------------------------
print("\n=== Global mask ===")
global_mask = gp.Mask(
    mask_type="global",
    f_img=f_img,
    boundary=boundary,
    exclusions=[exclusion, circle],
)
print(f"  binary shape: {global_mask.binary.shape}, active pixels: {int(global_mask.binary.sum())}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(f_img.image_gs, cmap="gist_gray")
ax.imshow(np.ma.masked_where(global_mask.binary > 0, global_mask.binary), cmap="autumn", alpha=0.4)
ax.set_title("Global mask (masked pixels shaded)")
plt.show()

# ---------------------------------------------------------------------------
# Speckle: synthetic images with a known deformation
# ---------------------------------------------------------------------------
print("\n=== Speckle ===")
# The demo images were made this way; speckle_shear.pyv holds their ground truth.
truth = gp.load(str(IMAGES / "speckle_shear.pyv"))
print(f"  loaded: {truth}")

# A small new series: 401 x 401 px, 1 % simple shear over 3 images.
speckle = gp.Speckle(
    image_cfg={"image_dir": str(OUTPUT / "speckle"), "name": "demo", "image_size": (401, 401)},
    speckle_cfg={"speckle_size": 6, "speckle_number": 1500},
    progression="deformation",
    deformation_cfg={"comp": [0.0, 0.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    noise_cfg=(0.0, 0.0),
    scale_cfg={"scale": "lin", "n": 3},
)
(OUTPUT / "speckle").mkdir(exist_ok=True)
speckle.solve(seed=42)
print(f"  wrote: {sorted(p.name for p in (OUTPUT / 'speckle').glob('demo_*'))}")
