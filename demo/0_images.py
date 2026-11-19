"""Demo 0 - Images, masks and regions: what PIV works with.

PIV (particle image velocimetry, also called DIC - digital image correlation)
measures how a surface moves by photographing it twice and working out where
each small patch of the first photograph went in the second. It needs:

  * a REFERENCE image - the surface before it moves (the starting state);
  * a TARGET image - the same surface after it has moved;
  * texture on the surface - a random pattern (here, sand grains), so that
    every patch looks different from its neighbours and can be found again.

The patch that is tracked is called a SUBSET. Its shape is set by a LOCAL
mask; which parts of the photograph are analysed at all is set by a GLOBAL
mask, built from a boundary and any exclusions (REGIONS).

Run this script and close each figure to move on.
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
# 1. Reference and target images
# ---------------------------------------------------------------------------
print("=== 1. Reference and target images ===")
print("The demo images are five synthetic photographs of Leighton Buzzard sand,")
print("made with gp.Speckle (see images/make_shear_images.py). Between the first")
print("and the last the sand is sheared by 1 % - too little to see by eye.")
print()
print("Image 0 is the REFERENCE: every displacement is measured from it.")
print("Image 4 is the TARGET: the state we want to measure.")

# gp.Image loads a photograph as greyscale and prepares it for sub-pixel
# work: it fits a smooth surface through the pixels so that the brightness
# can be read between pixel centres (a displacement of 0.37 px is normal).
f_img = gp.Image(filepath=str(IMAGES / "shear_0.png"))
g_img = gp.Image(filepath=str(IMAGES / "shear_4.png"))
print(f"  size: {f_img.image_gs.shape[1]} x {f_img.image_gs.shape[0]} px")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(f_img.image_gs, cmap="gist_gray")
axes[0].set_title("Reference (shear_0)")
axes[1].imshow(g_img.image_gs, cmap="gist_gray")
axes[1].set_title("Target (shear_4)")
diff = g_img.image_gs.astype(float) - f_img.image_gs.astype(float)
axes[2].imshow(diff, cmap="RdBu", vmin=-80, vmax=80)
axes[2].set_title("Target - reference")
for ax in axes:
    ax.set_axis_off()
fig.suptitle("The two images look the same; their difference shows the sand has moved,\n"
             "most at the edges (the shear is about the centre)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 2. What makes a usable texture
# ---------------------------------------------------------------------------
print("\n=== 2. Texture ===")
print("PIV recognises a patch by its pattern of light and dark. Rules of thumb:")
print("  * each grain (speckle) should be at least 3-4 px across, or the")
print("    measured displacements lock onto whole pixels;")
print("  * a subset should be about 4 grains in radius, so it holds enough")
print("    independent detail to be unique. Here the median grain is 8 px, so")
print("    the demos use a subset radius of 30 px.")

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(f_img.image_gs[400:600, 400:600], cmap="gist_gray")
ax.add_patch(plt.Circle((100, 100), 30, fill=False, color="tab:red", lw=2))
ax.set_title("A 200 x 200 px close-up; red: one subset (radius 30 px)")
ax.set_axis_off()
plt.show()

# ---------------------------------------------------------------------------
# 3. Local masks: the shape of a subset
# ---------------------------------------------------------------------------
print("\n=== 3. Local masks ===")
print('A LOCAL mask (mask_type="local") is the shape of one subset. The same')
print("shape is used at every point measured. shape is \"circle\" (size = radius)")
print('or "square" (size = half-width), in px.')
mask_c = gp.Mask(mask_type="local", shape="circle", size=30)
mask_s = gp.Mask(mask_type="local", shape="square", size=30)
print(f"  {mask_c}: {mask_c.n_px} pixels")
print(f"  {mask_s}: {mask_s.n_px} pixels")

# ---------------------------------------------------------------------------
# 4. Regions: the boundary and exclusions
# ---------------------------------------------------------------------------
print("\n=== 4. Regions ===")
print("A REGION is an outline: the BOUNDARY of the area to measure, or an")
print("EXCLUSION inside it (a hole - e.g. a footing, a pile, a marker).")
print("  PathRegion: any polygon, from its corner points [x, y].")
print("  CircleRegion: a circle, from its centre and radius.")
print("hard=True also removes the pixels from the analysis; hard=False only")
print("shapes where points are measured. option says how the outline follows")
print('the material through a test: "F" flexible (moves and deforms with it),')
print('"R" rigid (moves without deforming), "S" static (stays put).')
boundary = gp.PathRegion(
    nodes=np.array([[250.0, 250.0], [250.0, 750.0], [750.0, 750.0], [750.0, 250.0]]),
    hard=True,
)
exclusion = gp.PathRegion(
    nodes=np.array([[560.0, 560.0], [560.0, 680.0], [680.0, 680.0], [680.0, 560.0]]),
    hard=True,
)
circle = gp.CircleRegion(centre=[400.0, 400.0], radius=60.0, size=15.0, option="R", hard=True)
print(f"  boundary: {boundary.current_nodes.shape[0]} corners")
print(f"  circle:   option {circle.option!r}, centre {circle.current_centre}")

# ---------------------------------------------------------------------------
# 5. Global mask: which pixels are analysed
# ---------------------------------------------------------------------------
print("\n=== 5. Global mask ===")
print('A GLOBAL mask (mask_type="global") marks every pixel of the photograph')
print("as used (1) or not (0): inside the boundary and outside the exclusions.")
print("A Mesh or Sequence builds one for you from its boundary and exclusions;")
print("making one by hand is useful for checking them, or for a single Subset.")
global_mask = gp.Mask(
    mask_type="global",
    f_img=f_img,
    boundary=boundary,
    exclusions=[exclusion, circle],
)
active = int(global_mask.binary.sum())
print(f"  {active} of {global_mask.binary.size} pixels are analysed")

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(f_img.image_gs, cmap="gist_gray")
ax.imshow(np.ma.masked_where(global_mask.binary > 0, global_mask.binary), cmap="autumn", alpha=0.5)
ax.set_title("Global mask: shaded pixels are not analysed")
ax.set_axis_off()
plt.show()

# ---------------------------------------------------------------------------
# 6. Making your own synthetic images
# ---------------------------------------------------------------------------
print("\n=== 6. Synthetic images ===")
print("gp.Speckle draws a texture and moves it by a known amount, so a result")
print("can be checked against the truth (see 6_validation.py). The truth for")
print("the demo images is stored next to them:")
truth = gp.load(str(IMAGES / "speckle_shear.pyv"))
print(f"  {truth}")

print("A small painted-speckle example, sheared by 1 % over 3 images:")
(OUTPUT / "speckle").mkdir(exist_ok=True)
speckle = gp.Speckle(
    image_cfg={"image_dir": str(OUTPUT / "speckle"), "name": "demo", "image_size": (401, 401),
               "file_format": ".png"},
    speckle_cfg={"speckle_size": 6, "speckle_number": 1500},
    progression="deformation",
    deformation_cfg={"comp": [0.0, 0.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    noise_cfg=(0.0, 0.0),
    scale_cfg={"scale": "lin", "n": 3},
)
speckle.solve(seed=42)
print(f"  wrote {sorted(p.name for p in (OUTPUT / 'speckle').glob('demo_*'))} to output/speckle/")
print("\nNext: 1_subset.py tracks a single subset between the two images.")
