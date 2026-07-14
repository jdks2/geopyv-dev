#!/usr/bin/env python3
"""
Generate geopyv-dev tutorial Jupyter notebooks.
Run from the geopyv-dev repo root:
    python tutorials/make_tutorials.py
"""

import json
import os

os.makedirs("tutorials", exist_ok=True)

# ---------------------------------------------------------------------------
# Notebook format helpers
# ---------------------------------------------------------------------------

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def save_nb(path, notebook):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Shared snippets
# ---------------------------------------------------------------------------

COMMON_IMPORTS = """\
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import geopyv_dev as gp

%matplotlib inline
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "animation.embed_limit": 50,
})
"""

GEN_SINGLE = """\
# Generate a reference and a deformed speckle image.
# 5 px x-translation + slight shear — fast enough to run in < 2 s.
os.makedirs("data/single", exist_ok=True)
speckle = gp.Speckle(
    image_cfg={"image_dir": "data/single", "name": "demo", "image_size": (401, 401)},
    speckle_cfg={"speckle_size": 3.5, "speckle_number": 700},
    progression="deformation",
    deformation_cfg={"comp": [5.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    noise_cfg=(0.0, 1.5),
    scale_cfg={"scale": "lin", "n": 2},
)
speckle.solve(seed=42)
f_img = gp.Image(filepath="data/single/demo_0.jpg")
g_img = gp.Image(filepath="data/single/demo_1.jpg")
print(f"Reference : {f_img}")
print(f"Target    : {g_img}")
"""

GEN_SEQUENCE = """\
# Generate a 5-image shear-band sequence (images seq_0 … seq_4).
os.makedirs("data/sequence", exist_ok=True)
speckle_seq = gp.Speckle(
    image_cfg={"image_dir": "data/sequence", "name": "seq", "image_size": (401, 401)},
    speckle_cfg={"speckle_size": 3.5, "speckle_number": 700},
    progression="deformation",
    deformation_cfg={
        "comp": [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mode": "SB",
        "option": "sin",
        "width": 100.0,
    },
    noise_cfg=(0.2, 1.5),
    scale_cfg={"scale": "lin", "n": 5},
)
speckle_seq.solve(seed=42)
print("Generated: data/sequence/seq_0.jpg … seq_4.jpg")
"""

MESH_PARAMS = """\
boundary_nodes = np.array([
    [50.0, 50.0], [50.0, 350.0], [350.0, 350.0], [350.0, 50.0]
])
boundary  = gp.PathRegion(nodes=boundary_nodes, hard=False)
local_mask = gp.Mask(mask_type="local", shape="circle", size=25)
seed_coord = [200.0, 200.0]
"""

# ===========================================================================
# Notebook 0 — Introduction
# ===========================================================================

NB0 = nb([

    md("""\
# 00 — Introduction to `geopyv_dev`

**Digital Image Correlation (DIC)** measures full-field displacements and strains from
photographs of a deforming surface — no strain gauges, no contact.  `geopyv_dev` is a
high-performance Python package (Rust core, PyO3 bindings) that implements the complete
DIC pipeline.

## Tutorial map

| Notebook | Topic |
|----------|-------|
| **00 — you are here** | What is DIC? Package overview |
| **01** | `Image`, `Mask`, `CircleRegion`, `PathRegion` |
| **02** | `Subset` — single-point correlation |
| **03** | `Mesh` — full-field single image pair |
| **04** | `Sequence` — multi-image time series |
| **05** | `Particle` & `Field` — strain-path tracking |

> **Setup:** activate your virtual environment and ensure `geopyv_dev` is installed
> (`pip install geopyv_dev` or `maturin develop` from the repo root).
"""),

    code("""\
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

%matplotlib inline
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "animation.embed_limit": 50,
})
"""),

    md("""\
## What is DIC?

The core idea is simple:

1. **Photograph** the surface before deformation (the *reference* image).
2. **Deform** the specimen — mechanically, thermally, or any other way.
3. **Photograph** again (the *target* image).
4. **Correlate** small square or circular patches (*subsets*) between the two images
   to find where each part of the surface moved.

The animation below shows this process. Each red box is a subset — a small region whose
intensity pattern is matched to the target image to recover the local displacement.
"""),

    code("""\
# --- Animated DIC concept diagram ---
rng = np.random.RandomState(42)
n = 300
xs, ys = rng.uniform(10, 190, n), rng.uniform(10, 190, n)
sz = rng.uniform(20, 120, n)

DX, DY = 12, 6           # true displacement applied to target
SUBSETS = [              # subset centres to animate through
    (60, 60), (130, 60), (60, 130), (130, 130), (95, 95),
]

fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")
for ax, title in [(ax_r, "Reference image  $f$"),
                   (ax_t, "Target image  $g$")]:
    ax.scatter(xs, ys, s=sz, c="k", alpha=0.85, linewidths=0)
    if ax is ax_t:
        ax.scatter(xs + DX, ys + DY, s=sz, c="k", alpha=0.85, linewidths=0)
        ax.set_facecolor("#f9f9f9")
    ax.set_xlim(0, 200); ax.set_ylim(0, 200)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

W = 36   # subset width/height
ref_box = patches.Rectangle((0, 0), W, W, lw=2, ec="crimson", fc="crimson", alpha=0.15)
tar_box = patches.Rectangle((0, 0), W, W, lw=2, ec="seagreen", fc="none", ls="--")
ax_r.add_patch(ref_box)
ax_t.add_patch(tar_box)
arrow = ax_t.annotate("", xy=(0, 0), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->", color="royalblue", lw=2))
status = ax_t.text(100, 5, "", ha="center", va="bottom", fontsize=9,
                   color="royalblue", fontweight="bold")

N_FRAMES = len(SUBSETS) * 20   # 20 frames per subset (search + lock)

def update(frame):
    si = frame // 20
    phase = frame % 20
    cx, cy = SUBSETS[si]
    ref_box.set_xy((cx - W/2, cy - W/2))

    if phase < 10:           # "searching" phase — box slides in from reference position
        frac = phase / 9
        tx = cx - W/2 + frac * DX * 0.8
        ty = cy - W/2 + frac * DY * 0.8
        tar_box.set(xy=(tx, ty), ec="crimson", alpha=0.5)
        status.set_text("searching…")
    else:                    # "locked on" phase
        tar_box.set(xy=(cx - W/2 + DX, cy - W/2 + DY), ec="seagreen", alpha=1.0)
        arrow.xy = (cx + DX, cy + DY)
        arrow.xyann = (cx, cy)
        status.set_text(f"Δu={DX}, Δv={DY} px")

    return ref_box, tar_box, arrow, status

anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=80, blit=True)
plt.tight_layout()
os.makedirs("../docs/assets", exist_ok=True)
anim.save("../docs/assets/nb0_dic_concept.gif", writer="pillow", fps=12)
HTML(anim.to_jshtml())
"""),

    md("""\
## Why does DIC need a speckle pattern?

A plain, uniform surface has no features to track — every patch looks the same.
DIC requires a **random, high-contrast texture** so each subset has a unique intensity
fingerprint that can be matched reliably.

The most common approach: spray a thin coat of white paint, then speckle with black
aerosol from a distance.  Key rules of thumb:

- Speckle diameter ≈ **3–7 pixels** (too small → aliasing; too large → low contrast gradient)
- Coverage ≈ **40–60 %** of the surface
- No repeating pattern (lattices, gratings, etc. cause false matches)

`geopyv_dev` includes a `Speckle` class to generate synthetic speckle images for testing.
"""),

    code("""\
# Generate a quick synthetic speckle pair to illustrate the pattern
import os
import geopyv_dev as gp

os.makedirs("data/single", exist_ok=True)
_sp = gp.Speckle(
    image_cfg={"image_dir": "data/single", "name": "intro", "image_size": (301, 301)},
    speckle_cfg={"speckle_size": 3.5, "speckle_number": 500},
    progression="deformation",
    deformation_cfg={"comp": [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    noise_cfg=(0.0, 2.0),
    scale_cfg={"scale": "lin", "n": 2},
)
_sp.solve(seed=7)

_f = gp.Image(filepath="data/single/intro_0.jpg")
_g = gp.Image(filepath="data/single/intro_1.jpg")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, img, lbl in [(axes[0], _f.image_gs, "Reference $f$"),
                      (axes[1], _g.image_gs, "Target $g$  (+8 px)")]:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(lbl, fontsize=12, fontweight="bold")
    ax.axis("off")

diff = _g.image_gs - _f.image_gs
im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-80, vmax=80)
axes[2].set_title("Difference $g - f$", fontsize=12, fontweight="bold")
axes[2].axis("off")
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Intensity")
plt.suptitle("Synthetic speckle images", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig("../docs/assets/nb0_speckle_pair.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
## Package structure

`geopyv_dev` objects form a clear hierarchy — start at the bottom and work up:

```
Image  ──►  Mask (local / global)
               │
               ▼
            Subset  ◄──── single point, single pair
               │
               ▼
             Mesh   ◄──── full field, single pair
               │
               ▼
           Sequence ◄──── full field, image time series
               │
          ┌────┴────┐
          ▼          ▼
       Particle    Field ◄──── strain-path tracking
```

The diagram below maps these objects onto the two main workflows.
"""),

    code("""\
# Package hierarchy and workflow overview
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.5); ax.axis("off")

COLORS = {
    "data":     "#dbeafe",   # blue-tinted
    "core":     "#dcfce7",   # green-tinted
    "analysis": "#fef9c3",   # yellow-tinted
}

def box(ax, x, y, w, h, label, sublabel="", color="#f0f0f0"):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                   fc=color, ec="steelblue", lw=1.4)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0), label,
            ha="center", va="center", fontsize=10, fontweight="bold")
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha="center", va="center", fontsize=8, color="#555")

def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))

# Data layer
box(ax, 0.2, 3.8, 2.2, 1.2, "Image", "greyscale + B-spline", COLORS["data"])
box(ax, 2.7, 3.8, 2.2, 1.2, "Mask", "local or global", COLORS["data"])

# Core layer
box(ax, 0.2, 2.2, 2.2, 1.2, "Subset", "single point", COLORS["core"])
box(ax, 2.7, 2.2, 2.2, 1.2, "Mesh", "full field", COLORS["core"])
box(ax, 5.2, 2.2, 2.2, 1.2, "Sequence", "time series", COLORS["core"])

# Analysis layer
box(ax, 2.7, 0.5, 2.2, 1.2, "Particle", "1 tracking pt", COLORS["analysis"])
box(ax, 5.2, 0.5, 2.2, 1.2, "Field", "N tracking pts", COLORS["analysis"])

# Arrows
arrow(ax, 1.3, 3.8, 1.3, 3.4)
arrow(ax, 3.8, 3.8, 3.8, 3.4)
arrow(ax, 2.2, 2.8, 2.7, 2.8)        # Image → Mask (shared)
arrow(ax, 3.8, 2.2, 3.8, 1.7)        # Mesh → Particle
arrow(ax, 6.3, 2.2, 6.3, 1.7)        # Sequence → Field
arrow(ax, 4.9, 2.8, 5.2, 2.8)        # Mesh → Sequence

# Workflow labels
ax.text(3.8, 0.2, "Workflow A: single pair", ha="center", va="bottom",
        fontsize=9, color="steelblue", style="italic")
ax.text(6.3, 0.2, "Workflow B: time series", ha="center", va="bottom",
        fontsize=9, color="steelblue", style="italic")

plt.title("geopyv_dev object hierarchy", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
fig.savefig("../docs/assets/nb0_hierarchy.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
## What can DIC measure?

Once you have a solved `Mesh` or `Sequence`, you have access to:

| Quantity | Where | Notes |
|----------|-------|-------|
| Displacement **u**, **v** | `mesh.displacements` | Per-node, pixels |
| Warp vector **p** | `mesh.p` | 6 (order-1) or 12 (order-2) params per node |
| Correlation score C_ZNCC | `mesh.c_zncc` | 1.0 = perfect; < 0.7 = suspect |
| Strain components | `particle.strains` | Via `Particle` or `Field` |
| Volumetric strain | `particle.vol_strains` | Via `Particle` or `Field` |

Next up: **Notebook 01** — loading images and building masks.
"""),

])

# ===========================================================================
# Notebook 1 — Images and Masks
# ===========================================================================

NB1 = nb([

    md("""\
# 01 — Images and Masks

Before any correlation, you need two things:
- An **`Image`** — a greyscale photograph with pre-computed interpolation data.
- A **`Mask`** — defining which pixels to include in a subset or which region to analyse.

This notebook covers: `Image`, `Mask` (local and global), `CircleRegion`, `PathRegion`.
"""),

    code(COMMON_IMPORTS),

    code(GEN_SINGLE),

    md("""\
## The `Image` class

`Image` loads a JPEG/PNG, converts to greyscale, applies a mild Gaussian blur
(σ = 1.1), and pre-computes the **bi-quintic B-spline coefficient matrix** (`qcqt`).
This matrix lets the DIC solver evaluate sub-pixel intensity values extremely quickly
without recomputing the spline at every iteration.

```python
img = gp.Image(filepath="photo.jpg")   # from file
img = gp.Image(image_gs=array)         # from numpy array (already greyscale)
```

Key attributes:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `image_gs` | (H, W) | Greyscale pixel values 0–255 |
| `qcqt` | (H·6, W·6) | Pre-computed B-spline coefficient blocks |
| `border` | int | Padding used for coefficient computation |
| `filepath` | str \\| None | Source path |
"""),

    code("""\
# Explore the Image object
print(f_img)
print(f"  image_gs shape : {f_img.image_gs.shape}")
print(f"  qcqt shape     : {f_img.qcqt.shape}")
print(f"  border         : {f_img.border}")
print(f"  filepath       : {f_img.filepath}")
"""),

    code("""\
# Display the reference image alongside a zoomed crop
fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.5))

img_arr = f_img.image_gs
ax_full.imshow(img_arr, cmap="gray", vmin=0, vmax=255)
ax_full.set_title("Full reference image", fontsize=12, fontweight="bold")
ax_full.axis("off")

# Zoom into the centre — shows individual speckle blobs
cx, cy, R = 200, 200, 60
ax_zoom.imshow(img_arr[cy-R:cy+R, cx-R:cx+R], cmap="gray",
               extent=[cx-R, cx+R, cy+R, cy-R], vmin=0, vmax=255)
rect = patches.Rectangle((cx-R, cy-R), 2*R, 2*R, lw=2, ec="crimson", fc="none")
ax_full.add_patch(rect)
ax_zoom.set_title("Zoomed crop (120 × 120 px)", fontsize=12, fontweight="bold")
ax_zoom.set_xlabel("x (px)"); ax_zoom.set_ylabel("y (px)")
plt.tight_layout()
os.makedirs("../docs/assets", exist_ok=True)
fig.savefig("../docs/assets/nb1_image_zoom.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
### Under the hood: B-spline interpolation

The DIC solver must evaluate intensities at **sub-pixel** positions (e.g. x = 127.43).
`Image` pre-computes the B-spline coefficient matrix `qcqt` so that any sub-pixel
lookup is just a small matrix multiply — no recomputation needed per iteration.

The block for pixel (i, j) lives at `qcqt[i*6 : i*6+6, j*6 : j*6+6]`.
"""),

    code("""\
# Visualise the B-spline coefficient block for one pixel
# and show how it enables smooth sub-pixel interpolation.
import matplotlib.gridspec as gridspec
from scipy.interpolate import RectBivariateSpline

px, py = 140, 140   # pixel we're zooming into
W = 4               # window radius (pixels)
crop = f_img.image_gs[py-W:py+W+1, px-W:px+W+1]

# Build a smooth interpolant over the crop using scipy for illustration
xi = np.arange(px-W, px+W+1, dtype=float)
yi = np.arange(py-W, py+W+1, dtype=float)
interp = RectBivariateSpline(yi, xi, crop)
xf = np.linspace(px-W, px+W, 120)
yf = np.linspace(py-W, py+W, 120)
Zf = interp(yf, xf)

fig = plt.figure(figsize=(11, 4.5))
gs = gridspec.GridSpec(1, 2, wspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Pixel grid
im1 = ax1.imshow(crop, cmap="gray", vmin=0, vmax=255,
                  extent=[px-W-0.5, px+W+0.5, py+W+0.5, py-W-0.5], interpolation="nearest")
ax1.scatter(*np.meshgrid(xi, yi), s=10, c="red", zorder=5, label="pixel centres")
ax1.set_title("Pixel grid (discrete samples)", fontsize=11, fontweight="bold")
ax1.set_xlabel("x (px)"); ax1.set_ylabel("y (px)")
ax1.legend(fontsize=9)
fig.colorbar(im1, ax=ax1, fraction=0.046, label="Intensity")

# Smooth surface
im2 = ax2.imshow(Zf, cmap="gray", vmin=0, vmax=255,
                  extent=[px-W, px+W, py+W, py-W])
ax2.scatter(*np.meshgrid(xi, yi), s=10, c="red", zorder=5)
ax2.set_title("B-spline surface (sub-pixel)", fontsize=11, fontweight="bold")
ax2.set_xlabel("x (px)"); ax2.set_ylabel("y (px)")
fig.colorbar(im2, ax=ax2, fraction=0.046, label="Intensity")
plt.suptitle("From discrete pixels to a smooth intensity surface", fontsize=12, y=1.02)
fig.savefig("../docs/assets/nb1_bspline.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
## The `Mask` class

A `Mask` defines which pixels belong to a subset.  There are two fundamentally
different types, both accessed through the same class via `mask_type`:

| `mask_type` | Purpose | Key parameters |
|-------------|---------|----------------|
| `"local"` | Pixel offsets relative to a subset centre | `shape` (`"circle"` / `"square"`), `size` |
| `"global"` | Binary image-space mask (entire ROI at once) | `f_img`, `boundary`, `exclusions` |

> **Rule of thumb**: use `"local"` for `Subset` and `Mesh` (the default).
> Use `"global"` only when you need pixel-level masking of the image itself.
"""),

    code("""\
# --- Local masks: circle vs square ---
mask_c = gp.Mask(mask_type="local", shape="circle", size=25)
mask_s = gp.Mask(mask_type="local", shape="square", size=25)
print(mask_c)
print(mask_s)
print(f"Circle: {mask_c.n_px} active pixels  |  Square: {mask_s.n_px} active pixels")
"""),

    code("""\
# Visualise local mask pixel grids
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
SIZE = 25

for ax, mask, title in [
    (axes[0], mask_c, f"Local circle  (size={SIZE}, n_px={mask_c.n_px})"),
    (axes[1], mask_s, f"Local square  (size={SIZE}, n_px={mask_s.n_px})"),
]:
    # Build a 2D grid from the flat coordinate offsets
    coords = mask.coords          # shape (n_px, 2) — [dx, dy] offsets
    sm     = mask.subset_mask     # shape (2*size+1, 2*size+1) binary grid
    ax.imshow(sm, cmap="Blues", vmin=-0.2, vmax=1.2, interpolation="nearest")

    # Overlay active pixel markers
    ys, xs = np.where(sm)
    ax.scatter(xs, ys, s=6, c="navy", alpha=0.5, linewidths=0)

    # Mark centre
    c = SIZE
    ax.plot(c, c, "r+", ms=12, mew=2, label="centre")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9)

plt.suptitle("Local mask templates — active pixels in blue", fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig("../docs/assets/nb1_local_masks.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
### Global masks

A **global mask** computes a binary image of shape (H, W) based on a boundary polygon
and optional exclusion regions.  Pixels inside the boundary (and outside exclusions)
are set to 1 (active); all others are 0.

Global masks are passed to `Subset` or created standalone with `Mask(mask_type="global")`.
"""),

    code("""\
# Define a boundary and an exclusion region
boundary_nodes = np.array([
    [50.0, 50.0], [50.0, 350.0], [350.0, 350.0], [350.0, 50.0]
])
excl_nodes = np.array([
    [280.0, 280.0], [280.0, 340.0], [340.0, 340.0], [340.0, 280.0]
])

global_mask = gp.Mask(
    mask_type="global",
    f_img=f_img,
    boundary=gp.PathRegion(nodes=boundary_nodes, hard=True),
    exclusions=[gp.PathRegion(nodes=excl_nodes, hard=True)],
)
print(global_mask)
print(f"Binary mask shape: {global_mask.binary.shape}")
print(f"Active pixels: {global_mask.binary.sum()} / {global_mask.binary.size}")
"""),

    code("""\
# Show the global mask overlaid on the image
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].imshow(f_img.image_gs, cmap="gray", vmin=0, vmax=255)
axes[0].add_patch(patches.Polygon(boundary_nodes[:, [0, 1]], closed=True,
                                   lw=2, ec="crimson", fc="crimson", alpha=0.15))
axes[0].add_patch(patches.Polygon(excl_nodes[:, [0, 1]], closed=True,
                                   lw=2, ec="royalblue", fc="royalblue", alpha=0.3))
axes[0].set_title("Boundary (red) + exclusion (blue)", fontsize=11, fontweight="bold")
axes[0].axis("off")

masked = f_img.image_gs.copy().astype(float)
masked[global_mask.binary == 0] = np.nan
im = axes[1].imshow(masked, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Image after global mask", fontsize=11, fontweight="bold")
axes[1].axis("off")
plt.tight_layout()
fig.savefig("../docs/assets/nb1_global_mask.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
## `CircleRegion` and `PathRegion`

Regions are **tracked** boundary/exclusion objects.  Unlike plain numpy arrays, they:
- Record their full deformation history (`history_nodes`, `history_centres`)
- Can be updated when the reference image advances in a `Sequence`

The `option` parameter controls how the region moves between frames:

| `option` | Meaning |
|----------|---------|
| `"F"` | **Flexible** — region deforms with the local mesh warp (default) |
| `"R"` | **Rigid** — region translates rigidly |
| `"S"` | **Static** — region never moves |
| `"D"` | **Displacement-only** — translates by mean displacement |
"""),

    code("""\
# CircleRegion: circle centred at (200, 200), radius 80 px
circle_region = gp.CircleRegion(centre=[200.0, 200.0], radius=80.0, size=15.0, option="F")
print(circle_region)
print(f"  Nodes shape : {circle_region.current_nodes.shape}")
print(f"  Centre      : {circle_region.current_centre}")
print(f"  hard        : {circle_region.hard}")

# PathRegion: arbitrary polygon
path_region = gp.PathRegion(nodes=boundary_nodes, option="R")
print(path_region)
"""),

    code("""\
# Visualise both region types on the image
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(f_img.image_gs, cmap="gray", vmin=0, vmax=255, alpha=0.7)

cn = circle_region.current_nodes
ax.fill(cn[:, 0], cn[:, 1], fc="crimson", ec="crimson", alpha=0.2, lw=2, label="CircleRegion")
ax.plot(np.append(cn[:, 0], cn[0, 0]), np.append(cn[:, 1], cn[0, 1]), "r-", lw=2)

pn = path_region.current_nodes
ax.fill(pn[:, 0], pn[:, 1], fc="royalblue", ec="royalblue", alpha=0.15, lw=2, label="PathRegion")
ax.plot(np.append(pn[:, 0], pn[0, 0]), np.append(pn[:, 1], pn[0, 1]), "b-", lw=2)

ax.legend(fontsize=10); ax.axis("off")
ax.set_title("CircleRegion vs PathRegion on the speckle image", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig("../docs/assets/nb1_regions.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
## Summary

| Object | When to use |
|--------|------------|
| `Image` | Always — wraps every photograph |
| `Mask(mask_type="local")` | Passed to `Subset`, `Mesh`, `Sequence` as the subset template |
| `Mask(mask_type="global")` | When you need pixel-level masking of the image |
| `CircleRegion` | Circular boundary or exclusion that should track the deformation |
| `PathRegion` | Polygonal boundary or exclusion with optional tracking |

**Next:** Notebook 02 — correlating a single subset to find a point displacement.
"""),

])

# ===========================================================================
# Notebook 2 — Subset
# ===========================================================================

NB2 = nb([

    md("""\
# 02 — Subset: single-point correlation

A `Subset` is a small patch of pixels centred on one point in the reference image.
The DIC solver finds the **warp parameters** that best deform this patch to match the
same region in the target image — giving you the local displacement and strain at that point.

This notebook covers: warp parameters, `solve_icgn` vs `solve_fagn`,
correlation scores, convergence history, `save` / `load`.
"""),

    code(COMMON_IMPORTS),

    code(GEN_SINGLE + "\n" + MESH_PARAMS),

    md("""\
## The warp function

The DIC solver describes how a subset deforms using a **warp vector** `p`.

For **order 1** (affine, 6 parameters):

```
u(x,y) = p[0] + p[1]·Δx + p[2]·Δy
v(x,y) = p[3] + p[4]·Δx + p[5]·Δy
```

where Δx = x − x₀, Δy = y − y₀ (offsets from the subset centre x₀, y₀).

For **order 2** (quadratic, 12 parameters), second-order terms p[3..5] and p[9..11]
are added.  Most analyses use order 1 for `Subset` and order 2 for `Mesh`/`Sequence`.

The diagram below labels each parameter.
"""),

    code("""\
# Warp parameter diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
for ax in (ax1, ax2):
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5); ax.set_aspect("equal"); ax.axis("off")

# Reference square
sq = patches.Rectangle((0.5, 0.5), 2, 2, lw=2, ec="steelblue", fc="steelblue", alpha=0.12)
ax1.add_patch(sq)
ax1.text(1.5, 1.5, "$f$ subset\\n(reference)", ha="center", va="center",
         fontsize=10, color="steelblue", fontweight="bold")
ax1.set_title("Reference subset", fontsize=11, fontweight="bold")
ax1.plot(1.5, 1.5, "r+", ms=14, mew=2)
ax1.text(1.6, 1.6, "$(x_0, y_0)$", fontsize=9, color="red")

# Deformed (translate + slight shear)
from matplotlib.patches import FancyArrowPatch
pts = np.array([[0.5, 0.5], [2.5, 0.5], [2.7, 2.5], [0.5, 2.5], [0.5, 0.5]])
pts[:, 0] += 0.3   # u displacement
pts[:, 1] -= 0.25  # v displacement
ax2.add_patch(patches.Polygon(pts[:-1], closed=True, lw=2, ec="seagreen",
                               fc="seagreen", alpha=0.12))
ax2.set_title("Deformed subset (in target $g$)", fontsize=11, fontweight="bold")

# Arrows for u, v
ax2.annotate("", xy=(1.5+0.3, 1.5-0.25), xytext=(1.5, 1.5),
              arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2))
ax2.text(1.75, 1.25, "$u = p_0$", color="crimson", fontsize=10, fontweight="bold")
ax2.annotate("", xy=(1.5+0.3, 1.5-0.25-0.5), xytext=(1.5+0.3, 1.5-0.25),
              arrowprops=dict(arrowstyle="-|>", color="royalblue", lw=2))
ax2.text(1.85, 0.9, "$v = p_3$", color="royalblue", fontsize=10, fontweight="bold")

ax2.text(0.1, 3.2,
         r"$u(\\Delta x,\\Delta y)=p_0+p_1\\Delta x+p_2\\Delta y$",
         fontsize=9, color="#333")
ax2.text(0.1, 2.95,
         r"$v(\\Delta x,\\Delta y)=p_3+p_4\\Delta x+p_5\\Delta y$",
         fontsize=9, color="#333")

plt.suptitle("Order-1 warp: 6 parameters", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
os.makedirs("../docs/assets", exist_ok=True)
fig.savefig("../docs/assets/nb2_warp_diagram.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
## Creating and solving a `Subset`

```python
subset = gp.Subset(
    coord      = [x, y],       # centre of the subset in the reference image
    local_mask = mask,          # a local Mask object defining the patch shape
    f_img      = f_img,         # reference Image
    g_img      = g_img,         # target Image
    subset_order = 1,           # 1 (affine) or 2 (quadratic)
)
subset.solve()                  # runs ICGN by default
```

`solve()` is non-destructive to the images — you can create many subsets from the same pair.
"""),

    code("""\
# Create a subset at the image centre
subset = gp.Subset(
    coord      = [200.0, 200.0],
    local_mask = local_mask,
    f_img      = f_img,
    g_img      = g_img,
    subset_order = 1,
)
print(subset)
print(f"\\n  n_px (active pixels) : {subset.n_px}")
print(f"  sssig (intensity STD) : {subset.sssig:.2f}  (> 30 is healthy for DIC)")
print(f"  solved               : {subset.solved}")
"""),

    code("""\
# Inspect the reference patch before solving
subset.inspect()
"""),

    code("""\
# Solve with ICGN (Inverse Compositional Gauss-Newton) — the standard algorithm
subset.solve(algorithm="icgn", max_norm=1e-3, max_iterations=50, tolerance=0.75)
print(subset)
print(f"\\n  p      : {[f'{v:.5f}' for v in subset.p]}")
print(f"  u      : {subset.p[0]:.4f} px   (x-displacement)")
print(f"  v      : {subset.p[3]:.4f} px   (y-displacement)")
print(f"  C_ZNCC : {subset.c_zncc:.6f}  (1.0 = perfect match)")
print(f"  iters  : {subset.iterations}")
print(f"  converged : {subset.converged}")
"""),

    md("""\
### ICGN vs FAGN

`geopyv_dev` implements two DIC algorithms:

| Algorithm | Full name | Notes |
|-----------|-----------|-------|
| `"icgn"` | Inverse Compositional Gauss-Newton | Faster — Hessian computed once |
| `"fagn"` | Forward Additive Gauss-Newton | Slower — Hessian rebuilt each iteration |

Both converge to the same answer for well-textured subsets.  FAGN can sometimes
recover from poor initial guesses that ICGN struggles with.
"""),

    code("""\
# Convergence animation — replay the iteration history
# history = list of (iteration_no, C_ZNCC, C_ZNSSD, delta_norm)
history = subset.history
iters   = [h[0] for h in history]
zncc    = [h[1] for h in history]
norms   = [h[3] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.set_xlim(-0.5, max(iters) + 0.5); ax1.set_ylim(min(zncc)*0.999, 1.0005)
ax2.set_xlim(-0.5, max(iters) + 0.5); ax2.set_ylim(0, max(norms)*1.15)

for ax, ylabel in [(ax1, "C_ZNCC"), (ax2, "Δ norm (convergence)")]:
    ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
    ax.axhline(0 if ax is ax2 else 1.0, ls=":", c="grey", alpha=0.5)

line1, = ax1.plot([], [], "o-", c="steelblue", ms=5, label="C_ZNCC")
line2, = ax2.plot([], [], "o-", c="crimson",   ms=5, label="Δ norm")
dot1   = ax1.plot([], [], "o", c="steelblue", ms=10)[0]
dot2   = ax2.plot([], [], "o", c="crimson",   ms=10)[0]
ax1.axhline(0.75, ls="--", c="orange", lw=1.2, label="tolerance=0.75")
ax1.legend(fontsize=9); ax2.legend(fontsize=9)

def update(frame):
    i = frame + 1
    line1.set_data(iters[:i], zncc[:i]);  dot1.set_data([iters[i-1]], [zncc[i-1]])
    line2.set_data(iters[:i], norms[:i]); dot2.set_data([iters[i-1]], [norms[i-1]])
    return line1, line2, dot1, dot2

anim = FuncAnimation(fig, update, frames=len(history), interval=250, blit=True)
plt.suptitle("ICGN convergence history", fontsize=12, fontweight="bold")
plt.tight_layout()
anim.save("../docs/assets/nb2_convergence.gif", writer="pillow", fps=5)
HTML(anim.to_jshtml())
"""),

    code("""\
# Compare ICGN and FAGN on the same subset
subset_fagn = gp.Subset(coord=[200.0, 200.0], local_mask=local_mask,
                         f_img=f_img, g_img=g_img)
subset_fagn.solve(algorithm="fagn", max_norm=1e-3, max_iterations=50)

print(f"ICGN: p[0]={subset.p[0]:.5f}  iters={subset.iterations}  C_ZNCC={subset.c_zncc:.6f}")
print(f"FAGN: p[0]={subset_fagn.p[0]:.5f}  iters={subset_fagn.iterations}  C_ZNCC={subset_fagn.c_zncc:.6f}")
"""),

    code("""\
# Save to disk and reload
subset.save("data/subset.pyv")

loaded = gp.load("data/subset.pyv")
print(f"Loaded: {loaded}")
print(f"  p[0]   : {loaded.p[0]:.5f}")
print(f"  C_ZNCC : {loaded.c_zncc:.6f}")
"""),

    md("""\
## Key takeaways

- `p[0]` = u-displacement, `p[3]` = v-displacement (order-1).
- `C_ZNCC` close to 1.0 means a reliable match; below ~0.7 is suspect.
- `converged` is `True` when the Δ norm dropped below `max_norm` within `max_iterations`.
- ICGN ≈ 5–10× faster than FAGN for well-conditioned problems.

**Next:** Notebook 03 — extending to full-field measurements with `Mesh`.
"""),

])

# ===========================================================================
# Notebook 3 — Mesh
# ===========================================================================

NB3 = nb([

    md("""\
# 03 — Mesh: full-field DIC

A `Mesh` covers an entire region of interest with a triangulated network of nodes.
Each node is a `Subset` centre — the solver correlates all of them using a
**reliability-guided** strategy: start from a "seed" node with a known good initial
guess, then propagate outward in order of correlation quality.

This notebook covers: `Mesh` construction, geometry, `solve()`, displacement
contours, ZNCC score maps, and save/load.
"""),

    code(COMMON_IMPORTS),

    code(GEN_SINGLE + "\n" + MESH_PARAMS),

    md("""\
## Constructing a `Mesh`

```python
mesh = gp.Mesh(
    boundary    = boundary,       # PathRegion, CircleRegion, or (N,2) array
    target_nodes = 80,            # approximate number of nodes (solver binary-searches for element size)
    f_img       = f_img,          # reference Image
    g_img       = g_img,          # target Image
    size        = (10., 80.),     # (min, max) element edge length in pixels
    exclusions  = [excl_region],  # optional holes in the mesh
    mesh_order  = 2,              # 1 = linear triangles; 2 = quadratic (mid-side nodes)
)
```

The mesh geometry is computed immediately at construction using a constrained
Delaunay triangulation (spade).  No correlation happens yet.

### Linear vs quadratic elements

| `mesh_order` | Nodes per element | DOF per node | Best for |
|---|---|---|---|
| 1 | 3 (vertices) | 6 | Large deformations, coarse grids |
| 2 | 6 (vertices + mid-sides) | 12 | Strain gradients, fine grids |
"""),

    code("""\
# Build two meshes: linear (order 1) and quadratic (order 2)
mesh1 = gp.Mesh(boundary=boundary, target_nodes=60, f_img=f_img, g_img=g_img,
                 size=(15., 70.), mesh_order=1)
mesh2 = gp.Mesh(boundary=boundary, target_nodes=60, f_img=f_img, g_img=g_img,
                 size=(15., 70.), mesh_order=2)

print(f"Order-1  : {mesh1}")
print(f"Order-2  : {mesh2}")
print(f"\\nOrder-1 has {mesh1.nodes.shape[0]} nodes, {mesh1.elements.shape[0]} elements")
print(f"Order-2 has {mesh2.nodes.shape[0]} nodes, {mesh2.elements.shape[0]} elements")
"""),

    code("""\
# Visualise the mesh geometry before solving
def plot_mesh_geometry(mesh, ax, title, img):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, alpha=0.55)
    nodes = mesh.nodes
    elems = mesh.elements
    # Draw element edges
    for elem in elems:
        corners = elem[:3]  # first 3 nodes are always vertices
        tri_pts = nodes[corners]
        tri_closed = np.vstack([tri_pts, tri_pts[0]])
        ax.plot(tri_closed[:, 0], tri_closed[:, 1], "b-", lw=0.6, alpha=0.6)
    # Draw nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], s=12, c="crimson", zorder=5, linewidths=0)
    # Mark boundary nodes
    bn = nodes[mesh.boundary]
    ax.scatter(bn[:, 0], bn[:, 1], s=30, c="navy", marker="s", zorder=6,
               linewidths=0, label="boundary")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    ax.legend(fontsize=8, loc="lower right")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_mesh_geometry(mesh1, ax1, f"Linear mesh  (order 1)  –  {mesh1.nodes.shape[0]} nodes",
                   f_img.image_gs)
plot_mesh_geometry(mesh2, ax2, f"Quadratic mesh (order 2) –  {mesh2.nodes.shape[0]} nodes",
                   f_img.image_gs)
plt.tight_layout()
os.makedirs("../docs/assets", exist_ok=True)
fig.savefig("../docs/assets/nb3_mesh_geometry.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
## Reliability-guided propagation

The solver does **not** correlate all nodes independently.  Instead it:

1. Solves the **seed node** (the node closest to `seed_coord`) with a tight tolerance.
2. Ranks all unsolved neighbours by their C_ZNCC (best first).
3. Uses the best neighbour's warp as an initial guess for the next node.
4. Repeats until all nodes are solved or fail the `tolerance` threshold.

This dramatically improves robustness — each node gets a warm start from an already-converged
neighbour rather than starting from zero.

The animation below illustrates the concept.
"""),

    code("""\
# Animated propagation concept
# We approximate the solve order by BFS distance from the seed node.
import collections

nodes_arr = mesh1.nodes
elems_arr = mesh1.elements

# Build adjacency list (vertex-to-vertex)
n_nodes = nodes_arr.shape[0]
adj = collections.defaultdict(set)
for elem in elems_arr:
    verts = list(elem[:3])
    for i, a in enumerate(verts):
        for b in verts[i+1:]:
            adj[a].add(b); adj[b].add(a)

# BFS from seed node
seed_node = int(np.argmin(np.linalg.norm(nodes_arr - np.array([200., 200.]), axis=1)))
order, visited, queue = [], {seed_node}, collections.deque([seed_node])
while queue:
    n = queue.popleft()
    order.append(n)
    for nb_n in sorted(adj[n]):
        if nb_n not in visited:
            visited.add(nb_n); queue.append(nb_n)

# Animate nodes appearing in BFS order
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(f_img.image_gs, cmap="gray", vmin=0, vmax=255, alpha=0.5)
for elem in elems_arr:
    tri = nodes_arr[elem[:3]]
    ax.plot(*np.append(tri, [tri[0]], axis=0).T, "b-", lw=0.5, alpha=0.35)

scatter_solved  = ax.scatter([], [], s=35, c="seagreen",  zorder=5, linewidths=0, label="solved")
scatter_pending = ax.scatter(nodes_arr[:, 0], nodes_arr[:, 1],
                              s=35, c="#bbb", zorder=4, linewidths=0, label="pending")
seed_mark = ax.scatter([nodes_arr[seed_node, 0]], [nodes_arr[seed_node, 1]],
                        s=100, c="crimson", zorder=6, marker="*", label="seed")

frame_label = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                       va="top", fontsize=10, color="black",
                       bbox=dict(fc="white", ec="none", alpha=0.7))
ax.legend(fontsize=9, loc="lower right")
ax.axis("off")
ax.set_title("Reliability-guided propagation (BFS order)", fontsize=11, fontweight="bold")

STEP = max(1, len(order) // 40)   # ~40 animation frames

def update_prop(frame):
    n_done = min((frame + 1) * STEP, len(order))
    done = order[:n_done]
    pending = order[n_done:]
    scatter_solved.set_offsets(nodes_arr[done])
    scatter_pending.set_offsets(nodes_arr[pending] if pending else np.empty((0, 2)))
    frame_label.set_text(f"{n_done}/{len(order)} nodes solved")
    return scatter_solved, scatter_pending, frame_label

anim_prop = FuncAnimation(fig, update_prop, frames=40, interval=120, blit=True)
plt.tight_layout()
anim_prop.save("../docs/assets/nb3_propagation.gif", writer="pillow", fps=10)
HTML(anim_prop.to_jshtml())
"""),

    code("""\
# Solve the mesh  (takes a few seconds)
mesh1.solve(
    local_mask    = local_mask,
    seed_coord    = seed_coord,
    seed_warp     = [0.0] * 6,
    subset_order  = 1,
    tolerance     = 0.75,
    seed_tolerance= 0.9,
    method        = "icgn",
)
print(mesh1)
print(f"\\n  mean C_ZNCC : {mesh1.c_zncc.mean():.4f}")
print(f"  min  C_ZNCC : {mesh1.c_zncc.min():.4f}")
print(f"  seed node   : {mesh1.seed_node}")
"""),

    code("""\
# Displacement field — u (horizontal) and v (vertical)
nodes = mesh1.nodes
disps = mesh1.displacements

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
titles = ["u-displacement (px)", "v-displacement (px)", "C_ZNCC score"]
data   = [disps[:, 0], disps[:, 1], mesh1.c_zncc]
cmaps  = ["RdBu_r", "RdBu_r", "RdYlGn"]

for ax, d, title, cmap in zip(axes, data, titles, cmaps):
    ax.imshow(f_img.image_gs, cmap="gray", vmin=0, vmax=255, alpha=0.4)
    sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=d, cmap=cmap, s=30,
                    linewidths=0, vmin=d.min(), vmax=d.max())
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11, fontweight="bold"); ax.axis("off")

plt.suptitle("Mesh solution — colour by per-node value", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig("../docs/assets/nb3_displacement.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    code("""\
# Save and reload the mesh
mesh1.save("data/mesh.pyv")
mesh_loaded = gp.load("data/mesh.pyv")
print(f"Loaded: {mesh_loaded}")
print(f"  mean C_ZNCC : {mesh_loaded.c_zncc.mean():.4f}  (matches original)")
"""),

    md("""\
## Key takeaways

- `target_nodes` controls mesh density; `size` bounds element edge lengths.
- `mesh_order=2` (quadratic) uses more nodes but captures strain gradients better.
- Reliability-guided propagation makes the solver robust to initial-guess errors.
- Always check `mesh.c_zncc.min()` — values below ~0.7 indicate problem nodes.

**Next:** Notebook 04 — extending to a full image sequence with `Sequence`.
"""),

])

# ===========================================================================
# Notebook 4 — Sequence
# ===========================================================================

NB4 = nb([

    md("""\
# 04 — Sequence: multi-image time series

A `Sequence` strings together many image pairs and solves them efficiently,
re-using mesh geometry and warp guesses between frames.

This notebook covers: `Sequence`, `SequenceOptions` (guide / sequential / sync),
per-pair data access, and an animated displacement history.
"""),

    code(COMMON_IMPORTS),

    code(GEN_SEQUENCE + "\n" + """\
# Mesh configuration shared across all pairs
boundary_nodes = np.array([
    [50.0, 50.0], [50.0, 350.0], [350.0, 350.0], [350.0, 50.0]
])
boundary  = gp.PathRegion(nodes=boundary_nodes, hard=False)
local_mask = gp.Mask(mask_type="local", shape="circle", size=25)
seed_coord = [200.0, 200.0]
"""),

    md("""\
## `SequenceOptions` — how pairs connect

`SequenceOptions` controls how information flows from one pair to the next:

| Option | Default | Effect |
|--------|---------|--------|
| `guide` | `True` | Use the previous pair's warp field to seed the next pair's solve |
| `sequential` | `False` | Advance the reference image after each successful pair |
| `sync` | `True` | Re-use the previous mesh geometry (skip CDT) for subsequent pairs |
| `override_` | `False` | Relax tolerance on retry after a failed consecutive pair |

`sync=True` is almost always what you want — it saves time and keeps the node
numbering consistent across pairs so you can directly index `sequence.displacements(i)`.
"""),

    code("""\
# Show SequenceOptions defaults
opts = gp.SequenceOptions()
print(opts)

# For a standard analysis: guide warp + sync geometry
opts_standard = gp.SequenceOptions(guide=True, sync=True, sequential=False)
print(opts_standard)
"""),

    code("""\
# Build the Sequence (discovers all images in data/sequence/ automatically)
sequence = gp.Sequence(
    image_dir   = "data/sequence",
    boundary    = boundary,
    target_nodes= 60,
    size        = (15., 70.),
    mesh_order  = 1,
)
print(f"{sequence}")
print(f"  Image paths found: {sequence.n_pairs + 1}")
for p in sequence.image_paths[:3]:
    print(f"    {os.path.basename(p)}")
print("    …")
"""),

    code("""\
# Solve all pairs  (a few seconds per pair)
sequence.solve(
    local_mask    = local_mask,
    seed_coord    = seed_coord,
    seed_warp     = [0.0] * 6,
    subset_order  = 1,
    tolerance     = 0.75,
    seed_tolerance= 0.9,
    method        = "icgn",
    options       = gp.SequenceOptions(guide=True, sync=True),
)
print(sequence)
print(f"  solved     : {sequence.solved}")
print(f"  n_pairs    : {sequence.n_pairs}")
print(f"  pair 0 mean C_ZNCC : {sequence.c_zncc(0).mean():.4f}")
"""),

    code("""\
# Access per-pair data without loading full Mesh objects
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, ax in enumerate(axes.flat):
    nodes = sequence.nodes(idx)
    u     = sequence.displacements(idx)[:, 0]   # x-displacements
    zncc  = sequence.c_zncc(idx)
    # Quick per-node scatter coloured by u
    sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=u, cmap="RdBu_r",
                    s=35, linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.046, label="u (px)")
    ax.set_title(f"Pair {idx}: u-displacement", fontsize=11, fontweight="bold")
    ax.set_aspect("equal"); ax.set_xlim(0, 401); ax.set_ylim(0, 401)
    ax.invert_yaxis(); ax.axis("off")

plt.suptitle("u-displacement for each image pair", fontsize=13, fontweight="bold")
plt.tight_layout()
os.makedirs("../docs/assets", exist_ok=True)
fig.savefig("../docs/assets/nb4_displacement_grid.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    code("""\
# Animated u-displacement across pairs
nodes = sequence.nodes(0)
all_u = [sequence.displacements(i)[:, 0] for i in range(sequence.n_pairs)]
vmin, vmax = min(u.min() for u in all_u), max(u.max() for u in all_u)

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=all_u[0], cmap="RdBu_r",
                s=40, linewidths=0, vmin=vmin, vmax=vmax)
cb = plt.colorbar(sc, ax=ax, fraction=0.046, label="u-displacement (px)")
title = ax.set_title("Pair 0: u-displacement", fontsize=11, fontweight="bold")
ax.set_aspect("equal"); ax.set_xlim(0, 401); ax.set_ylim(0, 401)
ax.invert_yaxis(); ax.axis("off")

def update_seq(frame):
    sc.set_array(all_u[frame])
    title.set_text(f"Pair {frame}: u-displacement  (mean={all_u[frame].mean():.3f} px)")
    return sc, title

anim_seq = FuncAnimation(fig, update_seq, frames=sequence.n_pairs, interval=600, blit=True)
plt.tight_layout()
anim_seq.save("../docs/assets/nb4_sequence_animation.gif", writer="pillow", fps=2)
HTML(anim_seq.to_jshtml())
"""),

    code("""\
# Save and reload
sequence.save("data/sequence.pyv")
seq_loaded = gp.load("data/sequence.pyv")
print(f"Loaded: {seq_loaded}")
print(f"  solved  : {seq_loaded.solved}")
print(f"  n_pairs : {seq_loaded.n_pairs}")
print(f"  pair 0 mean C_ZNCC : {seq_loaded.c_zncc(0).mean():.4f}")
"""),

    md("""\
## Key takeaways

- `Sequence` auto-discovers images from a directory by trailing integer sort.
- `SequenceOptions(guide=True, sync=True)` is the recommended default.
- Access per-pair arrays directly via `sequence.nodes(i)`, `sequence.displacements(i)`,
  `sequence.c_zncc(i)` — faster than `sequence.mesh_solutions[i]` for large sequences.
- The `.pyv` format stores all pairs compactly; reload with `gp.load()`.

**Next:** Notebook 05 — tracking material points through time with `Particle` and `Field`.
"""),

])

# ===========================================================================
# Notebook 5 — Particle & Field
# ===========================================================================

NB5 = nb([

    md("""\
# 05 — Particle & Field: strain-path tracking

The `Sequence` gives you displacements at **fixed mesh nodes** for each image pair.
`Particle` and `Field` go further: they integrate those displacements along
**material point trajectories** to build up the full strain history.

This notebook covers: Lagrangian vs Eulerian, `Particle`, `Field`, strain paths,
volumetric strain, and animated tracking.
"""),

    code(COMMON_IMPORTS),

    code("""\
# Reload the sequence saved in notebook 04
# (If you haven't run notebook 04, generate + solve the sequence first.)
try:
    sequence = gp.load("data/sequence.pyv")
    print(f"Loaded: {sequence}")
except FileNotFoundError:
    print("Running sequence generation and solve (first time)...")
    import os, numpy as np
    os.makedirs("data/sequence", exist_ok=True)
    _sp = gp.Speckle(
        image_cfg={"image_dir": "data/sequence", "name": "seq", "image_size": (401, 401)},
        speckle_cfg={"speckle_size": 3.5, "speckle_number": 700},
        progression="deformation",
        deformation_cfg={"comp": [8.0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          "mode": "SB", "option": "sin", "width": 100.0},
        noise_cfg=(0.2, 1.5),
        scale_cfg={"scale": "lin", "n": 5},
    )
    _sp.solve(seed=42)
    boundary = gp.PathRegion(nodes=np.array([[50.,50.],[50.,350.],[350.,350.],[350.,50.]]),
                              hard=False)
    sequence = gp.Sequence(image_dir="data/sequence", boundary=boundary,
                            target_nodes=60, size=(15., 70.), mesh_order=1)
    sequence.solve(local_mask=gp.Mask(mask_type="local", shape="circle", size=25),
                   seed_coord=[200., 200.], subset_order=1, tolerance=0.75,
                   options=gp.SequenceOptions(guide=True, sync=True))
    sequence.save("data/sequence.pyv")
    print(f"Done: {sequence}")
"""),

    md("""\
## Lagrangian vs Eulerian tracking

DIC gives you displacement vectors at **fixed node positions** for each frame.
There are two ways to integrate these into a strain path:

### Lagrangian (track=True — default)
The particle **moves with the material**.  Its coordinate updates after each increment
using the interpolated displacement at its current position.

```
frame 0:  x₀ = [200, 200]
frame 1:  x₁ = x₀ + u(x₀)   ←  displacement evaluated at x₀
frame 2:  x₂ = x₁ + u(x₁)   ←  displacement evaluated at new position x₁
```

### Eulerian (track=False)
The particle **stays fixed in space**.  The displacement at its original position
is evaluated for every frame.

```
frame 0:  x₀ = [200, 200]
frame 1:  u(x₀)
frame 2:  u(x₀)   ←  always the same reference point
```

The animation below illustrates the difference.
"""),

    code("""\
# Conceptual Lagrangian vs Eulerian diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")

FRAMES = 4
DX = 6.0   # displacement per frame (px, simplified)

for ax, mode, color in [(ax1, "Lagrangian (track=True)", "seagreen"),
                         (ax2, "Eulerian  (track=False)", "royalblue")]:
    x = np.linspace(40, 360, 8)
    for xi in x:
        for frame in range(FRAMES + 1):
            alpha = 0.15 + 0.17 * frame
            ax.add_patch(patches.Circle((xi + frame * DX, 200 - frame * 1.5), 8,
                                         fc="grey", ec="none", alpha=alpha * 0.3))

    # Draw one tracked particle's path
    x0 = 200.
    if mode.startswith("L"):
        xs = [x0 + i * DX for i in range(FRAMES + 1)]
        ys = [200. - i * 1.5 for i in range(FRAMES + 1)]
    else:
        xs = [x0] * (FRAMES + 1)
        ys = [200.] * (FRAMES + 1)

    ax.plot(xs, ys, "o-", c=color, ms=10, lw=2, zorder=5)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        ax.text(xi + 3, yi - 10, f"$t_{i}$", fontsize=9, color=color)

    ax.set_xlim(0, 410); ax.set_ylim(150, 250)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(mode, fontsize=11, fontweight="bold", color=color)

plt.suptitle("Lagrangian vs Eulerian particle tracking", fontsize=12, fontweight="bold")
plt.tight_layout()
os.makedirs("../docs/assets", exist_ok=True)
fig.savefig("../docs/assets/nb5_lag_eul.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
## The `Particle` class

```python
particle = gp.Particle(
    source     = sequence,          # solved Sequence or Mesh
    coordinate = [x, y],           # initial position [x, y]
    track      = True,             # True = Lagrangian, False = Eulerian
)
particle.solve(
    factor    = 0.0,               # volumetric correction factor
    true_incs = True,              # logarithmic strain increments
)
```

After solving, `particle.coordinates` has shape `(n_frames+1, 2)` — one row per frame
including the initial position.
"""),

    code("""\
# Create a Lagrangian particle at the centre of the ROI
particle_lag = gp.Particle(source=sequence, coordinate=[200.0, 200.0], track=True)
particle_lag.solve(factor=0.0, true_incs=True)
print(particle_lag)
print(f"\\n  coordinates shape : {particle_lag.coordinates.shape}")
print(f"  warps shape       : {particle_lag.warps.shape}")
print(f"  strains shape     : {particle_lag.strains.shape}")

# Eulerian version at the same location
particle_eul = gp.Particle(source=sequence, coordinate=[200.0, 200.0], track=False)
particle_eul.solve(factor=0.0, true_incs=True)
print(f"\\nEulerian particle: {particle_eul}")
"""),

    code("""\
# Plot coordinate trajectory and strain history
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Trajectory (Lagrangian only)
coords = particle_lag.coordinates   # shape (n_frames+1, 2)
axes[0].plot(coords[:, 0], coords[:, 1], "o-", c="seagreen", ms=7, lw=1.5)
for i, (xi, yi) in enumerate(coords):
    axes[0].text(xi + 1, yi - 2, str(i), fontsize=9, color="seagreen")
axes[0].set_title("Lagrangian trajectory", fontsize=11, fontweight="bold")
axes[0].set_xlabel("x (px)"); axes[0].set_ylabel("y (px)")
axes[0].invert_yaxis()

# u-displacement over time: Lagrangian vs Eulerian
# warps shape: (n_frames+1, warp_len) — row 0 is initial; rows 1..N are post-solve
warp_frames = range(particle_lag.warps.shape[0])
u_lag = particle_lag.warps[:, 0]
u_eul = particle_eul.warps[:, 0]
axes[1].plot(warp_frames, u_lag, "o-", c="seagreen", label="Lagrangian", ms=6)
axes[1].plot(warp_frames, u_eul, "s--", c="royalblue", label="Eulerian", ms=6)
axes[1].set_title("u-displacement history", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Frame"); axes[1].set_ylabel("u (px)"); axes[1].legend()

# Strain history: one row per solved increment
strains_lag = particle_lag.strains   # shape (n_frames, n_components)
n_str = strains_lag.shape[0]
strain_frames = list(range(1, n_str + 1))   # frames 1 … n_frames
axes[2].plot(strain_frames, strains_lag[:, 0], "o-", c="seagreen", label="exx Lag.")
axes[2].plot(strain_frames, strains_lag[:, 3], "s--", c="crimson",  label="eyy Lag.")
axes[2].set_title("Strain components", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Frame"); axes[2].set_ylabel("Strain"); axes[2].legend()

plt.suptitle("Particle solution", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig("../docs/assets/nb5_particle.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("""\
## The `Field` class

`Field` places **many particles** across the mesh simultaneously and solves all of
them in one call.  By default it distributes particles at element centroids
(one per element) — use `gp.field_distribute_particles()` or supply explicit coordinates.

```python
field = gp.Field(sequence_solution=sequence, track=True, depth=1.0)
field.solve(factor=0.0, true_incs=True)
```

After solving, `field.particles` gives you a list of `ParticleSolution` objects,
one per particle.
"""),

    code("""\
# Distribute particles at element centroids of pair-0 mesh
m0 = sequence.mesh_solution_at(0)
coords, volumes = gp.field_distribute_particles(
    nodes=m0.nodes, elements=m0.elements, depth=1.0
)
print(f"Distributed {len(volumes)} particles across {m0.elements.shape[0]} elements")

field = gp.Field(sequence_solution=sequence, track=True, depth=1.0,
                  coordinates=coords, volumes=volumes)
field.solve(factor=0.0, true_incs=True)
print(field)
print(f"  n_particles : {field.n_particles}")
print(f"  inc_no      : {field.inc_no}")
"""),

    code("""\
# Final-frame volumetric strain map
particles = field.particles

# Each particle has accumulated strain; take the last increment
final_vol_strains = np.array([p.vol_strains[-1] for p in particles])
init_coords = field.coordinates   # (N, 2) initial positions

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sc1 = ax1.scatter(init_coords[:, 0], init_coords[:, 1],
                   c=final_vol_strains, cmap="RdBu_r", s=40, linewidths=0)
plt.colorbar(sc1, ax=ax1, fraction=0.046, label="εᵥₒₗ")
ax1.set_title(f"Volumetric strain — frame {field.inc_no}", fontsize=11, fontweight="bold")
ax1.set_aspect("equal"); ax1.set_xlim(0, 401); ax1.set_ylim(0, 401); ax1.invert_yaxis()
ax1.axis("off")

# u-displacement at final frame per particle
final_u = np.array([p.warps[-1, 0] for p in particles])
sc2 = ax2.scatter(init_coords[:, 0], init_coords[:, 1],
                   c=final_u, cmap="RdBu_r", s=40, linewidths=0)
plt.colorbar(sc2, ax=ax2, fraction=0.046, label="u (px)")
ax2.set_title("u-displacement — final frame", fontsize=11, fontweight="bold")
ax2.set_aspect("equal"); ax2.set_xlim(0, 401); ax2.set_ylim(0, 401); ax2.invert_yaxis()
ax2.axis("off")

plt.suptitle("Field solution — per-particle quantities", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig("../docs/assets/nb5_field.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    code("""\
# Animated: volumetric strain evolving over all frames
all_vol_strains = np.array([[p.vol_strains[t] for p in particles]
                             for t in range(field.inc_no)])   # shape (n_frames, n_particles)
vmin_v, vmax_v = all_vol_strains.min(), all_vol_strains.max()

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(init_coords[:, 0], init_coords[:, 1],
                c=all_vol_strains[0], cmap="RdBu_r", s=40, linewidths=0,
                vmin=vmin_v, vmax=vmax_v)
cb = plt.colorbar(sc, ax=ax, fraction=0.046, label="εᵥₒₗ")
ttl = ax.set_title("Frame 1: volumetric strain", fontsize=11, fontweight="bold")
ax.set_aspect("equal"); ax.set_xlim(0, 401); ax.set_ylim(0, 401); ax.invert_yaxis()
ax.axis("off")

def update_field(frame):
    sc.set_array(all_vol_strains[frame])
    ttl.set_text(f"Increment {frame + 1}: volumetric strain")
    return sc, ttl

anim_f = FuncAnimation(fig, update_field, frames=field.inc_no, interval=600, blit=True)
plt.tight_layout()
anim_f.save("../docs/assets/nb5_vol_strain.gif", writer="pillow", fps=2)
HTML(anim_f.to_jshtml())
"""),

    code("""\
# Save and reload
field.save("data/field.pyv")
field_loaded = gp.load("data/field.pyv")
print(f"Loaded: {field_loaded}")
print(f"  n_particles : {field_loaded.n_particles}")
print(f"  inc_no      : {field_loaded.inc_no}")
"""),

    md("""\
## Summary: the full workflow

```
gp.Speckle(...)          → generate synthetic images (or use real ones)
gp.Image(filepath=...)   → load images

gp.Mask(mask_type="local", shape="circle", size=25)    → subset template

gp.Sequence(image_dir, boundary, target_nodes, ...)
    .solve(local_mask, seed_coord, ...)               → correlation over time

gp.Field(sequence, track=True, coordinates=coords, volumes=vols)
    .solve()                                          → strain paths for all particles

gp.save("result.pyv", field)   /   gp.load("result.pyv")
```

From here you can explore:
- **Calibration** (`gp.CalibrationParams`) to convert pixel displacements to physical units.
- **Validation** (`gp.Validation`) against known synthetic deformations.
- **Visualisation** via `geopyv_dev.plots` — `contour_field`, `history_field`, `trace_field`.
"""),

])

# ===========================================================================
# Write all notebooks
# ===========================================================================

notebooks = [
    ("tutorials/00_introduction.ipynb",      NB0),
    ("tutorials/01_images_and_masks.ipynb",  NB1),
    ("tutorials/02_subset.ipynb",            NB2),
    ("tutorials/03_mesh.ipynb",              NB3),
    ("tutorials/04_sequence.ipynb",          NB4),
    ("tutorials/05_particle_and_field.ipynb",NB5),
]

print("Writing tutorial notebooks …")
for path, notebook in notebooks:
    save_nb(path, notebook)
print(f"\nDone — {len(notebooks)} notebooks written to tutorials/")
