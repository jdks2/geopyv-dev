"""Build the gif/figure assets for docs/tutorials/01_images_and_masks.rst.

Generates:
- quad_deformation.gif — u / rotation / dv/dy / shear, each looping seamlessly
  forever, from four small dedicated Speckle series (kept separate from
  images/shear so regenerating this thumbnail never touches the canonical
  tutorial dataset).
- mask_local_vs_global.png — static comparison of a local mask (patch shape
  around a subset centre) vs a global mask (binary image-space region).

Each `quad_deformation.gif` panel is rendered frame-by-frame: every frame is
its own tiny ``Speckle(scale_cfg={"n": 2})`` instance (image 0 = undeformed
reference, image 1 = the fully-scaled deformation for that instant), built
with the same RNG seed every time so every frame starts from the identical
base speckle point cloud. `Speckle` renders each output image directly from
that one fixed reference cloud rather than chaining frame-to-frame (see
`src/speckle.rs::solve` — `compute_displaced` always takes the same
`positions` view), so a deformation value that returns to its starting value
reproduces the starting frame exactly, which is what makes each panel loop
with no visible seam or snap-back.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO_ROOT, GIF_DIR, IMG_DIR, fig_to_rgba, save_gif, save_png, load_downsampled

sys.path.insert(0, REPO_ROOT)
import geopyv_dev as gp

QUAD_DIR = "/tmp/geopyv_docs_quad_speckle"
QUAD_SIZE = 400
DISPLAY_SIZE = 180           # panel thumbnail size after downsampling
CANVAS = 1.9
DPI = 130                    # -> crisp panel titles/labels
N_FRAMES = 32

# Tuned so rendered coverage (~67% of pixels above a mid-grey threshold)
# visually matches the canonical images/shear series at this canvas size —
# naive linear scaling of images/shear's own (speckle_size=10,
# speckle_number=7000 @ 1001x1001) undershoots badly, since coverage grows
# roughly with speckle_size**2, not linearly.
SPECKLE_NUMBER = 2400
SPECKLE_SIZE = 7.0

SEED = 7
ROT_MAX = 2 * np.pi   # one full turn per loop
DVDY_MAX = 0.08
SHEAR_MAX = 0.06


def _render_frame_raw(name, comp, angle=None):
    """Solve a fresh 2-image Speckle series and return its full-resolution
    frame 1 as a plain grayscale array (image 0 is always the undeformed
    reference; image 1 carries `comp`/`angle` at full strength, since
    scale_cfg n=2 gives mult=1 at index 1)."""
    os.makedirs(QUAD_DIR, exist_ok=True)
    deformation_cfg = {"comp": comp}
    if angle is not None:
        deformation_cfg["mode"] = "rotation"
        deformation_cfg["angle"] = angle
    speckle = gp.Speckle(
        image_cfg={"image_dir": QUAD_DIR, "name": name, "image_size": (QUAD_SIZE, QUAD_SIZE)},
        speckle_cfg={"speckle_size": SPECKLE_SIZE, "speckle_number": SPECKLE_NUMBER},
        progression="deformation",
        deformation_cfg=deformation_cfg,
        noise_cfg=(0.0, 0.0),
        scale_cfg={"scale": "lin", "n": 2},
    )
    speckle.solve(seed=SEED)
    from PIL import Image as _PILImage
    return np.array(_PILImage.open(os.path.join(QUAD_DIR, f"{name}_1.jpg")).convert("L"))


def _downsample_blur(arr, size=DISPLAY_SIZE, blur=0.5):
    from PIL import Image as _PILImage, ImageFilter
    img = _PILImage.fromarray(arr).resize((size, size), _PILImage.LANCZOS)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    return np.array(img)


def _render_frame(name, comp, angle=None):
    return _downsample_blur(_render_frame_raw(name, comp, angle=angle))


def _u_frames():
    # A pure x-translation would just slide points off one edge with nothing
    # replacing them on the other — not a seamless "conveyor belt". Instead,
    # render the same base point cloud (same seed) shifted by `s` and by
    # `s - QUAD_SIZE`, and take the elementwise max: for any point, exactly
    # one of those two shifts lands it back in [0, QUAD_SIZE), so together
    # they reconstruct a true periodic (wraparound) translation. At s=0 the
    # second render is a full-width-away shift (nothing visible), so frame 0
    # is the clean, undisturbed reference, and s=QUAD_SIZE is identical to
    # s=0 — the loop has no seam at any point in the cycle, not just at the
    # wrap instant.
    frames = []
    for i in range(N_FRAMES):
        s = QUAD_SIZE * i / N_FRAMES
        a = _render_frame_raw("u_a", comp=[s, 0.0] + [0.0] * 10)
        b = _render_frame_raw("u_b", comp=[s - QUAD_SIZE, 0.0] + [0.0] * 10)
        frames.append(_downsample_blur(np.maximum(a, b)))
    return frames


def _rot_frames():
    frames = []
    for i in range(N_FRAMES):
        angle = ROT_MAX * i / N_FRAMES
        frames.append(_render_frame("rot", comp=[0.0] * 12, angle=angle))
    return frames


def _dvdy_frames():
    # Raised-cosine envelope: 0 -> max -> 0 once per loop (squash and relax).
    frames = []
    for i in range(N_FRAMES):
        mult = 0.5 * (1 - np.cos(2 * np.pi * i / N_FRAMES))
        comp = [0.0] * 12
        comp[5] = DVDY_MAX * mult  # v_y = dv/dy
        frames.append(_render_frame("dvdy", comp=comp))
    return frames


def _shear_frames():
    # Sine envelope: smoothly reverses sign each half-cycle.
    frames = []
    for i in range(N_FRAMES):
        mult = np.sin(2 * np.pi * i / N_FRAMES)
        comp = [0.0] * 12
        comp[3] = SHEAR_MAX * mult  # v_x = dv/dx
        comp[4] = SHEAR_MAX * mult  # u_y = du/dy
        frames.append(_render_frame("shear", comp=comp))
    return frames


def build_quad_gif():
    panels = {
        "u": _u_frames(),
        "rot": _rot_frames(),
        "dvdy": _dvdy_frames(),
        "shear": _shear_frames(),
    }
    frames = []
    for i in range(N_FRAMES):
        fig, axes = plt.subplots(2, 2, figsize=(CANVAS * 2, CANVAS * 2), dpi=DPI)
        for ax, (label, imgs) in zip(axes.flat, panels.items()):
            ax.imshow(imgs[i], cmap="gist_gray")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(label, fontsize=11, pad=3)
        plt.tight_layout(pad=0.3)
        frames.append(fig_to_rgba(fig))
        plt.close(fig)
    out = os.path.join(GIF_DIR, "quad_deformation.gif")
    save_gif(frames, out, duration_ms=70, colors=40)
    return out


def build_mask_comparison():
    f_img = gp.Image(filepath=os.path.join(REPO_ROOT, "images", "shear", "shear_0.jpg"))
    img = load_downsampled(os.path.join(REPO_ROOT, "images", "shear", "shear_0.jpg"), size=400, blur=1.0)

    boundary_nodes = np.array([
        [250.0, 250.0], [250.0, 750.0], [750.0, 750.0], [750.0, 250.0],
    ])
    excl_nodes = np.array([
        [560.0, 560.0], [560.0, 680.0], [680.0, 680.0], [680.0, 560.0],
    ])
    global_mask = gp.Mask(
        mask_type="global",
        f_img=f_img,
        boundary=gp.PathRegion(nodes=boundary_nodes, hard=True),
        exclusions=[gp.PathRegion(nodes=excl_nodes, hard=True)],
    )
    binary = np.asarray(global_mask.binary)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.8), dpi=200)

    ax = axes[0]
    ax.imshow(img, cmap="gist_gray", extent=(0, 1001, 1001, 0))
    cx, cy, r = 500.0, 500.0, 90.0
    circle = plt.Circle((cx, cy), r, fill=False, color="#d62728", linewidth=2.5)
    square = plt.Rectangle((cx - r, cy - r), 2 * r, 2 * r, fill=False, color="#1f77b4", linewidth=2.5)
    ax.add_patch(circle)
    ax.add_patch(square)
    ax.plot(cx, cy, "+", color="black", markersize=10, markeredgewidth=2)
    ax.set_xlim(300, 700); ax.set_ylim(700, 300)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("local mask\n(patch shape around a subset centre)", fontsize=9)

    ax = axes[1]
    ax.imshow(binary, cmap="gist_gray", extent=(0, 1001, 1001, 0), vmin=0, vmax=1)
    ax.set_xlim(150, 850); ax.set_ylim(850, 150)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("global mask\n(binary image-space region)", fontsize=9)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "mask_local_vs_global.png")
    save_png(fig, out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    build_quad_gif()
    build_mask_comparison()
