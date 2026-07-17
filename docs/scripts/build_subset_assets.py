"""Build the image assets for docs/tutorials/02_subset.rst.

Generates:
- subset_shape_order.gif — circle & square subsets growing/shrinking, then fixed
  and wiggling through 0th/1st/2nd order deformation.
- subset_quality.png — a real speckle field whose point density falls off from
  one corner (rather than an artificial flat patch pasted onto a uniform
  image), with a genuine well-textured Subset in the dense corner and a
  genuine poorly-textured one far from it, flanked by their real
  ``inspect()`` renders.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO_ROOT, GIF_DIR, IMG_DIR, PATH_COLOR, fig_to_rgba, save_gif, save_png, load_downsampled, hold

sys.path.insert(0, REPO_ROOT)
import geopyv_dev as gp

CANVAS = 2.2
DPI = 80
SHEAR_DIR = os.path.join(REPO_ROOT, "images", "shear")
CENTRE = [500.0, 500.0]
WINDOW = 160.0  # half-width of the cropped view around CENTRE, in source px


def _shape_points(shape, cx, cy, r, warp=None, n_pts=100):
    t = np.linspace(0, 2 * np.pi, n_pts)
    if shape == "circle":
        x, y = np.cos(t), np.sin(t)
    else:
        n = 8.0  # superellipse exponent -> rounded square
        x = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2.0 / n)
        y = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2.0 / n)
    px, py = cx + r * x, cy + r * y
    if warp is not None:
        dx, dy = warp(r * x, r * y)
        px, py = px + dx, py + dy
    return px, py


def _order0_warp(amp):
    def f(x, y):
        return np.full_like(x, amp), np.full_like(y, 0.4 * amp)
    return f


def _order1_warp(amp):
    def f(x, y):
        return amp * 0.02 * y, amp * 0.02 * x
    return f


def _order2_warp(amp):
    """A parabolic "banana" bend — curvature varies across the shape, which
    distinguishes 2nd-order bulging from 1st-order's uniform ellipse/skew."""
    def f(x, y):
        r = max(1.0, float(np.sqrt(x ** 2 + y ** 2).max()))
        dx = np.zeros_like(x)
        dy = amp * 1.1 * (x / r) ** 2
        return dx, dy
    return f


def build_shape_order_gif():
    img = load_downsampled(os.path.join(SHEAR_DIR, "shear_0.jpg"), size=280, blur=0.8)
    scale = 280.0 / (2 * WINDOW)  # display px per source px, image already cropped conceptually
    # We display the full downsampled image but zoom the axes to a window around CENTRE
    # in *source* pixel coordinates (image is 1001 source px downsampled to 280 display px).
    src_size = 1001.0
    disp = 280.0

    def to_disp(v):
        return v * disp / src_size

    cx, cy = to_disp(CENTRE[0]), to_disp(CENTRE[1])
    win = to_disp(WINDOW)

    frames, durations = [], []

    def render(r_c, r_s, warp_c=None, warp_s=None):
        fig, axes = plt.subplots(1, 2, figsize=(CANVAS * 2, CANVAS), dpi=DPI)
        for ax, shape, r, warp, title in (
            (axes[0], "circle", r_c, warp_c, "circle"),
            (axes[1], "square", r_s, warp_s, "square"),
        ):
            ax.imshow(img, cmap="gist_gray")
            px, py = _shape_points(shape, cx, cy, r, warp=warp)
            ax.plot(px, py, color=PATH_COLOR, linewidth=3.0)
            ax.plot(cx, cy, "+", color=PATH_COLOR, markersize=12, markeredgewidth=2.5)
            ax.set_xlim(cx - win, cx + win); ax.set_ylim(cy + win, cy - win)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(title, fontsize=10)
        plt.tight_layout(pad=0.3)
        frames.append(fig_to_rgba(fig))
        durations.append(80)
        plt.close(fig)

    # Grow / shrink breathing, one cycle.
    n_breathe = 10
    r0, r1 = to_disp(35), to_disp(85)
    for i in range(n_breathe):
        t = (i / n_breathe) * 2 * np.pi
        r = r0 + (r1 - r0) * (0.5 - 0.5 * np.cos(t))
        render(r, r)

    # Fixed radius from here on.
    r_fixed = to_disp(60)

    # Wiggle through order 0, 1, 2 — one oscillation cycle each.
    for warp_fn in (_order0_warp, _order1_warp, _order2_warp):
        n_wiggle = 10
        for i in range(n_wiggle):
            t = (i / n_wiggle) * 2 * np.pi
            amp = np.sin(t) * win * 0.35
            render(r_fixed, r_fixed, warp_c=warp_fn(amp), warp_s=warp_fn(amp))

    durations = hold(durations, 400)
    out = os.path.join(GIF_DIR, "subset_shape_order.gif")
    save_gif(frames, out, duration_ms=durations, colors=64)
    return out


QUALITY_SIZE = 1001
QUALITY_CORNER = np.array([0.0, 0.0])       # speckle density falls off from here
QUALITY_DECAY = 150.0                        # px — controls how fast density fades
QUALITY_N_CANDIDATES = 60_000
QUALITY_N_KEEP = 7_000
QUALITY_BLOB_SIGMA = 3.2
# Note: right at the corner itself, density saturates into near-solid white
# (overlapping blobs merge, killing contrast) — 250,250 is far enough out to
# sit in the "well-textured, distinct speckles" band rather than the
# oversaturated one right at the corner.
GOOD_COORD = [250.0, 250.0]
POOR_COORD = [860.0, 860.0]                   # far from the corner, sparse


def _corner_weighted_positions(seed=11):
    """Sample point positions whose density falls off exponentially with
    distance from QUALITY_CORNER — a genuine density gradient (not a pasted
    flat patch), so the resulting good/poor Subset metrics are computed by
    the real Rust code on real (if syntheticaly generated) image content."""
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(0.0, QUALITY_SIZE, size=(QUALITY_N_CANDIDATES, 2))
    dist = np.hypot(*(candidates - QUALITY_CORNER).T)
    weights = np.exp(-dist / QUALITY_DECAY)
    weights /= weights.sum()
    idx = rng.choice(QUALITY_N_CANDIDATES, size=QUALITY_N_KEEP, replace=False, p=weights)
    return candidates[idx]


def _splat_blobs(positions, size, sigma, amp=230.0):
    """Additively render soft Gaussian blobs at `positions` onto a `size`x`size`
    canvas — the same kind of blob speckle rendering `Speckle` does internally,
    reimplemented here in plain numpy so the density gradient above (which
    `Speckle`'s own uniform sampler doesn't support) can drive it."""
    canvas = np.zeros((size, size), dtype=np.float64)
    k = int(np.ceil(sigma * 3))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    kernel = amp * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    for x, y in positions:
        ix, iy = int(round(x)), int(round(y))
        x0, x1 = max(0, ix - k), min(size, ix + k + 1)
        y0, y1 = max(0, iy - k), min(size, iy + k + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        kx0, kx1 = x0 - (ix - k), x1 - (ix - k)
        ky0, ky1 = y0 - (iy - k), y1 - (iy - k)
        canvas[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1]
    return np.clip(canvas, 0, 255).astype(np.uint8)


def build_quality_figure():
    """A corner-concentrated speckle field with a genuinely well-textured
    Subset (in the dense corner) and a genuinely poorly-textured one (far
    from it), flanked by their real inspect() renders."""
    positions = _corner_weighted_positions()
    canvas = _splat_blobs(positions, QUALITY_SIZE, QUALITY_BLOB_SIGMA)

    from PIL import Image as PILImage
    # Persisted under images/, like images/shear/ — so the tutorial's code
    # block (which loads "images/quality/quality_speckle.jpg") is actually
    # runnable, not just illustrative.
    quality_img_dir = os.path.join(REPO_ROOT, "images", "quality")
    os.makedirs(quality_img_dir, exist_ok=True)
    img_path = os.path.join(quality_img_dir, "quality_speckle.jpg")
    PILImage.fromarray(canvas).save(img_path, quality=95)

    f_img = gp.Image(filepath=img_path)
    mask = gp.Mask(mask_type="local", shape="circle", size=45)
    good_sub = gp.Subset(coord=GOOD_COORD, local_mask=mask, f_img=f_img, g_img=f_img, subset_order=1)
    poor_sub = gp.Subset(coord=POOR_COORD, local_mask=mask, f_img=f_img, g_img=f_img, subset_order=1)

    disp_img = load_downsampled(img_path, size=500, blur=0.4)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), dpi=170,
                              gridspec_kw={"width_ratios": [1, 1.3, 1]})

    gp.plots.inspect_subset(poor_sub, ax=axes[0], show=False)
    axes[0].set_title("poorly-textured", fontsize=10)

    ax = axes[1]
    ax.imshow(disp_img, cmap="gist_gray", extent=(0, QUALITY_SIZE, QUALITY_SIZE, 0))
    for coord, label in ((GOOD_COORD, "good"), (POOR_COORD, "poor")):
        circle = plt.Circle(coord, 45.0, fill=False, color=PATH_COLOR, linewidth=2.2)
        ax.add_patch(circle)
        ax.annotate(label, xy=coord, xytext=(coord[0], coord[1] - 60),
                    color=PATH_COLOR, ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("reference image (corner-concentrated speckle)", fontsize=9)

    gp.plots.inspect_subset(good_sub, ax=axes[2], show=False)
    axes[2].set_title("well-textured", fontsize=10)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "subset_quality.png")
    save_png(fig, out, dpi=170)
    plt.close(fig)
    return out, good_sub, poor_sub


if __name__ == "__main__":
    build_shape_order_gif()
    _, good_sub, poor_sub = build_quality_figure()
    print("good sssig", good_sub.sssig, "sigma", good_sub.sigma_intensity)
    print("poor sssig", poor_sub.sssig, "sigma", poor_sub.sigma_intensity)
