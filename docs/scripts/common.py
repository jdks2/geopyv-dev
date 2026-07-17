"""Shared helpers for building the tutorial gif/figure assets under docs/_static/.

Run the ``build_*.py`` scripts in this directory to (re)generate the assets
referenced by ``docs/tutorials/*.rst``. Requires ``geopyv_dev`` to be importable
(``maturin develop`` from the repo root) and a plain ``matplotlib`` + ``Pillow``
environment — no extra doc-build dependencies.
"""
import io
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from PIL import Image

plt.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATIC_DIR = os.path.join(REPO_ROOT, "docs", "_static")
GIF_DIR = os.path.join(STATIC_DIR, "gifs")
IMG_DIR = os.path.join(STATIC_DIR, "img")

# House palette, kept consistent with geopyv_dev/plots.py.
CMAP_SEQUENTIAL = "viridis"      # magnitude fields (R, contours) — matches plots.py default
MESH_COLOR = "black"             # mesh/element overlay lines
PATH_COLOR = "#d62728"           # particle path highlight (explicitly requested: red)
GOOD_COLOR = "#1f77b4"           # well-textured subset outline (blue)
POOR_COLOR = "#d62728"           # poorly-textured subset outline (red) — always paired with a text/marker cue


def fig_to_rgba(fig, dpi=None):
    """Render a matplotlib Figure to an (H, W, 4) uint8 array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi or fig.dpi)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGBA"))


def save_gif(frames, path, duration_ms=80, loop=0, colors=64):
    """Assemble a list of (H,W,3|4) uint8 arrays or PIL Images into a looping gif.

    Frames are quantized to a shared adaptive palette (`colors`) — plain
    per-frame quantization bloats file size badly on photographic speckle
    backgrounds, which don't have the flat color regions GIF compresses well.
    `duration_ms` may be a scalar or a per-frame list (use a list to hold the
    last frame of a beat via a longer duration instead of duplicating frames).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil_frames = [
        Image.fromarray(f).convert("RGB") if not isinstance(f, Image.Image) else f.convert("RGB")
        for f in frames
    ]
    # Build one shared palette from a sample spread across the whole sequence —
    # using just the first frame degenerates badly when it's a near-blank fade-in
    # (its palette ends up with too few real colors for the busy frames later on).
    sample_idx = np.linspace(0, len(pil_frames) - 1, min(8, len(pil_frames))).astype(int)
    w, h = pil_frames[0].size
    strip = Image.new("RGB", (w, h * len(sample_idx)))
    for k, idx in enumerate(sample_idx):
        strip.paste(pil_frames[idx], (0, k * h))
    palette_src = strip.quantize(colors=colors, method=Image.MEDIANCUT)

    quantized = [f.quantize(colors=colors, palette=palette_src, dither=Image.NONE) for f in pil_frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
        disposal=2,
    )
    size_kb = os.path.getsize(path) / 1024
    print(f"wrote {path} ({len(quantized)} frames, {size_kb:.0f} KB)")


def load_downsampled(path, size=240, blur=1.5):
    """Load an image, downsample, and softly blur it.

    Full-res photographic speckle noise compresses very badly in a GIF
    palette (every pixel is independently random). A blurred thumbnail still
    reads as a speckle pattern at animation scale but is far cheaper to encode.
    """
    from PIL import ImageFilter
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    return np.array(img)


def hold(duration_list, extra_ms):
    """Extend the last entry of a per-frame duration list by extra_ms —
    use instead of duplicating frames to pause on a beat."""
    duration_list[-1] += extra_ms
    return duration_list


def save_png(fig, path, dpi=150):
    """Save a static figure to docs/_static/img/, creating the directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Flow-diagram primitives, shared by build_mesh_flow.py / build_sequence_flow.py.
# ---------------------------------------------------------------------------

FLOW_BOX_COLOR = "#eef2f7"
FLOW_BOX_EDGE = "#333333"
FLOW_LOOP_COLOR = "#eaf3ea"
FLOW_ERR_COLOR = "#fbeaea"
FLOW_ERR_EDGE = "#a33"
FLOW_TEXT_SIZE = 8.5


def flow_box(ax, xy, w, h, text, color=FLOW_BOX_COLOR, edge=FLOW_BOX_EDGE, fontsize=FLOW_TEXT_SIZE):
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, linespacing=1.35)
    return (x, y, w, h)


def flow_diamond(ax, xy, w, h, text, fontsize=FLOW_TEXT_SIZE):
    x, y = xy
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.1,
                          edgecolor=FLOW_BOX_EDGE, facecolor="#fff6e6"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, linespacing=1.3)
    return (x, y, w, h)


def flow_arrow(ax, p0, p1, text=None, rad=0.0, color="#333333", text_dx=0.15):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                         linewidth=1.1, color=color,
                         connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + text_dx, my, text, ha="left", va="center",
                fontsize=FLOW_TEXT_SIZE - 1, color=color)
