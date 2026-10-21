"""Build the adaptive-masking figures for docs/tutorials/03_mesh.rst:

- ``adaptive_neighbour_check.png``: the still-pair screen. A node's own warp
  predicts its neighbour's displacement; the tolerance circle is what noise
  alone allows. (a) Uniform ground: the neighbour lands inside. (b) A band
  between them: it lands outside, so both go on to the tests.
- ``adaptive_flow.png``: what happens to one node, from the screen through the
  tests, the jolt and the corrections — reflects src/screen.rs, src/seam.rs and
  Mesh::adaptive_masking.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    IMG_DIR, save_png, flow_box as box, flow_diamond as diamond, flow_arrow as arrow,
    FLOW_LOOP_COLOR as LOOP_COLOR, FLOW_TEXT_SIZE as TEXT_SIZE, GOOD_COLOR, POOR_COLOR,
)


def _vec(ax, p, d, color, ls="-", lw=1.6, z=4):
    ax.add_patch(FancyArrowPatch(p, (p[0] + d[0], p[1] + d[1]), arrowstyle="-|>", mutation_scale=12,
                                 color=color, linestyle=ls, linewidth=lw, zorder=z))


def build_neighbour_check():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    a, b = np.array([1.0, 1.0]), np.array([4.0, 1.6])
    scale = 1.0
    u_a = np.array([1.1, 1.3])
    grad = np.array([[0.06, -0.04], [0.10, 0.02]])          # du/dx, du/dy terms
    predicted = u_a + grad @ (b - a)
    tol = 0.42
    cases = [("(a) Uniform ground: within tolerance", predicted + np.array([0.18, -0.12]), GOOD_COLOR),
             ("(b) Band between A and B: beyond tolerance", predicted + np.array([0.9, -1.15]), POOR_COLOR)]
    for ax, (title, u_b, col) in zip(axes, cases):
        if "Band" in title:
            ax.axvspan(2.25, 2.85, color="#f3d9c8", zorder=0)
            ax.text(2.55, 3.85, "band", ha="center", fontsize=TEXT_SIZE + 1, color="#a0522d")
        ax.plot(*zip(a, b), color="#999999", lw=0.8, ls=":", zorder=1)
        for p, lab in ((a, "A"), (b, "B")):
            ax.plot(*p, "o", color="k", ms=5, zorder=5)
            ax.text(p[0] - 0.15, p[1] - 0.3, lab, fontsize=TEXT_SIZE + 3)
        _vec(ax, a, scale * u_a, "k")
        tip = b + scale * predicted
        _vec(ax, b, scale * predicted, "#555555", ls="--", lw=1.2)
        ax.add_patch(Circle(tip, tol, facecolor="#e7f3ff", edgecolor="#4a6fa5", lw=1.0, zorder=2))
        _vec(ax, b, scale * u_b, col, lw=1.8, z=6)
        ax.set_xlim(0.3, 6.6)
        ax.set_ylim(0.3, 4.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=TEXT_SIZE + 2)
    handles = [plt.Line2D([], [], color="k", lw=1.6, label="A's measured displacement"),
               plt.Line2D([], [], color="#555555", lw=1.2, ls="--", label="B's displacement predicted from A's warp"),
               plt.Line2D([], [], color="#4a6fa5", marker="o", ls="", mfc="#e7f3ff", ms=10,
                          label="tolerance: what noise alone allows (from frames 0–1)"),
               plt.Line2D([], [], color=GOOD_COLOR, lw=1.8, label="B's measured displacement")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=TEXT_SIZE + 1, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def build_flow():
    fig, ax = plt.subplots(figsize=(7.5, 10.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(4.6, 26)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    cx = 5.0

    box(ax, (cx, 25), 6.8, 1.0, "Plain order-1 solve of every node (Mesh::solve as usual)")
    box(ax, (cx, 23.1), 6.8, 1.9,
        "Screen: for each pair of neighbours, does A's warp predict B?\n"
        "mismatch beyond the noise tolerance (noise from the still pair,\n"
        "frames 0–1, or a patch around the seed) and beyond 0.01 px of\n"
        "under-fit → flag both; then keep ⅓ / add ⅔ of neighbours")
    d_sent = diamond(ax, (cx, 20.6), 4.6, 1.6, "Sent on?")
    box(ax, (8.7, 20.6), 2.2, 0.9, "keep the\nplain solve", color="#e7f3ff")
    box(ax, (cx, 18.5), 6.8, 1.4,
        "Tests on the plain subset's residual:\nbending score, and hinge and jump scores along their best line")
    d_dec = diamond(ax, (cx, 16.0), 5.6, 1.9,
                    "Edge beats its alarm\nand beats bending by\nthe margin?")
    d_bend = diamond(ax, (8.4, 13.6), 3.0, 1.6, "Either score\nbeyond its alarm?")
    box(ax, (8.4, 11.2), 2.4, 0.9, "re-solve the full\ndisc at order 2", color=LOOP_COLOR)
    box(ax, (8.4, 9.6), 2.4, 0.8, "leave as is", color="#e7f3ff")
    jx = cx - 0.6
    diamond(ax, (jx, 13.2), 4.2, 1.7, "Line within 5 px\nof the centre?")
    box(ax, (1.45, 11.6), 2.5, 1.8,
        "Jolt: move node and\nsubset away from the\nline (along the edge on\na boundary; not at a\ncorner), then re-test",
        color=LOOP_COLOR, fontsize=TEXT_SIZE - 0.5)
    box(ax, (jx, 10.0), 4.0, 1.2, "Cut: mask the far side of the line\n(3 px gap) and re-solve")
    box(ax, (jx, 7.9), 5.0, 1.4,
        "Pass 2, tests on the cut subset:\nsecond cut (narrow band, keeps \u2265 15 %),\norder 2 on the cut subset, or nothing")
    box(ax, (cx, 5.6), 6.8, 1.3,
        "Mesh updated: final warps (12 columns), jolted node positions;\n"
        "mesh.seams keeps the plain solve and every node's record", color="#e7f3ff")

    arrow(ax, (cx, 24.5), (cx, 24.05))
    arrow(ax, (cx, 22.15), (cx, 21.4))
    arrow(ax, (cx + 2.3, 20.6), (7.6, 20.6), text="no", text_dx=-0.5)
    arrow(ax, (cx, 19.8), (cx, 19.2), text="yes", text_dx=0.3)
    arrow(ax, (cx, 17.8), (cx, 16.95))
    arrow(ax, (cx + 2.8, 16.0), (8.4, 14.4), text="no", text_dx=0.1)
    arrow(ax, (8.4, 12.8), (8.4, 11.65), text="yes", text_dx=0.3)
    ax.plot([9.9, 9.9], [13.6, 9.6], color="#333333", lw=1.1)
    arrow(ax, (9.9, 9.6), (9.6, 9.6))
    ax.text(10.05, 11.6, "no", fontsize=TEXT_SIZE - 1, rotation=90, va="center")
    arrow(ax, (jx, 15.05), (jx, 14.05), text="yes", text_dx=0.3)
    arrow(ax, (jx - 2.1, 13.2), (1.45, 12.5), text="yes", text_dx=-0.6)
    ax.plot([1.45, 1.45], [10.7, 10.0], color="#333333", lw=1.1)
    arrow(ax, (1.45, 10.0), (jx - 2.0, 10.0))
    arrow(ax, (jx, 12.35), (jx, 10.6), text="no", text_dx=0.3)
    arrow(ax, (jx, 9.4), (jx, 8.6))
    arrow(ax, (jx, 7.2), (jx, 6.25))
    ax.set_title("Adaptive masking — solver_options={'masking': 'adaptive'}", fontsize=11, pad=14)
    return fig


if __name__ == "__main__":
    save_png(build_neighbour_check(), os.path.join(IMG_DIR, "adaptive_neighbour_check.png"))
    save_png(build_flow(), os.path.join(IMG_DIR, "adaptive_flow.png"))
