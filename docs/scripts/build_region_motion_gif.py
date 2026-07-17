"""Build region_motion.gif for docs/tutorials/01_images_and_masks.rst.

Shows how each `option` value ("F"lexible, "R"igid, "S"tatic) actually moves a
tracked `CircleRegion`, by driving a real `CircleRegion` instance through its
own `store_flexible`/`store_rigid` methods (the same calls `Mesh`/`Sequence`
make internally, per `store_region_step` in `src/sequence.rs`) rather than
hand-drawing shapes. "Defined" ("D") has no working automatic store/update
path today (see `region_option_d_investigation.md`) — its panel instead
writes directly to `current_nodes` each frame, which is the only way to drive
a region's shape from outside input right now, and the tutorial text says so
explicitly rather than implying an automatic "D" pipeline exists.

Every panel is parametrised as a periodic function of frame index so the
whole gif loops with no seam: the state implied by the "wrap" transition
(last frame back to frame 0) uses exactly the same per-step increment as
every other transition.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO_ROOT, GIF_DIR, fig_to_rgba, save_gif

sys.path.insert(0, REPO_ROOT)
import geopyv_dev as gp

N_FRAMES = 24
RADIUS = 1.0
LIM = 2.0


def _base_region(option):
    return gp.CircleRegion(centre=[0.0, 0.0], radius=RADIUS, size=0.15, option=option)


def _flexible_history():
    """Small drift + a squash-then-relax pulse (scale x down / y up and back),
    driven entirely through repeated store_flexible() calls."""
    region = _base_region("F")
    base_nodes = np.asarray(region.current_nodes)
    centre0 = np.asarray(region.current_centre)

    def shape_at(i):
        e = 0.5 * (1 - np.cos(2 * np.pi * i / N_FRAMES))       # 0 -> 1 -> 0
        sx, sy = 1 - 0.22 * e, 1 + 0.22 * e
        trans = np.array([0.5 * np.sin(2 * np.pi * i / N_FRAMES),
                           0.25 * np.sin(2 * np.pi * i / N_FRAMES)])
        local = base_nodes - centre0
        return centre0 + trans + np.column_stack((local[:, 0] * sx, local[:, 1] * sy))

    prev = shape_at(0)
    history = [prev]
    for i in range(1, N_FRAMES):
        cur = shape_at(i)
        region.store_flexible(cur - prev)
        # store_* only appends to history_nodes; current_nodes only advances
        # on update() (matching the real Sequence pipeline, which calls it
        # once the next reference image is loaded).
        region.update(f"frame_{i}")
        history.append(np.asarray(region.current_nodes))
        prev = cur
    return history


def _rigid_history():
    """Constant angular velocity spin + a small circular translation, driven
    entirely through repeated store_rigid() calls."""
    region = _base_region("R")
    dtheta = 2 * np.pi / N_FRAMES

    def centre_at(i):
        return 0.4 * np.array([np.cos(2 * np.pi * i / N_FRAMES) - 1,
                                np.sin(2 * np.pi * i / N_FRAMES)])

    history = [np.asarray(region.current_nodes)]
    prev_centre = centre_at(0)
    for i in range(1, N_FRAMES):
        cur_centre = centre_at(i)
        du, dv = cur_centre - prev_centre
        region.store_rigid(np.array([du, dv, 0.0, dtheta, -dtheta]))
        region.update(f"frame_{i}")
        history.append(np.asarray(region.current_nodes))
        prev_centre = cur_centre
    return history


def _static_history():
    region = _base_region("S")
    nodes = np.asarray(region.current_nodes)
    return [nodes for _ in range(N_FRAMES)]


def _defined_history():
    """No automatic store/update path exists for "D" today (it's a no-op —
    see region_option_d_investigation.md); this drives current_nodes directly
    every frame, the supported way to feed a region external, per-frame node
    positions. A 3-lobed radial wobble is used so its motion is visibly not
    reducible to the uniform squash (Flexible) or pure rotation (Rigid)."""
    region = _base_region("D")
    base_nodes = np.asarray(region.current_nodes)
    centre0 = np.asarray(region.current_centre)
    theta = np.arctan2(base_nodes[:, 1] - centre0[1], base_nodes[:, 0] - centre0[0])
    r = np.hypot(base_nodes[:, 0] - centre0[0], base_nodes[:, 1] - centre0[1])

    history = []
    for i in range(N_FRAMES):
        wobble = 0.15 * np.sin(2 * np.pi * i / N_FRAMES + 3 * theta)
        rr = r + wobble
        nodes = np.column_stack((centre0[0] + rr * np.cos(theta), centre0[1] + rr * np.sin(theta)))
        region.current_nodes = nodes
        history.append(np.asarray(region.current_nodes))
    return history


def build_region_motion_gif():
    panels = {
        "flexible (F)": _flexible_history(),
        "rigid (R)": _rigid_history(),
        "static (S)": _static_history(),
        "defined (D)": _defined_history(),
    }
    frames = []
    for i in range(N_FRAMES):
        fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.2), dpi=130)
        for ax, (label, history) in zip(axes.flat, panels.items()):
            nodes = history[i]
            centre = nodes.mean(axis=0)
            xs = np.append(nodes[:, 0], nodes[0, 0])
            ys = np.append(nodes[:, 1], nodes[0, 1])
            ax.fill(xs, ys, facecolor="#1f77b4", alpha=0.35, edgecolor="#1f77b4", linewidth=1.8)
            # Spoke from centre to node 0 — otherwise a near-circular shape
            # makes rotation (rigid) invisible against a squash (flexible).
            ax.plot([centre[0], nodes[0, 0]], [centre[1], nodes[0, 1]],
                    color="#d62728", linewidth=1.8)
            ax.plot(*centre, "+", color="black", markersize=7, markeredgewidth=1.5)
            ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(label, fontsize=10, pad=3)
        plt.tight_layout(pad=0.4)
        frames.append(fig_to_rgba(fig))
        plt.close(fig)
    out = os.path.join(GIF_DIR, "region_motion.gif")
    save_gif(frames, out, duration_ms=90, colors=32)
    return out


if __name__ == "__main__":
    build_region_motion_gif()
