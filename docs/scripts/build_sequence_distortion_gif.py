"""Build the "mesh that never gets re-triangulated" distortion gif for
docs/tutorials/04_sequence.rst.

Solves the real committed images/SB/* series (6 frames) with sync=True, so the
same node connectivity is reused for every pair rather than being regenerated —
then replays that one triangulation at each frame's accumulated node position.
Demonstrates why large accumulated strain needs periodic re-meshing: the
elements visibly skew even though the DIC correlation itself stays healthy
(high C_ZNCC throughout — this is a geometry problem, not a correlation one).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO_ROOT, GIF_DIR, CMAP_SEQUENTIAL, MESH_COLOR, save_gif, hold

sys.path.insert(0, REPO_ROOT)
import geopyv_dev as gp

CANVAS = 4.2
DPI = 100
SB_DIR = os.path.join(REPO_ROOT, "images", "SB")
BOUNDARY = np.array([[300.0, 300.0], [300.0, 700.0], [700.0, 700.0], [700.0, 300.0]])


def solve_sequence():
    boundary = gp.PathRegion(nodes=BOUNDARY, hard=False)
    local_mask = gp.Mask(mask_type="local", shape="circle", size=25)
    seq = gp.Sequence(image_dir=SB_DIR, boundary=boundary, target_nodes=70,
                       size=(15., 70.), mesh_order=1)
    seq.solve(local_mask=local_mask, seed_coord=[500.0, 500.0], seed_warp=[0.0] * 6,
              subset_order=1, tolerance=0.75, seed_tolerance=0.9, method="icgn",
              options=gp.SequenceOptions(guide=True, sync=True, sequential=False))
    return seq


def frame_positions(seq):
    """Accumulate each pair's displacement onto frame 0's fixed node positions.

    Only walks pairs up to (but not including) the first automatic reference
    catch-up (see the Sequence::solve flow diagram): that fallback resets
    `sync_sol`, so the mesh actually gets re-triangulated there — a different
    node count — which is a second, *good* example of the fix this page is
    building up to, not something this "never re-meshed" demo should paper
    over by re-indexing across it.
    """
    nodes0 = np.asarray(seq.nodes(0))
    updates = seq.reference_updates
    n_fixed_pairs = updates.index(True) if True in updates else seq.n_pairs
    cum = [np.zeros_like(nodes0)]
    for i in range(n_fixed_pairs):
        cum.append(np.asarray(seq.displacements(i)))
    return nodes0, cum


def min_angle_deg(nodes, elements):
    """Smallest interior angle across all triangles, in degrees — a simple,
    honest distortion metric (matches the intuition behind the Jacobian
    fold-over check in Mesh::solve: it degrades toward 0 well before any
    element actually inverts)."""
    p0, p1, p2 = nodes[elements[:, 0]], nodes[elements[:, 1]], nodes[elements[:, 2]]
    worst = 180.0
    for a, b, c in ((p0, p1, p2), (p1, p2, p0), (p2, p0, p1)):
        v1 = a - b
        v2 = c - b
        cos_ang = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12)
        ang = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        worst = min(worst, ang.min())
    return worst


def build_frames(seq):
    nodes0, cum = frame_positions(seq)
    elements = np.asarray(seq.elements(0))[:, :3]
    bx0, bx1 = BOUNDARY[:, 0].min() - 40, BOUNDARY[:, 0].max() + 40
    by0, by1 = BOUNDARY[:, 1].min() - 40, BOUNDARY[:, 1].max() + 40
    r_max = max(np.sqrt((c ** 2).sum(axis=1)).max() for c in cum) or 1.0

    frames, durations = [], []
    for k, c in enumerate(cum):
        dx = nodes0[:, 0] + c[:, 0]
        dy = nodes0[:, 1] + c[:, 1]
        R = np.sqrt((c ** 2).sum(axis=1))
        ang = min_angle_deg(np.column_stack([dx, dy]), elements)

        fig, ax = plt.subplots(figsize=(CANVAS, CANVAS), dpi=DPI)
        ax.set_facecolor("#f2f2f2")
        tri_obj = tri.Triangulation(dx, dy, elements)
        ax.tricontourf(tri_obj, R, levels=12, cmap=CMAP_SEQUENTIAL, vmin=0, vmax=r_max)
        ax.triplot(tri_obj, color=MESH_COLOR, linewidth=0.6, alpha=0.6)
        ax.set_xlim(bx0, bx1)
        ax.set_ylim(by1, by0)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"frame {k}  —  same triangulation every frame\n"
                     f"worst element angle: {ang:4.1f}°", fontsize=9)
        plt.tight_layout(pad=0.4)

        buf_fig = fig
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(frame.copy())
        durations.append(600)
        plt.close(fig)

    durations = hold(durations, 900)
    return frames, durations


def build():
    seq = solve_sequence()
    frames, durations = build_frames(seq)
    out = os.path.join(GIF_DIR, "sequence_distortion.gif")
    save_gif(frames, out, duration_ms=durations, colors=48)
    return out


if __name__ == "__main__":
    build()
