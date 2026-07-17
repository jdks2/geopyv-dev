"""Build the hero storyboard gif for docs/tutorials/00_introduction.rst.

Phased build — run with e.g. `python docs/scripts/build_intro_gif.py --phase A`
to render/inspect one phase's frames before moving on to the next. `--phase all`
(default) renders every phase and stitches the full loop.

Uses the real canonical shear series (images/shear/) and a real solved Sequence
for every scene. The subset in phase A is fixed at the exact coordinate of a
real mesh node (phase B's anchor), so "subset" and "mesh node" are literally
the same point throughout. Phase D (shear -> rotation handoff, particle path)
blends in analytically computed rigid-rotation displacement on top of the same
mesh geometry, exaggerated for visibility — this is a concept animation, not a
second DIC solve, so the blend is computed directly rather than re-run through
image correlation.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
import matplotlib.tri as tri

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    REPO_ROOT, GIF_DIR, MESH_COLOR, PATH_COLOR,
    fig_to_rgba, save_gif, hold, load_downsampled,
)

sys.path.insert(0, REPO_ROOT)
import geopyv_dev as gp

CANVAS = 4.0          # figure size in inches (square)
DPI = 120             # -> 480x480 px frames (holds up at full text-column width)
IMG_SIZE = 1001.0
BOUNDARY = np.array([[300.0, 300.0], [300.0, 700.0], [700.0, 700.0], [700.0, 300.0]])
FIX_COORD = [640.0, 380.0]   # initial guess; snapped to the nearest real mesh node below
SHEAR_DIR = os.path.join(REPO_ROOT, "images", "shear")
CMAP_DIVERGING = "RdBu_r"    # signed u-displacement contour


def _blank_axes(ax):
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(IMG_SIZE, 0)  # row-down image convention
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _superellipse(cx, cy, r, n, n_pts=100):
    """Vertices of |x/r|^n + |y/r|^n = 1, centred at (cx, cy). n=2 -> circle, n large -> square."""
    t = np.linspace(0, 2 * np.pi, n_pts)
    ct, st = np.cos(t), np.sin(t)
    x = np.sign(ct) * np.abs(ct) ** (2.0 / n)
    y = np.sign(st) * np.abs(st) ** (2.0 / n)
    return cx + r * x, cy + r * y


def ease(t):
    """Smoothstep easing, t in [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Phase A: speckle reveal, subset fixed at the anchor node, template morphs
# square -> circle
# ---------------------------------------------------------------------------

def phase_a(anchor_coord, n_reveal=7, n_morph=14):
    img = load_downsampled(os.path.join(SHEAR_DIR, "shear_0.jpg"))
    frames, durations = [], []

    # Reveal: fade the speckle image in.
    for i in range(n_reveal):
        alpha = ease((i + 1) / n_reveal)
        fig, ax = plt.subplots(figsize=(CANVAS, CANVAS), dpi=DPI)
        _blank_axes(ax)
        ax.imshow(img, cmap="gist_gray", extent=(0, IMG_SIZE, IMG_SIZE, 0), alpha=alpha)
        plt.tight_layout(pad=0)
        frames.append(fig_to_rgba(fig))
        durations.append(70)
        plt.close(fig)

    # Morph: square -> circle while shrinking, fixed at the anchor node.
    r0, r1 = 190.0, 70.0
    n0, n1 = 10.0, 2.0
    for i in range(n_morph):
        t = ease((i + 1) / n_morph)
        r = r0 + (r1 - r0) * t
        n = n0 + (n1 - n0) * t
        fig, ax = plt.subplots(figsize=(CANVAS, CANVAS), dpi=DPI)
        _blank_axes(ax)
        ax.imshow(img, cmap="gist_gray", extent=(0, IMG_SIZE, IMG_SIZE, 0))
        xs, ys = _superellipse(anchor_coord[0], anchor_coord[1], r, n)
        ax.plot(xs, ys, color=PATH_COLOR, linewidth=4.0)
        ax.plot(*anchor_coord, marker="+", color=PATH_COLOR, markersize=16, markeredgewidth=3)
        plt.tight_layout(pad=0)
        frames.append(fig_to_rgba(fig))
        durations.append(70)
        plt.close(fig)

    durations = hold(durations, 500)
    return frames, durations


def build_sequence():
    boundary = gp.PathRegion(nodes=BOUNDARY, hard=False)
    local_mask = gp.Mask(mask_type="local", shape="circle", size=25)
    seq = gp.Sequence(image_dir=SHEAR_DIR, boundary=boundary, target_nodes=70,
                       size=(15., 70.), mesh_order=1)
    seq.solve(local_mask=local_mask, seed_coord=[500.0, 500.0], seed_warp=[0.0] * 6,
              subset_order=1, tolerance=0.75, seed_tolerance=0.9, method="icgn",
              options=gp.SequenceOptions(guide=True, sync=True))
    return seq


def _anchor_index(nodes, guess_coord):
    return int(np.argmin(np.sum((nodes - np.array(guess_coord)) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Phase B: camera zooms out from the anchor node while the real mesh
# triangulation grows outward from it, edge by edge (a wavefront over the
# actual element connectivity — not radial lines to unrelated nodes).
# ---------------------------------------------------------------------------

def _bfs_hops(elements, anchor_idx, n_nodes):
    adj = [set() for _ in range(n_nodes)]
    for a, b, c in elements:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    hops = np.full(n_nodes, -1.0)
    hops[anchor_idx] = 0.0
    frontier = [anchor_idx]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if hops[v] == -1:
                    hops[v] = d
                    nxt.append(v)
        frontier = nxt
    return hops


def _unique_edges(elements):
    edges = set()
    for a, b, c in elements:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((u, v) if u < v else (v, u))
    return np.array(sorted(edges))


def phase_b(seq, anchor_idx, n_zoom=16):
    img = load_downsampled(os.path.join(SHEAR_DIR, "shear_0.jpg"))
    nodes = np.asarray(seq.nodes(0))
    elements = np.asarray(seq.elements(0))[:, :3].astype(int)
    anchor_coord = nodes[anchor_idx]

    hops = _bfs_hops(elements, anchor_idx, len(nodes))
    max_hop = hops.max()
    edges = _unique_edges(elements)
    edge_hops = hops[edges].max(axis=1)
    mesh_rgb = to_rgb(MESH_COLOR)
    path_rgb = to_rgb(PATH_COLOR)

    x0, x1 = anchor_coord[0] - 110, anchor_coord[0] + 110
    y0, y1 = anchor_coord[1] - 110, anchor_coord[1] + 110
    bx0, bx1 = BOUNDARY[:, 0].min() - 40, BOUNDARY[:, 0].max() + 40
    by0, by1 = BOUNDARY[:, 1].min() - 40, BOUNDARY[:, 1].max() + 40

    frames, durations = [], []
    for i in range(n_zoom):
        t = ease((i + 1) / n_zoom)
        xl = x0 + (bx0 - x0) * t
        xr = x1 + (bx1 - x1) * t
        yl = y0 + (by0 - y0) * t
        yr = y1 + (by1 - y1) * t

        fig, ax = plt.subplots(figsize=(CANVAS, CANVAS), dpi=DPI)
        ax.imshow(img, cmap="gist_gray", extent=(0, IMG_SIZE, IMG_SIZE, 0))

        # Wavefront of real mesh edges growing outward from the anchor node,
        # in step with the pull-back.
        level = t * (max_hop + 1.0)
        edge_alpha = np.clip(level - edge_hops, 0.0, 1.0) * 0.55
        visible = edge_alpha > 0.01
        if visible.any():
            segments = nodes[edges[visible]]
            colors = np.zeros((visible.sum(), 4))
            colors[:, :3] = mesh_rgb
            colors[:, 3] = edge_alpha[visible]
            ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.7))

        node_alpha = np.clip(level - hops + 1.0, 0.0, 1.0)
        node_alpha[anchor_idx] = 1.0
        colors = np.zeros((len(nodes), 4))
        colors[:, :3] = path_rgb
        colors[:, 3] = node_alpha * 0.7
        ax.scatter(nodes[:, 0], nodes[:, 1], s=6.0, c=colors, linewidths=0)
        ax.plot(*anchor_coord, marker="+", color=PATH_COLOR, markersize=16, markeredgewidth=3)

        ax.set_xlim(xl, xr)
        ax.set_ylim(yr, yl)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        plt.tight_layout(pad=0)
        frames.append(fig_to_rgba(fig))
        durations.append(80)
        plt.close(fig)

    durations = hold(durations, 400)
    return frames, durations


DISPLAY_SCALE = 40.0  # exaggerate real (sub-px to few-px) displacements for visibility


def _sequence_states(seq):
    """List of (nodes, u, v) per frame, frame 0 = undeformed reference."""
    nodes0 = np.asarray(seq.nodes(0))
    states = [(nodes0, np.zeros(len(nodes0)), np.zeros(len(nodes0)))]
    for i in range(seq.n_pairs):
        d = np.asarray(seq.displacements(i))
        states.append((nodes0, d[:, 0], d[:, 1]))
    return states


# ---------------------------------------------------------------------------
# Phase C: mesh deforms under real (solved) shear, u-displacement contour
# growing over the real deforming speckle pattern; the traced particle
# already appears as the contour starts to deflect.
# ---------------------------------------------------------------------------

def phase_c(seq, anchor_idx, u_vmax_c, n_sub=6):
    states = _sequence_states(seq)
    elements = np.asarray(seq.elements(0))[:, :3].astype(int)
    nodes0 = states[0][0]

    bx0, bx1 = BOUNDARY[:, 0].min() - 40, BOUNDARY[:, 0].max() + 40
    by0, by1 = BOUNDARY[:, 1].min() - 40, BOUNDARY[:, 1].max() + 40

    imgs = [load_downsampled(os.path.join(SHEAR_DIR, f"shear_{i}.jpg")) for i in range(len(states))]

    trail_x = [nodes0[anchor_idx, 0]]
    trail_y = [nodes0[anchor_idx, 1]]

    frames, durations = [], []
    for k in range(len(states) - 1):
        nodes, u0, v0 = states[k]
        _, u1, v1 = states[k + 1]
        for s in range(n_sub):
            t = ease((s + 1) / n_sub)
            u = u0 + (u1 - u0) * t
            v = v0 + (v1 - v0) * t
            dx = nodes[:, 0] + DISPLAY_SCALE * u
            dy = nodes[:, 1] + DISPLAY_SCALE * v
            trail_x.append(dx[anchor_idx]); trail_y.append(dy[anchor_idx])

            bg = ((1 - t) * imgs[k] + t * imgs[k + 1]).astype(np.uint8)

            fig, ax = plt.subplots(figsize=(CANVAS, CANVAS), dpi=DPI)
            ax.imshow(bg, cmap="gist_gray", extent=(0, IMG_SIZE, IMG_SIZE, 0))
            tri_obj = tri.Triangulation(dx, dy, elements)
            ax.tricontourf(tri_obj, u, levels=12, cmap=CMAP_DIVERGING,
                            vmin=-u_vmax_c, vmax=u_vmax_c, alpha=0.7)
            ax.triplot(tri_obj, color=MESH_COLOR, linewidth=0.4, alpha=0.3)
            ax.plot(trail_x, trail_y, color=PATH_COLOR, linewidth=2.2)
            ax.plot(trail_x[-1], trail_y[-1], "o", color=PATH_COLOR, markersize=5)

            ax.set_xlim(bx0, bx1); ax.set_ylim(by1, by0)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            plt.tight_layout(pad=0)
            frames.append(fig_to_rgba(fig))
            durations.append(90)
            plt.close(fig)

    durations = hold(durations, 400)
    return frames, durations, trail_x, trail_y


# ---------------------------------------------------------------------------
# Phase D: shear -> rotation handoff, mesh/contour fade, red particle path
# continues from phase C
# ---------------------------------------------------------------------------
#
# Analytically blended displacement fields on the same mesh geometry (not a
# second DIC solve — see module docstring). ORIGIN is the shear field's centre
# (matches images/shear's Speckle origin = image centre).

ORIGIN = np.array([IMG_SIZE / 2.0, IMG_SIZE / 2.0])
SHEAR_K = 0.01     # matches comp[3]=comp[4] used to generate images/shear
ROT_THETA = 0.05   # radians, small-angle rigid rotation


def _field(nodes, weights):
    """weights = (shear_w, rot_w) in [0,1]; returns (u, v)."""
    dx = nodes[:, 0] - ORIGIN[0]
    dy = nodes[:, 1] - ORIGIN[1]
    ws, wr = weights
    u = ws * SHEAR_K * dy + wr * (-ROT_THETA * dy)
    v = ws * SHEAR_K * dx + wr * (ROT_THETA * dx)
    return u, v


def _weights_at(t):
    """Single handoff over t in [0,1]: shear (1->0) to rotation (0->1)."""
    tt = ease(t)
    return (1 - tt, tt)


def _seq_u_vmax(seq):
    """Color scale for phase C's u-contour, from the real (small, sub-pixel)
    solved displacements — kept separate from phase D's scale below because
    the exaggerated analytic rotation field is an order of magnitude larger
    and would wash phase C out to near-white on a shared scale."""
    vmax = 1e-6
    for i in range(seq.n_pairs):
        d = np.asarray(seq.displacements(i))
        vmax = max(vmax, np.abs(d[:, 0]).max())
    return vmax


def _field_u_vmax(nodes, n_probe=60):
    """Color scale for phase D's u-contour, from its own analytic field."""
    vmax = 1e-6
    for t in np.linspace(0, 1, n_probe):
        u, _ = _field(nodes, _weights_at(t))
        vmax = max(vmax, np.abs(u).max())
    return vmax


def phase_d(seq, anchor_idx, u_vmax_d, trail_x0, trail_y0, n_frames=30):
    nodes = np.asarray(seq.nodes(0))
    elements = np.asarray(seq.elements(0))[:, :3].astype(int)
    bx0, bx1 = BOUNDARY[:, 0].min() - 40, BOUNDARY[:, 0].max() + 40
    by0, by1 = BOUNDARY[:, 1].min() - 40, BOUNDARY[:, 1].max() + 40
    bg = load_downsampled(os.path.join(SHEAR_DIR, "shear_4.jpg"))

    trail_x, trail_y = list(trail_x0), list(trail_y0)

    frames, durations = [], []
    for i in range(n_frames):
        t = (i + 1) / n_frames
        u, v = _field(nodes, _weights_at(t))
        dx = nodes[:, 0] + DISPLAY_SCALE * u
        dy = nodes[:, 1] + DISPLAY_SCALE * v
        trail_x.append(dx[anchor_idx]); trail_y.append(dy[anchor_idx])

        fade = max(0.0, 1.0 - 1.15 * t)  # mesh/contour fade out over the phase
        fig, ax = plt.subplots(figsize=(CANVAS, CANVAS), dpi=DPI)
        ax.imshow(bg, cmap="gist_gray", extent=(0, IMG_SIZE, IMG_SIZE, 0))
        if fade > 0.02:
            tri_obj = tri.Triangulation(dx, dy, elements)
            ax.tricontourf(tri_obj, u, levels=12, cmap=CMAP_DIVERGING,
                            vmin=-u_vmax_d, vmax=u_vmax_d, alpha=0.7 * fade)
            ax.triplot(tri_obj, color=MESH_COLOR, linewidth=0.4, alpha=fade * 0.3)

        ax.plot(trail_x, trail_y, color=PATH_COLOR, linewidth=2.2)
        ax.plot(trail_x[-1], trail_y[-1], "o", color=PATH_COLOR, markersize=5)

        ax.set_xlim(bx0, bx1); ax.set_ylim(by1, by0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        plt.tight_layout(pad=0)
        frames.append(fig_to_rgba(fig))
        durations.append(80)
        plt.close(fig)

    durations = hold(durations, 500)
    return frames, durations


# ---------------------------------------------------------------------------
# Results panel: particle strain history + subset inspect (real solved data)
# ---------------------------------------------------------------------------

def build_results_panel(seq, anchor_coord):
    coord = [float(anchor_coord[0]), float(anchor_coord[1])]

    particle = gp.Particle(source=seq, coordinate=coord)
    particle.solve()

    f_img = gp.Image(filepath=os.path.join(SHEAR_DIR, "shear_0.jpg"))
    g_img = gp.Image(filepath=os.path.join(SHEAR_DIR, "shear_4.jpg"))
    local_mask = gp.Mask(mask_type="local", shape="circle", size=25)
    subset = gp.Subset(coord=coord, local_mask=local_mask, f_img=f_img, g_img=g_img)
    subset.solve()

    # Kept at the same overall pixel width as every other scene (CANVAS x CANVAS)
    # so the stitched gif has one consistent frame size — two narrow subplots
    # rather than a wider canvas.
    fig, axes = plt.subplots(1, 2, figsize=(CANVAS, CANVAS), dpi=DPI)
    particle.history(quantity="strains", ax=axes[0], show=False)
    axes[0].set_title("particle strain history", fontsize=9)
    subset.inspect(ax=axes[1], show=False)
    axes[1].set_title("subset inspect", fontsize=9)
    fig.patch.set_facecolor("#f2f2f2")
    plt.tight_layout(pad=0.3)
    panel = fig_to_rgba(fig)
    plt.close(fig)
    return panel


# ---------------------------------------------------------------------------
# Phase E: slide left/right to reveal the results panel
# ---------------------------------------------------------------------------

def phase_e(last_d_frame, panel, n_slide=14):
    from PIL import Image as PILImage
    h, w = last_d_frame.shape[0], last_d_frame.shape[1]
    left = PILImage.fromarray(last_d_frame).convert("RGB")
    right = PILImage.fromarray(panel).convert("RGB").resize((w, h))

    frames, durations = [], []
    for i in range(n_slide):
        t = ease((i + 1) / n_slide)
        off = int(t * w)
        canvas = PILImage.new("RGB", (w, h), "#f2f2f2")
        canvas.paste(left, (-off, 0))
        canvas.paste(right, (w - off, 0))
        frames.append(np.array(canvas))
        durations.append(70)

    durations = hold(durations, 600)
    return frames, durations


# ---------------------------------------------------------------------------
# Phase F: fade results panel -> black -> speckle reveal (seamless loop)
# ---------------------------------------------------------------------------

def phase_f(panel_frame, n_fade=8):
    """Fade the results panel to black. The loop wrap-around into phase_a's own
    fade-in supplies the "speckle pattern reappears" beat — no need to fade
    back up here too (that would double the fade-in and hitch the loop)."""
    black = np.zeros_like(panel_frame[..., :3])
    frames, durations = [], []
    for i in range(n_fade):
        a = ease((i + 1) / n_fade)
        frames.append(((1 - a) * panel_frame[..., :3] + a * black).astype(np.uint8))
        durations.append(60)
    durations = hold(durations, 300)
    return frames, durations


def _crossfade(frame_a, frame_b, n=5, duration_ms=60):
    frames, durations = [], []
    a3, b3 = frame_a[..., :3].astype(float), frame_b[..., :3].astype(float)
    for i in range(n):
        t = ease((i + 1) / n)
        frames.append(((1 - t) * a3 + t * b3).astype(np.uint8))
        durations.append(duration_ms)
    return frames, durations


def build_full_gif():
    """Stitch every phase into the final looping hero gif."""
    seq = build_sequence()
    nodes0 = np.asarray(seq.nodes(0))
    anchor_idx = _anchor_index(nodes0, FIX_COORD)
    anchor_coord = nodes0[anchor_idx]
    u_vmax_c = _seq_u_vmax(seq)
    u_vmax_d = _field_u_vmax(nodes0)

    a_frames, a_dur = phase_a(anchor_coord)
    b_frames, b_dur = phase_b(seq, anchor_idx)
    c_frames, c_dur, trail_x, trail_y = phase_c(seq, anchor_idx, u_vmax_c)
    d_frames, d_dur = phase_d(seq, anchor_idx, u_vmax_d, trail_x, trail_y)
    panel = build_results_panel(seq, anchor_coord)
    e_frames, e_dur = phase_e(d_frames[-1], panel)
    f_frames, f_dur = phase_f(panel)

    xf_bc, xf_bc_dur = _crossfade(b_frames[-1], c_frames[0], n=5)

    frames = a_frames + b_frames + xf_bc + c_frames + d_frames + e_frames + f_frames
    durations = a_dur + b_dur + xf_bc_dur + c_dur + d_dur + e_dur + f_dur

    out = os.path.join(GIF_DIR, "intro_hero.gif")
    save_gif(frames, out, duration_ms=durations, colors=80)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="A")
    args = ap.parse_args()

    if args.phase == "A":
        seq = build_sequence()
        nodes0 = np.asarray(seq.nodes(0))
        anchor_idx = _anchor_index(nodes0, FIX_COORD)
        frames, durations = phase_a(nodes0[anchor_idx])
        save_gif(frames, os.path.join(GIF_DIR, "_preview_phase_a.gif"), duration_ms=durations)
    elif args.phase == "B":
        seq = build_sequence()
        nodes0 = np.asarray(seq.nodes(0))
        anchor_idx = _anchor_index(nodes0, FIX_COORD)
        frames, durations = phase_b(seq, anchor_idx)
        save_gif(frames, os.path.join(GIF_DIR, "_preview_phase_b.gif"), duration_ms=durations)
    elif args.phase == "C":
        seq = build_sequence()
        nodes0 = np.asarray(seq.nodes(0))
        anchor_idx = _anchor_index(nodes0, FIX_COORD)
        u_vmax_c = _seq_u_vmax(seq)
        frames, durations, _, _ = phase_c(seq, anchor_idx, u_vmax_c)
        save_gif(frames, os.path.join(GIF_DIR, "_preview_phase_c.gif"), duration_ms=durations)
    elif args.phase == "D":
        seq = build_sequence()
        nodes0 = np.asarray(seq.nodes(0))
        anchor_idx = _anchor_index(nodes0, FIX_COORD)
        u_vmax_c = _seq_u_vmax(seq)
        u_vmax_d = _field_u_vmax(nodes0)
        _, _, trail_x, trail_y = phase_c(seq, anchor_idx, u_vmax_c)
        frames, durations = phase_d(seq, anchor_idx, u_vmax_d, trail_x, trail_y)
        save_gif(frames, os.path.join(GIF_DIR, "_preview_phase_d.gif"), duration_ms=durations)
    elif args.phase == "E":
        seq = build_sequence()
        nodes0 = np.asarray(seq.nodes(0))
        anchor_idx = _anchor_index(nodes0, FIX_COORD)
        anchor_coord = nodes0[anchor_idx]
        u_vmax_c = _seq_u_vmax(seq)
        u_vmax_d = _field_u_vmax(nodes0)
        _, _, trail_x, trail_y = phase_c(seq, anchor_idx, u_vmax_c)
        d_frames, _ = phase_d(seq, anchor_idx, u_vmax_d, trail_x, trail_y)
        panel = build_results_panel(seq, anchor_coord)
        frames, durations = phase_e(d_frames[-1], panel)
        save_gif(frames, os.path.join(GIF_DIR, "_preview_phase_e.gif"), duration_ms=durations)
    elif args.phase == "F":
        seq = build_sequence()
        nodes0 = np.asarray(seq.nodes(0))
        anchor_idx = _anchor_index(nodes0, FIX_COORD)
        anchor_coord = nodes0[anchor_idx]
        panel = build_results_panel(seq, anchor_coord)
        frames, durations = phase_f(panel)
        save_gif(frames, os.path.join(GIF_DIR, "_preview_phase_f.gif"), duration_ms=durations)
    elif args.phase == "all":
        build_full_gif()
