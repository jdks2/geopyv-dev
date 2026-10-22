import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
from matplotlib.collections import LineCollection

from . import _geopyv_dev as _core

plt.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

def _imshow_or_blank(ax, img_path, **kwargs):
    """Helper: show image from path, or blank background if path is None."""
    kwargs.setdefault("cmap", "gist_gray")
    if img_path is not None:
        import cv2
        img_gs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(img_gs, **kwargs)
    else:
        blank = np.zeros((50, 50), dtype=np.uint8)
        ax.imshow(blank, **kwargs)


def _show_save_close(fig, show, block, save, owned=True):
    if save:
        plt.savefig(save, dpi=600)
    if show:
        plt.show(block=block)
    elif owned:
        plt.close(fig)


def _require_solved(obj):
    """Raise RuntimeError if `obj` has not been solved yet.

    Defense-in-depth: most solve-dependent attribute access already raises
    via the underlying Rust getters, but this catches paths (e.g. a
    `quantity` that only reads always-available fields) that wouldn't
    otherwise error on unsolved data.
    """
    if not getattr(obj, 'solved', False):
        raise RuntimeError(
            f"{type(obj).__name__} has not been solved; call solve() first"
        )


def inspect_subset(subset, ax=None, show=True, block=True, save=False,
                    residual=False, colorbar=True, seams=False, **kwargs):
    """Visualise a solved ``Subset``.

    ``residual=True`` shows the converged per-pixel zero-normalised image
    residual (recomputed on demand via ``Subset.residual_map()`` -- never
    persisted, see ``omitted_mode_diagnostic`` in `src/subset.rs`) instead of
    the reference-image crop, on a diverging colormap centred at zero. This
    is the map the warp-adequacy score ``eta_u``/``eta_v`` is a projection
    of: a spatially coherent patch of large residual within an otherwise
    quiet subset is the signature the "partial masking" follow-up work
    (`mds/...` -- see the plan) would cluster on.

    A subset solved with ``solver_options={"masking": "adaptive"}`` shows its
    cut line(s) in red and the masked side in grey; ``seams=True`` adds each
    pass's scores against their alarm levels and the decision.
    """
    coord = subset.coord
    f_coords = np.asarray(subset.f_coords)
    f = np.asarray(subset.f)
    sssig = subset.sssig
    sigma_intensity = subset.sigma_intensity

    f_img_path = getattr(subset, 'f_img_path', None)
    template_size = getattr(subset, 'template_size', None)
    template_shape = getattr(subset, 'template_shape', None)
    if template_size is None:
        template_size = int(np.ceil(np.sqrt(len(f) / np.pi)))

    if residual:
        values = np.asarray(subset.residual_map())
        xi = f_coords[:, 0]
        yi = f_coords[:, 1]
        x_min = int(np.floor(xi.min()))
        y_min = int(np.floor(yi.min()))
        x_max = int(np.ceil(xi.max()))
        y_max = int(np.ceil(yi.max()))
        h = y_max - y_min + 1
        w = x_max - x_min + 1
        display = np.full((h, w), np.nan)
        for k in range(len(values)):
            ix = int(round(float(xi[k]))) - x_min
            iy = int(round(float(yi[k]))) - y_min
            if 0 <= iy < h and 0 <= ix < w:
                display[iy, ix] = float(values[k])
        vmax = np.nanmax(np.abs(display)) if np.isfinite(display).any() else 1.0
        imshow_kwargs = {"cmap": "coolwarm", "vmin": -vmax, "vmax": vmax}
        imshow_kwargs.update(kwargs)

        owned = ax is None
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        im = ax.imshow(display, **imshow_kwargs)
        if colorbar:
            fig.colorbar(im, ax=ax, label="Residual (zn)")
        eta_u = getattr(subset, 'eta_u', None)
        eta_v = getattr(subset, 'eta_v', None)
        if eta_u is not None and eta_v is not None:
            label = f"η_u = {eta_u:.4f}; η_v = {eta_v:.4f}"
        else:
            label = "η not available (unsolved, or unsupported subset_order)"
        ax.text(0.5, -0.05, label, transform=ax.transAxes, ha="center")
        ax.set_axis_off()
        plt.tight_layout()
        _show_save_close(fig, show, block, save, owned=owned)
        return fig, ax

    # Build display image (square crop centred on subset)
    if f_img_path is not None:
        import cv2
        img = cv2.imread(f_img_path, cv2.IMREAD_COLOR)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        x, y = coord
        r = template_size
        x0, x1 = max(0, int(round(x)) - r), min(img_gs.shape[1], int(round(x)) + r + 1)
        y0, y1 = max(0, int(round(y)) - r), min(img_gs.shape[0], int(round(y)) + r + 1)
        display = img_gs[y0:y1, x0:x1].astype(float)
    else:
        xi = f_coords[:, 0]
        yi = f_coords[:, 1]
        x_min = int(np.floor(xi.min()))
        y_min = int(np.floor(yi.min()))
        x_max = int(np.ceil(xi.max()))
        y_max = int(np.ceil(yi.max()))
        h = y_max - y_min + 1
        w = x_max - x_min + 1
        display = np.full((h, w), float(np.mean(f)))
        for k in range(len(f)):
            ix = int(round(float(xi[k]))) - x_min
            iy = int(round(float(yi[k]))) - y_min
            if 0 <= iy < h and 0 <= ix < w:
                display[iy, ix] = float(f[k])

    # For circular templates, mask pixels outside the circle with NaN
    if template_shape == "circle":
        h, w = display.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        row_idx, col_idx = np.ogrid[:h, :w]
        outside = (row_idx - cy) ** 2 + (col_idx - cx) ** 2 > template_size ** 2
        display = display.copy()
        display[outside] = np.nan

    imshow_kwargs = {"cmap": "gist_gray"}
    imshow_kwargs.update(kwargs)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    rec = getattr(subset, "seams", None)
    if seams and rec is None:
        _seams_of(subset)
    r = template_size
    cx, cy = float(coord[0]), float(coord[1])
    ax.imshow(display, extent=[cx - r - 0.5, cx + r + 0.5, cy + r + 0.5, cy - r - 0.5], **imshow_kwargs)
    _draw_seam_overlay(ax, rec, coord, template_size)
    caption = f"Size: {template_size} px; \u03c3_s = {sigma_intensity:.2f}; SSSIG = {sssig:.2E}"
    if seams:
        caption += "\n" + f"action: {rec['action']}" + "\n" + _seam_score_text(rec)
    ax.text(0.5, -0.05, caption, transform=ax.transAxes, ha="center", va="top")
    ax.set_axis_off()
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


# ---------------------------------------------------------------------------
# Adaptive masking (seams) helpers
# ---------------------------------------------------------------------------

# Action -> (label, colour); "unsent" is a node the screen did not send on.
_SEAM_ACTIONS = {
    "unsent": ("not sent on", "#c8c8c8"),
    "nothing": ("tested, left", "#4a6fa5"),
    "order2": ("order 2", "#2a9d8f"),
    "cut": ("cut", "#e76f51"),
    "cutorder2": ("cut + order 2", "#9b5de5"),
    "cut2": ("two cuts", "#d62828"),
}


def _seams_of(obj):
    seams = getattr(obj, "seams", None)
    if seams is None:
        raise ValueError(f"{type(obj).__name__} was not solved with "
                         "solver_options={'masking': 'adaptive'}")
    return seams


def _seam_segment(centre, theta, c, half):
    """End points of the line x cos(theta) + y sin(theta) = c (offsets from
    ``centre``), ``half`` px either side of its closest point to the centre."""
    n = np.array([np.cos(theta), np.sin(theta)])
    t = np.array([-n[1], n[0]])
    mid = np.asarray(centre, float) + c * n
    return mid - half * t, mid + half * t


def _draw_seam_overlay(ax, rec, centre, size):
    """Cut lines (red) and the masked side (grey) of one subset's seam record
    on a crop centred on ``centre`` (absolute coordinates)."""
    if rec is None or not rec["lines"]:
        return
    r = int(size)
    xs = np.arange(-r, r + 1)
    X, Y = np.meshgrid(xs, xs)
    masked = np.zeros(X.shape, bool)
    for line in rec["lines"]:
        s_ = np.cos(line["theta"]) * X + np.sin(line["theta"]) * Y
        c = line["c"]
        keep = s_ < c - rec["gap"] if c > 0 else s_ > c + rec["gap"]
        masked |= ~keep
    masked &= X ** 2 + Y ** 2 <= r ** 2
    overlay = np.zeros(X.shape + (4,))
    overlay[masked] = (0.6, 0.6, 0.6, 0.75)
    cx, cy = float(centre[0]), float(centre[1])
    ax.imshow(overlay, extent=[cx - r - 0.5, cx + r + 0.5, cy + r + 0.5, cy - r - 0.5], zorder=5)
    for line in rec["lines"]:
        a, b = _seam_segment(centre, line["theta"], line["c"], r)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#d62828", lw=1.5, zorder=7)


_STAGE_NAMES = {"pass1": "pass 1", "pass1jolted": "pass 1 after jolt", "pass2": "pass 2 (cut, moved)"}


def _seam_score_text(rec):
    """One line per pass: the three scores against their alarm levels."""
    rows = []
    for ps in rec["passes"]:
        lv = rec["levels"]["pass2" if ps["stage"] == "pass2" else "pass1"]
        th = ps["hinge"]["T"] if ps["hinge"] else 0.0
        tj = ps["jump"]["T"] if ps["jump"] else 0.0
        rows.append(f"{_STAGE_NAMES.get(ps['stage'], ps['stage'])}: bend {ps['T_bend']:.0f}/{lv['bend']:.0f}, "
                    f"hinge {th:.0f}/{lv['hinge']:.0f}, jump {tj:.0f}/{lv['jump']:.0f} "
                    f"-> {ps['decision'].replace('_', ' ')}")
    return "\n".join(rows)


def _picked_pass1(rec, lv):
    """(T_edge, T_bend) of the pass-1 decision."""
    ps = [p for p in rec["passes"] if p["stage"] != "pass2"][-1]
    th = ps["hinge"]["T"] if ps["hinge"] else 0.0
    tj = ps["jump"]["T"] if ps["jump"] else 0.0
    te = tj if tj / lv["jump"] > th / lv["hinge"] else th
    return te, ps["T_bend"]


def _load_img_gs(img_path):
    """Grayscale reference image as float ndarray, or None if no path."""
    if img_path is None:
        return None
    import cv2
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)


def _render_template_crop(ax, img_gs, coord, size, shape):
    """Draw a subset template as a square crop of ``img_gs`` centred on
    ``coord`` (absolute ``[x, y]``); ``circle`` / ``semicircle`` templates
    are masked to their own shape.

    Mirrors :func:`inspect_subset`'s crop rendering; kept separate because
    that function also has an image-less (f/f_coords) path meshes never hit.
    """
    x, y = float(coord[0]), float(coord[1])
    r = int(size)
    if img_gs is None:
        blank = np.zeros((2 * r + 1, 2 * r + 1))
        x0, y0 = int(round(x)) - r, int(round(y)) - r
        x1, y1 = x0 + blank.shape[1], y0 + blank.shape[0]
        display = blank
    else:
        h, w = img_gs.shape
        x0, x1 = max(0, int(round(x)) - r), min(w, int(round(x)) + r + 1)
        y0, y1 = max(0, int(round(y)) - r), min(h, int(round(y)) + r + 1)
        display = img_gs[y0:y1, x0:x1].astype(float).copy()

    hh, ww = display.shape
    cy, cx = y - y0, x - x0
    if shape in ("circle", "semicircle"):
        row_idx, col_idx = np.ogrid[:hh, :ww]
        outside = (row_idx - cy) ** 2 + (col_idx - cx) ** 2 > r ** 2
        if shape == "semicircle":
            outside = outside | (row_idx < cy)  # keep y-offset >= 0 (bottom half)
        display[outside] = np.nan

    ax.imshow(display, cmap="gist_gray", extent=[x0, x1, y1, y0])
    ax.scatter([x], [y], color="red", s=40, marker="x", zorder=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_axis_off()


def inspect_mesh(mesh, subset_idx=None, show_areas=False, ax=None,
                 show=True, block=True, save=False, seams=None, sent=False, **kwargs):
    """Render the mesh over its reference image.

    ``subset_idx`` : int, optional
        Highlight one node. When the mesh records its subset template
        (every solve from schema ``0x07`` on) the plot becomes a crop of
        that node's subset template (like ``subset.inspect()``), with its cut
        line(s) and masked side after an adaptive solve.
    ``seams`` : {"actions", "scores"}, optional
        After ``solver_options={"masking": "adaptive"}``. ``"actions"``: nodes
        coloured by what was done (left, order 2, cut, cut + order 2, two
        cuts), cut lines in red, moved nodes linked to where they started;
        ``sent=True`` also rings the nodes the screen sent on. ``"scores"``:
        edge score against bending score for every node sent on, with the
        alarm and margin lines.
    """
    if seams is not None:
        if seams == "actions":
            return _inspect_mesh_actions(mesh, sent, ax, show, block, save, **kwargs)
        if seams == "scores":
            return _inspect_mesh_scores(mesh, ax, show, block, save, **kwargs)
        raise ValueError(f"seams must be 'actions' or 'scores', got {seams!r}")
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    f_img_path = getattr(mesh, 'f_img_path', None)
    tmpl_shape = getattr(mesh, 'template_shape', None)
    tmpl_sizes = getattr(mesh, 'template_sizes', None)
    if tmpl_sizes is not None:
        tmpl_sizes = np.asarray(tmpl_sizes)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    has_idx = subset_idx is not None and 0 <= subset_idx < len(nodes)

    # --- Per-subset template crop -----------------------------------------
    if has_idx and tmpl_shape is not None and tmpl_sizes is not None:
        coord = nodes[subset_idx]
        size = int(tmpl_sizes[subset_idx])
        img_gs = _load_img_gs(f_img_path)
        caption = f"Subset {subset_idx}: {tmpl_shape} r={size}"
        _render_template_crop(ax, img_gs, coord, size, str(tmpl_shape))
        rec = (getattr(mesh, "seams", None) or {}).get("records", [None] * len(nodes))[subset_idx]
        _draw_seam_overlay(ax, rec, coord, size)
        if rec is not None:
            caption += f"; action: {rec['action']}"
        ax.text(0.5, -0.06, caption, transform=ax.transAxes, ha="center")
        plt.tight_layout()
        _show_save_close(fig, show, block, save, owned=owned)
        return fig, ax

    # --- Full-mesh view --------------------------------------------------
    imshow_kwargs = {"cmap": "gist_gray"}
    imshow_kwargs.update(kwargs)
    _imshow_or_blank(ax, f_img_path, **imshow_kwargs)

    # Element edges -- one LineCollection, not tens of thousands of
    # ax.plot() calls (an 8000-node mesh has ~16k elements).
    tri = nodes[elements[:, :3]]                   # (M, 3, 2)
    segs = np.concatenate(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    ax.add_collection(LineCollection(
        segs, colors="b", linewidths=0.5 if len(elements) <= 2000 else 0.2))

    if len(elements) <= 400:
        # Element index annotations (skipped on a dense mesh, where 10^4
        # labels are unreadable and slow anyway).
        for i, elem in enumerate(elements):
            cx = nodes[elem[:3], 0].mean()
            cy = nodes[elem[:3], 1].mean()
            ax.text(cx, cy, str(i), ha='center', va='center', color='red', fontsize=8)

    if show_areas:
        from matplotlib.patches import Circle as MplCircle
        radius = 10
        for nd in nodes:
            ax.add_patch(MplCircle((nd[0], nd[1]), radius, alpha=0.2, color='blue'))

    if has_idx:
        node = nodes[subset_idx]
        ax.scatter([node[0]], [node[1]], color='red', s=60, zorder=10)

    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def _seam_action_per_node(sm):
    act = np.array(["unsent"] * len(sm["sent"]), dtype=object)
    for i, a in enumerate(sm["action"]):
        if a is not None:
            act[i] = a
    return act


def _inspect_mesh_actions(mesh, sent, ax, show, block, save, **kwargs):
    _require_solved(mesh)
    sm = _seams_of(mesh)
    nodes = np.asarray(mesh.nodes)
    plain = np.asarray(sm["plain_nodes"])
    elements = np.asarray(mesh.elements)
    act = _seam_action_per_node(sm)
    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    _imshow_or_blank(ax, getattr(mesh, "f_img_path", None), alpha=0.6)
    t = nodes[elements[:, :3]]
    ax.add_collection(LineCollection(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]),
                                     colors="k", linewidths=0.1, alpha=0.3))
    for key, (label, colour) in _SEAM_ACTIONS.items():
        m = act == key
        if m.any():
            ax.scatter(nodes[m, 0], nodes[m, 1], s=2 if key == "unsent" else 6, color=colour,
                       label=f"{label} ({int(m.sum())})", zorder=4 if key == "unsent" else 6, lw=0)
    if sent:
        m = np.asarray(sm["sent"])
        ax.scatter(nodes[m, 0], nodes[m, 1], s=24, facecolors="none", edgecolors="k", lw=0.4,
                   label=f"sent on ({int(m.sum())})", zorder=5)
    size = int(np.asarray(mesh.template_sizes)[0]) if getattr(mesh, "template_sizes", None) is not None else 10
    segs = []
    for i, rec in enumerate(sm["records"]):
        if rec is not None:
            for line in rec["lines"]:
                segs.append(_seam_segment(nodes[i], line["theta"], line["c"], 0.5 * size))
    if segs:
        ax.add_collection(LineCollection(segs, colors="#d62828", linewidths=0.8, zorder=7))
    moved = np.hypot(*(nodes - plain).T) > 0
    if moved.any():
        ax.add_collection(LineCollection(np.stack([plain[moved], nodes[moved]], axis=1),
                                         colors="k", linewidths=0.8, zorder=8))
        ax.scatter(plain[moved, 0], plain[moved, 1], s=10, facecolors="none", edgecolors="k",
                   lw=0.6, zorder=8, label=f"moved from ({int(moved.sum())})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, markerscale=2, frameon=False)
    ax.set_title(f"Noise: {sm['noise_source']}", fontsize=9)
    ax.set_axis_off()
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def _inspect_mesh_scores(mesh, ax, show, block, save, **kwargs):
    _require_solved(mesh)
    sm = _seams_of(mesh)
    recs = [(i, r) for i, r in enumerate(sm["records"]) if r is not None and r["passes"]]
    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if not recs:
        ax.text(0.5, 0.5, "no node was sent on", ha="center", transform=ax.transAxes)
    else:
        lv = recs[0][1]["levels"]["pass1"]
        te, tb, acts = [], [], []
        for _, r in recs:
            e, b = _picked_pass1(r, lv)
            te.append(max(e, 1e-1))
            tb.append(max(b, 1e-1))
            acts.append(r["action"])
        te, tb, acts = np.array(te), np.array(tb), np.array(acts)
        for key, (label, colour) in _SEAM_ACTIONS.items():
            m = acts == key
            if m.any():
                ax.scatter(tb[m], te[m], s=6, color=colour, label=f"{label} ({int(m.sum())})", lw=0)
        lo = 0.5 * min(te.min(), tb.min(), lv["bend"])
        hi = 2.0 * max(te.max(), tb.max(), lv["edge"] + lv["margin"])
        x = np.geomspace(lo, hi, 200)
        ax.axhline(lv["edge"], color="k", ls="--", lw=0.8, label="edge alarm")
        ax.axvline(lv["bend"], color="k", ls=":", lw=0.8, label="bending alarm")
        ax.plot(x, x + lv["margin"], color="#d62828", lw=0.8, label="cut margin")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"Bending score, $T_{bend}$ (-)")
        ax.set_ylabel(r"Edge score, $T_{edge}$ (-)")
        ax.legend(fontsize=7, markerscale=2, framealpha=0.8)
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def inspect_sequence(sequence, mesh_idx=None, subset_idx=None, ax=None, show=True, block=True, save=False, **kwargs):
    if mesh_idx is None:
        raise ValueError("mesh_idx is required")
    mesh = sequence.mesh_solution_at(mesh_idx)
    return inspect_mesh(mesh, subset_idx=subset_idx, ax=ax, show=show, block=block, save=save, **kwargs)


def inspect_particle(particle, ax=None, show=True, block=True, save=False, **kwargs):
    coords = np.asarray(particle.coordinates)
    initial = coords[0]
    img_path = getattr(particle, 'image_0_path', None)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    _imshow_or_blank(ax, img_path)
    ax.scatter([initial[0]], [initial[1]], marker='x', color='red', s=100, zorder=10, **kwargs)
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def inspect_field(field, particle_idx=None, ax=None, show=True, block=True, save=False, **kwargs):
    coords = np.asarray(field.coordinates)
    img_path = getattr(field, 'image_0_path', None)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    _imshow_or_blank(ax, img_path)
    ax.scatter(coords[:, 0], coords[:, 1], **kwargs)
    if particle_idx is not None:
        ax.scatter([coords[particle_idx, 0]], [coords[particle_idx, 1]],
                   color='red', s=100, zorder=10)
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def convergence_subset(subset, axes=None, show=True, block=True, save=False, **kwargs):
    _require_solved(subset)
    history = subset.history
    max_norm = 1e-3

    iters = [h[0] for h in history]
    norms = [h[1] for h in history]
    znccs = [h[2] for h in history]

    owned = axes is None
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True)
    else:
        fig = axes[0].get_figure()
    ax = axes
    ax[0].semilogy(iters, norms, marker="o", **kwargs)
    ax[0].semilogy([min(iters), max(iters)], [max_norm, max_norm], "--r")
    ax[0].set_ylabel(r"$\Delta$ Norm (-)")
    ax[1].plot(iters, znccs, marker="o", **kwargs)
    ax[1].plot([min(iters), max(iters)], [0.75, 0.75], "--r")
    ax[1].set_ylabel(r"$C_{ZNCC}$ (-)")
    ax[1].set_xlabel("Iteration (-)")
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, np.array(axes)


def convergence_mesh(mesh, quantity, ax=None, show=True, block=True, save=False, **kwargs):
    _require_solved(mesh)
    valid = {"C_ZNCC", "iterations", "norm"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")

    if quantity == "C_ZNCC":
        data = np.asarray(mesh.c_zncc)
        xlabel = r"$C_{ZNCC}$ (-)"
    elif quantity == "iterations":
        data = np.asarray(mesh.iterations, dtype=float)
        xlabel = "Iterations (-)"
    else:
        data = np.asarray(mesh.norms)
        xlabel = r"$\Delta$ Norm (-)"

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.hist(data, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count (-)")
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


_NO_TIME_MSG = (
    "rate=True needs image times, and this solution has none: pass time= (a time "
    "increment, or one time per image) to the Sequence/Mesh, or use image file names "
    "that carry a timestamp, then re-solve"
)


def contour_sequence(sequence, mesh_idx, quantity, mode="leapfrog", rate=False, **kwargs):
    """Contour pair ``mesh_idx``. For ``u``/``v``/``R``, ``mode`` sets what the
    displacement is measured from: ``"leapfrog"`` (default) the pair's reference
    image, ``"sequential"`` the previous target image. ``rate=True`` divides by
    the matching time step (velocity, px/s)."""
    _require_solved(sequence)
    if mesh_idx is None:
        raise ValueError("mesh_idx is required")
    if mode not in ("leapfrog", "sequential"):
        raise ValueError(f"mode must be 'leapfrog' or 'sequential', got {mode!r}")
    mesh = sequence.mesh_solution_at(mesh_idx)
    if quantity in ("u", "v", "R"):
        displacements, dt = sequence.increment_displacements(mesh_idx, mode)
        return _contour_mesh(mesh, quantity, np.asarray(displacements), dt, rate, **kwargs)
    return contour_mesh(mesh, quantity, rate=rate, **kwargs)


def convergence_sequence(sequence, mesh_idx=None, subset_idx=None, quantity="C_ZNCC",
                          ax=None, show=True, block=True, save=False, **kwargs):
    _require_solved(sequence)
    if mesh_idx is not None:
        mesh = sequence.mesh_solution_at(mesh_idx)
        return convergence_mesh(mesh, quantity, ax=ax, show=show, block=block, save=save, **kwargs)
    else:
        all_data = np.concatenate([np.asarray(arr) for arr in sequence.all_c_zncc()])
        owned = ax is None
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        ax.hist(all_data, **kwargs)
        ax.set_xlabel(r"$C_{ZNCC}$ (-)")
        ax.set_ylabel("Count (-)")
        plt.tight_layout()
        _show_save_close(fig, show, block, save, owned=owned)
        return fig, ax


# Adaptive-masking quantities for ``Mesh.contour``.
_SEAM_QUANTITIES = ("noise", "still_mismatch", "correction")


def _seam_quantity(mesh, quantity):
    sm = _seams_of(mesh)
    if quantity == "noise":
        return np.asarray(sm["noise_scale"])
    if quantity == "still_mismatch":
        if sm["still_mismatch"] is None:
            raise ValueError("still_mismatch needs a still pair (Mesh(..., noise=...) or a "
                             f"Sequence); this mesh used {sm['noise_source']}")
        return np.asarray(sm["still_mismatch"])
    d = np.asarray(mesh.displacements) - np.asarray(sm["plain_p"])[:, :2]
    return np.hypot(d[:, 0], d[:, 1])


def contour_mesh(mesh, quantity, rate=False, **kwargs):
    """Contour a solved mesh. ``rate=True`` plots ``u``/``v``/``R`` as
    velocities: displacement / (``g_time - f_time``), in px/s."""
    _require_solved(mesh)
    displacements = np.asarray(mesh.displacements) if quantity in ("u", "v", "R") else None
    return _contour_mesh(mesh, quantity, displacements, mesh.dt, rate, **kwargs)


def _contour_mesh(mesh, quantity, displacements, dt, rate, ax=None, show=True, block=True,
                  save=False, mesh_overlay=False, calibration=None, **kwargs):
    _require_solved(mesh)
    valid = {"C_ZNCC", "iterations", "norm", "u", "v", "R", *_SEAM_QUANTITIES}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")
    if calibration is not None and quantity not in {"u", "v", "R"}:
        raise ValueError(
            f"calibration only applies to quantity in ['R', 'u', 'v'], not {quantity!r}"
        )
    if rate and quantity not in {"u", "v", "R"}:
        raise ValueError(f"rate only applies to quantity in ['R', 'u', 'v'], not {quantity!r}")
    if rate and dt is None:
        raise ValueError(_NO_TIME_MSG)

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    if calibration is not None:
        # Node positions stay in pixel space (the background image and mesh layout are
        # never re-projected — lens distortion means "calibrated positions" would be a
        # nonlinear warp, not just a scale, and would no longer align with the image).
        # Only the displacement values plotted for u/v/R are converted, per node.
        displacements = calibration.i2o(nodes + displacements) - calibration.i2o(nodes)

    unit = "calibrated" if calibration is not None else "px"
    if rate:
        unit += "/s"
        displacements = displacements / dt if dt != 0 else np.full_like(displacements, np.nan)
    labels = {
        "C_ZNCC": r"$C_{ZNCC}$ (-)",
        "iterations": "Iterations (-)",
        "norm": r"$\Delta$ Norm (-)",
        "u": f"u ({unit})",
        "v": f"v ({unit})",
        "R": f"R ({unit})",
        "noise": "Mismatch noise (px)",
        "still_mismatch": "Frame 0\u20131 mismatch (px)",
        "correction": "Change from the plain solve (px)",
    }
    if quantity == "C_ZNCC":
        values = np.asarray(mesh.c_zncc)
    elif quantity == "iterations":
        values = np.asarray(mesh.iterations, dtype=float)
    elif quantity == "norm":
        values = np.asarray(mesh.norms)
    elif quantity == "u":
        values = displacements[:, 0]
    elif quantity == "v":
        values = displacements[:, 1]
    elif quantity == "R":
        values = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
    else:
        values = _seam_quantity(mesh, quantity)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    f_img_path = getattr(mesh, 'f_img_path', None)
    _imshow_or_blank(ax, f_img_path)

    tri_obj = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements[:, :3])
    cf = ax.tricontourf(tri_obj, values, **kwargs)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(labels[quantity])

    if mesh_overlay:
        ax.triplot(tri_obj, color="k", alpha=0.25, linewidth=0.5)

    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


_WARP_LABELS_HISTORY = [
    r"$u$ ($px$)",
    r"$v$ ($px$)",
    r"$du/dx$ ($-$)",
    r"$dv/dx$ ($-$)",
    r"$du/dy$ ($-$)",
    r"$dv/dy$ ($-$)",
    r"$d^2u/dx^2$ ($-$)",
    r"$d^2v/dx^2$ ($-$)",
    r"$d^2u/dxdy$ ($-$)",
    r"$d^2v/dxdy$ ($-$)",
    r"$d^2u/dy^2$ ($-$)",
    r"$d^2v/dy^2$ ($-$)",
]

_STRAIN_LABELS_HISTORY = [
    r"$\epsilon_{xx}$ ($-$)",
    r"$\epsilon_{yy}$ ($-$)",
    r"$\epsilon_{zz}$ ($-$)",
    r"$\epsilon_{yz}$ ($-$)",
    r"$\epsilon_{xz}$ ($-$)",
    r"$\epsilon_{xy}$ ($-$)",
]


def _time_axis(particle, n):
    """``(x, xlabel)``: the particle's image times when stored, else image number."""
    t = getattr(particle, "time", None)
    if t is not None:
        return np.asarray(t)[:n], r"Time, $t$ ($s$)"
    return np.arange(n, dtype=float), r"Image Number, $i$ ($-$)"


def _rate_label(label, displacement):
    """``label`` with its unit replaced by a per-second one: ``px/s`` for a
    displacement, ``/s`` otherwise."""
    name = label.split(" (")[0]
    return f"{name} ($px/s$)" if displacement else f"{name} ($/s$)"


def _check_mode(mode, allowed=("leapfrog", "sequential")):
    if mode not in allowed:
        raise ValueError(f"mode must be one of {list(allowed)!r}, got {mode!r}")


def history_particle(particle, quantity="warps", components=None, rate=False, mode="leapfrog",
                     ax=None, show=True, block=True, save=None,
                     xlim=None, ylim=None, **kwargs):
    """Plot a particle's series against time (image number if it has no
    stored times). ``rate=True`` plots the rate over each increment instead,
    measured from that row's reference image (``mode="leapfrog"``, default)
    or from the previous image (``mode="sequential"``, drawn as steps)."""
    _require_solved(particle)
    valid = {"warps", "strains", "vol_strains"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")
    _check_mode(mode)
    if rate and getattr(particle, "time", None) is None:
        raise ValueError(_NO_TIME_MSG)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    n = np.asarray(particle.coordinates).shape[0]
    x, xlabel = _time_axis(particle, n)

    def line(series_quantity, component, label, displacement=False):
        if not rate:
            if series_quantity == "warps":
                y = np.asarray(particle.warps)[:, component]
            elif series_quantity == "strains":
                y = np.asarray(particle.strains)[:, component]
            else:
                y = np.asarray(particle.vol_strains)
            ax.plot(x[:len(y)], y, label=label, **kwargs)
            return
        y = np.asarray(particle.series_increments(series_quantity, component, mode, True))
        label = _rate_label(label, displacement)
        if mode == "sequential":
            ax.stairs(y, x[:len(y) + 1], label=label, baseline=None, **kwargs)
        else:
            ax.plot(x[1:len(y) + 1], y, marker=".", label=label, **kwargs)

    if quantity == "warps":
        data = np.asarray(particle.warps)
        if components is None:
            components = range(data.shape[1])
        for c in components:
            label = _WARP_LABELS_HISTORY[c] if c < len(_WARP_LABELS_HISTORY) else str(c)
            line("warps", c, label, displacement=c < 2)
        ax.set_ylabel("Rate" if rate else "Value")
        ax.legend()

    elif quantity == "strains":
        for c in (0, 1, 5):
            line("strains", c, _STRAIN_LABELS_HISTORY[c])
        line("vol_strains", 0, r"$\epsilon_{vol}$ ($-$)")
        ax.set_ylabel(r"Strain rate, $\dot{\epsilon}$ ($/s$)" if rate else r"Strain, $\epsilon$")
        ax.legend()

    elif quantity == "vol_strains":
        line("vol_strains", 0, r"$\epsilon_{vol}$ ($-$)")
        ax.set_ylabel(r"Volumetric strain rate, $\dot{\epsilon}_{vol}$ ($/s$)" if rate
                      else r"Volumetric strain, $\epsilon_{vol}$ ($-$)")

    ax.set_xlabel(xlabel)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def history_field(field, particle_index, quantity="warps", components=None,
                  ax=None, show=True, block=True, save=None,
                  xlim=None, ylim=None, **kwargs):
    """``history_particle`` for particle ``particle_index`` of a field --
    ``rate=`` and ``mode=`` are passed through."""
    _require_solved(field)
    particle = field.particles[particle_index]
    return history_particle(particle, quantity, components=components,
                            ax=ax, show=show, block=block, save=save,
                            xlim=xlim, ylim=ylim, **kwargs)


def _trace_values(particle, quantity, component, mode, rate):
    """One colour value per path segment (row ``k - 1`` -> ``k``): the change
    of the series since the segment's reference image (``"leapfrog"``) or the
    previous image (``"sequential"``), per second if ``rate``; or the time
    (image number without times) the segment ends at (``"time"``)."""
    if mode == "time":
        n = np.asarray(particle.coordinates).shape[0]
        return _time_axis(particle, n)[0][1:]
    return np.asarray(particle.series_increments(quantity, component, mode, rate))


def _trace_label(particle, quantity, component, mode, rate):
    if mode == "time":
        return _time_axis(particle, 1)[1]
    if quantity == "warps":
        label = _WARP_LABELS_HISTORY[component] if component < len(_WARP_LABELS_HISTORY) else str(component)
    elif quantity == "strains":
        label = _STRAIN_LABELS_HISTORY[component] if component < len(_STRAIN_LABELS_HISTORY) else str(component)
    else:
        label = r"$\epsilon_{vol}$ ($-$)"
    return _rate_label(label, quantity == "warps" and component < 2) if rate else label


def _check_trace_args(particle, quantity, mode, rate):
    valid = {"warps", "strains", "vol_strains"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")
    _check_mode(mode, ("leapfrog", "sequential", "time"))
    if rate and mode == "time":
        raise ValueError("rate=True needs mode='leapfrog' or 'sequential', not 'time'")
    if rate and getattr(particle, "time", None) is None:
        raise ValueError(_NO_TIME_MSG)


def trace_particle(particle, quantity="warps", component=0, mode="leapfrog", rate=False,
                   imshow=True, ax=None, show=True, block=True, save=None,
                   xlim=None, ylim=None, **kwargs):
    """Draw the particle's path, each segment coloured per ``mode``: the
    change since its reference image (``"leapfrog"``, default), since the
    previous image (``"sequential"``), or the time it ends at (``"time"``).
    ``rate=True`` divides the leapfrog/sequential change by its time step."""
    _require_solved(particle)
    _check_trace_args(particle, quantity, mode, rate)

    coords = np.asarray(particle.coordinates)
    values = _trace_values(particle, quantity, component, mode, rate)
    label = _trace_label(particle, quantity, component, mode, rate)

    points = coords.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(np.nanmin(values), np.nanmax(values))
    lc = LineCollection(segments, cmap="viridis", norm=norm, **kwargs)
    lc.set_array(values)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if imshow:
        img_path = getattr(particle, 'image_0_path', None)
        _imshow_or_blank(ax, img_path)
    else:
        ax.set_aspect("equal", "box")
        ax.autoscale()

    ax.add_collection(lc)
    if not imshow:
        ax.autoscale_view()
    fig.colorbar(lc, ax=ax, label=label)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def trace_field(field, quantity="warps", component=0, mode="leapfrog", rate=False,
                imshow=True, ax=None, show=True, block=True, save=None,
                xlim=None, ylim=None, **kwargs):
    """Every particle's path, coloured as in ``trace_particle``."""
    _require_solved(field)
    particles = field.particles
    if not particles:
        raise ValueError("field has no particles")
    _check_trace_args(particles[0], quantity, mode, rate)

    all_segments = []
    all_values = []
    for p in particles:
        coords = np.asarray(p.coordinates)
        pts = coords.reshape(-1, 1, 2)
        all_segments.append(np.concatenate([pts[:-1], pts[1:]], axis=1))
        all_values.append(_trace_values(p, quantity, component, mode, rate))

    all_segments = np.concatenate(all_segments, axis=0)
    all_values = np.concatenate(all_values)
    label = _trace_label(particles[0], quantity, component, mode, rate)

    norm = plt.Normalize(np.nanmin(all_values), np.nanmax(all_values))
    lc = LineCollection(all_segments, cmap="viridis", norm=norm, **kwargs)
    lc.set_array(all_values)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if imshow:
        img_path = getattr(field, 'image_0_path', None)
        _imshow_or_blank(ax, img_path)
    else:
        ax.set_aspect("equal", "box")

    ax.add_collection(lc)
    if not imshow:
        ax.autoscale_view()
    fig.colorbar(lc, ax=ax, label=label)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def _field_contour_label(quantity, rate, absolute, calibrated):
    name = quantity + (" (abs)" if absolute else "")
    if not rate:
        return name
    unit = "calibrated" if calibrated else "px"
    if quantity == "theta_p":
        return "theta_p of strain rate (rad)"
    if quantity == "gamma_max_grad":
        return f"{name} (/{unit}/s)"
    if quantity in ("u", "v", "R"):
        return f"{name} ({unit}/s)"
    return f"{name} (/s)"


def _resolve_window(window, inc_no):
    """contour_field's ``window`` as a concrete half-open ``(start, stop)``
    increment range (``None`` -> ``None``, i.e. all increments). Accepts a
    2-element list/tuple (slice bounds, ``None``/negative allowed), a
    ``slice`` (step 1 only), or an int (that single increment)."""
    if window is None:
        return None
    if isinstance(window, (list, tuple)) and len(window) == 2:
        window = slice(window[0], window[1])
    if isinstance(window, slice):
        start, stop, step = window.indices(inc_no)
        if step != 1:
            raise ValueError(f"window slice step must be 1, got {step}")
        return (start, stop)
    k = int(window)
    if k < 0:
        k += inc_no
    return (k, k + 1)


def contour_field(field, quantity, window=None, increment=None, mode="leapfrog", rate=False,
                  absolute=False, deformed=False, ax=None, show=True, block=True, save=False,
                  **kwargs):
    """Filled contour of one reduced value per particle.

    Rows: ``window`` (``None``: all, i.e. the total; an int ``k``: the value
    at row ``k``; a pair/slice: last-minus-first over that half-open range),
    or ``increment=k``, the change *into* row ``k`` -- measured from that
    row's reference image (``mode="leapfrog"``, default) or from the previous
    image (``mode="sequential"``). ``absolute`` sums the absolute row-to-row changes.

    ``rate=True`` divides by the elapsed time between the first and last rows
    (the stored image times): strain rates in /s, ``u``/``v``/``R`` as
    velocities in px/s (``R`` the speed). Principal quantities (``ep1``,
    ``ep2``, ``gamma_max``, ``theta_p``) are then those of the strain-rate
    tensor, not the change in the principal strains.

    Values, positions and triangles all come from the core
    (``Field.contour_values`` / ``contour_coordinates`` /
    ``contour_triangles``), the same functions the GUI draws from. The
    triangulation covers only the field's region (exclusions and concave
    boundary sections left empty); a field saved before regions were stored
    falls back to the full convex hull, with a warning.

    ``deformed=True`` draws at the particle positions at the window's last
    increment instead of the initial ones (same triangles).

    gamma_max_grad requires the Field to have been solved via
    Field.solve() with strain_method left as the default or a
    MeshlessParams -- the spatial gradient of gamma_max is computed once,
    at solve time, with those MeshlessParams (the defaults under the
    default subset average).
    """
    _require_solved(field)
    valid = {"u", "v", "R", "ep_xx", "ep_yy", "ep_xy", "ep_vol",
             "ep1", "ep2", "gamma_max", "theta_p", "gamma_max_grad"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")

    _check_mode(mode)
    if window is not None and increment is not None:
        raise ValueError("give window or increment, not both")
    if rate and getattr(field, "time", None) is None:
        raise ValueError(_NO_TIME_MSG)
    w = _resolve_window(window, field.inc_no)
    values = np.asarray(field.contour_values(
        quantity, window=w, increment=increment, mode=mode, rate=rate, absolute=absolute))
    coords = np.asarray(field.contour_coordinates(
        deformed=deformed, window=w, increment=increment, mode=mode))
    triangles = np.asarray(field.contour_triangles())
    if field.region is None:
        warnings.warn(
            "this Field predates stored regions (.pyv < 0x08): the contour fills the "
            "particles' full convex hull, including any exclusions -- re-solve to fix",
            stacklevel=2,
        )

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    img_path = getattr(field, 'image_0_path', None)
    _imshow_or_blank(ax, img_path)

    if len(triangles) > 0:
        mpl_tri = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
        cf = ax.tricontourf(mpl_tri, values, **kwargs)
    else:
        cf = ax.scatter(coords[:, 0], coords[:, 1], c=values, **kwargs)

    label = _field_contour_label(quantity, rate, absolute, getattr(field, "calibrated", False))

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(label)

    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


# ---------------------------------------------------------------------------
# Validation plot helpers
# ---------------------------------------------------------------------------

_WARP_LABELS = [
    r"Horizontal displacement, $u$ ($px$)",
    r"Vertical displacement, $v$ ($px$)",
    r"Horizontal normal strain, $\epsilon_{xx}$ ($-$)",
    r"Shear strain component, $dv/dx$ ($-$)",
    r"Shear strain component, $du/dy$ ($-$)",
    r"Vertical normal strain, $\epsilon_{yy}$ ($-$)",
    r"Strain gradient component, $d^2u/dx^2$ ($-$)",
    r"Strain gradient component, $d^2v/dx^2$ ($-$)",
    r"Strain gradient component, $d^2u/dxdy$ ($-$)",
    r"Strain gradient component, $d^2v/dxdy$ ($-$)",
    r"Strain gradient component, $d^2u/dy^2$ ($-$)",
    r"Strain gradient component, $d^2v/dy^2$ ($-$)",
    r"Rotation, $\theta$ ($^o$)",
    r"Pure shear strain, $\epsilon_{xy}$ ($-$)",
]

_NOISE_AXES_TITLES = [
    r"(a) $1^{st}$ Order Subsets, $1^{st}$ Order Mesh",
    r"(b) $2^{nd}$ Order Subsets, $1^{st}$ Order Mesh",
    r"(c) $1^{st}$ Order Subsets, $2^{nd}$ Order Mesh",
    r"(d) $2^{nd}$ Order Subsets, $2^{nd}$ Order Mesh",
]

_COLOURS = ["r", "b", "g", "orange", "purple", "k", "brown"]
_MARKERS = ["o", "^", "s", "v", "D", "P"]


def _x_series(solution, component):
    pm = np.asarray(solution.pm)     # (image_no, 12)
    if component == 12:
        # pm[i, 0] = angle (radians) at step i for WarpMode::Rotation
        return np.degrees(pm[1:, 0])
    elif component == 13:
        return np.abs(pm[1:, 3])
    else:
        return np.abs(pm[1:, component])


def _std_error_series(field_data, component):
    applied  = np.asarray(field_data.applied)   # (n_frames, n_particles, 12)
    observed = np.asarray(field_data.observed)
    if component >= 12:
        l2 = np.sqrt(np.sum((applied[:, :, :2] - observed[:, :, :2]) ** 2, axis=2))
        return np.std(l2, axis=1)
    err = applied[:, :, component] - observed[:, :, component]
    return np.std(err, axis=1)


def _mean_error_series(field_data):
    applied  = np.asarray(field_data.applied)
    observed = np.asarray(field_data.observed)
    diff = applied[:, :, :2] - observed[:, :, :2]
    l2 = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.mean(l2, axis=1)


def standard_error_validation(solution, component, observing=None,
                               scale="log", plot="scatter",
                               xlim=None, ylim=None,
                               prev_series=None, prev_series_label=None,
                               ax=None, show=True, block=True, save=None, **kwargs):
    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    x = _x_series(solution, component)
    for idx, fd in enumerate(solution.fields):
        y = _std_error_series(fd, component if observing is None else observing)
        colour = _COLOURS[idx % len(_COLOURS)]
        marker = _MARKERS[idx % len(_MARKERS)]
        label  = solution.labels[idx] if idx < len(solution.labels) else str(idx)
        if plot == "scatter":
            ax.scatter(x, y, color=colour, marker=marker, label=label, **kwargs)
        else:
            ax.plot(x, y, color=colour, label=label, **kwargs)
    if prev_series is not None:
        ax.plot(prev_series[:,0], prev_series[:,1], color="gray", linestyle="--",
                label=prev_series_label or "previous")
    ax.set_xscale(scale)
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(_WARP_LABELS[component] if component < len(_WARP_LABELS) else "")
    if observing is not None:
        ax.set_ylabel(r"Error, $\Delta$" + _WARP_LABELS[observing])
    else:
        ax.set_ylabel(r"Standard error, $\rho_{px}$ ($px$)")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def mean_error_validation(solution, component,
                          scale="log", plot="scatter",
                          xlim=None, ylim=None,
                          prev_series=None, prev_series_label=None,
                          ax=None, show=True, block=True, save=None, **kwargs):
    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    x = _x_series(solution, component)
    for idx, fd in enumerate(solution.fields):
        y = _mean_error_series(fd)
        colour = _COLOURS[idx % len(_COLOURS)]
        marker = _MARKERS[idx % len(_MARKERS)]
        label  = solution.labels[idx] if idx < len(solution.labels) else str(idx)
        if plot == "scatter":
            ax.scatter(x, y, color=colour, marker=marker, label=label, **kwargs)
        else:
            ax.plot(x, y, color=colour, marker=marker, label=label, **kwargs)
    if prev_series is not None:
        ax.plot(x, prev_series, color="gray", linestyle="--",
                label=prev_series_label or "previous")
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(_WARP_LABELS[component] if component < len(_WARP_LABELS) else "")
    ax.set_ylabel("Mean L2 displacement error ($px$)")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def noise_standard_error_validation(solution, component, observing=None,
                                     scale="log", plot="scatter",
                                     xlim=None, ylim=None,
                                     axes=None, show=True, block=True, save=None, **kwargs):
    owned = axes is None
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    else:
        fig = axes.flat[0].get_figure()
    x = _x_series(solution, component)
    for panel_idx, ax in enumerate(axes.flat):
        for sub_idx in range(len(solution.fields)):
            if sub_idx // 4 != panel_idx:
                continue
            fd = solution.fields[sub_idx]
            y  = _std_error_series(fd, component if observing is None else observing)
            colour = _COLOURS[sub_idx % len(_COLOURS)]
            marker = _MARKERS[sub_idx % len(_MARKERS)]
            label  = solution.labels[sub_idx] if sub_idx < len(solution.labels) else str(sub_idx)
            if plot == "scatter":
                ax.scatter(x, y, color=colour, marker=marker, label=label, **kwargs)
            else:
                ax.plot(x, y, color=colour, marker=marker, label=label, **kwargs)
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if panel_idx < len(_NOISE_AXES_TITLES):
            ax.set_title(_NOISE_AXES_TITLES[panel_idx])
        ax.set_xlabel(_WARP_LABELS[component] if component < len(_WARP_LABELS) else "")
        ax.set_ylabel("Standard error")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
    axes[1, 1].legend(loc="upper left")
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, axes


def noise_mean_error_validation(solution, component,
                                 scale="log", plot="scatter",
                                 xlim=None, ylim=None,
                                 axes=None, show=True, block=True, save=None, **kwargs):
    owned = axes is None
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    else:
        fig = axes.flat[0].get_figure()
    x = _x_series(solution, component)
    for panel_idx, ax in enumerate(axes.flat):
        for sub_idx in range(len(solution.fields)):
            if sub_idx // 4 != panel_idx:
                continue
            fd = solution.fields[sub_idx]
            y  = _mean_error_series(fd)
            colour = _COLOURS[sub_idx % len(_COLOURS)]
            marker = _MARKERS[sub_idx % len(_MARKERS)]
            label  = solution.labels[sub_idx] if sub_idx < len(solution.labels) else str(sub_idx)
            if plot == "scatter":
                ax.scatter(x, y, color=colour, marker=marker, label=label, **kwargs)
            else:
                ax.plot(x, y, color=colour, marker=marker, label=label, **kwargs)
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if panel_idx < len(_NOISE_AXES_TITLES):
            ax.set_title(_NOISE_AXES_TITLES[panel_idx])
        ax.set_xlabel(_WARP_LABELS[component] if component < len(_WARP_LABELS) else "")
        ax.set_ylabel("Mean L2 displacement error ($px$)")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
    axes[1, 1].legend(loc="upper left")
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, axes


def strain_error_validation(solution,
                             scale="log", plot="scatter",
                             xlim=None, ylim=None,
                             axes=None, show=True, block=True, save=None, **kwargs):
    owned = axes is None
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    else:
        fig = axes[0].get_figure()
    mult = np.asarray(solution.mult)[1:]  # (n_frames,) x-axis

    for idx, fd in enumerate(solution.fields):
        applied  = np.asarray(fd.applied)
        observed = np.asarray(fd.observed)
        colour = _COLOURS[idx % len(_COLOURS)]
        marker = _MARKERS[idx % len(_MARKERS)]
        label  = solution.labels[idx] if idx < len(solution.labels) else str(idx)

        diff_disp = applied[:, :, :2] - observed[:, :, :2]
        l2 = np.sqrt(np.sum(diff_disp ** 2, axis=2))
        y0 = np.std(l2, axis=1)

        err3 = applied[:, :, 3] - observed[:, :, 3]
        err4 = applied[:, :, 4] - observed[:, :, 4]
        y1 = np.std(0.5 * (err3 + err4), axis=1)

        def det_err(w):
            return (1.0 + w[:, :, 2]) * (1.0 + w[:, :, 5]) - w[:, :, 3] * w[:, :, 4]
        vol_err = det_err(applied) - det_err(observed)
        y2 = np.std(vol_err, axis=1)

        for ax, y in zip(axes, [y0, y1, y2]):
            if plot == "scatter":
                ax.scatter(mult, y, color=colour, marker=marker, label=label, **kwargs)
            else:
                ax.plot(mult, y, color=colour, marker=marker, label=label, **kwargs)

    for ax in axes:
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Std L2 displacement error ($px$)")
    axes[1].set_ylabel(r"Std shear strain error ($-$)")
    axes[2].set_ylabel(r"Std volumetric strain error ($-$)")
    axes[2].set_xlabel("Warp multiplier")
    axes[0].legend(loc="upper left")
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, axes


def spatial_error_validation(solution, field_index, time_index, quantity="R",
                              imshow=True, colorbar=True,
                              ticks=None, alpha=0.5, levels=None,
                              xlim=None, ylim=None,
                              ax=None, show=True, block=True, save=None, **kwargs):
    fd = solution.fields[field_index]
    applied  = np.asarray(fd.applied)   # (n_frames, n_particles, 12)
    observed = np.asarray(fd.observed)
    coords   = np.asarray(fd.coordinates)  # (n_frames+1, n_particles, 2)

    # Particle positions at time_index (clamped to available frames)
    t = min(time_index, coords.shape[0] - 1)
    xy = coords[t]  # (n_particles, 2)
    x_pos = xy[:, 0]
    y_pos = xy[:, 1]

    frame = min(time_index, applied.shape[0] - 1)
    if quantity == "u":
        err = applied[frame, :, 0] - observed[frame, :, 0]
    elif quantity == "v":
        err = applied[frame, :, 1] - observed[frame, :, 1]
    else:  # "R" — L2 displacement error
        du = applied[frame, :, 0] - observed[frame, :, 0]
        dv = applied[frame, :, 1] - observed[frame, :, 1]
        err = np.sqrt(du**2 + dv**2)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if imshow:
        img_path = fd.image_0_path
        _imshow_or_blank(ax, img_path)

    triang = tri.Triangulation(x_pos, y_pos)
    contour_kwargs = {}
    if levels is not None:
        contour_kwargs["levels"] = levels
    contour_kwargs.update(kwargs)
    cf = ax.tricontourf(triang, err, alpha=alpha, **contour_kwargs)
    if colorbar:
        cb = fig.colorbar(cf, ax=ax)
        if ticks is not None:
            cb.set_ticks(ticks)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


# ---------------------------------------------------------------------------
# Calibration plots
# ---------------------------------------------------------------------------

def inspect_calibration(calibration, image_index=0, ax=None, show=True, block=True, save=None):
    """One calibration image with its detected ChArUco corners overlaid."""
    _require_solved(calibration)
    import cv2

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    path = calibration._accepted_images[image_index]
    frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ax.imshow(frame)
    corners = calibration._all_corners[image_index]
    ax.scatter(corners[:, :, 0], corners[:, :, 1], color="r", s=12)
    ax.set_title(path)
    ax.set_axis_off()
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def visualise_calibration(calibration, ax=None, show=True, block=True, save=None):
    """Detected ChArUco corners across every accepted image — a coverage map."""
    _require_solved(calibration)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for corners in calibration._all_corners:
        ax.scatter(corners[:, :, 0], corners[:, :, 1], color="r", s=6, alpha=0.6)
    h, w = calibration._imsize
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # row-down image convention
    ax.set_aspect("equal")
    ax.set_title("Detected ChArUco corners — all accepted images")
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def _calibration_quantity_label(quantity):
    return {
        "R": r"Resultant, $R$ ($px$)",
        "u": r"Horizontal displacement, $u$ ($px$)",
        "v": r"Vertical displacement, $v$ ($px$)",
    }[quantity]


def contour_calibration(calibration, quantity="R", points=True, colorbar=True, ticks=None,
                         alpha=0.75, levels=None, axis=True, xlim=None, ylim=None,
                         ax=None, show=True, block=True, save=None):
    """Per-corner undistortion-magnitude map — how much the lens model warps
    each detected board corner, independent of the extrinsic pose/reprojection."""
    _require_solved(calibration)
    valid = {"u", "v", "R"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")
    import cv2

    image_points = np.concatenate(calibration._all_corners, axis=0).reshape(-1, 2)
    undistorted = cv2.undistortImagePoints(
        image_points, calibration._intmat, calibration._dist
    ).reshape(-1, 2)
    delta = image_points - undistorted
    if quantity == "R":
        values = np.sqrt(np.sum(delta**2, axis=1))
    elif quantity == "u":
        values = delta[:, 0]
    else:
        values = delta[:, 1]

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots(num="Distortion magnitude")
    else:
        fig = ax.get_figure()
    cf = ax.tricontourf(image_points[:, 0], image_points[:, 1], values,
                         alpha=alpha, levels=levels, extend="both")
    if points:
        ax.scatter(image_points[:, 0], image_points[:, 1], color="k", s=4)
    if not axis:
        ax.set_axis_off()
    ax.set_aspect("equal")
    h, w = calibration._imsize
    ax.set_xlim(xlim if xlim is not None else (0, w))
    ax.set_ylim(ylim if ylim is not None else (h, 0))
    if colorbar:
        fig.colorbar(cf, ax=ax, label=_calibration_quantity_label(quantity), ticks=ticks)
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def error_calibration(calibration, quantity="R", points=True, colorbar=True, ticks=None,
                       alpha=0.75, levels=None, axis=True, xlim=None, ylim=None,
                       ax=None, show=True, block=True, save=None):
    """Reprojection error map — detected corners vs. the solved camera model's
    own reprojection of the corresponding board points, per accepted image."""
    _require_solved(calibration)
    valid = {"u", "v", "R"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")

    reimgpnts = np.concatenate(calibration._reimgpnts, axis=0)
    imgpnts = np.concatenate(
        [c.reshape(-1, 2) for c in calibration._all_corners], axis=0
    )
    delta = reimgpnts - imgpnts
    if quantity == "R":
        error = np.sqrt(np.sum(delta**2, axis=1))
    elif quantity == "u":
        error = delta[:, 0]
    else:
        error = delta[:, 1]

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots(num="Reprojection error")
    else:
        fig = ax.get_figure()
    cf = ax.tricontourf(imgpnts[:, 0], imgpnts[:, 1], error, alpha=alpha, levels=levels, extend="both")
    if points:
        start = 0
        for path, corners in zip(calibration._accepted_images, calibration._all_corners):
            n = corners.reshape(-1, 2).shape[0]
            ax.scatter(imgpnts[start:start + n, 0], imgpnts[start:start + n, 1], label=path, s=8)
            start += n
        ax.legend(fontsize=6)
    if not axis:
        ax.set_axis_off()
    ax.set_aspect("equal")
    h, w = calibration._imsize
    ax.set_xlim(xlim if xlim is not None else (0, w))
    ax.set_ylim(ylim if ylim is not None else (h, 0))
    if colorbar:
        fig.colorbar(cf, ax=ax, label=_calibration_quantity_label(quantity), ticks=ticks)
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax
