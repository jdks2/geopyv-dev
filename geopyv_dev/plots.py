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
                    residual=False, colorbar=True, **kwargs):
    """Visualise a solved ``Subset``.

    ``residual=True`` shows the converged per-pixel zero-normalised image
    residual (recomputed on demand via ``Subset.residual_map()`` -- never
    persisted, see ``omitted_mode_diagnostic`` in `src/subset.rs`) instead of
    the reference-image crop, on a diverging colormap centred at zero. This
    is the map the warp-adequacy score ``eta_u``/``eta_v`` is a projection
    of: a spatially coherent patch of large residual within an otherwise
    quiet subset is the signature the "partial masking" follow-up work
    (`mds/...` -- see the plan) would cluster on.
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
    ax.imshow(display, **imshow_kwargs)
    ax.text(0.5, -0.05,
            f"Size: {template_size} px; \u03c3_s = {sigma_intensity:.2f}; SSSIG = {sssig:.2E}",
            transform=ax.transAxes, ha="center")
    ax.set_axis_off()
    plt.tight_layout()
    _show_save_close(fig, show, block, save, owned=owned)
    return fig, ax


def _load_img_gs(img_path):
    """Grayscale reference image as float ndarray, or None if no path."""
    if img_path is None:
        return None
    import cv2
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)


def _render_template_crop(ax, img_gs, coord, size, shape, cut_coords=None):
    """Draw a subset template as a square crop of ``img_gs`` centred on
    ``coord`` (absolute ``[x, y]``); ``circle`` / ``semicircle`` templates
    are masked to their own shape. ``cut_coords`` (absolute ``[x, y]`` from
    :func:`_geopyv_dev.zoned_subset_footprint`) are the zone-removed pixels,
    overlaid in red.

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
    if cut_coords is not None and len(cut_coords):
        ax.scatter(cut_coords[:, 0], cut_coords[:, 1], s=10, c="red", marker="s",
                   linewidths=0, alpha=0.55, zorder=5)
    ax.scatter([x], [y], color="red", s=40, marker="x", zorder=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_axis_off()


_ZONE_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
    "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075",
]


def _overlay_zone_map(ax, zonal, legend=True):
    """Sharp, high-contrast per-zone colour overlay + black zone-boundary
    lines over the current axes.

    Raw zone ids are sparse (connected-component labels: 1, 5, 21, ...), so
    ``imshow``-ing them directly on a continuous colormap puts neighbouring
    zones on near-identical colours. This remaps them to a dense range and
    assigns each a distinct saturated hue.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    zi = np.asarray(zonal.zone_image)
    ids = np.array(sorted(int(v) for v in np.unique(zi) if v != 0))
    if ids.size == 0:
        return
    remap = np.zeros(int(zi.max()) + 1, dtype=int)
    for k, zid in enumerate(ids):
        remap[zid] = k
    dense = np.ma.array(remap[zi], mask=(zi == 0))

    colours = [_ZONE_COLOURS[k % len(_ZONE_COLOURS)] for k in range(ids.size)]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(-0.5, ids.size + 0.5), ids.size)
    ax.imshow(dense, cmap=cmap, norm=norm, alpha=0.5, interpolation="nearest")

    # Crisp black boundary lines wherever the label changes. Contour on a
    # coarsened copy -- marching-squares over a full 2000^2 grid is seconds
    # per call and the lines only need to read as sharp, not pixel-exact.
    if ids.size > 1:
        h, w = zi.shape
        step = max(1, int(round(max(h, w) / 700)))
        lab = remap[zi][::step, ::step].astype(float)
        ax.contour(np.arange(0, w, step), np.arange(0, h, step), lab,
                   levels=np.arange(0.5, ids.size - 0.5),
                   colors="k", linewidths=1.1)

    if legend and ids.size > 1:
        ax.legend(handles=[Patch(facecolor=c, edgecolor="k", label=f"zone {zid}")
                           for c, zid in zip(colours, ids)],
                  loc="upper right", fontsize=8, framealpha=0.9)


def inspect_mesh(mesh, subset_idx=None, show_areas=False, zones=False, ax=None,
                 show=True, block=True, save=False, **kwargs):
    """Render the mesh over its reference image.

    ``subset_idx`` : int, optional
        Highlight one node. When the mesh records its subset template
        (every solve from schema ``0x07`` on), and ``zones`` is not set,
        the plot becomes a crop of that node's subset template (like
        ``subset.inspect()``); for a zonally-solved mesh the zone-removed
        template pixels are shaded red and the pre/post pixel counts,
        ``|∇γ|`` and any minimum-pixel-guard fallback are captioned.
    ``zones`` : bool, default False
        Overlay the whole-image zone-label map from a
        ``solver_options={"masking": "zonal"}`` solve. Raises
        ``RuntimeError`` if the mesh was not solved that way (or its pass 2
        fell back). When combined with ``subset_idx`` the chosen subset's
        kept (green) / removed (red) footprint is drawn at its true
        location on the full map.
    """
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    f_img_path = getattr(mesh, 'f_img_path', None)
    zonal = getattr(mesh, 'zonal_masking', None)
    tmpl_shape = getattr(mesh, 'template_shape', None)
    tmpl_sizes = getattr(mesh, 'template_sizes', None)
    if tmpl_sizes is not None:
        tmpl_sizes = np.asarray(tmpl_sizes)

    if zones and zonal is None:
        raise RuntimeError(
            "mesh was not solved with solver_options={'masking': 'zonal'} "
            "(or its pass 2 fell back) -- no zone map to inspect"
        )

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    has_idx = subset_idx is not None and 0 <= subset_idx < len(nodes)

    # --- Per-subset template crop (only when not showing the zone map) -----
    if has_idx and not zones and tmpl_shape is not None and tmpl_sizes is not None:
        coord = nodes[subset_idx]
        size = int(tmpl_sizes[subset_idx])
        img_gs = _load_img_gs(f_img_path)
        cut = None
        caption = f"Subset {subset_idx}: {tmpl_shape} r={size}"
        if zonal is not None:
            fp = _core.zoned_subset_footprint(
                np.asarray(zonal.zone_image), [float(coord[0]), float(coord[1])],
                str(tmpl_shape), size,
            )
            cut = np.asarray(fp["coords_cut"])
            pre = int(np.asarray(zonal.node_pre_px)[subset_idx])
            post = int(np.asarray(zonal.node_post_px)[subset_idx])
            ggrad = float(np.asarray(zonal.node_gamma_max_grad)[subset_idx])
            zid = int(np.asarray(zonal.node_zone)[subset_idx])
            fell_back = bool(np.asarray(zonal.node_guard_fallback)[subset_idx])
            caption = (f"Subset {subset_idx}  zone {zid}: kept {post}/{pre} px, "
                       f"$|\\nabla\\gamma|$={ggrad:.3g}")
            if fell_back:
                caption += "  — GUARD FALLBACK (solved un-zoned)"
        _render_template_crop(ax, img_gs, coord, size, str(tmpl_shape), cut_coords=cut)
        ax.text(0.5, -0.06, caption, transform=ax.transAxes, ha="center")
        plt.tight_layout()
        _show_save_close(fig, show, block, save, owned=owned)
        return fig, ax

    # --- Full-mesh view --------------------------------------------------
    imshow_kwargs = {"cmap": "gist_gray"}
    imshow_kwargs.update(kwargs)
    _imshow_or_blank(ax, f_img_path, **imshow_kwargs)

    if zones:
        _overlay_zone_map(ax, zonal)
    else:
        # Element edges -- one LineCollection, not tens of thousands of
        # ax.plot() calls (an 8000-node mesh has ~16k elements). Omitted
        # under the zone overlay, where they only obscure the zones.
        tri = nodes[elements[:, :3]]                   # (M, 3, 2)
        segs = np.concatenate(
            [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
        ax.add_collection(LineCollection(
            segs, colors="b", linewidths=0.5 if len(elements) <= 2000 else 0.2))

    if not zones and len(elements) <= 400:
        # Element index annotations (skipped under the zone wash, and on a
        # dense mesh where 10^4 labels are unreadable and slow anyway).
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
        if zones and zonal is not None and tmpl_shape is not None and tmpl_sizes is not None:
            fp = _core.zoned_subset_footprint(
                np.asarray(zonal.zone_image), [float(node[0]), float(node[1])],
                str(tmpl_shape), int(tmpl_sizes[subset_idx]),
            )
            kept = np.asarray(fp["coords_kept"])
            cut = np.asarray(fp["coords_cut"])
            if len(kept):
                ax.scatter(kept[:, 0], kept[:, 1], s=6, c="lime", marker="s",
                           linewidths=0, alpha=0.5, zorder=8)
            if len(cut):
                ax.scatter(cut[:, 0], cut[:, 1], s=6, c="red", marker="s",
                           linewidths=0, alpha=0.5, zorder=8)

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


def contour_sequence(sequence, mesh_idx, quantity, **kwargs):
    _require_solved(sequence)
    if mesh_idx is None:
        raise ValueError("mesh_idx is required")
    mesh = sequence.mesh_solution_at(mesh_idx)
    return contour_mesh(mesh, quantity, **kwargs)


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


def contour_mesh(mesh, quantity, ax=None, show=True, block=True, save=False, mesh_overlay=False,
                  calibration=None, **kwargs):
    _require_solved(mesh)
    valid = {"C_ZNCC", "iterations", "norm", "u", "v", "R"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")
    if calibration is not None and quantity not in {"u", "v", "R"}:
        raise ValueError(
            f"calibration only applies to quantity in ['R', 'u', 'v'], not {quantity!r}"
        )

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    displacements = np.asarray(mesh.displacements)
    if calibration is not None:
        # Node positions stay in pixel space (the background image and mesh layout are
        # never re-projected — lens distortion means "calibrated positions" would be a
        # nonlinear warp, not just a scale, and would no longer align with the image).
        # Only the displacement values plotted for u/v/R are converted, per node.
        displacements = calibration.i2o(nodes + displacements) - calibration.i2o(nodes)

    unit = "calibrated" if calibration is not None else "px"
    labels = {
        "C_ZNCC": r"$C_{ZNCC}$ (-)",
        "iterations": "Iterations (-)",
        "norm": r"$\Delta$ Norm (-)",
        "u": f"u ({unit})",
        "v": f"v ({unit})",
        "R": f"R ({unit})",
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


def history_particle(particle, quantity="warps", components=None,
                     ax=None, show=True, block=True, save=None,
                     xlim=None, ylim=None, **kwargs):
    _require_solved(particle)
    valid = {"warps", "strains", "vol_strains"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if quantity == "warps":
        data = np.asarray(particle.warps)
        if components is None:
            components = range(data.shape[1])
        for c in components:
            label = _WARP_LABELS_HISTORY[c] if c < len(_WARP_LABELS_HISTORY) else str(c)
            ax.plot(range(data.shape[0]), data[:, c], label=label, **kwargs)
        ax.set_ylabel("Value")
        ax.legend()

    elif quantity == "strains":
        strains = np.asarray(particle.strains)
        vol_strains = np.asarray(particle.vol_strains)
        ax.plot(range(strains.shape[0]), strains[:, 0], label=_STRAIN_LABELS_HISTORY[0], **kwargs)
        ax.plot(range(strains.shape[0]), strains[:, 1], label=_STRAIN_LABELS_HISTORY[1], **kwargs)
        ax.plot(range(strains.shape[0]), strains[:, 5], label=_STRAIN_LABELS_HISTORY[5], **kwargs)
        ax.plot(range(len(vol_strains)), vol_strains, label=r"$\epsilon_{vol}$ ($-$)", **kwargs)
        ax.set_ylabel(r"Strain, $\epsilon$")
        ax.legend()

    elif quantity == "vol_strains":
        vol_strains = np.asarray(particle.vol_strains)
        ax.plot(range(len(vol_strains)), vol_strains, **kwargs)
        ax.set_ylabel(r"Volumetric strain, $\epsilon_{vol}$ ($-$)")

    ax.set_xlabel(r"Image Number, $i$ ($-$)")
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
    _require_solved(field)
    particle = field.particles[particle_index]
    return history_particle(particle, quantity, components=components,
                            ax=ax, show=show, block=block, save=save,
                            xlim=xlim, ylim=ylim, **kwargs)


def trace_particle(particle, quantity="warps", component=0,
                   imshow=True, ax=None, show=True, block=True, save=None,
                   xlim=None, ylim=None, **kwargs):
    _require_solved(particle)
    valid = {"warps", "strains", "vol_strains"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")

    coords = np.asarray(particle.coordinates)
    if quantity == "warps":
        values = np.diff(np.asarray(particle.warps)[:, component])
        label = _WARP_LABELS_HISTORY[component] if component < len(_WARP_LABELS_HISTORY) else str(component)
    elif quantity == "strains":
        values = np.diff(np.asarray(particle.strains)[:, component])
        label = _STRAIN_LABELS_HISTORY[component] if component < len(_STRAIN_LABELS_HISTORY) else str(component)
    else:
        values = np.diff(np.asarray(particle.vol_strains))
        label = _STRAIN_LABELS_HISTORY[3]

    points = coords.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(values.min(), values.max())
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


def trace_field(field, quantity="warps", component=0,
                imshow=True, ax=None, show=True, block=True, save=None,
                xlim=None, ylim=None, **kwargs):
    _require_solved(field)
    valid = {"warps", "strains", "vol_strains"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")

    all_segments = []
    all_values = []
    for p in field.particles:
        coords = np.asarray(p.coordinates)
        if quantity == "warps":
            v = np.diff(np.asarray(p.warps)[:, component])
        elif quantity == "strains":
            v = np.diff(np.asarray(p.strains)[:, component])
        else:
            v = np.diff(np.asarray(p.vol_strains))
        pts = coords.reshape(-1, 1, 2)
        all_segments.append(np.concatenate([pts[:-1], pts[1:]], axis=1))
        all_values.append(v)

    all_segments = np.concatenate(all_segments, axis=0)
    all_values = np.concatenate(all_values)

    if quantity == "warps":
        label = _WARP_LABELS_HISTORY[component] if component < len(_WARP_LABELS_HISTORY) else str(component)
    elif quantity == "strains":
        label = _STRAIN_LABELS_HISTORY[component] if component < len(_STRAIN_LABELS_HISTORY) else str(component)
    else:
        label = _STRAIN_LABELS_HISTORY[3]

    norm = plt.Normalize(all_values.min(), all_values.max())
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


def contour_field(field, quantity, window=None, dt=None, absolute=False, deformed=False,
                   ax=None, show=True, block=True, save=False, **kwargs):
    """Filled contour of one reduced value per particle.

    Values, positions and triangles all come from the core
    (``Field.contour_values`` / ``contour_coordinates`` /
    ``contour_triangles``), the same functions the GUI draws from. The
    triangulation covers only the field's region (exclusions and concave
    boundary sections left empty); a field saved before regions were stored
    falls back to the full convex hull, with a warning.

    ``deformed=True`` draws at the particle positions at the window's last
    increment instead of the initial ones (same triangles).

    gamma_max_grad requires the Field to have been solved via
    Field.solve() with strain_method left as meshless (the default) --
    the spatial gradient of gamma_max is computed once, at solve time,
    using the exact MeshlessParams the solve itself used.
    """
    _require_solved(field)
    valid = {"u", "v", "R", "ep_xx", "ep_yy", "ep_xy", "ep_vol",
             "ep1", "ep2", "gamma_max", "theta_p", "gamma_max_grad"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")

    w = _resolve_window(window, field.inc_no)
    values = np.asarray(field.contour_values(quantity, window=w, dt=dt, absolute=absolute))
    coords = np.asarray(field.contour_coordinates(deformed=deformed, window=w))
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

    if dt is None:
        label = f"{quantity}" + (" (abs)" if absolute else "")
    else:
        label = f"{quantity} /s"

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
