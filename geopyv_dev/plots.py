import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection

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


def inspect_subset(subset, ax=None, show=True, block=True, save=False, **kwargs):
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


def inspect_mesh(mesh, subset_idx=None, show_areas=False, ax=None, show=True, block=True, save=False, **kwargs):
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    f_img_path = getattr(mesh, 'f_img_path', None)

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    imshow_kwargs = {"cmap": "gist_gray"}
    imshow_kwargs.update(kwargs)
    _imshow_or_blank(ax, f_img_path, **imshow_kwargs)

    # Element edges
    n_corner = 3
    for elem in elements:
        corners = nodes[elem[:n_corner]]
        for i in range(n_corner):
            j = (i + 1) % n_corner
            ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]],
                    'b-', linewidth=0.5)

    # Element index annotations
    for i, elem in enumerate(elements):
        cx = nodes[elem[:3], 0].mean()
        cy = nodes[elem[:3], 1].mean()
        ax.text(cx, cy, str(i), ha='center', va='center', color='red', fontsize=8)

    if show_areas:
        from matplotlib.patches import Circle as MplCircle
        radius = 10
        for nd in nodes:
            ax.add_patch(MplCircle((nd[0], nd[1]), radius, alpha=0.2, color='blue'))

    if subset_idx is not None and subset_idx < len(nodes):
        node = nodes[subset_idx]
        ax.scatter([node[0]], [node[1]], color='red', s=60, zorder=10)

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
    history = getattr(subset, 'history', None)
    if not getattr(subset, 'solved', False) or history is None:
        raise ValueError("Subset has not been solved.")
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
    if mesh_idx is None:
        raise ValueError("mesh_idx is required")
    mesh = sequence.mesh_solution_at(mesh_idx)
    return contour_mesh(mesh, quantity, **kwargs)


def convergence_sequence(sequence, mesh_idx=None, subset_idx=None, quantity="C_ZNCC",
                          ax=None, show=True, block=True, save=False, **kwargs):
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


def contour_mesh(mesh, quantity, ax=None, show=True, block=True, save=False, **kwargs):
    valid = {"C_ZNCC", "iterations", "norm", "u", "v", "R"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    displacements = np.asarray(mesh.displacements)

    labels = {
        "C_ZNCC": r"$C_{ZNCC}$ (-)",
        "iterations": "Iterations (-)",
        "norm": r"$\Delta$ Norm (-)",
        "u": "u (px)",
        "v": "v (px)",
        "R": "R (px)",
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
    particle = field.particles[particle_index]
    return history_particle(particle, quantity, components=components,
                            ax=ax, show=show, block=block, save=save,
                            xlim=xlim, ylim=ylim, **kwargs)


def trace_particle(particle, quantity="warps", component=0,
                   imshow=True, ax=None, show=True, block=True, save=None,
                   xlim=None, ylim=None, **kwargs):
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


def contour_field(field, quantity, window=None, dt=None, absolute=False,
                   ax=None, show=True, block=True, save=False, **kwargs):
    valid = {"u", "v", "R", "ep_xx", "ep_yy", "ep_xy", "ep_vol"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")

    particles = field.particles
    n = len(particles)
    coords = np.array([[p.coordinates[0, 0], p.coordinates[0, 1]] for p in particles])

    strain_col = {"ep_xx": 0, "ep_yy": 1, "ep_xy": 5}

    # Build window slice
    if window is not None:
        if isinstance(window, (list, tuple)) and len(window) == 2:
            w = slice(window[0], window[1])
        else:
            w = window
    else:
        w = slice(None)

    values = np.zeros(n)
    for i, p in enumerate(particles):
        warps = np.asarray(p.warps)  # (inc_no, warp_len)
        vol_strains = np.asarray(p.vol_strains)
        strains = np.asarray(p.strains)  # (inc_no, 6)

        if quantity == "u":
            v = warps[w, 0]
        elif quantity == "v":
            v = warps[w, 1]
        elif quantity == "R":
            v = np.sqrt(warps[w, 0]**2 + warps[w, 1]**2)
        elif quantity in strain_col:
            v = strains[w, strain_col[quantity]]
        elif quantity == "ep_vol":
            v = vol_strains[w]

        v = np.atleast_1d(v)
        if dt is None:
            if absolute:
                values[i] = float(np.sum(np.abs(np.diff(v)))) if len(v) > 1 else 0.0
            else:
                values[i] = float(v[-1] - v[0]) if len(v) > 1 else float(v[-1])
        else:
            if len(v) > 1:
                values[i] = float((v[-1] - v[0]) / (len(v) * dt))
            else:
                values[i] = 0.0

    # Delaunay triangulation
    if n >= 3:
        delaunay = Delaunay(coords)
        triangles = delaunay.simplices
    else:
        triangles = None

    owned = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    img_path = getattr(field, 'image_0_path', None)
    _imshow_or_blank(ax, img_path)

    if triangles is not None and len(triangles) > 0:
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
