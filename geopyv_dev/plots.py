import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.spatial import Delaunay


def _imshow_or_blank(ax, img_path, **kwargs):
    """Helper: show image from path, or blank background if path is None."""
    kwargs.setdefault("cmap", "gist_gray")
    if img_path is not None:
        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        ax.imshow(img_gs, **kwargs)
    else:
        blank = np.zeros((50, 50), dtype=float)
        ax.imshow(blank, **kwargs)


def _show_save_close(fig, show, block, save):
    if save:
        plt.savefig(save, dpi=600)
    if show:
        plt.show(block=block)
    else:
        plt.close(fig)


def inspect_subset(subset, show=True, block=True, save=False, **kwargs):
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

    fig, ax = plt.subplots()
    ax.imshow(display, **imshow_kwargs)
    ax.text(0.5, -0.05,
            f"Size: {template_size} px; \u03c3_s = {sigma_intensity:.2f}; SSSIG = {sssig:.2E}",
            transform=ax.transAxes, ha="center")
    ax.set_axis_off()
    plt.tight_layout()
    _show_save_close(fig, show, block, save)
    return fig, ax


def inspect_mesh(mesh, subset_idx=None, show_areas=False, show=True, block=True, save=False, **kwargs):
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    f_img_path = getattr(mesh, 'f_img_path', None)

    fig, ax = plt.subplots()
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

    plt.tight_layout()
    _show_save_close(fig, show, block, save)
    return fig, ax


def inspect_sequence(sequence, mesh_idx=None, subset_idx=None, show=True, block=True, save=False, **kwargs):
    if mesh_idx is None:
        raise ValueError("mesh_idx is required")
    mesh = sequence.mesh_solutions[mesh_idx]
    return inspect_mesh(mesh, subset_idx=subset_idx, show=show, block=block, save=save, **kwargs)


def inspect_particle(particle, show=True, block=True, save=False, **kwargs):
    coords = np.asarray(particle.coordinates)
    initial = coords[0]
    img_path = getattr(particle, 'image_0_path', None)

    fig, ax = plt.subplots()
    _imshow_or_blank(ax, img_path)
    ax.scatter([initial[0]], [initial[1]], marker='x', color='red', s=100, zorder=10, **kwargs)
    plt.tight_layout()
    _show_save_close(fig, show, block, save)
    return fig, ax


def inspect_field(field, particle_idx=None, show=True, block=True, save=False, **kwargs):
    coords = np.asarray(field.coordinates)
    img_path = getattr(field, 'image_0_path', None)

    fig, ax = plt.subplots()
    _imshow_or_blank(ax, img_path)
    ax.scatter(coords[:, 0], coords[:, 1], **kwargs)
    if particle_idx is not None:
        ax.scatter([coords[particle_idx, 0]], [coords[particle_idx, 1]],
                   color='red', s=100, zorder=10)
    plt.tight_layout()
    _show_save_close(fig, show, block, save)
    return fig, ax


def convergence_subset(subset, show=True, block=True, save=False, **kwargs):
    solve_result = getattr(subset, 'solve_result', None)
    if solve_result is None:
        raise ValueError("Subset has not been solved.")
    history = solve_result["history"]
    max_norm = solve_result.get("max_norm", 1e-3)
    max_iterations = solve_result.get("max_iterations", 50)

    iters = [h[0] for h in history]
    norms = [h[1] for h in history]
    znccs = [h[2] for h in history]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].semilogy(iters, norms, marker="o", **kwargs)
    ax[0].semilogy([min(iters), max(iters)], [max_norm, max_norm], "--r")
    ax[0].set_ylabel(r"$\Delta$ Norm (-)")
    ax[1].plot(iters, znccs, marker="o", **kwargs)
    ax[1].plot([min(iters), max(iters)], [0.75, 0.75], "--r")
    ax[1].set_ylabel(r"$C_{ZNCC}$ (-)")
    ax[1].set_xlabel("Iteration (-)")
    plt.tight_layout()
    _show_save_close(fig, show, block, save)
    return fig, np.array(ax)


def convergence_mesh(mesh, quantity, show=True, block=True, save=False, **kwargs):
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

    fig, ax = plt.subplots()
    ax.hist(data, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count (-)")
    plt.tight_layout()
    _show_save_close(fig, show, block, save)
    return fig, ax


def convergence_sequence(sequence, mesh_idx=None, subset_idx=None, quantity="C_ZNCC",
                          show=True, block=True, save=False, **kwargs):
    if mesh_idx is not None:
        mesh = sequence.mesh_solutions[mesh_idx]
        return convergence_mesh(mesh, quantity, show=show, block=block, save=save, **kwargs)
    else:
        all_data = np.concatenate([np.asarray(m.c_zncc) for m in sequence.mesh_solutions])
        fig, ax = plt.subplots()
        ax.hist(all_data, **kwargs)
        ax.set_xlabel(r"$C_{ZNCC}$ (-)")
        ax.set_ylabel("Count (-)")
        plt.tight_layout()
        _show_save_close(fig, show, block, save)
        return fig, ax


def contour_mesh(mesh, quantity, show=True, block=True, save=False, **kwargs):
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

    fig, ax = plt.subplots()
    f_img_path = getattr(mesh, 'f_img_path', None)
    _imshow_or_blank(ax, f_img_path)

    tri_obj = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements[:, :3])
    cf = ax.tricontourf(tri_obj, values, **kwargs)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(labels[quantity])

    plt.tight_layout()
    _show_save_close(fig, show, block, save)
    return fig, ax


def contour_field(field, quantity, window=None, dt=None, absolute=False,
                   show=True, block=True, save=False, **kwargs):
    valid = {"u", "v", "R", "ep_xx", "ep_yy", "ep_xy", "ep_vol"}
    if quantity not in valid:
        raise ValueError(f"quantity must be one of {sorted(valid)!r}, got {quantity!r}")

    particles = field.particles
    n = len(particles)
    coords = np.array([[p.coordinates[0, 0], p.coordinates[0, 1]] for p in particles])

    strain_col = {"ep_xx": 0, "ep_yy": 1, "ep_xy": 2}

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

    fig, ax = plt.subplots()
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
    _show_save_close(fig, show, block, save)
    return fig, ax
