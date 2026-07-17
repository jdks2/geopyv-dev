"""Build the Mesh::solve reliability-guided propagation flow diagram for
docs/tutorials/03_mesh.rst.

Static box/arrow diagram (no gif) — reflects the actual control flow in
src/mesh.rs::Mesh::solve, not the simplified textbook description.
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    IMG_DIR, save_png, flow_box as box, flow_diamond as diamond, flow_arrow as arrow,
    FLOW_LOOP_COLOR as LOOP_COLOR, FLOW_ERR_COLOR as ERR_COLOR, FLOW_ERR_EDGE as ERR_EDGE,
    FLOW_TEXT_SIZE as TEXT_SIZE,
)


def build():
    fig, ax = plt.subplots(figsize=(7.5, 11.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cx = 5.0

    b_seed = box(ax, (cx, 23), 6.6, 1.3,
                 "Seed node solve\nnearest node to seed_coord, forced_p0=seed_warp,\nstrict seed_tolerance")
    b_seednb = box(ax, (cx, 21), 6.6, 1.3,
                    "Solve seed's neighbours\neach preconditioned from its own trusted\nneighbourhood, pushed to heap")
    b_pop = box(ax, (cx, 18.7), 6.6, 1.4,
                "Pop best-C_ZNCC unpropagated\nnode from max-heap\n(stale re-pushed entries are skipped)",
                color=LOOP_COLOR)
    b_solve_nb = box(ax, (cx, 16.2), 6.6, 2.3,
                      "Solve its unsolved neighbours\n≥ 3 trusted neighbours → local least-squares\naffine/quadratic field fit\n1–2 trusted → Taylor projection from best\nfailed attempt → retry once with zero warp,\nkeep whichever c_zncc is higher",
                      color=LOOP_COLOR)
    b_push = box(ax, (cx, 13.9), 4.4, 0.9, "Push newly solved\nneighbours onto heap", color=LOOP_COLOR)

    b_corr = box(ax, (cx, 11.9), 6.6, 1.9,
                 "Corrections pass (after heap empties)\nLane 1: re-solve IQR-low-C_ZNCC outliers,\n  keep only if strictly improved\nLane 2: re-solve flow/R IQR outliers,\n  always overwrite")

    d_quality = diamond(ax, (cx, 9.3), 5.6, 1.9,
                         "All nodes\nquality_ok?\n(or override_active)")
    b_err1 = box(ax, (8.6, 9.3), 2.6, 1.1, "Err:\nmesh unsolvable", color=ERR_COLOR, edge=ERR_EDGE)

    d_compat = diamond(ax, (cx, 6.7), 5.6, 1.9, "Every element's\nJacobian\ndeterminant > 0?")
    b_err2 = box(ax, (8.6, 6.7), 2.6, 1.1, "Err:\nelement fold-over", color=ERR_COLOR, edge=ERR_EDGE)

    b_done = box(ax, (cx, 4.3), 6.6, 1.3,
                 "MeshSolution assembled\nnodes, displacements, c_zncc, p, elements,\ncentroids, areas, strains", color="#e7f3ff")

    # main spine
    arrow(ax, (cx, 22.35), (cx, 21.65))
    arrow(ax, (cx, 20.35), (cx, 19.4))
    arrow(ax, (cx, 18.0), (cx, 17.35))
    arrow(ax, (cx, 15.25), (cx, 14.35))
    # loop back from push -> pop, routed clear of every box's left edge
    # (the narrowest box here is b_push, at cx-2.2; give the spine another
    # clean gap past the widest box's edge at cx-3.3 so it can't graze either)
    x_loop = cx - 4.3
    pop_left = b_pop[0] - b_pop[2] / 2
    push_left = b_push[0] - b_push[2] / 2
    arrow(ax, (x_loop, 13.9), (x_loop, 18.7), rad=0.0, color="#4a7a4a")
    ax.plot([x_loop, push_left], [13.9, 13.9], color="#4a7a4a", linewidth=1.1)
    ax.plot([x_loop, pop_left], [18.7, 18.7], color="#4a7a4a", linewidth=1.1)
    ax.text(x_loop - 0.3, 16.3, "heap not empty",
            ha="center", va="center", fontsize=TEXT_SIZE - 1, color="#4a7a4a", rotation=90)

    arrow(ax, (cx, 13.45), (cx, 12.85))
    arrow(ax, (cx, 10.95), (cx, 10.25))
    arrow(ax, (7.8, 9.3), (b_err1[0] - b_err1[2] / 2, 9.3), text="no", text_dx=-0.9)
    arrow(ax, (cx, 8.35), (cx, 7.65), text="yes", text_dx=0.3)
    arrow(ax, (7.8, 6.7), (b_err2[0] - b_err2[2] / 2, 6.7), text="no", text_dx=-0.9)
    arrow(ax, (cx, 5.75), (cx, 4.95), text="yes", text_dx=0.3)

    ax.set_title("Mesh::solve — reliability-guided propagation", fontsize=11, pad=14)
    return fig


if __name__ == "__main__":
    fig = build()
    save_png(fig, os.path.join(IMG_DIR, "mesh_solve_flow.png"))
