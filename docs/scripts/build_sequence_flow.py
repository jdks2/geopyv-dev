"""Build the Sequence::solve per-pair flow diagram for docs/tutorials/04_sequence.rst.

Static box/arrow diagram — reflects src/sequence.rs::Sequence::solve, including the
non-consecutive-failure reference catch-up (which the `sequential` flag does not
control) and the curtail-on-consecutive-failure termination path.
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
    fig, ax = plt.subplots(figsize=(9.5, 12.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 25)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cx = 5.2

    box(ax, (cx, 24), 7.2, 1.1, "Initialise: f_index = 0, g_index = 1\nsync_sol = None")
    b_mesh = box(ax, (cx, 21.7), 7.6, 2.3,
                 "Mesh::solve(f_img=images[f_index], g_img=images[g_index])\n"
                 "(embeds the full reliability-guided propagation\nin 03_mesh.rst)\n"
                 "guide=True → seed_warp preconditioned from previous\n"
                 "  pair's projected particle displacement\n"
                 "sync=True → reuse cached geometry if available\n"
                 "  (skip CDT), else regenerate",
                 color=LOOP_COLOR)
    d_ok = diamond(ax, (cx, 18.6), 5.4, 1.7, "Pair solved\nsuccessfully?")

    d_seq = diamond(ax, (2.6, 15.9), 4.6, 1.7, "sequential\n== True?")
    b_adv = box(ax, (0.9, 13.3), 4.2, 1.5,
                "Advance reference\nf_index = g_index − 1\nreference_updates[next]=True\nsync_sol reset")
    b_hold = box(ax, (5.4, 13.3), 3.4, 1.1, "Hold reference\nf_index unchanged")
    b_next = box(ax, (2.6, 11.1), 3.6, 1.0, "g_index += 1")

    d_more = diamond(ax, (2.6, 8.9), 4.6, 1.7, "g_index <\nn_images?")

    b_done = box(ax, (2.6, 6.3), 6.0, 1.6,
                 "SequenceSolution assembled\nper-pair nodes / displacements / c_zncc / p,\n"
                 "reference_updates, override_log, mesh_solutions",
                 color="#e7f3ff")

    d_consec = diamond(ax, (9.8, 15.9), 5.2, 2.1,
                        "f_index + 1 ≥ g_index?\n(reference already the\nimmediately-preceding frame)")
    b_curtail = box(ax, (9.8, 12.7), 5.0, 1.7,
                     "Curtail sequence\nunsolvable = true\nreturn Ok(partial SequenceSolution)",
                     color=ERR_COLOR, edge=ERR_EDGE)
    b_fallback = box(ax, (9.8, 9.3), 5.4, 2.3,
                      "Fallback catch-up (independent of `sequential`)\n"
                      "f_index = g_index − 1\nreference_updates[this pair] = True\n"
                      "sync_sol reset\n"
                      "override_=True → next SolveConfig uses\n"
                      "  tolerance=0, override_active=true",
                      color=LOOP_COLOR)

    # main spine: init -> mesh -> ok?
    arrow(ax, (cx, 23.45), (cx, 22.85))
    arrow(ax, (cx, 20.55), (cx, 19.45))

    # yes -> sequential branch
    arrow(ax, (d_ok[0] - d_ok[2] / 2, 18.6), (d_seq[0], 16.75), text="yes", text_dx=0.15)
    arrow(ax, (d_seq[0] - d_seq[2] / 2, 15.9), (b_adv[0], 14.05), text="yes", text_dx=-1.1)
    arrow(ax, (d_seq[0] + d_seq[2] / 2, 15.9), (b_hold[0], 13.85), text="no", text_dx=0.2)
    arrow(ax, (b_adv[0], 12.55), (b_next[0] - 0.5, 11.6))
    arrow(ax, (b_hold[0], 12.75), (b_next[0] + 0.9, 11.6))
    arrow(ax, (2.6, 10.6), (2.6, 9.75))

    # loop back to Mesh::solve if more pairs, else done
    arrow(ax, (d_more[0] - d_more[2] / 2, 8.9), (0.4, 8.9), rad=0.0)
    ax.plot([0.4, 0.4], [8.9, 21.7], color="#333333", linewidth=1.1)
    arrow(ax, (0.4, 21.7), (b_mesh[0] - b_mesh[2] / 2, 21.7))
    ax.text(0.15, 15.0, "yes — next pair", ha="center", va="center",
            fontsize=TEXT_SIZE - 1, rotation=90, color="#333333")

    arrow(ax, (2.6, 8.05), (2.6, 7.1), text="no", text_dx=0.2)

    # no -> consecutive-failure check
    arrow(ax, (d_ok[0] + d_ok[2] / 2, 18.6), (d_consec[0] - 1.0, 16.95), text="no", text_dx=0.15)
    arrow(ax, (d_consec[0] - 1.2, 15.05), (b_curtail[0] - 1.2, 13.55), text="yes", text_dx=-0.35)
    arrow(ax, (d_consec[0] + 1.9, 15.05), (b_fallback[0] + 1.9, 10.45), rad=0.28)
    ax.text(11.9, 14.3, "no", ha="center", va="center", fontsize=TEXT_SIZE - 1, color="#333333")

    # fallback retries the same pair
    ax.plot([12.7, 12.7], [9.3, 21.7], color="#333333", linewidth=1.1)
    arrow(ax, (b_fallback[0] + b_fallback[2] / 2, 9.3), (12.7, 9.3), rad=0.0)
    arrow(ax, (12.7, 21.7), (b_mesh[0] + b_mesh[2] / 2, 21.7))
    ax.text(12.95, 15.0, "retry same g_index", ha="center", va="center",
            fontsize=TEXT_SIZE - 1, rotation=90, color="#333333")

    ax.set_title("Sequence::solve — per-pair loop", fontsize=11, pad=14)
    return fig


if __name__ == "__main__":
    fig = build()
    save_png(fig, os.path.join(IMG_DIR, "sequence_solve_flow.png"))
