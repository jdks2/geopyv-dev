"""Demo 3 - Sequence: correlating a whole image series.

Saves output/sequence.pyv, which demos 4-6 load.
"""

from pathlib import Path

import numpy as np

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images" / "shear"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

local_mask = gp.Mask(mask_type="local", shape="circle", size=25)
boundary = gp.PathRegion(
    nodes=np.array([[300.0, 300.0], [300.0, 700.0], [700.0, 700.0], [700.0, 300.0]]),
    hard=False,
)

# ---------------------------------------------------------------------------
# Build: one mesh per image pair, images discovered from the folder
# ---------------------------------------------------------------------------
print("=== Sequence ===")
sequence = gp.Sequence(
    image_dir=str(IMAGES),  # Finds shear_0.jpg ... shear_4.jpg.
    boundary=boundary,
    target_nodes=100,
    size=(15.0, 70.0),
    mesh_order=2,
    time=0.1,  # Images taken at 10 Hz.
)
print(f"{sequence}  ({sequence.n_pairs} pairs)")
print(f"  time: {sequence.time} s")

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
sequence.solve(
    local_mask=local_mask,
    seed_coord=[500.0, 500.0],
    seed_warp=[0.0] * 12,
    subset_order=2,
    tolerance=0.75,
    seed_tolerance=0.9,
    options=gp.SequenceOptions(guide=True, sync=True, sequential=False),
)
print(sequence)
print(f"  solved:            {sequence.solved}")
print(f"  pair 0 C_ZNCC:     {sequence.c_zncc(0).mean():.4f}")
print(f"  reference updates: {sequence.reference_updates}")

last = sequence.n_pairs - 1
sequence.inspect(mesh_idx=0)
sequence.convergence(quantity="C_ZNCC")  # All pairs pooled.
sequence.contour(quantity="R", mesh_idx=last)  # Total since frame 0.
sequence.contour(quantity="u", mesh_idx=last, mode="sequential", rate=True)  # Last increment, px/s.

# One pair as a full Mesh object (same API as demo 2).
m0 = sequence.mesh_solution_at(0)
print(f"  pair 0 mesh: {m0}")

# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------
path = OUTPUT / "sequence.pyv"
sequence.save(str(path))
sequence = gp.load(str(path))
print(f"\nLoaded: {sequence}")
print(f"  pair 0 C_ZNCC: {sequence.c_zncc(0).mean():.4f}")
sequence.contour(quantity="R", mesh_idx=last)
