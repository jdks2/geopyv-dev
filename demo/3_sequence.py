"""Demo 3 - Sequence: following the material through a series of images.

A test is photographed many times. A Sequence makes one mesh per image PAIR:
image 0 with 1, 0 with 2, 0 with 3 ... - every image measured from the same
reference ("leapfrog"). If the material changes too much for a pair to match
(C_ZNCC falls too low), the reference moves on to a later image and the
mesh is carried with the material ("reference update").

The images are found in a folder and ordered by the number in their names
(or by a timestamp, if the names have one).

Saves output/sequence.pyv, which demos 4-6 load.
"""

from pathlib import Path

import numpy as np

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images" / "shear"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

local_mask = gp.Mask(mask_type="local", shape="circle", size=30)
boundary = gp.PathRegion(
    nodes=np.array([[300.0, 300.0], [300.0, 700.0], [700.0, 700.0], [700.0, 300.0]]),
    hard=False,
)

# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------
print("=== 1. Building a sequence ===")
sequence = gp.Sequence(
    image_dir=str(IMAGES),  # Finds shear_0.png ... shear_4.png.
    boundary=boundary,
    target_nodes=150,
    size=(15.0, 200.0),
    mesh_order=1,
    time=0.1,  # Seconds between images (taken at 10 Hz); or a list of times.
)
print(f"{sequence}: {sequence.n_pairs} image pairs")
print(f"  image times: {sequence.time} s")

# ---------------------------------------------------------------------------
# 2. Solve
# ---------------------------------------------------------------------------
print("\n=== 2. Solving ===")
sequence.solve(
    local_mask=local_mask,
    seed_coord=[500.0, 500.0],
    seed_warp=[0.0] * 6,
    subset_order=1,
    tolerance=0.75,
    seed_tolerance=0.9,
    # guide: use each pair's result as the next pair's first guess (faster);
    # sync: keep the same nodes in every pair while the reference is unchanged.
    options=gp.SequenceOptions(guide=True, sync=True, sequential=False),
)
print(sequence)
for m in range(sequence.n_pairs):
    print(f"  pair {m}: C_ZNCC mean {sequence.c_zncc(m).mean():.4f}")
print(f"  reference updated: {sequence.reference_updates} (no: every pair matched image 0)")

last = sequence.n_pairs - 1
sequence.inspect(mesh_idx=0)
sequence.convergence(quantity="C_ZNCC")  # Match quality of all pairs together.
print("\nTotal displacement since image 0, at the last image:")
sequence.contour(quantity="R", mesh_idx=last)
print("Just the last step (image 3 to 4), as a velocity in px/s:")
sequence.contour(quantity="u", mesh_idx=last, mode="sequential", rate=True)

# One pair as a full Mesh object, with everything from demo 2.
m0 = sequence.mesh_solution_at(0)
print(f"\nPair 0 as a mesh: {m0}")

# ---------------------------------------------------------------------------
# 3. Save and reload
# ---------------------------------------------------------------------------
path = OUTPUT / "sequence.pyv"
sequence.save(str(path))
sequence = gp.load(str(path))
print(f"Loaded from output/sequence.pyv: {sequence}")
print("\nNext: 4_particle.py follows one point of sand; 5_field.py many.")
