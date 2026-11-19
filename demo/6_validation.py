"""Demo 6 - Validation: how accurate is it?

The demo images are synthetic: gp.Speckle moved the sand by a known amount,
stored in images/shear/speckle_shear.pyv. Validation compares, particle by
particle and image by image, the APPLIED (true) warp with the OBSERVED
(measured) one. Doing this on synthetic images like your own - same grain
size, subset size and mesh - tells you what accuracy to expect in a test.

Run 3_sequence.py and 5_field.py first.
"""

import sys
from pathlib import Path

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images" / "shear"
OUTPUT = HERE / "output"
if not (OUTPUT / "field.pyv").exists():
    sys.exit("output/field.pyv not found - run 3_sequence.py then 5_field.py first.")

speckle = gp.load(str(IMAGES / "speckle_shear.pyv"))
field = gp.load(str(OUTPUT / "field.pyv"))

# ---------------------------------------------------------------------------
# Compare applied and observed
# ---------------------------------------------------------------------------
print("=== Validation ===")
val = gp.Validation(speckle=speckle, fields=[field], labels=["demo"])
val.solve(cumulative=True)  # True: totals since image 0; False: each step on its own.

fd = val.solution.fields[0]
n_frames, n_particles, _ = fd.applied.shape
print(f"{n_particles} particles compared over {n_frames} images")
for k in range(n_frames):
    du = abs(fd.applied[k, :, 0] - fd.observed[k, :, 0])
    dv = abs(fd.applied[k, :, 1] - fd.observed[k, :, 1])
    print(f"  image {k + 1}: mean |u error| {du.mean():.4f} px, mean |v error| {dv.mean():.4f} px")
print("Errors of thousandths of a pixel, as here, are typical for clean synthetic")
print("images; camera noise, blur and larger strains make them bigger.")

val.standard_error(component=0)  # Spread of the u error, image by image.
val.mean_error(component=0)  # Average (bias) of the u error, image by image.
val.spatial_error(field_index=0, time_index=n_frames - 1, quantity="R")  # Where the errors are.
print("\nNext: 7_calibration.py converts pixels into millimetres.")
