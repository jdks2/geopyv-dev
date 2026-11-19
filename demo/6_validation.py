"""Demo 6 - Validation: measured warps against the known applied deformation.

The demo images are synthetic (gp.Speckle), so their true deformation is
stored in images/shear/speckle_shear.pyv. Run 3_sequence.py and 5_field.py first.
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
# Solve: compare applied and observed warps particle by particle
# ---------------------------------------------------------------------------
print("=== Validation ===")
val = gp.Validation(speckle=speckle, fields=[field], labels=["demo"])
val.solve(cumulative=True)

fd = val.solution.fields[0]
print(f"  applied:  {fd.applied.shape}  (frames, particles, warp components)")
u_error = fd.applied[-1, :, 0] - fd.observed[-1, :, 0]
print(f"  mean |u error|, last frame: {abs(u_error).mean():.4f} px")

val.standard_error(component=0)  # Spread of u error against frame.
val.mean_error(component=0)  # Bias of u error against frame.
val.spatial_error(field_index=0, time_index=fd.applied.shape[0] - 1, quantity="R")
