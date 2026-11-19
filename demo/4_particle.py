"""Demo 4 - Particle: the history of one point of material.

A mesh gives displacements at fixed places in the image. Soil mechanics
usually wants the history of a piece of soil: where it went and how much it
strained. A Particle starts at a point, is carried along by the measured
displacements from image to image, and adds up its strain on the way.

STRAIN is the change of shape: eps_xx, eps_yy stretching along x and y
(negative = shortening), eps_xy shear. Volumetric strain is the change of
area (in 2-D, with a fixed depth).

Run 3_sequence.py first.
"""

import sys
from pathlib import Path

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
if not (OUTPUT / "sequence.pyv").exists():
    sys.exit("output/sequence.pyv not found - run 3_sequence.py first.")

sequence = gp.load(str(OUTPUT / "sequence.pyv"))

# ---------------------------------------------------------------------------
# 1. Build and solve
# ---------------------------------------------------------------------------
print("=== 1. Following one point ===")
particle = gp.Particle(
    source=sequence,  # A solved Sequence (or a single Mesh).
    coordinate=[400.0, 650.0],  # Where it starts, px; it need not be a mesh node.
    track=True,  # True: follow the material (Lagrangian); False: stay at this spot (Eulerian).
)
particle.solve()
print(particle)
print("Row 0 is the start; one row per image after it.")
for i, (xy, eps) in enumerate(zip(particle.coordinates, particle.strains)):
    print(f"  image {i}: at ({xy[0]:7.2f}, {xy[1]:7.2f}) px, shear strain eps_xy = {eps[5]: .5f}")
print(f"Final volumetric strain: {particle.vol_strains[-1]:.1e} (shear alone hardly changes the area)")
print("The images were sheared so that eps_xy reaches 0.01 in size; see the last row.")

particle.inspect()  # Where it starts, on the reference image.
particle.history(quantity="warps", components=[0, 1])  # u and v against time.
particle.history(quantity="strains", rate=True, mode="sequential")  # Strain rates, /s, per step.
particle.trace(quantity="strains", component=5)  # Its path, coloured by eps_xy.

# ---------------------------------------------------------------------------
# 2. Save and reload
# ---------------------------------------------------------------------------
path = OUTPUT / "particle.pyv"
particle.save(str(path))
particle = gp.load(str(path))
print(f"\nLoaded from output/particle.pyv: {particle}")
print("\nNext: 5_field.py does the same for points spread over the whole area.")
