"""Demo 4 - Particle: following one material point through a solved Sequence.

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
# Build and solve
# ---------------------------------------------------------------------------
print("=== Particle ===")
particle = gp.Particle(
    source=sequence,  # A solved Sequence or Mesh.
    coordinate=[400.0, 650.0],  # Start position [x, y]; need not be a node.
    track=True,  # True = follows the material (Lagrangian).
)
print(particle)

particle.solve()
print(particle)
print(f"  coordinates: {particle.coordinates.shape}")
print(f"  strains:     {particle.strains.shape}  [xx, yy, zz, yz, xz, xy]")
print(f"  final position:    {particle.coordinates[-1]}")
print(f"  final eps_xy:      {particle.strains[-1, 5]:.5f}")
print(f"  final vol. strain: {particle.vol_strains[-1]:.2e}")

particle.inspect()
particle.history(quantity="warps", components=[0, 1])  # u, v against time.
particle.history(quantity="strains", rate=True, mode="sequential")  # Strain rates, /s.
particle.trace(quantity="strains", component=5)  # Path coloured by eps_xy.

# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------
path = OUTPUT / "particle.pyv"
particle.save(str(path))
particle = gp.load(str(path))
print(f"\nLoaded: {particle}")
particle.history(quantity="warps", components=[0, 1])
