"""Demo 5 - Field: many particles distributed over the region.

Run 3_sequence.py first. Saves output/field.pyv, which demo 6 loads.
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
# Build: one particle per element, placed from the first mesh
# ---------------------------------------------------------------------------
print("=== Field ===")
m0 = sequence.mesh_solution_at(0)
coords, volumes = gp.field_distribute_particles(nodes=m0.nodes, elements=m0.elements, depth=1.0)
print(f"{len(volumes)} particles distributed from mesh 0")

field = gp.Field(sequence, track=True, depth=1.0, coordinates=coords, volumes=volumes)
print(field)

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
field.solve()
print(field)
print(f"  n_particles: {field.n_particles}")
print(f"  increments:  {field.inc_no}")
p0 = field.particle_at(0)
print(f"  particle 0 final eps_xy: {p0.strains[-1, 5]:.5f}")

field.inspect()
field.contour("gamma_max")  # Total over the test.
field.contour("gamma_max", increment=2, mode="sequential", rate=True)  # One increment, /s.
field.contour("ep_vol")
field.history(particle_index=0, quantity="strains")
field.trace(quantity="warps", component=0)  # Paths coloured by u.

# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------
path = OUTPUT / "field.pyv"
field.save(str(path))
field = gp.load(str(path))
print(f"\nLoaded: {field}")
field.contour("gamma_max")
