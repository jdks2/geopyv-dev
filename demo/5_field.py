"""Demo 5 - Field: strain histories over the whole area.

A Field is many particles (demo 4) spread over the measured area, one per
triangle of the first mesh, each standing for that triangle's share of the
area. Together they give maps of strain - where the soil is shearing,
compressing or dilating - and how those maps grow through the test.

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
# 1. Place the particles
# ---------------------------------------------------------------------------
print("=== 1. Placing particles ===")
m0 = sequence.mesh_solution_at(0)
coords, volumes = gp.field_distribute_particles(nodes=m0.nodes, elements=m0.elements, depth=1.0)
print(f"{len(volumes)} particles, one per triangle of the first mesh")
field = gp.Field(sequence, track=True, depth=1.0, coordinates=coords, volumes=volumes)

# ---------------------------------------------------------------------------
# 2. Solve
# ---------------------------------------------------------------------------
print("\n=== 2. Solving ===")
field.solve()
print(field)
p0 = field.particle_at(0)
print(f"  particle 0 ends with shear strain eps_xy = {p0.strains[-1, 5]:.5f}")

field.inspect()  # Where the particles are.
print("\nMaximum shear strain (gamma_max) over the whole test - uniform here,")
print("as the whole image was sheared evenly:")
field.contour("gamma_max")
print("The same over one step (image 2 to 3), as a rate in /s:")
field.contour("gamma_max", increment=2, mode="sequential", rate=True)
print("Volumetric strain (ep_vol): about zero, as shear does not change area:")
field.contour("ep_vol")
field.history(particle_index=0, quantity="strains")  # One particle's strains against time.
field.trace(quantity="warps", component=0)  # All paths, coloured by u.

# ---------------------------------------------------------------------------
# 3. Save and reload
# ---------------------------------------------------------------------------
path = OUTPUT / "field.pyv"
field.save(str(path))
field = gp.load(str(path))
print(f"\nLoaded from output/field.pyv: {field}")
print("\nNext: 6_validation.py checks these results against the true deformation.")
