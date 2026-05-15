import numpy as np
import geopyv_dev as gp

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
ref = gp.Image(filepath="images/comp/compression_0.jpg")
tar = gp.Image(filepath="images/comp/compression_1.jpg")
local_mask = gp.Mask(mask_type="local", shape="circle", size=50)

boundary_nodes = np.array(
    [[200.0, 200.0], [200.0, 800.0], [800.0, 800.0], [800.0, 200.0]]
)
boundary = gp.PathRegion(nodes=boundary_nodes, hard=False)
exclusion = gp.CircleRegion(
    centre=[700.0, 700.0], radius=50.0, size=20.0, option="F", hard=True,
)
seed_coord = [501.0, 501.0]
seed_warp = [0.0] * 6

# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------
# print("=== Subset ===")
# print(gp.Subset.__doc__)
# subset = gp.Subset(coord=[500.0, 500.0], local_mask=local_mask, f_img=ref, g_img=tar)
# print(subset)
# subset.inspect()
# 
# subset.solve(algorithm="icgn")
# print(subset)
# print(f"  p:      {subset.p:.4f}")
# print(f"  c_zncc: {subset.c_zncc:.4f}")
# subset.convergence()
# 
# gp.save("subset.pyv", subset)
# del subset
# subset = gp.load("subset.pyv")
# print(f"\nLoaded: {subset}")
# print(f"  p:      {subset.p}")
# print(f"  c_zncc: {subset.c_zncc:.4f}")
# subset.inspect()
# subset.convergence()

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
# print("\n=== Mesh ===")
# mesh = gp.Mesh(
#     boundary=boundary, target_nodes=100, f_img=ref, g_img=tar,
#     size=(20.0, 200.0), exclusions=[exclusion], mesh_order=1,
# )
# print(mesh)
# 
# mesh.solve(local_mask, seed_coord, seed_warp=seed_warp, tolerance=0.7)
# print(mesh)
# print(f"  mean C_ZNCC: {mesh.c_zncc.mean():.4f}")
# mesh.inspect()
# mesh.convergence()
# mesh.contour("u")
# 
# gp.save("mesh.pyv", mesh)
# del mesh
# mesh = gp.load("mesh.pyv")
# print(f"\nLoaded: {mesh}")
# print(f"  mean C_ZNCC: {mesh.c_zncc.mean():.4f}")
# mesh.inspect()
# mesh.convergence()
# mesh.contour("u")

# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------
print("\n=== Sequence ===")
sequence = gp.Sequence(
    image_dir="images/comp", # Where the images are stored.
    boundary=boundary_nodes, # A boundary object or a coordinate array.
    exclusions=[exclusion], # A list of exclusion objects or coordinate arrays. Defaults to None.
    size=(20.0, 200.0), # Nodal spacing tuple (lower, upper).
    target_nodes=100, # An integer number of target nodes. 
    mesh_order=1, # The mesh order, 1 or 2. 
)
print(f"{sequence}  ({sequence.n_pairs} pairs)")

sequence.solve(
    local_mask=local_mask,
    seed_coord=seed_coord,
    seed_warp=seed_warp,
    tolerance=0.75,
    options=gp.SequenceOptions(sync=True),
)
print(sequence)
print(f"  solved:          {sequence.solved}")
print(f"  n_pairs:         {sequence.n_pairs}")
print(f"  mesh[0] C_ZNCC:  {sequence.mesh_solutions[0].c_zncc.mean():.4f}")
sequence.inspect(mesh_idx=0)
sequence.convergence()
sequence.convergence(mesh_idx=0)

gp.save("sequence.pyv", sequence)
del sequence
sequence = gp.load("sequence.pyv")
print(f"\nLoaded: {sequence}")
print(f"  solved:          {sequence.solved}")
print(f"  n_pairs:         {sequence.n_pairs}")
print(f"  mesh[0] C_ZNCC:  {sequence.mesh_solutions[0].c_zncc.mean():.4f}")
sequence.inspect(mesh_idx=0)
sequence.convergence()
sequence.convergence(mesh_idx=0)

# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------
print("\n=== Particle ===")
particle = gp.Particle(source=sequence, coordinate=[500.0, 500.0])
print(particle)

particle.solve()
print(particle)
print(f"  coordinates shape: {particle.coordinates.shape}")
print(f"  warps shape:       {particle.warps.shape}")
print(f"  strains shape:     {particle.strains.shape}")
print(f"  volumes:           {particle.volumes}")
particle.inspect()

particle.save("particle.pyv")
del particle
particle = gp.load("particle.pyv")
print(f"\nLoaded: {particle}")
print(f"  coordinates shape: {particle.coordinates.shape}")
particle.inspect()

# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------
print("\n=== Field ===")
first_mesh = sequence.mesh_solutions[0]
coords, volumes = gp.field_distribute_particles(
    nodes=first_mesh.nodes, elements=first_mesh.elements,
)
print(f"{len(volumes)} particles distributed from mesh[0]")

field = gp.Field(sequence, coordinates=coords, volumes=volumes)
print(field)

field.solve()
print(field)
print(f"  n_particles:      {len(field.particles)}")
print(f"  vol_totals shape: {field.vol_totals.shape}")
field.inspect()
field.contour("ep_vol")

field.save("field.pyv")
del field
field = gp.load("field.pyv")
print(f"\nLoaded: {field}")
print(f"  n_particles:     {len(field.particles)}")
field.inspect()
field.contour("ep_vol")
