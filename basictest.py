import glob
import numpy as np
import geopyv_dev as gp

# ---------------------------------------------------------------------------
# Subset test
# ---------------------------------------------------------------------------
# Setup.
ref = gp.Image(filepath="images/comp/compression_0.jpg")
tar = gp.Image(filepath="images/comp/compression_1.jpg")
template = gp.Circle(radius=50)

# Subset instantiation.
subset = gp.Subset(
    coord=[500.0, 500.0],
    template=template,
    f_img=ref,
    template_size=template.size,
    template_shape=template.shape,
    f_img_path=ref.filepath,
)

# Subset inspection.
print(subset)
print(f"  n_px={subset.n_px}, sssig={subset.sssig:.4f}")
subset.inspect()

# Subset solving (ICGN).
result = subset.solve_icgn(g_img=tar, p_0=[0.0] * 6)
print(f"  C_ZNCC={result['c_zncc']:.4f}, converged={result['converged']}, "
      f"iterations={result['iterations']}")
subset.convergence()
print(f"p={result['p']:.4f}")
# ---------------------------------------------------------------------------
# Mesh test
# ---------------------------------------------------------------------------
# ROI definition (boundary polygon → borders/segments/curves arrays).
boundary_nodes = np.array(
    [[200.0, 200.0], [200.0, 800.0], [800.0, 800.0], [800.0, 200.0]]
)
exclusion_circle = gp.CircleRegion(
    centre=[700.0, 700.0],
    radius=50.0,
    size=20.0,
    option="F",
    hard=True,
)
borders, segments, curves = gp.define_roi(
    boundary_nodes=boundary_nodes,
    boundary_hard=True,
    exclusion_nodes=[exclusion_circle.current_nodes],
)

# Mesh instantiation.
mesh = gp.Mesh(
    borders=borders,
    segments=segments,
    curves=curves,
    size_lower=20.0,
    size_upper=200.0,
    target_nodes=100,
    mesh_order=1,
)
print(mesh)

seed_coord = [501.0, 501.0]
seed_warp = [0.0] * 6

# Mesh solving.
mesh_sol = mesh.solve(
    f_img=ref,
    g_img=tar,
    template=template,
    seed_coord=seed_coord,
    seed_warp=seed_warp,
    tolerance=0.7,
)
print(mesh_sol)
print(f"  mean C_ZNCC={mesh_sol.c_zncc.mean():.4f}")

# Mesh saving / loading.
gp.save("mesh.pyv", mesh_sol)
del mesh_sol
mesh_sol = gp.load("mesh.pyv")
print(f"Loaded: {mesh_sol}")

# Mesh plots.
mesh_sol.inspect()
mesh_sol.inspect(show_areas=True)
mesh_sol.convergence()
mesh_sol.convergence(quantity="iterations")
mesh_sol.convergence(quantity="norm")
mesh_sol.contour("u")
mesh_sol.contour("v")
mesh_sol.contour("R")
mesh_sol.contour("C_ZNCC")

# ---------------------------------------------------------------------------
# Sequence test
# ---------------------------------------------------------------------------
# Setup: all compression images, sorted by frame number.
image_paths = sorted(
    glob.glob("images/comp/compression_*.jpg"),
    key=lambda p: int(p.split("_")[-1].split(".")[0]),
)
print(f"\nSequence: {len(image_paths)} images, {len(image_paths) - 1} pairs")

# Sequence instantiation.
sequence = gp.Sequence(
    image_paths=image_paths,
    borders=borders,
    segments=segments,
    curves=curves,
    size_lower=20.0,
    size_upper=200.0,
    target_nodes=100,
    mesh_order=1,
)
print(sequence)

# Sequence solving.
seq_sol = sequence.solve(
    template=template,
    seed_coord=seed_coord,
    seed_warp=seed_warp,
    adaptive_iterations=0,
    method="icgn",
    alpha=0.2,
    tolerance=0.75,
    sync=True,
)
print(seq_sol)

# Sequence saving / loading.
gp.save("sequence.pyv", seq_sol)
del seq_sol
seq_sol = gp.load("sequence.pyv")
print(f"Loaded: {seq_sol}")

# Sequence plots.
seq_sol.inspect(mesh_idx=0)
seq_sol.inspect(mesh_idx=5)
seq_sol.convergence()
seq_sol.convergence(mesh_idx=0)
seq_sol.convergence(mesh_idx=0, quantity="iterations")

# ---------------------------------------------------------------------------
# Particle test
# ---------------------------------------------------------------------------
# Extract mesh arrays from the sequence solution.
mesh_solutions = seq_sol.mesh_solutions
nodes_list        = [m.nodes        for m in mesh_solutions]
elements_list     = [m.elements     for m in mesh_solutions]
displacements_list= [m.displacements for m in mesh_solutions]
mesh_order_list   = [int(m.mesh_order) for m in mesh_solutions]

n_pairs = len(mesh_solutions)
inc_no  = n_pairs + 1  # initial state + one per pair

# Particle instantiation.
particle = gp.Particle(
    coordinate=[500.0, 500.0],
    initial_warp=[0.0] * 6,
    initial_volume=1.0,
    inc_no=inc_no,
    mesh_order=1,
    track=True,
)
print(f"\n{particle}")

# Particle solving.
particle_sol = particle.solve(
    nodes_list=nodes_list,
    elements_list=elements_list,
    displacements_list=displacements_list,
    mesh_order_list=mesh_order_list,
)
print(particle_sol)
print(f"  coordinates shape: {particle_sol.coordinates.shape}")
print(f"  warps shape: {particle_sol.warps.shape}")
print(f"  strains shape: {particle_sol.strains.shape}")
print(f"  volumes: {particle_sol.volumes}")

# Particle plots (image_0_path not propagated from sequence — blank background).
particle_sol.inspect()

# ---------------------------------------------------------------------------
# Field test
# ---------------------------------------------------------------------------
# Distribute particles at element centroids of the first mesh.
first_mesh = mesh_solutions[0]
coords, volumes = gp.field_distribute_particles(
    nodes=first_mesh.nodes,
    elements=first_mesh.elements,
)
print(f"\nField: {len(volumes)} particles distributed")

# Field instantiation.
field = gp.Field(
    coordinates=coords,
    volumes=volumes,
    inc_no=inc_no,
    track=True,
)
print(field)

# Field solving.
field_sol = field.solve(
    nodes_list=nodes_list,
    elements_list=elements_list,
    displacements_list=displacements_list,
    mesh_order_list=mesh_order_list,
)
print(field_sol)

# Field saving / loading.
gp.save("field.pyv", field_sol)
del field_sol
field_sol = gp.load("field.pyv")
print(f"Loaded: {field_sol}")

# Inspect data from a specific particle.
particle_idx = 4
psol = field_sol.particles[particle_idx]
print(f"\nParticle {particle_idx}:")
print(f"  coordinates:\n{psol.coordinates}")
print(f"  warps (u, v, ...):\n{psol.warps[:, :2]}")
print(f"  strains:\n{psol.strains}")
print(f"  volumes: {psol.volumes}")

# Field plots (image_0_path not propagated — blank background for inspect).
field_sol.inspect()
field_sol.inspect(particle_idx=4)
field_sol.contour("u")
field_sol.contour("v")
field_sol.contour("ep_xx")
field_sol.contour("ep_vol")
field_sol.contour("u", window=[0, 5])
field_sol.contour("ep_xx", absolute=True)
