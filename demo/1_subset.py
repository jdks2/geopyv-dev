"""Demo 1 - Subset: correlating a single patch between two images."""

from pathlib import Path

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images" / "shear"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

f_img = gp.Image(filepath=str(IMAGES / "shear_0.jpg"))
g_img = gp.Image(filepath=str(IMAGES / "shear_4.jpg"))
local_mask = gp.Mask(mask_type="local", shape="circle", size=25)

# ---------------------------------------------------------------------------
# Build: the template is extracted from the reference image
# ---------------------------------------------------------------------------
print("=== Subset ===")
subset = gp.Subset(
    coord=[400.0, 650.0],  # Centre [x, y] in the reference image.
    local_mask=local_mask,  # Subset shape and size.
    f_img=f_img,  # Reference image.
    g_img=g_img,  # Target image.
    subset_order=1,  # 1 (affine) or 2 (quadratic) warp.
)
print(subset)
print(f"  sigma_intensity: {subset.sigma_intensity:.2f}")
print(f"  SSSIG:           {subset.sssig:.2e}")
subset.inspect()

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
subset.solve(
    solver="icgn",  # "icgn" or "fagn".
    max_norm=1e-3,  # Convergence threshold on the warp update.
    max_iterations=50,
    tolerance=0.75,  # Minimum C_ZNCC to count as solved.
)
print(subset)
print(f"  u, v:       {subset.p[0]:.4f}, {subset.p[1]:.4f} px")
print(f"  C_ZNCC:     {subset.c_zncc:.6f}")
print(f"  iterations: {subset.iterations}")
print(f"  converged:  {subset.converged}")
subset.inspect()
subset.convergence()

# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------
path = OUTPUT / "subset.pyv"
subset.save(str(path))
subset = gp.load(str(path))
print(f"\nLoaded: {subset}")
print(f"  u, v:   {subset.p[0]:.4f}, {subset.p[1]:.4f} px")
print(f"  C_ZNCC: {subset.c_zncc:.6f}")
subset.inspect()
