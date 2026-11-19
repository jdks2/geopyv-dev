"""Demo 1 - Subset: tracking one patch between two images.

A Subset is a patch of the reference image (the TEMPLATE). The solver
searches the target image for where that patch went and how it changed shape.
The answer is a WARP - for a first-order subset six numbers:

    u, v              displacement of the subset centre (px)
    du/dx, dv/dx,     how the displacement changes across the subset, i.e.
    du/dy, dv/dy      the stretch and shear of the patch

The match quality is C_ZNCC (zero-normalised cross-correlation): 1 is a
perfect match, 0 no match. Above about 0.9 is usual for a good image pair.
"""

from pathlib import Path

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images" / "shear"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

f_img = gp.Image(filepath=str(IMAGES / "shear_0.png"))  # reference
g_img = gp.Image(filepath=str(IMAGES / "shear_4.png"))  # target
local_mask = gp.Mask(mask_type="local", shape="circle", size=30)

# ---------------------------------------------------------------------------
# 1. Build: cut the template out of the reference image
# ---------------------------------------------------------------------------
print("=== 1. Building a subset ===")
subset = gp.Subset(
    coord=[400.0, 650.0],  # Centre [x, y] in the reference image, px (x right, y down).
    local_mask=local_mask,  # Its shape and size (here a circle of radius 30 px).
    f_img=f_img,  # Reference image.
    g_img=g_img,  # Target image.
    subset_order=1,  # 1: the patch may stretch and shear uniformly; 2: also bend.
)
print(subset)
print("Before solving, check the template has enough texture to be found again:")
print(f"  sigma_intensity = {subset.sigma_intensity:.1f}  (brightness spread; above ~15 is healthy)")
print(f"  SSSIG           = {subset.sssig:.2e}  (strength of brightness gradients; above ~1e5 is healthy)")
subset.inspect()  # The template on the reference image.

# ---------------------------------------------------------------------------
# 2. Solve
# ---------------------------------------------------------------------------
print("\n=== 2. Solving ===")
print("The solver starts from a guess (no motion) and improves the warp step by")
print("step until the change is below max_norm or max_iterations is reached.")
subset.solve(
    solver="icgn",  # "icgn" (fast, the usual choice) or "fagn".
    max_norm=1e-3,  # Stop when the warp changes by less than this.
    max_iterations=50,
    tolerance=0.75,  # Lowest C_ZNCC accepted as a match.
)
print(subset)
p = subset.p
print(f"  displacement u, v: {p[0]:.3f}, {p[1]:.3f} px")
print(f"  gradients du/dx, dv/dx, du/dy, dv/dy: {p[2]:.4f}, {p[3]:.4f}, {p[4]:.4f}, {p[5]:.4f}")
print(f"  C_ZNCC:            {subset.c_zncc:.5f}")
print(f"  iterations:        {subset.iterations} (converged: {subset.converged})")
print("The images were sheared with du/dy = dv/dx = 0.01 about the centre (500, 500),")
print("so here u should be 0.01 x (650 - 500) = 1.5 px and v = 0.01 x (400 - 500) = -1.0 px.")
subset.inspect()  # Now also shows the result.
subset.convergence()  # How C_ZNCC and the warp settled, iteration by iteration.

# ---------------------------------------------------------------------------
# 3. Save and reload
# ---------------------------------------------------------------------------
print("\n=== 3. Saving ===")
print("Every geopyv result saves to a .pyv file and loads back with gp.load.")
path = OUTPUT / "subset.pyv"
subset.save(str(path))
subset = gp.load(str(path))
print(f"Loaded from output/subset.pyv: {subset}")
print("\nNext: 2_mesh.py measures a whole area at once.")
