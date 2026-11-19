"""Demo 7 - Calibration: from pixels to millimetres.

Everything so far is in pixels, and a camera lens bends straight lines a
little (most near the edges of the picture). Calibration measures the camera
from photographs of a printed ChArUco board - a chessboard with a coded
marker in every white square, so each corner can be identified even when
part of the board is hidden. From the known board size it finds:

  * the INTRINSIC parameters: focal length and image centre, in px;
  * the lens DISTORTION: how far the lens moves each point;
  * the EXTRINSIC pose: where the board (and so the specimen face) is.

With these, image positions convert to millimetres on the specimen face.

Take at least 10 photographs of the board, tilted and moved about, through
the viewing window at the test's focus, aperture and zoom, and do not touch
the camera settings afterwards. One photograph with the board flat on the
specimen face sets the object coordinates (ext_id).

Needs OpenCV: pip install "geopyv_dev[calibration]". images/calibration/
holds example board photographs; replace them with your own.
"""

import sys
from pathlib import Path

import numpy as np

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "images" / "calibration"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)
if not CALIBRATION.is_dir() or not any(CALIBRATION.iterdir()):
    sys.exit("images/calibration/ is empty - add ChArUco board images first.")

# ---------------------------------------------------------------------------
# 1. Describe the printed board
# ---------------------------------------------------------------------------
print("=== 1. The board ===")
cal = gp.Calibration(
    calibration_dir=str(CALIBRATION),
    common_name="",  # Use images whose names start with this ("" = all).
    file_format=".jpg",
    # Squares across, squares down, square side, marker side. The lengths set
    # the units of every calibrated result - here millimetres.
    board_parameters=(29, 18, 10, 8),
)

# ---------------------------------------------------------------------------
# 2. Solve
# ---------------------------------------------------------------------------
print("=== 2. Solving (finding the board in every image, then fitting the camera) ===")
cal.solve(
    ext_id=None,  # Number in the name of the board-on-specimen photo (None: the last used).
    binary_threshold=110,  # Grey level to binarise at before finding markers (None: off).
    acceptance_threshold=50,  # Images with fewer corners found are left out.
)
k = cal.params.intmat
print(f"  focal length:   {k[0, 0]:.0f}, {k[1, 1]:.0f} px")
print(f"  image centre:   {k[0, 2]:.0f}, {k[1, 2]:.0f} px")
print(f"  lens distortion k1, k2, p1, p2, k3: {np.round(cal.params.dist, 4)}")

cal.inspect(image_index=0)  # The corners found in one image.
cal.visualise()  # Corners from every image: they should cover the whole frame.
cal.contour(quantity="R")  # How far the lens moves each corner, px.
cal.error(quantity="R")  # Found corner vs. where the camera model puts it: should be small.

# ---------------------------------------------------------------------------
# 3. Use it
# ---------------------------------------------------------------------------
print("\n=== 3. Pixels to millimetres ===")
img_pts = np.array([[500.0, 500.0], [1500.0, 1000.0]])
obj_pts = cal.params.i2o(img_pts)  # image -> object (mm); o2i goes back.
for a, b in zip(img_pts, obj_pts):
    print(f"  pixel ({a[0]:.0f}, {a[1]:.0f}) -> ({b[0]:.2f}, {b[1]:.2f}) mm")
print(f"  round trip error: {np.abs(cal.params.o2i(obj_pts) - img_pts).max():.1e} px")
print("A Field solved with field.solve(calibration=cal.params) has its positions")
print('and displacements in mm; mesh.contour("u", calibration=cal.params) plots')
print("a mesh's displacements in mm.")

path = OUTPUT / "calibration.pyv"
cal.save(str(path))
print("\nSaved to output/calibration.pyv - the GUI's Calibration tab can import it.")
