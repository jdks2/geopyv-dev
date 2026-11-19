"""Demo 7 - Calibration: pixels to physical units with a ChArUco board.

Needs OpenCV (pip install "geopyv_dev[calibration]"). images/calibration/ holds
example ChArUco board photographs; for a real test, replace them with images
taken through the viewing window at the test's focus, aperture and zoom.
"""

import sys
from pathlib import Path

import numpy as np

import geopyv_dev as gp

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "images" / "calibration"
if not CALIBRATION.is_dir() or not any(CALIBRATION.iterdir()):
    sys.exit("images/calibration/ is empty - add ChArUco board images first.")

# ---------------------------------------------------------------------------
# Build: describe the printed board
# ---------------------------------------------------------------------------
print("=== Calibration ===")
cal = gp.Calibration(
    calibration_dir=str(CALIBRATION),
    common_name="",  # Shared file-name prefix ("" = every image in the folder).
    file_format=".jpg",
    board_parameters=(29, 18, 10, 8),  # Columns, rows, square and marker length (mm).
)

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
cal.solve(
    ext_id=None,  # Image that sets the object-space frame (None = last accepted).
    binary_threshold=110,  # Binarisation threshold before marker detection.
    acceptance_threshold=50,  # Minimum corners for an image to be used.
)
print(f"  solved: {cal.solved}")
print(f"  intrinsic matrix:\n{cal.params.intmat}")
print(f"  distortion: {cal.params.dist}")

cal.inspect(image_index=0)  # Detected corners on one image.
cal.visualise()  # Corner coverage across all accepted images.
cal.contour(quantity="R")  # Lens undistortion per corner.
cal.error(quantity="R")  # Reprojection error per corner.

# ---------------------------------------------------------------------------
# Map points between image (px) and object (mm) space
# ---------------------------------------------------------------------------
img_pts = np.array([[500.0, 500.0], [1500.0, 1000.0]])
obj_pts = cal.params.i2o(img_pts)
print(f"\n  image {img_pts.tolist()} ->\n  object {np.round(obj_pts, 3).tolist()}")
print(f"  round trip error: {np.abs(cal.params.o2i(obj_pts) - img_pts).max():.2e} px")
