08 — Calibration: pixel to physical coordinates
=================================================

Every DIC measurement starts in **image space** — dimensions are in pixels and things are
distorted by the camera lens. ``Calibration`` maps those measurements to the
**object space** — undistorted measurements in physically meaningful units. For this, a 
known target is used, namely a ChArUco board. 

Why calibrate?
---------------

Uncalibrated PIV/DIC yields pixel displacements and distorted strains. Without calibration,
you cannot:

- Measure undistorted, real-world deformations.
- Compare results across different tests.

Calibration solves the full camera model — intrinsic matrix, lens distortion, and
extrinsic pose — from a set of images of a known planar target, then provides the
``i2o`` (image-to-object) and ``o2i`` (object-to-image) mappings.

Why ChArUco over dot grids?
-----------------------------

Traditional calibration grids use a regular array of dots. In a geotechnics lab the
camera looks at the specimen through a transparent acrylic viewing window — which contributes
to overall image distortion — and dots on the inner wall provide scaling. 

Instead, ChArUco boards combine a chessboard pattern with embedded ArUco markers. Images of the
board are taken by the camera through the viewing window. This gives three advantages:

1. **Sub-pixel corner accuracy** — chessboard corners are detected more precisely
   than dot centroids. 
2. **Partial occlusion tolerance** — ArUco markers identify which corner is which
   even when part of the board is hidden by equipment or the window frame. Dots
   require the full grid to be visible or manual specification of positions. 
3. **No test occlusions** — permanent markers can create occlusions within the region of interest.

Incorporating the viewing window
----------------------------------

The viewing window is a refracting element in the optical path. Calibrating with the
board **without the window** does not capture the window's contribution to the overall distortion.

Instead, image the ChArUco board **through the viewing window** — in the same
optical configuration used during the test. The pinhole model then absorbs the
window's refractive effect into the intrinsic matrix and distortion coefficients,
so the calibration automatically corrects for it.

Camera stability after calibration
-------------------------------------

The calibration is valid only for the exact camera configuration used when the
calibration images were taken. Changing any of the following invalidates it:

- **Aperture** — alters the depth of field and the effective principal point.
- **Focus** — shifts the image plane and changes the focal length.
- **Zoom** — rescales the focal length.
- **Relative camera position or orientation** — changes the distortion effects.

Set focus, aperture, and zoom before calibration and do not touch them again.
If any of these must be changed (e.g. focus for the actual specimen distance), repeat
the calibration at the new settings (this can be done after the test). 

Setup
------

Acquire at least 10 images of the ChArUco board at varied orientations and distances
— from roughly the same working distance as the test — through the viewing window.
Name them consistently (e.g. ``cal_0.jpg``, ``cal_1.jpg``, …) in one directory.

.. code-block:: python

   import geopyv_dev as gp

   cal = gp.Calibration(
       calibration_dir  = "data/calibration/",
       common_name      = "cal",
       file_format      = ".jpg",
       board_parameters = (11, 8, 0.016, 0.012),   # columns, rows, sq_len, marker_len (m)
       show             = False,
   )

``board_parameters`` must match the physical board you printed:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``columns``
     - Number of chessboard columns
   * - ``rows``
     - Number of chessboard rows
   * - ``sq_len``
     - Physical side length of one chessboard square (metres)
   * - ``marker_len``
     - Physical side length of one ArUco marker (metres)

``show=True`` displays the board pattern at construction — useful to verify you are
using the correct board specification before collecting images.

Solving the calibration
------------------------

.. code-block:: python

   cal.solve(
       ext_id              = None,   # image index to use for the extrinsic pose
       binary_threshold    = None,   # pixel threshold for binarisation before detection
       acceptance_threshold= 50,     # minimum detected corners to accept an image
   )

   print(cal.params)   # CalibrationParams(fx=..., fy=..., cx=..., cy=...)

``acceptance_threshold`` controls how many ChArUco corners must be detected for an
image to contribute to the calibration.  Images with fewer corners are discarded.
Increase it if poor images are degrading the fit; decrease it if too few images pass.

``ext_id`` selects which accepted image is used to set the object-space coordinate
frame (the extrinsic pose). The default is the last accepted image.  Choose an image
taken with the board at the specimen face — in the same plane as the object you want
to track — to minimise depth-of-field and perspective errors.

After ``solve()``, ``cal.params`` is a ``CalibrationParams`` object containing:

.. list-table::
   :header-rows: 1

   * - Attribute
     - Shape
     - Description
   * - ``params.intmat``
     - (3, 3)
     - Intrinsic camera matrix (focal lengths + principal point)
   * - ``params.extmat``
     - (4, 4)
     - Extrinsic pose matrix (rotation + translation)
   * - ``params.dist``
     - (5,)
     - Distortion coefficients ``[k1, k2, p1, p2, k3]``

Inspecting the solve
-----------------------

Four plots help you judge whether the solve is any good, all available as methods
on ``cal`` (and as free functions in ``geopyv_dev.plots``):

.. code-block:: python

   cal.inspect(image_index=0)      # one accepted image + its detected ChArUco corners
   cal.visualise()                 # detected corners across every accepted image — coverage map
   cal.contour(quantity="R")       # per-corner undistortion magnitude (lens model only)
   cal.error(quantity="R")         # reprojection error: detected vs. model-reprojected corners

``inspect``/``visualise`` are purely about corner **detection** — did enough of the
board get seen, and how well is it spread across the frame? ``contour``/``error`` are
about the **fit** — ``contour`` shows how much the lens model itself distorts each
corner (independent of pose), while ``error`` shows the actual residual between each
detected corner and where the solved camera model predicts it should be; a good
calibration keeps this low and evenly spread, with no strong spatial pattern (a
pattern usually means the board wasn't rigid, or coverage was too sparse in that
region).

Adjusting the pose
----------------------

``modify()`` perturbs the solved extrinsic matrix by an additional rotation and/or
translation — e.g. to compensate for a small, known camera movement between the
calibration and the test, without re-shooting the whole board sequence:

.. code-block:: python

   cal.modify(dangles=[0.0, 0.0, 0.01], centre=[960.0, 540.0])

``dangles`` is an axis-angle rotation vector (Rodrigues form, as OpenCV produces)
added to the pose's own; ``centre`` is an image-space point used to re-anchor the
translation after rotating. This replaces ``cal.params`` with the perturbed result —
it does not mutate the original. Call ``cal.params.modify(...)`` directly instead if
you want the perturbed ``CalibrationParams`` without replacing ``cal.params``.

Calibrating objects
---------------------

Not every object in the pipeline should be calibrated:

- **``Subset``, ``Mesh``, and ``Sequence`` are never calibrated.** They are the DIC
  *solve* layer — reference/target images, node correlation, and propagation all
  happen in pixel space, and nothing about that process depends on physical units.
  None of them has a ``calibrate()`` method.
- **``Region`` can be calibrated — but only *after* it has been used to build a
  ``Mesh``/``Sequence``, as a reporting step**, not before. A boundary/exclusion
  polygon is defined in pixel space and fully consumed the moment ``Mesh``/``Sequence``
  construction runs — its coordinates are copied into the mesh's own node/element
  arrays, and the ``Region`` object itself is never touched again afterwards.
  Calibrating it at that point is safe, and lets you report its coordinates in
  physical units:

.. code-block:: python

   boundary_nodes = np.array([
       [50.0, 50.0], [50.0, 350.0], [350.0, 350.0], [350.0, 50.0]
   ])
   boundary = gp.PathRegion(nodes=boundary_nodes, hard=False)

   mesh = gp.Mesh(boundary=boundary, target_nodes=80, f_img=f_img, g_img=g_img,
                  size=(15.0, 70.0), mesh_order=2)
   mesh.solve(local_mask=local_mask, seed_coord=[200.0, 200.0], seed_warp=[0.0] * 12,
              subset_order=2, tolerance=0.75, seed_tolerance=0.9, method="icgn")

   # boundary has already done its job — Mesh copied its pixel-space nodes at
   # construction — so calibrating it now is safe, and purely for reporting.
   cal.calibrate(boundary)
   print(boundary.current_nodes)   # now in metres (or mm, depending on sq_len)

``cal.calibrate()`` accepts ``CircleRegion`` and ``PathRegion`` objects and converts
their node coordinates and centre history in-place using ``params.i2o``.

.. warning::
   Never pass an already-calibrated ``Region`` (``region.calibrated is True``) into
   ``Mesh(...)`` or ``Sequence(...)`` — its coordinates are in physical units, not
   pixels, and the mesh would be built at the wrong scale. This is enforced: doing
   so raises a ``TypeError``.

- **``Particle``/``Field`` calibrate at solve time.** This is where a physically
  meaningful, continuous position and strain path is actually constructed, so it's
  the natural place to convert to object space — see below.

Calibrated Mesh contours
----------------------------

For a quick full-field look at calibrated displacement without building a
``Particle``/``Field``, pass ``calibration=`` to ``mesh.contour()``:

.. code-block:: python

   mesh.contour("R", calibration=cal.params)

Only the **plotted value** changes — the background image and the mesh's own node
layout stay in pixel space. The image is a photograph, and lens distortion means a
calibrated node position isn't just a rescaled pixel position; overlaying calibrated
node positions on the pixel-space image would misalign the two. ``calibration=``
only applies to ``quantity="u"``/``"v"``/``"R"``; passing it alongside
``"C_ZNCC"``, ``"iterations"``, or ``"norm"`` raises a ``ValueError``, since those
aren't spatial quantities.

Applying calibration to a Field
----------------------------------

Pass the ``CalibrationParams`` to ``Field.solve()``:

.. code-block:: python

   sequence = gp.load("data/sequence.pyv")

   field = gp.Field(sequence_solution=sequence, track=True, depth=0.050)
   field.solve(
       factor      = 0.0,
       true_incs   = True,
       calibration = cal.params,
   )
   print(f"calibrated : {field.calibrated}")

With ``calibration`` supplied:

- Particle initial coordinates are converted from pixels to object space before
  the first increment.
- Displacement increments are converted from pixels to physical units at each
  increment.
- ``field.calibrated`` is set to ``True``.
- ``field.depth`` should be set to the specimen thickness in **physical units**
  (matching ``sq_len``) so that volumetric strains are computed correctly.

Save and reload
-----------------

.. code-block:: python

   cal.save("data/calibration.pyv")
   cal = gp.load("data/calibration.pyv")

This persists the camera model (``intmat``/``extmat``/``dist``) plus the diagnostic
data behind ``inspect``/``visualise``/``contour``/``error`` (detected corners, ids,
accepted image paths, reprojection points) — a reloaded ``Calibration`` supports all
of those immediately. It does **not** persist the ChArUco board/dictionary setup, so
a reloaded object is ready for using the result, but not for a fresh ``solve()``.

Using ``CalibrationParams`` directly
--------------------------------------

``CalibrationParams`` can be constructed without a ``Calibration`` if you already
have the intrinsic and extrinsic matrices from another source (e.g. a previous
OpenCV calibration):

.. code-block:: python

   import numpy as np

   params = gp.CalibrationParams(
       intmat = np.array([[2150.0, 0.0, 960.0],
                          [0.0, 2150.0, 540.0],
                          [0.0,    0.0,   1.0]]),
       extmat = np.eye(4),
       dist   = np.array([-0.12, 0.05, 0.0, 0.0, 0.0]),
   )

   # Map image pixels to object coordinates
   imgpnts = np.array([[960.0, 540.0]])   # image centre
   objpnts = params.i2o(imgpnts)
   print(objpnts)

   # Round-trip check
   recovered = params.o2i(objpnts)
