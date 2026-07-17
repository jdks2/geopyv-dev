01 — Images and Masks
=======================

Before measuring deformation, you need two things:

- An **``Image``** — a greyscale photograph with pre-computed interpolation data.
- A **``Mask``** — defining which pixels to include in a subset or which region to analyse.

Before that, let's think about the surface we want to track.

Speckle
--------

Why texture?
~~~~~~~~~~~~~~~~~~~~~~~~

A plain, uniform surface has no features to track — every patch looks the same.
PIV/DIC requires a **random, high-contrast texture** so each subset has a unique intensity
fingerprint that can be matched reliably between images. In geotechnics, this is usually achieved
using dyed sand (mixed in or distributed across the surface), or flicking paint onto a
contrasting surface (e.g. a membrane or structure). In general:

- Speckle diameter — pixel size : **d/p > 4**
- Coverage ≈ **40–60%** of the surface
- Avoid repeated patterns (lattices, gratings, etc. cause false matches)

Why validate against a known deformation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real footage never comes with a ground truth attached — you can measure how a real surface
moved, but you can't independently check the answer against what it *actually* did. A
deformation you specify yourself, on the other hand, is exactly known at every pixel, which
makes it possible to quantify a solver's accuracy and precision directly (bias, standard
error, breakdown at large strains) rather than just eyeballing plausibility. See
:doc:`07_validation` for the full treatment of validating solves this way.

The ``Speckle`` class
~~~~~~~~~~~~~~~~~~~~~~~~

``geopyv_dev`` includes a ``Speckle`` class that generates exactly this kind of synthetic
speckle series — a known, reproducible deformation baked into a stack of images — so that
solver accuracy can be checked against ground truth, and so that tutorial examples don't
depend on a camera. Every image pair in these tutorials is produced this way.

``Speckle`` takes six groups of configuration:

.. list-table::
   :header-rows: 1

   * - Argument
     - Contents
   * - ``image_cfg``
     - ``image_dir``, ``name``, ``image_size`` (default ``(1001, 1001)``), ``file_format``
       (default ``".jpg"``)
   * - ``speckle_cfg``
     - ``speckle_size`` (px blob radius), ``speckle_number``
   * - ``progression``
     - ``"deformation"`` (warp progresses, noise held at its final value) or ``"noise"``
       (noise progresses, deformation held at full strength)
   * - ``deformation_cfg``
     - ``comp`` — the 12-element final warp vector (see below); optionally ``origin``,
       or ``mode="rotation"`` (with ``angle``) / ``mode="SB"`` (shear-band, with ``option``
       and ``width``) in place of a general ``comp`` warp
   * - ``noise_cfg``
     - ``(noise_pos, noise_int)`` — final Gaussian noise standard deviations
   * - ``scale_cfg``
     - ``scale`` (``"lin"`` or ``"log"``), ``n`` (total images — image 0 is always the
       undeformed reference), and ``min`` for log scale

The generation process itself is simple: a fixed cloud of reference speckle positions is
sampled once, up front. Every output image is then rendered **directly from that same
reference cloud** — the warp implied by ``comp`` (scaled by how far along the ``n``-image
progression that frame is) is applied analytically to the reference positions, not chained
frame-to-frame from the previous image. Image 0 is always the undeformed reference; each
subsequent image carries progressively more deformation and/or noise.

What a deforming image series looks like
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/gifs/quad_deformation.gif
   :alt: Four side-by-side speckle panels, each looping forever. Top-left (u):
         the texture scrolls sideways like a conveyor belt. Top-right (rot):
         the texture spins continuously about its centre. Bottom-left (dvdy):
         the texture squashes vertically and relaxes back, pulsing. Bottom-right
         (shear): a shear deformation grows, relaxes, reverses direction, and
         relaxes again.
   :width: 460px
   :align: center

Four of the warp components an ``Image`` series might carry: a rigid translation
(**u**), a **rotation**, an isolated **dv/dy** term, and a full **shear**
combining ``dv/dx`` and ``du/dy`` together. The last of these — pure shear — is the
deformation used to build the example series below.

The rest of these tutorials use one shared example: **five images of increasing
shear strain**, at ``images/shear/shear_0.jpg`` … ``shear_4.jpg``. It's a synthetic
``Speckle`` series with a pure-shear warp (``comp[3]`` = dv/dx, ``comp[4]`` = du/dy,
following the warp-vector ordering ``[u, v, u_x, v_x, u_y, v_y, ...]``), so it's exactly
reproducible:

.. code-block:: python

   import geopyv_dev as gp

   speckle = gp.Speckle(
       image_cfg={"image_dir": "images/shear", "name": "shear", "image_size": (1001, 1001)},
       speckle_cfg={"speckle_size": 10, "speckle_number": 7000},
       progression="deformation",
       deformation_cfg={"comp": [0.0, 0.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
       noise_cfg=(0.0, 0.0),
       scale_cfg={"scale": "lin", "n": 5},
   )
   speckle.solve(seed=42)

   f_img = gp.Image(filepath="images/shear/shear_0.jpg")
   g_img = gp.Image(filepath="images/shear/shear_4.jpg")

Image
------

Why images are preloaded
~~~~~~~~~~~~~~~~~~~~~~~~~~~

DIC solves are iterative: the same reference/target image pair gets re-sampled at
sub-pixel positions thousands of times over the course of solving one mesh, and
millions of times over a full sequence. Re-deriving an interpolant from raw pixels on
every lookup would make that hopelessly slow, so ``Image`` does the expensive part —
**once, up front, at load time** — and every subsequent lookup is then a cheap matrix
multiply.

Image preprocessing
~~~~~~~~~~~~~~~~~~~~~~~

Loading an ``Image`` does three things:

1. Converts to greyscale.
2. Applies a mild **Gaussian blur** (σ = 1.1) — this suppresses pixel-level aliasing
   that would otherwise leak into the sub-pixel interpolation.
3. Pre-computes the **bi-quintic B-spline coefficient matrix** (``qcqt``) — the DIC
   solver needs intensities at arbitrary sub-pixel positions (e.g. x = 127.43); this
   matrix lets any such lookup be evaluated as a small matrix multiply instead of
   re-fitting a spline every time. The block for pixel (i, j) lives at
   ``qcqt[i*6 : i*6+6, j*6 : j*6+6]``.

.. code-block:: python

   import geopyv_dev as gp

   img = gp.Image(filepath="photo.jpg")   # from file
   img = gp.Image(image_gs=array)         # from numpy array (already greyscale)

Key attributes:

.. list-table::
   :header-rows: 1

   * - Attribute
     - Shape
     - Description
   * - ``image_gs``
     - (H, W)
     - Greyscale pixel values 0–255
   * - ``qcqt``
     - (H·6, W·6)
     - Pre-computed B-spline coefficient blocks
   * - ``border``
     - int
     - Padding used for coefficient computation
   * - ``filepath``
     - str | None
     - Source path

What you can extract from an ``Image``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once loaded, an ``Image`` is ready to hand to a ``Subset``, ``Mesh``, or ``Sequence``
— but you can also inspect it directly:

.. code-block:: python

   import numpy as np

   print(f_img.image_gs.shape)        # (1001, 1001) — greyscale array
   print(f_img.image_gs.dtype)        # uint8
   print(f_img.qcqt.shape)            # (1001*6, 1001*6) — B-spline coefficients
   print(f_img.border)                # interpolation border padding, px
   print(f_img.filepath)              # 'images/shear/shear_0.jpg'

   # A quick visual sanity check with matplotlib:
   import matplotlib.pyplot as plt
   plt.imshow(f_img.image_gs, cmap="gist_gray")
   plt.show()

Mask
-----

A ``Mask`` defines a region of pixels using a binary mask. There are two fundamentally
different types, both accessed through the same class via ``mask_type``:

.. list-table::
   :header-rows: 1

   * - ``mask_type``
     - Purpose
     - Key parameters
   * - ``"local"``
     - Defining a subset
     - ``shape`` (``"circle"`` / ``"square"``), ``size``
   * - ``"global"``
     - Defining a Region of Interest (RoI)
     - ``f_img``, ``boundary``, ``exclusions``

.. tip::
   Use ``"local"`` for ``Subset`` and ``Mesh`` (the default). Use ``"global"`` only when
   you need pixel-level masking of the image itself.

.. code-block:: python

   # Local: reused as the template shape at every subset centre.
   mask_c = gp.Mask(mask_type="local", shape="circle", size=25)
   mask_s = gp.Mask(mask_type="local", shape="square", size=25)

Local vs global — the difference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/img/mask_local_vs_global.png
   :alt: Left, a local mask — a circle and a square patch shape drawn around a
         single subset centre on the speckle image. Right, a global mask — a
         binary black-and-white image showing a rectangular active region with
         a smaller rectangular exclusion hole cut out of it.
   :width: 600px
   :align: center

A **local** mask is a small *shape* — just pixel offsets relative to wherever a
subset ends up centred; the same ``Mask`` object is reused at every subset/node in a
mesh. A **global** mask is a *pre-computed binary image* the size of the whole
photograph — every pixel is resolved once, up front, against a boundary polygon and
any exclusions, rather than re-evaluated per subset.

Global masks
~~~~~~~~~~~~~

A **global mask** computes a binary image of shape (H, W) based on a boundary polygon
and optional exclusion regions. Pixels inside the boundary (and outside exclusions)
are set to 1 (active); all others are 0.

Global masks are passed to ``Subset`` or created standalone with ``Mask(mask_type="global")``.

.. code-block:: python

   import numpy as np

   boundary_nodes = np.array([
       [250.0, 250.0], [250.0, 750.0], [750.0, 750.0], [750.0, 250.0]
   ])
   excl_nodes = np.array([
       [560.0, 560.0], [560.0, 680.0], [680.0, 680.0], [680.0, 560.0]
   ])

   global_mask = gp.Mask(
       mask_type="global",
       f_img=f_img,
       boundary=gp.PathRegion(nodes=boundary_nodes, hard=True),
       exclusions=[gp.PathRegion(nodes=excl_nodes, hard=True)],
   )
   print(global_mask.binary.shape)   # (1001, 1001), dtype uint8 — 1 = active

Region
~~~~~~~

``CircleRegion`` and ``PathRegion`` are **tracked** boundary/exclusion objects. Unlike
plain numpy arrays, they record their full deformation history (``history_nodes``,
``history_centres``) and can be updated as a ``Sequence`` advances through its images.

.. list-table:: ``CircleRegion(centre, radius=50.0, size=20.0, option="F", hard=True, compensate=True)``
   :header-rows: 1

   * - Parameter
     - Description
   * - ``centre``
     - ``[x, y]`` centre of the circle
   * - ``radius``
     - Radius of the boundary polygon (default ``50.0``)
   * - ``size``
     - Approximate arc spacing between polygon vertices (default ``20.0``) —
       at least 6 vertices are always generated
   * - ``option``
     - Tracking mode: ``"D"``, ``"S"``, ``"R"``, or ``"F"`` (default ``"F"``)
   * - ``hard``
     - Include the boundary in a global mask's binary image (default ``True``)
   * - ``compensate``
     - Reserved for compensation handling (default ``True``)

.. list-table:: ``PathRegion(nodes, centre=None, option="F", hard=True, compensate=True, radius=25.0)``
   :header-rows: 1

   * - Parameter
     - Description
   * - ``nodes``
     - ``(N, 2)`` polygon vertices ``[x, y]``
   * - ``centre``
     - ``[x, y]``, optional — defaults to the mean of ``nodes``
   * - ``option``
     - Tracking mode: ``"D"``, ``"S"``, ``"R"``, or ``"F"`` (default ``"F"``)
   * - ``hard``
     - Include the boundary in a global mask's binary image (default ``True``)
   * - ``compensate``
     - Reserved for compensation handling (default ``True``)
   * - ``radius``
     - Characteristic radius stored for mesh use (default ``25.0``) — not derived
       from ``nodes``, so pass the actual scale of the polygon if it matters downstream

The ``option`` parameter controls how the region is *intended* to move between frames:

.. list-table::
   :header-rows: 1

   * - ``option``
     - Meaning
   * - ``"F"``
     - **Flexible** — region deforms with the local mesh warp (default)
   * - ``"R"``
     - **Rigid** — region translates and rotates, but never deforms
   * - ``"S"``
     - **Static** — region never moves
   * - ``"D"``
     - **Defined** — intended to follow a pre-supplied, user-specified trajectory

.. code-block:: python

   # CircleRegion: circle centred at (500, 500), radius 200 px
   circle_region = gp.CircleRegion(centre=[500.0, 500.0], radius=200.0, size=15.0, option="F")

   # PathRegion: arbitrary polygon
   path_region = gp.PathRegion(nodes=boundary_nodes, option="R")

.. image:: /_static/gifs/region_motion.gif
   :alt: Four panels, each a circular region with a red spoke marking
         orientation. Flexible squashes into an ellipse while drifting.
         Rigid rotates (the spoke sweeps around) and translates while
         staying perfectly circular. Static never changes. Defined ripples
         through an irregular three-lobed wobble.
   :width: 460px
   :align: center

Each panel is driven through the region's real API — repeated ``store_flexible``/
``store_rigid`` calls (followed by ``update()``, which is what actually advances
``current_nodes`` — see below) for Flexible and Rigid, no calls at all for Static.
**Except Defined**: as of today, ``"D"`` has no working automatic store/update path
(``store_region_step`` treats ``"S"`` and ``"D"`` identically — a no-op — and there's
no constructor that accepts the multi-snapshot trajectory the option is meant to
represent). The Defined panel above is instead driven by writing directly to
``current_nodes`` every frame, which is the only supported way to feed a region
externally-specified positions right now. See ``region_option_d_investigation.md``
in the repository root for the full comparison against the original Python behaviour.

What you can extract from a ``Region``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(circle_region.option)            # 'F'
   print(circle_region.current_nodes)     # (N, 2) — current working node positions
   print(circle_region.current_centre)    # [x, y] — current working centre
   print(circle_region.history_nodes)     # list of (N, 2) arrays, one per store_* call
   print(circle_region.history_centres)   # list of [x, y], one per store_* call
   print(circle_region.counter)           # number of store_* calls so far
   print(circle_region.solved)            # True once at least one store_* call has run
   print(circle_region.hard)              # whether it's included in a global mask
   print(circle_region.compensate)

.. note::
   ``current_nodes`` only reflects a stored step once ``update()`` has been called
   with the corresponding image filepath — ``store_flexible``/``store_rigid`` only
   append to ``history_nodes``, matching how ``Sequence`` calls them internally.
   Also note: ``radius``/``size`` passed to ``CircleRegion`` aren't retrievable
   afterwards — there's no getter for them.

Loading an image in the GUI
-------------------------------

.. note::
   A short screen recording of this workflow will be added here.

``geopyv-gui`` (``cargo run -p geopyv-gui``) opens with the **Images** tab active:

1. Click **Import** at the bottom of the left-hand file list.
2. Pick one or more image files (``.jpg``/``.jpeg``/``.png``/``.tif``/``.tiff``) from
   the native file dialog — they're copied into the project's image directory and
   appear in the list.
3. Click an entry in the list to select it — it's shown immediately in the central
   viewer.

Summary
--------

.. list-table::
   :header-rows: 1

   * - Object
     - When to use
   * - ``Image``
     - Always — wraps every photograph
   * - ``Mask(mask_type="local")``
     - Passed to ``Subset``, ``Mesh``, ``Sequence`` as the subset template
   * - ``Mask(mask_type="global")``
     - When you need pixel-level masking of the image
   * - ``CircleRegion``
     - Circular boundary or exclusion that should track the deformation
   * - ``PathRegion``
     - Polygonal boundary or exclusion with optional tracking

**Next:** :doc:`02_subset` — correlating a single subset to find a point displacement.
