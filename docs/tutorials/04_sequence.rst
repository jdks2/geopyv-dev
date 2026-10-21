04 — Sequence: multi-image time series
=========================================

A ``Mesh`` (:doc:`03_mesh`) solves one image pair. A ``Sequence`` is a temporal
series of meshes: it walks through every frame of an image directory, solving
each successive pair and carrying geometry and warp information forward between
them — the natural extension when you want to track *progressive* deformation
rather than a single before/after snapshot.

This tutorial covers: image timing, reference-image strategy, ``SequenceOptions``,
per-pair solving, and extracting/plotting results — including rates — across a
sequence.

Setup
------

Continuing with the canonical shear series (:doc:`01_images_and_masks`) — now
using the full 5-frame series rather than just the first/last frame:

.. code-block:: python

   import numpy as np
   import geopyv_dev as gp

   boundary_nodes = np.array([
       [300.0, 300.0], [300.0, 700.0], [700.0, 700.0], [700.0, 300.0]
   ])
   boundary = gp.PathRegion(nodes=boundary_nodes, hard=False)
   local_mask = gp.Mask(mask_type="local", shape="circle", size=25)
   seed_coord = [500.0, 500.0]

Constructing a ``Sequence``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sequence = gp.Sequence(
       image_dir    = "images/shear",   # auto-discovers shear_0.jpg ... shear_4.jpg
       boundary     = boundary,
       target_nodes = 80,
       size         = (15.0, 70.0),
       mesh_order   = 2,
       time         = 0.1,              # images taken at 10 Hz — see below
   )

The geometry for the first pair is generated immediately, the same as a standalone
``Mesh`` (:doc:`03_mesh`) — no correlation has happened yet.

Image timing
~~~~~~~~~~~~~~

``time`` records when each image was taken, so results can later be plotted as
rates (velocities, strain rates). It takes one of:

- **A number** — the time step between consecutive images, in seconds:
  ``time=0.1`` for images taken at 10 Hz gives times ``0.0, 0.1, 0.2, …``. If you
  have thinned the image directory (say, kept every 10th frame), give the spacing
  of the images you kept.
- **A list** — the time of every image in the directory, in sorted order, e.g.
  ``time=[0.0, 0.5, 2.0, 2.1, 6.0]``. The spacing can be irregular.
- **Nothing** (``None``, the default) — the times are read from the file names,
  when every name carries a timestamp. Three forms are recognised:

  .. list-table::
     :header-rows: 1

     * - Form
       - Examples
     * - Date and time, with optional fractional seconds
       - ``img_2026-10-14T13-45-02.123``, ``cam_20261014_134502123_0001``
     * - Unix epoch in s, ms, µs or ns (10, 13, 16 or 19 digits)
       - ``frame_1697283902123``
     * - Time of day, with separators (a midnight crossing is handled)
       - ``shot_13-45-02.500``

  The timestamp must be found in the same place in every name. A bare number such
  as ``shear_3`` or ``frame_000123`` is always taken as a frame counter, never a
  time, so the canonical shear series above needs ``time`` to be given. When
  timestamps are found, the images are ordered by them, and a one-line message
  says which form was matched, so a wrong match is easy to spot.

.. code-block:: python

   print(sequence.time)          # [0.  0.1 0.2 0.3 0.4] — seconds, one per image
   print(sequence.time_source)   # "given", "filename (<format>)" or "none"

Times are stored relative to the first image (except an explicit list, which is
kept as given) and are carried into every pair's ``MeshSolution`` as
``(f_time, g_time)`` — the times of *that pair's* reference and target images —
and on into ``Particle`` and ``Field`` (:doc:`05_particle`, :doc:`06_field`).
They survive ``save``/``load``.

Solve
------

The problem: a fixed triangulation distorts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tracking progressive deformation sounds like it should mean "solve one mesh, then
just keep displacing its same nodes frame after frame." It doesn't work, because
the triangulation itself is only valid geometry for however much the region has
deformed *so far*. Push it far enough and elements skew, and eventually fold over
— exactly the compatibility failure ``Mesh::solve`` guards against in
:doc:`03_mesh`.

.. image:: /_static/gifs/sequence_distortion.gif
   :alt: The same fixed triangulation, generated once from a shear-band image
         series, replayed at each frame's accumulated node position. The interior
         mesh visibly skews into a parallelogram-like shear band across frames,
         and the worst element angle drops, even though every underlying
         correlation stayed above 0.998 C_ZNCC — this is a geometry problem, not
         a correlation one.
   :width: 460px
   :align: center

This uses the real ``images/SB`` shear-band series: one ``Mesh`` solved once
(``sync=True`` reuses that same node connectivity for every pair, rather than
re-triangulating), then its nodes are displaced by each frame's real solved
displacement and replotted with the *same* elements throughout. Correlation
quality never drops — the DIC solve is fine — but the geometry it's reporting
strain on becomes progressively less trustworthy. This is precisely why
``Sequence`` exists: it re-triangulates when the geometry actually needs it,
rather than never.

.. tip::
   To reproduce a shear-band series like the distortion demo above, use
   ``mode="SB"`` in ``deformation_cfg`` — but note the shear-band strain is read
   from **``comp[4]``**, not ``comp[0]``: ``deformation_cfg={"comp": [0,0,0,0,
   0.08, 0,0,0,0,0,0,0], "mode": "SB", "option": "sin", "width": 100.0}``.

Fixed reference vs advancing it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every pair in a sequence is a reference image ``f`` correlated against a target
image ``g``. There are two basic strategies for choosing ``f`` as you move through
the frames, discussed generally (for image-based deformation measurement in
geotechnics) in Stanier, Blaber, Take & White (2016), *Improved image-based
deformation measurement for geotechnical applications*, Canadian Geotechnical
Journal:

- **Fixed reference** — always correlate against frame 0. Displacements are
  cumulative and drift-free, but correlation gets harder every frame as the
  deformation from frame 0 grows.
- **Incremental reference** — advance ``f`` to the previous frame after every
  pair. Each individual correlation stays easy (small increments), at the cost of
  needing to sum per-pair displacements to get anything cumulative, and small
  per-pair errors can accumulate ("drift").

``geopyv_dev`` defaults to the fixed-reference strategy (``sequential=False``),
but doesn't commit to it rigidly: if a pair fails to correlate directly against
the fixed reference, the solver automatically falls back to the previous frame as
a one-off reference and retries — it only advances, or "leapfrogs" the reference
forward, when the fixed reference actually stops working, rather than either every
frame (``sequential=True``) or never. There's no separate ``leapfrog`` flag; it's
what the default already does, and :ref:`the flow diagram below <sequence-flow>`
shows exactly where that fallback sits in the per-pair loop.

``SequenceOptions``
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Effect
   * - ``guide``
     - ``True``
     - Precondition the next pair's seed warp from the previous pair's projected
       displacement (particle-based warm start).
   * - ``sequential``
     - ``False``
     - Advance the reference to the previous frame after **every** successful
       pair, rather than only on a correlation failure.
   * - ``sync``
     - ``True``
     - Reuse the previous pair's mesh geometry (skip CDT) as long as the
       reference hasn't changed; reset automatically whenever it does.
   * - ``override_``
     - ``False``
     - On the failure-triggered reference fallback specifically, relax the next
       pair's tolerance to 0 and bypass the post-corrections quality gate from
       :doc:`03_mesh`, rather than letting that retry fail the whole sequence.

.. code-block:: python

   options = gp.SequenceOptions(guide=True, sync=True, sequential=False)

.. _sequence-flow:

How ``Sequence::solve`` walks through the frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/img/sequence_solve_flow.png
   :alt: Flow diagram of Sequence::solve. Initialise f_index=0, g_index=1, then
         loop over Mesh::solve for the current pair. On success, branch on
         sequential (advance the reference every frame, or hold it fixed) before
         incrementing g_index and looping while more frames remain. On failure,
         check whether the reference was already the immediately-preceding frame
         — if so, curtail the sequence as unsolvable; if not, fall back to that
         frame as the new reference (optionally relaxing tolerance via override_)
         and retry the same pair.
   :width: 620px
   :align: center

Solving
~~~~~~~~~

.. code-block:: python

   sequence.solve(
       local_mask     = local_mask,
       seed_coord     = seed_coord,
       seed_warp      = [0.0] * 12,
       subset_order   = 2,
       tolerance      = 0.75,
       seed_tolerance = 0.9,
       method         = "icgn",
       options        = gp.SequenceOptions(guide=True, sync=True, sequential=False),
   )
   print(f"solved  : {sequence.solved}")
   print(f"n_pairs : {sequence.n_pairs}")                       # 4 pairs from 5 images
   print(f"pair 0 mean C_ZNCC : {sequence.c_zncc(0).mean():.4f}")
   print(f"reference_updates  : {sequence.reference_updates}")  # all False here — every
                                                                 # pair correlated fine
                                                                 # against frame 0 directly

Results
--------

Two ways to reach a given pair's data, matching the general pattern from
:doc:`00_introduction`:

.. code-block:: python

   # 1. Direct per-pair array getters — no full Mesh object materialised
   nodes = sequence.nodes(0)                    # (N, 2)
   u     = sequence.displacements(0)[:, 0]       # x-displacements
   zncc  = sequence.c_zncc(0)
   node5 = sequence.nodes(0, subset_index=5)     # a single row, (2,)

   # 2. A full MeshWrapper for one pair — same object type/API as 03_mesh
   m0 = sequence.mesh_solution_at(0)             # loads just pair 0
   # sequence.mesh_solutions loads every pair at once — prefer mesh_solution_at()
   # for a single pair, especially on a large sequence.

Plotting
~~~~~~~~~~

Same two conventions as :doc:`03_mesh` — free function or method, and ``ax=``/``show=``
composition:

.. code-block:: python

   sequence.inspect(mesh_idx=0, subset_idx=sequence.seed_node(0))
   sequence.convergence(mesh_idx=0, quantity="C_ZNCC")   # one pair's histogram
   sequence.convergence(quantity="C_ZNCC")               # all pairs pooled together
   sequence.contour(quantity="R", mesh_idx=3)             # displacement magnitude, last pair

Leapfrog and sequential increments, and rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each pair's displacements are measured from that pair's reference image — frame 0
throughout, unless the reference leapfrogged forward. That is often not what you
want to look at: in a progressive test you usually want *what happened between
this frame and the last one*. ``mode`` chooses, for ``u``, ``v`` and ``R``:

- ``mode="leapfrog"`` (default) — from the pair's own reference image, i.e. the
  displacements as solved. The time step is ``g_time - f_time``.
- ``mode="sequential"`` — from the previous target image, whatever the reference
  was. While the reference is unchanged this is ``u(pair m) − u(pair m−1)`` at
  each node (interpolated onto pair *m*'s nodes if the meshes differ); where the
  reference has just moved to the previous frame, the pair's own displacement
  already is that increment. The time step is the time between the two target
  images.

So a leapfrog-solved sequence can still be plotted sequentially, and vice versa.
``rate=True`` divides by the matching time step, giving a velocity in px/s:

.. code-block:: python

   sequence.contour(quantity="R", mesh_idx=3, mode="sequential")              # last increment only
   sequence.contour(quantity="u", mesh_idx=3, mode="sequential", rate=True)   # its velocity, px/s
   sequence.contour(quantity="u", mesh_idx=3, rate=True)                      # mean velocity since frame 0

   # The same numbers without plotting: an (N, 2) increment and its time step
   du, dt = sequence.increment_displacements(3, mode="sequential")

Rates need the sequence's times (see *Image timing* above); without them
``rate=True`` raises an error saying how to supply them. Strain rates come from a
``Field`` — see :doc:`06_field`.

Save and reload
~~~~~~~~~~~~~~~~~

.. code-block:: python

   sequence.save("data/sequence.pyv")
   sequence_loaded = gp.load("data/sequence.pyv")

Adaptive masking
------------------

``sequence.solve(..., solver_options={"masking": "adaptive"})`` applies adaptive
masking (:doc:`03_mesh`) to every pair. The **first two images are taken as a still
pair**: recording should start before anything moves (small rigid camera movement
is fine). That first pair is solved plainly and only supplies the camera noise; every
later pair is screened against it, including after a reference update. Each pair's
``mesh_solution_at(i).seams`` holds its record, and ``sequence.contour(...)`` /
``sequence.inspect(...)`` take the same adaptive quantities and options as a
``Mesh``. In the GUI, **Masking: Adaptive** in the Solver section does the same.

Key takeaways
--------------

- A ``Sequence`` is a series of independently-generated ``Mesh``\ es, not one mesh
  reused frame after frame — that reuse is exactly what causes the distortion
  demonstrated above.
- The default reference strategy is a fixed frame-0 reference that automatically
  falls back ("leapfrogs") to the previous frame only when direct correlation
  against frame 0 fails — not every-frame advancement (``sequential=True``) and
  not a rigid fixed reference either.
- ``sync=True`` reuses mesh geometry between pairs when the reference hasn't
  changed, and resets automatically when it has.
- Two ways to reach a pair's data: direct array getters (``sequence.nodes(i, ...)``),
  or a full ``mesh_solution_at(i)`` when you want the whole ``Mesh``-like API.
- ``time`` (a time step, one time per image, or timestamps read from the file
  names) lets results be plotted as rates; ``mode="leapfrog"``/``"sequential"``
  chooses whether a pair's displacement is measured from its reference image or
  from the previous frame.

**Next:** :doc:`05_particle` — tracking material points through time with
``Particle``.
