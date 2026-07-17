04 — Sequence: multi-image time series
=========================================

A ``Mesh`` (:doc:`03_mesh`) solves one image pair. A ``Sequence`` is a temporal
series of meshes: it walks through every frame of an image directory, solving
each successive pair and carrying geometry and warp information forward between
them — the natural extension when you want to track *progressive* deformation
rather than a single before/after snapshot.

This tutorial covers: reference-image strategy, ``SequenceOptions``, per-pair
solving, and extracting/plotting results across a sequence.

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
   )

The geometry for the first pair is generated immediately, the same as a standalone
``Mesh`` (:doc:`03_mesh`) — no correlation has happened yet.

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

Save and reload
~~~~~~~~~~~~~~~~~

.. code-block:: python

   sequence.save("data/sequence.pyv")
   sequence_loaded = gp.load("data/sequence.pyv")

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

**Next:** :doc:`05_particle` — tracking material points through time with
``Particle``.
