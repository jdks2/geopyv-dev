03 — Mesh: full-field DIC
============================

A ``Subset`` (:doc:`02_subset`) gives you displacement at one point. A ``Mesh``
covers a whole region of interest at once: a triangulated network of nodes, each
one a ``Subset`` centre, correlated together rather than independently. Two things
make that worthwhile rather than just "many subsets in a loop": **reliability-guided
propagation** (each new node gets a warm-start initial guess built from its
already-solved neighbours, not a cold zero warp) and **a basis for strain** (the
triangulation lets ``Mesh`` compute element-wise strain directly, and lets
:doc:`05_particle` interpolate a continuous displacement field *between* nodes).

Setup
------

Using the canonical shear series introduced in :doc:`01_images_and_masks`:

.. code-block:: python

   import numpy as np
   import geopyv_dev as gp

   f_img = gp.Image(filepath="images/shear/shear_0.jpg")
   g_img = gp.Image(filepath="images/shear/shear_4.jpg")

   boundary_nodes = np.array([
       [300.0, 300.0], [300.0, 700.0], [700.0, 700.0], [700.0, 300.0]
   ])
   boundary = gp.PathRegion(nodes=boundary_nodes, hard=False)

   excl_nodes = np.array([
       [480.0, 480.0], [480.0, 520.0], [520.0, 520.0], [520.0, 480.0]
   ])
   exclusion = gp.PathRegion(nodes=excl_nodes, hard=True)

   local_mask = gp.Mask(mask_type="local", shape="circle", size=25)
   seed_coord = [500.0, 500.0]

How a mesh is generated
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Mesh`` construction does two independent things with ``boundary``/``exclusions``,
immediately and before any correlation happens:

1. **Triangulation.** The boundary polygon and every exclusion polygon are combined
   into one constrained-Delaunay-triangulation (CDT) problem — solved by ``spade`` —
   producing the node coordinates and element connectivity. ``target_nodes`` drives a
   binary search over element edge length (within the ``size`` bounds) until the
   triangulation lands close to the requested node count.
2. **A binary pixel mask**, the size of the whole image, built from the same
   boundary + exclusion polygons: **1 everywhere**, then **0 outside the boundary**
   if ``boundary_hard`` is ``True``, then **0 inside each exclusion** whose own
   ``exclusions_hard`` flag is ``True``. This is applied, alongside the ``local_mask``
   template, to every node's subset during solving — it isn't exposed to Python
   directly.

.. important::
   This pixel mask is built and used **automatically and internally** by ``Mesh`` —
   unlike ``Subset`` (:doc:`01_images_and_masks`), you never construct a
   ``Mask(mask_type="global", ...)`` yourself and hand it to ``Mesh``. If you want to
   inspect the equivalent mask outside of a mesh solve, build one standalone with
   the same boundary/exclusions, as shown in :doc:`01_images_and_masks`.

.. note::
   **Hard vs soft, and a gotcha for exclusions.** ``boundary_hard`` is read from the
   boundary ``Region`` object's own ``hard`` attribute. Exclusions are different:
   their hard/soft flag comes **only** from the separate ``exclusions_hard`` list
   argument (default: all ``True``) — whatever ``hard=`` you set on the exclusion
   ``Region`` object itself is ignored by ``Mesh``. A *soft* exclusion still removes
   that area from the triangulation (no nodes/elements placed inside it), but its
   pixels remain correlatable if a neighbouring subset's template happens to overlap
   the hole; a *hard* exclusion additionally zeros those pixels out of the mask.

Constructing a ``Mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mesh = gp.Mesh(
       boundary      = boundary,          # PathRegion, CircleRegion, or (N,2) array
       target_nodes  = 80,                # approximate node count (binary-searched)
       f_img         = f_img,             # reference Image
       g_img         = g_img,             # target Image
       size          = (15.0, 70.0),      # (min, max) element edge length, px
       exclusions    = [exclusion],       # optional holes in the mesh
       exclusions_hard = [True],          # per-exclusion hard flag (default: all True)
       mesh_order    = 2,                 # 1 = linear triangles; 2 = quadratic
       time          = 0.4,               # optional: seconds between the two images
   )
   print(f"{mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} elements")

The geometry (``mesh.nodes``, ``mesh.elements``, ``mesh.boundary``,
``mesh.exclusions``) is available immediately — no correlation has happened yet.

Image timing
~~~~~~~~~~~~~~

``time`` tells the mesh when the two images were taken, so that results can be
plotted as rates (velocities) as well as displacements. It takes one of:

- **A number** — the time step between the two images, in seconds. ``time=0.4``
  is the same as ``time=(0.0, 0.4)``.
- **Two numbers** — the time of each image, ``(f_time, g_time)``, on any clock.
- **Nothing** (``None``, the default) — the time is read from the image file
  names, if both carry a timestamp. :doc:`04_sequence` lists the formats
  recognised. ``shear_0.jpg`` and ``shear_4.jpg`` don't carry one (a bare number
  is read as a frame counter, never a time), which is why ``time`` is given above.

.. code-block:: python

   print(mesh.time)          # (0.0, 0.4)  — (f_time, g_time), seconds
   print(mesh.dt)            # 0.4         — g_time - f_time
   print(mesh.time_source)   # "given", "filename (<format>)" or "none"

The time is stored with the solution, so it survives ``save``/``load``. Without
it, everything else works as before; only the rate plots below are unavailable.

Linear vs quadratic elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - ``mesh_order``
     - Nodes per element
     - Warp params per node
   * - 1 (default)
     - 3 (vertices)
     - 6 (affine)
   * - 2
     - 6 (vertices + mid-sides)
     - 12 (quadratic)

Solve
------

The solver does **not** correlate all nodes independently or in a simple sweep:

1. **Seed node** — the mesh node nearest ``seed_coord`` is solved first, with a
   caller-supplied ``seed_warp`` and the stricter ``seed_tolerance``.
2. **Seed's neighbours** are solved next, each preconditioned from its own trusted
   neighbourhood, and pushed onto a max-heap keyed by C_ZNCC.
3. **Main loop:** pop the *best-scoring* unpropagated node from the heap (not FIFO —
   propagation always expands from wherever the solver is currently most confident),
   solve its still-unsolved neighbours, and push those onto the heap. Each neighbour's
   initial guess comes from a local least-squares affine (or quadratic, for
   ``mesh_order=2``) field fit over its ≥3 trusted neighbours, or a Taylor projection
   from its single best neighbour if it has only 1–2. If that first attempt fails
   the ``tolerance``, one retry is made from a zero warp, keeping whichever result
   scored higher.
4. Repeat until the heap is empty — every reachable node has an entry.

.. image:: /_static/img/mesh_solve_flow.png
   :alt: Flow diagram of Mesh::solve. Seed node solve, then seed neighbours solve,
         feeding a loop that pops the best-C_ZNCC unpropagated node from a max-heap
         and solves its unsolved neighbours (least-squares field fit from trusted
         neighbours, or Taylor projection, with a zero-warp retry on failure),
         pushing newly solved neighbours back onto the heap until it empties. Then
         a corrections pass, a quality gate that can fail the mesh, a compatibility
         (Jacobian) check that can also fail it, and finally the assembled
         MeshSolution.
   :width: 560px
   :align: center

.. code-block:: python

   mesh.solve(
       local_mask     = local_mask,
       seed_coord     = seed_coord,
       seed_warp      = [0.0] * 12,   # length 6*mesh_order
       subset_order   = 2,
       tolerance      = 0.75,
       seed_tolerance = 0.9,
       method         = "icgn",
   )
   print(f"solved      : {mesh.solved}")
   print(f"mean C_ZNCC : {mesh.c_zncc.mean():.4f}")
   print(f"min  C_ZNCC : {mesh.c_zncc.min():.4f}")
   print(f"seed node   : {mesh.seed_node}")

The corrections pass
~~~~~~~~~~~~~~~~~~~~~~~~

Once the propagation queue empties, ``Mesh`` runs one more pass over the whole
mesh, in two lanes, *before* deciding whether the solve as a whole succeeded:

- **Lane 1 — low-correlation outliers.** Nodes whose C_ZNCC is an IQR-low outlier
  are re-solved (worst first); the new result only replaces the propagated one if
  it's both a strict improvement *and* clears quality — otherwise the original
  propagation result is kept.
- **Lane 2 — spatial outliers.** Nodes flagged either by anomalous displacement
  "flow" (relative to their neighbourhood) or by an element-local displacement
  magnitude that's an IQR-high outlier are re-solved and their result is **always**
  overwritten — being flagged already means the propagated result was a known
  spatial discontinuity.

After corrections, two checks can still fail the whole mesh (raise an error rather
than return a partial result): every node must be within tolerance (unless the
caller has set ``override_active``, which only ``Sequence`` does internally — see
:doc:`04_sequence`), and every element's deformation-gradient Jacobian determinant
must stay positive (no folded-over elements).

Results
--------

``Mesh`` plotting follows the two conventions from :doc:`00_introduction`'s
*General approaches*: every method is also a free function in ``geopyv_dev.plots``,
and every plot accepts ``ax=``/``show=`` for composing into a larger figure. There
isn't a way to pull a live, independent ``Subset`` back out of a solved ``Mesh`` —
per-node convergence history doesn't survive into ``MeshSolution`` — but every
per-node array is there, both by direct indexing and located visually on the mesh:

.. code-block:: python

   import matplotlib.pyplot as plt

   i = mesh.seed_node

   # Direct array indexing — fast, no plotting
   print(mesh.nodes[i], mesh.displacements[i], mesh.c_zncc[i], mesh.p[i])
   print(mesh.iterations[i], mesh.norms[i])

   # The same data, located visually on the mesh
   mesh.inspect(subset_idx=i)

   # contour(): valid quantity strings are "C_ZNCC", "iterations", "norm", "u", "v", "R"
   mesh.contour("R")                         # displacement magnitude, own figure

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
   mesh.contour("u", ax=ax1, show=False)
   mesh.contour("C_ZNCC", ax=ax2, show=False)
   plt.tight_layout()

   # rate=True plots u, v or R as a velocity: displacement / mesh.dt, in px/s.
   # It needs the mesh's time (see "Image timing" above).
   mesh.contour("R", rate=True)

   # convergence(): a histogram across all nodes (not a per-iteration trace like
   # Subset.convergence() — a Mesh has many nodes, each with its own history)
   mesh.convergence(quantity="C_ZNCC")

Adaptive masking
------------------

A subset fits one warp to the whole disc: a stretch, shear and shift (order 1) or
also a bend (order 2). Where the disc straddles the edge of a shear band, no single
warp fits both sides, and the measurement is smeared across the edge. With
``solver_options={"masking": "adaptive"}``, ``Mesh.solve`` finds those subsets and
corrects them:

1. **Plain solve.** Every node is solved at order 1, exactly as above.
2. **Screen.** Each node's warp predicts where its neighbours should have moved. If
   a neighbour lands further from that prediction than camera noise alone allows
   (and by more than 0.01 px of under-fit), both nodes are flagged. A node then
   stays flagged only if at least a third of its neighbours are flagged, and an
   unflagged node is added if at least two-thirds are, until nothing changes.
3. **Tests.** On each node sent on, three scores ask whether the leftover after the
   fit looks like a bend across the whole disc, or like a kink or step along one
   line.
4. **Correct.** A clear edge is **cut**: the far side of the line is masked (with a
   3 px gap) and the subset re-solved; a second look may cut again (a narrow band)
   or add a bend. A bend is re-solved at **order 2**. A cut subset's result describes
   the side it keeps, so the node and its subset then **move** to the middle of the
   pixels it keeps (as far as the mesh allows without distorting an element; along
   the edge for a node on the boundary or an exclusion; never at a corner) and are
   solved again there, before the second look. The moved disc can reach an edge the
   first one did not, which the second look then cuts.

.. image:: /_static/img/adaptive_neighbour_check.png
   :alt: Two panels. Node A's arrow and B's displacement predicted from A's warp,
         ending in a tolerance circle. Left: B's measured arrow ends inside the
         circle. Right: a band lies between A and B, and B's measured arrow ends
         far outside the circle, so both nodes go on to the tests.
   :width: 640px
   :align: center

.. image:: /_static/img/adaptive_flow.png
   :alt: Flow diagram of adaptive masking for one node: plain solve, screen, tests,
         then cut, move to the middle of the kept pixels and re-solve, order 2, or
         nothing, and a second pass on the moved cut subset.
   :width: 520px
   :align: center

**Where the noise comes from.** The screen's tolerance is set by camera noise, so it
needs a measure of it:

- **A still pair** (recommended): the reference image and one taken just before
  anything moved. Small rigid camera movement between them is fine. Solve that pair
  plainly and hand it to the moving mesh:

  .. code-block:: python

     still = gp.Mesh(boundary, target_nodes=8000, f_img=f_img, g_img=gp.Image(filepath="still.png"))
     still.solve(local_mask, seed_coord)

     mesh = gp.Mesh(boundary, target_nodes=8000, f_img=f_img, g_img=g_img, noise=still)
     mesh.solve(local_mask, seed_coord, solver_options={"masking": "adaptive"})

- **No still pair:** the noise is estimated from a small patch around the seed point,
  which is assumed to move only rigidly. Choose the seed accordingly.

A ``Sequence`` always uses its first two images as the still pair (see
:doc:`04_sequence`).

.. note::
   Adaptive masking needs **lossless images** (PNG, TIFF or BMP): JPEG compression
   changes between two images of moving texture and looks like under-fit. It solves
   at ``subset_order=1`` and applies order 2 node by node; asking for
   ``subset_order=2`` gives a warning and uses 1. ``mesh.p`` then has 12 columns,
   with zeros in the bend terms wherever a node stayed at order 1.

**What it did.** ``mesh.seams`` holds the record: the plain warps and node
positions, which nodes were flagged and sent on, each node's action (``"nothing"``,
``"order2"``, ``"cut"``, ``"cutorder2"``, ``"cut2"``) and its scores. The plots:

.. code-block:: python

   mesh.inspect(seams="actions", sent=True)  # what was done where, cut lines, moves
   mesh.inspect(seams="scores")              # edge vs bending score, alarm and margin lines
   mesh.inspect(subset_idx=i)                # one node: its cut line(s) and masked side
   mesh.contour("noise")                     # each node's noise level (px)
   mesh.contour("still_mismatch")            # frame 0-1 neighbour mismatch (px; still pair only)
   mesh.contour("correction")                # how far the correction moved u (px)

``still_mismatch`` is the check that the still frame really was still: it should
look like noise everywhere.

Building, solving, and inspecting a mesh in the GUI
--------------------------------------------------------

.. note::
   A short screen recording of this workflow will be added here.

1. Switch to the **Meshes** tab and click **New**.
2. In **Timing**, enter the time between the two images as either **Δt (s)** or
   **Rate (Hz)** — editing one fills in the other. Leave both blank to read the
   times from the image file names; the line underneath says which timestamp
   format was found, if any.
3. In **Mesh Generation**, set target nodes, element size bounds, and mesh order.
4. In **Solver**, choose ICGN/FAGN, subset order, and the tolerance/iteration limits.
   **Masking: Adaptive** turns on adaptive masking (the order is then fixed at 1);
   **Noise** picks a still frame from the image list, or leaves the noise to be
   estimated near the seed.
5. In **Define Region**, pick a drawing mode (Rectangular / Circular / Free), then
   click **Boundary ▶** and draw the region of interest on the image viewer;
   optionally click **Exclusion ▶** to add holes, and **Seed ▶** to place the seed
   node. Status indicators confirm boundary/exclusions/seed are set.
6. Click **Run** once the mesh is fully specified — it solves in a background
   thread so the UI stays responsive.
7. Once solved, switch to **View** mode: the metadata row shows node count, mesh
   order, and seed node; the **Displacements** / **Quality** buttons and their
   radios switch the overlay — the same information as ``mesh.contour(...)``
   above. Under **Displacements**, **Rate (velocity, px/s)** divides by the
   pair's Δt (shown underneath); it is greyed out when the mesh has no times.
   After an adaptive solve, the metadata adds the noise source and how many nodes
   were sent on, cut, bent and moved; **Quality** adds **Noise**, **Frame 0–1** and
   **Correction**, and **Show adaptive actions and cuts** draws each node's action
   and cut lines over the plot.

**Next:** :doc:`04_sequence` — extending to a full image sequence with ``Sequence``.
