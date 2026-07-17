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
   )
   print(f"{mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} elements")

The geometry (``mesh.nodes``, ``mesh.elements``, ``mesh.boundary``,
``mesh.exclusions``) is available immediately — no correlation has happened yet.

Linear vs quadratic elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - ``mesh_order``
     - Nodes per element
     - Warp params per node
   * - 1
     - 3 (vertices)
     - 6 (affine)
   * - 2 (default)
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

   # convergence(): a histogram across all nodes (not a per-iteration trace like
   # Subset.convergence() — a Mesh has many nodes, each with its own history)
   mesh.convergence(quantity="C_ZNCC")

Building, solving, and inspecting a mesh in the GUI
--------------------------------------------------------

.. note::
   A short screen recording of this workflow will be added here.

1. Switch to the **Meshes** tab and click **New**.
2. In **Mesh Generation**, set target nodes, element size bounds, and mesh order.
3. In **Solver**, choose ICGN/FAGN, subset order, and the tolerance/iteration limits.
4. In **Define Region**, pick a drawing mode (Rectangular / Circular / Free), then
   click **Boundary ▶** and draw the region of interest on the image viewer;
   optionally click **Exclusion ▶** to add holes, and **Seed ▶** to place the seed
   node. Status indicators confirm boundary/exclusions/seed are set.
5. Click **Run** once the mesh is fully specified — it solves in a background
   thread so the UI stays responsive.
6. Once solved, switch to **View** mode: the metadata row shows node count, mesh
   order, and seed node; **Displacements** / **Strains** / **Quality** buttons (plus
   **Iterations** / **Norms** radios) switch the overlay — the same information as
   ``mesh.contour(...)`` above.

**Next:** :doc:`04_sequence` — extending to a full image sequence with ``Sequence``.
