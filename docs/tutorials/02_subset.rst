02 — Subset: single-point correlation
========================================

A ``Subset`` is a small patch of pixels centred on a point in the
reference image providing a sample of deformation. Defined by a coordinate
and a local mask (or template), the PIV/DIC solver finds the **warp parameters**
that best deform this patch to match the same region in the target image,
spotting the difference between the two.

.. image:: /_static/img/c2_subset.png
   :alt: A 2x3 grid of subset template crops from a speckle image — top row
         circular templates, bottom row square templates, each column a
         larger size than the last.
   :width: 480px
   :align: center

Setup
------

A ``Subset`` is defined by four things:

.. list-table::
   :header-rows: 1

   * - Input
     - Meaning
   * - Coordinate
     - The ``[x, y]`` centre of the subset in the reference image
   * - Local mask (template)
     - A ``Mask(mask_type="local", ...)`` object defining the patch shape —
       ``"circle"`` or ``"square"`` (see :doc:`01_images_and_masks`)
   * - Size
     - The patch's radius (circle) or half-width (square), set on the
       ``local_mask`` itself
   * - Order
     - ``subset_order`` — 1 (affine) or 2 (quadratic); how complex a warp the
       solver is allowed to fit (see *The warp function*, below)

.. image:: /_static/gifs/subset_shape_order.gif
   :alt: A circle and a square subset template growing and shrinking, then fixed
         in place and wiggling to show 0th-order (rigid), 1st-order (affine —
         ellipse/parallelogram), and 2nd-order (curved, banana-bend) deformation.
   :width: 480px
   :align: center

Using the shear series generated in :doc:`01_images_and_masks`:

.. code-block:: python

   import numpy as np
   import geopyv_dev as gp

   f_img = gp.Image(filepath="images/shear/shear_0.jpg")
   g_img = gp.Image(filepath="images/shear/shear_4.jpg")

   local_mask = gp.Mask(mask_type="local", shape="circle", size=25)
   coord = [500.0, 500.0]

Nothing has been solved yet — Setup only gathers the ingredients above; a
``Subset`` isn't actually created until *Solve*, below.

Image quality: is this a good place to put a subset?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before solving, it's worth checking whether a subset is actually sitting on
usable texture — a patch dropped over a flat, low-contrast region has nothing
for the solver to lock onto and will fail (or worse, converge confidently to
the wrong answer).

.. image:: /_static/img/subset_quality.png
   :alt: A speckle field whose point density falls off from the top-left
         corner. A subset in that dense corner (good) and one far from it
         (poor) are marked; on either side, their real inspect() renders show
         the good one full of distinct speckles with a high sigma and SSSIG,
         and the poor one almost blank with both near zero.
   :width: 720px
   :align: center

Two metrics are available on every ``Subset``, before solving:

.. list-table::
   :header-rows: 1

   * - Attribute
     - Meaning
     - Guidance
   * - ``sigma_intensity``
     - Standard deviation of pixel intensity within the template
     - Healthy texture is typically **> 15–20**; near 0 means a flat patch
   * - ``sssig``
     - Sum of squared intensity gradients (SSSIG)
     - Healthy texture is typically **> 1e5**; low values mean weak/no gradient
       for the solver to key on

.. code-block:: python

   import geopyv_dev as gp

   # quality_speckle.jpg: speckle density concentrated toward one corner,
   # so one region is genuinely well-textured and another genuinely isn't —
   # see docs/scripts/build_subset_assets.py for how it's generated.
   f_quality = gp.Image(filepath="images/quality/quality_speckle.jpg")
   mask = gp.Mask(mask_type="local", shape="circle", size=45)

   good_sub = gp.Subset(coord=[250.0, 250.0], local_mask=mask, f_img=f_quality, g_img=f_quality)
   poor_sub = gp.Subset(coord=[860.0, 860.0], local_mask=mask, f_img=f_quality, g_img=f_quality)

   for name, sub in [("good", good_sub), ("poor", poor_sub)]:
       print(f"{name}: sigma_intensity={sub.sigma_intensity:.2f}  sssig={sub.sssig:.2e}")
       sub.inspect()   # shows the template crop with these values annotated

Both metrics are cheap to compute from the reference image alone — check them
before committing to a solve, especially for subsets placed automatically (e.g.
by a ``Mesh``) rather than hand-picked.

The warp function
-------------------

The DIC solver describes how a subset deforms using a **warp vector** ``p``. For
**order 1** (affine, 6 parameters)::

   u(x,y) = p[0] + p[2]·Δx + p[4]·Δy
   v(x,y) = p[1] + p[3]·Δx + p[5]·Δy

where Δx = x − x₀, Δy = y − y₀ (offsets from the subset centre x₀, y₀). So
``p[0]`` = u-displacement, ``p[1]`` = v-displacement, ``p[2]`` = du/dx,
``p[3]`` = dv/dx, ``p[4]`` = du/dy, ``p[5]`` = dv/dy — the same component
ordering used by ``Speckle``'s ``comp`` array in :doc:`00_introduction` and
:doc:`01_images_and_masks`.

For **order 2** (quadratic, 12 parameters), six second-order terms
(``p[6..11]``) are added. Most analyses use order 1 for ``Subset`` and order 2
for ``Mesh``/``Sequence``.

Solve
------

``solve()`` takes five inputs, all optional except the choice is yours to make
explicit:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Meaning
     - Default
   * - ``p_0``
     - Initial warp vector to start iterating from
     - ``None`` — zeros matching ``subset_order``
   * - ``algorithm``
     - ``"icgn"`` (Inverse Compositional Gauss-Newton — Hessian computed once,
       typically 5–10x faster) or ``"fagn"`` (Forward Additive Gauss-Newton —
       Hessian rebuilt every iteration, slower but can sometimes recover from a
       poor initial guess ICGN struggles with). Both converge to the same
       answer for well-textured subsets.
     - ``"icgn"``
   * - ``max_norm``
     - Convergence threshold on the warp-update norm ``||Δp||``
     - ``1e-3``
   * - ``max_iterations``
     - Iteration limit before giving up
     - ``50``
   * - ``tolerance``
     - Minimum ``C_ZNCC`` for the result to be marked ``solved=True``
     - ``0.75``

.. code-block:: python

   subset = gp.Subset(
       coord      = coord,          # centre of the subset in the reference image
       local_mask = local_mask,     # a local Mask object defining the patch shape
       f_img      = f_img,          # reference Image
       g_img      = g_img,          # target Image
       subset_order = 1,            # 1 (affine) or 2 (quadratic)
   )
   subset.solve(
       p_0            = None,       # initial warp guess; None -> zeros
       algorithm      = "icgn",     # "icgn" or "fagn"
       max_norm       = 1e-3,       # convergence threshold on ||Δp||
       max_iterations = 50,         # iteration limit
       tolerance      = 0.75,       # minimum C_ZNCC to mark solved=True
   )

   print(f"u      : {subset.p[0]:.4f} px   (x-displacement)")
   print(f"v      : {subset.p[1]:.4f} px   (y-displacement)")
   print(f"C_ZNCC : {subset.c_zncc:.6f}  (1.0 = perfect match)")
   print(f"iters  : {subset.iterations}")
   print(f"converged : {subset.converged}")

``solve()`` is non-destructive to the images — you can create many subsets from the same pair.

Results
--------

Everything above is available on the solved ``subset``:

.. list-table::
   :header-rows: 1

   * - Attribute
     - Meaning
   * - ``p``
     - Solved warp vector (see *The warp function*, above)
   * - ``c_zncc``
     - Zero-Normalised Cross-Correlation coefficient — 1.0 is a perfect match
   * - ``c_znssd``
     - Zero-Normalised Sum of Squared Differences — complementary error metric
   * - ``converged``
     - ``True`` if ``||Δp||`` dropped below ``max_norm`` within ``max_iterations``
       *and* ``c_zncc`` reached ``tolerance``
   * - ``iterations``
     - Number of iterations actually run
   * - ``history``
     - Per-iteration ``(iteration, norm, C_ZNCC, C_ZNSSD)`` tuples — the full
       path the solver took, useful for diagnosing slow or failed convergence
   * - ``max_norm`` / ``tolerance``
     - The thresholds this particular solve was run with (see below)

Why these particular thresholds? ``max_norm=1e-3`` stops iterating once a
further update would move the warp by less than a thousandth of a pixel —
tight enough that no further iteration would change the reported displacement
at any precision that matters, but not so tight that well-behaved solves waste
iterations chasing noise. ``tolerance=0.75`` is a floor on ``C_ZNCC``: below
it, the "match" is more likely a coincidental correlation than a genuine
tracked patch, so the result is flagged ``converged=False`` even if the norm
criterion was satisfied. Both dashed reference lines are drawn directly on the
convergence plot below, precisely because they're the pass/fail lines a real
solve is judged against.

``subset.inspect()`` shows the solved template with its quality metrics
annotated, and ``subset.convergence()`` plots ``history`` against both
thresholds above. Following the axes convention from :doc:`00_introduction`'s
*General approaches*, you can let either create its own figure, or hand it an
axes you built yourself to combine with other plots:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Auto-generated figures
   subset.inspect()
   subset.convergence()

   # Or embedded in a figure you control
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
   subset.inspect(ax=ax1, show=False)
   subset.convergence(ax=ax2, show=False)
   plt.tight_layout()
   plt.savefig("subset_summary.png", dpi=300)

Selecting and solving a subset in the GUI
----------------------------------------------

.. note::
   A short screen recording of this workflow will be added here.

1. Switch to the **Subsets** tab and click **New**.
2. Name the subset, and pick reference/target images from the dropdowns.
3. Choose a template shape (circle/square) and size, and a solver
   configuration (ICGN/FAGN, order 1/2, max norm, max iterations, ZNCC
   tolerance).
4. Click on the reference image in the viewer to place the subset centre.
5. Click **Run** to solve.
6. Once solved, switch between the **Displacement**, **Inspect**, and
   **Convergence** view buttons to read off results — the same information as
   ``subset.p``, ``subset.inspect()``, and ``subset.convergence()`` above.

**Next:** :doc:`03_mesh` — extending to full-field measurements with ``Mesh``.
