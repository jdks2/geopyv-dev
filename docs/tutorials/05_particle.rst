05 — Particle: strain-path tracking
======================================

A solved ``Sequence`` (:doc:`04_sequence`) gives you displacements at fixed mesh
nodes, for each pair, relative to whatever reference image that particular pair
used. That last part is the catch: :doc:`04_sequence` showed that the reference
image can change mid-sequence (``sequential=True``, or the automatic fallback on a
failed pair) — which is exactly what large-deformation correlation needs, but it
means the raw per-pair displacements aren't directly stackable into one continuous
strain history. ``Particle`` is the object that does that stacking properly: it
interpolates displacement from the mesh at a point (not necessarily a node) and
integrates it across every pair, tracking whichever reference was actually used
each time.

This tutorial covers: Lagrangian vs Eulerian tracking, ``Particle`` setup and
solving, and extracting/plotting strain-path results.

Setup
------

Continuing from the sequence solved in :doc:`04_sequence`:

.. code-block:: python

   import geopyv_dev as gp

   sequence = gp.load("data/sequence.pyv")

Lagrangian vs Eulerian tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to interpret "displacement at a point" across several frames,
selected via the ``track`` constructor argument below:

**Lagrangian (``track=True`` — default)**

The particle **moves with the material**. Its coordinate updates after each
increment using the interpolated displacement at its *current* position::

   frame 0:  x₀ = [500, 500]
   frame 1:  x₁ = x₀ + u(x₀)   ←  displacement evaluated at x₀
   frame 2:  x₂ = x₁ + u(x₁)   ←  displacement evaluated at the new position x₁

**Eulerian (``track=False``)**

The particle **stays fixed in space**; the displacement at its original position
is evaluated fresh every frame::

   frame 0:  x₀ = [500, 500]
   frame 1:  u(x₀)
   frame 2:  u(x₀)   ←  always the same reference point

Constructing a ``Particle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   particle = gp.Particle(
       source     = sequence,        # a solved Sequence or Mesh
       coordinate = [500.0, 500.0],  # initial position [x, y] — need not be a mesh node
       track      = True,            # True = Lagrangian, False = Eulerian
   )

   particle_eul = gp.Particle(source=sequence, coordinate=[500.0, 500.0], track=False)

Solve
------

.. code-block:: python

   particle.solve(
       factor    = 0.0,    # volumetric correction on the reported normal strains — see below
       true_incs = True,   # linearise the first two strain increments — see below
   )
   particle_eul.solve(factor=0.0, true_incs=True)

``factor`` and ``true_incs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both only affect the reported **in-plane normal strains** (``eps_xx``, ``eps_yy``)
— shear and the separately-tracked volumetric strain are untouched:

- ``true_incs`` (default ``True``) linearises *only the first two* strain-increment
  rows from log-strain back toward an engineering-strain increment; every later
  increment is unaffected either way.
- ``factor`` (default ``0.0``) subtracts ``factor × mean(eps_xx_inc, eps_yy_inc)``
  from both normal-strain increments at every step, then re-accumulates the
  cumulative ``strains`` from the corrected increments. ``factor=0`` (default)
  leaves increments unchanged; ``factor=1`` removes the mean volumetric increment
  from both normal components entirely (a fully-deviatoric normal-strain report).
- ``vol_strains`` is tracked completely separately, directly from the
  element-Jacobian-based volume at each increment — ``factor``/``true_incs`` have
  **no effect on it**.

Results
--------

What a solved ``Particle`` gives you
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   print(particle.coordinates.shape)   # (n_frames+1, 2) — trajectory; row 0 = initial position
   print(particle.warps.shape)         # (n_frames+1, 6*mesh_order) — accumulated warp per frame
   print(particle.incs.shape)          # per-increment (not accumulated) warp; row 0 unused
   print(particle.strains.shape)       # (n_frames+1, 6): [eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy]
   print(particle.strain_incs.shape)   # (n_frames, 6) — per-increment strains
   print(particle.vol_strains)         # (n_frames+1,) — from Jacobian-based volume, not `strains`

``strains``' out-of-plane components (``eps_zz``, ``eps_yz``, ``eps_xz``, indices
2–4) are always zero — planar DIC only measures in-plane deformation; ``eps_xx``
(index 0), ``eps_yy`` (index 1), and ``eps_xy`` (index 5) are the ones that carry
real data.

Plotting
~~~~~~~~~~

Same conventions as :doc:`03_mesh`/:doc:`04_sequence` — free function or method,
``ax=``/``show=`` composition:

.. code-block:: python

   particle.inspect()                                    # initial position on the reference image

   # history(): quantity is "warps", "strains", or "vol_strains" — full trajectory
   particle.history(quantity="warps")                     # all components over frames
   particle.history(quantity="strains", components=[0, 1, 5])   # just eps_xx, eps_yy, eps_xy

   # trace(): one component's per-increment value, drawn along the particle's path
   particle.trace(quantity="strains", component=5)         # shear strain along the trajectory

Save and reload
~~~~~~~~~~~~~~~~~

.. code-block:: python

   particle.save("data/particle.pyv")
   particle_loaded = gp.load("data/particle.pyv")

Key takeaways
--------------

- ``Particle`` exists because a ``Sequence``'s reference image can change
  mid-sequence — it's what turns per-pair, possibly-different-reference
  displacements into one continuous, trackable strain path.
- Lagrangian (``track=True``) follows the material point; Eulerian (``track=False``)
  re-samples the same fixed location every frame.
- ``factor``/``true_incs`` only adjust the reported in-plane normal strains
  (``eps_xx``/``eps_yy``); ``vol_strains`` is independent, Jacobian-based.

**Next:** :doc:`06_field` — placing many particles at once for a full spatial
strain picture.
