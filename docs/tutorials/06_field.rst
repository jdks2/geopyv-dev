06 — Field: spatial arrangement of particles
==============================================

A ``Field`` places many particles simultaneously across the mesh and solves all
of their strain paths in a single call. Where a ``Particle`` (tutorial 05) tracks
**one material point**, a ``Field`` tracks **every element centroid** — covering the
full volume of the specimen at once, giving a continuous spatial picture of how
strain distributes and concentrates rather than just a reading at one point.

This tutorial covers: particle distribution, ``Field`` setup and solving, accessing
per-particle results, spatial contour plots, and save/load.

Setup
------

.. code-block:: python

   import numpy as np
   import geopyv_dev as gp

   # Reload a solved sequence from the previous tutorial
   sequence = gp.load("data/sequence.pyv")

Distributing particles
~~~~~~~~~~~~~~~~~~~~~~~~

By default ``Field`` places one particle at the centroid of each element in the
first mesh of the sequence:

.. code-block:: python

   field = gp.Field(sequence_solution=sequence, track=True, depth=1.0)

To override the distribution, supply explicit ``coordinates`` and ``volumes``:

.. code-block:: python

   m0 = sequence.mesh_solution_at(0)
   coords, volumes = gp.field_distribute_particles(
       nodes    = m0.nodes,
       elements = m0.elements,
       depth    = 1.0,
   )

   field = gp.Field(
       sequence_solution = sequence,
       track             = True,
       depth             = 1.0,
       coordinates       = coords,
       volumes           = volumes,
   )
   print(f"n_particles : {field.n_particles}")

``field_distribute_particles`` returns the element centroids together with the
corresponding element areas scaled by ``depth``.  Particle volumes are used for
volumetric strain tracking; set ``depth`` to the specimen thickness in
physical units once calibration is applied.

Constructor parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``sequence_solution``
     - —
     - Solved ``Sequence`` or ``Mesh``
   * - ``track``
     - ``True``
     - ``True`` = Lagrangian (particles move); ``False`` = Eulerian (fixed in space)
   * - ``depth``
     - 1.0
     - Out-of-plane depth; must be > 0
   * - ``coordinates``
     - ``None``
     - ``(N, 2)`` array of initial particle positions; if omitted, uses element centroids
   * - ``volumes``
     - ``None``
     - ``(N,)`` array of initial particle volumes; required when ``coordinates`` is supplied

Solve
------

.. code-block:: python

   field.solve(
       factor    = 0.0,   # volumetric correction factor
       true_incs = True,  # logarithmic strain increments
   )
   print(f"solved      : {field.solved}")
   print(f"n_particles : {field.n_particles}")
   print(f"inc_no      : {field.inc_no}")

``field.solve()`` processes every mesh increment sequentially, advancing all
particles in parallel (one mesh loaded at a time — memory-efficient for large
sequences).

``factor`` scales a volumetric correction applied during the strain-path integration.
``true_incs=True`` computes logarithmic (true) strain increments; set to ``False``
for engineering strain.

Results
--------

Solved attributes
~~~~~~~~~~~~~~~~~~~

After solving, the following attributes are available:

.. list-table::
   :header-rows: 1

   * - Attribute
     - Shape / type
     - Description
   * - ``field.particles``
     - list of ``ParticleSolution``
     - One entry per particle
   * - ``field.vol_totals``
     - ``(inc_no,)``
     - Sum of all particle volumes at each increment
   * - ``field.reference_update_register``
     - list of int
     - Increment indices where the reference image was advanced
   * - ``field.calibrated``
     - bool
     - Whether calibration was applied during solve
   * - ``field.inc_no``
     - int
     - Number of increments solved (= number of meshes + 1)

Per-particle access
~~~~~~~~~~~~~~~~~~~~~

Each element of ``field.particles`` is a ``ParticleSolution`` with the same
attributes as a solved ``Particle`` (see tutorial 05):

.. code-block:: python

   p0 = field.particles[0]
   print(p0.coordinates)    # (inc_no+1, 2) trajectory
   print(p0.strains[-1])    # final-increment strain components
   print(p0.vol_strains[-1])  # final volumetric strain

Volumetric strain at each increment for all particles:

.. code-block:: python

   vol_strains_final = [p.vol_strains[-1] for p in field.particles]

Plotting
~~~~~~~~~~

The ``geopyv_dev.plots`` module provides three plot types for fields.

**Spatial contour at a given increment**

.. code-block:: python

   import geopyv_dev.plots as gpp

   # Filled contour of volumetric strain at the last increment
   gpp.contour_field(
       field,
       quantity  = "vol_strains",
       window    = -1,             # increment index; -1 = last
       cmap      = "RdBu_r",
   )

``quantity`` accepts any ``ParticleSolution`` attribute that has shape
``(inc_no,)`` or ``(inc_no, n_components)``.  Use ``absolute=True`` to plot the
magnitude rather than the signed value.

**Strain path for one particle**

.. code-block:: python

   # Evolution of warp component 0 (u-displacement) over time for particle 5
   gpp.history_field(field, particle_index=5, quantity="warps", components=[0])

**Spatial traces**

.. code-block:: python

   # Trajectory of all particles (Lagrangian paths overlaid on the reference image)
   gpp.trace_field(field, quantity="warps", component=0)

Save and reload
~~~~~~~~~~~~~~~~~

.. code-block:: python

   field.save("data/field.pyv")
   field_loaded = gp.load("data/field.pyv")

Key takeaways
--------------

- ``Field`` distributes many particles at element centroids by default; supply
  explicit ``coordinates`` and ``volumes`` to override.
- ``depth`` scales particle volumes for out-of-plane thickness; set it to the
  specimen thickness after applying calibration.
- Per-particle results are in ``field.particles``; index ``-1`` gives the final-increment value.
- ``contour_field``, ``history_field``, and ``trace_field`` cover spatial, temporal,
  and trajectory views respectively.

**Next:** :doc:`07_validation` — checking results against known synthetic deformations
with ``Validation``.
