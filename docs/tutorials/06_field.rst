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
   * - ``field.time``
     - ``(inc_no,)`` or ``None``
     - Time of each row's image, seconds — from the sequence's ``time``
       (:doc:`04_sequence`)

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

**Spatial contours: total, incremental, and rates**

``quantity`` is one of ``"u"``, ``"v"``, ``"R"`` (displacement magnitude),
``"ep_xx"``, ``"ep_yy"``, ``"ep_xy"``, ``"ep_vol"``, ``"ep1"``, ``"ep2"``,
``"gamma_max"``, ``"theta_p"`` (principal strains, maximum shear strain and the
principal angle) or ``"gamma_max_grad"`` (the spatial gradient of
``gamma_max``). Which rows of each particle's history are used is set by
``window`` or ``increment``:

.. code-block:: python

   import geopyv_dev.plots as gpp

   # Total: the change over the whole test (the default)
   gpp.contour_field(field, quantity="ep_vol", cmap="RdBu_r")

   # The value at one row (increment index; -1 = last)
   gpp.contour_field(field, quantity="gamma_max", window=-1)

   # The change over a range of rows (half-open, like a slice)
   gpp.contour_field(field, quantity="gamma_max", window=(1, 3))

   # Incremental: the change into row 3 ...
   gpp.contour_field(field, quantity="gamma_max", increment=3)                      # ... since its reference image
   gpp.contour_field(field, quantity="gamma_max", increment=3, mode="sequential")   # ... since the previous image

``mode`` has the same meaning as for a sequence (:doc:`04_sequence`):
``"leapfrog"`` (default) measures the increment from the reference image of the
pair that produced row 3; ``"sequential"`` from row 2, the previous image. The two
agree whenever the reference image was advanced every frame. ``absolute=True``
sums the size of every row-to-row change instead of taking last minus first.

``rate=True`` divides by the time elapsed between the first and last rows used,
taken from ``field.time`` — so spacing between images can be irregular:

.. code-block:: python

   gpp.contour_field(field, quantity="ep_xx", rate=True)                    # mean strain rate over the test, /s
   gpp.contour_field(field, quantity="gamma_max", increment=3,
                     mode="sequential", rate=True)                          # strain rate over one increment, /s
   gpp.contour_field(field, quantity="R", increment=3, rate=True)           # speed, px/s

Two quantities change meaning under ``rate=True``:

- ``ep1``, ``ep2``, ``gamma_max`` and ``theta_p`` become the principal values,
  maximum shear rate and principal direction of the **strain-rate tensor**
  (the change in strain divided by the time step) — not the rate of change of
  the principal strains, which differs whenever the principal directions rotate,
  as they do in a developing shear band. ``theta_p`` stays an angle.
- ``R`` becomes the speed, the size of the change in displacement per second,
  rather than the rate of change of the displacement's magnitude.

Rates need image times (:doc:`04_sequence`); without them ``rate=True`` raises an
error saying how to supply them. The colour bar shows the units: ``/s`` for
strain rates, ``px/s`` for velocities.

**Strain path for one particle**

.. code-block:: python

   # u-displacement against time (image number if the sequence has no times), particle 5
   gpp.history_field(field, particle_index=5, quantity="warps", components=[0])

   # Its strain rates over each increment, from the previous image (drawn as steps)
   gpp.history_field(field, particle_index=5, quantity="strains", rate=True, mode="sequential")

**Spatial traces**

.. code-block:: python

   # All particle paths, each segment coloured by the change in u since its reference image
   gpp.trace_field(field, quantity="warps", component=0)

   # ... since the previous image, per second
   gpp.trace_field(field, quantity="warps", component=0, mode="sequential", rate=True)

   # ... or by the time each segment ends at
   gpp.trace_field(field, mode="time")

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
- ``window`` gives totals over a range of rows, ``increment`` the change into one
  row (``mode="leapfrog"`` from its reference image, ``"sequential"`` from the
  previous image), and ``rate=True`` turns either into a rate using the stored
  image times — with principal quantities taken from the strain-rate tensor.

**Next:** :doc:`07_validation` — checking results against known synthetic deformations
with ``Validation``.
