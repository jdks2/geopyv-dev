API Reference
=============

Public classes and functions exported from ``geopyv_dev``. This page covers the
Python-facing surface used throughout the tutorials; low-level geometry and solver
helpers re-exported from the compiled Rust core are considered internal and are not
documented here.

Core objects
------------

Loaded images and the regions used to mask or seed a subset/mesh — see
:doc:`../tutorials/01_images_and_masks`.

.. autosummary::
   :nosignatures:

   geopyv_dev.Image
   geopyv_dev.Mask
   geopyv_dev.CircleRegion
   geopyv_dev.PathRegion

.. autoclass:: geopyv_dev.Image
   :members:

.. autoclass:: geopyv_dev.Mask
   :members:

.. autoclass:: geopyv_dev.CircleRegion
   :members:

.. autoclass:: geopyv_dev.PathRegion
   :members:

Correlation
-----------

Single-point and full-field DIC — see :doc:`../tutorials/02_subset`,
:doc:`../tutorials/03_mesh`, :doc:`../tutorials/04_sequence`.

.. autosummary::
   :nosignatures:

   geopyv_dev.Subset
   geopyv_dev.Mesh
   geopyv_dev.Sequence

.. autoclass:: geopyv_dev.Subset
   :members:

.. autoclass:: geopyv_dev.Mesh
   :members:

.. autoclass:: geopyv_dev.Sequence
   :members:

Adaptive masking results
~~~~~~~~~~~~~~~~~~~~~~~~

After ``solve(..., solver_options={"masking": "adaptive"})`` (see
:doc:`../tutorials/03_mesh`):

- ``Subset.seams`` — dict: ``plain_coord``, ``plain_p``, ``moved`` (total move of
  the centre after its cuts, or ``None``), ``action``, ``lines`` (cut lines: ``kind``, ``theta``, ``c``, ``T``),
  ``gap``, ``passes`` (scores and decision per pass), ``h``, ``levels``, ``flags``.
  ``None`` after a uniform solve.
- ``Mesh.seams`` (and each ``Sequence`` pair's ``mesh_solution_at(i).seams``) —
  dict of per-node arrays: ``plain_nodes``, ``plain_p``, ``flagged``, ``sent``,
  ``noise_scale``, ``still_mismatch`` (``None`` without a still pair), ``action``,
  ``records`` (each node's ``Subset.seams`` record, or ``None``), plus
  ``noise_source``, ``rounds`` and ``h``.
- ``Mesh(..., noise=still_mesh)`` — the solved still pair whose residuals set the
  noise; without it the noise is estimated near the seed.
- Plots: ``Mesh.contour`` quantities ``"noise"``, ``"still_mismatch"``,
  ``"correction"``; ``Mesh.inspect(seams="actions" | "scores", sent=...)``;
  ``Subset.inspect(seams=True)``.

Tracking
--------

Lagrangian/Eulerian strain-path tracking — see :doc:`../tutorials/05_particle`,
:doc:`../tutorials/06_field`.

.. autosummary::
   :nosignatures:

   geopyv_dev.Particle
   geopyv_dev.Field

.. autoclass:: geopyv_dev.Particle
   :members:

.. autoclass:: geopyv_dev.Field
   :members:

Validation & calibration
-------------------------

See :doc:`../tutorials/07_validation`, :doc:`../tutorials/08_calibration`.

.. autosummary::
   :nosignatures:

   geopyv_dev.Validation
   geopyv_dev.Calibration
   geopyv_dev.CalibrationParams

.. autoclass:: geopyv_dev.Validation
   :members:

.. autoclass:: geopyv_dev.Calibration
   :members:

.. autoclass:: geopyv_dev.CalibrationParams
   :members:

Image timing and rates
----------------------

Every ``Subset``, ``Mesh`` and ``Sequence`` takes a ``time`` argument (see the class
docstrings above and :doc:`../tutorials/04_sequence`): a time step in seconds, one
time per image, or ``None`` to read timestamps from the image file names. The
results then carry these attributes and methods (they live on the compiled objects,
so they are listed here rather than under each class):

.. py:attribute:: Subset.time
                  Mesh.time
                  MeshSolution.time
   :noindex:

   ``(f_time, g_time)`` — the times of the pair's reference and target images, in
   seconds, or ``None``.

.. py:attribute:: Mesh.dt
                  MeshSolution.dt
   :noindex:

   ``g_time - f_time``, or ``None``.

.. py:attribute:: Sequence.time
                  Field.time
                  Particle.time
   :noindex:

   The time of each image (``Sequence``) or row (``Field``/``Particle`` — row *k*
   is image *k*), in seconds, as a 1-D array; or ``None``.

.. py:attribute:: Subset.time_source
                  Mesh.time_source
                  Sequence.time_source
   :noindex:

   Where ``time`` came from: ``"given"``, ``"filename (<format>)"`` or ``"none"``.

.. py:method:: Sequence.increment_displacements(mesh_index, mode="leapfrog")
   :noindex:

   Pair ``mesh_index``'s nodal displacement increment ``(N, 2)`` and its time step
   in seconds (``None`` without times). ``mode="leapfrog"`` measures it from the
   pair's reference image; ``mode="sequential"`` from the previous target image.

.. py:method:: Field.contour_values(quantity, window=None, increment=None, mode="leapfrog", rate=False, absolute=False)
   :noindex:

   The per-particle values ``contour_field`` plots, as an ``(N,)`` array — see
   :func:`geopyv_dev.contour_field` for the meaning of each argument.

.. py:method:: Field.contour_range(window=None, increment=None, mode="leapfrog")
   :noindex:

   The half-open ``(start, stop)`` rows those arguments select.

.. py:method:: Particle.series_increments(quantity="warps", component=0, mode="leapfrog", rate=False)
   :noindex:

   The change of ``quantity`` (``"warps"``, ``"strains"`` or ``"vol_strains"``;
   column ``component``) into each row ``k = 1 … inc_no − 1``: from that row's
   reference image (``"leapfrog"``) or the previous image (``"sequential"``); with
   ``rate``, divided by that time step.

The plotting functions below take ``rate=True`` (and, where there is a choice of
increment, ``mode=``) to plot rates; units are ``/s`` for strain rates and ``px/s``
for velocities.

I/O
---

.. autosummary::
   :nosignatures:

   geopyv_dev.save
   geopyv_dev.load

.. autofunction:: geopyv_dev.save

.. autofunction:: geopyv_dev.load

Plotting & inspection
----------------------

Called via the ``.inspect()`` / ``.convergence()`` / ``.contour()`` / ``.history()`` /
``.trace()`` methods on the objects above; listed here individually for reference.

.. autosummary::
   :toctree: generated
   :nosignatures:

   geopyv_dev.inspect_subset
   geopyv_dev.inspect_mesh
   geopyv_dev.inspect_sequence
   geopyv_dev.inspect_particle
   geopyv_dev.inspect_field
   geopyv_dev.convergence_subset
   geopyv_dev.convergence_mesh
   geopyv_dev.convergence_sequence
   geopyv_dev.contour_mesh
   geopyv_dev.contour_sequence
   geopyv_dev.contour_field
   geopyv_dev.history_particle
   geopyv_dev.history_field
   geopyv_dev.trace_particle
   geopyv_dev.trace_field
   geopyv_dev.standard_error_validation
   geopyv_dev.mean_error_validation
   geopyv_dev.noise_standard_error_validation
   geopyv_dev.noise_mean_error_validation
   geopyv_dev.strain_error_validation
   geopyv_dev.spatial_error_validation
