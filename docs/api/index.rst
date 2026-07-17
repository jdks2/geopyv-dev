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
