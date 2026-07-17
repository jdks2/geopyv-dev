00 — Introduction to ``geopyv_dev``
=====================================

.. image:: /_static/gifs/intro_hero.gif
   :alt: Looping animation — a speckle pattern; a subset fixed on a node of the
         real DIC mesh, its template morphing from a square to a circle and
         shrinking; the camera zooming out as the surrounding mesh grows outward
         from that same node along its real triangulation; the mesh deforming
         under shear with a u-displacement contour drawn over the still-visible
         deforming speckle pattern, a red particle already tracing that node's
         path as the contour deflects; the deformation handing off from shear to
         rotation while the mesh fades; a slide across to the particle's strain
         history and a subset-inspect crop; and a fade back to the speckle pattern.
   :width: 100%
   :align: center

What is PIV/DIC?
------------------

In ``geopyv_dev``, **Particle Image Velocimetry (PIV)**/ **Digital Image Correlation (DIC)** recovers 
full-field deformation from images of a deforming surface, by tracking textured patterns. 
It provides contactless strain measurement — no gauges or sensors. Photograph progressive 
surface deformation under external loading across a series of images. Correlate small patches of pixels
(*subsets*) across space (*meshes*) and time (*sequences*) to sample incremental deformation. Sample
through interpolation (*particles* and *fields*) to recreate strain paths. 

Rusty Python
---------------------------------

``geopyv_dev`` is a Python package with a **Rust core**. For performance, the underlying algorithms 
run as compiled Rust (via `PyO3 <https://pyo3.rs>`_ bindings). For accessibility, a Python interface 
facilitates interactive, scriptable workflow. 

Modular by design
--------------------

As an Object-Oriented Program (OOP), ``geopyv_dev`` consists of hierarchical module objects:

Deformation measurement:
- **A subset** samples deformation at a single point for a single image pair.
- **A mesh** of subsets, samples a deformation across a region of interest for a single image pair.
- **A sequence** of meshes, samples incremental deformation across time.

Strain path generation: 
- **A particle** interpolates strain from a mesh or sequence to generate the strain path for a single point.
- **A field** of particles interpolates strain for a region. 

In other words, each level builds on the one below it. 

A geotechnical focus
-----------------------

``geopyv_dev`` was developed for geotechnical physical modelling — the analysis of progressive 
plane-strain deformation image data. **PIV/DIC** has been used in geotechnics for many purposes, 
from qualitatively describing a failure mechanism to quantitative work calculations. ``geopyv_dev`` is
equipped to this end, integrating a variety of techniques to maximise measurement accuracy across large
deformation problems. 

Tutorial map
------------

.. list-table::
   :header-rows: 1

   * - Tutorial
     - Topic
   * - **00 — you are here**
     - Geotechnical PIV/DIC Package overview
   * - **01**
     - ``Image``, ``Mask``, ``CircleRegion``, ``PathRegion``
   * - **02**
     - ``Subset`` — single-point correlation
   * - **03**
     - ``Mesh`` — full-field single image pair
   * - **04**
     - ``Sequence`` — multi-image time series
   * - **05**
     - ``Particle`` — single-point strain path tracking
   * - **06** 
     - ``Field`` — full-field strain-path tracking
   * - **07**
     - ``Validation`` — proving accuracy
   * - **06** 
     - ``Calibration`` — from image to reality

.. note::
   **Setup:** The following tutorial pages have scripts you can run locally.
   See :doc:`../installation` for full setup instructions (pip or from
   source, Windows/Linux).

General approaches
--------------------

Before we go through, it is worth mentioning a few conventions that hold across the whole package.

**Save and reload**

Every solved object serialises to a ``.pyv`` file and loads back in one call:

.. code-block:: python

   mesh.save("data/mesh.pyv") # Of course, you need to solve a mesh first!
   mesh = gp.load("data/mesh.pyv")

   sequence.save("data/sequence.pyv")
   sequence = gp.load("data/sequence.pyv")

``.pyv`` is a compact binary format (bincode); it is not compatible with the
pickle-based ``.pyv`` files from the original Python ``geopyv`` package. The loaded
object is returned as the appropriate wrapper type, ready to use immediately.

**Plots: supply an axes or let the function create one**

Every plot function accepts an optional ``ax=`` argument (``axes=`` for multi-panel
functions). If you omit it, the function creates its own figure and
returns ``(fig, ax)``. Pass an existing axes to embed the plot inside a larger
figure you are building or adapt the plot formatting. Also, every plot function accepts standard ``kwargs``
for ``matplotlib``. 

.. code-block:: python

   import matplotlib.pyplot as plt
   import geopyv_dev as gp

   mesh = gp.load() # Point to a mesh you've run! 

   fig, (ax1, ax2) = plt.subplots(1, 2)
   mesh.contour(
       quantity = "u",
       ax       = ax1, 
       show     = False
   )
   mesh.contour(
       quantity = "c_zncc",
       levels   = np.linspace(0.75,1.0, 21), # A kwarg!
       ax       = ax2, 
       show     = False
   )
   plt.tight_layout()
   plt.savefig("summary.png", dpi=300)

   mesh.contour(
       quantity = "R", 
       show     = False,
       save     = "example_mesh.png"
   )

``show=True`` (default) calls ``plt.show()`` at the end; set ``show=False`` when
embedding. ``save="path.png"`` writes to disk at 600 dpi before showing. We'll return to plotting for
each object. 

**Hierarchy access: free function or object method**

Plots are available as free functions in ``geopyv_dev.plots`` and as convenience
methods on the wrapper objects. The method just delegates to the free function —
choose whichever form reads more naturally:

.. code-block:: python

   # Free function
   gp.plots.contour_sequence(sequence, mesh_idx=2, quantity="displacements", component=0)

   # Equivalent method
   sequence.contour(quantity="displacements", mesh_idx=2, component=0)

   # Or another...
   sequence.meshes[i](quantity="displacements", component=0)

The same pattern applies throughout: ``mesh.contour(...)``, ``mesh.convergence(...)``,
``field.trace(...)``, ``particle.history(...)``, and so on.

Next up: :doc:`01_images_and_masks` — loading images and building masks.
