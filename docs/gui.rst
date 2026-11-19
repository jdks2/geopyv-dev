GUI
===

``geopyv_dev`` includes a desktop GUI for the same workflow as the scripting
interface. It is installed with the package — no extra steps:

.. code-block:: bash

   pip install geopyv_dev
   geopyv-gui

Run it from a terminal with the virtual environment active. It needs a display,
so it will not start over a plain SSH session.

Projects
--------

The start screen offers **New Project** and **Open Project** (plus recently
opened projects). A project is a folder containing a ``project.json`` and one
sub-folder per stage:

.. code-block:: text

   my_project/
     project.json
     Images/Data/          test images
     Images/Calibration/   ChArUco board images
     Subsets/
     Meshes/
     Sequences/
     Particles/
     Fields/
     Out/

Results are saved as ``.pyv`` files — the same format as ``gp.save`` — so
anything made in the GUI can be loaded in a script with ``gp.load``, and the
other way round. Calibrations are kept in ``Calibrations/`` (created when an
older project is opened).

Tabs
----

The tabs follow the tutorials in order:

.. list-table::
   :header-rows: 1

   * - Tab
     - Use
     - Tutorial
   * - Images
     - Import test images
     - :doc:`tutorials/01_images_and_masks`
   * - Calibration
     - Solve a ChArUco camera calibration from board photographs, or import one
       saved from a script; inspect its corners, lens distortion and reprojection error
     - :doc:`tutorials/08_calibration`
   * - Subsets
     - Correlate a single subset and inspect it
     - :doc:`tutorials/02_subset`
   * - Meshes
     - Draw a boundary and exclusions, mesh and solve one image pair
     - :doc:`tutorials/03_mesh`
   * - Sequences
     - Solve a whole image series
     - :doc:`tutorials/04_sequence`
   * - Particles
     - Follow single points through a solved sequence
     - :doc:`tutorials/05_particle`
   * - Fields
     - Distribute and solve particles; contour, inspect, history and trace plots
     - :doc:`tutorials/06_field`

Each tab lists its saved files on the left; **New** starts a new object and
long solves show progress with a **Cancel** button. The settings in each form
are the arguments described in the matching tutorial.

Viewing results
---------------

- **Frame** (Meshes, Sequences): draw over the *reference* image with the mesh
  where it was built, or over the *target* image with the mesh moved by the
  measured displacements.
- **Opacity**: how strongly the contours cover the image.
- **Units** (Meshes, Sequences; the Fields form): pixels, or one of the
  project's calibrations for displacements — and, for a Field, positions and
  strains — in the board's units (e.g. mm). A calibrated Field is drawn without
  the image behind it, as its positions are no longer pixels.
- **Range**: *Auto* fits the colour scale to the data, rounded outwards to
  round numbers; *Manual* sets it.

Calibrations
------------

Photograph the board as described in :doc:`tutorials/08_calibration`, then in the
**Calibration** tab press **New**, import the photographs (they are copied into
``Images/Calibration/``), enter the board's squares across and down and its square
and marker side lengths, and **Solve**. Detection and the camera fit run in OpenCV,
so solving needs ``pip install "geopyv_dev[calibration]"``; without it the tab can
still **Import** and use a calibration saved from a script (``cal.save(...)``).
