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
other way round.

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
