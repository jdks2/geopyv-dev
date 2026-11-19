geopyv-dev
===========

``geopyv_dev`` is a high-performance Python package for Digital Image Correlation (DIC)
analysis in geotechnics, built on a Rust core with PyO3 bindings.

Installation
------------

.. code-block:: bash

   pip install geopyv_dev

See :doc:`installation` for prerequisites, Windows/Linux-specific commands,
and installing from source via GitHub.

Work through the :doc:`tutorials/index` in order — each builds on the previous one,
from loading images through to full strain-path tracking across an image sequence —
then consult the :doc:`api/index` for full class and function documentation.

Prefer a point-and-click workflow? The :doc:`gui` is installed with the package —
run ``geopyv-gui``. Runnable scripts covering each tutorial are in the
`demo folder <https://github.com/jdks2/geopyv-dev/tree/main/demo>`_ of the repository.

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   tutorials/index
   gui
   api/index
