Installation
=============

There are two ways to install ``geopyv_dev``:

- **pip**, from PyPI — prebuilt wheels, no Rust toolchain required. The
  right choice for using the package.
- **GitHub (from source)** — needed if you want the desktop GUI
  (``geopyv-gui``), unreleased changes, or to contribute.

Both are covered below for **Linux** and **Windows**.

Prerequisites
--------------

.. list-table::
   :header-rows: 1

   * - Route
     - Requirements
   * - pip
     - Python ≥ 3.8, pip
   * - GitHub (source)
     - Everything pip needs, plus `git <https://git-scm.com/downloads>`_ and
       `Rust <https://rustup.rs>`_ (stable, ≥ 1.85 — installed via ``rustup``).
       **Windows only:** the Visual Studio Build Tools "Desktop development
       with C++" workload (Rust needs its linker, ``link.exe``). Linux
       usually already has a working C toolchain; if not, install your
       distribution's build-essentials package (e.g. ``sudo apt install
       build-essential`` on Debian/Ubuntu).

Installing with pip
---------------------

Prebuilt wheels are published for Linux (x86_64, aarch64) and Windows
(Python 3.8–3.13) — no Rust toolchain needed for this route.

Linux
~~~~~~

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install geopyv_dev

Windows (PowerShell)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: powershell

   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install geopyv_dev

Verify the install (either OS, with the virtual environment active):

.. code-block:: bash

   python -c "import geopyv_dev"

Installing from GitHub (source)
-----------------------------------

Use this route to build the GUI, try unreleased changes, or contribute.

Linux
~~~~~~

.. code-block:: bash

   # 1. Install Rust (skip if you already have it)
   curl https://sh.rustup.rs -sSf | sh

   # 2. Clone the repository
   git clone https://github.com/jdks2/geopyv-dev.git
   cd geopyv-dev

   # 3. Create and activate a virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

   # 4. Build and install the Python package into the venv
   pip install maturin
   maturin develop --release --manifest-path python/Cargo.toml

Windows (PowerShell)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: powershell

   # 1. Install Rust (skip if you already have it) — download and run
   #    rustup-init.exe from https://rustup.rs, and make sure the Visual
   #    Studio Build Tools "Desktop development with C++" workload is
   #    installed (rustup will prompt for it if missing)

   # 2. Clone the repository
   git clone https://github.com/jdks2/geopyv-dev.git
   cd geopyv-dev

   # 3. Create and activate a virtual environment
   py -m venv .venv
   .venv\Scripts\Activate.ps1

   # 4. Build and install the Python package into the venv
   pip install maturin
   maturin develop --release --manifest-path python/Cargo.toml

Once built this way, the same clone also builds the desktop GUI:

.. code-block:: bash

   cargo run -p geopyv-gui

.. note::
   macOS isn't currently covered by a prebuilt wheel — use the from-source
   route above; the Linux steps apply almost verbatim (Rust via ``rustup``,
   Xcode Command Line Tools provide the C toolchain).
