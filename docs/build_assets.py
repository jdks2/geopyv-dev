#!/usr/bin/env python3
"""
Build pre-executed tutorial notebooks and static assets for the docs site.

Run from the repo root:
    python3 docs/build_assets.py
    # or:
    make docs-assets

This script executes each tutorial notebook (saving outputs and static assets
to docs/assets/ as a side-effect) and writes the executed notebooks to
docs/tutorials/ for use by Jupyter Book with execute_notebooks: "off".
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TUTORIALS = REPO_ROOT / "tutorials"
DOCS_TUTORIALS = REPO_ROOT / "docs" / "tutorials"
DOCS_ASSETS = REPO_ROOT / "docs" / "assets"

NOTEBOOKS = [
    "00_introduction.ipynb",
    "01_images_and_masks.ipynb",
    "02_subset.ipynb",
    "03_mesh.ipynb",
    "04_sequence.ipynb",
    "05_particle_and_field.ipynb",
]


def main():
    DOCS_TUTORIALS.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)

    failed = []
    for nb in NOTEBOOKS:
        src = TUTORIALS / nb
        print(f"\n--- Executing {nb} ---")
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=600",
                "--ExecutePreprocessor.kernel_name=python3",
                "--output-dir", str(DOCS_TUTORIALS),
                str(src),
            ],
            cwd=str(TUTORIALS),
        )
        if result.returncode != 0:
            print(f"  ERROR: {nb} failed.")
            failed.append(nb)
        else:
            print(f"  OK -> docs/tutorials/{nb}")

    print(f"\nAssets  : {DOCS_ASSETS}")
    print(f"Notebooks: {DOCS_TUTORIALS}")
    if failed:
        print(f"\nFailed ({len(failed)}):", ", ".join(failed))
        sys.exit(1)
    else:
        print(f"\nAll {len(NOTEBOOKS)} notebooks executed successfully.")


if __name__ == "__main__":
    main()
