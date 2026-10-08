# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Build & develop

```bash
# Build the Rust core library only
cargo build

# Build the Python extension (debug)
cargo build -p geopyv-dev-python

# Build and install the Python package into the venv (required after any Rust change)
source .venv/bin/activate
maturin develop --manifest-path python/Cargo.toml

# Run all Rust unit tests
cargo test

# Run a single Rust test by name
cargo test test_solve_icgn_order1_golden

# Run Python tests
source .venv/bin/activate
pytest tests/python/

# Run a single Python test file
pytest tests/python/mesh/test_mesh.py

# Run the integration smoke-test (requires images/ directory)
source .venv/bin/activate && python basictest.py
```

### Build the GUI (separate binary crate)

```bash
cargo build -p geopyv-gui
cargo run -p geopyv-gui
```

### Publishing a release to PyPI

The `.github/workflows/release.yml` workflow triggers **only on version tags** — pushing a branch does nothing. Steps:

1. Update the version string in all three places (they must match):
   - `Cargo.toml` (workspace root) — `version = "x.y.z"`
   - `pyproject.toml` — `version = "x.y.z"`
   - `python/Cargo.toml` — `version = "x.y.z"`

2. Commit the version bump and push the branch:
   ```bash
   git add Cargo.toml pyproject.toml python/Cargo.toml
   git commit -m "Bump version to x.y.z"
   git push origin <branch>
   ```

3. Create and push the version tag:
   ```bash
   git tag vx.y.z
   git push origin vx.y.z
   ```

The tag push triggers the workflow: it builds manylinux + aarch64 + Windows wheels and an sdist, then publishes everything to PyPI via trusted publishing (no API token needed).

## Architecture

### Three-crate workspace

```
geopyv-dev/          ← workspace root
  src/               ← Rust core library (crate: geopyv-dev)
  python/src/        ← PyO3 extension (crate: geopyv-dev-python → _geopyv_dev.so)
  geopyv-gui/src/    ← egui desktop app (crate: geopyv-gui)
  geopyv_dev/        ← Python package (imports .so + adds wrappers/plots)
  tests/python/      ← pytest suite for the Python API
```

The Python package that users import (`geopyv_dev`) is **not** the `.so` directly. It lives in `geopyv_dev/` and consists of:
- `__init__.py` — re-exports `_geopyv_dev.*`, then shadows the raw Rust classes (`Subset`, `Mesh`, `Sequence`, `Field`, `Particle`) with thin Python wrappers that add `.solve()`, `.inspect()`, `.convergence()`, `.contour()`.
- `wrappers.py` — `MeshWrapper`, `SequenceSolutionWrapper`, `FieldWrapper`, `ParticleWrapper`, `SubsetWrapper` — returned by `.solve()` on the wrapper classes.
- `plots.py` — all matplotlib visualisation functions.
- `_geopyv_dev.cpython-*.so` — the compiled Rust extension.

The installed `.so` is also shadowed: `maturin` writes to `geopyv_dev/_geopyv_dev.so` directly, which means `maturin develop` + the existing `geopyv_dev/` package are the runtime. There is a `python/geopyv-dev/` stub that is only used during the maturin build and can be ignored.

### Rust core → PyO3 boundary

Each domain module in `src/` has a corresponding `python/src/py_*.rs` wrapper:

| Core (`src/`) | Wrapper (`python/src/`) | Python class |
|---|---|---|
| `image.rs` | `py_image.rs` | `Image` |
| `templates.rs` | `py_templates.rs` | `Template` |
| `geometry/` | `py_geometry.rs` | `CircleRegion`, `PathRegion`, free fns |
| `subset.rs` | `py_subset.rs` | `Subset` |
| `mesh.rs` | `py_mesh.rs` | `Mesh`, `MeshSolution` |
| `sequence.rs` | `py_sequence.rs` | `Sequence`, `SequenceSolution` |
| `particle.rs` | `py_particle.rs` | `Particle`, `ParticleSolution` |
| `field.rs` | `py_field.rs` | `Field`, `FieldSolution` |
| `io.rs` | `py_io.rs` | `save`, `load` |
| `validation.rs` | `py_validation.rs` | free validation fns |

Error bridging: `geopyv_dev::Error` cannot implement `From<PyErr>` directly (orphan rule). The `python/src/lib.rs` newtype `Error(geopyv_dev::Error)` bridges via two `From` impls. Every wrapper method uses `.map_err(Error::from)?`.

### Data flow: DIC solve

`Image::from_file` → FFT → B-spline coefficients (QCQT matrix, stored on `Image`) → `Subset::new` takes QCQT + template coordinates + centre → computes `f`, `f_m`, `delta_f`, `grad_f` → `solve_icgn` / `solve_fagn` takes target QCQT + initial warp → returns `SolveResult`.

At the Python level: `gp.Image(filepath=...)` → `gp.Template(...)` → `gp.Subset(coord, template, f_img)` → `subset.solve(g_img, p_0)`.

### Mesh pipeline

`Mesh.generate(borders, segments, curves, ...)` calls `spade` (Delaunay/CDT) internally. `borders`/`segments`/`curves` are produced by `define_roi` (in `geometry/meshing.rs` / `py_geometry.rs`), which takes raw boundary nodes and exclusion-region node arrays and computes the constraint topology. `Mesh.solve(f_img, g_img, template, seed_coord, seed_warp)` runs reliability-guided propagation across all nodes, calling `Subset::new` + `solve_icgn/fagn` per node.

### Serialisation

`.pyv` files: 4-byte magic `GPYV` + 1-byte version `0x01` + bincode-v2 encoded `GeopyvObject` (a serde enum over `MeshSolution`, `FieldSolution`, `SequenceSolution`, `SubsetSolution`, `ParticleSolution`). Incompatible with the Python `geopyv` pickle format.

### Coordinate convention

`coord[0]` = x (column), `coord[1]` = y (row). `bspline_eval` uses `coords[:,0]` as the column index and `coords[:,1]` as the row index. This matches `_subset.cpp` in the original Python package.

### Tolerance tiers (used in test names and comments)

- **Tier A** `atol = 1e-12` — pure matrix algebra, no floating-point chains.
- **Tier B** `rtol = 1e-8` — FFT / B-spline coefficient computation.
- **Tier C** `rtol = 1e-5` — solver outputs (ZNCC, converged warp).

### Original Python package reference

The Rust code is a rewrite of `geopyv` (Python). C++ extension functions (`_subset.cpp`, etc.) are the numerical ground-truth. Comments throughout `src/` note which Python/C++ function each Rust function matches (e.g. `// Matches _intensity in _subset.cpp`). Golden test values come from running the Python package on the same inputs. The Python test images live at `geopyv/tests/ref.jpg` and `tar.jpg` relative to `CARGO_MANIFEST_DIR`; integration tests skip gracefully if the files are absent.
