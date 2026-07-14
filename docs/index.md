# geopyv-dev

`geopyv_dev` is a high-performance Python package for Digital Image Correlation (DIC)
analysis in geotechnics, built on a Rust core with PyO3 bindings.

## Installation

```bash
pip install geopyv_dev
```

## Tutorials

Work through the notebooks in order — each builds on the previous one, from
loading images through to full strain-path tracking across an image sequence.

| Notebook | Topic |
|----------|-------|
| 00 — Introduction | What is DIC? Package overview and object hierarchy |
| 01 — Images & Masks | `Image`, `Mask`, `CircleRegion`, `PathRegion` |
| 02 — Subset | Single-point correlation, warp parameters, ICGN/FAGN |
| 03 — Mesh | Full-field DIC, reliability-guided propagation |
| 04 — Sequence | Multi-image time series |
| 05 — Particle & Field | Lagrangian/Eulerian strain-path tracking |
