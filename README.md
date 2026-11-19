# geopyv-dev
PIV/DIC package for geotechnics, with a Rust core. Under active development.

## Install

```bash
pip install geopyv_dev
```

Prebuilt wheels for Linux and Windows, Python 3.10–3.13. Camera calibration also
needs OpenCV: `pip install geopyv_dev[calibration]`.

## Use

- **Scripts:** `import geopyv_dev as gp`. Runnable examples for each step are in
  [`demo/`](demo/).
- **GUI:** run `geopyv-gui` (installed with the package).
- **Docs and tutorials:** <https://geopyv-dev.readthedocs.io>
