//! PyO3 wrapper for `geopyv_dev::speckle`.

use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use geopyv_dev::speckle::{Progression, ScaleType, ShearBandOption, Speckle, WarpMode};


use crate::Error;

// ---------------------------------------------------------------------------
// Dict-extraction helpers
// ---------------------------------------------------------------------------

fn req<'py, T: pyo3::FromPyObject<'py>>(
    d: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<T> {
    d.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing key '{key}'")))?
        .extract::<T>()
}

fn opt<'py, T: pyo3::FromPyObject<'py>>(
    d: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<T>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract::<T>()?)),
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// PySpeckle
// ---------------------------------------------------------------------------

#[pyclass(name = "Speckle")]
pub struct PySpeckle {
    pub(crate) inner: Arc<Speckle>,
}

#[pymethods]
impl PySpeckle {
    /// Create a synthetic speckle image generator.
    ///
    /// Parameters
    /// ----------
    /// image_cfg : dict
    ///     ``image_dir`` (str), ``name`` (str), ``image_size`` ((int,int), default (1001,1001)),
    ///     ``file_format`` (str, default ".jpg").
    /// speckle_cfg : dict
    ///     ``speckle_size`` (float, px blob radius), ``speckle_number`` (int).
    /// progression : str
    ///     ``"deformation"`` — warp progresses; noise constant at final values.
    ///     ``"noise"``       — noise progresses; deformation constant at full comp.
    /// deformation_cfg : dict
    ///     ``comp`` (list[float], 12 elements) — final warp vector (applied at mult=1).
    ///     ``origin`` ((float,float), optional) — warp reference origin; defaults to image centre.
    ///     ``mode`` (str, optional) — ``None`` / ``"SB"`` for shear-band.
    ///     ``option`` (str, optional) — ``"sin"`` / ``"lin"`` / ``"quad"`` when mode="SB".
    ///     ``width`` (float, optional) — shear-band full width in px (default 100.0).
    /// noise_cfg : (float, float)
    ///     ``(noise_pos, noise_int)`` — final Gaussian noise std-devs.
    /// scale_cfg : dict
    ///     ``scale`` (str) — ``"lin"`` or ``"log"``.
    ///     ``n`` (int) — total number of images (image 0 is always the reference).
    ///     ``min`` (float, optional) — minimum multiplier for log scale (default 1e-5).
    #[new]
    #[pyo3(signature = (image_cfg, speckle_cfg, progression, deformation_cfg, noise_cfg, scale_cfg))]
    fn new(
        image_cfg: &Bound<'_, PyDict>,
        speckle_cfg: &Bound<'_, PyDict>,
        progression: &str,
        deformation_cfg: &Bound<'_, PyDict>,
        noise_cfg: (f64, f64),
        scale_cfg: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        // image_cfg
        let image_dir: String = req(image_cfg, "image_dir")?;
        let name: String = req(image_cfg, "name")?;
        let image_size: (usize, usize) =
            opt(image_cfg, "image_size")?.unwrap_or((1001, 1001));
        let file_format_raw: String =
            opt(image_cfg, "file_format")?.unwrap_or_else(|| ".jpg".into());
        let file_format = if file_format_raw.starts_with('.') {
            file_format_raw
        } else {
            format!(".{file_format_raw}")
        };

        // speckle_cfg
        let speckle_size: f64 = req(speckle_cfg, "speckle_size")?;
        let speckle_number: usize = req(speckle_cfg, "speckle_number")?;

        // progression
        let prog = match progression {
            "deformation" => Progression::Deformation,
            "noise" => Progression::Noise,
            other => return Err(PyValueError::new_err(format!(
                "progression must be 'deformation' or 'noise', got '{other}'"
            ))),
        };

        // deformation_cfg
        let comp_vec: Vec<f64> = opt(deformation_cfg, "comp")?.unwrap_or_else(|| vec![0.0; 12]);
        let mut comp: [f64; 12] = comp_vec.try_into().map_err(|v: Vec<f64>| {
            PyValueError::new_err(format!(
                "comp must have exactly 12 elements, got {}",
                v.len()
            ))
        })?;
        let origin: [f64; 2] = match opt::<(f64, f64)>(deformation_cfg, "origin")? {
            Some((ox, oy)) => [ox, oy],
            None => [image_size.0 as f64 / 2.0, image_size.1 as f64 / 2.0],
        };
        let mode_str: Option<String> = opt(deformation_cfg, "mode")?;
        let warp_mode = match mode_str.as_deref() {
            None | Some("") | Some("none") | Some("None") => WarpMode::None,
            Some(s) if s.eq_ignore_ascii_case("rotation") => {
                // Store angle in comp[0]; other comp entries are ignored for rotation.
                let angle: f64 = req(deformation_cfg, "angle")?;
                comp[0] = angle;
                WarpMode::Rotation
            }
            Some(s) if s.eq_ignore_ascii_case("sb") => {
                let option_str: String =
                    opt(deformation_cfg, "option")?.unwrap_or_else(|| "sin".into());
                let sb_opt = match option_str.as_str() {
                    "sin" => ShearBandOption::Sin,
                    "lin" => ShearBandOption::Lin,
                    "quad" => ShearBandOption::Quad,
                    other => return Err(PyValueError::new_err(format!(
                        "option must be 'sin', 'lin', or 'quad', got '{other}'"
                    ))),
                };
                let width: f64 = opt(deformation_cfg, "width")?.unwrap_or(100.0);
                WarpMode::ShearBand { option: sb_opt, width }
            }
            Some(other) => return Err(PyValueError::new_err(format!(
                "mode must be None, 'rotation', or 'SB', got '{other}'"
            ))),
        };

        // noise_cfg
        let (noise_pos, noise_int) = noise_cfg;

        // scale_cfg
        let scale_str: String = req(scale_cfg, "scale")?;
        let image_no: usize = req(scale_cfg, "n")?;
        let scale = match scale_str.as_str() {
            "lin" => ScaleType::Lin,
            "log" => {
                let min: f64 = opt(scale_cfg, "min")?.unwrap_or(1e-5);
                ScaleType::Log { min }
            }
            other => return Err(PyValueError::new_err(format!(
                "scale must be 'lin' or 'log', got '{other}'"
            ))),
        };

        let inner = Speckle::new(
            image_dir,
            name,
            file_format,
            image_size,
            speckle_size,
            speckle_number,
            prog,
            comp,
            origin,
            noise_pos,
            noise_int,
            warp_mode,
            image_no,
            scale,
        )
        .map_err(Error::from)?;

        Ok(PySpeckle { inner: Arc::new(inner) })
    }

    /// Generate all images and write them to ``image_dir``.
    ///
    /// Parameters
    /// ----------
    /// seed : int, optional
    ///     RNG seed for reproducibility. Default 42.
    #[pyo3(signature = (seed=None))]
    fn solve(&mut self, seed: Option<u64>) -> PyResult<()> {
        Arc::get_mut(&mut self.inner)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "cannot call solve(): Speckle is shared with a Validation instance"
            ))?
            .solve(seed.unwrap_or(42))
            .map_err(Error::from)?;
        Ok(())
    }

    /// Reference speckle positions ``(N, 2)`` — available after :meth:`solve`.
    #[getter]
    fn speckle_positions<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pos = self.inner.speckle_positions.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("call solve() first")
        })?;
        Ok(pos.clone().into_pyarray_bound(py))
    }

    #[getter]
    fn solved(&self) -> bool {
        self.inner.solved
    }

    fn __repr__(&self) -> String {
        format!(
            "Speckle(image_no={}, speckle_number={}, speckle_size={}, solved={})",
            self.inner.image_no,
            self.inner.speckle_number,
            self.inner.speckle_size,
            if self.inner.solved { "True" } else { "False" },
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpeckle>()?;
    Ok(())
}
