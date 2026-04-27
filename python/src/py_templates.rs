//! PyO3 wrapper for `geopyv_dev::templates::Template`.
//!
//! Exposes a single `Template` Python class; shape ("circle" or "square") is
//! passed as the first argument.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use geopyv_dev::templates::Template;

use crate::Error;

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

/// Subset template.
///
/// Parameters
/// ----------
/// shape : str
///     ``"circle"`` or ``"square"``.
/// size : int, optional
///     Radius (circle) or half-side-length (square) in pixels. Default 25.
///
/// Examples
/// --------
/// >>> t = Template("circle", size=50)
/// >>> t = Template("square", size=30)
#[pyclass(name = "Template")]
pub struct PyTemplate {
    pub(crate) inner: Template,
}

#[pymethods]
impl PyTemplate {
    #[new]
    #[pyo3(signature = (shape, size = 25))]
    fn new(shape: &str, size: usize) -> PyResult<Self> {
        let t = match shape {
            "circle" => Template::circle(size).map_err(Error::from)?,
            "square" => Template::square(size).map_err(Error::from)?,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown template shape '{}': expected 'circle' or 'square'",
                    other
                )))
            }
        };
        Ok(PyTemplate { inner: t })
    }

    /// ``"circle"`` or ``"square"``.
    #[getter]
    fn shape(&self) -> &str {
        match self.inner.shape {
            geopyv_dev::templates::TemplateShape::Circle => "circle",
            geopyv_dev::templates::TemplateShape::Square => "square",
        }
    }

    /// ``"radius"`` (circle) or ``"length"`` (square).
    #[getter]
    fn dimension(&self) -> &str {
        &self.inner.dimension
    }

    /// Radius (circle) or half-side-length (square) in pixels.
    #[getter]
    fn size(&self) -> usize {
        self.inner.size
    }

    /// Number of active pixels in the unmasked template.
    #[getter]
    fn n_px(&self) -> usize {
        self.inner.n_px
    }

    /// Pixel offset coordinates, shape ``(n_px, 2)``, dtype float64.
    /// Column 0 = x-offset, column 1 = y-offset.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.coords.clone().into_pyarray_bound(py)
    }

    /// Binary subset mask, shape ``((2*size+1), (2*size+1))``, dtype int32.
    #[getter]
    fn subset_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i32>> {
        self.inner.subset_mask.clone().into_pyarray_bound(py)
    }

    /// Number of pixels remaining after the most recent :meth:`mask` call.
    #[getter]
    fn m_n_px(&self) -> Option<usize> {
        self.inner.m_n_px
    }

    /// Apply a binary image mask, updating ``coords`` and ``m_n_px``.
    ///
    /// Parameters
    /// ----------
    /// centre : array-like [x, y]
    ///     Centre coordinates (integer pixel).
    /// mask : np.ndarray (H, W), dtype uint8
    ///     Binary image mask: 0 = inactive, 1 = active.
    fn mask(&mut self, centre: [f64; 2], mask: PyReadonlyArray2<u8>) {
        self.inner.mask_update(centre, mask.as_array());
    }

    fn __repr__(&self) -> String {
        format!("Template('{}', size={})", self.shape(), self.inner.size)
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTemplate>()?;
    Ok(())
}
