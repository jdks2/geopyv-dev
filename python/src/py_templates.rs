//! PyO3 wrappers for `geopyv_dev::templates::Template`.
//!
//! Exposes `Circle` and `Square` as separate Python classes matching the
//! `geopyv.templates` public API.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use geopyv_dev::templates::Template;

use crate::Error;

// ---------------------------------------------------------------------------
// Circle template
// ---------------------------------------------------------------------------

/// Circular subset template.
///
/// Parameters
/// ----------
/// radius : int, optional
///     Radius of the subset in pixels. Default 25.
#[pyclass(name = "Circle")]
pub struct PyCircle {
    inner: Template,
}

#[pymethods]
impl PyCircle {
    #[new]
    #[pyo3(signature = (radius = 25))]
    fn new(radius: usize) -> PyResult<Self> {
        let t = Template::circle(radius).map_err(Error::from)?;
        Ok(PyCircle { inner: t })
    }

    #[getter]
    fn shape(&self) -> &str {
        "circle"
    }

    #[getter]
    fn dimension(&self) -> &str {
        "radius"
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size
    }

    #[getter]
    fn n_px(&self) -> usize {
        self.inner.n_px
    }

    /// Pixel offset coordinates, shape (n_px, 2), dtype float64.
    /// Column 0 = row-offset (y), column 1 = col-offset (x).
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.coords.clone().into_pyarray_bound(py)
    }

    /// Binary subset mask, shape ((2*radius+1), (2*radius+1)), dtype int32.
    #[getter]
    fn subset_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i32>> {
        self.inner.subset_mask.clone().into_pyarray_bound(py)
    }

    /// Number of pixels after the last `mask` call.
    #[getter]
    fn m_n_px(&self) -> Option<usize> {
        self.inner.m_n_px
    }

    /// Apply a binary image mask, updating `coords` and `m_n_px`.
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
        format!("Circle(radius={})", self.inner.size)
    }
}

// ---------------------------------------------------------------------------
// Square template
// ---------------------------------------------------------------------------

/// Square subset template.
///
/// Parameters
/// ----------
/// length : int, optional
///     Half side-length of the subset in pixels. Default 25.
#[pyclass(name = "Square")]
pub struct PySquare {
    inner: Template,
}

#[pymethods]
impl PySquare {
    #[new]
    #[pyo3(signature = (length = 25))]
    fn new(length: usize) -> PyResult<Self> {
        let t = Template::square(length).map_err(Error::from)?;
        Ok(PySquare { inner: t })
    }

    #[getter]
    fn shape(&self) -> &str {
        "square"
    }

    #[getter]
    fn dimension(&self) -> &str {
        "length"
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size
    }

    #[getter]
    fn n_px(&self) -> usize {
        self.inner.n_px
    }

    /// Pixel offset coordinates, shape (n_px, 2), dtype float64.
    /// Column 0 = x-offset, column 1 = y-offset.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.coords.clone().into_pyarray_bound(py)
    }

    /// Binary subset mask, all ones, shape ((2*length+1), (2*length+1)), dtype int32.
    #[getter]
    fn subset_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i32>> {
        self.inner.subset_mask.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn m_n_px(&self) -> Option<usize> {
        self.inner.m_n_px
    }

    fn mask(&mut self, centre: [f64; 2], mask: PyReadonlyArray2<u8>) {
        self.inner.mask_update(centre, mask.as_array());
    }

    fn __repr__(&self) -> String {
        format!("Square(length={})", self.inner.size)
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCircle>()?;
    m.add_class::<PySquare>()?;
    Ok(())
}
