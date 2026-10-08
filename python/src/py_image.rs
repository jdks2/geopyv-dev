//! PyO3 wrapper for `geopyv_dev::image::Image`.

use std::path::Path;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2}; // IntoPyArray brings into_pyarray_bound into scope
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use geopyv_dev::image::Image;

use crate::Error;

// ---------------------------------------------------------------------------
// Python class
// ---------------------------------------------------------------------------

/// Greyscale image with pre-computed bi-quintic B-spline interpolation data.
///
/// Parameters
/// ----------
/// filepath : str or pathlib.Path, optional
///     Path to a JPEG/PNG/etc image file. The image is loaded, converted to
///     greyscale (OpenCV BGR2GRAY weights), and pre-filtered with a 5×5
///     Gaussian (σ = 1.1).
/// image_gs : numpy.ndarray of shape (H, W), dtype float64, optional
///     Pre-loaded greyscale image (values 0–255). Use this when cv2 has
///     already done the loading and blur on the Python side.
/// border : int, optional
///     Padding border for B-spline coefficient computation. Default 20.
///
/// Exactly one of `filepath` or `image_gs` must be supplied.
#[pyclass(name = "Image")]
pub struct PyImage {
    pub(crate) inner: Arc<Image>,
    pub(crate) filepath: Option<String>,
}

#[pymethods]
impl PyImage {
    #[new]
    #[pyo3(signature = (filepath=None, image_gs=None, border=20))]
    fn new(
        filepath: Option<&str>,
        image_gs: Option<PyReadonlyArray2<f64>>,
        border: usize,
    ) -> PyResult<Self> {
        match (filepath, image_gs) {
            (Some(fp), None) => {
                let img = Image::from_file(Path::new(fp), border).map_err(Error::from)?;
                Ok(PyImage { inner: Arc::new(img), filepath: Some(fp.to_owned()) })
            }
            (None, Some(arr)) => {
                let owned = arr.as_array().to_owned();
                Ok(PyImage { inner: Arc::new(Image::from_array(owned, border)), filepath: None })
            }
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "supply exactly one of `filepath` or `image_gs`, not both",
            )),
            (None, None) => Err(PyValueError::new_err(
                "one of `filepath` or `image_gs` is required",
            )),
        }
    }

    /// Greyscale pixel values; shape (H, W), dtype float64.
    #[getter]
    fn image_gs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.image_gs.clone().into_pyarray_bound(py)
    }

    /// Pre-computed Q·C_block·Qᵀ matrix; shape (H*6, W*6), dtype float64.
    ///
    /// The B-spline block for pixel (i, j) is `qcqt[i*6:i*6+6, j*6:j*6+6]`.
    ///
    /// Internally the core stores this block-contiguous `(H, W, 36)`
    /// (`layer_rg_plan.md` §11.5); this getter re-interleaves to the
    /// historical `(H*6, W*6)` layout so the Python-visible shape and
    /// block-indexing convention are unchanged.
    #[getter]
    fn qcqt<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let q3 = &self.inner.qcqt; // (rows, cols, 36), block-contiguous
        let (rows, cols, _) = q3.dim();
        let mut out = numpy::ndarray::Array2::<f64>::zeros((rows * 6, cols * 6));
        for i in 0..rows {
            for j in 0..cols {
                for r in 0..6 {
                    for c in 0..6 {
                        out[[i * 6 + r, j * 6 + c]] = q3[[i, j, r * 6 + c]];
                    }
                }
            }
        }
        out.into_pyarray_bound(py)
    }

    /// Border padding used during B-spline coefficient computation.
    #[getter]
    fn border(&self) -> usize {
        self.inner.border
    }

    /// File path used to load the image, or ``None`` if constructed from array.
    #[getter]
    fn filepath(&self) -> Option<String> {
        self.filepath.clone()
    }

    fn __repr__(&self) -> String {
        let (h, w) = self.inner.image_gs.dim();
        format!("Image(shape=({h}, {w}), border={})", self.inner.border)
    }
}

// ---------------------------------------------------------------------------
// Module registration helper
// ---------------------------------------------------------------------------

/// Register the `Image` class into the given (sub)module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImage>()?;
    Ok(())
}
