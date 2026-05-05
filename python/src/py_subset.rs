//! PyO3 wrapper for `geopyv_dev::subset`.

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use geopyv_dev::subset::Subset;

use crate::{
    py_image::PyImage,
    py_templates::PyTemplate,
    Error,
};

// ---------------------------------------------------------------------------
// Python class
// ---------------------------------------------------------------------------

/// Reference subset for DIC.
///
/// Parameters
/// ----------
/// coord : array_like of shape (2,)
///     Subset centre coordinate ``[x, y]``.
/// template : Template
///     Template object whose pixel offsets define the subset shape.
/// f_img : Image
///     Reference image (pre-computed B-spline data).
///
/// Attributes
/// ----------
/// coord : list[float]
///     Subset centre ``[x, y]``.
/// template_shape : str
///     ``"circle"`` or ``"square"``.
/// template_size : int
///     Radius or half-side-length of the template.
/// n_px : int
///     Number of pixels in the subset.
/// f_coords : numpy.ndarray, shape (n_px, 2)
///     Absolute coordinates of each subset pixel in the reference image.
/// f : numpy.ndarray, shape (n_px,)
///     Reference intensities (B-spline interpolated).
/// f_m : float
///     Mean reference intensity.
/// delta_f : float
///     ``sqrt(Σ(f_i − f_m)²)`` normalisation factor.
/// grad_f : numpy.ndarray, shape (n_px, 2)
///     Image gradient at each subset pixel ``[grad_x, grad_y]``.
/// sssig : float
///     Sum of squared intensity gradients (quality metric).
/// sigma_intensity : float
///     Standard deviation of reference intensities (quality metric).
#[pyclass(name = "Subset")]
pub struct PySubset {
    inner: Subset,
    pub(crate) f_img_path: Option<String>,
    pub(crate) template_size: usize,
    pub(crate) template_shape: String,
    solve_result: Option<pyo3::PyObject>,
}

#[pymethods]
impl PySubset {
    #[new]
    #[pyo3(signature = (coord, template, f_img))]
    fn new(
        coord: [f64; 2],
        template: &Bound<'_, PyAny>,
        f_img: PyRef<'_, PyImage>,
    ) -> PyResult<Self> {
        let tmpl = template
            .extract::<PyRef<'_, PyTemplate>>()
            .map_err(|_| PyTypeError::new_err("template must be a Template"))?;
        let template_shape = match tmpl.inner.shape {
            geopyv_dev::templates::TemplateShape::Circle => "circle",
            geopyv_dev::templates::TemplateShape::Square => "square",
        }.to_string();
        let template_size = tmpl.inner.size;
        let s = Subset::new(coord, &tmpl.inner.coords, &f_img.inner.qcqt)
            .map_err(Error::from)?;
        Ok(PySubset {
            inner: s,
            f_img_path: f_img.filepath.clone(),
            template_size,
            template_shape,
            solve_result: None,
        })
    }

    /// Subset centre ``[x, y]``.
    #[getter]
    fn coord(&self) -> [f64; 2] {
        self.inner.coord
    }

    /// ``"circle"`` or ``"square"``.
    #[getter]
    fn template_shape(&self) -> &str {
        &self.template_shape
    }

    /// Radius (circle) or half-side-length (square) in pixels.
    #[getter]
    fn template_size(&self) -> usize {
        self.template_size
    }

    /// Number of pixels.
    #[getter]
    fn n_px(&self) -> usize {
        self.inner.n_px()
    }

    /// Absolute subset pixel coordinates, shape (n_px, 2).
    #[getter]
    fn f_coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.f_coords.clone().into_pyarray_bound(py)
    }

    /// Reference intensities, shape (n_px,).
    #[getter]
    fn f<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.f.clone().into_pyarray_bound(py)
    }

    /// Mean reference intensity.
    #[getter]
    fn f_m(&self) -> f64 {
        self.inner.f_m
    }

    /// ``sqrt(Σ(f_i − f_m)²)`` normalisation factor.
    #[getter]
    fn delta_f(&self) -> f64 {
        self.inner.delta_f
    }

    /// Reference image gradients, shape (n_px, 2) — ``[grad_x, grad_y]``.
    #[getter]
    fn grad_f<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.grad_f.clone().into_pyarray_bound(py)
    }

    /// Sum of squared intensity gradients (quality metric).
    #[getter]
    fn sssig(&self) -> f64 {
        self.inner.sssig
    }

    /// Standard deviation of reference intensities.
    #[getter]
    fn sigma_intensity(&self) -> f64 {
        self.inner.sigma_intensity
    }

    /// Inverse Compositional Gauss-Newton solver.
    ///
    /// Parameters
    /// ----------
    /// g_img : Image
    ///     Target image (pre-computed B-spline data).
    /// p_0 : list[float]
    ///     Initial warp vector. Length 6 for order-1, 12 for order-2.
    /// max_norm : float, optional
    ///     Convergence criterion on ``||Δp||``. Default 1e-3.
    /// max_iterations : int, optional
    ///     Iteration limit. Default 50.
    ///
    /// Returns
    /// -------
    /// dict with keys:
    ///     ``p`` (list[float]), ``c_zncc`` (float), ``c_znssd`` (float),
    ///     ``iterations`` (int), ``converged`` (bool),
    ///     ``history`` (list of (iter, norm, zncc, znssd)).
    #[pyo3(signature = (g_img, p_0, max_norm=1e-3, max_iterations=50))]
    fn solve_icgn<'py>(
        &mut self,
        py: Python<'py>,
        g_img: PyRef<'_, PyImage>,
        p_0: Vec<f64>,
        max_norm: f64,
        max_iterations: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self
            .inner
            .solve_icgn(&g_img.inner.qcqt, &p_0, max_norm, max_iterations)
            .map_err(Error::from)?;
        let d = result_to_dict(py, result)?;
        d.set_item("max_norm", max_norm)?;
        d.set_item("max_iterations", max_iterations)?;
        self.solve_result = Some(d.clone().into_any().unbind());
        Ok(d)
    }

    /// Forward Additive Gauss-Newton solver.
    ///
    /// Parameters and return value same as :meth:`solve_icgn`.
    #[pyo3(signature = (g_img, p_0, max_norm=1e-3, max_iterations=50))]
    fn solve_fagn<'py>(
        &mut self,
        py: Python<'py>,
        g_img: PyRef<'_, PyImage>,
        p_0: Vec<f64>,
        max_norm: f64,
        max_iterations: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self
            .inner
            .solve_fagn(&g_img.inner.qcqt, &p_0, max_norm, max_iterations)
            .map_err(Error::from)?;
        let d = result_to_dict(py, result)?;
        d.set_item("max_norm", max_norm)?;
        d.set_item("max_iterations", max_iterations)?;
        self.solve_result = Some(d.clone().into_any().unbind());
        Ok(d)
    }

    /// File path of the reference image, or ``None``.
    #[getter]
    fn f_img_path(&self) -> Option<String> {
        self.f_img_path.clone()
    }

    /// Solve result dict from the last solve call, or ``None``.
    #[getter]
    fn solve_result<'py>(&self, py: Python<'py>) -> Option<Bound<'py, pyo3::types::PyDict>> {
        self.solve_result.as_ref().map(|obj| {
            obj.bind(py).downcast::<pyo3::types::PyDict>().unwrap().clone()
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Subset(coord={:?}, n_px={})",
            self.inner.coord,
            self.inner.n_px()
        )
    }
}

// ---------------------------------------------------------------------------
// Helper: convert SolveResult → Python dict
// ---------------------------------------------------------------------------

fn result_to_dict<'py>(
    py: Python<'py>,
    result: geopyv_dev::subset::SolveResult,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("p", result.p)?;
    d.set_item("c_zncc", result.c_zncc)?;
    d.set_item("c_znssd", result.c_znssd)?;
    d.set_item("iterations", result.iterations)?;
    d.set_item("converged", result.converged)?;
    let history: Vec<(usize, f64, f64, f64)> = result.history;
    d.set_item("history", history)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// Module registration helper
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySubset>()?;
    Ok(())
}
