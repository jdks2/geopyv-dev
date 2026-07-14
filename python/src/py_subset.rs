//! PyO3 wrapper for `geopyv_dev::subset`.

use std::path::PathBuf;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use geopyv_dev::{
    image::Image,
    io::{self as gp_io, GeopyvObject},
    subset::{Subset, SubsetSolution, SolveResult},
    masks::{LocalMask, MaskShape},
};

use crate::{
    py_image::PyImage,
    py_mask::PyMask,
    Error,
};

// ---------------------------------------------------------------------------
// Python class
// ---------------------------------------------------------------------------

#[pyclass(name = "Subset")]
pub struct PySubset {
    // Always present; image-dependent fields within inner may be None.
    // Solve state (SolveResult wrapped in a SubsetSolution) lives on `inner`
    // itself — see Subset::solved()/Subset::solution().
    pub(crate) inner: Subset,

    // Python object handles (not in Rust core).
    local_mask: Option<Py<PyMask>>,
    pub(crate) f_img: Option<Py<PyImage>>,
    pub(crate) g_img: Option<Py<PyImage>>,
}

// ---------------------------------------------------------------------------
// Non-pymethods impl (Rust-internal helpers)
// ---------------------------------------------------------------------------

impl PySubset {
    /// Reconstruct a `PySubset` from a loaded `SubsetSolution`.
    pub(crate) fn from_solution(py: Python<'_>, sol: SubsetSolution) -> PyResult<Self> {
        let order = (sol.result.p.len() / 6).max(1);

        // Reconstruct LocalMask from template summary.
        let local_mask = match sol.mask.shape {
            MaskShape::Circle => LocalMask::circle(sol.mask.size),
            MaskShape::Square => LocalMask::square(sol.mask.size),
        }.map_err(Error::from)?;
        let py_mask: Py<PyMask> = Py::new(py, PyMask::from_local(local_mask.clone()))?;

        // Attempt to load reference image.
        let py_f_img: Option<Py<PyImage>> =
            Image::from_file(&sol.ref_image, 20).ok().map(|img| {
                Py::new(py, PyImage {
                    inner: Arc::new(img),
                    filepath: Some(sol.ref_image.to_string_lossy().into_owned()),
                })
            }).transpose()?;

        // Attempt to load target image.
        let py_g_img: Option<Py<PyImage>> =
            Image::from_file(&sol.target_image, 20).ok().map(|img| {
                Py::new(py, PyImage {
                    inner: Arc::new(img),
                    filepath: Some(sol.target_image.to_string_lossy().into_owned()),
                })
            }).transpose()?;

        // Build inner — always Subset. Reconstruct with live image data when
        // available (for potential re-solving), but always mark it solved
        // with the loaded solution.
        let mut inner = if let (Some(ref f_py), Some(ref g_py)) = (&py_f_img, &py_g_img) {
            let f_ref = f_py.bind(py).borrow();
            let g_ref = g_py.bind(py).borrow();
            Subset::new(
                sol.coord, &local_mask, None,
                Arc::clone(&f_ref.inner), Arc::clone(&g_ref.inner), order,
            ).unwrap_or_else(|_| Subset::from_subset_solution(&sol, &local_mask))
        } else {
            Subset::from_subset_solution(&sol, &local_mask)
        };
        inner.set_solution(sol);

        Ok(PySubset {
            inner,
            local_mask: Some(py_mask),
            f_img: py_f_img,
            g_img: py_g_img,
        })
    }

    /// Build a `SubsetSolution` for serialisation.
    pub(crate) fn to_subset_solution(&self, py: Python<'_>) -> PyResult<SubsetSolution> {
        let mut sol = self.inner.solution().cloned().ok_or_else(|| {
            PyRuntimeError::new_err("Subset has not been solved")
        })?;
        // Prefer the wrapper's own image handles (may have been swapped since
        // the solve) over whatever was baked into the stored solution.
        if let Some(f_img_path) = self.f_img.as_ref()
            .and_then(|img| img.bind(py).borrow().filepath.clone())
        {
            sol.ref_image = PathBuf::from(f_img_path);
        }
        if let Some(g_img_path) = self.g_img.as_ref()
            .and_then(|img| img.bind(py).borrow().filepath.clone())
        {
            sol.target_image = PathBuf::from(g_img_path);
        }
        Ok(sol)
    }

    /// The underlying `SolveResult`, if solved.
    fn result(&self) -> Option<&SolveResult> {
        self.inner.solution().map(|s| &s.result)
    }
}

// ---------------------------------------------------------------------------
// pymethods
// ---------------------------------------------------------------------------

#[pymethods]
impl PySubset {
    #[new]
    #[pyo3(signature = (coord, local_mask, f_img, g_img, subset_order = 1, global_mask = None))]
    fn new(
        py: Python<'_>,
        coord: [f64; 2],
        local_mask: &Bound<'_, PyAny>,
        f_img: &Bound<'_, PyImage>,
        g_img: &Bound<'_, PyImage>,
        subset_order: usize,
        global_mask: Option<Py<PyMask>>,
    ) -> PyResult<Self> {
        if subset_order != 1 && subset_order != 2 {
            return Err(PyValueError::new_err(format!(
                "subset_order must be 1 or 2, got {subset_order}"
            )));
        }

        let lm_bound = local_mask
            .downcast::<PyMask>()
            .map_err(|_| PyTypeError::new_err("local_mask must be a Mask"))?;

        let inner = {
            let lm_ref = lm_bound.borrow();
            let lm = lm_ref.local_mask_ref()?;
            let gm_guard = global_mask.as_ref().map(|gm| gm.bind(py).borrow());
            let gm_view = gm_guard.as_ref().and_then(|g| g.global_view());
            let f_img_ref = f_img.borrow();
            let g_img_ref = g_img.borrow();
            Subset::new(
                coord,
                lm,
                gm_view,
                Arc::clone(&f_img_ref.inner),
                Arc::clone(&g_img_ref.inner),
                subset_order,
            ).map_err(Error::from)?
        };

        let lm_py: Py<PyMask> = lm_bound.clone().unbind();
        let f_img_py: Py<PyImage> = f_img.clone().unbind();
        let g_img_py: Py<PyImage> = g_img.clone().unbind();

        Ok(PySubset {
            inner,
            local_mask: Some(lm_py),
            f_img: Some(f_img_py),
            g_img: Some(g_img_py),
        })
    }

    // -----------------------------------------------------------------------
    // Solvers
    // -----------------------------------------------------------------------

    /// Inverse Compositional Gauss-Newton solver. Mutates object; returns None.
    #[pyo3(signature = (p_0 = None, max_norm = 1e-3, max_iterations = 50, tolerance = 0.75))]
    fn solve_icgn(
        &mut self,
        p_0: Option<Vec<f64>>,
        max_norm: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<()> {
        self.inner
            .solve_icgn(p_0.as_deref(), tolerance, max_norm, max_iterations)
            .map_err(Error::from)?;
        Ok(())
    }

    /// Forward Additive Gauss-Newton solver. Mutates object; returns None.
    #[pyo3(signature = (p_0 = None, max_norm = 1e-3, max_iterations = 50, tolerance = 0.75))]
    fn solve_fagn(
        &mut self,
        p_0: Option<Vec<f64>>,
        max_norm: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<()> {
        self.inner
            .solve_fagn(p_0.as_deref(), tolerance, max_norm, max_iterations)
            .map_err(Error::from)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Save
    // -----------------------------------------------------------------------

    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        if !self.inner.solved() {
            return Err(PyRuntimeError::new_err(
                "Subset has not been solved; cannot save.",
            ));
        }
        let sol = self.to_subset_solution(py)?;
        gp_io::save(path, &GeopyvObject::Subset(sol)).map_err(Error::from)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Always-available getters
    // -----------------------------------------------------------------------

    #[getter]
    fn coord(&self) -> [f64; 2] { self.inner.coord }

    #[getter]
    fn n_px(&self) -> usize { self.inner.n_px() }

    #[getter]
    fn subset_order(&self) -> usize { self.inner.subset_order }

    #[getter]
    fn sssig(&self) -> f64 { self.inner.sssig }

    #[getter]
    fn solved(&self) -> bool { self.inner.solved() }

    #[getter]
    fn template_shape(&self) -> String {
        match &self.inner.mask.shape {
            MaskShape::Circle => "circle".to_string(),
            MaskShape::Square => "square".to_string(),
        }
    }

    #[getter]
    fn template_size(&self) -> usize { self.inner.mask.size }

    #[getter]
    fn template_n_px(&self) -> usize { self.inner.mask.n_px }

    #[getter]
    fn f_img(&self, py: Python<'_>) -> Option<PyObject> {
        self.f_img.as_ref().map(|img| img.clone_ref(py).into_any())
    }

    #[getter]
    fn g_img(&self, py: Python<'_>) -> Option<PyObject> {
        self.g_img.as_ref().map(|img| img.clone_ref(py).into_any())
    }

    #[getter]
    fn local_mask(&self, py: Python<'_>) -> Option<PyObject> {
        self.local_mask.as_ref().map(|m| m.clone_ref(py).into_any())
    }

    #[getter]
    fn f_img_path(&self, py: Python<'_>) -> Option<String> {
        self.f_img.as_ref()
            .and_then(|img| img.bind(py).borrow().filepath.clone())
    }

    #[getter]
    fn g_img_path(&self, py: Python<'_>) -> Option<String> {
        self.g_img.as_ref()
            .and_then(|img| img.bind(py).borrow().filepath.clone())
    }

    // -----------------------------------------------------------------------
    // Image-dependent getters (None when images missing after load)
    // -----------------------------------------------------------------------

    #[getter]
    fn f_coords<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner.f_coords.clone().map(|fc| fc.into_pyarray_bound(py))
    }

    #[getter]
    fn f<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.f.clone().map(|f| f.into_pyarray_bound(py))
    }

    #[getter]
    fn f_m(&self) -> Option<f64> { self.inner.f_m }

    #[getter]
    fn delta_f(&self) -> Option<f64> { Some(self.inner.delta_f) }

    #[getter]
    fn grad_f<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner.grad_f.clone().map(|gf| gf.into_pyarray_bound(py))
    }

    #[getter]
    fn sigma_intensity(&self) -> Option<f64> { self.inner.sigma_intensity }

    // -----------------------------------------------------------------------
    // Getters requiring a solve result (None when unsolved)
    // -----------------------------------------------------------------------

    #[getter]
    fn p(&self) -> Option<Vec<f64>> { self.result().map(|r| r.p.clone()) }

    #[getter]
    fn c_zncc(&self) -> Option<f64> { self.result().map(|r| r.c_zncc) }

    #[getter]
    fn c_znssd(&self) -> Option<f64> { self.result().map(|r| r.c_znssd) }

    #[getter]
    fn converged(&self) -> Option<bool> { self.result().map(|r| r.converged) }

    #[getter]
    fn iterations(&self) -> Option<usize> { self.result().map(|r| r.iterations) }

    #[getter]
    fn history(&self) -> Option<Vec<(usize, f64, f64, f64)>> { self.result().map(|r| r.history.clone()) }

    #[getter]
    fn max_norm(&self) -> Option<f64> { self.result().map(|r| r.max_norm) }

    #[getter]
    fn tolerance(&self) -> Option<f64> { self.result().map(|r| r.tolerance) }

    // -----------------------------------------------------------------------
    // __repr__
    // -----------------------------------------------------------------------

    fn __repr__(&self) -> String {
        let shape_str = match &self.inner.mask.shape {
            MaskShape::Circle => "circle",
            MaskShape::Square => "square",
        };
        let tmpl_str = format!("{}({})", shape_str, self.inner.mask.size);
        if let Some(r) = self.result() {
            let p_str = {
                let parts: Vec<String> = r.p.iter().map(|v| format!("{:.6}", v)).collect();
                format!("[{}]", parts.join(", "))
            };
            let zncc_str = format!("{:.5}", r.c_zncc);
            format!(
                "Subset(coord=[{:.2}, {:.2}], n_px={}, template={}, sssig={:.1},\n       solved=True, c_zncc={}, p={})",
                self.inner.coord[0], self.inner.coord[1],
                self.inner.n_px(),
                tmpl_str,
                self.inner.sssig,
                zncc_str,
                p_str,
            )
        } else {
            format!(
                "Subset(coord=[{:.2}, {:.2}], n_px={}, template={}, sssig={:.1}, solved=False)",
                self.inner.coord[0], self.inner.coord[1],
                self.inner.n_px(),
                tmpl_str,
                self.inner.sssig,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySubset>()?;
    Ok(())
}
