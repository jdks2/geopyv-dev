use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

use geopyv_dev::validation::{self, Validation, ValidationFieldData, ValidationSolution};

use crate::{py_field::PyField, py_speckle::PySpeckle, Error};

// ---------------------------------------------------------------------------
// anomalies (free function — unchanged)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn anomalies<'py>(
    py: Python<'py>,
    applied: PyReadonlyArray3<'py, f64>,
    observed: PyReadonlyArray3<'py, f64>,
    skim: usize,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<f64>>)> {
    let result = validation::anomalies(
        applied.as_array(),
        observed.as_array(),
        skim,
    )
    .map_err(Error::from)?;
    Ok((
        result.applied.into_pyarray_bound(py),
        result.observed.into_pyarray_bound(py),
    ))
}

// ---------------------------------------------------------------------------
// PyValidationFieldData
// ---------------------------------------------------------------------------

#[pyclass(name = "ValidationFieldData")]
pub struct PyValidationFieldData {
    pub(crate) inner: ValidationFieldData,
}

#[pymethods]
impl PyValidationFieldData {
    #[getter]
    fn applied<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.result.applied.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn observed<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.result.observed.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.coordinates.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn image_0_path(&self) -> Option<String> {
        self.inner.image_0_path.clone()
    }

    fn __repr__(&self) -> String {
        let s = self.inner.result.applied.shape();
        format!("ValidationFieldData(n_frames={}, n_particles={})", s[0], s[1])
    }
}

// ---------------------------------------------------------------------------
// PyValidationSolution
// ---------------------------------------------------------------------------

#[pyclass(name = "ValidationSolution")]
pub struct PyValidationSolution {
    pub(crate) inner: ValidationSolution,
}

#[pymethods]
impl PyValidationSolution {
    #[getter]
    fn fields(&self) -> Vec<PyValidationFieldData> {
        self.inner.fields.iter().map(|f| PyValidationFieldData { inner: f.clone() }).collect()
    }

    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner.labels.clone()
    }

    #[getter]
    fn pm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.pm.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn mult<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.mult.clone().into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "ValidationSolution(n_fields={}, image_no={})",
            self.inner.fields.len(),
            self.inner.mult.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyValidation
// ---------------------------------------------------------------------------

#[pyclass(name = "Validation")]
pub struct PyValidation {
    inner: Validation,
}

#[pymethods]
impl PyValidation {
    #[new]
    fn new(
        speckle: &Bound<'_, PySpeckle>,
        field_solutions: Vec<Bound<'_, PyField>>,
        labels: Vec<String>,
    ) -> PyResult<Self> {
        let speckle_arc = Arc::clone(&speckle.borrow().inner);
        let field_sols: Result<Vec<_>, _> = field_solutions
            .iter()
            .map(|f| {
                f.borrow().inner.solution().cloned().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(
                        "Field has not been solved; call solve() first",
                    )
                })
            })
            .collect();
        let validation = Validation::new(speckle_arc, field_sols?, labels)
            .map_err(Error::from)?;
        Ok(PyValidation { inner: validation })
    }

    #[pyo3(signature = (cumulative=true, skim=None))]
    fn solve(&self, cumulative: bool, skim: Option<usize>) -> PyResult<PyValidationSolution> {
        let sol = self.inner.solve(cumulative, skim).map_err(Error::from)?;
        Ok(PyValidationSolution { inner: sol })
    }

    fn __repr__(&self) -> String {
        "Validation()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(anomalies, m)?)?;
    m.add_class::<PyValidationFieldData>()?;
    m.add_class::<PyValidationSolution>()?;
    m.add_class::<PyValidation>()?;
    Ok(())
}
