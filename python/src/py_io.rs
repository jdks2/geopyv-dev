use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use geopyv_dev::io::{self, GeopyvObject};

use std::sync::Arc;

use crate::{
    py_calibration::PyCalibrationSolution,
    py_field::PyField,
    py_mesh::PyMesh,
    py_particle::PyParticle,
    py_sequence::PySequence,
    py_speckle::PySpeckle,
    py_subset::PySubset,
    Error,
};

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------

/// Serialise a ``Subset``, ``Mesh``, ``Sequence``, ``Particle``, or ``Field``
/// to a ``.pyv`` file.
///
/// The file format is: 4-byte magic ``b"GPYV"`` + 1-byte version ``0x01`` +
/// bincode v2 payload. This format is intentionally incompatible with the
/// Python ``pickle``-based format used by the original ``geopyv`` package.
#[pyfunction]
pub fn save(path: &str, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = obj.py();
    let gobj = if let Ok(m) = obj.extract::<PyRef<PyMesh>>() {
        let sol = m.inner.solution().ok_or_else(|| {
            PyRuntimeError::new_err("Mesh has not been solved; cannot save.")
        })?;
        GeopyvObject::Mesh((**sol).clone())
    } else if let Ok(s) = obj.extract::<PyRef<PySequence>>() {
        let sol = s.inner.solution().ok_or_else(|| {
            PyRuntimeError::new_err("Sequence has not been solved; cannot save.")
        })?;
        GeopyvObject::Sequence(sol.clone())
    } else if let Ok(s) = obj.extract::<PyRef<PySubset>>() {
        GeopyvObject::Subset(s.to_subset_solution(py)?)
    } else if let Ok(p) = obj.extract::<PyRef<PyParticle>>() {
        let sol = p.inner.solution().ok_or_else(|| {
            PyRuntimeError::new_err("Particle has not been solved; cannot save.")
        })?;
        GeopyvObject::Particle((**sol).clone())
    } else if let Ok(f) = obj.extract::<PyRef<PyField>>() {
        let sol = f.inner.solution().ok_or_else(|| {
            PyRuntimeError::new_err("Field has not been solved; cannot save.")
        })?;
        GeopyvObject::Field(sol.clone())
    } else if let Ok(s) = obj.extract::<PyRef<PySpeckle>>() {
        GeopyvObject::Speckle((*s.inner).clone())
    } else {
        return Err(PyRuntimeError::new_err(
            "expected Subset, Mesh, Sequence, Particle, Field, or Speckle",
        ));
    };
    io::save(path, &gobj).map_err(Error::from)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

/// Load a ``Subset``, ``Mesh``, ``Sequence``, ``Particle``, or ``Field`` from a
/// ``.pyv`` file.
///
/// Returns the appropriate Python object depending on the type tag embedded in
/// the file.
///
/// Raises ``RuntimeError`` if the file does not have the expected magic header,
/// uses an unsupported version, or the data is malformed.
#[pyfunction]
pub fn load(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let obj = io::load(path).map_err(Error::from)?;
    match obj {
        GeopyvObject::Mesh(m) => Ok(Py::new(py, PyMesh::from_solution(py, Arc::new(m))?)?.into_py(py)),
        GeopyvObject::Field(f) => Ok(Py::new(py, PyField::from_solution(f)?)?.into_py(py)),
        GeopyvObject::Sequence(s) => Ok(Py::new(py, PySequence::from_solution(s)?)?.into_py(py)),
        GeopyvObject::Subset(s) => {
            Ok(Py::new(py, PySubset::from_solution(py, s)?)?.into_py(py))
        }
        GeopyvObject::Particle(p) => Ok(Py::new(py, PyParticle::from_solution(p)?)?.into_py(py)),
        GeopyvObject::Speckle(s) => Ok(Py::new(py, PySpeckle { inner: Arc::new(s) })?.into_py(py)),
        GeopyvObject::Calibration(c) => {
            Ok(Py::new(py, PyCalibrationSolution { inner: c })?.into_py(py))
        }
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    Ok(())
}
