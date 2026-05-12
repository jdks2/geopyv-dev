use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use geopyv_dev::io::{self, GeopyvObject};

use crate::{
    py_field::PyFieldSolution,
    py_mesh::PyMesh,
    py_sequence::PySequenceSolution,
    py_subset::PySubset,
    Error,
};

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------

/// Serialise a ``Subset``, ``Mesh``, ``FieldSolution``, or
/// ``SequenceSolution`` to a ``.pyv`` file.
///
/// The file format is: 4-byte magic ``b"GPYV"`` + 1-byte version ``0x01`` +
/// bincode v2 payload. This format is intentionally incompatible with the
/// Python ``pickle``-based format used by the original ``geopyv`` package.
#[pyfunction]
pub fn save(path: &str, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = obj.py();
    let gobj = if let Ok(m) = obj.extract::<PyRef<PyMesh>>() {
        if m.solution.is_none() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Mesh has not been solved; cannot save.",
            ));
        }
        GeopyvObject::Mesh(m.solution.clone().unwrap())
    } else if let Ok(f) = obj.extract::<PyRef<PyFieldSolution>>() {
        GeopyvObject::Field(f.inner.clone())
    } else if let Ok(s) = obj.extract::<PyRef<PySequenceSolution>>() {
        GeopyvObject::Sequence(s.inner.clone())
    } else if let Ok(s) = obj.extract::<PyRef<PySubset>>() {
        if s.result.is_none() {
            return Err(PyRuntimeError::new_err(
                "Subset has not been solved; cannot save.",
            ));
        }
        GeopyvObject::Subset(s.to_subset_solution(py)?)
    } else {
        return Err(PyRuntimeError::new_err(
            "expected Subset, Mesh, FieldSolution, or SequenceSolution",
        ));
    };
    io::save(path, &gobj).map_err(Error::from)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

/// Load a ``Subset``, ``Mesh``, ``FieldSolution``, or
/// ``SequenceSolution`` from a ``.pyv`` file.
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
        GeopyvObject::Mesh(m) => Ok(Py::new(py, PyMesh::from_solution(py, m)?)?.into_py(py)),
        GeopyvObject::Field(f) => Ok(PyFieldSolution { inner: f }.into_py(py)),
        GeopyvObject::Sequence(s) => Ok(PySequenceSolution { inner: s }.into_py(py)),
        GeopyvObject::Subset(s) => {
            Ok(Py::new(py, PySubset::from_solution(py, s)?)?.into_py(py))
        }
        GeopyvObject::Particle(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "loading Particle objects is not yet supported via Python",
        )),
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
