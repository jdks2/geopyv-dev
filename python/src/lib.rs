mod py_image;
mod py_templates;
mod py_geometry;
mod py_subset;
mod py_mesh;
mod py_particle;
mod py_field;
mod py_sequence;
mod py_io;
mod py_validation;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Newtype that bridges `geopyv_dev::Error` → `PyErr`.
///
/// The orphan rule prevents `impl From<geopyv_dev::Error> for PyErr` directly
/// (neither type is local to this crate). Every PyO3 wrapper method converts
/// with `.map_err(Error::from)?` which chains
/// `geopyv_dev::Error → Error → PyErr` in two cheap steps.
pub(crate) struct Error(pub geopyv_dev::Error);

impl From<geopyv_dev::Error> for Error {
    fn from(e: geopyv_dev::Error) -> Self {
        Error(e)
    }
}

impl From<Error> for PyErr {
    fn from(e: Error) -> Self {
        PyRuntimeError::new_err(e.0.to_string())
    }
}

/// geopyv_dev extension module.
#[pymodule]
fn _geopyv_dev(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_image::register(m)?;
    py_templates::register(m)?;
    py_geometry::register(m)?;
    py_subset::register(m)?;
    py_mesh::register(m)?;
    py_particle::register(m)?;
    py_field::register(m)?;
    py_sequence::register(m)?;
    py_io::register(m)?;
    py_validation::register(m)?;
    Ok(())
}
