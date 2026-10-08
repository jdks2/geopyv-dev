//! Desktop GUI launcher — exposes `geopyv_gui::run` so the Python package can
//! ship the `geopyv-gui` console script without a separate binary.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Open the geopyv desktop GUI and block until its window is closed.
///
/// Releases the GIL while the window is open. Must be called from the main
/// thread, at most once per process. Normally launched via the
/// ``geopyv-gui`` command rather than called directly.
#[pyfunction]
pub fn run_gui(py: Python<'_>) -> PyResult<()> {
    py.allow_threads(|| geopyv_gui::run().map_err(|e| e.to_string()))
        .map_err(|e| PyRuntimeError::new_err(format!("geopyv-gui failed: {e}")))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_gui, m)?)?;
    Ok(())
}
