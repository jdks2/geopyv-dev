use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};

use geopyv_dev::validation;

use crate::Error;

// ---------------------------------------------------------------------------
// anomalies
// ---------------------------------------------------------------------------

/// Remove the ``skim`` particles with the largest displacement error from each
/// frame.
///
/// Parameters
/// ----------
/// applied : ndarray, shape (n_frames, n_particles, 12)
///     Ground-truth warp vectors.
/// observed : ndarray, shape (n_frames, n_particles, 12)
///     DIC-measured warp vectors.
/// skim : int
///     Number of outlier particles to remove per frame.
///
/// Returns
/// -------
/// tuple[ndarray, ndarray]
///     ``(applied_trimmed, observed_trimmed)`` each of shape
///     ``(n_frames, n_particles - skim, 12)``.
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
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(anomalies, m)?)?;
    Ok(())
}
