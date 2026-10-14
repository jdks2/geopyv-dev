use std::sync::Arc;
use ndarray::{Array1, Array2};
use numpy::{Element, PyArray1, PyArray2};
use pyo3::{prelude::*, types::PyCapsule};

pub fn arc_array1<'py, T, E, F>(
    py: Python<'py>,
    arc: &Arc<T>,
    getter: F,
) -> PyResult<Bound<'py, PyArray1<E>>>
where
    T: 'static + Send + Sync,
    E: Element,
    F: FnOnce(&T) -> &Array1<E>,
{
    let array_ref = getter(arc.as_ref());
    let capsule = PyCapsule::new_bound(py, Arc::clone(arc), None)?;
    // SAFETY: capsule holds Arc::clone keeping the allocation alive past the array lifetime.
    // T is never mutated after Arc creation, so no aliased-mutable reference exists,
    // including under GIL-released access by numpy C extensions.
    Ok(unsafe { PyArray1::borrow_from_array_bound(array_ref, capsule.into_any()) })
}

pub fn arc_array2<'py, T, E, F>(
    py: Python<'py>,
    arc: &Arc<T>,
    getter: F,
) -> PyResult<Bound<'py, PyArray2<E>>>
where
    T: 'static + Send + Sync,
    E: Element,
    F: FnOnce(&T) -> &Array2<E>,
{
    let array_ref = getter(arc.as_ref());
    let capsule = PyCapsule::new_bound(py, Arc::clone(arc), None)?;
    // SAFETY: same invariant as arc_array1 — see above.
    Ok(unsafe { PyArray2::borrow_from_array_bound(array_ref, capsule.into_any()) })
}

/// A Python `time=` argument: `None`, a scalar time increment (seconds
/// between consecutive images), or one time per image (any sequence).
pub fn extract_time_input(
    time: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<geopyv_dev::timing::TimeInput>> {
    use geopyv_dev::timing::TimeInput;
    let Some(t) = time.filter(|t| !t.is_none()) else { return Ok(None) };
    if t.is_instance_of::<pyo3::types::PyBool>() {
        return Err(pyo3::exceptions::PyTypeError::new_err("time must be a number or a sequence of numbers"));
    }
    if let Ok(dt) = t.extract::<f64>() {
        return Ok(Some(TimeInput::Increment(dt)));
    }
    t.extract::<Vec<f64>>().map(|v| Some(TimeInput::Times(v))).map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "time must be a time increment (number) or one time per image (sequence of numbers)",
        )
    })
}

/// `"leapfrog"` / `"sequential"` as an [`IncrementMode`](geopyv_dev::timing::IncrementMode).
pub fn parse_mode(mode: &str) -> PyResult<geopyv_dev::timing::IncrementMode> {
    mode.parse().map_err(|e: geopyv_dev::Error| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
