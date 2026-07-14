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
