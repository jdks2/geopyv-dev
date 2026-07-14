//! PyO3 wrappers for `geopyv_dev::geometry` (utilities, region, meshing).

use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use geopyv_dev::geometry::{meshing, region, utilities};

use crate::Error;

// ===========================================================================
// Utilities — free functions
// ===========================================================================

/// Return the characteristic element length for an equilateral triangle of
/// the given area: ``sqrt(4 * |area| / sqrt(3))``.
#[pyfunction]
pub fn area_to_length(area: f64) -> f64 {
    utilities::area_to_length(area)
}

/// Compute the area of a polygon (Shoelace formula).
///
/// Parameters
/// ----------
/// pts : np.ndarray (N, 2)
///     Vertices as ``[x, y]`` rows (clockwise or counter-clockwise).
#[pyfunction]
pub fn poly_area(pts: PyReadonlyArray2<f64>) -> f64 {
    utilities::poly_area(pts.as_array())
}

/// Test whether three points A, B, C are arranged counter-clockwise.
#[pyfunction]
pub fn ccw(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    utilities::ccw(a, b, c)
}

/// Test whether line segment AB intersects line segment CD.
#[pyfunction]
pub fn intersect(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    utilities::intersect(a, b, c, d)
}

/// Check a 6-node polygon for self-intersection between non-adjacent segments.
///
/// Parameters
/// ----------
/// n : np.ndarray (6, 2)
///     Six ``[x, y]`` polygon vertices.
///
/// Returns
/// -------
/// list[int] or None
///     ``[i, j]`` segment-index pair of the first intersection, or ``None``.
#[pyfunction]
pub fn polysect(n: PyReadonlyArray2<f64>) -> PyResult<Option<[usize; 2]>> {
    let arr = n.as_array();
    if arr.nrows() != 6 {
        return Err(PyValueError::new_err("polysect requires exactly 6 points"));
    }
    let pts: [[f64; 2]; 6] =
        std::array::from_fn(|i| [arr[[i, 0]], arr[[i, 1]]]);
    Ok(utilities::polysect(&pts))
}

/// Compute the centroid of a polygon.
///
/// Note: replicates a typo in the Python source where the y-centroid
/// accumulation uses the x-component of the first vertex instead of y.
/// This is preserved for numerical equivalence.
///
/// Parameters
/// ----------
/// coords : np.ndarray (N, 2)
///     Polygon vertices as ``[x, y]`` rows.
///
/// Returns
/// -------
/// np.ndarray (2,)
#[pyfunction]
pub fn polycentroid<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let c = utilities::polycentroid(coords.as_array());
    ndarray::Array1::from_vec(vec![c[0], c[1]]).into_pyarray_bound(py)
}

/// Compute a plottable triangulation from a mesh element connectivity array.
///
/// Parameters
/// ----------
/// elements : np.ndarray (N, 3 or 6), dtype uint32
///     Element connectivity (node indices).
/// x, y : np.ndarray (M,)
///     Node coordinates.
/// mesh_order : int
///     1 for linear (3-node) or 2 for quadratic (6-node) triangles.
///
/// Returns
/// -------
/// (triangulation, x_path, y_path) — all numpy arrays.
#[pyfunction]
pub fn plot_triangulation<'py>(
    py: Python<'py>,
    elements: PyReadonlyArray2<u32>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    mesh_order: u32,
) -> PyResult<(
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let (tri, xp, yp) = utilities::plot_triangulation(
        elements.as_array(),
        x.as_array(),
        y.as_array(),
        mesh_order,
    );
    Ok((
        tri.into_pyarray_bound(py),
        xp.into_pyarray_bound(py),
        yp.into_pyarray_bound(py),
    ))
}

// ===========================================================================
// Region classes
// ===========================================================================

macro_rules! impl_region_pymethods {
    ($ty:ty, $repr_name:literal) => {
        #[pymethods]
        impl $ty {
            #[getter]
            fn option(&self) -> String {
                self.inner.option.to_string()
            }

            #[getter]
            fn hard(&self) -> bool {
                self.inner.hard
            }

            #[getter]
            fn compensate(&self) -> bool {
                self.inner.compensate
            }

            #[getter]
            fn solved(&self) -> bool {
                self.inner.solved
            }

            #[getter]
            fn counter(&self) -> usize {
                self.inner.counter
            }

            #[getter]
            fn calibrated(&self) -> bool {
                self.inner.calibrated
            }

            #[setter]
            fn set_calibrated(&mut self, val: bool) {
                self.inner.calibrated = val;
            }

            #[getter]
            fn current_nodes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
                self.inner.current_nodes.clone().into_pyarray_bound(py)
            }

            #[setter]
            fn set_current_nodes(&mut self, nodes: PyReadonlyArray2<f64>) {
                self.inner.current_nodes = nodes.as_array().to_owned();
            }

            #[getter]
            fn current_centre<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                ndarray::arr1(&self.inner.current_centre).into_pyarray_bound(py)
            }

            #[setter]
            fn set_current_centre(&mut self, centre: [f64; 2]) {
                self.inner.current_centre = centre;
            }

            #[getter]
            fn history_nodes<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
                self.inner.history_nodes.iter()
                    .map(|n| n.clone().into_pyarray_bound(py))
                    .collect()
            }

            #[setter]
            fn set_history_nodes(&mut self, nodes_list: Vec<PyReadonlyArray2<f64>>) {
                self.inner.history_nodes = nodes_list.iter()
                    .map(|n| n.as_array().to_owned())
                    .collect();
            }

            #[getter]
            fn history_centres(&self) -> Vec<[f64; 2]> {
                self.inner.history_centres.clone()
            }

            #[setter]
            fn set_history_centres(&mut self, centres: Vec<[f64; 2]>) {
                self.inner.history_centres = centres;
            }

            #[setter]
            fn set_counter(&mut self, val: usize) {
                self.inner.counter = val;
            }

            #[setter]
            fn set_ref_index(&mut self, val: Option<usize>) {
                self.inner.ref_index = val;
            }

            fn store_rigid(&mut self, warp: PyReadonlyArray1<f64>) -> PyResult<()> {
                let w: Vec<f64> = warp.as_array().to_vec();
                self.inner.store_rigid(&w).map_err(Error::from)?;
                Ok(())
            }

            fn store_flexible(&mut self, warp: PyReadonlyArray2<f64>) -> PyResult<()> {
                self.inner.store_flexible(warp.as_array()).map_err(Error::from)?;
                Ok(())
            }

            fn update(&mut self, filepath: &str) {
                self.inner.update(filepath);
            }

            fn __repr__(&self) -> String {
                format!(
                    concat!($repr_name, "(n_nodes={}, option='{}')"),
                    self.inner.current_nodes.nrows(),
                    self.inner.option,
                )
            }
        }
    };
}

/// Circular tracked boundary region.
///
/// Parameters
/// ----------
/// centre : array-like [x, y]
/// radius : float, optional (default 50.0)
/// size : float, optional — arc spacing between vertices (default 20.0)
/// option : str, optional — 'D', 'S', 'R', or 'F' (default 'F')
/// hard : bool, optional (default True)
/// compensate : bool, optional (default True)
#[pyclass(name = "CircleRegion")]
pub struct PyCircleRegion {
    inner: region::Region,
}

#[pymethods]
impl PyCircleRegion {
    #[new]
    #[pyo3(signature = (centre, radius=50.0, size=20.0, option="F", hard=true, compensate=true))]
    fn new(
        centre: [f64; 2],
        radius: f64,
        size: f64,
        option: &str,
        hard: bool,
        compensate: bool,
    ) -> PyResult<Self> {
        let opt: region::RegionOption = option.parse().map_err(Error::from)?;
        let r = region::Region::circle(centre, radius, size, opt, hard, compensate)
            .map_err(Error::from)?;
        Ok(PyCircleRegion { inner: r })
    }
}

impl_region_pymethods!(PyCircleRegion, "CircleRegion");

/// Arbitrary-polygon tracked boundary region.
///
/// Parameters
/// ----------
/// nodes : np.ndarray (N, 2)
///     Polygon vertices ``[x, y]``.
/// centre : array-like [x, y], optional
///     If not supplied, computed as the mean of `nodes`.
/// option : str, optional (default 'F')
/// hard : bool, optional (default True)
/// compensate : bool, optional (default True)
/// radius : float, optional (default 25.0) — stored for mesh use.
#[pyclass(name = "PathRegion")]
pub struct PyPathRegion {
    inner: region::Region,
}

#[pymethods]
impl PyPathRegion {
    #[new]
    #[pyo3(signature = (nodes, centre=None, option="F", hard=true, compensate=true, radius=25.0))]
    fn new(
        nodes: PyReadonlyArray2<f64>,
        centre: Option<[f64; 2]>,
        option: &str,
        hard: bool,
        compensate: bool,
        radius: f64,
    ) -> PyResult<Self> {
        let opt: region::RegionOption = option.parse().map_err(Error::from)?;
        let n = nodes.as_array().to_owned();
        let r = region::Region::path(centre, n, opt, hard, compensate, radius)
            .map_err(Error::from)?;
        Ok(PyPathRegion { inner: r })
    }
}

impl_region_pymethods!(PyPathRegion, "PathRegion");

// ===========================================================================
// Extraction helpers (pub(crate) — used by py_mesh and py_sequence)
// ===========================================================================

/// Extract `(nodes, hard)` from a region argument.
/// Accepts PyCircleRegion, PyPathRegion, or a raw numpy array (hard=false).
pub(crate) fn extract_region(obj: &Bound<'_, PyAny>) -> PyResult<(Array2<f64>, bool)> {
    if let Ok(r) = obj.extract::<PyRef<PyCircleRegion>>() {
        return Ok((r.inner.current_nodes.clone(), r.inner.hard));
    }
    if let Ok(r) = obj.extract::<PyRef<PyPathRegion>>() {
        return Ok((r.inner.current_nodes.clone(), r.inner.hard));
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray2<f64>>() {
        return Ok((a.as_array().to_owned(), false));
    }
    Err(PyTypeError::new_err(
        "must be a CircleRegion, PathRegion, or numpy array (N, 2)",
    ))
}

/// Extract a full `Region` (not just `nodes`/`hard`) for `Sequence`-level
/// displacement tracking across reference-image updates.
///
/// `CircleRegion`/`PathRegion` -> clone of `.inner` (preserves the region's
/// tracking `option`). Raw ndarray -> `Region` with `option = S`
/// (static/untracked), which preserves today's frozen-boundary behaviour for
/// plain-array callers (there is no object to track displacement into).
pub(crate) fn extract_region_full(obj: &Bound<'_, PyAny>) -> PyResult<region::Region> {
    if let Ok(r) = obj.extract::<PyRef<PyCircleRegion>>() {
        return Ok(r.inner.clone());
    }
    if let Ok(r) = obj.extract::<PyRef<PyPathRegion>>() {
        return Ok(r.inner.clone());
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray2<f64>>() {
        let nodes = a.as_array().to_owned();
        let region = region::Region::path(None, nodes, region::RegionOption::S, false, false, 0.0)
            .map_err(Error::from)?;
        return Ok(region);
    }
    Err(PyTypeError::new_err(
        "must be a CircleRegion, PathRegion, or numpy array (N, 2)",
    ))
}

/// Write a solved `Region`'s final tracked state back into a Python
/// `CircleRegion`/`PathRegion` object — a no-op if `obj` is a raw ndarray
/// (there's no tracked object to write into).
///
/// Used by `Sequence::solve` so the caller's original region object reflects
/// the displaced boundary/exclusion positions accumulated over the whole run,
/// matching the fact that Python's `Mesh._store_region`/`_update_region`
/// mutate the same `boundary_obj`/`exclusion_objs` instances the caller
/// passed in.
pub(crate) fn write_region_state(obj: &Bound<'_, PyAny>, region: &region::Region) -> PyResult<()> {
    if let Ok(mut r) = obj.extract::<PyRefMut<PyCircleRegion>>() {
        r.inner = region.clone();
        return Ok(());
    }
    if let Ok(mut r) = obj.extract::<PyRefMut<PyPathRegion>>() {
        r.inner = region.clone();
        return Ok(());
    }
    Ok(())
}

// ===========================================================================
// Meshing free functions
// ===========================================================================

/// Build a binary image mask from a boundary polygon and optional exclusions.
///
/// Parameters
/// ----------
/// img_shape : (int, int)
///     ``(height, width)`` of the output mask.
/// boundary_nodes : np.ndarray (N, 2)
///     Boundary polygon ``[x, y]`` vertices.
/// boundary_hard : bool
///     If ``True``, fill inside the boundary polygon.
///     If ``False``, fill the entire image.
/// exclusion_nodes : list[np.ndarray (K, 2)], optional
///     Exclusion polygons whose interiors are zeroed out.
///
/// Returns
/// -------
/// np.ndarray (H, W), dtype uint8
#[pyfunction]
#[pyo3(signature = (img_shape, boundary_nodes, boundary_hard=true, exclusion_nodes=None, exclusions_hard=None))]
pub fn mask_image<'py>(
    py: Python<'py>,
    img_shape: (usize, usize),
    boundary_nodes: PyReadonlyArray2<f64>,
    boundary_hard: bool,
    exclusion_nodes: Option<Vec<PyReadonlyArray2<f64>>>,
    exclusions_hard: Option<Vec<bool>>,
) -> Bound<'py, PyArray2<u8>> {
    let bn = boundary_nodes.as_array();
    let excl_list = exclusion_nodes.unwrap_or_default();
    let excl_owned: Vec<Array2<f64>> = excl_list
        .iter()
        .map(|a| a.as_array().to_owned())
        .collect();
    let excl_views: Vec<ArrayView2<f64>> =
        excl_owned.iter().map(|a: &Array2<f64>| a.view()).collect();
    let excl_hard = exclusions_hard.unwrap_or_else(|| vec![false; excl_owned.len()]);
    let mask = meshing::mask_image(img_shape, bn, boundary_hard, &excl_views, &excl_hard);
    mask.into_pyarray_bound(py)
}

// ===========================================================================
// Module registration
// ===========================================================================

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Utility functions.
    m.add_function(wrap_pyfunction!(area_to_length, m)?)?;
    m.add_function(wrap_pyfunction!(poly_area, m)?)?;
    m.add_function(wrap_pyfunction!(ccw, m)?)?;
    m.add_function(wrap_pyfunction!(intersect, m)?)?;
    m.add_function(wrap_pyfunction!(polysect, m)?)?;
    m.add_function(wrap_pyfunction!(polycentroid, m)?)?;
    m.add_function(wrap_pyfunction!(plot_triangulation, m)?)?;
    // Region classes.
    m.add_class::<PyCircleRegion>()?;
    m.add_class::<PyPathRegion>()?;
    // Meshing functions.
    m.add_function(wrap_pyfunction!(mask_image, m)?)?;
    Ok(())
}
