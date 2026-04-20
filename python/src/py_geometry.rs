//! PyO3 wrappers for `geopyv_dev::geometry` (utilities, region, meshing).

use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

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

fn option_from_str(s: &str) -> PyResult<region::RegionOption> {
    match s {
        "D" => Ok(region::RegionOption::D),
        "S" => Ok(region::RegionOption::S),
        "R" => Ok(region::RegionOption::R),
        "F" => Ok(region::RegionOption::F),
        other => Err(PyValueError::new_err(format!(
            "option must be 'D', 'S', 'R', or 'F'; got '{other}'"
        ))),
    }
}

fn option_to_str(opt: &region::RegionOption) -> &'static str {
    match opt {
        region::RegionOption::D => "D",
        region::RegionOption::S => "S",
        region::RegionOption::R => "R",
        region::RegionOption::F => "F",
    }
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
        let opt = option_from_str(option)?;
        let r = region::Region::circle(centre, radius, size, opt, hard, compensate)
            .map_err(Error::from)?;
        Ok(PyCircleRegion { inner: r })
    }

    #[getter]
    fn option(&self) -> &str {
        option_to_str(&self.inner.option)
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
    fn current_nodes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.current_nodes.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn current_centre<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.current_centre.clone().into_pyarray_bound(py)
    }

    /// Store a rigid-body update. `warp` is a 1-D array with ≥ 5 elements
    /// ``[u, v, _, du_dx, du_dy, ...]``.
    fn store_rigid(&mut self, warp: PyReadonlyArray1<f64>) -> PyResult<()> {
        let w: Vec<f64> = warp.as_array().to_vec();
        self.inner.store_rigid(&w).map_err(Error::from)?;
        Ok(())
    }

    /// Store a flexible update. `warp` is a 2-D array of shape (N, 2).
    fn store_flexible(&mut self, warp: PyReadonlyArray2<f64>) -> PyResult<()> {
        let w = warp.as_array().to_owned();
        self.inner.store_flexible(&w).map_err(Error::from)?;
        Ok(())
    }

    /// Update working nodes/centre from history using the frame index in `filepath`.
    fn update(&mut self, filepath: &str) {
        self.inner.update(filepath);
    }

    fn __repr__(&self) -> String {
        format!(
            "CircleRegion(n_nodes={}, option='{}')",
            self.inner.current_nodes.nrows(),
            option_to_str(&self.inner.option)
        )
    }
}

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
        let opt = option_from_str(option)?;
        let n = nodes.as_array().to_owned();
        let r = region::Region::path(centre, n, opt, hard, compensate, radius)
            .map_err(Error::from)?;
        Ok(PyPathRegion { inner: r })
    }

    #[getter]
    fn option(&self) -> &str {
        option_to_str(&self.inner.option)
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
    fn current_nodes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.current_nodes.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn current_centre<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.current_centre.clone().into_pyarray_bound(py)
    }

    fn store_rigid(&mut self, warp: PyReadonlyArray1<f64>) -> PyResult<()> {
        let w: Vec<f64> = warp.as_array().to_vec();
        self.inner.store_rigid(&w).map_err(Error::from)?;
        Ok(())
    }

    fn store_flexible(&mut self, warp: PyReadonlyArray2<f64>) -> PyResult<()> {
        let w = warp.as_array().to_owned();
        self.inner.store_flexible(&w).map_err(Error::from)?;
        Ok(())
    }

    fn update(&mut self, filepath: &str) {
        self.inner.update(filepath);
    }

    fn __repr__(&self) -> String {
        format!(
            "PathRegion(n_nodes={}, option='{}')",
            self.inner.current_nodes.nrows(),
            option_to_str(&self.inner.option)
        )
    }
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
#[pyo3(signature = (img_shape, boundary_nodes, boundary_hard=true, exclusion_nodes=None))]
pub fn mask_image<'py>(
    py: Python<'py>,
    img_shape: (usize, usize),
    boundary_nodes: PyReadonlyArray2<f64>,
    boundary_hard: bool,
    exclusion_nodes: Option<Vec<PyReadonlyArray2<f64>>>,
) -> Bound<'py, PyArray2<u8>> {
    let bn = boundary_nodes.as_array();
    let excl_owned: Vec<Array2<f64>> = exclusion_nodes
        .unwrap_or_default()
        .iter()
        .map(|a| a.as_array().to_owned())
        .collect();
    let excl_views: Vec<ArrayView2<f64>> =
        excl_owned.iter().map(|a: &Array2<f64>| a.view()).collect();
    let mask = meshing::mask_image(img_shape, bn, boundary_hard, &excl_views);
    mask.into_pyarray_bound(py)
}

/// Prepare segment and curve data for the mesh generator.
///
/// Parameters
/// ----------
/// boundary_nodes : np.ndarray (N, 2)
/// boundary_hard : bool, optional (default True)
/// exclusion_nodes : list[np.ndarray], optional
/// img_shape : (int, int) or None, optional
///     If supplied, a binary mask is also returned.
///
/// Returns
/// -------
/// tuple: (borders, segments, curves[, mask])
///   borders  : np.ndarray (M, 2), float64
///   segments : np.ndarray (M, 2), int32
///   curves   : list[list[int]]
///   mask     : np.ndarray (H, W) uint8  — only present when img_shape given
#[pyfunction]
#[pyo3(signature = (boundary_nodes, boundary_hard=true, exclusion_nodes=None, img_shape=None))]
pub fn define_roi<'py>(
    py: Python<'py>,
    boundary_nodes: PyReadonlyArray2<f64>,
    boundary_hard: bool,
    exclusion_nodes: Option<Vec<PyReadonlyArray2<f64>>>,
    img_shape: Option<(usize, usize)>,
) -> PyResult<PyObject> {
    let bn = boundary_nodes.as_array();
    let excl_owned: Vec<Array2<f64>> = exclusion_nodes
        .unwrap_or_default()
        .iter()
        .map(|a| a.as_array().to_owned())
        .collect();
    let excl_views: Vec<ArrayView2<f64>> =
        excl_owned.iter().map(|a: &Array2<f64>| a.view()).collect();
    let roi = meshing::define_roi(bn, boundary_hard, &excl_views, img_shape);

    use pyo3::IntoPy;
    let borders_py = roi.borders.into_pyarray_bound(py).into_py(py);
    let segments_py = roi.segments.into_pyarray_bound(py).into_py(py);
    let curves_py: PyObject = roi.curves.into_py(py);

    if let Some(mask) = roi.mask {
        let mask_py: PyObject = mask.into_pyarray_bound(py).into_py(py);
        Ok(PyTuple::new_bound(py, [borders_py, segments_py, curves_py, mask_py]).into_py(py))
    } else {
        Ok(PyTuple::new_bound(py, [borders_py, segments_py, curves_py]).into_py(py))
    }
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
    m.add_function(wrap_pyfunction!(define_roi, m)?)?;
    Ok(())
}
