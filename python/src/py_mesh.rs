//! PyO3 wrapper for `geopyv_dev::mesh`.
//!
//! Exposes:
//! - `Mesh` class: constructed via `generate()`, runs `solve()`.
//! - `MeshSolution` class: read-only result from `Mesh.solve()`.
//! - Free functions mirroring the pure-math helpers in `mesh.rs`, prefixed
//!   `mesh_` to avoid name collisions at the module level.

use std::collections::HashSet;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
// PyArray1<u32> is used for iterations getter
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use geopyv_dev::mesh::{
    self, Mesh, MeshSolution, SolveConfig, SolveMethod,
};

use crate::{py_image::PyImage, Error};

// ---------------------------------------------------------------------------
// Mesh class
// ---------------------------------------------------------------------------

/// DIC mesh: geometry + reliability-guided solver.
///
/// Construct with :meth:`Mesh.generate`; then call :meth:`Mesh.solve`.
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    inner: Mesh,
}

#[pymethods]
impl PyMesh {
    /// Generate a constrained-Delaunay triangulation and return a `Mesh`.
    ///
    /// Parameters
    /// ----------
    /// borders : numpy.ndarray, shape (N, 2), float64
    ///     Outer boundary polygon vertices.
    /// segments : numpy.ndarray, shape (S, 2), int32
    ///     Constraint segment endpoint indices into `borders`.
    /// curves : list[list[int]]
    ///     Grouped constraint curve indices.
    /// size_lower : float
    ///     Minimum element edge length.
    /// size_upper : float
    ///     Maximum element edge length.
    /// target_nodes : int
    ///     Target node count for binary-search sizing.
    /// mesh_order : int, optional
    ///     1 (linear) or 2 (quadratic). Default 1.
    #[new]
    #[pyo3(signature = (borders, segments, curves, size_lower, size_upper, target_nodes, mesh_order=1))]
    fn new(
        borders: PyReadonlyArray2<f64>,
        segments: PyReadonlyArray2<i32>,
        curves: Vec<Vec<i32>>,
        size_lower: f64,
        size_upper: f64,
        target_nodes: usize,
        mesh_order: u8,
    ) -> PyResult<Self> {
        let b = borders.as_array();
        let s = segments.as_array();
        let m = Mesh::generate(b, s, &curves, size_lower, size_upper, target_nodes, mesh_order)
            .map_err(Error::from)?;
        Ok(PyMesh { inner: m })
    }

    /// Run the reliability-guided DIC solver.
    ///
    /// Parameters
    /// ----------
    /// f_img : Image
    ///     Reference image (pre-computed B-spline data).
    /// g_img : Image
    ///     Target image.
    /// template_coords : numpy.ndarray, shape (n_px, 2), float64
    ///     Subset template pixel offsets.
    /// seed_coord : list[float]
    ///     Image coordinate ``[x, y]`` near a region of low deformation.
    /// seed_warp : list[float]
    ///     Initial warp vector for the seed node (length 6 or 12).
    /// max_norm : float, optional
    ///     Convergence criterion. Default 1e-5.
    /// max_iterations : int, optional
    ///     Iteration limit per node. Default 50.
    /// subset_order : int, optional
    ///     1 (affine) or 2 (quadratic). Default 1.
    /// tolerance : float, optional
    ///     Minimum acceptable C_ZNCC. Default 0.75.
    /// method : str, optional
    ///     ``"icgn"`` (default) or ``"fagn"``.
    ///
    /// Returns
    /// -------
    /// MeshSolution
    #[pyo3(signature = (f_img, g_img, template_coords, seed_coord, seed_warp,
                         max_norm=1e-5, max_iterations=50, subset_order=1,
                         tolerance=0.75, method="icgn"))]
    fn solve(
        &self,
        f_img: PyRef<'_, PyImage>,
        g_img: PyRef<'_, PyImage>,
        template_coords: PyReadonlyArray2<f64>,
        seed_coord: [f64; 2],
        seed_warp: Vec<f64>,
        max_norm: f64,
        max_iterations: usize,
        subset_order: usize,
        tolerance: f64,
        method: &str,
    ) -> PyResult<PyMeshSolution> {
        let solve_method = if method == "fagn" { SolveMethod::Fagn } else { SolveMethod::Icgn };
        let cfg = SolveConfig {
            max_norm,
            max_iterations,
            subset_order,
            tolerance,
            method: solve_method,
        };
        let tc = template_coords.as_array().to_owned();
        let f_path = f_img.filepath.clone();
        let g_path = g_img.filepath.clone();
        let sol = self.inner
            .solve(&f_img.inner, &g_img.inner, &tc, seed_coord, &seed_warp, &cfg)
            .map_err(Error::from)?;
        Ok(PyMeshSolution { inner: sol, f_img_path: f_path, g_img_path: g_path })
    }

    /// Node coordinates, shape ``(N, 2)``.
    #[getter]
    fn nodes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.nodes().clone().into_pyarray_bound(py)
    }

    /// Element connectivity, shape ``(M, 3)`` or ``(M, 6)``.
    #[getter]
    fn elements<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let e: Array2<i64> = self.inner.elements().map(|&x| x as i64);
        e.into_pyarray_bound(py)
    }

    /// Boundary node indices.
    #[getter]
    fn boundary(&self) -> Vec<i64> {
        self.inner.boundary().iter().map(|&x| x as i64).collect()
    }

    /// Exclusion node index groups.
    #[getter]
    fn exclusions(&self) -> Vec<Vec<i64>> {
        self.inner
            .exclusions()
            .iter()
            .map(|g| g.iter().map(|&x| x as i64).collect())
            .collect()
    }

    /// Mesh element order (1 or 2).
    #[getter]
    fn mesh_order(&self) -> u8 {
        self.inner.mesh_order()
    }

    fn __repr__(&self) -> String {
        format!(
            "Mesh(nodes={}, elements={}, order={})",
            self.inner.nodes().nrows(),
            self.inner.elements().nrows(),
            self.inner.mesh_order(),
        )
    }
}

// ---------------------------------------------------------------------------
// MeshSolution class
// ---------------------------------------------------------------------------

/// Read-only DIC solve result from :meth:`Mesh.solve`.
#[pyclass(name = "MeshSolution")]
pub struct PyMeshSolution {
    pub(crate) inner: MeshSolution,
    pub(crate) f_img_path: Option<String>,
    pub(crate) g_img_path: Option<String>,
}

#[pymethods]
impl PyMeshSolution {
    /// Construct a `MeshSolution` directly from arrays.
    ///
    /// Used primarily in tests to build synthetic results without running a
    /// full DIC solve.
    #[new]
    #[pyo3(signature = (nodes, elements, boundary, exclusions,
                         areas, warps, displacements, c_zncc, p,
                         seed_node=0, mesh_order=1, subset_order=1,
                         f_img_path=None, g_img_path=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nodes: PyReadonlyArray2<f64>,
        elements: PyReadonlyArray2<i64>,
        boundary: Vec<i64>,
        exclusions: Vec<Vec<i64>>,
        areas: PyReadonlyArray1<f64>,
        warps: PyReadonlyArray2<f64>,
        displacements: PyReadonlyArray2<f64>,
        c_zncc: PyReadonlyArray1<f64>,
        p: PyReadonlyArray2<f64>,
        seed_node: i64,
        mesh_order: u8,
        subset_order: u8,
        f_img_path: Option<String>,
        g_img_path: Option<String>,
    ) -> Self {
        let n_nodes = nodes.as_array().nrows();
        PyMeshSolution {
            inner: MeshSolution {
                nodes: nodes.as_array().to_owned(),
                elements: elements.as_array().map(|&x| x as usize),
                boundary: boundary.iter().map(|&x| x as usize).collect(),
                exclusions: exclusions
                    .iter()
                    .map(|g| g.iter().map(|&x| x as usize).collect())
                    .collect(),
                areas: areas.as_array().to_owned(),
                warps: warps.as_array().to_owned(),
                displacements: displacements.as_array().to_owned(),
                c_zncc: c_zncc.as_array().to_owned(),
                p: p.as_array().to_owned(),
                seed_node: seed_node as usize,
                mesh_order,
                subset_order,
                iterations: Array1::<u32>::zeros(n_nodes),
                norms: Array1::<f64>::zeros(n_nodes),
            },
            f_img_path,
            g_img_path,
        }
    }

    /// Node coordinates ``(N, 2)``.
    #[getter]
    fn nodes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.nodes.clone().into_pyarray_bound(py)
    }

    /// Element connectivity ``(M, 3)`` or ``(M, 6)``.
    #[getter]
    fn elements<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let e: Array2<i64> = self.inner.elements.map(|&x| x as i64);
        e.into_pyarray_bound(py)
    }

    /// Boundary node indices.
    #[getter]
    fn boundary(&self) -> Vec<i64> {
        self.inner.boundary.iter().map(|&x| x as i64).collect()
    }

    /// Exclusion node index groups.
    #[getter]
    fn exclusions(&self) -> Vec<Vec<i64>> {
        self.inner
            .exclusions
            .iter()
            .map(|g| g.iter().map(|&x| x as i64).collect())
            .collect()
    }

    /// Signed element areas ``(M,)``.
    #[getter]
    fn areas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.areas.clone().into_pyarray_bound(py)
    }

    /// Element warp vectors ``(M, 12)``.
    #[getter]
    fn warps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.warps.clone().into_pyarray_bound(py)
    }

    /// Per-node displacements ``(N, 2)``.
    #[getter]
    fn displacements<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.displacements.clone().into_pyarray_bound(py)
    }

    /// Per-node ZNCC scores ``(N,)``.
    #[getter]
    fn c_zncc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.c_zncc.clone().into_pyarray_bound(py)
    }

    /// Per-node warp parameters ``(N, 6)`` or ``(N, 12)``.
    #[getter]
    fn p<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.p.clone().into_pyarray_bound(py)
    }

    /// Index of the seed node.
    #[getter]
    fn seed_node(&self) -> i64 {
        self.inner.seed_node as i64
    }

    /// Mesh element order.
    #[getter]
    fn mesh_order(&self) -> u8 {
        self.inner.mesh_order
    }

    /// Subset warp order.
    #[getter]
    fn subset_order(&self) -> u8 {
        self.inner.subset_order
    }

    /// Reference image file path, or ``None``.
    #[getter]
    fn f_img_path(&self) -> Option<String> {
        self.f_img_path.clone()
    }

    /// Target image file path, or ``None``.
    #[getter]
    fn g_img_path(&self) -> Option<String> {
        self.g_img_path.clone()
    }

    /// Per-node iteration counts ``(N,)``.
    #[getter]
    fn iterations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.iterations.clone().into_pyarray_bound(py)
    }

    /// Per-node final ∆norm values ``(N,)``.
    #[getter]
    fn norms<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.norms.clone().into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "MeshSolution(nodes={}, elements={}, mesh_order={}, subset_order={})",
            self.inner.nodes.nrows(),
            self.inner.elements.nrows(),
            self.inner.mesh_order,
            self.inner.subset_order,
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions — mirroring Phase 1 fixture function signatures
// ---------------------------------------------------------------------------

/// Signed element areas: ``0.5 * det([[1,x0,y0],[1,x1,y1],[1,x2,y2]])``.
///
/// Parameters
/// ----------
/// nodes : numpy.ndarray, shape (N, 2), float64
/// elements : numpy.ndarray, shape (M, 3) or (M, 6), int64
///
/// Returns
/// -------
/// numpy.ndarray, shape (M,), float64
#[pyfunction]
fn mesh_element_area<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    elements: PyReadonlyArray2<i64>,
) -> Bound<'py, PyArray1<f64>> {
    let n = nodes.as_array().to_owned();
    let e: Array2<usize> = elements.as_array().map(|&x| x as usize);
    mesh::element_area(&n, &e).into_pyarray_bound(py)
}

/// Element warp vectors ``(M, 12)`` from nodal displacements.
///
/// Parameters
/// ----------
/// nodes : numpy.ndarray, shape (N, 2), float64
/// elements : numpy.ndarray, shape (M, 3) or (M, 6), int64
/// displacements : numpy.ndarray, shape (N, 2), float64
/// mesh_order : int
///     1 (linear) or 2 (quadratic).
///
/// Returns
/// -------
/// numpy.ndarray, shape (M, 12), float64
#[pyfunction]
fn mesh_element_strains<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    elements: PyReadonlyArray2<i64>,
    displacements: PyReadonlyArray2<f64>,
    mesh_order: u8,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = nodes.as_array().to_owned();
    let e: Array2<usize> = elements.as_array().map(|&x| x as usize);
    let d = displacements.as_array().to_owned();
    mesh::element_strains(&n, &e, &d, mesh_order)
        .map(|w| w.into_pyarray_bound(py))
        .map_err(|e| PyErr::from(Error::from(e)))
}

/// Shape functions and their derivatives at the element centroid.
///
/// Parameters
/// ----------
/// mesh_order : int
///     1 or 2.
///
/// Returns
/// -------
/// tuple of (N, dN, d2N):
///     N   – shape (3,) or (6,)
///     dN  – shape (2, 3) or (2, 6)
///     d2N – shape (3, 6) for order-2; ``None`` for order-1
#[pyfunction]
fn mesh_shape_function(py: Python<'_>, mesh_order: u8) -> PyResult<Bound<'_, PyTuple>> {
    let (n_vec, dn_vec, d2n_opt) = mesh::shape_function(mesh_order);

    // N
    let n_arr: Array1<f64> = Array1::from(n_vec);
    let n_any = n_arr.into_pyarray_bound(py).into_any();

    // dN
    let dn_r = dn_vec.len();
    let dn_c = if dn_r > 0 { dn_vec[0].len() } else { 0 };
    let dn_flat: Vec<f64> = dn_vec.into_iter().flatten().collect();
    let dn_arr = Array2::from_shape_vec((dn_r, dn_c), dn_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let dn_any = dn_arr.into_pyarray_bound(py).into_any();

    // d2N
    let d2n_any = match d2n_opt {
        Some(d2n_vec) => {
            let r = d2n_vec.len();
            let c = if r > 0 { d2n_vec[0].len() } else { 0 };
            let flat: Vec<f64> = d2n_vec.into_iter().flatten().collect();
            Array2::from_shape_vec((r, c), flat)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                .into_pyarray_bound(py)
                .into_any()
        }
        None => py.None().into_bound(py),
    };

    Ok(PyTuple::new_bound(py, [n_any, dn_any, d2n_any]))
}

/// Node connectivity for a given node index.
///
/// Parameters
/// ----------
/// elements : numpy.ndarray, shape (M, 3) or (M, 6), int64
/// mesh_order : int
/// idx : int
///     Node index.
/// full : bool, optional
///     If True, return all element-sharing nodes. Default False.
///
/// Returns
/// -------
/// list[int]  — sorted neighbour indices
#[pyfunction]
#[pyo3(signature = (elements, mesh_order, idx, full=false))]
fn mesh_connectivity(
    elements: PyReadonlyArray2<i64>,
    mesh_order: u8,
    idx: usize,
    full: bool,
) -> Vec<i64> {
    let e: Array2<usize> = elements.as_array().map(|&x| x as usize);
    mesh::connectivity(&e, mesh_order, idx, full)
        .into_iter()
        .map(|x| x as i64)
        .collect()
}

/// Return the node index closest to ``seed_coord``.
///
/// Parameters
/// ----------
/// nodes : numpy.ndarray, shape (N, 2), float64
/// seed_coord : list[float]
///     ``[x, y]`` query coordinate.
///
/// Returns
/// -------
/// int
#[pyfunction]
fn mesh_find_seed_node(nodes: PyReadonlyArray2<f64>, seed_coord: [f64; 2]) -> i64 {
    let n = nodes.as_array().to_owned();
    mesh::find_seed_node(&n, seed_coord) as i64
}

/// IQR-based outlier detection on C_ZNCC scores.
///
/// Returns indices where ``C_ZNCC < LQ − 2.5 × IQR``.
///
/// Parameters
/// ----------
/// c_zncc : numpy.ndarray, shape (N,), float64
///
/// Returns
/// -------
/// numpy.ndarray of int64, outlier indices
#[pyfunction]
fn mesh_corr<'py>(
    py: Python<'py>,
    c_zncc: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<i64>> {
    let vals = c_zncc.as_array().to_owned();
    let ids = mesh::corr(vals.view());
    let ids_i64: Vec<i64> = ids.into_iter().map(|x| x as i64).collect();
    Array1::from(ids_i64).into_pyarray_bound(py)
}

/// Flow score: dot product between node displacement direction and mean neighbour direction.
///
/// Parameters
/// ----------
/// idx : int
///     Node index.
/// displacements : numpy.ndarray, shape (N, 2), float64
/// elements : numpy.ndarray, shape (M, 3) or (M, 6), int64
/// mesh_order : int
/// exclude : list[int], optional
///     Neighbour indices to exclude. Default ``[]``.
/// displacement : list[float] or numpy array (2,), optional
///     Override for the node's own displacement. Default uses ``displacements[idx]``.
///
/// Returns
/// -------
/// float  (−1 if no valid neighbours)
#[pyfunction]
#[pyo3(signature = (idx, displacements, elements, mesh_order, exclude=None, displacement=None))]
fn mesh_flow_calc(
    idx: usize,
    displacements: PyReadonlyArray2<f64>,
    elements: PyReadonlyArray2<i64>,
    mesh_order: u8,
    exclude: Option<Vec<usize>>,
    displacement: Option<Vec<f64>>,
) -> f64 {
    let e: Array2<usize> = elements.as_array().map(|&x| x as usize);
    let d = displacements.as_array();
    let excl: HashSet<usize> = exclude.unwrap_or_default().into_iter().collect();
    let disp_override: Option<[f64; 2]> =
        displacement.map(|v| [v[0], v[1]]);
    mesh::flow_calc(idx, &e, mesh_order, d, &excl, disp_override)
}

/// Displacement magnitude: ``sqrt(dx² + dy²)``.
///
/// Parameters
/// ----------
/// displacement : list[float]
///     ``[dx, dy]``.
///
/// Returns
/// -------
/// float
#[pyfunction]
fn mesh_r_calc(displacement: [f64; 2]) -> f64 {
    mesh::r_calc(displacement)
}

/// Compute adaptive target areas from shear-strain × area products.
///
/// ``D[e] = |warps[e,3] + warps[e,4]| * |areas[e]|``
///
/// ``target[e] = areas[e] * clip(D[e] / mean(D), alpha, 1/alpha)^-2``
///
/// Parameters
/// ----------
/// warps : numpy.ndarray, shape (M, 12), float64
/// areas : numpy.ndarray, shape (M,), float64
/// alpha : float
///
/// Returns
/// -------
/// numpy.ndarray, shape (M,), float64
#[pyfunction]
fn mesh_adaptive_target_areas<'py>(
    py: Python<'py>,
    warps: PyReadonlyArray2<f64>,
    areas: PyReadonlyArray1<f64>,
    alpha: f64,
) -> Bound<'py, PyArray1<f64>> {
    let w = warps.as_array().to_owned();
    let a = areas.as_array().to_owned();
    mesh::adaptive_target_areas(&w, &a, alpha).into_pyarray_bound(py)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMesh>()?;
    m.add_class::<PyMeshSolution>()?;
    m.add_function(wrap_pyfunction!(mesh_element_area, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_element_strains, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_shape_function, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_find_seed_node, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_corr, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_flow_calc, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_r_calc, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_adaptive_target_areas, m)?)?;
    Ok(())
}
