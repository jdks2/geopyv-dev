//! PyO3 wrapper for `geopyv_dev::mesh`.
//!
//! Exposes:
//! - `Mesh` class: single mutable object, constructed with images, mutated in
//!   place by `solve()` which returns `None`.
//! - Free functions mirroring the pure-math helpers in `mesh.rs`, prefixed
//!   `mesh_` to avoid name collisions at the module level.
//! - `MeshSolution` class: retained as the serialisation boundary (IO tests
//!   and `py_io.rs` use `MeshSolution`), but removed from `register()` so it
//!   is not user-visible.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use geopyv_dev::image::Image;
use geopyv_dev::io::{save as io_save, GeopyvObject};
use geopyv_dev::mesh::{
    self, Mesh, MeshSolution, SeedConfig, SolveConfig, SolveMethod,
};

use crate::{
    py_geometry::extract_region,
    py_image::PyImage,
    py_mask::PyMask,
    utils::{arc_array1, arc_array2},
    Error,
};

// ---------------------------------------------------------------------------
// Mesh class
// ---------------------------------------------------------------------------

/// DIC mesh: geometry + reliability-guided solver.
///
/// Construct with ``Mesh(boundary, target_nodes, f_img, g_img, ...)``;
/// then call ``mesh.solve(template, seed_coord)`` which mutates in place.
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    pub(crate) inner: Mesh,
    pub(crate) f_img: Option<Py<PyImage>>,
    pub(crate) g_img: Option<Py<PyImage>>,
    pub(crate) solution: Option<Arc<MeshSolution>>,
}

#[pymethods]
impl PyMesh {
    /// Construct a mesh: triangulate the boundary, compute the binary mask,
    /// and store the reference and target images.
    ///
    /// Parameters
    /// ----------
    /// boundary : CircleRegion, PathRegion, or numpy.ndarray (N, 2)
    ///     Boundary region. Raw arrays imply ``boundary_hard=False``.
    /// target_nodes : int
    ///     Target node count for binary-search sizing.
    /// f_img : Image
    ///     Reference image.
    /// g_img : Image
    ///     Target image.
    /// size : (float, float), optional
    ///     ``(size_lower, size_upper)`` element edge lengths. Default ``(1.0, 1000.0)``.
    /// exclusions : list[CircleRegion | PathRegion | numpy.ndarray], optional
    ///     Exclusion regions. Default ``None``.
    /// exclusions_hard : list[bool], optional
    ///     Per-exclusion hard flag. Default all ``True``.
    /// mesh_order : int, optional
    ///     1 (linear) or 2 (quadratic). Default 2.
    #[new]
    #[pyo3(signature = (boundary, target_nodes, f_img, g_img,
                         size=(1.0, 1000.0), exclusions=None, exclusions_hard=None,
                         mesh_order=2))]
    fn new(
        _py: Python<'_>,
        boundary: &Bound<'_, PyAny>,
        target_nodes: usize,
        f_img: &Bound<'_, PyImage>,
        g_img: &Bound<'_, PyImage>,
        size: (f64, f64),
        exclusions: Option<Vec<Bound<'_, PyAny>>>,
        exclusions_hard: Option<Vec<bool>>,
        mesh_order: u8,
    ) -> PyResult<Self> {
        let (boundary_nodes, boundary_hard) = extract_region(boundary)?;
        let excl_owned: Vec<Array2<f64>> = exclusions
            .unwrap_or_default()
            .iter()
            .map(|obj| extract_region(obj).map(|(nodes, _)| nodes))
            .collect::<PyResult<Vec<_>>>()?;
        let n_excl = excl_owned.len();
        let excl_views: Vec<_> = excl_owned.iter().map(|a| a.view()).collect();
        let excl_hard: Vec<bool> = exclusions_hard.unwrap_or_else(|| vec![true; n_excl]);

        let f_py = f_img.clone().unbind();
        let g_py = g_img.clone().unbind();
        let f_ref = f_img.borrow();
        let g_ref = g_img.borrow();
        let f_arc = Arc::clone(&f_ref.inner);
        let g_arc = Arc::clone(&g_ref.inner);

        let inner = Mesh::new(
            boundary_nodes.view(),
            boundary_hard,
            &excl_views,
            &excl_hard,
            size,
            target_nodes,
            mesh_order,
            f_arc,
            g_arc,
        )
        .map_err(Error::from)?;

        Ok(PyMesh {
            inner,
            f_img: Some(f_py),
            g_img: Some(g_py),
            solution: None,
        })
    }

    /// Run the reliability-guided DIC solver. Mutates in place; returns ``None``.
    ///
    /// Parameters
    /// ----------
    /// template : Template
    ///     Subset template whose pixel offsets define the subset shape.
    /// seed_coord : list[float]
    ///     Image coordinate ``[x, y]`` near a region of low deformation.
    /// seed_warp : list[float], optional
    ///     Initial warp vector for the seed node. Defaults to zeros.
    /// max_norm : float, optional
    ///     Convergence criterion. Default 1e-5.
    /// max_iterations : int, optional
    ///     Iteration limit per node. Default 50.
    /// subset_order : int, optional
    ///     1 (affine) or 2 (quadratic). Default 2.
    /// tolerance : float, optional
    ///     Minimum acceptable C_ZNCC for propagated nodes. Default 0.75.
    /// seed_tolerance : float, optional
    ///     Minimum acceptable C_ZNCC for the seed node. Default 0.9.
    /// method : str, optional
    ///     ``"icgn"`` (default) or ``"fagn"``.
    #[pyo3(signature = (local_mask, seed_coord, seed_warp=None,
                        max_norm=1e-5, max_iterations=50, subset_order=2,
                        tolerance=0.75, seed_tolerance=0.9, method="icgn"))]
    fn solve(
        &mut self,
        _py: Python<'_>,
        local_mask: &Bound<'_, PyAny>,
        seed_coord: [f64; 2],
        seed_warp: Option<Vec<f64>>,
        max_norm: f64,
        max_iterations: usize,
        subset_order: usize,
        tolerance: f64,
        seed_tolerance: f64,
        method: &str,
    ) -> PyResult<()> {
        let tmpl = local_mask
            .extract::<PyRef<'_, PyMask>>()
            .map_err(|_| PyTypeError::new_err("local_mask must be a Mask"))?;
        let local_mask_ref = tmpl.local_mask_ref()?;
        let p_len = 6 * subset_order;
        let warp = seed_warp.unwrap_or_else(|| vec![0.0; p_len]);
        let seed = SeedConfig {
            coord: seed_coord,
            warp,
            tolerance: seed_tolerance,
        };
        let solve_method = if method == "fagn" { SolveMethod::Fagn } else { SolveMethod::Icgn };
        let cfg = SolveConfig {
            max_norm,
            max_iterations,
            subset_order,
            tolerance,
            method: solve_method,
        };
        let sol = self.inner.solve(local_mask_ref, &seed, &cfg, None).map_err(Error::from)?;
        self.solution = Some(Arc::new(sol));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Geometry getters — always available
    // -----------------------------------------------------------------------

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

    /// Reference image used at construction, or ``None``.
    #[getter]
    fn f_img(&self, py: Python<'_>) -> Option<Py<PyImage>> {
        self.f_img.as_ref().map(|img| img.clone_ref(py))
    }

    /// Target image used at construction, or ``None``.
    #[getter]
    fn g_img(&self, py: Python<'_>) -> Option<Py<PyImage>> {
        self.g_img.as_ref().map(|img| img.clone_ref(py))
    }

    /// ``True`` once ``solve()`` has been called successfully.
    #[getter]
    fn solved(&self) -> bool {
        self.solution.is_some()
    }

    // -----------------------------------------------------------------------
    // Solve-result getters — raise AttributeError if not yet solved
    // -----------------------------------------------------------------------

    /// Signed element areas ``(M,)``.
    #[getter]
    fn areas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        arc_array1(py, self.require_solved_arc()?, |s| &s.areas)
    }

    /// Element warp vectors ``(M, 12)``.
    #[getter]
    fn warps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arc_array2(py, self.require_solved_arc()?, |s| &s.warps)
    }

    /// Per-node displacements ``(N, 2)``.
    #[getter]
    fn displacements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arc_array2(py, self.require_solved_arc()?, |s| &s.displacements)
    }

    /// Per-node ZNCC scores ``(N,)``.
    #[getter]
    fn c_zncc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        arc_array1(py, self.require_solved_arc()?, |s| &s.c_zncc)
    }

    /// Per-node warp parameters ``(N, 6)`` or ``(N, 12)``.
    #[getter]
    fn p<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arc_array2(py, self.require_solved_arc()?, |s| &s.p)
    }

    /// Index of the seed node.
    #[getter]
    fn seed_node(&self) -> PyResult<i64> {
        Ok(self.require_solved()?.seed_node as i64)
    }

    /// Subset warp order (1 or 2).
    #[getter]
    fn subset_order(&self) -> PyResult<u8> {
        Ok(self.require_solved()?.subset_order)
    }

    /// Per-node iteration counts ``(N,)``.
    #[getter]
    fn iterations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        arc_array1(py, self.require_solved_arc()?, |s| &s.iterations)
    }

    /// Per-node final ∆norm values ``(N,)``.
    #[getter]
    fn norms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        arc_array1(py, self.require_solved_arc()?, |s| &s.norms)
    }

    /// Reference image file path, or ``None`` if not set.
    #[getter]
    fn f_img_path(&self) -> PyResult<Option<String>> {
        let s = self.require_solved()?.f_img_path.to_string_lossy().into_owned();
        Ok(if s.is_empty() { None } else { Some(s) })
    }

    /// Target image file path, or ``None`` if not set.
    #[getter]
    fn g_img_path(&self) -> PyResult<Option<String>> {
        let s = self.require_solved()?.g_img_path.to_string_lossy().into_owned();
        Ok(if s.is_empty() { None } else { Some(s) })
    }

    // -----------------------------------------------------------------------
    // IO
    // -----------------------------------------------------------------------

    /// Save the solved mesh to a ``.pyv`` file.
    fn save(&self, path: &str) -> PyResult<()> {
        let sol = self.solution.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Mesh has not been solved; cannot save.")
        })?;
        io_save(path, &GeopyvObject::Mesh((**sol).clone())).map_err(Error::from)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Repr
    // -----------------------------------------------------------------------

    fn __repr__(&self) -> String {
        if self.solution.is_some() {
            let sol = self.solution.as_ref().unwrap();
            let zncc_min = sol.c_zncc.iter().cloned().fold(f64::INFINITY, f64::min);
            let zncc_max = sol.c_zncc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            format!(
                "Mesh(nodes={}, elements={}, mesh_order={}, subset_order={}, solved=True, c_zncc=[{:.4}..{:.4}])",
                self.inner.nodes().nrows(),
                self.inner.elements().nrows(),
                self.inner.mesh_order(),
                sol.subset_order,
                zncc_min,
                zncc_max,
            )
        } else {
            format!(
                "Mesh(nodes={}, elements={}, mesh_order={}, solved=False)",
                self.inner.nodes().nrows(),
                self.inner.elements().nrows(),
                self.inner.mesh_order(),
            )
        }
    }
}

impl PyMesh {
    fn require_solved(&self) -> PyResult<&MeshSolution> {
        self.solution.as_deref().ok_or_else(|| {
            PyAttributeError::new_err("Mesh has not been solved; call solve() first")
        })
    }

    fn require_solved_arc(&self) -> PyResult<&Arc<MeshSolution>> {
        self.solution.as_ref().ok_or_else(|| {
            PyAttributeError::new_err("Mesh has not been solved; call solve() first")
        })
    }

    /// Restore a solved `PyMesh` from a serialised `MeshSolution`.
    ///
    /// Images are NOT loaded — only geometry and solution arrays are restored.
    /// `f_img` / `g_img` will be `None`; `f_img_path` / `g_img_path` remain
    /// accessible for display purposes.  Call `reload_images()` explicitly if
    /// the full `Image` (with B-spline coefficients) is needed for re-solving.
    pub(crate) fn from_solution(_py: Python<'_>, sol: Arc<MeshSolution>) -> PyResult<Self> {
        let dummy = Arc::new(Image::from_array(ndarray::Array2::<f64>::zeros((10, 10)), 3));
        let inner = Mesh::from_solution(&sol, Arc::clone(&dummy), dummy);
        Ok(PyMesh {
            inner,
            f_img: None,
            g_img: None,
            solution: Some(sol),
        })
    }

    /// Load (or reload) the reference and target images from their stored paths.
    ///
    /// Required before re-solving a restored mesh.  Reads both images from disk
    /// and computes B-spline coefficients.
    pub(crate) fn reload_images(&mut self, py: Python<'_>) -> PyResult<()> {
        let sol = self.require_solved_arc()?.clone();
        let dummy = || Arc::new(Image::from_array(ndarray::Array2::<f64>::zeros((10, 10)), 3));

        let mut f_arc = dummy();
        if !sol.f_img_path.as_os_str().is_empty() {
            if let Ok(img) = Image::from_file(&sol.f_img_path, 20) {
                let arc = Arc::new(img);
                f_arc = Arc::clone(&arc);
                self.f_img = Py::new(py, PyImage {
                    inner: arc,
                    filepath: Some(sol.f_img_path.to_string_lossy().into_owned()),
                }).ok();
            }
        }

        let mut g_arc = dummy();
        if !sol.g_img_path.as_os_str().is_empty() {
            if let Ok(img) = Image::from_file(&sol.g_img_path, 20) {
                let arc = Arc::new(img);
                g_arc = Arc::clone(&arc);
                self.g_img = Py::new(py, PyImage {
                    inner: arc,
                    filepath: Some(sol.g_img_path.to_string_lossy().into_owned()),
                }).ok();
            }
        }

        self.inner = Mesh::from_solution(&sol, f_arc, g_arc);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MeshSolution class — kept for serialisation boundary; NOT registered.
// ---------------------------------------------------------------------------

/// Serialisation-only result type.  Not exposed to Python users.
#[pyclass(name = "MeshSolution")]
pub struct PyMeshSolution {
    pub(crate) inner: MeshSolution,
}

#[pymethods]
impl PyMeshSolution {
    /// Construct a `MeshSolution` directly from arrays (used in tests).
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
        let nodes_owned = nodes.as_array().to_owned();
        let elements_owned: Array2<usize> = elements.as_array().map(|&x| x as usize);
        let centroids = geopyv_dev::mesh::compute_centroids(&nodes_owned, &elements_owned);
        PyMeshSolution {
            inner: MeshSolution {
                nodes: nodes_owned,
                elements: elements_owned,
                boundary: boundary.iter().map(|&x| x as usize).collect(),
                exclusions: exclusions
                    .iter()
                    .map(|g| g.iter().map(|&x| x as usize).collect())
                    .collect(),
                centroids,
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
                f_img_path: f_img_path.map(PathBuf::from).unwrap_or_default(),
                g_img_path: g_img_path.map(PathBuf::from).unwrap_or_default(),
            },
        }
    }

    #[getter]
    fn nodes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.nodes.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn elements<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let e: Array2<i64> = self.inner.elements.map(|&x| x as i64);
        e.into_pyarray_bound(py)
    }

    #[getter]
    fn boundary(&self) -> Vec<i64> {
        self.inner.boundary.iter().map(|&x| x as i64).collect()
    }

    #[getter]
    fn exclusions(&self) -> Vec<Vec<i64>> {
        self.inner
            .exclusions
            .iter()
            .map(|g| g.iter().map(|&x| x as i64).collect())
            .collect()
    }

    #[getter]
    fn areas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.areas.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn warps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.warps.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn displacements<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.displacements.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn c_zncc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.c_zncc.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn p<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.p.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn seed_node(&self) -> i64 {
        self.inner.seed_node as i64
    }

    #[getter]
    fn mesh_order(&self) -> u8 {
        self.inner.mesh_order
    }

    #[getter]
    fn subset_order(&self) -> u8 {
        self.inner.subset_order
    }

    #[getter]
    fn f_img_path(&self) -> Option<String> {
        let s = self.inner.f_img_path.to_string_lossy().into_owned();
        if s.is_empty() { None } else { Some(s) }
    }

    #[getter]
    fn g_img_path(&self) -> Option<String> {
        let s = self.inner.g_img_path.to_string_lossy().into_owned();
        if s.is_empty() { None } else { Some(s) }
    }

    #[getter]
    fn iterations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.iterations.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn norms<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.norms.clone().into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        let zncc = &self.inner.c_zncc;
        let zncc_min = zncc.iter().cloned().fold(f64::INFINITY, f64::min);
        let zncc_max = zncc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        format!(
            "MeshSolution(nodes={}, elements={}, mesh_order={}, subset_order={}, c_zncc=[{:.4}..{:.4}])",
            self.inner.nodes.nrows(),
            self.inner.elements.nrows(),
            self.inner.mesh_order,
            self.inner.subset_order,
            zncc_min,
            zncc_max,
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

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

#[pyfunction]
fn mesh_shape_function(py: Python<'_>, mesh_order: u8) -> PyResult<Bound<'_, PyTuple>> {
    let (n_vec, dn_vec, d2n_opt) = mesh::shape_function(mesh_order);

    let n_arr: Array1<f64> = Array1::from(n_vec);
    let n_any = n_arr.into_pyarray_bound(py).into_any();

    let dn_r = dn_vec.len();
    let dn_c = if dn_r > 0 { dn_vec[0].len() } else { 0 };
    let dn_flat: Vec<f64> = dn_vec.into_iter().flatten().collect();
    let dn_arr = Array2::from_shape_vec((dn_r, dn_c), dn_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let dn_any = dn_arr.into_pyarray_bound(py).into_any();

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

#[pyfunction]
fn mesh_find_seed_node(nodes: PyReadonlyArray2<f64>, seed_coord: [f64; 2]) -> i64 {
    let n = nodes.as_array().to_owned();
    mesh::find_seed_node(&n, seed_coord) as i64
}

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
    let disp_override: Option<[f64; 2]> = displacement.map(|v| [v[0], v[1]]);
    mesh::flow_calc(idx, &e, mesh_order, d, &excl, disp_override)
}

#[pyfunction]
fn mesh_r_calc(displacement: [f64; 2]) -> f64 {
    mesh::r_calc(displacement)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMesh>()?;
    // PyMeshSolution intentionally NOT registered — serialisation boundary only.
    m.add_function(wrap_pyfunction!(mesh_element_area, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_element_strains, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_shape_function, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_find_seed_node, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_corr, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_flow_calc, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_r_calc, m)?)?;
    Ok(())
}
