//! PyO3 wrapper for `geopyv_dev::particle`.
//!
//! Exposes:
//! - `Particle` class: constructed with initial state, runs `solve_increment`
//!   (one step) or `solve` (full sequence from pre-built mesh data arrays).
//! - `ParticleSolution` class: read-only result from `Particle.solve`.
//! - Free functions mirroring the pure-math helpers in `particle.rs`, prefixed
//!   `particle_` to avoid name collisions at the module level.

use std::path::PathBuf;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use geopyv_dev::particle::{
    self, MeshData, Particle, ParticleConfig, ParticleSolution,
};

use crate::Error;

// ---------------------------------------------------------------------------
// ParticleSolution class
// ---------------------------------------------------------------------------

/// Read-only result from :meth:`Particle.solve`.
#[pyclass(name = "ParticleSolution")]
pub struct PyParticleSolution {
    pub(crate) inner: ParticleSolution,
}

#[pymethods]
impl PyParticleSolution {
    /// Particle positions ``(inc_no, 2)``.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.coordinates.clone().into_pyarray_bound(py)
    }

    /// Accumulated warp vectors ``(inc_no, 6*mesh_order)``.
    #[getter]
    fn warps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.warps.clone().into_pyarray_bound(py)
    }

    /// Per-increment warp vectors ``(inc_no, 6*mesh_order)``.
    #[getter]
    fn incs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.incs.clone().into_pyarray_bound(py)
    }

    /// Particle volumes ``(inc_no,)``.
    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.volumes.clone().into_pyarray_bound(py)
    }

    /// Strains ``(inc_no, 6)``.
    #[getter]
    fn strains<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.strains.clone().into_pyarray_bound(py)
    }

    /// Strain increments ``(inc_no-1, 6)``.
    #[getter]
    fn strain_incs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.strain_incs.clone().into_pyarray_bound(py)
    }

    /// Volumetric strains ``(inc_no,)``.
    #[getter]
    fn vol_strains<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.vol_strains.clone().into_pyarray_bound(py)
    }

    /// Increment indices at which the reference mesh was updated.
    #[getter]
    fn reference_update_register(&self) -> Vec<i64> {
        self.inner
            .reference_update_register
            .iter()
            .map(|&x| x as i64)
            .collect()
    }

    /// Initial coordinate of the particle ``[x, y]``.
    #[getter]
    fn initial_coord(&self) -> [f64; 2] {
        if self.inner.coordinates.nrows() > 0 {
            [self.inner.coordinates[[0, 0]], self.inner.coordinates[[0, 1]]]
        } else {
            [0.0, 0.0]
        }
    }

    /// Path of the initial (reference) image, or ``None``.
    #[getter]
    fn image_0_path(&self) -> Option<String> {
        self.inner.image_0_path.as_ref().map(|p| p.to_string_lossy().into_owned())
    }

    fn __repr__(&self) -> String {
        format!(
            "ParticleSolution(inc_no={}, warp_len={})",
            self.inner.coordinates.nrows(),
            self.inner.warps.ncols(),
        )
    }
}

// ---------------------------------------------------------------------------
// Particle class
// ---------------------------------------------------------------------------

/// Lagrangian / Eulerian particle tracking and strain-path computation.
///
/// Parameters
/// ----------
/// coordinate : list[float]
///     Initial position ``[x, y]``.
/// initial_warp : list[float]
///     Initial warp vector (length 6 or 12); zero-padded if shorter than
///     ``6 * mesh_order``.
/// initial_volume : float
///     Initial volume (must be > 0).
/// inc_no : int
///     Total number of increments (frames), including the initial state.
/// mesh_order : int, optional
///     1 (linear) or 2 (quadratic). Default 1.
/// track : bool, optional
///     ``True`` for Lagrangian (coordinate moves). Default ``True``.
#[pyclass(name = "Particle")]
pub struct PyParticle {
    inner: Particle,
}

#[pymethods]
impl PyParticle {
    #[new]
    #[pyo3(signature = (coordinate, initial_warp, initial_volume, inc_no, mesh_order=1, track=true, image_0_path=None))]
    fn new(
        coordinate: [f64; 2],
        initial_warp: Vec<f64>,
        initial_volume: f64,
        inc_no: usize,
        mesh_order: u8,
        track: bool,
        image_0_path: Option<String>,
    ) -> PyResult<Self> {
        let path = image_0_path.map(PathBuf::from);
        let p = Particle::new(coordinate, &initial_warp, initial_volume, inc_no, mesh_order, track, path)
            .map_err(Error::from)?;
        Ok(PyParticle { inner: p })
    }

    /// Solve a single increment.
    ///
    /// Parameters
    /// ----------
    /// m : int
    ///     Increment index (0-based); result is stored at ``m + 1``.
    /// nodes : numpy.ndarray, shape (N, 2), float64
    ///     Mesh node coordinates.
    /// elements : numpy.ndarray, shape (M, 3) or (M, 6), int64
    ///     Element connectivity (corner + midpoint nodes for order-2).
    /// displacements : numpy.ndarray, shape (N, 2), float64
    ///     Per-node displacements ``[u, v]``.
    /// mesh_order : int
    ///     1 or 2 — must match the element columns.
    /// ref_update : bool, optional
    ///     ``True`` if the reference mesh changed at step ``m``. Default ``False``.
    ///
    /// Returns
    /// -------
    /// bool  — always ``True`` on success.
    #[pyo3(signature = (m, nodes, elements, displacements, mesh_order, ref_update=false))]
    fn solve_increment(
        &mut self,
        m: usize,
        nodes: PyReadonlyArray2<f64>,
        elements: PyReadonlyArray2<i64>,
        displacements: PyReadonlyArray2<f64>,
        mesh_order: u8,
        ref_update: bool,
    ) -> bool {
        let n = nodes.as_array().to_owned();
        let e: Array2<usize> = elements.as_array().map(|&x| x as usize);
        let d = displacements.as_array().to_owned();
        let cm = MeshData {
            nodes: &n,
            elements: &e,
            displacements: &d,
            mesh_order,
        };
        self.inner.solve_increment(m, &cm, ref_update)
    }

    /// Solve for all increments over a sequence of meshes.
    ///
    /// Parameters
    /// ----------
    /// nodes_list : list[numpy.ndarray]
    ///     One ``(N_m, 2)`` float64 array per increment.
    /// elements_list : list[numpy.ndarray]
    ///     One ``(M_m, 3 or 6)`` int64 array per increment.
    /// displacements_list : list[numpy.ndarray]
    ///     One ``(N_m, 2)`` float64 array per increment.
    /// mesh_order_list : list[int]
    ///     One ``mesh_order`` per increment.
    /// factor : float, optional
    ///     Volumetric correction factor. Default 1.0.
    /// true_incs : bool, optional
    ///     Use logarithmic strain increments. Default ``True``.
    ///
    /// Returns
    /// -------
    /// ParticleSolution
    #[pyo3(signature = (nodes_list, elements_list, displacements_list, mesh_order_list, factor=1.0, true_incs=true))]
    fn solve(
        &mut self,
        nodes_list: Vec<PyReadonlyArray2<f64>>,
        elements_list: Vec<PyReadonlyArray2<i64>>,
        displacements_list: Vec<PyReadonlyArray2<f64>>,
        mesh_order_list: Vec<u8>,
        factor: f64,
        true_incs: bool,
    ) -> PyResult<PyParticleSolution> {
        let n_inc = nodes_list.len();
        // Convert each increment's arrays into owned storage so MeshData can borrow them.
        let nodes_owned: Vec<Array2<f64>> =
            nodes_list.iter().map(|a| a.as_array().to_owned()).collect();
        let elements_owned: Vec<Array2<usize>> = elements_list
            .iter()
            .map(|a| a.as_array().map(|&x| x as usize))
            .collect();
        let displacements_owned: Vec<Array2<f64>> = displacements_list
            .iter()
            .map(|a| a.as_array().to_owned())
            .collect();

        let mesh_data: Vec<MeshData<'_>> = (0..n_inc)
            .map(|i| MeshData {
                nodes: &nodes_owned[i],
                elements: &elements_owned[i],
                displacements: &displacements_owned[i],
                mesh_order: mesh_order_list[i],
            })
            .collect();

        let cfg = ParticleConfig { factor, true_incs };
        let sol = self.inner.solve(&mesh_data, &cfg).map_err(Error::from)?;
        Ok(PyParticleSolution { inner: sol })
    }

    // --- State getters ---

    /// Particle coordinates ``(inc_no, 2)``.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.coordinates.clone().into_pyarray_bound(py)
    }

    /// Accumulated warp vectors ``(inc_no, 6*mesh_order)``.
    #[getter]
    fn warps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.warps.clone().into_pyarray_bound(py)
    }

    /// Per-increment warp vectors ``(inc_no, 6*mesh_order)``.
    #[getter]
    fn incs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.incs.clone().into_pyarray_bound(py)
    }

    /// Particle volumes ``(inc_no,)``.
    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.volumes.clone().into_pyarray_bound(py)
    }

    /// Whether :meth:`solve` has been called.
    #[getter]
    fn solved(&self) -> bool {
        self.inner.solved
    }

    /// Mesh element order (1 or 2).
    #[getter]
    fn mesh_order(&self) -> u8 {
        self.inner.mesh_order
    }

    /// ``True`` if Lagrangian (coordinate tracks material).
    #[getter]
    fn track(&self) -> bool {
        self.inner.track
    }

    /// Current solve step index.
    #[getter]
    fn current_step(&self) -> usize {
        self.inner.current_step
    }

    /// Path of the initial (reference) image, or ``None``.
    #[getter]
    fn image_0_path(&self) -> Option<String> {
        self.inner.image_0_path.as_ref().map(|p| p.to_string_lossy().into_owned())
    }

    /// Initial coordinate of the particle ``[x, y]``.
    #[getter]
    fn initial_coord(&self) -> [f64; 2] {
        if self.inner.coordinates.nrows() > 0 {
            [self.inner.coordinates[[0, 0]], self.inner.coordinates[[0, 1]]]
        } else {
            [0.0, 0.0]
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Particle(inc_no={}, mesh_order={}, track={}, solved={})",
            self.inner.coordinates.nrows(),
            self.inner.mesh_order,
            self.inner.track,
            self.inner.solved,
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute barycentric coordinates of a point inside a triangle.
///
/// Replicates ``Particle._local_coordinates``.
///
/// Parameters
/// ----------
/// coordinate : list[float]
///     Particle position ``[x, y]``.
/// element_nodes : numpy.ndarray, shape (N>=3, 2), float64
///     Element node coordinates; only the first 3 rows (corners) are used.
///
/// Returns
/// -------
/// tuple (zeta, eta, theta, det_denom) : (float, float, float, float)
#[pyfunction]
fn particle_local_coordinates(
    coordinate: [f64; 2],
    element_nodes: PyReadonlyArray2<f64>,
) -> (f64, f64, f64, f64) {
    particle::local_coordinates(coordinate, element_nodes.as_array())
}

/// Evaluate element shape functions and their derivatives.
///
/// Replicates ``Particle._shape_function``.
///
/// Parameters
/// ----------
/// mesh_order : int
///     1 or 2.
/// zeta, eta, theta : float
///     Barycentric coordinates.
///
/// Returns
/// -------
/// tuple (N, dN, d2N):
///     N   – shape ``(3,)`` or ``(6,)``
///     dN  – shape ``(2, 3)`` or ``(2, 6)``
///     d2N – shape ``(3, 6)`` for order-2; ``None`` for order-1
#[pyfunction]
fn particle_shape_function(
    py: Python<'_>,
    mesh_order: u8,
    zeta: f64,
    eta: f64,
    theta: f64,
) -> PyResult<Bound<'_, PyTuple>> {
    let (n, dn, d2n_opt) = particle::shape_function(mesh_order, zeta, eta, theta);

    let n_any = n.into_pyarray_bound(py).into_any();
    let dn_any = dn.into_pyarray_bound(py).into_any();
    let d2n_any = match d2n_opt {
        Some(d2n) => d2n.into_pyarray_bound(py).into_any(),
        None => py.None().into_bound(py),
    };

    Ok(PyTuple::new_bound(py, [n_any, dn_any, d2n_any]))
}

/// Compute the warp increment for a particle at ``coordinate``.
///
/// Replicates ``Particle._warp_increment``.
///
/// Parameters
/// ----------
/// coordinate : list[float]
///     Current particle position ``[x, y]``.
/// element_nodes : numpy.ndarray, shape (3 or 6, 2), float64
/// element_disps : numpy.ndarray, shape (3 or 6, 2), float64
/// mesh_order : int
///
/// Returns
/// -------
/// numpy.ndarray, shape (6,) or (12,), float64
#[pyfunction]
fn particle_warp_increment<'py>(
    py: Python<'py>,
    coordinate: [f64; 2],
    element_nodes: PyReadonlyArray2<f64>,
    element_disps: PyReadonlyArray2<f64>,
    mesh_order: u8,
) -> Bound<'py, PyArray1<f64>> {
    particle::warp_increment(
        coordinate,
        element_nodes.as_array(),
        element_disps.as_array(),
        mesh_order,
    )
    .into_pyarray_bound(py)
}

/// Find the index of the element containing ``coordinate``.
///
/// Replicates ``Particle._element_locator``.
///
/// Parameters
/// ----------
/// coordinate : list[float]
/// nodes : numpy.ndarray, shape (N, 2), float64
/// elements : numpy.ndarray, shape (M, 3) or (M, 6), int64
/// centroids : numpy.ndarray, shape (M, 2), float64
///
/// Returns
/// -------
/// int
#[pyfunction]
fn particle_element_locator(
    coordinate: [f64; 2],
    nodes: PyReadonlyArray2<f64>,
    elements: PyReadonlyArray2<i64>,
    centroids: PyReadonlyArray2<f64>,
) -> i64 {
    let n = nodes.as_array().to_owned();
    let e: Array2<usize> = elements.as_array().map(|&x| x as usize);
    let c = centroids.as_array().to_owned();
    particle::element_locator(coordinate, &n, &e, &c) as i64
}

/// Compute element centroids (mean of corner node positions).
///
/// Parameters
/// ----------
/// nodes : numpy.ndarray, shape (N, 2), float64
/// elements : numpy.ndarray, shape (M, 3) or (M, 6), int64
///
/// Returns
/// -------
/// numpy.ndarray, shape (M, 2), float64
#[pyfunction]
fn particle_compute_centroids<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    elements: PyReadonlyArray2<i64>,
) -> Bound<'py, PyArray2<f64>> {
    let n = nodes.as_array().to_owned();
    let e: Array2<usize> = elements.as_array().map(|&x| x as usize);
    particle::compute_centroids(&n, &e).into_pyarray_bound(py)
}

/// Compute strains and strain increments from accumulated warp vectors.
///
/// Replicates ``Particle._strain_def``.
///
/// Parameters
/// ----------
/// warps : numpy.ndarray, shape (inc_no, 6*mesh_order), float64
/// factor : float, optional
///     Volumetric correction factor. Default 1.0.
/// true_incs : bool, optional
///     Logarithmic strain increments. Default ``True``.
///
/// Returns
/// -------
/// tuple (strains, strain_incs):
///     strains     – shape ``(inc_no, 6)``
///     strain_incs – shape ``(inc_no-1, 6)``
#[pyfunction]
#[pyo3(signature = (warps, factor=1.0, true_incs=true))]
fn particle_strain_def<'py>(
    py: Python<'py>,
    warps: PyReadonlyArray2<f64>,
    factor: f64,
    true_incs: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let w = warps.as_array().to_owned();
    let (strains, strain_incs) = particle::strain_def(&w, factor, true_incs);
    let s_any = strains.into_pyarray_bound(py).into_any();
    let si_any = strain_incs.into_pyarray_bound(py).into_any();
    Ok(PyTuple::new_bound(py, [s_any, si_any]))
}

/// Compute volumetric strains from a volumes array.
///
/// Parameters
/// ----------
/// volumes : numpy.ndarray, shape (inc_no,), float64
///
/// Returns
/// -------
/// numpy.ndarray, shape (inc_no,), float64
#[pyfunction]
fn particle_vol_strains<'py>(
    py: Python<'py>,
    volumes: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let v = volumes.as_array().to_owned();
    particle::vol_strains(&v).into_pyarray_bound(py)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyParticle>()?;
    m.add_class::<PyParticleSolution>()?;
    m.add_function(wrap_pyfunction!(particle_local_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(particle_shape_function, m)?)?;
    m.add_function(wrap_pyfunction!(particle_warp_increment, m)?)?;
    m.add_function(wrap_pyfunction!(particle_element_locator, m)?)?;
    m.add_function(wrap_pyfunction!(particle_compute_centroids, m)?)?;
    m.add_function(wrap_pyfunction!(particle_strain_def, m)?)?;
    m.add_function(wrap_pyfunction!(particle_vol_strains, m)?)?;
    Ok(())
}
