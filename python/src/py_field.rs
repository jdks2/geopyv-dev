//! PyO3 wrapper for `geopyv_dev::field`.
//!
//! Exposes:
//! - `Field` class: constructed with initial particle state, runs `solve`.
//! - `FieldSolution` class: read-only result from `Field.solve`.
//! - `field_distribute_particles` free function.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use geopyv_dev::{
    field::{self, Field, FieldSolution},
    particle::MeshData,
};

use crate::{py_particle::PyParticleSolution, Error};

// ---------------------------------------------------------------------------
// FieldSolution class
// ---------------------------------------------------------------------------

/// Read-only result from :meth:`Field.solve`.
#[pyclass(name = "FieldSolution")]
pub struct PyFieldSolution {
    pub(crate) inner: FieldSolution,
    pub(crate) image_0_path: Option<String>,
}

#[pymethods]
impl PyFieldSolution {
    /// Per-particle solutions as a list of :class:`ParticleSolution`.
    #[getter]
    fn particles(&self) -> Vec<PyParticleSolution> {
        self.inner
            .particles
            .iter()
            .map(|p| PyParticleSolution { inner: p.clone(), image_0_path: self.image_0_path.clone() })
            .collect()
    }

    /// Sum of all particle volumes at each increment, shape ``(inc_no,)``.
    #[getter]
    fn vol_totals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.vol_totals.clone().into_pyarray_bound(py)
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

    /// Initial coordinates of all particles (row 0 of each particle's coordinate array), shape ``(N, 2)``.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.particles.len();
        let mut coords = ndarray::Array2::<f64>::zeros((n, 2));
        for (i, p) in self.inner.particles.iter().enumerate() {
            if p.coordinates.nrows() > 0 {
                coords[[i, 0]] = p.coordinates[[0, 0]];
                coords[[i, 1]] = p.coordinates[[0, 1]];
            }
        }
        coords.into_pyarray_bound(py)
    }

    /// Path of the initial (reference) image, or ``None``.
    #[getter]
    fn image_0_path(&self) -> Option<String> {
        self.image_0_path.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "FieldSolution(particles={}, inc_no={})",
            self.inner.particles.len(),
            self.inner.vol_totals.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Field class
// ---------------------------------------------------------------------------

/// Distributed particle field for strain-path tracking.
///
/// Parameters
/// ----------
/// coordinates : numpy.ndarray, shape (N, 2), float64
///     Initial particle positions.
/// volumes : numpy.ndarray, shape (N,), float64
///     Initial particle volumes; all must be > 0.
/// track : bool, optional
///     ``True`` for Lagrangian (coordinates move). Default ``True``.
/// depth : float, optional
///     Depth multiplier; must be > 0. Default 1.0.
/// inc_no : int
///     Total number of frames (≥ 2).
#[pyclass(name = "Field")]
pub struct PyField {
    inner: Field,
    pub(crate) image_0_path: Option<String>,
}

#[pymethods]
impl PyField {
    #[new]
    #[pyo3(signature = (coordinates, volumes, inc_no, track=true, depth=1.0, image_0_path=None))]
    fn new(
        coordinates: PyReadonlyArray2<f64>,
        volumes: PyReadonlyArray1<f64>,
        inc_no: usize,
        track: bool,
        depth: f64,
        image_0_path: Option<String>,
    ) -> PyResult<Self> {
        let c = coordinates.as_array().to_owned();
        let v = volumes.as_array().to_owned();
        let f = Field::new(c, v, track, depth, inc_no).map_err(Error::from)?;
        Ok(PyField { inner: f, image_0_path })
    }

    /// Solve strain paths for all particles over a sequence of mesh increments.
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
    ///     One ``mesh_order`` (1 or 2) per increment.
    /// ref_updates : list[bool], optional
    ///     One ``bool`` per increment; ``True`` if the reference mesh changed.
    ///     If empty or omitted, all steps use ``False``. Default ``[]``.
    /// factor : float, optional
    ///     Volumetric correction factor passed to strain computation. Default 1.0.
    /// true_incs : bool, optional
    ///     Logarithmic strain increments. Default ``True``.
    ///
    /// Returns
    /// -------
    /// FieldSolution
    #[pyo3(signature = (nodes_list, elements_list, displacements_list, mesh_order_list,
                         ref_updates=None, factor=1.0, true_incs=true))]
    fn solve(
        &mut self,
        nodes_list: Vec<PyReadonlyArray2<f64>>,
        elements_list: Vec<PyReadonlyArray2<i64>>,
        displacements_list: Vec<PyReadonlyArray2<f64>>,
        mesh_order_list: Vec<u8>,
        ref_updates: Option<Vec<bool>>,
        factor: f64,
        true_incs: bool,
    ) -> PyResult<PyFieldSolution> {
        let n_inc = nodes_list.len();
        let nodes_owned: Vec<ndarray::Array2<f64>> =
            nodes_list.iter().map(|a| a.as_array().to_owned()).collect();
        let elements_owned: Vec<Array2<usize>> = elements_list
            .iter()
            .map(|a| a.as_array().map(|&x| x as usize))
            .collect();
        let displacements_owned: Vec<ndarray::Array2<f64>> = displacements_list
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

        let ru: Vec<bool> = ref_updates.unwrap_or_default();
        let sol = self
            .inner
            .solve(&mesh_data, &ru, factor, true_incs)
            .map_err(Error::from)?;
        Ok(PyFieldSolution { inner: sol, image_0_path: self.image_0_path.clone() })
    }

    // --- Getters ---

    /// Initial particle coordinates ``(N, 2)``.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.coordinates.clone().into_pyarray_bound(py)
    }

    /// Initial particle volumes ``(N,)``.
    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.volumes.clone().into_pyarray_bound(py)
    }

    /// Number of particles.
    #[getter]
    fn n_particles(&self) -> usize {
        self.inner.n_particles()
    }

    /// ``True`` if Lagrangian.
    #[getter]
    fn track(&self) -> bool {
        self.inner.track
    }

    /// Depth multiplier.
    #[getter]
    fn depth(&self) -> f64 {
        self.inner.depth
    }

    /// Total number of frames.
    #[getter]
    fn inc_no(&self) -> usize {
        self.inner.inc_no
    }

    /// Whether :meth:`solve` has been called.
    #[getter]
    fn solved(&self) -> bool {
        self.inner.solved
    }

    /// Path of the initial (reference) image, or ``None``.
    #[getter]
    fn image_0_path(&self) -> Option<String> {
        self.image_0_path.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Field(n_particles={}, inc_no={}, track={}, solved={})",
            self.inner.n_particles(),
            self.inner.inc_no,
            self.inner.track,
            self.inner.solved,
        )
    }
}

// ---------------------------------------------------------------------------
// Free function
// ---------------------------------------------------------------------------

/// Distribute particles at element centroids and compute representative volumes.
///
/// Replicates ``Field._distribute_particles``.
///
/// Parameters
/// ----------
/// nodes : numpy.ndarray, shape (N, 2), float64
///     Mesh node coordinates.
/// elements : numpy.ndarray, shape (M, 3) or (M, 6), int64
///     Element connectivity; only the first 3 columns (corner nodes) are used.
/// depth : float, optional
///     Depth multiplier for volume. Default 1.0.
///
/// Returns
/// -------
/// tuple (coordinates, volumes):
///     coordinates – shape ``(M, 2)``
///     volumes     – shape ``(M,)``
#[pyfunction]
#[pyo3(signature = (nodes, elements, depth=1.0))]
fn field_distribute_particles<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    elements: PyReadonlyArray2<i64>,
    depth: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let n = nodes.as_array().to_owned();
    let e: Array2<usize> = elements.as_array().map(|&x| x as usize);
    let (coords, vols) = field::distribute_particles(&n, &e, depth);
    Ok((coords.into_pyarray_bound(py), vols.into_pyarray_bound(py)))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyField>()?;
    m.add_class::<PyFieldSolution>()?;
    m.add_function(wrap_pyfunction!(field_distribute_particles, m)?)?;
    Ok(())
}
