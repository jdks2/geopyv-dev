//! PyO3 wrapper for `geopyv_dev::particle`.

use std::sync::Arc;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use crate::utils::{arc_array1, arc_array2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use geopyv_dev::particle::{
    self, Particle, ParticleConfig, ParticleSolution, ParticleSource,
};
use geopyv_dev::mesh;

use crate::{py_calibration::PyCalibrationParams, py_mesh::{PyMesh, PyMeshSolution}, py_sequence::PySequence, Error};


// ---------------------------------------------------------------------------
// ParticleSolution class
// ---------------------------------------------------------------------------

/// Read-only result from :meth:`Particle.solve`.
#[pyclass(name = "ParticleSolution")]
pub struct PyParticleSolution {
    pub(crate) inner: Arc<ParticleSolution>,
}

#[pymethods]
impl PyParticleSolution {
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arc_array2(py, &self.inner, |s| &s.coordinates)
    }

    #[getter]
    fn warps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arc_array2(py, &self.inner, |s| &s.warps)
    }

    #[getter]
    fn incs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arc_array2(py, &self.inner, |s| &s.incs)
    }

    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        arc_array1(py, &self.inner, |s| &s.volumes)
    }

    #[getter]
    fn strains<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arc_array2(py, &self.inner, |s| &s.strains)
    }

    #[getter]
    fn strain_incs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arc_array2(py, &self.inner, |s| &s.strain_incs)
    }

    #[getter]
    fn vol_strains<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        arc_array1(py, &self.inner, |s| &s.vol_strains)
    }

    #[getter]
    fn reference_update_register(&self) -> Vec<i64> {
        self.inner.reference_update_register.iter().map(|&x| x as i64).collect()
    }

    #[getter]
    fn initial_coord(&self) -> [f64; 2] {
        if self.inner.coordinates.nrows() > 0 {
            [self.inner.coordinates[[0, 0]], self.inner.coordinates[[0, 1]]]
        } else {
            [0.0, 0.0]
        }
    }

    #[getter]
    fn image_0_path(&self) -> Option<String> {
        self.inner.image_0_path.as_ref().map(|p| p.to_string_lossy().into_owned())
    }

    #[getter]
    fn calibrated(&self) -> bool {
        self.inner.calibrated
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
/// source : SequenceSolution or MeshSolution
///     Data source for mesh geometry and displacements.
/// coordinate : list[float]
///     Initial position ``[x, y]``.
/// initial_warp : list[float], optional
///     Initial warp vector; zero-padded if shorter than ``6 * mesh_order``.
/// track : bool, optional
///     ``True`` for Lagrangian (coordinate moves). Default ``True``.
#[pyclass(name = "Particle")]
pub struct PyParticle {
    pub(crate) inner: Particle,
}

#[pymethods]
impl PyParticle {
    #[new]
    #[pyo3(signature = (source, coordinate, initial_warp=None, track=true))]
    fn new(
        _py: Python<'_>,
        source: &Bound<'_, PyAny>,
        coordinate: [f64; 2],
        initial_warp: Option<Vec<f64>>,
        track: bool,
    ) -> PyResult<Self> {
        let ps: ParticleSource = if let Ok(seq) = source.downcast::<PySequence>() {
            let borrowed = seq.borrow();
            let sol = borrowed.solution.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Sequence has not been solved")
            })?;
            ParticleSource::Sequence(Arc::new(sol.clone()))
        } else if let Ok(mesh) = source.downcast::<PyMeshSolution>() {
            ParticleSource::Mesh(Arc::new(mesh.borrow().inner.clone()))
        } else if let Ok(mesh) = source.downcast::<PyMesh>() {
            let borrowed = mesh.borrow();
            let sol = borrowed.solution.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Mesh has not been solved")
            })?;
            ParticleSource::Mesh(Arc::clone(sol))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "source must be a solved Sequence or Mesh",
            ));
        };
        let mesh_order = ps.mesh_order();
        let warp = initial_warp.unwrap_or_else(|| vec![0.0; 6 * mesh_order as usize]);
        let p = Particle::new(ps, coordinate, &warp, 1.0, track).map_err(Error::from)?;
        Ok(PyParticle { inner: p })
    }

    /// Solve for all increments using mesh data from the source.
    ///
    /// Parameters
    /// ----------
    /// factor : float, optional
    ///     Volumetric correction factor. Default 0.0.
    /// true_incs : bool, optional
    ///     Logarithmic strain increments. Default ``True``.
    #[pyo3(signature = (factor=0.0, true_incs=true, calibration=None))]
    fn solve(
        &mut self,
        factor: f64,
        true_incs: bool,
        calibration: Option<Bound<'_, PyCalibrationParams>>,
    ) -> PyResult<()> {
        let cfg = ParticleConfig { factor, true_incs };
        let borrowed = calibration.as_ref().map(|b| b.borrow());
        let cal = borrowed.as_ref().map(|b| &b.inner);
        Ok(self.inner.solve(&cfg, cal).map_err(Error::from)?)
    }

    /// Solve a single increment.
    ///
    /// Parameters
    /// ----------
    /// m : int
    ///     Increment index (0-based); result stored at ``m + 1``.
    ///
    /// Returns
    /// -------
    /// bool  — ``True`` on success; raises ``RuntimeError`` on I/O failure.
    fn solve_increment(&mut self, m: usize) -> PyResult<bool> {
        let mesh = self.inner.source.load_mesh_at(m).map_err(Error::from)?;
        Ok(self.inner.solve_increment(m, &mesh, None))
    }

    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.coordinates.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn warps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.warps.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn incs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.incs.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.volumes.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn solved(&self) -> bool { self.inner.solved() }

    #[getter]
    fn mesh_order(&self) -> u8 { self.inner.mesh_order() }

    #[getter]
    fn track(&self) -> bool { self.inner.track }

    #[getter]
    fn image_0_path(&self) -> Option<String> {
        let s = self.inner.image_0_path()?.to_string_lossy().into_owned();
        if s.is_empty() { None } else { Some(s) }
    }

    #[getter]
    fn initial_coord(&self) -> [f64; 2] {
        if self.inner.coordinates.nrows() > 0 {
            [self.inner.coordinates[[0, 0]], self.inner.coordinates[[0, 1]]]
        } else {
            [0.0, 0.0]
        }
    }

    #[getter]
    fn strains<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arc_array2(py, self.require_solved()?, |s| &s.strains)
    }

    #[getter]
    fn strain_incs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arc_array2(py, self.require_solved()?, |s| &s.strain_incs)
    }

    #[getter]
    fn vol_strains<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        arc_array1(py, self.require_solved()?, |s| &s.vol_strains)
    }

    #[getter]
    fn reference_update_register(&self) -> PyResult<Vec<i64>> {
        Ok(self.require_solved()?.reference_update_register.iter().map(|&x| x as i64).collect())
    }

    #[getter]
    fn calibrated(&self) -> PyResult<bool> {
        Ok(self.require_solved()?.calibrated)
    }

    fn __repr__(&self) -> String {
        format!(
            "Particle(inc_no={}, mesh_order={}, track={}, solved={})",
            self.inner.inc_no(),
            self.inner.mesh_order(),
            self.inner.track,
            self.inner.solved(),
        )
    }
}

impl PyParticle {
    fn require_solved(&self) -> PyResult<&Arc<geopyv_dev::particle::ParticleSolution>> {
        self.inner.solution().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Particle has not been solved; call solve() first",
            )
        })
    }

    pub fn from_solution(sol: geopyv_dev::particle::ParticleSolution) -> PyResult<Self> {
        Ok(PyParticle { inner: geopyv_dev::particle::Particle::from_solution(sol) })
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

#[pyfunction]
fn particle_local_coordinates(
    coordinate: [f64; 2],
    element_nodes: PyReadonlyArray2<f64>,
) -> (f64, f64, f64, f64) {
    particle::local_coordinates(coordinate, element_nodes.as_array())
}

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
    let (idx, _) = particle::element_locator(coordinate, &n, &e, &c);
    idx as i64
}

#[pyfunction]
fn particle_compute_centroids<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f64>,
    elements: PyReadonlyArray2<i64>,
) -> Bound<'py, PyArray2<f64>> {
    let n = nodes.as_array().to_owned();
    let e: Array2<usize> = elements.as_array().map(|&x| x as usize);
    mesh::compute_centroids(&n, &e).into_pyarray_bound(py)
}

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
