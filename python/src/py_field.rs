//! PyO3 wrapper for `geopyv_dev::field`.

use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use geopyv_dev::field::{self, Field, FieldDistribution, FieldSolution};
use ndarray::Array2;

use geopyv_dev::sequence::SequenceSolution;

use crate::{py_calibration::PyCalibrationParams, py_mesh::PyMesh, py_particle::{PyParticle, strain_method_from_py}, py_sequence::PySequence, Error};

// ---------------------------------------------------------------------------
// FieldSolution class
// ---------------------------------------------------------------------------

/// Read-only result from :meth:`Field.solve`.
#[pyclass(name = "FieldSolution")]
pub struct PyFieldSolution {
    pub(crate) inner: FieldSolution,
}

#[pymethods]
impl PyFieldSolution {
    /// Every particle's strain-path solution, each reconstructed as a live
    /// ``Particle`` (see [`PyField::particles`] for why — this mirrors the
    /// same reconstruction Subset/Mesh/Sequence already use).
    #[getter]
    fn particles(&self) -> PyResult<Vec<PyParticle>> {
        self.inner.particles.iter()
            .map(|p| PyParticle::from_solution((**p).clone()))
            .collect()
    }

    /// Load a single particle's strain-path solution by index without
    /// materialising the rest — cheaper than ``particles[idx]`` for a large field.
    fn particle_at(&self, idx: usize) -> PyResult<PyParticle> {
        let p = self.inner.particles.get(idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!("particle index {idx} out of range"))
        })?;
        PyParticle::from_solution((**p).clone())
    }

    #[getter]
    fn vol_totals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.vol_totals.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn reference_update_register(&self) -> Vec<i64> {
        self.inner.reference_update_register.iter().map(|&x| x as i64).collect()
    }

    /// Initial coordinates of all particles ``(N, 2)``.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.initial_coordinates.clone().into_pyarray_bound(py)
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
/// sequence_solution : Sequence or Mesh
///     Solved sequence or single solved mesh providing the DIC increment data.
///     A ``Mesh`` is wrapped internally as a one-increment sequence.
/// track : bool, optional
///     ``True`` for Lagrangian (coordinates move). Default ``True``.
/// depth : float, optional
///     Depth multiplier; must be > 0. Default 1.0.
/// coordinates : numpy.ndarray (N, 2), optional
///     Explicit initial particle positions.
/// volumes : numpy.ndarray (N,), optional
///     Explicit initial particle volumes (required with ``coordinates``).
#[pyclass(name = "Field")]
pub struct PyField {
    pub(crate) inner: Field,
}

#[pymethods]
impl PyField {
    #[new]
    #[pyo3(signature = (sequence_solution, track=true, depth=1.0,
                         coordinates=None, volumes=None))]
    fn new(
        sequence_solution: &Bound<'_, PyAny>,
        track: bool,
        depth: f64,
        coordinates: Option<PyReadonlyArray2<f64>>,
        volumes: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let source: Arc<SequenceSolution> =
            if let Ok(seq) = sequence_solution.extract::<PyRef<PySequence>>() {
                let sol = seq.inner.solution().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Sequence has not been solved")
                })?;
                Arc::new(sol.clone())
            } else if let Ok(mesh) = sequence_solution.extract::<PyRef<PyMesh>>() {
                let mesh_sol = mesh.inner.solution().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Mesh has not been solved")
                })?;
                Arc::new(SequenceSolution::from_mesh_solution(mesh_sol.clone()))
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "sequence_solution must be a solved Sequence or a solved Mesh",
                ));
            };
        let distribution = match (coordinates, volumes) {
            (Some(c), Some(v)) => FieldDistribution::Explicit {
                coordinates: c.as_array().to_owned(),
                volumes: v.as_array().to_owned(),
            },
            (None, None) => FieldDistribution::FromSequence,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                "coordinates and volumes must both be supplied or both omitted",
            )),
        };
        let f = Field::new(source, distribution, track, depth).map_err(Error::from)?;
        Ok(PyField { inner: f })
    }

    /// Solve strain paths for all particles.
    ///
    /// Parameters
    /// ----------
    /// factor : float, optional
    ///     Volumetric correction factor. Default 0.0.
    /// true_incs : bool, optional
    ///     Logarithmic strain increments. Default ``True``.
    /// strain_method : MeshlessParams or False, optional
    ///     ``None`` (default): meshless (IRLS-robust MLS) strain estimator
    ///     with default ``MeshlessParams`` -- the package-wide default. A
    ///     ``MeshlessParams`` instance: meshless with those parameters.
    ///     ``False``: force the original mesh-element (shape-function)
    ///     interpolation instead -- retained for comparison, no longer the
    ///     default.
    #[pyo3(signature = (factor=0.0, true_incs=true, calibration=None, strain_method=None))]
    fn solve(
        &mut self,
        factor: f64,
        true_incs: bool,
        calibration: Option<Bound<'_, PyCalibrationParams>>,
        strain_method: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let borrowed = calibration.as_ref().map(|b| b.borrow());
        let cal = borrowed.as_ref().map(|b| &b.inner);
        let sm = strain_method_from_py(strain_method.as_ref())?;
        Ok(self.inner.solve(factor, true_incs, cal, sm).map_err(Error::from)?)
    }

    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.coordinates.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.volumes.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn n_particles(&self) -> usize { self.inner.n_particles() }

    #[getter]
    fn track(&self) -> bool { self.inner.track }

    #[getter]
    fn depth(&self) -> f64 { self.inner.depth }

    #[getter]
    fn inc_no(&self) -> usize { self.inner.inc_no() }

    #[getter]
    fn solved(&self) -> bool { self.inner.solved() }

    #[getter]
    fn image_0_path(&self) -> Option<String> {
        let s = self.inner.image_0_path()?.to_string_lossy().into_owned();
        if s.is_empty() { None } else { Some(s) }
    }

    /// Every particle's strain-path solution, each reconstructed as a live
    /// ``Particle`` — matches the Subset/Mesh/Sequence precedent of always
    /// handing back the same live class (with a genuine ``solved`` getter)
    /// rather than a separate read-only result type. Prefer ``particle_at(idx)``
    /// when only one particle is needed on a large field.
    #[getter]
    fn particles(&self) -> PyResult<Vec<PyParticle>> {
        self.require_solved()?.particles.iter()
            .map(|p| PyParticle::from_solution((**p).clone()))
            .collect()
    }

    /// Load a single particle's strain-path solution by index without
    /// materialising the rest — cheaper than ``particles[idx]`` for a large field.
    fn particle_at(&self, idx: usize) -> PyResult<PyParticle> {
        let sol = self.require_solved()?;
        let p = sol.particles.get(idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!("particle index {idx} out of range"))
        })?;
        PyParticle::from_solution((**p).clone())
    }

    #[getter]
    fn vol_totals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.require_solved()?.vol_totals.clone().into_pyarray_bound(py))
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
            "Field(n_particles={}, inc_no={}, track={}, solved={})",
            self.inner.n_particles(),
            self.inner.inc_no(),
            self.inner.track,
            self.inner.solved(),
        )
    }
}

impl PyField {
    fn require_solved(&self) -> PyResult<&geopyv_dev::field::FieldSolution> {
        self.inner.solution().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Field has not been solved; call solve() first",
            )
        })
    }

    pub fn from_solution(sol: geopyv_dev::field::FieldSolution) -> PyResult<Self> {
        Ok(PyField { inner: geopyv_dev::field::Field::from_solution(sol) })
    }
}

// ---------------------------------------------------------------------------
// Free function
// ---------------------------------------------------------------------------

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
