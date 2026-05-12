//! PyO3 wrapper for `geopyv_dev::sequence`.
//!
//! Exposes:
//! - `SequenceOptions` class: temporal coupling options for `Sequence.solve`.
//! - `Sequence` class: constructed with image directory + mesh config, runs `solve`.
//! - `SequenceSolution` class: read-only result from `Sequence.solve`.
//! - `sequence_deformation_preconditioning` free function.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use geopyv_dev::{
    mesh::{SeedConfig, SolveConfig, SolveMethod},
    sequence::{self, Sequence, SequenceMeshConfig, SequenceOptions, SequenceSolveConfig},
    masks::LocalMask,
};

use crate::{
    py_geometry::extract_region,
    py_mesh::PyMesh,
    py_mask::PyMask,
    Error,
};

// ---------------------------------------------------------------------------
// SequenceOptions class
// ---------------------------------------------------------------------------

/// Temporal coupling options for :meth:`Sequence.solve`.
///
/// Parameters
/// ----------
/// guide : bool, optional
///     Particle-based warp preconditioning between pairs. Default ``True``.
/// sequential : bool, optional
///     Advance reference after each successful solve. Default ``False``.
/// sync : bool, optional
///     Reuse previous mesh geometry for next pair (skips CDT). Default ``True``.
/// override_ : bool, optional
///     Relax tolerance on retry after a failed non-consecutive pair. Default ``False``.
#[pyclass(name = "SequenceOptions")]
#[derive(Clone)]
pub struct PySequenceOptions {
    pub guide: bool,
    pub sequential: bool,
    pub sync: bool,
    pub override_: bool,
}

#[pymethods]
impl PySequenceOptions {
    #[new]
    #[pyo3(signature = (guide=true, sequential=false, sync=true, override_=false))]
    fn new(guide: bool, sequential: bool, sync: bool, override_: bool) -> Self {
        PySequenceOptions { guide, sequential, sync, override_ }
    }

    #[getter]
    fn guide(&self) -> bool { self.guide }

    #[getter]
    fn sequential(&self) -> bool { self.sequential }

    #[getter]
    fn sync(&self) -> bool { self.sync }

    #[getter]
    fn override_(&self) -> bool { self.override_ }

    fn __repr__(&self) -> String {
        format!(
            "SequenceOptions(guide={}, sequential={}, sync={}, override_={})",
            self.guide, self.sequential, self.sync, self.override_,
        )
    }
}

// ---------------------------------------------------------------------------
// SequenceSolution class
// ---------------------------------------------------------------------------

/// Read-only result from :meth:`Sequence.solve`.
#[pyclass(name = "SequenceSolution")]
pub struct PySequenceSolution {
    pub(crate) inner: geopyv_dev::sequence::SequenceSolution,
}

#[pymethods]
impl PySequenceSolution {
    /// Per-pair DIC solutions; ``mesh_solutions[i]`` is for pair ``(i, i+1)``.
    /// Empty when the sequence was solved with ``save`` set to a directory path.
    #[getter]
    fn mesh_solutions(&self, py: Python<'_>) -> PyResult<Vec<Py<PyMesh>>> {
        self.inner
            .mesh_solutions
            .iter()
            .map(|s| Py::new(py, PyMesh::from_solution(py, s.clone())?))
            .collect()
    }

    /// File paths of per-frame ``.pyv`` files when solved with ``save`` set.
    /// Empty list when meshes are held in memory.
    #[getter]
    fn mesh_paths(&self) -> Vec<String> {
        self.inner
            .mesh_paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect()
    }

    /// ``True`` when all image pairs solved successfully.
    #[getter]
    fn solved(&self) -> bool {
        self.inner.solved
    }

    /// ``True`` when a consecutive pair was unsolvable and the sequence was curtailed.
    #[getter]
    fn unsolvable(&self) -> bool {
        self.inner.unsolvable
    }

    /// g-indices (1-based) at which tolerance override was active.
    #[getter]
    fn override_log(&self) -> Vec<i64> {
        self.inner.override_log.iter().map(|&x| x as i64).collect()
    }

    fn __repr__(&self) -> String {
        let n_pairs = self.inner.mesh_solutions.len() + self.inner.mesh_paths.len();
        let geom = self.inner.mesh_solutions.first().map(|m| {
            format!(", nodes={}, mesh_order={}, subset_order={}",
                m.nodes.nrows(), m.mesh_order, m.subset_order)
        }).unwrap_or_default();
        format!(
            "SequenceSolution(pairs={}{}, solved={})",
            n_pairs, geom, self.inner.solved,
        )
    }
}

// ---------------------------------------------------------------------------
// Sequence class
// ---------------------------------------------------------------------------

/// Multi-image-pair DIC sequence controller.
///
/// Parameters
/// ----------
/// image_dir : str
///     Directory containing the images.  All ``*.jpg``, ``*.jpeg``, and
///     ``*.png`` files are collected and sorted by trailing integer in
///     their filename stem.
/// boundary : CircleRegion, PathRegion, or numpy.ndarray (N, 2)
///     Boundary region or raw polygon vertices. Raw arrays imply ``boundary_hard=False``.
/// target_nodes : int
///     Target node count for binary-search sizing.
/// size : tuple[float, float], optional
///     ``(size_lower, size_upper)`` element edge lengths. Default ``(1.0, 1000.0)``.
/// exclusions : list[CircleRegion | PathRegion | numpy.ndarray], optional
///     Exclusion regions. Hard/soft flag is taken from each region object;
///     raw arrays default to soft. Default None.
/// mesh_order : int, optional
///     1 (linear) or 2 (quadratic). Default 2.
#[pyclass(name = "Sequence")]
pub struct PySequence {
    inner: Sequence,
}

#[pymethods]
impl PySequence {
    #[new]
    #[pyo3(signature = (image_dir, boundary, target_nodes,
                         size=(1.0, 1000.0), exclusions=None, mesh_order=2))]
    fn new(
        image_dir: &str,
        boundary: &Bound<'_, PyAny>,
        target_nodes: usize,
        size: (f64, f64),
        exclusions: Option<Vec<Bound<'_, PyAny>>>,
        mesh_order: u8,
    ) -> PyResult<Self> {
        let (boundary_nodes, boundary_hard) = extract_region(boundary)?;
        let excl_list = exclusions.unwrap_or_default();
        let (excl_owned, excl_hard): (Vec<Array2<f64>>, Vec<bool>) = excl_list
            .iter()
            .map(|obj| extract_region(obj))
            .collect::<PyResult<Vec<_>>>()?
            .into_iter()
            .unzip();
        let mesh_cfg = SequenceMeshConfig {
            boundary_nodes,
            boundary_hard,
            exclusion_nodes: excl_owned,
            exclusions_hard: excl_hard,
            size,
            target_nodes,
            mesh_order,
        };
        let seq = Sequence::from_dir(std::path::Path::new(image_dir), mesh_cfg)
            .map_err(Error::from)?;
        Ok(PySequence { inner: seq })
    }

    /// Solve all image pairs.
    ///
    /// Parameters
    /// ----------
    /// template : Template
    ///     Subset template whose pixel offsets define the subset shape.
    /// seed_coord : list[float]
    ///     Initial ``[x, y]`` seed coordinate near low-deformation region.
    /// seed_warp : list[float], optional
    ///     Initial warp vector for seed node (length 6 or 12). Defaults to zeros.
    /// max_norm : float, optional
    ///     Convergence criterion. Default 1e-5.
    /// max_iterations : int, optional
    ///     Iteration limit per subset node. Default 50.
    /// subset_order : int, optional
    ///     1 (affine) or 2 (quadratic). Default 2.
    /// tolerance : float, optional
    ///     Minimum acceptable C_ZNCC for propagated nodes. Default 0.75.
    /// seed_tolerance : float, optional
    ///     Minimum acceptable C_ZNCC for the seed node. Default 0.9.
    /// method : str, optional
    ///     ``"icgn"`` (default) or ``"fagn"``.
    /// options : SequenceOptions, optional
    ///     Temporal coupling strategy. Default ``SequenceOptions()``.
    /// border : int, optional
    ///     Image border (pixels) for B-spline precomputation. Default 20.
    ///
    /// Returns
    /// -------
    /// SequenceSolution
    #[pyo3(signature = (local_mask, seed_coord, seed_warp=None,
                         max_norm=1e-5, max_iterations=50, subset_order=2,
                         tolerance=0.75, seed_tolerance=0.9, method="icgn",
                         options=None, border=20, save=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        local_mask: &Bound<'_, PyAny>,
        seed_coord: [f64; 2],
        seed_warp: Option<Vec<f64>>,
        max_norm: f64,
        max_iterations: usize,
        subset_order: usize,
        tolerance: f64,
        seed_tolerance: f64,
        method: &str,
        options: Option<PyRef<'_, PySequenceOptions>>,
        border: usize,
        save: Option<&str>,
    ) -> PyResult<PySequenceSolution> {
        let solve_method = if method == "fagn" {
            SolveMethod::Fagn
        } else {
            SolveMethod::Icgn
        };
        let tmpl = local_mask
            .extract::<PyRef<'_, PyMask>>()
            .map_err(|_| PyTypeError::new_err("local_mask must be a Mask"))?;
        let local_mask: LocalMask = tmpl.local_mask_ref()?.clone();
        let warp_len = 6 * subset_order;
        let warp = seed_warp.unwrap_or_else(|| vec![0.0; warp_len]);
        let seq_options = options.map(|o| SequenceOptions {
            guide: o.guide,
            sequential: o.sequential,
            sync: o.sync,
            override_: o.override_,
        }).unwrap_or_default();
        let cfg = SequenceSolveConfig {
            mesh_cfg: SolveConfig {
                max_norm,
                max_iterations,
                subset_order,
                tolerance,
                method: solve_method,
            },
            local_mask,
            seed: SeedConfig {
                coord: seed_coord,
                warp,
                tolerance: seed_tolerance,
            },
            options: seq_options,
            border,
            save: save.map(PathBuf::from),
        };
        let sol = self.inner.solve(&cfg).map_err(Error::from)?;
        Ok(PySequenceSolution { inner: sol })
    }

    /// Number of image pairs (= number of meshes to solve).
    #[getter]
    fn n_pairs(&self) -> usize {
        self.inner.n_pairs()
    }

    /// Ordered list of image file paths discovered from the directory.
    #[getter]
    fn image_paths(&self) -> Vec<String> {
        self.inner
            .image_paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sequence(images={}, pairs={})",
            self.inner.image_paths.len(),
            self.inner.n_pairs(),
        )
    }
}

// ---------------------------------------------------------------------------
// Free function: sequence_deformation_preconditioning
// ---------------------------------------------------------------------------

/// Extract seed displacement and warp from a solved mesh for the next pair.
///
/// Replicates ``Sequence._deformation_preconditioning``.
///
/// Parameters
/// ----------
/// mesh : Mesh
///     Solved mesh result from the previous pair.
/// seed_coord : list[float]
///     ``[x, y]`` seed coordinate.
/// mesh_order : int
///     Mesh element order (1 or 2).
/// subset_order : int
///     Subset warp order (1 or 2).
///
/// Returns
/// -------
/// tuple (seed_displacement, seed_warp):
///     seed_displacement – shape ``(2,)`` float64
///     seed_warp         – shape ``(6*subset_order,)`` float64
#[pyfunction]
#[pyo3(signature = (mesh, seed_coord, mesh_order, subset_order))]
fn sequence_deformation_preconditioning<'py>(
    py: Python<'py>,
    mesh: PyRef<'_, PyMesh>,
    seed_coord: [f64; 2],
    mesh_order: u8,
    subset_order: u8,
) -> PyResult<Bound<'py, PyTuple>> {
    let sol = mesh.solution.as_ref().ok_or_else(|| {
        PyRuntimeError::new_err("Mesh has not been solved")
    })?;
    let (disp, warp) =
        sequence::deformation_preconditioning(sol, seed_coord, mesh_order, subset_order);
    let disp_arr = Array1::from(vec![disp[0], disp[1]]);
    let warp_arr = Array1::from(warp);
    Ok(PyTuple::new_bound(
        py,
        [
            disp_arr.into_pyarray_bound(py).into_any(),
            warp_arr.into_pyarray_bound(py).into_any(),
        ],
    ))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySequenceOptions>()?;
    m.add_class::<PySequence>()?;
    m.add_class::<PySequenceSolution>()?;
    m.add_function(wrap_pyfunction!(sequence_deformation_preconditioning, m)?)?;
    Ok(())
}
