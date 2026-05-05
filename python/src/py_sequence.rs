//! PyO3 wrapper for `geopyv_dev::sequence`.
//!
//! Exposes:
//! - `Sequence` class: constructed with image paths + mesh config, runs `solve`.
//! - `SequenceSolution` class: read-only result from `Sequence.solve`.
//! - `sequence_deformation_preconditioning` free function.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use geopyv_dev::{
    geometry::meshing,
    mesh::{SolveConfig, SolveMethod},
    sequence::{self, Sequence, SequenceMeshConfig, SequenceSolveConfig},
};

use crate::{
    py_geometry::extract_region,
    py_mesh::PyMeshSolution,
    py_templates::PyTemplate,
    Error,
};

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
    fn mesh_solutions(&self) -> Vec<PyMeshSolution> {
        self.inner
            .mesh_solutions
            .iter()
            .map(|s| PyMeshSolution { inner: s.clone() })
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
        let n = self.inner.mesh_solutions.len() + self.inner.mesh_paths.len();
        format!(
            "SequenceSolution(pairs={}, solved={}, unsolvable={})",
            n,
            self.inner.solved,
            self.inner.unsolvable,
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
/// image_paths : list[str]
///     Ordered list of image file paths (≥ 2).
/// boundary : CircleRegion, PathRegion, or numpy.ndarray (N, 2)
///     Boundary region or raw polygon vertices. Raw arrays imply ``boundary_hard=False``.
/// size_lower : float
///     Minimum element edge length.
/// size_upper : float
///     Maximum element edge length.
/// target_nodes : int
///     Target node count for binary-search sizing.
/// exclusions : list[CircleRegion | PathRegion | numpy.ndarray], optional
///     Exclusion regions or raw polygon arrays. Default None.
/// mesh_order : int, optional
///     1 (linear) or 2 (quadratic). Default 1.
#[pyclass(name = "Sequence")]
pub struct PySequence {
    inner: Sequence,
}

#[pymethods]
impl PySequence {
    #[new]
    #[pyo3(signature = (image_paths, boundary, size_lower, size_upper, target_nodes,
                         exclusions=None, mesh_order=1))]
    fn new(
        image_paths: Vec<String>,
        boundary: &Bound<'_, PyAny>,
        size_lower: f64,
        size_upper: f64,
        target_nodes: usize,
        exclusions: Option<Vec<Bound<'_, PyAny>>>,
        mesh_order: u8,
    ) -> PyResult<Self> {
        let (boundary_nodes, boundary_hard) = extract_region(boundary)?;
        let excl_owned: Vec<Array2<f64>> = exclusions
            .unwrap_or_default()
            .iter()
            .map(|obj| extract_region(obj).map(|(nodes, _)| nodes))
            .collect::<PyResult<Vec<_>>>()?;
        let excl_views: Vec<_> = excl_owned.iter().map(|a| a.view()).collect();
        let roi = meshing::define_roi(boundary_nodes.view(), boundary_hard, &excl_views, None);
        let paths: Vec<PathBuf> = image_paths.into_iter().map(PathBuf::from).collect();
        let mesh_cfg = SequenceMeshConfig {
            borders: roi.borders,
            segments: roi.segments,
            curves: roi.curves,
            size_lower,
            size_upper,
            target_nodes,
            mesh_order,
        };
        let seq = Sequence::new(paths, mesh_cfg).map_err(Error::from)?;
        Ok(PySequence { inner: seq })
    }

    /// Construct a Sequence by scanning a directory for image files.
    ///
    /// Parameters
    /// ----------
    /// image_dir : str
    ///     Directory containing the images.  All ``*.jpg``, ``*.jpeg``, and
    ///     ``*.png`` files are collected and sorted by trailing integer in
    ///     their filename stem.
    /// boundary : CircleRegion, PathRegion, or numpy.ndarray (N, 2)
    ///     Boundary region.
    /// size_lower, size_upper, target_nodes, exclusions, mesh_order
    ///     Same as :meth:`__init__`.
    #[staticmethod]
    #[pyo3(signature = (image_dir, boundary, size_lower, size_upper, target_nodes,
                         exclusions=None, mesh_order=1))]
    fn from_dir(
        image_dir: &str,
        boundary: &Bound<'_, PyAny>,
        size_lower: f64,
        size_upper: f64,
        target_nodes: usize,
        exclusions: Option<Vec<Bound<'_, PyAny>>>,
        mesh_order: u8,
    ) -> PyResult<Self> {
        let (boundary_nodes, boundary_hard) = extract_region(boundary)?;
        let excl_owned: Vec<Array2<f64>> = exclusions
            .unwrap_or_default()
            .iter()
            .map(|obj| extract_region(obj).map(|(nodes, _)| nodes))
            .collect::<PyResult<Vec<_>>>()?;
        let excl_views: Vec<_> = excl_owned.iter().map(|a| a.view()).collect();
        let roi = meshing::define_roi(boundary_nodes.view(), boundary_hard, &excl_views, None);
        let mesh_cfg = SequenceMeshConfig {
            borders: roi.borders,
            segments: roi.segments,
            curves: roi.curves,
            size_lower,
            size_upper,
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
    /// seed_warp : list[float]
    ///     Initial warp vector for seed node (length 6 or 12).
    /// max_norm : float, optional
    ///     Convergence criterion. Default 1e-5.
    /// max_iterations : int, optional
    ///     Iteration limit per subset node. Default 50.
    /// subset_order : int, optional
    ///     1 (affine) or 2 (quadratic). Default 1.
    /// tolerance : float, optional
    ///     Minimum acceptable C_ZNCC. Default 0.75.
    /// method : str, optional
    ///     ``"icgn"`` (default) or ``"fagn"``.
    /// guide : bool, optional
    ///     Particle-based warp preconditioning between pairs. Default ``True``.
    /// sequential : bool, optional
    ///     Advance reference after each successful solve. Default ``False``.
    /// sync : bool, optional
    ///     Reuse previous mesh geometry in sync mode. Default ``True``.
    /// override_ : bool, optional
    ///     Relax tolerance on retry after a failed non-consecutive pair. Default ``False``.
    /// border : int, optional
    ///     Image border (pixels) for B-spline precomputation. Default 20.
    ///
    /// Returns
    /// -------
    /// SequenceSolution
    #[pyo3(signature = (template, seed_coord, seed_warp,
                         max_norm=1e-5, max_iterations=50, subset_order=1,
                         tolerance=0.75, method="icgn",
                         guide=true, sequential=false, sync=true,
                         override_=false, border=20, save=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        template: &Bound<'_, PyAny>,
        seed_coord: [f64; 2],
        seed_warp: Vec<f64>,
        max_norm: f64,
        max_iterations: usize,
        subset_order: usize,
        tolerance: f64,
        method: &str,
        guide: bool,
        sequential: bool,
        sync: bool,
        override_: bool,
        border: usize,
        save: Option<&str>,
    ) -> PyResult<PySequenceSolution> {
        let solve_method = if method == "fagn" {
            SolveMethod::Fagn
        } else {
            SolveMethod::Icgn
        };
        let tmpl = template
            .extract::<PyRef<'_, PyTemplate>>()
            .map_err(|_| PyTypeError::new_err("template must be a Template"))?;
        let template_coords = tmpl.inner.coords.clone();
        let cfg = SequenceSolveConfig {
            mesh_cfg: SolveConfig {
                max_norm,
                max_iterations,
                subset_order,
                tolerance,
                method: solve_method,
            },
            template_coords,
            seed_coord,
            seed_warp,
            guide,
            sequential,
            sync,
            override_,
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

    /// Ordered list of image file paths.
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
            "Sequence(n_images={}, n_pairs={})",
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
/// sol : MeshSolution
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
#[pyo3(signature = (sol, seed_coord, mesh_order, subset_order))]
fn sequence_deformation_preconditioning<'py>(
    py: Python<'py>,
    sol: PyRef<'_, PyMeshSolution>,
    seed_coord: [f64; 2],
    mesh_order: u8,
    subset_order: u8,
) -> PyResult<Bound<'py, PyTuple>> {
    let (disp, warp) =
        sequence::deformation_preconditioning(&sol.inner, seed_coord, mesh_order, subset_order);
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
    m.add_class::<PySequence>()?;
    m.add_class::<PySequenceSolution>()?;
    m.add_function(wrap_pyfunction!(sequence_deformation_preconditioning, m)?)?;
    Ok(())
}
