//! PyO3 wrapper for `geopyv_dev::sequence`.
//!
//! Exposes:
//! - `SequenceOptions` class: temporal coupling options for `Sequence.solve`.
//! - `Sequence` class: constructed with image directory + mesh config, runs `solve`
//!   (mutates in place; solution stored on the object).
//! - `sequence_deformation_preconditioning` free function.

use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use geopyv_dev::{
    image::Image,
    mesh::{Preconditioning, SeedConfig, SolveConfig, SolveMethod},
    sequence::{self, Sequence, SequenceMeshConfig, SequenceOptions, SequenceSolveConfig},
    masks::LocalMask,
};

use crate::{
    py_geometry::{extract_region_full, write_region_state},
    py_image::PyImage,
    py_mesh::PyMesh,
    py_mask::PyMask,
    py_particle::PyMeshlessParams,
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
    pub(crate) inner: Sequence,
    /// Original boundary/exclusion objects passed to `new()`, retained so
    /// `solve()` can write the final tracked region state back into them.
    /// `None` when reconstructed from a saved solution (`from_solution`).
    boundary_obj: Option<Py<PyAny>>,
    exclusion_objs: Vec<Py<PyAny>>,
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
        let boundary_region = extract_region_full(boundary)?;
        let excl_list = exclusions.unwrap_or_default();
        let excl_regions: Vec<geopyv_dev::geometry::region::Region> = excl_list
            .iter()
            .map(|obj| extract_region_full(obj))
            .collect::<PyResult<Vec<_>>>()?;
        let mesh_cfg = SequenceMeshConfig {
            boundary: boundary_region,
            exclusions: excl_regions,
            size,
            target_nodes,
            mesh_order,
        };
        let seq = Sequence::from_dir(std::path::Path::new(image_dir), mesh_cfg)
            .map_err(Error::from)?;
        let boundary_obj = Some(boundary.clone().unbind());
        let exclusion_objs = excl_list.iter().map(|o| o.clone().unbind()).collect();
        Ok(PySequence { inner: seq, boundary_obj, exclusion_objs })
    }

    /// Solve all image pairs. Mutates in place; returns ``None``.
    ///
    /// Parameters
    /// ----------
    /// local_mask : Mask
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
    /// preconditioning : str, optional
    ///     Intra-mesh RG frontier traversal strategy, forwarded to every
    ///     pair's `Mesh.solve` unchanged -- ``"RG"`` (default) or
    ///     ``"layer-RG"`` (layer-parallel; see
    ///     `geopyv_dev_fresh/layer_rg_plan.md`).
    /// masking : str, optional
    ///     The ``masking`` axis, forwarded to every pair's `Mesh.solve`
    ///     unchanged -- ``"uniform"`` (default) or ``"zonal"``. See
    ///     `Mesh.solve`'s own docstring and
    ///     `geopyv_dev_fresh/solver_options_restructure.md` §4's Stage B.
    ///     `Sequence::solve`'s per-pair loop needed no structural change to
    ///     support this -- it already calls `mesh.solve` once per pair.
    /// zonal_k, zonal_smoothing_sigma, zonal_meshless_params, zonal_iterations
    ///     : optional. Same meaning as on `Mesh.solve`; ``masking="zonal"``
    ///     only (``ValueError`` with ``masking="uniform"``). There is no
    ///     ``zonal_zone_map`` here: one caller-supplied map has no sensible
    ///     meaning across a sequence of increments.
    #[pyo3(signature = (local_mask=None, seed_coord=None, seed_warp=None,
                         max_norm=1e-5, max_iterations=50, subset_order=2,
                         tolerance=0.75, seed_tolerance=0.9, method="icgn",
                         options=None, border=20, save=None,
                         preconditioning="RG",
                         masking="uniform",
                         zonal_k=None,
                         zonal_smoothing_sigma=None,
                         zonal_meshless_params=None,
                         zonal_iterations=None,
                         layer_rg_batch_factor=4,
                         layer_rg_root_rel_eps=0.02,
                         layer_rg_max_workers=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        py: Python<'_>,
        local_mask: Option<&Bound<'_, PyAny>>,
        seed_coord: Option<[f64; 2]>,
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
        preconditioning: &str,
        masking: &str,
        zonal_k: Option<f64>,
        zonal_smoothing_sigma: Option<f64>,
        zonal_meshless_params: Option<PyRef<'_, PyMeshlessParams>>,
        zonal_iterations: Option<usize>,
        layer_rg_batch_factor: usize,
        layer_rg_root_rel_eps: f64,
        layer_rg_max_workers: Option<usize>,
    ) -> PyResult<()> {
        let local_mask = local_mask.ok_or_else(|| {
            PyTypeError::new_err("local_mask is required")
        })?;
        let seed_coord = seed_coord.ok_or_else(|| {
            PyTypeError::new_err("seed_coord is required")
        })?;

        let solve_method = if method == "fagn" {
            SolveMethod::Fagn
        } else {
            SolveMethod::Icgn
        };
        let preconditioning = match preconditioning {
            "RG" => Preconditioning::Rg,
            "layer-RG" => Preconditioning::LayerRg,
            other => {
                return Err(PyValueError::new_err(format!(
                    "preconditioning must be 'RG' or 'layer-RG', got {other:?}"
                )))
            }
        };
        if layer_rg_batch_factor == 0 {
            return Err(PyValueError::new_err("layer_rg_batch_factor must be >= 1"));
        }
        let layer_rg = geopyv_dev::mesh::LayerRgConfig {
            batch_factor: layer_rg_batch_factor,
            root_rel_eps: layer_rg_root_rel_eps,
            max_workers: layer_rg_max_workers,
        };
        let masking = crate::py_mesh::masking_from_py(
            masking,
            zonal_k,
            zonal_smoothing_sigma,
            zonal_meshless_params.map(|p| p.inner.clone()),
            zonal_iterations,
            None,
        )?;
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
                override_active: false,
                preconditioning,
                layer_rg,
                masking,
                ..Default::default()
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
        self.inner.solve(&cfg, None).map_err(Error::from)?;
        let sol = self.inner.solution().expect("solve() succeeded, solution must be Some");

        // Write the final tracked boundary/exclusion state back into the
        // original Python region objects passed to `new()`, so the caller
        // can inspect e.g. `boundary_obj.current_nodes` post-solve. No-op
        // for raw-ndarray inputs (nothing to write back into) or when this
        // `Sequence` was reconstructed from a saved solution.
        if let Some(ref obj) = self.boundary_obj {
            write_region_state(obj.bind(py), &sol.boundary_region)?;
        }
        for (obj, region) in self.exclusion_objs.iter().zip(&sol.exclusion_regions) {
            write_region_state(obj.bind(py), region)?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Solution getters — guarded by require_solved()
    // -----------------------------------------------------------------------

    /// Per-pair DIC solutions; ``mesh_solutions[i]`` is for pair ``(i, i+1)``.
    /// Empty list when the sequence was solved with ``save`` set to a directory path.
    #[getter]
    fn mesh_solutions(&self, py: Python<'_>) -> PyResult<Vec<Py<PyMesh>>> {
        let sol = self.require_solved()?;
        (0..sol.n_meshes())
            .map(|i| {
                let ms = sol.load_mesh_at(i).map_err(Error::from)?;
                Py::new(py, PyMesh::from_solution(py, ms)?)
            })
            .collect()
    }

    /// Load a single per-pair DIC solution by index without materialising the rest.
    ///
    /// Prefer this over ``mesh_solutions[i]`` when only one pair is needed — the
    /// full ``mesh_solutions`` getter loads every pair (including re-reading both
    /// images from disk for each) before slicing.
    fn mesh_solution_at(&self, py: Python<'_>, idx: usize) -> PyResult<Py<PyMesh>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(idx).map_err(Error::from)?;
        Py::new(py, PyMesh::from_solution(py, ms)?)
    }

    /// Per-pair C_ZNCC arrays without loading images.
    ///
    /// Returns a list of 1-D numpy arrays (one per pair).  Much cheaper than
    /// ``mesh_solutions`` when only correlation scores are needed.
    fn all_c_zncc(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let sol = self.require_solved()?;
        (0..sol.n_meshes())
            .map(|i| {
                let ms = sol.load_mesh_at(i).map_err(Error::from)?;
                Ok(ms.c_zncc.clone().into_pyarray_bound(py).into_any().unbind())
            })
            .collect()
    }

    /// File paths of per-frame ``.pyv`` files when solved with ``save`` set.
    /// Empty list when meshes are held in memory.
    #[getter]
    fn mesh_paths(&self) -> PyResult<Vec<String>> {
        let sol = self.require_solved()?;
        Ok(sol.mesh_paths.iter().map(|p| p.to_string_lossy().into_owned()).collect())
    }

    /// ``True`` once ``solve()`` has been called (regardless of quality).
    /// See ``all_converged`` for whether every pair succeeded.
    #[getter]
    fn solved(&self) -> bool {
        self.inner.solved()
    }

    /// ``True`` when all image pairs solved successfully.
    #[getter]
    fn all_converged(&self) -> PyResult<bool> {
        Ok(self.require_solved()?.all_converged)
    }

    /// ``True`` when a consecutive pair was unsolvable and the sequence was curtailed.
    #[getter]
    fn unsolvable(&self) -> PyResult<bool> {
        Ok(self.require_solved()?.unsolvable)
    }

    /// g-indices (1-based) at which tolerance override was active.
    #[getter]
    fn override_log(&self) -> PyResult<Vec<i64>> {
        Ok(self.require_solved()?.override_log.iter().map(|&x| x as i64).collect())
    }

    /// One entry per mesh pair; ``True`` if the reference image advanced at that step.
    #[getter]
    fn reference_updates(&self) -> PyResult<Vec<bool>> {
        Ok(self.require_solved()?.reference_updates.clone())
    }

    // -----------------------------------------------------------------------
    // Geometry / metadata getters — always available
    // -----------------------------------------------------------------------

    /// Number of image pairs (= number of meshes to solve).
    #[getter]
    fn n_pairs(&self) -> usize {
        if self.inner.image_paths.is_empty() {
            return self.inner.solution().map(|s| s.n_meshes()).unwrap_or(0);
        }
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

    // -----------------------------------------------------------------------
    // Per-mesh field accessors — mirror PyMesh getters with mesh_index
    // -----------------------------------------------------------------------

    /// Node coordinates for mesh ``mesh_index``, shape ``(N, 2)``.
    /// Pass ``subset_index`` to extract a single row ``(2,)``.
    #[pyo3(signature = (mesh_index, subset_index=None))]
    fn nodes<'py>(&self, py: Python<'py>, mesh_index: usize, subset_index: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(match subset_index {
            None => ms.nodes.clone().into_pyarray_bound(py).into_any(),
            Some(i) => {
                if i >= ms.nodes.nrows() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        format!("subset_index {i} out of range (N={})", ms.nodes.nrows())
                    ));
                }
                ms.nodes.row(i).to_owned().into_pyarray_bound(py).into_any()
            }
        })
    }

    /// Element connectivity for mesh ``mesh_index``, shape ``(M, 3)`` or ``(M, 6)``.
    /// Pass ``element_index`` to extract a single row.
    #[pyo3(signature = (mesh_index, element_index=None))]
    fn elements<'py>(&self, py: Python<'py>, mesh_index: usize, element_index: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        if let Some(i) = element_index {
            if i >= ms.elements.nrows() {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    format!("element_index {i} out of range (M={})", ms.elements.nrows())
                ));
            }
            return Ok(ms.elements.row(i).mapv(|x| x as i64).into_pyarray_bound(py).into_any());
        }
        let e: Array2<i64> = ms.elements.map(|&x| x as i64);
        Ok(e.into_pyarray_bound(py).into_any())
    }

    /// Boundary node indices for mesh ``mesh_index``.
    fn boundary(&self, mesh_index: usize) -> PyResult<Vec<i64>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(ms.boundary.iter().map(|&x| x as i64).collect())
    }

    /// Exclusion node index groups for mesh ``mesh_index``.
    fn exclusions(&self, mesh_index: usize) -> PyResult<Vec<Vec<i64>>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(ms.exclusions.iter().map(|g| g.iter().map(|&x| x as i64).collect()).collect())
    }

    /// Signed element areas for mesh ``mesh_index``, shape ``(M,)``.
    /// Pass ``element_index`` to get a single scalar.
    #[pyo3(signature = (mesh_index, element_index=None))]
    fn areas<'py>(&self, py: Python<'py>, mesh_index: usize, element_index: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(match element_index {
            None => ms.areas.clone().into_pyarray_bound(py).into_any(),
            Some(i) => {
                if i >= ms.areas.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        format!("element_index {i} out of range (M={})", ms.areas.len())
                    ));
                }
                pyo3::types::PyFloat::new_bound(py, ms.areas[i]).into_any()
            }
        })
    }

    /// Element warp vectors for mesh ``mesh_index``, shape ``(M, 12)``.
    /// Pass ``element_index`` to extract a single row.
    #[pyo3(signature = (mesh_index, element_index=None))]
    fn warps<'py>(&self, py: Python<'py>, mesh_index: usize, element_index: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(match element_index {
            None => ms.warps.clone().into_pyarray_bound(py).into_any(),
            Some(i) => {
                if i >= ms.warps.nrows() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        format!("element_index {i} out of range (M={})", ms.warps.nrows())
                    ));
                }
                ms.warps.row(i).to_owned().into_pyarray_bound(py).into_any()
            }
        })
    }

    /// Per-node displacements for mesh ``mesh_index``, shape ``(N, 2)``.
    /// Pass ``subset_index`` to extract a single row ``(2,)``.
    #[pyo3(signature = (mesh_index, subset_index=None))]
    fn displacements<'py>(&self, py: Python<'py>, mesh_index: usize, subset_index: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(match subset_index {
            None => ms.displacements.clone().into_pyarray_bound(py).into_any(),
            Some(i) => {
                if i >= ms.displacements.nrows() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        format!("subset_index {i} out of range (N={})", ms.displacements.nrows())
                    ));
                }
                ms.displacements.row(i).to_owned().into_pyarray_bound(py).into_any()
            }
        })
    }

    /// Per-node ZNCC scores for mesh ``mesh_index``, shape ``(N,)``.
    /// Pass ``subset_index`` to get a single scalar.
    #[pyo3(signature = (mesh_index, subset_index=None))]
    fn c_zncc<'py>(&self, py: Python<'py>, mesh_index: usize, subset_index: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(match subset_index {
            None => ms.c_zncc.clone().into_pyarray_bound(py).into_any(),
            Some(i) => {
                if i >= ms.c_zncc.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        format!("subset_index {i} out of range (N={})", ms.c_zncc.len())
                    ));
                }
                pyo3::types::PyFloat::new_bound(py, ms.c_zncc[i]).into_any()
            }
        })
    }

    /// Per-node warp parameters for mesh ``mesh_index``, shape ``(N, 6)`` or ``(N, 12)``.
    /// Pass ``subset_index`` to extract a single row.
    #[pyo3(signature = (mesh_index, subset_index=None))]
    fn p<'py>(&self, py: Python<'py>, mesh_index: usize, subset_index: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(match subset_index {
            None => ms.p.clone().into_pyarray_bound(py).into_any(),
            Some(i) => {
                if i >= ms.p.nrows() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        format!("subset_index {i} out of range (N={})", ms.p.nrows())
                    ));
                }
                ms.p.row(i).to_owned().into_pyarray_bound(py).into_any()
            }
        })
    }

    /// Per-node iteration counts for mesh ``mesh_index``, shape ``(N,)``.
    /// Pass ``subset_index`` to get a single integer.
    #[pyo3(signature = (mesh_index, subset_index=None))]
    fn iterations<'py>(&self, py: Python<'py>, mesh_index: usize, subset_index: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(match subset_index {
            None => ms.iterations.clone().into_pyarray_bound(py).into_any(),
            Some(i) => {
                if i >= ms.iterations.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        format!("subset_index {i} out of range (N={})", ms.iterations.len())
                    ));
                }
                (ms.iterations[i] as i64).into_py(py).into_bound(py)
            }
        })
    }

    /// Per-node final Δnorm values for mesh ``mesh_index``, shape ``(N,)``.
    /// Pass ``subset_index`` to get a single scalar.
    #[pyo3(signature = (mesh_index, subset_index=None))]
    fn norms<'py>(&self, py: Python<'py>, mesh_index: usize, subset_index: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(match subset_index {
            None => ms.norms.clone().into_pyarray_bound(py).into_any(),
            Some(i) => {
                if i >= ms.norms.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        format!("subset_index {i} out of range (N={})", ms.norms.len())
                    ));
                }
                pyo3::types::PyFloat::new_bound(py, ms.norms[i]).into_any()
            }
        })
    }

    /// Seed node index for mesh ``mesh_index``.
    fn seed_node(&self, mesh_index: usize) -> PyResult<i64> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(ms.seed_node as i64)
    }

    /// Mesh element order for mesh ``mesh_index`` (1 or 2).
    fn mesh_order(&self, mesh_index: usize) -> PyResult<u8> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(ms.mesh_order)
    }

    /// Subset warp order for mesh ``mesh_index`` (1 or 2).
    fn subset_order(&self, mesh_index: usize) -> PyResult<u8> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(ms.subset_order)
    }

    /// Reference image path for mesh ``mesh_index``, or ``None``.
    fn f_img_path(&self, mesh_index: usize) -> PyResult<Option<String>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        let s = ms.f_img_path.to_string_lossy().into_owned();
        Ok(if s.is_empty() { None } else { Some(s) })
    }

    /// Target image path for mesh ``mesh_index``, or ``None``.
    fn g_img_path(&self, mesh_index: usize) -> PyResult<Option<String>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        let s = ms.g_img_path.to_string_lossy().into_owned();
        Ok(if s.is_empty() { None } else { Some(s) })
    }

    /// Reference image for mesh ``mesh_index``, or ``None`` if path missing or unreadable.
    fn f_img(&self, py: Python<'_>, mesh_index: usize) -> PyResult<Option<Py<PyImage>>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(load_image_opt(py, &ms.f_img_path))
    }

    /// Target image for mesh ``mesh_index``, or ``None`` if path missing or unreadable.
    fn g_img(&self, py: Python<'_>, mesh_index: usize) -> PyResult<Option<Py<PyImage>>> {
        let sol = self.require_solved()?;
        let ms = sol.load_mesh_at(mesh_index).map_err(Error::from)?;
        Ok(load_image_opt(py, &ms.g_img_path))
    }

    fn __repr__(&self) -> String {
        if let Some(sol) = self.inner.solution() {
            let n = sol.n_meshes();
            let n_images = if self.inner.image_paths.is_empty() { n + 1 } else { self.inner.image_paths.len() };
            format!("Sequence(images={}, pairs={}, all_converged={})", n_images, n, sol.all_converged)
        } else {
            format!(
                "Sequence(images={}, pairs={})",
                self.inner.image_paths.len(),
                self.inner.n_pairs(),
            )
        }
    }
}

fn load_image_opt(py: Python<'_>, path: &std::path::Path) -> Option<Py<PyImage>> {
    if path.as_os_str().is_empty() {
        return None;
    }
    let img = Image::from_file(path, 20).ok()?;
    let py_img = PyImage {
        inner: Arc::new(img),
        filepath: Some(path.to_string_lossy().into_owned()),
    };
    Py::new(py, py_img).ok()
}

impl PySequence {
    fn require_solved(&self) -> PyResult<&geopyv_dev::sequence::SequenceSolution> {
        self.inner.solution().ok_or_else(|| {
            PyRuntimeError::new_err("Sequence has not been solved; call solve() first")
        })
    }

    /// Reconstruct a `PySequence` shell from a deserialised `SequenceSolution`.
    ///
    /// Used by `load()` in `py_io.rs`. `SequenceMeshConfig` cannot be fully recovered
    /// from the solution (boundary polygon is not stored), so a minimal placeholder is
    /// used — it is never consulted once a solution is already present.
    pub(crate) fn from_solution(
        sol: geopyv_dev::sequence::SequenceSolution,
    ) -> PyResult<Self> {
        let image_paths: Vec<PathBuf> = if !sol.mesh_solutions.is_empty() {
            let mut paths: Vec<_> = sol.mesh_solutions.iter()
                .map(|m| m.f_img_path.clone())
                .collect();
            if let Some(last) = sol.mesh_solutions.last() {
                paths.push(last.g_img_path.clone());
            }
            paths
        } else {
            vec![]
        };
        let mesh_cfg = SequenceMeshConfig {
            boundary: sequence::default_boundary_region(),
            exclusions: vec![],
            size: (1.0, 1000.0),
            target_nodes: 0,
            mesh_order: sol.mesh_order,
        };
        let inner = Sequence::from_solution(image_paths, mesh_cfg, sol);
        Ok(PySequence { inner, boundary_obj: None, exclusion_objs: vec![] })
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
    let sol = mesh.inner.solution().ok_or_else(|| {
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
    m.add_function(wrap_pyfunction!(sequence_deformation_preconditioning, m)?)?;
    Ok(())
}
