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
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use geopyv_dev::image::Image;
use geopyv_dev::io::{save as io_save, GeopyvObject};
use geopyv_dev::masks::{LocalMask, MaskShape};
use geopyv_dev::mesh::{
    self, LayerRgConfig, Masking, Mesh, MeshSolution, Preconditioning, SeedConfig, SolveConfig,
    SolveMethod, ZonalConfig, ZonalMaskingRecord,
};

use crate::{
    py_geometry::extract_region,
    py_image::PyImage,
    py_mask::PyMask,
    py_particle::PyMeshlessParams,
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
    /// override_active : bool, optional
    ///     When ``True``, a mesh with one or more ``quality_ok == False``
    ///     subsets remaining after corrections is accepted instead of
    ///     raising -- those nodes' `p`/warps stay whatever the last solve
    ///     attempt produced, individually still flagged via
    ///     ``mesh.quality_ok`` (unaffected by this flag; only the
    ///     mesh-level accept/reject gate changes), rather than the whole
    ///     mesh failing outright. Intended for a small number of subsets
    ///     genuinely straddling a sharp discontinuity that no unmasked
    ///     warp can fit well -- not a general substitute for fixing a
    ///     mesh that's mostly failing. Default ``False`` (matches prior
    ///     behaviour exactly).
    /// masking : str, optional
    ///     The ``masking`` axis of ``solver_options`` -- ``"uniform"``
    ///     (default, today's plain solve) or ``"zonal"``. ``"zonal"`` runs
    ///     an ordinary solve, builds a ``Field`` at the mesh's own node
    ///     positions to get each node's shear strain, classifies nodes via
    ///     a robust threshold, rasterises that classification to a
    ///     whole-image zone label (meshless nearest-node lookup, not mesh-
    ///     element interpolation), then solves again with that label
    ///     applied as an internal per-subset exclusion mask. See
    ///     ``geopyv_dev_fresh/solver_options_restructure.md`` §4's Stage B.
    ///     Internal plumbing: the ``geopyv_dev`` Python package's own
    ///     ``Mesh.solve()`` wrapper validates ``solver_options`` before
    ///     reaching this binding. Every ``zonal_*`` argument below belongs
    ///     to ``masking="zonal"`` alone: passing any of them with
    ///     ``masking="uniform"`` raises ``ValueError``.
    /// zonal_k : float, optional
    ///     Robust-threshold multiplier for ``masking="zonal"``'s ridge
    ///     classification: a node sits on an interface ridge when its
    ///     ``|∇γ|`` exceeds ``median + zonal_k * 1.4826 * MAD`` across all
    ///     nodes. Default ``2.0``.
    /// zonal_smoothing_sigma : float, optional
    ///     Gaussian σ (pixels) applied to the rasterised ``|∇γ|`` image
    ///     before thresholding. ``None`` ⇒ the mesh's own node spacing.
    /// zonal_meshless_params : MeshlessParams, optional
    ///     Params for ``masking="zonal"``'s own internal field solve (the
    ///     one that computes each node's shear strain and its gradient).
    ///     Default ``None`` (``MeshlessParams``'s own defaults).
    /// zonal_iterations : int, optional
    ///     ``masking="zonal"`` classifier refit passes. ``1`` (default) =
    ///     classify once -- identical to the pre-iterator behaviour. ``>1``
    ///     re-solves the internal field ``zone_aware`` (same-zone neighbours
    ///     only), seeded by the current zone map, and reclassifies, up to
    ///     this many times or until the per-node zone partition converges.
    ///     Ignored when ``zonal_zone_map`` is supplied (nothing to refit).
    /// zonal_zone_map : np.ndarray (H, W), dtype uint8, optional
    ///     Caller-supplied whole-image zone-label grid matching the
    ///     reference image. When given (and ``masking="zonal"``), the
    ///     classifier is skipped and this grid drives pass 2's per-subset
    ///     masking directly -- use it to mask from a partition you already
    ///     know (e.g. from specimen geometry) rather than one detected from
    ///     the strain field. Reserve label ``0`` for "no zone". Default
    ///     ``None``.
    /// preconditioning : str, optional
    ///     Intra-mesh RG frontier traversal strategy -- ``"RG"`` (default,
    ///     today's serial cascade) or ``"layer-RG"`` (layer-parallel
    ///     independent-set rounds; see `geopyv_dev_fresh/layer_rg_plan.md`).
    ///     ``"layer-RG"`` agrees with ``"RG"`` within Tier C and is
    ///     run-to-run deterministic.
    /// layer_rg_batch_factor : int, optional
    ///     ``"layer-RG"`` only: frontier roots merged per parallel round
    ///     ``≈ batch_factor * workers``. Default ``4``.
    /// layer_rg_root_rel_eps : float, optional
    ///     ``"layer-RG"`` only: ε-band width -- a queued root joins the
    ///     round while its ``c_zncc`` is ``>= top * (1 - eps)``.
    ///     Default ``0.02``.
    /// layer_rg_max_workers : int, optional
    ///     ``"layer-RG"`` only: cap rayon workers for this solve's parallel
    ///     rounds. Default ``None`` (global pool).
    #[pyo3(signature = (local_mask, seed_coord, seed_warp=None,
                        max_norm=1e-5, max_iterations=50, subset_order=2,
                        tolerance=0.75, seed_tolerance=0.9, method="icgn",
                        override_active=false,
                        preconditioning="RG",
                        masking="uniform",
                        zonal_k=None,
                        zonal_smoothing_sigma=None,
                        zonal_meshless_params=None,
                        zonal_iterations=None,
                        zonal_zone_map=None,
                        layer_rg_batch_factor=4,
                        layer_rg_root_rel_eps=0.02,
                        layer_rg_max_workers=None))]
    #[allow(clippy::too_many_arguments)]
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
        override_active: bool,
        preconditioning: &str,
        masking: &str,
        zonal_k: Option<f64>,
        zonal_smoothing_sigma: Option<f64>,
        zonal_meshless_params: Option<PyRef<'_, PyMeshlessParams>>,
        zonal_iterations: Option<usize>,
        zonal_zone_map: Option<PyReadonlyArray2<u8>>,
        layer_rg_batch_factor: usize,
        layer_rg_root_rel_eps: f64,
        layer_rg_max_workers: Option<usize>,
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
        let solve_method = match method {
            "fagn" => SolveMethod::Fagn,
            _ => SolveMethod::Icgn,
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
        let layer_rg = LayerRgConfig {
            batch_factor: layer_rg_batch_factor,
            root_rel_eps: layer_rg_root_rel_eps,
            max_workers: layer_rg_max_workers,
        };
        let masking = masking_from_py(
            masking,
            zonal_k,
            zonal_smoothing_sigma,
            zonal_meshless_params.map(|p| p.inner.clone()),
            zonal_iterations,
            zonal_zone_map.map(|a| a.as_array().to_owned()),
        )?;
        let cfg = SolveConfig {
            max_norm,
            max_iterations,
            subset_order,
            tolerance,
            method: solve_method,
            override_active,
            preconditioning,
            layer_rg,
            masking,
        };
        self.inner.solve(local_mask_ref, &seed, &cfg, None).map_err(Error::from)?;
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
        self.inner.solved()
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

    /// Subset template shape (``"circle"`` / ``"square"`` /
    /// ``"semicircle"``), or ``None`` for a pre-``0x07`` (released-format) loaded ``.pyv``.
    #[getter]
    fn template_shape(&self) -> PyResult<Option<String>> {
        Ok(self.require_solved()?.template_shape.as_ref().map(mask_shape_name))
    }

    /// Per-node subset template size (radius / half-side), ``(N,)``, or
    /// ``None`` for a pre-``0x07`` (released-format) loaded ``.pyv``.
    #[getter]
    fn template_sizes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<u32>>>> {
        Ok(self
            .require_solved()?
            .template_sizes
            .as_ref()
            .map(|a| a.clone().into_pyarray_bound(py)))
    }

    /// Zonal-masking diagnostics, or ``None`` unless this mesh was solved
    /// with ``solver_options={"masking": "zonal"}`` and pass 2 succeeded.
    #[getter]
    fn zonal_masking(&self) -> PyResult<Option<PyZonalMaskingRecord>> {
        Ok(self
            .require_solved()?
            .zonal_masking
            .clone()
            .map(|inner| PyZonalMaskingRecord { inner }))
    }

    // -----------------------------------------------------------------------
    // IO
    // -----------------------------------------------------------------------

    /// Save the solved mesh to a ``.pyv`` file.
    fn save(&self, path: &str) -> PyResult<()> {
        let sol = self.require_solved_arc()?;
        io_save(path, &GeopyvObject::Mesh((**sol).clone())).map_err(Error::from)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Repr
    // -----------------------------------------------------------------------

    fn __repr__(&self) -> String {
        if let Some(sol) = self.inner.solution() {
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
        self.inner.solution().map(|arc| arc.as_ref()).ok_or_else(|| {
            PyRuntimeError::new_err("Mesh has not been solved; call solve() first")
        })
    }

    fn require_solved_arc(&self) -> PyResult<&Arc<MeshSolution>> {
        self.inner.solution().ok_or_else(|| {
            PyRuntimeError::new_err("Mesh has not been solved; call solve() first")
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
        let inner = Mesh::from_solution(sol, Arc::clone(&dummy), dummy);
        Ok(PyMesh {
            inner,
            f_img: None,
            g_img: None,
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

        self.inner = Mesh::from_solution(sol, f_arc, g_arc);
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
                solve_config: None,
                seed: None,
                template_shape: None,
                template_sizes: None,
                zonal_masking: None,
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

    #[getter]
    fn template_shape(&self) -> Option<String> {
        self.inner.template_shape.as_ref().map(mask_shape_name)
    }

    #[getter]
    fn template_sizes<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u32>>> {
        self.inner
            .template_sizes
            .as_ref()
            .map(|a| a.clone().into_pyarray_bound(py))
    }

    #[getter]
    fn zonal_masking(&self) -> Option<PyZonalMaskingRecord> {
        self.inner
            .zonal_masking
            .clone()
            .map(|inner| PyZonalMaskingRecord { inner })
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
// Zonal-masking inspection
// ---------------------------------------------------------------------------

fn mask_shape_name(shape: &MaskShape) -> String {
    match shape {
        MaskShape::Circle => "circle",
        MaskShape::Square => "square",
        MaskShape::Semicircle => "semicircle",
    }
    .to_string()
}

/// Read-only view of a solved mesh's zonal-masking diagnostics — see
/// `geopyv_dev::mesh::ZonalMaskingRecord`. Returned by
/// ``Mesh.zonal_masking`` / ``MeshSolution.zonal_masking``; all per-node
/// arrays are indexed like ``mesh.nodes``.
#[pyclass(name = "ZonalMaskingRecord", module = "geopyv_dev._geopyv_dev")]
pub struct PyZonalMaskingRecord {
    inner: ZonalMaskingRecord,
}

#[pymethods]
impl PyZonalMaskingRecord {
    /// Stage-2 meshless shear strain (``gamma_max``) per node, ``(N,)`` —
    /// context only; the classifier keys off its gradient.
    #[getter]
    fn node_gamma_max<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.node_gamma_max.clone().into_pyarray_bound(py)
    }

    /// Stage-2 meshless shear-strain gradient magnitude ``|∇γ|`` per node,
    /// ``(N,)`` — the classifying signal.
    #[getter]
    fn node_gamma_max_grad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.node_gamma_max_grad.clone().into_pyarray_bound(py)
    }

    /// ``True`` where the node's ``|∇γ|`` exceeded ``grad_cutoff`` (sits on
    /// an interface ridge rather than inside a regime), ``(N,)``.
    #[getter]
    fn node_boundary(&self) -> Vec<bool> {
        self.inner.node_boundary.clone()
    }

    /// Zone id under each node's own centre in ``zone_image``, ``(N,)``.
    #[getter]
    fn node_zone<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.node_zone.clone().into_pyarray_bound(py)
    }

    /// Retained pixel count before the zone cut (shape ∩ boundary), ``(N,)``.
    #[getter]
    fn node_pre_px<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.node_pre_px.clone().into_pyarray_bound(py)
    }

    /// Retained pixel count after the zone cut
    /// (shape ∩ boundary ∩ zone), ``(N,)``.
    #[getter]
    fn node_post_px<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.node_post_px.clone().into_pyarray_bound(py)
    }

    /// ``True`` where the minimum-pixel guard rejected the zone cut and the
    /// node was solved un-zoned, ``(N,)``.
    #[getter]
    fn node_guard_fallback(&self) -> Vec<bool> {
        self.inner.node_guard_fallback.clone()
    }

    /// Whole-image zone-label grid pass 2 applied, ``(H, W)`` uint8.
    #[getter]
    fn zone_image<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        self.inner.zone_image.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn k(&self) -> f64 {
        self.inner.k
    }

    /// Median of the per-node ``|∇γ|``.
    #[getter]
    fn grad_median(&self) -> f64 {
        self.inner.grad_median
    }

    /// Median absolute deviation of the per-node ``|∇γ|``.
    #[getter]
    fn grad_mad(&self) -> f64 {
        self.inner.grad_mad
    }

    /// ``grad_median + k * 1.4826 * grad_mad`` — the ridge cut-off on ``|∇γ|``.
    #[getter]
    fn grad_cutoff(&self) -> f64 {
        self.inner.grad_cutoff
    }

    fn __repr__(&self) -> String {
        let n = self.inner.node_zone.len();
        let n_ridge = self.inner.node_boundary.iter().filter(|&&e| e).count();
        let n_fallback = self.inner.node_guard_fallback.iter().filter(|&&e| e).count();
        let n_zones = self
            .inner
            .zone_image
            .iter()
            .copied()
            .collect::<HashSet<u8>>()
            .len();
        format!(
            "ZonalMaskingRecord(nodes={n}, zones={n_zones}, ridge_nodes={n_ridge}, \
             guard_fallback={n_fallback}, k={:.3}, grad_cutoff={:.4})",
            self.inner.k, self.inner.grad_cutoff,
        )
    }
}

/// Reconstruct one subset's template footprint under a zone-label image:
/// which pixels of a ``(template_shape, template_size)`` ``LocalMask``
/// centred at ``node_coord`` survive ``zone_image``'s per-subset zone cut
/// (`LocalMask::zone_mask_update`), and which are removed.
///
/// Pure geometry — the mesh's boundary/exclusion mask is deliberately NOT
/// applied here (it only trims pixels at the mesh edge; the authoritative
/// pre/post counts are in ``ZonalMaskingRecord``). Returns a dict with
/// ``coords_kept`` ``(K, 2)`` and ``coords_cut`` ``(L, 2)``, both absolute
/// ``[x, y]`` pixel coordinates.
#[pyfunction]
fn zoned_subset_footprint<'py>(
    py: Python<'py>,
    zone_image: PyReadonlyArray2<u8>,
    node_coord: [f64; 2],
    template_shape: &str,
    template_size: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let base = match template_shape {
        "circle" => LocalMask::circle(template_size),
        "square" => LocalMask::square(template_size),
        "semicircle" => LocalMask::semicircle(template_size),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown template_shape {other:?} (expected circle/square/semicircle)"
            )))
        }
    }
    .map_err(Error::from)?;

    let full: Vec<(i64, i64)> = base
        .coords
        .outer_iter()
        .map(|r| (r[0] as i64, r[1] as i64))
        .collect();

    let mut cut = base.clone();
    cut.zone_mask_update(node_coord, zone_image.as_array());
    let kept_set: HashSet<(i64, i64)> = cut
        .coords
        .outer_iter()
        .map(|r| (r[0] as i64, r[1] as i64))
        .collect();

    let to_abs = |offsets: &[(i64, i64)]| -> Array2<f64> {
        let mut a = Array2::<f64>::zeros((offsets.len(), 2));
        for (i, &(dx, dy)) in offsets.iter().enumerate() {
            a[[i, 0]] = node_coord[0] + dx as f64;
            a[[i, 1]] = node_coord[1] + dy as f64;
        }
        a
    };

    let kept: Vec<(i64, i64)> = full.iter().copied().filter(|o| kept_set.contains(o)).collect();
    let removed: Vec<(i64, i64)> =
        full.iter().copied().filter(|o| !kept_set.contains(o)).collect();

    let d = PyDict::new_bound(py);
    d.set_item("coords_kept", to_abs(&kept).into_pyarray_bound(py))?;
    d.set_item("coords_cut", to_abs(&removed).into_pyarray_bound(py))?;
    Ok(d)
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

/// Build the `masking` axis from the binding's flat `masking=` /
/// `zonal_*` arguments (shared by `Mesh.solve` and `Sequence.solve`). Every
/// `zonal_*` argument belongs to `masking="zonal"` alone: supplying one with
/// `masking="uniform"` is an error rather than silently ignored.
pub(crate) fn masking_from_py(
    masking: &str,
    zonal_k: Option<f64>,
    zonal_smoothing_sigma: Option<f64>,
    zonal_meshless_params: Option<geopyv_dev::particle::MeshlessParams>,
    zonal_iterations: Option<usize>,
    zonal_zone_map: Option<Array2<u8>>,
) -> PyResult<Masking> {
    match masking {
        "uniform" => {
            let given: Vec<&str> = [
                ("zonal_k", zonal_k.is_some()),
                ("zonal_smoothing_sigma", zonal_smoothing_sigma.is_some()),
                ("zonal_meshless_params", zonal_meshless_params.is_some()),
                ("zonal_iterations", zonal_iterations.is_some()),
                ("zonal_zone_map", zonal_zone_map.is_some()),
            ]
            .iter()
            .filter(|(_, g)| *g)
            .map(|(n, _)| *n)
            .collect();
            if !given.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "{given:?} only apply with masking='zonal' (got masking='uniform')"
                )));
            }
            Ok(Masking::Uniform)
        }
        "zonal" => {
            let defaults = ZonalConfig::default();
            let iterations = zonal_iterations.unwrap_or(defaults.iterations);
            if iterations == 0 {
                return Err(PyValueError::new_err("zonal_iterations must be >= 1"));
            }
            Ok(Masking::Zonal(ZonalConfig {
                meshless_params: zonal_meshless_params.unwrap_or(defaults.meshless_params),
                k: zonal_k.unwrap_or(defaults.k),
                smoothing_sigma: zonal_smoothing_sigma,
                iterations,
                zone_map: zonal_zone_map,
            }))
        }
        other => Err(PyValueError::new_err(format!(
            "masking must be 'uniform' or 'zonal', got {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMesh>()?;
    m.add_class::<PyZonalMaskingRecord>()?;
    // PyMeshSolution intentionally NOT registered — serialisation boundary only.
    m.add_function(wrap_pyfunction!(zoned_subset_footprint, m)?)?;
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
