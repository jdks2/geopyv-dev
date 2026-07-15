//! DIC mesh solver and mesh geometry.
//!
//! Translates `geopyv/src/geopyv/mesh.py` (`MeshBase` + `Mesh` classes).
//! gmsh is **not** used; mesh generation delegates to
//! [`crate::geometry::triangulation`].
//!
//! # Architecture
//!
//! - [`Mesh`] is constructed from pre-computed geometry (nodes/elements) plus
//!   the reference/target QCQT arrays.
//! - [`Mesh::solve`] runs the reliability-guided DIC algorithm, stores per-node
//!   results in [`SolveState`], computes element areas / strains, and checks
//!   compatibility.
//! - Pure-math helpers (`element_area`, `element_strains`, etc.) are free
//!   functions so they can be called by `adaptive_remesh` logic without a
//!   `Mesh` borrow.

use std::collections::{BinaryHeap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use nalgebra::{DMatrix, DVector};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::{
    geometry::{meshing, triangulation},
    image::Image,
    subset::{Subset, SolveResult},
    masks::LocalMask,
    Error,
};

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Per-node DIC solve output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResult {
    pub p: Vec<f64>,
    pub c_zncc: f64,
    pub u: f64,
    pub v: f64,
}

/// Output of [`Mesh::solve`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshSolution {
    /// Node coordinates `(N, 2)`.
    pub nodes: Array2<f64>,
    /// Element connectivity `(M, 3)` or `(M, 6)`.
    pub elements: Array2<usize>,
    /// Boundary node indices.
    pub boundary: Vec<usize>,
    /// Exclusion node index groups.
    pub exclusions: Vec<Vec<usize>>,
    /// Element centroids `(M, 2)`, computed at solve time.
    pub centroids: Array2<f64>,
    /// Signed element areas `(M,)`.
    pub areas: Array1<f64>,
    /// Element warp vectors `(M, 12)`.
    pub warps: Array2<f64>,
    /// Per-node displacements `(N, 2)`.
    pub displacements: Array2<f64>,
    /// Per-node ZNCC scores `(N,)`.
    pub c_zncc: Array1<f64>,
    /// Per-node warp parameters `(N, 6 or 12)`.
    pub p: Array2<f64>,
    /// Index of the seed node.
    pub seed_node: usize,
    pub mesh_order: u8,
    pub subset_order: u8,
    /// Per-node iteration counts `(N,)`.
    pub iterations: Array1<u32>,
    /// Per-node final ∆norm values `(N,)`.
    pub norms: Array1<f64>,
    /// Reference image file path.
    pub f_img_path: PathBuf,
    /// Target image file path.
    pub g_img_path: PathBuf,
    /// Per-subset solver settings used to produce this solution.
    #[serde(default)]
    pub solve_config: Option<SolveConfig>,
    /// Seed-node parameters used to produce this solution.
    #[serde(default)]
    pub seed: Option<SeedConfig>,
}

// ---------------------------------------------------------------------------
// Solve configuration
// ---------------------------------------------------------------------------

/// Parameters forwarded to each subset solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveConfig {
    pub max_norm: f64,
    pub max_iterations: usize,
    pub subset_order: usize,
    pub tolerance: f64,
    pub method: SolveMethod,
    /// When `true`, `Mesh::solve` accepts a mesh with `quality_ok == false`
    /// subsets remaining after corrections instead of returning `Err`
    /// (mirrors `mesh.py`'s `self._override` bypass for status-3 "subset
    /// decorrelation"). Set by `sequence.rs` on a reference-update retry;
    /// `false` for a normal/first-attempt solve.
    #[serde(default)]
    pub override_active: bool,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            max_norm: 1e-5,
            max_iterations: 50,
            subset_order: 2,
            tolerance: 0.75,
            method: SolveMethod::Icgn,
            override_active: false,
        }
    }
}

/// Seed-node parameters for reliability-guided DIC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedConfig {
    /// Image coordinate `[x, y]` near a region of low deformation.
    pub coord: [f64; 2],
    /// Initial warp vector for the seed node (normalised to p_len before use).
    pub warp: Vec<f64>,
    /// Minimum acceptable C_ZNCC for the seed node (default 0.9).
    /// Stricter than the propagated-node tolerance in SolveConfig.
    pub tolerance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveMethod {
    Icgn,
    Fagn,
}

// ---------------------------------------------------------------------------
// Mesh struct
// ---------------------------------------------------------------------------

/// Mesh-level DIC solver.
///
/// Construct via [`Mesh::new`].
pub struct Mesh {
    nodes: Array2<f64>,
    elements: Array2<usize>,
    boundary: Vec<usize>,
    exclusions: Vec<Vec<usize>>,
    mesh_order: u8,
    /// Binary image mask (`Some` after `Mesh::new`; `None` after `from_solution`).
    mask: Option<Array2<u8>>,
    pub f_img: Arc<Image>,
    pub g_img: Arc<Image>,
    solution: Option<Arc<MeshSolution>>,
}

impl Mesh {
    /// Create a mesh: run `define_roi` + CDT, compute binary mask, store images.
    ///
    /// This is the single Rust entry point that replaces the previous separate
    /// `define_roi` + `Mesh::generate` call sequence.
    ///
    /// # Arguments
    /// * `boundary_nodes` — `(N, 2)` boundary polygon vertices `[x, y]`.
    /// * `boundary_hard`  — If `true`, fill only inside the boundary for the mask.
    /// * `exclusion_nodes` — Exclusion polygon arrays.
    /// * `exclusions_hard` — Per-exclusion hard flag (soft = meshing only, not mask).
    /// * `size`           — `(size_lower, size_upper)` element edge lengths.
    /// * `target_nodes`   — Target node count for binary-search sizing.
    /// * `mesh_order`     — 1 (linear) or 2 (quadratic).
    /// * `f_img`          — Reference image (shared ownership).
    /// * `g_img`          — Target image (shared ownership).
    pub fn new(
        boundary_nodes: ArrayView2<f64>,
        boundary_hard: bool,
        exclusion_nodes: &[ArrayView2<f64>],
        exclusions_hard: &[bool],
        size: (f64, f64),
        target_nodes: usize,
        mesh_order: u8,
        f_img: Arc<Image>,
        g_img: Arc<Image>,
    ) -> Result<Self, Error> {
        if size.0 <= 0.0 || size.0 >= size.1 {
            return Err(Error::InvalidInput(
                "size.0 must be > 0 and < size.1".to_string(),
            ));
        }
        if target_nodes < 1 {
            return Err(Error::InvalidInput(
                "target_nodes must be >= 1".to_string(),
            ));
        }
        let img_shape = f_img.image_gs.dim(); // (height, width)
        let roi = meshing::define_roi(
            boundary_nodes,
            boundary_hard,
            exclusion_nodes,
            exclusions_hard,
            Some(img_shape),
        );
        let tm = triangulation::generate_mesh(
            roi.borders.view(),
            roi.segments.view(),
            &roi.curves,
            size.0,
            size.1,
            target_nodes,
            mesh_order,
        )?;
        let order = if tm.elements.ncols() == 6 { 2 } else { 1 };
        Ok(Mesh {
            nodes: tm.nodes,
            elements: tm.elements,
            boundary: tm.boundary,
            exclusions: tm.exclusions,
            mesh_order: order,
            mask: roi.mask,
            f_img,
            g_img,
            solution: None,
        })
    }

    /// Reconstruct a `Mesh` topology from a previous [`MeshSolution`].
    ///
    /// Used by the sequence solver in `sync` mode. `mask` is `None` because
    /// the polygon data is not available; subsets are unmasked in this mode.
    /// The resulting mesh is marked as solved (`sol` becomes its stored solution).
    pub fn from_solution(sol: Arc<MeshSolution>, f_img: Arc<Image>, g_img: Arc<Image>) -> Self {
        Mesh {
            nodes: sol.nodes.clone(),
            elements: sol.elements.clone(),
            boundary: sol.boundary.clone(),
            exclusions: sol.exclusions.clone(),
            mesh_order: sol.mesh_order,
            mask: None,
            f_img,
            g_img,
            solution: Some(sol),
        }
    }

    /// `true` if this mesh has been solved (a `MeshSolution` is available).
    pub fn solved(&self) -> bool {
        self.solution.is_some()
    }

    /// The stored solve result, if any. Cheap to clone (`Arc`).
    pub fn solution(&self) -> Option<&Arc<MeshSolution>> {
        self.solution.as_ref()
    }

    /// Run the reliability-guided DIC solver.
    ///
    /// Images are taken from `self.f_img` / `self.g_img` (set at construction).
    ///
    /// # Arguments
    /// * `local_mask` — Subset local mask (cloned per node when a global mask is present).
    /// * `seed`       — Seed-node parameters (coord, initial warp, tolerance).
    /// * `cfg`        — Solver configuration.
    /// * `progress`   — Optional external progress bar (used by Sequence).
    pub fn solve(
        &mut self,
        local_mask: &LocalMask,
        seed: &SeedConfig,
        cfg: &SolveConfig,
        progress: Option<&indicatif::ProgressBar>,
    ) -> Result<(), Error> {
        let n_nodes = self.nodes.nrows();
        let p_len = 6 * cfg.subset_order;

        // Normalise seed.warp to exactly p_len elements.
        let seed_warp_norm: Vec<f64> = {
            let mut w = seed.warp.clone();
            w.resize(p_len, 0.0);
            w
        };

        let mut stored = vec![false; n_nodes];
        let mut propagated = vec![false; n_nodes];
        let mut quality_ok = Array1::<bool>::from_elem(n_nodes, false);
        let mut c_zncc = Array1::<f64>::zeros(n_nodes);
        let mut queue: BinaryHeap<(u64, usize)> = BinaryHeap::new();
        let mut p = Array2::<f64>::zeros((n_nodes, p_len));
        let mut displacements = Array2::<f64>::zeros((n_nodes, 2));
        let mut iterations = Array1::<u32>::zeros(n_nodes);
        let mut norms = Array1::<f64>::zeros(n_nodes);

        // Build per-node subsets, delegating masking to Subset::new.
        let subsets: Vec<Subset> = (0..n_nodes)
            .map(|i| {
                let coord = [self.nodes[[i, 0]], self.nodes[[i, 1]]];
                Subset::new(
                    coord,
                    local_mask,
                    self.mask.as_ref().map(|m| m.view()),
                    Arc::clone(&self.f_img),
                    Arc::clone(&self.g_img),
                    cfg.subset_order,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Progress bar: use provided one, or create a local one.
        let own_pb;
        let pb: &indicatif::ProgressBar = if let Some(p) = progress {
            p
        } else {
            own_pb = indicatif::ProgressBar::new(n_nodes as u64);
            own_pb.set_style(
                indicatif::ProgressStyle::with_template(
                    "  Solving mesh:  [{bar:40.cyan}] {pos}/{len} subsets  eta {eta}"
                )
                .unwrap()
                .progress_chars("█░"),
            );
            &own_pb
        };

        // --- Seed node (uses seed.tolerance, stricter than cfg.tolerance).
        // Forced preconditioning warp = the user's seed_warp; falls back to a
        // zero-warp second attempt on failure, same as any other node (see
        // `solve_node`) — unlike the old code, which had no seed fallback.
        let seed_node = find_seed_node(&self.nodes, seed.coord);
        let seed_cfg = SolveConfig { tolerance: seed.tolerance, ..cfg.clone() };
        let seed_result = self.solve_node(
            &subsets[seed_node], seed_node, &[], &c_zncc, &p, &seed_cfg, Some(&seed_warp_norm),
        )?;
        store_result(seed_node, &seed_result, &mut quality_ok, &mut c_zncc, &mut p, &mut displacements, &mut iterations, &mut norms);
        pb.inc(1);
        stored[seed_node] = true;
        propagated[seed_node] = true;
        queue.push((c_zncc[seed_node].to_bits(), seed_node));

        // --- Seed neighbours.
        self.solve_neighbours_from(
            seed_node,
            &subsets,
            cfg,
            pb,
            &mut stored,
            &mut quality_ok,
            &mut c_zncc,
            &mut queue,
            &mut p,
            &mut displacements,
            &mut iterations,
            &mut norms,
        )?;

        // --- Reliability-guided queue: highest-C_ZNCC first.
        while let Some((_, cur_idx)) = queue.pop() {
            if propagated[cur_idx] {
                continue;
            }
            propagated[cur_idx] = true;
            self.solve_neighbours_from(
                cur_idx,
                &subsets,
                cfg,
                pb,
                &mut stored,
                &mut quality_ok,
                &mut c_zncc,
                &mut queue,
                &mut p,
                &mut displacements,
                &mut iterations,
                &mut norms,
            )?;
        }

        // --- Corrections (outlier re-solve).
        self.corrections(
            &subsets,
            cfg,
            pb,
            &mut stored,
            &mut quality_ok,
            &mut c_zncc,
            &mut p,
            &mut displacements,
            &mut iterations,
            &mut norms,
        )?;

        if progress.is_none() {
            pb.finish_and_clear();
        }

        // --- Post-corrections quality gate ("subset decorrelation").
        //
        // Mirrors `mesh.py::_reliability_guided`'s `if any(self._C_ZNCC <
        // self._tolerance): self._unsolvable = True; self._status = 3` check
        // — but on `quality_ok` (convergence *and* correlation) rather than
        // correlation alone, since that's the whole point of `quality_ok`
        // meaning something now. `cfg.override_active` mirrors Python's
        // `self._override` bypass: a reference-update retry from
        // `sequence.rs` sets it so a still-imperfect mesh isn't rejected a
        // second time.
        if !cfg.override_active {
            let n_bad = quality_ok.iter().filter(|&&ok| !ok).count();
            if n_bad > 0 {
                return Err(Error::InvalidInput(format!(
                    "mesh unsolvable: {n_bad} subset(s) failed to reach quality_ok after corrections"
                )));
            }
        }

        // --- Element areas, centroids and strains.
        let centroids = compute_centroids(&self.nodes, &self.elements);
        let areas = element_area(&self.nodes, &self.elements);
        let warps = element_strains(
            &self.nodes,
            &self.elements,
            &displacements,
            self.mesh_order,
        )?;

        // --- Compatibility check.
        self.check_compatibility(&warps)?;

        let f_img_path = self.f_img.filepath.clone().unwrap_or_default();
        let g_img_path = self.g_img.filepath.clone().unwrap_or_default();

        self.solution = Some(Arc::new(MeshSolution {
            nodes: self.nodes.clone(),
            elements: self.elements.clone(),
            boundary: self.boundary.clone(),
            exclusions: self.exclusions.clone(),
            centroids,
            areas,
            warps,
            displacements,
            c_zncc,
            p,
            seed_node,
            mesh_order: self.mesh_order,
            subset_order: cfg.subset_order as u8,
            iterations,
            norms,
            f_img_path,
            g_img_path,
            solve_config: Some(cfg.clone()),
            seed: Some(seed.clone()),
        }));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal solver helpers
    // -----------------------------------------------------------------------

    fn solve_one_with_tolerance(
        &self,
        subset: &Subset,
        warp_0: &[f64],
        tolerance: f64,
        cfg: &SolveConfig,
    ) -> Result<SolveResult, Error> {
        match cfg.method {
            SolveMethod::Icgn => subset.solve_icgn_result(Some(warp_0), tolerance, cfg.max_norm, cfg.max_iterations),
            SolveMethod::Fagn => subset.solve_fagn_result(Some(warp_0), tolerance, cfg.max_norm, cfg.max_iterations),
        }
    }

    fn solve_one(
        &self,
        subset: &Subset,
        warp_0: &[f64],
        cfg: &SolveConfig,
    ) -> Result<SolveResult, Error> {
        self.solve_one_with_tolerance(subset, warp_0, cfg.tolerance, cfg)
    }

    /// Shared node-solve primitive used by RG propagation
    /// ([`Self::solve_neighbours_from`]) and both corrections lanes
    /// ([`Self::correlation_improvements`], [`Self::corrections`]).
    ///
    /// `trusted` — indices of `target_idx`'s neighbours already stored with
    /// `quality_ok == true`; used to build a preconditioning warp via
    /// [`Self::precondition_warp`]. Ignored when `forced_p0` is `Some`
    /// (the seed-node call site: the preconditioning source there is the
    /// user's `seed_warp`, not a neighbourhood).
    ///
    /// Tries the preconditioned warp, then a zero warp on failure, then
    /// returns whichever attempt scored higher on `c_zncc`. No separate
    /// "reliable" flag: `quality_ok` on the returned result already means
    /// "converged and correlated" (see `subset.rs::SolveResult::quality_ok`)
    /// — if this function reaches its last attempt, both tries already
    /// failed that check, so whichever is kept is honestly `quality_ok ==
    /// false` by construction.
    fn solve_node(
        &self,
        subset: &Subset,
        target_idx: usize,
        trusted: &[usize],
        c_zncc: &Array1<f64>,
        p: &Array2<f64>,
        cfg: &SolveConfig,
        forced_p0: Option<&[f64]>,
    ) -> Result<SolveResult, Error> {
        let p_len = 6 * cfg.subset_order;
        let p_a: Vec<f64> = match forced_p0 {
            Some(p0) => {
                let mut v = p0.to_vec();
                v.resize(p_len, 0.0);
                v
            }
            None => {
                let target_pos = [self.nodes[[target_idx, 0]], self.nodes[[target_idx, 1]]];
                precondition_warp(&self.nodes, target_pos, trusted, c_zncc, p, p_len)
            }
        };

        let result_a = self.solve_one(subset, &p_a, cfg)?;
        if result_a.quality_ok {
            return Ok(result_a);
        }
        if p_a.iter().all(|&v| v == 0.0) {
            // p_a was already the zero warp — a second zero-warp attempt
            // would be a wasted, identical solve.
            return Ok(result_a);
        }
        let zeros = vec![0.0f64; p_len];
        let result_b = self.solve_one(subset, &zeros, cfg)?;
        if result_b.quality_ok {
            return Ok(result_b);
        }
        Ok(if result_b.c_zncc > result_a.c_zncc { result_b } else { result_a })
    }


    /// Solve all unsolved neighbours of `cur_idx`, preconditioning each from
    /// its *own* trusted neighbourhood (not just `cur_idx`) via
    /// [`Self::solve_node`].
    fn solve_neighbours_from(
        &self,
        cur_idx: usize,
        subsets: &[Subset],
        cfg: &SolveConfig,
        pb: &indicatif::ProgressBar,
        stored: &mut Vec<bool>,
        quality_ok: &mut Array1<bool>,
        c_zncc: &mut Array1<f64>,
        queue: &mut BinaryHeap<(u64, usize)>,
        p: &mut Array2<f64>,
        displacements: &mut Array2<f64>,
        iterations: &mut Array1<u32>,
        norms: &mut Array1<f64>,
    ) -> Result<(), Error> {
        let neighbours = connectivity(&self.elements, self.mesh_order, cur_idx, false);
        for &nb_idx in &neighbours {
            if stored[nb_idx] {
                continue;
            }
            let candidates = connectivity(&self.elements, self.mesh_order, nb_idx, true);
            let trusted: Vec<usize> = candidates
                .into_iter()
                .filter(|&n| stored[n] && quality_ok[n])
                .collect();
            let result = self.solve_node(&subsets[nb_idx], nb_idx, &trusted, c_zncc, p, cfg, None)?;
            store_result(nb_idx, &result, quality_ok, c_zncc, p, displacements, iterations, norms);
            pb.inc(1);
            stored[nb_idx] = true;
            // Pushed regardless of `quality_ok` — for mesh-graph connectivity
            // completeness (a not-`quality_ok` node can still be the only
            // path to unsolved territory). It's never counted as *trusted*
            // for anyone else's preconditioning, since that set is filtered
            // on `quality_ok == true` already, so this doesn't leak an
            // untrustworthy result into anyone else's precondition.
            queue.push((c_zncc[nb_idx].to_bits(), nb_idx));
        }
        Ok(())
    }

    /// Improve anomalous poor-correlation nodes without discarding the RG result.
    ///
    /// Lane 1 of corrections (matches `mesh.py::_correlation_improvements`):
    /// re-solves each `_corr()`-flagged (IQR-outlier C_ZNCC) node via
    /// [`Self::solve_node`], but only overwrites the node's stored solve if
    /// the new C_ZNCC both beats the previous value and clears
    /// `quality_ok` — otherwise the node's existing (RG-cascade) result is
    /// left untouched and the node is excluded from preconditioning for the
    /// remainder of this pass. No urgency to force anything here: these
    /// nodes aren't necessarily spatially inconsistent, just relatively low
    /// versus their peers.
    fn correlation_improvements(
        &self,
        subsets: &[Subset],
        cfg: &SolveConfig,
        solved: &mut Vec<bool>,
        quality_ok: &mut Array1<bool>,
        c_zncc: &mut Array1<f64>,
        p: &mut Array2<f64>,
        displacements: &mut Array2<f64>,
        iterations: &mut Array1<u32>,
        norms: &mut Array1<f64>,
    ) -> Result<(), Error> {
        let mut order = corr(c_zncc.view());
        // Ascending by C_ZNCC (worst first), matching Python's np.argsort.
        order.sort_by(|&a, &b| c_zncc[a].partial_cmp(&c_zncc[b]).unwrap());

        let mut unimproved: HashSet<usize> = HashSet::new();
        for i in 0..order.len() {
            let j = order[i];
            let excluded: HashSet<usize> =
                order[i + 1..].iter().copied().chain(unimproved.iter().copied()).collect();
            let full_neighbours = connectivity(&self.elements, self.mesh_order, j, true);
            let trusted: Vec<usize> = full_neighbours
                .iter()
                .copied()
                .filter(|&nb| !excluded.contains(&nb) && quality_ok[nb])
                .collect();

            let result = self.solve_node(&subsets[j], j, &trusted, c_zncc, p, cfg, None)?;
            if result.c_zncc > c_zncc[j] && result.quality_ok {
                store_result(j, &result, quality_ok, c_zncc, p, displacements, iterations, norms);
                solved[j] = true;
            } else {
                unimproved.insert(j);
            }
        }
        Ok(())
    }

    /// Outlier correction: re-solve nodes flagged by `_flow`, `_R`.
    ///
    /// Lane 2 of corrections. Unlike Lane 1, this **always** overwrites the
    /// stored result with whatever [`Self::solve_node`] returns: the
    /// existing value is a known spatial discontinuity (that's what flagged
    /// it), so leaving it in place would defeat the point and risks failing
    /// the downstream compatibility check outright. `quality_ok` on the
    /// stored result is the honest signal for whether it should be trusted
    /// downstream — often `false` here, and that's now visible instead of
    /// silently indistinguishable from a clean solve.
    fn corrections(
        &self,
        subsets: &[Subset],
        cfg: &SolveConfig,
        _pb: &indicatif::ProgressBar,
        solved: &mut Vec<bool>,
        quality_ok: &mut Array1<bool>,
        c_zncc: &mut Array1<f64>,
        p: &mut Array2<f64>,
        displacements: &mut Array2<f64>,
        iterations: &mut Array1<u32>,
        norms: &mut Array1<f64>,
    ) -> Result<(), Error> {
        self.correlation_improvements(
            subsets, cfg, solved, quality_ok, c_zncc, p, displacements, iterations, norms,
        )?;

        let (_, flow_ids, flow_lq, flow_iqr) =
            flow_stats(&self.elements, self.mesh_order, displacements.view());
        let (r_vals, r_ids) =
            r_stats(&self.elements, self.mesh_order, &self.nodes, displacements.view());

        // Merge and sort ascending by R — mildest displacement-magnitude
        // anomalies corrected first (not worst-first: `r_ids` flags only
        // *upper*-tail IQR outliers in `r_calc`, so ascending-R means the
        // more-plausible flagged nodes get fixed first, giving the wilder
        // ones better-informed neighbours to precondition from later in
        // this same loop).
        let mut full_id: Vec<usize> = {
            let s: HashSet<usize> = flow_ids.iter().chain(r_ids.iter()).copied().collect();
            s.into_iter().collect()
        };
        full_id.sort_by(|&a, &b| r_vals[a].partial_cmp(&r_vals[b]).unwrap());

        // Exempt nodes whose flow is within normal bounds and not in R outliers.
        let full_id_set: HashSet<usize> = full_id.iter().copied().collect();
        let exempt: HashSet<usize> = full_id
            .iter()
            .copied()
            .filter(|&j| {
                if r_ids.contains(&j) {
                    return false;
                }
                let f = flow_calc_excluding(
                    j,
                    &full_id_set,
                    &self.elements,
                    self.mesh_order,
                    displacements.view(),
                    None,
                );
                f > flow_lq - 2.5 * flow_iqr
            })
            .collect();

        let active_ids: Vec<usize> =
            full_id.into_iter().filter(|id| !exempt.contains(id)).collect();

        for i in 0..active_ids.len() {
            let j = active_ids[i];
            let full_neighbours = connectivity(&self.elements, self.mesh_order, j, true);
            // Exclude current and later outliers from preconditioning.
            let later: HashSet<usize> = active_ids[i..].iter().copied().collect();
            let trusted: Vec<usize> = full_neighbours
                .iter()
                .copied()
                .filter(|&nb| !later.contains(&nb) && quality_ok[nb])
                .collect();

            let result = self.solve_node(&subsets[j], j, &trusted, c_zncc, p, cfg, None)?;
            store_result(j, &result, quality_ok, c_zncc, p, displacements, iterations, norms);
            solved[j] = true;
        }
        Ok(())
    }

    /// Compatibility check: reject mesh if any element has det(F) ≤ 0 (fold-over).
    fn check_compatibility(&self, warps: &Array2<f64>) -> Result<(), Error> {
        for e in 0..warps.nrows() {
            let j00 = 1.0 + warps[[e, 2]]; // 1 + du/dx
            let j01 =       warps[[e, 3]]; //     dv/dx
            let j10 =       warps[[e, 4]]; //     du/dy
            let j11 = 1.0 + warps[[e, 5]]; // 1 + dv/dy
            let det = j00 * j11 - j01 * j10;
            if det <= 0.0 {
                let cx = self.nodes.slice(s![.., 0]).mean().unwrap_or(0.0);
                let cy = self.nodes.slice(s![.., 1]).mean().unwrap_or(0.0);
                return Err(Error::InvalidInput(format!(
                    "mesh compatibility violated at element {e} near ({cx:.1},{cy:.1}): det(J)={det:.4}"
                )));
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Field accessors (used by the PyO3 wrapper)
    // -----------------------------------------------------------------------

    /// Node coordinates `(N, 2)`.
    pub fn nodes(&self) -> &Array2<f64> { &self.nodes }

    /// Element connectivity `(M, 3)` or `(M, 6)`.
    pub fn elements(&self) -> &Array2<usize> { &self.elements }

    /// Boundary node indices.
    pub fn boundary(&self) -> &[usize] { &self.boundary }

    /// Exclusion node index groups.
    pub fn exclusions(&self) -> &[Vec<usize>] { &self.exclusions }

    /// Mesh element order (1 or 2).
    pub fn mesh_order(&self) -> u8 { self.mesh_order }

    /// Reference image (shared ownership).
    pub fn f_img(&self) -> &Arc<Image> { &self.f_img }

    /// Target image (shared ownership).
    pub fn g_img(&self) -> &Arc<Image> { &self.g_img }
}

// ---------------------------------------------------------------------------
// Free functions — pure math, no Image dependency.
// Used in unit tests and externally (e.g. sequence.rs).
// ---------------------------------------------------------------------------

/// Signed element areas: `0.5 * det(M)` where M has ones in col 0,
/// x-coords in col 1, y-coords in col 2 (rows = corner nodes).
pub fn element_area(nodes: &Array2<f64>, elements: &Array2<usize>) -> Array1<f64> {
    let n_elems = elements.nrows();
    let mut areas = Array1::<f64>::zeros(n_elems);
    for e in 0..n_elems {
        let (i0, i1, i2) = (elements[[e, 0]], elements[[e, 1]], elements[[e, 2]]);
        let (x0, y0) = (nodes[[i0, 0]], nodes[[i0, 1]]);
        let (x1, y1) = (nodes[[i1, 0]], nodes[[i1, 1]]);
        let (x2, y2) = (nodes[[i2, 0]], nodes[[i2, 1]]);
        // det([[1,x0,y0],[1,x1,y1],[1,x2,y2]])
        areas[e] = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    }
    areas
}

/// Shape functions at the element centroid for the given mesh order.
///
/// Returns `(N, dN, d2N)`:
/// - Order-1: `N` (3,), `dN` (2,3), `d2N = None`
/// - Order-2: `N` (6,), `dN` (2,6), `d2N = Some` (3,6)
pub fn shape_function(
    mesh_order: u8,
) -> (Vec<f64>, Vec<Vec<f64>>, Option<Vec<Vec<f64>>>) {
    match mesh_order {
        1 => {
            let n = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
            let dn = vec![
                vec![1.0, 0.0, -1.0],
                vec![0.0, 1.0, -1.0],
            ];
            (n, dn, None)
        }
        2 => {
            let n = vec![-1.0 / 9.0, -1.0 / 9.0, -1.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0];
            let dn = vec![
                vec![1.0/3.0, 0.0, -1.0/3.0, 4.0/3.0, -4.0/3.0, 0.0],
                vec![0.0, 1.0/3.0, -1.0/3.0, 4.0/3.0, 0.0, -4.0/3.0],
            ];
            let d2n = vec![
                vec![4.0, 0.0, 4.0, 0.0, 0.0, -8.0],
                vec![0.0, 0.0, 4.0, 4.0, -4.0, -4.0],
                vec![0.0, 4.0, 4.0, 0.0, -8.0, 0.0],
            ];
            (n, dn, Some(d2n))
        }
        _ => panic!("mesh_order must be 1 or 2"),
    }
}

/// Element centroid local coordinates. Returns `A` with shape `(n_elems, 3, 4)`:
/// row 0 = all ones; rows 1-2 = `[centroid, x-coords, y-coords]`.
pub fn local_coordinates(nodes: &Array2<f64>, elements: &Array2<usize>) -> Vec<[[f64; 4]; 3]> {
    let n_elems = elements.nrows();
    let mut a = vec![[[1.0f64; 4]; 3]; n_elems];
    for e in 0..n_elems {
        let n_verts = elements.ncols().min(3); // only first 3 (corner) nodes
        // Row 0 stays all ones.
        // Rows 1-2, col 0 = centroid x, y.
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mean_n = elements.ncols() as f64;
        for k in 0..elements.ncols() {
            let ni = elements[[e, k]];
            cx += nodes[[ni, 0]];
            cy += nodes[[ni, 1]];
        }
        a[e][1][0] = cx / mean_n;
        a[e][2][0] = cy / mean_n;
        // Rows 1-2, cols 1-3 = transposed corner node coordinates.
        for k in 0..n_verts {
            let ni = elements[[e, k]];
            a[e][1][k + 1] = nodes[[ni, 0]]; // x-row
            a[e][2][k + 1] = nodes[[ni, 1]]; // y-row
        }
    }
    a
}

/// Element warp vectors `(n_elems, 12)`, translating `MeshBase._element_strains`.
///
/// Indices 0-1: mean displacement; 2-5: 1st-order strain; 6-11: 2nd-order strain
/// (always zero for mesh_order == 1).
///
/// The 2×2 Jacobian `J_x_T = dN @ x` is inverted analytically for order-1 and
/// numerically for order-2.
pub fn element_strains(
    nodes: &Array2<f64>,
    elements: &Array2<usize>,
    displacements: &Array2<f64>,
    mesh_order: u8,
) -> Result<Array2<f64>, Error> {
    let n_elems = elements.nrows();
    let (n_sf, dn, d2n) = shape_function(mesh_order);
    let n_sf_len = n_sf.len(); // 3 or 6
    let mut warps = Array2::<f64>::zeros((n_elems, 12));
    let a = local_coordinates(nodes, elements);

    for e in 0..n_elems {
        // Gather node positions and displacements.
        let mut x = vec![[0.0f64; 2]; n_sf_len]; // x[k] = [xi, yi]
        let mut u = vec![[0.0f64; 2]; n_sf_len]; // u[k] = [uk, vk]
        for k in 0..n_sf_len {
            let ni = elements[[e, k]];
            x[k] = [nodes[[ni, 0]], nodes[[ni, 1]]];
            u[k] = [displacements[[ni, 0]], displacements[[ni, 1]]];
        }

        // Displacement centroid: N @ u (2 components).
        warps[[e, 0]] = n_sf.iter().zip(&u).map(|(&nk, &[uk, _])| nk * uk).sum();
        warps[[e, 1]] = n_sf.iter().zip(&u).map(|(&nk, &[_, vk])| nk * vk).sum();

        // 1st-order strains: J_u_T = dN @ u; J_x_T = dN @ x; strain = inv(J_x_T) @ J_u_T.
        // Both are 2×2 when dN is (2, n_sf_len) and x/u are (n_sf_len, 2).
        let mut j_x = [[0.0f64; 2]; 2]; // J_x_T[row][col]
        let mut j_u = [[0.0f64; 2]; 2];
        for row in 0..2 {
            for col in 0..2 {
                j_x[row][col] =
                    dn[row].iter().zip(&x).map(|(&d, xi)| d * xi[col]).sum();
                j_u[row][col] =
                    dn[row].iter().zip(&u).map(|(&d, ui)| d * ui[col]).sum();
            }
        }
        // 2×2 analytic inverse.
        let det = j_x[0][0] * j_x[1][1] - j_x[0][1] * j_x[1][0];
        if det.abs() < f64::EPSILON * 1e6 {
            return Err(Error::MeshGeneration(format!(
                "degenerate Jacobian at element {e}: det = {det:.2e}"
            )));
        }
        let inv = [
            [j_x[1][1] / det, -j_x[0][1] / det],
            [-j_x[1][0] / det, j_x[0][0] / det],
        ];
        // strain = inv(J_x_T) @ J_u_T → (2×2) @ (2×2) = (2×2), flatten row-major.
        // warps[e, 2..6] = [strain[0][0], strain[0][1], strain[1][0], strain[1][1]]
        // i.e. [du/dx, dv/dx, du/dy, dv/dy]
        for row in 0..2 {
            for col in 0..2 {
                warps[[e, 2 + row * 2 + col]] =
                    inv[row][0] * j_u[0][col] + inv[row][1] * j_u[1][col];
            }
        }

        // 2nd-order strains (mesh_order == 2 only).
        if let Some(ref d2n_mat) = d2n {
            // K_u = d2N @ u → (3, 2)
            let mut k_u = [[0.0f64; 2]; 3];
            for row in 0..3 {
                for col in 0..2 {
                    k_u[row][col] =
                        d2n_mat[row].iter().zip(&u).map(|(&d, ui)| d * ui[col]).sum();
                }
            }
            // dz = 2×2 matrix from corner node coords.
            let (x0, y0) = (x[0][0], x[0][1]);
            let (x1, y1) = (x[1][0], x[1][1]);
            let (x2, y2) = (x[2][0], x[2][1]);
            // det(A[e, :, [1,2,3]]) — the 3×3 submatrix of local_coordinates cols 1-3.
            // a[e][row][col+1] for row=0,1,2 and col=0,1,2.
            let ae = a[e];
            let det_a = ae[0][1] * (ae[1][2] * ae[2][3] - ae[1][3] * ae[2][2])
                - ae[0][2] * (ae[1][1] * ae[2][3] - ae[1][3] * ae[2][1])
                + ae[0][3] * (ae[1][1] * ae[2][2] - ae[1][2] * ae[2][1]);
            let dz = [
                [(y1 - y2) / det_a, (y2 - y0) / det_a],
                [(x2 - x1) / det_a, (x0 - x2) / det_a],
            ];
            // K_x_inv (3×3).
            let kx = [
                [
                    dz[0][0] * dz[0][0],
                    2.0 * dz[0][0] * dz[0][1],
                    dz[0][1] * dz[0][1],
                ],
                [
                    dz[0][0] * dz[1][0],
                    dz[0][0] * dz[1][1] + dz[0][1] * dz[1][0],
                    dz[0][1] * dz[1][1],
                ],
                [
                    dz[1][0] * dz[1][0],
                    2.0 * dz[1][0] * dz[1][1],
                    dz[1][1] * dz[1][1],
                ],
            ];
            // K_x_inv @ K_u → (3,2), flatten.
            for row in 0..3 {
                for col in 0..2 {
                    warps[[e, 6 + row * 2 + col]] =
                        kx[row][0] * k_u[0][col]
                            + kx[row][1] * k_u[1][col]
                            + kx[row][2] * k_u[2][col];
                }
            }
        }
    }
    Ok(warps)
}

/// Compute element centroids (mean of corner node positions).
pub fn compute_centroids(nodes: &Array2<f64>, elements: &Array2<usize>) -> Array2<f64> {
    let n_elem = elements.nrows();
    let mut centroids = Array2::<f64>::zeros((n_elem, 2));
    for i in 0..n_elem {
        let n0 = elements[[i, 0]];
        let n1 = elements[[i, 1]];
        let n2 = elements[[i, 2]];
        centroids[[i, 0]] = (nodes[[n0, 0]] + nodes[[n1, 0]] + nodes[[n2, 0]]) / 3.0;
        centroids[[i, 1]] = (nodes[[n0, 1]] + nodes[[n1, 1]] + nodes[[n2, 1]]) / 3.0;
    }
    centroids
}

/// Find the node index closest to `seed_coord`.
pub fn find_seed_node(nodes: &Array2<f64>, seed_coord: [f64; 2]) -> usize {
    (0..nodes.nrows())
        .min_by(|&a, &b| {
            let da = dist2(nodes, a, seed_coord);
            let db = dist2(nodes, b, seed_coord);
            da.partial_cmp(&db).unwrap()
        })
        .unwrap_or(0)
}

#[inline]
fn dist2(nodes: &Array2<f64>, idx: usize, coord: [f64; 2]) -> f64 {
    let dx = nodes[[idx, 0]] - coord[0];
    let dy = nodes[[idx, 1]] - coord[1];
    dx * dx + dy * dy
}

/// Return node indices connected to `idx`.
///
/// - Order-1 or `full=true`: all unique nodes sharing an element with `idx`,
///   excluding `idx` itself.
/// - Order-2, `full=false`: midpoint neighbours if `idx` is a corner; corner
///   neighbours if `idx` is a midpoint (based on column position in element).
pub fn connectivity(
    elements: &Array2<usize>,
    mesh_order: u8,
    idx: usize,
    full: bool,
) -> Vec<usize> {
    if mesh_order == 1 || full {
        let mut result: HashSet<usize> = HashSet::new();
        for e in 0..elements.nrows() {
            let n_cols = elements.ncols();
            let mut found = false;
            for k in 0..n_cols {
                if elements[[e, k]] == idx {
                    found = true;
                    break;
                }
            }
            if found {
                for k in 0..n_cols {
                    let nb = elements[[e, k]];
                    if nb != idx {
                        result.insert(nb);
                    }
                }
            }
        }
        let mut v: Vec<usize> = result.into_iter().collect();
        v.sort_unstable();
        v
    } else {
        // Order-2, full=false: edge-adjacent connectivity only.
        let mut result: HashSet<usize> = HashSet::new();
        for e in 0..elements.nrows() {
            let ncols = elements.ncols();
            for col in 0..ncols {
                if elements[[e, col]] != idx {
                    continue;
                }
                // Append based on which column position idx occupies.
                let row = &elements.row(e);
                let adds: &[usize] = match col {
                    0 => &[row[3], row[5]],     // corner 0 → mid01, mid20
                    1 => &[row[3], row[4]],     // corner 1 → mid01, mid12
                    2 => &[row[4], row[5]],     // corner 2 → mid12, mid20
                    3 => &[row[0], row[1]],     // mid01 → corner 0, corner 1
                    4 => &[row[1], row[2]],     // mid12 → corner 1, corner 2
                    5 => &[row[0], row[2]],     // mid20 → corner 0, corner 2
                    _ => &[],
                };
                for &nb in adds {
                    if nb != idx {
                        result.insert(nb);
                    }
                }
            }
        }
        let mut v: Vec<usize> = result.into_iter().collect();
        v.sort_unstable();
        v
    }
}

/// IQR-based outlier detection on C_ZNCC scores.
/// Returns indices where `C_ZNCC < LQ - 2.5 * IQR`.
pub fn corr(c_zncc: ArrayView1<f64>) -> Vec<usize> {
    let vals: Vec<f64> = c_zncc.iter().copied().collect();
    corr_1d(&vals)
}

pub fn corr_1d(vals: &[f64]) -> Vec<usize> {
    let mut sorted: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lq = percentile(&sorted, 25.0);
    let uq = percentile(&sorted, 75.0);
    let iqr = uq - lq;
    let fence = lq - 2.5 * iqr;
    vals.iter().enumerate().filter(|(_, v)| **v < fence).map(|(i, _)| i).collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let pos = p / 100.0 * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Flow for node `i`: dot(unit(disp_i), unit(mean_neighbour_disp)).
/// Returns `-1` if there are no valid neighbours.
pub fn flow_calc(
    idx: usize,
    elements: &Array2<usize>,
    mesh_order: u8,
    displacements: ArrayView2<f64>,
    exclude: &HashSet<usize>,
    displacement_override: Option<[f64; 2]>,
) -> f64 {
    let disp = displacement_override
        .unwrap_or([displacements[[idx, 0]], displacements[[idx, 1]]]);
    if !disp[0].is_finite() || !disp[1].is_finite() {
        return -1.0;
    }
    let neighbours = connectivity(elements, mesh_order, idx, true);
    let valid: Vec<usize> = neighbours
        .into_iter()
        .filter(|nb| !exclude.contains(nb))
        .filter(|nb| displacements[[*nb, 0]].is_finite() && displacements[[*nb, 1]].is_finite())
        .collect();
    if valid.is_empty() {
        return -1.0;
    }
    let n = valid.len() as f64;
    let vx: f64 = valid.iter().map(|&nb| displacements[[nb, 0]]).sum::<f64>() / n;
    let vy: f64 = valid.iter().map(|&nb| displacements[[nb, 1]]).sum::<f64>() / n;
    let v_mag = (vx * vx + vy * vy).sqrt();
    let d_mag = (disp[0] * disp[0] + disp[1] * disp[1]).sqrt();
    if v_mag < f64::EPSILON || d_mag < f64::EPSILON {
        return -1.0;
    }
    (disp[0] * vx + disp[1] * vy) / (d_mag * v_mag)
}

fn flow_calc_excluding(
    idx: usize,
    exclude: &HashSet<usize>,
    elements: &Array2<usize>,
    mesh_order: u8,
    displacements: ArrayView2<f64>,
    displacement_override: Option<[f64; 2]>,
) -> f64 {
    flow_calc(idx, elements, mesh_order, displacements, exclude, displacement_override)
}

/// Absolute displacement magnitude.
pub fn r_calc(displacement: [f64; 2]) -> f64 {
    (displacement[0] * displacement[0] + displacement[1] * displacement[1]).sqrt()
}

/// Flow outlier IDs, LQ, IQR for the full mesh (no exclusions).
fn flow_stats(
    elements: &Array2<usize>,
    mesh_order: u8,
    displacements: ArrayView2<f64>,
) -> (Vec<f64>, Vec<usize>, f64, f64) {
    let n = displacements.nrows();
    let empty = HashSet::new();
    let flow: Vec<f64> = (0..n)
        .map(|i| flow_calc(i, elements, mesh_order, displacements, &empty, None))
        .collect();
    let mut sorted: Vec<f64> = flow.iter().copied().filter(|f| f.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lq = percentile(&sorted, 25.0);
    let uq = percentile(&sorted, 75.0);
    let iqr = uq - lq;
    let fence = lq - 1.5 * iqr;
    let ids: Vec<usize> = flow.iter().enumerate().filter(|(_, f)| **f < fence).map(|(i, _)| i).collect();
    (flow, ids, lq, iqr)
}

/// R (displacement magnitude) outlier IDs using element-local IQR.
fn r_stats(
    elements: &Array2<usize>,
    mesh_order: u8,
    _nodes: &Array2<f64>,
    displacements: ArrayView2<f64>,
) -> (Vec<f64>, HashSet<usize>) {
    let n = displacements.nrows();
    let r: Vec<f64> = (0..n)
        .map(|i| r_calc([displacements[[i, 0]], displacements[[i, 1]]]))
        .collect();
    let mut r_ids: HashSet<usize> = HashSet::new();
    for e in 0..elements.nrows() {
        // Collect all neighbourhood nodes from corners of this element.
        let mut local: HashSet<usize> = HashSet::new();
        for k in 0..3 {
            let corner = elements[[e, k]];
            for nb in connectivity(elements, mesh_order, corner, true) {
                local.insert(nb);
            }
        }
        let local_r: Vec<f64> = local.iter().map(|&nb| r[nb]).collect();
        if local_r.is_empty() {
            continue;
        }
        let mut sr = local_r.clone();
        sr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lq = percentile(&sr, 25.0);
        let uq = percentile(&sr, 75.0);
        let iqr = uq - lq;
        let fence = uq + 4.0 * iqr;
        for &nb in &local {
            if r[nb] > fence {
                r_ids.insert(nb);
            }
        }
    }
    (r, r_ids)
}

// ---------------------------------------------------------------------------
// Internal helper — warp propagation
// ---------------------------------------------------------------------------

/// Project warp `p` by a Taylor expansion over displacement `(dx, dy)`
/// from the node `p` was solved at to the target position (mirrors
/// `Mesh._neighbours` attempt 2).
fn project_warp(p: &[f64], dx: f64, dy: f64) -> Vec<f64> {
    let mut p_proj = p.to_vec();
    if p.len() >= 6 {
        p_proj[0] = p[0] + p[2] * dx + p[4] * dy;
        p_proj[1] = p[1] + p[3] * dx + p[5] * dy;
    }
    if p.len() >= 12 {
        p_proj[0] += 0.5 * p[6] * dx * dx + p[8] * dx * dy + 0.5 * p[10] * dy * dy;
        p_proj[1] += 0.5 * p[7] * dx * dx + p[9] * dx * dy + 0.5 * p[11] * dy * dy;
        p_proj[2] = p[2] + p[6] * dx + p[8] * dy;
        p_proj[3] = p[3] + p[7] * dx + p[9] * dy;
        p_proj[4] = p[4] + p[8] * dx + p[10] * dy;
        p_proj[5] = p[5] + p[9] * dx + p[11] * dy;
    }
    p_proj
}

/// Build a preconditioning warp for `target_pos` from `trusted` neighbour
/// indices (already-solved nodes with `quality_ok == true`).
///
/// - `>= 3` trusted — local least-squares affine field fit
///   ([`affine_field_fit`]): derives the target's `(u, v, ux, vx, uy, vy)`
///   from how displacement actually varies across the trusted neighbourhood,
///   rather than trusting any one neighbour's own individually-fit gradient
///   terms.
/// - `1` or `2` trusted, or the fit above is singular (e.g. collinear
///   neighbours) — Taylor-projected extrapolation from whichever trusted
///   neighbour has the highest `c_zncc` ([`project_warp`]).
/// - `0` trusted — zero warp.
fn precondition_warp(
    nodes: &Array2<f64>,
    target_pos: [f64; 2],
    trusted: &[usize],
    c_zncc: &Array1<f64>,
    p: &Array2<f64>,
    p_len: usize,
) -> Vec<f64> {
    if trusted.len() >= 3 {
        if let Some(fit) = affine_field_fit(nodes, target_pos, trusted, p, p_len) {
            return fit;
        }
    }
    if let Some(&best) = trusted
        .iter()
        .max_by(|&&a, &&b| c_zncc[a].partial_cmp(&c_zncc[b]).unwrap())
    {
        let dx = target_pos[0] - nodes[[best, 0]];
        let dy = target_pos[1] - nodes[[best, 1]];
        return project_warp(&p.row(best).to_vec(), dx, dy);
    }
    vec![0.0; p_len]
}

/// Local least-squares affine field fit through `trusted` neighbours,
/// evaluated at `target_pos`. Regresses each neighbour's own `(u, v)`
/// against its position offset from the target — the fitted intercepts give
/// the target's `(u, v)`, and the fitted slopes give `(ux, uy, vx, vy)`
/// directly, derived from how displacement actually varies across the
/// trusted neighbourhood rather than any one neighbour's own gradient
/// estimate.
///
/// For `p_len >= 12` (`subset_order == 2`), also attempts the full
/// quadratic fit (6 basis terms per component: `1, dx, dy, dx², dx·dy,
/// dy²`) when `trusted.len() >= 6`; otherwise the 2nd-order terms are left
/// at zero.
///
/// Returns `None` if the normal-equations matrix is singular (e.g. all
/// trusted neighbours collinear) — the caller falls back to single-neighbour
/// Taylor projection in that case.
fn affine_field_fit(
    nodes: &Array2<f64>,
    target_pos: [f64; 2],
    trusted: &[usize],
    p: &Array2<f64>,
    p_len: usize,
) -> Option<Vec<f64>> {
    let quadratic = p_len >= 12 && trusted.len() >= 6;
    let rows: Vec<Vec<f64>> = trusted
        .iter()
        .map(|&i| {
            let dx = nodes[[i, 0]] - target_pos[0];
            let dy = nodes[[i, 1]] - target_pos[1];
            if quadratic {
                vec![1.0, dx, dy, dx * dx, dx * dy, dy * dy]
            } else {
                vec![1.0, dx, dy]
            }
        })
        .collect();
    let u: Vec<f64> = trusted.iter().map(|&i| p[[i, 0]]).collect();
    let v: Vec<f64> = trusted.iter().map(|&i| p[[i, 1]]).collect();

    let cu = ols_solve(&rows, &u)?;
    let cv = ols_solve(&rows, &v)?;

    let mut warp = vec![0.0; p_len];
    warp[0] = cu[0]; // u
    warp[1] = cv[0]; // v
    warp[2] = cu[1]; // ux
    warp[3] = cv[1]; // vx
    warp[4] = cu[2]; // uy
    warp[5] = cv[2]; // vy
    if quadratic {
        // u(dx,dy) = u + ux·dx + uy·dy + 0.5·uxx·dx² + uxy·dx·dy + 0.5·uyy·dy²
        // (see `apply_warp`, subset.rs) vs. the fit's u(dx,dy) = cu[0] +
        // cu[1]·dx + cu[2]·dy + cu[3]·dx² + cu[4]·dx·dy + cu[5]·dy² — so
        // uxx = 2·cu[3], uxy = cu[4], uyy = 2·cu[5] (and likewise for v).
        warp[6] = 2.0 * cu[3];  // uxx
        warp[7] = 2.0 * cv[3];  // vxx
        warp[8] = cu[4];        // uxy
        warp[9] = cv[4];        // vxy
        warp[10] = 2.0 * cu[5]; // uyy
        warp[11] = 2.0 * cv[5]; // vyy
    }
    Some(warp)
}

/// Ordinary least squares via the normal equations `(AᵀA) x = Aᵀy`.
/// `rows` are the design-matrix rows (all the same length). Returns `None`
/// if `AᵀA` is singular.
fn ols_solve(rows: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let n = rows.len();
    let m = rows.first()?.len();
    let a = DMatrix::from_fn(n, m, |r, c| rows[r][c]);
    let yv = DVector::from_fn(n, |r, _| y[r]);
    let ata = a.transpose() * &a;
    let aty = a.transpose() * yv;
    let inv = ata.try_inverse()?;
    Some((inv * aty).iter().copied().collect())
}

/// Store a `SolveResult` into the per-node arrays.
fn store_result(
    idx: usize,
    result: &SolveResult,
    quality_ok: &mut Array1<bool>,
    c_zncc: &mut Array1<f64>,
    p: &mut Array2<f64>,
    displacements: &mut Array2<f64>,
    iterations: &mut Array1<u32>,
    norms: &mut Array1<f64>,
) {
    quality_ok[idx] = result.quality_ok;
    c_zncc[idx] = result.c_zncc.max(0.0);
    for (k, &v) in result.p.iter().enumerate() {
        p[[idx, k]] = v;
    }
    displacements[[idx, 0]] = result.p[0];
    displacements[[idx, 1]] = result.p[1];
    iterations[idx] = result.iterations as u32;
    norms[idx] = result.history.last().map(|h| h.1).unwrap_or(0.0);
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn unit_triangle_mesh() -> (Array2<f64>, Array2<usize>) {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        (nodes, elements)
    }

    // -----------------------------------------------------------------------
    // element_area
    // -----------------------------------------------------------------------

    #[test]
    fn element_area_unit_triangle() {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let elements = array![[0usize, 1, 2]];
        let areas = element_area(&nodes, &elements);
        assert!((areas[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn element_area_two_triangles() {
        let (nodes, elements) = unit_triangle_mesh();
        let areas = element_area(&nodes, &elements);
        assert_eq!(areas.len(), 2);
        for a in areas.iter() {
            assert!((a.abs() - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn element_area_scaled_triangle() {
        let nodes = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let elements = array![[0usize, 1, 2]];
        let areas = element_area(&nodes, &elements);
        assert!((areas[0] - 2.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // shape_function
    // -----------------------------------------------------------------------

    #[test]
    fn shape_function_order1_n_sums_to_one() {
        let (n, _, _) = shape_function(1);
        assert!((n.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn shape_function_order1_d2n_none() {
        let (_, _, d2n) = shape_function(1);
        assert!(d2n.is_none());
    }

    #[test]
    fn shape_function_order2_n_values() {
        let (n, _, _) = shape_function(2);
        let expected = [-1.0/9.0, -1.0/9.0, -1.0/9.0, 4.0/9.0, 4.0/9.0, 4.0/9.0];
        for (a, b) in n.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn shape_function_order2_has_d2n() {
        let (_, _, d2n) = shape_function(2);
        assert!(d2n.is_some());
        assert_eq!(d2n.unwrap().len(), 3);
    }

    // -----------------------------------------------------------------------
    // element_strains
    // -----------------------------------------------------------------------

    #[test]
    fn element_strains_pure_translation_order1() {
        let (nodes, elements) = unit_triangle_mesh();
        let disps = array![[3.5, -2.1], [3.5, -2.1], [3.5, -2.1], [3.5, -2.1]];
        let warps = element_strains(&nodes, &elements, &disps, 1).unwrap();
        // Displacement centroid = (3.5, -2.1), strains = 0.
        for e in 0..2 {
            assert!((warps[[e, 0]] - 3.5).abs() < 1e-10);
            assert!((warps[[e, 1]] - (-2.1)).abs() < 1e-10);
            for k in 2..6 {
                assert!(warps[[e, k]].abs() < 1e-10, "warps[{e},{k}] = {:.2e}", warps[[e,k]]);
            }
        }
    }

    #[test]
    fn element_strains_x_shear_order1() {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let elements = array![[0usize, 1, 2]];
        let disps = array![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]];
        let warps = element_strains(&nodes, &elements, &disps, 1).unwrap();
        // Centroid u = 1/3.
        assert!((warps[[0, 0]] - 1.0 / 3.0).abs() < 1e-12);
        assert!(warps[[0, 1]].abs() < 1e-12);
        // du/dx = 1, rest = 0.
        assert!((warps[[0, 2]] - 1.0).abs() < 1e-10);
        assert!(warps[[0, 3]].abs() < 1e-10);
        assert!(warps[[0, 4]].abs() < 1e-10);
        assert!(warps[[0, 5]].abs() < 1e-10);
    }

    #[test]
    fn element_strains_shape_order1() {
        let (nodes, elements) = unit_triangle_mesh();
        let disps = Array2::<f64>::zeros((4, 2));
        let warps = element_strains(&nodes, &elements, &disps, 1).unwrap();
        assert_eq!(warps.shape(), &[2, 12]);
    }

    #[test]
    fn element_strains_pure_translation_order2() {
        // 9-node mesh: 4 corners + 5 midpoints
        let nodes = array![
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1.0, 0.5], [0.5, 1.0]
        ];
        let elements = array![
            [0usize, 1, 2, 4, 5, 6],
            [1, 3, 2, 7, 8, 5]
        ];
        let disps = array![
            [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0],
            [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]
        ];
        let warps = element_strains(&nodes, &elements, &disps, 2).unwrap();
        // Pure translation: centroid disp = (1,2), all strains = 0.
        for e in 0..2 {
            assert!((warps[[e, 0]] - 1.0).abs() < 1e-10);
            assert!((warps[[e, 1]] - 2.0).abs() < 1e-10);
            for k in 2..12 {
                assert!(
                    warps[[e, k]].abs() < 1e-8,
                    "warps[{e},{k}] = {:.2e}",
                    warps[[e, k]]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // connectivity
    // -----------------------------------------------------------------------

    #[test]
    fn connectivity_order1_corner() {
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let result = connectivity(&elements, 1, 0, false);
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn connectivity_order1_shared_node() {
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let result = connectivity(&elements, 1, 1, false);
        assert_eq!(result, vec![0, 2, 3]);
    }

    #[test]
    fn connectivity_order2_corner_gets_midpoints() {
        // elem 0 = [0,1,2,4,5,6] — corner 0 at position 0 → mids at positions 3 and 5: [4,6]
        let elements = array![[0usize, 1, 2, 4, 5, 6], [1, 3, 2, 7, 8, 5]];
        let result = connectivity(&elements, 2, 0, false);
        assert_eq!(result, vec![4, 6]);
    }

    #[test]
    fn connectivity_order2_midpoint_gets_corners() {
        // elem 0 = [0,1,2,4,5,6] — midpoint 4 at position 3 → corners [:2] = [0,1]
        let elements = array![[0usize, 1, 2, 4, 5, 6], [1, 3, 2, 7, 8, 5]];
        let result = connectivity(&elements, 2, 4, false);
        assert_eq!(result, vec![0, 1]);
    }

    // -----------------------------------------------------------------------
    // find_seed_node
    // -----------------------------------------------------------------------

    #[test]
    fn find_seed_node_exact() {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert_eq!(find_seed_node(&nodes, [1.0, 0.0]), 1);
    }

    #[test]
    fn find_seed_node_nearest() {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        // (0.4, 0.4): d0=0.566, d1=0.721, d2=0.721, d3=0.849 → node 0
        assert_eq!(find_seed_node(&nodes, [0.4, 0.4]), 0);
    }

    // -----------------------------------------------------------------------
    // corr_1d (IQR outlier detection)
    // -----------------------------------------------------------------------

    #[test]
    fn corr_no_outliers() {
        let vals = vec![0.9f64; 10];
        assert!(corr_1d(&vals).is_empty());
    }

    #[test]
    fn corr_detects_single_outlier() {
        let mut vals = vec![0.9f64; 19];
        vals.push(-10.0);
        let result = corr_1d(&vals);
        assert!(result.contains(&19));
    }

    // -----------------------------------------------------------------------
    // flow_calc
    // -----------------------------------------------------------------------

    #[test]
    fn flow_calc_aligned_returns_one() {
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let disps = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
        let result = flow_calc(0, &elements, 1, disps.view(), &HashSet::new(), None);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn flow_calc_anti_aligned_returns_minus_one() {
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let disps = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
        let result =
            flow_calc(0, &elements, 1, disps.view(), &HashSet::new(), Some([-1.0, 0.0]));
        assert!((result - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn flow_calc_no_neighbours_returns_minus_one() {
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let disps = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
        // Exclude all of node 0's neighbours.
        let exclude: HashSet<usize> = [1, 2].iter().copied().collect();
        let result = flow_calc(0, &elements, 1, disps.view(), &exclude, None);
        assert_eq!(result, -1.0);
    }

    // -----------------------------------------------------------------------
    // r_calc
    // -----------------------------------------------------------------------

    #[test]
    fn r_calc_unit() {
        assert!((r_calc([1.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn r_calc_pythagorean() {
        assert!((r_calc([3.0, 4.0]) - 5.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // project_warp
    // -----------------------------------------------------------------------

    #[test]
    fn project_warp_second_order_terms() {
        // u = x*y: du/dy=1, d2u/dxdy=1, all others zero.
        // Shift by dx=2, dy=3. Expected du/dy_new = 1 + 1*2 + 0*3 = 3.
        let mut p = vec![0.0f64; 12];
        p[4] = 1.0; // du/dy
        p[8] = 1.0; // d2u/dxdy
        let p_proj = project_warp(&p, 2.0, 3.0);
        assert!((p_proj[4] - 3.0).abs() < 1e-12, "du/dy projected: {}", p_proj[4]);
    }

    // -----------------------------------------------------------------------
    // affine_field_fit / ols_solve
    // -----------------------------------------------------------------------

    #[test]
    fn affine_field_fit_recovers_exact_linear_field() {
        // Synthetic linear displacement field: u = 0.1*x - 0.05*y + 2.0,
        // v = -0.02*x + 0.08*y - 1.0. A noiseless affine fit should recover
        // it exactly at any target position, including one not among the
        // sampled nodes.
        let u = |x: f64, y: f64| 0.1 * x - 0.05 * y + 2.0;
        let v = |x: f64, y: f64| -0.02 * x + 0.08 * y - 1.0;
        let nodes = array![
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 2.0],
        ];
        let mut p = Array2::<f64>::zeros((5, 6));
        for i in 0..5 {
            let (x, y) = (nodes[[i, 0]], nodes[[i, 1]]);
            p[[i, 0]] = u(x, y);
            p[[i, 1]] = v(x, y);
        }
        let target = [3.0, 7.0];
        let trusted = [0usize, 1, 2, 3, 4];
        let fit = affine_field_fit(&nodes, target, &trusted, &p, 6).expect("fit should succeed");
        assert!((fit[0] - u(target[0], target[1])).abs() < 1e-9, "u: {}", fit[0]);
        assert!((fit[1] - v(target[0], target[1])).abs() < 1e-9, "v: {}", fit[1]);
        assert!((fit[2] - 0.1).abs() < 1e-9, "ux: {}", fit[2]);
        assert!((fit[3] - (-0.02)).abs() < 1e-9, "vx: {}", fit[3]);
        assert!((fit[4] - (-0.05)).abs() < 1e-9, "uy: {}", fit[4]);
        assert!((fit[5] - 0.08).abs() < 1e-9, "vy: {}", fit[5]);
    }

    #[test]
    fn affine_field_fit_none_when_collinear() {
        // All trusted neighbours on a single line: the `dy` column (and
        // hence the normal-equations matrix) is degenerate.
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let p = Array2::<f64>::zeros((3, 6));
        let trusted = [0usize, 1, 2];
        assert!(affine_field_fit(&nodes, [1.0, 5.0], &trusted, &p, 6).is_none());
    }

    #[test]
    fn precondition_warp_thresholds() {
        // 0 trusted -> zero warp.
        let nodes = array![[0.0, 0.0], [1.0, 0.0]];
        let c_zncc = Array1::from_vec(vec![0.9, 0.95]);
        let p = array![[1.0, 2.0, 0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]];

        let zero = precondition_warp(&nodes, [5.0, 5.0], &[], &c_zncc, &p, 6);
        assert!(zero.iter().all(|&v| v == 0.0));

        // 1 trusted -> Taylor projection from that neighbour (no gradient
        // terms in `p` here, so projection is a flat copy of u,v).
        let one = precondition_warp(&nodes, [5.0, 5.0], &[1], &c_zncc, &p, 6);
        assert!((one[0] - 3.0).abs() < 1e-12);
        assert!((one[1] - 4.0).abs() < 1e-12);
    }

}
