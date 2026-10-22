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
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    geometry::{meshing, triangulation},
    image::Image,
    screen::{self, NoiseSource, ScreenInput, StillPair},
    seam::{self, SeamConfig, SeamPending, SeamRecord},
    subset::{Subset, SolveResult},
    timing::{resolve_pair_time, TimeInput, TimeSource},
    masks::{LocalMask, MaskShape},
    progress::{self, SolveProgress, TerminalProgress},
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
    /// Subset template shape (uniform across nodes). Backs
    /// `Mesh.inspect(subset_idx=...)`'s template crop.
    #[serde(default)]
    pub template_shape: Option<MaskShape>,
    /// Per-node subset template size (radius / half-side), `(N,)`. Every
    /// entry is currently equal (one shared template per solve).
    #[serde(default)]
    pub template_sizes: Option<Array1<u32>>,
    /// `(f_time, g_time)` of the image pair, seconds — see [`crate::timing`].
    #[serde(default)]
    pub time: Option<(f64, f64)>,
    /// Adaptive masking record — see [`Masking::Adaptive`]. `None` for a
    /// uniform solve.
    #[serde(default)]
    pub seams: Option<MeshSeams>,
}

/// What adaptive masking did to one mesh.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshSeams {
    /// Node positions of the plain solve (before any move).
    pub plain_nodes: Array2<f64>,
    /// Warps of the plain order-1 solve, `(N, 6)`.
    pub plain_p: Array2<f64>,
    /// Flagged by the still-pair tolerance.
    pub flagged: Vec<bool>,
    /// Sent on to the seam tests (after keep ⅓ / add ⅔).
    pub sent: Vec<bool>,
    /// Rounds of the neighbour rule.
    pub rounds: usize,
    pub noise_source: NoiseSource,
    /// Per node: typical noise of a neighbour mismatch (px).
    pub noise_scale: Array1<f64>,
    /// Per node: largest frame 0–1 neighbour mismatch (px); `None` without a
    /// still pair.
    pub still_mismatch: Option<Array1<f64>>,
    /// Speckle correlation length of the reference image (px).
    pub h: f64,
    /// Per node: the seam record of a node sent on.
    pub records: Vec<Option<SeamRecord>>,
}

#[cfg(test)]
impl MeshSeams {
    /// A record holding only per-node seam records (other fields empty).
    pub(crate) fn for_test(records: Vec<Option<SeamRecord>>) -> Self {
        let n = records.len();
        MeshSeams {
            plain_nodes: Array2::zeros((n, 2)),
            plain_p: Array2::zeros((n, 6)),
            flagged: vec![false; n],
            sent: records.iter().map(Option::is_some).collect(),
            rounds: 0,
            noise_source: NoiseSource::StillPair,
            noise_scale: Array1::zeros(n),
            still_mismatch: None,
            h: 3.0,
            records,
        }
    }
}

impl MeshSolution {
    /// `g_time - f_time`, if the pair's times are known.
    pub fn dt(&self) -> Option<f64> {
        self.time.map(|(f, g)| g - f)
    }
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
    /// Intra-mesh RG frontier traversal strategy — see [`Preconditioning`].
    #[serde(default)]
    pub preconditioning: Preconditioning,
    /// Tunables for `preconditioning == LayerRg` — see [`LayerRgConfig`].
    /// Ignored otherwise.
    #[serde(default)]
    pub layer_rg: LayerRgConfig,
    /// Uniform (plain) or adaptive masking — see [`Masking`].
    #[serde(default)]
    pub masking: Masking,
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
            preconditioning: Preconditioning::Rg,
            layer_rg: LayerRgConfig::default(),
            masking: Masking::Uniform,
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

impl SeedConfig {
    /// Default seed-node C_ZNCC tolerance.
    pub const DEFAULT_TOLERANCE: f64 = 0.9;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveMethod {
    Icgn,
    Fagn,
}

/// Intra-mesh RG frontier traversal strategy — the `preconditioning` axis
/// of `solver_options` (see `geopyv_dev_fresh/solver_options_restructure.md`).
/// `Rg` is the serial cascade (`solve_impl`'s `Rg` branch), unchanged.
/// `LayerRg` is the layer-parallel variant
/// ([`Mesh::expand_parallel`], `geopyv_dev_fresh/layer_rg_plan.md` §3): the
/// frontier is solved a graph-independent colour at a time, concurrently
/// via `rayon`. It agrees with `Rg` within Tier C and is run-to-run
/// deterministic (see the `layer_rg_*` tests in this module).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Preconditioning {
    #[default]
    Rg,
    LayerRg,
}

/// The `masking` axis of `solver_options`. `Adaptive`: after the plain
/// order-1 solve, the still-pair screen picks nodes whose neighbours disagree
/// beyond noise, and each is investigated and corrected (cut, then moved to
/// the middle of what it keeps; or order 2) — see [`crate::screen`] and
/// [`crate::seam`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Masking {
    #[default]
    Uniform,
    Adaptive,
}

/// Tunables for `Preconditioning::LayerRg` — see
/// `geopyv_dev_fresh/layer_rg_plan.md` §3.2. Ignored when
/// `preconditioning == Rg`. `#[serde(default)]` on the `SolveConfig` field
/// keeps old `.pyv` configs loading unchanged.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LayerRgConfig {
    /// Frontier roots merged into one parallel round ≈ `batch_factor *
    /// workers`. Over-decomposing (>1) lets rayon work-steal across a slow
    /// node — see `layer_rg_plan.md` §5.7.
    pub batch_factor: usize,
    /// ε-band width: a queued root joins the current round while its
    /// `c_zncc` key is `>= top_key * (1 - root_rel_eps)`.
    pub root_rel_eps: f64,
    /// Cap on rayon workers for this solve's parallel rounds. `None` uses
    /// the global pool (shared with `Field`/`Image`/`Speckle`).
    pub max_workers: Option<usize>,
}

impl Default for LayerRgConfig {
    fn default() -> Self {
        Self { batch_factor: 4, root_rel_eps: 0.02, max_workers: None }
    }
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
    /// `(f_time, g_time)`, seconds; `None` until [`Mesh::set_time`].
    pub time: Option<(f64, f64)>,
    pub time_source: TimeSource,
    /// Still pair (frames 0 → 1) supplying the noise for adaptive masking;
    /// `None` estimates it from a patch around the seed node.
    still: Option<StillPair>,
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
            time: None,
            time_source: TimeSource::None,
            still: None,
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
            time: sol.time,
            time_source: if sol.time.is_some() { TimeSource::Given } else { TimeSource::None },
            still: None,
            solution: Some(sol),
        }
    }

    /// Set the still pair (frames 0 → 1) whose residuals give the noise for
    /// adaptive masking; `None` estimates it from a patch around the seed node.
    pub fn set_noise(&mut self, still: Option<StillPair>) {
        self.still = still;
    }

    /// Reset node positions to the plain solve's (before any move), so a
    /// mesh reused from an adaptive solution starts from its own geometry.
    pub(crate) fn reset_plain_nodes(&mut self) {
        if let Some(seams) = self.solution.as_ref().and_then(|s| s.seams.as_ref()) {
            self.nodes = seams.plain_nodes.clone();
        }
    }

    /// Set the pair's `(f_time, g_time)` from `time=` (`None`: parse the
    /// image file names) — see [`crate::timing`].
    pub fn set_time(&mut self, input: Option<&TimeInput>) -> Result<(), Error> {
        let (time, source) = resolve_pair_time(
            self.f_img.filepath.as_deref(),
            self.g_img.filepath.as_deref(),
            input,
        )?;
        self.time = time;
        self.time_source = source;
        Ok(())
    }

    /// `true` if this mesh has been solved (a `MeshSolution` is available).
    pub fn solved(&self) -> bool {
        self.solution.is_some()
    }

    /// Construct the `Subset` for node `idx` on demand.
    ///
    /// `Mesh::solve` used to build every node's `Subset` eagerly and hold
    /// the whole `Vec<Subset>` for the call's duration; at large meshes /
    /// large local-mask radii that OOMs (see `mds/subset_lazy_construction.md`).
    /// Each node's `Subset` is only ever needed transiently for the single
    /// call that solves (or diagnoses) *that* node, so it's built here on
    /// demand and dropped by the caller instead.
    fn subset_at(
        &self,
        idx: usize,
        local_mask: &LocalMask,
        subset_order: usize,
    ) -> Result<Subset, Error> {
        let coord = [self.nodes[[idx, 0]], self.nodes[[idx, 1]]];
        Subset::new(
            coord,
            local_mask,
            self.mask.as_ref().map(|m| m.view()),
            Arc::clone(&self.f_img),
            Arc::clone(&self.g_img),
            subset_order,
        )
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
    /// * `progress`   — Progress/cancellation observer; `None` shows a
    ///                  terminal progress bar. A cancelled solve returns
    ///                  [`Error::Cancelled`] and stores no solution.
    pub fn solve(
        &mut self,
        local_mask: &LocalMask,
        seed: &SeedConfig,
        cfg: &SolveConfig,
        progress: Option<&dyn SolveProgress>,
    ) -> Result<(), Error> {
        let terminal;
        let progress: &dyn SolveProgress = match progress {
            Some(p) => p,
            None => {
                terminal = TerminalProgress::mesh();
                &terminal
            }
        };
        self.solve_impl(local_mask, seed, cfg, progress)
    }

    fn solve_impl(
        &mut self,
        masks: &LocalMask,
        seed: &SeedConfig,
        cfg: &SolveConfig,
        progress: &dyn SolveProgress,
    ) -> Result<(), Error> {
        let n_nodes = self.nodes.nrows();
        let p_len = 6 * cfg.subset_order;

        // Per-node adjacency, precomputed once -- see `Adjacency` and
        // `layer_rg_plan.md` §11.6 (was a per-call `connectivity_indexed`
        // HashSet build, ~5-10x per node across propagation + corrections).
        let adj = Adjacency::build(&self.elements, self.mesh_order, n_nodes);

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

        let pb = progress;
        pb.mesh_start(n_nodes);

        // --- Seed node (uses seed.tolerance, stricter than cfg.tolerance).
        // Forced preconditioning warp = the user's seed_warp; falls back to a
        // zero-warp second attempt on failure, same as any other node (see
        // `solve_node`) — unlike the old code, which had no seed fallback.
        let seed_node = find_seed_node(&self.nodes, seed.coord);
        let seed_cfg = SolveConfig { tolerance: seed.tolerance, ..cfg.clone() };
        let seed_result = self.solve_node(
            &self.subset_at(seed_node, masks, cfg.subset_order)?,
            seed_node, &[], &c_zncc, &p, &seed_cfg, Some(&seed_warp_norm),
        )?;
        store_result(seed_node, &seed_result, &mut quality_ok, &mut c_zncc, &mut p, &mut displacements, &mut iterations, &mut norms);
        progress::node_done(pb)?;
        stored[seed_node] = true;
        propagated[seed_node] = true;
        queue.push((c_zncc[seed_node].to_bits(), seed_node));

        // --- Frontier propagation. `Rg`: serial cascade (unchanged).
        //     `LayerRg`: parallel independent-set rounds -- see
        //     `geopyv_dev_fresh/layer_rg_plan.md` §3.
        match cfg.preconditioning {
            Preconditioning::Rg => {
                // --- Seed neighbours.
                self.solve_neighbours_from(
                    seed_node, masks, cfg, pb,
                    &mut stored, &mut quality_ok, &mut c_zncc, &mut queue,
                    &mut p, &mut displacements, &mut iterations, &mut norms,
                    &adj,
                )?;

                // --- Reliability-guided queue: highest-C_ZNCC first.
                while let Some((_, cur_idx)) = queue.pop() {
                    if propagated[cur_idx] {
                        continue;
                    }
                    propagated[cur_idx] = true;
                    self.solve_neighbours_from(
                        cur_idx, masks, cfg, pb,
                        &mut stored, &mut quality_ok, &mut c_zncc, &mut queue,
                        &mut p, &mut displacements, &mut iterations, &mut norms,
                        &adj,
                    )?;
                }
            }
            Preconditioning::LayerRg => {
                let root_cap = cfg.layer_rg.batch_factor.max(1).saturating_mul(
                    cfg.layer_rg
                        .max_workers
                        .unwrap_or_else(rayon::current_num_threads)
                        .max(1),
                );
                let eps = cfg.layer_rg.root_rel_eps;

                // A capped worker pool is built ONCE here, not per round --
                // `ThreadPoolBuilder::build` spawns OS threads and is far too
                // expensive to call per colour. `None` = use the global pool.
                let pool = match cfg.layer_rg.max_workers {
                    Some(w) => Some(
                        rayon::ThreadPoolBuilder::new()
                            .num_threads(w.max(1))
                            .build()
                            .map_err(|e| Error::InvalidInput(format!("rayon pool build: {e}")))?,
                    ),
                    None => None,
                };

                // --- Seed neighbours (first parallel round).
                self.expand_parallel(
                    &[seed_node], masks, cfg, pool.as_ref(), pb,
                    &mut stored, &mut quality_ok, &mut c_zncc, &mut queue,
                    &mut p, &mut displacements, &mut iterations, &mut norms,
                    &adj,
                )?;

                // --- ε-band frontier loop.
                while let Some((top_key, top_idx)) = queue.pop() {
                    if propagated[top_idx] {
                        continue;
                    }
                    propagated[top_idx] = true;
                    let mut roots = vec![top_idx];
                    let threshold = f64::from_bits(top_key) * (1.0 - eps);
                    while roots.len() < root_cap {
                        match queue.peek() {
                            Some(&(k, _)) if f64::from_bits(k) >= threshold => {
                                let (_, r) = queue.pop().unwrap();
                                if propagated[r] {
                                    continue;
                                }
                                propagated[r] = true;
                                roots.push(r);
                            }
                            _ => break,
                        }
                    }
                    self.expand_parallel(
                        &roots, masks, cfg, pool.as_ref(), pb,
                        &mut stored, &mut quality_ok, &mut c_zncc, &mut queue,
                        &mut p, &mut displacements, &mut iterations, &mut norms,
                        &adj,
                    )?;
                }
            }
        }

        // --- Corrections (outlier re-solve).
        self.corrections(
            masks,
            cfg,
            pb,
            &mut stored,
            &mut quality_ok,
            &mut c_zncc,
            &mut p,
            &mut displacements,
            &mut iterations,
            &mut norms,
            &adj,
        )?;

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


        // --- Adaptive masking: screen, then investigate and correct.
        let (p, seams_rec) = if cfg.masking == Masking::Adaptive {
            let (p12, rec) = self.adaptive_masking(
                masks, cfg, &adj, seed_node, quality_ok.as_slice().expect("contiguous"), &p,
                &mut displacements, &mut c_zncc, &mut iterations, &mut norms, pb,
            )?;
            (p12, Some(rec))
        } else {
            (p, None)
        };

        // --- Element areas, centroids and strains.
        let centroids = compute_centroids(&self.nodes, &self.elements);
        let areas = element_area(&self.nodes, &self.elements);
        let warps = element_strains(&self.nodes, &self.elements, &displacements, self.mesh_order)?;

        // --- Compatibility check.
        self.check_compatibility(&warps, &centroids)?;

        let f_img_path = self.f_img.filepath.clone().unwrap_or_default();
        let g_img_path = self.g_img.filepath.clone().unwrap_or_default();

        // Template shape/size, retained for `Mesh.inspect(subset_idx=...)`.
        let template_shape = Some(masks.shape.clone());
        let template_sizes = Some(Array1::from_elem(n_nodes, masks.size as u32));

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
            // Adaptive masking mixes order-1 and order-2 nodes; `p` then holds
            // 12 columns with zero quadratic terms on order-1 nodes.
            subset_order: if seams_rec.is_some() { 2 } else { cfg.subset_order as u8 },
            iterations,
            norms,
            f_img_path,
            g_img_path,
            solve_config: Some(cfg.clone()),
            seed: Some(seed.clone()),
            template_shape,
            template_sizes,
            time: self.time,
            seams: seams_rec,
        }));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Adaptive masking
    // -----------------------------------------------------------------------

    /// Screen the plain solve, then investigate and correct the nodes sent
    /// on. Cut nodes move (`self.nodes`); returns the `(N, 12)` warps and
    /// the record. Matches the sandbox chain (`screening_summary.pdf`).
    #[allow(clippy::too_many_arguments)]
    fn adaptive_masking(
        &mut self,
        masks: &LocalMask,
        cfg: &SolveConfig,
        adj: &Adjacency,
        seed_node: usize,
        usable: &[bool],
        p: &Array2<f64>,
        displacements: &mut Array2<f64>,
        c_zncc: &mut Array1<f64>,
        iterations: &mut Array1<u32>,
        norms: &mut Array1<f64>,
        pb: &dyn SolveProgress,
    ) -> Result<(Array2<f64>, MeshSeams), Error> {
        if cfg.subset_order != 1 {
            return Err(Error::InvalidInput("adaptive masking needs subset_order = 1".into()));
        }
        seam::require_lossless(&self.f_img)?;
        seam::require_lossless(&self.g_img)?;
        if let Some(still) = &self.still {
            seam::require_lossless(&still.f_img)?;
            seam::require_lossless(&still.g_img)?;
        }
        if pb.cancelled() {
            return Err(Error::Cancelled);
        }
        let n = self.nodes.nrows();
        let h = seam::correlation_length(&self.f_img)?;
        let out = screen::screen(&ScreenInput {
            nodes: &self.nodes,
            neighbours: &adj.full,
            p,
            usable,
            local_mask: masks,
            global_mask: self.mask.as_ref().map(|m| m.view()),
            f_img: &self.f_img,
            g_img: &self.g_img,
            h,
            still: self.still.as_ref(),
            seed: seed_node,
        })?;
        if pb.cancelled() {
            return Err(Error::Cancelled);
        }

        let boundary_nb = boundary_neighbours(&self.elements, self.mesh_order, n);
        let incident = build_node_element_index(&self.elements, n);
        let seam_cfg = SeamConfig::default();
        let gm = self.mask.as_ref().map(|m| m.view());
        let plain_nodes = self.nodes.clone();
        let sent: Vec<usize> = (0..n).filter(|&i| out.sent[i] && usable[i]).collect();

        // Pass 1 and the first cut, in parallel, at the plain positions.
        let first = |i: usize| -> Option<(Subset, Option<SeamPending>)> {
            let coord = [plain_nodes[[i, 0]], plain_nodes[[i, 1]]];
            let mut sub = Subset::new(coord, masks, gm, Arc::clone(&self.f_img), Arc::clone(&self.g_img), 1).ok()?;
            sub.time = self.time;
            let p0: Vec<f64> = (0..6).map(|k| p[[i, k]]).collect();
            match cfg.method {
                SolveMethod::Icgn => sub.solve_icgn(Some(&p0), cfg.tolerance, cfg.max_norm, cfg.max_iterations),
                SolveMethod::Fagn => sub.solve_fagn(Some(&p0), cfg.tolerance, cfg.max_norm, cfg.max_iterations),
            }
            .ok()?;
            let pending = sub.seam_first(masks, gm, Some(h), &seam_cfg).ok()?;
            Some((sub, pending))
        };
        let mut done: Vec<(usize, Option<(Subset, Option<SeamPending>)>)> =
            sent.par_iter().map(|&i| (i, first(i))).collect();
        if pb.cancelled() {
            return Err(Error::Cancelled);
        }

        // The cut subsets move one node at a time in node order, each checked
        // against the nodes as they then stand. Only element-sharing
        // neighbours interact, so they run in waves: a cut node waits for its
        // lower-numbered cut neighbours, which gives the sequential result.
        let mut cut_at = vec![false; n];
        for (i, d) in &done {
            cut_at[*i] = matches!(d, Some((_, Some(_))));
        }
        let mut wave = vec![0usize; n];
        for i in 0..n {
            if cut_at[i] {
                wave[i] = adj.full[i].iter().filter(|&&j| j < i && cut_at[j]).map(|&j| wave[j] + 1).max().unwrap_or(0);
            }
        }
        let n_waves = (0..n).filter(|&i| cut_at[i]).map(|i| wave[i] + 1).max().unwrap_or(0);
        for w in 0..n_waves {
            let nodes_now = self.nodes.clone();
            let elements = &self.elements;
            done.par_iter_mut().filter(|(i, _)| cut_at[*i] && wave[*i] == w).for_each(|(i, d)| {
                let i = *i;
                let Some((sub, pending)) = d.as_mut() else { return };
                let pending = pending.take().expect("cut subsets are pending");
                let mover = |from: [f64; 2], want: [f64; 2]| {
                    allowed_shift(i, from, want, &nodes_now, &plain_nodes, elements, &boundary_nb, &incident)
                };
                if sub.seam_rest(masks, gm, &seam_cfg, pending, &mover).is_err() {
                    *d = None;
                }
            });
            for (i, d) in &done {
                if cut_at[*i] && wave[*i] == w {
                    if let Some((s, _)) = d {
                        self.nodes[[*i, 0]] = s.coord[0];
                        self.nodes[[*i, 1]] = s.coord[1];
                    }
                }
            }
            if pb.cancelled() {
                return Err(Error::Cancelled);
            }
        }

        let mut p12 = Array2::<f64>::zeros((n, 12));
        for i in 0..n {
            for k in 0..6 {
                p12[[i, k]] = p[[i, k]];
            }
        }
        let mut records: Vec<Option<SeamRecord>> = vec![None; n];
        for (i, sub) in done {
            let Some((s, _)) = sub else { continue };
            let sol = s.solution().expect("solved");
            let r = &sol.result;
            for (k, &v) in r.p.iter().enumerate() {
                p12[[i, k]] = v;
            }
            for k in r.p.len()..12 {
                p12[[i, k]] = 0.0;
            }
            displacements[[i, 0]] = r.p[0];
            displacements[[i, 1]] = r.p[1];
            c_zncc[i] = r.c_zncc.max(0.0);
            iterations[i] = r.iterations as u32;
            norms[i] = r.history.last().map(|h| h.1).unwrap_or(0.0);
            records[i] = sol.seams.clone();
        }
        let plain_p = Array2::from_shape_fn((n, 6), |(i, k)| p[[i, k]]);
        Ok((p12, MeshSeams {
            plain_nodes,
            plain_p,
            flagged: out.flagged,
            sent: out.sent,
            rounds: out.rounds,
            noise_source: out.source,
            noise_scale: out.noise_scale,
            still_mismatch: out.still_mismatch,
            h,
            records,
        }))
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
            SolveMethod::Icgn => {
                subset.solve_icgn_result(Some(warp_0), tolerance, cfg.max_norm, cfg.max_iterations)
            }
            SolveMethod::Fagn => {
                subset.solve_fagn_result(Some(warp_0), tolerance, cfg.max_norm, cfg.max_iterations)
            }
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
        masks: &LocalMask,
        cfg: &SolveConfig,
        pb: &dyn SolveProgress,
        stored: &mut Vec<bool>,
        quality_ok: &mut Array1<bool>,
        c_zncc: &mut Array1<f64>,
        queue: &mut BinaryHeap<(u64, usize)>,
        p: &mut Array2<f64>,
        displacements: &mut Array2<f64>,
        iterations: &mut Array1<u32>,
        norms: &mut Array1<f64>,
        adj: &Adjacency,
    ) -> Result<(), Error> {
        for &nb_idx in adj.get(cur_idx, false) {
            if stored[nb_idx] {
                continue;
            }
            let trusted: Vec<usize> = adj
                .get(nb_idx, true)
                .iter()
                .copied()
                .filter(|&n| stored[n] && quality_ok[n])
                .collect();
            let result = self.solve_node(
                &self.subset_at(nb_idx, masks, cfg.subset_order)?,
                nb_idx, &trusted, c_zncc, p, cfg, None,
            )?;
            store_result(nb_idx, &result, quality_ok, c_zncc, p, displacements, iterations, norms);
            progress::node_done(pb)?;
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

    /// `Preconditioning::LayerRg` frontier expansion — see
    /// `geopyv_dev_fresh/layer_rg_plan.md` §3.3.
    ///
    /// Solves every currently-unsolved neighbour of every node in `roots`
    /// (the same total set `solve_neighbours_from` would, called once per
    /// root), but partitioned into graph-independent colours: within a
    /// colour no two nodes share a mesh edge, so none can be in another's
    /// `trusted` set and they are solved concurrently with `rayon`. Each
    /// colour is folded serially, in ascending node-index order, before the
    /// next colour starts — so a later-coloured node still sees an
    /// adjacent earlier-coloured node's result, matching the serial
    /// "lower index first" ordering. Fully deterministic (§5 of the plan).
    #[allow(clippy::too_many_arguments)]
    fn expand_parallel(
        &self,
        roots: &[usize],
        masks: &LocalMask,
        cfg: &SolveConfig,
        pool: Option<&rayon::ThreadPool>,
        pb: &dyn SolveProgress,
        stored: &mut Vec<bool>,
        quality_ok: &mut Array1<bool>,
        c_zncc: &mut Array1<f64>,
        queue: &mut BinaryHeap<(u64, usize)>,
        p: &mut Array2<f64>,
        displacements: &mut Array2<f64>,
        iterations: &mut Array1<u32>,
        norms: &mut Array1<f64>,
        adj: &Adjacency,
    ) -> Result<(), Error> {
        // 1. Candidates: sorted-unique unsolved neighbours of all roots.
        let mut candidates: Vec<usize> = roots
            .iter()
            .flat_map(|&r| adj.get(r, false).iter().copied())
            .filter(|&nb| !stored[nb])
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(());
        }

        // 2. Greedy graph colouring over the full (edge-sharing) adjacency
        //    -- the same relation `trusted` is built from.
        let colour = greedy_graph_colour(adj, &candidates);
        let n_colours = colour.iter().copied().max().map_or(0, |c| c + 1);

        // 3. Per colour: freeze trusted sets, solve in parallel, fold serially.
        for c in 0..n_colours {
            let inputs: Vec<(usize, Vec<usize>)> = candidates
                .iter()
                .copied()
                .zip(colour.iter().copied())
                .filter(|&(_, col)| col == c)
                .map(|(nb, _)| {
                    let trusted: Vec<usize> = adj
                        .get(nb, true)
                        .iter()
                        .copied()
                        .filter(|&t| stored[t] && quality_ok[t])
                        .collect();
                    (nb, trusted)
                })
                .collect();

            let c_zncc_ro: &Array1<f64> = c_zncc;
            let p_ro: &Array2<f64> = p;
            let solve_colour = || -> Result<Vec<(usize, SolveResult)>, Error> {
                inputs
                    .par_iter()
                    .map(|(nb, trusted)| {
                        let subset = self.subset_at(*nb, masks, cfg.subset_order)?;
                        let r = self.solve_node(
                            &subset, *nb, trusted, c_zncc_ro, p_ro, cfg, None,
                        )?;
                        Ok::<(usize, SolveResult), Error>((*nb, r))
                    })
                    .collect()
            };
            let mut results = match pool {
                Some(p) => p.install(solve_colour)?,
                None => solve_colour()?,
            };

            // Serial fold, canonical (ascending node-index) order.
            results.sort_unstable_by_key(|(nb, _)| *nb);
            for (nb, r) in results {
                store_result(
                    nb, &r, quality_ok, c_zncc, p, displacements, iterations, norms,
                );
                progress::node_done(pb)?;
                stored[nb] = true;
                queue.push((c_zncc[nb].to_bits(), nb));
            }
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
        masks: &LocalMask,
        cfg: &SolveConfig,
        solved: &mut Vec<bool>,
        quality_ok: &mut Array1<bool>,
        c_zncc: &mut Array1<f64>,
        p: &mut Array2<f64>,
        displacements: &mut Array2<f64>,
        iterations: &mut Array1<u32>,
        norms: &mut Array1<f64>,
        adj: &Adjacency,
    ) -> Result<(), Error> {
        let mut order = corr(c_zncc.view());
        // Ascending by C_ZNCC (worst first), matching Python's np.argsort.
        // `total_cmp`, not `partial_cmp().unwrap_or(Equal)`: this array
        // isn't pre-filtered to exclude non-finite values (unlike the
        // *_stats functions' local sort inputs), so treating "NaN vs
        // anything" as Equal is inconsistent with genuine orderings
        // between two other finite, unequal values elsewhere in the same
        // array -- not just wrong, but an invalid total order that Rust's
        // sort now actively detects and panics on (see line ~748).
        order.sort_by(|&a, &b| c_zncc[a].total_cmp(&c_zncc[b]));

        let mut unimproved: HashSet<usize> = HashSet::new();
        for i in 0..order.len() {
            let j = order[i];
            let excluded: HashSet<usize> =
                order[i + 1..].iter().copied().chain(unimproved.iter().copied()).collect();
            let trusted: Vec<usize> = adj
                .get(j, true)
                .iter()
                .copied()
                .filter(|&nb| !excluded.contains(&nb) && quality_ok[nb])
                .collect();

            let result = self.solve_node(
                &self.subset_at(j, masks, cfg.subset_order)?,
                j, &trusted, c_zncc, p, cfg, None,
            )?;
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
        masks: &LocalMask,
        cfg: &SolveConfig,
        _pb: &dyn SolveProgress,
        solved: &mut Vec<bool>,
        quality_ok: &mut Array1<bool>,
        c_zncc: &mut Array1<f64>,
        p: &mut Array2<f64>,
        displacements: &mut Array2<f64>,
        iterations: &mut Array1<u32>,
        norms: &mut Array1<f64>,
        adj: &Adjacency,
    ) -> Result<(), Error> {
        self.correlation_improvements(
            masks, cfg, solved, quality_ok, c_zncc, p, displacements, iterations, norms,
            adj,
        )?;

        let (_, flow_ids, flow_lq, flow_iqr) =
            flow_stats(adj, displacements.view());
        let (r_vals, r_ids) =
            r_stats(adj, &self.elements, &self.nodes, displacements.view());

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
        // A node can enter `full_id` via `flow_ids` alone: flow_calc
        // substitutes a *finite* sentinel (-1.0) for a non-finite
        // displacement, so such a node can clear the flow fence even
        // though its own `r_vals` entry (raw displacement magnitude, no
        // such substitution) is still genuinely NaN. `total_cmp`, not
        // `partial_cmp().unwrap_or(Equal)` -- this array mixes that NaN
        // with multiple distinct finite values, and treating "NaN vs
        // anything" as Equal isn't a valid total order there (confirmed:
        // Rust's sort panics with "does not correctly implement a total
        // order" on exactly this pattern, it doesn't just silently
        // misorder).
        full_id.sort_by(|&a, &b| r_vals[a].total_cmp(&r_vals[b]));

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
                    adj,
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
            // Exclude current and later outliers from preconditioning.
            let later: HashSet<usize> = active_ids[i..].iter().copied().collect();
            let trusted: Vec<usize> = adj
                .get(j, true)
                .iter()
                .copied()
                .filter(|&nb| !later.contains(&nb) && quality_ok[nb])
                .collect();

            let result = self.solve_node(
                &self.subset_at(j, masks, cfg.subset_order)?,
                j, &trusted, c_zncc, p, cfg, None,
            )?;
            store_result(j, &result, quality_ok, c_zncc, p, displacements, iterations, norms);
            solved[j] = true;
        }
        Ok(())
    }

    /// Compatibility check: reject mesh if any element has det(F) ≤ 0 (fold-over).
    fn check_compatibility(&self, warps: &Array2<f64>, centroids: &Array2<f64>) -> Result<(), Error> {
        for e in 0..warps.nrows() {
            let j00 = 1.0 + warps[[e, 2]]; // 1 + du/dx
            let j01 =       warps[[e, 3]]; //     dv/dx
            let j10 =       warps[[e, 4]]; //     du/dy
            let j11 = 1.0 + warps[[e, 5]]; // 1 + dv/dy
            let det = j00 * j11 - j01 * j10;
            if det <= 0.0 {
                let cx = centroids[[e, 0]];
                let cy = centroids[[e, 1]];
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

/// `node_id -> [element row indices containing it]`, built once per
/// `Mesh::solve()` call (O(n_elements)) so that the solve pipeline's many
/// per-node/per-element [`connectivity_indexed`] lookups don't each rescan
/// the whole element array ([`connectivity`]'s brute-force scan is fine for the occasional one-off
/// lookup `python/src/py_mesh.rs` exposes, but made the whole solve
/// pipeline O(n_nodes * n_elements) when called ~once per node throughout
/// propagation/corrections -- this index turns each lookup into
/// O(node degree) instead).
fn build_node_element_index(elements: &Array2<usize>, n_nodes: usize) -> Vec<Vec<usize>> {
    let mut index = vec![Vec::new(); n_nodes];
    for e in 0..elements.nrows() {
        for k in 0..elements.ncols() {
            index[elements[[e, k]]].push(e);
        }
    }
    index
}

/// Identical output to [`connectivity`] for the same `(elements, mesh_order,
/// idx, full)` -- only iterates the elements `incident[idx]` already says
/// contain `idx` (from [`build_node_element_index`]) instead of rescanning
/// every element to rediscover that membership. See
/// `test_connectivity_indexed_matches_connectivity` for a direct
/// equivalence check against the original.
fn connectivity_indexed(
    elements: &Array2<usize>,
    mesh_order: u8,
    idx: usize,
    full: bool,
    incident: &[Vec<usize>],
) -> Vec<usize> {
    let mut result: HashSet<usize> = HashSet::new();
    if mesh_order == 1 || full {
        for &e in &incident[idx] {
            for k in 0..elements.ncols() {
                let nb = elements[[e, k]];
                if nb != idx {
                    result.insert(nb);
                }
            }
        }
    } else {
        // Order-2, full=false: edge-adjacent connectivity only.
        for &e in &incident[idx] {
            let ncols = elements.ncols();
            for col in 0..ncols {
                if elements[[e, col]] != idx {
                    continue;
                }
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
    }
    let mut v: Vec<usize> = result.into_iter().collect();
    v.sort_unstable();
    v
}

/// Per-node adjacency lists, precomputed once per `solve_impl`
/// (`geopyv_dev_fresh/layer_rg_plan.md` §11.6). [`connectivity_indexed`]
/// builds a `HashSet` + sorts on every call and is hit ~5–10× per node
/// across propagation and corrections; this hoists all of that to O(n) up
/// front. `full[i]` = every node sharing an element with `i` (the relation
/// `trusted` and the corrections stats use); `edge[i]` = edge-adjacent
/// only (order-2 `full=false`; identical to `full` for order-1). Both
/// ascending-sorted, byte-for-byte what `connectivity_indexed` returned.
pub struct Adjacency {
    full: Vec<Vec<usize>>,
    edge: Vec<Vec<usize>>,
}

impl Adjacency {
    fn build(elements: &Array2<usize>, mesh_order: u8, n_nodes: usize) -> Self {
        let incident = build_node_element_index(elements, n_nodes);
        let full: Vec<Vec<usize>> = (0..n_nodes)
            .map(|i| connectivity_indexed(elements, mesh_order, i, true, &incident))
            .collect();
        let edge = if mesh_order == 1 {
            full.clone()
        } else {
            (0..n_nodes)
                .map(|i| connectivity_indexed(elements, mesh_order, i, false, &incident))
                .collect()
        };
        Adjacency { full, edge }
    }

    #[inline]
    fn get(&self, idx: usize, full: bool) -> &[usize] {
        if full { &self.full[idx] } else { &self.edge[idx] }
    }
}

/// Greedy proper graph colouring of `candidates` under the full
/// (edge-sharing) adjacency — see `geopyv_dev_fresh/layer_rg_plan.md` §3.3.
///
/// `candidates` must be sorted ascending. Each node is visited in that
/// order and assigned the lowest colour not already used by one of its
/// already-coloured `candidates` neighbours. Deterministic: [`Adjacency`]
/// lists are sorted and the visit order is fixed, so the same input always
/// yields the same `Vec`.
///
/// Returned: `colour[i]` for `candidates[i]`, in `0..k`.
fn greedy_graph_colour(adj: &Adjacency, candidates: &[usize]) -> Vec<usize> {
    let mut colour = vec![usize::MAX; candidates.len()];
    let mut used: Vec<bool> = Vec::new();
    for i in 0..candidates.len() {
        used.clear();
        for &nb in adj.get(candidates[i], true) {
            if let Ok(pos) = candidates.binary_search(&nb) {
                let cn = colour[pos];
                if cn != usize::MAX {
                    if cn >= used.len() {
                        used.resize(cn + 1, false);
                    }
                    used[cn] = true;
                }
            }
        }
        colour[i] = (0..).find(|&k| k >= used.len() || !used[k]).unwrap();
    }
    colour
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

/// Same as [`flow_calc`] but takes a precomputed `incident` index
/// ([`build_node_element_index`]) and uses [`connectivity_indexed`] instead
/// of [`connectivity`]'s per-call element rescan. `flow_calc` itself stays
/// untouched for its one external caller (`python/src/py_mesh.rs`'s
/// single-node lookup); this variant is for the solve pipeline's hot loops
/// (`flow_stats`, `flow_calc_excluding`), called ~once per node.
fn flow_calc_indexed(
    idx: usize,
    adj: &Adjacency,
    displacements: ArrayView2<f64>,
    exclude: &HashSet<usize>,
    displacement_override: Option<[f64; 2]>,
) -> f64 {
    let disp = displacement_override
        .unwrap_or([displacements[[idx, 0]], displacements[[idx, 1]]]);
    if !disp[0].is_finite() || !disp[1].is_finite() {
        return -1.0;
    }
    let valid: Vec<usize> = adj
        .get(idx, true)
        .iter()
        .copied()
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
    adj: &Adjacency,
    displacements: ArrayView2<f64>,
    displacement_override: Option<[f64; 2]>,
) -> f64 {
    flow_calc_indexed(idx, adj, displacements, exclude, displacement_override)
}

/// Absolute displacement magnitude.
pub fn r_calc(displacement: [f64; 2]) -> f64 {
    (displacement[0] * displacement[0] + displacement[1] * displacement[1]).sqrt()
}

/// Flow outlier IDs, LQ, IQR for the full mesh (no exclusions).
fn flow_stats(
    adj: &Adjacency,
    displacements: ArrayView2<f64>,
) -> (Vec<f64>, Vec<usize>, f64, f64) {
    let n = displacements.nrows();
    let empty = HashSet::new();
    let flow: Vec<f64> = (0..n)
        .map(|i| flow_calc_indexed(i, adj, displacements, &empty, None))
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
    adj: &Adjacency,
    elements: &Array2<usize>,
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
            for &nb in adj.get(corner, true) {
                local.insert(nb);
            }
        }
        // Non-quality_ok nodes can carry a non-finite displacement (e.g. a
        // diverged ICGN solve) at this point in the pipeline -- the final
        // quality_ok gate runs after corrections(), not before, so this
        // statistics pass must not assume every node converged. Mirrors
        // flow_stats's identical `is_finite` filter just above; an
        // unfiltered sort here previously panicked on NaN comparisons for
        // meshes with any unconverged node (e.g. a subset straddling a
        // sharp discontinuity that ICGN can't fit at all).
        let local_r: Vec<f64> = local.iter().map(|&nb| r[nb]).filter(|f| f.is_finite()).collect();
        if local_r.is_empty() {
            continue;
        }
        let mut sr = local_r.clone();
        sr.sort_by(|a, b| a.total_cmp(b));
        let lq = percentile(&sr, 25.0);
        let uq = percentile(&sr, 75.0);
        let iqr = uq - lq;
        let fence = uq + 4.0 * iqr;
        for &nb in &local {
            // r[nb] > fence is false for NaN (IEEE-754), so a non-finite
            // node is simply never flagged here -- it stays whatever
            // quality_ok already says, handled by the caller's gate.
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
/// Per node, its neighbours along the mesh boundary (outer boundary and
/// exclusion edges): the other ends of the element sides (sub-sides for
/// 6-node elements) that belong to one element only. Empty for interior
/// nodes.
fn boundary_neighbours(elements: &Array2<usize>, mesh_order: u8, n_nodes: usize) -> Vec<Vec<usize>> {
    let sides: &[(usize, usize)] = if mesh_order == 2 {
        // [c0, c1, c2, m01, m12, m20]
        &[(0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0)]
    } else {
        &[(0, 1), (1, 2), (2, 0)]
    };
    let mut count: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for e in 0..elements.nrows() {
        for &(a, b) in sides {
            let (u, v) = (elements[[e, a]], elements[[e, b]]);
            *count.entry((u.min(v), u.max(v))).or_insert(0) += 1;
        }
    }
    let mut out = vec![Vec::new(); n_nodes];
    for ((u, v), c) in count {
        if c == 1 {
            out[u].push(v);
            out[v].push(u);
        }
    }
    out
}

/// Largest turn (degrees) along the boundary at a node that still counts as
/// lying on an edge rather than defining a corner of the geometry.
const BOUNDARY_CORNER_DEG: f64 = 5.0;

/// The part of `want` that node `i`, now at `from`, may move. Interior nodes
/// move towards `from + want`. Boundary nodes move only along their edge (the
/// part of `want` along it, kept on the straight edge or on the circle through
/// the node and its two boundary neighbours, and at most half way to either),
/// so the geometry is unchanged; a node where the boundary turns by more than
/// [`BOUNDARY_CORNER_DEG`] defines the geometry and does not move. The move
/// then goes only as far (found by bisection) as keeps every incident element
/// the same way round and at least a quarter of its area in the plain mesh.
#[allow(clippy::too_many_arguments)]
fn allowed_shift(
    i: usize,
    from: [f64; 2],
    want: [f64; 2],
    nodes: &Array2<f64>,
    plain: &Array2<f64>,
    elements: &Array2<usize>,
    boundary_nb: &[Vec<usize>],
    incident: &[Vec<usize>],
) -> [f64; 2] {
    let none = [0.0, 0.0];
    let target: Box<dyn Fn(f64) -> [f64; 2]> = match boundary_nb[i].as_slice() {
        [] => Box::new(move |f: f64| [from[0] + f * want[0], from[1] + f * want[1]]),
        &[a, b] => {
            let pa = [nodes[[a, 0]], nodes[[a, 1]]];
            let pb = [nodes[[b, 0]], nodes[[b, 1]]];
            let u = [from[0] - pa[0], from[1] - pa[1]];
            let v = [pb[0] - from[0], pb[1] - from[1]];
            let (lu, lv) = (u[0].hypot(u[1]), v[0].hypot(v[1]));
            let cos = ((u[0] * v[0] + u[1] * v[1]) / (lu * lv)).clamp(-1.0, 1.0);
            if cos.acos().to_degrees() > BOUNDARY_CORNER_DEG {
                return none;
            }
            let lt = (pb[0] - pa[0]).hypot(pb[1] - pa[1]);
            let t = [(pb[0] - pa[0]) / lt, (pb[1] - pa[1]) / lt];
            let s_max = 0.5 * lu.min(lv);
            let s_full = (t[0] * want[0] + t[1] * want[1]).clamp(-s_max, s_max);
            let circle = circumcircle(pa, from, pb);
            Box::new(move |f: f64| {
                let q = [from[0] + f * s_full * t[0], from[1] + f * s_full * t[1]];
                match circle {
                    Some((o, r)) => {
                        let d = (q[0] - o[0]).hypot(q[1] - o[1]);
                        [o[0] + r * (q[0] - o[0]) / d, o[1] + r * (q[1] - o[1]) / d]
                    }
                    None => q,
                }
            })
        }
        _ => return none,
    };
    let ok = |f: f64| element_ok_move(i, target(f), nodes, plain, elements, &incident[i]);
    let f = if ok(1.0) {
        1.0
    } else {
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..12 {
            let mid = 0.5 * (lo + hi);
            if ok(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    if f == 0.0 {
        return none;
    }
    let q = target(f);
    [q[0] - from[0], q[1] - from[1]]
}

/// Centre and radius of the circle through three points, or `None` if they
/// are (nearly) collinear.
fn circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<([f64; 2], f64)> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    let scale = (b[0] - a[0]).hypot(b[1] - a[1]).max((c[0] - b[0]).hypot(c[1] - b[1]));
    if d.abs() < 1e-9 * scale * scale {
        return None;
    }
    let (a2, b2, c2) = (a[0] * a[0] + a[1] * a[1], b[0] * b[0] + b[1] * b[1], c[0] * c[0] + c[1] * c[1]);
    let o = [
        (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
        (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d,
    ];
    let r = (a[0] - o[0]).hypot(a[1] - o[1]);
    (r < 1e6 * scale).then_some((o, r))
}

/// Whether moving node `i` to `to` (the other nodes where they now stand)
/// keeps every incident element (each of the four linear parts of a 6-node
/// element) the same way round as in the plain mesh `plain` and at least a
/// quarter of its area there, however many moves have shaped it.
fn element_ok_move(i: usize, to: [f64; 2], nodes: &Array2<f64>, plain: &Array2<f64>, elements: &Array2<usize>, incident: &[usize]) -> bool {
    let parts: &[[usize; 3]] = if elements.ncols() == 6 {
        &[[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]]
    } else {
        &[[0, 1, 2]]
    };
    let pos = |k: usize, moved: bool| match (moved, k == i) {
        (false, _) => [plain[[k, 0]], plain[[k, 1]]],
        (true, true) => to,
        (true, false) => [nodes[[k, 0]], nodes[[k, 1]]],
    };
    let area = |t: [usize; 3], moved: bool| {
        let (a, b, c) = (pos(t[0], moved), pos(t[1], moved), pos(t[2], moved));
        0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
    };
    incident.iter().all(|&e| {
        parts.iter().all(|part| {
            let t = part.map(|k| elements[[e, k]]);
            if !t.contains(&i) {
                return true;
            }
            let (before, after) = (area(t, false), area(t, true));
            before * after > 0.0 && after.abs() >= 0.25 * before.abs()
        })
    })
}

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

    // ---- Adaptive masking: move rule ----

    /// A 3×3-node grid of order-1 triangles on [0, 20]², spacing 10.
    fn move_grid() -> (Array2<f64>, Array2<usize>) {
        let nodes = Array2::from_shape_fn((9, 2), |(i, c)| if c == 0 { (i % 3) as f64 * 10.0 } else { (i / 3) as f64 * 10.0 });
        let mut el = Vec::new();
        for r in 0..2 {
            for c in 0..2 {
                let a = r * 3 + c;
                el.push([a, a + 1, a + 4]);
                el.push([a, a + 4, a + 3]);
            }
        }
        (nodes, Array2::from_shape_fn((8, 3), |(e, k)| el[e][k]))
    }

    fn at(nodes: &Array2<f64>, i: usize) -> [f64; 2] {
        [nodes[[i, 0]], nodes[[i, 1]]]
    }

    #[test]
    fn move_interior_node_moves_all_the_way() {
        let (nodes, el) = move_grid();
        let (bnd, inc) = (boundary_neighbours(&el, 1, 9), build_node_element_index(&el, 9));
        let s = allowed_shift(4, at(&nodes, 4), [1.8, 2.4], &nodes, &nodes, &el, &bnd, &inc);
        assert!((s[0] - 1.8).abs() < 1e-12 && (s[1] - 2.4).abs() < 1e-12);
    }

    #[test]
    fn move_limited_to_keep_elements_sound() {
        let (nodes, el) = move_grid();
        let (bnd, inc) = (boundary_neighbours(&el, 1, 9), build_node_element_index(&el, 9));
        let s = allowed_shift(4, at(&nodes, 4), [9.0, 0.0], &nodes, &nodes, &el, &bnd, &inc);
        assert!(s[0] > 1.0 && s[0] < 9.0 && s[1] == 0.0, "{s:?}");
        let q = [10.0 + s[0], 10.0];
        assert!(element_ok_move(4, q, &nodes, &nodes, &el, &inc[4]));
        assert!(!element_ok_move(4, [19.0, 10.0], &nodes, &nodes, &el, &inc[4]));
    }

    #[test]
    fn second_move_is_held_to_the_plain_mesh() {
        // a node moved as far as allowed cannot move further the same way:
        // elements are judged against their plain-mesh areas, not the shrunk ones
        let (nodes, el) = move_grid();
        let (bnd, inc) = (boundary_neighbours(&el, 1, 9), build_node_element_index(&el, 9));
        let one = allowed_shift(4, at(&nodes, 4), [9.0, 0.0], &nodes, &nodes, &el, &bnd, &inc);
        let mut moved = nodes.clone();
        moved[[4, 0]] += one[0];
        let two = allowed_shift(4, at(&moved, 4), [9.0, 0.0], &moved, &nodes, &el, &bnd, &inc);
        assert!(two[0].abs() < 0.01, "moved on past the limit: {two:?}");
    }

    #[test]
    fn move_edge_node_slides_along_its_edge() {
        let (nodes, el) = move_grid();
        let (bnd, inc) = (boundary_neighbours(&el, 1, 9), build_node_element_index(&el, 9));
        // node 1 = (10, 0) on the bottom edge: only the part along the edge is used
        let s = allowed_shift(1, at(&nodes, 1), [2.0, 3.0], &nodes, &nodes, &el, &bnd, &inc);
        assert!(s[1].abs() < 1e-12 && (s[0] - 2.0).abs() < 1e-12, "{s:?}");
    }

    #[test]
    fn move_corner_does_not_move() {
        let (nodes, el) = move_grid();
        let (bnd, inc) = (boundary_neighbours(&el, 1, 9), build_node_element_index(&el, 9));
        assert_eq!(allowed_shift(0, at(&nodes, 0), [2.0, 2.0], &nodes, &nodes, &el, &bnd, &inc), [0.0, 0.0]);
    }

    #[test]
    fn move_on_an_arc_stays_on_the_circle() {
        // Three boundary nodes on a circle of radius 400 about the origin, plus
        // an inner node to close two triangles.
        let ang = [-0.04_f64, 0.0, 0.04];
        let mut pts: Vec<[f64; 2]> = ang.iter().map(|a| [400.0 * a.cos(), 400.0 * a.sin()]).collect();
        pts.push([380.0, 0.0]);
        let nodes = Array2::from_shape_fn((4, 2), |(i, c)| pts[i][c]);
        let el = Array2::from_shape_fn((2, 3), |(e, k)| [[0, 1, 3], [1, 2, 3]][e][k]);
        let (bnd, inc) = (boundary_neighbours(&el, 1, 4), build_node_element_index(&el, 4));
        let s = allowed_shift(1, at(&nodes, 1), [-1.0, 3.0], &nodes, &nodes, &el, &bnd, &inc);
        let q = [400.0 + s[0], s[1]];
        assert!((q[0].hypot(q[1]) - 400.0).abs() < 1e-9, "left the circle: {q:?}");
        assert!(s[1] > 2.9);
    }

    #[test]
    fn boundary_neighbours_order2_uses_sub_sides() {
        // one 6-node element: [c0, c1, c2, m01, m12, m20]
        let el = Array2::from_shape_vec((1, 6), vec![0, 1, 2, 3, 4, 5]).unwrap();
        let b = boundary_neighbours(&el, 2, 6);
        let mut m3 = b[3].clone();
        m3.sort();
        assert_eq!(m3, vec![0, 1]);
        assert_eq!(b[0].len(), 2);
    }
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

    /// [`connectivity_indexed`] (used throughout the solve pipeline to avoid
    /// connectivity's O(n_elements)-per-call rescan, see
    /// `build_node_element_index`'s doc comment) must return byte-identical
    /// results to the original [`connectivity`] for every node and both
    /// `full` settings, in both mesh orders -- this is the correctness
    /// guarantee the perf fix rests on.
    #[test]
    fn test_connectivity_indexed_matches_connectivity() {
        let elements1 = array![[0usize, 1, 2], [1, 3, 2]];
        let incident1 = build_node_element_index(&elements1, 4);
        for idx in 0..4 {
            for &full in &[false, true] {
                assert_eq!(
                    connectivity_indexed(&elements1, 1, idx, full, &incident1),
                    connectivity(&elements1, 1, idx, full),
                    "order=1 idx={idx} full={full}",
                );
            }
        }

        let elements2 = array![[0usize, 1, 2, 4, 5, 6], [1, 3, 2, 7, 8, 5]];
        let incident2 = build_node_element_index(&elements2, 9);
        for idx in 0..9 {
            for &full in &[false, true] {
                assert_eq!(
                    connectivity_indexed(&elements2, 2, idx, full, &incident2),
                    connectivity(&elements2, 2, idx, full),
                    "order=2 idx={idx} full={full}",
                );
            }
        }
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

    // -----------------------------------------------------------------------
    // r_stats
    // -----------------------------------------------------------------------

    #[test]
    fn r_stats_ignores_non_finite_displacement_without_panicking() {
        // Node 1 failed to converge (e.g. ICGN diverged on a subset
        // straddling a sharp discontinuity) and carries a NaN displacement
        // -- corrections() runs r_stats before the final quality_ok gate,
        // so this is a real, reachable state, not a hypothetical one.
        // Previously panicked (`sr.sort_by(...).unwrap()` on a NaN
        // comparison); must now complete and simply not flag node 1.
        let (nodes, elements) = unit_triangle_mesh();
        let displacements = array![
            [0.0, 0.0],
            [f64::NAN, f64::NAN],
            [0.05, 0.0],
            [0.05, 0.0],
        ];
        let adj = Adjacency::build(&elements, 1, nodes.nrows());
        let (r, r_ids) = r_stats(&adj, &elements, &nodes, displacements.view());
        assert!(r[1].is_nan());
        assert!(!r_ids.contains(&1));
    }

    // -----------------------------------------------------------------------
    // Synthetic-texture solve fixtures
    // -----------------------------------------------------------------------

    fn make_synthetic_texture_mesh(size: usize) -> Array2<f64> {
        let mut img = Array2::<f64>::zeros((size, size));
        for y in 0..size {
            for x in 0..size {
                let xf = x as f64;
                let yf = y as f64;
                let v = (0.7 * xf).sin() * (0.5 * yf).cos()
                    + (0.31 * xf + 0.2 * yf).sin()
                    + (0.13 * xf - 0.42 * yf).cos();
                img[[y, x]] = 128.0 + 40.0 * v;
            }
        }
        img
    }

    /// Rigid horizontal shift via bilinear resampling -- doesn't need to
    /// match the production B-spline evaluator, just needs realistic
    /// shifted texture for the solver to correlate against (same approach
    /// `subset.rs`'s own synthetic-image tests use).
    fn shift_image_uniform(src: &Array2<f64>, u: f64) -> Array2<f64> {
        let (h, w) = src.dim();
        let mut out = Array2::<f64>::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let src_x = x as f64 - u;
                let x0 = src_x.floor();
                let frac = src_x - x0;
                let x0i = (x0.max(0.0) as usize).min(w - 1);
                let x1i = ((x0 + 1.0).max(0.0) as usize).min(w - 1);
                out[[y, x]] = src[[y, x0i]] * (1.0 - frac) + src[[y, x1i]] * frac;
            }
        }
        out
    }

    /// Fresh `(Mesh, SeedConfig, SolveConfig)` over a deterministic
    /// synthetic uniformly-shifted texture -- `Mesh::new` is deterministic
    /// (no RNG anywhere in `generate_mesh`), so two calls with identical
    /// arguments produce an identical node/element layout, letting tests
    /// build two independent `Mesh`es and compare their solved results
    /// node-for-node.
    fn make_test_mesh_and_solve_inputs() -> (Mesh, SeedConfig, SolveConfig) {
        use crate::image::Image;
        let size = 100usize;
        let texture = make_synthetic_texture_mesh(size);
        let shifted = shift_image_uniform(&texture, 1.5);
        let ref_img = Arc::new(Image::from_array(texture, 15));
        let tar_img = Arc::new(Image::from_array(shifted, 15));

        let boundary = array![[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]];
        let mesh = Mesh::new(
            boundary.view(), false, &[], &[], (1.0, 100.0), 12, 1,
            Arc::clone(&ref_img), Arc::clone(&tar_img),
        ).unwrap();

        let seed = SeedConfig { coord: [50.0, 50.0], warp: vec![0.0; 6], tolerance: 0.5 };
        let cfg = SolveConfig {
            tolerance: 0.5, subset_order: 1, override_active: true, ..Default::default()
        };
        (mesh, seed, cfg)
    }

    // -----------------------------------------------------------------------
    // Step-discontinuity image pair
    // -----------------------------------------------------------------------

    /// Step-discontinuity texture: rows `0..split_row` shift by `u_top`,
    /// rows `split_row..` shift by `u_bottom` -- a genuine horizontal-
    /// displacement discontinuity localised at `split_row`, so nodes near
    /// it have real, elevated shear strain relative to nodes far from it
    /// (unlike `shift_image_uniform`'s uniform shift, which has ~zero
    /// strain everywhere).
    fn shift_image_band(src: &Array2<f64>, split_row: usize, u_top: f64, u_bottom: f64) -> Array2<f64> {
        let (h, w) = src.dim();
        let mut out = Array2::<f64>::zeros((h, w));
        for y in 0..h {
            let u = if y < split_row { u_top } else { u_bottom };
            for x in 0..w {
                let src_x = x as f64 - u;
                let x0 = src_x.floor();
                let frac = src_x - x0;
                let x0i = (x0.max(0.0) as usize).min(w - 1);
                let x1i = ((x0 + 1.0).max(0.0) as usize).min(w - 1);
                out[[y, x]] = src[[y, x0i]] * (1.0 - frac) + src[[y, x1i]] * frac;
            }
        }
        out
    }

    /// Counts `node_done` calls; cancels once `stop_after` are reported.
    struct CountingProgress {
        stop_after: usize,
        started: std::sync::atomic::AtomicUsize,
        done: std::sync::atomic::AtomicUsize,
    }
    impl crate::progress::SolveProgress for CountingProgress {
        fn mesh_start(&self, n: usize) {
            self.started.store(n, std::sync::atomic::Ordering::Relaxed);
        }
        fn node_done(&self) {
            self.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn cancelled(&self) -> bool {
            self.done.load(std::sync::atomic::Ordering::Relaxed) >= self.stop_after
        }
    }

    #[test]
    fn solve_progress_reports_every_node_and_cancels() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let local_mask = LocalMask::circle(6).unwrap();

        // Uncancelled: one mesh_start(n) and at least n node_done calls
        // (corrections may re-report), same result as the terminal bar.
        let (mut mesh, seed, cfg) = make_band_test_mesh_and_solve_inputs();
        let n = mesh.nodes().nrows();
        let p = CountingProgress { stop_after: usize::MAX, started: AtomicUsize::new(0), done: AtomicUsize::new(0) };
        mesh.solve(&local_mask, &seed, &cfg, Some(&p)).unwrap();
        assert_eq!(p.started.load(Ordering::Relaxed), n);
        assert!(p.done.load(Ordering::Relaxed) >= n);
        let (mut reference, _, _) = make_band_test_mesh_and_solve_inputs();
        reference.solve(&local_mask, &seed, &cfg, Some(&())).unwrap();
        assert_eq!(mesh.solution().unwrap().displacements, reference.solution().unwrap().displacements);

        // Cancelled part-way: Error::Cancelled, nothing stored.
        let (mut mesh, seed, cfg) = make_band_test_mesh_and_solve_inputs();
        let p = CountingProgress { stop_after: 5, started: AtomicUsize::new(0), done: AtomicUsize::new(0) };
        assert!(matches!(mesh.solve(&local_mask, &seed, &cfg, Some(&p)), Err(Error::Cancelled)));
        assert_eq!(p.done.load(Ordering::Relaxed), 5);
        assert!(mesh.solution().is_none());
    }

    fn make_band_test_mesh_and_solve_inputs() -> (Mesh, SeedConfig, SolveConfig) {
        let size = 100usize;
        let texture = make_synthetic_texture_mesh(size);
        let shifted = shift_image_band(&texture, 50, 1.5, -1.5);
        let ref_img = Arc::new(Image::from_array(texture, 15));
        let tar_img = Arc::new(Image::from_array(shifted, 15));

        // A FIXED structured grid built directly (not `Mesh::new`, whose
        // Ruppert refinement is not deterministic across calls), so A/B
        // tests see identical nodes on both sides. 7.5px spacing over 20..80,
        // denser than make_test_mesh_and_solve_inputs so "near the band" vs
        // "far" has enough nodes to separate.
        let n_side = 9usize;
        let at = |k: usize| 20.0 + 7.5 * k as f64;
        let nodes = Array2::from_shape_fn((n_side * n_side, 2), |(i, j)| {
            if j == 0 { at(i % n_side) } else { at(i / n_side) }
        });
        let id = |c: usize, r: usize| r * n_side + c;
        let mut elems: Vec<usize> = Vec::new();
        for r in 0..n_side - 1 {
            for c in 0..n_side - 1 {
                elems.extend_from_slice(&[id(c, r), id(c + 1, r), id(c, r + 1)]);
                elems.extend_from_slice(&[id(c + 1, r), id(c + 1, r + 1), id(c, r + 1)]);
            }
        }
        let elements = Array2::from_shape_vec((elems.len() / 3, 3), elems).unwrap();
        let boundary: Vec<usize> = (0..n_side * n_side)
            .filter(|&i| {
                let (c, r) = (i % n_side, i / n_side);
                c == 0 || r == 0 || c == n_side - 1 || r == n_side - 1
            })
            .collect();
        let corners = array![[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]];
        let roi = meshing::define_roi(corners.view(), false, &[], &[], Some((size, size)));
        let mesh = Mesh {
            nodes,
            elements,
            boundary,
            exclusions: vec![],
            mesh_order: 1,
            mask: roi.mask,
            f_img: Arc::clone(&ref_img),
            g_img: Arc::clone(&tar_img),
            time: None,
            time_source: TimeSource::None,
            solution: None,
            still: None,
        };

        // Seeded well clear of the y=50 discontinuity (circle(6) subsets,
        // seeded at y=30 -- 20px clearance) so the seed node itself isn't
        // straddling the band.
        let seed = SeedConfig { coord: [50.0, 30.0], warp: vec![0.0; 6], tolerance: 0.5 };
        let cfg = SolveConfig {
            tolerance: 0.5, subset_order: 1, override_active: true, ..Default::default()
        };
        (mesh, seed, cfg)
    }

    // -----------------------------------------------------------------------
    // Preconditioning::LayerRg (geopyv_dev_fresh/layer_rg_plan.md)
    // -----------------------------------------------------------------------

    fn layer_rg_cfg(base: &SolveConfig, batch_factor: usize, eps: f64) -> SolveConfig {
        SolveConfig {
            preconditioning: Preconditioning::LayerRg,
            layer_rg: LayerRgConfig { batch_factor, root_rel_eps: eps, max_workers: None },
            ..base.clone()
        }
    }

    /// §6.3: `LayerRg` agrees with serial `Rg` within Tier C, and matches
    /// exactly on `quality_ok` and the seed node.
    #[test]
    fn layer_rg_matches_serial_rg_within_tier_c() {
        let (mut m_rg, seed, cfg) = make_test_mesh_and_solve_inputs();
        let local_mask = LocalMask::circle(8).unwrap();
        m_rg.solve(&local_mask, &seed, &cfg, None).unwrap();
        let a = Arc::clone(m_rg.solution().unwrap());

        let (mut m_lrg, seed2, cfg2) = make_test_mesh_and_solve_inputs();
        m_lrg.solve(&local_mask, &seed2, &layer_rg_cfg(&cfg2, 4, 0.02), None).unwrap();
        let b = Arc::clone(m_lrg.solution().unwrap());

        assert_eq!(a.seed_node, b.seed_node, "seed node must match");
        assert_eq!(a.p.nrows(), b.p.nrows());
        for i in 0..a.p.nrows() {
            let du = (a.displacements[[i, 0]] - b.displacements[[i, 0]]).abs();
            let dv = (a.displacements[[i, 1]] - b.displacements[[i, 1]]).abs();
            let scale = a.displacements[[i, 0]].abs().max(a.displacements[[i, 1]].abs()).max(1.0);
            assert!(
                du <= 1e-5 * scale && dv <= 1e-5 * scale,
                "node {i}: displacement diff ({du}, {dv}) exceeds Tier C"
            );
            let dc = (a.c_zncc[i] - b.c_zncc[i]).abs();
            assert!(dc <= 1e-5, "node {i}: c_zncc diff {dc} exceeds Tier C");
        }
    }

    /// §6.6: `LayerRg` is run-to-run deterministic (byte-identical p/c_zncc).
    #[test]
    fn layer_rg_is_deterministic() {
        let local_mask = LocalMask::circle(8).unwrap();
        let mut prev: Option<Arc<MeshSolution>> = None;
        for _ in 0..3 {
            let (mut m, seed, cfg) = make_test_mesh_and_solve_inputs();
            m.solve(&local_mask, &seed, &layer_rg_cfg(&cfg, 4, 0.02), None).unwrap();
            let sol = Arc::clone(m.solution().unwrap());
            if let Some(p) = &prev {
                assert_eq!(p.p, sol.p, "LayerRg p not run-to-run identical");
                assert_eq!(p.c_zncc, sol.c_zncc, "LayerRg c_zncc not run-to-run identical");
                assert_eq!(p.iterations, sol.iterations);
            }
            prev = Some(sol);
        }
    }

    /// §6.5: `batch_factor = 1` (one root, minimal batching) is still a
    /// valid RG order — within Tier C of serial `Rg`.
    #[test]
    fn layer_rg_batch_factor_1_matches_rg() {
        let local_mask = LocalMask::circle(8).unwrap();
        let (mut m_rg, seed, cfg) = make_test_mesh_and_solve_inputs();
        m_rg.solve(&local_mask, &seed, &cfg, None).unwrap();
        let a = Arc::clone(m_rg.solution().unwrap());

        let (mut m_lrg, seed2, cfg2) = make_test_mesh_and_solve_inputs();
        m_lrg.solve(&local_mask, &seed2, &layer_rg_cfg(&cfg2, 1, 0.0), None).unwrap();
        let b = Arc::clone(m_lrg.solution().unwrap());

        for i in 0..a.p.nrows() {
            assert!((a.c_zncc[i] - b.c_zncc[i]).abs() <= 1e-5, "node {i} c_zncc");
        }
    }

    /// §3.3: `greedy_graph_colour` produces a proper colouring — no edge
    /// joins two same-colour candidates — and is deterministic.
    #[test]
    fn greedy_graph_colour_is_proper_and_deterministic() {
        let (mesh, _seed, _cfg) = make_test_mesh_and_solve_inputs();
        let adj = Adjacency::build(&mesh.elements, mesh.mesh_order, mesh.nodes.nrows());
        // Use every node as a candidate — the densest possible conflict graph.
        let candidates: Vec<usize> = (0..mesh.nodes.nrows()).collect();
        let c1 = greedy_graph_colour(&adj, &candidates);
        let c2 = greedy_graph_colour(&adj, &candidates);
        assert_eq!(c1, c2, "colouring must be deterministic");
        for &i in &candidates {
            for &nb in adj.get(i, true) {
                assert_ne!(
                    c1[i], c1[nb],
                    "edge ({i},{nb}) joins two colour-{} nodes", c1[i]
                );
            }
        }
    }
}
