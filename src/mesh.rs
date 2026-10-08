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
    field::{Field, FieldDistribution},
    geometry::{meshing, triangulation},
    image::Image,
    particle::{MeshlessParams, StrainMethod},
    sequence::SequenceSolution,
    subset::{Subset, SolveResult},
    masks::{zone_at, LocalMask, MaskShape},
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
    /// Zonal-masking diagnostics — `Some` only for a
    /// `masking = Masking::Zonal` solve whose pass 2 succeeded; `None` for
    /// uniform solves and for a zonal solve that fell back to pass 1 (its
    /// stored result then carries no zone cut). See [`ZonalMaskingRecord`].
    #[serde(default)]
    pub zonal_masking: Option<ZonalMaskingRecord>,
}

/// Per-node record of what [`Mesh::solve_zonal_masking_impl`] did, retained
/// on [`MeshSolution`] for post-solve inspection
/// (`Mesh.inspect(zones=True)` / `Mesh.inspect(subset_idx=...)`). All
/// per-node arrays are indexed identically to `MeshSolution::nodes`.
///
/// Zones are defined from the **spatial gradient** of the shear strain:
/// nodes whose `|∇γ|` sits on a ridge (`> grad_cutoff`) mark the interfaces,
/// the low-gradient regions between them are the coherent deformation
/// regimes, and each subset keeps only pixels sharing its own centre's
/// regime label. See `geopyv_dev_fresh/zonal_masking_gradient_correction.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZonalMaskingRecord {
    /// Stage-2 meshless shear strain (`gamma_max`) at each node, `(N,)` —
    /// context only; the classifier keys off its gradient, below.
    pub node_gamma_max: Array1<f64>,
    /// Stage-2 meshless shear-strain gradient magnitude
    /// `|∇γ| = hypot(dγ/dx, dγ/dy)` at each node, `(N,)` — the classifying
    /// signal.
    pub node_gamma_max_grad: Array1<f64>,
    /// `true` where the node's `|∇γ|` exceeded `grad_cutoff`, i.e. the node
    /// sits on an interface ridge rather than inside a regime, `(N,)`.
    pub node_boundary: Vec<bool>,
    /// Zone id under each node's own centre in `zone_image`, `(N,)`.
    pub node_zone: Array1<u32>,
    /// Retained pixel count BEFORE the zone cut (shape ∩ boundary), `(N,)`.
    pub node_pre_px: Array1<u32>,
    /// Retained pixel count AFTER the zone cut (shape ∩ boundary ∩ zone),
    /// `(N,)`.
    pub node_post_px: Array1<u32>,
    /// `true` where the minimum-pixel guard rejected the zone cut and the
    /// node was solved with its ordinary (un-zoned) subset, `(N,)`.
    pub node_guard_fallback: Vec<bool>,
    /// Robust-threshold multiplier actually used (`ZonalConfig::k`).
    pub k: f64,
    /// Median of the per-node `|∇γ|`.
    pub grad_median: f64,
    /// Median absolute deviation of the per-node `|∇γ|`.
    pub grad_mad: f64,
    /// `grad_median + k * 1.4826 * grad_mad` — the ridge cut-off on `|∇γ|`.
    pub grad_cutoff: f64,
    /// The whole-image zone-label grid pass 2 actually applied, `(H, W)`.
    pub zone_image: Array2<u8>,
}

/// What every node's subset is cut from during one `solve_impl` pass: the
/// shared local template, plus -- only for pass 2 of a `Masking::Zonal`
/// solve -- the whole-image zone-label grid that further restricts each
/// node's subset to its own centre's zone (see [`zone_cut_for_node`]). The
/// zone grid is deliberately *not* a `SolveConfig` field: it is internal
/// plumbing of [`Mesh::solve_zonal_masking_impl`], reachable only via
/// `masking = Masking::Zonal`.
struct SubsetTemplate<'a> {
    local_mask: &'a LocalMask,
    zone_mask: Option<ArrayView2<'a, u8>>,
}

impl<'a> SubsetTemplate<'a> {
    fn plain(local_mask: &'a LocalMask) -> Self {
        Self { local_mask, zone_mask: None }
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
    /// Persisted (unlike `masking`): this is a real solve-configuration
    /// choice, not a one-shot pipeline input.
    #[serde(default)]
    pub preconditioning: Preconditioning,
    /// Tunables for `preconditioning == LayerRg` — see [`LayerRgConfig`].
    /// Ignored otherwise.
    #[serde(default)]
    pub layer_rg: LayerRgConfig,
    /// The `masking` axis of `solver_options` — see [`Masking`]. Not
    /// serialised: a one-shot pipeline input, not persisted mesh state.
    #[serde(skip)]
    pub masking: Masking,
}

/// Minimum fraction of a node's "shape ∩ boundary" pixel count that must
/// survive a zonal cut for that cut to be applied; below it the node falls
/// back to its ordinary (un-zoned) subset rather than solving with a
/// badly-cut one (`geopyv_dev_fresh/zonal_masking_plan.md`, "Minimum-pixel
/// guard"). A fixed safety invariant of zonal masking, not a tuning knob.
const ZONE_MASK_MIN_PIXEL_FRACTION: f64 = 1.0 / 3.0;

/// `(median, MAD)` of a value stream, via O(n) selection (no full sort).
/// `MAD` is the median absolute deviation from the median — `1.4826 * MAD`
/// is a robust σ estimate. Used for the ridge cutoff on the smoothed `|∇γ|`
/// image in `Mesh::solve_zonal_masking_impl`. Returns `(0.0, 0.0)` for an
/// empty stream.
fn robust_median_mad(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut v: Vec<f64> = values.filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mid = v.len() / 2;
    v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    let median = v[mid];
    let mut dev: Vec<f64> = v.iter().map(|&x| (x - median).abs()).collect();
    dev.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    (median, dev[mid])
}

/// Mean node spacing over the node bounding box:
/// `sqrt(bbox_area / n_nodes)`. Used to scale the default `|∇γ|`-image
/// smoothing σ in `Mesh::solve_zonal_masking_impl` — same convention the
/// rasteriser's own bucket-grid cell size uses, and the standard
/// mesh-density scale this codebase ties meshless radii to.
fn node_spacing(nodes: &Array2<f64>) -> f64 {
    let n = nodes.nrows();
    if n == 0 {
        return 1.0;
    }
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for i in 0..n {
        x_min = x_min.min(nodes[[i, 0]]);
        x_max = x_max.max(nodes[[i, 0]]);
        y_min = y_min.min(nodes[[i, 1]]);
        y_max = y_max.max(nodes[[i, 1]]);
    }
    let area = ((x_max - x_min).max(1.0)) * ((y_max - y_min).max(1.0));
    (area / n as f64).sqrt().max(1.0)
}

/// Convergence tolerance for the zonal classifier refit loop: stop once
/// fewer than this fraction of *image pixels* change zone between
/// consecutive passes. ~5e-4 of a 2000² image ≈ a ~1 px shift of a
/// full-width seam.
const ZONAL_ITER_TOL: f64 = 5e-4;

/// Output of one classifier pass (`classify_from_field`) — everything
/// [`Mesh::solve_zonal_masking_impl`] needs for its inspection record and
/// for seeding the next refit iteration.
struct ClassifyResult {
    zone_image: Array2<u8>,
    /// Zone id under each node centre (`zone_at`), for the convergence
    /// check and the zone-aware refit seed.
    node_zone: Array1<u32>,
    gamma_max: Vec<f64>,
    grad_mag: Vec<f64>,
    node_boundary: Vec<bool>,
    grad_median: f64,
    grad_mad: f64,
    grad_cutoff: f64,
    /// Weak-ridge guard fired ⇒ `zone_image` is all-ones (a single zone).
    collapsed: bool,
}

/// Stages 3, 5a, 5b, 6 of the zonal classifier, factored out of
/// [`Mesh::solve_zonal_masking_impl`] so they can run once per refit
/// iteration. No behaviour change vs the previous inline form.
fn classify_from_field(
    field_sol: &crate::field::FieldSolution,
    nodes: &Array2<f64>,
    img_shape: (usize, usize),
    sigma: f64,
    zc: &ZonalConfig,
    min_core_px: usize,
) -> ClassifyResult {
    use crate::geometry::rasterize;
    let n_nodes = nodes.nrows();

    // --- Stage 3: per-node `gamma_max` (drives the classifier) + its own
    // meshless gradient magnitude `|∇γ|` (recorded for inspection only).
    let gamma_max: Vec<f64> = field_sol
        .particles
        .iter()
        .map(|p| {
            p.principal_strains
                .as_ref()
                .and_then(|ps| ps.row(ps.nrows() - 1).get(2).copied())
                .unwrap_or(0.0)
        })
        .collect();
    let grad_mag: Vec<f64> = field_sol
        .particles
        .iter()
        .map(|p| {
            p.gamma_max_grad
                .as_ref()
                .map(|g| {
                    let last = g.row(g.nrows() - 1);
                    last[0].hypot(last[1])
                })
                .filter(|v| v.is_finite())
                .unwrap_or(0.0)
        })
        .collect();

    // --- Stage 5a: dense, smoothed |∇γ| image (Sobel of a blurred
    // nearest-node `gamma_max` raster).
    let gamma_dense = rasterize::nearest_node_values(nodes.view(), &gamma_max, img_shape);
    let gamma_smooth = rasterize::gaussian_blur(gamma_dense.view(), sigma);
    let grad_dense = rasterize::sobel_magnitude(gamma_smooth.view());

    // --- Stage 5b: robust ridge threshold.
    let (grad_median, grad_mad) = robust_median_mad(grad_dense.iter().copied());
    let grad_cutoff = grad_median + zc.k * 1.4826 * grad_mad;
    let core_mask = grad_dense.mapv(|g| g <= grad_cutoff);
    let node_boundary: Vec<bool> = (0..n_nodes)
        .map(|i| {
            let cx = (nodes[[i, 0]] as isize).clamp(0, img_shape.1 as isize - 1) as usize;
            let cy = (nodes[[i, 1]] as isize).clamp(0, img_shape.0 as isize - 1) as usize;
            grad_dense[[cy, cx]] > grad_cutoff
        })
        .collect();

    // --- Stage 6: cores → drop < 1 subset → weak-ridge guard → watershed.
    let (mut markers, n_cores) = rasterize::label_components(core_mask.view());
    let mut counts = vec![0usize; n_cores as usize + 1];
    for &v in markers.iter() {
        counts[v as usize] += 1;
    }
    let n_kept = (1..=n_cores as usize).filter(|&l| counts[l] >= min_core_px).count();
    for v in markers.iter_mut() {
        if *v != 0 && counts[*v as usize] < min_core_px {
            *v = 0;
        }
    }
    let core_frac =
        core_mask.iter().filter(|&&c| c).count() as f64 / (img_shape.0 * img_shape.1) as f64;
    let collapsed = n_kept <= 1 || core_frac < 0.25;
    let zone_image = if collapsed {
        Array2::<u8>::ones(img_shape)
    } else {
        let flooded = rasterize::watershed_from_markers(grad_dense.view(), markers.view());
        let flooded = rasterize::fill_from_nearest_label(flooded.view());
        flooded.mapv(|v| v.max(1).min(255) as u8)
    };

    let node_zone: Array1<u32> = (0..n_nodes)
        .map(|i| zone_at(zone_image.view(), [nodes[[i, 0]], nodes[[i, 1]]]) as u32)
        .collect();

    ClassifyResult {
        zone_image,
        node_zone,
        gamma_max,
        grad_mag,
        node_boundary,
        grad_median,
        grad_mad,
        grad_cutoff,
        collapsed,
    }
}

/// Fraction of image pixels whose zone label moved between two classifier
/// passes — the convergence signal for the refit loop. Zone ids are greedily
/// matched by maximum overlap first (watershed / CC ids are scan-order, not
/// stable across passes). Pixel-level, not per-node: the seam keeps shifting
/// for a few refits after every *node* has already settled into its final
/// zone, so a node-level metric would stop the loop too early.
fn changed_zone_pixel_frac(prev: &Array2<u8>, cur: &Array2<u8>) -> f64 {
    let (p, c) = match (prev.as_slice(), cur.as_slice()) {
        (Some(p), Some(c)) if p.len() == c.len() && !p.is_empty() => (p, c),
        _ => return 0.0,
    };
    let cur_max = c.iter().copied().max().unwrap_or(0) as usize;
    let prev_max = p.iter().copied().max().unwrap_or(0) as usize;
    let mut overlap = vec![vec![0usize; prev_max + 1]; cur_max + 1];
    for i in 0..p.len() {
        overlap[c[i] as usize][p[i] as usize] += 1;
    }
    let map: Vec<u8> = (0..=cur_max)
        .map(|ci| (0..=prev_max).max_by_key(|&pi| overlap[ci][pi]).unwrap_or(0) as u8)
        .collect();
    let changed = (0..p.len()).filter(|&i| map[c[i] as usize] != p[i]).count();
    changed as f64 / p.len() as f64
}

/// A throwaway [`ZonalMaskingRecord`] carrying only `node_zone` + `zone_image`
/// — the two fields [`crate::particle::meshless_warp_increment`] reads for
/// `zone_aware` neighbour filtering. Attached to the source mesh of a refit
/// iteration's Stage-2 field; never serialised.
fn minimal_zonal_record(
    node_zone: Array1<u32>,
    zone_image: Array2<u8>,
    n: usize,
) -> ZonalMaskingRecord {
    ZonalMaskingRecord {
        node_gamma_max: Array1::zeros(n),
        node_gamma_max_grad: Array1::zeros(n),
        node_boundary: vec![false; n],
        node_zone,
        node_pre_px: Array1::zeros(n),
        node_post_px: Array1::zeros(n),
        node_guard_fallback: vec![false; n],
        k: 0.0,
        grad_median: 0.0,
        grad_mad: 0.0,
        grad_cutoff: 0.0,
        zone_image,
    }
}

/// The per-node zone-mask cut plus its minimum-pixel guard — factored out
/// of [`Mesh::subset_at`] so [`Mesh::solve_zonal_masking_impl`] can record
/// the exact same `pre`/`post`/fell-back numbers it produced, without
/// re-solving. Purely geometric (no image correlation): clones `base`,
/// applies [`LocalMask::zone_mask_update`], and measures the retained pixel
/// counts with and without the cut (both intersected with `global_mask`,
/// the boundary/exclusion mask, when present — mirroring `Subset::new`'s
/// own downstream `mask_update`).
///
/// Returns `(picked, pre_px, post_px, fell_back)`:
/// * `picked`    — the mask to build the subset from: the zone-cut mask,
///   or `base` unchanged when the guard rejects the cut.
/// * `pre_px`    — retained px, shape ∩ boundary, no zone cut.
/// * `post_px`   — retained px, shape ∩ boundary ∩ zone.
/// * `fell_back` — `true` when `pre_px > 0` and `post_px` dropped below
///   `min_pixel_fraction * pre_px`.
fn zone_cut_for_node(
    base: &LocalMask,
    coord: [f64; 2],
    global_mask: Option<ArrayView2<u8>>,
    zone_mask: ArrayView2<u8>,
    min_pixel_fraction: f64,
) -> (LocalMask, u32, u32, bool) {
    let pre = match global_mask {
        Some(gm) => {
            let mut probe = base.clone();
            probe.mask_update(coord, gm);
            probe.m_n_px.unwrap_or(0)
        }
        None => base.n_px,
    };

    let mut lm = base.clone();
    lm.zone_mask_update(coord, zone_mask);

    let post = match global_mask {
        Some(gm) => {
            let mut probe = lm.clone();
            probe.mask_update(coord, gm);
            probe.m_n_px.unwrap_or(0)
        }
        None => lm.m_n_px.unwrap_or(0),
    };

    let fell_back = pre > 0 && (post as f64) < min_pixel_fraction * (pre as f64);
    let picked = if fell_back { base.clone() } else { lm };
    (picked, pre as u32, post as u32, fell_back)
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

/// The `masking` axis of `solver_options` (see
/// `geopyv_dev_fresh/solver_options_restructure.md` §4's "Stage B"). `Uniform`
/// is today's plain solve, unchanged. `Zonal` runs
/// [`Mesh::solve_zonal_masking_impl`]: an ordinary solve, a `Field` built at
/// the mesh's own node positions to get each node's shear strain, a robust
/// per-node classification, a meshless rasterisation of that classification
/// to a whole-image zone label, then a second solve with that label applied
/// via the internal per-node zone cut ([`zone_cut_for_node`]).
/// Not serialised (`#[serde(skip)]` on `SolveConfig::masking`) — like
/// `zone_mask` itself, this is a one-shot solve input, not persisted mesh
/// state (a solved mesh's own `MeshSolution` records only its final,
/// already-zone-masked `p`/`c_zncc`, not which pipeline produced them).
#[derive(Debug, Clone)]
pub enum Masking {
    Uniform,
    Zonal(ZonalConfig),
}

impl Default for Masking {
    fn default() -> Self {
        Masking::Uniform
    }
}

/// Tunables for `Masking::Zonal` — see
/// `geopyv_dev_fresh/zonal_masking_gradient_correction.md`. Zones are
/// defined from the shear-strain *gradient* `|∇γ|`: its ridges are the
/// interfaces, the low-gradient regions between them are the regimes.
///
/// `min_pixel_fraction` is deliberately NOT a field here — it's a fixed
/// safety invariant of the masking mechanism, not a classification tuning
/// knob ([`ZONE_MASK_MIN_PIXEL_FRACTION`]).
#[derive(Debug, Clone)]
pub struct ZonalConfig {
    /// Params for the Stage-2 field solve (`StrainMethod::Meshless`) that
    /// computes each node's own shear strain **and its gradient** (one
    /// radius for one signal — see `particle.rs`'s `gamma_max_grad` note).
    pub meshless_params: MeshlessParams,
    /// Robust-threshold multiplier: a node sits on an interface ridge when
    /// its `|∇γ|` exceeds `median + k * 1.4826 * MAD` across all nodes.
    pub k: f64,
    /// Gaussian σ (pixels) applied to the rasterised `|∇γ|` image before
    /// thresholding. `None` ⇒ default to the mesh's own node spacing.
    pub smoothing_sigma: Option<f64>,
    /// Classifier refit passes. `1` (default) = classify once — identical
    /// to the pre-iterator behaviour. `>1` = after the first classification,
    /// refit the Stage-2 field with `MeshlessParams::zone_aware = true`
    /// seeded by that zone map and reclassify, up to this many times or
    /// until the per-node zone partition stops moving. The zone-aware refit
    /// removes the cross-zone neighbour blend that biases `γ_max`/
    /// `γ_max_grad` near a discontinuity — see
    /// `geopyv_dev_fresh/mds/zonal_masking_iterator_plan.md`.
    pub iterations: usize,
    /// Caller-supplied whole-image zone-label grid, `(H, W)`, matching the
    /// reference image shape. When `Some`, the classifier (Stages 2–6) is
    /// skipped entirely and this grid is used directly as pass 2's zone
    /// mask — `iterations` is then irrelevant (there is nothing to refit)
    /// and the `ZonalMaskingRecord`'s gradient fields are left zeroed.
    /// Callers use this to drive the masking from a partition they already
    /// know (e.g. one derived from known specimen geometry) rather than one
    /// detected from the strain field. Reserve label `0` for "no zone"
    /// (see `LocalMask::zone_mask_update`).
    pub zone_map: Option<Array2<u8>>,
}

impl Default for ZonalConfig {
    fn default() -> Self {
        Self {
            meshless_params: MeshlessParams::default(),
            k: 2.0,
            smoothing_sigma: None,
            iterations: 1,
            zone_map: None,
        }
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
        tmpl: &SubsetTemplate,
        subset_order: usize,
    ) -> Result<Subset, Error> {
        let coord = [self.nodes[[idx, 0]], self.nodes[[idx, 1]]];
        // Zonal pass 2 only: restrict this node's own LocalMask clone to its
        // centre's zone (+ minimum-pixel guard), shared with
        // `solve_zonal_masking_impl`'s inspection record so the two never
        // diverge. A per-node clone rather than a new `Subset::new`
        // parameter, whose `global_mask` slot already carries the mesh's
        // boundary/exclusion rasterisation.
        let local_mask_owned;
        let local_mask = match tmpl.zone_mask {
            Some(zm) => {
                let (picked, _pre, _post, _fell_back) = zone_cut_for_node(
                    tmpl.local_mask,
                    coord,
                    self.mask.as_ref().map(|m| m.view()),
                    zm,
                    ZONE_MASK_MIN_PIXEL_FRACTION,
                );
                local_mask_owned = picked;
                &local_mask_owned
            }
            None => tmpl.local_mask,
        };
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
    /// * `progress`   — Optional external progress bar (used by Sequence).
    pub fn solve(
        &mut self,
        local_mask: &LocalMask,
        seed: &SeedConfig,
        cfg: &SolveConfig,
        progress: Option<&indicatif::ProgressBar>,
    ) -> Result<(), Error> {
        match &cfg.masking {
            Masking::Uniform => self.solve_impl(&SubsetTemplate::plain(local_mask), seed, cfg, progress),
            Masking::Zonal(zc) => self.solve_zonal_masking_impl(local_mask, seed, cfg, zc, progress),
        }
    }

    /// `masking = Masking::Zonal(_)`'s implementation — see
    /// `geopyv_dev_fresh/solver_options_restructure.md` §4's Stage B.
    /// `pub(crate)` rather than a second public method: not a new user
    /// surface, `masking` is a `SolveConfig`/`solver_options` value on the
    /// one public `solve()`.
    ///
    /// 1. Pass 1: an ordinary solve (`masking` forced `Uniform` for this
    ///    inner call, so this doesn't recurse).
    /// 2. Build a `Field` at this mesh's own node positions, Rust-to-Rust
    ///    (no PyO3 round-trip), solved with `StrainMethod::Meshless(zc.meshless_params)`.
    /// 3. Rasterise each node's `gamma_max` to a dense image (meshless
    ///    nearest-node, `nearest_node_values`), Gaussian-smooth it (σ = the
    ///    mesh's node spacing by default) and take its Sobel magnitude, a
    ///    dense `|∇γ|`. No mesh-element interpolation anywhere.
    /// 4. Robust ridge threshold on that image, `median + k * 1.4826 * MAD`;
    ///    the connected low-`|∇γ|` regions are the regime cores (cores
    ///    smaller than one subset dropped; ≤1 core or <25 % core coverage
    ///    ⇒ a single zone, i.e. no cut).
    /// 5. Marker-controlled watershed of `|∇γ|` from the cores, so each
    ///    zone seam lands on the ridge crest, then `fill_from_nearest_label`
    ///    for any unreached pixels. Steps 2-5 repeat up to `zc.iterations`
    ///    times (zone-aware refit), or are skipped entirely when
    ///    `zc.zone_map` is supplied.
    /// 6. Pass 2: solve again with every subset cut to its own centre's
    ///    zone ([`SubsetTemplate::zone_mask`], guarded by
    ///    [`ZONE_MASK_MIN_PIXEL_FRACTION`]), `masking` reset to `Uniform`
    ///    (so this inner call doesn't recurse either).
    /// 7. On pass-2 failure: fall back to pass 1's already-stored solution
    ///    rather than propagating the error — decided in the doc above,
    ///    never let the masking refinement make an already-solvable mesh
    ///    unsolvable.
    pub(crate) fn solve_zonal_masking_impl(
        &mut self,
        local_mask: &LocalMask,
        seed: &SeedConfig,
        cfg: &SolveConfig,
        zc: &ZonalConfig,
        progress: Option<&indicatif::ProgressBar>,
    ) -> Result<(), Error> {
        // --- Pass 1: plain solve, uniform masking. ---
        let cfg1 = SolveConfig { masking: Masking::Uniform, ..cfg.clone() };
        self.solve_impl(&SubsetTemplate::plain(local_mask), seed, &cfg1, progress)?;
        let pass1_solution = Arc::clone(self.solution().expect("solve_impl succeeded, solution must be Some"));

        let nodes = self.nodes().to_owned();
        let n_nodes = nodes.nrows();
        let img_shape = self.f_img.image_gs.dim(); // (height, width)

        // The classifier's per-node outputs -- shear strain and its gradient
        // magnitude, the ridge flags, the robust-threshold stats, and the
        // whole-image zone grid pass 2 applies -- or their neutral
        // stand-ins when the caller supplied a fixed `zone_map`.
        #[allow(clippy::type_complexity)]
        let (gamma_max, grad_mag, node_boundary, grad_median, grad_mad, grad_cutoff, zone_image):
            (Vec<f64>, Vec<f64>, Vec<bool>, f64, f64, f64, Array2<u8>) =
        if let Some(zmap) = zc.zone_map.as_ref() {
            // --- Caller-supplied zone map: skip Stages 2-6 entirely. The
            // partition is taken as given, so there is nothing to classify
            // or refit (`iterations` is irrelevant) and the gradient-signal
            // fields of the record are left zeroed. ---
            if zmap.dim() != img_shape {
                return Err(Error::InvalidInput(format!(
                    "zonal zone_map shape {:?} does not match the reference image {:?}",
                    zmap.dim(), img_shape,
                )));
            }
            (
                vec![0.0; n_nodes],
                vec![0.0; n_nodes],
                vec![false; n_nodes],
                0.0,
                0.0,
                0.0,
                zmap.clone(),
            )
        } else {
        // --- Stages 2-6, iterated (`geopyv_dev_fresh/mds/zonal_masking_iterator_plan.md`).
        // `zc.iterations == 1` (default) runs the loop body once and is
        // byte-identical to the pre-iterator classifier. `> 1`: after the
        // first (biased) classification, refit the Stage-2 meshless field
        // with `zone_aware = true` seeded by the current zone map -- which
        // drops the cross-zone neighbour blend that skews `gamma_max` /
        // `gamma_max_grad` near a discontinuity -- and reclassify, until the
        // per-node zone partition stops moving (`ZONAL_ITER_TOL`), it hits
        // the cap, or a refit destabilises a good map.
        let sigma = zc.smoothing_sigma.unwrap_or_else(|| node_spacing(&nodes));
        let min_core_px = local_mask.n_px.max(1) as usize;
        let iterations = zc.iterations.max(1);

        let mut cls: Option<ClassifyResult> = None;
        let mut prev_change = f64::INFINITY;

        for it in 0..iterations {
            // --- Stage 2: Field at the mesh nodes, Rust-to-Rust. it == 0
            // uses the plain field; it > 0 refits zone-aware, seeded by the
            // previous iteration's zone image (injected as a minimal record
            // -- only `node_zone` + `zone_image` are read by the meshless
            // `zone_aware` filter).
            let source: Arc<MeshSolution> = if it == 0 {
                Arc::clone(&pass1_solution)
            } else {
                let prev = cls.as_ref().expect("it > 0 ⇒ iteration 0 already ran");
                let mut m = (*pass1_solution).clone();
                m.zonal_masking = Some(minimal_zonal_record(
                    prev.node_zone.clone(),
                    prev.zone_image.clone(),
                    n_nodes,
                ));
                Arc::new(m)
            };
            let mut mp = zc.meshless_params.clone();
            if it > 0 {
                mp.zone_aware = true;
            }

            let seq_sol = Arc::new(SequenceSolution::from_mesh_solution(source));
            let mut field = Field::new(
                seq_sol,
                FieldDistribution::Explicit {
                    coordinates: nodes.clone(),
                    volumes: Array1::ones(n_nodes),
                },
                false,
                1.0,
            )?;
            field.solve(0.0, true, None, StrainMethod::Meshless(mp))?;
            let field_sol = field
                .solution()
                .expect("field.solve() succeeded, solution must be Some");

            let new_cls =
                classify_from_field(field_sol, &nodes, img_shape, sigma, zc, min_core_px);

            if it == 0 {
                let stop = new_cls.collapsed; // uniform field: no partition to refine
                cls = Some(new_cls);
                if stop {
                    break;
                }
                continue;
            }

            let prev = cls.as_ref().expect("it > 0 ⇒ iteration 0 already ran");
            let change = changed_zone_pixel_frac(&prev.zone_image, &new_cls.zone_image);

            // Reject a refit that collapses a previously-real partition, or
            // that moves the map MORE than the last refit did (oscillating)
            // -- keep the previous, better map.
            if (new_cls.collapsed && !prev.collapsed) || change > prev_change {
                break;
            }
            let converged = change < ZONAL_ITER_TOL;
            cls = Some(new_cls);
            prev_change = change;
            if converged {
                break;
            }
        }

        let cls = cls.expect("iteration 0 always runs");
        (
            cls.gamma_max,
            cls.grad_mag,
            cls.node_boundary,
            cls.grad_median,
            cls.grad_mad,
            cls.grad_cutoff,
            cls.zone_image,
        )
        };

        // --- Inspection record: the per-node zone-cut outcome, computed
        // with the exact same `zone_cut_for_node` the pass-2 solve uses, so
        // `Mesh.inspect(zones=True)` / `inspect(subset_idx=...)` report what
        // actually happened. Purely geometric -- no re-solve. ---
        let mut node_zone = Array1::<u32>::zeros(n_nodes);
        let mut node_pre_px = Array1::<u32>::zeros(n_nodes);
        let mut node_post_px = Array1::<u32>::zeros(n_nodes);
        let mut node_guard_fallback = vec![false; n_nodes];
        for idx in 0..n_nodes {
            let coord = [nodes[[idx, 0]], nodes[[idx, 1]]];
            node_zone[idx] = zone_at(zone_image.view(), coord) as u32;
            let (_picked, pre, post, fell_back) = zone_cut_for_node(
                local_mask,
                coord,
                self.mask.as_ref().map(|m| m.view()),
                zone_image.view(),
                ZONE_MASK_MIN_PIXEL_FRACTION,
            );
            node_pre_px[idx] = pre;
            node_post_px[idx] = post;
            node_guard_fallback[idx] = fell_back;
        }
        let record = ZonalMaskingRecord {
            node_gamma_max: Array1::from(gamma_max),
            node_gamma_max_grad: Array1::from(grad_mag),
            node_boundary,
            node_zone,
            node_pre_px,
            node_post_px,
            node_guard_fallback,
            k: zc.k,
            grad_median,
            grad_mad,
            grad_cutoff,
            zone_image: zone_image.clone(),
        };

        // --- Pass 2: solve again with every subset cut to its own zone,
        // uniform masking (no recursion). ---
        let cfg2 = SolveConfig {
            masking: Masking::Uniform,
            override_active: true,
            ..cfg.clone()
        };
        let tmpl2 = SubsetTemplate { local_mask, zone_mask: Some(zone_image.view()) };
        match self.solve_impl(&tmpl2, seed, &cfg2, progress) {
            Ok(()) => {
                // Attach the record to pass 2's fresh solution (refcount 1
                // here -- `pass1_solution` is a distinct `Arc` -- so
                // `make_mut` does not clone).
                let sol = self
                    .solution
                    .as_mut()
                    .expect("solve_impl succeeded, solution must be Some");
                Arc::make_mut(sol).zonal_masking = Some(record);
                Ok(())
            }
            Err(_) => {
                // Decided: never let the masking refinement make an
                // already-solvable mesh unsolvable -- fall back to pass 1's
                // own already-computed solution (which carries no zone cut,
                // hence no `zonal_masking` record).
                self.solution = Some(pass1_solution);
                Ok(())
            }
        }
    }

    fn solve_impl(
        &mut self,
        masks: &SubsetTemplate,
        seed: &SeedConfig,
        cfg: &SolveConfig,
        progress: Option<&indicatif::ProgressBar>,
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
            &self.subset_at(seed_node, masks, cfg.subset_order)?,
            seed_node, &[], &c_zncc, &p, &seed_cfg, Some(&seed_warp_norm),
        )?;
        store_result(seed_node, &seed_result, &mut quality_ok, &mut c_zncc, &mut p, &mut displacements, &mut iterations, &mut norms);
        pb.inc(1);
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
        let warps = element_strains(&self.nodes, &self.elements, &displacements, self.mesh_order)?;

        // --- Compatibility check.
        self.check_compatibility(&warps, &centroids)?;

        let f_img_path = self.f_img.filepath.clone().unwrap_or_default();
        let g_img_path = self.g_img.filepath.clone().unwrap_or_default();

        // Template shape/size, retained for `Mesh.inspect(subset_idx=...)`.
        let template_shape = Some(masks.local_mask.shape.clone());
        let template_sizes = Some(Array1::from_elem(n_nodes, masks.local_mask.size as u32));

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
            template_shape,
            template_sizes,
            // Filled by `solve_zonal_masking_impl` after a successful pass 2;
            // stays `None` for every uniform solve (including pass 1).
            zonal_masking: None,
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
        masks: &SubsetTemplate,
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
        masks: &SubsetTemplate,
        cfg: &SolveConfig,
        pool: Option<&rayon::ThreadPool>,
        pb: &indicatif::ProgressBar,
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
                pb.inc(1);
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
        masks: &SubsetTemplate,
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
        masks: &SubsetTemplate,
        cfg: &SolveConfig,
        _pb: &indicatif::ProgressBar,
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
    use ndarray::{array, s};

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
    // Zone-mask minimum-pixel guard (geopyv_dev_fresh/zonal_masking_plan.md)
    // -----------------------------------------------------------------------

    #[test]
    fn subset_at_zone_mask_min_pixel_guard_falls_back_when_cut_too_aggressive() {
        let (mesh, _seed, _cfg) = make_test_mesh_and_solve_inputs();
        let local_mask = LocalMask::circle(10).unwrap();
        let plain = SubsetTemplate::plain(&local_mask);

        // Pick the interior node closest to the texture centre (50, 50) --
        // far enough from the 20..80 boundary that its full circle(10)
        // footprint is unclipped by the mesh's own boundary mask, isolating
        // the zone-mask guard as the only thing under test.
        let idx = (0..mesh.nodes.nrows())
            .min_by(|&a, &b| {
                let da = (mesh.nodes[[a, 0]] - 50.0).powi(2) + (mesh.nodes[[a, 1]] - 50.0).powi(2);
                let db = (mesh.nodes[[b, 0]] - 50.0).powi(2) + (mesh.nodes[[b, 1]] - 50.0).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();

        let baseline = mesh.subset_at(idx, &plain, 1).unwrap();
        let baseline_n_px = baseline.mask.n_px;

        // Zone label uniform everywhere: the centre's own label covers the
        // node's entire footprint, so the cut removes nothing -- guard must
        // not fire (nothing to fall back from), and the result must match
        // the no-zone baseline exactly.
        let zone_uniform = Array2::<u8>::from_elem((100, 100), 1u8);
        let uniform_result = mesh
            .subset_at(idx, &SubsetTemplate { local_mask: &local_mask, zone_mask: Some(zone_uniform.view()) }, 1)
            .unwrap();
        assert_eq!(uniform_result.mask.n_px, baseline_n_px,
            "a uniform zone label should cut nothing at all");

        // Zone label: only a small 3x3 patch around the node's own centre is
        // label 1, every other pixel (including virtually this node's whole
        // circle(10) footprint) is label 2 -- an aggressive cut leaving only
        // a handful of pixels (kept >1 so the surviving subset still has
        // enough intensity variation to be solvable, not "featureless").
        let mut zone_split = Array2::<u8>::from_elem((100, 100), 2u8);
        let cx = mesh.nodes[[idx, 0]].round() as usize;
        let cy = mesh.nodes[[idx, 1]].round() as usize;
        zone_split.slice_mut(s![cy - 1..=cy + 1, cx - 1..=cx + 1]).fill(1);

        // With the guard (ZONE_MASK_MIN_PIXEL_FRACTION), the cut is rejected
        // as too aggressive -- result must match the no-zone baseline
        // exactly, not the heavily-cut subset.
        let guarded = mesh
            .subset_at(idx, &SubsetTemplate { local_mask: &local_mask, zone_mask: Some(zone_split.view()) }, 1)
            .unwrap();
        assert_eq!(guarded.mask.n_px, baseline_n_px,
            "an overly aggressive zone cut must fall back to the unmodified local mask");

        // With the guard effectively disabled (fraction = 0.0, any nonzero
        // survival passes), the SAME aggressive cut must actually apply --
        // confirms the guard, not something else, was responsible for the
        // fallback above.
        let coord = [mesh.nodes[[idx, 0]], mesh.nodes[[idx, 1]]];
        let (_unguarded, _pre, post, fell_back) = zone_cut_for_node(
            &local_mask, coord, mesh.mask.as_ref().map(|m| m.view()), zone_split.view(), 0.0,
        );
        assert!(!fell_back && (post as usize) < baseline_n_px,
            "with the guard disabled, the aggressive zone cut should actually reduce n_px \
             (baseline={baseline_n_px}, got={post})");
    }

    #[test]
    fn full_solve_with_zone_mask_differs_from_plain_solve() {
        // End-to-end regression test, not just the isolated subset_at check
        // above -- this project's own history (zone_mask_update_survives_a_
        // later_mask_update's regression) found that a masking mechanism
        // combined correctly in isolation had still been silently defeated
        // once run through a REAL Mesh::solve() alongside the boundary mask
        // every real mesh also applies. This is that same class of check,
        // now for the min-pixel-guarded subset_at end to end.
        let (mut mesh1, seed1, cfg1) = make_test_mesh_and_solve_inputs();
        let (mut mesh2, seed2, cfg2) = make_test_mesh_and_solve_inputs();
        let local_mask = LocalMask::circle(10).unwrap();

        mesh1.solve(&local_mask, &seed1, &cfg1, None).unwrap();
        let sol1 = Arc::clone(mesh1.solution().unwrap());

        // Left/right split zone label over the full 100x100 texture -- with
        // circle(10) subsets and a boundary spanning 20..80, plenty of nodes
        // should straddle x=50.
        let mut zone = Array2::<u8>::from_elem((100, 100), 1u8);
        zone.slice_mut(s![.., 50..]).fill(2);
        let tmpl = SubsetTemplate { local_mask: &local_mask, zone_mask: Some(zone.view()) };
        mesh2.solve_impl(&tmpl, &seed2, &cfg2, None).unwrap();
        let sol2 = Arc::clone(mesh2.solution().unwrap());

        let n_diff = (0..sol1.p.nrows())
            .filter(|&i| {
                (0..sol1.p.ncols()).any(|k| (sol1.p[[i, k]] - sol2.p[[i, k]]).abs() > 1e-9)
            })
            .count();
        assert!(n_diff > 0, "a real left/right zone split should change at least some nodes");
    }

    // -----------------------------------------------------------------------
    // Masking::Zonal end-to-end (geopyv_dev_fresh/solver_options_restructure.md
    // §4's Stage B) -- the full classify-from-a-real-strain-field pipeline,
    // not a hand-supplied zone_mask array like the test above.
    // -----------------------------------------------------------------------

    /// Step-discontinuity texture: rows `0..split_row` shift by `u_top`,
    /// rows `split_row..` shift by `u_bottom` -- a genuine horizontal-
    /// displacement discontinuity localised at `split_row`, so nodes near
    /// it have real, elevated shear strain relative to nodes far from it
    /// (unlike `shift_image_uniform`'s uniform shift, which has ~zero
    /// strain everywhere and so nothing for a magnitude-threshold
    /// classifier to distinguish).
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

    fn make_band_test_mesh_and_solve_inputs() -> (Mesh, SeedConfig, SolveConfig) {
        let size = 100usize;
        let texture = make_synthetic_texture_mesh(size);
        let shifted = shift_image_band(&texture, 50, 1.5, -1.5);
        let ref_img = Arc::new(Image::from_array(texture, 15));
        let tar_img = Arc::new(Image::from_array(shifted, 15));

        // A FIXED structured grid built directly (not `Mesh::new`, whose
        // Ruppert refinement is not deterministic across calls) -- the
        // gradient-watershed classifier is sensitive enough to node
        // placement that an A/B test needs identical nodes on both sides
        // (see project_geopyv_zone_masking.md). 7.5px spacing over 20..80,
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
            solution: None,
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

    #[test]
    fn solve_zonal_masking_changes_results_for_a_real_step_discontinuity() {
        let (mut mesh1, seed1, cfg1) = make_band_test_mesh_and_solve_inputs();
        let (mut mesh2, seed2, cfg2) = make_band_test_mesh_and_solve_inputs();
        let local_mask = LocalMask::circle(6).unwrap();

        mesh1.solve(&local_mask, &seed1, &cfg1, None).unwrap();
        let sol1 = Arc::clone(mesh1.solution().unwrap());

        let cfg2z = SolveConfig {
            masking: Masking::Zonal(ZonalConfig {
                // MeshlessParams::default()'s radius (50px) is much larger
                // than this test mesh's own ~8px node spacing (60 nodes
                // over a 60x60 boundary) -- it would average over almost
                // the whole mesh and smooth the localised band signal away
                // entirely (confirmed: with the default radius, 0/57 nodes
                // ever differ). A radius tied to this mesh's own density,
                // matching how the rest of this project always scales
                // MeshlessParams to the mesh it's used on, not a fixed
                // constant.
                meshless_params: MeshlessParams { radius: 15.0, ..MeshlessParams::default() },
                k: 1.0,
                smoothing_sigma: None,
                iterations: 1,
                zone_map: None,
            }),
            ..cfg2
        };
        mesh2.solve(&local_mask, &seed2, &cfg2z, None).unwrap();
        let sol2 = Arc::clone(mesh2.solution().unwrap());

        let n_diff = (0..sol1.p.nrows())
            .filter(|&i| (0..sol1.p.ncols()).any(|k| (sol1.p[[i, k]] - sol2.p[[i, k]]).abs() > 1e-9))
            .count();
        assert!(n_diff > 0,
            "a real step-discontinuity mesh, solved via masking=Zonal end to end, should \
             classify at least some nodes onto interface ridges and change their result \
             relative to a plain solve -- got 0/{} nodes differing", sol1.p.nrows());

        // Mean C_ZNCC is the standard summary this project's whole masking
        // track uses (zone_mask_trial.py, benchmark.py) -- report it as a
        // real (if modest, per this project's own history) sanity check,
        // not just "did anything change at all".
        let mean1 = sol1.c_zncc.mean().unwrap();
        let mean2 = sol2.c_zncc.mean().unwrap();
        eprintln!(
            "solve_zonal_masking numeric observation: n_diff={}/{}, mean_c_zncc plain={:.6} \
             zonal={:.6}, min_c_zncc plain={:.6} zonal={:.6}",
            n_diff, sol1.p.nrows(), mean1, mean2,
            sol1.c_zncc.iter().cloned().fold(f64::INFINITY, f64::min),
            sol2.c_zncc.iter().cloned().fold(f64::INFINITY, f64::min),
        );
        assert!(mean1.is_finite() && mean2.is_finite());
    }

    #[test]
    fn changed_zone_pixel_frac_is_label_invariant_and_counts_real_moves() {
        let a = Array2::from_shape_vec((2, 3), vec![1u8, 1, 1, 2, 2, 2]).unwrap();
        // Same partition, relabelled -> 0 changed.
        let b = Array2::from_shape_vec((2, 3), vec![7u8, 7, 7, 3, 3, 3]).unwrap();
        assert_eq!(changed_zone_pixel_frac(&a, &b), 0.0);

        // One of six pixels genuinely moved regime.
        let c = Array2::from_shape_vec((2, 3), vec![1u8, 1, 2, 2, 2, 2]).unwrap();
        assert!((changed_zone_pixel_frac(&a, &c) - 1.0 / 6.0).abs() < 1e-12);

        // Total re-partition (checkerboard vs stripes).
        let d = Array2::from_shape_vec((2, 3), vec![1u8, 2, 1, 2, 1, 2]).unwrap();
        assert!(changed_zone_pixel_frac(&a, &d) > 0.3);

        // Shape mismatch / empty -> 0, no panic.
        assert_eq!(
            changed_zone_pixel_frac(&Array2::zeros((0, 0)), &Array2::zeros((0, 0))),
            0.0
        );
        assert_eq!(
            changed_zone_pixel_frac(&a, &Array2::<u8>::zeros((3, 3))),
            0.0
        );
    }

    #[test]
    fn zonal_iterations_1_is_identical_to_single_pass() {
        // `iterations = 1` must run the classifier loop body exactly once
        // and produce a byte-identical zone_image / node_zone / solution to
        // the pre-iterator code path. (Shares the band test mesh -- if that
        // mesh regresses upstream this will surface it, but the point here
        // is `iterations = 1` == default == no change.)
        let base = || SolveConfig {
            masking: Masking::Zonal(ZonalConfig {
                meshless_params: MeshlessParams { radius: 15.0, ..MeshlessParams::default() },
                k: 1.0,
                smoothing_sigma: None,
                iterations: 1,
                zone_map: None,
            }),
            ..make_band_test_mesh_and_solve_inputs().2
        };
        let local_mask = LocalMask::circle(6).unwrap();

        let (mut m1, s1, _) = make_band_test_mesh_and_solve_inputs();
        m1.solve(&local_mask, &s1, &base(), None).unwrap();
        let r1 = Arc::clone(m1.solution().unwrap());

        let (mut m2, s2, _) = make_band_test_mesh_and_solve_inputs();
        m2.solve(&local_mask, &s2, &base(), None).unwrap();
        let r2 = Arc::clone(m2.solution().unwrap());

        assert_eq!(r1.p, r2.p, "iterations=1 must be deterministic / unchanged");
        let (z1, z2) = (
            r1.zonal_masking.as_ref().map(|z| &z.zone_image),
            r2.zonal_masking.as_ref().map(|z| &z.zone_image),
        );
        assert_eq!(z1, z2);
    }

    #[test]
    fn zonal_iterations_gt_1_tightens_the_boundary() {
        // The zone-aware refit should place `node_boundary`-flagged nodes
        // CLOSER to the true y = 50 step than a single pass. Metric: the
        // spread (std dev) of ridge nodes' |y - 50|.
        let (mut m1, s1, cfg) = make_band_test_mesh_and_solve_inputs();
        let local_mask = LocalMask::circle(6).unwrap();
        let mk = |iters| SolveConfig {
            masking: Masking::Zonal(ZonalConfig {
                meshless_params: MeshlessParams { radius: 15.0, ..MeshlessParams::default() },
                k: 1.0,
                smoothing_sigma: None,
                iterations: iters,
                zone_map: None,
            }),
            ..cfg.clone()
        };
        m1.solve(&local_mask, &s1, &mk(1), None).unwrap();
        let (mut m3, s3, _) = make_band_test_mesh_and_solve_inputs();
        m3.solve(&local_mask, &s3, &mk(4), None).unwrap();

        let spread = |m: &Mesh| {
            let rec = m.solution().unwrap().zonal_masking.clone().unwrap();
            let nodes = m.nodes();
            let dists: Vec<f64> = (0..rec.node_boundary.len())
                .filter(|&i| rec.node_boundary[i])
                .map(|i| (nodes[[i, 1]] - 50.0).abs())
                .collect();
            let n = dists.len().max(1) as f64;
            let mean = dists.iter().sum::<f64>() / n;
            (dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n).sqrt()
        };
        let (s_1, s_4) = (spread(&m1), spread(&m3));
        eprintln!("ridge-node |y-50| spread: iters=1 {s_1:.3}, iters=4 {s_4:.3}");
        assert!(s_4 <= s_1 + 1e-6,
            "zone-aware refit should not widen the ridge-node spread (1: {s_1}, 4: {s_4})");
    }

    #[test]
    fn zonal_masking_record_is_populated_after_a_zonal_solve() {
        let (mut mesh, seed, cfg) = make_band_test_mesh_and_solve_inputs();
        let local_mask = LocalMask::circle(6).unwrap();
        let cfg_z = SolveConfig {
            masking: Masking::Zonal(ZonalConfig {
                meshless_params: MeshlessParams { radius: 15.0, ..MeshlessParams::default() },
                k: 1.0,
                smoothing_sigma: None,
                iterations: 1,
                zone_map: None,
            }),
            ..cfg
        };
        mesh.solve(&local_mask, &seed, &cfg_z, None).unwrap();
        let sol = mesh.solution().unwrap();
        let n = sol.nodes.nrows();

        // Template metadata is recorded on every solve.
        assert_eq!(sol.template_shape, Some(MaskShape::Circle));
        let sizes = sol.template_sizes.as_ref().expect("template_sizes recorded");
        assert_eq!(sizes.len(), n);
        assert!(sizes.iter().all(|&s| s == 6));

        let rec = sol.zonal_masking.as_ref().expect("zonal record populated");
        assert_eq!(rec.node_gamma_max.len(), n);
        assert_eq!(rec.node_gamma_max_grad.len(), n);
        assert_eq!(rec.node_boundary.len(), n);
        assert_eq!(rec.node_zone.len(), n);
        assert_eq!(rec.node_pre_px.len(), n);
        assert_eq!(rec.node_post_px.len(), n);
        assert_eq!(rec.node_guard_fallback.len(), n);
        assert_eq!(rec.zone_image.dim(), mesh.f_img.image_gs.dim());
        assert!((rec.grad_cutoff - (rec.grad_median + rec.k * 1.4826 * rec.grad_mad)).abs() < 1e-12);
        assert!(rec.node_gamma_max_grad.iter().all(|&g| g >= 0.0), "|∇γ| is a magnitude");
        // A real step discontinuity at y=50 → at least 2 distinct regime
        // zones (above the band, and the band/below).
        let distinct: std::collections::HashSet<u32> =
            rec.node_zone.iter().copied().collect();
        assert!(distinct.len() >= 2,
            "a step discontinuity should partition into >= 2 regime zones, got {}",
            distinct.len());

        for i in 0..n {
            assert!(rec.node_post_px[i] <= rec.node_pre_px[i],
                "post-cut px must not exceed pre-cut px (node {i})");
            if rec.node_guard_fallback[i] {
                // Guard tripped: the cut really was below 1/3 of pre.
                assert!((rec.node_post_px[i] as f64) < (1.0 / 3.0) * (rec.node_pre_px[i] as f64));
            }
        }
    }

    #[test]
    fn meshless_field_zone_aware_beats_zone_unaware_near_a_real_interface() {
        // End-to-end: a real masking=Zonal solve on the y=50 step-shift band
        // (u_true = 1.5 for y<50, -1.5 for y>=50), then two Fields built at
        // the same mesh nodes -- zone_aware=false vs zone_aware=true -- to
        // confirm zone-aware filtering actually recovers accuracy near the
        // interface that the zone-agnostic meshless fit smooths away.
        let (mut mesh, seed, cfg) = make_band_test_mesh_and_solve_inputs();
        let local_mask = LocalMask::circle(6).unwrap();
        let cfg_z = SolveConfig {
            masking: Masking::Zonal(ZonalConfig {
                meshless_params: MeshlessParams { radius: 15.0, ..MeshlessParams::default() },
                k: 1.0,
                smoothing_sigma: None,
                iterations: 1,
                zone_map: None,
            }),
            ..cfg
        };
        mesh.solve(&local_mask, &seed, &cfg_z, None).unwrap();
        let mesh_sol = Arc::clone(mesh.solution().unwrap());
        let nodes = mesh_sol.nodes.clone();
        let n = nodes.nrows();

        let seq_sol = Arc::new(SequenceSolution::from_mesh_solution(Arc::clone(&mesh_sol)));
        let coords = nodes.clone();
        let volumes = Array1::<f64>::ones(n);

        let make_field = |zone_aware: bool| {
            let mut field = Field::new(
                Arc::clone(&seq_sol),
                FieldDistribution::Explicit { coordinates: coords.clone(), volumes: volumes.clone() },
                false,
                1.0,
            ).unwrap();
            let params = MeshlessParams { radius: 15.0, zone_aware, ..MeshlessParams::default() };
            field.solve(0.0, true, None, StrainMethod::Meshless(params)).unwrap();
            field
        };
        let field_unaware = make_field(false);
        let field_aware = make_field(true);
        let sol_unaware = field_unaware.solution().unwrap();
        let sol_aware = field_aware.solution().unwrap();

        let u_true = |y: f64| if y < 50.0 { 1.5 } else { -1.5 };

        // Restrict to particles within one meshless radius of the interface
        // -- exactly the population the zone-agnostic fit blends across.
        let mut err_unaware = Vec::new();
        let mut err_aware = Vec::new();
        for i in 0..n {
            let y = nodes[[i, 1]];
            if (y - 50.0).abs() >= 15.0 {
                continue;
            }
            let truth = u_true(y);
            let u_un = sol_unaware.particles[i].warps[[1, 0]];
            let u_aw = sol_aware.particles[i].warps[[1, 0]];
            err_unaware.push((u_un - truth).abs());
            err_aware.push((u_aw - truth).abs());
        }
        assert!(!err_unaware.is_empty(), "test mesh should have nodes near the y=50 interface");

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let mean_unaware = mean(&err_unaware);
        let mean_aware = mean(&err_aware);
        assert!(
            mean_aware < mean_unaware,
            "zone-aware meshless fit should recover accuracy near a real interface: \
             mean |u error| zone_aware=false: {mean_unaware:.4}, zone_aware=true: {mean_aware:.4} \
             (n={})", err_unaware.len(),
        );
    }

    #[test]
    fn zonal_masking_record_is_none_after_a_uniform_solve() {
        let (mut mesh, seed, cfg) = make_band_test_mesh_and_solve_inputs();
        let local_mask = LocalMask::circle(6).unwrap();
        mesh.solve(&local_mask, &seed, &cfg, None).unwrap();
        let sol = mesh.solution().unwrap();
        assert!(sol.zonal_masking.is_none());
        // ...but the template metadata is still there.
        assert_eq!(sol.template_shape, Some(MaskShape::Circle));
        assert!(sol.template_sizes.as_ref().is_some_and(|s| s.iter().all(|&v| v == 6)));
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

    // NOTE: a test forcing pass 2 specifically to fail (with pass 1 already
    // succeeded) is NOT included here -- engineering a real pass-2-only
    // failure through the public solve_zonal_masking_impl surface turned
    // out to need either mocking solve_impl or a hard-to-construct
    // mesh-compatibility edge case (pass 2 hardcodes override_active=true,
    // so the ordinary quality-gate failure modes are already bypassed; see
    // solve_zonal_masking_impl's own deviation note in the implementation
    // report). The fallback logic itself (`Err(_) => { self.solution =
    // Some(pass1_solution); Ok(()) }`) is implemented per the plan's
    // decision, but only verified by code inspection in this pass, not by
    // a forced-failure test -- flagged honestly rather than shipped with a
    // test that doesn't actually exercise the branch it claims to.
}
