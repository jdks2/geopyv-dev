//! Still-pair neighbour screen: which mesh nodes go on to the seam tests.
//!
//! Port of the sandbox reference (`geopyv_dev_pub/sandbox_break_test`:
//! `static_noise/mesh_stage.py`, `static_noise/pairs.py::sandwich`,
//! `png_three_way/pair_noise.py::cross`, `static_noise/rounds.py::run_two_sided`).
//! Plan: `geopyv_dev_pub/mds/geopyv_dev_integration_plan.md` §1 step 2.
//!
//! # Mismatch
//!
//! For neighbours A, C (element-sharing) and both directions, A's affine warp
//! predicts C's displacement; the mismatch is
//! `e = u_C − u_A − ∇u_A (x_C − x_A)`.
//!
//! # Noise of the mismatch
//!
//! Linearised ICGN gives `Cov(p_A, p_C) = H_A⁻¹ M_AC H_C⁻¹` with the sandwich
//! `M_AC = Σ_x Σ_y J_A(x) r_A(x) w(x − y) r_C(y) J_C(y)'` (Gaussian kernel of
//! width h, the speckle correlation length), plus the 8-bit quantisation floor
//! `(1/6)/δ_f² Σ_{shared x} J_A J_C'` and the small-sample factor `N/(N − 6)`.
//! The residual `r` is taken from the **still pair** (frames 0–1: camera noise
//! only), never from the moving pair, whose residual hides its own under-fit.
//! With no still pair, `r` is replaced by its expectation for white noise
//! passed through the package's 5×5 load blur, at a level estimated from a
//! small patch around the seed node, where only rigid-body motion is assumed
//! ([`seed_patch_sigma`]); [`DEFAULT_NOISE_SIGMA`] if the patch is unusable.
//!
//! # Decision
//!
//! A pair fails if `|e|` exceeds the size floor `2 · 0.01 · s² / R²` (s: the
//! node's shortest neighbour distance, R: subset radius) **and** `e' S⁻¹ e`
//! exceeds the χ²₂ level at `(1 − 0.05)^(1/(2·degree))` of the node; both nodes
//! of a failing pair are flagged. Then **keep ⅓, add ⅔** until stable: a
//! flagged node stays flagged only if ≥ ⅓ of its neighbours are flagged, an
//! unflagged node is added if ≥ ⅔ are.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    image::Image,
    masks::LocalMask,
    mesh::MeshSolution,
    seam::{gauss_conv, lu_factor, lu_solve, shift_warp, solve_dense},
    subset::Subset,
    Error,
};

/// Noise (grey levels, 8-bit) assumed when there is neither a still pair nor
/// a usable seed patch.
pub const DEFAULT_NOISE_SIGMA: f64 = 2.0;
/// Share of still nodes the noise alone sends on.
const ALPHA: f64 = 0.05;
/// Under-fit (px) behind the size floor (×1).
const L_STAR: f64 = 0.01;
const KEEP_SHARE: f64 = 1.0 / 3.0;
const ADD_SHARE: f64 = 2.0 / 3.0;
const MAX_ROUNDS: usize = 30;

/// A solved still pair (frames 0 → 1) and its images.
#[derive(Clone)]
pub struct StillPair {
    pub solution: Arc<MeshSolution>,
    pub f_img: Arc<Image>,
    pub g_img: Arc<Image>,
}

/// Where the screen's noise came from.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NoiseSource {
    /// Residuals of the still pair (frames 0 → 1).
    StillPair,
    /// White noise through the load blur, `sigma` grey levels estimated from
    /// the seed patch.
    SeedPatch { sigma: f64 },
    /// White noise of [`DEFAULT_NOISE_SIGMA`] grey levels (seed patch unusable).
    Default { sigma: f64 },
}

/// Everything the screen needs from one solved moving mesh.
pub(crate) struct ScreenInput<'a> {
    pub nodes: &'a Array2<f64>,
    pub neighbours: &'a [Vec<usize>],
    /// Moving warps `(N, ≥ 6)`; only the order-1 part is used.
    pub p: &'a Array2<f64>,
    pub usable: &'a [bool],
    pub local_mask: &'a LocalMask,
    pub global_mask: Option<ArrayView2<'a, u8>>,
    pub f_img: &'a Arc<Image>,
    pub g_img: &'a Arc<Image>,
    pub h: f64,
    pub still: Option<&'a StillPair>,
    /// Seed node, whose patch gives the noise level without a still pair.
    pub seed: usize,
}

/// The screen's verdict per node.
pub(crate) struct ScreenOutput {
    /// Flagged by the tolerance (before the neighbour rule).
    pub flagged: Vec<bool>,
    /// Sent on to the seam tests (after keep ⅓ / add ⅔).
    pub sent: Vec<bool>,
    pub rounds: usize,
    /// Per node: median over its pair directions of the mismatch noise,
    /// `sqrt(trace(S)/2)` (px).
    pub noise_scale: Array1<f64>,
    /// Per node: largest frame 0–1 mismatch over its pairs (px); `None`
    /// without a still pair.
    pub still_mismatch: Option<Array1<f64>>,
    pub source: NoiseSource,
}

/// One node's subset as the screen sees it.
struct NodeData {
    /// Global pixel positions `round(coord + offset)`.
    gx: Vec<i64>,
    gy: Vec<i64>,
    /// `J = [f_x, f_y, f_x x, f_y x, f_x y, f_y y]`, `f = ∇f / δ_f`.
    j: Vec<[f64; 6]>,
    /// Score columns: `J r` (still residual) or `J √2σ/δ_f` (default model).
    s: Vec<[f64; 6]>,
    df: f64,
    /// Small-sample factor `N/(N − 6)`, `N = n_px / (2π h²)`.
    ssf: f64,
}

/// Per-node inverse Hessian, self sandwich and carried still warp.
#[derive(Clone)]
struct NodeStats {
    hi: [[f64; 6]; 6],
    meat: [[f64; 6]; 6],
    /// Frame 0–1 warp carried to this node (order 1), if a still pair.
    p_still: Option<[f64; 6]>,
}

/// The smoothing kernel applied to the score columns, as a 1-D separable
/// factor and its half-width.
struct Kernel {
    w1: Vec<f64>,
    rr: isize,
}

impl Kernel {
    /// Still residuals: the HAC kernel `exp(−t²/2h²)`, truncated at 3h.
    fn hac(h: f64) -> Self {
        let rr = (3.0 * h).ceil() as isize;
        Kernel { w1: (-rr..=rr).map(|t| (-((t * t) as f64) / (2.0 * h * h)).exp()).collect(), rr }
    }

    /// Default model: the HAC kernel times the autocorrelation of the 5×5
    /// load blur (σ = 1.1), `(g ⋆ g)(t)`; zero beyond |t| > 4.
    fn default_noise(h: f64) -> Self {
        let g = blur_taps();
        let auto = |t: isize| -> f64 {
            (-2..=2)
                .filter_map(|i: isize| {
                    let j = i + t;
                    (-2..=2).contains(&j).then(|| g[(i + 2) as usize] * g[(j + 2) as usize])
                })
                .sum()
        };
        let rr = 4;
        Kernel {
            w1: (-rr..=rr).map(|t| (-((t * t) as f64) / (2.0 * h * h)).exp() * auto(t)).collect(),
            rr,
        }
    }
}

fn build_node(
    inp: &ScreenInput,
    idx: usize,
    still_index: &Option<NearestIndex>,
    sigma: f64,
) -> Option<(NodeData, Option<[f64; 6]>)> {
    let coord = [inp.nodes[[idx, 0]], inp.nodes[[idx, 1]]];
    let mov = Subset::new(coord, inp.local_mask, inp.global_mask, Arc::clone(inp.f_img), Arc::clone(inp.g_img), 1).ok()?;
    let fc = mov.f_coords.as_ref()?;
    let grad = mov.grad_f.as_ref()?;
    let n = fc.nrows();
    let df = mov.delta_f;
    let mut gx = Vec::with_capacity(n);
    let mut gy = Vec::with_capacity(n);
    let mut j = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y) = (fc[[i, 0]] - coord[0], fc[[i, 1]] - coord[1]);
        gx.push(fc[[i, 0]].round() as i64);
        gy.push(fc[[i, 1]].round() as i64);
        let (fx, fy) = (grad[[i, 0]] / df, grad[[i, 1]] / df);
        j.push([fx, fy, fx * x, fy * x, fx * y, fy * y]);
    }
    let (s, p_still) = match (inp.still, still_index) {
        (Some(still), Some(index)) => {
            let k = index.nearest(coord);
            let sp = &still.solution.p;
            let pk: Vec<f64> = (0..6).map(|c| sp[[k, c]]).collect();
            let d = [coord[0] - still.solution.nodes[[k, 0]], coord[1] - still.solution.nodes[[k, 1]]];
            let pc = shift_warp(&pk, d);
            let st = Subset::new(coord, inp.local_mask, inp.global_mask, Arc::clone(&still.f_img), Arc::clone(&still.g_img), 1).ok()?;
            let r = st.residual_with(&pc).ok()?;
            if r.len() != n {
                return None;
            }
            let s = (0..n).map(|i| j[i].map(|v| v * r[i])).collect();
            (s, Some(std::array::from_fn(|c| pc[c])))
        }
        _ => {
            let scale = std::f64::consts::SQRT_2 * sigma / df;
            ((0..n).map(|i| j[i].map(|v| v * scale)).collect(), None)
        }
    };
    let ne = n as f64 / (2.0 * std::f64::consts::PI * inp.h * inp.h);
    Some((NodeData { gx, gy, j, s, df, ssf: ne / (ne - 6.0) }, p_still))
}

/// A node's score columns convolved with the kernel, and its raw `J`, on its
/// bounding box padded by the kernel half-width.
struct Grid {
    x0: i64,
    y0: i64,
    w: usize,
    h: usize,
    conv: [Vec<f64>; 6],
    jraw: [Vec<f64>; 6],
    occupied: Vec<bool>,
}

impl Grid {
    /// `padded`: extend the box by the kernel half-width so the smoothed
    /// columns are complete wherever a neighbour's pixels fall; a node's own
    /// self-sandwich needs only its own box (the input is zero outside it).
    fn new(d: &NodeData, k: &Kernel, padded: bool) -> Self {
        let pad = if padded { k.rr as i64 } else { 0 };
        let x0 = d.gx.iter().min().unwrap() - pad;
        let y0 = d.gy.iter().min().unwrap() - pad;
        let w = (d.gx.iter().max().unwrap() + pad - x0 + 1) as usize;
        let h = (d.gy.iter().max().unwrap() + pad - y0 + 1) as usize;
        let mut imgs: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; w * h]);
        let mut jraw: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; w * h]);
        let mut occupied = vec![false; w * h];
        for i in 0..d.gx.len() {
            let at = (d.gy[i] - y0) as usize * w + (d.gx[i] - x0) as usize;
            for c in 0..6 {
                imgs[c][at] = d.s[i][c];
                jraw[c][at] = d.j[i][c];
            }
            occupied[at] = true;
        }
        let conv = std::array::from_fn(|c| gauss_conv(&imgs[c], w, h, &k.w1, k.rr));
        Grid { x0, y0, w, h, conv, jraw, occupied }
    }

    fn at(&self, x: i64, y: i64) -> Option<usize> {
        let (cx, cy) = (x - self.x0, y - self.y0);
        (cx >= 0 && cy >= 0 && (cx as usize) < self.w && (cy as usize) < self.h).then(|| cy as usize * self.w + cx as usize)
    }

    /// `Σ_x S_A(x) conv(S_C)(x)'` and `Σ_{shared x} J_A(x) J_C(x)'` for A's pixels.
    fn cross(&self, a: &NodeData) -> ([[f64; 6]; 6], [[f64; 6]; 6]) {
        let mut m = [[0.0; 6]; 6];
        let mut fl = [[0.0; 6]; 6];
        for i in 0..a.gx.len() {
            let Some(at) = self.at(a.gx[i], a.gy[i]) else { continue };
            for p in 0..6 {
                let sp = a.s[i][p];
                for q in 0..6 {
                    m[p][q] += sp * self.conv[q][at];
                }
            }
            if self.occupied[at] {
                for p in 0..6 {
                    for q in 0..6 {
                        fl[p][q] += a.j[i][p] * self.jraw[q][at];
                    }
                }
            }
        }
        (m, fl)
    }
}

/// Nearest still-pair node for any position (uniform bucket grid).
struct NearestIndex {
    nodes: Array2<f64>,
    cell: f64,
    x0: f64,
    y0: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<usize>>,
}

impl NearestIndex {
    fn new(nodes: &Array2<f64>) -> Self {
        let n = nodes.nrows().max(1);
        let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for r in nodes.rows() {
            x0 = x0.min(r[0]);
            y0 = y0.min(r[1]);
            x1 = x1.max(r[0]);
            y1 = y1.max(r[1]);
        }
        let area = ((x1 - x0) * (y1 - y0)).max(1.0);
        let cell = (area / n as f64).sqrt().max(1.0);
        let nx = ((x1 - x0) / cell).floor() as usize + 1;
        let ny = ((y1 - y0) / cell).floor() as usize + 1;
        let mut buckets = vec![Vec::new(); nx * ny];
        for (k, r) in nodes.rows().into_iter().enumerate() {
            let bx = ((r[0] - x0) / cell) as usize;
            let by = ((r[1] - y0) / cell) as usize;
            buckets[by * nx + bx].push(k);
        }
        NearestIndex { nodes: nodes.clone(), cell, x0, y0, nx, ny, buckets }
    }

    fn nearest(&self, p: [f64; 2]) -> usize {
        let bx = (((p[0] - self.x0) / self.cell).floor() as isize).clamp(0, self.nx as isize - 1);
        let by = (((p[1] - self.y0) / self.cell).floor() as isize).clamp(0, self.ny as isize - 1);
        let mut best = (f64::INFINITY, 0usize);
        let mut ring = 0isize;
        loop {
            for yy in by - ring..=by + ring {
                for xx in bx - ring..=bx + ring {
                    if yy < 0 || xx < 0 || yy >= self.ny as isize || xx >= self.nx as isize {
                        continue;
                    }
                    if (yy - by).abs() != ring && (xx - bx).abs() != ring {
                        continue;
                    }
                    for &k in &self.buckets[yy as usize * self.nx + xx as usize] {
                        let d = (self.nodes[[k, 0]] - p[0]).hypot(self.nodes[[k, 1]] - p[1]);
                        if d < best.0 {
                            best = (d, k);
                        }
                    }
                }
            }
            // Every unvisited bucket is at least `ring * cell` away.
            if best.0 <= ring as f64 * self.cell || ring as usize > self.nx.max(self.ny) {
                return best.1;
            }
            ring += 1;
        }
    }
}

fn mat_mul(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|k| a[i][k] * b[k][j]).sum()))
}

fn transpose(a: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

/// `e = B(d, false) p_C − B(d, true) p_A`, `p = [u, v, u_x, v_x, u_y, v_y]`.
fn b_mat(d: [f64; 2], own: bool) -> [[f64; 6]; 2] {
    let mut m = [[0.0; 6]; 2];
    m[0][0] = 1.0;
    m[1][1] = 1.0;
    if own {
        m[0][2] = d[0];
        m[1][3] = d[0];
        m[0][4] = d[1];
        m[1][5] = d[1];
    }
    m
}

/// `B V B'` for 2×6 B and 6×6 V.
fn bvb(b1: &[[f64; 6]; 2], v: &[[f64; 6]; 6], b2: &[[f64; 6]; 2]) -> [[f64; 2]; 2] {
    std::array::from_fn(|r| {
        std::array::from_fn(|c| (0..6).map(|i| (0..6).map(|k| b1[r][i] * v[i][k] * b2[c][k]).sum::<f64>()).sum())
    })
}

fn mismatch(pa: &[f64], pc: &[f64], d: [f64; 2]) -> [f64; 2] {
    [pc[0] - pa[0] - (pa[2] * d[0] + pa[4] * d[1]), pc[1] - pa[1] - (pa[3] * d[0] + pa[5] * d[1])]
}

/// χ²₂ quantile, closed form.
fn chi2_2_ppf(p: f64) -> f64 {
    -2.0 * (1.0 - p).ln()
}

/// Keep ⅓, add ⅔ until stable (`rounds.py::run_two_sided`): a flagged node
/// stays flagged only if at least a third of its neighbours are flagged; an
/// unflagged node is added if at least two-thirds are. Both parts apply
/// together each round. Returns the final set and the number of rounds.
fn neighbour_rule(flagged: &[bool], neighbours: &[Vec<usize>]) -> (Vec<bool>, usize) {
    let mut cur = flagged.to_vec();
    let mut rounds = 0;
    for _ in 0..MAX_ROUNDS {
        let next: Vec<bool> = (0..cur.len())
            .map(|i| {
                let nb = &neighbours[i];
                let f = if nb.is_empty() { 0.0 } else { nb.iter().filter(|&&k| cur[k]).count() as f64 / nb.len() as f64 };
                if cur[i] { f >= KEEP_SHARE } else { f >= ADD_SHARE }
            })
            .collect();
        if next == cur {
            break;
        }
        cur = next;
        rounds += 1;
    }
    (cur, rounds)
}

/// The 1-D 5-tap load blur (σ = 1.1), as `image.rs::gaussian_blur_5x5`.
fn blur_taps() -> [f64; 5] {
    let raw: [f64; 5] = std::array::from_fn(|i| (-(((i as f64) - 2.0).powi(2)) / (2.0 * 1.1 * 1.1)).exp());
    let sum: f64 = raw.iter().sum();
    raw.map(|v| v / sum)
}

/// Camera noise (grey levels) from the moving pair on a small patch around the
/// seed node: the seed node and its neighbours, where nothing beyond
/// rigid-body motion is assumed, so each converged order-1 residual is noise
/// only. Per pixel, `Var(e) = σ_b² (1/δ_f² + 1/δ_g²) ≈ 2 σ_b²/δ_f²`, with σ_b²
/// the blurred noise of one image (less 1/12 for the 8-bit rounding, which the
/// screen adds separately) and `σ² = σ_b² / (Σ g²)²` for the 2-D load blur.
/// `None` if no patch node is usable.
pub(crate) fn seed_patch_sigma(inp: &ScreenInput) -> Option<f64> {
    let patch: Vec<usize> = std::iter::once(inp.seed)
        .chain(inp.neighbours.get(inp.seed)?.iter().copied())
        .filter(|&i| inp.usable[i])
        .collect();
    let (mut ss, mut dof) = (0.0, 0.0);
    for i in patch {
        let coord = [inp.nodes[[i, 0]], inp.nodes[[i, 1]]];
        let Ok(sub) = Subset::new(coord, inp.local_mask, inp.global_mask, Arc::clone(inp.f_img), Arc::clone(inp.g_img), 1)
        else { continue };
        let p: Vec<f64> = (0..6).map(|k| inp.p[[i, k]]).collect();
        let Ok(e) = sub.residual_with(&p) else { continue };
        ss += e.iter().map(|v| v * v).sum::<f64>() * sub.delta_f * sub.delta_f / 2.0;
        dof += e.len() as f64 - 6.0;
    }
    if dof <= 0.0 {
        return None;
    }
    let sigma_b2 = (ss / dof - 1.0 / 12.0).max(0.0);
    let g2: f64 = blur_taps().iter().map(|v| v * v).sum();
    Some((sigma_b2 / (g2 * g2)).sqrt())
}

/// Run the screen on one solved mesh.
pub(crate) fn screen(inp: &ScreenInput) -> Result<ScreenOutput, Error> {
    let n = inp.nodes.nrows();
    let still_index = inp.still.map(|s| NearestIndex::new(&s.solution.nodes));
    let (sigma, source) = match inp.still {
        Some(_) => (f64::NAN, NoiseSource::StillPair),
        None => match seed_patch_sigma(inp) {
            Some(sigma) => (sigma, NoiseSource::SeedPatch { sigma }),
            None => (DEFAULT_NOISE_SIGMA, NoiseSource::Default { sigma: DEFAULT_NOISE_SIGMA }),
        },
    };
    let kernel = match inp.still {
        Some(_) => Kernel::hac(inp.h),
        None => Kernel::default_noise(inp.h),
    };

    // One pass over nodes: each node's inverse Hessian and self sandwich, and
    // the cross sandwich with each lower-index neighbour (on this node's grid).
    struct NodeOut {
        stats: NodeStats,
        cross: Vec<(usize, [[f64; 6]; 6])>,
    }
    let per_node: Vec<Option<NodeOut>> = (0..n)
        .into_par_iter()
        .map(|c| {
            if !inp.usable[c] {
                return None;
            }
            let (dc, p_still) = build_node(inp, c, &still_index, sigma)?;
            let mut jj = vec![0.0; 36];
            for row in &dc.j {
                for p in 0..6 {
                    for q in 0..6 {
                        jj[p * 6 + q] += row[p] * row[q];
                    }
                }
            }
            let lu = lu_factor(jj.clone(), 6)?;
            let cols: Vec<Vec<f64>> = (0..6)
                .map(|k| {
                    let mut e = vec![0.0; 6];
                    e[k] = 1.0;
                    lu_solve(&lu, 6, &e)
                })
                .collect();
            let hi: [[f64; 6]; 6] = std::array::from_fn(|r| std::array::from_fn(|k| cols[k][r]));
            let grid = Grid::new(&dc, &kernel, true);
            let (m, _) = grid.cross(&dc);
            let q = (1.0 / 6.0) / (dc.df * dc.df);
            let meat = std::array::from_fn(|p| std::array::from_fn(|r| (m[p][r] + q * jj[p * 6 + r]) * dc.ssf));
            let mut cross = Vec::new();
            for &a in inp.neighbours[c].iter().filter(|&&a| a < c && inp.usable[a]) {
                let Some((da, _)) = build_node(inp, a, &still_index, sigma) else { continue };
                let (m, fl) = grid.cross(&da);
                let q = (1.0 / 6.0) / (da.df * da.df);
                let sc = (da.ssf * dc.ssf).sqrt();
                cross.push((a, std::array::from_fn(|p| std::array::from_fn(|r| (m[p][r] + q * fl[p][r]) * sc))));
            }
            Some(NodeOut { stats: NodeStats { hi, meat, p_still }, cross })
        })
        .collect();

    // Node degree and shortest neighbour distance (all neighbours).
    let radius = inp.local_mask.size as f64;
    let dist = |a: usize, b: usize| (inp.nodes[[b, 0]] - inp.nodes[[a, 0]]).hypot(inp.nodes[[b, 1]] - inp.nodes[[a, 1]]);
    let size: Vec<f64> = (0..n)
        .map(|i| {
            let smin = inp.neighbours[i].iter().map(|&k| dist(i, k)).fold(f64::INFINITY, f64::min);
            2.0 * L_STAR * smin * smin / (radius * radius)
        })
        .collect();
    let level: Vec<f64> = (0..n)
        .map(|i| chi2_2_ppf((1.0 - ALPHA).powf(1.0 / (2 * inp.neighbours[i].len()).max(1) as f64)))
        .collect();

    // Each pair, both directions: mismatch, its noise covariance, Mahalanobis.
    struct PairOut {
        a: usize,
        c: usize,
        /// Per direction (a→c, c→a): |e|, e'S⁻¹e, sqrt(trace S / 2), |e_still|.
        dirs: [(f64, f64, f64, Option<f64>); 2],
    }
    let pairs: Vec<PairOut> = (0..n)
        .into_par_iter()
        .flat_map_iter(|c| {
            let mut out = Vec::new();
            let Some(nc) = per_node[c].as_ref() else { return out.into_iter() };
            for &(a, mac) in &nc.cross {
                let Some(na) = per_node[a].as_ref() else { continue };
                let (sa, sc_) = (&na.stats, &nc.stats);
                let mut dirs = [(0.0, 0.0, 0.0, None); 2];
                for (k, (i, j, mij)) in [(a, c, mac), (c, a, transpose(&mac))].into_iter().enumerate() {
                    let (si, sj) = if i == a { (sa, sc_) } else { (sc_, sa) };
                    let dd = [inp.nodes[[j, 0]] - inp.nodes[[i, 0]], inp.nodes[[j, 1]] - inp.nodes[[i, 1]]];
                    let (ba, bc) = (b_mat(dd, true), b_mat(dd, false));
                    let vaa = mat_mul(&mat_mul(&si.hi, &si.meat), &si.hi);
                    let vcc = mat_mul(&mat_mul(&sj.hi, &sj.meat), &sj.hi);
                    let vac = mat_mul(&mat_mul(&si.hi, &mij), &sj.hi);
                    let t1 = bvb(&bc, &vcc, &bc);
                    let t2 = bvb(&ba, &vaa, &ba);
                    let t3 = bvb(&bc, &transpose(&vac), &ba);
                    let t4 = bvb(&ba, &vac, &bc);
                    let s: [[f64; 2]; 2] = std::array::from_fn(|r| std::array::from_fn(|cc| t1[r][cc] + t2[r][cc] - t3[r][cc] - t4[r][cc]));
                    let s = [[s[0][0], 0.5 * (s[0][1] + s[1][0])], [0.5 * (s[0][1] + s[1][0]), s[1][1]]];
                    let pi: Vec<f64> = (0..6).map(|q| inp.p[[i, q]]).collect();
                    let pj: Vec<f64> = (0..6).map(|q| inp.p[[j, q]]).collect();
                    let e = mismatch(&pi, &pj, dd);
                    let mahal = solve_dense(vec![s[0][0], s[0][1], s[1][0], s[1][1]], 2, &e)
                        .map_or(f64::NAN, |x| e[0] * x[0] + e[1] * x[1]);
                    let e_still = match (si.p_still, sj.p_still) {
                        (Some(ps_i), Some(ps_j)) => {
                            let es = mismatch(&ps_i, &ps_j, dd);
                            Some(es[0].hypot(es[1]))
                        }
                        _ => None,
                    };
                    dirs[k] = (e[0].hypot(e[1]), mahal, (0.5 * (s[0][0] + s[1][1])).max(0.0).sqrt(), e_still);
                }
                out.push(PairOut { a, c, dirs });
            }
            out.into_iter()
        })
        .collect();

    // Flag, as `mesh_stage.py::screens`.
    let mut flagged = vec![false; n];
    let mut scales: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut still_mm: Vec<f64> = vec![0.0; n];
    for pr in &pairs {
        for (k, (i, _)) in [(pr.a, pr.c), (pr.c, pr.a)].into_iter().enumerate() {
            let (e, mahal, scale, e_still) = pr.dirs[k];
            scales[pr.a].push(scale);
            scales[pr.c].push(scale);
            if let Some(es) = e_still {
                still_mm[pr.a] = still_mm[pr.a].max(es);
                still_mm[pr.c] = still_mm[pr.c].max(es);
            }
            if e <= size[i] || !mahal.is_finite() {
                continue;
            }
            for node in [pr.a, pr.c] {
                if mahal > level[node] {
                    flagged[node] = true;
                }
            }
        }
    }

    let (sent, rounds) = neighbour_rule(&flagged, inp.neighbours);

    let noise_scale = Array1::from_iter(scales.iter_mut().map(|v| {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = v.len();
        if m % 2 == 1 { v[m / 2] } else { 0.5 * (v[m / 2 - 1] + v[m / 2]) }
    }));
    Ok(ScreenOutput {
        flagged,
        sent,
        rounds,
        noise_scale,
        still_mismatch: inp.still.map(|_| Array1::from_vec(still_mm)),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi2_level_matches_scipy() {
        // scipy.stats.chi2.ppf(0.95 ** (1 / 12), 2)
        let got = chi2_2_ppf(0.95_f64.powf(1.0 / 12.0));
        assert!((got - 10.914476716288904).abs() < 1e-9, "{got}");
    }

    /// A ring of 6 neighbours around node 0, each also linked to its ring
    /// neighbours.
    fn wheel() -> Vec<Vec<usize>> {
        let mut nb = vec![(1..=6).collect::<Vec<_>>()];
        for k in 1..=6 {
            let prev = if k == 1 { 6 } else { k - 1 };
            let next = if k == 6 { 1 } else { k + 1 };
            nb.push(vec![0, prev, next]);
        }
        nb
    }

    #[test]
    fn neighbour_rule_fills_holes_and_drops_isolated_pairs() {
        let nb = wheel();
        // hole: all ring nodes flagged, centre not -> centre added (6/6 >= 2/3)
        let mut f = vec![true; 7];
        f[0] = false;
        let (s, rounds) = neighbour_rule(&f, &nb);
        assert!(s.iter().all(|&v| v) && rounds == 1);
        // an isolated pair (1, 2) with nothing else: each has 1 of 3 flagged
        // neighbours (1/3 >= 1/3) -> kept
        let mut f = vec![false; 7];
        f[1] = true;
        f[2] = true;
        assert_eq!(neighbour_rule(&f, &nb).0, f);
        // a lone flagged node with no flagged neighbour is dropped
        let mut f = vec![false; 7];
        f[3] = true;
        let (s, _) = neighbour_rule(&f, &nb);
        assert!(s.iter().all(|&v| !v));
    }

    #[test]
    fn mismatch_zero_for_affine_field() {
        // u = a + G x: A's warp predicts C exactly.
        let g = [0.01, -0.02, 0.03, 0.004];
        let warp = |x: f64, y: f64| [0.5 + g[0] * x + g[2] * y, -0.2 + g[1] * x + g[3] * y, g[0], g[1], g[2], g[3]];
        let (pa, pc) = (warp(10.0, 20.0), warp(24.0, 13.0));
        let e = mismatch(&pa, &pc, [14.0, -7.0]);
        assert!(e[0].abs() < 1e-15 && e[1].abs() < 1e-15);
    }

    #[test]
    fn nearest_index_finds_nearest() {
        let nodes = Array2::from_shape_fn((50, 2), |(i, c)| if c == 0 { (i % 10) as f64 * 7.3 } else { (i / 10) as f64 * 5.1 });
        let idx = NearestIndex::new(&nodes);
        for p in [[0.0, 0.0], [33.3, 12.0], [100.0, -40.0], [15.0, 20.4]] {
            let brute = (0..50)
                .min_by(|&a, &b| {
                    let da = (nodes[[a, 0]] - p[0]).hypot(nodes[[a, 1]] - p[1]);
                    let db = (nodes[[b, 0]] - p[0]).hypot(nodes[[b, 1]] - p[1]);
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap();
            assert_eq!(idx.nearest(p), brute, "{p:?}");
        }
    }

    #[test]
    fn default_kernel_is_blur_autocorrelation_times_hac() {
        let k = Kernel::default_noise(6.0);
        assert_eq!(k.w1.len(), 9);
        // symmetric, peak at 0, zero outside the blur autocorrelation's support
        for t in 0..4 {
            assert!((k.w1[t] - k.w1[8 - t]).abs() < 1e-15);
        }
        assert!(k.w1[4] > k.w1[5] && k.w1[0] > 0.0);
    }
}
