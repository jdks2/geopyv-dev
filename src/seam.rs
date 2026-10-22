//! Seam investigation and correction for a solved order-1 subset.
//!
//! Port of the sandbox reference (`geopyv_dev_pub/sandbox_break_test`:
//! `break_test.py`, `refine.py`, the `break_test_rs` crate,
//! `fresh_comparison/run_fresh.py::robust_T`/`corr_length`,
//! `png_three_way/run.py::decide` and `png_three_way/deployed.py::image_free_levels`).
//! Plan: `geopyv_dev_pub/mds/geopyv_dev_integration_plan.md`.
//!
//! # Tests
//!
//! Three robust scores on the converged order-1 residual `e`:
//! - **T_bend**: the 6 quadratic columns `[f_x x², f_y x², f_x xy, f_y xy, f_x y², f_y y²]`.
//! - **T_hinge** / **T_jump**: a kink `H(s − c)(s − c)[f_x, f_y]` or a step
//!   `H(s − c)[f_x, f_y]` along the line `s = x cos θ + y sin θ = c`, each at its
//!   own line, found by a coarse 15° / 3 px scan then LM-damped (Gauss-)Newton on a
//!   smoothed step of width `w`.
//!
//! with `f = ∇f / δ_f` and `(x, y)` the pixel offsets from the subset centre.
//! Each score is a Wald statistic whose variance is a spatial HAC sum (Gaussian
//! kernel of width `h`, the speckle correlation length) plus the 8-bit
//! quantisation floor.
//!
//! # Decision (margin rule)
//!
//! `T_edge` is whichever of hinge/jump has the larger `T / alarm`. **Cut** if
//! `T_edge > alarm_edge` and `T_edge − T_bend > margin`; otherwise **order 2** if
//! `T_edge > alarm_edge` or `T_bend > alarm_bend`; otherwise nothing.
//!
//! # Correction
//!
//! A cut masks the far side of the line (gap `g`) and re-solves. The cut
//! subset then **moves** (and so its node) to the centre of area of the
//! pixels it keeps, with the line fixed in the image, and is solved again
//! there: its result describes the kept side, so it is reported at the
//! middle of that side, not at a centre that may lie on the line itself. The
//! moved subset is tested again (pass 2): a second cut (then moved and
//! re-solved again), order 2 on the cut subset, or nothing. Testing after the
//! move matters: the moved disc can reach an edge the first disc did not
//! (`geopyv_dev_pub/docs/centroid_jolt_report.pdf`).
//!
//! # Levels
//!
//! Image-free: `N_eff = n_disc / (2π h²)`; `T_bend` at the small-sample F level;
//! the other levels at power laws `a N_eff^−b` fitted on simulated pure-noise
//! nulls (`deployed.py::LAW`).

use std::sync::Arc;

use ndarray::{Array2, ArrayView2};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};

use crate::{image::Image, masks::LocalMask, mesh::SolveMethod, subset::{SolveResult, Subset}, Error};

// ---------------------------------------------------------------------------
// Configuration, levels and records
// ---------------------------------------------------------------------------

/// Settings for [`Subset::solve_seams`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeamConfig {
    /// Gap (px) left on the kept side of a cut line.
    pub gap: f64,
    /// Move each cut subset to the centre of area of its kept pixels.
    pub recentre: bool,
    /// Smallest kept fraction of the full template for a second cut.
    pub min_keep: f64,
    /// Maximum number of cuts (1 or 2).
    pub max_cuts: usize,
    /// Whether order 2 may be applied (full disc at pass 1, cut subset at pass 2).
    pub allow_order2: bool,
    /// Explicit levels; `None` uses the image-free levels.
    pub levels: Option<SeamLevels>,
}

impl Default for SeamConfig {
    fn default() -> Self {
        SeamConfig { gap: 3.0, recentre: true, min_keep: 0.15, max_cuts: 2, allow_order2: true, levels: None }
    }
}

/// Alarm levels for one pass.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PassLevels {
    pub bend: f64,
    pub hinge: f64,
    pub jump: f64,
    pub edge: f64,
    pub margin: f64,
}

/// Alarm levels for pass 1 (the plain subset) and pass 2 (the cut subset).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeamLevels {
    pub pass1: PassLevels,
    pub pass2: PassLevels,
}

/// Power-law coefficients `(a, b)`, level = `a N_eff^-b`; `deployed.py::LAW`
/// (simulated pure-noise nulls, d = 10 px, R = 25/40/60, σ = 0/2/5 pooled,
/// 99th percentiles; validated at speckle sizes 6–14 px).
const LAW_PASS1: [(f64, f64); 4] = [(353.29, 0.6553), (194.828, 0.4813), (323.769, 0.5906), (307.867, 0.6535)];
const LAW_PASS2: [(f64, f64); 4] = [(249.408, 0.4875), (342.542, 0.604), (340.575, 0.5433), (289.755, 0.556)];

impl SeamLevels {
    /// Image-free levels for a template of `n_disc` pixels and speckle
    /// correlation length `h` (px). Matches `deployed.py::image_free_levels`.
    pub fn image_free(n_disc: usize, h: f64) -> Self {
        let n = n_disc as f64 / (2.0 * std::f64::consts::PI * h * h);
        let f = f_level(n, 6);
        let pass = |law: &[(f64, f64); 4]| {
            let l = |k: usize| law[k].0 * n.powf(-law[k].1);
            PassLevels { bend: f, hinge: l(0), jump: l(1), edge: l(2), margin: l(3) }
        };
        SeamLevels { pass1: pass(&LAW_PASS1), pass2: pass(&LAW_PASS2) }
    }
}

/// Which edge model a line belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    Hinge,
    Jump,
}

/// A line `x cos θ + y sin θ = c` in the subset's own offsets, with its score.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeamLine {
    pub kind: EdgeKind,
    pub theta: f64,
    pub c: f64,
    pub t: f64,
}

/// The decision at one pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeamDecision {
    Nothing,
    Order2,
    Cut(EdgeKind),
}

/// Which subset a pass scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeamStage {
    /// The plain subset at its original position.
    Pass1,
    /// No longer produced: the plain subset after the earlier pre-cut jolt
    /// (kept so older files still load).
    Pass1Jolted,
    /// The cut subset, after its move.
    Pass2,
}

/// Scores and decision of one pass.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeamPass {
    pub stage: SeamStage,
    pub t_bend: f64,
    pub hinge: Option<SeamLine>,
    pub jump: Option<SeamLine>,
    pub decision: SeamDecision,
}

/// What was finally applied to the subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeamAction {
    Nothing,
    Order2,
    Cut,
    CutOrder2,
    Cut2,
}

/// Something that did not go as planned; the subset keeps the last good solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeamFlag {
    /// The plain solve had not converged; nothing was investigated.
    PlainNotConverged,
    /// No longer produced (the earlier pre-cut jolt; kept so older files
    /// still load).
    JoltFailed,
    /// No longer produced (as `JoltFailed`).
    JoltBlocked,
    /// The cut subset could not be solved; the plain solve is kept.
    CutFailed,
    /// The order-2 re-solve did not converge; the previous solve is kept.
    Order2Failed,
    /// A second cut would keep less than `min_keep` of the template.
    SecondCutTooSmall,
    /// No longer produced (as `JoltFailed`).
    SecondLineNearCentre,
    /// The mesh allowed only part of a move (it would otherwise distort an
    /// element, or the node is on the boundary); the subset moved that far.
    MoveLimited,
    /// The mesh allowed no move; the subset stays where it was cut.
    MoveBlocked,
    /// The moved subset could not be solved; it stays where it was cut.
    MoveFailed,
}

/// Per-subset record of the seam investigation and correction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeamRecord {
    /// Centre of the plain solve, before any move.
    pub plain_coord: [f64; 2],
    /// Warp of the plain order-1 solve at `plain_coord`.
    pub plain_p: Vec<f64>,
    /// Total move of the centre after its cuts, if any (final centre =
    /// `plain_coord + moved`). Files from before the move hold the earlier
    /// pre-cut jolt here.
    pub moved: Option<[f64; 2]>,
    pub action: SeamAction,
    /// Cut lines applied, in the final subset's offsets.
    pub lines: Vec<SeamLine>,
    /// Gap (px) the cuts were made with.
    pub gap: f64,
    pub passes: Vec<SeamPass>,
    /// Speckle correlation length used (px).
    pub h: f64,
    pub levels: SeamLevels,
    pub flags: Vec<SeamFlag>,
}

// ---------------------------------------------------------------------------
// Lossless images
// ---------------------------------------------------------------------------

const LOSSLESS: [&str; 4] = ["png", "tif", "tiff", "bmp"];

/// Refuse lossy image files: JPEG's 8×8 block compression differs between two
/// images of moving texture and the tests read it as under-fit (context.md
/// key observation 3). Images built from arrays have no file and pass.
pub(crate) fn require_lossless(img: &Image) -> Result<(), Error> {
    if let Some(path) = &img.filepath {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        if !LOSSLESS.contains(&ext.as_str()) {
            return Err(Error::InvalidInput(format!(
                "seam investigation needs lossless images (PNG, TIFF or BMP), got {}",
                path.display()
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Speckle correlation length
// ---------------------------------------------------------------------------

/// Speckle correlation length (px) of a reference image: the lag at which the
/// autocorrelation of the central (up to 512 px) crop first falls to 0.5,
/// averaged over x and y lags, linearly interpolated. Matches
/// `run_fresh.py::corr_length`.
pub fn correlation_length(img: &Image) -> Result<f64, Error> {
    let a = &img.image_gs;
    let (rows, cols) = a.dim();
    let side = 512.min(rows).min(cols) & !1;
    if side < 16 {
        return Err(Error::InvalidInput("image too small to measure the speckle correlation length".into()));
    }
    let half = side / 2;
    let (cy, cx) = (rows / 2, cols / 2);
    let mut crop: Vec<Complex<f64>> = Vec::with_capacity(side * side);
    for r in cy - half..cy + half {
        for c in cx - half..cx + half {
            crop.push(Complex::new(a[[r, c]], 0.0));
        }
    }
    let mean = crop.iter().map(|z| z.re).sum::<f64>() / (side * side) as f64;
    crop.iter_mut().for_each(|z| z.re -= mean);

    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(side);
    let inv = planner.plan_fft_inverse(side);
    let fft2 = |buf: &mut Vec<Complex<f64>>, plan: &Arc<dyn rustfft::Fft<f64>>| {
        for row in buf.chunks_mut(side) {
            plan.process(row);
        }
        let mut col = vec![Complex::new(0.0, 0.0); side];
        for c in 0..side {
            for r in 0..side {
                col[r] = buf[r * side + c];
            }
            plan.process(&mut col);
            for r in 0..side {
                buf[r * side + c] = col[r];
            }
        }
    };
    fft2(&mut crop, &fwd);
    crop.iter_mut().for_each(|z| *z *= z.conj());
    fft2(&mut crop, &inv);
    // Circular autocorrelation, unshifted: lag (dy, dx) at [dy * side + dx].
    let ac0 = crop[0].re;
    let prof: Vec<f64> = (0..half).map(|k| 0.5 * (crop[k].re + crop[k * side].re) / ac0).collect();
    let k = prof.iter().position(|&v| v < 0.5).ok_or_else(|| {
        Error::InvalidInput("speckle autocorrelation never falls to 0.5".into())
    })?;
    Ok(k as f64 - 1.0 + (prof[k - 1] - 0.5) / (prof[k - 1] - prof[k]))
}

// ---------------------------------------------------------------------------
// F level (small-sample, k fixed columns)
// ---------------------------------------------------------------------------

/// `k (N − 1)/(N − k) · F_{k, N−k}(0.99)`, or ∞ for `N ≤ k + 1`.
/// Matches `speckle_analyse.py::f_level`.
pub fn f_level(n: f64, k: usize) -> f64 {
    let kf = k as f64;
    if n <= kf + 1.0 {
        return f64::INFINITY;
    }
    kf * (n - 1.0) / (n - kf) * f_ppf(0.99, kf, n - kf)
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos (g = 7, n = 9).
    const C: [f64; 9] = [
        0.999_999_999_999_809_9, 676.520_368_121_885_1, -1_259.139_216_722_402_8,
        771.323_428_777_653_1, -176.615_029_162_140_6, 12.507_343_278_686_905,
        -0.138_571_095_265_720_12, 9.984_369_578_019_572e-6, 1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let s = C[0] + (1..9).map(|i| C[i] / (x + i as f64)).sum::<f64>();
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + s.ln()
}

/// Continued fraction for the incomplete beta function (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Regularised incomplete beta `I_x(a, b)`.
fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    if x < (a + 1.0) / (a + b + 2.0) {
        ln_front.exp() * beta_cf(a, b, x) / a
    } else {
        1.0 - ln_front.exp() * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Quantile of the F distribution with `(d1, d2)` degrees of freedom.
fn f_ppf(p: f64, d1: f64, d2: f64) -> f64 {
    // F = (d2/d1) x/(1 − x) with x ~ Beta(d1/2, d2/2): bisect on x.
    let (a, b) = (0.5 * d1, 0.5 * d2);
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if beta_inc(a, b, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let x = 0.5 * (lo + hi);
    d2 / d1 * x / (1.0 - x)
}

// ---------------------------------------------------------------------------
// Subset data for the tests
// ---------------------------------------------------------------------------

/// Pixel offsets, scaled reference gradients `∇f / δ_f` and the converged
/// zero-normalised residual of one solved order-1 subset.
pub(crate) struct SeamData {
    x: Vec<f64>,
    y: Vec<f64>,
    fx: Vec<f64>,
    fy: Vec<f64>,
    e: Vec<f64>,
    delta_f: f64,
}

impl SeamData {
    fn from_subset(s: &Subset) -> Result<Self, Error> {
        let unavail = || Error::InvalidInput("seam investigation: images unavailable".to_string());
        let fc = s.f_coords.as_ref().ok_or_else(unavail)?;
        let grad = s.grad_f.as_ref().ok_or_else(unavail)?;
        let e = s.residual_map()?;
        let n = fc.nrows();
        Ok(SeamData {
            x: (0..n).map(|i| fc[[i, 0]] - s.coord[0]).collect(),
            y: (0..n).map(|i| fc[[i, 1]] - s.coord[1]).collect(),
            fx: (0..n).map(|i| grad[[i, 0]] / s.delta_f).collect(),
            fy: (0..n).map(|i| grad[[i, 1]] / s.delta_f).collect(),
            e: e.to_vec(),
            delta_f: s.delta_f,
        })
    }

    fn n(&self) -> usize {
        self.e.len()
    }

    #[inline(always)]
    fn j_row(&self, i: usize) -> [f64; 6] {
        let (fx, fy, x, y) = (self.fx[i], self.fy[i], self.x[i], self.y[i]);
        [fx, fy, fx * x, fy * x, fx * y, fy * y]
    }

    fn radius(&self) -> f64 {
        (0..self.n()).map(|i| self.x[i].hypot(self.y[i])).fold(f64::NEG_INFINITY, f64::max)
    }

    fn quad_columns(&self) -> Vec<Vec<f64>> {
        let n = self.n();
        let col = |f: &dyn Fn(usize) -> f64| (0..n).map(f).collect::<Vec<f64>>();
        vec![
            col(&|i| self.fx[i] * self.x[i] * self.x[i]),
            col(&|i| self.fy[i] * self.x[i] * self.x[i]),
            col(&|i| self.fx[i] * self.x[i] * self.y[i]),
            col(&|i| self.fy[i] * self.x[i] * self.y[i]),
            col(&|i| self.fx[i] * self.y[i] * self.y[i]),
            col(&|i| self.fy[i] * self.y[i] * self.y[i]),
        ]
    }

    /// Hard-step edge columns at the line (`run_fresh.py::edge_columns`).
    fn edge_columns(&self, kind: EdgeKind, theta: f64, c: f64) -> Vec<Vec<f64>> {
        let (ct, st) = (theta.cos(), theta.sin());
        let base: Vec<f64> = (0..self.n())
            .map(|i| {
                let s = self.x[i] * ct + self.y[i] * st;
                if s > c {
                    match kind {
                        EdgeKind::Jump => 1.0,
                        EdgeKind::Hinge => s - c,
                    }
                } else {
                    0.0
                }
            })
            .collect();
        vec![
            base.iter().zip(&self.fx).map(|(b, f)| b * f).collect(),
            base.iter().zip(&self.fy).map(|(b, f)| b * f).collect(),
        ]
    }
}

// ---------------------------------------------------------------------------
// Small dense linear algebra
// ---------------------------------------------------------------------------

/// LU with partial pivoting (as LAPACK `gesv`). `None` only for an exactly
/// singular matrix, as `np.linalg.solve` raises only then.
pub(crate) fn lu_factor(mut a: Vec<f64>, m: usize) -> Option<(Vec<f64>, Vec<usize>)> {
    let mut piv: Vec<usize> = (0..m).collect();
    for k in 0..m {
        let mut p = k;
        let mut best = a[k * m + k].abs();
        for r in k + 1..m {
            if a[r * m + k].abs() > best {
                best = a[r * m + k].abs();
                p = r;
            }
        }
        if best == 0.0 {
            return None;
        }
        if p != k {
            for c in 0..m {
                a.swap(p * m + c, k * m + c);
            }
            piv.swap(p, k);
        }
        let d = a[k * m + k];
        for r in k + 1..m {
            let f = a[r * m + k] / d;
            a[r * m + k] = f;
            for c in k + 1..m {
                a[r * m + c] -= f * a[k * m + c];
            }
        }
    }
    Some((a, piv))
}

pub(crate) fn lu_solve(lu: &(Vec<f64>, Vec<usize>), m: usize, b: &[f64]) -> Vec<f64> {
    let (a, piv) = lu;
    let mut x: Vec<f64> = (0..m).map(|i| b[piv[i]]).collect();
    for i in 0..m {
        for j in 0..i {
            x[i] -= a[i * m + j] * x[j];
        }
    }
    for i in (0..m).rev() {
        for j in i + 1..m {
            x[i] -= a[i * m + j] * x[j];
        }
        x[i] /= a[i * m + i];
    }
    x
}

pub(crate) fn solve_dense(a: Vec<f64>, m: usize, b: &[f64]) -> Option<Vec<f64>> {
    lu_factor(a, m).map(|lu| lu_solve(&lu, m, b))
}

// ---------------------------------------------------------------------------
// Coarse scan (hard step, LM statistic at the converged affine solution)
// ---------------------------------------------------------------------------

/// Per pixel `b = [J (6), f_x s, f_y s]`.
const NB: usize = 8;
/// Unique entries of the symmetric `b b'`.
const NSYM: usize = NB * (NB + 1) / 2;
const COARSE_THETA_DEG: f64 = 15.0;
const COARSE_C_PX: f64 = 3.0;
const TRIM: f64 = 0.15;

#[inline(always)]
fn add_outer(acc: &mut [f64], b: &[f64; NB]) {
    let mut k = 0;
    for i in 0..NB {
        for j in i..NB {
            acc[k] += b[i] * b[j];
            k += 1;
        }
    }
}

fn unpack(acc: &[f64]) -> [[f64; NB]; NB] {
    let mut g = [[0.0; NB]; NB];
    let mut k = 0;
    for i in 0..NB {
        for j in i..NB {
            g[i][j] = acc[k];
            g[j][i] = acc[k];
            k += 1;
        }
    }
    g
}

/// The affine (null) block, factored once per subset (no guard band, so every
/// candidate keeps every pixel).
struct Affine {
    lu: (Vec<f64>, Vec<usize>),
    xinv_xe: Vec<f64>,
    eme: f64,
    n: f64,
}

/// Rows of the `2 × 8` selector `A`: `Z = b A'` for the jump / hinge columns.
fn a_rows(kind: EdgeKind, c: f64) -> [[f64; NB]; 2] {
    let mut a = [[0.0; NB]; 2];
    match kind {
        EdgeKind::Jump => {
            a[0][0] = 1.0;
            a[1][1] = 1.0;
        }
        EdgeKind::Hinge => {
            // (s - c) f = f s - c f
            a[0][6] = 1.0;
            a[1][7] = 1.0;
            a[0][0] = -c;
            a[1][1] = -c;
        }
    }
    a
}

/// Explained sum of squares of the 2 edge columns at one candidate.
fn explained(kind: EdgeKind, c: f64, g_up: &[[f64; NB]; NB], v_up: &[f64; NB], aff: &Affine) -> Option<f64> {
    let a = a_rows(kind, c);
    let mut ag = [[0.0; NB]; 2];
    for k in 0..2 {
        for l in 0..NB {
            let al = a[k][l];
            if al != 0.0 {
                for m in 0..NB {
                    ag[k][m] += al * g_up[l][m];
                }
            }
        }
    }
    let mut zz = [[0.0; 2]; 2];
    for k in 0..2 {
        for q in 0..2 {
            zz[k][q] = (0..NB).map(|m| ag[k][m] * a[q][m]).sum();
        }
    }
    let ze: [f64; 2] = std::array::from_fn(|k| (0..NB).map(|l| a[k][l] * v_up[l]).sum());
    let w: [Vec<f64>; 2] = std::array::from_fn(|q| lu_solve(&aff.lu, 6, &ag[q][..6]));
    let mut zmz = vec![0.0; 4];
    let mut zme = [0.0; 2];
    for k in 0..2 {
        for q in 0..2 {
            zmz[k * 2 + q] = zz[k][q] - (0..6).map(|m| ag[k][m] * w[q][m]).sum::<f64>();
        }
        zme[k] = ze[k] - (0..6).map(|m| ag[k][m] * aff.xinv_xe[m]).sum::<f64>();
    }
    let bt = solve_dense(zmz, 2, &zme)?;
    Some(zme[0] * bt[0] + zme[1] * bt[1])
}

/// Coarse start for one edge model: the argmax of the profile likelihood over
/// a 15° × 3 px grid. `None` if no candidate leaves `TRIM` of the pixels on
/// each side. Matches `break_test_rs::scan_subset` (fast options) followed by
/// `summarise`, which `refine.py::coarse_start` uses.
fn coarse_start(d: &SeamData, kind: EdgeKind) -> Option<(f64, f64)> {
    let n = d.n();
    let r = d.radius();
    let thetas: Vec<f64> = (0..)
        .map(|k| (k as f64 * COARSE_THETA_DEG).to_radians())
        .take_while(|&t| t < std::f64::consts::PI - 1e-12)
        .collect();
    let rf = r.floor();
    let cs: Vec<f64> = (0..).map(|k| -rf + k as f64 * COARSE_C_PX).take_while(|&c| c < rf + 0.25).collect();
    let nc = cs.len();

    // Affine block (identical for every candidate).
    let mut g = vec![0.0; 36];
    let mut v = vec![0.0; 6];
    let mut e2 = 0.0;
    for i in 0..n {
        let j = d.j_row(i);
        for p in 0..6 {
            for q in 0..6 {
                g[p * 6 + q] += j[p] * j[q];
            }
            v[p] += j[p] * d.e[i];
        }
        e2 += d.e[i] * d.e[i];
    }
    let lu = lu_factor(g, 6)?;
    let xinv_xe = lu_solve(&lu, 6, &v);
    let eme = e2 - (0..6).map(|p| v[p] * xinv_xe[p]).sum::<f64>();
    let aff = Affine { lu, xinv_xe, eme, n: n as f64 };

    let mut best: Option<(f64, usize, usize)> = None; // (LR, it, ic)
    let mut bin_g = vec![0.0; (nc + 1) * NSYM];
    let mut bin_v = vec![0.0; (nc + 1) * NB];
    let mut bin_n = vec![0usize; nc + 1];
    for (it, &th) in thetas.iter().enumerate() {
        let (ct, st) = (th.cos(), th.sin());
        bin_g.iter_mut().for_each(|x| *x = 0.0);
        bin_v.iter_mut().for_each(|x| *x = 0.0);
        bin_n.iter_mut().for_each(|x| *x = 0);
        for i in 0..n {
            let s = d.x[i] * ct + d.y[i] * st;
            // pixel is on the upper side of c_j iff c_j < s
            let k = cs.partition_point(|&c| c < s);
            let j = d.j_row(i);
            let b = [j[0], j[1], j[2], j[3], j[4], j[5], d.fx[i] * s, d.fy[i] * s];
            add_outer(&mut bin_g[k * NSYM..(k + 1) * NSYM], &b);
            for p in 0..NB {
                bin_v[k * NB + p] += b[p] * d.e[i];
            }
            bin_n[k] += 1;
        }
        let mut run_g = vec![0.0; NSYM];
        let mut run_v = [0.0; NB];
        let mut n_up = 0usize;
        for jc in (0..nc).rev() {
            let k = jc + 1;
            for p in 0..NSYM {
                run_g[p] += bin_g[k * NSYM + p];
            }
            for p in 0..NB {
                run_v[p] += bin_v[k * NB + p];
            }
            n_up += bin_n[k];
            let n_low = n - n_up;
            let nk = n as f64;
            if !((n_low as f64) >= TRIM * nk && (n_up as f64) >= TRIM * nk && n > 20) {
                continue;
            }
            let Some(ex) = explained(kind, cs[jc], &unpack(&run_g), &run_v, &aff) else { continue };
            let lr = aff.n * (aff.eme / (aff.eme - ex).max(1e-300)).ln();
            // np.nanargmax: first occurrence of the maximum in row-major order
            let idx = it * nc + jc;
            match best {
                Some((b, bi, bj)) if !(lr > b || (lr == b && idx < bi * nc + bj)) => {}
                _ => best = Some((lr, it, jc)),
            }
        }
    }
    // The refinement starts from the argmax itself (`refine.py::coarse_start`
    // uses c_argmax, not the profile-interval midpoint).
    let (_, it, ic) = best?;
    Some((thetas[it], cs[ic]))
}

// ---------------------------------------------------------------------------
// Refinement of (θ, c): variable projection, LM-damped (Gauss-)Newton
// ---------------------------------------------------------------------------

const INV_SQRT2PI: f64 = 0.398_942_280_401_432_7;
/// Beyond |u/w| > Z_CUT the smoothed step is 0 or 1 to double precision.
const Z_CUT: f64 = 9.0;
/// Width (px) of the smoothed step.
const W: f64 = 0.5;
const REFINE_TOL: f64 = 1e-3;
const REFINE_MAX_ITER: usize = 50;

struct Problem<'a> {
    d: &'a SeamData,
    kind: EdgeKind,
    n: usize,
    r: f64,
    jrows: Vec<[f64; 6]>,
}

/// One linear sub-problem at fixed (θ, c).
struct Fit {
    f: f64,
    xtx: Vec<f64>, // 8 x 8
    xtd: Vec<f64>, // 8 x 2
    dtd: [[f64; 2]; 2],
    dtr: [f64; 2],
}

const M: usize = 8;

impl<'a> Problem<'a> {
    fn new(d: &'a SeamData, kind: EdgeKind) -> Self {
        let n = d.n();
        Problem { d, kind, n, r: d.radius(), jrows: (0..n).map(|i| d.j_row(i)).collect() }
    }

    /// Columns for pixel i at u = s − c: values and d/du (2 each).
    #[inline(always)]
    fn cols(&self, i: usize, u: f64) -> ([f64; 2], [f64; 2]) {
        let z = u / W;
        let (h, phi) = if z > Z_CUT {
            (1.0, 0.0)
        } else if z < -Z_CUT {
            (0.0, 0.0)
        } else {
            (0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2)), INV_SQRT2PI * (-0.5 * z * z).exp())
        };
        let dh = phi / W;
        let j = &self.jrows[i];
        match self.kind {
            EdgeKind::Jump => ([h * j[0], h * j[1]], [dh * j[0], dh * j[1]]),
            EdgeKind::Hinge => {
                let r = u * h + W * phi;
                ([r * j[0], r * j[1]], [h * j[0], h * j[1]])
            }
        }
    }

    fn split_ok(&self, theta: f64, c: f64) -> bool {
        let (ct, st) = (theta.cos(), theta.sin());
        let lo = (0..self.n).filter(|&i| self.d.x[i] * ct + self.d.y[i] * st <= c).count();
        let nf = self.n as f64;
        (lo as f64) >= TRIM * nf && ((self.n - lo) as f64) >= TRIM * nf
    }

    fn fit(&self, theta: f64, c: f64) -> Option<Fit> {
        let n = self.n;
        let (ct, st) = (theta.cos(), theta.sin());
        let mut zs = vec![[0.0; 2]; n];
        let mut dzs = vec![[0.0; 2]; n];
        let mut tt = vec![0.0; n];
        let mut xtx = vec![0.0; M * M];
        let mut xte = vec![0.0; M];
        let mut x = [0.0; M];
        for i in 0..n {
            let (xi, yi) = (self.d.x[i], self.d.y[i]);
            let s = xi * ct + yi * st;
            tt[i] = -xi * st + yi * ct;
            let (z, dz) = self.cols(i, s - c);
            zs[i] = z;
            dzs[i] = dz;
            x[..6].copy_from_slice(&self.jrows[i]);
            x[6..].copy_from_slice(&z);
            let e = self.d.e[i];
            for p in 0..M {
                for q in p..M {
                    xtx[p * M + q] += x[p] * x[q];
                }
                xte[p] += x[p] * e;
            }
        }
        for p in 0..M {
            for q in 0..p {
                xtx[p * M + q] = xtx[q * M + p];
            }
        }
        let a = solve_dense(xtx.clone(), M, &xte)?;
        let beta = &a[6..];
        let mut f = 0.0;
        let mut xtd = vec![0.0; M * 2];
        let mut dtd = [[0.0; 2]; 2];
        let mut dtr = [0.0; 2];
        for i in 0..n {
            x[..6].copy_from_slice(&self.jrows[i]);
            x[6..].copy_from_slice(&zs[i]);
            let dz = dzs[i][0] * beta[0] + dzs[i][1] * beta[1];
            let dv = [dz * tt[i], -dz];
            let r = self.d.e[i] - (0..M).map(|p| x[p] * a[p]).sum::<f64>();
            f += r * r;
            for p in 0..M {
                xtd[p * 2] += x[p] * dv[0];
                xtd[p * 2 + 1] += x[p] * dv[1];
            }
            for u in 0..2 {
                for v in 0..2 {
                    dtd[u][v] += dv[u] * dv[v];
                }
                dtr[u] += dv[u] * r;
            }
        }
        Some(Fit { f, xtx, xtd, dtd, dtr })
    }

    /// Kaufman VarPro normal equations: A = D'D − D'X (X'X)⁻¹ X'D, g = D'r.
    fn step_matrix(&self, fit: &Fit) -> Option<([[f64; 2]; 2], [f64; 2])> {
        let lu = lu_factor(fit.xtx.clone(), M)?;
        let w0 = lu_solve(&lu, M, &(0..M).map(|p| fit.xtd[p * 2]).collect::<Vec<_>>());
        let w1 = lu_solve(&lu, M, &(0..M).map(|p| fit.xtd[p * 2 + 1]).collect::<Vec<_>>());
        let ws = [w0, w1];
        let mut a = fit.dtd;
        for u in 0..2 {
            for v in 0..2 {
                a[u][v] -= (0..M).map(|p| fit.xtd[p * 2 + u] * ws[v][p]).sum::<f64>();
            }
        }
        Some((a, fit.dtr))
    }

    /// Central-difference exact curvature (½ Hessian of F) from g = D'r.
    fn newton_matrix(&self, theta: f64, c: f64) -> Option<[[f64; 2]; 2]> {
        let h = [1e-3 / self.r, 1e-3];
        let mut hm = [[0.0; 2]; 2];
        for kk in 0..2 {
            let mut dphi = [0.0; 2];
            dphi[kk] = h[kk];
            let fp = self.fit(theta + dphi[0], c + dphi[1])?;
            let fm = self.fit(theta - dphi[0], c - dphi[1])?;
            if !(fp.f.is_finite() && fm.f.is_finite()) {
                return None;
            }
            for u in 0..2 {
                hm[u][kk] = -(fp.dtr[u] - fm.dtr[u]) / (2.0 * h[kk]);
            }
        }
        let off = 0.5 * (hm[0][1] + hm[1][0]);
        Some([[hm[0][0], off], [off, hm[1][1]]])
    }
}

fn solve2(a: [[f64; 2]; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let x = solve_dense(vec![a[0][0], a[0][1], a[1][0], a[1][1]], 2, &b)?;
    Some([x[0], x[1]])
}

/// Refine (θ, c) from a coarse start; returns the line folded into θ ∈ [0, π).
/// Matches `refine.py::refine` (hessian = "newton").
fn refine(d: &SeamData, kind: EdgeKind, theta0: f64, c0: f64) -> Option<(f64, f64)> {
    let p = Problem::new(d, kind);
    let (mut theta, mut c) = (theta0, c0);
    let mut fit = p.fit(theta, c)?;
    let mut lam = 1e-3;
    for _ in 0..REFINE_MAX_ITER {
        let Some((mut a, g)) = p.step_matrix(&fit) else { break };
        if let Some(hn) = p.newton_matrix(theta, c) {
            if hn[0][0] > 0.0 && hn[0][0] * hn[1][1] - hn[0][1] * hn[1][0] > 0.0 {
                a = hn;
            }
        }
        let mut accepted: Option<([f64; 2], Fit)> = None;
        for _ in 0..12 {
            let ad = [[a[0][0] * (1.0 + lam), a[0][1]], [a[1][0], a[1][1] * (1.0 + lam)]];
            let Some(dd) = solve2(ad, g) else {
                lam *= 10.0;
                continue;
            };
            match p.fit(theta + dd[0], c + dd[1]) {
                Some(fnew) if fnew.f < fit.f => {
                    accepted = Some((dd, fnew));
                    break;
                }
                _ => lam *= 10.0,
            }
        }
        let Some((dd, fnew)) = accepted else { break };
        theta += dd[0];
        c += dd[1];
        fit = fnew;
        lam = (lam / 10.0).max(1e-12);
        if !p.split_ok(theta, c) || dd[1].hypot(p.r * dd[0]) < REFINE_TOL {
            break;
        }
    }
    let kf = (theta / std::f64::consts::PI).floor();
    let flipped = (kf as i64).rem_euclid(2) == 1;
    Some((theta - kf * std::f64::consts::PI, if flipped { -c } else { c }))
}

/// Coarse scan then refinement for one edge model; `None` if no line fits.
fn locate(d: &SeamData, kind: EdgeKind) -> Option<(f64, f64)> {
    let (t0, c0) = coarse_start(d, kind)?;
    refine(d, kind, t0, c0)
}

// ---------------------------------------------------------------------------
// Robust score (spatial HAC Wald statistic + quantisation floor)
// ---------------------------------------------------------------------------

/// Score statistic for the omitted columns `z`, robust to spatially correlated
/// residuals. Matches `run_fresh.py::robust_T`: Wald form (variance from the
/// residual after fitting [J, Z]), Gaussian kernel of width `h` truncated at
/// 3h (applied separably), plus 8-bit quantisation noise of both images,
/// (1/6)/δ_f² per pixel.
fn robust_t(d: &SeamData, z: &[Vec<f64>], h: f64) -> f64 {
    let n = d.n();
    let k = z.len();
    // Project out the affine columns: M = I − J (J'J)⁻¹ J'.
    let mut jj = vec![0.0; 36];
    for i in 0..n {
        let j = d.j_row(i);
        for p in 0..6 {
            for q in 0..6 {
                jj[p * 6 + q] += j[p] * j[q];
            }
        }
    }
    let Some(lu) = lu_factor(jj, 6) else { return 0.0 };
    let project = |v: &dyn Fn(usize) -> f64| -> Vec<f64> {
        let mut jv = [0.0; 6];
        for i in 0..n {
            let j = d.j_row(i);
            let vi = v(i);
            for p in 0..6 {
                jv[p] += j[p] * vi;
            }
        }
        let a = lu_solve(&lu, 6, &jv);
        (0..n)
            .map(|i| {
                let j = d.j_row(i);
                v(i) - (0..6).map(|p| j[p] * a[p]).sum::<f64>()
            })
            .collect()
    };
    let mz: Vec<Vec<f64>> = z.iter().map(|col| project(&|i| col[i])).collect();
    let me = project(&|i| d.e[i]);
    let mut a = vec![0.0; k * k];
    let mut g = vec![0.0; k];
    for p in 0..k {
        for q in 0..k {
            a[p * k + q] = (0..n).map(|i| mz[p][i] * mz[q][i]).sum();
        }
        g[p] = (0..n).map(|i| mz[p][i] * me[i]).sum();
    }
    let Some(coef) = solve_dense(a.clone(), k, &g) else { return 0.0 };
    let r1: Vec<f64> = (0..n).map(|i| me[i] - (0..k).map(|p| mz[p][i] * coef[p]).sum::<f64>()).collect();

    // Score images on the pixel lattice.
    let (xmin, ymin) = (d.x.iter().cloned().fold(f64::INFINITY, f64::min), d.y.iter().cloned().fold(f64::INFINITY, f64::min));
    let ix: Vec<(usize, usize)> = (0..n)
        .map(|i| ((d.x[i] - xmin).round() as usize, (d.y[i] - ymin).round() as usize))
        .collect();
    let width = ix.iter().map(|p| p.0).max().unwrap_or(0) + 1;
    let height = ix.iter().map(|p| p.1).max().unwrap_or(0) + 1;
    let rr = (3.0 * h).ceil() as isize;
    let w1: Vec<f64> = (-rr..=rr).map(|t| (-((t * t) as f64) / (2.0 * h * h)).exp()).collect();
    let mut imgs = vec![vec![0.0; width * height]; k];
    for (i, &(cx, cy)) in ix.iter().enumerate() {
        for p in 0..k {
            imgs[p][cy * width + cx] = mz[p][i] * r1[i];
        }
    }
    let conv: Vec<Vec<f64>> = imgs.iter().map(|im| gauss_conv(im, width, height, &w1, rr)).collect();
    let mut cm = vec![0.0; k * k];
    for p in 0..k {
        for q in 0..k {
            cm[p * k + q] = imgs[q].iter().zip(&conv[p]).map(|(x, y)| x * y).sum();
        }
    }
    let floor = (1.0 / 6.0) / (d.delta_f * d.delta_f);
    let mut csym = vec![0.0; k * k];
    for p in 0..k {
        for q in 0..k {
            csym[p * k + q] = 0.5 * (cm[p * k + q] + cm[q * k + p]) + floor * a[p * k + q];
        }
    }
    match solve_dense(csym, k, &g) {
        Some(x) => g.iter().zip(&x).map(|(a, b)| a * b).sum(),
        None => 0.0,
    }
}

/// Separable "same"-size convolution with a symmetric 1-D kernel (zero outside).
pub(crate) fn gauss_conv(im: &[f64], width: usize, height: usize, w: &[f64], rr: isize) -> Vec<f64> {
    let rr = rr as usize;
    // Rows: each output pixel sums only the taps that land inside the row;
    // all-zero rows are skipped.
    let mut tmp = vec![0.0; width * height];
    for y in 0..height {
        let row = &im[y * width..(y + 1) * width];
        if row.iter().all(|&v| v == 0.0) {
            continue;
        }
        let out = &mut tmp[y * width..(y + 1) * width];
        for (x, o) in out.iter_mut().enumerate() {
            let lo = x.saturating_sub(rr);
            let hi = (x + rr).min(width - 1);
            *o = (lo..=hi).map(|xx| w[xx + rr - x] * row[xx]).sum();
        }
    }
    // Columns, as whole-row multiply-adds.
    let mut out = vec![0.0; width * height];
    for y in 0..height {
        let lo = y.saturating_sub(rr);
        let hi = (y + rr).min(height - 1);
        let (o, _) = out[y * width..].split_at_mut(width);
        for yy in lo..=hi {
            let k = w[yy + rr - y];
            let r = &tmp[yy * width..(yy + 1) * width];
            for (ov, rv) in o.iter_mut().zip(r) {
                *ov += k * rv;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// One pass: scores and decision
// ---------------------------------------------------------------------------

fn scores(d: &SeamData, h: f64) -> (f64, Option<SeamLine>, Option<SeamLine>) {
    let t_bend = robust_t(d, &d.quad_columns(), h);
    let line = |kind| {
        locate(d, kind).map(|(theta, c)| SeamLine { kind, theta, c, t: robust_t(d, &d.edge_columns(kind, theta, c), h) })
    };
    (t_bend, line(EdgeKind::Hinge), line(EdgeKind::Jump))
}

/// Margin rule at one pass. Matches `png_three_way/run.py::decide`.
fn decide(t_bend: f64, hinge: Option<SeamLine>, jump: Option<SeamLine>, lv: &PassLevels) -> SeamDecision {
    let th = hinge.map_or(0.0, |l| l.t);
    let tj = jump.map_or(0.0, |l| l.t);
    // max() over (hinge, jump) keeps the first on a tie
    let (kind, te) = if tj / lv.jump > th / lv.hinge { (EdgeKind::Jump, tj) } else { (EdgeKind::Hinge, th) };
    if te > lv.edge && te - t_bend > lv.margin {
        SeamDecision::Cut(kind)
    } else if te > lv.edge || t_bend > lv.bend {
        SeamDecision::Order2
    } else {
        SeamDecision::Nothing
    }
}

fn investigate_pass(s: &Subset, h: f64, lv: &PassLevels, stage: SeamStage) -> Result<SeamPass, Error> {
    let d = SeamData::from_subset(s)?;
    let (t_bend, hinge, jump) = scores(&d, h);
    Ok(SeamPass { stage, t_bend, hinge, jump, decision: decide(t_bend, hinge, jump, lv) })
}

// ---------------------------------------------------------------------------
// Masks
// ---------------------------------------------------------------------------

impl LocalMask {
    /// Keep only the centre's side of the line `x cos θ + y sin θ = c` (pixel
    /// offsets), less a gap of `gap` px next to the line. Matches the
    /// sandbox's `masked_solve` rule: keep `s < c − gap` if `c > 0`, else
    /// `s > c + gap`. Applies to both `coords` and `subset_mask`, so a later
    /// `mask_update` keeps the cut.
    pub fn cut_half_plane(&mut self, theta: f64, c: f64, gap: f64) {
        let (ct, st) = (theta.cos(), theta.sin());
        let keep = |x: f64, y: f64| {
            let s = ct * x + st * y;
            if c > 0.0 { s < c - gap } else { s > c + gap }
        };
        let size = self.size as isize;
        let side = self.subset_mask.nrows();
        for row in 0..side {
            for col in 0..side {
                let (x, y) = ((col as isize - size) as f64, (row as isize - size) as f64);
                if !keep(x, y) {
                    self.subset_mask[[row, col]] = 0;
                }
            }
        }
        let kept: Vec<[f64; 2]> = self
            .coords
            .rows()
            .into_iter()
            .filter(|r| keep(r[0], r[1]))
            .map(|r| [r[0], r[1]])
            .collect();
        self.coords = Array2::from_shape_fn((kept.len(), 2), |(i, j)| kept[i][j]);
        self.m_n_px = Some(kept.len());
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// The shift a cut subset at the given centre may make towards the given
/// full shift (to the centre of area of its kept pixels).
pub(crate) type Mover<'m> = dyn Fn([f64; 2], [f64; 2]) -> [f64; 2] + Sync + 'm;

/// A subset cut at pass 1, not yet moved: what [`Subset::seam_rest`] needs.
pub(crate) struct SeamPending {
    rec: SeamRecord,
    /// Cut lines in the subset's current offsets.
    lines: Vec<SeamLine>,
    settings: Settings,
}

/// Solve settings of the plain solve, reused for every re-solve.
#[derive(Clone, Copy)]
struct Settings {
    method: SolveMethod,
    tolerance: f64,
    max_norm: f64,
    max_iterations: usize,
}

impl Settings {
    fn of(r: &SolveResult) -> Self {
        Settings { method: r.method, tolerance: r.tolerance, max_norm: r.max_norm, max_iterations: r.max_iterations }
    }
}

/// Everything needed to build re-solved subsets around one node.
struct Builder<'a> {
    local_mask: &'a LocalMask,
    global_mask: Option<ArrayView2<'a, u8>>,
    f_img: Arc<Image>,
    g_img: Arc<Image>,
    settings: Settings,
    time: Option<(f64, f64)>,
    time_source: crate::timing::TimeSource,
}

impl Builder<'_> {
    /// Build and solve a subset at `coord` with the given cuts and order;
    /// `None` if it cannot be built or does not converge.
    fn solve(&self, coord: [f64; 2], cuts: &[SeamLine], gap: f64, order: usize, p0: &[f64]) -> Option<Subset> {
        let mut lm = self.local_mask.clone();
        for l in cuts {
            lm.cut_half_plane(l.theta, l.c, gap);
        }
        let mut s = Subset::new(coord, &lm, self.global_mask, Arc::clone(&self.f_img), Arc::clone(&self.g_img), order).ok()?;
        s.time = self.time;
        s.time_source = self.time_source.clone();
        let st = self.settings;
        let r = match st.method {
            SolveMethod::Icgn => s.solve_icgn(Some(p0), st.tolerance, st.max_norm, st.max_iterations),
            SolveMethod::Fagn => s.solve_fagn(Some(p0), st.tolerance, st.max_norm, st.max_iterations),
        };
        r.ok()?;
        s.solution().filter(|sol| sol.result.converged)?;
        Some(s)
    }

    fn n_kept(&self, cuts: &[SeamLine], gap: f64) -> usize {
        let mut lm = self.local_mask.clone();
        for l in cuts {
            lm.cut_half_plane(l.theta, l.c, gap);
        }
        lm.coords.nrows()
    }
}

/// Order-1 warp `p = [u, v, u_x, v_x, u_y, v_y]` carried to a point shifted by `d`.
pub(crate) fn shift_warp(p: &[f64], d: [f64; 2]) -> Vec<f64> {
    let mut q = p.to_vec();
    q[0] += p[2] * d[0] + p[4] * d[1];
    q[1] += p[3] * d[0] + p[5] * d[1];
    q
}

fn with_zero_quadratic(p: &[f64]) -> Vec<f64> {
    let mut q = p[..6].to_vec();
    q.extend_from_slice(&[0.0; 6]);
    q
}

fn pick(pass: &SeamPass, kind: EdgeKind) -> Option<SeamLine> {
    match kind {
        EdgeKind::Hinge => pass.hinge,
        EdgeKind::Jump => pass.jump,
    }
}

impl Subset {
    /// Speckle correlation length of this subset's reference image.
    pub(crate) fn correlation_length(&self) -> Result<f64, Error> {
        let f = self.f_img.as_ref().ok_or_else(|| Error::InvalidInput("images unavailable".into()))?;
        correlation_length(f)
    }

    fn check_seam_inputs(&self) -> Result<(), Error> {
        if self.subset_order != 1 {
            return Err(Error::InvalidInput("seam investigation needs an order-1 subset".into()));
        }
        let sol = self.solution().ok_or_else(|| Error::NotSolved("Subset".into()))?;
        if sol.seams.is_some() {
            return Err(Error::InvalidInput("subset already carries a seam correction; re-solve it first".into()));
        }
        for img in [&self.f_img, &self.g_img] {
            require_lossless(img.as_ref().ok_or_else(|| Error::InvalidInput("images unavailable".into()))?)?;
        }
        Ok(())
    }

    /// Investigate and correct this solved order-1 subset (see the module
    /// docs). On return the subset is the corrected one: its centre (after
    /// any move), mask (after cuts), order and solution all change
    /// accordingly, and the solution carries a [`SeamRecord`] with the plain
    /// solve.
    ///
    /// `local_mask` / `global_mask` must be the masks this subset was built
    /// with. `h` is the reference image's speckle correlation length
    /// (`None`: measured here).
    pub fn solve_seams<'a>(
        &mut self,
        local_mask: &'a LocalMask,
        global_mask: Option<ArrayView2<'a, u8>>,
        h: Option<f64>,
        cfg: &SeamConfig,
    ) -> Result<(), Error> {
        self.solve_seams_with(local_mask, global_mask, h, cfg, &|_, d| d)
    }

    /// [`Subset::solve_seams`] with a rule for the moves: given the current
    /// centre and the full shift to the centre of area of the kept pixels,
    /// `mover` returns the shift to apply (all of it, part of it, or none; a
    /// mesh limits it so no element is distorted).
    pub(crate) fn solve_seams_with<'a>(
        &mut self,
        local_mask: &'a LocalMask,
        global_mask: Option<ArrayView2<'a, u8>>,
        h: Option<f64>,
        cfg: &SeamConfig,
        mover: &Mover<'_>,
    ) -> Result<(), Error> {
        if let Some(pending) = self.seam_first(local_mask, global_mask, h, cfg)? {
            self.seam_rest(local_mask, global_mask, cfg, pending, mover)?;
        }
        Ok(())
    }

    /// Pass 1 and its correction. If the decision is a cut, the subset is
    /// left cut but not yet moved and the returned [`SeamPending`] carries
    /// the record on to [`Subset::seam_rest`]; otherwise the subset is final
    /// and carries its record.
    pub(crate) fn seam_first<'a>(
        &mut self,
        local_mask: &'a LocalMask,
        global_mask: Option<ArrayView2<'a, u8>>,
        h: Option<f64>,
        cfg: &SeamConfig,
    ) -> Result<Option<SeamPending>, Error> {
        self.check_seam_inputs()?;
        let h = match h {
            Some(h) => h,
            None => self.correlation_length()?,
        };
        let levels = cfg.levels.unwrap_or_else(|| SeamLevels::image_free(local_mask.n_px, h));
        let plain = self.solution().cloned().expect("checked");
        let b = self.builder(local_mask, global_mask, Settings::of(&plain.result));
        let mut rec = SeamRecord {
            plain_coord: self.coord,
            plain_p: plain.result.p.clone(),
            moved: None,
            action: SeamAction::Nothing,
            lines: Vec::new(),
            gap: cfg.gap,
            passes: Vec::new(),
            h,
            levels,
            flags: Vec::new(),
        };
        if !plain.result.converged {
            rec.flags.push(SeamFlag::PlainNotConverged);
            self.finish(None, rec)?;
            return Ok(None);
        }

        let pass = investigate_pass(self, h, &levels.pass1, SeamStage::Pass1)?;
        rec.passes.push(pass);
        let p = plain.result.p.clone();
        let coord = self.coord;
        match pass.decision {
            SeamDecision::Nothing => {}
            SeamDecision::Order2 => {
                if cfg.allow_order2 {
                    match b.solve(coord, &[], cfg.gap, 2, &with_zero_quadratic(&p)) {
                        Some(s) => {
                            rec.action = SeamAction::Order2;
                            self.finish(Some(s), rec)?;
                            return Ok(None);
                        }
                        None => rec.flags.push(SeamFlag::Order2Failed),
                    }
                }
            }
            SeamDecision::Cut(kind) => {
                let l1 = pick(&pass, kind).expect("a cut has its line");
                match b.solve(coord, &[l1], cfg.gap, 1, &p) {
                    None => rec.flags.push(SeamFlag::CutFailed),
                    Some(cut) => {
                        rec.action = SeamAction::Cut;
                        *self = cut;
                        return Ok(Some(SeamPending { rec, lines: vec![l1], settings: b.settings }));
                    }
                }
            }
        }
        self.finish(None, rec)?;
        Ok(None)
    }

    /// After the first cut: move to the kept side's centre of area and
    /// re-solve, then pass 2 on the moved subset (a second cut, which is
    /// moved and re-solved again, or order 2 on the cut subset, or nothing).
    pub(crate) fn seam_rest<'a>(
        &mut self,
        local_mask: &'a LocalMask,
        global_mask: Option<ArrayView2<'a, u8>>,
        cfg: &SeamConfig,
        pending: SeamPending,
        mover: &Mover<'_>,
    ) -> Result<(), Error> {
        let SeamPending { mut rec, mut lines, settings } = pending;
        let b = self.builder(local_mask, global_mask, settings);
        self.recentre(&b, &mut lines, &mut rec, cfg, mover);

        let p2 = investigate_pass(self, rec.h, &rec.levels.pass2, SeamStage::Pass2)?;
        rec.passes.push(p2);
        let cut_p = self.solution().expect("solved").result.p.clone();
        let coord = self.coord;
        match p2.decision {
            SeamDecision::Nothing => {}
            SeamDecision::Order2 => {
                if cfg.allow_order2 {
                    match b.solve(coord, &lines, cfg.gap, 2, &with_zero_quadratic(&cut_p)) {
                        Some(s) => {
                            *self = s;
                            rec.action = SeamAction::CutOrder2;
                        }
                        None => rec.flags.push(SeamFlag::Order2Failed),
                    }
                }
            }
            SeamDecision::Cut(k2) if cfg.max_cuts >= 2 => {
                let l2 = pick(&p2, k2).expect("a cut has its line");
                let mut both = lines.clone();
                both.push(l2);
                if (b.n_kept(&both, cfg.gap) as f64) < cfg.min_keep * local_mask.n_px as f64 {
                    rec.flags.push(SeamFlag::SecondCutTooSmall);
                } else {
                    match b.solve(coord, &both, cfg.gap, 1, &cut_p) {
                        Some(s) => {
                            *self = s;
                            lines = both;
                            rec.action = SeamAction::Cut2;
                            self.recentre(&b, &mut lines, &mut rec, cfg, mover);
                        }
                        None => rec.flags.push(SeamFlag::CutFailed),
                    }
                }
            }
            SeamDecision::Cut(_) => {}
        }
        rec.lines = lines;
        self.finish(None, rec)
    }

    /// Move this cut subset towards the centre of area of the pixels it
    /// keeps (as far as `mover` allows), with its cut lines fixed in the
    /// image, and solve it again there. On failure it stays where it was.
    fn recentre(&mut self, b: &Builder, lines: &mut Vec<SeamLine>, rec: &mut SeamRecord, cfg: &SeamConfig, mover: &Mover<'_>) {
        if !cfg.recentre {
            return;
        }
        let Some(fc) = self.f_coords.as_ref() else { return };
        let n = fc.nrows() as f64;
        let d = [fc.column(0).sum() / n - self.coord[0], fc.column(1).sum() / n - self.coord[1]];
        let shift = mover(self.coord, d);
        if shift[0].hypot(shift[1]) < 1e-9 {
            rec.flags.push(SeamFlag::MoveBlocked);
            return;
        }
        let moved: Vec<SeamLine> = lines
            .iter()
            .map(|l| SeamLine { c: l.c - (l.theta.cos() * shift[0] + l.theta.sin() * shift[1]), ..*l })
            .collect();
        let p = self.solution().expect("solved").result.p.clone();
        let coord = [self.coord[0] + shift[0], self.coord[1] + shift[1]];
        match b.solve(coord, &moved, cfg.gap, 1, &shift_warp(&p, shift)) {
            Some(s) => {
                if (shift[0] - d[0]).hypot(shift[1] - d[1]) > 1e-9 {
                    rec.flags.push(SeamFlag::MoveLimited);
                }
                *self = s;
                *lines = moved;
                let total = [coord[0] - rec.plain_coord[0], coord[1] - rec.plain_coord[1]];
                rec.moved = Some(total);
            }
            None => rec.flags.push(SeamFlag::MoveFailed),
        }
    }

    fn builder<'a>(&self, local_mask: &'a LocalMask, global_mask: Option<ArrayView2<'a, u8>>, settings: Settings) -> Builder<'a> {
        Builder {
            local_mask,
            global_mask,
            f_img: Arc::clone(self.f_img.as_ref().expect("checked")),
            g_img: Arc::clone(self.g_img.as_ref().expect("checked")),
            settings,
            time: self.time,
            time_source: self.time_source.clone(),
        }
    }

    /// Adopt the corrected subset (if any) and attach the record.
    fn finish(&mut self, fin: Option<Subset>, rec: SeamRecord) -> Result<(), Error> {
        if let Some(s) = fin {
            // Built from the same images, time and masks, so it replaces this
            // subset wholesale.
            *self = s;
        }
        let mut sol = self.solution().cloned().expect("solved");
        sol.seams = Some(rec);
        self.set_solution(sol);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rtol: f64) -> bool {
        (a - b).abs() <= rtol * b.abs().max(1e-300)
    }

    /// F quantiles and the small-sample level against `scipy.stats.f.ppf`.
    #[test]
    fn f_level_matches_scipy() {
        assert!(close(f_ppf(0.5, 3.0, 7.5), 0.8651040512738714, 1e-9));
        assert!(close(f_ppf(0.9, 2.0, 50.0), 2.411954903579627, 1e-9));
        for (n, want) in [(10.0, 205.29267562562657), (21.7, 33.500124482365095), (40.0, 23.305776785699837),
                          (100.0, 18.957260963291656)] {
            let got = f_level(n, 6);
            assert!(close(got, want, 1e-9), "N = {n}: {got} vs {want}");
        }
        assert!(f_level(7.0, 6).is_infinite());
    }

    /// Image-free levels against `deployed.py::image_free_levels(40, 6.0)`.
    #[test]
    fn image_free_levels_match_sandbox() {
        let n_disc = LocalMask::circle(40).unwrap().n_px;
        assert_eq!(n_disc, 5025); // speckle_analyse.disc_px(40)
        let lv = SeamLevels::image_free(n_disc, 6.0);
        let want1 = PassLevels { bend: 32.80397333124534, hinge: 46.30940244813814, jump: 43.80331188906506,
                                 edge: 51.86824940634522, margin: 40.581205947793734 };
        let want2 = PassLevels { bend: 32.80397333124534, hinge: 55.00684075083169, jump: 52.64231700874733,
                                 edge: 63.17944575728352, margin: 51.67630863200445 };
        for (got, want) in [(lv.pass1, want1), (lv.pass2, want2)] {
            for (g, w) in [(got.bend, want.bend), (got.hinge, want.hinge), (got.jump, want.jump),
                           (got.edge, want.edge), (got.margin, want.margin)] {
                assert!(close(g, w, 1e-9), "{g} vs {w}");
            }
        }
    }

    #[test]
    fn decide_margin_rule() {
        let lv = PassLevels { bend: 30.0, hinge: 40.0, jump: 40.0, edge: 50.0, margin: 40.0 };
        let line = |kind, t| Some(SeamLine { kind, theta: 0.0, c: 10.0, t });
        // strong edge, weak bend: cut along the larger T / alarm
        assert_eq!(decide(10.0, line(EdgeKind::Hinge, 100.0), line(EdgeKind::Jump, 60.0), &lv),
                   SeamDecision::Cut(EdgeKind::Hinge));
        // edge fires but bending explains nearly as much: order 2
        assert_eq!(decide(80.0, line(EdgeKind::Hinge, 100.0), None, &lv), SeamDecision::Order2);
        // only bending fires
        assert_eq!(decide(35.0, line(EdgeKind::Hinge, 20.0), None, &lv), SeamDecision::Order2);
        assert_eq!(decide(5.0, None, None, &lv), SeamDecision::Nothing);
        // tie on T / alarm keeps hinge (Python max() keeps the first)
        assert_eq!(decide(0.0, line(EdgeKind::Hinge, 100.0), line(EdgeKind::Jump, 100.0), &lv),
                   SeamDecision::Cut(EdgeKind::Hinge));
    }

    #[test]
    fn cut_half_plane_keeps_centre_side_with_gap() {
        let full = LocalMask::circle(10).unwrap();
        // vertical line x = 4: keep x < 4 - 3 = 1
        let mut m = full.clone();
        m.cut_half_plane(0.0, 4.0, 3.0);
        assert!(m.coords.rows().into_iter().all(|r| r[0] < 1.0));
        let want = full.coords.rows().into_iter().filter(|r| r[0] < 1.0).count();
        assert_eq!(m.coords.nrows(), want);
        assert_eq!(m.m_n_px, Some(want));
        assert_eq!(m.subset_mask.iter().filter(|&&v| v != 0).count(), want);
        // negative c: keep s > c + gap
        let mut m = full.clone();
        m.cut_half_plane(std::f64::consts::FRAC_PI_2, -6.5, 3.0);
        assert!(m.coords.rows().into_iter().all(|r| r[1] > -3.5));
        assert!(m.coords.rows().into_iter().any(|r| r[1] == -3.0));
    }

    #[test]
    fn shift_warp_carries_affine_motion() {
        let p = [1.0, -2.0, 0.01, 0.02, -0.03, 0.04];
        let q = shift_warp(&p, [3.0, -4.0]);
        assert!((q[0] - (1.0 + 0.01 * 3.0 + (-0.03) * (-4.0))).abs() < 1e-15);
        assert!((q[1] - (-2.0 + 0.02 * 3.0 + 0.04 * (-4.0))).abs() < 1e-15);
        assert_eq!(&q[2..], &p[2..]);
    }

    /// Synthetic disc with speckle-like gradients (no images), as
    /// `break_test.py::_synthetic_subset`.
    fn synthetic(r: i32, seed: u64) -> SeamData {
        use rand::{rngs::SmallRng, Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        let (mut x, mut y, mut fx, mut fy) = (vec![], vec![], vec![], vec![]);
        // smooth random texture: sum of random plane waves
        let waves: Vec<(f64, f64, f64)> = (0..40)
            .map(|_| (rng.gen_range(-0.6..0.6), rng.gen_range(-0.6..0.6), rng.gen_range(0.0..6.28)))
            .collect();
        for yy in -r..=r {
            for xx in -r..=r {
                if xx * xx + yy * yy > r * r {
                    continue;
                }
                let (xf, yf) = (xx as f64, yy as f64);
                let (mut gx, mut gy) = (0.0, 0.0);
                for &(kx, ky, ph) in &waves {
                    let c = (kx * xf + ky * yf + ph).cos();
                    gx -= kx * c;
                    gy -= ky * c;
                }
                x.push(xf);
                y.push(yf);
                fx.push(gx);
                fy.push(gy);
            }
        }
        let n = x.len();
        SeamData { x, y, fx, fy, e: vec![0.0; n], delta_f: 1.0 }
    }

    /// Residual of an injected hinge, made orthogonal to the affine columns
    /// (a converged affine solve), plus a little noise.
    fn inject_hinge(d: &mut SeamData, theta: f64, c: f64, beta: [f64; 2], noise: f64, seed: u64) {
        use rand::{rngs::SmallRng, SeedableRng};
        use rand_distr::{Distribution, Normal};
        let mut rng = SmallRng::seed_from_u64(seed);
        let nd = Normal::new(0.0, 1.0).unwrap();
        let n = d.n();
        let (ct, st) = (theta.cos(), theta.sin());
        let mut e: Vec<f64> = (0..n)
            .map(|i| {
                let s = d.x[i] * ct + d.y[i] * st;
                let h = if s > c { s - c } else { 0.0 };
                h * (beta[0] * d.fx[i] + beta[1] * d.fy[i])
            })
            .collect();
        let sd = (e.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        e.iter_mut().for_each(|v| *v += noise * sd * nd.sample(&mut rng));
        let mut jj = vec![0.0; 36];
        let mut je = vec![0.0; 6];
        for i in 0..n {
            let j = d.j_row(i);
            for p in 0..6 {
                for q in 0..6 {
                    jj[p * 6 + q] += j[p] * j[q];
                }
                je[p] += j[p] * e[i];
            }
        }
        let a = solve_dense(jj, 6, &je).unwrap();
        for i in 0..n {
            let j = d.j_row(i);
            e[i] -= (0..6).map(|p| j[p] * a[p]).sum::<f64>();
        }
        d.e = e;
    }

    /// The line search recovers an injected hinge line (break_test.py self-check).
    #[test]
    fn locate_recovers_injected_hinge() {
        for (theta_deg, c) in [(37.5_f64, -8.0), (130.0, 3.0), (90.0, 12.0)] {
            let mut d = synthetic(32, 7);
            let theta = theta_deg.to_radians();
            inject_hinge(&mut d, theta, c, [0.02, -0.01], 0.3, 1);
            let (th, cc) = locate(&d, EdgeKind::Hinge).expect("a line");
            let dth = ((th - theta + std::f64::consts::FRAC_PI_2).rem_euclid(std::f64::consts::PI)
                - std::f64::consts::FRAC_PI_2)
                .abs()
                .to_degrees();
            let c_cmp = if (th - theta).cos() < 0.0 { -cc } else { cc };
            assert!(dth < 2.5 && (c_cmp - c).abs() < 1.0, "θ {theta_deg}: got {} / {cc}", th.to_degrees());
        }
    }

    /// The robust score is large for an injected hinge and small for noise.
    #[test]
    fn robust_score_separates_signal_from_noise() {
        let mut d = synthetic(32, 3);
        inject_hinge(&mut d, 0.4, 5.0, [0.03, 0.0], 0.2, 2);
        let t_sig = robust_t(&d, &d.edge_columns(EdgeKind::Hinge, 0.4, 5.0), 2.0);
        let mut d0 = synthetic(32, 3);
        inject_hinge(&mut d0, 0.4, 5.0, [0.0, 0.0], 0.0, 2);
        // pure noise residual
        use rand::{rngs::SmallRng, SeedableRng};
        use rand_distr::{Distribution, Normal};
        let mut rng = SmallRng::seed_from_u64(9);
        let nd = Normal::new(0.0, 1e-3).unwrap();
        d0.e = (0..d0.n()).map(|_| nd.sample(&mut rng)).collect();
        let t_null = robust_t(&d0, &d0.edge_columns(EdgeKind::Hinge, 0.4, 5.0), 2.0);
        assert!(t_sig > 100.0 && t_null < 30.0, "signal {t_sig}, null {t_null}");
    }

    #[test]
    fn lossless_check() {
        let mut img = Image::from_array(Array2::zeros((40, 40)), 20);
        assert!(require_lossless(&img).is_ok());
        img.filepath = Some("a/b/ref.PNG".into());
        assert!(require_lossless(&img).is_ok());
        img.filepath = Some("a/b/ref.jpg".into());
        assert!(require_lossless(&img).is_err());
    }
}
