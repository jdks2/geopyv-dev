//! Subset module for geopyv-dev.
//!
//! Performs DIC/PIV algorithms on pixel patches.
//!
//! # Solvers
//!
//! - **ICGN** (Inverse Compositional Gauss-Newton): Hessian precomputed once
//!   from the reference image gradient. Warp updated via matrix composition.
//!   Fast per-iteration; standard choice for DIC.
//! - **FAGN** (Forward Additive Gauss-Newton): Hessian recomputed each
//!   iteration from the target image gradient. Slower but sometimes more
//!   robust for large deformations.
//!
//! # Warp orders
//!
//! | Order | Parameters | Description |
//! |-------|------------|-------------|
//! | 1     | 6          | `[u, v, u_x, v_x, u_y, v_y]` |
//! | 2     | 12         | `[u, v, u_x, v_x, u_y, v_y, u_xx, v_xx, u_xy, v_xy, u_yy, v_yy]` |
//!
//! # Coordinate convention
//!
//! `coord[0]` and `coord[1]` are the two components of the subset centre, in
//! the same order as `template.coords` columns. `f_coords[:,0]` and
//! `f_coords[:,1]` are the two coordinate components of each subset pixel.
//! `bspline_eval` uses `coords[:,0]` as the QCQT column index and
//! `coords[:,1]` as the QCQT row index (i.e. intensities).
//!
//! # Tolerance tiers (per plan)
//! - Tier B  rtol = 1e-8   intensity interpolation (same B-spline as image)
//! - Tier C  rtol = 1e-5   solver outputs (ZNCC, warp at convergence)

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use ndarray::{Array1, Array2, ArrayView2, ArrayView3, Axis};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::{image::Image, masks::{LocalMask, MaskShape}, mesh::SolveMethod, Error};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// 3×3 warp matrix for order-1 homogeneous representation.
type WarpMat3 = SMatrix<f64, 3, 3>;
/// 6×6 warp matrix for order-2 homogeneous representation.
type WarpMat6 = SMatrix<f64, 6, 6>;
/// 6-element B-spline polynomial vector.
type SplineVec = SVector<f64, 6>;
/// 6×6 QCQT block (B-spline coefficient matrix for one pixel).
type QcqtBlock = SMatrix<f64, 6, 6>;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Serialisable summary of the local mask used by a subset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskSummary {
    pub shape: MaskShape,
    /// Radius (Circle) or half-side-length (Square) in pixels.
    pub size: usize,
    /// Number of active pixels in the mask.
    pub n_px: usize,
}

/// Reference subset quantities.
///
/// Created by [`Subset::new`] (full constructor) or
/// [`Subset::from_subset_solution`] (partial constructor, no images).
pub struct Subset {
    // Always populated — image-independent or recoverable from SubsetSolution.
    /// Centre coordinate `[coord0, coord1]`.
    pub coord: [f64; 2],
    /// Warp order: 1 (affine, 6 params) or 2 (quadratic, 12 params).
    pub subset_order: usize,
    /// Local mask shape/size/n_px summary.
    pub mask: MaskSummary,
    /// Sum of squared intensity gradients (SSSIG quality metric).
    pub sssig: f64,
    /// `sqrt(Σ(f_i − f_m)²)` — normalisation factor for ZNSSD. Equal to `std_dev * sqrt(n_px)`.
    pub delta_f: f64,

    // Image handles — None when images unavailable (load-from-disk path).
    pub f_img: Option<Arc<Image>>,
    pub g_img: Option<Arc<Image>>,

    // Image-dependent — None when images unavailable.
    /// Subset pixel coordinates, shape `(n_px, 2)`.
    pub f_coords: Option<Array2<f64>>,
    /// Reference subset intensities (B-spline interpolated), shape `(n_px,)`.
    pub f: Option<Array1<f64>>,
    /// Mean reference intensity.
    pub f_m: Option<f64>,
    /// Reference image gradients at `f_coords`, shape `(n_px, 2)`.
    pub grad_f: Option<Array2<f64>>,
    /// Standard deviation of reference intensities.
    pub sigma_intensity: Option<f64>,

    /// Result of the most recent [`Subset::solve_icgn`]/[`Subset::solve_fagn`]
    /// call. `None` until solved.
    solution: Option<SubsetSolution>,
}

/// Serialisable result of a single-coordinate, single image-pair DIC solve.
///
/// Saved as [`crate::io::GeopyvObject::Subset`] to a `.pyv` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsetSolution {
    pub coord: [f64; 2],
    #[serde(alias = "template")]
    pub mask: MaskSummary,
    pub ref_image: PathBuf,
    pub target_image: PathBuf,
    pub result: SolveResult,
    /// Standard deviation of reference intensities: `delta_f / sqrt(n_px)`.
    #[serde(default)]
    pub std_dev: f64,
    /// Sum of squared intensity gradients (quality metric).
    #[serde(default)]
    pub sssig: f64,
    /// `sqrt(Σ(f_i − f_m)²)` — normalisation factor for ZNSSD.
    #[serde(default)]
    pub delta_f: f64,
}

/// Output of a DIC solve ([`Subset::solve_icgn`] / [`Subset::solve_fagn`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveResult {
    /// Final warp parameter vector (6 or 12 elements).
    pub p: Vec<f64>,
    /// Final ZNCC score: 1 = perfect correlation.
    pub c_zncc: f64,
    /// Final ZNSSD score: 0 = perfect correlation.
    pub c_znssd: f64,
    /// Number of iterations completed.
    pub iterations: usize,
    /// Whether the norm criterion `||Δp|| < max_norm` was satisfied.
    pub converged: bool,
    /// Whether `converged && c_zncc >= tolerance` — both the norm criterion
    /// and the quality threshold passed. The single trust signal read by the
    /// RG mesh solver: a subset that exhausted `max_iterations` without
    /// converging is never treated as trustworthy here, even if its
    /// (possibly garbage) final `p` happens to still correlate well.
    #[serde(default, alias = "solved")]
    pub quality_ok: bool,
    /// Per-iteration history: `(iteration, norm, C_ZNCC, C_ZNSSD)`.
    pub history: Vec<(usize, f64, f64, f64)>,
    #[serde(default)]
    pub max_norm: f64,
    #[serde(default)]
    pub tolerance: f64,
    /// Warp order: 1 (affine, 6 params) or 2 (quadratic, 12 params).
    #[serde(default)]
    pub subset_order: usize,
    /// Iteration limit passed to the solver.
    #[serde(default)]
    pub max_iterations: usize,
    /// Which solver produced this result.
    #[serde(default = "default_solve_method")]
    pub method: SolveMethod,
    /// Gradient-weighted omitted-mode warp-adequacy score `(eta_u, eta_v)`
    /// -- see [`omitted_mode_diagnostic`]. `Some` for `subset_order` 1 or 2;
    /// `None` for any other order, and for results loaded from a
    /// pre-existing `.pyv` saved before this field existed.
    #[serde(default)]
    pub eta_omitted: Option<(f64, f64)>,
}

fn default_solve_method() -> SolveMethod {
    SolveMethod::Icgn
}

// ---------------------------------------------------------------------------
// B-spline helpers
// ---------------------------------------------------------------------------

/// Extract the 6×6 QCQT block for pixel (row = `yf`, col = `xf`).
///
/// QCQT is stored block-contiguous `(rows, cols, 36)` — the block is one
/// contiguous 36-run, `[r*6 + c]` (see [`crate::image::Image::qcqt`]).
#[inline]
fn qcqt_block(qcqt: &ArrayView3<f64>, yf: usize, xf: usize) -> QcqtBlock {
    let b = qcqt.index_axis(Axis(0), yf);
    let b = b.index_axis(Axis(0), xf); // contiguous 36-run for pixel (yf, xf)
    QcqtBlock::from_fn(|r, c| b[r * 6 + c])
}

/// Evaluate the bi-quintic B-spline at sub-pixel coordinate `(x, y)`.
///
/// `x` drives the QCQT **column** index (`xf = x.floor()`).
/// `y` drives the QCQT **row**    index (`yf = y.floor()`).
///
/// Formula: `intensity = dy_vec^T · QCQT_block · dx_vec`
/// where `dx_vec = [1, dx, dx², dx³, dx⁴, dx⁵]` and similarly for `dy_vec`.
///
/// Matches `_intensity` in `_subset.cpp` exactly, including the clamp to avoid
/// out-of-bounds access. `(x, y)` are clamped into the valid domain *before*
/// `xf`/`dx` are derived (not just `xf` on its own) — clamping only `xf`
/// while leaving `dx = x - xf` unbounded lets a far-out-of-range query
/// produce a `dx` of thousands or millions, and `dx⁵` in the quintic basis
/// then explodes instead of saturating. Clamping `x`/`y` first keeps
/// `dx`/`dy` in `[0, 1)` always, so in-bounds queries are completely
/// unaffected and out-of-bounds ones saturate to the boundary pixel's edge
/// value instead of blowing up.
#[inline]
fn bspline_eval(x: f64, y: f64, qcqt: &ArrayView3<f64>) -> f64 {
    let qrows = qcqt.shape()[0];
    let qcols = qcqt.shape()[1];
    let x = x.clamp(0.0, qcols as f64 - 1e-9);
    let y = y.clamp(0.0, qrows as f64 - 1e-9);
    let xf = x.floor() as usize;
    let yf = y.floor() as usize;
    let dx = x - xf as f64;
    let dy = y - yf as f64;

    let dx_v = poly_vec(dx);
    let dy_v = poly_vec(dy);
    let block = qcqt_block(qcqt, yf, xf);
    dy_v.dot(&(block * dx_v))
}

/// Evaluate the B-spline gradient in both coordinate directions at `(x, y)`.
///
/// Returns `[grad_x, grad_y]`.
/// - `grad_x`: derivative in the `x` (column) direction.
///   `dx_vec = [0, 1, 0, 0, 0, 0]`, `dy_vec = [1, dy, dy², ...]`.
/// - `grad_y`: derivative in the `y` (row) direction.
///   `dx_vec = [1, dx, dx², ...]`, `dy_vec = [0, 1, 0, 0, 0, 0]`.
///
/// Matches `_grad` in `_subset.cpp` exactly. Clamps `(x, y)` before deriving
/// `xf`/`dx` for the same reason as `bspline_eval` — see its doc comment.
#[inline]
fn bspline_grad(x: f64, y: f64, qcqt: &ArrayView3<f64>) -> [f64; 2] {
    let qrows = qcqt.shape()[0];
    let qcols = qcqt.shape()[1];
    let x = x.clamp(0.0, qcols as f64 - 1e-9);
    let y = y.clamp(0.0, qrows as f64 - 1e-9);
    let xf = x.floor() as usize;
    let yf = y.floor() as usize;
    let dx = x - xf as f64;
    let dy = y - yf as f64;
    let block = qcqt_block(qcqt, yf, xf);

    // x-direction: derivative basis in x, standard basis in y
    let dx_deriv = SplineVec::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    let dy_std = poly_vec(dy);
    let grad_x = dy_std.dot(&(block * dx_deriv));

    // y-direction: standard basis in x, derivative basis in y
    let dx_std = poly_vec(dx);
    let dy_deriv = SplineVec::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    let grad_y = dy_deriv.dot(&(block * dx_std));

    [grad_x, grad_y]
}

/// Fused [`bspline_eval`] + [`bspline_grad`] — one clamp, one block gather,
/// one pair of basis vectors (`geopyv_dev_fresh/layer_rg_plan.md` §11.3).
///
/// Bit-identical to calling both separately: `value` uses the exact
/// expression `bspline_eval` does; `grad_x` / `grad_y` use the exact
/// expressions `bspline_grad` does (`block · dx_v` and `block · deriv` are
/// each formed once and reused across the three dot products).
#[inline]
fn bspline_eval_grad(x: f64, y: f64, qcqt: &ArrayView3<f64>) -> (f64, [f64; 2]) {
    let qrows = qcqt.shape()[0];
    let qcols = qcqt.shape()[1];
    let x = x.clamp(0.0, qcols as f64 - 1e-9);
    let y = y.clamp(0.0, qrows as f64 - 1e-9);
    let xf = x.floor() as usize;
    let yf = y.floor() as usize;
    let dx = x - xf as f64;
    let dy = y - yf as f64;
    let block = qcqt_block(qcqt, yf, xf);

    let dx_v = poly_vec(dx);
    let dy_v = poly_vec(dy);
    let deriv = SplineVec::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);

    let bx = block * dx_v; // block · [1, dx, …, dx⁵]
    let bd = block * deriv; // block · [0, 1, 0, 0, 0, 0]

    let value = dy_v.dot(&bx);
    let grad_x = dy_v.dot(&bd);
    let grad_y = deriv.dot(&bx);
    (value, [grad_x, grad_y])
}

/// Build the B-spline polynomial basis vector `[1, t, t², t³, t⁴, t⁵]`.
#[inline]
fn poly_vec(t: f64) -> SplineVec {
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    SplineVec::new(1.0, t, t2, t3, t4, t5)
}

// ---------------------------------------------------------------------------
// Warp application
// ---------------------------------------------------------------------------

/// Apply a first- or second-order warp to a set of reference coordinates.
///
/// `coord` = subset centre; `p` = warp vector (6 or 12 elements);
/// `f_coords` shape = `(n, 2)`.
///
/// Matches `_g_coords` in `_subset.cpp`.
fn apply_warp(coord: [f64; 2], p: &[f64], f_coords: &Array2<f64>) -> Array2<f64> {
    let mut gc = Array2::zeros((f_coords.nrows(), 2));
    apply_warp_into(&mut gc, coord, p, f_coords);
    gc
}

/// [`apply_warp`] writing into a caller-owned `(n, 2)` buffer — lets the
/// ICGN loop reuse one allocation across iterations
/// (`geopyv_dev_fresh/layer_rg_plan.md` §11.4). Every element of `gc` is
/// overwritten, so its prior contents are irrelevant.
fn apply_warp_into(gc: &mut Array2<f64>, coord: [f64; 2], p: &[f64], f_coords: &Array2<f64>) {
    let n = f_coords.nrows();
    let xc = coord[0];
    let yc = coord[1];

    if p.len() <= 7 {
        // Order 1
        let (u, v) = (p[0], p[1]);
        let (ux, vx, uy, vy) = (p[2], p[3], p[4], p[5]);
        for i in 0..n {
            let x = f_coords[[i, 0]];
            let y = f_coords[[i, 1]];
            let (dx, dy) = (x - xc, y - yc);
            gc[[i, 0]] = x + u + ux * dx + uy * dy;
            gc[[i, 1]] = y + v + vx * dx + vy * dy;
        }
    } else {
        // Order 2
        let (u, v) = (p[0], p[1]);
        let (ux, vx, uy, vy) = (p[2], p[3], p[4], p[5]);
        let (uxx, vxx, uxy, vxy, uyy, vyy) = (p[6], p[7], p[8], p[9], p[10], p[11]);
        for i in 0..n {
            let x = f_coords[[i, 0]];
            let y = f_coords[[i, 1]];
            let (dx, dy) = (x - xc, y - yc);
            gc[[i, 0]] = x + u + ux * dx + uy * dy
                + 0.5 * uxx * dx * dx + uxy * dx * dy + 0.5 * uyy * dy * dy;
            gc[[i, 1]] = y + v + vx * dx + vy * dy
                + 0.5 * vxx * dx * dx + vxy * dx * dy + 0.5 * vyy * dy * dy;
        }
    }
}

// ---------------------------------------------------------------------------
// Steepest descent images + Hessian
// ---------------------------------------------------------------------------

/// Compute steepest descent images (SDI) for a set of coordinates.
///
/// Returns a `(n, 6)` or `(n, 12)` array depending on `order`.
/// Matches `_sdi` in `_subset.cpp`.
fn steepest_descent(
    coord: [f64; 2],
    coords: &Array2<f64>,
    grad: &Array2<f64>,
    order: usize,
) -> Array2<f64> {
    let n = coords.nrows();
    let m = if order == 1 { 6 } else { 12 };
    let mut sdi = Array2::zeros((n, m));
    for i in 0..n {
        let dx = coords[[i, 0]] - coord[0];
        let dy = coords[[i, 1]] - coord[1];
        let (gx, gy) = (grad[[i, 0]], grad[[i, 1]]);
        sdi[[i, 0]] = gx;
        sdi[[i, 1]] = gy;
        sdi[[i, 2]] = gx * dx;
        sdi[[i, 3]] = gy * dx;
        sdi[[i, 4]] = gx * dy;
        sdi[[i, 5]] = gy * dy;
        if m == 12 {
            sdi[[i, 6]] = gx * 0.5 * dx * dx;
            sdi[[i, 7]] = gy * 0.5 * dx * dx;
            sdi[[i, 8]] = gx * dx * dy;
            sdi[[i, 9]] = gy * dx * dy;
            sdi[[i, 10]] = gx * 0.5 * dy * dy;
            sdi[[i, 11]] = gy * 0.5 * dy * dy;
        }
    }
    sdi
}

/// Gauss-Newton Hessian approximation: `H = SDI^T · SDI` (symmetric).
///
/// Matches `_hessian` in `_subset.cpp`.
fn compute_hessian(sdi: &Array2<f64>) -> DMatrix<f64> {
    let m = sdi.ncols();
    let mut h = DMatrix::zeros(m, m);
    for i in 0..m {
        for j in i..m {
            let dot: f64 = sdi
                .column(i)
                .iter()
                .zip(sdi.column(j).iter())
                .map(|(a, b)| a * b)
                .sum();
            h[(i, j)] = dot;
            h[(j, i)] = dot;
        }
    }
    h
}


// ---------------------------------------------------------------------------
// Parameter update vectors
// ---------------------------------------------------------------------------

/// ICGN parameter increment: `Δp = −H⁻¹ · ∇(ZNSSD)`.
///
/// Matches `_Delta_p_ICGN` in `_subset.cpp`. Kept as the reference
/// implementation the fused [`delta_p_and_znssd_icgn`] is checked against
/// (see `delta_p_and_znssd_icgn_matches_split`); the ICGN loop itself uses
/// the fused version.
#[cfg(test)]
fn delta_p_icgn(
    hessian: &DMatrix<f64>,
    f: &Array1<f64>,
    g: &Array1<f64>,
    f_m: f64,
    g_m: f64,
    delta_f: f64,
    delta_g: f64,
    sdi: &Array2<f64>,
) -> DVector<f64> {
    let m = sdi.ncols();
    let n = f.len();
    let mut grad_z = DVector::zeros(m);
    for j in 0..m {
        for i in 0..n {
            grad_z[j] +=
                sdi[[i, j]] * ((f[i] - f_m) - (delta_f / delta_g) * (g[i] - g_m));
        }
    }
    let inv_h = hessian.clone().try_inverse().expect("ICGN Hessian singular");
    -(inv_h * grad_z)
}

/// Fused ICGN update + ZNSSD for one iteration — see
/// `geopyv_dev_fresh/layer_rg_plan.md` §11.1/§11.2.
///
/// Bit-identical to `-(inv_h * grad_z)` from [`delta_p_icgn`] paired with a
/// separate [`znssd`] call, but:
///
/// - takes a pre-inverted `inv_h` (ICGN's Hessian is loop-invariant, so the
///   caller inverts once instead of once per iteration — §11.1);
/// - runs a single `i`-outer sweep, reusing `f_i − f_m` / `g_i − g_m` for
///   both the `grad_z` bracket and the ZNSSD term instead of two sweeps
///   (§11.2). `i` is the outer loop so each `grad_z[j]` still accumulates
///   in `i`-ascending order (identical to `delta_p_icgn`'s `for j { for i }`)
///   and `znssd_acc` accumulates in the same order as `znssd`'s `.sum()`.
///   The two per-pixel residual forms are kept *separate* (division for the
///   ZNSSD term, `delta_f/delta_g` factor for the bracket) so neither sum's
///   rounding changes; `delta_f / delta_g` is hoisted (was recomputed
///   `n·m` times).
fn delta_p_and_znssd_icgn(
    inv_h: &DMatrix<f64>,
    f: &Array1<f64>,
    g: &Array1<f64>,
    f_m: f64,
    g_m: f64,
    delta_f: f64,
    delta_g: f64,
    sdi: &Array2<f64>,
) -> (DVector<f64>, f64) {
    let m = sdi.ncols();
    let n = f.len();
    let ratio = delta_f / delta_g;
    let mut grad_z = DVector::zeros(m);
    let mut znssd_acc = 0.0_f64;
    for i in 0..n {
        let a = f[i] - f_m;
        let b = g[i] - g_m;
        // ZNSSD per-pixel term — identical grouping to `znssd()`.
        let z = a / delta_f - b / delta_g;
        znssd_acc += z * z;
        // grad_z bracket — identical grouping to `delta_p_icgn()`.
        let bracket = a - ratio * b;
        for j in 0..m {
            grad_z[j] += sdi[[i, j]] * bracket;
        }
    }
    (-(inv_h * grad_z), znssd_acc)
}


/// FAGN parameter increment: `Δp = H⁻¹ · ∇(ZNSSD)`.
///
/// Matches `_Delta_p_FAGN` in `_subset.cpp`.
fn delta_p_fagn(
    hessian: &DMatrix<f64>,
    f: &Array1<f64>,
    g: &Array1<f64>,
    f_m: f64,
    g_m: f64,
    delta_f: f64,
    delta_g: f64,
    sdi: &Array2<f64>,
) -> DVector<f64> {
    let m = sdi.ncols();
    let n = f.len();
    let mut grad_z = DVector::zeros(m);
    for j in 0..m {
        for i in 0..n {
            grad_z[j] +=
                sdi[[i, j]] * ((f[i] - f_m) * (delta_g / delta_f) - (g[i] - g_m));
        }
    }
    let inv_h = hessian.clone().try_inverse().expect("FAGN Hessian singular");
    inv_h * grad_z
}

// ---------------------------------------------------------------------------
// ICGN warp composition
// ---------------------------------------------------------------------------

/// Compositional warp update for ICGN: `p_new = compose(p_old, Δp⁻¹)`.
///
/// Uses the homogeneous warp matrix representation for order 1 (3×3) and
/// order 2 (6×6). Matches `_p_new_ICGN` in `_subset.cpp`.
fn compose_icgn(p: &[f64], delta_p: &[f64]) -> Vec<f64> {
    if p.len() == 6 {
        // Order 1: 3×3 homogeneous warp matrix
        let (u, v) = (p[0], p[1]);
        let (ux, vx, uy, vy) = (p[2], p[3], p[4], p[5]);
        let (du, dv) = (delta_p[0], delta_p[1]);
        let (dux, dvx, duy, dvy) = (delta_p[2], delta_p[3], delta_p[4], delta_p[5]);

        // W_old = [[1+ux, uy, u], [vx, 1+vy, v], [0, 0, 1]]
        let w_old = WarpMat3::from_fn(|r, c| match (r, c) {
            (0, 0) => 1.0 + ux,
            (0, 1) => uy,
            (0, 2) => u,
            (1, 0) => vx,
            (1, 1) => 1.0 + vy,
            (1, 2) => v,
            (2, 2) => 1.0,
            _ => 0.0,
        });
        // W_delta = [[1+dux, duy, du], [dvx, 1+dvy, dv], [0, 0, 1]]
        let w_del = WarpMat3::from_fn(|r, c| match (r, c) {
            (0, 0) => 1.0 + dux,
            (0, 1) => duy,
            (0, 2) => du,
            (1, 0) => dvx,
            (1, 1) => 1.0 + dvy,
            (1, 2) => dv,
            (2, 2) => 1.0,
            _ => 0.0,
        });
        let w_new = w_old * w_del.try_inverse().expect("W_delta singular (order-1)");
        vec![
            w_new[(0, 2)],
            w_new[(1, 2)],
            w_new[(0, 0)] - 1.0,
            w_new[(1, 0)],
            w_new[(0, 1)],
            w_new[(1, 1)] - 1.0,
        ]
    } else {
        // Order 2: 6×6 homogeneous warp matrix
        let (u, v) = (p[0], p[1]);
        let (ux, vx, uy, vy) = (p[2], p[3], p[4], p[5]);
        let (uxx, vxx, uxy, vxy, uyy, vyy) = (p[6], p[7], p[8], p[9], p[10], p[11]);
        let (du, dv) = (delta_p[0], delta_p[1]);
        let (dux, dvx, duy, dvy) = (delta_p[2], delta_p[3], delta_p[4], delta_p[5]);
        let (duxx, dvxx, duxy, dvxy, duyy, dvyy) = (
            delta_p[6], delta_p[7], delta_p[8],
            delta_p[9], delta_p[10], delta_p[11],
        );

        /// Compute the 18 auxiliary terms for the 6×6 homogeneous warp matrix.
        fn s_terms(
            u: f64, v: f64, ux: f64, vx: f64, uy: f64, vy: f64,
            uxx: f64, vxx: f64, uxy: f64, vxy: f64, uyy: f64, vyy: f64,
        ) -> [f64; 18] {
            [
                2.0 * ux + ux * ux + u * uxx,                         // S1
                2.0 * u * uxy + 2.0 * (1.0 + ux) * uy,               // S2
                uy * uy + u * uyy,                                     // S3
                2.0 * u * (1.0 + ux),                                  // S4
                2.0 * u * uy,                                          // S5
                u * u,                                                 // S6
                0.5 * v * uxx + 2.0 * (1.0 + ux) * vx + u * vxx,    // S7
                uy * vx + ux * vy + v * uxy + u * vxy + vy + ux,     // S8
                0.5 * (v * uyy + 2.0 * uy * (1.0 + vy) + u * vyy),  // S9
                v + v * ux + u * vx,                                   // S10
                u + v * uy + u * vy,                                   // S11
                u * v,                                                 // S12
                vx * vx + v * vxx,                                    // S13
                2.0 * v * vxy + 2.0 * vx * (1.0 + vy),              // S14
                2.0 * vy + vy * vy + v * vyy,                        // S15
                2.0 * v * vx,                                         // S16
                2.0 * v * (1.0 + vy),                                 // S17
                v * v,                                                 // S18
            ]
        }

        let s = s_terms(u, v, ux, vx, uy, vy, uxx, vxx, uxy, vxy, uyy, vyy);
        let ds = s_terms(du, dv, dux, dvx, duy, dvy, duxx, dvxx, duxy, dvxy, duyy, dvyy);

        #[rustfmt::skip]
        let w_old = WarpMat6::from_row_slice(&[
            1.0+s[0], s[1],     s[2],       s[3],    s[4],    s[5],
            s[6],     1.0+s[7], s[8],       s[9],    s[10],   s[11],
            s[12],    s[13],    1.0+s[14],  s[15],   s[16],   s[17],
            0.5*uxx,  uxy,      0.5*uyy,    1.0+ux,  uy,      u,
            0.5*vxx,  vxy,      0.5*vyy,    vx,      1.0+vy,  v,
            0.0,      0.0,      0.0,        0.0,     0.0,     1.0,
        ]);
        #[rustfmt::skip]
        let w_del = WarpMat6::from_row_slice(&[
            1.0+ds[0], ds[1],     ds[2],       ds[3],   ds[4],   ds[5],
            ds[6],     1.0+ds[7], ds[8],       ds[9],   ds[10],  ds[11],
            ds[12],    ds[13],    1.0+ds[14],  ds[15],  ds[16],  ds[17],
            0.5*duxx,  duxy,      0.5*duyy,    1.0+dux, duy,     du,
            0.5*dvxx,  dvxy,      0.5*dvyy,    dvx,     1.0+dvy, dv,
            0.0,       0.0,       0.0,         0.0,     0.0,     1.0,
        ]);
        let w_new =
            w_old * w_del.try_inverse().expect("W_delta singular (order-2)");
        vec![
            w_new[(3, 5)],
            w_new[(4, 5)],
            w_new[(3, 3)] - 1.0,
            w_new[(4, 3)],
            w_new[(3, 4)],
            w_new[(4, 4)] - 1.0,
            w_new[(3, 0)] * 2.0,
            w_new[(4, 0)] * 2.0,
            w_new[(3, 1)],
            w_new[(4, 1)],
            w_new[(3, 2)] * 2.0,
            w_new[(4, 2)] * 2.0,
        ]
    }
}

// ---------------------------------------------------------------------------
// Convergence metrics
// ---------------------------------------------------------------------------

/// Gao et al. (2015) convergence norm for the warp parameter increment.
///
/// Matches `_norm` in `_subset.cpp`. `size` = `sqrt(n_px)` (representative
/// subset dimension).
pub fn convergence_norm(delta_p: &[f64], size: f64) -> f64 {
    let n = delta_p.len();
    if n <= 7 {
        (delta_p[0].powi(2)
            + delta_p[1].powi(2)
            + (delta_p[2] * size).powi(2)
            + (delta_p[3] * size).powi(2)
            + (delta_p[4] * size).powi(2)
            + (delta_p[5] * size).powi(2))
        .sqrt()
    } else {
        (delta_p[0].powi(2)
            + delta_p[1].powi(2)
            + (delta_p[2] * size).powi(2)
            + (delta_p[3] * size).powi(2)
            + (delta_p[4] * size).powi(2)
            + (delta_p[5] * size).powi(2)
            + (0.5 * delta_p[6] * size * size).powi(2)
            + (0.5 * delta_p[7] * size * size).powi(2)
            + (0.5 * delta_p[8] * size * size).powi(2)
            + (0.5 * delta_p[9] * size * size).powi(2)
            + (0.5 * delta_p[10] * size * size).powi(2)
            + (0.5 * delta_p[11] * size * size).powi(2))
        .sqrt()
    }
}

/// Zero-normalised sum of squared differences (ZNSSD).
///
/// Matches `_ZNSSD` in `_subset.cpp`. ZNCC = 1 − ZNSSD/2.
pub fn znssd(
    f: &Array1<f64>,
    g: &Array1<f64>,
    f_m: f64,
    g_m: f64,
    delta_f: f64,
    delta_g: f64,
) -> f64 {
    f.iter()
        .zip(g.iter())
        .map(|(&fi, &gi)| {
            let v = (fi - f_m) / delta_f - (gi - g_m) / delta_g;
            v * v
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Warp-adequacy diagnostic (order-1 and order-2)
// ---------------------------------------------------------------------------
//
// Gradient-weighted omitted-mode score: Sam Stanier, "Detecting
// Warp-Function Under-Fitting in Local Digital Image Correlation" (2026-09-22,
// `Warp_Function_Underfitting_Report.pdf`). For a converged subset, a high
// ZNCC does not by itself prove the warp is a kinematically adequate model
// of the true displacement field -- a localised or higher-order field can be
// averaged into a smooth, well-correlating fit while leaving a large
// displacement/shear error behind. This projects the converged residual onto
// the modes the fitted warp order cannot represent (quadratic for order-1,
// cubic for order-2 -- the report's own vertical-only benchmark only covers
// the order-1 case; the order-2/cubic extension follows the same
// first-order-Taylor argument one degree further out, per the report's own
// §8: "other deformation mechanisms would require corresponding omitted
// modes", but is not itself validated by the report), weighted by local
// image gradient, to score how much of that residual "wants" to be
// explained by a mode the model wasn't allowed to use.
// Manufactured-benchmark AUC ~0.77 for adequate/under-fit classification in
// the report (order-1 only) -- a real but moderate signal, not a calibrated
// test.

/// Zero-normalise a slice: subtract the mean, divide by the population
/// standard deviation. Falls back to an all-zero result (instead of
/// dividing by ~0) when the input has negligible variance (`1e-18`
/// variance floor).
fn zn_slice(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.max(1e-18).sqrt();
    x.iter().map(|&v| (v - mean) / std).collect()
}

/// One component (`u` or `v`) of [`omitted_mode_diagnostic`]: gradient-weight
/// the `K` omitted modes (3 quadratic modes `[ξ², ξζ, ζ²]` for order-1, 4
/// cubic modes `[ξ³, ξ²ζ, ξζ², ζ³]` for order-2), standardise each of the
/// `K` resulting columns over the subset (report eq. 8), project the
/// mean-centred residual onto them and divide by `N`, then take the L2 norm
/// (report eq. 9, `η = ‖ Sᵀ(e − ē) / N ‖₂`). Generic over `K` so order-1 and
/// order-2 share this one implementation rather than two near-duplicates.
fn eta_component<const K: usize>(modes: &[[f64; K]], grad_zn: &[f64], e_centered: &[f64]) -> f64 {
    let n = modes.len();
    let raw_cols: [Vec<f64>; K] =
        std::array::from_fn(|k| (0..n).map(|i| grad_zn[i] * modes[i][k]).collect());
    let s_cols: [Vec<f64>; K] = std::array::from_fn(|k| zn_slice(&raw_cols[k]));
    let proj: [f64; K] = std::array::from_fn(|k| {
        s_cols[k]
            .iter()
            .zip(e_centered.iter())
            .map(|(&s, &e)| s * e)
            .sum::<f64>()
            / n as f64
    });
    proj.iter().map(|&p| p * p).sum::<f64>().sqrt()
}

/// Gradient-weighted omitted-mode warp-adequacy score, generalised from the
/// report's vertical-only, order-1-only manufactured case to both
/// displacement components and both warp orders geopyv-dev supports:
/// `eta_u` uses the reference horizontal gradient `I_x` (omitted `u`
/// modes), `eta_v` uses the reference vertical gradient `I_y` (omitted `v`
/// modes). The omitted-mode basis is the *next* polynomial degree up from
/// `subset_order` -- quadratic for order-1, cubic for order-2 (see the
/// section doc comment above for the order-2 caveat). Returns `None` for
/// any other order (there is no order-3 solver to be "one below").
///
/// Uses the reference-frame gradient `grad_f` (not a per-iteration target
/// gradient) because inverse-compositional ICGN holds the reference frame,
/// and thus its own linearisation, fixed -- and because `grad_f` is the one
/// gradient always available on `Subset` regardless of which solver
/// (ICGN or FAGN) produced the converged `p`.
///
/// Returns `Some((eta_u, eta_v))`, or `None` if `subset_order` isn't 1 or 2.
fn omitted_mode_diagnostic(
    coord: [f64; 2],
    subset_order: usize,
    f_coords: &Array2<f64>,
    grad_f: &Array2<f64>,
    f: &Array1<f64>,
    f_m: f64,
    delta_f: f64,
    g: &Array1<f64>,
    g_m: f64,
    delta_g: f64,
) -> Option<(f64, f64)> {
    let n = f_coords.nrows();

    // Per-pixel zero-normalised image residual, mean-centred.
    let e: Vec<f64> = (0..n)
        .map(|i| (f[i] - f_m) / delta_f - (g[i] - g_m) / delta_g)
        .collect();
    let e_bar = e.iter().sum::<f64>() / n as f64;
    let e_centered: Vec<f64> = e.iter().map(|&ei| ei - e_bar).collect();

    let zn_ix = zn_slice(&(0..n).map(|i| grad_f[[i, 0]]).collect::<Vec<f64>>());
    let zn_iy = zn_slice(&(0..n).map(|i| grad_f[[i, 1]]).collect::<Vec<f64>>());

    let local_coords = |i: usize| {
        let xi = f_coords[[i, 0]] - coord[0];
        let zeta = f_coords[[i, 1]] - coord[1];
        (xi, zeta)
    };

    match subset_order {
        1 => {
            // Omitted (order-2) modes: [ξ², ξζ, ζ²].
            let modes: Vec<[f64; 3]> = (0..n)
                .map(|i| {
                    let (xi, zeta) = local_coords(i);
                    [xi * xi, xi * zeta, zeta * zeta]
                })
                .collect();
            Some((
                eta_component(&modes, &zn_ix, &e_centered),
                eta_component(&modes, &zn_iy, &e_centered),
            ))
        }
        2 => {
            // Omitted (order-3) modes: [ξ³, ξ²ζ, ξζ², ζ³].
            let modes: Vec<[f64; 4]> = (0..n)
                .map(|i| {
                    let (xi, zeta) = local_coords(i);
                    [xi * xi * xi, xi * xi * zeta, xi * zeta * zeta, zeta * zeta * zeta]
                })
                .collect();
            Some((
                eta_component(&modes, &zn_ix, &e_centered),
                eta_component(&modes, &zn_iy, &e_centered),
            ))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Subset
// ---------------------------------------------------------------------------

impl Subset {
    /// Full constructor — called when images are available.
    ///
    /// Masking logic: if `global_mask` is `Some`, clones `local_mask`, applies
    /// `mask_update`, and uses the masked coords. If `None`, uses `local_mask`
    /// coords directly (all pixels within the local mask shape are active).
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if the effective pixel set is empty or
    /// `delta_f == 0` (featureless subset).
    pub fn new(
        coord: [f64; 2],
        local_mask: &LocalMask,
        global_mask: Option<ArrayView2<u8>>,
        f_img: Arc<Image>,
        g_img: Arc<Image>,
        subset_order: usize,
    ) -> Result<Self, Error> {
        let coords = match global_mask {
            Some(mask) => {
                let mut lm = local_mask.clone();
                lm.mask_update(coord, mask);
                lm.coords
            }
            None => local_mask.coords.clone(),
        };

        let n = coords.nrows();
        if n == 0 {
            return Err(Error::InvalidInput("effective mask coords is empty".to_string()));
        }

        let mask = MaskSummary {
            shape: local_mask.shape.clone(),
            size: local_mask.size,
            n_px: n,
        };

        let qv = f_img.qcqt.view();

        // f_coords[i] = coords[i] + coord  (matches _f_coords in C++)
        let mut f_coords = Array2::zeros((n, 2));
        for i in 0..n {
            f_coords[[i, 0]] = coords[[i, 0]] + coord[0];
            f_coords[[i, 1]] = coords[[i, 1]] + coord[1];
        }

        // Reference intensities + gradients (Tier B) -- one fused B-spline
        // pass (layer_rg_plan.md §11.3). grad_f is now computed before the
        // delta_f == 0 check rather than after; that path is a rare fatal
        // error return, so the extra work there is immaterial.
        let mut f = Array1::<f64>::zeros(n);
        let mut grad_f = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let (fi, [gx, gy]) = bspline_eval_grad(f_coords[[i, 0]], f_coords[[i, 1]], &qv);
            f[i] = fi;
            grad_f[[i, 0]] = gx;
            grad_f[[i, 1]] = gy;
        }
        let f_m = f.sum() / n as f64;
        let delta_f = f.iter().map(|&fi| (fi - f_m).powi(2)).sum::<f64>().sqrt();

        if delta_f == 0.0 {
            return Err(Error::InvalidInput(
                "reference subset is featureless (delta_f = 0)".to_string(),
            ));
        }

        // Quality metrics
        let sssig: f64 = (0..n)
            .map(|i| 0.5 * (grad_f[[i, 0]].powi(2) + grad_f[[i, 1]].powi(2)))
            .sum();
        let sigma_intensity =
            (f.iter().map(|&fi| (fi - f_m).powi(2)).sum::<f64>() / n as f64).sqrt();

        Ok(Subset {
            coord,
            subset_order,
            mask,
            sssig,
            delta_f,
            f_img: Some(f_img),
            g_img: Some(g_img),
            f_coords: Some(f_coords),
            f: Some(f),
            f_m: Some(f_m),
            grad_f: Some(grad_f),
            sigma_intensity: Some(sigma_intensity),
            solution: None,
        })
    }

    /// Partial constructor — called on load when images cannot be found on disk.
    ///
    /// Populates only the always-populated tier from `sol`. All image handles
    /// and image-dependent fields are `None`. The resulting `Subset` is marked
    /// as solved (`sol` becomes its stored solution).
    pub fn from_subset_solution(sol: &SubsetSolution, _local_mask: &LocalMask) -> Self {
        let subset_order = if sol.result.p.is_empty() { 1 } else { sol.result.p.len() / 6 };
        Subset {
            coord: sol.coord,
            subset_order,
            mask: sol.mask.clone(),
            sssig: sol.sssig,
            delta_f: sol.delta_f,
            f_img: None,
            g_img: None,
            f_coords: None,
            f: None,
            f_m: None,
            grad_f: None,
            sigma_intensity: None,
            solution: Some(sol.clone()),
        }
    }

    /// `true` if this subset has been solved (a `SubsetSolution` is available).
    pub fn solved(&self) -> bool {
        self.solution.is_some()
    }

    /// The stored solve result, if any.
    pub fn solution(&self) -> Option<&SubsetSolution> {
        self.solution.as_ref()
    }

    /// Directly mark this subset as solved with a pre-computed solution.
    ///
    /// Used when reconstructing a `Subset` that has live image data (via
    /// [`Subset::new`], e.g. for potential re-solving) but whose solved state
    /// comes from a previously saved [`SubsetSolution`] rather than a fresh
    /// call to [`Subset::solve_icgn`]/[`Subset::solve_fagn`].
    pub fn set_solution(&mut self, solution: SubsetSolution) {
        self.solution = Some(solution);
    }

    /// Number of active pixels (from template summary; always available).
    #[inline]
    pub fn n_px(&self) -> usize {
        self.mask.n_px
    }

    /// Inverse Compositional Gauss-Newton solver (ICGN).
    ///
    /// Mutating entry point: runs the solve and stores the resulting
    /// [`SubsetSolution`] on `self` (see [`Subset::solved`]/[`Subset::solution`]).
    /// For the underlying multi-call-safe computation (used internally by
    /// [`Mesh::solve`](crate::mesh::Mesh::solve), which solves the same
    /// `Subset` repeatedly with different initial warps without persisting
    /// any single attempt), see [`Subset::solve_icgn_result`].
    ///
    /// # Arguments
    /// * `p_0` — initial warp vector; `None` → zeros of length `6 * subset_order`;
    ///   provided slice is silently resized to `6 * subset_order`.
    /// * `tolerance` — minimum ZNCC for `quality_ok = true`.
    /// * `max_norm` — convergence criterion on `||Δp||`.
    /// * `max_iterations` — iteration limit.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if any image-dependent field is `None`.
    pub fn solve_icgn(
        &mut self,
        p_0: Option<&[f64]>,
        tolerance: f64,
        max_norm: f64,
        max_iterations: usize,
    ) -> Result<(), Error> {
        let result = self.solve_icgn_result(p_0, tolerance, max_norm, max_iterations)?;
        self.solution = Some(self.build_solution(result));
        Ok(())
    }

    /// Forward Additive Gauss-Newton solver (FAGN).
    ///
    /// Mutating entry point; see [`Subset::solve_icgn`] for the split between
    /// this and [`Subset::solve_fagn_result`].
    ///
    /// # Arguments: same as [`Subset::solve_icgn`].
    pub fn solve_fagn(
        &mut self,
        p_0: Option<&[f64]>,
        tolerance: f64,
        max_norm: f64,
        max_iterations: usize,
    ) -> Result<(), Error> {
        let result = self.solve_fagn_result(p_0, tolerance, max_norm, max_iterations)?;
        self.solution = Some(self.build_solution(result));
        Ok(())
    }

    /// Build a [`SubsetSolution`] from a computed [`SolveResult`], pulling in
    /// this subset's static metadata (coordinates, mask, image paths).
    fn build_solution(&self, result: SolveResult) -> SubsetSolution {
        let ref_image = self.f_img.as_ref()
            .and_then(|img| img.filepath.clone())
            .unwrap_or_default();
        let target_image = self.g_img.as_ref()
            .and_then(|img| img.filepath.clone())
            .unwrap_or_default();
        let n_px = self.n_px();
        SubsetSolution {
            coord: self.coord,
            mask: self.mask.clone(),
            ref_image,
            target_image,
            result,
            std_dev: self.delta_f / (n_px as f64).sqrt(),
            sssig: self.sssig,
            delta_f: self.delta_f,
        }
    }

    /// Inverse Compositional Gauss-Newton solve computation (no mutation).
    ///
    /// Safe to call repeatedly on the same `Subset` with different initial
    /// warps without disturbing any previously stored solution — this is what
    /// [`Mesh::solve`](crate::mesh::Mesh::solve) relies on internally when it
    /// retries a node with several candidate warps and keeps only the best
    /// [`SolveResult`]. External callers wanting a single-shot, solved-tracking
    /// API should use [`Subset::solve_icgn`] instead.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if any image-dependent field is `None`.
    pub(crate) fn solve_icgn_result(
        &self,
        p_0: Option<&[f64]>,
        tolerance: f64,
        max_norm: f64,
        max_iterations: usize,
    ) -> Result<SolveResult, Error> {
        let unavail = || Error::InvalidInput("solve_icgn: images unavailable".to_string());
        let f_coords = self.f_coords.as_ref().ok_or_else(unavail)?;
        let f       = self.f.as_ref().ok_or_else(unavail)?;
        let f_m     = self.f_m.ok_or_else(unavail)?;
        let grad_f  = self.grad_f.as_ref().ok_or_else(unavail)?;
        let g_img   = self.g_img.as_ref().ok_or_else(unavail)?;

        let n = f_coords.nrows();
        let size = (n as f64).sqrt();
        let expected = 6 * self.subset_order;
        let mut p = p_0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; expected]);
        p.resize(expected, 0.0);
        let gv = g_img.qcqt.view();

        // Precompute SDI and Hessian (constant in ICGN); invert once --
        // ICGN's Hessian is loop-invariant (layer_rg_plan.md §11.1).
        let sdi_ref = steepest_descent(self.coord, f_coords, grad_f, self.subset_order);
        let hessian = compute_hessian(&sdi_ref);
        let inv_h = hessian.try_inverse().expect("ICGN Hessian singular");

        let mut history = Vec::with_capacity(max_iterations);
        let mut converged = false;

        // Buffers reused across iterations (layer_rg_plan.md §11.4).
        let mut gc = Array2::<f64>::zeros((n, 2));
        let mut g = Array1::<f64>::zeros(n);

        for iteration in 1..=max_iterations {
            apply_warp_into(&mut gc, self.coord, &p, f_coords);
            for i in 0..n {
                g[i] = bspline_eval(gc[[i, 0]], gc[[i, 1]], &gv);
            }
            let g_m = g.sum() / n as f64;
            let delta_g = g.iter().map(|&gi| (gi - g_m).powi(2)).sum::<f64>().sqrt();

            // Fused update + ZNSSD, one i-outer sweep (layer_rg_plan.md §11.2).
            let (dp_vec, c_znssd) = delta_p_and_znssd_icgn(
                &inv_h, f, &g, f_m, g_m, self.delta_f, delta_g, &sdi_ref,
            );
            let dp: Vec<f64> = dp_vec.iter().copied().collect();
            p = compose_icgn(&p, &dp);

            let norm = convergence_norm(&dp, size);
            let c_zncc = 1.0 - c_znssd / 2.0;
            history.push((iteration, norm, c_zncc, c_znssd));

            if norm < max_norm {
                converged = true;
                break;
            }
        }

        let (c_zncc, c_znssd) = history
            .last()
            .map(|&(_, _, zncc, z)| (zncc, z))
            .unwrap_or((0.0, 4.0));

        let quality_ok = converged && c_zncc >= tolerance;

        // Warp-adequacy diagnostic -- `g` still holds the
        // final iteration's buffer after the loop exits (reused, not
        // reallocated per-iteration); g_m/delta_g are cheap to recompute
        // from it rather than threading them out of the hot loop above.
        let g_m_final = g.sum() / n as f64;
        let delta_g_final = g.iter().map(|&gi| (gi - g_m_final).powi(2)).sum::<f64>().sqrt();
        let eta_omitted = omitted_mode_diagnostic(
            self.coord, self.subset_order, f_coords, grad_f, f, f_m, self.delta_f,
            &g, g_m_final, delta_g_final,
        );

        Ok(SolveResult {
            p, c_zncc, c_znssd, iterations: history.len(), converged, quality_ok, history,
            max_norm, tolerance,
            subset_order: self.subset_order, max_iterations, method: SolveMethod::Icgn,
            eta_omitted,
        })
    }

    /// Forward Additive Gauss-Newton solve computation (no mutation).
    ///
    /// See [`Subset::solve_icgn_result`] for why this exists alongside the
    /// mutating [`Subset::solve_fagn`].
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if any image-dependent field is `None`.
    pub(crate) fn solve_fagn_result(
        &self,
        p_0: Option<&[f64]>,
        tolerance: f64,
        max_norm: f64,
        max_iterations: usize,
    ) -> Result<SolveResult, Error> {
        let unavail = || Error::InvalidInput("solve_fagn: images unavailable".to_string());
        let f_coords = self.f_coords.as_ref().ok_or_else(unavail)?;
        let f       = self.f.as_ref().ok_or_else(unavail)?;
        let f_m     = self.f_m.ok_or_else(unavail)?;
        let grad_f  = self.grad_f.as_ref().ok_or_else(unavail)?;
        let g_img   = self.g_img.as_ref().ok_or_else(unavail)?;

        let n = f_coords.nrows();
        let size = (n as f64).sqrt();
        let expected = 6 * self.subset_order;
        let mut p = p_0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; expected]);
        p.resize(expected, 0.0);
        let gv = g_img.qcqt.view();

        let mut history = Vec::with_capacity(max_iterations);
        let mut converged = false;
        // Lifted out of the loop (rather than shadowed each iteration) so
        // the final iteration's target intensities survive past `for` --
        // the warp-adequacy diagnostic below needs them.
        let mut g = Array1::<f64>::zeros(n);

        for iteration in 1..=max_iterations {
            let g_center = [self.coord[0] + p[0], self.coord[1] + p[1]];

            let gc = apply_warp(self.coord, &p, f_coords);
            g = (0..n)
                .map(|i| bspline_eval(gc[[i, 0]], gc[[i, 1]], &gv))
                .collect();
            let g_m = g.sum() / n as f64;
            let delta_g = g.iter().map(|&gi| (gi - g_m).powi(2)).sum::<f64>().sqrt();

            // Gradient at target positions
            let mut grad_g = Array2::zeros((n, 2));
            for i in 0..n {
                let [gx, gy] = bspline_grad(gc[[i, 0]], gc[[i, 1]], &gv);
                grad_g[[i, 0]] = gx;
                grad_g[[i, 1]] = gy;
            }

            let sdi_cur = steepest_descent(g_center, &gc, &grad_g, self.subset_order);
            let hessian = compute_hessian(&sdi_cur);
            let dp_vec = delta_p_fagn(
                &hessian, f, &g, f_m, g_m, self.delta_f, delta_g, &sdi_cur,
            );

            // Additive update
            for (pi, dpi) in p.iter_mut().zip(dp_vec.iter()) {
                *pi += dpi;
            }
            let dp: Vec<f64> = dp_vec.iter().copied().collect();

            let norm = convergence_norm(&dp, size);
            let c_znssd = znssd(f, &g, f_m, g_m, self.delta_f, delta_g);
            let c_zncc = 1.0 - c_znssd / 2.0;
            history.push((iteration, norm, c_zncc, c_znssd));

            if norm < max_norm {
                converged = true;
                break;
            }
        }

        let (c_zncc, c_znssd) = history
            .last()
            .map(|&(_, _, zncc, z)| (zncc, z))
            .unwrap_or((0.0, 4.0));

        let quality_ok = converged && c_zncc >= tolerance;

        // Warp-adequacy diagnostic -- see solve_icgn_result.
        let g_m_final = g.sum() / n as f64;
        let delta_g_final = g.iter().map(|&gi| (gi - g_m_final).powi(2)).sum::<f64>().sqrt();
        let eta_omitted = omitted_mode_diagnostic(
            self.coord, self.subset_order, f_coords, grad_f, f, f_m, self.delta_f,
            &g, g_m_final, delta_g_final,
        );

        Ok(SolveResult {
            p, c_zncc, c_znssd, iterations: history.len(), converged, quality_ok, history,
            max_norm, tolerance,
            subset_order: self.subset_order, max_iterations, method: SolveMethod::Fagn,
            eta_omitted,
        })
    }

    /// Recompute the converged per-pixel zero-normalised image residual
    /// `e_i = zn(f_i) - zn(g_i)` for this subset's stored solution.
    ///
    /// Deliberately **not** persisted anywhere (`SolveResult` stores only the
    /// scalar [`SolveResult::eta_omitted`]) -- a full per-pixel map is
    /// recomputed on demand from data the subset already needs to hold for
    /// solving in the first place (`f`, `f_m`, `delta_f`, `f_coords`,
    /// `g_img`, and the converged `p`), so it costs nothing to store and one
    /// cheap B-spline pass to regenerate. Backs `Subset.inspect(residual=True)`.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if any image-dependent field is `None`
    /// (e.g. a `Subset` reloaded from a `.pyv` whose original images are no
    /// longer found on disk) or if the subset has not been solved.
    pub fn residual_map(&self) -> Result<Array1<f64>, Error> {
        let unavail = || Error::InvalidInput("residual_map: images unavailable".to_string());
        let f_coords = self.f_coords.as_ref().ok_or_else(unavail)?;
        let f       = self.f.as_ref().ok_or_else(unavail)?;
        let f_m     = self.f_m.ok_or_else(unavail)?;
        let g_img   = self.g_img.as_ref().ok_or_else(unavail)?;
        let sol = self.solution.as_ref().ok_or_else(|| {
            Error::InvalidInput("residual_map: subset has not been solved".to_string())
        })?;
        let p = &sol.result.p;

        let n = f_coords.nrows();
        let gv = g_img.qcqt.view();
        let gc = apply_warp(self.coord, p, f_coords);
        let g: Array1<f64> = (0..n)
            .map(|i| bspline_eval(gc[[i, 0]], gc[[i, 1]], &gv))
            .collect();
        let g_m = g.sum() / n as f64;
        let delta_g = g.iter().map(|&gi| (gi - g_m).powi(2)).sum::<f64>().sqrt();

        Ok((0..n)
            .map(|i| (f[i] - f_m) / self.delta_f - (g[i] - g_m) / delta_g)
            .collect())
    }

}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    // ---- Pure function tests (no I/O) ----

    /// Tier C: convergence_norm with pure translation = ||(u, v)||.
    #[test]
    fn test_norm_pure_translation() {
        let dp = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let norm = convergence_norm(&dp, 1.0);
        assert!((norm - 5.0).abs() < 1e-12, "norm = {norm}");
    }

    /// Tier C: convergence_norm = 0 for zero increment.
    #[test]
    fn test_norm_zero() {
        let dp = vec![0.0; 6];
        assert_eq!(convergence_norm(&dp, 44.3), 0.0);
    }

    /// convergence_norm uses size for gradient terms.
    #[test]
    fn test_norm_with_gradient_terms() {
        // Only ux term: norm = |ux * size|
        let dp = vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        let size = 5.0;
        let expected = 10.0;
        assert!((convergence_norm(&dp, size) - expected).abs() < 1e-12);
    }

    // ---- Warp-adequacy diagnostic (Tier A: pure computation, atol = 1e-12) ----

    #[test]
    fn test_zn_slice_standardises() {
        let x = [1.0, 2.0, 3.0];
        let zn = zn_slice(&x);
        let std = (2.0_f64 / 3.0).sqrt();
        assert!((zn[0] - (-1.0 / std)).abs() < 1e-12, "zn[0] = {}", zn[0]);
        assert!(zn[1].abs() < 1e-12, "zn[1] = {}", zn[1]);
        assert!((zn[2] - (1.0 / std)).abs() < 1e-12, "zn[2] = {}", zn[2]);
        // Standardised: mean 0, population std 1.
        let mean: f64 = zn.iter().sum::<f64>() / 3.0;
        let var: f64 = zn.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-12, "mean = {mean}");
        assert!((var - 1.0).abs() < 1e-10, "var = {var}");
    }

    /// Zero-variance input (e.g. a purely horizontal texture gradient's `I_y`
    /// column) must not produce NaN/inf -- the `1e-18` variance floor should
    /// collapse it to all-zero instead.
    #[test]
    fn test_zn_slice_zero_variance_floor() {
        let x = [5.0, 5.0, 5.0, 5.0];
        let zn = zn_slice(&x);
        for &v in &zn {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
            assert!(v.is_finite());
        }
    }

    /// `eta_component` against a hand-computed projection. Four pixels at
    /// local coordinates forming a symmetric cross `(±1, 0), (0, ±1)`, so the
    /// omitted-mode columns `[ξ², ξζ, ζ²]` reduce to a clean closed form:
    /// `ξ²` column = `[1,1,0,0]`, `ζ²` column = `[0,0,1,1]`, `ξζ` column all
    /// zero. With `grad_zn = [1,-1,1,-1]` the gradient-weighted `ξ²` column
    /// is `[1,-1,0,0]` (population std `sqrt(0.5)`, so standardised it's
    /// `[√2,-√2,0,0]`); projecting onto `e_centered = [1,0,0,0]` and dividing
    /// by `N=4` gives `proj[0] = √2/4`, `proj[1] = proj[2] = 0`, so
    /// `eta = √2/4` exactly.
    #[test]
    fn test_eta_component_hand_computed() {
        let modes = [
            [1.0, 0.0, 0.0], // (ξ,ζ) = (1, 0)
            [1.0, 0.0, 0.0], // (ξ,ζ) = (-1, 0)  -- ξ² still 1
            [0.0, 0.0, 1.0], // (ξ,ζ) = (0, 1)
            [0.0, 0.0, 1.0], // (ξ,ζ) = (0, -1)  -- ζ² still 1
        ];
        let grad_zn = [1.0, -1.0, 1.0, -1.0];
        let e_centered = [1.0, 0.0, 0.0, 0.0];
        let eta = eta_component(&modes, &grad_zn, &e_centered);
        let expected = std::f64::consts::SQRT_2 / 4.0;
        assert!((eta - expected).abs() < 1e-12, "eta = {eta}, expected {expected}");
    }

    /// `eta_component` is non-negative (it's an L2 norm) regardless of sign
    /// patterns in the inputs.
    #[test]
    fn test_eta_component_non_negative() {
        let modes = [[1.0, 0.5, 0.2], [-0.3, 0.1, -0.7], [0.4, -0.4, 0.9]];
        let grad_zn = [0.5, -1.2, 0.8];
        let e_centered = [-0.6, 1.1, 0.2];
        assert!(eta_component(&modes, &grad_zn, &e_centered) >= 0.0);
    }

    /// `eta_component` at `K=4` (the order-2/cubic-omitted-mode case) against
    /// a hand-computed projection -- same symmetric-cross pixel layout as
    /// [`test_eta_component_hand_computed`], but with cubic modes `[ξ³, ξ²ζ,
    /// ξζ², ζ³]`: only the `ξ³` and `ζ³` columns are non-degenerate on this
    /// layout (`[1,1,0,0]` and `[0,0,1,1]` after gradient weighting), giving
    /// standardised columns `±[1,1,-1,-1]`. Projecting onto
    /// `e_centered = [1,0,0,0]` and dividing by `N=4` gives
    /// `proj = [0.25, 0, 0, -0.25]`, so `eta = sqrt(0.25² + 0.25²) = √2/4`.
    #[test]
    fn test_eta_component_hand_computed_order2_cubic() {
        let modes = [
            [1.0, 0.0, 0.0, 0.0],  // (ξ,ζ) = (1, 0)
            [-1.0, 0.0, 0.0, 0.0], // (ξ,ζ) = (-1, 0)
            [0.0, 0.0, 0.0, 1.0],  // (ξ,ζ) = (0, 1)
            [0.0, 0.0, 0.0, -1.0], // (ξ,ζ) = (0, -1)
        ];
        let grad_zn = [1.0, -1.0, 1.0, -1.0];
        let e_centered = [1.0, 0.0, 0.0, 0.0];
        let eta = eta_component(&modes, &grad_zn, &e_centered);
        let expected = std::f64::consts::SQRT_2 / 4.0;
        assert!((eta - expected).abs() < 1e-12, "eta = {eta}, expected {expected}");
    }

    /// End-to-end sanity check on [`omitted_mode_diagnostic`]: when the
    /// target subset is pixel-for-pixel identical to the reference, the
    /// zero-normalised residual is exactly zero everywhere regardless of the
    /// omitted-mode/gradient weighting, so both components must be ~0 --
    /// checked at both order-1 (quadratic omitted modes) and order-2 (cubic).
    #[test]
    fn test_omitted_mode_diagnostic_zero_for_identical_images() {
        let coord = [0.0, 0.0];
        let f_coords = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0],
        )
        .unwrap();
        let grad_f = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 0.5, -1.0, 0.3, 0.2, 1.0, -0.2, -1.0],
        )
        .unwrap();
        let f = Array1::from(vec![10.0, 12.0, 9.0, 11.0]);
        let f_m = f.mean().unwrap();
        let delta_f = f.iter().map(|&fi: &f64| (fi - f_m).powi(2)).sum::<f64>().sqrt();

        for order in [1, 2] {
            let (eta_u, eta_v) = omitted_mode_diagnostic(
                coord, order, &f_coords, &grad_f, &f, f_m, delta_f, &f, f_m, delta_f,
            )
            .unwrap_or_else(|| panic!("expected Some for order {order}"));
            assert!(eta_u.abs() < 1e-10, "order {order}: eta_u = {eta_u}");
            assert!(eta_v.abs() < 1e-10, "order {order}: eta_v = {eta_v}");
        }
    }

    /// `omitted_mode_diagnostic` returns `None` for any order other than 1
    /// or 2 (there is no order-3 solver to be "one below").
    #[test]
    fn test_omitted_mode_diagnostic_none_for_unsupported_order() {
        let coord = [0.0, 0.0];
        let f_coords = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, -1.0, 0.0]).unwrap();
        let grad_f = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, -1.0, 0.3]).unwrap();
        let f = Array1::from(vec![10.0, 12.0]);
        let f_m = f.mean().unwrap();
        let delta_f = f.iter().map(|&fi: &f64| (fi - f_m).powi(2)).sum::<f64>().sqrt();
        assert!(omitted_mode_diagnostic(
            coord, 3, &f_coords, &grad_f, &f, f_m, delta_f, &f, f_m, delta_f,
        )
        .is_none());
    }

    /// ZNSSD = 0 for identical subsets.
    #[test]
    fn test_znssd_identical() {
        let f = Array1::from(vec![10.0, 50.0, 100.0, 200.0, 150.0]);
        let f_m = f.mean().unwrap();
        let df = f.iter().map(|&fi: &f64| (fi - f_m).powi(2)).sum::<f64>().sqrt();
        let z = znssd(&f, &f, f_m, f_m, df, df);
        assert!(z.abs() < 1e-12, "znssd = {z}");
    }

    /// ZNSSD is in [0, 4] for typical inputs.
    #[test]
    fn test_znssd_range() {
        let f = Array1::from(vec![10.0, 50.0, 100.0, 200.0]);
        let g = Array1::from(vec![12.0, 48.0, 105.0, 190.0]);
        let f_m = f.mean().unwrap();
        let g_m = g.mean().unwrap();
        let df = f.iter().map(|&fi: &f64| (fi - f_m).powi(2)).sum::<f64>().sqrt();
        let dg = g.iter().map(|&gi: &f64| (gi - g_m).powi(2)).sum::<f64>().sqrt();
        let z = znssd(&f, &g, f_m, g_m, df, dg);
        assert!(z >= 0.0 && z <= 4.0, "znssd out of [0, 4]: {z}");
    }

    /// §11.2: the fused update+ZNSSD is *byte*-identical to the split
    /// `delta_p_icgn` + `znssd` (a pre-inverted Hessian, not the raw one).
    #[test]
    fn delta_p_and_znssd_icgn_matches_split() {
        // Deterministic pseudo-random-ish inputs, order 1 (m = 6).
        let n = 37usize;
        let m = 6usize;
        let f: Array1<f64> = (0..n).map(|i| 40.0 + (i as f64 * 0.7).sin() * 30.0).collect();
        let g: Array1<f64> = (0..n).map(|i| 42.0 + (i as f64 * 0.61 + 0.3).sin() * 28.0).collect();
        let f_m = f.mean().unwrap();
        let g_m = g.mean().unwrap();
        let delta_f = f.iter().map(|&x: &f64| (x - f_m).powi(2)).sum::<f64>().sqrt();
        let delta_g = g.iter().map(|&x: &f64| (x - g_m).powi(2)).sum::<f64>().sqrt();
        let mut sdi = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                sdi[[i, j]] = ((i * 7 + j * 3) as f64 * 0.13).cos() * (1.0 + j as f64 * 0.1);
            }
        }
        let hessian = compute_hessian(&sdi);
        let inv_h = hessian.clone().try_inverse().unwrap();

        let dp_ref = delta_p_icgn(&hessian, &f, &g, f_m, g_m, delta_f, delta_g, &sdi);
        let z_ref = znssd(&f, &g, f_m, g_m, delta_f, delta_g);
        let (dp_fused, z_fused) =
            delta_p_and_znssd_icgn(&inv_h, &f, &g, f_m, g_m, delta_f, delta_g, &sdi);

        assert_eq!(z_ref.to_bits(), z_fused.to_bits(), "ZNSSD not byte-identical");
        for k in 0..m {
            assert_eq!(
                dp_ref[k].to_bits(), dp_fused[k].to_bits(),
                "Δp[{k}] not byte-identical: {} vs {}", dp_ref[k], dp_fused[k]
            );
        }
    }

    /// §11.3: fused value+gradient is byte-identical to the two separate calls.
    #[test]
    fn bspline_eval_grad_matches_separate() {
        // A small non-degenerate QCQT: build one from a real Image.
        let img = crate::image::Image::from_array(
            Array2::from_shape_fn((24, 28), |(y, x)| {
                60.0 + ((x as f64) * 0.9).sin() * 20.0 + ((y as f64) * 0.7).cos() * 15.0
            }),
            6,
        );
        let qv = img.qcqt.view();
        for &(x, y) in &[
            (3.25, 4.75), (10.0, 12.0), (0.0, 0.0), (15.999, 9.001),
            (7.5, 0.5), (-3.0, 40.0), (1e6, -1e6),
        ] {
            let v_ref = bspline_eval(x, y, &qv);
            let g_ref = bspline_grad(x, y, &qv);
            let (v, g) = bspline_eval_grad(x, y, &qv);
            assert_eq!(v_ref.to_bits(), v.to_bits(), "value at ({x},{y})");
            assert_eq!(g_ref[0].to_bits(), g[0].to_bits(), "grad_x at ({x},{y})");
            assert_eq!(g_ref[1].to_bits(), g[1].to_bits(), "grad_y at ({x},{y})");
        }
    }

    /// compose_icgn: zero delta → p unchanged (within floating-point precision).
    #[test]
    fn test_compose_icgn_zero_delta_identity() {
        let p = vec![1.0, 2.0, 0.1, 0.05, -0.1, 0.02];
        let dp = vec![0.0; 6];
        let p_new = compose_icgn(&p, &dp);
        for (a, b) in p.iter().zip(p_new.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "compose_icgn identity failed: {a} != {b}"
            );
        }
    }

    /// compose_icgn order-1 with pure translation from zero warp.
    ///
    /// Starting from identity warp, applying Δp = [du, dv, 0, ...] gives
    /// p_new = [−du, −dv, 0, ...] (the compositional inverse).
    #[test]
    fn test_compose_icgn_pure_translation() {
        let p = vec![0.0; 6];
        let dp = vec![0.5, 0.3, 0.0, 0.0, 0.0, 0.0];
        let p_new = compose_icgn(&p, &dp);
        assert!((p_new[0] - (-0.5)).abs() < 1e-10, "u: {}", p_new[0]);
        assert!((p_new[1] - (-0.3)).abs() < 1e-10, "v: {}", p_new[1]);
        for &val in &p_new[2..] {
            assert!(val.abs() < 1e-12, "grad term non-zero: {val}");
        }
    }

    /// compose_icgn order-2 has the right output length.
    #[test]
    fn test_compose_icgn_order2_length() {
        let p = vec![0.0; 12];
        let dp = vec![0.5, 0.3, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001];
        let p_new = compose_icgn(&p, &dp);
        assert_eq!(p_new.len(), 12);
    }

    /// Tier C golden value: compose_icgn with specific p and Δp.
    ///
    /// Golden values from Phase 1 fixtures:
    ///   p = [1.0, 0.5, 0.01, 0.02, -0.01, 0.005]
    ///   Δp = [0.1, 0.05, 0.001, 0.002, -0.001, 0.0005]
    ///   p_new = [0.89954843, 0.44797691, 0.00900896, 0.017973, -0.0089865, 0.00451572]
    #[test]
    fn test_compose_icgn_golden() {
        let p = vec![1.0, 0.5, 0.01, 0.02, -0.01, 0.005];
        let dp = vec![0.1, 0.05, 0.001, 0.002, -0.001, 0.0005];
        let expected = [
            0.89954843_f64, 0.44797691, 0.00900896, 0.01797300, -0.00898650, 0.00451572,
        ];
        let p_new = compose_icgn(&p, &dp);
        for (i, (&got, &exp)) in p_new.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "p_new[{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// B-spline evaluation on a constant image returns that constant.
    ///
    /// For a constant image `val`, all B-spline coefficients = `val`, and
    /// the QCQT block [0,0] entry = `val` (Q row 0 sums to 1; all others = 0).
    /// Evaluating at integer coordinates (dx=dy=0): intensity = val.
    #[test]
    fn test_bspline_eval_constant_image() {
        // Build a tiny constant QCQT (enough for 2×2 pixels)
        // For a constant image `val`, QCQT block[0, 0] = val, rest 0
        // (because Q row-0 = [1/120, 13/60, 11/20, 13/60, 1/120, 0] sums to 1,
        //  and at dx=dy=0 only the constant term contributes).
        let val = 42.0_f64;
        let mut qcqt = ndarray::Array3::zeros((2, 2, 36)); // 2×2 pixels
        // At each pixel: block[0] (== old block[0,0]) = val, rest 0
        qcqt[[0, 0, 0]] = val;
        qcqt[[0, 1, 0]] = val;
        qcqt[[1, 0, 0]] = val;
        qcqt[[1, 1, 0]] = val;

        let result = bspline_eval(0.0, 0.0, &qcqt.view());
        assert!(
            (result - val).abs() < 1e-12,
            "expected {val}, got {result}"
        );
    }

    /// A far out-of-range query must saturate to the boundary pixel's edge
    /// value, not explode.
    ///
    /// Before the clamp fix, only `xf` was clamped into range while
    /// `dx = x - xf` was left unbounded, so a query far outside the QCQT
    /// domain (e.g. `x = 1_000_000`) produced a `dx` of the same enormous
    /// magnitude, and `dx⁵` in the quintic basis exploded instead of
    /// saturating (originally found via TV-DIC's `warp_image`, which could
    /// sample far outside the image once a displacement estimate diverged —
    /// see `mds/TV_DIC_abandoned.md`).
    #[test]
    fn test_bspline_eval_clamps_far_out_of_range_query() {
        // 1×1-pixel QCQT block: constant term 100, plus a nonzero dx⁵
        // coefficient so an unbounded dx would actually blow the result up
        // (a purely constant image wouldn't distinguish the two behaviours).
        let mut qcqt = ndarray::Array3::zeros((1, 1, 36));
        qcqt[[0, 0, 0]] = 100.0; // constant term
        qcqt[[0, 0, 5]] = 1.0; // dx⁵ coefficient

        let far_out = bspline_eval(1_000_000.0, 0.0, &qcqt.view());
        assert!(far_out.is_finite(), "far out-of-range query exploded: {far_out}");
        assert!(
            far_out.abs() < 200.0,
            "expected the query to saturate near the boundary pixel value \
             (~101), got {far_out} -- clamp is not bounding dx"
        );

        // Should saturate to (essentially) the same value as querying right
        // at the boundary edge.
        let at_edge = bspline_eval(0.999_999, 0.0, &qcqt.view());
        assert!(
            (far_out - at_edge).abs() < 1e-3,
            "far out-of-range query should match the boundary edge value: \
             at_edge={at_edge}, far_out={far_out}"
        );
    }

    /// Same guarantee as `test_bspline_eval_clamps_far_out_of_range_query`,
    /// for `bspline_grad` — a far out-of-range gradient query must stay
    /// bounded rather than exploding via the same unclamped-`dx` mechanism.
    #[test]
    fn test_bspline_grad_clamps_far_out_of_range_query() {
        let mut qcqt = ndarray::Array3::zeros((1, 1, 36));
        qcqt[[0, 0, 0]] = 100.0;
        qcqt[[0, 0, 5]] = 1.0;
        qcqt[[0, 0, 30]] = 1.0; // dy⁵ coefficient ([5*6+0]), for the y-direction half

        let [gx, gy] = bspline_grad(1_000_000.0, -1_000_000.0, &qcqt.view());
        assert!(gx.is_finite() && gy.is_finite(), "grad exploded: [{gx}, {gy}]");
        assert!(
            gx.abs() < 200.0 && gy.abs() < 200.0,
            "expected a bounded gradient near the boundary pixel, got [{gx}, {gy}]"
        );
    }

    // ---- Integration tests: require real images ----

    /// Helper: get path to the DIC test images shipped with geopyv.
    fn test_image_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../geopyv/tests")
            .join(name)
    }

    /// Tier C: ICGN order-1 converges on the real DIC test pair.
    ///
    /// Golden values (x-first coordinate convention) — computed from
    /// geopyv_dev, which differs from `geopyv` by the one-pixel B-spline
    /// prefilter fix in `image.rs::build_kernel` (see its doc comment):
    ///   iterations = 3
    ///   ZNCC ≈ 0.9999869  (Tier C: rtol = 1e-5)
    ///   p ≈ [0.03404, 0.03570, 1.2e-4, -3.1e-5, 4.0e-5, -4.9e-5]
    #[test]
    fn test_solve_icgn_order1_golden() {
        let ref_path = test_image_path("ref.jpg");
        let tar_path = test_image_path("tar.jpg");
        if !ref_path.exists() || !tar_path.exists() {
            eprintln!("Skipping: test images not found");
            return;
        }

        use crate::image::Image;
        use crate::masks::LocalMask;
        let ref_img = Arc::new(Image::from_file(&ref_path, 20).unwrap());
        let tar_img = Arc::new(Image::from_file(&tar_path, 20).unwrap());

        let coord = [200.43, 200.76];
        let local_mask = LocalMask::circle(25).unwrap();
        let subset = Subset::new(
            coord, &local_mask, None,
            Arc::clone(&ref_img), Arc::clone(&tar_img), 1,
        ).unwrap();

        // Verify reference quantities match golden values (x-first coordinate convention)
        assert_eq!(subset.n_px(), 1961, "n_px mismatch");
        assert!(
            (subset.f_m.unwrap() - 72.4611033781).abs() < 1e-4,
            "f_m = {:?}", subset.f_m
        );
        assert!(
            (subset.delta_f - 3166.8112575921).abs() < 0.01,
            "delta_f = {}", subset.delta_f
        );
        assert!(
            (subset.sssig - 430204.0240975655).abs() < 1.0,
            "sssig = {}", subset.sssig
        );

        // Solve
        let result = subset.solve_icgn_result(None, 0.75, 1e-3, 50).unwrap();

        // Tier C: ZNCC ≈ 0.9999869, rtol = 1e-5
        assert!(result.converged, "ICGN did not converge");
        assert!(result.quality_ok, "subset should meet quality tolerance (c_zncc > 0.75)");
        assert!(
            (result.c_zncc - 0.9999869).abs() < 1e-4,
            "ZNCC = {} (expected ~0.9999869)", result.c_zncc
        );
        // Displacement: p[0] ≈ 0.034043, p[1] ≈ 0.035697
        assert!(
            (result.p[0] - 0.034043).abs() < 1e-4,
            "p[0] = {} (expected ~0.034043)", result.p[0]
        );
        assert!(
            (result.p[1] - 0.035697).abs() < 1e-4,
            "p[1] = {} (expected ~0.035697)", result.p[1]
        );
    }

    /// Tier C: FAGN order-1 converges on the real DIC test pair.
    ///
    /// Golden values (x-first coordinate convention) — computed from
    /// geopyv_dev with the one-pixel B-spline prefilter fix (see
    /// `image.rs::build_kernel`):
    ///   iterations = 3, ZNCC ≈ 0.9999869
    ///   p ≈ [0.033595, 0.035016, 1.2e-4, -2.9e-5, 3.9e-5, -4.1e-5]
    #[test]
    fn test_solve_fagn_order1_golden() {
        let ref_path = test_image_path("ref.jpg");
        let tar_path = test_image_path("tar.jpg");
        if !ref_path.exists() || !tar_path.exists() {
            eprintln!("Skipping: test images not found");
            return;
        }

        use crate::image::Image;
        use crate::masks::LocalMask;
        let ref_img = Arc::new(Image::from_file(&ref_path, 20).unwrap());
        let tar_img = Arc::new(Image::from_file(&tar_path, 20).unwrap());

        let coord = [200.43, 200.76];
        let local_mask = LocalMask::circle(25).unwrap();
        let subset = Subset::new(
            coord, &local_mask, None,
            Arc::clone(&ref_img), Arc::clone(&tar_img), 1,
        ).unwrap();

        let result = subset.solve_fagn_result(None, 0.75, 1e-3, 50).unwrap();

        assert!(result.converged, "FAGN did not converge");
        assert!(result.quality_ok, "subset should meet quality tolerance (c_zncc > 0.75)");
        assert!(
            (result.c_zncc - 0.9999869).abs() < 1e-4,
            "ZNCC = {} (expected ~0.9999869)", result.c_zncc
        );
        assert!(
            (result.p[0] - 0.033595).abs() < 1e-4,
            "p[0] = {} (expected ~0.033595)", result.p[0]
        );
        assert!(
            (result.p[1] - 0.035016).abs() < 1e-4,
            "p[1] = {} (expected ~0.035016)", result.p[1]
        );
    }

    /// Tier C: ICGN order-2 converges on the real DIC test pair.
    #[test]
    fn test_solve_icgn_order2_golden() {
        let ref_path = test_image_path("ref.jpg");
        let tar_path = test_image_path("tar.jpg");
        if !ref_path.exists() || !tar_path.exists() {
            eprintln!("Skipping: test images not found");
            return;
        }

        use crate::image::Image;
        use crate::masks::LocalMask;
        let ref_img = Arc::new(Image::from_file(&ref_path, 20).unwrap());
        let tar_img = Arc::new(Image::from_file(&tar_path, 20).unwrap());

        let coord = [200.43, 200.76];
        let local_mask = LocalMask::circle(25).unwrap();
        let subset = Subset::new(
            coord, &local_mask, None,
            Arc::clone(&ref_img), Arc::clone(&tar_img), 2,
        ).unwrap();

        let result = subset.solve_icgn_result(None, 0.75, 1e-3, 50).unwrap();

        assert!(result.converged, "ICGN order-2 did not converge");
        assert!(result.quality_ok, "subset should meet quality tolerance (c_zncc > 0.75)");
        assert!(
            result.c_zncc > 0.999,
            "ZNCC = {} (expected > 0.999)", result.c_zncc
        );
    }

    /// Tier C: FAGN order-2 converges on the real DIC test pair.
    #[test]
    fn test_solve_fagn_order2_golden() {
        let ref_path = test_image_path("ref.jpg");
        let tar_path = test_image_path("tar.jpg");
        if !ref_path.exists() || !tar_path.exists() {
            eprintln!("Skipping: test images not found");
            return;
        }

        use crate::image::Image;
        use crate::masks::LocalMask;
        let ref_img = Arc::new(Image::from_file(&ref_path, 20).unwrap());
        let tar_img = Arc::new(Image::from_file(&tar_path, 20).unwrap());

        let coord = [200.43, 200.76];
        let local_mask = LocalMask::circle(25).unwrap();
        let subset = Subset::new(
            coord, &local_mask, None,
            Arc::clone(&ref_img), Arc::clone(&tar_img), 2,
        ).unwrap();

        let result = subset.solve_fagn_result(None, 0.75, 1e-3, 50).unwrap();

        assert!(result.converged, "FAGN order-2 did not converge");
        assert!(result.quality_ok, "subset should meet quality tolerance (c_zncc > 0.75)");
        assert!(
            result.c_zncc > 0.999,
            "ZNCC = {} (expected > 0.999)", result.c_zncc
        );
    }

    /// solved=false when tolerance is set above the actual ZNCC score.
    #[test]
    fn test_solved_false_below_tolerance() {
        let ref_path = test_image_path("ref.jpg");
        let tar_path = test_image_path("tar.jpg");
        if !ref_path.exists() || !tar_path.exists() {
            eprintln!("Skipping: test images not found");
            return;
        }

        use crate::image::Image;
        use crate::masks::LocalMask;
        let ref_img = Arc::new(Image::from_file(&ref_path, 20).unwrap());
        let tar_img = Arc::new(Image::from_file(&tar_path, 20).unwrap());

        let coord = [200.43, 200.76];
        let local_mask = LocalMask::circle(25).unwrap();
        let subset = Subset::new(
            coord, &local_mask, None,
            Arc::clone(&ref_img), Arc::clone(&tar_img), 1,
        ).unwrap();

        // Set tolerance above 1.0 so no solve can ever pass.
        let result = subset.solve_icgn_result(None, 2.0, 1e-3, 50).unwrap();

        assert!(!result.quality_ok, "quality_ok should be false when tolerance > max possible ZNCC");
    }

    /// `quality_ok` requires convergence, not just correlation: a subset
    /// that exhausts its iteration budget without converging must not
    /// report `quality_ok == true` even when its c_zncc clears a lenient
    /// tolerance. Regression test for the bug documented in
    /// `mds/rg_quality_gate_missing_convergence_check.md`.
    #[test]
    fn test_quality_ok_requires_convergence() {
        let ref_path = test_image_path("ref.jpg");
        let tar_path = test_image_path("tar.jpg");
        if !ref_path.exists() || !tar_path.exists() {
            eprintln!("Skipping: test images not found");
            return;
        }

        use crate::image::Image;
        use crate::masks::LocalMask;
        let ref_img = Arc::new(Image::from_file(&ref_path, 20).unwrap());
        let tar_img = Arc::new(Image::from_file(&tar_path, 20).unwrap());

        let coord = [200.43, 200.76];
        let local_mask = LocalMask::circle(25).unwrap();
        let subset = Subset::new(
            coord, &local_mask, None,
            Arc::clone(&ref_img), Arc::clone(&tar_img), 1,
        ).unwrap();

        // Lenient tolerance (any positive correlation passes) but a single
        // iteration — nowhere near enough for `||Δp|| < max_norm` to hold.
        let result = subset.solve_icgn_result(None, 0.0, 1e-8, 1).unwrap();

        assert!(!result.converged, "single-iteration solve should not have converged");
        assert!(
            !result.quality_ok,
            "quality_ok must be false when not converged, regardless of c_zncc ({})",
            result.c_zncc
        );
    }

    /// Fresh subsets report unsolved; the mutating `solve_icgn` stores a
    /// `SubsetSolution` and flips `solved()` to true.
    #[test]
    fn test_mutating_solve_icgn_sets_solved_and_solution() {
        let ref_path = test_image_path("ref.jpg");
        let tar_path = test_image_path("tar.jpg");
        if !ref_path.exists() || !tar_path.exists() {
            eprintln!("Skipping: test images not found");
            return;
        }

        use crate::image::Image;
        use crate::masks::LocalMask;
        let ref_img = Arc::new(Image::from_file(&ref_path, 20).unwrap());
        let tar_img = Arc::new(Image::from_file(&tar_path, 20).unwrap());

        let coord = [200.43, 200.76];
        let local_mask = LocalMask::circle(25).unwrap();
        let mut subset = Subset::new(
            coord, &local_mask, None,
            Arc::clone(&ref_img), Arc::clone(&tar_img), 1,
        ).unwrap();

        assert!(!subset.solved(), "freshly constructed subset must be unsolved");
        assert!(subset.solution().is_none());

        subset.solve_icgn(None, 0.75, 1e-3, 50).unwrap();

        assert!(subset.solved(), "solved() must be true after solve_icgn");
        let sol = subset.solution().expect("solution must be Some after solve_icgn");
        assert_eq!(sol.coord, coord);
        assert!(
            (sol.result.c_zncc - 0.9999869).abs() < 1e-4,
            "ZNCC = {} (expected ~0.9999869)", sol.result.c_zncc
        );
    }

}
