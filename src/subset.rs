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

use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::{templates::TemplateShape, Error};

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

/// Reference subset quantities computed once from the reference image.
///
/// Created by [`Subset::new`]. All fields reflect the coordinate convention
/// from `geopyv`: `coord[0]` and `coord[1]` are the centre; `f_coords[:,0]`
/// and `f_coords[:,1]` are the two coordinate components per pixel.
pub struct Subset {
    /// Centre coordinate `[coord0, coord1]`.
    pub coord: [f64; 2],
    /// Subset pixel coordinates, shape `(n_px, 2)`.
    pub f_coords: Array2<f64>,
    /// Reference subset intensities (B-spline interpolated), shape `(n_px,)`.
    pub f: Array1<f64>,
    /// Mean reference intensity `f_m`.
    pub f_m: f64,
    /// `sqrt(Σ(f_i − f_m)²)` — normalisation factor for ZNSSD.
    pub delta_f: f64,
    /// Reference image gradients at `f_coords`, shape `(n_px, 2)`.
    pub grad_f: Array2<f64>,
    /// Sum of squared intensity gradients (SSSIG quality metric).
    pub sssig: f64,
    /// Standard deviation of reference intensities (quality metric).
    pub sigma_intensity: f64,
}

/// Serialisable summary of the template used in a [`SubsetSolution`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateSummary {
    pub shape: TemplateShape,
    /// Radius (Circle) or half-side-length (Square) in pixels.
    pub size: usize,
    /// Number of active pixels in the template.
    pub n_px: usize,
}

/// Serialisable result of a single-coordinate, single image-pair DIC solve.
///
/// Saved as [`crate::io::GeopyvObject::Subset`] to a `.pyv` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsetSolution {
    pub coord: [f64; 2],
    pub template: TemplateSummary,
    pub ref_image: PathBuf,
    pub target_image: PathBuf,
    pub result: SolveResult,
    /// Standard deviation of reference intensities: `delta_f / sqrt(n_px)`.
    #[serde(default)]
    pub std_dev: f64,
    /// Sum of squared intensity gradients (quality metric).
    #[serde(default)]
    pub sssig: f64,
    /// Convergence norm threshold used in the solve (for display in convergence plot).
    #[serde(default = "default_max_norm")]
    pub max_norm: f64,
}

fn default_max_norm() -> f64 {
    1e-3
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
    /// Per-iteration history: `(iteration, norm, C_ZNCC, C_ZNSSD)`.
    pub history: Vec<(usize, f64, f64, f64)>,
}

// ---------------------------------------------------------------------------
// B-spline helpers
// ---------------------------------------------------------------------------

/// Extract the 6×6 QCQT block for pixel (row = `yf`, col = `xf`).
///
/// QCQT is stored as `qcqt[yf*6 : yf*6+6, xf*6 : xf*6+6]` — matching the
/// storage layout produced by [`crate::image::Image`].
#[inline]
fn qcqt_block(qcqt: &ArrayView2<f64>, yf: usize, xf: usize) -> QcqtBlock {
    QcqtBlock::from_fn(|r, c| qcqt[[yf * 6 + r, xf * 6 + c]])
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
/// out-of-bounds access.
#[inline]
fn bspline_eval(x: f64, y: f64, qcqt: &ArrayView2<f64>) -> f64 {
    let qcols = qcqt.ncols() / 6;
    let qrows = qcqt.nrows() / 6;
    let xf = (x.floor() as i64).clamp(0, qcols as i64 - 1) as usize;
    let yf = (y.floor() as i64).clamp(0, qrows as i64 - 1) as usize;
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
/// Matches `_grad` in `_subset.cpp` exactly.
#[inline]
fn bspline_grad(x: f64, y: f64, qcqt: &ArrayView2<f64>) -> [f64; 2] {
    let qcols = qcqt.ncols() / 6;
    let qrows = qcqt.nrows() / 6;
    let xf = (x.floor() as i64).clamp(0, qcols as i64 - 1) as usize;
    let yf = (y.floor() as i64).clamp(0, qrows as i64 - 1) as usize;
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
    let n = f_coords.nrows();
    let mut gc = Array2::zeros((n, 2));
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
    gc
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
/// Matches `_Delta_p_ICGN` in `_subset.cpp`.
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
// Subset
// ---------------------------------------------------------------------------

impl Subset {
    /// Create a new `Subset` from a centre coordinate, template coordinates,
    /// and the reference image QCQT.
    ///
    /// # Arguments
    /// * `coord` — subset centre `[coord0, coord1]`
    /// * `template_coords` — pixel offsets relative to centre, shape `(n_px, 2)`
    /// * `f_qcqt` — B-spline coefficient matrix from [`crate::image::Image`],
    ///   shape `(rows*6, cols*6)`
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if `template_coords` is empty or
    /// `delta_f == 0` (featureless subset).
    pub fn new(
        coord: [f64; 2],
        template_coords: &Array2<f64>,
        f_qcqt: &Array2<f64>,
    ) -> Result<Self, Error> {
        let n = template_coords.nrows();
        if n == 0 {
            return Err(Error::InvalidInput("template_coords is empty".to_string()));
        }
        let qv = f_qcqt.view();

        // f_coords[i] = template_coords[i] + coord  (matches _f_coords in C++)
        let mut f_coords = Array2::zeros((n, 2));
        for i in 0..n {
            f_coords[[i, 0]] = template_coords[[i, 0]] + coord[0];
            f_coords[[i, 1]] = template_coords[[i, 1]] + coord[1];
        }

        // Reference intensities (Tier B)
        let f: Array1<f64> = (0..n)
            .map(|i| bspline_eval(f_coords[[i, 0]], f_coords[[i, 1]], &qv))
            .collect();
        let f_m = f.sum() / n as f64;
        let delta_f = f.iter().map(|&fi| (fi - f_m).powi(2)).sum::<f64>().sqrt();

        if delta_f == 0.0 {
            return Err(Error::InvalidInput(
                "reference subset is featureless (delta_f = 0)".to_string(),
            ));
        }

        // Reference gradients (Tier B)
        let mut grad_f = Array2::zeros((n, 2));
        for i in 0..n {
            let [gx, gy] = bspline_grad(f_coords[[i, 0]], f_coords[[i, 1]], &qv);
            grad_f[[i, 0]] = gx;
            grad_f[[i, 1]] = gy;
        }

        // Quality metrics
        let sssig: f64 = (0..n)
            .map(|i| 0.5 * (grad_f[[i, 0]].powi(2) + grad_f[[i, 1]].powi(2)))
            .sum();
        let sigma_intensity =
            (f.iter().map(|&fi| (fi - f_m).powi(2)).sum::<f64>() / n as f64).sqrt();

        Ok(Subset {
            coord,
            f_coords,
            f,
            f_m,
            delta_f,
            grad_f,
            sssig,
            sigma_intensity,
        })
    }

    /// Number of pixels in the subset.
    #[inline]
    pub fn n_px(&self) -> usize {
        self.f_coords.nrows()
    }

    /// Inverse Compositional Gauss-Newton solver (ICGN).
    ///
    /// The Hessian is precomputed once from the reference image gradient.
    /// The warp is updated via matrix composition.
    ///
    /// # Arguments
    /// * `g_qcqt` — target image B-spline coefficient matrix
    /// * `p_0` — initial warp vector (6 elements for order 1, 12 for order 2)
    /// * `max_norm` — convergence criterion on `||Δp||`
    /// * `max_iterations` — iteration limit
    ///
    /// # Errors
    /// Propagates any [`Error`] from internal operations.
    pub fn solve_icgn(
        &self,
        g_qcqt: &Array2<f64>,
        p_0: &[f64],
        max_norm: f64,
        max_iterations: usize,
    ) -> Result<SolveResult, Error> {
        let n = self.n_px();
        let size = (n as f64).sqrt();
        let order = if p_0.len() <= 7 { 1 } else { 2 };
        let gv = g_qcqt.view();

        // Precompute SDI and Hessian (constant in ICGN)
        let sdi_ref = steepest_descent(self.coord, &self.f_coords, &self.grad_f, order);
        let hessian = compute_hessian(&sdi_ref);

        let mut p = p_0.to_vec();
        let mut history = Vec::with_capacity(max_iterations);
        let mut converged = false;

        for iteration in 1..=max_iterations {
            let gc = apply_warp(self.coord, &p, &self.f_coords);
            let g: Array1<f64> = (0..n)
                .map(|i| bspline_eval(gc[[i, 0]], gc[[i, 1]], &gv))
                .collect();
            let g_m = g.sum() / n as f64;
            let delta_g = g.iter().map(|&gi| (gi - g_m).powi(2)).sum::<f64>().sqrt();

            let dp_vec = delta_p_icgn(
                &hessian,
                &self.f,
                &g,
                self.f_m,
                g_m,
                self.delta_f,
                delta_g,
                &sdi_ref,
            );
            let dp: Vec<f64> = dp_vec.iter().copied().collect();
            p = compose_icgn(&p, &dp);

            let norm = convergence_norm(&dp, size);
            let c_znssd = znssd(&self.f, &g, self.f_m, g_m, self.delta_f, delta_g);
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

        Ok(SolveResult { p, c_zncc, c_znssd, iterations: history.len(), converged, history })
    }

    /// Forward Additive Gauss-Newton solver (FAGN).
    ///
    /// The Hessian is recomputed each iteration from the target image gradient.
    /// The warp is updated additively.
    ///
    /// # Arguments: same as [`solve_icgn`].
    pub fn solve_fagn(
        &self,
        g_qcqt: &Array2<f64>,
        p_0: &[f64],
        max_norm: f64,
        max_iterations: usize,
    ) -> Result<SolveResult, Error> {
        let n = self.n_px();
        let size = (n as f64).sqrt();
        let order = if p_0.len() <= 7 { 1 } else { 2 };
        let gv = g_qcqt.view();

        let mut p = p_0.to_vec();
        let mut history = Vec::with_capacity(max_iterations);
        let mut converged = false;

        for iteration in 1..=max_iterations {
            // Warped centre coordinate
            let g_center = [self.coord[0] + p[0], self.coord[1] + p[1]];

            let gc = apply_warp(self.coord, &p, &self.f_coords);
            let g: Array1<f64> = (0..n)
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

            let sdi_cur = steepest_descent(g_center, &gc, &grad_g, order);
            let hessian = compute_hessian(&sdi_cur);
            let dp_vec = delta_p_fagn(
                &hessian,
                &self.f,
                &g,
                self.f_m,
                g_m,
                self.delta_f,
                delta_g,
                &sdi_cur,
            );

            // Additive update
            for (pi, dpi) in p.iter_mut().zip(dp_vec.iter()) {
                *pi += dpi;
            }
            let dp: Vec<f64> = dp_vec.iter().copied().collect();

            let norm = convergence_norm(&dp, size);
            let c_znssd = znssd(&self.f, &g, self.f_m, g_m, self.delta_f, delta_g);
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

        Ok(SolveResult { p, c_zncc, c_znssd, iterations: history.len(), converged, history })
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
        // For a constant image `val`, QCQT block[0,0] = val, rest 0
        // (because Q row-0 = [1/120, 13/60, 11/20, 13/60, 1/120, 0] sums to 1,
        //  and at dx=dy=0 only the constant term contributes).
        let val = 42.0_f64;
        let mut qcqt = Array2::zeros((12, 12)); // 2×2 pixels
        // At each pixel: block[0, 0] = val, rest 0
        qcqt[[0, 0]] = val;
        qcqt[[0, 6]] = val;
        qcqt[[6, 0]] = val;
        qcqt[[6, 6]] = val;

        let result = bspline_eval(0.0, 0.0, &qcqt.view());
        assert!(
            (result - val).abs() < 1e-12,
            "expected {val}, got {result}"
        );
    }

    // ---- Integration tests: require real images ----

    /// Helper: get path to the DIC test images shipped with geopyv.
    fn test_image_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../geopyv/tests")
            .join(name)
    }

    /// Circle template coordinates matching `geopyv.templates.Circle(25)`.
    fn circle_coords_25() -> Array2<f64> {
        use crate::templates::Template;
        Template::circle(25).unwrap().coords
    }

    /// Tier C: ICGN order-1 converges on the real DIC test pair.
    ///
    /// Golden values (x-first coordinate convention):
    ///   iterations = 3
    ///   ZNCC ≈ 0.999987  (Tier C: rtol = 1e-5)
    ///   p ≈ [0.03420, 0.03583, 1.2e-4, -7.5e-5, 2.6e-5, -7.3e-5]
    #[test]
    fn test_solve_icgn_order1_golden() {
        let ref_path = test_image_path("ref.jpg");
        let tar_path = test_image_path("tar.jpg");
        if !ref_path.exists() || !tar_path.exists() {
            eprintln!("Skipping: test images not found");
            return;
        }

        use crate::image::Image;
        let ref_img = Image::from_file(&ref_path, 20).unwrap();
        let tar_img = Image::from_file(&tar_path, 20).unwrap();

        let coord = [200.43, 200.76];
        let tmpl = circle_coords_25();
        let subset = Subset::new(coord, &tmpl, &ref_img.qcqt).unwrap();

        // Verify reference quantities match golden values (x-first coordinate convention)
        assert_eq!(subset.n_px(), 1961, "n_px mismatch");
        assert!(
            (subset.f_m - 70.8561329162).abs() < 1e-4,
            "f_m = {}", subset.f_m
        );
        assert!(
            (subset.delta_f - 3179.4465451209).abs() < 0.01,
            "delta_f = {}", subset.delta_f
        );
        assert!(
            (subset.sssig - 420066.064).abs() < 1.0,
            "sssig = {}", subset.sssig
        );

        // Solve
        let p_0 = vec![0.0f64; 6];
        let result = subset
            .solve_icgn(&tar_img.qcqt, &p_0, 1e-3, 50)
            .unwrap();

        // Tier C: ZNCC ≈ 0.999987, rtol = 1e-5
        assert!(result.converged, "ICGN did not converge");
        assert!(
            (result.c_zncc - 0.999987).abs() < 1e-4,
            "ZNCC = {} (expected ~0.999987)", result.c_zncc
        );
        // Displacement: p[0] ≈ 0.034201, p[1] ≈ 0.035834
        assert!(
            (result.p[0] - 0.034201).abs() < 1e-4,
            "p[0] = {} (expected ~0.034201)", result.p[0]
        );
        assert!(
            (result.p[1] - 0.035834).abs() < 1e-4,
            "p[1] = {} (expected ~0.035834)", result.p[1]
        );
    }

    /// Tier C: FAGN order-1 converges on the real DIC test pair.
    ///
    /// Golden values (x-first coordinate convention):
    ///   iterations = 3, ZNCC ≈ 0.999987
    ///   p ≈ [0.033741, 0.035105, 1.2e-4, -7.1e-5, 2.3e-5, -6.6e-5]
    #[test]
    fn test_solve_fagn_order1_golden() {
        let ref_path = test_image_path("ref.jpg");
        let tar_path = test_image_path("tar.jpg");
        if !ref_path.exists() || !tar_path.exists() {
            eprintln!("Skipping: test images not found");
            return;
        }

        use crate::image::Image;
        let ref_img = Image::from_file(&ref_path, 20).unwrap();
        let tar_img = Image::from_file(&tar_path, 20).unwrap();

        let coord = [200.43, 200.76];
        let tmpl = circle_coords_25();
        let subset = Subset::new(coord, &tmpl, &ref_img.qcqt).unwrap();

        let p_0 = vec![0.0f64; 6];
        let result = subset
            .solve_fagn(&tar_img.qcqt, &p_0, 1e-3, 50)
            .unwrap();

        assert!(result.converged, "FAGN did not converge");
        assert!(
            (result.c_zncc - 0.999987).abs() < 1e-4,
            "ZNCC = {} (expected ~0.999987)", result.c_zncc
        );
        assert!(
            (result.p[0] - 0.033741).abs() < 1e-4,
            "p[0] = {} (expected ~0.033741)", result.p[0]
        );
        assert!(
            (result.p[1] - 0.035105).abs() < 1e-4,
            "p[1] = {} (expected ~0.035105)", result.p[1]
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
        let ref_img = Image::from_file(&ref_path, 20).unwrap();
        let tar_img = Image::from_file(&tar_path, 20).unwrap();

        let coord = [200.43, 200.76];
        let tmpl = circle_coords_25();
        let subset = Subset::new(coord, &tmpl, &ref_img.qcqt).unwrap();

        let p_0 = vec![0.0f64; 12];
        let result = subset
            .solve_icgn(&tar_img.qcqt, &p_0, 1e-3, 50)
            .unwrap();

        assert!(result.converged, "ICGN order-2 did not converge");
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
        let ref_img = Image::from_file(&ref_path, 20).unwrap();
        let tar_img = Image::from_file(&tar_path, 20).unwrap();

        let coord = [200.43, 200.76];
        let tmpl = circle_coords_25();
        let subset = Subset::new(coord, &tmpl, &ref_img.qcqt).unwrap();

        let p_0 = vec![0.0f64; 12];
        let result = subset
            .solve_fagn(&tar_img.qcqt, &p_0, 1e-3, 50)
            .unwrap();

        assert!(result.converged, "FAGN order-2 did not converge");
        assert!(
            result.c_zncc > 0.999,
            "ZNCC = {} (expected > 0.999)", result.c_zncc
        );
    }

}
