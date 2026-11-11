//! Rigid-texture template (`solver_options={"texture": "rigid"}`).
//!
//! `gp.Speckle` draws each speckle as a round blob that moves as a solid lump
//! with the material under its centre. The subset solver assumes the pattern
//! stretches with the material, so where the material strains the target never
//! matches any warped copy of the reference exactly. To first order the
//! left-over is (geopyv-glob-rs `docs/grain_noise.pdf`, `src/adapt/grain.rs`):
//!
//! ```text
//! g(x + u(x)) − f(x) ≈ G(x) = −(s²/2) E : ∇∇f(x) − tr(E) (f(x) − f̄)
//! ```
//!
//! with `E` the Green-Lagrange strain (zero for a rigid rotation, which leaves
//! a round blob unchanged), `f̄` the subset mean and `s` the blob width, from
//! the measured speckle correlation length `ℓ`: `s² = ℓ² / (2 ln 2)`. `∇∇f` is
//! taken once per reference image with a Gaussian derivative filter of width
//! `SMOOTH` and scaled by `((s²/2 + σ²)/(s²/2))²` to undo the extra smoothing.
//! The rigid solve fits `g` to the template `f + G(E(p))`.
//!
//! `texture="grains"` drops the round-Gaussian assumption. For opaque rigid
//! grains of any shape that turn with the material, every pixel moves with
//! its grain's centre `c`, so to first order
//!
//! ```text
//! g(x + u(x)) - f(x) ≈ ∇f(x) · (U - I)(x - c(x))
//! ```
//!
//! with `U` the stretch of the polar decomposition `F = R U` (zero for any
//! rotation). The grain centre of every pixel comes from splitting the
//! reference into grains once ([`GrainCentres`]). Where grains add rather than
//! hide each other, a pixel's gradient mixes grains and this does not hold.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::image::Image;
use crate::seam::correlation_length;

/// Width (px) of the Gaussian derivative filter for ∇∇f.
const SMOOTH: f64 = 1.0;

/// The reference image's second derivatives at every pixel (already scaled by
/// `corr`), and the speckle width.
pub(crate) struct GrainMaps {
    pub fxx: Array2<f64>,
    pub fyy: Array2<f64>,
    pub fxy: Array2<f64>,
    /// s² (px²).
    pub s2: f64,
    /// Smoothing correction already applied to the maps.
    pub corr: f64,
}

/// 1-D Gaussian kernel (order 0, 1 or 2 derivative) of width `SMOOTH`, radius
/// 4σ, as correlation weights: `Σ_t k[t] a[x + t]` is the smoothed value or
/// derivative at `x`.
fn kernel(order: usize) -> Vec<f64> {
    kernel_with(order, SMOOTH)
}

/// As [`kernel`], of width `sigma`.
fn kernel_with(order: usize, sigma: f64) -> Vec<f64> {
    let r = (4.0 * sigma).ceil() as i64;
    let s2 = sigma * sigma;
    let g: Vec<f64> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * s2)).exp()).collect();
    let norm: f64 = g.iter().sum();
    (-r..=r)
        .zip(&g)
        .map(|(i, &gi)| {
            let t = i as f64;
            let v = match order {
                0 => gi,
                1 => t / s2 * gi,
                _ => (t * t / (s2 * s2) - 1.0 / s2) * gi,
            };
            v / norm
        })
        .collect()
}

/// Correlate every row (`along_rows`) or column of `a` with `k`, replicating
/// the edge pixels.
fn filter_1d(a: &Array2<f64>, k: &[f64], along_rows: bool) -> Array2<f64> {
    let (rows, cols) = a.dim();
    let r = (k.len() / 2) as i64;
    let mut out = Array2::<f64>::zeros((rows, cols));
    out.as_slice_mut()
        .expect("standard layout")
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let mut s = 0.0;
                for (j, w) in k.iter().enumerate() {
                    let t = j as i64 - r;
                    s += w * if along_rows {
                        a[[y, (x as i64 + t).clamp(0, cols as i64 - 1) as usize]]
                    } else {
                        a[[(y as i64 + t).clamp(0, rows as i64 - 1) as usize, x]]
                    };
                }
                *o = s;
            }
        });
    out
}

impl GrainMaps {
    /// Build from a reference image, or `None` when the image has no
    /// measurable speckle (too small, flat, or an autocorrelation that never
    /// falls to 0.5): the rigid solve then falls back to the deformable one.
    pub(crate) fn build(img: &Image) -> Option<Self> {
        let ell = correlation_length(img).ok()?;
        let s2 = ell * ell / (2.0 * std::f64::consts::LN_2);
        if !s2.is_finite() || s2 <= 0.0 {
            return None;
        }
        let corr = ((s2 / 2.0 + SMOOTH * SMOOTH) / (s2 / 2.0)).powi(2);
        let k = [kernel(0), kernel(1), kernel(2)];
        let a = &img.image_gs;
        // x = column (along a row), y = row.
        let h = [filter_1d(a, &k[0], true), filter_1d(a, &k[1], true), filter_1d(a, &k[2], true)];
        let fxx = filter_1d(&h[2], &k[0], false) * corr;
        let fyy = filter_1d(&h[0], &k[2], false) * corr;
        let fxy = filter_1d(&h[1], &k[1], false) * corr;
        Some(Self { fxx, fyy, fxy, s2, corr })
    }

    /// As [`Self::sample`], with the smoothing correction for blob width^2
    /// `s2` instead of the correlation length's.
    pub(crate) fn sample_for(&self, coords: &Array2<f64>, s2: f64) -> Vec<[f64; 3]> {
        let k = smoothing_corr(s2) / self.corr;
        self.sample(coords).into_iter().map(|h| h.map(|v| v * k)).collect()
    }

    /// `[fxx, fyy, fxy]` at each `(x = column, y = row)` of `coords`
    /// (bilinear, clamped to the image).
    pub(crate) fn sample(&self, coords: &Array2<f64>) -> Vec<[f64; 3]> {
        let (rows, cols) = self.fxx.dim();
        (0..coords.nrows())
            .map(|i| {
                let x = coords[[i, 0]].clamp(0.0, cols as f64 - 1.0);
                let y = coords[[i, 1]].clamp(0.0, rows as f64 - 1.0);
                let (x0, y0) = ((x.floor() as usize).min(cols.saturating_sub(2)), (y.floor() as usize).min(rows.saturating_sub(2)));
                let (x1, y1) = ((x0 + 1).min(cols - 1), (y0 + 1).min(rows - 1));
                let (tx, ty) = (x - x0 as f64, y - y0 as f64);
                let bl = |m: &Array2<f64>| {
                    (1.0 - ty) * ((1.0 - tx) * m[[y0, x0]] + tx * m[[y0, x1]])
                        + ty * ((1.0 - tx) * m[[y1, x0]] + tx * m[[y1, x1]])
                };
                [bl(&self.fxx), bl(&self.fyy), bl(&self.fxy)]
            })
            .collect()
    }
}

/// Factor undoing the extra spread the `SMOOTH` derivative filter gives a
/// blob of width^2 `s2 / 2`.
pub(crate) fn smoothing_corr(s2: f64) -> f64 {
    ((s2 / 2.0 + SMOOTH * SMOOTH) / (s2 / 2.0)).powi(2)
}

/// The coarse part of the reference (the image smoothed by a Gaussian of
/// width `sigma_l`) at `coords`: its value, and its second derivatives after
/// the `SMOOTH` filter (i.e. with a Gaussian of width sqrt(SMOOTH^2 +
/// sigma_l^2)), not yet scaled. Worked out on a window around `coords` only.
pub(crate) fn coarse_part(img: &Image, coords: &Array2<f64>, sigma_l: f64) -> (Vec<f64>, Vec<[f64; 3]>) {
    let a = &img.image_gs;
    let (rows, cols) = a.dim();
    let sigma_t = (SMOOTH * SMOOTH + sigma_l * sigma_l).sqrt();
    let m = (4.0 * sigma_t).ceil() as usize + 2;
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for i in 0..coords.nrows() {
        x0 = x0.min(coords[[i, 0]]);
        x1 = x1.max(coords[[i, 0]]);
        y0 = y0.min(coords[[i, 1]]);
        y1 = y1.max(coords[[i, 1]]);
    }
    let c0 = (x0.floor().max(0.0) as usize).saturating_sub(m);
    let c1 = ((x1.ceil().max(0.0) as usize) + m + 1).min(cols);
    let r0 = (y0.floor().max(0.0) as usize).saturating_sub(m);
    let r1 = ((y1.ceil().max(0.0) as usize) + m + 1).min(rows);
    let w = a.slice(ndarray::s![r0..r1, c0..c1]).to_owned();
    let kl = kernel_with(0, sigma_l);
    let mut kt = [kernel_with(0, sigma_t), kernel_with(1, sigma_t), kernel_with(2, sigma_t)];
    // cut at 4 widths, the second-derivative kernel does not quite sum to zero
    // and would see a constant brightness; take that out
    let (s0, s2) = (kt[0].iter().sum::<f64>(), kt[2].iter().sum::<f64>());
    let k0 = kt[0].clone();
    kt[2].iter_mut().zip(&k0).for_each(|(v, g)| *v -= s2 * g / s0);
    let low = filter_1d(&filter_1d(&w, &kl, true), &kl, false);
    let h = [filter_1d(&w, &kt[0], true), filter_1d(&w, &kt[1], true), filter_1d(&w, &kt[2], true)];
    let (lxx, lyy, lxy) = (filter_1d(&h[2], &kt[0], false), filter_1d(&h[0], &kt[2], false), filter_1d(&h[1], &kt[1], false));
    let (wr, wc) = low.dim();
    let bl = |mtx: &Array2<f64>, x: f64, y: f64| {
        let x = (x - c0 as f64).clamp(0.0, wc as f64 - 1.0);
        let y = (y - r0 as f64).clamp(0.0, wr as f64 - 1.0);
        let (xa, ya) = ((x.floor() as usize).min(wc.saturating_sub(2)), (y.floor() as usize).min(wr.saturating_sub(2)));
        let (xb, yb) = ((xa + 1).min(wc - 1), (ya + 1).min(wr - 1));
        let (tx, ty) = (x - xa as f64, y - ya as f64);
        (1.0 - ty) * ((1.0 - tx) * mtx[[ya, xa]] + tx * mtx[[ya, xb]]) + ty * ((1.0 - tx) * mtx[[yb, xa]] + tx * mtx[[yb, xb]])
    };
    (0..coords.nrows())
        .map(|i| {
            let (x, y) = (coords[[i, 0]], coords[[i, 1]]);
            (bl(&low, x, y), [bl(&lxx, x, y), bl(&lyy, x, y), bl(&lxy, x, y)])
        })
        .unzip()
}

/// Green-Lagrange strain `[Exx, Eyy, Exy]` from the displacement gradients.
#[inline]
fn green_lagrange(ux: f64, vx: f64, uy: f64, vy: f64) -> [f64; 3] {
    [
        ux + 0.5 * (ux * ux + vx * vx),
        vy + 0.5 * (uy * uy + vy * vy),
        0.5 * (uy + vx) + 0.5 * (ux * uy + vx * vy),
    ]
}

/// The rigid-texture template `f + G` for warp `p` (same layout as
/// `apply_warp_into`: 2, 6 or 12 parameters). `h` holds `[fxx, fyy, fxy]` at
/// each subset pixel ([`GrainMaps::sample`]). Order 1 uses its uniform strain;
/// order 2 the strain of its warp at each pixel. Order 0 carries no strain
/// and returns `f` unchanged.
pub(crate) fn grain_template(
    f: &Array1<f64>,
    f_m: f64,
    base: Option<&[f64]>,
    h: &[[f64; 3]],
    s2: f64,
    p: &[f64],
    coord: [f64; 2],
    f_coords: &Array2<f64>,
) -> Array1<f64> {
    if p.len() == 2 {
        return f.clone();
    }
    // area term measured from the subset mean, or from `base` (the coarse part)
    let b = |i: usize| base.map_or(f_m, |b| b[i]);
    let g = |e: [f64; 3], hi: &[f64; 3], fi: f64, bi: f64| {
        -0.5 * s2 * (e[0] * hi[0] + e[1] * hi[1] + 2.0 * e[2] * hi[2]) - (e[0] + e[1]) * (fi - bi)
    };
    if p.len() <= 7 {
        let e = green_lagrange(p[2], p[3], p[4], p[5]);
        Array1::from_shape_fn(f.len(), |i| f[i] + g(e, &h[i], f[i], b(i)))
    } else {
        Array1::from_shape_fn(f.len(), |i| {
            let (dx, dy) = (f_coords[[i, 0]] - coord[0], f_coords[[i, 1]] - coord[1]);
            let ux = p[2] + p[6] * dx + p[8] * dy;
            let vx = p[3] + p[7] * dx + p[9] * dy;
            let uy = p[4] + p[8] * dx + p[10] * dy;
            let vy = p[5] + p[9] * dx + p[11] * dy;
            f[i] + g(green_lagrange(ux, vx, uy, vy), &h[i], f[i], b(i))
        })
    }
}

/// Grain centre of every pixel of a reference image (`NaN` where no grain).
pub(crate) struct GrainCentres {
    pub cx: Array2<f32>,
    pub cy: Array2<f32>,
    pub n_grains: usize,
    /// Grain number of every pixel (1..=n_grains), 0 where no grain.
    pub label: Array2<i64>,
    /// Centre `[x, y]` of grain number k + 1.
    pub centres: Vec<[f64; 2]>,
}

impl GrainCentres {
    /// Split the image into grains: smooth with the `SMOOTH` Gaussian, let
    /// every pixel climb to its brightest 3 x 3 neighbour until it reaches a
    /// peak, and take that peak's grain. Peaks no higher than a quarter of the
    /// way from the image's dark to its bright level (1st to 99th percentile)
    /// are background: no grain. Each
    /// grain's centre is the brightness-weighted centroid of its pixels at
    /// least half as bright as its brightest one. `None` if no grain is found.
    pub(crate) fn build(img: &Image) -> Option<Self> {
        let a = &img.image_gs;
        let (rows, cols) = a.dim();
        if rows < 3 || cols < 3 {
            return None;
        }
        let k0 = kernel(0);
        let sm = filter_1d(&filter_1d(a, &k0, true), &k0, false);
        let n = rows * cols;
        let mut ptr: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|q| {
                let (r, c) = (q / cols, q % cols);
                let mut best = (sm[[r, c]], q);
                for dr in -1i64..=1 {
                    for dc in -1i64..=1 {
                        let (rr, cc) = (r as i64 + dr, c as i64 + dc);
                        if rr < 0 || cc < 0 || rr >= rows as i64 || cc >= cols as i64 {
                            continue;
                        }
                        let v = sm[[rr as usize, cc as usize]];
                        if v > best.0 {
                            best = (v, rr as usize * cols + cc as usize);
                        }
                    }
                }
                best.1
            })
            .collect();
        loop {
            let next: Vec<usize> = ptr.par_iter().map(|&q| ptr[q]).collect();
            if next == ptr {
                break;
            }
            ptr = next;
        }
        let mut sorted: Vec<f64> = sm.iter().copied().collect();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let (lo, hi) = (sorted[((n as f64) * 0.01) as usize], sorted[((n as f64) * 0.99) as usize]);
        if !(hi > lo) {
            return None;
        }
        let floor = lo + 0.25 * (hi - lo);
        // one slot per root pixel
        let mut peak = vec![f64::NEG_INFINITY; n];
        for q in 0..n {
            let root = ptr[q];
            peak[root] = peak[root].max(a[[q / cols, q % cols]]);
        }
        let mut acc = vec![[0.0f64; 3]; n];
        for q in 0..n {
            let root = ptr[q];
            if sm[[root / cols, root % cols]] <= floor {
                continue;
            }
            let v = a[[q / cols, q % cols]];
            if v >= 0.5 * peak[root] {
                acc[root][0] += v;
                acc[root][1] += v * (q % cols) as f64;
                acc[root][2] += v * (q / cols) as f64;
            }
        }
        let n_grains = acc.iter().filter(|w| w[0] > 0.0).count();
        if n_grains == 0 {
            return None;
        }
        let centre = |q: usize, axis: usize| -> f32 {
            let w = &acc[ptr[q]];
            if w[0] > 0.0 { (w[axis] / w[0]) as f32 } else { f32::NAN }
        };
        let cx = Array2::from_shape_fn((rows, cols), |(r, c)| centre(r * cols + c, 1));
        let cy = Array2::from_shape_fn((rows, cols), |(r, c)| centre(r * cols + c, 2));
        // number the grains in order of their peak pixel
        let mut number = vec![0i64; n];
        let mut centres = Vec::with_capacity(n_grains);
        for (root, w) in acc.iter().enumerate() {
            if w[0] > 0.0 {
                centres.push([w[1] / w[0], w[2] / w[0]]);
                number[root] = centres.len() as i64;
            }
        }
        let label = Array2::from_shape_fn((rows, cols), |(r, c)| number[ptr[r * cols + c]]);
        Some(Self { cx, cy, n_grains, label, centres })
    }

    /// Offset `x - c(x)` of each of `coords` from its grain centre (nearest
    /// pixel; `[0, 0]`, i.e. no correction, where there is no grain).
    pub(crate) fn offsets(&self, coords: &Array2<f64>) -> Vec<[f64; 2]> {
        let (rows, cols) = self.cx.dim();
        (0..coords.nrows())
            .map(|i| {
                let (x, y) = (coords[[i, 0]], coords[[i, 1]]);
                let (c, r) = ((x.round().max(0.0) as usize).min(cols - 1), (y.round().max(0.0) as usize).min(rows - 1));
                let (gx, gy) = (self.cx[[r, c]], self.cy[[r, c]]);
                if gx.is_nan() { [0.0, 0.0] } else { [x - gx as f64, y - gy as f64] }
            })
            .collect()
    }
}

/// Displacement gradients `(ux, vx, uy, vy)` of warp `p` at subset pixel `i`.
#[inline]
fn warp_gradient(p: &[f64], i: usize, coord: [f64; 2], f_coords: &Array2<f64>) -> (f64, f64, f64, f64) {
    if p.len() <= 7 {
        return (p[2], p[3], p[4], p[5]);
    }
    let (dx, dy) = (f_coords[[i, 0]] - coord[0], f_coords[[i, 1]] - coord[1]);
    (p[2] + p[6] * dx + p[8] * dy, p[3] + p[7] * dx + p[9] * dy,
     p[4] + p[8] * dx + p[10] * dy, p[5] + p[9] * dx + p[11] * dy)
}

/// `U - I` for `F = I + grad u = R U` (2-D polar decomposition), as
/// `[[k00, k01], [k10, k11]]`.
#[inline]
fn stretch_minus_identity(ux: f64, vx: f64, uy: f64, vy: f64) -> [[f64; 2]; 2] {
    let f = [[1.0 + ux, uy], [vx, 1.0 + vy]];
    let th = (f[1][0] - f[0][1]).atan2(f[0][0] + f[1][1]);
    let (c, s) = (th.cos(), th.sin());
    // U = R^T F
    [[c * f[0][0] + s * f[1][0] - 1.0, c * f[0][1] + s * f[1][1]],
     [-s * f[0][0] + c * f[1][0], -s * f[0][1] + c * f[1][1] - 1.0]]
}

/// The grains template `f + grad f . (U - I)(x - c)` for warp `p`; `off`
/// holds `x - c` and `grad` the reference gradient at each subset pixel.
/// Order 0 returns `f` unchanged.
pub(crate) fn grains_template(
    f: &Array1<f64>,
    grad: &Array2<f64>,
    off: &[[f64; 2]],
    p: &[f64],
    coord: [f64; 2],
    f_coords: &Array2<f64>,
) -> Array1<f64> {
    if p.len() == 2 {
        return f.clone();
    }
    let uniform = (p.len() <= 7).then(|| stretch_minus_identity(p[2], p[3], p[4], p[5]));
    Array1::from_shape_fn(f.len(), |i| {
        let k = uniform.unwrap_or_else(|| {
            let (ux, vx, uy, vy) = warp_gradient(p, i, coord, f_coords);
            stretch_minus_identity(ux, vx, uy, vy)
        });
        let (dx, dy) = (off[i][0], off[i][1]);
        f[i] + grad[[i, 0]] * (k[0][0] * dx + k[0][1] * dy) + grad[[i, 1]] * (k[1][0] * dx + k[1][1] * dy)
    })
}

/// The template a rigid-texture solve builds at every iteration.
pub(crate) enum TemplateModel {
    /// Round Gaussian speckle: curvature `[fxx, fyy, fxy]` and `s^2`;
    /// `base`: what the area term measures brightness from at each pixel
    /// (`None`: the subset mean).
    Curvature { h: Vec<[f64; 3]>, s2: f64, base: Option<Vec<f64>> },
    /// Opaque grains of any shape: `x - c` and the reference gradient.
    Grains { off: Vec<[f64; 2]>, grad: Array2<f64> },
    /// Flat grains in a matrix, from their edges ([`edge_offsets`]): `x - c`
    /// and the reference gradient, as `Grains`.
    Edges { off: Vec<[f64; 2]>, grad: Array2<f64> },
}

/// `x - c` at each subset pixel for flat grains seen at their edges: the grain
/// centre lies across the edge, on the grain's side, at the grains' mean
/// edge distance `rho`. `dark`: the grains are darker than the matrix (so the
/// centre lies down the gradient). Zero where the brightness is flat (no edge,
/// no correction).
pub(crate) fn edge_offsets(grad: &Array2<f64>, rho: f64, dark: bool) -> Vec<[f64; 2]> {
    let sign = if dark { rho } else { -rho };
    (0..grad.nrows())
        .map(|i| {
            let (gx, gy) = (grad[[i, 0]], grad[[i, 1]]);
            let n = gx.hypot(gy);
            if n > 1e-9 { [sign * gx / n, sign * gy / n] } else { [0.0, 0.0] }
        })
        .collect()
}

impl TemplateModel {
    pub(crate) fn template(&self, f: &Array1<f64>, f_m: f64, p: &[f64], coord: [f64; 2], f_coords: &Array2<f64>) -> Array1<f64> {
        match self {
            Self::Curvature { h, s2, base } => grain_template(f, f_m, base.as_deref(), h, *s2, p, coord, f_coords),
            Self::Grains { off, grad } | Self::Edges { off, grad } => grains_template(f, grad, off, p, coord, f_coords),
        }
    }

    pub(crate) fn texture(&self) -> crate::mesh::Texture {
        match self {
            Self::Curvature { .. } => crate::mesh::Texture::RigidSphere,
            Self::Grains { .. } => crate::mesh::Texture::RigidShape,
            Self::Edges { .. } => crate::mesh::Texture::RigidEdge,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 160;
    const S: f64 = 4.0;

    /// 700 round Gaussian speckles (s = 4 px) on N×N, fixed seed.
    fn centres() -> Vec<[f64; 2]> {
        let mut state = 12345u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..700).map(|_| [rnd() * N as f64, rnd() * N as f64]).collect()
    }

    fn draw(cs: &[[f64; 2]], x: f64, y: f64) -> f64 {
        100.0 * cs.iter().map(|c| (-((x - c[0]).powi(2) + (y - c[1]).powi(2)) / (S * S)).exp()).sum::<f64>()
    }

    /// Compare the model G (at the true order-1 warp `p` about the image
    /// centre) with the exact mismatch g(x + u(x)) − f(x) of speckles each
    /// moved rigidly by u(centre), on the inner 80×80, both with their means
    /// removed: the model uses f − f̄ where the exact first-order mismatch has
    /// f, a constant tr(E)·f̄ apart that ZNCC/ZNSSD ignore. Returns
    /// (correlation, size ratio model/exact, rms exact).
    fn compare(p: [f64; 6]) -> (f64, f64, f64) {
        let cs = centres();
        let c0 = [N as f64 / 2.0, N as f64 / 2.0];
        let u = |x: f64, y: f64| {
            let (dx, dy) = (x - c0[0], y - c0[1]);
            [p[2] * dx + p[4] * dy, p[3] * dx + p[5] * dy]
        };
        let img = Image::from_array(Array2::from_shape_fn((N, N), |(r, c)| draw(&cs, c as f64, r as f64)), 8);
        let maps = GrainMaps::build(&img).expect("speckle image has a model");
        let moved: Vec<[f64; 2]> = cs.iter().map(|c| { let d = u(c[0], c[1]); [c[0] + d[0], c[1] + d[1]] }).collect();
        let pts: Vec<[f64; 2]> = (40..120).flat_map(|y| (40..120).map(move |x| [x as f64, y as f64])).collect();
        let coords = Array2::from_shape_fn((pts.len(), 2), |(i, j)| pts[i][j]);
        let f = Array1::from_shape_fn(pts.len(), |i| img.image_gs[[pts[i][1] as usize, pts[i][0] as usize]]);
        let f_m = f.mean().unwrap();
        let h = maps.sample(&coords);
        let t = grain_template(&f, f_m, None, &h, maps.s2, &p, c0, &coords);
        let exact: Vec<f64> = pts.iter().enumerate().map(|(i, q)| { let d = u(q[0], q[1]); draw(&moved, q[0] + d[0], q[1] + d[1]) - f[i] }).collect();
        let model: Vec<f64> = (0..pts.len()).map(|i| t[i] - f[i]).collect();
        let (em, mm) = (exact.iter().sum::<f64>() / pts.len() as f64, model.iter().sum::<f64>() / pts.len() as f64);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for i in 0..pts.len() {
            let (exact, model) = (exact[i] - em, model[i] - mm);
            sxy += exact * model;
            sxx += exact * exact;
            syy += model * model;
        }
        (sxy / (sxx * syy).sqrt(), (syy / sxx).sqrt(), (sxx / pts.len() as f64).sqrt())
    }

    #[test]
    fn model_matches_rigid_speckles_under_shear() {
        let (corr, ratio, rms) = compare([0.0, 0.0, 0.0, 0.0, 0.05, 0.0]);
        eprintln!("shear 0.05: correlation {corr:.3}, size ratio {ratio:.3}, rms exact {rms:.2}");
        assert!(corr > 0.75, "correlation {corr} (rms exact {rms})");
        assert!((ratio - 1.0).abs() < 0.25, "size ratio {ratio}");
    }

    #[test]
    fn model_matches_rigid_speckles_under_stretch() {
        let (corr, ratio, rms) = compare([0.0, 0.0, 0.05, 0.0, 0.0, 0.0]);
        eprintln!("stretch 0.05: correlation {corr:.3}, size ratio {ratio:.3}, rms exact {rms:.2}");
        assert!(corr > 0.75, "correlation {corr} (rms exact {rms})");
        assert!((ratio - 1.0).abs() < 0.25, "size ratio {ratio}");
    }

    /// A rigid rotation leaves round speckles unchanged: G is zero to
    /// round-off (Green-Lagrange strain).
    #[test]
    fn rotation_gives_no_correction() {
        let cs = centres();
        let img = Image::from_array(Array2::from_shape_fn((N, N), |(r, c)| draw(&cs, c as f64, r as f64)), 8);
        let maps = GrainMaps::build(&img).unwrap();
        let coords = Array2::from_shape_fn((400, 2), |(i, j)| 60.0 + if j == 0 { (i % 20) as f64 } else { (i / 20) as f64 } * 2.0);
        let f = Array1::from_shape_fn(400, |i| img.image_gs[[coords[[i, 1]] as usize, coords[[i, 0]] as usize]]);
        let h = maps.sample(&coords);
        for th in [0.01f64, 0.05, 0.3] {
            let (c, s) = (th.cos(), th.sin());
            let p = [0.0, 0.0, c - 1.0, s, -s, c - 1.0];
            let t = grain_template(&f, f.mean().unwrap(), None, &h, maps.s2, &p, [80.0, 80.0], &coords);
            let worst = (&t - &f).iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(worst < 1e-10, "rotation {th}: |G| up to {worst}");
        }
    }

    /// Order 0 and a zero warp return `f` exactly.
    #[test]
    fn order0_and_zero_warp_return_f() {
        let f = Array1::from_vec(vec![10.0, 20.0, 35.0]);
        let coords = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0]).unwrap();
        let h = vec![[1.0, -2.0, 0.5]; 3];
        assert_eq!(grain_template(&f, 21.0, None, &h, 9.0, &[0.3, -0.2], [1.0, 1.0], &coords), f);
        assert_eq!(grain_template(&f, 21.0, None, &h, 9.0, &[0.3, -0.2, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0], &coords), f);
        assert_eq!(grain_template(&f, 21.0, None, &h, 9.0, &[0.0; 12], [1.0, 1.0], &coords), f);
    }

    /// Order 2 with zero second-order terms equals order 1.
    #[test]
    fn order2_reduces_to_order1() {
        let f = Array1::from_vec(vec![10.0, 20.0, 35.0]);
        let coords = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 4.0, 1.0, 1.0, 5.0]).unwrap();
        let h = vec![[1.0, -2.0, 0.5], [0.3, 0.1, -0.4], [-1.0, 2.0, 0.0]];
        let p1 = [0.1, 0.2, 0.01, -0.02, 0.03, 0.005];
        let mut p2 = [0.0; 12];
        p2[..6].copy_from_slice(&p1);
        let a = grain_template(&f, 21.0, None, &h, 9.0, &p1, [2.0, 2.0], &coords);
        let b = grain_template(&f, 21.0, None, &h, 9.0, &p2, [2.0, 2.0], &coords);
        assert!((&a - &b).iter().all(|v| v.abs() < 1e-12));
    }

    /// Curvature maps of a quadratic image are its exact second derivatives
    /// (times `corr`) away from the edges.
    #[test]
    fn curvature_of_quadratic() {
        let (a, b, c) = (0.02, -0.03, 0.05);
        let img = Image::from_array(Array2::from_shape_fn((40, 40), |(y, x)| {
            let (x, y) = (x as f64 - 20.0, y as f64 - 20.0);
            a * x * x + b * x * y + c * y * y
        }), 8);
        // Correlation length of a smooth quadratic is not speckle; build the
        // filter part directly.
        let k = [kernel(0), kernel(1), kernel(2)];
        let h = [filter_1d(&img.image_gs, &k[0], true), filter_1d(&img.image_gs, &k[1], true), filter_1d(&img.image_gs, &k[2], true)];
        let fxx = filter_1d(&h[2], &k[0], false);
        let fyy = filter_1d(&h[0], &k[2], false);
        let fxy = filter_1d(&h[1], &k[1], false);
        for (m, want) in [(&fxx, 2.0 * a), (&fyy, 2.0 * c), (&fxy, b)] {
            let got = m[[20, 20]];
            assert!((got - want).abs() < 0.01 * want.abs(), "got {got}, want {want}");
        }
        // First derivative sign: x-gradient of x at the centre is +1.
        let ramp = Array2::from_shape_fn((40, 40), |(_, x)| x as f64);
        let gx = filter_1d(&filter_1d(&ramp, &k[1], true), &k[0], false);
        assert!((gx[[20, 20]] - 1.0).abs() < 0.01);
    }

    /// A flat image has no speckle model (no panic): the rigid solve falls
    /// back to the deformable one.
    #[test]
    fn flat_image_has_no_model() {
        let img = Image::from_array(Array2::from_elem((64, 64), 128.0), 8);
        assert!(GrainMaps::build(&img).is_none());
        let tiny = Image::from_array(Array2::from_elem((10, 10), 128.0), 8);
        assert!(GrainMaps::build(&tiny).is_none());
    }

    /// Rigid-blob pair (s = 4 px, 1500 blobs on 200×200) under a uniform
    /// simple shear `gamma` about the centre: each blob drawn at its centre
    /// moved by u = gamma (y − 100). Returns the reference and target images.
    fn shear_pair(gamma: f64) -> (std::sync::Arc<Image>, std::sync::Arc<Image>) {
        let n = 200usize;
        let mut state = 777u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let cs: Vec<[f64; 2]> = (0..1500).map(|_| [rnd() * n as f64, rnd() * n as f64]).collect();
        let moved: Vec<[f64; 2]> = cs.iter().map(|c| [c[0] + gamma * (c[1] - 100.0), c[1]]).collect();
        let f = Array2::from_shape_fn((n, n), |(r, c)| draw(&cs, c as f64, r as f64));
        let g = Array2::from_shape_fn((n, n), |(r, c)| draw(&moved, c as f64, r as f64));
        (std::sync::Arc::new(Image::from_array(f, 8)), std::sync::Arc::new(Image::from_array(g, 8)))
    }

    /// Solve 20 subsets (radius 15, order `order`) both ways; returns per
    /// texture (mean |uy − gamma|, mean ZNSSD, mean iterations, all converged).
    fn solve_grid(gamma: f64, order: usize) -> [(f64, f64, f64, bool); 2] {
        solve_grid_with(gamma, order, false)
    }

    fn solve_grid_with(gamma: f64, order: usize, fagn: bool) -> [(f64, f64, f64, bool); 2] {
        use crate::masks::LocalMask;
        use crate::mesh::Texture;
        use crate::subset::Subset;
        let (fi, gi) = shear_pair(gamma);
        let mask = LocalMask::circle(15).unwrap();
        let mut out = [(0.0, 0.0, 0.0, true); 2];
        for (k, tex) in [Texture::Deformable, Texture::RigidSphere].into_iter().enumerate() {
            let mut cnt = 0.0;
            for yi in 0..4 {
                for xi in 0..5 {
                    let c = [55.0 + 22.5 * xi as f64 + 0.3, 62.0 + 25.0 * yi as f64 + 0.6];
                    let sub = Subset::new(c, &mask, None, fi.clone(), gi.clone(), order).unwrap();
                    let mut p0 = vec![0.0; if order == 1 { 6 } else { 12 }];
                    p0[0] = gamma * (c[1] - 100.0);
                    let r = if fagn {
                        sub.solve_fagn_result(Some(&p0), 0.75, 1e-6, 100, tex).unwrap()
                    } else {
                        sub.solve_icgn_result(Some(&p0), 0.75, 1e-6, 100, tex).unwrap()
                    };
                    out[k].0 += (r.p[4] - gamma).abs();
                    out[k].1 += r.c_znssd;
                    out[k].2 += r.iterations as f64;
                    out[k].3 &= r.converged;
                    cnt += 1.0;
                }
            }
            out[k].0 /= cnt;
            out[k].1 /= cnt;
            out[k].2 /= cnt;
        }
        out
    }

    /// On rigid speckles under shear the rigid template fits better (lower
    /// ZNSSD) and recovers the shear more closely, without costing
    /// convergence.
    #[test]
    fn rigid_icgn_helps_on_rigid_speckle_shear() {
        for order in [1usize, 2] {
            for gamma in [0.02, 0.05, 0.1] {
                let [d, r] = solve_grid(gamma, order);
                eprintln!(
                    "order {order} shear {gamma}: |uy err| def {:.2e} rig {:.2e}; ZNSSD def {:.2e} rig {:.2e}; iters def {:.1} rig {:.1}; converged def {} rig {}",
                    d.0, r.0, d.1, r.1, d.2, r.2, d.3, r.3
                );
                // Order 2 at large shear may leave a subset unconverged: the
                // template follows the warp at every iteration and its last
                // digits can creep (accepted: a reference update follows).
                if order == 1 {
                    assert!(r.3, "order 1 shear {gamma}: rigid did not converge everywhere");
                    assert!(r.1 < d.1, "rigid ZNSSD {} not below deformable {}", r.1, d.1);
                    assert!(r.0 < d.0, "rigid shear error {} not below deformable {}", r.0, d.0);
                }
            }
        }
    }

    /// At zero strain the two textures agree (G = 0 at the solution).
    #[test]
    fn rigid_icgn_matches_deformable_at_zero_strain() {
        for order in [1usize, 2] {
            let [d, r] = solve_grid(0.0, order);
            assert!(d.3 && r.3);
            assert!((d.0 - r.0).abs() < 1e-6 && (d.1 - r.1).abs() < 1e-9, "order {order}: {d:?} vs {r:?}");
        }
    }

    /// FAGN: the same gain as ICGN, and agreement at zero strain.
    #[test]
    fn rigid_fagn_helps_on_rigid_speckle_shear() {
        for order in [1usize, 2] {
            for gamma in [0.0, 0.05, 0.1] {
                let [d, r] = solve_grid_with(gamma, order, true);
                eprintln!(
                    "FAGN order {order} shear {gamma}: |uy err| def {:.2e} rig {:.2e}; ZNSSD def {:.2e} rig {:.2e}; iters def {:.1} rig {:.1}; converged def {} rig {}",
                    d.0, r.0, d.1, r.1, d.2, r.2, d.3, r.3
                );
                assert!(r.3, "order {order} shear {gamma}: rigid did not converge everywhere");
                if gamma == 0.0 {
                    assert!((d.0 - r.0).abs() < 1e-6 && (d.1 - r.1).abs() < 1e-9);
                } else if order == 1 {
                    assert!(r.1 < d.1 && r.0 < d.0, "rigid {r:?} not better than deformable {d:?}");
                }
            }
        }
    }

    /// `residual_map()` after a rigid solve uses the rigid template (smaller
    /// residual than the deformable solve's), and the result records it.
    #[test]
    fn rigid_residual_map_and_record() {
        use crate::masks::LocalMask;
        use crate::mesh::Texture;
        use crate::subset::Subset;
        let (fi, gi) = shear_pair(0.05);
        let mask = LocalMask::circle(15).unwrap();
        let c = [100.3, 100.6];
        let mut rms = [0.0; 2];
        for (k, tex) in [Texture::Deformable, Texture::RigidSphere].into_iter().enumerate() {
            let mut sub = Subset::new(c, &mask, None, fi.clone(), gi.clone(), 1).unwrap();
            sub.solve_icgn(None, 0.75, 1e-6, 100, tex).unwrap();
            assert_eq!(sub.solution().unwrap().result.texture, tex);
            let e = sub.residual_map().unwrap();
            rms[k] = (e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64).sqrt();
        }
        assert!(rms[1] < 0.5 * rms[0], "residual rms deformable {} rigid {}", rms[0], rms[1]);
    }

    /// A rigid solve against a reference with no measurable speckle runs and
    /// falls back to the deformable template (recorded as such); order 0 too.
    #[test]
    fn rigid_falls_back_without_speckle_and_for_order0() {
        use crate::masks::LocalMask;
        use crate::mesh::Texture;
        use crate::subset::Subset;
        // Textured but too small (< 16 px) to measure a correlation length.
        let small = Array2::from_shape_fn((14, 14), |(r, c)| 100.0 + 40.0 * (0.9 * c as f64).sin() * (0.7 * r as f64).cos());
        let img = std::sync::Arc::new(Image::from_array(small, 8));
        assert!(img.grain().is_none());
        let sub = Subset::new([7.0, 7.0], &LocalMask::circle(3).unwrap(), None, img.clone(), img.clone(), 1).unwrap();
        let r = sub.solve_icgn_result(None, 0.0, 1e-5, 20, Texture::RigidSphere).unwrap();
        assert_eq!(r.texture, Texture::Deformable);
        let mask = LocalMask::circle(10).unwrap();
        let (fi, gi) = shear_pair(0.0);
        let sub0 = Subset::new([100.0, 100.0], &mask, None, fi, gi, 0).unwrap();
        let r0 = sub0.solve_icgn_result(None, 0.75, 1e-6, 50, Texture::RigidSphere).unwrap();
        assert!(r0.converged);
        assert_eq!(r0.texture, Texture::Deformable);
    }

    // ---- texture="grains" ----

    /// Opaque elliptical grains (aspect 2, random orientation) on a jittered
    /// grid, each drawn as `exp(-(e1^2/a^2 + e2^2/b^2))` (x200) and read at
    /// `Ud` about its own centre (`U = I` for the reference); a pixel shows its
    /// brightest grain. Returns (image, centres).
    fn ellipse_grains(u: [[f64; 2]; 2]) -> (Array2<f64>, Vec<[f64; 2]>) {
        let mut state = 99u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let (a, b) = (5.0 * 2f64.sqrt(), 5.0 / 2f64.sqrt());
        let mut grains = Vec::new();
        for gy in 0..9 {
            for gx in 0..9 {
                let c = [12.0 + 22.0 * gx as f64 + 4.0 * (rnd() - 0.5), 12.0 + 22.0 * gy as f64 + 4.0 * (rnd() - 0.5)];
                grains.push((c, rnd() * std::f64::consts::PI));
            }
        }
        let img = Array2::from_shape_fn((200, 200), |(r, col)| {
            grains.iter().fold(0.0f64, |m, &(c, phi)| {
                let (dx, dy) = (col as f64 - c[0], r as f64 - c[1]);
                let (qx, qy) = (u[0][0] * dx + u[0][1] * dy, u[1][0] * dx + u[1][1] * dy);
                let (e1, e2) = (phi.cos() * qx + phi.sin() * qy, -phi.sin() * qx + phi.cos() * qy);
                m.max(200.0 * (-(e1 * e1 / (a * a) + e2 * e2 / (b * b))).exp())
            })
        });
        (img, grains.into_iter().map(|g| g.0).collect())
    }

    /// The grain split finds every grain's centre.
    #[test]
    fn grain_centres_found() {
        let (f, centres) = ellipse_grains([[1.0, 0.0], [0.0, 1.0]]);
        let img = Image::from_array(f, 8);
        let gc = GrainCentres::build(&img).expect("grains found");
        let mut worst = 0.0f64;
        for c in &centres {
            let (r, col) = (c[1].round() as usize, c[0].round() as usize);
            let e = (gc.cx[[r, col]] as f64 - c[0]).hypot(gc.cy[[r, col]] as f64 - c[1]);
            worst = worst.max(e);
        }
        assert!(gc.n_grains >= centres.len() && worst < 0.3, "{} grains, worst centre error {worst} px", gc.n_grains);
        // numbering: every pixel's centre is its numbered grain's centre
        assert_eq!(gc.centres.len(), gc.n_grains);
        for ((r, col), &k) in gc.label.indexed_iter() {
            if k > 0 {
                let c = gc.centres[k as usize - 1];
                assert!((c[0] as f32 - gc.cx[[r, col]]).abs() < 1e-3 && (c[1] as f32 - gc.cy[[r, col]]).abs() < 1e-3);
            } else {
                assert!(gc.cx[[r, col]].is_nan());
            }
        }
    }

    /// The grains template matches the exact mismatch of rigid turning grains
    /// under shear and stretch, and is zero for a rotation.
    #[test]
    fn grains_template_matches_exact() {
        let (f0, _) = ellipse_grains([[1.0, 0.0], [0.0, 1.0]]);
        let img = Image::from_array(f0.clone(), 8);
        let gc = GrainCentres::build(&img).unwrap();
        let pts: Vec<[f64; 2]> = (30..170).flat_map(|y| (30..170).map(move |x| [x as f64, y as f64])).collect();
        let coords = Array2::from_shape_fn((pts.len(), 2), |(i, j)| pts[i][j]);
        let f = Array1::from_shape_fn(pts.len(), |i| f0[[pts[i][1] as usize, pts[i][0] as usize]]);
        let grad = Array2::from_shape_fn((pts.len(), 2), |(i, j)| {
            let (x, y) = (pts[i][0] as usize, pts[i][1] as usize);
            if j == 0 { 0.5 * (f0[[y, x + 1]] - f0[[y, x - 1]]) } else { 0.5 * (f0[[y + 1, x]] - f0[[y - 1, x]]) }
        });
        let off = gc.offsets(&coords);
        for (name, p) in [("shear", [0.0, 0.0, 0.0, 0.0, 0.05, 0.0]), ("stretch", [0.0, 0.0, 0.05, 0.0, 0.0, 0.0])] {
            let k = stretch_minus_identity(p[2], p[3], p[4], p[5]);
            let (exact_img, _) = ellipse_grains([[1.0 + k[0][0], k[0][1]], [k[1][0], 1.0 + k[1][1]]]);
            let t = grains_template(&f, &grad, &off, &p, [100.0, 100.0], &coords);
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for (i, q) in pts.iter().enumerate() {
                let e = exact_img[[q[1] as usize, q[0] as usize]] - f[i];
                let m = t[i] - f[i];
                sxy += e * m;
                sxx += e * e;
                syy += m * m;
            }
            let corr = sxy / (sxx * syy).sqrt();
            let ratio = (syy / sxx).sqrt();
            eprintln!("grains {name} 0.05: correlation {corr:.3}, size ratio {ratio:.3}");
            assert!(corr > 0.9 && (ratio - 1.0).abs() < 0.2, "{name}: correlation {corr}, size ratio {ratio}");
        }
        let th = 0.05f64;
        let p = [0.0, 0.0, th.cos() - 1.0, th.sin(), -th.sin(), th.cos() - 1.0];
        let t = grains_template(&f, &grad, &off, &p, [100.0, 100.0], &coords);
        assert!((&t - &f).iter().all(|v| v.abs() < 1e-9), "rotation must give no correction");
    }

    /// No grains in a flat image: `None`, and a grains solve falls back.
    #[test]
    fn grains_flat_image_has_no_grains() {
        use crate::masks::LocalMask;
        use crate::mesh::Texture;
        use crate::subset::Subset;
        let flat = Image::from_array(Array2::from_elem((40, 40), 128.0), 8);
        assert!(GrainCentres::build(&flat).is_none());
        let (f0, _) = ellipse_grains([[1.0, 0.0], [0.0, 1.0]]);
        let img = std::sync::Arc::new(Image::from_array(f0, 8));
        let sub = Subset::new([100.0, 100.0], &LocalMask::circle(15).unwrap(), None, img.clone(), img.clone(), 0).unwrap();
        let r = sub.solve_icgn_result(None, 0.0, 1e-5, 20, Texture::RigidShape).unwrap();
        assert_eq!(r.texture, Texture::Deformable, "order 0: no correction");
    }

    /// On rigid elliptical grains under shear, the grains texture recovers the
    /// shear better than both the deformable and the rigid (round) template.
    #[test]
    fn grains_icgn_beats_deformable_and_rigid_on_ellipses() {
        use crate::masks::LocalMask;
        use crate::mesh::Texture;
        use crate::subset::Subset;
        let gamma = 0.05;
        // reference: grains at their centres; target: grains moved by the shear
        // about the image centre, each turned with the material (shape U about
        // its moved centre). Drawn analytically.
        let (f0, centres) = ellipse_grains([[1.0, 0.0], [0.0, 1.0]]);
        let k = stretch_minus_identity(0.0, 0.0, gamma, 0.0);
        let th = (-gamma).atan2(2.0);
        let rot = |d: [f64; 2]| [th.cos() * d[0] - th.sin() * d[1], th.sin() * d[0] + th.cos() * d[1]];
        let _ = rot;
        let g0 = {
            // target pixel y: grain k at c_k + u(c_k); its own frame offset is R^T (y - c_k - u(c_k))
            let mut state = 99u64;
            let mut rnd = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            let (a, b) = (5.0 * 2f64.sqrt(), 5.0 / 2f64.sqrt());
            let mut grains = Vec::new();
            for gy in 0..9 {
                for gx in 0..9 {
                    let c = [12.0 + 22.0 * gx as f64 + 4.0 * (rnd() - 0.5), 12.0 + 22.0 * gy as f64 + 4.0 * (rnd() - 0.5)];
                    grains.push((c, rnd() * std::f64::consts::PI));
                }
            }
            let _ = (&centres, k);
            Array2::from_shape_fn((200, 200), |(r, col)| {
                grains.iter().fold(0.0f64, |m, &(c, phi)| {
                    let moved = [c[0] + gamma * (c[1] - 100.0), c[1]];
                    let (dx, dy) = (col as f64 - moved[0], r as f64 - moved[1]);
                    // back to the grain's own frame: undo the turn, then the grain's orientation
                    let (qx, qy) = (th.cos() * dx + th.sin() * dy, -th.sin() * dx + th.cos() * dy);
                    let (e1, e2) = (phi.cos() * qx + phi.sin() * qy, -phi.sin() * qx + phi.cos() * qy);
                    m.max(200.0 * (-(e1 * e1 / (a * a) + e2 * e2 / (b * b))).exp())
                })
            })
        };
        let fi = std::sync::Arc::new(Image::from_array(f0, 8));
        let gi = std::sync::Arc::new(Image::from_array(g0, 8));
        let mask = LocalMask::circle(25).unwrap();
        let mut err = [0.0f64; 3];
        let mut used = Vec::new();
        for (t, tex) in [Texture::Deformable, Texture::RigidSphere, Texture::RigidShape].into_iter().enumerate() {
            for yi in 0..4 {
                for xi in 0..4 {
                    let c = [55.0 + 30.0 * xi as f64 + 0.3, 55.0 + 30.0 * yi as f64 + 0.6];
                    let sub = Subset::new(c, &mask, None, fi.clone(), gi.clone(), 1).unwrap();
                    let p0 = [gamma * (c[1] - 100.0), 0.0, 0.0, 0.0, 0.0, 0.0];
                    let r = sub.solve_icgn_result(Some(&p0), 0.0, 1e-6, 100, tex).unwrap();
                    err[t] += (r.p[4] - gamma).abs() / 16.0;
                    used.push(r.texture);
                }
            }
        }
        eprintln!("shear error: deformable {:.2e}, rigid {:.2e}, grains {:.2e}", err[0], err[1], err[2]);
        assert!(used[32..].iter().all(|&t| t == Texture::RigidShape));
        assert!(err[2] < 0.6 * err[0] && err[2] < err[1], "{err:?}");
    }

    /// The coarse part of a linear ramp is the ramp, with no curvature; of a
    /// single small grain, a faint smear.
    #[test]
    fn coarse_part_of_ramp_and_grain() {
        let ramp = Image::from_array(Array2::from_shape_fn((80, 80), |(r, c)| 50.0 + 0.5 * c as f64 + 0.25 * r as f64), 8);
        let coords = ndarray::array![[40.0, 40.0], [35.5, 44.25]];
        let (low, h) = coarse_part(&ramp, &coords, 5.0);
        for i in 0..2 {
            let want = 50.0 + 0.5 * coords[[i, 0]] + 0.25 * coords[[i, 1]];
            assert!((low[i] - want).abs() < 1e-6, "{} vs {want}", low[i]);
            assert!(h[i].iter().all(|v| v.abs() < 1e-9), "{:?}", h[i]);
        }
        let grain = Image::from_array(Array2::from_shape_fn((80, 80), |(r, c)| {
            if ((r as f64 - 40.0).powi(2) + (c as f64 - 40.0).powi(2)).sqrt() < 3.0 { 100.0 } else { 200.0 }
        }), 8);
        let (low, _) = coarse_part(&grain, &ndarray::array![[40.0, 40.0]], 5.0);
        assert!(low[0] > 180.0 && low[0] < 200.0, "{}", low[0]);
    }

    /// Edge offsets point from the grain's centre to the pixel, at the given
    /// distance, for a dark grain; the other way for a bright one.
    #[test]
    fn edge_offsets_point_to_the_grain() {
        let (c, r) = ([40.3, 39.6], 6.0);
        let img = |x: f64, y: f64, dark: bool| {
            let inside = 0.5 * libm::erfc(((x - c[0]).hypot(y - c[1]) - r) / 0.8);
            if dark { 200.0 - 100.0 * inside } else { 100.0 + 100.0 * inside }
        };
        for dark in [true, false] {
            let pts: Vec<[f64; 2]> = (0..80).flat_map(|y| (0..80).map(move |x| [x as f64, y as f64])).collect();
            let grad = Array2::from_shape_fn((pts.len(), 2), |(i, j)| {
                let (x, y) = (pts[i][0], pts[i][1]);
                let h = 1e-3;
                if j == 0 { (img(x + h, y, dark) - img(x - h, y, dark)) / (2.0 * h) } else { (img(x, y + h, dark) - img(x, y - h, dark)) / (2.0 * h) }
            });
            let off = edge_offsets(&grad, r, dark);
            let gmax = (0..pts.len()).map(|i| grad[[i, 0]].hypot(grad[[i, 1]])).fold(0.0, f64::max);
            for (i, p) in pts.iter().enumerate() {
                if grad[[i, 0]].hypot(grad[[i, 1]]) < 0.5 * gmax {
                    continue;
                }
                let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
                let cos = (off[i][0] * dx + off[i][1] * dy) / (off[i][0].hypot(off[i][1]) * dx.hypot(dy));
                assert!(cos > 0.999, "pixel {p:?}: offset {:?} not along x - c", off[i]);
                assert!((off[i][0].hypot(off[i][1]) - r).abs() < 1e-9);
            }
        }
    }
}
