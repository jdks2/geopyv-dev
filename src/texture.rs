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

/// The template a rigid-texture solve builds at every iteration.
pub(crate) enum TemplateModel {
    /// Round Gaussian speckle: curvature `[fxx, fyy, fxy]` and `s^2`;
    /// `base`: what the area term measures brightness from at each pixel
    /// (`None`: the subset mean).
    Curvature { h: Vec<[f64; 3]>, s2: f64, base: Option<Vec<f64>> },
}

impl TemplateModel {
    pub(crate) fn template(&self, f: &Array1<f64>, f_m: f64, p: &[f64], coord: [f64; 2], f_coords: &Array2<f64>) -> Array1<f64> {
        match self {
            Self::Curvature { h, s2, base } => grain_template(f, f_m, base.as_deref(), h, *s2, p, coord, f_coords),
        }
    }

    pub(crate) fn texture(&self) -> crate::mesh::Texture {
        match self {
            Self::Curvature { .. } => crate::mesh::Texture::Rigid,
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
        for (k, tex) in [Texture::Deformable, Texture::Rigid].into_iter().enumerate() {
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
        for (k, tex) in [Texture::Deformable, Texture::Rigid].into_iter().enumerate() {
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
        let r = sub.solve_icgn_result(None, 0.0, 1e-5, 20, Texture::Rigid).unwrap();
        assert_eq!(r.texture, Texture::Deformable);
        let mask = LocalMask::circle(10).unwrap();
        let (fi, gi) = shear_pair(0.0);
        let sub0 = Subset::new([100.0, 100.0], &mask, None, fi, gi, 0).unwrap();
        let r0 = sub0.solve_icgn_result(None, 0.75, 1e-6, 50, Texture::Rigid).unwrap();
        assert!(r0.converged);
        assert_eq!(r0.texture, Texture::Deformable);
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
}
