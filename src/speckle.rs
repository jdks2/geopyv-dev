//! Synthetic speckle-pattern image generator.

use std::path::Path;

use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::Normal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::Error;

// ---------------------------------------------------------------------------
// Warp mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShearBandOption {
    Sin,
    Lin,
    Quad,
    /// `Lin`-shaped band whose corners are replaced by a linear ramp of
    /// length `tau` (px) in `du/dy`, rather than jumping straight from the
    /// flat `strain` plateau to zero: `du/dy` is `0` for `|dy| >= width/2`,
    /// ramps linearly to `strain` over `width/2 - tau <= |dy| < width/2`,
    /// and is flat at `strain` for `|dy| < width/2 - tau`. `tau -> 0`
    /// recovers `Lin` exactly; `tau == width/2` removes the flat plateau
    /// entirely, giving a pure triangular ("sawtooth") `du/dy` profile
    /// peaking at the band centre. Requires `0 < tau <= width/2`.
    Smooth { tau: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarpMode {
    /// Full second-order polynomial warp; uses all 12 `comp` parameters.
    None,
    /// Rigid-body rotation around `origin`. `comp[0]` stores the final angle in radians;
    /// the actual angle at image `i` is `comp[0] * deform_mult_at(i)`.
    Rotation,
    /// 1-D shear-band displacement varying across the band width.
    ShearBand { option: ShearBandOption, width: f64 },
    /// Circular ("rotating core") shear band around `origin`: a rigid disc of
    /// radius `r1` rotates by `comp[0] * deform_mult_at(i)` radians; an
    /// annulus from `r1` to `r2` linearly tapers that rotation to zero;
    /// beyond `r2` the field is exactly stationary. Closed form -- see
    /// `mds/circular_shear_band_speckle.md` for the derivation and its
    /// verification against the old Python package's brute-force
    /// (1000-substep) numerical integration of the same mode.
    ///
    /// `tau`, when `Some`, rounds the two corners of the annulus's `dphi/dr`
    /// box (at `r1` and `r2`) with a linear ramp of length `tau` (px),
    /// analogous to `ShearBandOption::Smooth` -- see
    /// `geopyv_dev_run/code/dev/performance_evaluation.md`'s "0b" section
    /// for the derivation. Unlike `Smooth` (which lets its outer saturation
    /// value drift by `O(tau)`), this keeps `phi(r1)=theta` (core rotation)
    /// and `phi(r2)=0` (far-field) exactly fixed -- both are externally
    /// specified physical parameters here, not derived quantities -- by
    /// steepening the flat-region rate `s = -theta/(r2-r1-tau)` to
    /// compensate. Requires `0 < tau <= (r2-r1)/2`. `None` is bit-for-bit
    /// identical to the pre-existing hard-edged behaviour.
    CircularShear { r1: f64, r2: f64, tau: Option<f64> },
}

// ---------------------------------------------------------------------------
// Progression and scale type
// ---------------------------------------------------------------------------

/// Which attribute progresses across the image series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Progression {
    /// Warp magnitude progresses (lin/log); noise is constant at final values.
    Deformation,
    /// Noise magnitude progresses (lin/log); deformation is constant at full comp.
    Noise,
}

/// Spacing of the progression from 0 → 1 across `image_no` images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleType {
    /// Image 0 is reference (mult = 0); images 1..image_no-1 are linear from 0 to 1.
    Lin,
    /// Image 0 is reference (mult = 0); images 1..image_no-1 are log-spaced from `min` to 1.
    Log { min: f64 },
}

// ---------------------------------------------------------------------------
// Speckle struct
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Speckle {
    // image config
    pub image_dir: String,
    pub name: String,
    pub file_format: String,
    pub image_size: (usize, usize),   // (width, height)
    // speckle config
    pub speckle_size: f64,
    pub speckle_number: usize,
    /// When `true`, each speckle's Gaussian is rendered only within
    /// `ceil(1.5 * speckle_size)` px of its centre instead of the fixed
    /// ±100px window -- safe (not lossy) at the speckle sizes this project
    /// uses, since a pixel's contribution is already below the `u8`
    /// rounding threshold by `r ~ 1.15 * speckle_size`; see
    /// `geopyv_dev_run/code/dev/performance_evaluation.md`'s "0a" section.
    pub speckle_limit: bool,
    // progression
    pub progression: Progression,
    // deformation config
    pub comp: [f64; 12],
    pub origin: [f64; 2],
    pub mode: WarpMode,
    // noise config
    pub noise_pos: f64,
    pub noise_int: f64,
    // scale config
    pub image_no: usize,
    pub scale: ScaleType,
    // post-solve
    pub speckle_positions: Option<Array2<f64>>,
    pub solved: bool,
}

impl Speckle {
    pub fn new(
        image_dir: String,
        name: String,
        file_format: String,
        image_size: (usize, usize),
        speckle_size: f64,
        speckle_number: usize,
        speckle_limit: bool,
        progression: Progression,
        comp: [f64; 12],
        origin: [f64; 2],
        noise_pos: f64,
        noise_int: f64,
        mode: WarpMode,
        image_no: usize,
        scale: ScaleType,
    ) -> Result<Self, Error> {
        if speckle_size <= 0.0 {
            return Err(Error::InvalidInput("speckle_size must be > 0".into()));
        }
        if speckle_number == 0 {
            return Err(Error::InvalidInput("speckle_number must be >= 1".into()));
        }
        if image_no == 0 {
            return Err(Error::InvalidInput("image_no must be >= 1".into()));
        }
        if let WarpMode::ShearBand { option: ShearBandOption::Smooth { tau }, width } = &mode {
            if *tau <= 0.0 {
                return Err(Error::InvalidInput(
                    "ShearBandOption::Smooth requires tau > 0 (use Lin for tau == 0)".into(),
                ));
            }
            if *tau > width / 2.0 {
                return Err(Error::InvalidInput(
                    "ShearBandOption::Smooth requires tau <= width/2 (tau == width/2 gives \
                     a pure triangular/sawtooth profile with no flat plateau; a larger tau \
                     would need the two ramps to cross, which is not supported)".into(),
                ));
            }
        }
        if let WarpMode::CircularShear { r1, r2, tau } = &mode {
            if *r1 <= 0.0 {
                return Err(Error::InvalidInput(
                    "CircularShear requires r1 > 0".into(),
                ));
            }
            if *r2 <= *r1 {
                return Err(Error::InvalidInput(
                    "CircularShear requires r2 > r1".into(),
                ));
            }
            if let Some(t) = tau {
                if *t <= 0.0 {
                    return Err(Error::InvalidInput(
                        "CircularShear tau must be > 0 (use None for a hard edge)".into(),
                    ));
                }
                if *t > (r2 - r1) / 2.0 {
                    return Err(Error::InvalidInput(
                        "CircularShear requires tau <= (r2-r1)/2 (tau == (r2-r1)/2 gives \
                         a pure triangular dphi/dr profile with no flat plateau; a larger tau \
                         would need the two ramps to cross, which is not supported)".into(),
                    ));
                }
            }
        }
        Ok(Self {
            image_dir,
            name,
            file_format,
            image_size,
            speckle_size,
            speckle_number,
            speckle_limit,
            progression,
            comp,
            origin,
            mode,
            noise_pos,
            noise_int,
            image_no,
            scale,
            speckle_positions: Option::None,
            solved: false,
        })
    }

    /// Raw lin/log scale multiplier at image `i` (0 → 1, regardless of progression type).
    /// Image 0 is always 0 (reference).
    pub(crate) fn scale_mult_at(&self, i: usize) -> f64 {
        let n = self.image_no;
        if n <= 1 {
            return if i == 0 { 0.0 } else { 1.0 };
        }
        match &self.scale {
            ScaleType::Lin => i as f64 / (n - 1) as f64,
            ScaleType::Log { min } => {
                if i == 0 {
                    0.0
                } else {
                    let log_min = min.max(f64::MIN_POSITIVE).log10();
                    let t = (i - 1) as f64 / (n - 2).max(1) as f64;
                    10.0_f64.powf(log_min + t * (0.0 - log_min))
                }
            }
        }
    }

    /// Deformation multiplier at image `i`.
    ///
    /// For `Progression::Deformation`: equals `scale_mult_at(i)`.
    /// For `Progression::Noise`: 0 for image 0 (reference), 1 for all others.
    pub(crate) fn deform_mult_at(&self, i: usize) -> f64 {
        match self.progression {
            Progression::Deformation => self.scale_mult_at(i),
            Progression::Noise => if i == 0 { 0.0 } else { 1.0 },
        }
    }

    pub fn solve(&mut self, seed: u64) -> Result<(), Error> {
        std::fs::create_dir_all(&self.image_dir)?;

        let positions = generate_speckles(self.speckle_number, self.image_size, seed);
        self.speckle_positions = Some(positions.clone());

        let pb = indicatif::ProgressBar::new(self.image_no as u64);
        pb.set_style(
            indicatif::ProgressStyle::with_template(
                "  Generating:    [{bar:40.cyan}] {pos}/{len} images  eta {eta}"
            )
            .unwrap()
            .progress_chars("█░"),
        );

        for i in 0..self.image_no {
            let deform_mult = self.deform_mult_at(i);
            let noise_mult = match self.progression {
                Progression::Deformation => 1.0_f64,
                Progression::Noise => self.scale_mult_at(i),
            };

            let pm_i: [f64; 12] = std::array::from_fn(|k| self.comp[k] * deform_mult);
            let noise_pos_i = self.noise_pos * noise_mult;
            let noise_int_i = self.noise_int * noise_mult;

            let displaced = compute_displaced(&self.mode, positions.view(), &pm_i, self.origin);
            let img = render_image(
                displaced.view(),
                self.image_size,
                self.speckle_size,
                self.speckle_limit,
                noise_pos_i,
                noise_int_i,
                seed.wrapping_add(i as u64),
            )?;
            let path = Path::new(&self.image_dir)
                .join(format!("{}_{}{}", self.name, i, self.file_format));
            save_image(&img, &path)?;
            pb.inc(1);
        }
        pb.finish_and_clear();

        self.solved = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `u(dy)` for [`ShearBandOption::Smooth`]: `du/dy` is flat at `strain` for
/// `a = |dy| < p` (`p = b - tau`, the plateau half-width), ramps linearly
/// to zero over `p <= a < b`, and is zero for `a >= b` -- see the
/// `ShearBandOption::Smooth` doc comment. `u` is odd in `dy` (matching
/// `Sin`/`Lin`'s `u(0) = 0` convention); computed via the positive-side
/// magnitude `u_mag(a)` (itself carrying `strain`'s own sign) with the
/// sign of `dy` applied explicitly, since `strain` may itself be negative
/// (a naive `copysign` would discard that).
fn smooth_step_u(dy: f64, strain: f64, b: f64, tau: f64) -> f64 {
    let a = dy.abs();
    let p = b - tau;
    let u_mag = if a < p {
        strain * a
    } else if a < b {
        strain / tau * (b * a - 0.5 * a * a) - strain * p * p / (2.0 * tau)
    } else {
        strain * (b - 0.5 * tau)
    };
    if dy >= 0.0 { u_mag } else { -u_mag }
}

/// `(du/dy, d2u/dy2)` at `dy` for [`ShearBandOption::Smooth`] -- exact
/// derivatives of [`smooth_step_u`]'s piecewise-linear `du/dy` profile.
/// `du/dy` is even in `dy` (uses `a = |dy|` directly); `d2u/dy2` is odd
/// (nonzero, constant magnitude `strain/tau`, only within the two ramp
/// regions -- `du/dy` is piecewise *linear* there, unlike `Lin`'s
/// piecewise-*constant* profile whose second derivative is zero
/// everywhere except unrepresented breakpoint deltas).
fn smooth_step_derivatives(dy: f64, strain: f64, b: f64, tau: f64) -> (f64, f64) {
    let a = dy.abs();
    let p = b - tau;
    if a < p {
        (strain, 0.0)
    } else if a < b {
        (strain * (b - a) / tau, -strain / tau * dy.signum())
    } else {
        (0.0, 0.0)
    }
}

/// `(phi, dphi/dr)` for `WarpMode::CircularShear`'s annulus (`r1 <= r < r2`),
/// with corner-sharpness `tau` applied (`tau <= 0.0` recovers the hard-edged
/// `theta * (r2 - r) / (r2 - r1)` exactly). See `WarpMode::CircularShear`'s
/// doc comment and `geopyv_dev_run/code/dev/performance_evaluation.md`'s
/// "0b" section for the derivation: `phi(r1)=theta` and `phi(r2)=0` are kept
/// exactly fixed (unlike `smooth_step_u`'s drifting outer saturation) by
/// steepening the flat-region rate `s = -theta/(r2-r1-tau)`.
fn circular_shear_annulus(r: f64, theta: f64, r1: f64, r2: f64, tau: f64) -> (f64, f64) {
    if tau <= 0.0 {
        return (theta * (r2 - r) / (r2 - r1), -theta / (r2 - r1));
    }
    let s = -theta / (r2 - r1 - tau);
    if r < r1 + tau {
        let a = r - r1;
        (theta + s * a * a / (2.0 * tau), s * a / tau)
    } else if r < r2 - tau {
        (theta + s * (r - r1 - tau / 2.0), s)
    } else {
        let q = r2 - r;
        (-s * q * q / (2.0 * tau), s * q / tau)
    }
}

/// Generate speckle positions uniformly over the full image extent.
fn generate_speckles(
    count: usize,
    image_size: (usize, usize),
    seed: u64,
) -> Array2<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut out = Array2::zeros((count, 2));
    for i in 0..count {
        out[[i, 0]] = rng.r#gen::<f64>() * image_size.0 as f64;
        out[[i, 1]] = rng.r#gen::<f64>() * image_size.1 as f64;
    }
    out
}

/// Compute displaced speckle positions for increment `i`.
/// Returns (N, 2) array of [x, y] positions.
fn compute_displaced(
    mode: &WarpMode,
    ref_speckles: ArrayView2<f64>,
    pm_i: &[f64; 12],
    origin: [f64; 2],
) -> Array2<f64> {
    let n = ref_speckles.nrows();
    let mut out = Array2::zeros((n, 2));

    match mode {
        WarpMode::Rotation => {
            // pm_i[0] = comp[0] * deform_mult_at(i) = current angle in radians.
            let theta = pm_i[0];
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            for i in 0..n {
                let dx = ref_speckles[[i, 0]] - origin[0];
                let dy = ref_speckles[[i, 1]] - origin[1];
                out[[i, 0]] = origin[0] + cos_t * dx - sin_t * dy;
                out[[i, 1]] = origin[1] + sin_t * dx + cos_t * dy;
            }
        }
        WarpMode::None => {
            for i in 0..n {
                let dx = ref_speckles[[i, 0]] - origin[0];
                let dy = ref_speckles[[i, 1]] - origin[1];
                let u = pm_i[0]
                    + pm_i[2] * dx
                    + pm_i[4] * dy
                    + 0.5 * pm_i[6] * dx * dx
                    + pm_i[8] * dx * dy
                    + 0.5 * pm_i[10] * dy * dy;
                let v = pm_i[1]
                    + pm_i[3] * dx
                    + pm_i[5] * dy
                    + 0.5 * pm_i[7] * dx * dx
                    + pm_i[9] * dx * dy
                    + 0.5 * pm_i[11] * dy * dy;
                out[[i, 0]] = ref_speckles[[i, 0]] + u;
                out[[i, 1]] = ref_speckles[[i, 1]] + v;
            }
        }
        WarpMode::ShearBand { option, width } => {
            let b = width / 2.0;
            let strain = pm_i[4];
            for i in 0..n {
                let dy = ref_speckles[[i, 1]] - origin[1];
                let u = match option {
                    ShearBandOption::Sin => {
                        let a = b * strain;
                        if dy.abs() < b {
                            a * (std::f64::consts::PI * dy / width).sin()
                        } else if dy >= b {
                            a
                        } else {
                            -a
                        }
                    }
                    ShearBandOption::Lin => {
                        let a = b * strain;
                        if dy.abs() < b {
                            strain * dy
                        } else if dy >= b {
                            a
                        } else {
                            -a
                        }
                    }
                    ShearBandOption::Quad => {
                        if dy >= 0.0 {
                            0.5 * strain * dy * dy
                        } else {
                            0.0
                        }
                    }
                    ShearBandOption::Smooth { tau } => smooth_step_u(dy, strain, b, *tau),
                };
                out[[i, 0]] = ref_speckles[[i, 0]] + u;
                out[[i, 1]] = ref_speckles[[i, 1]];
            }
        }
        WarpMode::CircularShear { r1, r2, tau } => {
            // pm_i[0] = comp[0] * deform_mult_at(i) = current core rotation
            // angle in radians (same convention as WarpMode::Rotation).
            let theta = pm_i[0];
            let tau = tau.unwrap_or(0.0);
            for i in 0..n {
                let dx = ref_speckles[[i, 0]] - origin[0];
                let dy = ref_speckles[[i, 1]] - origin[1];
                let r = (dx * dx + dy * dy).sqrt();
                let phi = if r < *r1 {
                    theta
                } else if r < *r2 {
                    circular_shear_annulus(r, theta, *r1, *r2, tau).0
                } else {
                    out[[i, 0]] = ref_speckles[[i, 0]];
                    out[[i, 1]] = ref_speckles[[i, 1]];
                    continue;
                };
                let (sin_p, cos_p) = phi.sin_cos();
                out[[i, 0]] = ref_speckles[[i, 0]] + (cos_p * dx - sin_p * dy - dx);
                out[[i, 1]] = ref_speckles[[i, 1]] + (sin_p * dx + cos_p * dy - dy);
            }
        }
    }
    out
}

/// Per-speckle render half-window (px). When `speckle_limit` is true,
/// restricts to `ceil(1.5 * speckle_size)` -- safe, not lossy, since a
/// pixel's Gaussian contribution is already below the `u8` rounding
/// threshold by `r ~ 1.15 * speckle_size`; see
/// `geopyv_dev_run/code/dev/performance_evaluation.md`'s "0a" section.
/// Otherwise the pre-existing fixed ±100px window.
fn render_half_window(speckle_size: f64, speckle_limit: bool) -> isize {
    if speckle_limit {
        (1.5 * speckle_size).ceil() as isize
    } else {
        100
    }
}

/// Render displaced speckle positions as Gaussian blobs into a (H, W) u8 image.
/// Parallelised over speckles with rayon.
fn render_image(
    speckles: ArrayView2<f64>,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_limit: bool,
    noise_pos: f64,
    noise_int: f64,
    seed: u64,
) -> Result<Array2<u8>, Error> {
    let (w, h) = image_size;
    let sigma_sq_4 = speckle_size * speckle_size / 4.0;
    let half_window = render_half_window(speckle_size, speckle_limit);
    let n = speckles.nrows();

    // Apply position noise and collect to owned Vec for rayon.
    let positions: Vec<(f64, f64)> = if noise_pos > 0.0 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, noise_pos)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        (0..n)
            .map(|i| {
                (
                    speckles[[i, 0]] + normal.sample(&mut rng),
                    speckles[[i, 1]] + normal.sample(&mut rng),
                )
            })
            .collect()
    } else {
        (0..n).map(|i| (speckles[[i, 0]], speckles[[i, 1]])).collect()
    };

    // Parallel render: each thread owns its accumulation buffer.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (n / n_threads).max(1);

    let thread_buffers: Vec<Vec<f64>> = positions
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut buf = vec![0.0f64; w * h];
            for &(cx, cy) in chunk {
                let x_lo_s = cx as isize - half_window;
                let x_hi_s = cx as isize + half_window + 1;
                let y_lo_s = cy as isize - half_window;
                let y_hi_s = cy as isize + half_window + 1;
                // Skip speckles entirely outside the image.
                if x_hi_s <= 0 || x_lo_s >= w as isize
                    || y_hi_s <= 0 || y_lo_s >= h as isize
                {
                    continue;
                }
                let x_lo = x_lo_s.max(0) as usize;
                let x_hi = x_hi_s.min(w as isize) as usize;
                let y_lo = y_lo_s.max(0) as usize;
                let y_hi = y_hi_s.min(h as isize) as usize;
                for py in y_lo..y_hi {
                    for px in x_lo..x_hi {
                        let ddx = px as f64 - cx;
                        let ddy = py as f64 - cy;
                        buf[py * w + px] +=
                            (-(ddx * ddx + ddy * ddy) / sigma_sq_4).exp();
                    }
                }
            }
            buf
        })
        .collect();

    // Sum thread buffers.
    let mut total = vec![0.0f64; w * h];
    for buf in &thread_buffers {
        for (t, &b) in total.iter_mut().zip(buf.iter()) {
            *t += b;
        }
    }

    // Apply intensity noise, scale to [0, 255], clip and cast.
    let mut out = Array2::<u8>::zeros((h, w));
    if noise_int > 0.0 {
        let mut rng = SmallRng::seed_from_u64(seed ^ 0xDEAD_BEEF);
        let normal = Normal::new(0.0, noise_int)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        for py in 0..h {
            for px in 0..w {
                let v = total[py * w + px] + normal.sample(&mut rng);
                out[[py, px]] = ((v * 200.0).clamp(0.0, 255.0)) as u8;
            }
        }
    } else {
        for py in 0..h {
            for px in 0..w {
                out[[py, px]] = ((total[py * w + px] * 200.0).clamp(0.0, 255.0)) as u8;
            }
        }
    }
    Ok(out)
}

fn save_image(img: &Array2<u8>, path: &Path) -> Result<(), Error> {
    let (h, w) = (img.nrows(), img.ncols());
    let flat: Vec<u8> = img.iter().cloned().collect();
    image::GrayImage::from_raw(w as u32, h as u32, flat)
        .expect("buffer size mismatch")
        .save(path)
        .map_err(|e| Error::Io(e.to_string()))
}

// ---------------------------------------------------------------------------
// Ground-truth warp field computation (used by Validation)
// ---------------------------------------------------------------------------

impl Speckle {
    pub fn warp(&self, i: usize, coordinates: ArrayView2<f64>) -> Result<Array2<f64>, Error> {
        if coordinates.ncols() != 2 {
            return Err(Error::InvalidInput("coordinates must have 2 columns".into()));
        }
        let n = coordinates.nrows();
        let mult = self.deform_mult_at(i);
        let pm: [f64; 12] = std::array::from_fn(|k| self.comp[k] * mult);
        let ox = self.origin[0];
        let oy = self.origin[1];
        let mut out = Array2::zeros((n, 12));

        match &self.mode {
            WarpMode::Rotation => {
                // pm[0] = comp[0] * deform_mult_at(i) = current angle in radians.
                let theta = pm[0];
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                for p in 0..n {
                    let dx = coordinates[[p, 0]] - ox;
                    let dy = coordinates[[p, 1]] - oy;
                    out[[p, 0]] = (cos_t - 1.0) * dx - sin_t * dy;
                    out[[p, 1]] = sin_t * dx + (cos_t - 1.0) * dy;
                    out[[p, 2]] = cos_t - 1.0;
                    out[[p, 3]] = sin_t;
                    out[[p, 4]] = -sin_t;
                    out[[p, 5]] = cos_t - 1.0;
                    // second-order terms are zero for rigid rotation
                }
            }
            WarpMode::None => {
                for p in 0..n {
                    let dx = coordinates[[p, 0]] - ox;
                    let dy = coordinates[[p, 1]] - oy;
                    out[[p, 0]] = pm[0] + pm[2]*dx + pm[4]*dy
                        + 0.5*pm[6]*dx*dx + pm[8]*dx*dy + 0.5*pm[10]*dy*dy;
                    out[[p, 1]] = pm[1] + pm[3]*dx + pm[5]*dy
                        + 0.5*pm[7]*dx*dx + pm[9]*dx*dy + 0.5*pm[11]*dy*dy;
                    out[[p, 2]] = pm[2] + pm[6]*dx + pm[8]*dy;
                    out[[p, 3]] = pm[3] + pm[7]*dx + pm[9]*dy;
                    out[[p, 4]] = pm[4] + pm[8]*dx + pm[10]*dy;
                    out[[p, 5]] = pm[5] + pm[9]*dx + pm[11]*dy;
                    for k in 6..12usize {
                        out[[p, k]] = pm[k];
                    }
                }
            }
            WarpMode::ShearBand { option, width } => {
                let b = width / 2.0;
                let strain = pm[4];
                for p in 0..n {
                    let dy = coordinates[[p, 1]] - oy;
                    let (u, du_dy, d2u_dy2) = match option {
                        ShearBandOption::Sin => {
                            let a = b * strain;
                            if dy.abs() < b {
                                let pi_dy_w = std::f64::consts::PI * dy / width;
                                (
                                    a * pi_dy_w.sin(),
                                    a * std::f64::consts::PI / width * pi_dy_w.cos(),
                                    -a * (std::f64::consts::PI / width).powi(2) * pi_dy_w.sin(),
                                )
                            } else if dy >= b {
                                (a, 0.0, 0.0)
                            } else {
                                (-a, 0.0, 0.0)
                            }
                        }
                        ShearBandOption::Lin => {
                            let a = b * strain;
                            if dy.abs() < b {
                                (strain * dy, strain, 0.0)
                            } else if dy >= b {
                                (a, 0.0, 0.0)
                            } else {
                                (-a, 0.0, 0.0)
                            }
                        }
                        ShearBandOption::Quad => {
                            if dy >= 0.0 {
                                (0.5 * strain * dy * dy, strain * dy, strain)
                            } else {
                                (0.0, 0.0, 0.0)
                            }
                        }
                        ShearBandOption::Smooth { tau } => {
                            let u = smooth_step_u(dy, strain, b, *tau);
                            let (du_dy, d2u_dy2) = smooth_step_derivatives(dy, strain, b, *tau);
                            (u, du_dy, d2u_dy2)
                        }
                    };
                    out[[p, 0]] = u;
                    out[[p, 4]] = du_dy;
                    out[[p, 10]] = d2u_dy2;
                }
            }
            WarpMode::CircularShear { r1, r2, tau } => {
                // pm[0] = comp[0] * deform_mult_at(i) = current core rotation
                // angle. Closed-form displacement and Jacobian -- both
                // verified against the old package's brute-force incremental
                // integration / central finite differences respectively; see
                // mds/circular_shear_band_speckle.md.
                let theta = pm[0];
                let tau = tau.unwrap_or(0.0);
                for p in 0..n {
                    let dx = coordinates[[p, 0]] - ox;
                    let dy = coordinates[[p, 1]] - oy;
                    let r = (dx * dx + dy * dy).sqrt();
                    if r < *r1 {
                        let (sin_t, cos_t) = theta.sin_cos();
                        out[[p, 0]] = cos_t * dx - sin_t * dy - dx;
                        out[[p, 1]] = sin_t * dx + cos_t * dy - dy;
                        out[[p, 2]] = cos_t - 1.0;
                        out[[p, 3]] = sin_t;
                        out[[p, 4]] = -sin_t;
                        out[[p, 5]] = cos_t - 1.0;
                    } else if r < *r2 {
                        let (phi, k) = circular_shear_annulus(r, theta, *r1, *r2, tau);
                        let (sin_p, cos_p) = phi.sin_cos();
                        out[[p, 0]] = cos_p * dx - sin_p * dy - dx;
                        out[[p, 1]] = sin_p * dx + cos_p * dy - dy;
                        // k = dphi/dr (locally constant within whichever of
                        // the up-ramp/flat/down-ramp sub-regions `r` falls
                        // in -- see circular_shear_annulus); chain rule
                        // gives dphi/dx = k*dx/r, dphi/dy = k*dy/r exactly
                        // as in the hard-edged (tau=None) case.
                        out[[p, 2]] = cos_p - 1.0 - k * (dx / r) * (sin_p * dx + cos_p * dy);
                        out[[p, 3]] = sin_p + k * (dx / r) * (cos_p * dx - sin_p * dy);
                        out[[p, 4]] = -sin_p - k * (dy / r) * (sin_p * dx + cos_p * dy);
                        out[[p, 5]] = cos_p - 1.0 + k * (dy / r) * (cos_p * dx - sin_p * dy);
                    }
                    // r >= r2: stationary far field, out[[p, ..]] stays 0.
                    // Second-order terms are zero everywhere -- the annulus
                    // Jacobian has a kink at r1/r2 (piecewise-C0 only), same
                    // as ShearBand{Lin}'s corners.
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_deformation_speckle(comp: [f64; 12], image_no: usize) -> Speckle {
        Speckle::new(
            "/tmp".into(), "s".into(), ".jpg".into(),
            (100, 100),
            1.0,
            10,
            false,
            Progression::Deformation,
            comp,
            [50.0, 50.0],
            0.0, 0.0,
            WarpMode::None,
            image_no,
            ScaleType::Lin,
        ).unwrap()
    }

    #[test]
    fn test_generate_speckles_count_and_bounds() {
        let pos = generate_speckles(300, (1001, 1001), 0);
        assert_eq!(pos.nrows(), 300);
        for i in 0..300 {
            assert!(pos[[i, 0]] >= 0.0 && pos[[i, 0]] <= 1001.0);
            assert!(pos[[i, 1]] >= 0.0 && pos[[i, 1]] <= 1001.0);
        }
    }

    #[test]
    fn test_compute_displaced_none_zero_comp() {
        let ref_pos = ndarray::arr2(&[[100.0, 200.0], [300.0, 400.0]]);
        let pm_i = [0.0f64; 12];
        let out = compute_displaced(&WarpMode::None, ref_pos.view(), &pm_i, [500.0, 500.0]);
        assert!((out[[0, 0]] - 100.0).abs() < 1e-12);
        assert!((out[[0, 1]] - 200.0).abs() < 1e-12);
        assert!((out[[1, 0]] - 300.0).abs() < 1e-12);
        assert!((out[[1, 1]] - 400.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_displaced_sb_sin_at_origin() {
        let ref_pos = ndarray::arr2(&[[100.0, 500.0]]);
        let mut pm_i = [0.0f64; 12];
        pm_i[4] = 0.1;
        let out = compute_displaced(
            &WarpMode::ShearBand { option: ShearBandOption::Sin, width: 200.0 },
            ref_pos.view(),
            &pm_i,
            [500.0, 500.0],
        );
        assert!((out[[0, 0]] - 100.0).abs() < 1e-12);
        assert!((out[[0, 1]] - 500.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_displaced_sb_sin_saturation() {
        let ref_pos = ndarray::arr2(&[[100.0, 900.0]]);
        let mut pm_i = [0.0f64; 12];
        pm_i[4] = 0.2;
        let width = 100.0_f64;
        let b = width / 2.0;
        let out = compute_displaced(
            &WarpMode::ShearBand { option: ShearBandOption::Sin, width },
            ref_pos.view(),
            &pm_i,
            [500.0, 500.0],
        );
        let expected_u = b * pm_i[4];
        assert!((out[[0, 0]] - (100.0 + expected_u)).abs() < 1e-12);
    }

    #[test]
    fn test_scale_mult_linear() {
        let s = Speckle::new(
            "/tmp".into(), "s".into(), ".jpg".into(), (10,10), 1.0, 1,
            false, Progression::Deformation, [0.0;12], [5.0,5.0], 0.0, 0.0,
            WarpMode::None, 5, ScaleType::Lin,
        ).unwrap();
        assert!((s.scale_mult_at(0)).abs() < 1e-12);
        assert!((s.scale_mult_at(2) - 0.5).abs() < 1e-12);
        assert!((s.scale_mult_at(4) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_scale_mult_log() {
        let s = Speckle::new(
            "/tmp".into(), "s".into(), ".jpg".into(), (10,10), 1.0, 1,
            false, Progression::Deformation, [0.0;12], [5.0,5.0], 0.0, 0.0,
            WarpMode::None, 4, ScaleType::Log { min: 0.01 },
        ).unwrap();
        assert!((s.scale_mult_at(0)).abs() < 1e-12);
        assert!((s.scale_mult_at(1) - 0.01).abs() < 1e-10);
        assert!((s.scale_mult_at(3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_deform_mult_noise_progression() {
        let s = Speckle::new(
            "/tmp".into(), "s".into(), ".jpg".into(), (10,10), 1.0, 1,
            false, Progression::Noise, [0.0;12], [5.0,5.0], 0.0, 0.0,
            WarpMode::None, 5, ScaleType::Lin,
        ).unwrap();
        assert!((s.deform_mult_at(0)).abs() < 1e-12);
        assert!((s.deform_mult_at(1) - 1.0).abs() < 1e-12);
        assert!((s.deform_mult_at(4) - 1.0).abs() < 1e-12);
        // scale_mult_at progresses normally for noise axis
        assert!((s.scale_mult_at(0)).abs() < 1e-12);
        assert!((s.scale_mult_at(4) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_render_image_nonzero() {
        let speckles = ndarray::arr2(&[[100.0, 100.0], [300.0, 200.0]]);
        let img = render_image(speckles.view(), (401, 301), 8.0, false, 0.0, 0.0, 0).unwrap();
        assert_eq!(img.shape(), &[301, 401]);
        assert!(img.iter().any(|&p| p > 0), "image should have non-zero pixels");
    }

    #[test]
    fn test_render_half_window() {
        assert_eq!(render_half_window(10.0, false), 100);
        assert_eq!(render_half_window(10.0, true), 15);
        assert_eq!(render_half_window(8.0, true), 12);
        assert_eq!(render_half_window(7.0, true), 11); // ceil(10.5) = 11
        assert_eq!(render_half_window(80.0, true), 120); // can exceed the unlimited default
    }

    #[test]
    fn test_render_image_speckle_limit_matches_unlimited() {
        // At the speckle sizes this project uses (d=10), a pixel's
        // contribution is already below the u8 rounding threshold well
        // inside 1.5x speckle_size, so speckle_limit must not change the
        // rendered image at all.
        let speckles = ndarray::arr2(&[[100.3, 100.7], [250.0, 180.0], [5.0, 5.0]]);
        let img_unlimited = render_image(speckles.view(), (401, 301), 10.0, false, 0.0, 0.0, 0).unwrap();
        let img_limited = render_image(speckles.view(), (401, 301), 10.0, true, 0.0, 0.0, 0).unwrap();
        assert_eq!(img_unlimited, img_limited);
    }

    #[test]
    fn test_warp_zero_comp() {
        let speckle = make_deformation_speckle([0.0; 12], 5);
        let coords = ndarray::arr2(&[[10.0, 20.0], [30.0, 40.0]]);
        let w = speckle.warp(2, coords.view()).unwrap();
        assert_eq!(w.shape(), &[2, 12]);
        for v in w.iter() { assert!(v.abs() < 1e-12); }
    }

    #[test]
    fn test_warp_unit_translation() {
        // comp[0]=1.0, image_no=3, ScaleType::Lin → scale_mult_at(2) = 1.0
        let mut comp = [0.0f64; 12];
        comp[0] = 1.0;
        let speckle = make_deformation_speckle(comp, 3);
        let coords = ndarray::arr2(&[[10.0, 20.0]]);
        let w = speckle.warp(2, coords.view()).unwrap();
        assert!((w[[0, 0]] - 1.0).abs() < 1e-12, "u={}", w[[0, 0]]);
        for k in 2..12usize { assert!(w[[0, k]].abs() < 1e-12); }
    }

    #[test]
    fn test_warp_gradient_recovery() {
        let mut comp = [0.0f64; 12];
        comp[2] = 0.1;
        let speckle = make_deformation_speckle(comp, 3);
        let coords = ndarray::arr2(&[[60.0, 70.0]]);
        let w = speckle.warp(2, coords.view()).unwrap();
        assert!((w[[0, 2]] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_warp_shearband_zero_on_centreline() {
        let mut comp = [0.0f64; 12];
        comp[4] = 0.1;
        let speckle = Speckle::new(
            "/tmp".into(), "s".into(), ".jpg".into(),
            (100, 100), 1.0, 10,
            false,
            Progression::Deformation,
            comp, [50.0, 50.0], 0.0, 0.0,
            WarpMode::ShearBand { option: ShearBandOption::Sin, width: 100.0 },
            3, ScaleType::Lin,
        ).unwrap();
        let coords = ndarray::arr2(&[[50.0, 50.0]]);
        let w = speckle.warp(2, coords.view()).unwrap();
        assert!(w[[0, 0]].abs() < 1e-12, "u at centreline should be 0");
    }

    // ShearBandOption::Smooth -----------------------------------------------

    #[test]
    fn test_speckle_new_rejects_nonpositive_tau() {
        let mut comp = [0.0f64; 12];
        comp[4] = 0.1;
        let err = Speckle::new(
            "/tmp".into(), "s".into(), ".jpg".into(),
            (100, 100), 1.0, 10,
            false,
            Progression::Deformation,
            comp, [50.0, 50.0], 0.0, 0.0,
            WarpMode::ShearBand { option: ShearBandOption::Smooth { tau: 0.0 }, width: 100.0 },
            3, ScaleType::Lin,
        );
        assert!(err.is_err());
    }

    #[test]
    fn test_compute_displaced_sb_smooth_at_origin() {
        let ref_pos = ndarray::arr2(&[[100.0, 500.0]]);
        let mut pm_i = [0.0f64; 12];
        pm_i[4] = 0.1;
        let out = compute_displaced(
            &WarpMode::ShearBand {
                option: ShearBandOption::Smooth { tau: 5.0 },
                width: 100.0,
            },
            ref_pos.view(),
            &pm_i,
            [500.0, 500.0],
        );
        assert!((out[[0, 0]] - 100.0).abs() < 1e-12, "u at band centre should be 0");
        assert!((out[[0, 1]] - 500.0).abs() < 1e-12);
    }

    #[test]
    fn test_warp_shearband_smooth_zero_on_centreline() {
        let mut comp = [0.0f64; 12];
        comp[4] = 0.1;
        let speckle = Speckle::new(
            "/tmp".into(), "s".into(), ".jpg".into(),
            (100, 100), 1.0, 10,
            false,
            Progression::Deformation,
            comp, [50.0, 50.0], 0.0, 0.0,
            WarpMode::ShearBand { option: ShearBandOption::Smooth { tau: 5.0 }, width: 100.0 },
            3, ScaleType::Lin,
        ).unwrap();
        let coords = ndarray::arr2(&[[50.0, 50.0]]);
        let w = speckle.warp(2, coords.view()).unwrap();
        assert!(w[[0, 0]].abs() < 1e-12, "u at centreline should be 0");
    }

    #[test]
    fn test_speckle_new_rejects_tau_exceeding_half_width() {
        let mut comp = [0.0f64; 12];
        comp[4] = 0.1;
        let err = Speckle::new(
            "/tmp".into(), "s".into(), ".jpg".into(),
            (100, 100), 1.0, 10,
            false,
            Progression::Deformation,
            comp, [50.0, 50.0], 0.0, 0.0,
            // width=100 -> b=50; tau=60 > b is rejected (ramps would cross)
            WarpMode::ShearBand { option: ShearBandOption::Smooth { tau: 60.0 }, width: 100.0 },
            3, ScaleType::Lin,
        );
        assert!(err.is_err());
    }

    #[test]
    fn test_smooth_step_matches_worked_example() {
        // Directly encodes the intended du/dy shape: flat at `strain` for
        // |dy| < b-tau, linear ramp to zero over b-tau <= |dy| < b, zero
        // beyond. width=20 (b=10), strain=0.05.
        let strain = 0.05;
        let b = 10.0;

        // tau=5: 0 up to dy=-10, ramp to strain at dy=-5, flat until
        // dy=5, ramp back to 0 at dy=10.
        let tau = 5.0;
        for &(dy, expected) in &[
            (-10.0, 0.0), (-7.5, 0.025), (-5.0, 0.05),
            (0.0, 0.05),
            (5.0, 0.05), (7.5, 0.025), (10.0, 0.0),
        ] {
            let (du_dy, _) = smooth_step_derivatives(dy, strain, b, tau);
            assert!(
                (du_dy - expected).abs() < 1e-12,
                "tau=5, dy={dy}: du_dy={du_dy}, expected={expected}"
            );
        }

        // tau=10 (== b): no flat plateau at all -- pure triangular/sawtooth
        // profile, linear from 0 at dy=-10 up to strain at dy=0, back down
        // to 0 at dy=10.
        let tau = 10.0;
        for &(dy, expected) in &[
            (-10.0, 0.0), (-5.0, 0.025), (0.0, 0.05), (5.0, 0.025), (10.0, 0.0),
        ] {
            let (du_dy, _) = smooth_step_derivatives(dy, strain, b, tau);
            assert!(
                (du_dy - expected).abs() < 1e-12,
                "tau=10, dy={dy}: du_dy={du_dy}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_smooth_step_recovers_lin_as_tau_shrinks() {
        let strain = 0.1;
        let b = 50.0;
        let tau = 0.1;

        // Strictly inside the plateau (|dy| < b-tau), u matches Lin's
        // strain*dy exactly for any tau -- both formulas agree there by
        // construction, no tau-dependent error at all.
        for &dy in &[-40.0_f64, -20.0, 0.0, 20.0, 40.0] {
            let u_smooth = smooth_step_u(dy, strain, b, tau);
            let u_lin = dy * strain;
            assert!(
                (u_smooth - u_lin).abs() < 1e-10,
                "dy={dy}: smooth={u_smooth}, lin={u_lin}"
            );
        }

        // Outside the band (|dy| >= b), Smooth saturates at
        // strain*(b - tau/2) rather than Lin's strain*b -- a fixed O(tau)
        // offset (not shrinking with distance from the corner, since the
        // profile is exactly flat out there), so the tolerance must scale
        // with tau rather than being a fixed small number.
        for &dy in &[-80.0_f64, -60.0, 60.0, 80.0] {
            let u_smooth = smooth_step_u(dy, strain, b, tau);
            let u_lin = dy.clamp(-b, b) * strain;
            assert!(
                (u_smooth - u_lin).abs() < strain * tau,
                "dy={dy}: smooth={u_smooth}, lin={u_lin}"
            );
        }
    }

    #[test]
    fn test_smooth_step_symmetry() {
        // u is odd about dy=0, du_dy is even -- both follow directly from
        // the piecewise construction using a=|dy| (see smooth_step_u/
        // smooth_step_derivatives doc comments).
        let strain = 0.15;
        let b = 30.0;
        let tau = 8.0;
        for &dy in &[3.0_f64, 15.0, 26.0, 40.0, 70.0] {
            let u_pos = smooth_step_u(dy, strain, b, tau);
            let u_neg = smooth_step_u(-dy, strain, b, tau);
            assert!((u_pos + u_neg).abs() < 1e-10, "dy={dy}: u(dy)+u(-dy)={}", u_pos + u_neg);

            let (dudy_pos, _) = smooth_step_derivatives(dy, strain, b, tau);
            let (dudy_neg, _) = smooth_step_derivatives(-dy, strain, b, tau);
            assert!((dudy_pos - dudy_neg).abs() < 1e-10);
        }
    }

    #[test]
    fn test_smooth_step_derivatives_match_finite_difference() {
        // b=25, tau=6 -> plateau |dy|<19, ramp 19<=|dy|<25, outside |dy|>=25.
        // Points deliberately avoid landing exactly on a breakpoint (+-19,
        // +-25): du/dy is only piecewise-linear, so d2u/dy2 has a genuine
        // kink there that a central finite difference straddles and can't
        // resolve -- not a bug, see test_smooth_step_matches_worked_example
        // for exact (non-FD) checks that include the breakpoints themselves.
        let strain = 0.12;
        let b = 25.0;
        let tau = 6.0;
        let h = 1e-4;
        // -40, 45: outside; -22, 22: ramp; -5, 0, 12: plateau.
        for &dy in &[-40.0_f64, -22.0, -5.0, 0.0, 12.0, 22.0, 45.0] {
            let (du_dy, d2u_dy2) = smooth_step_derivatives(dy, strain, b, tau);

            let u_plus = smooth_step_u(dy + h, strain, b, tau);
            let u_minus = smooth_step_u(dy - h, strain, b, tau);
            let du_dy_fd = (u_plus - u_minus) / (2.0 * h);
            assert!(
                (du_dy - du_dy_fd).abs() < 1e-6,
                "dy={dy}: du_dy={du_dy}, fd={du_dy_fd}"
            );

            // du/dy is exactly linear within any single region (plateau is
            // constant, ramp is linear, outside is constant), so u is
            // exactly quadratic-or-linear there and a central second
            // difference has no truncation error beyond floating-point
            // noise -- tolerance can be tight.
            let u_mid = smooth_step_u(dy, strain, b, tau);
            let d2u_dy2_fd = (u_plus - 2.0 * u_mid + u_minus) / (h * h);
            assert!(
                (d2u_dy2 - d2u_dy2_fd).abs() < 1e-3,
                "dy={dy}: d2u_dy2={d2u_dy2}, fd={d2u_dy2_fd}"
            );
        }
    }

    // WarpMode::CircularShear ------------------------------------------------

    #[test]
    fn test_circular_shear_rejects_bad_radii() {
        let mut comp = [0.0f64; 12];
        comp[0] = 1.0;
        let base = |mode: WarpMode| {
            Speckle::new(
                "/tmp".into(), "s".into(), ".jpg".into(),
                (100, 100), 1.0, 10,
            false,
                Progression::Deformation,
                comp, [50.0, 50.0], 0.0, 0.0,
                mode, 3, ScaleType::Lin,
            )
        };
        assert!(base(WarpMode::CircularShear { r1: 0.0, r2: 10.0, tau: None }).is_err());
        assert!(base(WarpMode::CircularShear { r1: -5.0, r2: 10.0, tau: None }).is_err());
        assert!(base(WarpMode::CircularShear { r1: 10.0, r2: 10.0, tau: None }).is_err());
        assert!(base(WarpMode::CircularShear { r1: 10.0, r2: 5.0, tau: None }).is_err());
        assert!(base(WarpMode::CircularShear { r1: 5.0, r2: 10.0, tau: None }).is_ok());
    }

    #[test]
    fn test_circular_shear_zero_at_centre() {
        let ref_pos = ndarray::arr2(&[[501.0, 501.0]]);
        let mut pm_i = [0.0f64; 12];
        pm_i[0] = 1.2345; // any nonzero core rotation angle
        let out = compute_displaced(
            &WarpMode::CircularShear { r1: 100.0, r2: 150.0, tau: None },
            ref_pos.view(),
            &pm_i,
            [501.0, 501.0],
        );
        // The centre point itself has dx=dy=0, so u=v=0 regardless of theta.
        assert!((out[[0, 0]] - 501.0).abs() < 1e-12);
        assert!((out[[0, 1]] - 501.0).abs() < 1e-12);
    }

    #[test]
    fn test_circular_shear_rigid_core() {
        // Two points inside r1 should keep their exact relative geometry
        // after warp -- a rigid rotation preserves distances and angles.
        let centre = [501.0, 501.0];
        let ref_pos = ndarray::arr2(&[[550.0, 501.0], [501.0, 560.0]]);
        let mut pm_i = [0.0f64; 12];
        pm_i[0] = 0.4;
        let out = compute_displaced(
            &WarpMode::CircularShear { r1: 100.0, r2: 150.0, tau: None },
            ref_pos.view(),
            &pm_i,
            centre,
        );
        let d_ref = ((ref_pos[[0, 0]] - ref_pos[[1, 0]]).powi(2)
            + (ref_pos[[0, 1]] - ref_pos[[1, 1]]).powi(2))
        .sqrt();
        let d_out = ((out[[0, 0]] - out[[1, 0]]).powi(2) + (out[[0, 1]] - out[[1, 1]]).powi(2))
            .sqrt();
        assert!((d_ref - d_out).abs() < 1e-9, "rigid rotation must preserve distance");

        // And each point individually should match a direct rotation matrix
        // application about the centre.
        let (s, c) = pm_i[0].sin_cos();
        for i in 0..2 {
            let dx = ref_pos[[i, 0]] - centre[0];
            let dy = ref_pos[[i, 1]] - centre[1];
            let expected_x = centre[0] + c * dx - s * dy;
            let expected_y = centre[1] + s * dx + c * dy;
            assert!((out[[i, 0]] - expected_x).abs() < 1e-9);
            assert!((out[[i, 1]] - expected_y).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circular_shear_far_field_stationary() {
        let ref_pos = ndarray::arr2(&[[501.0 + 151.0, 501.0], [501.0, 501.0 - 500.0]]);
        let mut pm_i = [0.0f64; 12];
        pm_i[0] = 2.5;
        let out = compute_displaced(
            &WarpMode::CircularShear { r1: 100.0, r2: 150.0, tau: None },
            ref_pos.view(),
            &pm_i,
            [501.0, 501.0],
        );
        for i in 0..2 {
            assert!((out[[i, 0]] - ref_pos[[i, 0]]).abs() < 1e-12);
            assert!((out[[i, 1]] - ref_pos[[i, 1]]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_circular_shear_continuity_at_r1_and_r2() {
        // m(r1) = 1 (matches the core's full-angle rotation); m(r2) = 0
        // (matches the stationary far field). Approach each boundary from
        // both sides and check the displacement matches to within the
        // step size used.
        let centre = [501.0, 501.0];
        let (r1, r2) = (100.0, 150.0);
        let mut pm_i = [0.0f64; 12];
        pm_i[0] = 0.7;
        let eps = 1e-6;

        for &(boundary, label) in &[(r1, "r1"), (r2, "r2")] {
            let just_inside = ndarray::arr2(&[[centre[0] + boundary - eps, centre[1]]]);
            let just_outside = ndarray::arr2(&[[centre[0] + boundary + eps, centre[1]]]);
            let w_in = compute_displaced(
                &WarpMode::CircularShear { r1, r2, tau: None }, just_inside.view(), &pm_i, centre,
            );
            let w_out = compute_displaced(
                &WarpMode::CircularShear { r1, r2, tau: None }, just_outside.view(), &pm_i, centre,
            );
            let du = (w_in[[0, 0]] - w_out[[0, 0]]).abs();
            let dv = (w_in[[0, 1]] - w_out[[0, 1]]).abs();
            assert!(du < 1e-3 && dv < 1e-3, "{label}: discontinuous jump du={du} dv={dv}");
        }
    }

    /// Test-only literal port of the old Python package's brute-force
    /// (1000-substep) numerical integration for the annulus, used solely to
    /// cross-check the closed form above. See
    /// mds/circular_shear_band_speckle.md for why these are mathematically
    /// equivalent (SO(2) composes additively; m(r) is invariant under pure
    /// rotation since it depends only on radius).
    fn circular_shear_brute_force(
        ref_pos: ArrayView2<f64>,
        centre: [f64; 2],
        r1: f64,
        r2: f64,
        theta_total: f64,
        n_sub: usize,
    ) -> Array2<f64> {
        let n = ref_pos.nrows();
        let mut warp = Array2::<f64>::zeros((n, 2));
        let mut delta = Array2::<f64>::zeros((n, 2));
        let mut dist0 = vec![0.0f64; n];
        let mut is_core = vec![false; n];
        let mut is_annulus = vec![false; n];
        for i in 0..n {
            let dx = ref_pos[[i, 0]] - centre[0];
            let dy = ref_pos[[i, 1]] - centre[1];
            delta[[i, 0]] = dx;
            delta[[i, 1]] = dy;
            let d = (dx * dx + dy * dy).sqrt();
            dist0[i] = d;
            is_core[i] = d < r1;
            is_annulus[i] = !is_core[i] && d < r2;
        }
        // Core: exact full-angle rotation (matches the old code's direct
        // pm[2:6]-based application, itself already exact for any angle).
        let (s, c) = theta_total.sin_cos();
        for i in 0..n {
            if is_core[i] {
                warp[[i, 0]] = (c - 1.0) * delta[[i, 0]] + (-s) * delta[[i, 1]];
                warp[[i, 1]] = s * delta[[i, 0]] + (c - 1.0) * delta[[i, 1]];
            }
        }
        // Annulus: n_sub tiny rotation increments, re-deriving m from the
        // (updated) current position before each increment -- as literally
        // written in geopyv/src/geopyv/speckle.py's mode "C" branch.
        let dtheta = theta_total / n_sub as f64;
        for _ in 0..n_sub {
            for i in 0..n {
                if !is_annulus[i] {
                    continue;
                }
                let m = (r2 - dist0[i]) / (r2 - r1);
                let (sm, cm) = (dtheta * m).sin_cos();
                let dx = delta[[i, 0]];
                let dy = delta[[i, 1]];
                let du = cm * dx - sm * dy - dx;
                let dv = sm * dx + cm * dy - dy;
                warp[[i, 0]] += du;
                warp[[i, 1]] += dv;
                let new_dx = dx + du;
                let new_dy = dy + dv;
                delta[[i, 0]] = new_dx;
                delta[[i, 1]] = new_dy;
                dist0[i] = (new_dx * new_dx + new_dy * new_dy).sqrt();
            }
        }
        warp
    }

    #[test]
    fn test_circular_shear_matches_brute_force_integration() {
        let centre = [501.0, 501.0];
        let (r1, r2) = (300.0, 350.0);
        let theta_total = 2.0 * std::f64::consts::PI * 0.37;
        let ref_pos = ndarray::arr2(&[
            [501.0, 501.0],
            [501.0 + 150.0, 501.0],
            [501.0 + 299.0, 501.0],
            [501.0 + 310.0, 501.0],
            [501.0 + 325.0, 501.0],
            [501.0 + 349.0, 501.0],
            [501.0 + 400.0, 501.0],
            [501.0 - 320.0, 501.0 + 15.0],
        ]);
        let mut pm_i = [0.0f64; 12];
        pm_i[0] = theta_total;
        let closed = compute_displaced(
            &WarpMode::CircularShear { r1, r2, tau: None }, ref_pos.view(), &pm_i, centre,
        );
        let brute = circular_shear_brute_force(ref_pos.view(), centre, r1, r2, theta_total, 1000);
        for i in 0..ref_pos.nrows() {
            let du = closed[[i, 0]] - (ref_pos[[i, 0]] + brute[[i, 0]]);
            let dv = closed[[i, 1]] - (ref_pos[[i, 1]] + brute[[i, 1]]);
            assert!(
                du.abs() < 1e-6 && dv.abs() < 1e-6,
                "point {i}: closed-form vs brute-force mismatch du={du} dv={dv}"
            );
        }
    }

    #[test]
    fn test_circular_shear_jacobian_matches_finite_difference() {
        let mut comp = [0.0f64; 12];
        comp[0] = 2.0 * std::f64::consts::PI * 0.37;
        let speckle = Speckle::new(
            "/tmp".into(), "s".into(), ".jpg".into(),
            (1001, 1001), 1.0, 10,
            false,
            Progression::Deformation,
            comp, [501.0, 501.0], 0.0, 0.0,
            WarpMode::CircularShear { r1: 300.0, r2: 350.0, tau: None },
            3, ScaleType::Lin,
        ).unwrap();

        let h = 1e-5;
        let points = [
            (501.0 + 150.0, 501.0),
            (501.0 + 310.0, 501.0),
            (501.0 + 325.0, 501.0),
            (501.0 + 325.0, 501.0 + 40.0),
            (501.0 - 320.0, 501.0 + 15.0),
        ];
        for &(x, y) in &points {
            let centre = ndarray::arr2(&[[x, y]]);
            let w = speckle.warp(2, centre.view()).unwrap();

            let xp = ndarray::arr2(&[[x + h, y]]);
            let xm = ndarray::arr2(&[[x - h, y]]);
            let yp = ndarray::arr2(&[[x, y + h]]);
            let ym = ndarray::arr2(&[[x, y - h]]);
            let wxp = speckle.warp(2, xp.view()).unwrap();
            let wxm = speckle.warp(2, xm.view()).unwrap();
            let wyp = speckle.warp(2, yp.view()).unwrap();
            let wym = speckle.warp(2, ym.view()).unwrap();

            let dudx_fd = (wxp[[0, 0]] - wxm[[0, 0]]) / (2.0 * h);
            let dvdx_fd = (wxp[[0, 1]] - wxm[[0, 1]]) / (2.0 * h);
            let dudy_fd = (wyp[[0, 0]] - wym[[0, 0]]) / (2.0 * h);
            let dvdy_fd = (wyp[[0, 1]] - wym[[0, 1]]) / (2.0 * h);

            assert!((w[[0, 2]] - dudx_fd).abs() < 1e-5, "dudx at ({x},{y}): {} vs fd {}", w[[0,2]], dudx_fd);
            assert!((w[[0, 3]] - dvdx_fd).abs() < 1e-5, "dvdx at ({x},{y}): {} vs fd {}", w[[0,3]], dvdx_fd);
            assert!((w[[0, 4]] - dudy_fd).abs() < 1e-5, "dudy at ({x},{y}): {} vs fd {}", w[[0,4]], dudy_fd);
            assert!((w[[0, 5]] - dvdy_fd).abs() < 1e-5, "dvdy at ({x},{y}): {} vs fd {}", w[[0,5]], dvdy_fd);
        }
    }

    // CircularShear tau (corner sharpness) -----------------------------------

    #[test]
    fn test_circular_shear_rejects_bad_tau() {
        let mut comp = [0.0f64; 12];
        comp[0] = 1.0;
        let base = |tau: f64| {
            Speckle::new(
                "/tmp".into(), "s".into(), ".jpg".into(),
                (100, 100), 1.0, 10,
                false,
                Progression::Deformation,
                comp, [50.0, 50.0], 0.0, 0.0,
                WarpMode::CircularShear { r1: 100.0, r2: 150.0, tau: Some(tau) },
                3, ScaleType::Lin,
            )
        };
        assert!(base(0.0).is_err());
        assert!(base(-1.0).is_err());
        assert!(base(25.0).is_ok()); // (r2-r1)/2 = 25, boundary allowed
        assert!(base(25.001).is_err()); // just over the bound
    }

    #[test]
    fn test_circular_shear_smooth_boundary_conditions_preserved() {
        // phi(r1)=theta exactly (core boundary unaffected by smoothing) and
        // phi(r2)=0 exactly (far-field boundary unaffected) -- unlike
        // ShearBandOption::Smooth's drifting outer saturation, these are
        // externally fixed physical parameters here and must not move; see
        // WarpMode::CircularShear's tau doc comment.
        let theta = 0.6;
        let (r1, r2, tau) = (100.0, 150.0, 15.0);
        let (phi_at_r1, _) = circular_shear_annulus(r1, theta, r1, r2, tau);
        assert!((phi_at_r1 - theta).abs() < 1e-12, "phi(r1)={phi_at_r1}, expected {theta}");
        let (phi_near_r2, _) = circular_shear_annulus(r2 - 1e-9, theta, r1, r2, tau);
        assert!(phi_near_r2.abs() < 1e-6, "phi near r2 = {phi_near_r2}, expected ~0");
    }

    #[test]
    fn test_circular_shear_smooth_continuity_at_breakpoints() {
        let theta = 0.6;
        let (r1, r2, tau) = (100.0, 150.0, 15.0);
        let eps = 1e-6;
        for &r_mid in &[r1 + tau, r2 - tau] {
            let (phi_minus, _) = circular_shear_annulus(r_mid - eps, theta, r1, r2, tau);
            let (phi_plus, _) = circular_shear_annulus(r_mid + eps, theta, r1, r2, tau);
            assert!(
                (phi_minus - phi_plus).abs() < 1e-5,
                "discontinuity at r={r_mid}: {phi_minus} vs {phi_plus}"
            );
        }
    }

    #[test]
    fn test_circular_shear_smooth_recovers_hard_edge_away_from_corners() {
        let theta = 0.5;
        let (r1, r2) = (100.0, 150.0);
        let tau = 0.01;
        for &r in &[110.0_f64, 125.0, 140.0] {
            let (phi_smooth, k_smooth) = circular_shear_annulus(r, theta, r1, r2, tau);
            let phi_hard = theta * (r2 - r) / (r2 - r1);
            let k_hard = -theta / (r2 - r1);
            assert!((phi_smooth - phi_hard).abs() < 1e-3, "r={r}: smooth={phi_smooth} hard={phi_hard}");
            assert!((k_smooth - k_hard).abs() < 1e-2, "r={r}: k_smooth={k_smooth} hard={k_hard}");
        }
    }

    #[test]
    fn test_circular_shear_smooth_jacobian_matches_finite_difference() {
        let mut comp = [0.0f64; 12];
        comp[0] = 2.0 * std::f64::consts::PI * 0.37;
        let (r1, r2, tau) = (300.0, 350.0, 10.0);
        let speckle = Speckle::new(
            "/tmp".into(), "s".into(), ".jpg".into(),
            (1001, 1001), 1.0, 10,
            false,
            Progression::Deformation,
            comp, [501.0, 501.0], 0.0, 0.0,
            WarpMode::CircularShear { r1, r2, tau: Some(tau) },
            3, ScaleType::Lin,
        ).unwrap();

        let h = 1e-5;
        let points = [
            (501.0 + 150.0, 501.0),        // core, r < r1
            (501.0 + 305.0, 501.0),        // up-ramp, r1 < r < r1+tau=310
            (501.0 + 325.0, 501.0),        // flat plateau
            (501.0 + 345.0, 501.0),        // down-ramp, r2-tau=340 < r < r2
            (501.0 + 325.0, 501.0 + 40.0), // flat plateau, off-axis
            (501.0 - 320.0, 501.0 + 15.0), // flat plateau, off-axis negative side
        ];
        for &(x, y) in &points {
            let centre = ndarray::arr2(&[[x, y]]);
            let w = speckle.warp(2, centre.view()).unwrap();

            let xp = ndarray::arr2(&[[x + h, y]]);
            let xm = ndarray::arr2(&[[x - h, y]]);
            let yp = ndarray::arr2(&[[x, y + h]]);
            let ym = ndarray::arr2(&[[x, y - h]]);
            let wxp = speckle.warp(2, xp.view()).unwrap();
            let wxm = speckle.warp(2, xm.view()).unwrap();
            let wyp = speckle.warp(2, yp.view()).unwrap();
            let wym = speckle.warp(2, ym.view()).unwrap();

            let dudx_fd = (wxp[[0, 0]] - wxm[[0, 0]]) / (2.0 * h);
            let dvdx_fd = (wxp[[0, 1]] - wxm[[0, 1]]) / (2.0 * h);
            let dudy_fd = (wyp[[0, 0]] - wym[[0, 0]]) / (2.0 * h);
            let dvdy_fd = (wyp[[0, 1]] - wym[[0, 1]]) / (2.0 * h);

            assert!((w[[0, 2]] - dudx_fd).abs() < 1e-5, "dudx at ({x},{y}): {} vs fd {}", w[[0,2]], dudx_fd);
            assert!((w[[0, 3]] - dvdx_fd).abs() < 1e-5, "dvdx at ({x},{y}): {} vs fd {}", w[[0,3]], dvdx_fd);
            assert!((w[[0, 4]] - dudy_fd).abs() < 1e-5, "dudy at ({x},{y}): {} vs fd {}", w[[0,4]], dudy_fd);
            assert!((w[[0, 5]] - dvdy_fd).abs() < 1e-5, "dvdy at ({x},{y}): {} vs fd {}", w[[0,5]], dvdy_fd);
        }
    }
}
