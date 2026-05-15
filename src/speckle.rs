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
        Ok(Self {
            image_dir,
            name,
            file_format,
            image_size,
            speckle_size,
            speckle_number,
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
                };
                out[[i, 0]] = ref_speckles[[i, 0]] + u;
                out[[i, 1]] = ref_speckles[[i, 1]];
            }
        }
    }
    out
}

/// Render displaced speckle positions as Gaussian blobs into a (H, W) u8 image.
/// Parallelised over speckles with rayon.
fn render_image(
    speckles: ArrayView2<f64>,
    image_size: (usize, usize),
    speckle_size: f64,
    noise_pos: f64,
    noise_int: f64,
    seed: u64,
) -> Result<Array2<u8>, Error> {
    let (w, h) = image_size;
    let sigma_sq_4 = speckle_size * speckle_size / 4.0;
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
                let x_lo_s = cx as isize - 100;
                let x_hi_s = cx as isize + 101;
                let y_lo_s = cy as isize - 100;
                let y_hi_s = cy as isize + 101;
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
                    };
                    out[[p, 0]] = u;
                    out[[p, 4]] = du_dy;
                    out[[p, 10]] = d2u_dy2;
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
            Progression::Deformation, [0.0;12], [5.0,5.0], 0.0, 0.0,
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
            Progression::Deformation, [0.0;12], [5.0,5.0], 0.0, 0.0,
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
            Progression::Noise, [0.0;12], [5.0,5.0], 0.0, 0.0,
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
        let img = render_image(speckles.view(), (401, 301), 8.0, 0.0, 0.0, 0).unwrap();
        assert_eq!(img.shape(), &[301, 401]);
        assert!(img.iter().any(|&p| p > 0), "image should have non-zero pixels");
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
            Progression::Deformation,
            comp, [50.0, 50.0], 0.0, 0.0,
            WarpMode::ShearBand { option: ShearBandOption::Sin, width: 100.0 },
            3, ScaleType::Lin,
        ).unwrap();
        let coords = ndarray::arr2(&[[50.0, 50.0]]);
        let w = speckle.warp(2, coords.view()).unwrap();
        assert!(w[[0, 0]].abs() < 1e-12, "u at centreline should be 0");
    }
}
