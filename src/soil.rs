//! Soil images for [`crate::speckle::Speckle`]: sand grains in a clay matrix,
//! or pure sand (mds/speckle_soils_plan.md).
//!
//! Specified by what can be measured from photographs of the bulk material:
//! the average brightness and spread of the **matrix** and of the **seeding**
//! (dyed grains), and the **artificial seeding ratio** `asr`: the share of the
//! image that shows dye (0 none, 1 all), plus the grains' characterisation
//! (size curve, aspect ratio, outline statistics) and their size in pixels `d_p`.
//!
//! - **Grains** are rigid and opaque, described as a grain characterisation
//!   (e.g. dynamic image analysis) gives them. Size: EQPC, the diameter of a
//!   circle with the grain's projected area, from a cumulative size curve
//!   `psd` (% by mass passing against size; empty: every grain the same size)
//!   scaled so that its median `x50` is `d_p` pixels. On the imaged surface a
//!   size's share of the area is its share of the volume (Delesse), so the
//!   curve sets how much of the image each size covers. Grains finer than
//!   [`MIN_GRAIN_PX`] (one pixel) cannot be drawn: deeper layers (sand) or the
//!   matrix (clay) stand in for them. Shape: an aspect ratio (narrowest /
//!   widest width, Feret) drawn per grain from a normal spread (mean,
//!   standard deviation), independently of size, and an outline drawn from
//!   [`OutlineStats`] (measured on real grains: Leighton Buzzard sand by
//!   default). Outline: an ellipse (elongation, axis ratio set so the outline
//!   has the drawn aspect ratio) times radius waves 1 + sum a_n cos(n t +
//!   ph_n), n = 3..[`WAVES`], drawn from the statistics with random phases,
//!   redrawn while its deepest dent is deeper than real grains' dents.
//!   Sphericity is not an input: a QicPic sphericity is set mostly by fine
//!   roughness and pixel steps (a traced perfect circle scores 0.948).
//!   Round grains (aspect ratio 1, no spread, no outline statistics given)
//!   are circles.
//!   Grains are placed at random without crossing (touching is allowed); each
//!   moves with the material at its centre and turns with it. Each grain is
//!   one brightness drawn from a bell-shaped spread, with pixels on its edge
//!   partly covered. Inside its outline the brightness varies (defaults
//!   calibrated so that real Leighton Buzzard grains' measurements come out
//!   after blur 0.8 px and noise 2 grey levels; 0 turns each off): a tilt (brighter towards one side:
//!   random direction, change over one equal-area radius drawn from a
//!   Rayleigh spread of median `grain_tilt`), a dome (brighter middle, or
//!   darker for a negative draw: change from edge to centre drawn from a
//!   normal spread of width `grain_dome`), a dark rim (`grain_rim` darker at
//!   the edge, half that at 92 % of the way out) and a patchy texture (a
//!   smooth random field of relative strength `grain_texture` and feature size
//!   `texture_size` grain diameters). All are fixed to the grain: they move
//!   and turn with it, and keep the grain's mean brightness.
//! - **Clay**: the matrix is the clay; its brightness carries a smooth random
//!   pattern (relative strength `matrix_spread`, feature size `pattern_size`:
//!   the shift at which the pattern's self-similarity falls to one half),
//!   attached to the material: in a deformed image it is worked out exactly at
//!   each pixel's original position, so it stretches and shears with the
//!   motion. Dyed grains cover the share `asr` of the image.
//! - **Sand**: the matrix is the natural sand grains (brightness spread
//!   `matrix_spread` from grain to grain); each grain is dyed with chance
//!   `asr`. `depth_layers` layers of grains, each filled until no more fit;
//!   a pixel shows the nearest grain covering it (behind all layers, the
//!   matrix brightness). Every layer moves with the same in-plane motion
//!   (plane strain).
//! - The whole image is then blurred by a Gaussian of width `blur` px (the
//!   lens), noise is added, and values are rounded to 8 bits.

use ndarray::{Array2, ArrayView2};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::Error;

/// Which soil an image shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoilKind {
    /// Sand grains on a clay matrix with its own deforming pattern.
    Clay,
    /// Packed sand: layers of grains, the nearest seen.
    Sand,
}

/// Soil image settings — see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoilConfig {
    pub kind: SoilKind,
    /// Average brightness of the matrix (clay: the clay; sand: the natural
    /// grains at the surface), grey levels 0-255.
    pub matrix_brightness: f64,
    /// Clay: strength of the clay's pattern; sand: grain-to-grain spread of
    /// the natural grains. A fraction of `matrix_brightness`.
    pub matrix_spread: f64,
    /// Average brightness of the dyed grains (seeding) at the surface.
    pub seed_brightness: f64,
    /// Grain-to-grain spread of the dyed grains, a fraction of
    /// `seed_brightness`.
    pub seed_spread: f64,
    /// Artificial seeding ratio: share of the image showing dye (clay: share
    /// covered by dyed grains; sand: chance that a grain is dyed).
    pub asr: f64,
    /// Grain size curve: `[size, % passing]` (cumulative, by mass; any length
    /// unit), sizes increasing. Empty: every grain is `d_p` px.
    pub psd: Vec<[f64; 2]>,
    /// Median grain size (x50 of `psd`) in pixels.
    pub d_p: f64,
    /// Aspect ratio (narrowest / widest Feret width, <= 1): mean and
    /// standard deviation over the grains.
    pub aspect_ratio: f64,
    pub aspect_ratio_sd: f64,
    /// Outline statistics (e.g. from a QicPic particle report); `None`: the
    /// built-in Leighton Buzzard statistics, or circles when the aspect ratio
    /// is 1 with no spread.
    pub outlines: Option<OutlineStats>,
    /// Sand: number of grain layers.
    pub depth_layers: usize,
    /// Clay: feature size of the clay's pattern (px).
    pub pattern_size: f64,
    /// Lens blur (Gaussian width, px); 0 for none.
    pub blur: f64,
    /// Brightness inside each grain (see the module docs), fractions of the
    /// grain's brightness; 0 for none (defaults 0.14, 0.17, 0, 0.085). Tilt: median change over one radius;
    /// dome: spread of the change from edge to centre; rim: darkening at
    /// the edge; texture: spread of the patchy texture.
    pub grain_tilt: f64,
    pub grain_dome: f64,
    pub grain_rim: f64,
    pub grain_texture: f64,
    /// Feature size of the texture, in grain diameters (where its
    /// self-similarity falls to one half).
    pub texture_size: f64,
}

impl SoilConfig {
    /// Clay with sand grains, defaults from the measured real images
    /// (mds/speckle_soils_plan.md, section H).
    pub fn clay(matrix_brightness: f64, seed_brightness: f64, asr: f64) -> Self {
        Self {
            kind: SoilKind::Clay,
            matrix_brightness,
            matrix_spread: 0.05,
            seed_brightness,
            seed_spread: 0.09,
            asr,
            psd: Vec::new(),
            d_p: 6.0,
            aspect_ratio: 1.0,
            aspect_ratio_sd: 0.0,
            outlines: None,
            depth_layers: 1,
            pattern_size: 10.0,
            blur: 0.0,
            // brightness inside grains: calibrated on real Leighton Buzzard
            // grains (blur 0.8, noise 2; final_trial/grain_texture/check_speckle_texture.py)
            grain_tilt: 0.14,
            grain_dome: 0.17,
            grain_rim: 0.0,
            grain_texture: 0.085,
            texture_size: 0.10,
        }
    }

    /// Pure sand, defaults as [`Self::clay`] where they apply.
    pub fn sand(matrix_brightness: f64, seed_brightness: f64, asr: f64) -> Self {
        Self { kind: SoilKind::Sand, matrix_spread: 0.15, depth_layers: 6, ..Self::clay(matrix_brightness, seed_brightness, asr) }
    }

    /// Spread of a grain's area about its centre, per axis (px^2), averaged
    /// over the grains by area: the second moment of area of an ellipse of
    /// equal-area diameter d and aspect ratio a is d^2 (a + 1/a) / 32 per axis
    /// (d^2 / 16 for a circle). The aspect ratio is averaged over its spread.
    /// What the Gaussian rigid-texture correction calls the blob width^2, for
    /// grains of these properties (before any image blur).
    pub fn grain_sigma2(&self) -> Result<f64, Error> {
        let d2 = SizeDraw::new(self)?.area_mean_d2();
        let shape = |a: f64| 0.5 * (a + 1.0 / a);
        let factor = if self.aspect_ratio_sd > 0.0 {
            // average over a normal spread clipped to [0.2, 1], as the grains are drawn
            let (mut w, mut sum) = (0.0, 0.0);
            for k in -40..=40 {
                let z = k as f64 / 10.0;
                let p = (-0.5 * z * z).exp();
                let a = (self.aspect_ratio + self.aspect_ratio_sd * z).clamp(0.2, 1.0);
                w += p;
                sum += p * shape(a);
            }
            sum / w
        } else {
            shape(self.aspect_ratio)
        };
        Ok(d2 * factor / 16.0)
    }

    /// Mean distance (px) from a grain's centre to its outline's tangent,
    /// for the edge correction: perimeter / 2 pi for a convex grain (Cauchy),
    /// i.e. d / (2 sphericity), with d the area-weighted root-mean-square
    /// equal-area diameter and the sphericity the mean over the outlines the
    /// images would draw.
    pub fn grain_edge_distance(&self) -> Result<f64, Error> {
        let lib = shape_library(self, 0);
        let sph = if lib.is_empty() {
            1.0
        } else {
            lib.iter().map(|(s, _, _)| sphericity_of(&s.polygon())).sum::<f64>() / lib.len() as f64
        };
        Ok(SizeDraw::new(self)?.area_mean_d2().sqrt() / (2.0 * sph))
    }

    pub fn validate(&self) -> Result<(), Error> {
        let bad = |m: &str| Err(Error::InvalidInput(m.into()));
        if !(0.0..=255.0).contains(&self.matrix_brightness) || !(0.0..=255.0).contains(&self.seed_brightness) {
            return bad("matrix and seed brightness must be between 0 and 255");
        }
        if !(self.matrix_spread >= 0.0) || !(self.seed_spread >= 0.0) {
            return bad("matrix_spread and seed_spread must be >= 0");
        }
        if !(self.d_p > 0.0) {
            return bad("d_p must be > 0");
        }
        if !(self.aspect_ratio > 0.0 && self.aspect_ratio <= 1.0) {
            return bad("aspect_ratio must be in (0, 1]");
        }
        if !(self.aspect_ratio_sd >= 0.0) {
            return bad("aspect_ratio_sd must be >= 0");
        }
        if let Some(o) = &self.outlines {
            o.validate()?;
        }
        SizeDraw::new(self)?;
        if !(0.0..=1.0).contains(&self.asr) {
            return bad("asr must be between 0 and 1");
        }
        if self.kind == SoilKind::Clay && !(self.asr < 0.55) {
            return bad("clay asr must be below 0.55: grains placed at random without crossing cannot cover more");
        }
        if self.kind == SoilKind::Sand && self.depth_layers == 0 {
            return bad("sand needs depth_layers >= 1");
        }
        if !(self.pattern_size > 0.0) || !(self.blur >= 0.0) {
            return bad("pattern_size > 0 and blur >= 0 are required");
        }
        if !(self.grain_tilt >= 0.0 && self.grain_dome >= 0.0 && self.grain_rim >= 0.0 && self.grain_texture >= 0.0) {
            return bad("grain_tilt, grain_dome, grain_rim and grain_texture must be >= 0");
        }
        if !(self.texture_size > 0.0) {
            return bad("texture_size must be > 0");
        }
        Ok(())
    }
}

/// One grain in its own frame: outline and brightness.
#[derive(Debug, Clone)]
pub(crate) struct Grain {
    outline: Outline,
    pub brightness: f64,
    look: Look,
    pub dyed: bool,
    pub layer: usize,
    /// Largest distance of the outline from the centre (px).
    pub reach: f64,
    /// Outline area (px^2): pi (size / 2)^2 for every shape.
    pub area: f64,
}

/// A grain's brightness variation (see [`SoilConfig`]'s `grain_*`
/// settings): tilt vector (change over one radius), dome (change from edge to
/// centre), and where in the texture field the grain sits.
#[derive(Debug, Clone, Default)]
struct Look {
    tilt: [f64; 2],
    dome: f64,
    offset: [f64; 2],
}

/// Area means of rho and rho^RIM_POWER over a grain, rho = distance from the
/// centre / outline radius in that direction (exact for any star-shaped
/// outline): 2/3 and 2/(RIM_POWER + 2).
const RHO_MEAN: f64 = 2.0 / 3.0;
const RIM_POWER: i32 = 8;
const RIM_MEAN: f64 = 2.0 / (RIM_POWER as f64 + 2.0);

impl Grain {
    /// Brightness at grain-frame offset (x, y) from its centre.
    fn shade(&self, cfg: &SoilConfig, x: f64, y: f64, texture: Option<&Pattern>) -> f64 {
        if cfg.grain_tilt == 0.0 && cfg.grain_dome == 0.0 && cfg.grain_rim == 0.0 && texture.is_none() {
            return self.brightness;
        }
        let req = (self.area / std::f64::consts::PI).sqrt();
        let rho = (x.hypot(y) / self.outline.radius(y.atan2(x))).min(1.0);
        let mut f = 1.0 + (self.look.tilt[0] * x + self.look.tilt[1] * y) / req + self.look.dome * (RHO_MEAN - rho)
            - cfg.grain_rim * (rho.powi(RIM_POWER) - RIM_MEAN);
        if let Some(p) = texture {
            let d = 2.0 * req;
            f += cfg.grain_texture * p.at(x / d + self.look.offset[0], y / d + self.look.offset[1]);
        }
        self.brightness * f.max(0.0)
    }
}

#[derive(Debug, Clone)]
enum Outline {
    Circle { r: f64 },
    /// r(t) = r0 x (unit outline of `shape`)
    Shaped { r0: f64, shape: Shape },
}

/// Highest radius wave of a grain outline (finer waves are below a pixel at
/// camera scale).
pub const WAVES: usize = 24;
/// Irregularity waves 3..=WAVES.
const NW: usize = WAVES - 2;
/// Angles at which an outline's radius is tabulated.
const TABLE_N: usize = 1024;

/// Statistics of real grain outlines, as radius waves about the centre of
/// area: r = r0 E(t) (1 + sum a_n cos(n t + ph_n)), E the ellipse of the
/// grain's second moments of area (its elongation). Measured from single grains
/// (geopyv_dev/_qicpic.py: QicPic particle report; figure-of-eight particles
/// left out).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutlineStats {
    /// Irregularity S, the rms of a_3..a_8: median and log standard deviation.
    pub irregularity: f64,
    pub irregularity_sd: f64,
    /// a_n / S for n = 3..=WAVES: median and log standard deviation.
    pub profile: Vec<f64>,
    pub profile_sd: Vec<f64>,
    /// Deepest dent of the outline smoothed to WAVES waves (largest distance from the
    /// outline to its convex hull / equal-area diameter): quantiles at 0, 5,
    /// ..., 100 %.
    pub dent: Vec<f64>,
}

impl OutlineStats {
    /// Leighton Buzzard sand: 1029 single grains of the QicPic report
    /// (docs/soil_characterisation/Leighton_Buzzard.docx, 2026-11-09).
    pub fn leighton_buzzard() -> Self {
        Self {
            irregularity: 0.093479,
            irregularity_sd: 0.364682,
            profile: vec![0.7675, 0.4101, 0.2840, 0.1883, 0.1401, 0.1022, 0.0748, 0.0602, 0.0492, 0.0403, 0.0327,
                          0.0301, 0.0246, 0.0220, 0.0181, 0.0170, 0.0159, 0.0138, 0.0122, 0.0114, 0.0107, 0.0104],
            profile_sd: vec![0.3971, 0.6667, 0.6269, 0.6695, 0.7373, 0.7329, 0.7422, 0.7908, 0.7443, 0.7813, 0.7860,
                             0.7801, 0.7679, 0.7713, 0.7915, 0.7890, 0.7956, 0.7683, 0.8101, 0.7798, 0.7845, 0.7716],
            dent: vec![0.0005, 0.0069, 0.0106, 0.0139, 0.0164, 0.0187, 0.0211, 0.0237, 0.0259, 0.0285, 0.0314,
                       0.0337, 0.0366, 0.0401, 0.0441, 0.0488, 0.0547, 0.0606, 0.0685, 0.0805, 0.1167],
        }
    }

    pub fn validate(&self) -> Result<(), Error> {
        let ok = self.irregularity > 0.0
            && self.irregularity_sd >= 0.0
            && self.profile.len() == NW
            && self.profile_sd.len() == NW
            && self.profile.iter().all(|v| *v > 0.0)
            && self.profile_sd.iter().all(|v| *v >= 0.0)
            && self.dent.len() == 21
            && self.dent.windows(2).all(|w| w[1] >= w[0])
            && self.dent[0] >= 0.0;
        if ok { Ok(()) } else { Err(Error::InvalidInput("invalid outline statistics".into())) }
    }

    /// Dent limit at probability `u` in [0, 1] (between the quantiles).
    fn dent_at(&self, u: f64) -> f64 {
        let x = u.clamp(0.0, 1.0) * 20.0;
        let k = (x.floor() as usize).min(19);
        self.dent[k] + (x - k as f64) * (self.dent[k + 1] - self.dent[k])
    }
}

/// A unit grain outline: its radius tabulated at [`TABLE_N`] even angles, in
/// the frame turned by `turn`.
#[derive(Debug, Clone)]
struct Shape {
    r: std::sync::Arc<[f64]>,
    turn: f64,
}

/// 1 + sum_{n=3..WAVES} amp_n cos(n t + ph_n) (index 0 = wave 3).
fn wave_radius(amp: &[f64], ph: &[f64], t: f64) -> f64 {
    1.0 + (0..amp.len()).map(|k| amp[k] * ((k + 3) as f64 * t + ph[k]).cos()).sum::<f64>()
}

/// Radius of an ellipse with semi-axes 1 along `phi` and `q` across it.
fn ellipse_radius(q: f64, phi: f64, t: f64) -> f64 {
    let (s, c) = (t - phi).sin_cos();
    1.0 / (c * c + (s / q).powi(2)).sqrt()
}

impl Shape {
    fn tabulate(f: impl Fn(f64) -> f64) -> Self {
        let r: Vec<f64> = (0..TABLE_N).map(|j| f(2.0 * std::f64::consts::PI * j as f64 / TABLE_N as f64)).collect();
        Shape { r: r.into(), turn: 0.0 }
    }

    fn radius(&self, t: f64) -> f64 {
        let x = (t - self.turn).rem_euclid(2.0 * std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * TABLE_N as f64;
        let (j, f) = (x.floor() as usize % TABLE_N, x - x.floor());
        self.r[j] * (1.0 - f) + self.r[(j + 1) % TABLE_N] * f
    }

    /// The same outline turned by `alpha`.
    fn turned(&self, alpha: f64) -> Self {
        Shape { r: self.r.clone(), turn: self.turn + alpha }
    }

    fn polygon(&self) -> Vec<[f64; 2]> {
        const M: usize = 360;
        (0..M)
            .map(|j| {
                let t = 2.0 * std::f64::consts::PI * j as f64 / M as f64;
                let r = self.radius(t);
                [r * t.cos(), r * t.sin()]
            })
            .collect()
    }
}

impl Outline {
    /// Distance from the centre to the outline in direction `t` (grain frame).
    fn radius(&self, t: f64) -> f64 {
        match self {
            Outline::Circle { r } => *r,
            Outline::Shaped { r0, shape } => r0 * shape.radius(t),
        }
    }

    /// Signed distance (px) from the outline at grain-frame offset (x, y):
    /// negative inside. Exact for a circle, first-order near the edge otherwise.
    fn distance(&self, x: f64, y: f64) -> f64 {
        match self {
            Outline::Circle { r } => x.hypot(y) - r,
            Outline::Shaped { .. } => {
                let t = y.atan2(x);
                const H: f64 = 1e-4;
                let r = self.radius(t);
                let dr = (self.radius(t + H) - self.radius(t - H)) / (2.0 * H);
                (x.hypot(y) - r) / (1.0 + (dr / r).powi(2)).sqrt()
            }
        }
    }
}

/// Grains smaller than this (px) are not drawn: they cover no pixel centre.
pub const MIN_GRAIN_PX: f64 = 1.0;

/// Standard normal cumulative probability.
fn phi(z: f64) -> f64 {
    0.5 * libm::erfc(-z / std::f64::consts::SQRT_2)
}

/// Inverse of [`phi`] (Acklam's rational approximation, |error| < 1.2e-9).
fn probit(p: f64) -> f64 {
    const A: [f64; 6] = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const B: [f64; 5] = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let tail = |q: f64| (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// Grain sizes (px) drawn so that each size covers the share of the image
/// area that the size curve gives it. The curve is interpolated as straight
/// lines between its points on a log-size / probability scale, and extended
/// past its ends the same way (a curve of only x10, x50 and x90 is then
/// log-normal), from 0.01 % to 99.99 %. Grains below [`MIN_GRAIN_PX`] are not
/// drawn.
pub(crate) enum SizeDraw {
    Fixed(f64),
    /// Log-size bins `(lo, hi)` (px) and the cumulative chance by count.
    Curve(Vec<(f64, f64)>, Vec<f64>),
}

impl SizeDraw {
    const ZMAX: f64 = 3.719; // 0.01 % / 99.99 %
    const BINS: usize = 400;

    pub(crate) fn new(cfg: &SoilConfig) -> Result<Self, Error> {
        let bad = |m: &str| Err(Error::InvalidInput(m.into()));
        if cfg.psd.is_empty() {
            if cfg.d_p < MIN_GRAIN_PX {
                return bad("d_p is below one pixel: no grain can be drawn");
            }
            return Ok(SizeDraw::Fixed(cfg.d_p));
        }
        for w in cfg.psd.windows(2) {
            if !(w[1][0] > w[0][0]) || !(w[1][1] >= w[0][1]) {
                return bad("psd sizes must increase and % passing must not decrease");
            }
        }
        if cfg.psd.iter().any(|p| !(p[0] > 0.0) || !(0.0..=100.0).contains(&p[1])) {
            return bad("psd sizes must be > 0 and % passing between 0 and 100");
        }
        // (ln size, z): keep the last 0 % point and the first 100 % point, then
        // only points where z rises
        let lo = cfg.psd.iter().rposition(|p| p[1] <= 0.0).unwrap_or(0);
        let hi = cfg.psd.iter().position(|p| p[1] >= 100.0).unwrap_or(cfg.psd.len() - 1);
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for p in &cfg.psd[lo..=hi] {
            let z = probit((p[1] / 100.0).clamp(1e-4, 1.0 - 1e-4));
            if pts.last().map_or(true, |&(_, zl)| z > zl + 1e-12) {
                pts.push((p[0].ln(), z));
            }
        }
        if pts.len() < 2 {
            return bad("psd needs at least two points with different % passing");
        }
        // ln size at probability z: straight lines, end segments extended
        let lnx = |z: f64| -> f64 {
            let k = pts.windows(2).position(|w| z <= w[1].1).unwrap_or(pts.len() - 2);
            let (a, b) = (pts[k], pts[k + 1]);
            a.0 + (z - a.1) * (b.0 - a.0) / (b.1 - a.1)
        };
        let shift = cfg.d_p.ln() - lnx(0.0); // x50 -> d_p px
        let mut bins = Vec::with_capacity(Self::BINS);
        let mut cum = Vec::with_capacity(Self::BINS);
        let mut total = 0.0;
        for k in 0..Self::BINS {
            let (z0, z1) = (-Self::ZMAX + 2.0 * Self::ZMAX * k as f64 / Self::BINS as f64,
                            -Self::ZMAX + 2.0 * Self::ZMAX * (k + 1) as f64 / Self::BINS as f64);
            let (a, b) = (lnx(z0) + shift, lnx(z1) + shift);
            let mid = (0.5 * (a + b)).exp();
            if mid < MIN_GRAIN_PX {
                continue;
            }
            // area share of the bin / area of one grain = count share
            total += (phi(z1) - phi(z0)) / (mid * mid);
            bins.push((a, b));
            cum.push(total);
        }
        if bins.is_empty() {
            return bad("every grain of the psd is below one pixel: raise d_p");
        }
        cum.iter_mut().for_each(|c| *c /= total);
        Ok(SizeDraw::Curve(bins, cum))
    }

    /// Mean of d^2 (px^2) over the grains drawn, weighted by area.
    fn area_mean_d2(&self) -> f64 {
        match self {
            SizeDraw::Fixed(d) => d * d,
            SizeDraw::Curve(bins, cum) => {
                // a bin's area share is its count share times the area of one grain
                let (mut a, mut ad2) = (0.0, 0.0);
                for (k, &(lo, hi)) in bins.iter().enumerate() {
                    let d2 = (lo + hi).exp(); // (geometric mid size)^2
                    let w = (cum[k] - if k == 0 { 0.0 } else { cum[k - 1] }) * d2;
                    a += w;
                    ad2 += w * d2;
                }
                ad2 / a
            }
        }
    }

    fn draw(&self, rng: &mut SmallRng) -> f64 {
        match self {
            SizeDraw::Fixed(d) => *d,
            SizeDraw::Curve(bins, cum) => {
                let u: f64 = rng.r#gen();
                let k = cum.partition_point(|&c| c < u).min(bins.len() - 1);
                let (a, b) = bins[k];
                (a + rng.r#gen::<f64>() * (b - a)).exp()
            }
        }
    }
}

/// Area and perimeter of a closed polygon.
fn area_perimeter(pts: &[[f64; 2]]) -> (f64, f64) {
    let m = pts.len();
    let (mut a, mut per) = (0.0, 0.0);
    for j in 0..m {
        let (p, q) = (pts[j], pts[(j + 1) % m]);
        a += 0.5 * (p[0] * q[1] - q[0] * p[1]);
        per += (q[0] - p[0]).hypot(q[1] - p[1]);
    }
    (a, per)
}

/// Sphericity: perimeter of the equal-area circle / perimeter.
fn sphericity_of(pts: &[[f64; 2]]) -> f64 {
    let (a, per) = area_perimeter(pts);
    2.0 * (std::f64::consts::PI * a).sqrt() / per
}

/// Aspect ratio: narrowest / widest Feret width (over 180 directions).
fn aspect_of(pts: &[[f64; 2]]) -> f64 {
    aspect_k(pts, 180)
}

/// Aspect ratio over `n` directions.
fn aspect_k(pts: &[[f64; 2]], n: usize) -> f64 {
    let (mut lo, mut hi) = (f64::MAX, 0.0f64);
    for k in 0..n {
        let t = std::f64::consts::PI * k as f64 / n as f64;
        let (c, s) = (t.cos(), t.sin());
        let (mut a, mut b) = (f64::MAX, f64::MIN);
        for p in pts {
            let v = p[0] * c + p[1] * s;
            a = a.min(v);
            b = b.max(v);
        }
        lo = lo.min(b - a);
        hi = hi.max(b - a);
    }
    lo / hi
}

/// Deepest dent of a closed outline given in angle order about an inside
/// point: the largest distance from its points to the sides of its convex
/// hull, over its equal-area diameter (as geopyv_dev/_qicpic.py measures it).
fn dent_of(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    // Graham scan: the points are already in angle order about an inside
    // point, and the farthest one is a hull corner
    let s = (0..n).max_by(|&a, &b| pts[a][0].hypot(pts[a][1]).total_cmp(&pts[b][0].hypot(pts[b][1]))).unwrap();
    let mut st: Vec<usize> = vec![s];
    for k in 1..=n {
        let j = (s + k) % n;
        while st.len() >= 2 {
            let (a, b, c) = (pts[st[st.len() - 2]], pts[st[st.len() - 1]], pts[j]);
            if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 0.0 {
                st.pop();
            } else {
                break;
            }
        }
        st.push(j);
    }
    st.pop(); // s again
    let mut deep = 0.0f64;
    for w in 0..st.len() {
        let (a, b) = (st[w], st[(w + 1) % st.len()]);
        let e = [pts[b][0] - pts[a][0], pts[b][1] - pts[a][1]];
        let len = e[0].hypot(e[1]);
        let mut m = (a + 1) % n;
        while m != b {
            let d = (e[0] * (pts[m][1] - pts[a][1]) - e[1] * (pts[m][0] - pts[a][0])).abs() / len;
            deep = deep.max(d);
            m = (m + 1) % n;
        }
    }
    deep / (2.0 * (area_perimeter(pts).0 / std::f64::consts::PI).sqrt())
}

/// Draws allowed per outline while it is more dented than its limit.
const MAX_REDRAW: usize = 50;

/// One unit outline from `stats` with aspect ratio `aspect`: an ellipse (the
/// elongation, turned at random) times irregularity waves 3 and up (level
/// log-normal, each wave's share of it log-normal, random phases), the
/// ellipse's axis ratio set so that the outline's aspect ratio is `aspect`
/// (an irregular outline may already be less round than asked; it is then
/// kept as it is), redrawn while its deepest dent is deeper than a limit drawn
/// from the real grains' dents; the least dented draw is kept.
fn draw_shape(stats: &OutlineStats, aspect: f64, rng: &mut SmallRng) -> Shape {
    use std::f64::consts::TAU;
    const M: usize = 180; // points for fitting the aspect ratio
    const MD: usize = 360; // points for the dent, as Shape::polygon
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let s = stats.irregularity * (stats.irregularity_sd * n01.sample(rng)).exp();
    let cap = stats.dent_at(rng.r#gen());
    // cos and sin of n t (n = 3..WAVES) at the MD outline points, once: a
    // wave's value is then cos(n t) cos(ph) - sin(n t) sin(ph)
    let trig: Vec<(f64, f64, Vec<(f64, f64)>)> = (0..MD)
        .map(|j| {
            let t = TAU * j as f64 / MD as f64;
            (t.cos(), t.sin(), (0..NW).map(|k| ((k + 3) as f64 * t).sin_cos()).map(|(s, c)| (c, s)).collect())
        })
        .collect();
    // outline at every `step`-th of the MD points
    let polygon = |q: f64, phi: f64, amp: &[f64], ph: &[f64], step: usize| -> Vec<[f64; 2]> {
        let cs: Vec<(f64, f64)> = ph.iter().map(|p| (p.cos(), p.sin())).collect();
        (0..MD)
            .step_by(step)
            .map(|j| {
                let (c, sn, nt) = &trig[j];
                let w = 1.0 + (0..NW).map(|k| amp[k] * (nt[k].0 * cs[k].0 - nt[k].1 * cs[k].1)).sum::<f64>();
                let t = TAU * j as f64 / MD as f64;
                let r = ellipse_radius(q, phi, t) * w;
                [r * c, r * sn]
            })
            .collect()
    };
    // redraw the waves while too dented, judged on the ellipse of axis ratio
    // `aspect` (the axis ratio is refined for the kept draw only)
    let mut best: Option<(f64, f64, Vec<f64>, Vec<f64>)> = None;
    for _ in 0..MAX_REDRAW {
        let phi = rng.r#gen::<f64>() * std::f64::consts::PI;
        let ph: Vec<f64> = (0..NW).map(|_| rng.r#gen::<f64>() * TAU).collect();
        let amp: Vec<f64> =
            (0..NW).map(|k| s * stats.profile[k] * (stats.profile_sd[k] * n01.sample(rng)).exp()).collect();
        if polygon(1.0, 0.0, &amp, &ph, 1).iter().any(|p| p[0].hypot(p[1]) < 0.15) {
            continue;
        }
        let dent = dent_of(&polygon(aspect, phi, &amp, &ph, 1));
        if best.as_ref().map_or(true, |b| dent < b.0) {
            best = Some((dent, phi, amp, ph));
        }
        if dent <= cap {
            break;
        }
    }
    let Some((_, phi, amp, ph)) = best else { return Shape::tabulate(|_| 1.0) };
    let aspect_with = |q: f64| aspect_k(&polygon(q, phi, &amp, &ph, MD / M), 90);
    let q = if aspect_with(1.0) <= aspect {
        1.0
    } else {
        let (mut lo, mut hi) = (0.1, 1.0);
        for _ in 0..25 {
            let m = 0.5 * (lo + hi);
            if aspect_with(m) > aspect { hi = m } else { lo = m }
        }
        0.5 * (lo + hi)
    };
    Shape::tabulate(|t| ellipse_radius(q, phi, t) * wave_radius(&amp, &ph, t))
}

/// Number of grain outlines drawn per image set.
const SHAPE_LIBRARY: usize = 256;

/// Outlines with aspect ratios drawn from their spread (independently), each
/// with its unit area and largest radius; empty when every grain is round
/// (aspect ratio 1, no spread, no outline statistics given).
fn shape_library(cfg: &SoilConfig, seed: u64) -> Vec<(Shape, f64, f64)> {
    if cfg.outlines.is_none() && cfg.aspect_ratio >= 1.0 && cfg.aspect_ratio_sd == 0.0 {
        return Vec::new();
    }
    let stats = cfg.outlines.clone().unwrap_or_else(OutlineStats::leighton_buzzard);
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x0C1D_5000_0000_0004);
    let n01 = Normal::new(0.0, 1.0).unwrap();
    (0..SHAPE_LIBRARY)
        .map(|_| {
            let a = (cfg.aspect_ratio + cfg.aspect_ratio_sd * n01.sample(&mut rng)).clamp(0.2, 1.0);
            let shape = draw_shape(&stats, a, &mut rng);
            let pts = shape.polygon();
            let reach = pts.iter().map(|p| p[0].hypot(p[1])).fold(0.0, f64::max);
            (shape, area_perimeter(&pts).0, reach)
        })
        .collect()
}

fn make_grain(cfg: &SoilConfig, rng: &mut SmallRng, layer: usize, sizes: &SizeDraw, lib: &[(Shape, f64, f64)]) -> Grain {
    let r = 0.5 * sizes.draw(rng);
    // Clay: every grain is dyed. Sand: dyed with chance asr, else natural.
    let dyed = cfg.kind == SoilKind::Clay || rng.r#gen::<f64>() < cfg.asr;
    let (mean, spread) = if dyed { (cfg.seed_brightness, cfg.seed_spread) } else { (cfg.matrix_brightness, cfg.matrix_spread) };
    let bright = (mean * (1.0 + spread * Normal::new(0.0, 1.0).unwrap().sample(rng))).max(0.0);
    let (outline, reach) = if lib.is_empty() {
        (Outline::Circle { r }, r)
    } else {
        // one of the fitted outlines, turned at random, scaled to the grain's area
        let (shape, unit_area, unit_reach) = &lib[rng.gen_range(0..lib.len())];
        let r0 = r / (unit_area / std::f64::consts::PI).sqrt();
        let shape = shape.turned(rng.r#gen::<f64>() * 2.0 * std::f64::consts::PI);
        (Outline::Shaped { r0, shape }, r0 * unit_reach)
    };
    Grain { outline, brightness: bright, look: Look::default(), dyed, layer, reach, area: std::f64::consts::PI * r * r }
}

/// Share of a sand layer's area that its drawn grains add up to (placed
/// largest first; the gaps show deeper layers).
const SAND_FILL: f64 = 0.5;
/// Random positions tried for each grain before it is dropped.
const TRIES: usize = 400;

/// Grains of every layer at their reference centres, placed at random so that
/// none crosses another in its layer (a pixel's centre inside one grain is not
/// inside another; touching is allowed). Each layer draws grains from the size
/// distribution until their total area reaches its target (clay: `asr` of the
/// image; sand: [`SAND_FILL`]), then places them largest first, each at the
/// first of [`TRIES`] random positions where it fits (dropped if none). The
/// placed grains so keep the drawn sizes - `grain_size` is their median -
/// instead of being mostly the small grains that still fit late in a fill.
/// Clay tops up until `asr` of the image is covered.
pub(crate) fn place(cfg: &SoilConfig, size: (usize, usize), seed: u64) -> (Vec<[f64; 2]>, Vec<Grain>) {
    let (w, h) = size;
    let mut rng = SmallRng::seed_from_u64(seed);
    let lib = shape_library(cfg, seed);
    let sizes = SizeDraw::new(cfg).expect("validated soil settings");
    let mut pos = Vec::new();
    let mut grains = Vec::new();
    let layers = if cfg.kind == SoilKind::Sand { cfg.depth_layers } else { 1 };
    let total = (w * h) as f64;
    for layer in 0..layers {
        let mut occupied = vec![false; w * h];
        let mut covered = 0usize;
        let target = if cfg.kind == SoilKind::Clay { cfg.asr * total } else { SAND_FILL * total };
        for _round in 0..20 {
            let want = target - covered as f64;
            if want < 0.5 {
                break;
            }
            let mut batch = Vec::new();
            let mut drawn = 0.0;
            while drawn < want {
                let g = make_grain(cfg, &mut rng, layer, &sizes, &lib);
                drawn += g.area;
                batch.push(g);
            }
            batch.sort_by(|a, b| b.area.partial_cmp(&a.area).unwrap());
            let before = covered;
            for g in batch {
                if cfg.kind == SoilKind::Clay && covered as f64 >= target {
                    break;
                }
                let reach = g.reach.ceil() as isize + 1;
                for _ in 0..TRIES {
                    let c = [rng.r#gen::<f64>() * w as f64, rng.r#gen::<f64>() * h as f64];
                    // quick rejection: the centre pixel is already taken
                    let (cx, cy) = ((c[0] as usize).min(w - 1), (c[1] as usize).min(h - 1));
                    if occupied[cy * w + cx] {
                        continue;
                    }
                    let (x0, x1) = ((c[0] as isize - reach).max(0), (c[0] as isize + reach + 1).min(w as isize));
                    let (y0, y1) = ((c[1] as isize - reach).max(0), (c[1] as isize + reach + 1).min(h as isize));
                    let mut cells = Vec::new();
                    let mut fits = true;
                    'scan: for py in y0..y1 {
                        for px in x0..x1 {
                            if g.outline.distance(px as f64 - c[0], py as f64 - c[1]) < 0.0 {
                                let q = py as usize * w + px as usize;
                                if occupied[q] {
                                    fits = false;
                                    break 'scan;
                                }
                                cells.push(q);
                            }
                        }
                    }
                    if fits && !cells.is_empty() {
                        covered += cells.len();
                        cells.into_iter().for_each(|q| occupied[q] = true);
                        pos.push(c);
                        grains.push(g);
                        break;
                    }
                }
            }
            // sand: one pass per layer; clay: stop when nothing more fits
            if cfg.kind == SoilKind::Sand || covered == before {
                break;
            }
        }
    }
    // brightness variation inside each grain, from its own random stream so
    // that placement does not depend on it
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x6C00_4B00_0000_0005);
    let normal = Normal::new(0.0, 1.0).unwrap();
    // Rayleigh of median m: each component normal of width m / sqrt(2 ln 2)
    let tilt_sd = cfg.grain_tilt / (2.0 * std::f64::consts::LN_2).sqrt();
    for g in &mut grains {
        g.look = Look {
            tilt: [tilt_sd * normal.sample(&mut rng), tilt_sd * normal.sample(&mut rng)],
            dome: cfg.grain_dome * normal.sample(&mut rng),
            offset: [rng.r#gen::<f64>() * 1000.0, rng.r#gen::<f64>() * 1000.0],
        };
    }
    (pos, grains)
}

/// What a grain is, as seen in one image (see [`describe`]), measured as a
/// grain characterisation measures it.
#[derive(Debug, Clone, PartialEq)]
pub struct GrainInfo {
    /// Centre (px) in this image.
    pub x: f64,
    pub y: f64,
    /// EQPC: diameter of a circle of the same area (px).
    pub size: f64,
    /// Narrowest / widest Feret width (1 for a circle).
    pub aspect_ratio: f64,
    /// Direction of the long axis (second moments) in this image (rad, from
    /// +x towards +y, in [0, pi)); 0 for a circle.
    pub orientation: f64,
    pub brightness: f64,
    pub dyed: bool,
    /// Depth layer, 0 nearest (always 0 in clay).
    pub layer: usize,
}

/// Size, aspect ratio and orientation of each grain from its
/// outline (720 points), turned by `turn` (rad), at `centres`.
pub(crate) fn describe(grains: &[Grain], centres: ArrayView2<f64>, turn: &[f64]) -> Vec<GrainInfo> {
    const M: usize = 720;
    grains
        .par_iter()
        .enumerate()
        .map(|(k, g)| {
            let pts: Vec<[f64; 2]> = (0..M)
                .map(|j| {
                    let t = 2.0 * std::f64::consts::PI * j as f64 / M as f64;
                    let r = g.outline.radius(t);
                    [r * t.cos(), r * t.sin()]
                })
                .collect();
            // centroid and second moments for the long axis
            let (mut a, mut cx, mut cy, mut ixx, mut iyy, mut ixy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for j in 0..M {
                let (p, q) = (pts[j], pts[(j + 1) % M]);
                let c = p[0] * q[1] - q[0] * p[1];
                a += 0.5 * c;
                cx += (p[0] + q[0]) * c / 6.0;
                cy += (p[1] + q[1]) * c / 6.0;
                ixx += (p[0] * p[0] + p[0] * q[0] + q[0] * q[0]) * c / 12.0;
                iyy += (p[1] * p[1] + p[1] * q[1] + q[1] * q[1]) * c / 12.0;
                ixy += (p[0] * q[1] + 2.0 * p[0] * p[1] + 2.0 * q[0] * q[1] + q[0] * p[1]) * c / 24.0;
            }
            let (cx, cy) = (cx / a, cy / a);
            let (sxx, syy, sxy) = (ixx / a - cx * cx, iyy / a - cy * cy, ixy / a - cx * cy);
            let round = 0.25 * (sxx - syy).powi(2) + sxy * sxy < 1e-12 * (sxx + syy).powi(2);
            let orientation = if round {
                0.0
            } else {
                (0.5 * (2.0 * sxy).atan2(sxx - syy) + turn[k]).rem_euclid(std::f64::consts::PI)
            };
            GrainInfo {
                x: centres[[k, 0]],
                y: centres[[k, 1]],
                size: 2.0 * (a / std::f64::consts::PI).sqrt(),
                aspect_ratio: aspect_of(&pts),
                orientation,
                brightness: g.brightness,
                dyed: g.dyed,
                layer: g.layer,
            }
        })
        .collect()
}

/// Which grain each pixel shows: grain index + 1 where that grain is seen over
/// more than half of the pixel (see [`pixel_mixes`]); 0 where the matrix is;
/// -1 where nothing is (a pixel shared along an edge). Row-major `(h, w)`.
pub(crate) fn ownership(size: (usize, usize), grains: &[Grain], centres: ArrayView2<f64>, turn: &[f64]) -> Array2<i64> {
    let (w, h) = size;
    let mixes = pixel_mixes(size, grains, centres, turn);
    let owner: Vec<i64> = mixes
        .par_iter()
        .map(|mix| match mix.iter().copied().fold((0i64, 0.0), |b, c| if c.1 > b.1 { c } else { b }) {
            (id, share) if share > 0.5 => id,
            _ => -1,
        })
        .collect();
    Array2::from_shape_vec((h, w), owner).unwrap()
}

/// Samples per pixel side for pixels shared between grains or with the matrix.
const SUB: usize = 8;

/// What each pixel shows: `(grain index + 1, share of the pixel)`, 0 for the
/// matrix, shares adding to 1. A share is the part of the pixel where that
/// grain is the nearest one seen: the pixel is sampled at SUB x SUB points and
/// each point shows the nearest grain covering it (nearer layers in front),
/// else the matrix. A pixel lying wholly inside one grain with nothing in
/// front of it is that grain alone, without sampling. Row-major.
pub(crate) fn pixel_mixes(size: (usize, usize), grains: &[Grain], centres: ArrayView2<f64>, turn: &[f64]) -> Vec<Vec<(i64, f64)>> {
    let (w, h) = size;
    let trig: Vec<(f64, f64)> = turn.iter().map(|t| (t.cos(), t.sin())).collect();
    // signed distance from grain k's outline at (x, y), in its own frame
    let dist = |k: usize, x: f64, y: f64| {
        let (dx, dy) = (x - centres[[k, 0]], y - centres[[k, 1]]);
        let (c, s) = trig[k];
        grains[k].outline.distance(c * dx + s * dy, -s * dx + c * dy)
    };
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); h];
    for (k, g) in grains.iter().enumerate() {
        let (cy, r) = (centres[[k, 1]], g.reach + 1.5);
        let (y0, y1) = ((cy - r).floor().max(0.0) as usize, ((cy + r).ceil().max(0.0) as usize).min(h));
        for row in rows.iter_mut().take(y1).skip(y0) {
            row.push(k);
        }
    }
    (0..h)
        .into_par_iter()
        .flat_map_iter(|py| {
            // grains that may reach into each pixel of this row
            let mut cand: Vec<Vec<(usize, f64)>> = vec![Vec::new(); w];
            for &k in &rows[py] {
                let (cx, r) = (centres[[k, 0]], grains[k].reach + 1.5);
                let (x0, x1) = ((cx - r).floor().max(0.0) as usize, ((cx + r).ceil().max(0.0) as usize).min(w));
                for (px, c) in cand.iter_mut().enumerate().take(x1).skip(x0) {
                    let d = dist(k, px as f64, py as f64);
                    if d < 1.0 {
                        c.push((k, d));
                    }
                }
            }
            cand.into_iter().enumerate().map(move |(px, mut c)| {
                if c.is_empty() {
                    return vec![(0, 1.0)];
                }
                c.sort_by_key(|&(k, _)| grains[k].layer);
                // the nearest grain covering the whole pixel (1 px inside its edge)
                if let Some(pos) = c.iter().position(|&(_, d)| d < -1.0) {
                    let layer = grains[c[pos].0].layer;
                    if c[..pos].iter().all(|&(k, _)| grains[k].layer == layer) {
                        // grains of its own layer do not cross it
                        return vec![(c[pos].0 as i64 + 1, 1.0)];
                    }
                    c.retain(|&(k, _)| grains[k].layer <= layer);
                }
                let mut count: Vec<(i64, usize)> = Vec::new();
                for i in 0..SUB {
                    for j in 0..SUB {
                        let (sx, sy) = (px as f64 + (i as f64 + 0.5) / SUB as f64 - 0.5, py as f64 + (j as f64 + 0.5) / SUB as f64 - 0.5);
                        let id = c.iter().find(|&&(k, _)| dist(k, sx, sy) < 0.0).map_or(0, |&(k, _)| k as i64 + 1);
                        match count.iter_mut().find(|e| e.0 == id) {
                            Some(e) => e.1 += 1,
                            None => count.push((id, 1)),
                        }
                    }
                }
                count.into_iter().map(|(id, n)| (id, n as f64 / (SUB * SUB) as f64)).collect()
            })
        })
        .collect()
}

/// The clay's own pattern: a smooth random field of unit spread whose
/// self-similarity falls to one half at `size` px (Gaussian spectrum, sum of
/// random waves), attached to the material.
pub(crate) struct Pattern {
    k: Vec<[f64; 2]>,
    ph: Vec<f64>,
    norm: f64,
}

impl Pattern {
    pub(crate) fn new(size: f64, seed: u64) -> Self {
        const K: usize = 400;
        let l = size / (2.0 * std::f64::consts::LN_2).sqrt();
        let mut rng = SmallRng::seed_from_u64(seed ^ 0xC1A7_0000_0000_0003);
        let nk = Normal::new(0.0, 1.0 / l).unwrap();
        let k = (0..K).map(|_| [nk.sample(&mut rng), nk.sample(&mut rng)]).collect();
        let ph = (0..K).map(|_| rng.r#gen::<f64>() * 2.0 * std::f64::consts::PI).collect();
        Self { k, ph, norm: (2.0 / K as f64).sqrt() }
    }

    #[inline]
    pub(crate) fn at(&self, x: f64, y: f64) -> f64 {
        self.norm * self.k.iter().zip(&self.ph).map(|(k, p)| (k[0] * x + k[1] * y + p).cos()).sum::<f64>()
    }
}

/// Render one image. `centres`: grain centres in this image; `turn`: each
/// grain's rotation (rad); `material`: each pixel's original (reference)
/// position, row-major `(h*w, 2)`, for the clay pattern (`None` for sand);
/// `texture`: the grains' texture field (in grain diameters; `None` for no
/// texture), the same for every image of a set.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render(
    cfg: &SoilConfig,
    size: (usize, usize),
    grains: &[Grain],
    centres: ArrayView2<f64>,
    turn: &[f64],
    material: Option<&Array2<f64>>,
    pattern: Option<&Pattern>,
    texture: Option<&Pattern>,
    noise_grey: f64,
    seed: u64,
) -> Result<Array2<u8>, Error> {
    let (w, h) = size;
    // background
    let mut img: Vec<f64> = match (material, pattern) {
        (Some(m), Some(p)) if cfg.kind == SoilKind::Clay && cfg.matrix_spread > 0.0 => (0..w * h)
            .into_par_iter()
            .map(|q| cfg.matrix_brightness * (1.0 + cfg.matrix_spread * p.at(m[[q, 0]], m[[q, 1]])))
            .collect(),
        _ => vec![cfg.matrix_brightness; w * h],
    };
    // each pixel: the grains and matrix seen in it, weighted by the part of
    // the pixel each is seen over
    let mixes = pixel_mixes(size, grains, centres, turn);
    let trig: Vec<(f64, f64)> = turn.iter().map(|t| (t.cos(), t.sin())).collect();
    img.par_iter_mut().zip(mixes.par_iter()).enumerate().for_each(|(q, (v, mix))| {
        let (x, y) = ((q % w) as f64, (q / w) as f64);
        *v = mix
            .iter()
            .map(|&(id, share)| {
                share * if id > 0 {
                    // the grain's brightness at the pixel centre, in its own frame
                    let k = id as usize - 1;
                    let (dx, dy) = (x - centres[[k, 0]], y - centres[[k, 1]]);
                    let (c, s) = trig[k];
                    grains[k].shade(cfg, c * dx + s * dy, -s * dx + c * dy, texture)
                } else {
                    *v
                }
            })
            .sum();
    });
    // lens blur
    if cfg.blur > 0.0 {
        let a = Array2::from_shape_vec((h, w), img).map_err(|e| Error::InvalidInput(e.to_string()))?;
        let rr = (4.0 * cfg.blur).ceil() as i64;
        let k: Vec<f64> = (-rr..=rr).map(|i| (-(i * i) as f64 / (2.0 * cfg.blur * cfg.blur)).exp()).collect();
        let ks: f64 = k.iter().sum();
        let k: Vec<f64> = k.into_iter().map(|v| v / ks).collect();
        let pass = |src: &Array2<f64>, along_rows: bool| -> Vec<f64> {
            let mut out = vec![0.0; w * h];
            out.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
                for (x, o) in line.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for (j, kv) in k.iter().enumerate() {
                        let t = j as i64 - rr;
                        s += kv * if along_rows {
                            src[[y, (x as i64 + t).clamp(0, w as i64 - 1) as usize]]
                        } else {
                            src[[(y as i64 + t).clamp(0, h as i64 - 1) as usize, x]]
                        };
                    }
                    *o = s;
                }
            });
            out
        };
        let once = Array2::from_shape_vec((h, w), pass(&a, true)).unwrap();
        img = pass(&once, false);
    }
    // noise, 8-bit
    let mut rng = SmallRng::seed_from_u64(seed ^ 0xDEAD_BEEF);
    let normal = Normal::new(0.0, noise_grey.max(0.0)).map_err(|e| Error::InvalidInput(e.to_string()))?;
    let mut out = Array2::<u8>::zeros((h, w));
    for q in 0..w * h {
        let v = img[q] + if noise_grey > 0.0 { normal.sample(&mut rng) } else { 0.0 };
        out[[q / w, q % w]] = v.round().clamp(0.0, 255.0) as u8;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drawn outlines hit the aspect ratio asked for (where an outline with no
    /// 2-bulge wave is not already less round), stay within the dent limits,
    /// and are measured as _qicpic.py measures them (an ellipse has no dent).
    #[test]
    fn outlines_hit_aspect_and_dents() {
        let stats = OutlineStats::leighton_buzzard();
        stats.validate().unwrap();
        let mut rng = SmallRng::seed_from_u64(4);
        for a in [0.5, 0.6, 0.7, 0.8] {
            let mut dents = Vec::new();
            for _ in 0..40 {
                let pts = draw_shape(&stats, a, &mut rng).polygon();
                let ga = aspect_of(&pts);
                assert!(ga < a + 0.01, "aspect {ga} for {a}");
                if a <= 0.7 {
                    assert!((ga - a).abs() < 0.01, "aspect {ga} for {a}");
                }
                dents.push(dent_of(&pts));
            }
            dents.sort_by(f64::total_cmp);
            // limits are drawn from the real dents, so the median dent is no
            // deeper than the real median (0.031)
            assert!(dents[20] < 0.035, "median dent {} at aspect {a}", dents[20]);
        }
        let ell = Shape::tabulate(|t| 1.0 / (t.cos().powi(2) + (t.sin() / 0.6).powi(2)).sqrt());
        assert!(dent_of(&ell.polygon()) < 1e-9);
        assert!((aspect_of(&ell.polygon()) - 0.6).abs() < 1e-3);
    }

    /// Turning an outline moves it round: radius(t + alpha) of the turned
    /// outline is radius(t) of the original. Round grains stay circles.
    #[test]
    fn outline_turns_and_round_grains_stay_circles() {
        let mut rng = SmallRng::seed_from_u64(11);
        let shape = draw_shape(&OutlineStats::leighton_buzzard(), 0.6, &mut rng);
        let t = shape.turned(0.7);
        for k in 0..50 {
            let th = 0.123 * k as f64;
            assert!((t.radius(th + 0.7) - shape.radius(th)).abs() < 1e-9);
        }
        let round = SoilConfig::sand(100.0, 60.0, 0.2);
        assert!(shape_library(&round, 1).is_empty());
        assert!(!shape_library(&SoilConfig { outlines: Some(OutlineStats::leighton_buzzard()), ..round.clone() }, 1).is_empty());
        assert!(!shape_library(&SoilConfig { aspect_ratio: 0.7, ..round }, 1).is_empty());
    }

    /// Placed grains follow the size curve by area: a log-normal curve given by
    /// x10, x50, x90 only; x50 lands on d_p pixels; nothing below a pixel.
    #[test]
    fn sizes_follow_curve_by_area() {
        let psd = vec![[200.0, 10.0], [312.5, 50.0], [500.0, 90.0]];
        let cfg = SoilConfig { psd, d_p: 10.0, ..SoilConfig::clay(180.0, 100.0, 0.3) };
        let (_, grains) = place(&cfg, (600, 600), 3);
        let mut sz: Vec<(f64, f64)> = grains.iter().map(|g| (2.0 * (g.area / std::f64::consts::PI).sqrt(), g.area)).collect();
        assert!(sz.iter().all(|s| s.0 >= MIN_GRAIN_PX));
        sz.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let total: f64 = sz.iter().map(|s| s.1).sum();
        let at = |q: f64| {
            let mut c = 0.0;
            sz.iter().find(|s| { c += s.1; c >= q * total }).unwrap().0
        };
        for (q, want) in [(0.1, 6.4), (0.5, 10.0), (0.9, 16.0)] {
            assert!((at(q) / want - 1.0).abs() < 0.05, "{q}: {} vs {want}", at(q));
        }
    }

    /// The probability scale round-trips.
    #[test]
    fn probit_inverts_phi() {
        for z in [-3.5, -1.2816, 0.0, 0.7, 2.9] {
            assert!((probit(phi(z)) - z).abs() < 1e-6);
        }
    }

    #[test]
    fn clay_coverage_and_no_crossing() {
        let cfg = SoilConfig { aspect_ratio: 0.5, ..SoilConfig::clay(180.0, 135.0, 0.2) };
        let (pos, grains) = place(&cfg, (200, 150), 1);
        let mut count = vec![0u8; 200 * 150];
        for (k, g) in grains.iter().enumerate() {
            for py in 0..150 {
                for px in 0..200 {
                    if g.outline.distance(px as f64 - pos[k][0], py as f64 - pos[k][1]) < 0.0 {
                        count[py * 200 + px] += 1;
                    }
                }
            }
        }
        assert!(count.iter().all(|&c| c <= 1), "a pixel centre lies inside two grains");
        let cov = count.iter().filter(|&&c| c > 0).count() as f64 / (200.0 * 150.0);
        assert!((cov - 0.2).abs() < 0.01, "share covered {cov}");
    }

    #[test]
    fn pattern_has_unit_spread_and_size() {
        let p = Pattern::new(10.0, 2);
        let n = 256;
        let v: Vec<f64> = (0..n * n).map(|q| p.at((q % n) as f64, (q / n) as f64)).collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        assert!((sd - 1.0).abs() < 0.2, "spread {sd}");
        // self-similarity at a shift of 10 px about one half
        let c10: f64 = (0..n * (n - 10)).map(|q| (v[q] - m) * (v[q + 10] - m)).sum::<f64>() / (n * (n - 10)) as f64 / (sd * sd);
        assert!((c10 - 0.5).abs() < 0.15, "self-similarity at 10 px {c10}");
    }

    #[test]
    fn sand_layers_leave_few_gaps() {
        let cfg = SoilConfig { depth_layers: 4, ..SoilConfig::sand(120.0, 60.0, 0.25) };
        let (pos, grains) = place(&cfg, (150, 150), 3);
        let mut seen = vec![false; 150 * 150];
        for (k, g) in grains.iter().enumerate() {
            for py in 0..150 {
                for px in 0..150 {
                    if g.outline.distance(px as f64 - pos[k][0], py as f64 - pos[k][1]) < 0.0 {
                        seen[py * 150 + px] = true;
                    }
                }
            }
        }
        let gaps = seen.iter().filter(|&&s| !s).count() as f64 / (150.0 * 150.0);
        assert!(gaps < 0.15, "share of pixels with no grain in any layer: {gaps}");
    }

    /// Sand: the share of grains dyed follows asr (0 none, 1 all).
    #[test]
    fn sand_asr_sets_dyed_share() {
        for asr in [0.0, 0.3, 1.0] {
            // narrow spreads so that natural (120) and dyed (60) grains separate cleanly at 90
            let cfg = SoilConfig { matrix_spread: 0.05, seed_spread: 0.05, ..SoilConfig::sand(120.0, 60.0, asr) };
            let (_, grains) = place(&cfg, (200, 200), 9);
            let front: Vec<&Grain> = grains.iter().filter(|g| g.layer == 0).collect();
            let dyed = front.iter().filter(|g| g.brightness < 90.0).count() as f64 / front.len() as f64;
            assert!((dyed - asr).abs() < 0.05, "asr {asr}: dyed share {dyed}");
        }
    }

    /// describe: a circle is exact; a two-bulge outline's long axis follows
    /// its turn.
    #[test]
    fn describe_simple_outlines() {
        let c = Grain { outline: Outline::Circle { r: 3.0 }, brightness: 1.0, look: Look::default(), dyed: true, layer: 0, reach: 3.0, area: 0.0 };
        let w = Grain { outline: Outline::Shaped { r0: 4.0, shape: Shape::tabulate(|t| 1.0 / ((t - 0.3).cos().powi(2) + ((t - 0.3).sin() / 0.5).powi(2)).sqrt()) },
                        brightness: 1.0, look: Look::default(), dyed: true, layer: 0, reach: 4.0, area: 0.0 };
        let centres = ndarray::array![[10.0, 20.0], [30.0, 40.0]];
        let d = describe(&[c, w], centres.view(), &[0.0, 0.2]);
        assert!((d[0].size - 6.0).abs() < 1e-3 && (d[0].aspect_ratio - 1.0).abs() < 1e-3);
        assert!((d[1].aspect_ratio - 0.5).abs() < 0.002, "{}", d[1].aspect_ratio);
        assert!((d[1].orientation - 0.5).abs() < 1e-6, "{}", d[1].orientation);
    }

    /// Ownership agrees with the rendered image: clay share owned by grains is
    /// the asr, and an owned pixel is nearer its grain's brightness than the
    /// matrix's (more than half of it is that grain).
    #[test]
    fn ownership_matches_render() {
        let cfg = flat(SoilConfig { matrix_spread: 0.0, ..SoilConfig::clay(180.0, 100.0, 0.2) });
        let (pos, grains) = place(&cfg, (200, 200), 5);
        let c = Array2::from_shape_fn((pos.len(), 2), |(k, j)| pos[k][j]);
        let turn = vec![0.4; pos.len()];
        let own = ownership((200, 200), &grains, c.view(), &turn);
        let img = render(&cfg, (200, 200), &grains, c.view(), &turn, None, None, None, 0.0, 1).unwrap();
        let owned = own.iter().filter(|&&o| o > 0).count() as f64 / 40000.0;
        assert!((owned - 0.2).abs() < 0.02, "owned share {owned}");
        for (q, &o) in own.iter().enumerate() {
            let v = img[[q / 200, q % 200]] as f64;
            if o > 0 {
                let g = grains[o as usize - 1].brightness;
                assert!((v - g).abs() <= (v - 180.0).abs() + 1.0, // + 8-bit rounding
                        "pixel {q}: {v} owned by grain at {g}");
            } else if o == 0 {
                // more than half matrix: above halfway to even the darkest grain
                let darkest = grains.iter().map(|g| g.brightness).fold(f64::MAX, f64::min);
                assert!(v >= 0.5 * (180.0 + darkest) - 1.0, "matrix pixel {q} at {v}");
            }
        }
        // sand: every pixel shows some grain or the matrix; few edge-shared
        let sand = SoilConfig::sand(120.0, 60.0, 0.2);
        let (pos, grains) = place(&sand, (150, 150), 3);
        let c = Array2::from_shape_fn((pos.len(), 2), |(k, j)| pos[k][j]);
        let own = ownership((150, 150), &grains, c.view(), &vec![0.0; pos.len()]);
        let shared = own.iter().filter(|&&o| o < 0).count() as f64 / 22500.0;
        assert!(shared < 0.1, "edge-shared share {shared}");
        assert!(own.iter().all(|&o| o >= -1 && o <= grains.len() as i64));
    }

    /// A pixel takes each grain's brightness weighted by the part of the pixel
    /// it is seen over: two grains of the top layer over 1/8 and 5/8 of it, and
    /// a grain underneath seen through the remaining 2/8 (it also lies behind
    /// part of the first top grain, which hides that part).
    #[test]
    fn pixel_weights_are_visible_areas() {
        // nearly straight edges across pixel (10, 10): huge circles
        let big = |layer, brightness| Grain { outline: Outline::Circle { r: 1000.0 }, brightness, look: Look::default(), dyed: false, layer, reach: 1000.0, area: 0.0 };
        let grains = vec![big(0, 40.0), big(0, 200.0), big(1, 120.0)];
        // A: x < 9.625 (1/8); B: x > 9.875 (5/8); C (behind): x < 9.875 (3/8, 2/8 seen)
        let centres = ndarray::array![[9.625 - 1000.0, 10.0], [9.875 + 1000.0, 10.0], [9.875 - 1000.0, 10.0]];
        let mixes = pixel_mixes((20, 20), &grains, centres.view(), &[0.0; 3]);
        let mut mix = mixes[10 * 20 + 10].clone();
        mix.sort_by_key(|m| m.0);
        assert_eq!(mix, vec![(1, 0.125), (2, 0.625), (3, 0.25)]);
        let cfg = flat(SoilConfig { matrix_spread: 0.0, ..SoilConfig::sand(80.0, 60.0, 0.0) });
        let img = render(&cfg, (20, 20), &grains, centres.view(), &[0.0; 3], None, None, None, 0.0, 1).unwrap();
        assert_eq!(img[[10, 10]], (0.125f64 * 40.0 + 0.625 * 200.0 + 0.25 * 120.0).round() as u8);
    }

    /// Grain spread: a circle of diameter d is d^2/16 per axis; an aspect
    /// ratio of 0.5 adds a factor 1.25; a log-normal curve gives the area mean
    /// of d^2, x50^2 exp(2 sigma_ln^2).
    #[test]
    fn grain_sigma2_from_properties() {
        let round = SoilConfig { d_p: 6.0, ..SoilConfig::sand(100.0, 60.0, 0.2) };
        assert!((round.grain_sigma2().unwrap() - 36.0 / 16.0).abs() < 1e-12);
        let ell = SoilConfig { aspect_ratio: 0.5, ..round.clone() };
        assert!((ell.grain_sigma2().unwrap() - 36.0 / 16.0 * 1.25).abs() < 1e-12);
        let s_ln = (500.0f64 / 312.5).ln() / 1.2816;
        let dist = SoilConfig { psd: vec![[200.0, 10.0], [312.5, 50.0], [500.0, 90.0]], d_p: 10.0, ..round.clone() };
        let want = 100.0 * (2.0 * s_ln * s_ln).exp() / 16.0;
        let got = dist.grain_sigma2().unwrap();
        assert!((got / want - 1.0).abs() < 0.02, "{got} vs {want}");
    }

    /// Grains flat inside (no brightness variation), for tests of geometry.
    fn flat(c: SoilConfig) -> SoilConfig {
        SoilConfig { grain_tilt: 0.0, grain_dome: 0.0, grain_rim: 0.0, grain_texture: 0.0, ..c }
    }

    /// Brightness variation inside grains: with every setting 0 grains are
    /// flat; placement does not depend on it; a large grain
    /// shows the asked tilt, dome, rim and texture spread, keeps its mean, and
    /// its texture turns and moves with it.
    #[test]
    fn grain_brightness_variation() {
        // placement and grain brightness unchanged by the settings
        let plain = flat(SoilConfig::sand(120.0, 60.0, 0.2));
        let tex = SoilConfig { grain_tilt: 0.2, grain_dome: 0.2, grain_rim: 0.1, grain_texture: 0.1, ..plain.clone() };
        let (p0, g0) = place(&plain, (120, 120), 7);
        let (p1, g1) = place(&tex, (120, 120), 7);
        assert_eq!(p0, p1);
        assert!(g0.iter().zip(&g1).all(|(a, b)| a.brightness == b.brightness));
        assert!(g0.iter().all(|g| g.shade(&plain, 1.0, 2.0, None) == g.brightness));
        // tilt over a radius: Rayleigh of the asked median
        let (_, many) = place(&SoilConfig { grain_tilt: 0.2, ..SoilConfig::clay(180.0, 100.0, 0.4) }, (400, 400), 3);
        let mut mags: Vec<f64> = many.iter().map(|g| g.look.tilt[0].hypot(g.look.tilt[1])).collect();
        mags.sort_by(f64::total_cmp);
        let med = mags[mags.len() / 2];
        assert!((med - 0.2).abs() < 0.03, "median tilt {med} of {} grains", mags.len());

        // one large round grain, 100 grey, fixed tilt and dome
        let n = 121;
        let grain = |look: Look| Grain { outline: Outline::Circle { r: 50.0 }, brightness: 100.0, look, dyed: true, layer: 0,
                                         reach: 50.0, area: std::f64::consts::PI * 2500.0 };
        let cfg = SoilConfig { matrix_spread: 0.0, grain_tilt: 1.0, grain_dome: 1.0, ..SoilConfig::clay(0.0, 100.0, 0.2) };
        let look = Look { tilt: [0.2, -0.1], dome: 0.3, offset: [3.0, 7.0] };
        let c = ndarray::array![[60.0, 60.0]];
        let img = render(&cfg, (n, n), &[grain(look.clone())], c.view(), &[0.0], None, None, None, 0.0, 1).unwrap();
        // fit v = a + b x/R + c y/R + e (2/3 - rho) over the inside
        let (mut ata, mut atb) = (nalgebra::Matrix4::<f64>::zeros(), nalgebra::Vector4::<f64>::zeros());
        let mut sum = (0.0, 0.0);
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (x as f64 - 60.0, y as f64 - 60.0);
                let rho = dx.hypot(dy) / 50.0;
                if rho > 0.97 {
                    continue;
                }
                let row = nalgebra::Vector4::new(1.0, dx / 50.0, dy / 50.0, RHO_MEAN - rho);
                let v = img[[y, x]] as f64;
                ata += row * row.transpose();
                atb += row * v;
                sum = (sum.0 + v, sum.1 + 1.0);
            }
        }
        let f = ata.try_inverse().unwrap() * atb;
        assert!((f[0] - 100.0).abs() < 0.5 && (sum.0 / sum.1 - 100.0).abs() < 1.0, "mean {} {}", f[0], sum.0 / sum.1);
        assert!((f[1] - 20.0).abs() < 0.3 && (f[2] + 10.0).abs() < 0.3, "tilt {} {}", f[1], f[2]);
        assert!((f[3] - 30.0).abs() < 0.5, "dome {}", f[3]);
        // rim: darker by grain_rim at the edge than in the middle
        let rim = SoilConfig { grain_tilt: 0.0, grain_dome: 0.0, grain_rim: 0.2, ..cfg.clone() };
        let flat = grain(Look::default());
        let edge = flat.shade(&rim, 50.0, 0.0, None);
        let mid = flat.shade(&rim, 0.0, 0.0, None);
        assert!((mid - edge - 20.0).abs() < 1e-9 && (flat.shade(&rim, 46.0, 0.0, None) - (mid - 10.0)).abs() < 1.0);

        // texture: spread as asked over a grain much larger than its features
        let t = SoilConfig { grain_tilt: 0.0, grain_dome: 0.0, grain_texture: 0.1, texture_size: 0.05, ..cfg.clone() };
        let field = Pattern::new(t.texture_size, 9);
        let img = render(&t, (n, n), &[flat.clone()], c.view(), &[0.0], None, None, Some(&field), 0.0, 1).unwrap();
        let inside: Vec<f64> = (0..n * n)
            .filter(|q| ((q % n) as f64 - 60.0).hypot((q / n) as f64 - 60.0) < 45.0)
            .map(|q| img[[q / n, q % n]] as f64)
            .collect();
        let m = inside.iter().sum::<f64>() / inside.len() as f64;
        let sd = (inside.iter().map(|v| (v - m).powi(2)).sum::<f64>() / inside.len() as f64).sqrt();
        assert!((sd / 10.0 - 1.0).abs() < 0.3, "texture spread {sd} grey for 10");
        // moved by (7, -4) and turned a quarter: the same texture at the moved, turned pixels
        let tilted = Grain { look: Look { tilt: [0.15, 0.05], ..flat.look.clone() }, ..flat.clone() };
        let t2 = SoilConfig { grain_tilt: 1.0, ..t.clone() };
        let a = render(&t2, (n, n), &[tilted.clone()], c.view(), &[0.0], None, None, Some(&field), 0.0, 1).unwrap();
        let c2 = ndarray::array![[67.0, 56.0]];
        let b = render(&t2, (n, n), &[tilted], c2.view(), &[std::f64::consts::FRAC_PI_2], None, None, Some(&field), 0.0, 1).unwrap();
        for dy in -30i64..=30 {
            for dx in -30i64..=30 {
                // grain-frame point (dx, dy) sits at centre + R(turn) (dx, dy)
                let (x2, y2) = (67 - dy, 56 + dx);
                let (va, vb) = (a[[(60 + dy) as usize, (60 + dx) as usize]] as i32, b[[y2 as usize, x2 as usize]] as i32);
                assert!((va - vb).abs() <= 1, "({dx}, {dy}): {va} vs {vb}");
            }
        }
    }
}
