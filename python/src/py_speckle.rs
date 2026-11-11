//! PyO3 wrapper for `geopyv_dev::speckle`.

use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use geopyv_dev::speckle::{GrainOverlap, GrainRotation, GrainShape, Progression, ScaleType, ShearBandOption, Speckle, WarpMode};


use crate::Error;

// ---------------------------------------------------------------------------
// Dict-extraction helpers
// ---------------------------------------------------------------------------

fn req<'py, T: pyo3::FromPyObject<'py>>(
    d: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<T> {
    d.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing key '{key}'")))?
        .extract::<T>()
}

fn opt<'py, T: pyo3::FromPyObject<'py>>(
    d: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<T>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract::<T>()?)),
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// PySpeckle
// ---------------------------------------------------------------------------


/// A `soil_properties=` argument (Subset, Mesh, Sequence): the dict parsed as
/// `parse_soil` does, image-only keys ignored; `None` stays `None`.
pub(crate) fn soil_properties_arg(d: Option<&Bound<'_, PyDict>>) -> PyResult<Option<geopyv_dev::soil::SoilConfig>> {
    let Some(d) = d else { return Ok(None) };
    let kind: String = d
        .get_item("soil")?
        .ok_or_else(|| PyValueError::new_err("soil_properties needs 'soil': 'clay' or 'sand'"))?
        .extract()?;
    Ok(Some(parse_soil(d, &kind, false)?))
}

/// The `soil_properties` getter's dict (the grain and brightness settings).
pub(crate) fn soil_properties_dict<'py>(py: Python<'py>, c: &geopyv_dev::soil::SoilConfig) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("soil", match c.kind { geopyv_dev::soil::SoilKind::Clay => "clay", geopyv_dev::soil::SoilKind::Sand => "sand" })?;
    d.set_item("grains", if c.psd.is_empty() && c.aspect_ratio_sd == 0.0 { "uniform" } else { "distribution" })?;
    if !c.psd.is_empty() {
        d.set_item("psd", c.psd.iter().map(|r| r.to_vec()).collect::<Vec<_>>())?;
    }
    for (k, v) in [("d_p", c.d_p), ("aspect_ratio", c.aspect_ratio), ("aspect_ratio_sd", c.aspect_ratio_sd),
                   ("matrix_brightness", c.matrix_brightness), ("matrix_spread", c.matrix_spread),
                   ("seed_brightness", c.seed_brightness), ("seed_spread", c.seed_spread), ("asr", c.asr)] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// Soil settings from a dict (`gp.Speckle`'s soil `speckle_cfg`, or a subset's
/// `soil_properties`), validated. `image_keys`: read the image-generation keys
/// (`depth_layers`, `pattern_size`, `blur`, and the brightness inside grains:
/// `grain_tilt`, `grain_dome`, `grain_rim`, `grain_texture`, `texture_size`); otherwise they are ignored, as is
/// any other key not about the soil.
pub(crate) fn parse_soil(cfg: &Bound<'_, PyDict>, kind: &str, image_keys: bool) -> PyResult<geopyv_dev::soil::SoilConfig> {
    for k in ["shape", "cluster_n", "rotation", "overlap", "coverage", "grain_brightness", "grain_spread", "pattern_strength", "depth_dimming", "grain", "grain_size", "grain_size_spread", "aspect", "roundness"] {
        if cfg.contains(k)? {
            return Err(PyValueError::new_err(format!(
                "'{k}' does not apply to soil images (brightness: matrix_brightness / matrix_spread, \
                 seed_brightness / seed_spread, asr; grains: psd, d_p, aspect_ratio / aspect_ratio_sd, \
                 outlines)")));
        }
    }
    for k in ["sphericity", "sphericity_sd"] {
        if cfg.contains(k)? {
            return Err(PyValueError::new_err(format!(
                "'{k}' is no longer a soil setting: a QicPic sphericity is set mostly by fine roughness and \
                 pixel steps, not by the outline a camera sees. Give the grains' aspect_ratio / aspect_ratio_sd, \
                 and optionally outlines=<QicPic particle report .docx> for the sand's own outline statistics")));
        }
    }
    let m: f64 = req(cfg, "matrix_brightness")?;
    let sb: f64 = req(cfg, "seed_brightness")?;
    let asr: f64 = req(cfg, "asr")?;
    let mut c = match kind {
        "clay" => geopyv_dev::soil::SoilConfig::clay(m, sb, asr),
        "sand" => geopyv_dev::soil::SoilConfig::sand(m, sb, asr),
        other => return Err(PyValueError::new_err(format!("soil must be 'clay' or 'sand', got {other:?}"))),
    };
    if let Some(v) = opt(cfg, "matrix_spread")? { c.matrix_spread = v; }
    if let Some(v) = opt(cfg, "seed_spread")? { c.seed_spread = v; }
    let grains = opt::<String>(cfg, "grains")?.unwrap_or_else(|| "uniform".into());
    match grains.as_str() {
        "uniform" => {
            for k in ["psd", "aspect_ratio_sd"] {
                if cfg.contains(k)? {
                    return Err(PyValueError::new_err(format!("'{k}' needs grains=\"distribution\"")));
                }
            }
        }
        "distribution" => {
            if !cfg.contains("psd")? {
                return Err(PyValueError::new_err("grains=\"distribution\" needs psd"));
            }
        }
        other => return Err(PyValueError::new_err(format!(
            "grains must be 'uniform' or 'distribution', got {other:?}"))),
    }
    if let Some(rows) = opt::<Vec<Vec<f64>>>(cfg, "psd")? {
        c.psd = rows
            .into_iter()
            .map(|r| <[f64; 2]>::try_from(r.as_slice()).map_err(|_| PyValueError::new_err("psd must be an (n, 2) array of size, % passing")))
            .collect::<PyResult<_>>()?;
    }
    if let Some(v) = opt(cfg, "d_p")? { c.d_p = v; }
    // a QicPic particle report: the sand's own outline statistics, and its
    // aspect ratio where not given
    if let Some(path) = cfg.get_item("outlines")?.filter(|v| !v.is_none()) {
        let st = cfg.py().import_bound("geopyv_dev._qicpic")?.call_method1("calibrate", (path,))?;
        let st = st.downcast::<PyDict>()?;
        let get = |k: &str| -> PyResult<Bound<'_, pyo3::PyAny>> {
            st.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("outline statistics lack {k:?}")))
        };
        c.outlines = Some(geopyv_dev::soil::OutlineStats {
            irregularity: get("irregularity")?.extract()?,
            irregularity_sd: get("irregularity_sd")?.extract()?,
            profile: get("profile")?.extract()?,
            profile_sd: get("profile_sd")?.extract()?,
            dent: get("dent")?.extract()?,
        });
        c.aspect_ratio = get("aspect_ratio")?.extract()?;
        if grains == "distribution" {
            c.aspect_ratio_sd = get("aspect_ratio_sd")?.extract()?;
        }
    }
    if let Some(v) = opt(cfg, "aspect_ratio")? { c.aspect_ratio = v; }
    if let Some(v) = opt(cfg, "aspect_ratio_sd")? { c.aspect_ratio_sd = v; }
    if image_keys {
        if let Some(v) = opt(cfg, "depth_layers")? { c.depth_layers = v; }
        if let Some(v) = opt(cfg, "pattern_size")? { c.pattern_size = v; }
        if let Some(v) = opt(cfg, "blur")? { c.blur = v; }
        if let Some(v) = opt(cfg, "grain_tilt")? { c.grain_tilt = v; }
        if let Some(v) = opt(cfg, "grain_dome")? { c.grain_dome = v; }
        if let Some(v) = opt(cfg, "grain_rim")? { c.grain_rim = v; }
        if let Some(v) = opt(cfg, "grain_texture")? { c.grain_texture = v; }
        if let Some(v) = opt(cfg, "texture_size")? { c.texture_size = v; }
    }
    c.validate().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(c)
}

#[pyclass(name = "Speckle")]
pub struct PySpeckle {
    pub(crate) inner: Arc<Speckle>,
}

#[pymethods]
impl PySpeckle {
    /// Create a synthetic speckle image generator.
    ///
    /// Parameters
    /// ----------
    /// image_cfg : dict
    ///     ``image_dir`` (str), ``name`` (str), ``image_size`` ((int,int), default (1001,1001)),
    ///     ``file_format`` (str, default ".jpg").
    /// speckle_cfg : dict
    ///     ``speckle_size`` (float, px blob radius), ``speckle_number`` (int),
    ///     ``speckle_limit`` (bool, optional, default False) -- when True, each
    ///     speckle's Gaussian is rendered only within ``ceil(1.5 * speckle_size)``
    ///     px of its centre instead of the default fixed ±100px window; safe (not
    ///     lossy) at typical speckle sizes, and substantially faster to render.
    ///     ``shape`` (str, optional) -- grain shape: ``"round"`` (default),
    ///     ``"ellipse"`` (needs ``aspect``, long/short axis ratio >= 1; random
    ///     orientation, area of the round grain) or ``"cluster"`` (``cluster_n``
    ///     overlapping round blobs at random offsets, default 4). Every shape has
    ///     the total brightness of one round speckle; grains never deform.
    ///     ``rotation`` (str, optional) -- ``"material"`` (default: a non-round
    ///     grain turns with the material at its centre) or ``"none"``.
    ///     ``overlap`` (str, optional) -- ``"add"`` (default: overlapping grains
    ///     add, clipped at 255) or ``"opaque"`` (each grain has a random depth;
    ///     a pixel shows the topmost grain covering it).
    ///
    ///     **Soil images** (``soil`` given; the speckle and grain keys above are then
    ///     not used and ``speckle_size`` / ``speckle_number`` may be left out):
    ///
    ///     ``soil`` (str) -- ``"clay"`` (dyed sand grains on a clay matrix whose own
    ///     fine pattern deforms with the clay) or ``"sand"`` (packed sand: layers of
    ///     grains, a share ``asr`` of them dyed, the nearest grain seen).
    ///     ``matrix_brightness`` (float, required) -- average brightness of the matrix
    ///     (clay: the clay; sand: the natural sand grains), grey levels 0-255.
    ///     ``matrix_spread`` (float, default 0.05 clay / 0.15 sand) -- clay: strength of the
    ///     clay's own pattern; sand: grain-to-grain spread of the natural grains; a fraction
    ///     of ``matrix_brightness``.
    ///     ``seed_brightness`` (float, required) -- average brightness of the dyed grains.
    ///     ``seed_spread`` (float, default 0.09) -- grain-to-grain spread of the dyed grains,
    ///     a fraction of ``seed_brightness``.
    ///     ``asr`` (float, required) -- artificial seeding ratio: the share of the image
    ///     showing dye, 0 (none) to 1 (all). Clay: share covered by dyed grains (below
    ///     0.55); sand: chance that a grain is dyed.
    ///     Grains (sand; or the seeding sand in clay) are described as a grain
    ///     characterisation (e.g. dynamic image analysis) gives them:
    ///     ``grains`` (str, default ``"uniform"``) -- ``"uniform"``: every grain the
    ///     same size (``d_p``) and aspect ratio (``aspect_ratio``);
    ///     ``"distribution"``: sizes from ``psd`` and aspect ratios spread by
    ///     ``aspect_ratio_sd``.
    ///     ``psd`` (array ``(n, 2)``, ``"distribution"`` only, required) -- cumulative size curve: particle size
    ///     (any length unit; EQPC, the diameter of a circle with the grain's projected
    ///     area) and % by mass passing, sizes increasing. A full curve, or only a few
    ///     points such as x10, x50 and x90 (the curve is extended log-normally past its
    ///     ends).
    ///     ``d_p`` (float, default 6) -- the median size x50 in pixels (particle
    ///     diameter / pixel). Grains smaller than a pixel cannot be drawn; deeper
    ///     layers (sand) or the matrix (clay) stand in for them.
    ///     ``aspect_ratio`` (float, default 1) -- aspect ratio (narrowest / widest
    ///     width, <= 1); with ``"distribution"``, the mean over the grains.
    ///     ``aspect_ratio_sd`` (float, ``"distribution"`` only, default 0) -- its
    ///     standard deviation over the grains; drawn per grain, independently of size.
    ///     ``outlines`` (str, optional) -- path to a QicPic particle report (.docx):
    ///     the outline statistics of that sand's single grains (two grains
    ///     overlapping at a pinch are left out), and its aspect ratio mean (and,
    ///     with ``"distribution"``, standard deviation) unless ``aspect_ratio`` /
    ///     ``aspect_ratio_sd`` are given. Without it, the statistics of Leighton Buzzard sand are used.
    ///     An outline is an ellipse (the aspect ratio) times irregularity waves
    ///     (3 to 24 bulges) drawn from the statistics, redrawn while it has a
    ///     deeper dent than real grains. Grains of aspect ratio 1 with no spread
    ///     and no ``outlines`` are circles. Sphericity is not a setting: a QicPic
    ///     sphericity is set mostly by fine roughness and pixel steps (a traced
    ///     perfect circle scores 0.948), not by the outline a camera sees.
    ///     ``depth_layers`` (int, sand, default 6) -- number of grain layers; deeper
    ///     grains show through the gaps between nearer ones.
    ///     ``pattern_size`` (float, clay, default 10 px) -- feature size of the clay's own
    ///     pattern.
    ///     ``blur`` (float, default 0) -- lens blur width (px) applied to the whole image;
    ///     0 for none. Blur and noise simulate experimental error and are both optional.
    ///     Brightness inside each grain, fixed to the grain (moves and turns with it,
    ///     keeps its mean brightness); fractions of the grain's brightness, 0 turns
    ///     each off. Defaults calibrated so that real Leighton Buzzard grains'
    ///     measurements come out after blur 0.8 px and noise 2 grey levels:
    ///     ``grain_tilt`` (float, default 0.14) -- brighter towards one side: median change over
    ///     one equal-area radius, random direction (Rayleigh spread);
    ///     ``grain_dome`` (float, default 0.17) -- spread of the change from edge to centre
    ///     (normal, mean 0: some grains brighter in the middle, some darker);
    ///     ``grain_rim`` (float, default 0) -- darker by this at the edge (half that at 92 % of
    ///     the way out). The lens blur already darkens edges, so leave 0 unless a
    ///     measured rim is wider;
    ///     ``grain_texture`` (float, default 0.085) -- spread of a patchy texture, of
    ///     feature size ``texture_size`` (float, default 0.10) grain diameters.
    ///     Grains never cross (they may touch), are rigid and turn with the material.
    ///     Noise from ``noise_cfg`` (intensity 0.01 = 2 grey levels).
    ///     ``coverage`` (float, optional) -- place grains so that none overlap
    ///     (no pixel inside two grains' 10 %-of-peak outlines), adding them at
    ///     random until this share of the image is covered; ``speckle_number``
    ///     is then not used.
    /// progression : str
    ///     ``"deformation"`` — warp progresses; noise constant at final values.
    ///     ``"noise"``       — noise progresses; deformation constant at full comp.
    /// deformation_cfg : dict
    ///     ``comp`` (list[float], 12 elements) — final warp vector (applied at mult=1).
    ///     ``origin`` ((float,float), optional) — warp reference origin; defaults to image centre.
    ///     ``mode`` (str, optional) — ``None`` / ``"rotation"`` / ``"SB"`` / ``"circular"``.
    ///     ``option`` (str, optional) — ``"sin"`` / ``"lin"`` / ``"quad"`` / ``"smooth"`` when mode="SB".
    ///     ``width`` (float, optional) — shear-band full width in px (default 100.0).
    ///     ``tau`` (float, required when option="smooth") — corner transition length in px;
    ///         a box (``"lin"``-shaped) band whose corners are rounded by a logistic
    ///         transition of length ``tau``. ``tau`` comparable to ``width/2`` erodes the
    ///         flat plateau at ``du/dy = strain`` into a smoothed triangular bump.
    ///     ``angle`` (float, required when mode="rotation" or mode="circular") — rotation
    ///         angle in radians at multiplier 1.
    ///     ``r1``, ``r2`` (float, required when mode="circular") — inner (rigid core) and
    ///         outer (stationary far field) radii in px of the circular shear band around
    ///         ``origin``; a disc of radius ``r1`` rotates rigidly by ``angle``, an annulus
    ///         from ``r1`` to ``r2`` linearly tapers that rotation to zero, and the field is
    ///         exactly stationary beyond ``r2``.
    ///     ``tau`` (float, optional, only when mode="circular") — corner transition
    ///         length in px, rounding the annulus's two corners (at ``r1`` and ``r2``)
    ///         the same way ``option="smooth"`` rounds a shear band's corners; ``r1``
    ///         (core rotation ``angle``) and ``r2`` (stationary far field) stay exact.
    ///         Requires ``0 < tau <= (r2-r1)/2``. Omit for the hard-edged annulus.
    /// noise_cfg : (float, float)
    ///     ``(noise_pos, noise_int)`` — final Gaussian noise std-devs.
    /// scale_cfg : dict
    ///     ``scale`` (str) — ``"lin"`` or ``"log"``.
    ///     ``n`` (int) — total number of images (image 0 is always the reference).
    ///     ``min`` (float, optional) — minimum multiplier for log scale (default 1e-5).
    #[new]
    #[pyo3(signature = (image_cfg, speckle_cfg, progression, deformation_cfg, noise_cfg, scale_cfg))]
    fn new(
        image_cfg: &Bound<'_, PyDict>,
        speckle_cfg: &Bound<'_, PyDict>,
        progression: &str,
        deformation_cfg: &Bound<'_, PyDict>,
        noise_cfg: (f64, f64),
        scale_cfg: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        // image_cfg
        let image_dir: String = req(image_cfg, "image_dir")?;
        let name: String = req(image_cfg, "name")?;
        let image_size: (usize, usize) =
            opt(image_cfg, "image_size")?.unwrap_or((1001, 1001));
        let file_format_raw: String =
            opt(image_cfg, "file_format")?.unwrap_or_else(|| ".jpg".into());
        let file_format = if file_format_raw.starts_with('.') {
            file_format_raw
        } else {
            format!(".{file_format_raw}")
        };

        // speckle_cfg
        let soil_kind: Option<String> = opt(speckle_cfg, "soil")?;
        let (speckle_size, speckle_number): (f64, usize) = if soil_kind.is_some() {
            (opt(speckle_cfg, "speckle_size")?.unwrap_or(10.0), opt(speckle_cfg, "speckle_number")?.unwrap_or(1))
        } else {
            (req(speckle_cfg, "speckle_size")?, req(speckle_cfg, "speckle_number")?)
        };
        let soil = match soil_kind.as_deref() {
            None => None,
            Some(kind) => Some(parse_soil(speckle_cfg, kind, true)?),
        };
        let speckle_limit: bool = opt(speckle_cfg, "speckle_limit")?.unwrap_or(false);
        let shape_str: String = if soil.is_some() { "round".into() } else { opt(speckle_cfg, "shape")?.unwrap_or_else(|| "round".into()) };
        let grain_shape = match shape_str.as_str() {
            "round" => GrainShape::Round,
            "ellipse" => GrainShape::Ellipse { aspect: req(speckle_cfg, "aspect")? },
            "cluster" => GrainShape::Cluster { n: opt(speckle_cfg, "cluster_n")?.unwrap_or(4) },
            other => return Err(PyValueError::new_err(format!(
                "shape must be 'round', 'ellipse' or 'cluster', got {other:?}"))),
        };
        let coverage: Option<f64> = opt(speckle_cfg, "coverage")?;
        let overlap = match opt::<String>(speckle_cfg, "overlap")?.as_deref().unwrap_or("add") {
            "add" => GrainOverlap::Add,
            "opaque" => GrainOverlap::Opaque,
            other => return Err(PyValueError::new_err(format!(
                "overlap must be 'add' or 'opaque', got {other:?}"))),
        };
        let rotation_str: String = opt(speckle_cfg, "rotation")?.unwrap_or_else(|| "material".into());
        let grain_rotation = match rotation_str.as_str() {
            "material" => GrainRotation::Material,
            "none" => GrainRotation::None,
            other => return Err(PyValueError::new_err(format!(
                "rotation must be 'material' or 'none', got {other:?}"))),
        };

        // progression
        let prog = match progression {
            "deformation" => Progression::Deformation,
            "noise" => Progression::Noise,
            other => return Err(PyValueError::new_err(format!(
                "progression must be 'deformation' or 'noise', got '{other}'"
            ))),
        };

        // deformation_cfg
        let comp_vec: Vec<f64> = opt(deformation_cfg, "comp")?.unwrap_or_else(|| vec![0.0; 12]);
        let mut comp: [f64; 12] = comp_vec.try_into().map_err(|v: Vec<f64>| {
            PyValueError::new_err(format!(
                "comp must have exactly 12 elements, got {}",
                v.len()
            ))
        })?;
        let origin: [f64; 2] = match opt::<(f64, f64)>(deformation_cfg, "origin")? {
            Some((ox, oy)) => [ox, oy],
            None => [image_size.0 as f64 / 2.0, image_size.1 as f64 / 2.0],
        };
        let mode_str: Option<String> = opt(deformation_cfg, "mode")?;
        let warp_mode = match mode_str.as_deref() {
            None | Some("") | Some("none") | Some("None") => WarpMode::None,
            Some(s) if s.eq_ignore_ascii_case("rotation") => {
                // Store angle in comp[0]; other comp entries are ignored for rotation.
                let angle: f64 = req(deformation_cfg, "angle")?;
                comp[0] = angle;
                WarpMode::Rotation
            }
            Some(s) if s.eq_ignore_ascii_case("sb") => {
                let option_str: String =
                    opt(deformation_cfg, "option")?.unwrap_or_else(|| "sin".into());
                let sb_opt = match option_str.as_str() {
                    "sin" => ShearBandOption::Sin,
                    "lin" => ShearBandOption::Lin,
                    "quad" => ShearBandOption::Quad,
                    "smooth" => {
                        let tau: f64 = req(deformation_cfg, "tau")?;
                        ShearBandOption::Smooth { tau }
                    }
                    other => return Err(PyValueError::new_err(format!(
                        "option must be 'sin', 'lin', 'quad', or 'smooth', got '{other}'"
                    ))),
                };
                let width: f64 = opt(deformation_cfg, "width")?.unwrap_or(100.0);
                WarpMode::ShearBand { option: sb_opt, width }
            }
            Some(s) if s.eq_ignore_ascii_case("circular") => {
                // Same convention as "rotation": angle in comp[0], other
                // comp entries unused. origin doubles as the band centre.
                let angle: f64 = req(deformation_cfg, "angle")?;
                comp[0] = angle;
                let r1: f64 = req(deformation_cfg, "r1")?;
                let r2: f64 = req(deformation_cfg, "r2")?;
                let tau: Option<f64> = opt(deformation_cfg, "tau")?;
                WarpMode::CircularShear { r1, r2, tau }
            }
            Some(other) => return Err(PyValueError::new_err(format!(
                "mode must be None, 'rotation', 'SB', or 'circular', got '{other}'"
            ))),
        };

        // noise_cfg
        let (noise_pos, noise_int) = noise_cfg;

        // scale_cfg
        let scale_str: String = req(scale_cfg, "scale")?;
        let image_no: usize = req(scale_cfg, "n")?;
        let scale = match scale_str.as_str() {
            "lin" => ScaleType::Lin,
            "log" => {
                let min: f64 = opt(scale_cfg, "min")?.unwrap_or(1e-5);
                ScaleType::Log { min }
            }
            other => return Err(PyValueError::new_err(format!(
                "scale must be 'lin' or 'log', got '{other}'"
            ))),
        };

        let inner = Speckle::new(
            image_dir,
            name,
            file_format,
            image_size,
            speckle_size,
            speckle_number,
            speckle_limit,
            prog,
            comp,
            origin,
            noise_pos,
            noise_int,
            warp_mode,
            image_no,
            scale,
        )
        .and_then(|sp| sp.with_grains(grain_shape, grain_rotation))
        .map(|sp| sp.with_overlap(overlap))
        .and_then(|sp| match (coverage, soil.is_some()) { (Some(c), false) => sp.with_no_overlap(c), _ => Ok(sp) })
        .and_then(|sp| match soil { Some(c) => sp.with_soil(c), None => Ok(sp) })
        .map_err(Error::from)?;

        Ok(PySpeckle { inner: Arc::new(inner) })
    }

    /// Generate all images and write them to ``image_dir``.
    ///
    /// Parameters
    /// ----------
    /// seed : int, optional
    ///     RNG seed for reproducibility. Default 42.
    #[pyo3(signature = (seed=None))]
    fn solve(&mut self, seed: Option<u64>) -> PyResult<()> {
        Arc::get_mut(&mut self.inner)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "cannot call solve(): Speckle is shared with a Validation instance"
            ))?
            .solve(seed.unwrap_or(42))
            .map_err(Error::from)?;
        Ok(())
    }

    /// Reference speckle positions ``(N, 2)`` — available after :meth:`solve`.
    #[getter]
    fn speckle_positions<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pos = self.inner.speckle_positions.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("call solve() first")
        })?;
        Ok(pos.clone().into_pyarray_bound(py))
    }

    /// Applied (ground-truth) warp at ``coordinates`` in image ``i``, relative
    /// to the reference image 0. Does not need :meth:`solve`.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///     Image index (0 gives the zero warp).
    /// coordinates : numpy.ndarray
    ///     ``(N, 2)`` reference-image coordinates ``[x, y]``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(N, 12)`` warps ``[u, v, du/dx, dv/dx, du/dy, dv/dy,
    ///     d2u/dx2, d2v/dx2, d2u/dxdy, d2v/dxdy, d2u/dy2, d2v/dy2]``.
    fn warp<'py>(
        &self,
        py: Python<'py>,
        i: usize,
        coordinates: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if i >= self.inner.image_no {
            return Err(PyValueError::new_err(format!(
                "image index {i} out of range (image_no = {})", self.inner.image_no
            )));
        }
        let w = self.inner.warp(i, coordinates.as_array()).map_err(Error::from)?;
        Ok(w.into_pyarray_bound(py))
    }

    /// The grains of soil image ``i`` (after :meth:`solve`), rebuilt from the
    /// solve's seed.
    ///
    /// Parameters
    /// ----------
    /// i : int, default 0
    ///     Image index (0 is the reference).
    ///
    /// Returns
    /// -------
    /// (dict, numpy.ndarray)
    ///     A table with one entry per grain, as arrays: ``x``, ``y`` (centre in
    ///     this image, px), ``size`` (EQPC: diameter of a circle of the same
    ///     area, px), ``aspect_ratio`` (narrowest / widest width),
    ///     ``orientation`` (long axis from +x towards +y, rad, in [0, pi)),
    ///     ``brightness``, ``dyed``
    ///     (bool), ``layer`` (0 nearest; always 0 in clay). And a
    ///     ``(height, width)`` int map of which grain each pixel shows: the
    ///     grain's row in the table + 1 where that grain covers more than half
    ///     of the pixel, 0 where the matrix does, -1 where nothing does (a
    ///     pixel shared along an edge).
    #[pyo3(signature = (i=0))]
    fn grains<'py>(&self, py: Python<'py>, i: usize) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyArray2<i64>>)> {
        let (info, owner) = self.inner.grains(i).map_err(Error::from)?;
        let d = PyDict::new_bound(py);
        let col = |f: fn(&geopyv_dev::soil::GrainInfo) -> f64| info.iter().map(f).collect::<Vec<f64>>().into_pyarray_bound(py);
        d.set_item("x", col(|g| g.x))?;
        d.set_item("y", col(|g| g.y))?;
        d.set_item("size", col(|g| g.size))?;
        d.set_item("aspect_ratio", col(|g| g.aspect_ratio))?;
        d.set_item("orientation", col(|g| g.orientation))?;
        d.set_item("brightness", col(|g| g.brightness))?;
        d.set_item("dyed", info.iter().map(|g| g.dyed).collect::<Vec<bool>>().into_pyarray_bound(py))?;
        d.set_item("layer", info.iter().map(|g| g.layer as i64).collect::<Vec<i64>>().into_pyarray_bound(py))?;
        Ok((d, owner.into_pyarray_bound(py)))
    }

    #[getter]
    fn solved(&self) -> bool {
        self.inner.solved
    }

    fn __repr__(&self) -> String {
        format!(
            "Speckle(image_no={}, speckle_number={}, speckle_size={}, solved={})",
            self.inner.image_no,
            self.inner.speckle_number,
            self.inner.speckle_size,
            if self.inner.solved { "True" } else { "False" },
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpeckle>()?;
    Ok(())
}
