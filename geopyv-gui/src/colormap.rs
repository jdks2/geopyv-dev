use eframe::egui;

// ---------------------------------------------------------------------------
// Colormap type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColormapType {
    #[default]
    Viridis,
    Plasma,
    Inferno,
}

impl ColormapType {
    pub const ALL: &'static [ColormapType] = &[
        ColormapType::Viridis,
        ColormapType::Plasma,
        ColormapType::Inferno,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ColormapType::Viridis => "Viridis",
            ColormapType::Plasma => "Plasma",
            ColormapType::Inferno => "Inferno",
        }
    }
}

// ---------------------------------------------------------------------------
// 9-keyframe LUTs (t = 0, 0.125, 0.25, ..., 1.0)
// Values are (R, G, B) in [0, 255].
// ---------------------------------------------------------------------------

static VIRIDIS: [[u8; 3]; 9] = [
    [68,   1,  84],
    [71,  44, 122],
    [59,  82, 139],
    [44, 113, 142],
    [32, 144, 140],
    [35, 168, 132],
    [93, 201,  99],
    [161, 218, 62],
    [253, 231, 37],
];

static PLASMA: [[u8; 3]; 9] = [
    [ 13,   8, 135],
    [ 84,   2, 163],
    [139,  10, 165],
    [185,  50, 137],
    [219,  92, 104],
    [244, 136,  73],
    [253, 183,  45],
    [249, 230,  51],
    [240, 249,  33],
];

static INFERNO: [[u8; 3]; 9] = [
    [  0,   0,   4],
    [ 40,  11,  84],
    [101,  21, 110],
    [159,  42,  99],
    [212,  72,  66],
    [245, 125,  21],
    [252, 185,  20],
    [253, 236,  98],
    [252, 255, 164],
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Map `t ∈ [0, 1]` to an opaque colour via the chosen colormap.
pub fn sample(t: f32, cmap: ColormapType) -> egui::Color32 {
    let lut: &[[u8; 3]; 9] = match cmap {
        ColormapType::Viridis => &VIRIDIS,
        ColormapType::Plasma => &PLASMA,
        ColormapType::Inferno => &INFERNO,
    };

    let t = t.clamp(0.0, 1.0);
    let scaled = t * 8.0; // 9 keyframes → 8 segments
    let lo = (scaled as usize).min(7);
    let hi = lo + 1;
    let frac = scaled - lo as f32;

    let r = lerp_u8(lut[lo][0], lut[hi][0], frac);
    let g = lerp_u8(lut[lo][1], lut[hi][1], frac);
    let b = lerp_u8(lut[lo][2], lut[hi][2], frac);
    egui::Color32::from_rgb(r, g, b)
}

/// Map a scalar value to a colour given a [vmin, vmax] range.
pub fn map_value(v: f64, vmin: f64, vmax: f64, cmap: ColormapType) -> egui::Color32 {
    if vmax <= vmin {
        return sample(0.5, cmap);
    }
    let t = ((v - vmin) / (vmax - vmin)) as f32;
    sample(t, cmap)
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}
