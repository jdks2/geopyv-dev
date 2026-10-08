//! Shared viewer overlays: colorbars, filled contour bands and number
//! formatting. Presentation only — the values drawn come from the core.

use eframe::egui;

use crate::colormap::{self, ColormapType};
use crate::draw::ImageCoord;

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

pub fn format_sci(v: f64) -> String {
    if v.abs() < 1e-12 {
        return "0".to_string();
    }
    if v.abs() >= 1000.0 || (v.abs() < 0.001 && v.abs() > 0.0) {
        format!("{:.2e}", v)
    } else {
        format!("{:.4}", v)
    }
}

// ---------------------------------------------------------------------------
// Coordinate transform without an image underlay
// ---------------------------------------------------------------------------

/// Fit image-space `points` into `viewport` (for drawing without an image).
pub fn coord_from_points(points: impl Iterator<Item = [f64; 2]>, viewport: egui::Rect) -> ImageCoord {
    let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    for [x, y] in points {
        let (x, y) = (x as f32, y as f32);
        if !(x.is_finite() && y.is_finite()) {
            continue;
        }
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if !min_x.is_finite() {
        return ImageCoord {
            canvas_center: viewport.center(),
            offset: egui::Vec2::ZERO,
            zoom: 1.0,
            img_size: egui::Vec2::ZERO,
        };
    }
    let w = (max_x - min_x).max(1.0);
    let h = (max_y - min_y).max(1.0);
    let zoom = (viewport.width() / w).min(viewport.height() / h) * 0.85;
    let cx = (min_x + max_x) * 0.5;
    let cy = (min_y + max_y) * 0.5;
    ImageCoord {
        canvas_center: viewport.center(),
        offset: egui::vec2(-cx * zoom, -cy * zoom),
        zoom,
        img_size: egui::Vec2::ZERO,
    }
}

// ---------------------------------------------------------------------------
// Contour levels
// ---------------------------------------------------------------------------

/// Default filled-contour levels for data spanning `[zmin, zmax]`, exactly
/// as matplotlib's `tricontourf(..., N)` picks them (`N = 7` when not
/// given): `ContourSet._autolev` over `MaxNLocator(N + 1, min_n_ticks=1)`.
/// Ported line for line (matplotlib 3.10) so the GUI's bands match
/// `Field.contour()`'s.
pub fn nice_levels(zmin: f64, zmax: f64, n: usize) -> Vec<f64> {
    let lev = max_n_locator_ticks(zmin, zmax, n + 1);
    // ContourSet._autolev: trim excess levels the locator supplied.
    let i0 = lev.iter().rposition(|&l| l < zmin).unwrap_or(0);
    let i1 = lev.iter().position(|&l| l > zmax).map_or(lev.len(), |i| i + 1);
    if i1 < i0 + 3 {
        lev
    } else {
        lev[i0..i1].to_vec()
    }
}

/// `MaxNLocator(nbins, min_n_ticks=1).tick_values(vmin, vmax)` with the
/// default steps `[1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]`.
fn max_n_locator_ticks(vmin: f64, vmax: f64, nbins: usize) -> Vec<f64> {
    const STEPS: [f64; 10] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
    let (vmin, vmax) = nonsingular(vmin, vmax, 1e-13, 1e-14);
    let nbins = nbins.max(1) as f64;

    // scale_range(vmin, vmax, nbins)
    let dv = (vmax - vmin).abs();
    let meanv = (vmax + vmin) / 2.0;
    let offset = if meanv.abs() / dv < 100.0 {
        0.0
    } else {
        10f64.powf(py_floordiv(meanv.abs().log10(), 1.0)).copysign(meanv)
    };
    let scale = 10f64.powf(py_floordiv((dv / nbins).log10(), 1.0));

    let (_vmin, _vmax) = (vmin - offset, vmax - offset);
    // _staircase: 0.1 * steps[:-1], steps, 10 * steps[1].
    let steps: Vec<f64> = STEPS[..9]
        .iter()
        .map(|s| 0.1 * s)
        .chain(STEPS.iter().copied())
        .chain(std::iter::once(10.0 * STEPS[1]))
        .map(|s| s * scale)
        .collect();
    let raw_step = (_vmax - _vmin) / nbins;
    let istep = steps.iter().position(|&s| s >= raw_step).unwrap_or(steps.len() - 1);

    let mut ticks = Vec::new();
    for &step in steps[..=istep].iter().rev() {
        let best_vmin = py_floordiv(_vmin, step) * step;
        let low = edge_le(_vmin - best_vmin, step, offset);
        let high = edge_ge(_vmax - best_vmin, step, offset);
        ticks = (0..=((high - low).round() as i64).max(0))
            .map(|k| (low + k as f64) * step + best_vmin)
            .collect();
        if ticks.iter().any(|&t| t <= _vmax && t >= _vmin) {
            break;
        }
    }
    ticks.into_iter().map(|t| t + offset).collect()
}

/// `matplotlib.transforms.nonsingular`.
fn nonsingular(vmin: f64, vmax: f64, expander: f64, tiny: f64) -> (f64, f64) {
    if !(vmin.is_finite() && vmax.is_finite()) {
        return (-expander, expander);
    }
    let (mut vmin, mut vmax) = (vmin.min(vmax), vmin.max(vmax));
    let maxabs = vmin.abs().max(vmax.abs());
    if maxabs < (1e6 / tiny) * f64::MIN_POSITIVE {
        return (-expander, expander);
    }
    if vmax - vmin <= maxabs * tiny {
        if vmax == 0.0 && vmin == 0.0 {
            return (-expander, expander);
        }
        vmin -= expander * vmin.abs();
        vmax += expander * vmax.abs();
    }
    (vmin, vmax)
}

/// Python's float `divmod(x, y)`.
fn py_divmod(x: f64, y: f64) -> (f64, f64) {
    let mut m = x % y;
    let mut div = (x - m) / y;
    if m != 0.0 {
        if (y < 0.0) != (m < 0.0) {
            m += y;
            div -= 1.0;
        }
    } else {
        m = 0.0f64.copysign(y);
    }
    let floordiv = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        0.0f64.copysign(x / y)
    };
    (floordiv, m)
}

fn py_floordiv(x: f64, y: f64) -> f64 {
    py_divmod(x, y).0
}

/// `_Edge_integer.closeto` tolerance.
fn edge_tol(step: f64, offset: f64) -> f64 {
    let offset = offset.abs();
    if offset > 0.0 {
        let digits = (offset / step).log10();
        1e-10f64.max(10f64.powf(digits - 12.0)).min(0.4999)
    } else {
        1e-10
    }
}

/// `_Edge_integer.le`: largest n with n * step <= x.
fn edge_le(x: f64, step: f64, offset: f64) -> f64 {
    let (d, m) = py_divmod(x, step);
    if (m / step - 1.0).abs() < edge_tol(step, offset) { d + 1.0 } else { d }
}

/// `_Edge_integer.ge`: smallest n with n * step >= x.
fn edge_ge(x: f64, step: f64, offset: f64) -> f64 {
    let (d, m) = py_divmod(x, step);
    if (m / step).abs() < edge_tol(step, offset) { d } else { d + 1.0 }
}

// ---------------------------------------------------------------------------
// Filled contour geometry (tricontourf equivalent)
// ---------------------------------------------------------------------------

/// Triangulated iso-bands in image space: every input triangle clipped to
/// each `[levels[k], levels[k+1]]` band it crosses.
pub struct ContourBands {
    pub levels: Vec<f64>,
    /// `(vertices, band index)` per output triangle.
    pub pieces: Vec<([egui::Pos2; 3], usize)>,
}

impl ContourBands {
    /// `points[i]` / `values[i]` per particle; `triangles` index into them.
    /// Values outside `[levels[0], levels[last]]` are left unfilled, like
    /// `tricontourf` without `extend`.
    pub fn build(points: &[[f64; 2]], values: &[f64], triangles: &[[usize; 3]], levels: Vec<f64>) -> Self {
        let mut pieces = Vec::new();
        let n_bands = levels.len().saturating_sub(1);
        for t in triangles {
            let tri: Vec<(egui::Pos2, f64)> = t
                .iter()
                .map(|&i| (egui::pos2(points[i][0] as f32, points[i][1] as f32), values[i]))
                .collect();
            if tri.iter().any(|(p, f)| !(f.is_finite() && p.x.is_finite() && p.y.is_finite())) {
                continue;
            }
            let fmin = tri.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
            let fmax = tri.iter().map(|v| v.1).fold(f64::NEG_INFINITY, f64::max);
            for k in 0..n_bands {
                let (a, b) = (levels[k], levels[k + 1]);
                if fmax < a || fmin > b {
                    continue;
                }
                let poly = clip(&clip(&tri, a, true), b, false);
                for j in 1..poly.len().saturating_sub(1) {
                    pieces.push(([poly[0].0, poly[j].0, poly[j + 1].0], k));
                }
            }
        }
        ContourBands { levels, pieces }
    }

    /// Colour of band `k`: the colormap spread over the band midpoints,
    /// first band at 0 and last at 1 (as matplotlib colours filled contours).
    pub fn band_colour(&self, k: usize, cmap: ColormapType) -> egui::Color32 {
        let n = self.levels.len().saturating_sub(1);
        let t = if n > 1 { k as f32 / (n - 1) as f32 } else { 0.5 };
        colormap::sample(t, cmap)
    }

    pub fn paint(&self, painter: &egui::Painter, coord: &ImageCoord, cmap: ColormapType, alpha: u8) {
        let mut mesh = egui::epaint::Mesh::default();
        for (pts, k) in &self.pieces {
            let c = self.band_colour(*k, cmap);
            let colour = egui::Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha);
            let base = mesh.vertices.len() as u32;
            for p in pts {
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: coord.to_screen(*p),
                    uv: egui::epaint::WHITE_UV,
                    color: colour,
                });
            }
            mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
        painter.add(egui::Shape::mesh(mesh));
    }
}

/// Sutherland–Hodgman clip of a convex polygon (with a linear scalar per
/// vertex) to `f >= level` (`keep_above`) or `f <= level`.
fn clip(poly: &[(egui::Pos2, f64)], level: f64, keep_above: bool) -> Vec<(egui::Pos2, f64)> {
    let inside = |f: f64| if keep_above { f >= level } else { f <= level };
    let n = poly.len();
    let mut out = Vec::with_capacity(n + 2);
    for i in 0..n {
        let cur = poly[i];
        let prev = poly[(i + n - 1) % n];
        let (ci, pi) = (inside(cur.1), inside(prev.1));
        if ci != pi {
            let t = ((level - prev.1) / (cur.1 - prev.1)) as f32;
            out.push((prev.0 + (cur.0 - prev.0) * t, level));
        }
        if ci {
            out.push(cur);
        }
    }
    out
}

/// Smooth (Gouraud) shading of per-vertex values — the "smooth" alternative
/// to [`ContourBands`].
pub fn paint_smooth(
    painter: &egui::Painter,
    coord: &ImageCoord,
    points: &[[f64; 2]],
    values: &[f64],
    triangles: &[[usize; 3]],
    vmin: f64,
    vmax: f64,
    cmap: ColormapType,
    alpha: u8,
) {
    let mut mesh = egui::epaint::Mesh::default();
    for (p, &v) in points.iter().zip(values) {
        let c = colormap::map_value(v, vmin, vmax, cmap);
        mesh.vertices.push(egui::epaint::Vertex {
            pos: coord.to_screen(egui::pos2(p[0] as f32, p[1] as f32)),
            uv: egui::epaint::WHITE_UV,
            color: egui::Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha),
        });
    }
    for t in triangles {
        if t.iter().all(|&i| values[i].is_finite()) {
            mesh.indices.extend(t.iter().map(|&i| i as u32));
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

// ---------------------------------------------------------------------------
// Colorbars
// ---------------------------------------------------------------------------

/// What a colorbar shows.
pub enum ColorbarScale<'a> {
    /// Continuous gradient, labelled at the ends.
    Continuous { vmin: f64, vmax: f64 },
    /// Discrete filled-contour bands, labelled at each level.
    Levels(&'a [f64]),
}

/// Render a vertical continuous colorbar in the top-right corner of the viewer.
pub fn render_colorbar(
    painter: &egui::Painter,
    viewer_rect: egui::Rect,
    vmin: f64,
    vmax: f64,
    cmap: ColormapType,
    label: &str,
) {
    render_colorbar_scale(painter, viewer_rect, ColorbarScale::Continuous { vmin, vmax }, cmap, label);
}

pub fn render_colorbar_scale(
    painter: &egui::Painter,
    viewer_rect: egui::Rect,
    scale: ColorbarScale<'_>,
    cmap: ColormapType,
    label: &str,
) {
    const BAR_W: f32 = 22.0;
    const MARGIN: f32 = 12.0;
    const TITLE_H: f32 = 22.0;
    const LABEL_W: f32 = 64.0;
    const LABEL_PAD: f32 = 4.0;
    const N: usize = 64;

    let bar_h = (viewer_rect.height() - MARGIN * 2.0 - TITLE_H - 20.0).max(40.0);
    let bar_x = viewer_rect.max.x - MARGIN - BAR_W - LABEL_W;
    let bar_y = viewer_rect.min.y + MARGIN + TITLE_H;

    let bg_rect = egui::Rect::from_min_max(
        egui::pos2(bar_x - 4.0, viewer_rect.min.y + MARGIN - 2.0),
        egui::pos2(viewer_rect.max.x - MARGIN + 2.0, bar_y + bar_h + 14.0),
    );
    painter.rect_filled(
        bg_rect,
        egui::CornerRadius::same(4),
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
    );

    // Title.
    painter.text(
        egui::pos2(bar_x + BAR_W * 0.5, viewer_rect.min.y + MARGIN + TITLE_H * 0.5),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::new(14.0, egui::FontFamily::Proportional),
        egui::Color32::from_rgb(210, 210, 210),
    );

    let lc = egui::Color32::from_rgb(220, 220, 220);
    let font = egui::FontId::new(12.0, egui::FontFamily::Proportional);
    let lx = bar_x + BAR_W + LABEL_PAD;
    let label_at = |y: f32, v: f64| {
        painter.text(egui::pos2(lx, y), egui::Align2::LEFT_CENTER, format_sci(v), font.clone(), lc);
    };

    match scale {
        ColorbarScale::Continuous { vmin, vmax } => {
            // Gradient bar (N strips, top = max).
            let strip_h = bar_h / N as f32;
            for i in 0..N {
                let t = 1.0 - i as f32 / (N - 1) as f32;
                let strip = egui::Rect::from_min_size(
                    egui::pos2(bar_x, bar_y + i as f32 * strip_h),
                    egui::vec2(BAR_W, strip_h + 0.5),
                );
                painter.rect_filled(strip, egui::CornerRadius::same(0), colormap::sample(t, cmap));
            }
            label_at(bar_y, vmax);
            label_at(bar_y + bar_h, vmin);
        }
        ColorbarScale::Levels(levels) => {
            let n = levels.len().saturating_sub(1);
            if n == 0 {
                return;
            }
            // Equal-height band per level interval, top = highest band.
            let band_h = bar_h / n as f32;
            for k in 0..n {
                let t = if n > 1 { k as f32 / (n - 1) as f32 } else { 0.5 };
                let y0 = bar_y + (n - 1 - k) as f32 * band_h;
                let band = egui::Rect::from_min_size(egui::pos2(bar_x, y0), egui::vec2(BAR_W, band_h + 0.5));
                painter.rect_filled(band, egui::CornerRadius::same(0), colormap::sample(t, cmap));
            }
            // Label every level, thinned so labels stay >= 14 px apart.
            let every = ((14.0 / band_h).ceil() as usize).max(1);
            for (k, &v) in levels.iter().enumerate() {
                if k % every == 0 || k == n {
                    label_at(bar_y + (n - k) as f32 * band_h, v);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference levels from matplotlib 3.10 `tricontourf(z).levels`
    /// (default N = 7) for data spanning `[zmin, zmax]`.
    #[test]
    fn nice_levels_match_matplotlib_tricontourf() {
        let cases: [(f64, f64, &[f64]); 5] = [
            (-0.006696022742552372, -0.003571614338133457,
             &[-0.0068, -0.0064, -0.006, -0.0056, -0.0052, -0.0048, -0.0044, -0.004, -0.0036, -0.0032]),
            (-417.0332416311019, -126.14766988896298,
             &[-440., -400., -360., -320., -280., -240., -200., -160., -120.]),
            (-9.37681190009151e-07, 4.154683306092982e-06,
             &[-1.6e-06, -8.0e-07, 0.0, 8.0e-07, 1.6e-06, 2.4e-06, 3.2e-06, 4.0e-06, 4.8e-06]),
            (-0.14102207275057466, 0.3643151532234372,
             &[-0.16, -0.08, 0.0, 0.08, 0.16, 0.24, 0.32, 0.4]),
            (-0.015024823999391895, 1.0784173634099086,
             &[-0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]),
        ];
        for (lo, hi, want) in cases {
            let got = nice_levels(lo, hi, 7);
            assert_eq!(got.len(), want.len(), "{lo} {hi}: {got:?}");
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() <= 1e-9 * w.abs().max((hi - lo).abs()), "{lo} {hi}: {got:?}");
            }
        }
        let flat = nice_levels(2.0, 2.0, 7);
        assert!(flat.len() >= 2 && flat[0] <= 2.0 && *flat.last().unwrap() >= 2.0, "{flat:?}");
    }

    #[test]
    fn bands_partition_a_triangle() {
        // f = x on the triangle (0,0),(2,0),(0,2): area 2, bands [0,1] and [1,2].
        let pts = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let vals = [0.0, 2.0, 0.0];
        let b = ContourBands::build(&pts, &vals, &[[0, 1, 2]], vec![0.0, 1.0, 2.0]);
        let area = |p: &[egui::Pos2; 3]| {
            0.5 * ((p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y)).abs()
        };
        let band_area = |k| b.pieces.iter().filter(|(_, j)| *j == k).map(|(p, _)| area(p)).sum::<f32>();
        // Band [0,1] is x <= 1: area 2 - 0.5 = 1.5; band [1,2] is the 0.5 corner.
        assert!((band_area(0) - 1.5).abs() < 1e-5, "{}", band_area(0));
        assert!((band_area(1) - 0.5).abs() < 1e-5, "{}", band_area(1));
    }

    #[test]
    fn values_outside_levels_unfilled() {
        let pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let b = ContourBands::build(&pts, &[5.0, 6.0, 7.0], &[[0, 1, 2]], vec![0.0, 1.0]);
        assert!(b.pieces.is_empty());
    }
}
