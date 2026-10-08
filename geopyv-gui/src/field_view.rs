//! Field tab — view mode. Mirrors the Python `Field` plotting API:
//! `inspect`, `contour`, `history` and `trace`. Contour values, positions
//! and triangles come from the core (`FieldSolution::contour_*`), the same
//! functions `Field.contour()` uses; this module only draws them.

use std::path::{Path, PathBuf};

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};

use geopyv_dev::field::{ContourReduction, FieldQuantity, FieldSolution, SeriesQuantity};
use geopyv_dev::io::GeopyvObject;

use crate::colormap::{self, ColormapType};
use crate::draw::ImageCoord;
use crate::image_viewer::{HoverInfo, ImageViewer, TextureCache};
use crate::overlay::{self, ColorbarScale, ContourBands};

// ---------------------------------------------------------------------------
// Plot modes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldPlotMode {
    #[default]
    Contour,
    Inspect,
    History,
    Trace,
}

impl FieldPlotMode {
    const ALL: [FieldPlotMode; 4] = [Self::Contour, Self::Inspect, Self::History, Self::Trace];

    fn label(self) -> &'static str {
        match self {
            Self::Contour => "Contour",
            Self::Inspect => "Inspect — particle positions",
            Self::History => "History — one particle",
            Self::Trace => "Trace — particle paths",
        }
    }
}

/// The per-particle series `history()` / `trace()` plot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SeriesKind {
    #[default]
    Warps,
    Strains,
    VolStrains,
}

impl SeriesKind {
    const ALL: [SeriesKind; 3] = [Self::Warps, Self::Strains, Self::VolStrains];

    fn label(self) -> &'static str {
        match self {
            Self::Warps => "Warps",
            Self::Strains => "Strains",
            Self::VolStrains => "Volumetric strain",
        }
    }
}

/// Matches `plots.py`'s `_WARP_LABELS_HISTORY`.
const WARP_LABELS: [&str; 12] = [
    "u (px)", "v (px)", "du/dx", "dv/dx", "du/dy", "dv/dy",
    "d²u/dx²", "d²v/dx²", "d²u/dxdy", "d²v/dxdy", "d²u/dy²", "d²v/dy²",
];
/// Matches `plots.py`'s `_STRAIN_LABELS_HISTORY`.
const STRAIN_LABELS: [&str; 6] = ["ε_xx", "ε_yy", "ε_zz", "ε_yz", "ε_xz", "ε_xy"];

fn warp_label(c: usize) -> String {
    WARP_LABELS.get(c).map_or_else(|| c.to_string(), |s| s.to_string())
}

// ---------------------------------------------------------------------------
// Contour settings + caches
// ---------------------------------------------------------------------------

struct ContourSettings {
    quantity: FieldQuantity,
    /// Half-open increment window `[start, stop)`.
    window: (usize, usize),
    use_dt: bool,
    dt_text: String,
    dt: f64,
    absolute: bool,
    deformed: bool,
    /// matplotlib's `tricontourf` default `N = 7` (→ `MaxNLocator(N + 1)`).
    n_levels: usize,
    smooth: bool,
    opacity: f32,
}

impl Default for ContourSettings {
    fn default() -> Self {
        Self {
            quantity: FieldQuantity::U,
            window: (0, 1),
            use_dt: false,
            dt_text: "1.0".to_string(),
            dt: 1.0,
            absolute: false,
            deformed: false,
            n_levels: 7,
            smooth: false,
            opacity: 1.0,
        }
    }
}

impl ContourSettings {
    fn reduction(&self) -> ContourReduction {
        ContourReduction {
            window: Some(self.window),
            dt: self.use_dt.then_some(self.dt),
            absolute: self.absolute,
        }
    }

    /// Colorbar label, as `contour_field` builds it.
    fn label(&self) -> String {
        let name = self.quantity.name();
        if self.use_dt {
            format!("{name} /s")
        } else if self.absolute {
            format!("{name} (abs)")
        } else {
            name.to_string()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ContourKey {
    quantity: FieldQuantity,
    reduction: ContourReduction,
    deformed: bool,
}

struct ContourData {
    values: Vec<f64>,
    points: Vec<[f64; 2]>,
    /// Finite data range of `values`.
    range: (f64, f64),
}

struct ContourCache {
    key: ContourKey,
    data: Result<ContourData, String>,
}

struct BandsCache {
    key: ContourKey,
    bands: ContourBands,
}

// ---------------------------------------------------------------------------
// View state
// ---------------------------------------------------------------------------

pub struct FieldViewState {
    pub loaded_path: Option<PathBuf>,
    pub solution: Option<FieldSolution>,
    mode: FieldPlotMode,
    contour: ContourSettings,
    colormap: ColormapType,
    range_auto: bool,
    range_min_text: String,
    range_max_text: String,
    range_min: f64,
    range_max: f64,
    selected_particle: Option<usize>,
    history_kind: SeriesKind,
    history_warps: [bool; 12],
    trace_kind: SeriesKind,
    trace_component: usize,
    /// Core `contour_triangles()` for the loaded solution.
    triangles: Vec<[usize; 3]>,
    contour_cache: Option<ContourCache>,
    bands_cache: Option<BandsCache>,
    viewer: ImageViewer,
    last_viewer_rect: Option<egui::Rect>,
    /// PNG to write from the next viewport screenshot.
    pending_png: Option<PathBuf>,
    screenshot_sent: bool,
    status: Option<String>,
}

impl FieldViewState {
    pub fn new() -> Self {
        let mut history_warps = [false; 12];
        history_warps[0] = true;
        history_warps[1] = true;
        Self {
            loaded_path: None,
            solution: None,
            mode: FieldPlotMode::default(),
            contour: ContourSettings::default(),
            colormap: ColormapType::default(),
            range_auto: true,
            range_min_text: String::new(),
            range_max_text: String::new(),
            range_min: 0.0,
            range_max: 1.0,
            selected_particle: None,
            history_kind: SeriesKind::default(),
            history_warps,
            trace_kind: SeriesKind::default(),
            trace_component: 0,
            triangles: Vec::new(),
            contour_cache: None,
            bands_cache: None,
            viewer: ImageViewer::new(),
            last_viewer_rect: None,
            pending_png: None,
            screenshot_sent: false,
            status: None,
        }
    }

    /// Show `solution` (just solved, or loaded from `path`).
    pub fn set_solution(&mut self, path: Option<PathBuf>, solution: Option<FieldSolution>) {
        self.loaded_path = path;
        self.triangles = solution
            .as_ref()
            .map(|s| s.contour_triangles().rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
            .unwrap_or_default();
        let inc_no = solution.as_ref().map_or(1, |s| s.inc_no().max(1));
        self.contour.window = (0, inc_no);
        self.selected_particle = None;
        self.contour_cache = None;
        self.bands_cache = None;
        self.status = None;
        self.solution = solution;
    }

    /// Save the viewer area (whatever mode is shown) to `dest` as a PNG,
    /// from a viewport screenshot taken on the next frame.
    pub fn request_png(&mut self, dest: PathBuf) {
        if self.solution.is_some() {
            self.pending_png = Some(dest);
            self.screenshot_sent = false;
        }
    }

    /// Drop a requested PNG export (the view is no longer on screen).
    pub fn cancel_png(&mut self) {
        self.pending_png = None;
    }

    /// `<field>_<mode>.png`, the default export file name.
    pub fn png_name(&self, selected_path: Option<&Path>) -> String {
        let stem = selected_path
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "field".to_string());
        format!("{stem}_{}.png", format!("{:?}", self.mode).to_lowercase())
    }

    fn ensure_loaded(&mut self, selected_path: Option<&Path>) {
        if self.loaded_path.as_deref() == selected_path {
            return;
        }
        let sol = selected_path.and_then(|p| match geopyv_dev::io::load(p) {
            Ok(GeopyvObject::Field(s)) => Some(s),
            _ => None,
        });
        self.set_solution(selected_path.map(Path::to_path_buf), sol);
    }

    fn contour_key(&self) -> ContourKey {
        ContourKey {
            quantity: self.contour.quantity,
            reduction: self.contour.reduction(),
            deformed: self.contour.deformed,
        }
    }

    fn ensure_contour_data(&mut self) {
        let key = self.contour_key();
        if self.contour_cache.as_ref().is_some_and(|c| c.key == key) {
            return;
        }
        let Some(sol) = &self.solution else { return };
        let data = (|| {
            let values = sol.contour_values(key.quantity, &key.reduction).map_err(|e| e.to_string())?;
            let coords = sol.contour_coordinates(key.deformed, &key.reduction).map_err(|e| e.to_string())?;
            let points: Vec<[f64; 2]> = coords.rows().into_iter().map(|r| [r[0], r[1]]).collect();
            let finite = values.iter().copied().filter(|v| v.is_finite());
            let lo = finite.clone().fold(f64::INFINITY, f64::min);
            let hi = finite.fold(f64::NEG_INFINITY, f64::max);
            Ok(ContourData { values: values.to_vec(), points, range: (lo, hi) })
        })();
        self.contour_cache = Some(ContourCache { key, data });
        self.bands_cache = None;
    }

    /// `[vmin, vmax]`: the data range, or the manual range.
    fn colour_range(&self, data_range: (f64, f64)) -> (f64, f64) {
        if self.range_auto {
            data_range
        } else {
            (self.range_min, self.range_max)
        }
    }

    // -----------------------------------------------------------------------
    // Central pane
    // -----------------------------------------------------------------------

    pub fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        viewer_rect: egui::Rect,
        selected_path: Option<&Path>,
        cache: &mut TextureCache,
    ) -> Option<HoverInfo> {
        self.ensure_loaded(selected_path);
        self.last_viewer_rect = Some(viewer_rect);
        self.poll_screenshot(ui.ctx(), viewer_rect);

        if self.solution.is_none() {
            ui.painter().rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("Select a field from the list")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                });
            });
            return None;
        }

        if self.mode == FieldPlotMode::History {
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                self.show_history_plot(ui);
            });
            return None;
        }

        if self.mode == FieldPlotMode::Contour {
            self.ensure_contour_data();
        }
        let (hover, coord) = self.show_canvas(ui, viewer_rect, cache);
        let painter = ui.painter_at(viewer_rect);
        match self.mode {
            FieldPlotMode::Contour => self.paint_contour(&painter, viewer_rect, &coord),
            FieldPlotMode::Inspect => {
                self.handle_pick(ui, viewer_rect, &coord);
                self.paint_inspect(&painter, &coord);
            }
            FieldPlotMode::Trace => self.paint_trace(&painter, viewer_rect, &coord),
            FieldPlotMode::History => {}
        }
        hover
    }

    /// The reference image (pan/zoom) or, without one, a dark background
    /// fitted to the particles. Returns the image→screen transform.
    fn show_canvas(
        &mut self,
        ui: &mut egui::Ui,
        viewer_rect: egui::Rect,
        cache: &mut TextureCache,
    ) -> (Option<HoverInfo>, ImageCoord) {
        let sol = self.solution.as_ref().expect("checked by caller");
        let image = sol.image_0_path.clone().filter(|p| p.exists());
        let mut hover = None;
        let mut coord = None;
        if let Some(path) = image {
            hover = ui
                .allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    self.viewer.show(ui, &path, cache, None)
                })
                .inner;
            coord = self.viewer.last_coord();
        }
        let coord = coord.unwrap_or_else(|| {
            ui.painter().rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
            let c = &sol.initial_coordinates;
            overlay::coord_from_points(c.rows().into_iter().map(|r| [r[0], r[1]]), viewer_rect)
        });
        (hover, coord)
    }

    fn paint_contour(&mut self, painter: &egui::Painter, viewer_rect: egui::Rect, coord: &ImageCoord) {
        let Some(cache) = &self.contour_cache else { return };
        let data = match &cache.data {
            Ok(d) => d,
            Err(msg) => {
                paint_message(painter, viewer_rect, msg);
                return;
            }
        };
        if !data.range.0.is_finite() {
            paint_message(painter, viewer_rect, "No finite values to contour");
            return;
        }
        let (lo, hi) = self.colour_range(data.range);
        let alpha = (self.contour.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
        let label = self.contour.label();

        if self.contour.smooth {
            let (lo, hi) = if hi > lo { (lo, hi) } else { (lo - 1e-12, hi + 1e-12) };
            overlay::paint_smooth(
                painter, coord, &data.points, &data.values, &self.triangles, lo, hi, self.colormap, alpha,
            );
            overlay::render_colorbar(painter, viewer_rect, lo, hi, self.colormap, &label);
        } else {
            let levels = overlay::nice_levels(lo, hi, self.contour.n_levels);
            if !self.bands_cache.as_ref().is_some_and(|b| b.key == cache.key && b.bands.levels == levels) {
                let bands = ContourBands::build(&data.points, &data.values, &self.triangles, levels);
                self.bands_cache = Some(BandsCache { key: cache.key.clone(), bands });
            }
            let bands = &self.bands_cache.as_ref().expect("just built").bands;
            bands.paint(painter, coord, self.colormap, alpha);
            overlay::render_colorbar_scale(
                painter, viewer_rect, ColorbarScale::Levels(&bands.levels), self.colormap, &label,
            );
        }

        if self.solution.as_ref().is_some_and(|s| s.region.is_none()) {
            paint_footer(
                painter,
                viewer_rect,
                "Region not stored (field saved before .pyv 0x08): full hull shown — re-solve to mask exclusions",
            );
        }
    }

    fn paint_inspect(&self, painter: &egui::Painter, coord: &ImageCoord) {
        let sol = self.solution.as_ref().expect("checked by caller");
        let c = &sol.initial_coordinates;
        let fill = egui::Color32::from_rgb(31, 119, 180);
        for r in c.rows() {
            let p = coord.to_screen(egui::pos2(r[0] as f32, r[1] as f32));
            painter.circle_filled(p, 3.0, fill);
        }
        if let Some(i) = self.selected_particle.filter(|&i| i < c.nrows()) {
            let p = coord.to_screen(egui::pos2(c[[i, 0]] as f32, c[[i, 1]] as f32));
            painter.circle_filled(p, 7.0, egui::Color32::WHITE);
            painter.circle_filled(p, 5.5, egui::Color32::RED);
        }
    }

    /// Click in Inspect mode → select the nearest particle (within 15 px).
    fn handle_pick(&mut self, ui: &egui::Ui, viewer_rect: egui::Rect, coord: &ImageCoord) {
        let click = ui.input(|i| i.pointer.primary_clicked().then(|| i.pointer.interact_pos()).flatten());
        let Some(pos) = click.filter(|p| viewer_rect.contains(*p)) else { return };
        let sol = self.solution.as_ref().expect("checked by caller");
        let nearest = sol
            .initial_coordinates
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, r)| (i, coord.to_screen(egui::pos2(r[0] as f32, r[1] as f32)).distance(pos)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, d)) = nearest {
            if d <= 15.0 {
                self.selected_particle = Some(i);
            }
        }
    }

    /// `trace_field`: every particle's path, each segment coloured by the
    /// change of the chosen series over that increment.
    fn paint_trace(&self, painter: &egui::Painter, viewer_rect: egui::Rect, coord: &ImageCoord) {
        let sol = self.solution.as_ref().expect("checked by caller");
        let q = self.trace_quantity();
        let mut segments = Vec::new();
        for p in &sol.particles {
            let s = p.series(q);
            for m in 1..p.coordinates.nrows().min(s.len()) {
                let a = egui::pos2(p.coordinates[[m - 1, 0]] as f32, p.coordinates[[m - 1, 1]] as f32);
                let b = egui::pos2(p.coordinates[[m, 0]] as f32, p.coordinates[[m, 1]] as f32);
                segments.push((a, b, s[m] - s[m - 1]));
            }
        }
        let finite = segments.iter().map(|s| s.2).filter(|v| v.is_finite());
        let data_range = (
            finite.clone().fold(f64::INFINITY, f64::min),
            finite.fold(f64::NEG_INFINITY, f64::max),
        );
        if !data_range.0.is_finite() {
            paint_message(painter, viewer_rect, "No increments to trace");
            return;
        }
        let (lo, hi) = self.colour_range(data_range);
        for (a, b, v) in segments {
            let colour = colormap::map_value(v, lo, hi, self.colormap);
            painter.line_segment([coord.to_screen(a), coord.to_screen(b)], egui::Stroke::new(1.5, colour));
        }
        overlay::render_colorbar(painter, viewer_rect, lo, hi, self.colormap, &self.trace_label());
    }

    fn trace_quantity(&self) -> SeriesQuantity {
        match self.trace_kind {
            SeriesKind::Warps => SeriesQuantity::Warp(self.trace_component),
            SeriesKind::Strains => SeriesQuantity::Strain(self.trace_component),
            SeriesKind::VolStrains => SeriesQuantity::VolStrain,
        }
    }

    fn trace_label(&self) -> String {
        match self.trace_kind {
            SeriesKind::Warps => warp_label(self.trace_component),
            SeriesKind::Strains => STRAIN_LABELS.get(self.trace_component).unwrap_or(&"?").to_string(),
            SeriesKind::VolStrains => "ε_vol".to_string(),
        }
    }

    /// `history_field`: the selected particle's series against image number.
    fn show_history_plot(&self, ui: &mut egui::Ui) {
        let sol = self.solution.as_ref().expect("checked by caller");
        let Some(p) = self.selected_particle.and_then(|i| sol.particles.get(i)) else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("Pick a particle in Inspect mode, or set one on the right")
                        .size(16.0)
                        .color(ui.visuals().weak_text_color()),
                );
            });
            return;
        };
        let lines: Vec<(String, SeriesQuantity)> = match self.history_kind {
            SeriesKind::Warps => (0..p.warps.ncols())
                .filter(|&c| self.history_warps.get(c).copied().unwrap_or(false))
                .map(|c| (warp_label(c), SeriesQuantity::Warp(c)))
                .collect(),
            // history_particle's "strains" set: ε_xx, ε_yy, ε_xy, ε_vol.
            SeriesKind::Strains => vec![
                (STRAIN_LABELS[0].to_string(), SeriesQuantity::Strain(0)),
                (STRAIN_LABELS[1].to_string(), SeriesQuantity::Strain(1)),
                (STRAIN_LABELS[5].to_string(), SeriesQuantity::Strain(5)),
                ("ε_vol".to_string(), SeriesQuantity::VolStrain),
            ],
            SeriesKind::VolStrains => vec![("ε_vol".to_string(), SeriesQuantity::VolStrain)],
        };
        let y_label = match self.history_kind {
            SeriesKind::Warps => "Value",
            SeriesKind::Strains => "Strain, ε",
            SeriesKind::VolStrains => "Volumetric strain, ε_vol",
        };
        Plot::new("field_history_plot")
            .legend(Legend::default())
            .x_axis_label("Image number, i")
            .y_axis_label(y_label)
            .show(ui, |plot_ui| {
                for (name, q) in lines {
                    let s = p.series(q);
                    let pts: Vec<[f64; 2]> = s.iter().enumerate().map(|(i, &v)| [i as f64, v]).collect();
                    plot_ui.line(Line::new(PlotPoints::from(pts)).name(name));
                }
            });
    }

    // -----------------------------------------------------------------------
    // Right pane
    // -----------------------------------------------------------------------

    pub fn show_right(&mut self, ui: &mut egui::Ui, selected_path: Option<&Path>, out_dir: &Path) {
        self.ensure_loaded(selected_path);
        let Some(sol) = &self.solution else {
            ui.label(
                egui::RichText::new("No field selected")
                    .size(15.0)
                    .color(ui.visuals().weak_text_color()),
            );
            return;
        };
        let n_particles = sol.particles.len();
        let inc_no = sol.inc_no();
        let n_warps = sol.particles.first().map_or(0, |p| p.warps.ncols());
        let has_region = sol.region.is_some();

        section_header(ui, "Field");
        meta_row(ui, "Particles", &n_particles.to_string());
        meta_row(ui, "Increments", &inc_no.saturating_sub(1).to_string());
        meta_row(ui, "Region stored", if has_region { "yes" } else { "no (re-solve)" });
        separator(ui);

        section_header(ui, "Display");
        for m in FieldPlotMode::ALL {
            ui.radio_value(&mut self.mode, m, m.label());
        }
        separator(ui);

        match self.mode {
            FieldPlotMode::Contour => self.contour_controls(ui, inc_no),
            FieldPlotMode::Inspect => self.inspect_controls(ui, n_particles),
            FieldPlotMode::History => self.history_controls(ui, n_particles, n_warps),
            FieldPlotMode::Trace => self.trace_controls(ui, n_warps),
        }

        separator(ui);
        let field_name = selected_path
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "field".to_string());
        ui.horizontal(|ui| {
            let png = egui::Button::new("Export PNG").min_size(egui::vec2(80.0, 24.0));
            if ui.add_enabled(self.pending_png.is_none(), png).clicked() {
                self.request_png(out_dir.join(self.png_name(selected_path)));
                ui.ctx().request_repaint();
            }
            ui.add_space(4.0);
            if ui.add(egui::Button::new("Export CSV").min_size(egui::vec2(80.0, 24.0))).clicked() {
                let dest = out_dir.join(format!("{field_name}.csv"));
                let sol = self.solution.as_ref().expect("checked above");
                self.status = Some(match export_csv(sol, &dest) {
                    Ok(()) => format!("Saved {}", dest.display()),
                    Err(e) => format!("CSV export error: {e}"),
                });
            }
        });
        if let Some(msg) = &self.status {
            ui.label(egui::RichText::new(msg).size(13.0).color(ui.visuals().weak_text_color()));
        }
    }

    fn contour_controls(&mut self, ui: &mut egui::Ui, inc_no: usize) {
        let c = &mut self.contour;
        section_header(ui, "Quantity");
        egui::ComboBox::from_id_salt("field_contour_quantity")
            .selected_text(c.quantity.name())
            .width(150.0)
            .show_ui(ui, |ui| {
                for q in FieldQuantity::ALL {
                    ui.selectable_value(&mut c.quantity, q, q.name());
                }
            });

        ui.add_space(4.0);
        section_header(ui, "Window (increments)");
        let max = inc_no.max(1);
        let (mut start, mut stop) = c.window;
        ui.add(egui::Slider::new(&mut start, 0..=max - 1).text("start"));
        ui.add(egui::Slider::new(&mut stop, 1..=max).text("stop (excl.)"));
        if start != c.window.0 {
            stop = stop.max(start + 1);
        } else if stop != c.window.1 {
            start = start.min(stop - 1);
        }
        c.window = (start.min(max - 1), stop.clamp(start + 1, max));

        ui.add_space(4.0);
        section_header(ui, "Reduction");
        ui.horizontal(|ui| {
            ui.checkbox(&mut c.use_dt, "Rate, dt =");
            let r = ui.add_enabled(c.use_dt, egui::TextEdit::singleline(&mut c.dt_text).desired_width(60.0));
            if r.changed() {
                if let Ok(v) = c.dt_text.trim().parse::<f64>() {
                    if v > 0.0 {
                        c.dt = v;
                    }
                }
            }
        });
        ui.add_enabled(!c.use_dt, egui::Checkbox::new(&mut c.absolute, "Absolute (sum of |Δ|)"));

        ui.add_space(4.0);
        section_header(ui, "Position");
        ui.radio_value(&mut c.deformed, false, "Reference");
        ui.radio_value(&mut c.deformed, true, "Deformed (window end)");

        ui.add_space(4.0);
        section_header(ui, "Style");
        ui.checkbox(&mut c.smooth, "Smooth shading");
        ui.add_enabled(!c.smooth, egui::Slider::new(&mut c.n_levels, 1..=30).text("levels"));
        ui.add(egui::Slider::new(&mut c.opacity, 0.0..=1.0).text("opacity"));
        self.colormap_and_range(ui);
    }

    fn inspect_controls(&mut self, ui: &mut egui::Ui, n_particles: usize) {
        section_header(ui, "Selected particle");
        match self.selected_particle.filter(|&i| i < n_particles) {
            Some(i) => {
                let sol = self.solution.as_ref().expect("checked by caller");
                let c = &sol.initial_coordinates;
                meta_row(ui, "Index", &i.to_string());
                meta_row(ui, "x, y", &format!("{:.1}, {:.1}", c[[i, 0]], c[[i, 1]]));
                ui.horizontal(|ui| {
                    if ui.button("Show history").clicked() {
                        self.mode = FieldPlotMode::History;
                    }
                    if ui.button("Clear").clicked() {
                        self.selected_particle = None;
                    }
                });
            }
            None => {
                ui.label(egui::RichText::new("Click a particle to select it").color(ui.visuals().weak_text_color()));
            }
        }
    }

    fn history_controls(&mut self, ui: &mut egui::Ui, n_particles: usize, n_warps: usize) {
        section_header(ui, "Particle");
        let mut idx = self.selected_particle.unwrap_or(0);
        if ui.add(egui::DragValue::new(&mut idx).range(0..=n_particles.saturating_sub(1))).changed()
            || self.selected_particle.is_none()
        {
            if n_particles > 0 && self.selected_particle != Some(idx) {
                self.selected_particle = Some(idx.min(n_particles - 1));
            }
        }
        ui.add_space(4.0);
        section_header(ui, "Quantity");
        for k in SeriesKind::ALL {
            ui.radio_value(&mut self.history_kind, k, k.label());
        }
        if self.history_kind == SeriesKind::Warps {
            ui.add_space(4.0);
            section_header(ui, "Components");
            egui::Grid::new("field_history_warps").num_columns(2).show(ui, |ui| {
                for c in 0..n_warps.min(self.history_warps.len()) {
                    ui.checkbox(&mut self.history_warps[c], warp_label(c));
                    if c % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
        }
    }

    fn trace_controls(&mut self, ui: &mut egui::Ui, n_warps: usize) {
        section_header(ui, "Quantity");
        for k in SeriesKind::ALL {
            if ui.radio_value(&mut self.trace_kind, k, k.label()).changed() {
                self.trace_component = 0;
            }
        }
        let labels: Vec<String> = match self.trace_kind {
            SeriesKind::Warps => (0..n_warps).map(warp_label).collect(),
            SeriesKind::Strains => STRAIN_LABELS.iter().map(|s| s.to_string()).collect(),
            SeriesKind::VolStrains => Vec::new(),
        };
        if !labels.is_empty() {
            ui.add_space(4.0);
            section_header(ui, "Component");
            self.trace_component = self.trace_component.min(labels.len() - 1);
            egui::ComboBox::from_id_salt("field_trace_component")
                .selected_text(labels[self.trace_component].clone())
                .width(120.0)
                .show_ui(ui, |ui| {
                    for (c, l) in labels.iter().enumerate() {
                        ui.selectable_value(&mut self.trace_component, c, l);
                    }
                });
        }
        self.colormap_and_range(ui);
    }

    fn colormap_and_range(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        section_header(ui, "Colormap");
        egui::ComboBox::from_id_salt("field_colormap")
            .selected_text(self.colormap.label())
            .width(100.0)
            .show_ui(ui, |ui| {
                for &cmap in ColormapType::ALL {
                    ui.selectable_value(&mut self.colormap, cmap, cmap.label());
                }
            });

        ui.add_space(4.0);
        section_header(ui, "Range");
        let was_auto = self.range_auto;
        ui.checkbox(&mut self.range_auto, "Auto");
        if self.range_auto {
            return;
        }
        if was_auto {
            // Seed the manual range from the current data range.
            let current = self
                .contour_cache
                .as_ref()
                .filter(|_| self.mode == FieldPlotMode::Contour)
                .and_then(|c| c.data.as_ref().ok())
                .map(|d| d.range)
                .filter(|r| r.0.is_finite());
            if let Some((lo, hi)) = current {
                self.range_min = lo;
                self.range_max = hi;
            }
            self.range_min_text = format!("{:.4e}", self.range_min);
            self.range_max_text = format!("{:.4e}", self.range_max);
        }
        egui::Grid::new("field_range_grid").num_columns(2).spacing([6.0, 4.0]).show(ui, |ui| {
            for (name, text, value) in [
                ("Min:", &mut self.range_min_text, &mut self.range_min),
                ("Max:", &mut self.range_max_text, &mut self.range_max),
            ] {
                ui.label(name);
                if ui.add(egui::TextEdit::singleline(text).desired_width(90.0)).changed() {
                    if let Ok(v) = text.trim().parse::<f64>() {
                        *value = v;
                    }
                }
                ui.end_row();
            }
        });
    }

    // -----------------------------------------------------------------------
    // PNG export (viewport screenshot cropped to the viewer)
    // -----------------------------------------------------------------------

    fn poll_screenshot(&mut self, ctx: &egui::Context, viewer_rect: egui::Rect) {
        if self.pending_png.is_none() {
            return;
        }
        if !self.screenshot_sent {
            // Taken at the end of this frame, so it shows the plot as drawn now.
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            self.screenshot_sent = true;
            ctx.request_repaint();
            return;
        }
        let shot = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        let Some(image) = shot else {
            ctx.request_repaint();
            return;
        };
        let dest = self.pending_png.take().expect("checked above");
        let rect = self.last_viewer_rect.unwrap_or(viewer_rect);
        self.status = Some(match save_cropped(&image, rect, ctx.pixels_per_point(), &dest) {
            Ok(()) => format!("Saved {}", dest.display()),
            Err(e) => format!("PNG export error: {e}"),
        });
    }
}

fn save_cropped(image: &egui::ColorImage, rect: egui::Rect, ppp: f32, dest: &Path) -> Result<(), String> {
    let [w, h] = image.size;
    let x0 = ((rect.min.x * ppp).round().max(0.0) as usize).min(w);
    let y0 = ((rect.min.y * ppp).round().max(0.0) as usize).min(h);
    let x1 = ((rect.max.x * ppp).round().max(0.0) as usize).min(w);
    let y1 = ((rect.max.y * ppp).round().max(0.0) as usize).min(h);
    if x1 <= x0 || y1 <= y0 {
        return Err("viewer area is empty".to_string());
    }
    let mut out = image::RgbaImage::new((x1 - x0) as u32, (y1 - y0) as u32);
    for y in y0..y1 {
        for x in x0..x1 {
            let c = image.pixels[y * w + x].to_srgba_unmultiplied();
            out.put_pixel((x - x0) as u32, (y - y0) as u32, image::Rgba(c));
        }
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    out.save(dest).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// CSV export (long format per §14 Option B)
// ---------------------------------------------------------------------------

fn export_csv(sol: &FieldSolution, dest: &Path) -> Result<(), String> {
    use std::io::Write;

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let mut f = std::fs::File::create(dest).map_err(|e| e.to_string())?;
    writeln!(
        f,
        "increment,particle,x,y,eps_xx,eps_yy,eps_xy,eps_1,eps_2,vol_strain"
    )
    .map_err(|e| e.to_string())?;

    for (p_idx, particle) in sol.particles.iter().enumerate() {
        let n_incs = particle.coordinates.nrows();
        for inc in 0..n_incs {
            let x = particle.coordinates[[inc, 0]];
            let y = particle.coordinates[[inc, 1]];
            let exx = particle.strains[[inc, 0]];
            let eyy = particle.strains[[inc, 1]];
            let exy = if inc < particle.strains.nrows() { particle.strains[[inc, 5]] } else { 0.0 };
            let mean = (exx + eyy) / 2.0;
            let dev = (((exx - eyy) / 2.0).powi(2) + exy.powi(2)).sqrt();
            let e1 = mean + dev;
            let e2 = mean - dev;
            let vol = if inc < particle.vol_strains.len() { particle.vol_strains[inc] } else { 0.0 };

            writeln!(
                f,
                "{inc},{p_idx},{x:.4},{y:.4},{exx:.6e},{eyy:.6e},{exy:.6e},{e1:.6e},{e2:.6e},{vol:.6e}"
            )
            .map_err(|e| e.to_string())?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn paint_message(painter: &egui::Painter, rect: egui::Rect, msg: &str) {
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        msg,
        egui::FontId::new(15.0, egui::FontFamily::Proportional),
        egui::Color32::from_rgb(230, 180, 80),
    );
}

fn paint_footer(painter: &egui::Painter, rect: egui::Rect, msg: &str) {
    let pos = egui::pos2(rect.min.x + 10.0, rect.max.y - 10.0);
    let galley = painter.layout_no_wrap(
        msg.to_string(),
        egui::FontId::new(13.0, egui::FontFamily::Proportional),
        egui::Color32::from_rgb(230, 180, 80),
    );
    let bg = egui::Rect::from_min_size(pos - egui::vec2(4.0, galley.size().y + 2.0), galley.size() + egui::vec2(8.0, 4.0));
    painter.rect_filled(bg, egui::CornerRadius::same(3), egui::Color32::from_rgba_unmultiplied(0, 0, 0, 170));
    painter.galley(pos - egui::vec2(0.0, galley.size().y), galley, egui::Color32::WHITE);
}

fn separator(ui: &mut egui::Ui) {
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(6.0);
}

fn section_header(ui: &mut egui::Ui, label: &str) {
    ui.label(
        egui::RichText::new(label)
            .size(15.0)
            .color(ui.visuals().weak_text_color()),
    );
    ui.add_space(2.0);
}

fn meta_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(format!("{label}:"))
                .size(16.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.label(egui::RichText::new(value).size(15.0));
    });
    ui.add_space(2.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use geopyv_dev::field::FieldRegion;
    use geopyv_dev::particle::ParticleSolution;
    use ndarray::{array, Array1, Array2};
    use std::sync::Arc;

    /// 6x6 grid of particles at 0.5..5.5 in a [0,6]^2 boundary with a hole
    /// over the centre cell, u = x at increment 1 (and moved by u).
    fn holed_field() -> FieldSolution {
        let mut particles = Vec::new();
        let mut init = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                let (x, y) = (i as f64 + 0.5, j as f64 + 0.5);
                init.extend([x, y]);
                let mut warps = Array2::<f64>::zeros((2, 6));
                warps[[1, 0]] = x;
                particles.push(Arc::new(ParticleSolution {
                    coordinates: array![[x, y], [2.0 * x, y]],
                    warps,
                    incs: Array2::zeros((2, 6)),
                    volumes: Array1::ones(2),
                    strains: Array2::zeros((2, 6)),
                    strain_incs: Array2::zeros((1, 6)),
                    vol_strains: Array1::zeros(2),
                    reference_update_register: vec![],
                    image_0_path: None,
                    calibrated: false,
                    config: None,
                    principal_strains: Some(Array2::zeros((2, 4))),
                    gamma_max_grad: None,
                }));
            }
        }
        FieldSolution {
            particles,
            initial_coordinates: Array2::from_shape_vec((36, 2), init).unwrap(),
            vol_totals: Array1::ones(2),
            reference_update_register: vec![],
            image_0_path: None,
            calibrated: false,
            depth: 1.0,
            track: true,
            region: Some(FieldRegion {
                boundary: array![[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]],
                exclusions: vec![array![[2.6, 2.6], [3.4, 2.6], [3.4, 3.4], [2.6, 3.4]]],
            }),
        }
    }

    fn area(p: &[egui::Pos2; 3]) -> f32 {
        0.5 * ((p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y)).abs()
    }

    #[test]
    fn contour_bands_cover_masked_triangles_only() {
        let mut view = FieldViewState::new();
        view.set_solution(None, Some(holed_field()));
        assert_eq!(view.contour.window, (0, 2), "window defaults to every increment");
        // 5x5 cells x 2 triangles, minus the 2 over the hole.
        assert_eq!(view.triangles.len(), 48);

        for deformed in [false, true] {
            view.contour.deformed = deformed;
            view.ensure_contour_data();
            let data = view.contour_cache.as_ref().unwrap().data.as_ref().unwrap();
            assert_eq!(data.range, (0.5, 5.5), "u = x");
            let levels = overlay::nice_levels(data.range.0, data.range.1, 7);
            let bands = ContourBands::build(&data.points, &data.values, &view.triangles, levels);

            let sx = if deformed { 2.0 } else { 1.0 };
            let hole = egui::Rect::from_min_max(egui::pos2(2.5 * sx, 2.5), egui::pos2(3.5 * sx, 3.5));
            for (p, _) in &bands.pieces {
                let c = egui::pos2((p[0].x + p[1].x + p[2].x) / 3.0, (p[0].y + p[1].y + p[2].y) / 3.0);
                assert!(!hole.contains(c), "band piece inside the hole at {c:?}");
            }
            let covered: f32 = bands.pieces.iter().map(|(p, _)| area(p)).sum();
            let expected = (25.0 - 1.0) * sx; // 5x5 hull minus the hole cell, stretched by sx
            assert!((covered - expected).abs() < 1e-3, "covered {covered}, expected {expected}");
        }
    }
}
