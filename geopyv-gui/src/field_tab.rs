use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use ndarray::Array2;

use geopyv_dev::field::{grid_particles, Field, FieldDistribution, FieldSolution};
use geopyv_dev::io::GeopyvObject;
use geopyv_dev::particle::ParticleSolution;

use crate::colormap::{self, ColormapType};
use crate::draw::{ActiveDrawMode, DrawState, DrawnRegion};
use crate::image_viewer::{HoverInfo, ImageViewer, TextureCache};

// ---------------------------------------------------------------------------
// Plot type / component
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldPlotType {
    #[default]
    Scatter,
    TimeSeries,
    VolTotals,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldComponent {
    #[default]
    Exx,
    Eyy,
    Exy,
    E1,
    E2,
    VolStrain,
}

impl FieldComponent {
    pub fn label(self) -> &'static str {
        match self {
            Self::Exx => "\u{03b5}_xx",
            Self::Eyy => "\u{03b5}_yy",
            Self::Exy => "\u{03b5}_xy",
            Self::E1 => "\u{03b5}\u{2081} (principal)",
            Self::E2 => "\u{03b5}\u{2082} (principal)",
            Self::VolStrain => "Vol. strain",
        }
    }

    pub fn value_at(self, sol: &ParticleSolution, frame: usize) -> f64 {
        let f = frame.min(sol.strains.nrows().saturating_sub(1));
        match self {
            Self::Exx => sol.strains[[f, 0]],
            Self::Eyy => sol.strains[[f, 1]],
            Self::Exy => sol.strains[[f, 5]],
            // Stored at solve time (core `principal_strains`); recomputed only
            // for solutions saved before that field existed.
            Self::E1 | Self::E2 => {
                let col = if self == Self::E1 { 0 } else { 1 };
                match &sol.principal_strains {
                    Some(ps) => ps[[f, col]],
                    None => geopyv_dev::particle::principal_strains(&sol.strains)[[f, col]],
                }
            }
            Self::VolStrain => {
                let f2 = frame.min(sol.vol_strains.len().saturating_sub(1));
                sol.vol_strains[f2]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grid generation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// New field form
// ---------------------------------------------------------------------------

pub struct NewFieldForm {
    pub name: String,
    pub seq_idx: Option<usize>,
    pub ref_image_idx: Option<usize>,
    pub draw: DrawState,
    pub spacing_text: String,
    pub spacing: f32,
    pub lagrangian: bool,
    pub depth_text: String,
    pub depth: f64,
    pub factor_text: String,
    pub factor: f64,
    pub true_incs: bool,
    pub form_error: Option<String>,
    cached_grid: Vec<egui::Pos2>,
    last_boundary_len: usize,
    last_exclusions_sig: usize,
    last_spacing: f32,
}

impl Default for NewFieldForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            seq_idx: None,
            ref_image_idx: None,
            draw: DrawState::new(),
            spacing_text: "20".to_string(),
            spacing: 20.0,
            lagrangian: true,
            depth_text: "1.0".to_string(),
            depth: 1.0,
            factor_text: geopyv_dev::particle::ParticleConfig::default().factor.to_string(),
            factor: geopyv_dev::particle::ParticleConfig::default().factor,
            true_incs: true,
            form_error: None,
            cached_grid: Vec::new(),
            last_boundary_len: 0,
            last_exclusions_sig: 0,
            last_spacing: 0.0,
        }
    }
}

/// A drawn region as an `(N, 2)` `[x, y]` vertex array (image-pixel space).
fn polygon_array(region: &DrawnRegion) -> Array2<f64> {
    let verts = region.to_nodes();
    Array2::from_shape_fn((verts.len(), 2), |(i, j)| verts[i][j])
}

impl NewFieldForm {
    pub fn ensure_grid_current(&mut self) {
        let boundary_len = self
            .draw
            .boundary
            .as_ref()
            .map(|b| b.to_egui_verts().len())
            .unwrap_or(0);
        let exclusions_sig: usize = self
            .draw
            .exclusions
            .iter()
            .map(|e| e.to_egui_verts().len() + 1)
            .sum();

        if boundary_len == self.last_boundary_len
            && exclusions_sig == self.last_exclusions_sig
            && (self.spacing - self.last_spacing).abs() < 0.5
        {
            return;
        }

        self.last_boundary_len = boundary_len;
        self.last_exclusions_sig = exclusions_sig;
        self.last_spacing = self.spacing;

        // Preview exactly what `FieldDistribution::Grid` will place.
        self.cached_grid = match &self.draw.boundary {
            Some(b) => {
                let boundary = polygon_array(b);
                let exclusions: Vec<Array2<f64>> =
                    self.draw.exclusions.iter().map(polygon_array).collect();
                let views: Vec<_> = exclusions.iter().map(|e| e.view()).collect();
                let (coords, _) = grid_particles(boundary.view(), &views, self.spacing as f64, 1.0);
                coords
                    .rows()
                    .into_iter()
                    .map(|r| egui::pos2(r[0] as f32, r[1] as f32))
                    .collect()
            }
            None => Vec::new(),
        };
    }

    pub fn invalidate_grid(&mut self) {
        self.last_boundary_len = usize::MAX;
    }

    pub fn can_run(&self, sequences: &[PathBuf]) -> bool {
        let name_ok = !self.name.trim().is_empty()
            && !self.name.contains('/')
            && !self.name.contains('\\');
        let seq_ok = self.seq_idx.map(|i| i < sequences.len()).unwrap_or(false);
        let boundary_ok = self.draw.boundary_ok();
        let grid_ok = !self.cached_grid.is_empty();
        let depth_ok = self.depth > 0.0;
        name_ok && seq_ok && boundary_ok && grid_ok && depth_ok
    }
}

// ---------------------------------------------------------------------------
// View state
// ---------------------------------------------------------------------------

pub struct FieldViewState {
    pub loaded_path: Option<PathBuf>,
    pub solution: Option<FieldSolution>,
    pub plot_type: FieldPlotType,
    pub frame: usize,
    pub component: FieldComponent,
    pub colormap: ColormapType,
    pub range_auto: bool,
    pub range_min_text: String,
    pub range_max_text: String,
    pub range_min: f64,
    pub range_max: f64,
    pub selected_particle: Option<usize>,
}

impl FieldViewState {
    fn new() -> Self {
        Self {
            loaded_path: None,
            solution: None,
            plot_type: FieldPlotType::default(),
            frame: 0,
            component: FieldComponent::default(),
            colormap: ColormapType::default(),
            range_auto: true,
            range_min_text: "0.0".to_string(),
            range_max_text: "1.0".to_string(),
            range_min: 0.0,
            range_max: 1.0,
            selected_particle: None,
        }
    }

    fn clamp_frame(&mut self) {
        if let Some(sol) = &self.solution {
            let max_frame = if sol.particles.is_empty() {
                0
            } else {
                sol.particles[0].coordinates.nrows().saturating_sub(1)
            };
            self.frame = self.frame.min(max_frame);
        }
    }

    /// Compute the [vmin, vmax] range for the current component + frame.
    fn compute_range(&self, sol: &FieldSolution) -> (f64, f64) {
        if !self.range_auto {
            return (self.range_min, self.range_max);
        }
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;
        for p in &sol.particles {
            let v = self.component.value_at(p, self.frame);
            if v < vmin { vmin = v; }
            if v > vmax { vmax = v; }
        }
        if vmin >= vmax {
            vmin -= 1e-10;
            vmax += 1e-10;
        }
        (vmin, vmax)
    }
}

// ---------------------------------------------------------------------------
// Background solve state
// ---------------------------------------------------------------------------

pub struct FieldSolveState {
    pub running: bool,
    pub progress: f32,
    pub message: String,
    pub cancel: Arc<AtomicBool>,
    pub result: Option<Result<FieldSolution, String>>,
}

impl FieldSolveState {
    fn new() -> Self {
        Self {
            running: false,
            progress: 0.0,
            message: String::new(),
            cancel: Arc::new(AtomicBool::new(false)),
            result: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Spawn parameters
// ---------------------------------------------------------------------------

pub struct FieldSpawnParams {
    pub name: String,
    pub sequence_path: PathBuf,
    /// Field boundary and exclusion polygons, `(N, 2)` `[x, y]`.
    pub boundary: Array2<f64>,
    pub exclusions: Vec<Array2<f64>>,
    /// Grid pitch for `FieldDistribution::Grid`.
    pub spacing: f64,
    pub track: bool,
    pub depth: f64,
    pub factor: f64,
    pub true_incs: bool,
    pub fields_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Tab state
// ---------------------------------------------------------------------------

pub struct FieldTabState {
    pub view: FieldViewState,
    pub new_form: NewFieldForm,
    pub image_viewer: ImageViewer,
    pub solve_state: Arc<Mutex<FieldSolveState>>,
    pub pending_save_name: Option<String>,
}

impl FieldTabState {
    pub fn new() -> Self {
        Self {
            view: FieldViewState::new(),
            new_form: NewFieldForm::default(),
            image_viewer: ImageViewer::new(),
            solve_state: Arc::new(Mutex::new(FieldSolveState::new())),
            pending_save_name: None,
        }
    }

    pub fn is_solving(&self) -> bool {
        self.solve_state.lock().map(|s| s.running).unwrap_or(false)
    }

    pub fn check_solve_complete(
        &mut self,
    ) -> Option<Result<(FieldSolution, String), String>> {
        let mut state = self.solve_state.lock().unwrap();
        if state.running || state.result.is_none() {
            return None;
        }
        let result = state.result.take()?;
        let name = self.pending_save_name.take().unwrap_or_default();
        Some(result.map(|sol| (sol, name)))
    }

    pub fn spawn_solve(&mut self, params: FieldSpawnParams) {
        self.pending_save_name = Some(params.name.clone());
        let mut state = self.solve_state.lock().unwrap();
        state.running = true;
        state.progress = 0.0;
        state.message = "Starting\u{2026}".to_string();
        state.cancel = Arc::new(AtomicBool::new(false));
        state.result = None;
        let cancel = state.cancel.clone();
        drop(state);
        let shared = self.solve_state.clone();
        std::thread::spawn(move || run_solve(params, shared, cancel));
    }

    pub fn cancel_solve(&self) {
        if let Ok(state) = self.solve_state.lock() {
            state.cancel.store(true, Ordering::Relaxed);
        }
    }

    // -----------------------------------------------------------------------
    // Central panel
    // -----------------------------------------------------------------------

    pub fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        viewer_rect: egui::Rect,
        mode: crate::main_window::PaneMode,
        selected_path: Option<&Path>,
        images: &[PathBuf],
        cache: &mut TextureCache,
    ) -> Option<HoverInfo> {
        let solve_running = self.is_solving();

        if mode == crate::main_window::PaneMode::View {
            if self.view.loaded_path.as_deref() != selected_path {
                self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
                self.view.solution = selected_path.and_then(|p| {
                    geopyv_dev::io::load(p).ok().and_then(|obj| {
                        if let GeopyvObject::Field(s) = obj { Some(s) } else { None }
                    })
                });
                self.view.frame = 0;
                self.view.selected_particle = None;
            }
            self.view.clamp_frame();

            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                match &self.view.solution {
                    None => {
                        ui.painter()
                            .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                        ui.centered_and_justified(|ui| {
                            ui.label(
                                egui::RichText::new("Select a field from the list")
                                    .size(16.0)
                                    .color(ui.visuals().weak_text_color()),
                            );
                        });
                    }
                    Some(_) => {
                        // Show scatter for all plot types (VolTotals uses its own central view).
                        if self.view.plot_type == FieldPlotType::VolTotals {
                            let sol = self.view.solution.as_ref().unwrap();
                            show_vol_totals_plot(ui, sol);
                        } else {
                            let sol = self.view.solution.as_ref().unwrap();
                            let frame = self.view.frame;
                            let component = self.view.component;
                            let colormap = self.view.colormap;
                            let (vmin, vmax) = self.view.compute_range(sol);
                            let selected = self.view.selected_particle;

                            let response = show_scatter_plot(
                                ui,
                                sol,
                                frame,
                                component,
                                colormap,
                                vmin,
                                vmax,
                                selected,
                                self.view.plot_type == FieldPlotType::TimeSeries,
                            );

                            // Click-to-select in time-series mode.
                            if self.view.plot_type == FieldPlotType::TimeSeries {
                                if let Some(click_pos) = response {
                                    // Find nearest particle at frame 0.
                                    let mut best_idx: Option<usize> = None;
                                    let mut best_dist = f64::INFINITY;
                                    for (i, p) in sol.particles.iter().enumerate() {
                                        let f0 = frame.min(p.coordinates.nrows().saturating_sub(1));
                                        let px = p.coordinates[[f0, 0]];
                                        let py = p.coordinates[[f0, 1]];
                                        let dist = ((px - click_pos[0]).powi(2)
                                            + (py + click_pos[1]).powi(2))
                                        .sqrt();
                                        if dist < best_dist {
                                            best_dist = dist;
                                            best_idx = Some(i);
                                        }
                                    }
                                    if best_dist < 30.0 {
                                        self.view.selected_particle = best_idx;
                                    }
                                }
                            }
                        }
                    }
                }
            });
            return None;
        }

        // ----------------------------------------------------------------
        // New mode: reference image + boundary/exclusion drawing + grid.
        // ----------------------------------------------------------------
        if self.new_form.ref_image_idx.is_none() && !images.is_empty() {
            self.new_form.ref_image_idx = Some(0);
        }

        let image_path = self
            .new_form
            .ref_image_idx
            .and_then(|i| images.get(i))
            .cloned();

        let hover = if let Some(path) = image_path {
            let draw = if !solve_running {
                Some(&mut self.new_form.draw)
            } else {
                None
            };
            let hover = ui
                .allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    self.image_viewer.show(ui, &path, cache, draw)
                })
                .inner;

            if !solve_running {
                self.new_form.ensure_grid_current();
                if let Some(coord) = self.image_viewer.last_coord() {
                    let painter = ui.painter_at(viewer_rect);
                    for &pt in &self.new_form.cached_grid {
                        let screen = coord.to_screen(pt);
                        painter.circle_filled(
                            screen,
                            2.5,
                            egui::Color32::from_rgb(255, 200, 50),
                        );
                    }
                }
            }

            hover
        } else {
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                ui.painter()
                    .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("No images in project")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                });
            });
            None
        };

        if solve_running {
            show_progress_overlay(ui, viewer_rect, &self.solve_state);
        }

        hover
    }

    // -----------------------------------------------------------------------
    // Right pane — view mode (§10.5)
    // -----------------------------------------------------------------------

    pub fn show_right_view(
        &mut self,
        ui: &mut egui::Ui,
        selected_path: Option<&Path>,
        out_dir: &Path,
    ) {
        if self.view.loaded_path.as_deref() != selected_path {
            self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
            self.view.solution = selected_path.and_then(|p| {
                geopyv_dev::io::load(p).ok().and_then(|obj| {
                    if let GeopyvObject::Field(s) = obj { Some(s) } else { None }
                })
            });
            self.view.frame = 0;
            self.view.selected_particle = None;
        }

        // Clone so the borrow on self.view ends before we call &mut self methods.
        let sol_opt = self.view.solution.clone();
        let Some(sol) = sol_opt else {
            ui.label(
                egui::RichText::new("No field selected")
                    .size(15.0)
                    .color(ui.visuals().weak_text_color()),
            );
            return;
        };

        let n_particles = sol.particles.len();
        let n_incs = if n_particles > 0 {
            sol.particles[0].coordinates.nrows().saturating_sub(1)
        } else {
            0
        };

        // ---- Metadata ----
        section_header(ui, "Field");
        meta_row(ui, "Particles", &n_particles.to_string());
        meta_row(ui, "Increments", &n_incs.to_string());
        if !sol.vol_totals.is_empty() {
            meta_row(
                ui,
                "Vol total (final)",
                &format_sci(sol.vol_totals[sol.vol_totals.len() - 1]),
            );
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // ---- Plot type selector ----
        section_header(ui, "Display");
        ui.add_space(3.0);

        let old_type = self.view.plot_type;
        ui.radio_value(&mut self.view.plot_type, FieldPlotType::Scatter, "Scatter overlay");
        ui.radio_value(
            &mut self.view.plot_type,
            FieldPlotType::TimeSeries,
            "Time series — single particle",
        );
        ui.radio_value(
            &mut self.view.plot_type,
            FieldPlotType::VolTotals,
            "Volumetric totals vs increment",
        );
        if old_type != self.view.plot_type {
            self.view.selected_particle = None;
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        match self.view.plot_type {
            FieldPlotType::Scatter => {
                self.show_right_scatter_controls(ui, n_incs, &sol);
            }
            FieldPlotType::TimeSeries => {
                self.show_right_timeseries(ui, &sol);
            }
            FieldPlotType::VolTotals => {
                // Vol-totals plot shown in central pane.
            }
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // ---- Export buttons ----
        let field_name = selected_path
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "field".to_string());

        ui.horizontal(|ui| {
            if ui
                .add(egui::Button::new("Export PNG").min_size(egui::vec2(80.0, 24.0)))
                .clicked()
            {
                let frame = self.view.frame;
                let component = self.view.component;
                let colormap = self.view.colormap;
                let (vmin, vmax) = self.view.compute_range(&sol);
                let dest = out_dir.join(format!("{field_name}_frame{frame:03}.png"));
                if let Err(e) = export_scatter_png(&sol, frame, component, colormap, vmin, vmax, &dest) {
                    eprintln!("PNG export error: {e}");
                }
            }
            ui.add_space(4.0);
            if ui
                .add(egui::Button::new("Export CSV").min_size(egui::vec2(80.0, 24.0)))
                .clicked()
            {
                let dest = out_dir.join(format!("{field_name}.csv"));
                if let Err(e) = export_csv(&sol, &dest) {
                    eprintln!("CSV export error: {e}");
                }
            }
        });
    }

    fn show_right_scatter_controls(&mut self, ui: &mut egui::Ui, n_incs: usize, sol: &FieldSolution) {
        // Frame slider.
        section_header(ui, "Frame");
        ui.add_space(3.0);

        let max_frame = n_incs;
        let mut frame = self.view.frame as i32;
        let slider = egui::Slider::new(&mut frame, 0..=max_frame as i32)
            .text(format!("{} / {}", self.view.frame, max_frame));
        if ui.add(slider).changed() {
            self.view.frame = frame as usize;
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Component.
        section_header(ui, "Component");
        ui.add_space(3.0);
        egui::ComboBox::from_id_salt("field_component")
            .selected_text(self.view.component.label())
            .width(140.0)
            .show_ui(ui, |ui| {
                for comp in [
                    FieldComponent::Exx,
                    FieldComponent::Eyy,
                    FieldComponent::Exy,
                    FieldComponent::E1,
                    FieldComponent::E2,
                    FieldComponent::VolStrain,
                ] {
                    ui.selectable_value(&mut self.view.component, comp, comp.label());
                }
            });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Colormap.
        section_header(ui, "Colormap");
        ui.add_space(3.0);
        egui::ComboBox::from_id_salt("field_colormap")
            .selected_text(self.view.colormap.label())
            .width(100.0)
            .show_ui(ui, |ui| {
                for &cmap in ColormapType::ALL {
                    ui.selectable_value(&mut self.view.colormap, cmap, cmap.label());
                }
            });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Range.
        section_header(ui, "Range");
        ui.add_space(3.0);
        ui.checkbox(&mut self.view.range_auto, "Auto");

        if !self.view.range_auto {
            let (vmin, vmax) = self.view.compute_range(sol);
            if self.view.range_min_text.parse::<f64>().is_err() {
                self.view.range_min_text = format!("{:.4e}", vmin);
            }
            if self.view.range_max_text.parse::<f64>().is_err() {
                self.view.range_max_text = format!("{:.4e}", vmax);
            }

            egui::Grid::new("field_range_grid")
                .num_columns(2)
                .spacing([6.0, 4.0])
                .show(ui, |ui| {
                    ui.label(lbl("Min:"));
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.view.range_min_text)
                            .desired_width(80.0),
                    );
                    if r.changed() {
                        if let Ok(v) = self.view.range_min_text.trim().parse::<f64>() {
                            self.view.range_min = v;
                        }
                    }
                    ui.end_row();
                    ui.label(lbl("Max:"));
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.view.range_max_text)
                            .desired_width(80.0),
                    );
                    if r.changed() {
                        if let Ok(v) = self.view.range_max_text.trim().parse::<f64>() {
                            self.view.range_max = v;
                        }
                    }
                    ui.end_row();
                });
        }
    }

    fn show_right_timeseries(&mut self, ui: &mut egui::Ui, sol: &FieldSolution) {
        match self.view.selected_particle {
            None => {
                ui.label(
                    egui::RichText::new("Click a particle on the scatter to select")
                        .size(16.0)
                        .color(ui.visuals().weak_text_color()),
                );
            }
            Some(idx) => {
                if let Some(p) = sol.particles.get(idx) {
                    ui.label(
                        egui::RichText::new(format!("Particle {idx}"))
                            .size(16.0)
                            .color(ui.visuals().text_color()),
                    );
                    ui.add_space(4.0);

                    // Show time-series of strain vs increment.
                    let n = p.strains.nrows();
                    let exx: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, p.strains[[i, 0]]]).collect();
                    let eyy: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, p.strains[[i, 1]]]).collect();

                    let available_h = ui.available_height().min(200.0).max(80.0);
                    Plot::new("field_ts_plot")
                        .legend(Legend::default())
                        .height(available_h)
                        .x_axis_label("Increment")
                        .y_axis_label("Strain")
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new(PlotPoints::from(exx))
                                    .name("\u{03b5}_xx")
                                    .color(egui::Color32::from_rgb(70, 150, 255)),
                            );
                            plot_ui.line(
                                Line::new(PlotPoints::from(eyy))
                                    .name("\u{03b5}_yy")
                                    .color(egui::Color32::from_rgb(255, 130, 70)),
                            );
                        });
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Right pane — new mode (§9.6)
    // -----------------------------------------------------------------------

    pub fn show_right_new(
        &mut self,
        ui: &mut egui::Ui,
        images: &[PathBuf],
        sequences: &[PathBuf],
    ) -> Option<FieldSpawnParams> {
        if self.is_solving() {
            let (progress, message) = {
                let s = self.solve_state.lock().unwrap();
                (s.progress, s.message.clone())
            };
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Solving field\u{2026}")
                    .size(16.0)
                    .color(ui.visuals().text_color()),
            );
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(&message)
                    .size(16.0)
                    .color(ui.visuals().weak_text_color()),
            );
            ui.add_space(8.0);
            let available = ui.available_width();
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(available, 16.0), egui::Sense::hover());
            ui.painter().rect_filled(
                rect,
                egui::CornerRadius::same(3),
                ui.visuals().widgets.inactive.bg_fill,
            );
            if progress > 0.0 {
                let fill = egui::Rect::from_min_size(
                    rect.min,
                    egui::vec2(rect.width() * progress, rect.height()),
                );
                ui.painter().rect_filled(
                    fill,
                    egui::CornerRadius::same(3),
                    egui::Color32::from_rgb(70, 130, 220),
                );
            }
            ui.add_space(12.0);
            if ui
                .add(egui::Button::new("Cancel").min_size(egui::vec2(60.0, 26.0)))
                .clicked()
            {
                self.cancel_solve();
            }
            return None;
        }

        let form = &mut self.new_form;
        let mut spawn: Option<FieldSpawnParams> = None;

        egui::Grid::new("field_new_top_grid")
            .num_columns(2)
            .spacing([8.0, 5.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(lbl("Name:"));
                ui.add(egui::TextEdit::singleline(&mut form.name).desired_width(150.0));
                ui.end_row();

                ui.label(lbl("Sequence:"));
                let seq_label = form
                    .seq_idx
                    .and_then(|i| sequences.get(i))
                    .and_then(|p| p.file_stem())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "\u{2014}".to_string());
                egui::ComboBox::from_id_salt("field_seq")
                    .selected_text(&seq_label)
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        for (i, p) in sequences.iter().enumerate() {
                            let name = p
                                .file_stem()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            ui.selectable_value(&mut form.seq_idx, Some(i), name);
                        }
                    });
                ui.end_row();

                ui.label(lbl("Ref image:"));
                let img_label = form
                    .ref_image_idx
                    .and_then(|i| images.get(i))
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "\u{2014}".to_string());
                egui::ComboBox::from_id_salt("field_ref_img")
                    .selected_text(&img_label)
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        for (i, p) in images.iter().enumerate() {
                            let name = p
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            ui.selectable_value(&mut form.ref_image_idx, Some(i), name);
                        }
                    });
                ui.end_row();
            });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        section_header(ui, "Field Boundary");
        ui.label(
            egui::RichText::new("(independent of mesh boundary)")
                .size(16.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            let boundary_active = form.draw.mode == Some(ActiveDrawMode::Boundary);
            let btn_b = egui::Button::new(
                egui::RichText::new(if boundary_active {
                    "Drawing\u{2026}"
                } else {
                    "Boundary \u{25b6}"
                })
                .size(16.0),
            )
            .selected(boundary_active);
            if ui.add(btn_b).clicked() {
                if boundary_active {
                    form.draw.cancel();
                } else {
                    form.draw.start(ActiveDrawMode::Boundary);
                }
            }

            let can_exclusion = form.draw.boundary_ok();
            let exclusion_active = form.draw.mode == Some(ActiveDrawMode::Exclusion);
            let btn_e = egui::Button::new(
                egui::RichText::new(if exclusion_active {
                    "Drawing\u{2026}"
                } else {
                    "Exclusion \u{25b6}"
                })
                .size(16.0),
            )
            .selected(exclusion_active);
            if ui.add_enabled(can_exclusion, btn_e).clicked() {
                if exclusion_active {
                    form.draw.cancel();
                } else {
                    form.draw.start(ActiveDrawMode::Exclusion);
                }
            }
        });

        ui.add_space(3.0);
        if form.draw.boundary_ok() {
            ui.label(
                egui::RichText::new("\u{2713} Boundary drawn")
                    .size(15.0)
                    .color(egui::Color32::from_rgb(80, 200, 80)),
            );
        } else {
            ui.label(
                egui::RichText::new("No boundary")
                    .size(15.0)
                    .color(ui.visuals().weak_text_color()),
            );
        }
        if !form.draw.exclusions.is_empty() {
            ui.label(
                egui::RichText::new(format!("{} exclusion(s)", form.draw.exclusions.len()))
                    .size(15.0)
                    .color(ui.visuals().weak_text_color()),
            );
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        section_header(ui, "Grid");
        ui.add_space(3.0);

        form.ensure_grid_current();

        let mut spacing_changed = false;
        egui::Grid::new("field_grid_config")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(lbl("Spacing (px):"));
                let r = ui
                    .add(egui::TextEdit::singleline(&mut form.spacing_text).desired_width(80.0));
                if r.changed() {
                    if let Ok(v) = form.spacing_text.trim().parse::<f32>() {
                        if v > 0.0 {
                            form.spacing = v;
                            spacing_changed = true;
                        }
                    }
                }
                ui.end_row();

                ui.label(lbl("Particles:"));
                ui.label(egui::RichText::new(form.cached_grid.len().to_string()).size(15.0));
                ui.end_row();
            });

        if spacing_changed {
            form.ensure_grid_current();
        }

        ui.add_space(4.0);
        if ui.button("Regenerate Grid").clicked() {
            form.invalidate_grid();
            form.ensure_grid_current();
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        section_header(ui, "Mode");
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            ui.radio_value(&mut form.lagrangian, true, "Lagrangian");
            ui.radio_value(&mut form.lagrangian, false, "Eulerian");
        });

        ui.add_space(4.0);
        egui::Grid::new("field_params_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(lbl("Depth:"));
                let r = ui
                    .add(egui::TextEdit::singleline(&mut form.depth_text).desired_width(80.0));
                if r.changed() {
                    if let Ok(v) = form.depth_text.trim().parse::<f64>() {
                        form.depth = v;
                    }
                }
                ui.end_row();

                ui.label(lbl("Factor:"));
                let r = ui
                    .add(egui::TextEdit::singleline(&mut form.factor_text).desired_width(80.0));
                if r.changed() {
                    if let Ok(v) = form.factor_text.trim().parse::<f64>() {
                        form.factor = v;
                    }
                }
                ui.end_row();
            });

        ui.add_space(3.0);
        ui.checkbox(&mut form.true_incs, "True increments (logarithmic strain)");

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        if let Some(err) = &form.form_error.clone() {
            ui.label(
                egui::RichText::new(err)
                    .size(15.0)
                    .color(ui.visuals().error_fg_color),
            );
            ui.add_space(4.0);
        }

        let can_run = form.can_run(sequences);
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            if ui
                .add_enabled(can_run, egui::Button::new("Run").min_size(egui::vec2(60.0, 26.0)))
                .clicked()
            {
                spawn = Some(FieldSpawnParams {
                    name: form.name.trim().to_string(),
                    sequence_path: sequences[form.seq_idx.unwrap()].clone(),
                    boundary: form.draw.boundary.as_ref().map(polygon_array).unwrap_or_else(|| Array2::zeros((0, 2))),
                    exclusions: form.draw.exclusions.iter().map(polygon_array).collect(),
                    spacing: form.spacing as f64,
                    track: form.lagrangian,
                    depth: form.depth,
                    factor: form.factor,
                    true_incs: form.true_incs,
                    fields_dir: PathBuf::new(),
                });
                form.form_error = None;
            }
        });

        spawn
    }

    // -----------------------------------------------------------------------
    // Save / Save As helpers (called from main_window)
    // -----------------------------------------------------------------------

    pub fn action_save(&self, out_dir: &Path, selected_path: Option<&Path>) {
        let Some(sol) = &self.view.solution else { return };
        let frame = self.view.frame;
        let component = self.view.component;
        let colormap = self.view.colormap;
        let (vmin, vmax) = self.view.compute_range(sol);
        let stem = selected_path
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "field".to_string());
        let dest = out_dir.join(format!("{stem}_frame{frame:03}.png"));
        if let Err(e) = export_scatter_png(sol, frame, component, colormap, vmin, vmax, &dest) {
            eprintln!("Save error: {e}");
        }
    }

    pub fn action_save_as(&self, selected_path: Option<&Path>) {
        let Some(sol) = &self.view.solution else { return };
        let frame = self.view.frame;
        let component = self.view.component;
        let colormap = self.view.colormap;
        let (vmin, vmax) = self.view.compute_range(sol);
        let stem = selected_path
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "field".to_string());

        let Some(dest) = rfd::FileDialog::new()
            .set_title("Save as PNG")
            .set_file_name(format!("{stem}_frame{frame:03}.png"))
            .add_filter("PNG", &["png"])
            .save_file()
        else {
            return;
        };

        if let Err(e) = export_scatter_png(sol, frame, component, colormap, vmin, vmax, &dest) {
            eprintln!("Save As error: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Scatter plot (egui_plot, central pane)
// Returns the clicked plot-position if a click occurred.
// ---------------------------------------------------------------------------

fn show_scatter_plot(
    ui: &mut egui::Ui,
    sol: &FieldSolution,
    frame: usize,
    component: FieldComponent,
    colormap: ColormapType,
    vmin: f64,
    vmax: f64,
    selected_particle: Option<usize>,
    clickable: bool,
) -> Option<[f64; 2]> {
    let mut clicked_pos: Option<[f64; 2]> = None;

    Plot::new("field_scatter_view")
        .x_axis_label("x (px)")
        .y_axis_label("y (px, \u{2193})")
        .data_aspect(1.0)
        .show_grid(false)
        .legend(Legend::default())
        .show(ui, |plot_ui| {
            for (i, particle) in sol.particles.iter().enumerate() {
                let f = frame.min(particle.coordinates.nrows().saturating_sub(1));
                let x = particle.coordinates[[f, 0]];
                let y = -particle.coordinates[[f, 1]]; // flip y for image coords
                let val = component.value_at(particle, frame);
                let color = colormap::map_value(val, vmin, vmax, colormap);

                let is_selected = selected_particle == Some(i);
                let radius: f32 = if is_selected { 7.0 } else { 4.5 };
                let outline_color = if is_selected {
                    egui::Color32::WHITE
                } else {
                    color
                };

                if is_selected {
                    plot_ui.points(
                        Points::new([x, y])
                            .radius(radius + 2.5)
                            .color(egui::Color32::WHITE)
                            .name(""),
                    );
                }
                plot_ui.points(
                    Points::new([x, y])
                        .radius(radius)
                        .color(if is_selected { color } else { outline_color })
                        .name(""),
                );
            }

            if clickable {
                if let Some(pos) = plot_ui.response().interact_pointer_pos() {
                    let plot_pos = plot_ui.plot_from_screen(pos);
                    if plot_ui.response().clicked() {
                        clicked_pos = Some([plot_pos.x, plot_pos.y]);
                    }
                }
            }
        });

    clicked_pos
}

// ---------------------------------------------------------------------------
// Volumetric totals plot (central pane)
// ---------------------------------------------------------------------------

fn show_vol_totals_plot(ui: &mut egui::Ui, sol: &FieldSolution) {
    let n = sol.vol_totals.len();
    let points: Vec<[f64; 2]> = (0..n)
        .map(|i| [i as f64, sol.vol_totals[i]])
        .collect();

    Plot::new("field_vol_totals")
        .legend(Legend::default())
        .x_axis_label("Increment")
        .y_axis_label("Total volume")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(PlotPoints::from(points))
                    .name("Vol. total")
                    .color(egui::Color32::from_rgb(70, 150, 255)),
            );
        });
}

// ---------------------------------------------------------------------------
// Progress overlay
// ---------------------------------------------------------------------------

fn show_progress_overlay(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    state: &Arc<Mutex<FieldSolveState>>,
) {
    let (progress, message) = {
        let s = state.lock().unwrap();
        (s.progress, s.message.clone())
    };

    let bar_h = 24.0;
    let pad = 20.0;
    let bar_w = rect.width() - pad * 2.0;
    let bar_y = rect.center().y - bar_h * 0.5;
    let bar_rect = egui::Rect::from_min_size(
        egui::pos2(rect.min.x + pad, bar_y),
        egui::vec2(bar_w, bar_h),
    );

    ui.painter()
        .rect_filled(rect, 0.0, egui::Color32::from_rgba_premultiplied(0, 0, 0, 160));
    ui.painter().rect_filled(
        bar_rect,
        egui::CornerRadius::same(4),
        egui::Color32::from_rgb(40, 40, 48),
    );
    if progress > 0.0 {
        let fill = egui::Rect::from_min_size(
            bar_rect.min,
            egui::vec2(bar_rect.width() * progress, bar_rect.height()),
        );
        ui.painter().rect_filled(
            fill,
            egui::CornerRadius::same(4),
            egui::Color32::from_rgb(70, 130, 220),
        );
    }
    if !message.is_empty() {
        ui.painter().text(
            egui::pos2(bar_rect.min.x, bar_rect.max.y + 6.0),
            egui::Align2::LEFT_TOP,
            &message,
            egui::FontId::new(15.0, egui::FontFamily::Proportional),
            egui::Color32::from_rgb(200, 200, 200),
        );
    }
}

// ---------------------------------------------------------------------------
// Background solve thread
// ---------------------------------------------------------------------------

fn set_progress(state: &Arc<Mutex<FieldSolveState>>, progress: f32, msg: &str) {
    if let Ok(mut s) = state.lock() {
        s.progress = progress;
        s.message = msg.to_string();
    }
}

fn set_error(state: &Arc<Mutex<FieldSolveState>>, msg: String) {
    if let Ok(mut s) = state.lock() {
        s.result = Some(Err(msg));
        s.running = false;
        s.progress = 0.0;
    }
}

fn run_solve(
    params: FieldSpawnParams,
    state: Arc<Mutex<FieldSolveState>>,
    cancel: Arc<AtomicBool>,
) {
    set_progress(&state, 0.05, "Loading sequence\u{2026}");

    let seq_sol = match geopyv_dev::io::load(&params.sequence_path) {
        Ok(GeopyvObject::Sequence(s)) => s,
        Ok(_) => {
            set_error(&state, "File is not a sequence".to_string());
            return;
        }
        Err(e) => {
            set_error(&state, format!("Could not load sequence: {e}"));
            return;
        }
    };

    if seq_sol.n_meshes() == 0 {
        set_error(&state, "Sequence contains no mesh solutions".to_string());
        return;
    }

    set_progress(&state, 0.1, "Building field\u{2026}");

    let depth = params.depth.max(f64::MIN_POSITIVE);
    let distribution = FieldDistribution::Grid {
        boundary_nodes: params.boundary,
        exclusion_nodes: params.exclusions,
        spacing: params.spacing,
    };
    let source = Arc::new(seq_sol);
    let mut field = match Field::new(source, distribution, params.track, depth) {
        Ok(f) => f,
        Err(e) => {
            set_error(&state, format!("Field init error: {e}"));
            return;
        }
    };

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() {
            s.running = false;
        }
        return;
    }

    set_progress(&state, 0.15, "Solving field\u{2026}");

    if let Err(e) = field.solve(params.factor, params.true_incs, None, geopyv_dev::particle::StrainMethod::default()) {
        set_error(&state, format!("Solve error: {e}"));
        return;
    }

    let solution = match field.solution() {
        Some(sol) => sol.clone(),
        None => {
            set_error(&state, "No solution after solve".to_string());
            return;
        }
    };

    set_progress(&state, 1.0, "Done");
    if let Ok(mut s) = state.lock() {
        s.result = Some(Ok(solution));
        s.running = false;
    }
}

// ---------------------------------------------------------------------------
// Export — PNG scatter
// ---------------------------------------------------------------------------

fn export_scatter_png(
    sol: &FieldSolution,
    frame: usize,
    component: FieldComponent,
    colormap: ColormapType,
    vmin: f64,
    vmax: f64,
    dest: &Path,
) -> Result<(), String> {
    if sol.particles.is_empty() {
        return Err("No particles to export".to_string());
    }

    // Compute bounding box of particle positions at this frame.
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for p in &sol.particles {
        let f = frame.min(p.coordinates.nrows().saturating_sub(1));
        let x = p.coordinates[[f, 0]];
        let y = p.coordinates[[f, 1]];
        if x < xmin { xmin = x; }
        if x > xmax { xmax = x; }
        if y < ymin { ymin = y; }
        if y > ymax { ymax = y; }
    }

    let margin = 20.0f64;
    xmin -= margin; ymin -= margin;
    xmax += margin; ymax += margin;

    let img_w = (xmax - xmin).ceil() as u32;
    let img_h = (ymax - ymin).ceil() as u32;
    if img_w == 0 || img_h == 0 {
        return Err("Degenerate particle extents".to_string());
    }

    let mut img = image::RgbImage::from_pixel(img_w, img_h, image::Rgb([20u8, 20u8, 22u8]));

    let radius = 3i32;
    for p in &sol.particles {
        let f = frame.min(p.coordinates.nrows().saturating_sub(1));
        let x = p.coordinates[[f, 0]];
        let y = p.coordinates[[f, 1]];
        let val = component.value_at(p, frame);
        let color = colormap::map_value(val, vmin, vmax, colormap);
        let [r, g, b, _] = color.to_array();

        let px = (x - xmin).round() as i32;
        let py = (y - ymin).round() as i32;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let ix = px + dx;
                    let iy = py + dy;
                    if ix >= 0 && iy >= 0 && (ix as u32) < img_w && (iy as u32) < img_h {
                        img.put_pixel(ix as u32, iy as u32, image::Rgb([r, g, b]));
                    }
                }
            }
        }
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    img.save(dest).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Export — CSV (long format per §14 Option B)
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

fn lbl(text: &str) -> egui::RichText {
    egui::RichText::new(text).size(16.0)
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

fn format_sci(v: f64) -> String {
    format!("{:.4e}", v)
}
