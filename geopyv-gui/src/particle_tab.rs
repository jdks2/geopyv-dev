use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};

use geopyv_dev::io::GeopyvObject;
use geopyv_dev::particle::{MeshData, Particle, ParticleConfig, ParticleSolution};

use crate::draw::{ActiveDrawMode, DrawState};
use crate::image_viewer::{HoverInfo, ImageViewer, TextureCache};

// ---------------------------------------------------------------------------
// Plot type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParticlePlotType {
    #[default]
    StrainPath,
    StrainVsIncrement,
    VolumetricVsIncrement,
    Trajectory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrainComponent {
    #[default]
    Exx,
    Eyy,
    Exy,
}

impl StrainComponent {
    fn label(self) -> &'static str {
        match self {
            Self::Exx => "\u{03b5}_xx",
            Self::Eyy => "\u{03b5}_yy",
            Self::Exy => "\u{03b5}_xy",
        }
    }

    fn col(self) -> usize {
        match self {
            Self::Exx => 0,
            Self::Eyy => 1,
            Self::Exy => 5,
        }
    }
}

// ---------------------------------------------------------------------------
// New particle form
// ---------------------------------------------------------------------------

pub struct NewParticleForm {
    pub name: String,
    pub seq_idx: Option<usize>,
    pub ref_image_idx: Option<usize>,
    pub draw: DrawState,
    pub lagrangian: bool,
    pub factor_text: String,
    pub factor: f64,
    pub true_incs: bool,
    pub form_error: Option<String>,
}

impl Default for NewParticleForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            seq_idx: None,
            ref_image_idx: None,
            draw: DrawState::new(),
            lagrangian: true,
            factor_text: "1.0".to_string(),
            factor: 1.0,
            true_incs: true,
            form_error: None,
        }
    }
}

impl NewParticleForm {
    fn can_run(&self, sequences: &[PathBuf]) -> bool {
        let name_ok = !self.name.trim().is_empty()
            && !self.name.contains('/')
            && !self.name.contains('\\');
        let seq_ok = self.seq_idx.map(|i| i < sequences.len()).unwrap_or(false);
        name_ok && seq_ok && self.draw.point_ok()
    }
}

// ---------------------------------------------------------------------------
// View state
// ---------------------------------------------------------------------------

pub struct ParticleViewState {
    pub loaded_path: Option<PathBuf>,
    pub solution: Option<ParticleSolution>,
    pub plot_type: ParticlePlotType,
    pub strain_component: StrainComponent,
}

impl ParticleViewState {
    fn new() -> Self {
        Self {
            loaded_path: None,
            solution: None,
            plot_type: ParticlePlotType::default(),
            strain_component: StrainComponent::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Background solve state
// ---------------------------------------------------------------------------

pub struct ParticleSolveState {
    pub running: bool,
    pub progress: f32,
    pub message: String,
    pub cancel: Arc<AtomicBool>,
    pub result: Option<Result<ParticleSolution, String>>,
}

impl ParticleSolveState {
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

pub struct ParticleSpawnParams {
    pub name: String,
    pub sequence_path: PathBuf,
    pub coord: [f64; 2],
    pub track: bool,
    pub factor: f64,
    pub true_incs: bool,
    pub particles_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Tab state
// ---------------------------------------------------------------------------

pub struct ParticleTabState {
    pub view: ParticleViewState,
    pub new_form: NewParticleForm,
    pub image_viewer: ImageViewer,
    pub solve_state: Arc<Mutex<ParticleSolveState>>,
    pub pending_save_name: Option<String>,
}

impl ParticleTabState {
    pub fn new() -> Self {
        Self {
            view: ParticleViewState::new(),
            new_form: NewParticleForm::default(),
            image_viewer: ImageViewer::new(),
            solve_state: Arc::new(Mutex::new(ParticleSolveState::new())),
            pending_save_name: None,
        }
    }

    pub fn is_solving(&self) -> bool {
        self.solve_state.lock().map(|s| s.running).unwrap_or(false)
    }

    pub fn check_solve_complete(
        &mut self,
    ) -> Option<Result<(ParticleSolution, String), String>> {
        let mut state = self.solve_state.lock().unwrap();
        if state.running || state.result.is_none() {
            return None;
        }
        let result = state.result.take()?;
        let name = self.pending_save_name.take().unwrap_or_default();
        Some(result.map(|sol| (sol, name)))
    }

    pub fn spawn_solve(&mut self, params: ParticleSpawnParams) {
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
            // Lazy-load solution.
            if self.view.loaded_path.as_deref() != selected_path {
                self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
                self.view.solution = selected_path.and_then(|p| {
                    geopyv_dev::io::load(p).ok().and_then(|obj| {
                        if let GeopyvObject::Particle(s) = obj { Some(s) } else { None }
                    })
                });
            }

            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                match &self.view.solution {
                    None => {
                        ui.painter()
                            .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                        ui.centered_and_justified(|ui| {
                            ui.label(
                                egui::RichText::new("Select a particle from the list")
                                    .size(14.0)
                                    .color(ui.visuals().weak_text_color()),
                            );
                        });
                    }
                    Some(_) => {
                        show_plot(ui, viewer_rect, self.view.plot_type, self.view.strain_component, self.view.solution.as_ref().unwrap());
                    }
                }
            });
            return None;
        }

        // New mode: reference image with PlacePoint.
        if self.new_form.ref_image_idx.is_none() && !images.is_empty() {
            self.new_form.ref_image_idx = Some(0);
        }

        let image_path = self.new_form.ref_image_idx
            .and_then(|i| images.get(i))
            .cloned();

        let hover = if let Some(path) = image_path {
            let draw = if !solve_running { Some(&mut self.new_form.draw) } else { None };
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                self.image_viewer.show(ui, &path, cache, draw)
            })
            .inner
        } else {
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                ui.painter()
                    .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("No images in project")
                            .size(14.0)
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
    // Right pane — view mode (§10.4)
    // -----------------------------------------------------------------------

    pub fn show_right_view(&mut self, ui: &mut egui::Ui, selected_path: Option<&Path>) {
        if self.view.loaded_path.as_deref() != selected_path {
            self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
            self.view.solution = selected_path.and_then(|p| {
                geopyv_dev::io::load(p).ok().and_then(|obj| {
                    if let GeopyvObject::Particle(s) = obj { Some(s) } else { None }
                })
            });
        }

        let Some(sol) = &self.view.solution else {
            ui.label(
                egui::RichText::new("No particle selected")
                    .size(13.0)
                    .color(ui.visuals().weak_text_color()),
            );
            return;
        };

        // Metadata.
        let n_incs = sol.strains.nrows();
        let final_row = n_incs.saturating_sub(1);

        section_header(ui, "Particle");
        meta_row(ui, "Increments", &n_incs.to_string());
        meta_row(ui, "Mesh order", &(sol.warps.ncols() / 6).to_string());
        if n_incs > 0 {
            meta_row(ui, "x\u{2080}", &format!("{:.1}", sol.coordinates[[0, 0]]));
            meta_row(ui, "y\u{2080}", &format!("{:.1}", sol.coordinates[[0, 1]]));
            meta_row(ui, "\u{03b5}_xx (final)", &format!("{:.4e}", sol.strains[[final_row, 0]]));
            meta_row(ui, "\u{03b5}_yy (final)", &format!("{:.4e}", sol.strains[[final_row, 1]]));
            meta_row(
                ui,
                "Vol. strain (final)",
                &format!("{:.4e}", sol.vol_strains[final_row]),
            );
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Plot type selector.
        section_header(ui, "Plot Type");
        ui.add_space(3.0);

        ui.radio_value(
            &mut self.view.plot_type,
            ParticlePlotType::StrainPath,
            "Strain path (\u{03b5}\u{2081} vs \u{03b5}\u{2082})",
        );
        ui.radio_value(
            &mut self.view.plot_type,
            ParticlePlotType::StrainVsIncrement,
            "Strain component vs increment",
        );

        if self.view.plot_type == ParticlePlotType::StrainVsIncrement {
            ui.add_space(2.0);
            egui::ComboBox::from_id_salt("particle_strain_comp")
                .selected_text(self.view.strain_component.label())
                .width(90.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.view.strain_component,
                        StrainComponent::Exx,
                        StrainComponent::Exx.label(),
                    );
                    ui.selectable_value(
                        &mut self.view.strain_component,
                        StrainComponent::Eyy,
                        StrainComponent::Eyy.label(),
                    );
                    ui.selectable_value(
                        &mut self.view.strain_component,
                        StrainComponent::Exy,
                        StrainComponent::Exy.label(),
                    );
                });
        }

        ui.radio_value(
            &mut self.view.plot_type,
            ParticlePlotType::VolumetricVsIncrement,
            "Volumetric strain vs increment",
        );
        ui.radio_value(
            &mut self.view.plot_type,
            ParticlePlotType::Trajectory,
            "Trajectory (x, y vs increment)",
        );

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.add_enabled(
                false,
                egui::Button::new("Export PNG").min_size(egui::vec2(80.0, 24.0)),
            );
            ui.add_space(4.0);
            ui.add_enabled(
                false,
                egui::Button::new("Export CSV").min_size(egui::vec2(80.0, 24.0)),
            );
        });
    }

    // -----------------------------------------------------------------------
    // Right pane — new mode (§9.5)
    // -----------------------------------------------------------------------

    pub fn show_right_new(
        &mut self,
        ui: &mut egui::Ui,
        images: &[PathBuf],
        sequences: &[PathBuf],
    ) -> Option<ParticleSpawnParams> {
        if self.is_solving() {
            let (progress, message) = {
                let s = self.solve_state.lock().unwrap();
                (s.progress, s.message.clone())
            };
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Solving particle\u{2026}")
                    .size(14.0)
                    .color(ui.visuals().text_color()),
            );
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(&message)
                    .size(12.0)
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
        let mut spawn: Option<ParticleSpawnParams> = None;

        // Name, sequence, reference image.
        egui::Grid::new("particle_new_grid")
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
                egui::ComboBox::from_id_salt("particle_seq")
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
                egui::ComboBox::from_id_salt("particle_ref_img")
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

        // Coordinate placement.
        section_header(ui, "Coordinate");
        ui.add_space(3.0);

        let pt_active = form.draw.mode == Some(ActiveDrawMode::Point);
        let btn_pt = egui::Button::new(
            egui::RichText::new(if pt_active { "Placing\u{2026}" } else { "Place point \u{25b6}" })
                .size(12.0),
        )
        .selected(pt_active);
        if ui.add(btn_pt).clicked() {
            if pt_active {
                form.draw.cancel();
            } else {
                form.draw.start(ActiveDrawMode::Point);
            }
        }
        ui.add_space(3.0);
        match form.draw.point {
            Some(p) => {
                ui.label(
                    egui::RichText::new(format!("x: {:.1}   y: {:.1}", p.x, p.y))
                        .size(12.0)
                        .color(egui::Color32::from_rgb(100, 200, 100)),
                );
            }
            None => {
                ui.label(
                    egui::RichText::new("Click image to place")
                        .size(12.0)
                        .color(ui.visuals().weak_text_color()),
                );
            }
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Mode and parameters.
        section_header(ui, "Mode");
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            ui.radio_value(&mut form.lagrangian, true, "Lagrangian");
            ui.radio_value(&mut form.lagrangian, false, "Eulerian");
        });

        ui.add_space(4.0);
        egui::Grid::new("particle_params_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
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
                    .size(11.0)
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
                let pt = form.draw.point.unwrap();
                spawn = Some(ParticleSpawnParams {
                    name: form.name.trim().to_string(),
                    sequence_path: sequences[form.seq_idx.unwrap()].clone(),
                    coord: [pt.x as f64, pt.y as f64],
                    track: form.lagrangian,
                    factor: form.factor,
                    true_incs: form.true_incs,
                    particles_dir: PathBuf::new(), // filled in main_window
                });
                form.form_error = None;
            }
        });

        spawn
    }
}

// ---------------------------------------------------------------------------
// Central plot renderer
// ---------------------------------------------------------------------------

fn show_plot(
    ui: &mut egui::Ui,
    _rect: egui::Rect,
    plot_type: ParticlePlotType,
    strain_component: StrainComponent,
    sol: &ParticleSolution,
) {
    match plot_type {
        ParticlePlotType::StrainPath => {
            let points: Vec<[f64; 2]> = (0..sol.strains.nrows())
                .map(|i| {
                    let eps_xx = sol.strains[[i, 0]];
                    let eps_yy = sol.strains[[i, 1]];
                    let eps_xy = sol.strains[[i, 5]];
                    let mean = (eps_xx + eps_yy) / 2.0;
                    let dev =
                        (((eps_xx - eps_yy) / 2.0).powi(2) + eps_xy.powi(2)).sqrt();
                    [mean + dev, mean - dev]
                })
                .collect();
            Plot::new("particle_strain_path")
                .legend(Legend::default())
                .x_axis_label("\u{03b5}\u{2081}")
                .y_axis_label("\u{03b5}\u{2082}")
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(PlotPoints::from(points))
                            .name("Strain path")
                            .color(egui::Color32::from_rgb(70, 150, 255)),
                    );
                });
        }

        ParticlePlotType::StrainVsIncrement => {
            let col = strain_component.col();
            let n = sol.strains.nrows();
            let points: Vec<[f64; 2]> = (0..n)
                .map(|i| [i as f64, sol.strains[[i, col]]])
                .collect();
            Plot::new("particle_strain_inc")
                .legend(Legend::default())
                .x_axis_label("Increment")
                .y_axis_label(strain_component.label())
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(PlotPoints::from(points))
                            .name(strain_component.label())
                            .color(egui::Color32::from_rgb(70, 150, 255)),
                    );
                });
        }

        ParticlePlotType::VolumetricVsIncrement => {
            let n = sol.vol_strains.len();
            let points: Vec<[f64; 2]> =
                (0..n).map(|i| [i as f64, sol.vol_strains[i]]).collect();
            Plot::new("particle_vol_strain")
                .legend(Legend::default())
                .x_axis_label("Increment")
                .y_axis_label("Vol. strain")
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(PlotPoints::from(points))
                            .name("Vol. strain")
                            .color(egui::Color32::from_rgb(70, 150, 255)),
                    );
                });
        }

        ParticlePlotType::Trajectory => {
            let n = sol.coordinates.nrows();
            let x_pts: Vec<[f64; 2]> =
                (0..n).map(|i| [i as f64, sol.coordinates[[i, 0]]]).collect();
            let y_pts: Vec<[f64; 2]> =
                (0..n).map(|i| [i as f64, sol.coordinates[[i, 1]]]).collect();
            Plot::new("particle_traj")
                .legend(Legend::default())
                .x_axis_label("Increment")
                .y_axis_label("Position (px)")
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(PlotPoints::from(x_pts))
                            .name("x")
                            .color(egui::Color32::from_rgb(70, 150, 255)),
                    );
                    plot_ui.line(
                        Line::new(PlotPoints::from(y_pts))
                            .name("y")
                            .color(egui::Color32::from_rgb(255, 130, 70)),
                    );
                });
        }
    }
}

// ---------------------------------------------------------------------------
// Progress overlay
// ---------------------------------------------------------------------------

fn show_progress_overlay(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    state: &Arc<Mutex<ParticleSolveState>>,
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
            egui::FontId::new(13.0, egui::FontFamily::Proportional),
            egui::Color32::from_rgb(200, 200, 200),
        );
    }
}

// ---------------------------------------------------------------------------
// Background solve thread
// ---------------------------------------------------------------------------

fn set_progress(state: &Arc<Mutex<ParticleSolveState>>, progress: f32, msg: &str) {
    if let Ok(mut s) = state.lock() {
        s.progress = progress;
        s.message = msg.to_string();
    }
}

fn set_error(state: &Arc<Mutex<ParticleSolveState>>, msg: String) {
    if let Ok(mut s) = state.lock() {
        s.result = Some(Err(msg));
        s.running = false;
        s.progress = 0.0;
    }
}

fn run_solve(
    params: ParticleSpawnParams,
    state: Arc<Mutex<ParticleSolveState>>,
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

    let n_meshes = seq_sol.mesh_solutions.len();
    if n_meshes == 0 {
        set_error(&state, "Sequence contains no mesh solutions".to_string());
        return;
    }

    let mesh_order = seq_sol.mesh_solutions[0].mesh_order;
    let p_len = 6 * mesh_order as usize;
    let initial_warp = vec![0.0f64; p_len];
    let inc_no = n_meshes + 1;

    let mut particle = match Particle::new(
        params.coord,
        &initial_warp,
        1.0,
        inc_no,
        mesh_order,
        params.track,
    ) {
        Ok(p) => p,
        Err(e) => {
            set_error(&state, format!("Particle init error: {e}"));
            return;
        }
    };

    set_progress(&state, 0.1, "Solving increments\u{2026}");
    let base = 0.1f32;
    let per_step = (1.0f32 - base) / n_meshes as f32;

    for (m, mesh_sol) in seq_sol.mesh_solutions.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            if let Ok(mut s) = state.lock() {
                s.running = false;
            }
            return;
        }

        set_progress(
            &state,
            base + per_step * m as f32,
            &format!("Step {} / {}", m + 1, n_meshes),
        );

        let mesh_data = MeshData {
            nodes: &mesh_sol.nodes,
            elements: &mesh_sol.elements,
            displacements: &mesh_sol.displacements,
            mesh_order: mesh_sol.mesh_order,
        };
        particle.solve_increment(m, &mesh_data, false);
    }

    let cfg = ParticleConfig {
        factor: params.factor,
        true_incs: params.true_incs,
    };
    let solution = particle.finalize(&cfg);

    set_progress(&state, 1.0, "Done");
    if let Ok(mut s) = state.lock() {
        s.result = Some(Ok(solution));
        s.running = false;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn lbl(text: &str) -> egui::RichText {
    egui::RichText::new(text).size(12.0)
}

fn section_header(ui: &mut egui::Ui, label: &str) {
    ui.label(
        egui::RichText::new(label)
            .size(11.0)
            .color(ui.visuals().weak_text_color()),
    );
    ui.add_space(2.0);
}

fn meta_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(format!("{label}:"))
                .size(12.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.label(egui::RichText::new(value).size(13.0));
    });
    ui.add_space(2.0);
}
