use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, Points};

use geopyv_dev::image::Image;
use geopyv_dev::io::GeopyvObject;
use geopyv_dev::subset::{Subset, SubsetSolution, MaskSummary};
use geopyv_dev::masks::{LocalMask, MaskShape};

use crate::draw::{ActiveDrawMode, ImageCoord};
use crate::image_viewer::{HoverInfo, ImageViewer, TextureCache};

// ---------------------------------------------------------------------------
// Solver configuration enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveMethod {
    Icgn,
    Fagn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsetOrder {
    First,
    Second,
}

// ---------------------------------------------------------------------------
// Plot mode for view state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsetPlotMode {
    Displacement,
    Inspect,
    Convergence,
}

impl Default for SubsetPlotMode {
    fn default() -> Self {
        SubsetPlotMode::Displacement
    }
}

// ---------------------------------------------------------------------------
// Solver config form state
// ---------------------------------------------------------------------------

pub struct SolverConfig {
    pub method: SolveMethod,
    pub order: SubsetOrder,
    pub max_norm_text: String,
    pub max_norm: f64,
    pub max_iterations_text: String,
    pub max_iterations: usize,
    pub zncc_tol_text: String,
    pub zncc_tol: f64,
    pub parse_error: Option<String>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            method: SolveMethod::Icgn,
            order: SubsetOrder::First,
            max_norm_text: "1e-5".to_string(),
            max_norm: 1e-5,
            max_iterations_text: "50".to_string(),
            max_iterations: 50,
            zncc_tol_text: "0.75".to_string(),
            zncc_tol: 0.75,
            parse_error: None,
        }
    }
}

impl SolverConfig {
    pub fn is_valid(&self) -> bool {
        self.parse_error.is_none() && self.max_norm > 0.0 && self.max_iterations > 0
    }
}

// ---------------------------------------------------------------------------
// New subset form state
// ---------------------------------------------------------------------------

pub struct NewSubsetForm {
    pub name: String,
    pub ref_idx: Option<usize>,
    pub target_idx: Option<usize>,
    pub template_shape: MaskShape,
    pub template_size_text: String,
    pub template_size: u32,
    pub template_size_error: Option<String>,
    pub solver: SolverConfig,
    pub draw: crate::draw::DrawState,
    pub form_error: Option<String>,
}

impl Default for NewSubsetForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            ref_idx: None,
            target_idx: None,
            template_shape: MaskShape::Circle,
            template_size_text: "20".to_string(),
            template_size: 20,
            template_size_error: None,
            solver: SolverConfig::default(),
            draw: crate::draw::DrawState::new(),
            form_error: None,
        }
    }
}

impl NewSubsetForm {
    fn can_run(&self, images: &[PathBuf]) -> bool {
        let name_ok = !self.name.trim().is_empty()
            && !self.name.contains('/')
            && !self.name.contains('\\');
        let ref_ok = self.ref_idx.map(|i| i < images.len()).unwrap_or(false);
        let tar_ok = self.target_idx.map(|i| i < images.len()).unwrap_or(false);
        let template_ok = !self.template_size_text.trim().is_empty()
            && self.template_size > 0
            && self.template_size_error.is_none();
        let coord_ok = self.draw.point_ok();
        let solver_ok = self.solver.is_valid();
        name_ok && ref_ok && tar_ok && template_ok && coord_ok && solver_ok
    }
}

// ---------------------------------------------------------------------------
// View mode state
// ---------------------------------------------------------------------------

pub struct SubsetViewState {
    pub loaded_path: Option<PathBuf>,
    pub solution: Option<SubsetSolution>,
    pub active_plot: SubsetPlotMode,
    pub displacement_alpha: f32,
    /// Cached masked-crop texture for the Inspect panel; cleared when selection changes.
    pub inspect_texture: Option<egui::TextureHandle>,
}

impl SubsetViewState {
    fn new() -> Self {
        Self {
            loaded_path: None,
            solution: None,
            active_plot: SubsetPlotMode::default(),
            displacement_alpha: 0.5,
            inspect_texture: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Background solve state (shared with thread)
// ---------------------------------------------------------------------------

pub struct SubsetSolveState {
    pub running: bool,
    pub progress: f32,
    pub message: String,
    pub cancel: Arc<AtomicBool>,
    pub result: Option<Result<SubsetSolution, String>>,
}

impl SubsetSolveState {
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

pub struct SubsetSpawnParams {
    pub name: String,
    pub ref_path: PathBuf,
    pub target_path: PathBuf,
    pub template_shape: MaskShape,
    pub template_size: u32,
    pub coord: [f64; 2],
    pub method: SolveMethod,
    pub order: SubsetOrder,
    pub max_norm: f64,
    pub max_iterations: usize,
}

// ---------------------------------------------------------------------------
// Tab state
// ---------------------------------------------------------------------------

pub struct SubsetTabState {
    pub view: SubsetViewState,
    pub new_form: NewSubsetForm,
    pub image_viewer: ImageViewer,
    pub solve_state: Arc<Mutex<SubsetSolveState>>,
    pub pending_save_name: Option<String>,
    /// Path to write when the next egui screenshot arrives (set by save actions).
    pub pending_screenshot_path: Option<PathBuf>,
    /// Viewer rect from the most recent frame (used for screenshot cropping).
    pub last_viewer_rect: egui::Rect,
}

impl SubsetTabState {
    pub fn new() -> Self {
        Self {
            view: SubsetViewState::new(),
            new_form: NewSubsetForm::default(),
            image_viewer: ImageViewer::new(),
            solve_state: Arc::new(Mutex::new(SubsetSolveState::new())),
            pending_save_name: None,
            pending_screenshot_path: None,
            last_viewer_rect: egui::Rect::NOTHING,
        }
    }

    pub fn is_solving(&self) -> bool {
        self.solve_state.lock().map(|s| s.running).unwrap_or(false)
    }

    pub fn check_solve_complete(
        &mut self,
    ) -> Option<Result<(SubsetSolution, String), String>> {
        let mut state = self.solve_state.lock().unwrap();
        if state.running || state.result.is_none() {
            return None;
        }
        let result = state.result.take()?;
        let name = self.pending_save_name.take().unwrap_or_default();
        Some(result.map(|sol| (sol, name)))
    }

    pub fn spawn_solve(&mut self, params: SubsetSpawnParams) {
        self.pending_save_name = Some(params.name.clone());

        let mut state = self.solve_state.lock().unwrap();
        state.running = true;
        state.progress = 0.0;
        state.message = "Starting...".to_string();
        state.cancel = Arc::new(AtomicBool::new(false));
        state.result = None;
        let cancel = state.cancel.clone();
        drop(state);

        let shared = self.solve_state.clone();
        std::thread::spawn(move || {
            run_solve(params, shared, cancel);
        });
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
        self.last_viewer_rect = viewer_rect;

        // Lazy-load view solution when selected path changes.
        if mode == crate::main_window::PaneMode::View {
            let cached = self.view.loaded_path.as_deref();
            if cached != selected_path {
                self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
                self.view.solution = selected_path.and_then(|p| {
                    geopyv_dev::io::load(p).ok().and_then(|obj| {
                        if let GeopyvObject::Subset(s) = obj { Some(s) } else { None }
                    })
                });
                self.view.active_plot = SubsetPlotMode::Displacement;
                self.view.inspect_texture = None;
            }
        }

        // In Convergence mode, show the plot full-panel instead of an image.
        if mode == crate::main_window::PaneMode::View
            && self.view.active_plot == SubsetPlotMode::Convergence
        {
            if let Some(sol) = self.view.solution.as_ref() {
                return self.show_convergence_panel(ui, viewer_rect, sol);
            }
        }

        // In Inspect mode, show masked reference crop with quality metrics.
        if mode == crate::main_window::PaneMode::View
            && self.view.active_plot == SubsetPlotMode::Inspect
        {
            if let Some(sol) = self.view.solution.as_ref() {
                let sol_clone = sol.clone();
                return self.show_inspect_panel(ui, viewer_rect, &sol_clone);
            }
        }

        // Determine which image path to show.
        let image_path: Option<PathBuf> = match mode {
            crate::main_window::PaneMode::New => {
                self.new_form
                    .ref_idx
                    .and_then(|i| images.get(i))
                    .cloned()
            }
            crate::main_window::PaneMode::View => {
                // Displacement mode: show reference image. Otherwise target.
                self.view.solution.as_ref().map(|s| {
                    if self.view.active_plot == SubsetPlotMode::Displacement {
                        s.ref_image.clone()
                    } else {
                        s.target_image.clone()
                    }
                })
            }
        };

        let hover = if let Some(ref path) = image_path {
            let draw = if mode == crate::main_window::PaneMode::New && !solve_running {
                Some(&mut self.new_form.draw)
            } else {
                None
            };
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                self.image_viewer.show(ui, path, cache, draw)
            })
            .inner
        } else {
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                ui.painter()
                    .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                ui.centered_and_justified(|ui| {
                    let msg = match mode {
                        crate::main_window::PaneMode::New => {
                            "Select a reference image to begin"
                        }
                        crate::main_window::PaneMode::View => "Select a subset from the list",
                    };
                    ui.label(
                        egui::RichText::new(msg)
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                });
            });
            None
        };

        // Overlays — only when image is visible.
        if image_path.is_some() {
            if let Some(coord) = self.image_viewer.last_coord() {
                match mode {
                    crate::main_window::PaneMode::New if !solve_running => {
                        // Semi-transparent template fill preview.
                        let size = self.new_form.template_size as f32;
                        let shape = self.new_form.template_shape.clone();
                        let screen_r = size * coord.zoom;

                        if let Some(pt) = self.new_form.draw.point {
                            let sp = coord.to_screen(pt);
                            paint_template_fill(ui.painter(), sp, screen_r, &shape, 80);
                            paint_template_outline_color(
                                ui.painter(), sp, screen_r, &shape,
                                egui::Color32::from_rgba_unmultiplied(255, 200, 0, 200),
                            );
                        }

                        // Ghost preview at cursor while in Point mode.
                        if self.new_form.draw.mode == Some(ActiveDrawMode::Point) {
                            if let Some(cursor) = ui.input(|i| i.pointer.hover_pos()) {
                                if viewer_rect.contains(cursor) {
                                    paint_template_fill(
                                        ui.painter(), cursor, screen_r, &shape, 45,
                                    );
                                }
                            }
                        }
                    }

                    crate::main_window::PaneMode::View => {
                        if let Some(sol) = self.view.solution.as_ref() {
                            let screen_pt = coord.to_screen(egui::pos2(
                                sol.coord[0] as f32,
                                sol.coord[1] as f32,
                            ));
                            if viewer_rect.contains(screen_pt) {
                                paint_crosshair_x(ui.painter(), screen_pt);
                                let screen_r = sol.mask.size as f32 * coord.zoom;

                                // Original template — hard boundary, no fill.
                                paint_template_outline_color(
                                    ui.painter(),
                                    screen_pt,
                                    screen_r,
                                    &sol.mask.shape,
                                    egui::Color32::from_rgba_unmultiplied(255, 200, 0, 200),
                                );

                                // In Displacement mode, also draw filled original + deformed.
                                if self.view.active_plot == SubsetPlotMode::Displacement {
                                    let alpha =
                                        (self.view.displacement_alpha * 255.0) as u8;
                                    paint_template_fill(
                                        ui.painter(),
                                        screen_pt,
                                        screen_r,
                                        &sol.mask.shape,
                                        alpha,
                                    );
                                    paint_deformed_template(
                                        ui.painter(),
                                        sol,
                                        &coord,
                                        viewer_rect,
                                    );
                                }
                            }
                        }
                    }

                    _ => {}
                }
            }
        }

        if solve_running {
            self.show_progress_overlay(ui, viewer_rect);
        }

        hover
    }

    fn show_convergence_panel(
        &self,
        ui: &mut egui::Ui,
        viewer_rect: egui::Rect,
        sol: &SubsetSolution,
    ) -> Option<HoverInfo> {
        ui.painter()
            .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));

        let history = &sol.result.history;
        if history.is_empty() {
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("No convergence history")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                });
            });
            return None;
        }

        let half = (viewer_rect.height() - 8.0) * 0.5;
        let norm_rect = egui::Rect::from_min_size(
            viewer_rect.min,
            egui::vec2(viewer_rect.width(), half),
        );
        let zncc_rect = egui::Rect::from_min_size(
            egui::pos2(viewer_rect.min.x, viewer_rect.min.y + half + 8.0),
            egui::vec2(viewer_rect.width(), half),
        );

        // Build data vecs — norm in log10 space so the plot behaves like semilogy.
        let norm_log: Vec<[f64; 2]> = history
            .iter()
            .map(|&(iter, norm, _, _)| [iter as f64, norm.abs().log10().max(-20.0)])
            .collect();
        let zncc_data: Vec<[f64; 2]> = history
            .iter()
            .map(|&(iter, _, zncc, _)| [iter as f64, zncc])
            .collect();

        let max_norm_log = sol.result.max_norm.log10();

        // ── Norm plot (top) ──────────────────────────────────────────────────
        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(norm_rect), |ui| {
            Plot::new("subset_conv_norm")
                .height(half)
                .y_axis_label("‖Δp‖")
                .y_axis_formatter(|gm, _| sig3((10.0_f64).powf(gm.value)))
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(norm_log.clone())
                            .name("norm")
                            .color(egui::Color32::from_rgb(100, 170, 255))
                            .width(2.0),
                    );
                    plot_ui.points(
                        Points::new(norm_log)
                            .radius(4.0)
                            .color(egui::Color32::from_rgb(100, 170, 255)),
                    );
                    plot_ui.hline(
                        HLine::new(max_norm_log)
                            .name("max_norm")
                            .color(egui::Color32::RED)
                            .style(LineStyle::Dashed { length: 8.0 })
                            .width(1.5),
                    );
                });
        });

        // ── C_ZNCC plot (bottom) ─────────────────────────────────────────────
        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(zncc_rect), |ui| {
            Plot::new("subset_conv_zncc")
                .height(half)
                .y_axis_label("C_ZNCC")
                .x_axis_label("Iteration")
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(zncc_data.clone())
                            .name("C_ZNCC")
                            .color(egui::Color32::from_rgb(100, 220, 130))
                            .width(2.0),
                    );
                    plot_ui.points(
                        Points::new(zncc_data)
                            .radius(4.0)
                            .color(egui::Color32::from_rgb(100, 220, 130)),
                    );
                    plot_ui.hline(
                        HLine::new(0.75_f64)
                            .name("threshold")
                            .color(egui::Color32::RED)
                            .style(LineStyle::Dashed { length: 8.0 })
                            .width(1.5),
                    );
                });
        });

        None
    }

    fn show_inspect_panel(
        &mut self,
        ui: &mut egui::Ui,
        viewer_rect: egui::Rect,
        sol: &SubsetSolution,
    ) -> Option<HoverInfo> {
        // Build the masked crop texture on first show (or after selection change).
        if self.view.inspect_texture.is_none() {
            self.view.inspect_texture = build_inspect_texture(sol, ui.ctx());
        }

        ui.painter()
            .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));

        let pad = 8.0;
        let caption_h = 22.0;
        let available_w = viewer_rect.width() - pad * 2.0;
        let available_h = viewer_rect.height() - caption_h - pad * 3.0;
        let side = available_w.min(available_h);

        let img_rect = egui::Rect::from_center_size(
            egui::pos2(
                viewer_rect.center().x,
                viewer_rect.min.y + pad + side * 0.5,
            ),
            egui::vec2(side, side),
        );

        if let Some(ref tex) = self.view.inspect_texture {
            ui.painter().image(
                tex.id(),
                img_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        } else {
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(img_rect), |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("Could not load image")
                            .color(ui.visuals().weak_text_color()),
                    );
                });
            });
        }

        let caption = format!(
            "Size: {} px  ·  σ_s = {:.2}  ·  SSSIG = {:.2E}",
            sol.mask.size,
            sol.std_dev,
            sol.sssig,
        );
        ui.painter().text(
            egui::pos2(viewer_rect.center().x, img_rect.max.y + pad),
            egui::Align2::CENTER_TOP,
            &caption,
            egui::FontId::new(14.0, egui::FontFamily::Proportional),
            egui::Color32::from_rgb(200, 200, 200),
        );

        None
    }

    fn show_progress_overlay(&self, ui: &mut egui::Ui, rect: egui::Rect) {
        let (progress, message) = {
            let s = self.solve_state.lock().unwrap();
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

        ui.painter().rect_filled(
            rect,
            0.0,
            egui::Color32::from_rgba_premultiplied(0, 0, 0, 160),
        );
        ui.painter().rect_filled(
            bar_rect,
            egui::CornerRadius::same(4),
            egui::Color32::from_rgb(40, 40, 48),
        );

        if progress > 0.0 {
            let fill_rect = egui::Rect::from_min_size(
                bar_rect.min,
                egui::vec2(bar_rect.width() * progress, bar_rect.height()),
            );
            ui.painter().rect_filled(
                fill_rect,
                egui::CornerRadius::same(4),
                egui::Color32::from_rgb(70, 130, 220),
            );
        }

        if !message.is_empty() {
            let msg_pos = egui::pos2(bar_rect.min.x, bar_rect.max.y + 6.0);
            ui.painter().text(
                msg_pos,
                egui::Align2::LEFT_TOP,
                &message,
                egui::FontId::new(15.0, egui::FontFamily::Proportional),
                egui::Color32::from_rgb(200, 200, 200),
            );
        }
    }

    // -----------------------------------------------------------------------
    // Right pane — view mode
    // -----------------------------------------------------------------------

    pub fn show_right_view(&mut self, ui: &mut egui::Ui, selected_path: Option<&Path>) {
        if self.view.loaded_path.as_deref() != selected_path {
            self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
            self.view.solution = selected_path.and_then(|p| {
                geopyv_dev::io::load(p).ok().and_then(|obj| {
                    if let GeopyvObject::Subset(s) = obj { Some(s) } else { None }
                })
            });
            self.view.active_plot = SubsetPlotMode::Displacement;
            self.view.inspect_texture = None;
        }

        let Some(sol) = self.view.solution.as_ref() else {
            ui.label(
                egui::RichText::new("No subset selected")
                    .size(15.0)
                    .color(ui.visuals().weak_text_color()),
            );
            return;
        };
        let sol = sol.clone();

        // ── Input ──────────────────────────────────────────────────────────
        pane_header(ui, "Input");

        let ref_name = sol.ref_image.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let tar_name = sol.target_image.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let shape_str = match sol.mask.shape {
            MaskShape::Circle => "Circle",
            MaskShape::Square => "Square",
        };

        meta_row(ui, "Reference", &ref_name);
        meta_row(ui, "Target", &tar_name);
        meta_row(
            ui,
            "Template",
            &format!("{}, size {} px", shape_str, sol.mask.size),
        );
        meta_row(
            ui,
            "Coord",
            &format!("x = {}, y = {}", sig3(sol.coord[0]), sig3(sol.coord[1])),
        );
        meta_row(ui, "Std dev", &sig3(sol.std_dev));
        meta_row(ui, "SSSIG", &sig3(sol.sssig));

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(4.0);

        // ── Results ────────────────────────────────────────────────────────
        pane_header(ui, "Results");

        section_header(ui, "Solving");
        meta_row(ui, "Iterations", &format!("{}", sol.result.iterations));
        meta_row(ui, "Converged", if sol.result.converged { "✓" } else { "✗" });

        ui.add_space(4.0);
        section_header(ui, "Values");
        meta_row(ui, "C_ZNCC", &sig3(sol.result.c_zncc));

        let p = &sol.result.p;
        if p.len() >= 6 {
            // First order: [u, v, du/dx, dv/dx, du/dy, dv/dy]
            let u        = p[0];
            let v        = p[1];
            let eps_xx   = p[2]; // du/dx
            let eps_yy   = p[5]; // dv/dy
            let gamma_xy = p[3] + p[4]; // dv/dx + du/dy
            meta_row(ui, "u", &sig3(u));
            meta_row(ui, "v", &sig3(v));
            meta_row(ui, "ε_xx", &sig3(eps_xx));
            meta_row(ui, "ε_yy", &sig3(eps_yy));
            meta_row(ui, "γ_xy", &sig3(gamma_xy));
        }
        if p.len() >= 12 {
            // Second-order: [u,v, ux,vx,uy,vy, uxx,vxx,uxy,vxy,uyy,vyy]
            meta_row(ui, "u_xx", &sig3(p[6]));
            meta_row(ui, "u_xy", &sig3(p[8]));
            meta_row(ui, "u_yy", &sig3(p[10]));
            meta_row(ui, "v_xx", &sig3(p[7]));
            meta_row(ui, "v_xy", &sig3(p[9]));
            meta_row(ui, "v_yy", &sig3(p[11]));
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(4.0);

        // ── Plot ───────────────────────────────────────────────────────────
        pane_header(ui, "Plot");

        ui.horizontal(|ui| {
            let active = self.view.active_plot;
            if ui
                .add(
                    egui::Button::new("Displacement")
                        .selected(active == SubsetPlotMode::Displacement),
                )
                .clicked()
            {
                self.view.active_plot = SubsetPlotMode::Displacement;
            }
            if ui
                .add(
                    egui::Button::new("Inspect")
                        .selected(active == SubsetPlotMode::Inspect),
                )
                .clicked()
            {
                self.view.active_plot = SubsetPlotMode::Inspect;
            }
            if ui
                .add(
                    egui::Button::new("Convergence")
                        .selected(active == SubsetPlotMode::Convergence),
                )
                .clicked()
            {
                self.view.active_plot = SubsetPlotMode::Convergence;
            }
        });

        if self.view.active_plot == SubsetPlotMode::Displacement {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Opacity:")
                        .size(15.0)
                        .color(ui.visuals().weak_text_color()),
                );
                ui.add(egui::Slider::new(&mut self.view.displacement_alpha, 0.0..=1.0).show_value(false));
            });
        }
    }

    // -----------------------------------------------------------------------
    // Right pane — new mode
    // -----------------------------------------------------------------------

    pub fn show_right_new(
        &mut self,
        ui: &mut egui::Ui,
        images: &[PathBuf],
    ) -> Option<SubsetSpawnParams> {
        let solving = self.is_solving();

        if solving {
            return self.show_solve_progress_panel(ui);
        }

        let form = &mut self.new_form;
        let mut spawn: Option<SubsetSpawnParams> = None;

        egui::Grid::new("subset_new_grid")
            .num_columns(2)
            .spacing([8.0, 5.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Name:")
                        .size(16.0)
                        .color(ui.visuals().weak_text_color()),
                );
                ui.add(egui::TextEdit::singleline(&mut form.name).desired_width(150.0));
                ui.end_row();

                ui.label(
                    egui::RichText::new("Ref image:")
                        .size(16.0)
                        .color(ui.visuals().weak_text_color()),
                );
                let ref_label = form
                    .ref_idx
                    .and_then(|i| images.get(i))
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "—".to_string());
                egui::ComboBox::from_id_salt("subset_ref")
                    .selected_text(&ref_label)
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        for (i, p) in images.iter().enumerate() {
                            let name = p
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            ui.selectable_value(&mut form.ref_idx, Some(i), name);
                        }
                    });
                ui.end_row();

                ui.label(
                    egui::RichText::new("Target image:")
                        .size(16.0)
                        .color(ui.visuals().weak_text_color()),
                );
                let tar_label = form
                    .target_idx
                    .and_then(|i| images.get(i))
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "—".to_string());
                egui::ComboBox::from_id_salt("subset_tar")
                    .selected_text(&tar_label)
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        for (i, p) in images.iter().enumerate() {
                            let name = p
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            ui.selectable_value(&mut form.target_idx, Some(i), name);
                        }
                    });
                ui.end_row();
            });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        section_header(ui, "Template");
        ui.add_space(4.0);

        egui::Grid::new("subset_template_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Shape:")
                        .size(16.0)
                        .color(ui.visuals().weak_text_color()),
                );
                ui.horizontal(|ui| {
                    ui.radio_value(&mut form.template_shape, MaskShape::Circle, "Circle");
                    ui.radio_value(&mut form.template_shape, MaskShape::Square, "Square");
                });
                ui.end_row();

                ui.label(
                    egui::RichText::new("Size (px):")
                        .size(16.0)
                        .color(ui.visuals().weak_text_color()),
                );
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut form.template_size_text).desired_width(80.0),
                );
                if resp.changed() {
                    let s = form.template_size_text.trim().to_string();
                    if s.is_empty() {
                        form.template_size_error = None;
                    } else {
                        match s.parse::<u32>() {
                            Ok(0) => {
                                form.template_size_error =
                                    Some("Size must be \u{2265} 1".to_string())
                            }
                            Ok(v) => {
                                form.template_size = v;
                                form.template_size_error = None;
                            }
                            Err(_) => {
                                form.template_size_error =
                                    Some("Enter a positive integer".to_string())
                            }
                        }
                    }
                }
                ui.end_row();
            });

        if let Some(err) = &form.template_size_error.clone() {
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(err)
                    .size(15.0)
                    .color(ui.visuals().error_fg_color),
            );
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        ui.label(
            egui::RichText::new("Solver")
                .size(16.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(4.0);

        {
            let s = &mut form.solver;
            egui::Grid::new("subset_solver_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .min_col_width(72.0)
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("Method:")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut s.method, SolveMethod::Icgn, "ICGN");
                        ui.radio_value(&mut s.method, SolveMethod::Fagn, "FAGN");
                    });
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Order:")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut s.order, SubsetOrder::First, "1");
                        ui.radio_value(&mut s.order, SubsetOrder::Second, "2");
                    });
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Max norm:")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut s.max_norm_text).desired_width(80.0),
                    );
                    if resp.changed() {
                        match s.max_norm_text.trim().parse::<f64>() {
                            Ok(v) if v > 0.0 => {
                                s.max_norm = v;
                                s.parse_error = None;
                            }
                            _ => {
                                s.parse_error =
                                    Some("max_norm must be a positive number".into())
                            }
                        }
                    }
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Max iters:")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut s.max_iterations_text)
                            .desired_width(80.0),
                    );
                    if resp.changed() {
                        match s.max_iterations_text.trim().parse::<usize>() {
                            Ok(v) if v > 0 => {
                                s.max_iterations = v;
                                s.parse_error = None;
                            }
                            _ => {
                                s.parse_error =
                                    Some("max_iterations must be a positive integer".into())
                            }
                        }
                    }
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("ZNCC tol:")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut s.zncc_tol_text).desired_width(80.0),
                    );
                    if resp.changed() {
                        if let Ok(v) = s.zncc_tol_text.trim().parse::<f64>() {
                            s.zncc_tol = v;
                        }
                    }
                    ui.end_row();
                });

            if let Some(err) = &s.parse_error.clone() {
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(err)
                        .size(15.0)
                        .color(ui.visuals().error_fg_color),
                );
            }
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        ui.label(
            egui::RichText::new("Coordinate")
                .size(16.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            let active = form.draw.mode == Some(ActiveDrawMode::Point);
            let btn_text = if active { "Placing…" } else { "Place point" };
            let btn =
                egui::Button::new(egui::RichText::new(btn_text).size(16.0)).selected(active);
            if ui.add(btn).clicked() {
                if active {
                    form.draw.cancel();
                } else {
                    form.draw.start(ActiveDrawMode::Point);
                }
            }
        });

        if let Some(pt) = form.draw.point {
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(format!("x: {:.1}  y: {:.1}", pt.x, pt.y))
                    .size(16.0)
                    .color(ui.visuals().text_color()),
            );
        }

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

        let can_run = form.can_run(images);
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            if ui
                .add_enabled(can_run, egui::Button::new("Run").min_size(egui::vec2(60.0, 26.0)))
                .clicked()
            {
                let pt = form.draw.point.unwrap();
                let ref_path = images[form.ref_idx.unwrap()].clone();
                let target_path = images[form.target_idx.unwrap()].clone();
                spawn = Some(SubsetSpawnParams {
                    name: form.name.trim().to_string(),
                    ref_path,
                    target_path,
                    template_shape: form.template_shape.clone(),
                    template_size: form.template_size,
                    coord: [pt.x as f64, pt.y as f64],
                    method: form.solver.method,
                    order: form.solver.order,
                    max_norm: form.solver.max_norm,
                    max_iterations: form.solver.max_iterations,
                });
                form.form_error = None;
            }
        });

        spawn
    }

    fn show_solve_progress_panel(&self, ui: &mut egui::Ui) -> Option<SubsetSpawnParams> {
        let (progress, message) = {
            let s = self.solve_state.lock().unwrap();
            (s.progress, s.message.clone())
        };

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Solving…")
                .size(16.0)
                .color(ui.visuals().text_color()),
        );
        ui.add_space(4.0);
        ui.label(egui::RichText::new(&message).size(16.0).color(ui.visuals().weak_text_color()));
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

        None
    }

    // -----------------------------------------------------------------------
    // Save helpers — called from main_window when Save/Save As... is clicked
    // -----------------------------------------------------------------------

    /// Request a screenshot of the central viewer for saving.
    /// `path` is stored; the caller handles the screenshot event.
    pub fn request_save(&mut self, path: PathBuf, ctx: &egui::Context) {
        self.pending_screenshot_path = Some(path);
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
    }

    /// Consume and write a screenshot event to the pending save path.
    /// Call this from main_window after `ctx.input()` yields a Screenshot event.
    pub fn handle_screenshot(
        &mut self,
        image: &egui::ColorImage,
    ) -> Option<Result<(), String>> {
        let path = self.pending_screenshot_path.take()?;
        let rect = self.last_viewer_rect;

        // Convert pixel rect to integer bounds (clamped to image dimensions).
        let img_w = image.size[0];
        let img_h = image.size[1];
        let x0 = (rect.min.x as usize).min(img_w);
        let y0 = (rect.min.y as usize).min(img_h);
        let x1 = (rect.max.x as usize).min(img_w);
        let y1 = (rect.max.y as usize).min(img_h);
        let crop_w = x1.saturating_sub(x0);
        let crop_h = y1.saturating_sub(y0);

        if crop_w == 0 || crop_h == 0 {
            return Some(Err("Viewer rect is empty — cannot save".into()));
        }

        // Build an RGBA buffer from the cropped region.
        let mut rgba: Vec<u8> = Vec::with_capacity(crop_w * crop_h * 4);
        for row in y0..y1 {
            for col in x0..x1 {
                let c = image.pixels[row * img_w + col];
                rgba.push(c.r());
                rgba.push(c.g());
                rgba.push(c.b());
                rgba.push(c.a());
            }
        }

        image::save_buffer_with_format(
            &path,
            &rgba,
            crop_w as u32,
            crop_h as u32,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .map_err(|e| format!("Save failed: {e}"))
        .into()
    }
}

// ---------------------------------------------------------------------------
// Background thread entry point
// ---------------------------------------------------------------------------

fn set_progress(state: &Arc<Mutex<SubsetSolveState>>, progress: f32, msg: &str) {
    if let Ok(mut s) = state.lock() {
        s.progress = progress;
        s.message = msg.to_string();
    }
}

fn set_error(state: &Arc<Mutex<SubsetSolveState>>, msg: String) {
    if let Ok(mut s) = state.lock() {
        s.result = Some(Err(msg));
        s.running = false;
        s.progress = 0.0;
    }
}

fn run_solve(
    params: SubsetSpawnParams,
    state: Arc<Mutex<SubsetSolveState>>,
    cancel: Arc<AtomicBool>,
) {
    let local_mask = match params.template_shape {
        MaskShape::Circle => LocalMask::circle(params.template_size as usize),
        MaskShape::Square => LocalMask::square(params.template_size as usize),
    };
    let local_mask = match local_mask {
        Ok(t) => t,
        Err(e) => {
            set_error(&state, format!("Mask build error: {e}"));
            return;
        }
    };

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() { s.running = false; }
        return;
    }

    set_progress(&state, 0.1, "Loading reference image…");
    let ref_img = match Image::from_file(&params.ref_path, 20) {
        Ok(img) => Arc::new(img),
        Err(e) => {
            set_error(&state, format!("Reference image error: {e}"));
            return;
        }
    };

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() { s.running = false; }
        return;
    }

    set_progress(&state, 0.3, "Loading target image…");
    let target_img = match Image::from_file(&params.target_path, 20) {
        Ok(img) => Arc::new(img),
        Err(e) => {
            set_error(&state, format!("Target image error: {e}"));
            return;
        }
    };

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() { s.running = false; }
        return;
    }

    let n_params = if matches!(params.order, SubsetOrder::First) { 6 } else { 12 };
    let subset_order = n_params / 6;

    set_progress(&state, 0.5, "Creating subset…");
    let subset = match Subset::new(
        params.coord,
        &local_mask,
        None,
        Arc::clone(&ref_img),
        Arc::clone(&target_img),
        subset_order,
    ) {
        Ok(s) => s,
        Err(e) => {
            set_error(&state, format!("Subset error: {e}"));
            return;
        }
    };

    // Capture quality metrics before solving.
    let n_px = subset.mask.n_px;
    let std_dev = if n_px > 0 {
        subset.delta_f / (n_px as f64).sqrt()
    } else {
        0.0
    };
    let sssig = subset.sssig;

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() { s.running = false; }
        return;
    }

    set_progress(&state, 0.7, "Solving…");
    let p_0 = vec![0.0f64; n_params];

    let solve_result = match params.method {
        SolveMethod::Icgn => {
            subset.solve_icgn(Some(&p_0), 0.75, params.max_norm, params.max_iterations)
        }
        SolveMethod::Fagn => {
            subset.solve_fagn(Some(&p_0), 0.75, params.max_norm, params.max_iterations)
        }
    };

    match solve_result {
        Ok(result) => {
            let solution = SubsetSolution {
                coord: params.coord,
                mask: MaskSummary {
                    shape: params.template_shape,
                    size: params.template_size as usize,
                    n_px,
                },
                ref_image: params.ref_path,
                target_image: params.target_path,
                result,
                std_dev,
                sssig,
                delta_f: subset.delta_f,
            };
            if let Ok(mut s) = state.lock() {
                s.result = Some(Ok(solution));
                s.progress = 1.0;
                s.message = "Done".to_string();
                s.running = false;
            }
        }
        Err(e) => {
            set_error(&state, format!("Solve error: {e}"));
        }
    }
}

// ---------------------------------------------------------------------------
// Deformed template overlay
// ---------------------------------------------------------------------------

fn paint_deformed_template(
    painter: &egui::Painter,
    sol: &SubsetSolution,
    coord: &ImageCoord,
    viewer_rect: egui::Rect,
) {
    let p = &sol.result.p;
    if p.len() < 6 {
        return;
    }
    let cx = sol.coord[0];
    let cy = sol.coord[1];
    let r = sol.mask.size as f64;
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(220, 60, 60));

    let screen_pts: Vec<egui::Pos2> = match sol.mask.shape {
        MaskShape::Circle => {
            const N: usize = 64;
            (0..=N)
                .map(|i| {
                    let theta = 2.0 * std::f64::consts::PI * i as f64 / N as f64;
                    let dx = r * theta.cos();
                    let dy = r * theta.sin();
                    // [u, v, ux, vx, uy, vy]: x' = x + u + ux*dx + uy*dy
                    let u = p[0] + p[2] * dx + p[4] * dy;
                    let v = p[1] + p[3] * dx + p[5] * dy;
                    coord.to_screen(egui::pos2(
                        (cx + dx + u) as f32,
                        (cy + dy + v) as f32,
                    ))
                })
                .collect()
        }
        MaskShape::Square => {
            let corners = [
                (-r, -r),
                ( r, -r),
                ( r,  r),
                (-r,  r),
                (-r, -r), // close
            ];
            corners
                .iter()
                .map(|&(dx, dy)| {
                    let u = p[0] + p[2] * dx + p[4] * dy;
                    let v = p[1] + p[3] * dx + p[5] * dy;
                    coord.to_screen(egui::pos2(
                        (cx + dx + u) as f32,
                        (cy + dy + v) as f32,
                    ))
                })
                .collect()
        }
    };

    // Only draw if at least one point is inside the viewer.
    if screen_pts.iter().any(|&p| viewer_rect.contains(p)) {
        for w in screen_pts.windows(2) {
            painter.line_segment([w[0], w[1]], stroke);
        }
    }
}

// ---------------------------------------------------------------------------
// Inspect patch renderer (standalone, no ImageViewer state)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Inspect texture builder
// ---------------------------------------------------------------------------

/// Build a masked grayscale crop of the reference image centred on the subset
/// coordinate.  For circle templates, pixels outside the radius have alpha=0.
fn build_inspect_texture(
    sol: &SubsetSolution,
    ctx: &egui::Context,
) -> Option<egui::TextureHandle> {
    let img = image::open(&sol.ref_image).ok()?.to_luma8();
    let img_w = img.width() as i64;
    let img_h = img.height() as i64;

    // coord[0] = x (column), coord[1] = y (row)
    let cx = sol.coord[0].round() as i64;
    let cy = sol.coord[1].round() as i64;
    let r = sol.mask.size as i64;
    let full = (2 * r + 1) as usize;

    let mut rgba = vec![0u8; full * full * 4];

    for dy in -r..=r {
        for dx in -r..=r {
            let px = cx + dx; // column
            let py = cy + dy; // row

            if px < 0 || py < 0 || px >= img_w || py >= img_h {
                continue; // stays transparent
            }

            let inside = match sol.mask.shape {
                MaskShape::Circle => dx * dx + dy * dy <= r * r,
                MaskShape::Square => true,
            };
            if !inside {
                continue; // stays transparent
            }

            let grey = img.get_pixel(px as u32, py as u32)[0];
            let lx = (dx + r) as usize;
            let ly = (dy + r) as usize;
            let base = (ly * full + lx) * 4;
            rgba[base] = grey;
            rgba[base + 1] = grey;
            rgba[base + 2] = grey;
            rgba[base + 3] = 255;
        }
    }

    let color_image =
        egui::ColorImage::from_rgba_unmultiplied([full, full], &rgba);
    Some(ctx.load_texture(
        "subset_inspect",
        color_image,
        egui::TextureOptions::default(),
    ))
}

// ---------------------------------------------------------------------------
// Painting helpers
// ---------------------------------------------------------------------------

const CROSSHAIR_ARM: f32 = 10.0;
const POINT_COLOR: egui::Color32 = egui::Color32::WHITE;

fn paint_crosshair_x(painter: &egui::Painter, center: egui::Pos2) {
    let d = CROSSHAIR_ARM / std::f32::consts::SQRT_2;
    let s = egui::Stroke::new(2.0, POINT_COLOR);
    painter.line_segment([center - egui::vec2(d, d), center + egui::vec2(d, d)], s);
    painter.line_segment([center - egui::vec2(d, -d), center + egui::vec2(d, -d)], s);
}

fn paint_template_outline_color(
    painter: &egui::Painter,
    center: egui::Pos2,
    screen_radius: f32,
    shape: &MaskShape,
    color: egui::Color32,
) {
    let stroke = egui::Stroke::new(1.5, color);
    match shape {
        MaskShape::Circle => {
            painter.circle_stroke(center, screen_radius, stroke);
        }
        MaskShape::Square => {
            let rect = egui::Rect::from_center_size(
                center,
                egui::vec2(screen_radius * 2.0, screen_radius * 2.0),
            );
            painter.rect_stroke(rect, 0.0, stroke, egui::StrokeKind::Middle);
        }
    }
}

fn paint_template_fill(
    painter: &egui::Painter,
    center: egui::Pos2,
    screen_radius: f32,
    shape: &MaskShape,
    alpha: u8,
) {
    let fill = egui::Color32::from_rgba_unmultiplied(255, 200, 0, alpha);
    match shape {
        MaskShape::Circle => {
            painter.circle_filled(center, screen_radius, fill);
        }
        MaskShape::Square => {
            let rect = egui::Rect::from_center_size(
                center,
                egui::vec2(screen_radius * 2.0, screen_radius * 2.0),
            );
            painter.rect_filled(rect, 0.0, fill);
        }
    }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

fn pane_header(ui: &mut egui::Ui, label: &str) {
    ui.label(
        egui::RichText::new(label)
            .size(15.0)
            .strong()
            .color(ui.visuals().text_color()),
    );
    ui.add_space(3.0);
}

fn section_header(ui: &mut egui::Ui, label: &str) {
    ui.label(
        egui::RichText::new(label)
            .size(14.0)
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

/// Format a float to 3 significant figures.
fn sig3(x: f64) -> String {
    if x == 0.0 {
        return "0.00".to_string();
    }
    let mag = x.abs().log10().floor() as i32;
    // For values between 0.001 and 9999, use decimal; otherwise scientific.
    if mag >= -3 && mag <= 3 {
        let decimals = (2 - mag).max(0) as usize;
        format!("{:.prec$}", x, prec = decimals)
    } else {
        format!("{:.2e}", x)
    }
}
