use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};

use geopyv_dev::image::Image;
use geopyv_dev::io::GeopyvObject;
use geopyv_dev::subset::{Subset, SubsetSolution, TemplateSummary};
use geopyv_dev::templates::{Template, TemplateShape};

use crate::draw::ActiveDrawMode;
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
    pub template_shape: TemplateShape,
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
            template_shape: TemplateShape::Circle,
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
}

impl SubsetViewState {
    fn new() -> Self {
        Self {
            loaded_path: None,
            solution: None,
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
    pub template_shape: TemplateShape,
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
    /// Name from the last spawned solve, used for auto-save on completion.
    pub pending_save_name: Option<String>,
}

impl SubsetTabState {
    pub fn new() -> Self {
        Self {
            view: SubsetViewState::new(),
            new_form: NewSubsetForm::default(),
            image_viewer: ImageViewer::new(),
            solve_state: Arc::new(Mutex::new(SubsetSolveState::new())),
            pending_save_name: None,
        }
    }

    pub fn is_solving(&self) -> bool {
        self.solve_state.lock().map(|s| s.running).unwrap_or(false)
    }

    /// Check if the background solve has finished.
    /// Returns `Some(Ok((solution, name)))` on success, `Some(Err(msg))` on failure.
    /// Clears the result from the state after returning it.
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

    /// Spawn the background solve thread.
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

    /// Cancel a running solve.
    pub fn cancel_solve(&self) {
        if let Ok(state) = self.solve_state.lock() {
            state.cancel.store(true, Ordering::Relaxed);
        }
    }

    // -----------------------------------------------------------------------
    // Central panel
    // -----------------------------------------------------------------------

    /// Render the central panel for the Subsets tab.
    ///
    /// Returns hover info for the status bar.
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

        // Determine which image path to show.
        let image_path: Option<PathBuf> = match mode {
            crate::main_window::PaneMode::New => {
                self.new_form
                    .ref_idx
                    .and_then(|i| images.get(i))
                    .cloned()
            }
            crate::main_window::PaneMode::View => {
                // Show the target image if a solution is loaded.
                self.view
                    .solution
                    .as_ref()
                    .map(|s| s.target_image.clone())
            }
        };

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
            }
        }

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
                            .size(14.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                });
            });
            None
        };

        // Overlay crosshair + template outline for view mode.
        if mode == crate::main_window::PaneMode::View {
            if let (Some(sol), Some(coord)) = (
                self.view.solution.as_ref(),
                self.image_viewer.last_coord(),
            ) {
                let screen_pt = coord.to_screen(egui::pos2(
                    sol.coord[0] as f32,
                    sol.coord[1] as f32,
                ));
                if viewer_rect.contains(screen_pt) {
                    paint_crosshair_x(ui.painter(), screen_pt);
                    let screen_radius = sol.template.size as f32 * coord.zoom;
                    paint_template_outline(
                        ui.painter(),
                        screen_pt,
                        screen_radius,
                        &sol.template.shape,
                    );
                }
            }
        }

        // Progress overlay during solve.
        if solve_running {
            self.show_progress_overlay(ui, viewer_rect);
        }

        hover
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

        // Dim overlay.
        ui.painter().rect_filled(
            rect,
            0.0,
            egui::Color32::from_rgba_premultiplied(0, 0, 0, 160),
        );

        // Progress bar track.
        ui.painter().rect_filled(
            bar_rect,
            egui::CornerRadius::same(4),
            egui::Color32::from_rgb(40, 40, 48),
        );

        // Progress bar fill.
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

        // Message text.
        if !message.is_empty() {
            let msg_pos = egui::pos2(bar_rect.min.x, bar_rect.max.y + 6.0);
            ui.painter().text(
                msg_pos,
                egui::Align2::LEFT_TOP,
                &message,
                egui::FontId::new(13.0, egui::FontFamily::Proportional),
                egui::Color32::from_rgb(200, 200, 200),
            );
        }
    }

    // -----------------------------------------------------------------------
    // Right pane — view mode
    // -----------------------------------------------------------------------

    pub fn show_right_view(&mut self, ui: &mut egui::Ui, selected_path: Option<&Path>) {
        // Lazy-load handled in show_central; just display what's loaded.
        if self.view.loaded_path.as_deref() != selected_path {
            self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
            self.view.solution = selected_path.and_then(|p| {
                geopyv_dev::io::load(p).ok().and_then(|obj| {
                    if let GeopyvObject::Subset(s) = obj { Some(s) } else { None }
                })
            });
        }

        let Some(sol) = self.view.solution.as_ref() else {
            ui.label(
                egui::RichText::new("No subset selected")
                    .size(13.0)
                    .color(ui.visuals().weak_text_color()),
            );
            return;
        };

        // Result summary.
        section_header(ui, "Result");
        meta_row(ui, "Coord", &format!("x = {:.2},  y = {:.2}", sol.coord[0], sol.coord[1]));
        meta_row(ui, "C_ZNCC", &format!("{:.6}", sol.result.c_zncc));
        meta_row(ui, "C_ZNSSD", &format!("{:.6}", sol.result.c_znssd));
        meta_row(ui, "Iterations", &format!("{}", sol.result.iterations));
        meta_row(ui, "Converged", if sol.result.converged { "✓" } else { "✗" });

        let p_str: Vec<String> = sol.result.p.iter().map(|v| format!("{v:.4e}")).collect();
        meta_row(ui, "p", &p_str.join(",  "));

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        section_header(ui, "Convergence");

        let history = &sol.result.history;
        if history.is_empty() {
            ui.label(
                egui::RichText::new("No history recorded")
                    .size(12.0)
                    .color(ui.visuals().weak_text_color()),
            );
            return;
        }

        // Norm vs iteration plot.
        let norm_pts: PlotPoints = history
            .iter()
            .map(|&(iter, norm, _, _)| [iter as f64, norm])
            .collect();
        let zncc_pts: PlotPoints = history
            .iter()
            .map(|&(iter, _, zncc, _)| [iter as f64, zncc])
            .collect();

        ui.label(
            egui::RichText::new("Norm  (left axis)  ·  C_ZNCC  (right axis approximate)")
                .size(11.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(4.0);

        Plot::new("subset_convergence")
            .height(180.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(norm_pts)
                        .name("norm")
                        .color(egui::Color32::from_rgb(100, 170, 255))
                        .width(2.0),
                );
                plot_ui.line(
                    Line::new(zncc_pts)
                        .name("C_ZNCC")
                        .color(egui::Color32::from_rgb(100, 220, 130))
                        .width(2.0),
                );
            });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);

        // Template summary.
        section_header(ui, "Template");
        let shape_str = match sol.template.shape {
            TemplateShape::Circle => "Circle",
            TemplateShape::Square => "Square",
        };
        meta_row(ui, "Shape", shape_str);
        meta_row(ui, "Size", &format!("{} px", sol.template.size));
        meta_row(ui, "n_px", &format!("{}", sol.template.n_px));

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);

        // Image paths.
        section_header(ui, "Images");
        let ref_name = sol
            .ref_image
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let tar_name = sol
            .target_image
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        meta_row(ui, "Reference", &ref_name);
        meta_row(ui, "Target", &tar_name);
    }

    // -----------------------------------------------------------------------
    // Right pane — new mode
    // -----------------------------------------------------------------------

    /// Returns spawn params when the user clicks Run, or `None` otherwise.
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
                // Name
                ui.label(
                    egui::RichText::new("Name:")
                        .size(12.0)
                        .color(ui.visuals().weak_text_color()),
                );
                ui.add(egui::TextEdit::singleline(&mut form.name).desired_width(150.0));
                ui.end_row();

                // Reference image
                ui.label(
                    egui::RichText::new("Ref image:")
                        .size(12.0)
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

                // Target image
                ui.label(
                    egui::RichText::new("Target image:")
                        .size(12.0)
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

        // Template section.
        section_header(ui, "Template");
        ui.add_space(4.0);

        egui::Grid::new("subset_template_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Shape:")
                        .size(12.0)
                        .color(ui.visuals().weak_text_color()),
                );
                ui.horizontal(|ui| {
                    ui.radio_value(&mut form.template_shape, TemplateShape::Circle, "Circle");
                    ui.radio_value(&mut form.template_shape, TemplateShape::Square, "Square");
                });
                ui.end_row();

                ui.label(
                    egui::RichText::new("Size (px):")
                        .size(12.0)
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
                                form.template_size_error = Some("Size must be \u{2265} 1".to_string())
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
                    .size(11.0)
                    .color(ui.visuals().error_fg_color),
            );
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Solver section.
        ui.label(
            egui::RichText::new("Solver")
                .size(12.0)
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
                            .size(12.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut s.method, SolveMethod::Icgn, "ICGN");
                        ui.radio_value(&mut s.method, SolveMethod::Fagn, "FAGN");
                    });
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Order:")
                            .size(12.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut s.order, SubsetOrder::First, "1");
                        ui.radio_value(&mut s.order, SubsetOrder::Second, "2");
                    });
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Max norm:")
                            .size(12.0)
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
                            _ => s.parse_error = Some("max_norm must be a positive number".into()),
                        }
                    }
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Max iters:")
                            .size(12.0)
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
                            .size(12.0)
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
                        .size(11.0)
                        .color(ui.visuals().error_fg_color),
                );
            }
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Coordinate placement.
        ui.label(
            egui::RichText::new("Coordinate")
                .size(12.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            let active = form.draw.mode == Some(ActiveDrawMode::Point);
            let btn_text = if active { "Placing…" } else { "Place point" };
            let btn =
                egui::Button::new(egui::RichText::new(btn_text).size(12.0)).selected(active);
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
                    .size(12.0)
                    .color(ui.visuals().text_color()),
            );
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Error display.
        if let Some(err) = &form.form_error.clone() {
            ui.label(
                egui::RichText::new(err)
                    .size(11.0)
                    .color(ui.visuals().error_fg_color),
            );
            ui.add_space(4.0);
        }

        // Run button.
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

    // Shown in place of the new form while solving.
    fn show_solve_progress_panel(&self, ui: &mut egui::Ui) -> Option<SubsetSpawnParams> {
        let (progress, message) = {
            let s = self.solve_state.lock().unwrap();
            (s.progress, s.message.clone())
        };

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Solving…")
                .size(14.0)
                .color(ui.visuals().text_color()),
        );
        ui.add_space(4.0);
        ui.label(egui::RichText::new(&message).size(12.0).color(ui.visuals().weak_text_color()));
        ui.add_space(8.0);

        let available = ui.available_width();
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(available, 16.0),
            egui::Sense::hover(),
        );
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
    // Build template pixel offsets.
    let template = match params.template_shape {
        TemplateShape::Circle => Template::circle(params.template_size as usize),
        TemplateShape::Square => Template::square(params.template_size as usize),
    };
    let template = match template {
        Ok(t) => t,
        Err(e) => {
            set_error(&state, format!("Template build error: {e}"));
            return;
        }
    };

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() { s.running = false; }
        return;
    }

    // Load reference image.
    set_progress(&state, 0.1, "Loading reference image…");
    let ref_img = match Image::from_file(&params.ref_path, 20) {
        Ok(img) => img,
        Err(e) => {
            set_error(&state, format!("Reference image error: {e}"));
            return;
        }
    };

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() { s.running = false; }
        return;
    }

    // Create subset from reference.
    set_progress(&state, 0.3, "Creating subset…");
    let subset = match Subset::new(params.coord, &template.coords, &ref_img.qcqt) {
        Ok(s) => s,
        Err(e) => {
            set_error(&state, format!("Subset error: {e}"));
            return;
        }
    };

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() { s.running = false; }
        return;
    }

    // Load target image.
    set_progress(&state, 0.5, "Loading target image…");
    let target_img = match Image::from_file(&params.target_path, 20) {
        Ok(img) => img,
        Err(e) => {
            set_error(&state, format!("Target image error: {e}"));
            return;
        }
    };

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() { s.running = false; }
        return;
    }

    // Solve.
    set_progress(&state, 0.7, "Solving…");
    let n_params = if matches!(params.order, SubsetOrder::First) { 6 } else { 12 };
    let p_0 = vec![0.0f64; n_params];

    let solve_result = match params.method {
        SolveMethod::Icgn => {
            subset.solve_icgn(&target_img.qcqt, &p_0, params.max_norm, params.max_iterations)
        }
        SolveMethod::Fagn => {
            subset.solve_fagn(&target_img.qcqt, &p_0, params.max_norm, params.max_iterations)
        }
    };

    match solve_result {
        Ok(result) => {
            let solution = SubsetSolution {
                coord: params.coord,
                template: TemplateSummary {
                    shape: params.template_shape,
                    size: params.template_size as usize,
                    n_px: subset.n_px(),
                },
                ref_image: params.ref_path,
                target_image: params.target_path,
                result,
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

fn paint_template_outline(
    painter: &egui::Painter,
    center: egui::Pos2,
    screen_radius: f32,
    shape: &TemplateShape,
) {
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(255, 200, 0, 200));
    match shape {
        TemplateShape::Circle => {
            painter.circle_stroke(center, screen_radius, stroke);
        }
        TemplateShape::Square => {
            let half = screen_radius;
            let rect = egui::Rect::from_center_size(
                center,
                egui::vec2(half * 2.0, half * 2.0),
            );
            painter.rect_stroke(rect, 0.0, stroke, egui::StrokeKind::Middle);
        }
    }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

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
