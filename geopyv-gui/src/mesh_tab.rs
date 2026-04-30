use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;
use ndarray::Array2;

use geopyv_dev::geometry::meshing::define_roi;
use geopyv_dev::image::Image;
use geopyv_dev::io::GeopyvObject;
use geopyv_dev::mesh::{Mesh, MeshSolution, SolveConfig};
use geopyv_dev::mesh::SolveMethod as LibSolveMethod;
use geopyv_dev::templates::{Template, TemplateShape};

use crate::colormap::{self, ColormapType};
use crate::draw::{ActiveDrawMode, ImageCoord};
use crate::image_viewer::{HoverInfo, ImageViewer, TextureCache};
use crate::subset_tab::{SolveMethod, SolverConfig, SubsetOrder};

// ---------------------------------------------------------------------------
// Mesh generation config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshOrder {
    First,
    Second,
}

pub struct MeshGenConfig {
    pub mesh_order: MeshOrder,
    pub size_lower_text: String,
    pub size_lower: f64,
    pub size_upper_text: String,
    pub size_upper: f64,
    pub target_nodes_text: String,
    pub target_nodes: usize,
    pub parse_error: Option<String>,
}

impl Default for MeshGenConfig {
    fn default() -> Self {
        Self {
            mesh_order: MeshOrder::First,
            size_lower_text: "20".to_string(),
            size_lower: 20.0,
            size_upper_text: "40".to_string(),
            size_upper: 40.0,
            target_nodes_text: "200".to_string(),
            target_nodes: 200,
            parse_error: None,
        }
    }
}

impl MeshGenConfig {
    pub fn is_valid(&self) -> bool {
        self.parse_error.is_none()
            && self.size_lower > 0.0
            && self.size_upper > self.size_lower
            && self.target_nodes > 0
    }
}

// ---------------------------------------------------------------------------
// Plot type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshPlotType {
    // Displacement
    #[default]
    Ux,
    Uy,
    UMag,
    // Strain
    Exx,
    Eyy,
    Exy,
    EVM,
    // Quality
    CZncc,
    Iterations,
    Norms,
}

impl MeshPlotType {
    pub fn label(self) -> &'static str {
        match self {
            MeshPlotType::Ux => "u_x",
            MeshPlotType::Uy => "u_y",
            MeshPlotType::UMag => "|u|",
            MeshPlotType::Exx => "\u{03b5}_xx",
            MeshPlotType::Eyy => "\u{03b5}_yy",
            MeshPlotType::Exy => "\u{03b5}_xy",
            MeshPlotType::EVM => "\u{03b5}_VM",
            MeshPlotType::CZncc => "C_ZNCC",
            MeshPlotType::Iterations => "Iterations",
            MeshPlotType::Norms => "Norms",
        }
    }
}

// ---------------------------------------------------------------------------
// Range mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RangeMode {
    #[default]
    Auto,
    Manual,
}

// ---------------------------------------------------------------------------
// New mesh form state
// ---------------------------------------------------------------------------

pub struct NewMeshForm {
    pub name: String,
    pub ref_idx: Option<usize>,
    pub target_idx: Option<usize>,
    pub template_shape: TemplateShape,
    pub template_size_text: String,
    pub template_size: u32,
    pub template_size_error: Option<String>,
    pub gen_cfg: MeshGenConfig,
    pub solver: SolverConfig,
    pub draw: crate::draw::DrawState,
    pub form_error: Option<String>,
}

impl Default for NewMeshForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            ref_idx: None,
            target_idx: None,
            template_shape: TemplateShape::Circle,
            template_size_text: "20".to_string(),
            template_size: 20,
            template_size_error: None,
            gen_cfg: MeshGenConfig::default(),
            solver: SolverConfig::default(),
            draw: crate::draw::DrawState::new(),
            form_error: None,
        }
    }
}

impl NewMeshForm {
    fn can_run(&self, images: &[PathBuf]) -> bool {
        let name_ok = !self.name.trim().is_empty()
            && !self.name.contains('/')
            && !self.name.contains('\\');
        let ref_ok = self.ref_idx.map(|i| i < images.len()).unwrap_or(false);
        let tar_ok = self.target_idx.map(|i| i < images.len()).unwrap_or(false);
        let template_ok = !self.template_size_text.trim().is_empty()
            && self.template_size > 0
            && self.template_size_error.is_none();
        name_ok
            && ref_ok
            && tar_ok
            && template_ok
            && self.draw.boundary_ok()
            && !self.draw.has_self_intersection()
            && self.draw.seed_ok()
            && self.gen_cfg.is_valid()
            && self.solver.is_valid()
    }
}

// ---------------------------------------------------------------------------
// View state
// ---------------------------------------------------------------------------

pub struct MeshViewState {
    pub loaded_path: Option<PathBuf>,
    pub solution: Option<MeshSolution>,
    pub target_image: Option<PathBuf>,
    pub plot_type: MeshPlotType,
    pub colormap: ColormapType,
    pub range_mode: RangeMode,
    pub range_min_text: String,
    pub range_min: f64,
    pub range_max_text: String,
    pub range_max: f64,
    pub show_wireframe: bool,
    pub show_node_indices: bool,
    // Cached per-node scalar values (invalidated when plot_type changes).
    pub nodal_cache: Option<(MeshPlotType, Vec<f64>)>,
}

impl MeshViewState {
    fn new() -> Self {
        Self {
            loaded_path: None,
            solution: None,
            target_image: None,
            plot_type: MeshPlotType::default(),
            colormap: ColormapType::default(),
            range_mode: RangeMode::Auto,
            range_min_text: "0".to_string(),
            range_min: 0.0,
            range_max_text: "1".to_string(),
            range_max: 1.0,
            show_wireframe: false,
            show_node_indices: false,
            nodal_cache: None,
        }
    }

}

// ---------------------------------------------------------------------------
// Background solve state
// ---------------------------------------------------------------------------

pub struct MeshSolveState {
    pub running: bool,
    pub progress: f32,
    pub message: String,
    pub cancel: Arc<AtomicBool>,
    pub result: Option<Result<MeshSolution, String>>,
}

impl MeshSolveState {
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

pub struct MeshSpawnParams {
    pub name: String,
    pub ref_path: PathBuf,
    pub target_path: PathBuf,
    pub template_shape: TemplateShape,
    pub template_size: u32,
    pub boundary: Vec<[f64; 2]>,
    pub exclusions: Vec<Vec<[f64; 2]>>,
    pub seed: [f64; 2],
    pub mesh_order: u8,
    pub size_lower: f64,
    pub size_upper: f64,
    pub target_nodes: usize,
    pub method: SolveMethod,
    pub subset_order: SubsetOrder,
    pub max_norm: f64,
    pub max_iterations: usize,
    pub zncc_tol: f64,
}

// ---------------------------------------------------------------------------
// Tab state
// ---------------------------------------------------------------------------

pub struct MeshTabState {
    pub view: MeshViewState,
    pub new_form: NewMeshForm,
    pub image_viewer: ImageViewer,
    pub solve_state: Arc<Mutex<MeshSolveState>>,
    pub pending_save_name: Option<String>,
    pub pending_target_image: Option<PathBuf>,
}

impl MeshTabState {
    pub fn new() -> Self {
        Self {
            view: MeshViewState::new(),
            new_form: NewMeshForm::default(),
            image_viewer: ImageViewer::new(),
            solve_state: Arc::new(Mutex::new(MeshSolveState::new())),
            pending_save_name: None,
            pending_target_image: None,
        }
    }

    pub fn is_solving(&self) -> bool {
        self.solve_state.lock().map(|s| s.running).unwrap_or(false)
    }

    pub fn check_solve_complete(
        &mut self,
    ) -> Option<Result<(MeshSolution, String, Option<PathBuf>), String>> {
        let mut state = self.solve_state.lock().unwrap();
        if state.running || state.result.is_none() {
            return None;
        }
        let result = state.result.take()?;
        let name = self.pending_save_name.take().unwrap_or_default();
        let target = self.pending_target_image.take();
        Some(result.map(|sol| (sol, name, target)))
    }

    pub fn spawn_solve(&mut self, params: MeshSpawnParams) {
        self.pending_save_name = Some(params.name.clone());
        self.pending_target_image = Some(params.target_path.clone());
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

        if mode == crate::main_window::PaneMode::View {
            let cached = self.view.loaded_path.as_deref();
            if cached != selected_path {
                self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
                self.view.nodal_cache = None;
                self.view.solution = selected_path.and_then(|p| {
                    geopyv_dev::io::load(p).ok().and_then(|obj| {
                        if let GeopyvObject::Mesh(m) = obj { Some(m) } else { None }
                    })
                });
                // target_image is only available when loaded from the current session.
                // When opening from disk, target_image remains as previously set.
                if selected_path != self.view.loaded_path.as_deref() {
                    self.view.target_image = None;
                }
            }
        }

        // New mode: show reference image with draw overlay.
        let image_path_new: Option<PathBuf> = if mode == crate::main_window::PaneMode::New {
            self.new_form
                .ref_idx
                .and_then(|i| images.get(i))
                .cloned()
        } else {
            None
        };

        let hover = if let Some(path) = image_path_new {
            let draw = if !solve_running {
                Some(&mut self.new_form.draw)
            } else {
                None
            };
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                self.image_viewer.show(ui, &path, cache, draw)
            })
            .inner
        } else if mode == crate::main_window::PaneMode::View {
            self.show_view_overlay(ui, viewer_rect, cache)
        } else {
            // New mode, no ref image selected.
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                ui.painter()
                    .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("Select a reference image to begin")
                            .size(14.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                });
            });
            None
        };

        if solve_running {
            self.show_progress_overlay(ui, viewer_rect);
        }

        hover
    }

    fn show_view_overlay(
        &mut self,
        ui: &mut egui::Ui,
        viewer_rect: egui::Rect,
        cache: &mut TextureCache,
    ) -> Option<HoverInfo> {
        let has_solution = self.view.solution.is_some();

        if !has_solution {
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                ui.painter()
                    .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("Select a mesh from the list")
                            .size(14.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                });
            });
            return None;
        }

        // Show target image (if available) with the image viewer for pan/zoom.
        let hover = if let Some(target_path) = self.view.target_image.clone() {
            if target_path.exists() {
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    self.image_viewer.show(ui, &target_path, cache, None)
                })
                .inner
            } else {
                self.show_dark_background(ui, viewer_rect);
                None
            }
        } else {
            self.show_dark_background(ui, viewer_rect);
            None
        };

        // Now render the mesh overlay on top.
        let coord = self.image_viewer.last_coord().or_else(|| {
            // No image was shown; compute a coordinate transform from the mesh bounds.
            self.view.solution.as_ref().map(|sol| mesh_coord_from_bounds(sol, viewer_rect))
        });

        if let (Some(sol), Some(coord)) = (self.view.solution.as_ref(), coord) {
            // Compute nodal values and range (needs mutable borrow after immutable above).
            let plot_type = self.view.plot_type;
            let range_mode = self.view.range_mode;
            let manual_min = self.view.range_min;
            let manual_max = self.view.range_max;
            let colormap = self.view.colormap;
            let show_wireframe = self.view.show_wireframe;
            let show_node_indices = self.view.show_node_indices;

            // Ensure cache is up to date.
            if self.view.nodal_cache.as_ref().map(|(t, _)| *t) != Some(plot_type) {
                let vals = extract_nodal_values(sol, plot_type);
                self.view.nodal_cache = Some((plot_type, vals));
            }

            if let Some((_, ref vals)) = self.view.nodal_cache {
                let (vmin, vmax) = match range_mode {
                    RangeMode::Manual => (manual_min, manual_max),
                    RangeMode::Auto => {
                        let mn = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                        let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        (mn, mx)
                    }
                };

                let mut painter = ui.ctx().layer_painter(egui::LayerId::new(
                    egui::Order::Foreground,
                    egui::Id::new("mesh_overlay"),
                ));
                painter.set_clip_rect(viewer_rect);

                // Draw the solution ref: need to re-borrow sol here
                let sol = self.view.solution.as_ref().unwrap();
                render_mesh_overlay(
                    &painter,
                    sol,
                    vals,
                    vmin,
                    vmax,
                    colormap,
                    show_wireframe,
                    show_node_indices,
                    &coord,
                );

                render_colorbar(&painter, viewer_rect, vmin, vmax, colormap, plot_type.label());
            }
        }

        hover
    }

    fn show_dark_background(&self, ui: &mut egui::Ui, viewer_rect: egui::Rect) {
        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
            ui.painter()
                .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("No target image — re-run solve to enable image underlay")
                        .size(13.0)
                        .color(ui.visuals().weak_text_color()),
                );
            });
        });
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

    // -----------------------------------------------------------------------
    // Right pane — view mode (§10.2)
    // -----------------------------------------------------------------------

    pub fn show_right_view(&mut self, ui: &mut egui::Ui, selected_path: Option<&Path>) {
        if self.view.loaded_path.as_deref() != selected_path {
            self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
            self.view.nodal_cache = None;
            self.view.solution = selected_path.and_then(|p| {
                geopyv_dev::io::load(p).ok().and_then(|obj| {
                    if let GeopyvObject::Mesh(m) = obj { Some(m) } else { None }
                })
            });
        }

        let Some(sol) = self.view.solution.as_ref() else {
            ui.label(
                egui::RichText::new("No mesh selected")
                    .size(13.0)
                    .color(ui.visuals().weak_text_color()),
            );
            return;
        };

        // Basic metadata.
        section_header(ui, "Mesh");
        meta_row(ui, "Nodes", &sol.nodes.nrows().to_string());
        meta_row(ui, "Elements", &sol.elements.nrows().to_string());
        meta_row(ui, "Mesh order", &sol.mesh_order.to_string());
        meta_row(ui, "Subset order", &sol.subset_order.to_string());

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Plot type selector.
        section_header(ui, "Plot Type");
        ui.add_space(3.0);

        let old_plot = self.view.plot_type;

        ui.label(
            egui::RichText::new("── Displacement ──").size(11.0).color(ui.visuals().weak_text_color()),
        );
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::Ux, "u_x");
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::Uy, "u_y");
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::UMag, "|u|");
        });

        ui.add_space(3.0);
        ui.label(
            egui::RichText::new("── Strain ──").size(11.0).color(ui.visuals().weak_text_color()),
        );
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::Exx, "\u{03b5}_xx");
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::Eyy, "\u{03b5}_yy");
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::Exy, "\u{03b5}_xy");
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::EVM, "\u{03b5}_VM");
        });

        ui.add_space(3.0);
        ui.label(
            egui::RichText::new("── Quality ──").size(11.0).color(ui.visuals().weak_text_color()),
        );
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::CZncc, "C_ZNCC");
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::Iterations, "Iterations");
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::Norms, "Norms");
        });

        if self.view.plot_type != old_plot {
            self.view.nodal_cache = None;
        }

        ui.add_space(3.0);
        ui.label(
            egui::RichText::new("── Geometry ──").size(11.0).color(ui.visuals().weak_text_color()),
        );
        ui.checkbox(&mut self.view.show_wireframe, "Show wireframe");
        ui.checkbox(&mut self.view.show_node_indices, "Show node indices");

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Colormap selector.
        section_header(ui, "Colormap");
        ui.add_space(3.0);
        let current_label = self.view.colormap.label();
        egui::ComboBox::from_id_salt("mesh_colormap")
            .selected_text(current_label)
            .width(120.0)
            .show_ui(ui, |ui| {
                for &cmap in ColormapType::ALL {
                    ui.selectable_value(&mut self.view.colormap, cmap, cmap.label());
                }
            });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Range controls.
        section_header(ui, "Range");
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.view.range_mode, RangeMode::Auto, "Auto");
            ui.radio_value(&mut self.view.range_mode, RangeMode::Manual, "Manual");
        });

        if self.view.range_mode == RangeMode::Manual {
            ui.add_space(3.0);
            egui::Grid::new("mesh_range_grid")
                .num_columns(2)
                .spacing([6.0, 4.0])
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("Min:").size(12.0).color(ui.visuals().weak_text_color()),
                    );
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.view.range_min_text).desired_width(80.0),
                    );
                    if r.changed() {
                        if let Ok(v) = self.view.range_min_text.trim().parse::<f64>() {
                            self.view.range_min = v;
                        }
                    }
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Max:").size(12.0).color(ui.visuals().weak_text_color()),
                    );
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.view.range_max_text).desired_width(80.0),
                    );
                    if r.changed() {
                        if let Ok(v) = self.view.range_max_text.trim().parse::<f64>() {
                            self.view.range_max = v;
                        }
                    }
                    ui.end_row();
                });
        } else {
            // Show auto-computed range for reference.
            let (mn, mx) = {
                if self.view.nodal_cache.as_ref().map(|(t, _)| *t) != Some(self.view.plot_type) {
                    let vals = extract_nodal_values(sol, self.view.plot_type);
                    self.view.nodal_cache = Some((self.view.plot_type, vals));
                }
                self.view
                    .nodal_cache
                    .as_ref()
                    .map(|(_, v)| {
                        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
                        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        (mn, mx)
                    })
                    .unwrap_or((0.0, 1.0))
            };
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!("{} \u{2013} {}", format_sci(mn), format_sci(mx)))
                        .size(11.0)
                        .color(ui.visuals().weak_text_color()),
                );
            });
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Quality summary (below controls).
        section_header(ui, "Quality");
        let n = sol.c_zncc.len();
        if n > 0 {
            let mean: f64 = sol.c_zncc.iter().sum::<f64>() / n as f64;
            let min: f64 = sol.c_zncc.iter().cloned().fold(f64::INFINITY, f64::min);
            meta_row(ui, "Mean C_ZNCC", &format!("{mean:.4}"));
            meta_row(ui, "Min C_ZNCC", &format!("{min:.4}"));
            let n_below = sol.c_zncc.iter().filter(|&&v| v < 0.75).count();
            meta_row(ui, "Below 0.75", &n_below.to_string());
        }
        meta_row(ui, "Seed node", &sol.seed_node.to_string());

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Export buttons (wired in session 15).
        ui.horizontal(|ui| {
            ui.add_enabled(false, egui::Button::new("Export PNG").min_size(egui::vec2(80.0, 24.0)));
            ui.add_space(4.0);
            ui.add_enabled(false, egui::Button::new("Export CSV").min_size(egui::vec2(80.0, 24.0)));
        });
    }

    // -----------------------------------------------------------------------
    // Right pane — new mode
    // -----------------------------------------------------------------------

    pub fn show_right_new(
        &mut self,
        ui: &mut egui::Ui,
        images: &[PathBuf],
    ) -> Option<MeshSpawnParams> {
        if self.is_solving() {
            return self.show_solve_progress_panel(ui);
        }

        let form = &mut self.new_form;
        let mut spawn: Option<MeshSpawnParams> = None;

        // Basic fields grid.
        egui::Grid::new("mesh_new_grid")
            .num_columns(2)
            .spacing([8.0, 5.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Name:")
                        .size(12.0)
                        .color(ui.visuals().weak_text_color()),
                );
                ui.add(egui::TextEdit::singleline(&mut form.name).desired_width(150.0));
                ui.end_row();

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
                    .unwrap_or_else(|| "\u{2014}".to_string());
                egui::ComboBox::from_id_salt("mesh_ref")
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
                        .size(12.0)
                        .color(ui.visuals().weak_text_color()),
                );
                let tar_label = form
                    .target_idx
                    .and_then(|i| images.get(i))
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "\u{2014}".to_string());
                egui::ComboBox::from_id_salt("mesh_tar")
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

        egui::Grid::new("mesh_template_grid")
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
                    .size(11.0)
                    .color(ui.visuals().error_fg_color),
            );
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Mesh generation section.
        section_header(ui, "Mesh Generation");
        ui.add_space(4.0);

        {
            let g = &mut form.gen_cfg;
            egui::Grid::new("mesh_gen_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .min_col_width(72.0)
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("Mesh order:")
                            .size(12.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut g.mesh_order, MeshOrder::First, "1");
                        ui.radio_value(&mut g.mesh_order, MeshOrder::Second, "2");
                    });
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Size lower:")
                            .size(12.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut g.size_lower_text).desired_width(80.0),
                    );
                    if r.changed() {
                        match g.size_lower_text.trim().parse::<f64>() {
                            Ok(v) if v > 0.0 => {
                                g.size_lower = v;
                                g.parse_error = None;
                            }
                            _ => g.parse_error = Some("Size lower must be > 0".into()),
                        }
                    }
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Size upper:")
                            .size(12.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut g.size_upper_text).desired_width(80.0),
                    );
                    if r.changed() {
                        match g.size_upper_text.trim().parse::<f64>() {
                            Ok(v) if v > 0.0 => {
                                g.size_upper = v;
                                g.parse_error = None;
                            }
                            _ => g.parse_error = Some("Size upper must be > 0".into()),
                        }
                    }
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Target nodes:")
                            .size(12.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut g.target_nodes_text).desired_width(80.0),
                    );
                    if r.changed() {
                        match g.target_nodes_text.trim().parse::<usize>() {
                            Ok(v) if v > 0 => {
                                g.target_nodes = v;
                                g.parse_error = None;
                            }
                            _ => g.parse_error = Some("Target nodes must be a positive integer".into()),
                        }
                    }
                    ui.end_row();
                });

            if let Some(err) = &g.parse_error.clone() {
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

        // Solver section.
        section_header(ui, "Solver");
        ui.add_space(4.0);

        {
            let s = &mut form.solver;
            egui::Grid::new("mesh_solver_grid")
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
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut s.max_norm_text).desired_width(80.0),
                    );
                    if r.changed() {
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
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut s.max_iterations_text).desired_width(80.0),
                    );
                    if r.changed() {
                        match s.max_iterations_text.trim().parse::<usize>() {
                            Ok(v) if v > 0 => {
                                s.max_iterations = v;
                                s.parse_error = None;
                            }
                            _ => s.parse_error = Some("max_iterations must be a positive integer".into()),
                        }
                    }
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("ZNCC tol:")
                            .size(12.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut s.zncc_tol_text).desired_width(80.0),
                    );
                    if r.changed() {
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

        // Define region section.
        section_header(ui, "Define Region");
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            let b_active = form.draw.mode == Some(ActiveDrawMode::Boundary);
            let btn_b = egui::Button::new(
                egui::RichText::new(if b_active { "Drawing\u{2026}" } else { "Boundary \u{25b6}" }).size(12.0),
            )
            .selected(b_active);
            if ui.add(btn_b).clicked() {
                if b_active { form.draw.cancel(); } else { form.draw.start(ActiveDrawMode::Boundary); }
            }

            let e_active = form.draw.mode == Some(ActiveDrawMode::Exclusion);
            let btn_e = egui::Button::new(
                egui::RichText::new(if e_active { "Drawing\u{2026}" } else { "Exclusion \u{25b6}" }).size(12.0),
            )
            .selected(e_active);
            if ui.add(btn_e).clicked() {
                if e_active { form.draw.cancel(); } else { form.draw.start(ActiveDrawMode::Exclusion); }
            }

            let s_active = form.draw.mode == Some(ActiveDrawMode::Seed);
            let btn_s = egui::Button::new(
                egui::RichText::new(if s_active { "Placing\u{2026}" } else { "Seed \u{25b6}" }).size(12.0),
            )
            .selected(s_active);
            if ui.add(btn_s).clicked() {
                if s_active { form.draw.cancel(); } else { form.draw.start(ActiveDrawMode::Seed); }
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            let (b_text, b_color) = if form.draw.boundary_ok() {
                ("boundary \u{2713}", egui::Color32::from_rgb(100, 200, 100))
            } else {
                ("boundary \u{2014}", ui.visuals().weak_text_color())
            };
            ui.label(egui::RichText::new(b_text).size(11.0).color(b_color));
            ui.add_space(8.0);

            let exc_n = form.draw.exclusions.len();
            let e_color = if exc_n > 0 { egui::Color32::from_rgb(100, 200, 100) } else { ui.visuals().weak_text_color() };
            ui.label(egui::RichText::new(format!("exclusions: {exc_n}")).size(11.0).color(e_color));
            ui.add_space(8.0);

            let (s_text, s_color) = if form.draw.seed_ok() {
                ("seed \u{2713}", egui::Color32::from_rgb(100, 200, 100))
            } else {
                ("seed \u{2014}", ui.visuals().weak_text_color())
            };
            ui.label(egui::RichText::new(s_text).size(11.0).color(s_color));
        });

        if form.draw.has_self_intersection() {
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new("Self-intersecting polygon")
                    .size(11.0)
                    .color(ui.visuals().error_fg_color),
            );
        }

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

        let can_run = form.can_run(images);
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            if ui
                .add_enabled(can_run, egui::Button::new("Run").min_size(egui::vec2(60.0, 26.0)))
                .clicked()
            {
                let seed = form.draw.seed.unwrap();
                let boundary: Vec<[f64; 2]> = form
                    .draw
                    .boundary
                    .as_ref()
                    .unwrap()
                    .vertices
                    .iter()
                    .map(|p| [p.x as f64, p.y as f64])
                    .collect();
                let exclusions: Vec<Vec<[f64; 2]>> = form
                    .draw
                    .exclusions
                    .iter()
                    .map(|ex| ex.vertices.iter().map(|p| [p.x as f64, p.y as f64]).collect())
                    .collect();
                spawn = Some(MeshSpawnParams {
                    name: form.name.trim().to_string(),
                    ref_path: images[form.ref_idx.unwrap()].clone(),
                    target_path: images[form.target_idx.unwrap()].clone(),
                    template_shape: form.template_shape.clone(),
                    template_size: form.template_size,
                    boundary,
                    exclusions,
                    seed: [seed.x as f64, seed.y as f64],
                    mesh_order: if form.gen_cfg.mesh_order == MeshOrder::First { 1 } else { 2 },
                    size_lower: form.gen_cfg.size_lower,
                    size_upper: form.gen_cfg.size_upper,
                    target_nodes: form.gen_cfg.target_nodes,
                    method: form.solver.method,
                    subset_order: form.solver.order,
                    max_norm: form.solver.max_norm,
                    max_iterations: form.solver.max_iterations,
                    zncc_tol: form.solver.zncc_tol,
                });
                form.form_error = None;
            }
        });

        spawn
    }

    fn show_solve_progress_panel(&self, ui: &mut egui::Ui) -> Option<MeshSpawnParams> {
        let (progress, message) = {
            let s = self.solve_state.lock().unwrap();
            (s.progress, s.message.clone())
        };

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Solving mesh\u{2026}")
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

        None
    }
}

// ---------------------------------------------------------------------------
// Background thread
// ---------------------------------------------------------------------------

fn set_progress(state: &Arc<Mutex<MeshSolveState>>, progress: f32, msg: &str) {
    if let Ok(mut s) = state.lock() {
        s.progress = progress;
        s.message = msg.to_string();
    }
}

fn set_error(state: &Arc<Mutex<MeshSolveState>>, msg: String) {
    if let Ok(mut s) = state.lock() {
        s.result = Some(Err(msg));
        s.running = false;
        s.progress = 0.0;
    }
}

fn run_solve(
    params: MeshSpawnParams,
    state: Arc<Mutex<MeshSolveState>>,
    cancel: Arc<AtomicBool>,
) {
    set_progress(&state, 0.05, "Building template\u{2026}");
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

    set_progress(&state, 0.15, "Loading reference image\u{2026}");
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

    set_progress(&state, 0.30, "Generating mesh\u{2026}");
    let boundary_arr = Array2::from_shape_fn(
        (params.boundary.len(), 2),
        |(i, j)| params.boundary[i][j],
    );
    let exclusion_arrs: Vec<Array2<f64>> = params
        .exclusions
        .iter()
        .map(|ex| Array2::from_shape_fn((ex.len(), 2), |(i, j)| ex[i][j]))
        .collect();
    let exclusion_views: Vec<_> = exclusion_arrs.iter().map(|a| a.view()).collect();
    let roi = define_roi(boundary_arr.view(), true, &exclusion_views, None);

    let mesh = match Mesh::generate(
        roi.borders.view(),
        roi.segments.view(),
        &roi.curves,
        params.size_lower,
        params.size_upper,
        params.target_nodes,
        params.mesh_order,
    ) {
        Ok(m) => m,
        Err(e) => {
            set_error(&state, format!("Mesh generation error: {e}"));
            return;
        }
    };

    let n_nodes = mesh.nodes().nrows();

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() { s.running = false; }
        return;
    }

    set_progress(&state, 0.55, "Loading target image\u{2026}");
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

    set_progress(&state, 0.65, &format!("Solving {n_nodes} nodes\u{2026}"));
    let p_len = if matches!(params.subset_order, SubsetOrder::First) { 1 } else { 2 };
    let cfg = SolveConfig {
        max_norm: params.max_norm,
        max_iterations: params.max_iterations,
        subset_order: p_len,
        tolerance: params.zncc_tol,
        method: match params.method {
            SolveMethod::Icgn => LibSolveMethod::Icgn,
            SolveMethod::Fagn => LibSolveMethod::Fagn,
        },
    };
    let seed_warp = vec![0.0f64; 6 * p_len];

    match mesh.solve(
        &ref_img,
        &target_img,
        &template.coords,
        params.seed,
        &seed_warp,
        &cfg,
    ) {
        Ok(solution) => {
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
// Colormap overlay rendering
// ---------------------------------------------------------------------------

/// Compute per-node scalar values for the given plot type.
pub fn extract_nodal_values(sol: &MeshSolution, plot: MeshPlotType) -> Vec<f64> {
    let n = sol.nodes.nrows();
    let m = sol.warps.nrows();
    match plot {
        MeshPlotType::Ux => (0..n).map(|i| sol.displacements[[i, 0]]).collect(),
        MeshPlotType::Uy => (0..n).map(|i| sol.displacements[[i, 1]]).collect(),
        MeshPlotType::UMag => (0..n)
            .map(|i| {
                let u = sol.displacements[[i, 0]];
                let v = sol.displacements[[i, 1]];
                (u * u + v * v).sqrt()
            })
            .collect(),
        MeshPlotType::Exx => {
            let elem: Vec<f64> = (0..m).map(|e| sol.warps[[e, 2]]).collect();
            element_to_node_avg(n, &sol.elements, &elem)
        }
        MeshPlotType::Eyy => {
            let elem: Vec<f64> = (0..m).map(|e| sol.warps[[e, 5]]).collect();
            element_to_node_avg(n, &sol.elements, &elem)
        }
        MeshPlotType::Exy => {
            // ε_xy = (du/dy + dv/dx) / 2  (tensor shear strain)
            let elem: Vec<f64> = (0..m)
                .map(|e| (sol.warps[[e, 4]] + sol.warps[[e, 3]]) * 0.5)
                .collect();
            element_to_node_avg(n, &sol.elements, &elem)
        }
        MeshPlotType::EVM => {
            let elem: Vec<f64> = (0..m)
                .map(|e| {
                    let exx = sol.warps[[e, 2]];
                    let eyy = sol.warps[[e, 5]];
                    let exy = (sol.warps[[e, 4]] + sol.warps[[e, 3]]) * 0.5;
                    (exx * exx + eyy * eyy - exx * eyy + 3.0 * exy * exy).sqrt()
                })
                .collect();
            element_to_node_avg(n, &sol.elements, &elem)
        }
        MeshPlotType::CZncc => sol.c_zncc.to_vec(),
        MeshPlotType::Iterations => sol.iterations.iter().map(|&v| v as f64).collect(),
        MeshPlotType::Norms => sol.norms.to_vec(),
    }
}

/// Average element-level scalar values to nodes (sum / count over connected elements).
pub fn element_to_node_avg(
    n_nodes: usize,
    elements: &ndarray::Array2<usize>,
    elem_vals: &[f64],
) -> Vec<f64> {
    let mut sum = vec![0.0f64; n_nodes];
    let mut count = vec![0usize; n_nodes];
    for (e, &val) in elem_vals.iter().enumerate() {
        // Use only the first 3 columns (corner nodes) regardless of mesh order.
        for c in 0..3 {
            let ni = elements[[e, c]];
            sum[ni] += val;
            count[ni] += 1;
        }
    }
    sum.iter()
        .zip(count.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect()
}

/// Derive a coordinate transform for displaying the mesh without an image underlay.
pub fn mesh_coord_from_bounds(sol: &MeshSolution, viewport: egui::Rect) -> ImageCoord {
    let n = sol.nodes.nrows();
    if n == 0 {
        return ImageCoord {
            canvas_center: viewport.center(),
            offset: egui::Vec2::ZERO,
            zoom: 1.0,
            img_size: egui::Vec2::ZERO,
        };
    }

    let xs: Vec<f32> = (0..n).map(|i| sol.nodes[[i, 0]] as f32).collect();
    let ys: Vec<f32> = (0..n).map(|i| sol.nodes[[i, 1]] as f32).collect();
    let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

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

/// Render the coloured-triangle mesh overlay onto `painter`.
pub fn render_mesh_overlay(
    painter: &egui::Painter,
    sol: &MeshSolution,
    values: &[f64],
    vmin: f64,
    vmax: f64,
    cmap: ColormapType,
    show_wireframe: bool,
    show_node_indices: bool,
    coord: &ImageCoord,
) {
    let n_nodes = sol.nodes.nrows();
    let n_elems = sol.elements.nrows();

    if n_nodes == 0 || n_elems == 0 {
        return;
    }

    // Screen-space positions for all nodes.
    let positions: Vec<egui::Pos2> = (0..n_nodes)
        .map(|i| {
            coord.to_screen(egui::pos2(
                sol.nodes[[i, 0]] as f32,
                sol.nodes[[i, 1]] as f32,
            ))
        })
        .collect();

    // Per-node colours (semi-transparent for image underlay blending).
    let colors: Vec<egui::Color32> = values
        .iter()
        .map(|&v| {
            let c = colormap::map_value(v, vmin, vmax, cmap);
            egui::Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 200)
        })
        .collect();

    // Build smooth-shaded epaint::Mesh (per-vertex colour, shared nodes).
    let mut mesh = egui::epaint::Mesh::default();
    for i in 0..n_nodes {
        mesh.vertices.push(egui::epaint::Vertex {
            pos: positions[i],
            uv: egui::epaint::WHITE_UV,
            color: colors[i],
        });
    }
    for e in 0..n_elems {
        let n0 = sol.elements[[e, 0]] as u32;
        let n1 = sol.elements[[e, 1]] as u32;
        let n2 = sol.elements[[e, 2]] as u32;
        mesh.indices.extend_from_slice(&[n0, n1, n2]);
    }
    painter.add(egui::Shape::mesh(mesh));

    // Wireframe edges.
    if show_wireframe {
        let stroke = egui::Stroke::new(
            0.5,
            egui::Color32::from_rgba_unmultiplied(200, 200, 200, 80),
        );
        for e in 0..n_elems {
            let p0 = positions[sol.elements[[e, 0]]];
            let p1 = positions[sol.elements[[e, 1]]];
            let p2 = positions[sol.elements[[e, 2]]];
            painter.line_segment([p0, p1], stroke);
            painter.line_segment([p1, p2], stroke);
            painter.line_segment([p2, p0], stroke);
        }
    }

    // Node index labels.
    if show_node_indices {
        let font = egui::FontId::new(9.0, egui::FontFamily::Proportional);
        for i in 0..n_nodes {
            painter.text(
                positions[i],
                egui::Align2::CENTER_CENTER,
                i.to_string(),
                font.clone(),
                egui::Color32::WHITE,
            );
        }
    }
}

/// Render a vertical colorbar overlay in the top-right corner of the viewer.
pub fn render_colorbar(
    painter: &egui::Painter,
    viewer_rect: egui::Rect,
    vmin: f64,
    vmax: f64,
    cmap: ColormapType,
    label: &str,
) {
    const BAR_W: f32 = 14.0;
    const BAR_H: f32 = 110.0;
    const MARGIN: f32 = 10.0;
    const LABEL_W: f32 = 52.0;
    const TITLE_H: f32 = 16.0;

    let bar_x = viewer_rect.max.x - MARGIN - BAR_W - LABEL_W;
    let bar_y = viewer_rect.min.y + MARGIN + TITLE_H;

    let bg_rect = egui::Rect::from_min_max(
        egui::pos2(bar_x - 4.0, viewer_rect.min.y + MARGIN - 2.0),
        egui::pos2(viewer_rect.max.x - MARGIN + 2.0, bar_y + BAR_H + 14.0),
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
        egui::FontId::new(11.0, egui::FontFamily::Proportional),
        egui::Color32::from_rgb(210, 210, 210),
    );

    // Gradient bar (32 horizontal strips, top = max).
    const N: usize = 32;
    let strip_h = BAR_H / N as f32;
    for i in 0..N {
        let t = 1.0 - i as f32 / (N - 1) as f32;
        let color = colormap::sample(t, cmap);
        let y0 = bar_y + i as f32 * strip_h;
        let strip = egui::Rect::from_min_size(
            egui::pos2(bar_x, y0),
            egui::vec2(BAR_W, strip_h + 0.5),
        );
        painter.rect_filled(strip, egui::CornerRadius::same(0), color);
    }

    // Min / max labels.
    let lc = egui::Color32::from_rgb(220, 220, 220);
    let font = egui::FontId::new(10.0, egui::FontFamily::Proportional);
    let lx = bar_x + BAR_W + 3.0;
    painter.text(
        egui::pos2(lx, bar_y),
        egui::Align2::LEFT_CENTER,
        format_sci(vmax),
        font.clone(),
        lc,
    );
    painter.text(
        egui::pos2(lx, bar_y + BAR_H),
        egui::Align2::LEFT_CENTER,
        format_sci(vmin),
        font,
        lc,
    );
}

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
