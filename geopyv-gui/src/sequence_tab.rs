use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use eframe::egui;
use ndarray::Array2;

use geopyv_dev::geometry::meshing::define_roi;
use geopyv_dev::image::Image;
use geopyv_dev::io::GeopyvObject;
use geopyv_dev::mesh::{Mesh, MeshSolution, SolveConfig};
use geopyv_dev::mesh::SolveMethod as LibSolveMethod;
use geopyv_dev::sequence::{deformation_preconditioning, SequenceSolution};
use geopyv_dev::templates::{Template, TemplateShape};

use crate::colormap::ColormapType;
use crate::draw::{ActiveDrawMode, DrawShapeMode, ImageCoord};
use crate::image_viewer::{HoverInfo, ImageViewer, TextureCache};
use crate::mesh_tab::{
    extract_nodal_values, format_sci, mesh_coord_from_bounds, render_colorbar,
    render_mesh_overlay, MeshGenConfig, MeshOrder, MeshPlotType, RangeMode,
};
use crate::subset_tab::{SolveMethod, SolverConfig, SubsetOrder};

// ---------------------------------------------------------------------------
// Reference mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferenceMode {
    Fixed,
    #[default]
    Sequential,
}

// ---------------------------------------------------------------------------
// Sequence options
// ---------------------------------------------------------------------------

pub struct SequenceOptions {
    pub guide: bool,
    pub sequential: bool,
    pub sync: bool,
    pub override_: bool,
    pub border_text: String,
    pub border: usize,
    pub parse_error: Option<String>,
}

impl Default for SequenceOptions {
    fn default() -> Self {
        Self {
            guide: true,
            sequential: true,
            sync: true,
            override_: false,
            border_text: "20".to_string(),
            border: 20,
            parse_error: None,
        }
    }
}

// ---------------------------------------------------------------------------
// New sequence form state
// ---------------------------------------------------------------------------

pub struct NewSequenceForm {
    pub name: String,
    pub start_text: String,
    pub start_idx: Option<usize>,
    pub end_text: String,
    pub end_idx: Option<usize>,
    pub template_shape: TemplateShape,
    pub template_size_text: String,
    pub template_size: u32,
    pub template_size_error: Option<String>,
    pub gen_cfg: MeshGenConfig,
    pub solver: SolverConfig,
    pub seq_opts: SequenceOptions,
    pub draw: crate::draw::DrawState,
    pub form_error: Option<String>,
}

impl Default for NewSequenceForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            start_text: String::new(),
            start_idx: None,
            end_text: String::new(),
            end_idx: None,
            template_shape: TemplateShape::Circle,
            template_size_text: "20".to_string(),
            template_size: 20,
            template_size_error: None,
            gen_cfg: MeshGenConfig::default(),
            solver: SolverConfig::default(),
            seq_opts: SequenceOptions::default(),
            draw: crate::draw::DrawState::new(),
            form_error: None,
        }
    }
}

impl NewSequenceForm {
    fn can_run(&self, images: &[PathBuf]) -> bool {
        let name_ok = !self.name.trim().is_empty()
            && !self.name.contains('/')
            && !self.name.contains('\\');
        let range_ok = match (self.start_idx, self.end_idx) {
            (Some(s), Some(e)) => s < images.len() && e < images.len() && e > s,
            _ => false,
        };
        let template_ok = !self.template_size_text.trim().is_empty()
            && self.template_size > 0
            && self.template_size_error.is_none();
        name_ok
            && range_ok
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

pub struct SequenceViewState {
    pub loaded_path: Option<PathBuf>,
    pub solution: Option<SequenceSolution>,
    /// Image paths from the original solve; empty when loaded from disk.
    pub image_paths: Vec<PathBuf>,
    pub current_frame: usize,
    pub animate: bool,
    pub speed: f32,
    pub last_tick: Option<Instant>,
    pub plot_type: MeshPlotType,
    pub colormap: ColormapType,
    pub range_mode: RangeMode,
    pub range_min_text: String,
    pub range_min: f64,
    pub range_max_text: String,
    pub range_max: f64,
    pub reference_mode: ReferenceMode,
    pub show_wireframe: bool,
    pub show_node_indices: bool,
    /// Cached (frame_index, plot_type, values) — invalidated on frame/type change.
    pub nodal_cache: Option<(usize, MeshPlotType, Vec<f64>)>,
}

impl SequenceViewState {
    fn new() -> Self {
        Self {
            loaded_path: None,
            solution: None,
            image_paths: Vec::new(),
            current_frame: 0,
            animate: false,
            speed: 4.0,
            last_tick: None,
            plot_type: MeshPlotType::default(),
            colormap: ColormapType::default(),
            range_mode: RangeMode::default(),
            range_min_text: "0".to_string(),
            range_min: 0.0,
            range_max_text: "1".to_string(),
            range_max: 1.0,
            reference_mode: ReferenceMode::default(),
            show_wireframe: false,
            show_node_indices: false,
            nodal_cache: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Background solve state
// ---------------------------------------------------------------------------

pub struct SequenceSolveState {
    pub running: bool,
    pub progress: f32,
    pub message: String,
    pub cancel: Arc<AtomicBool>,
    pub result: Option<Result<SequenceSolution, String>>,
}

impl SequenceSolveState {
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

pub struct SequenceSpawnParams {
    pub name: String,
    pub image_paths: Vec<PathBuf>,
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
    pub guide: bool,
    pub sequential: bool,
    pub sync: bool,
    pub override_: bool,
    pub border: usize,
    /// Directory /Meshes/<name>/ for per-frame .pyv files.
    pub mesh_subdir: PathBuf,
    /// Directory /Sequences/ for the overall SequenceSolution.
    pub sequences_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Tab state
// ---------------------------------------------------------------------------

pub struct SequenceTabState {
    pub view: SequenceViewState,
    pub new_form: NewSequenceForm,
    pub image_viewer: ImageViewer,
    pub solve_state: Arc<Mutex<SequenceSolveState>>,
    pub pending_save_name: Option<String>,
    /// Image paths captured at spawn time; returned with the completed result.
    pub pending_image_paths: Option<Vec<PathBuf>>,
}

impl SequenceTabState {
    pub fn new() -> Self {
        Self {
            view: SequenceViewState::new(),
            new_form: NewSequenceForm::default(),
            image_viewer: ImageViewer::new(),
            solve_state: Arc::new(Mutex::new(SequenceSolveState::new())),
            pending_save_name: None,
            pending_image_paths: None,
        }
    }

    pub fn is_solving(&self) -> bool {
        self.solve_state.lock().map(|s| s.running).unwrap_or(false)
    }

    pub fn check_solve_complete(
        &mut self,
    ) -> Option<Result<(SequenceSolution, String, Vec<PathBuf>), String>> {
        let mut state = self.solve_state.lock().unwrap();
        if state.running || state.result.is_none() {
            return None;
        }
        let result = state.result.take()?;
        let name = self.pending_save_name.take().unwrap_or_default();
        let paths = self.pending_image_paths.take().unwrap_or_default();
        Some(result.map(|sol| (sol, name, paths)))
    }

    pub fn spawn_solve(&mut self, params: SequenceSpawnParams) {
        self.pending_save_name = Some(params.name.clone());
        self.pending_image_paths = Some(params.image_paths.clone());
        let mut state = self.solve_state.lock().unwrap();
        state.running = true;
        state.progress = 0.0;
        state.message = "Starting\u{2026}".to_string();
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
            if self.view.loaded_path.as_deref() != selected_path {
                self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
                self.view.solution = selected_path.and_then(|p| {
                    geopyv_dev::io::load(p).ok().and_then(|obj| {
                        if let GeopyvObject::Sequence(s) = obj { Some(s) } else { None }
                    })
                });
                self.view.current_frame = 0;
                self.view.nodal_cache = None;
                self.view.image_paths = Vec::new();
                self.view.animate = false;
            }
        }

        // New mode: show reference image with draw overlay.
        let image_path_new: Option<PathBuf> = if mode == crate::main_window::PaneMode::New {
            self.new_form.start_idx.and_then(|i| images.get(i)).cloned()
        } else {
            None
        };

        let hover = if let Some(path) = image_path_new {
            let draw = if !solve_running { Some(&mut self.new_form.draw) } else { None };
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                self.image_viewer.show(ui, &path, cache, draw)
            })
            .inner
        } else if mode == crate::main_window::PaneMode::View {
            self.show_view_central(ui, viewer_rect, cache)
        } else {
            // New mode, no start image selected.
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                ui.painter()
                    .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("Select start index to preview reference image")
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

    fn show_view_central(
        &mut self,
        ui: &mut egui::Ui,
        viewer_rect: egui::Rect,
        cache: &mut TextureCache,
    ) -> Option<HoverInfo> {
        let n_frames = self.view.solution.as_ref().map(|s| s.mesh_solutions.len()).unwrap_or(0);

        if n_frames == 0 {
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                ui.painter()
                    .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("Select a sequence from the list")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                });
            });
            return None;
        }

        // Clamp current frame.
        if self.view.current_frame >= n_frames {
            self.view.current_frame = n_frames - 1;
            self.view.nodal_cache = None;
        }
        let current_frame = self.view.current_frame;

        // Carve out the frame slider strip at the bottom.
        const SLIDER_H: f32 = 40.0;
        let image_rect = egui::Rect::from_min_max(
            viewer_rect.min,
            egui::pos2(viewer_rect.max.x, viewer_rect.max.y - SLIDER_H),
        );
        let slider_rect = egui::Rect::from_min_max(
            egui::pos2(viewer_rect.min.x, viewer_rect.max.y - SLIDER_H),
            viewer_rect.max,
        );

        // Target image for the current frame is image_paths[frame + 1].
        let target_path = self.view.image_paths.get(current_frame + 1).cloned();
        let show_image = target_path.as_ref().map(|p| p.exists()).unwrap_or(false);

        let hover = if show_image {
            let path = target_path.unwrap();
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(image_rect), |ui| {
                self.image_viewer.show(ui, &path, cache, None)
            })
            .inner
        } else {
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(image_rect), |ui| {
                ui.painter()
                    .rect_filled(image_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
            });
            None
        };

        // Colormap overlay.
        let coord: Option<ImageCoord> = self.image_viewer.last_coord().or_else(|| {
            self.view
                .solution
                .as_ref()
                .and_then(|s| s.mesh_solutions.get(current_frame))
                .map(|m| mesh_coord_from_bounds(m, image_rect))
        });

        if let Some(coord) = coord {
            let plot_type = self.view.plot_type;
            let range_mode = self.view.range_mode;
            let manual_min = self.view.range_min;
            let manual_max = self.view.range_max;
            let colormap = self.view.colormap;
            let show_wireframe = self.view.show_wireframe;
            let show_node_indices = self.view.show_node_indices;

            // Update nodal cache if frame or plot type changed.
            let cache_valid = self.view
                .nodal_cache
                .as_ref()
                .map(|(f, t, _)| *f == current_frame && *t == plot_type)
                .unwrap_or(false);

            if !cache_valid {
                if let Some(mesh_sol) = self
                    .view
                    .solution
                    .as_ref()
                    .and_then(|s| s.mesh_solutions.get(current_frame))
                {
                    let vals = extract_nodal_values(mesh_sol, plot_type);
                    self.view.nodal_cache = Some((current_frame, plot_type, vals));
                }
            }

            if let Some((_, _, ref vals)) = self.view.nodal_cache {
                let (vmin, vmax) = match range_mode {
                    RangeMode::Manual => (manual_min, manual_max),
                    RangeMode::Auto => {
                        let mn = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                        let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        (mn, mx)
                    }
                };

                if let Some(mesh_sol) = self
                    .view
                    .solution
                    .as_ref()
                    .and_then(|s| s.mesh_solutions.get(current_frame))
                {
                    let mut painter = ui.ctx().layer_painter(egui::LayerId::new(
                        egui::Order::Foreground,
                        egui::Id::new("seq_overlay"),
                    ));
                    painter.set_clip_rect(image_rect);
                    render_mesh_overlay(
                        &painter,
                        mesh_sol,
                        vals,
                        vmin,
                        vmax,
                        colormap,
                        show_wireframe,
                        show_node_indices,
                        &coord,
                    );
                    render_colorbar(&painter, image_rect, vmin, vmax, colormap, plot_type.label());
                }
            }
        }

        // Frame slider strip.
        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(slider_rect), |ui| {
            ui.painter()
                .rect_filled(slider_rect, 0.0, egui::Color32::from_rgb(24, 24, 26));
            ui.horizontal_centered(|ui| {
                ui.add_space(10.0);
                let max_frame = n_frames.saturating_sub(1);
                let old_frame = self.view.current_frame;
                ui.add(
                    egui::Slider::new(&mut self.view.current_frame, 0..=max_frame)
                        .show_value(false)
                        .trailing_fill(true),
                );
                if self.view.current_frame != old_frame {
                    self.view.nodal_cache = None;
                    self.view.animate = false;
                }
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(format!(
                        "{} / {}",
                        self.view.current_frame + 1,
                        n_frames
                    ))
                    .size(15.0)
                    .color(ui.visuals().weak_text_color()),
                );
                ui.add_space(10.0);
            });
        });

        // Animation tick.
        if self.view.animate && n_frames > 1 {
            let now = Instant::now();
            let should_advance = self
                .view
                .last_tick
                .map(|t| now.duration_since(t).as_secs_f32() >= 1.0 / self.view.speed)
                .unwrap_or(true);
            if should_advance {
                self.view.current_frame = (current_frame + 1) % n_frames;
                self.view.nodal_cache = None;
                self.view.last_tick = Some(now);
            }
            let delay = std::time::Duration::from_secs_f32(1.0 / self.view.speed);
            ui.ctx().request_repaint_after(delay);
        }

        hover
    }

    // -----------------------------------------------------------------------
    // Right pane — view mode (§10.3)
    // -----------------------------------------------------------------------

    pub fn show_right_view(&mut self, ui: &mut egui::Ui, selected_path: Option<&Path>) {
        if self.view.loaded_path.as_deref() != selected_path {
            self.view.loaded_path = selected_path.map(|p| p.to_path_buf());
            self.view.solution = selected_path.and_then(|p| {
                geopyv_dev::io::load(p).ok().and_then(|obj| {
                    if let GeopyvObject::Sequence(s) = obj { Some(s) } else { None }
                })
            });
            self.view.current_frame = 0;
            self.view.nodal_cache = None;
            self.view.image_paths = Vec::new();
            self.view.animate = false;
        }

        let n_frames = match self.view.solution.as_ref() {
            Some(s) => s.mesh_solutions.len(),
            None => {
                ui.label(
                    egui::RichText::new("No sequence selected")
                        .size(15.0)
                        .color(ui.visuals().weak_text_color()),
                );
                return;
            }
        };

        // --- Sequence metadata ---
        {
            let sol = self.view.solution.as_ref().unwrap();
            section_header(ui, "Sequence");
            meta_row(ui, "Frames", &n_frames.to_string());
            meta_row(ui, "Solved", if sol.solved { "yes" } else { "no" });
            meta_row(ui, "Unsolvable", if sol.unsolvable { "yes" } else { "no" });
            if !sol.override_log.is_empty() {
                meta_row(ui, "Override frames", &sol.override_log.len().to_string());
            }
            if let Some(first) = sol.mesh_solutions.first() {
                meta_row(ui, "Nodes", &first.nodes.nrows().to_string());
                meta_row(ui, "Elements", &first.elements.nrows().to_string());
                meta_row(ui, "Mesh order", &first.mesh_order.to_string());
            }
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // --- Frame navigation ---
        section_header(ui, "Frame");
        ui.add_space(3.0);

        let max_frame = n_frames.saturating_sub(1);

        // Navigation buttons row.
        ui.horizontal(|ui| {
            if ui
                .add(egui::Button::new("\u{23ee}").min_size(egui::vec2(24.0, 22.0)))
                .clicked()
            {
                self.view.current_frame = 0;
                self.view.nodal_cache = None;
                self.view.animate = false;
            }
            if ui
                .add(egui::Button::new("\u{25c4}").min_size(egui::vec2(24.0, 22.0)))
                .clicked()
                && self.view.current_frame > 0
            {
                self.view.current_frame -= 1;
                self.view.nodal_cache = None;
                self.view.animate = false;
            }
            ui.label(
                egui::RichText::new(format!(
                    " {} / {} ",
                    self.view.current_frame + 1,
                    n_frames
                ))
                .size(16.0),
            );
            if ui
                .add(egui::Button::new("\u{25ba}").min_size(egui::vec2(24.0, 22.0)))
                .clicked()
                && self.view.current_frame < max_frame
            {
                self.view.current_frame += 1;
                self.view.nodal_cache = None;
                self.view.animate = false;
            }
            if ui
                .add(egui::Button::new("\u{23ed}").min_size(egui::vec2(24.0, 22.0)))
                .clicked()
            {
                self.view.current_frame = max_frame;
                self.view.nodal_cache = None;
                self.view.animate = false;
            }
        });

        // Slider.
        if n_frames > 1 {
            ui.add_space(2.0);
            let old_frame = self.view.current_frame;
            ui.add(
                egui::Slider::new(&mut self.view.current_frame, 0..=max_frame)
                    .show_value(false)
                    .trailing_fill(true),
            );
            if self.view.current_frame != old_frame {
                self.view.nodal_cache = None;
                self.view.animate = false;
            }
        }

        // Animate row.
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let anim_changed = ui.checkbox(&mut self.view.animate, "Animate").changed();
            if anim_changed && self.view.animate {
                self.view.last_tick = None;
            }
            if self.view.animate {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("Speed:")
                        .size(16.0)
                        .color(ui.visuals().weak_text_color()),
                );
                ui.add(
                    egui::DragValue::new(&mut self.view.speed)
                        .range(0.5f32..=30.0)
                        .speed(0.5)
                        .suffix("\u{00d7}"),
                );
            }
        });

        // Reference mode.
        ui.add_space(4.0);
        section_header(ui, "Reference");
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.view.reference_mode, ReferenceMode::Fixed, "Fixed");
            ui.radio_value(
                &mut self.view.reference_mode,
                ReferenceMode::Sequential,
                "Sequential",
            );
        });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // --- Plot type selector ---
        section_header(ui, "Plot Type");
        ui.add_space(3.0);

        let old_plot = self.view.plot_type;

        ui.label(
            egui::RichText::new("\u{2500}\u{2500} Displacement \u{2500}\u{2500}")
                .size(15.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::Ux, "u_x");
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::Uy, "u_y");
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::UMag, "|u|");
        });

        ui.add_space(3.0);
        ui.label(
            egui::RichText::new("\u{2500}\u{2500} Strain \u{2500}\u{2500}")
                .size(15.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(
                &mut self.view.plot_type,
                MeshPlotType::Exx,
                "\u{03b5}_xx",
            );
            ui.radio_value(
                &mut self.view.plot_type,
                MeshPlotType::Eyy,
                "\u{03b5}_yy",
            );
            ui.radio_value(
                &mut self.view.plot_type,
                MeshPlotType::Exy,
                "\u{03b5}_xy",
            );
            ui.radio_value(
                &mut self.view.plot_type,
                MeshPlotType::EVM,
                "\u{03b5}_VM",
            );
        });

        ui.add_space(3.0);
        ui.label(
            egui::RichText::new("\u{2500}\u{2500} Quality \u{2500}\u{2500}")
                .size(15.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::CZncc, "C_ZNCC");
            ui.radio_value(
                &mut self.view.plot_type,
                MeshPlotType::Iterations,
                "Iterations",
            );
            ui.radio_value(&mut self.view.plot_type, MeshPlotType::Norms, "Norms");
        });

        if self.view.plot_type != old_plot {
            self.view.nodal_cache = None;
        }

        ui.add_space(3.0);
        ui.label(
            egui::RichText::new("\u{2500}\u{2500} Geometry \u{2500}\u{2500}")
                .size(15.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.checkbox(&mut self.view.show_wireframe, "Show wireframe");
        ui.checkbox(&mut self.view.show_node_indices, "Show node indices");

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // --- Colormap selector ---
        section_header(ui, "Colormap");
        ui.add_space(3.0);
        egui::ComboBox::from_id_salt("seq_colormap")
            .selected_text(self.view.colormap.label())
            .width(120.0)
            .show_ui(ui, |ui| {
                for &cmap in ColormapType::ALL {
                    ui.selectable_value(&mut self.view.colormap, cmap, cmap.label());
                }
            });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // --- Range controls ---
        section_header(ui, "Range");
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.view.range_mode, RangeMode::Auto, "Auto");
            ui.radio_value(&mut self.view.range_mode, RangeMode::Manual, "Manual");
        });

        if self.view.range_mode == RangeMode::Manual {
            ui.add_space(3.0);
            egui::Grid::new("seq_range_grid")
                .num_columns(2)
                .spacing([6.0, 4.0])
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("Min:")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
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
                    ui.label(
                        egui::RichText::new("Max:")
                            .size(16.0)
                            .color(ui.visuals().weak_text_color()),
                    );
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
        } else {
            // Show auto range for reference using the current frame's nodal cache.
            let (mn, mx) = self
                .view
                .nodal_cache
                .as_ref()
                .filter(|(f, t, _)| {
                    *f == self.view.current_frame && *t == self.view.plot_type
                })
                .map(|(_, _, v)| {
                    let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
                    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    (mn, mx)
                })
                .unwrap_or((0.0, 1.0));
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!(
                        "{} \u{2013} {}",
                        format_sci(mn),
                        format_sci(mx)
                    ))
                    .size(15.0)
                    .color(ui.visuals().weak_text_color()),
                );
            });
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // --- Quality for current frame ---
        {
            let sol = self.view.solution.as_ref().unwrap();
            if let Some(mesh_sol) = sol.mesh_solutions.get(self.view.current_frame) {
                let n = mesh_sol.c_zncc.len();
                if n > 0 {
                    section_header(ui, "Quality (current frame)");
                    let mean = mesh_sol.c_zncc.iter().sum::<f64>() / n as f64;
                    let min = mesh_sol.c_zncc.iter().cloned().fold(f64::INFINITY, f64::min);
                    meta_row(ui, "Mean C_ZNCC", &format!("{mean:.4}"));
                    meta_row(ui, "Min C_ZNCC", &format!("{min:.4}"));
                    let below = mesh_sol.c_zncc.iter().filter(|&&v| v < 0.75).count();
                    meta_row(ui, "Below 0.75", &below.to_string());
                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(6.0);
                }
            }
        }

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
    // Right pane — new mode
    // -----------------------------------------------------------------------

    pub fn show_right_new(
        &mut self,
        ui: &mut egui::Ui,
        images: &[PathBuf],
    ) -> Option<SequenceSpawnParams> {
        if self.is_solving() {
            let (progress, message) = {
                let s = self.solve_state.lock().unwrap();
                (s.progress, s.message.clone())
            };
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Solving sequence\u{2026}")
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
        let mut spawn: Option<SequenceSpawnParams> = None;

        // Basic fields.
        egui::Grid::new("seq_new_grid")
            .num_columns(2)
            .spacing([8.0, 5.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(lbl("Name:"));
                ui.add(egui::TextEdit::singleline(&mut form.name).desired_width(150.0));
                ui.end_row();
            });

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        // Template section.
        section_header(ui, "Template");
        ui.add_space(4.0);

        egui::Grid::new("seq_template_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(lbl("Shape:"));
                ui.horizontal(|ui| {
                    ui.radio_value(&mut form.template_shape, TemplateShape::Circle, "Circle");
                    ui.radio_value(&mut form.template_shape, TemplateShape::Square, "Square");
                });
                ui.end_row();

                ui.label(lbl("Size (px):"));
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

        ui.add_space(4.0);

        // Image range.
        section_header(ui, "Image Range");
        ui.add_space(3.0);
        ui.label(
            egui::RichText::new(format!("{} images available", images.len()))
                .size(15.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(3.0);

        egui::Grid::new("seq_range_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(lbl("Start (1-based):"));
                let r = ui.add(
                    egui::TextEdit::singleline(&mut form.start_text).desired_width(60.0),
                );
                if r.changed() {
                    form.start_idx = parse_1based_index(&form.start_text, images.len());
                }
                ui.end_row();

                ui.label(lbl("End (1-based):"));
                let r = ui.add(
                    egui::TextEdit::singleline(&mut form.end_text).desired_width(60.0),
                );
                if r.changed() {
                    form.end_idx = parse_1based_index(&form.end_text, images.len());
                }
                ui.end_row();
            });

        ui.add_space(2.0);
        match (form.start_idx, form.end_idx) {
            (Some(s), Some(e)) if e > s => {
                let n_images = e - s + 1;
                let n_pairs = n_images - 1;
                ui.label(
                    egui::RichText::new(format!("{n_images} images \u{2192} {n_pairs} pairs"))
                        .size(15.0)
                        .color(egui::Color32::from_rgb(100, 200, 100)),
                );
            }
            (Some(_), Some(_)) => {
                ui.label(
                    egui::RichText::new("End must be > Start")
                        .size(15.0)
                        .color(ui.visuals().error_fg_color),
                );
            }
            _ => {
                ui.label(
                    egui::RichText::new("Enter start and end indices")
                        .size(15.0)
                        .color(ui.visuals().weak_text_color()),
                );
            }
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Mesh generation.
        section_header(ui, "Mesh Generation");
        ui.add_space(4.0);
        {
            let g = &mut form.gen_cfg;
            egui::Grid::new("seq_gen_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .min_col_width(72.0)
                .show(ui, |ui| {
                    ui.label(lbl("Mesh order:"));
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut g.mesh_order, MeshOrder::First, "1");
                        ui.radio_value(&mut g.mesh_order, MeshOrder::Second, "2");
                    });
                    ui.end_row();

                    ui.label(lbl("Size lower:"));
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

                    ui.label(lbl("Size upper:"));
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

                    ui.label(lbl("Target nodes:"));
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
                        .size(15.0)
                        .color(ui.visuals().error_fg_color),
                );
            }
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Solver.
        section_header(ui, "Solver");
        ui.add_space(4.0);
        {
            let s = &mut form.solver;
            egui::Grid::new("seq_solver_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .min_col_width(72.0)
                .show(ui, |ui| {
                    ui.label(lbl("Method:"));
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut s.method, SolveMethod::Icgn, "ICGN");
                        ui.radio_value(&mut s.method, SolveMethod::Fagn, "FAGN");
                    });
                    ui.end_row();

                    ui.label(lbl("Order:"));
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut s.order, SubsetOrder::First, "1");
                        ui.radio_value(&mut s.order, SubsetOrder::Second, "2");
                    });
                    ui.end_row();

                    ui.label(lbl("Max norm:"));
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

                    ui.label(lbl("Max iters:"));
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut s.max_iterations_text).desired_width(80.0),
                    );
                    if r.changed() {
                        match s.max_iterations_text.trim().parse::<usize>() {
                            Ok(v) if v > 0 => {
                                s.max_iterations = v;
                                s.parse_error = None;
                            }
                            _ => s.parse_error =
                                Some("max_iterations must be a positive integer".into()),
                        }
                    }
                    ui.end_row();

                    ui.label(lbl("ZNCC tol:"));
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
                        .size(15.0)
                        .color(ui.visuals().error_fg_color),
                );
            }
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Sequence options.
        section_header(ui, "Sequence Options");
        ui.add_space(4.0);
        {
            let o = &mut form.seq_opts;
            egui::Grid::new("seq_opts_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .min_col_width(72.0)
                .show(ui, |ui| {
                    ui.label(lbl("Border (px):"));
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut o.border_text).desired_width(60.0),
                    );
                    if r.changed() {
                        match o.border_text.trim().parse::<usize>() {
                            Ok(v) => {
                                o.border = v;
                                o.parse_error = None;
                            }
                            _ => o.parse_error =
                                Some("Border must be a non-negative integer".into()),
                        }
                    }
                    ui.end_row();
                });

            ui.add_space(3.0);
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut o.guide, "Guide");
                ui.checkbox(&mut o.sequential, "Sequential");
                ui.checkbox(&mut o.sync, "Sync");
                ui.checkbox(&mut o.override_, "Override");
            });

            if let Some(err) = &o.parse_error.clone() {
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

        // Define region.
        section_header(ui, "Define Region");
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            for (label, mode) in [
                ("Rectangular", DrawShapeMode::Rectangular),
                ("Circular",    DrawShapeMode::Circular),
                ("Free",        DrawShapeMode::Free),
            ] {
                let resp = ui.selectable_value(&mut form.draw.shape_mode, mode, label);
                if resp.changed() {
                    form.draw.reset_in_progress();
                }
            }
            if form.draw.shape_mode == DrawShapeMode::Circular {
                ui.add_space(8.0);
                ui.label("Points:");
                ui.add(egui::DragValue::new(&mut form.draw.circle_n_points).range(6..=200).speed(1.0));
            }
        });
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            let b_active = form.draw.mode == Some(ActiveDrawMode::Boundary);
            let btn_b = egui::Button::new(
                egui::RichText::new(if b_active {
                    "Drawing\u{2026}"
                } else {
                    "Boundary \u{25b6}"
                })
                .size(16.0),
            )
            .selected(b_active);
            if ui.add(btn_b).clicked() {
                if b_active {
                    form.draw.cancel();
                } else {
                    form.draw.start(ActiveDrawMode::Boundary);
                }
            }

            let e_active = form.draw.mode == Some(ActiveDrawMode::Exclusion);
            let btn_e = egui::Button::new(
                egui::RichText::new(if e_active {
                    "Drawing\u{2026}"
                } else {
                    "Exclusion \u{25b6}"
                })
                .size(16.0),
            )
            .selected(e_active);
            if ui.add(btn_e).clicked() {
                if e_active {
                    form.draw.cancel();
                } else {
                    form.draw.start(ActiveDrawMode::Exclusion);
                }
            }

            let s_active = form.draw.mode == Some(ActiveDrawMode::Seed);
            let btn_s = egui::Button::new(
                egui::RichText::new(if s_active {
                    "Placing\u{2026}"
                } else {
                    "Seed \u{25b6}"
                })
                .size(16.0),
            )
            .selected(s_active);
            if ui.add(btn_s).clicked() {
                if s_active {
                    form.draw.cancel();
                } else {
                    form.draw.start(ActiveDrawMode::Seed);
                }
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
            ui.label(egui::RichText::new(b_text).size(15.0).color(b_color));
            ui.add_space(8.0);

            let exc_n = form.draw.exclusions.len();
            let e_color = if exc_n > 0 {
                egui::Color32::from_rgb(100, 200, 100)
            } else {
                ui.visuals().weak_text_color()
            };
            ui.label(
                egui::RichText::new(format!("exclusions: {exc_n}"))
                    .size(15.0)
                    .color(e_color),
            );
            ui.add_space(8.0);

            let (s_text, s_color) = if form.draw.seed_ok() {
                ("seed \u{2713}", egui::Color32::from_rgb(100, 200, 100))
            } else {
                ("seed \u{2014}", ui.visuals().weak_text_color())
            };
            ui.label(egui::RichText::new(s_text).size(15.0).color(s_color));
        });

        if form.draw.has_self_intersection() {
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new("Self-intersecting polygon")
                    .size(15.0)
                    .color(ui.visuals().error_fg_color),
            );
        }
        if form.draw.exclusion_out_of_bounds {
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new("Exclusion must be within boundary.")
                    .size(13.0)
                    .color(ui.visuals().error_fg_color),
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
                let seed = form.draw.seed.unwrap();
                let boundary: Vec<[f64; 2]> = form.draw.boundary.as_ref().unwrap().to_nodes();
                let exclusions: Vec<Vec<[f64; 2]>> =
                    form.draw.exclusions.iter().map(|r| r.to_nodes()).collect();
                let start = form.start_idx.unwrap();
                let end = form.end_idx.unwrap();
                let image_paths: Vec<PathBuf> = images[start..=end].to_vec();
                spawn = Some(SequenceSpawnParams {
                    name: form.name.trim().to_string(),
                    image_paths,
                    template_shape: form.template_shape.clone(),
                    template_size: form.template_size,
                    boundary,
                    exclusions,
                    seed: [seed.x as f64, seed.y as f64],
                    mesh_order: if form.gen_cfg.mesh_order == MeshOrder::First {
                        1
                    } else {
                        2
                    },
                    size_lower: form.gen_cfg.size_lower,
                    size_upper: form.gen_cfg.size_upper,
                    target_nodes: form.gen_cfg.target_nodes,
                    method: form.solver.method,
                    subset_order: form.solver.order,
                    max_norm: form.solver.max_norm,
                    max_iterations: form.solver.max_iterations,
                    zncc_tol: form.solver.zncc_tol,
                    guide: form.seq_opts.guide,
                    sequential: form.seq_opts.sequential,
                    sync: form.seq_opts.sync,
                    override_: form.seq_opts.override_,
                    border: form.seq_opts.border,
                    mesh_subdir: PathBuf::new(),
                    sequences_dir: PathBuf::new(),
                });
                form.form_error = None;
            }
        });

        spawn
    }
}

// ---------------------------------------------------------------------------
// Progress overlay (central panel)
// ---------------------------------------------------------------------------

fn show_progress_overlay(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    state: &Arc<Mutex<SequenceSolveState>>,
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
            egui::FontId::new(15.0, egui::FontFamily::Proportional),
            egui::Color32::from_rgb(200, 200, 200),
        );
    }
}

// ---------------------------------------------------------------------------
// Background solve thread
// ---------------------------------------------------------------------------

fn set_progress(state: &Arc<Mutex<SequenceSolveState>>, progress: f32, msg: &str) {
    if let Ok(mut s) = state.lock() {
        s.progress = progress;
        s.message = msg.to_string();
    }
}

fn set_error(state: &Arc<Mutex<SequenceSolveState>>, msg: String) {
    if let Ok(mut s) = state.lock() {
        s.result = Some(Err(msg));
        s.running = false;
        s.progress = 0.0;
    }
}

fn run_solve(
    params: SequenceSpawnParams,
    state: Arc<Mutex<SequenceSolveState>>,
    cancel: Arc<AtomicBool>,
) {
    // Build template.
    set_progress(&state, 0.02, "Building template\u{2026}");
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
        if let Ok(mut s) = state.lock() {
            s.running = false;
        }
        return;
    }

    // Build ROI from boundary / exclusions.
    set_progress(&state, 0.04, "Building mesh region\u{2026}");
    let boundary_arr = Array2::from_shape_fn((params.boundary.len(), 2), |(i, j)| {
        params.boundary[i][j]
    });
    let exclusion_arrs: Vec<Array2<f64>> = params
        .exclusions
        .iter()
        .map(|ex| Array2::from_shape_fn((ex.len(), 2), |(i, j)| ex[i][j]))
        .collect();
    let exclusion_views: Vec<_> = exclusion_arrs.iter().map(|a| a.view()).collect();
    let roi = define_roi(boundary_arr.view(), true, &exclusion_views, None);

    if cancel.load(Ordering::Relaxed) {
        if let Ok(mut s) = state.lock() {
            s.running = false;
        }
        return;
    }

    // Helper: generate a fresh mesh from the ROI for this sequence.
    let generate_fresh_mesh = || -> Result<Mesh, geopyv_dev::Error> {
        Mesh::generate(
            roi.borders.view(),
            roi.segments.view(),
            &roi.curves,
            params.size_lower,
            params.size_upper,
            params.target_nodes,
            params.mesh_order,
        )
    };

    // Prepare solve config.
    let subset_order_int = if matches!(params.subset_order, SubsetOrder::First) {
        1usize
    } else {
        2
    };
    let mesh_cfg = SolveConfig {
        max_norm: params.max_norm,
        max_iterations: params.max_iterations,
        subset_order: subset_order_int,
        tolerance: params.zncc_tol,
        method: match params.method {
            SolveMethod::Icgn => LibSolveMethod::Icgn,
            SolveMethod::Fagn => LibSolveMethod::Fagn,
        },
    };
    let seed_coord_init = params.seed;
    let mut seed_coord = seed_coord_init;
    let mut seed_warp = vec![0.0f64; 12];

    let n_images = params.image_paths.len();
    let n_pairs = n_images - 1;

    let mut mesh_solutions: Vec<MeshSolution> = Vec::with_capacity(n_pairs);
    let mut override_log: Vec<usize> = Vec::new();
    let mut sync_sol: Option<MeshSolution> = None;
    let mut mesh_override = false;

    let mut f_index = 0usize;
    let mut g_index = 1usize;

    // Spawn background I/O thread so frame writes don't block the solve loop.
    let (io_tx, io_rx) = std::sync::mpsc::sync_channel::<(PathBuf, MeshSolution)>(4);
    let io_state = Arc::clone(&state);
    let io_thread = std::thread::spawn(move || {
        while let Ok((path, mesh_sol)) = io_rx.recv() {
            if let Err(e) = geopyv_dev::io::save(&path, &GeopyvObject::Mesh(mesh_sol)) {
                set_error(&io_state, format!("Frame save error: {e}"));
                break;
            }
        }
    });

    // Load initial reference image.
    let mut f_img = match Image::from_file(&params.image_paths[f_index], params.border) {
        Ok(img) => img,
        Err(e) => {
            set_error(&state, format!("Reference image error: {e}"));
            return;
        }
    };

    let base_progress = 0.08f32;
    let pair_budget = 1.0f32 - base_progress;

    'outer: loop {
        let pair_num = g_index;
        let progress =
            base_progress + pair_budget * (mesh_solutions.len() as f32 / n_pairs as f32);
        set_progress(
            &state,
            progress,
            &format!("Frame {pair_num}/{n_pairs}\u{2026}"),
        );

        if cancel.load(Ordering::Relaxed) {
            if let Ok(mut s) = state.lock() {
                s.running = false;
            }
            return;
        }

        // Build / reuse mesh.
        let mesh = match (params.sync, &sync_sol) {
            (true, Some(prev)) => Mesh::from_solution(prev),
            _ => match generate_fresh_mesh() {
                Ok(m) => m,
                Err(e) => {
                    set_error(&state, format!("Mesh generation error: {e}"));
                    return;
                }
            },
        };

        // Load target image.
        let g_img = match Image::from_file(&params.image_paths[g_index], params.border) {
            Ok(img) => img,
            Err(e) => {
                set_error(&state, format!("Target image {g_index} error: {e}"));
                return;
            }
        };

        // Solve config (possibly override).
        let pair_cfg = if mesh_override {
            SolveConfig { tolerance: 0.0, ..mesh_cfg.clone() }
        } else {
            mesh_cfg.clone()
        };

        // Solve this pair.
        let pair_result = mesh.solve(
            &f_img,
            &g_img,
            &template.coords,
            seed_coord,
            &seed_warp,
            &pair_cfg,
            params.image_paths[f_index].clone(),
            params.image_paths[g_index].clone(),
        );

        let mesh_sol = match pair_result {
            Ok(sol) => sol,
            Err(_) => {
                if f_index + 1 < g_index {
                    f_index = g_index - 1;
                    f_img =
                        match Image::from_file(&params.image_paths[f_index], params.border) {
                            Ok(img) => img,
                            Err(e) => {
                                set_error(&state, format!("Image {f_index} error: {e}"));
                                return;
                            }
                        };
                    if params.sync {
                        sync_sol = None;
                    }
                    if params.override_ {
                        mesh_override = true;
                    }
                    continue 'outer;
                } else {
                    drop(io_tx);
                    let _ = io_thread.join();
                    let sol = SequenceSolution {
                        mesh_solutions,
                        mesh_paths: vec![],
                        solved: false,
                        unsolvable: true,
                        override_log,
                    };
                    save_frames_and_sequence(&sol, &params, &state);
                    return;
                }
            }
        };

        if mesh_override {
            if mesh_sol.c_zncc.iter().any(|&c| c < mesh_cfg.tolerance) {
                override_log.push(g_index);
            }
            mesh_override = false;
        }

        // Queue frame for background I/O.
        let frame_path = params
            .mesh_subdir
            .join(format!("frame_{:03}.pyv", mesh_solutions.len()));
        if io_tx.send((frame_path, mesh_sol.clone())).is_err() {
            return;
        }

        if params.sync {
            sync_sol = Some(mesh_sol.clone());
        }
        mesh_solutions.push(mesh_sol.clone());

        g_index += 1;
        if g_index >= n_images {
            break 'outer;
        }

        if params.guide {
            let (disp, new_warp) = deformation_preconditioning(
                &mesh_sol,
                seed_coord,
                params.mesh_order,
                subset_order_int as u8,
            );
            seed_coord[0] += disp[0];
            seed_coord[1] += disp[1];
            let n_copy = (6 * params.mesh_order as usize)
                .min(6 * subset_order_int)
                .min(new_warp.len())
                .min(12);
            for i in 0..n_copy {
                seed_warp[i] = new_warp[i];
            }
            for i in n_copy..12 {
                seed_warp[i] = 0.0;
            }
        }

        if params.sequential {
            f_index = g_index - 1;
            f_img =
                match Image::from_file(&params.image_paths[f_index], params.border) {
                    Ok(img) => img,
                    Err(e) => {
                        set_error(&state, format!("Image {f_index} error: {e}"));
                        return;
                    }
                };
            if params.sync {
                sync_sol = None;
            }
        }
    }

    let sol = SequenceSolution {
        mesh_solutions,
        mesh_paths: vec![],
        solved: true,
        unsolvable: false,
        override_log,
    };

    // Wait for all queued frame writes to complete before saving the sequence.
    drop(io_tx);
    let _ = io_thread.join();

    let seq_path = params.sequences_dir.join(format!("{}.pyv", params.name));
    if let Err(e) = geopyv_dev::io::save(&seq_path, &GeopyvObject::Sequence(sol.clone())) {
        set_error(&state, format!("Sequence save error: {e}"));
        return;
    }

    set_progress(&state, 1.0, "Done");
    if let Ok(mut s) = state.lock() {
        s.result = Some(Ok(sol));
        s.running = false;
    }
}

fn save_frames_and_sequence(
    sol: &SequenceSolution,
    params: &SequenceSpawnParams,
    state: &Arc<Mutex<SequenceSolveState>>,
) {
    let seq_path = params.sequences_dir.join(format!("{}.pyv", params.name));
    if let Err(e) = geopyv_dev::io::save(&seq_path, &GeopyvObject::Sequence(sol.clone())) {
        set_error(state, format!("Sequence save error: {e}"));
        return;
    }
    set_progress(state, 1.0, "Curtailed (unsolvable pair)");
    if let Ok(mut s) = state.lock() {
        s.result = Some(Ok(sol.clone()));
        s.running = false;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_1based_index(text: &str, n: usize) -> Option<usize> {
    text.trim().parse::<usize>().ok().and_then(|v| {
        if v >= 1 && v <= n {
            Some(v - 1)
        } else {
            None
        }
    })
}

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
