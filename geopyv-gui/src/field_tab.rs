use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;
use ndarray::Array2;

use geopyv_dev::field::{grid_particles, Field, FieldDistribution, FieldSolution};
use geopyv_dev::io::GeopyvObject;

use crate::field_view::FieldViewState;
use crate::draw::{point_in_polygon, ActiveDrawMode, DrawShapeMode, DrawState, DrawnRegion};
use crate::image_viewer::{HoverInfo, ImageViewer, TextureCache};

// ---------------------------------------------------------------------------
// Mesh outline (reference frame of the selected sequence)
// ---------------------------------------------------------------------------

/// Boundary and exclusion outlines of the sequence's first solved mesh, in
/// image-pixel space, plus the reference image they belong to.
pub struct MeshOutline {
    pub sequence_path: PathBuf,
    pub image: Option<PathBuf>,
    pub boundary: Vec<egui::Pos2>,
    pub exclusions: Vec<Vec<egui::Pos2>>,
    pub error: Option<String>,
}

impl MeshOutline {
    fn load(sequence_path: &Path) -> Self {
        let mut outline = MeshOutline {
            sequence_path: sequence_path.to_path_buf(),
            image: None,
            boundary: Vec::new(),
            exclusions: Vec::new(),
            error: None,
        };
        let seq = match geopyv_dev::io::load(sequence_path) {
            Ok(GeopyvObject::Sequence(s)) => s,
            Ok(_) => {
                outline.error = Some("File is not a sequence".to_string());
                return outline;
            }
            Err(e) => {
                outline.error = Some(format!("Could not load sequence: {e}"));
                return outline;
            }
        };
        if seq.n_meshes() == 0 {
            outline.error = Some("Sequence contains no mesh solutions".to_string());
            return outline;
        }
        let mesh = match seq.load_mesh_at(0) {
            Ok(m) => m,
            Err(e) => {
                outline.error = Some(format!("Could not load first mesh: {e}"));
                return outline;
            }
        };
        let loop_of = |idx: &[usize]| -> Vec<egui::Pos2> {
            idx.iter()
                .map(|&i| egui::pos2(mesh.nodes[[i, 0]] as f32, mesh.nodes[[i, 1]] as f32))
                .collect()
        };
        outline.boundary = loop_of(&mesh.boundary);
        outline.exclusions = mesh.exclusions.iter().map(|e| loop_of(e)).collect();
        outline.image = seq.first_f_img_path.clone().or_else(|| Some(mesh.f_img_path.clone()));
        outline
    }

    /// `true` if `p` lies inside the mesh boundary and outside every exclusion.
    pub fn contains(&self, p: egui::Pos2) -> bool {
        point_in_polygon(p, &self.boundary)
            && !self.exclusions.iter().any(|e| point_in_polygon(p, e))
    }
}

// ---------------------------------------------------------------------------
// New field form
// ---------------------------------------------------------------------------

pub struct NewFieldForm {
    pub name: String,
    pub seq_idx: Option<usize>,
    /// Outline of the selected sequence's first mesh; reloaded when the
    /// selection changes.
    pub mesh_outline: Option<MeshOutline>,
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
    /// Per preview particle: inside the mesh (boundary minus exclusions).
    cached_inside: Vec<bool>,
    last_signature: u64,
}

impl Default for NewFieldForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            seq_idx: None,
            mesh_outline: None,
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
            cached_inside: Vec::new(),
            last_signature: 0,
        }
    }
}

/// A drawn region as an `(N, 2)` `[x, y]` vertex array (image-pixel space).
fn polygon_array(region: &DrawnRegion) -> Array2<f64> {
    let verts = region.to_nodes();
    Array2::from_shape_fn((verts.len(), 2), |(i, j)| verts[i][j])
}

impl NewFieldForm {
    /// Hash of everything the preview depends on: drawn vertices, spacing and
    /// the mesh outline in use.
    fn grid_signature(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let mut put = |r: &DrawnRegion| {
            for p in r.to_egui_verts() {
                p.x.to_bits().hash(&mut h);
                p.y.to_bits().hash(&mut h);
            }
            u32::MAX.hash(&mut h);
        };
        if let Some(b) = &self.draw.boundary {
            put(b);
        }
        for e in &self.draw.exclusions {
            put(e);
        }
        self.spacing.to_bits().hash(&mut h);
        self.mesh_outline.as_ref().map(|o| &o.sequence_path).hash(&mut h);
        h.finish()
    }

    /// Reload the mesh outline if the selected sequence changed.
    pub fn ensure_outline_current(&mut self, sequences: &[PathBuf]) {
        let selected = self.seq_idx.and_then(|i| sequences.get(i));
        let current = self.mesh_outline.as_ref().map(|o| &o.sequence_path);
        if selected != current {
            self.mesh_outline = selected.map(|p| MeshOutline::load(p));
        }
    }

    /// Number of preview particles outside the mesh.
    pub fn n_outside(&self) -> usize {
        self.cached_inside.iter().filter(|&&b| !b).count()
    }

    pub fn ensure_grid_current(&mut self) {
        let signature = self.grid_signature();
        if signature == self.last_signature {
            return;
        }
        self.last_signature = signature;

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
        self.cached_inside = match &self.mesh_outline {
            Some(o) if !o.boundary.is_empty() => {
                self.cached_grid.iter().map(|&p| o.contains(p)).collect()
            }
            _ => vec![true; self.cached_grid.len()],
        };
    }

    pub fn invalidate_grid(&mut self) {
        self.last_signature = self.last_signature.wrapping_add(1);
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
        cache: &mut TextureCache,
    ) -> Option<HoverInfo> {
        let solve_running = self.is_solving();

        if mode == crate::main_window::PaneMode::View {
            return self.view.show_central(ui, viewer_rect, selected_path, cache);
        }
        self.view.cancel_png();

        // ----------------------------------------------------------------
        // New mode: the sequence's reference image + its mesh outline +
        // boundary/exclusion drawing + grid preview.
        // ----------------------------------------------------------------
        let image_path = self
            .new_form
            .mesh_outline
            .as_ref()
            .and_then(|o| o.image.clone());

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

            if let Some(coord) = self.image_viewer.last_coord() {
                let painter = ui.painter_at(viewer_rect);
                if let Some(outline) = &self.new_form.mesh_outline {
                    draw_outline(&painter, &coord, &outline.boundary, MESH_BOUNDARY_COLOUR);
                    for e in &outline.exclusions {
                        draw_outline(&painter, &coord, e, MESH_EXCLUSION_COLOUR);
                    }
                }
                if !solve_running {
                    self.new_form.ensure_grid_current();
                    for (&pt, &inside) in self
                        .new_form
                        .cached_grid
                        .iter()
                        .zip(&self.new_form.cached_inside)
                    {
                        let colour = if inside {
                            egui::Color32::from_rgb(255, 200, 50)
                        } else {
                            egui::Color32::from_rgb(230, 50, 50)
                        };
                        painter.circle_filled(coord.to_screen(pt), 2.5, colour);
                    }
                }
            }

            hover
        } else {
            let message = match &self.new_form.mesh_outline {
                Some(MeshOutline { error: Some(e), .. }) => e.clone(),
                _ => "Select a sequence".to_string(),
            };
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                ui.painter()
                    .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new(message)
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
        self.view.show_right(ui, selected_path, out_dir);
    }

    // -----------------------------------------------------------------------
    // Right pane — new mode (§9.6)
    // -----------------------------------------------------------------------

    pub fn show_right_new(
        &mut self,
        ui: &mut egui::Ui,
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
        if form.seq_idx.is_none() && !sequences.is_empty() {
            form.seq_idx = Some(0);
        }

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

            });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        form.ensure_outline_current(sequences);
        if let Some(MeshOutline { error: Some(e), .. }) = &form.mesh_outline {
            ui.label(egui::RichText::new(e).size(15.0).color(ui.visuals().error_fg_color));
        } else if form.mesh_outline.is_some() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("- - mesh boundary").size(14.0).color(MESH_BOUNDARY_COLOUR));
                ui.add_space(8.0);
                ui.label(egui::RichText::new("- - mesh exclusion").size(14.0).color(MESH_EXCLUSION_COLOUR));
            });
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

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
                egui::RichText::new(if b_active { "Drawing\u{2026}" } else { "Boundary \u{25b6}" }).size(16.0),
            )
            .selected(b_active);
            if ui.add(btn_b).clicked() {
                if b_active { form.draw.cancel(); } else { form.draw.start(ActiveDrawMode::Boundary); }
            }

            let e_active = form.draw.mode == Some(ActiveDrawMode::Exclusion);
            let btn_e = egui::Button::new(
                egui::RichText::new(if e_active { "Drawing\u{2026}" } else { "Exclusion \u{25b6}" }).size(16.0),
            )
            .selected(e_active);
            if ui.add_enabled(form.draw.boundary_ok(), btn_e).clicked() {
                if e_active { form.draw.cancel(); } else { form.draw.start(ActiveDrawMode::Exclusion); }
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
            let e_color = if exc_n > 0 { egui::Color32::from_rgb(100, 200, 100) } else { ui.visuals().weak_text_color() };
            ui.label(egui::RichText::new(format!("exclusions: {exc_n}")).size(15.0).color(e_color));
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

        let n_outside = form.n_outside();
        if n_outside > 0 {
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(format!("\u{26a0} {n_outside} particle(s) outside the mesh"))
                    .size(15.0)
                    .color(ui.visuals().warn_fg_color),
            );
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

    pub fn action_save(&mut self, out_dir: &Path, selected_path: Option<&Path>) {
        let dest = out_dir.join(self.view.png_name(selected_path));
        self.view.request_png(dest);
    }

    pub fn action_save_as(&mut self, selected_path: Option<&Path>) {
        let Some(dest) = rfd::FileDialog::new()
            .set_title("Save as PNG")
            .set_file_name(self.view.png_name(selected_path))
            .add_filter("PNG", &["png"])
            .save_file()
        else {
            return;
        };
        self.view.request_png(dest);
    }
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

const MESH_BOUNDARY_COLOUR: egui::Color32 = egui::Color32::from_rgb(0, 210, 230);
const MESH_EXCLUSION_COLOUR: egui::Color32 = egui::Color32::from_rgb(230, 70, 70);

/// Closed dashed outline of an image-space polygon.
fn draw_outline(
    painter: &egui::Painter,
    coord: &crate::draw::ImageCoord,
    verts: &[egui::Pos2],
    colour: egui::Color32,
) {
    if verts.len() < 2 {
        return;
    }
    let mut pts: Vec<egui::Pos2> = verts.iter().map(|&p| coord.to_screen(p)).collect();
    pts.push(pts[0]);
    painter.extend(egui::Shape::dashed_line(
        &pts,
        egui::Stroke::new(1.5, colour),
        6.0,
        4.0,
    ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use geopyv_dev::geometry::region::{Region, RegionOption};
    use geopyv_dev::masks::{LocalMask, MaskShape};
    use geopyv_dev::mesh::{SeedConfig, SolveConfig};
    use geopyv_dev::sequence::{Sequence, SequenceMeshConfig, SequenceOptions, SequenceSolveConfig};
    use ndarray::array;

    /// The outline drawn on the Field tab is the first mesh's boundary and
    /// exclusion, on that mesh's reference image.
    #[test]
    fn mesh_outline_matches_first_mesh_regions() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../images/shear");
        let images: Vec<PathBuf> = (0..2).map(|i| dir.join(format!("shear_{i}.jpg"))).collect();
        if !images.iter().all(|p| p.exists()) {
            eprintln!("images/shear not found; skipping");
            return;
        }
        let boundary = array![[300.0, 300.0], [700.0, 300.0], [700.0, 700.0], [300.0, 700.0]];
        let exclusion = array![[550.0, 550.0], [650.0, 550.0], [650.0, 650.0], [550.0, 650.0]];
        let region = |a: Array2<f64>, hard| Region::path(None, a, RegionOption::S, hard, false, 0.0).unwrap();
        let mut seq = Sequence::new(
            images.clone(),
            SequenceMeshConfig {
                boundary: region(boundary, true),
                exclusions: vec![region(exclusion, false)],
                size: (15.0, 70.0),
                target_nodes: 30,
                mesh_order: 1,
            },
        )
        .unwrap();
        let cfg = SequenceSolveConfig {
            mesh_cfg: SolveConfig { subset_order: 1, ..Default::default() },
            local_mask: LocalMask::new(MaskShape::Circle, 25).unwrap(),
            seed: SeedConfig {
                coord: [400.0, 400.0],
                warp: vec![0.0; 6],
                tolerance: SeedConfig::DEFAULT_TOLERANCE,
            },
            options: SequenceOptions::default(),
            border: geopyv_dev::image::DEFAULT_BORDER,
            save: None,
        };
        seq.solve(&cfg, Some(&())).unwrap();

        let path = std::env::temp_dir().join(format!("geopyv_field_outline_{}.pyv", std::process::id()));
        geopyv_dev::io::save(&path, &GeopyvObject::Sequence(seq.solution().unwrap().clone())).unwrap();
        let outline = MeshOutline::load(&path);
        std::fs::remove_file(&path).ok();

        assert!(outline.error.is_none(), "{:?}", outline.error);
        assert_eq!(outline.image.as_deref(), Some(images[0].as_path()));
        assert!(outline.boundary.len() >= 4);
        assert_eq!(outline.exclusions.len(), 1);
        assert!(outline.contains(egui::pos2(400.0, 400.0)));
        assert!(!outline.contains(egui::pos2(600.0, 600.0)), "inside the exclusion");
        assert!(!outline.contains(egui::pos2(200.0, 200.0)), "outside the boundary");
    }
}
