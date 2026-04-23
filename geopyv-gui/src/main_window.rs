use std::path::PathBuf;
use std::time::SystemTime;

use eframe::egui;

use crate::image_viewer::{HoverInfo, ImageViewer, TextureCache};
use crate::field_tab::{FieldSpawnParams, FieldTabState};
use crate::mesh_tab::{MeshSpawnParams, MeshTabState};
use crate::particle_tab::{ParticleSpawnParams, ParticleTabState};
use crate::project::Project;
use crate::sequence_tab::{SequenceSpawnParams, SequenceTabState};
use crate::subset_tab::{SubsetSpawnParams, SubsetTabState};
use crate::template::{shape_label, render_template_preview, TemplateConfig, TemplateShape};

// ---------------------------------------------------------------------------
// Tab
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Images,
    Templates,
    Subsets,
    Meshes,
    Sequences,
    Particles,
    Fields,
}

impl Tab {
    const ALL: &'static [Tab] = &[
        Tab::Images,
        Tab::Templates,
        Tab::Subsets,
        Tab::Meshes,
        Tab::Sequences,
        Tab::Particles,
        Tab::Fields,
    ];

    fn label(self) -> &'static str {
        match self {
            Tab::Images => "Images",
            Tab::Templates => "Templates",
            Tab::Subsets => "Subsets",
            Tab::Meshes => "Meshes",
            Tab::Sequences => "Sequences",
            Tab::Particles => "Particles",
            Tab::Fields => "Fields",
        }
    }
}

// ---------------------------------------------------------------------------
// Pane mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaneMode {
    View,
    New,
}

// ---------------------------------------------------------------------------
// File entry (left pane list item)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FileEntry {
    pub path: PathBuf,
    /// Short display name (stem for .pyv/.json, full filename for images).
    pub display: String,
    /// Compact relative age string.
    pub age: String,
    /// Whether this entry is a directory (Meshes sequence folders).
    pub is_dir: bool,
}

// ---------------------------------------------------------------------------
// Left pane state
// ---------------------------------------------------------------------------

pub struct LeftPaneState {
    pub entries: Vec<FileEntry>,
    pub selected: Option<usize>,
    last_tab: Option<Tab>,
}

impl LeftPaneState {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            selected: None,
            last_tab: None,
        }
    }

    pub fn refresh(&mut self, tab: Tab, project: &Project, force: bool) {
        if !force && self.last_tab == Some(tab) {
            return;
        }

        self.entries = match tab {
            Tab::Images => project
                .list_images()
                .into_iter()
                .map(|p| {
                    let display = p
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    FileEntry {
                        age: file_age(&p),
                        is_dir: false,
                        display,
                        path: p,
                    }
                })
                .collect(),

            Tab::Templates => project
                .list_templates()
                .into_iter()
                .map(file_entry_stem)
                .collect(),

            Tab::Subsets => project
                .list_subsets()
                .into_iter()
                .map(file_entry_stem)
                .collect(),

            Tab::Meshes => {
                let meshes = project.list_meshes();
                let standalone = meshes.standalone.into_iter().map(file_entry_stem);
                let folders = meshes.sequence_folders.into_iter().map(|p| {
                    let display = p
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    FileEntry {
                        age: String::new(),
                        is_dir: true,
                        display: format!("{display}/"),
                        path: p,
                    }
                });
                standalone.chain(folders).collect()
            }

            Tab::Sequences => project
                .list_sequences()
                .into_iter()
                .map(file_entry_stem)
                .collect(),

            Tab::Particles => project
                .list_particles()
                .into_iter()
                .map(file_entry_stem)
                .collect(),

            Tab::Fields => project
                .list_fields()
                .into_iter()
                .map(file_entry_stem)
                .collect(),
        };

        if self.last_tab != Some(tab) {
            self.selected = None;
        }
        self.last_tab = Some(tab);
    }
}

// ---------------------------------------------------------------------------
// Middle pane state
// ---------------------------------------------------------------------------

pub struct MiddlePaneState {
    pub image_viewer: ImageViewer,
}

impl MiddlePaneState {
    fn new() -> Self {
        Self {
            image_viewer: ImageViewer::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Right pane state
// ---------------------------------------------------------------------------

pub struct NewTemplateForm {
    pub name: String,
    pub shape: TemplateShape,
    pub size_text: String,
    pub size: u32,
    pub error: Option<String>,
}

impl Default for NewTemplateForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            shape: TemplateShape::Circle,
            size_text: "20".to_string(),
            size: 20,
            error: None,
        }
    }
}

pub struct TemplateTabState {
    /// Template currently shown in view mode, keyed by its path.
    pub loaded: Option<(PathBuf, TemplateConfig)>,
    pub new_form: NewTemplateForm,
}

impl TemplateTabState {
    fn new() -> Self {
        Self {
            loaded: None,
            new_form: NewTemplateForm::default(),
        }
    }
}

pub struct RightPaneState {
    pub template: TemplateTabState,
}

impl RightPaneState {
    fn new() -> Self {
        Self {
            template: TemplateTabState::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// PaneMode re-exported for subset_tab.rs
// ---------------------------------------------------------------------------
// (already defined above — just ensuring it is pub)


// ---------------------------------------------------------------------------
// Main window
// ---------------------------------------------------------------------------

pub struct MainWindow {
    pub active_tab: Tab,
    pub mode: PaneMode,
    pub left: LeftPaneState,
    pub middle: MiddlePaneState,
    pub right: RightPaneState,
    pub subset: SubsetTabState,
    pub mesh: MeshTabState,
    pub sequence: SequenceTabState,
    pub particle: ParticleTabState,
    pub field: FieldTabState,
    pub texture_cache: TextureCache,
    pending_refresh: bool,
    pub error_modal: Option<String>,
}

impl MainWindow {
    pub fn new() -> Self {
        Self {
            active_tab: Tab::Images,
            mode: PaneMode::View,
            left: LeftPaneState::new(),
            middle: MiddlePaneState::new(),
            right: RightPaneState::new(),
            subset: SubsetTabState::new(),
            mesh: MeshTabState::new(),
            sequence: SequenceTabState::new(),
            particle: ParticleTabState::new(),
            field: FieldTabState::new(),
            texture_cache: TextureCache::new(),
            pending_refresh: false,
            error_modal: None,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, project: &mut Project) {
        self.left.refresh(self.active_tab, project, false);

        // Request continuous repaint while a solve is running.
        if self.subset.is_solving()
            || self.mesh.is_solving()
            || self.sequence.is_solving()
            || self.particle.is_solving()
            || self.field.is_solving()
        {
            ctx.request_repaint();
        }

        // Poll subset solve completion.
        if let Some(outcome) = self.subset.check_solve_complete() {
            match outcome {
                Ok((solution, name)) => {
                    let save_path = project.subsets_dir().join(format!("{name}.pyv"));
                    if let Err(e) = geopyv_dev::io::save(
                        &save_path,
                        &geopyv_dev::io::GeopyvObject::Subset(solution.clone()),
                    ) {
                        self.error_modal = Some(format!("Save error: {e}"));
                    } else {
                        self.subset.view.solution = Some(solution);
                        self.subset.view.loaded_path = Some(save_path);
                        self.mode = PaneMode::View;
                        self.pending_refresh = true;
                    }
                }
                Err(e) => {
                    self.error_modal = Some(e);
                }
            }
        }

        // Poll mesh solve completion.
        if let Some(outcome) = self.mesh.check_solve_complete() {
            match outcome {
                Ok((solution, name, target_image)) => {
                    let save_path = project.meshes_dir().join(format!("{name}.pyv"));
                    if let Err(e) = geopyv_dev::io::save(
                        &save_path,
                        &geopyv_dev::io::GeopyvObject::Mesh(solution.clone()),
                    ) {
                        self.error_modal = Some(format!("Save error: {e}"));
                    } else {
                        self.mesh.view.solution = Some(solution);
                        self.mesh.view.loaded_path = Some(save_path);
                        self.mesh.view.target_image = target_image;
                        self.mesh.view.nodal_cache = None;
                        self.mode = PaneMode::View;
                        self.pending_refresh = true;
                    }
                }
                Err(e) => {
                    self.error_modal = Some(e);
                }
            }
        }

        // Poll sequence solve completion.
        if let Some(outcome) = self.sequence.check_solve_complete() {
            match outcome {
                Ok((solution, name, image_paths)) => {
                    self.sequence.view.solution = Some(solution);
                    self.sequence.view.loaded_path =
                        Some(project.sequences_dir().join(format!("{name}.pyv")));
                    self.sequence.view.image_paths = image_paths;
                    self.sequence.view.current_frame = 0;
                    self.sequence.view.nodal_cache = None;
                    self.sequence.view.animate = false;
                    self.mode = PaneMode::View;
                    self.pending_refresh = true;
                }
                Err(e) => {
                    self.error_modal = Some(e);
                }
            }
        }

        // Poll particle solve completion.
        if let Some(outcome) = self.particle.check_solve_complete() {
            match outcome {
                Ok((solution, name)) => {
                    let save_path = project.particles_dir().join(format!("{name}.pyv"));
                    if let Err(e) = geopyv_dev::io::save(
                        &save_path,
                        &geopyv_dev::io::GeopyvObject::Particle(solution.clone()),
                    ) {
                        self.error_modal = Some(format!("Save error: {e}"));
                    } else {
                        self.particle.view.solution = Some(solution);
                        self.particle.view.loaded_path = Some(save_path);
                        self.mode = PaneMode::View;
                        self.pending_refresh = true;
                    }
                }
                Err(e) => {
                    self.error_modal = Some(e);
                }
            }
        }

        // Poll field solve completion.
        if let Some(outcome) = self.field.check_solve_complete() {
            match outcome {
                Ok((solution, name)) => {
                    let save_path = project.fields_dir().join(format!("{name}.pyv"));
                    if let Err(e) = geopyv_dev::io::save(
                        &save_path,
                        &geopyv_dev::io::GeopyvObject::Field(solution.clone()),
                    ) {
                        self.error_modal = Some(format!("Save error: {e}"));
                    } else {
                        self.field.view.solution = Some(solution);
                        self.field.view.loaded_path = Some(save_path);
                        self.mode = PaneMode::View;
                        self.pending_refresh = true;
                    }
                }
                Err(e) => {
                    self.error_modal = Some(e);
                }
            }
        }

        // Snapshot cheap values before panel closures borrow self.
        let selected_path: Option<PathBuf> = self
            .left
            .selected
            .and_then(|i| self.left.entries.get(i))
            .map(|e| e.path.clone());
        let selected_index = self.left.selected;
        let templates_dir = project.templates_dir();
        let images = project.list_images();
        let templates = project.list_templates();
        let subsets_dir = project.subsets_dir();
        let meshes_dir = project.meshes_dir();
        let sequences_dir = project.sequences_dir();
        let sequences = project.list_sequences();
        let particles_dir = project.particles_dir();
        let fields_dir = project.fields_dir();
        let out_dir = project.out_dir();

        egui::TopBottomPanel::top("tab_bar")
            .exact_height(32.0)
            .show(ctx, |ui| {
                self.show_tab_bar(ui);
            });

        egui::SidePanel::left("left_pane")
            .resizable(true)
            .default_width(200.0)
            .width_range(140.0..=400.0)
            .show(ctx, |ui| {
                self.show_left_pane(ui, project);
            });

        let mut right_subset_spawn: Option<SubsetSpawnParams> = None;
        let mut right_mesh_spawn: Option<MeshSpawnParams> = None;
        let mut right_sequence_spawn: Option<SequenceSpawnParams> = None;
        let mut right_particle_spawn: Option<ParticleSpawnParams> = None;
        let mut right_field_spawn: Option<FieldSpawnParams> = None;
        egui::SidePanel::right("right_pane")
            .resizable(true)
            .default_width(260.0)
            .width_range(180.0..=500.0)
            .show(ctx, |ui| {
                (
                    right_subset_spawn,
                    right_mesh_spawn,
                    right_sequence_spawn,
                    right_particle_spawn,
                    right_field_spawn,
                ) = self.show_right_pane(
                    ui,
                    selected_path.as_deref(),
                    selected_index,
                    &templates_dir,
                    &images,
                    &templates,
                    &sequences,
                    &out_dir,
                );
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_central(ui, selected_path.as_deref(), &images);
        });

        // Handle Run from subset form.
        if let Some(params) = right_subset_spawn {
            let dest = subsets_dir.join(format!("{}.pyv", params.name));
            if dest.exists() {
                self.subset.new_form.form_error =
                    Some(format!("\"{}\" already exists — choose a different name", params.name));
            } else {
                self.subset.spawn_solve(params);
            }
        }

        // Handle Run from mesh form.
        if let Some(params) = right_mesh_spawn {
            let dest = project.meshes_dir().join(format!("{}.pyv", params.name));
            if dest.exists() {
                self.mesh.new_form.form_error =
                    Some(format!("\"{}\" already exists — choose a different name", params.name));
            } else {
                self.mesh.spawn_solve(params);
            }
        }

        // Handle Run from sequence form.
        if let Some(mut params) = right_sequence_spawn {
            let seq_dest = sequences_dir.join(format!("{}.pyv", params.name));
            if seq_dest.exists() {
                self.sequence.new_form.form_error =
                    Some(format!("\"{}\" already exists — choose a different name", params.name));
            } else {
                let mesh_subdir = meshes_dir.join(&params.name);
                if let Err(e) = std::fs::create_dir_all(&mesh_subdir) {
                    self.sequence.new_form.form_error =
                        Some(format!("Could not create Meshes subfolder: {e}"));
                } else {
                    params.mesh_subdir = mesh_subdir;
                    params.sequences_dir = sequences_dir.clone();
                    self.sequence.spawn_solve(params);
                }
            }
        }

        // Handle Run from particle form.
        if let Some(mut params) = right_particle_spawn {
            let dest = particles_dir.join(format!("{}.pyv", params.name));
            if dest.exists() {
                self.particle.new_form.form_error = Some(format!(
                    "\"{}\" already exists — choose a different name",
                    params.name
                ));
            } else {
                params.particles_dir = particles_dir.clone();
                self.particle.spawn_solve(params);
            }
        }

        // Handle Run from field form.
        if let Some(mut params) = right_field_spawn {
            let dest = fields_dir.join(format!("{}.pyv", params.name));
            if dest.exists() {
                self.field.new_form.form_error = Some(format!(
                    "\"{}\" already exists — choose a different name",
                    params.name
                ));
            } else {
                params.fields_dir = fields_dir.clone();
                self.field.spawn_solve(params);
            }
        }

        if self.pending_refresh {
            self.pending_refresh = false;
            self.left.refresh(self.active_tab, project, true);
        }

        // Error modal.
        self.show_error_modal(ctx);
    }

    fn show_error_modal(&mut self, ctx: &egui::Context) {
        let Some(msg) = self.error_modal.clone() else {
            return;
        };
        let mut open = true;
        egui::Window::new("Error")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(&msg);
                ui.add_space(8.0);
                if ui.button("OK").clicked() {
                    self.error_modal = None;
                }
            });
        if !open {
            self.error_modal = None;
        }
    }

    // -----------------------------------------------------------------------
    // Tab bar
    // -----------------------------------------------------------------------

    fn show_tab_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_centered(|ui| {
            ui.add_space(4.0);
            for &tab in Tab::ALL {
                let selected = self.active_tab == tab;
                let text = egui::RichText::new(tab.label()).size(13.0);
                let btn = egui::Button::new(text).selected(selected).corner_radius(4.0);
                if ui.add(btn).clicked() && !selected {
                    self.active_tab = tab;
                    self.mode = PaneMode::View;
                }
                ui.add_space(2.0);
            }
        });
    }

    // -----------------------------------------------------------------------
    // Left pane — file list + action buttons
    // -----------------------------------------------------------------------

    fn show_left_pane(&mut self, ui: &mut egui::Ui, project: &mut Project) {
        let tab = self.active_tab;

        egui::TopBottomPanel::bottom("left_buttons")
            .exact_height(36.0)
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(6, 4)))
            .show_inside(ui, |ui| {
                ui.horizontal_centered(|ui| {
                    let has_selection = self.left.selected.is_some();

                    if ui
                        .add_enabled(
                            has_selection,
                            egui::Button::new("Open").min_size(egui::vec2(54.0, 24.0)),
                        )
                        .clicked()
                    {
                        self.mode = PaneMode::View;
                    }

                    if tab == Tab::Images {
                        if ui.button("Import").clicked() {
                            self.action_import(project);
                        }
                    } else if ui.button("New").clicked() {
                        self.left.selected = None;
                        self.mode = PaneMode::New;
                    }
                });
            });

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());

                if self.left.entries.is_empty() {
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("(empty)")
                                .size(12.0)
                                .color(ui.visuals().weak_text_color()),
                        );
                    });
                    return;
                }

                for (i, entry) in self.left.entries.iter().enumerate() {
                    let selected = self.left.selected == Some(i);

                    let row_rect = egui::Rect::from_min_size(
                        ui.cursor().min,
                        egui::vec2(ui.available_width(), 22.0),
                    );

                    let response = ui.allocate_rect(row_rect, egui::Sense::click());

                    if response.clicked() {
                        self.left.selected = Some(i);
                    }
                    if response.double_clicked() {
                        self.left.selected = Some(i);
                        self.mode = PaneMode::View;
                    }

                    if selected {
                        ui.painter().rect_filled(
                            row_rect,
                            egui::CornerRadius::same(3),
                            ui.visuals().selection.bg_fill,
                        );
                    } else if response.hovered() {
                        ui.painter().rect_filled(
                            row_rect,
                            egui::CornerRadius::same(3),
                            ui.visuals().widgets.hovered.bg_fill,
                        );
                    }

                    let text_color = if selected {
                        ui.visuals().selection.stroke.color
                    } else if entry.is_dir {
                        ui.visuals().hyperlink_color
                    } else {
                        ui.visuals().text_color()
                    };

                    let galley_name = ui.painter().layout_no_wrap(
                        entry.display.clone(),
                        egui::FontId::new(13.0, egui::FontFamily::Proportional),
                        text_color,
                    );
                    let galley_age = ui.painter().layout_no_wrap(
                        entry.age.clone(),
                        egui::FontId::new(11.0, egui::FontFamily::Proportional),
                        ui.visuals().weak_text_color(),
                    );

                    let px = 6.0;
                    let name_pos =
                        row_rect.min + egui::vec2(px, (22.0 - galley_name.size().y) * 0.5);
                    let age_pos = egui::pos2(
                        row_rect.max.x - galley_age.size().x - px,
                        row_rect.min.y + (22.0 - galley_age.size().y) * 0.5,
                    );

                    ui.painter().galley(name_pos, galley_name, text_color);
                    ui.painter()
                        .galley(age_pos, galley_age, ui.visuals().weak_text_color());
                }
            });
    }

    // -----------------------------------------------------------------------
    // Central panel — image viewer (Images tab) + status bar
    // -----------------------------------------------------------------------

    fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        selected_path: Option<&std::path::Path>,
        images: &[PathBuf],
    ) {
        let tab = self.active_tab;
        let available = ui.available_rect_before_wrap();
        let status_h = 22.0;

        let viewer_rect = egui::Rect::from_min_max(
            available.min,
            egui::pos2(available.max.x, available.max.y - status_h),
        );
        let status_rect = egui::Rect::from_min_max(
            egui::pos2(available.min.x, available.max.y - status_h),
            available.max,
        );

        // Image viewer (Images tab) or stub for other tabs.
        let hover: Option<HoverInfo> = match (tab, selected_path) {
            (Tab::Images, Some(path)) => {
                let viewer = &mut self.middle.image_viewer;
                let cache = &mut self.texture_cache;
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    viewer.show(ui, path, cache, None)
                })
                .inner
            }
            (Tab::Images, None) => {
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    ui.painter().rect_filled(
                        viewer_rect,
                        0.0,
                        egui::Color32::from_rgb(15, 15, 15),
                    );
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new("Select an image from the list")
                                .size(14.0)
                                .color(ui.visuals().weak_text_color()),
                        );
                    });
                });
                None
            }
            (Tab::Templates, _) => {
                let preview: Option<(TemplateShape, u32)> = match self.mode {
                    PaneMode::New => {
                        let f = &self.right.template.new_form;
                        Some((f.shape.clone(), f.size))
                    }
                    PaneMode::View => self
                        .right
                        .template
                        .loaded
                        .as_ref()
                        .map(|(_, t)| (t.shape.clone(), t.size)),
                };
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    match preview {
                        Some((ref shape, size)) => {
                            render_template_preview(ui.painter(), viewer_rect, shape, size);
                        }
                        None => {
                            ui.painter()
                                .rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
                            ui.centered_and_justified(|ui| {
                                ui.label(
                                    egui::RichText::new("Select or create a template")
                                        .size(14.0)
                                        .color(ui.visuals().weak_text_color()),
                                );
                            });
                        }
                    }
                });
                None
            }

            (Tab::Subsets, _) => {
                let mode = self.mode;
                let cache = &mut self.texture_cache;
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    self.subset.show_central(
                        ui,
                        viewer_rect,
                        mode,
                        selected_path,
                        images,
                        cache,
                    )
                })
                .inner
            }

            (Tab::Meshes, _) => {
                let mode = self.mode;
                let cache = &mut self.texture_cache;
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    self.mesh.show_central(
                        ui,
                        viewer_rect,
                        mode,
                        selected_path,
                        images,
                        cache,
                    )
                })
                .inner
            }

            (Tab::Sequences, _) => {
                let mode = self.mode;
                let cache = &mut self.texture_cache;
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    self.sequence.show_central(
                        ui,
                        viewer_rect,
                        mode,
                        selected_path,
                        images,
                        cache,
                    )
                })
                .inner
            }

            (Tab::Particles, _) => {
                let mode = self.mode;
                let cache = &mut self.texture_cache;
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    self.particle.show_central(
                        ui,
                        viewer_rect,
                        mode,
                        selected_path,
                        images,
                        cache,
                    )
                })
                .inner
            }

            (Tab::Fields, _) => {
                let mode = self.mode;
                let cache = &mut self.texture_cache;
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| {
                    self.field.show_central(
                        ui,
                        viewer_rect,
                        mode,
                        selected_path,
                        images,
                        cache,
                    )
                })
                .inner
            }
        };

        // Status bar.
        ui.painter()
            .rect_filled(status_rect, 0.0, ui.visuals().faint_bg_color);

        let status_text = match &hover {
            Some(h) => format!(
                "x: {}   y: {}   intensity: {}",
                h.pixel_x, h.pixel_y, h.intensity
            ),
            None => "\u{2014}".to_string(),
        };

        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(status_rect), |ui| {
            ui.horizontal_centered(|ui| {
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(status_text)
                        .size(12.0)
                        .color(ui.visuals().weak_text_color()),
                );
            });
        });
    }

    // -----------------------------------------------------------------------
    // Right pane — Images metadata; stubs for other tabs
    // -----------------------------------------------------------------------

    fn show_right_pane(
        &mut self,
        ui: &mut egui::Ui,
        selected_path: Option<&std::path::Path>,
        selected_index: Option<usize>,
        templates_dir: &std::path::Path,
        images: &[PathBuf],
        templates: &[PathBuf],
        sequences: &[PathBuf],
        out_dir: &std::path::Path,
    ) -> (
        Option<SubsetSpawnParams>,
        Option<MeshSpawnParams>,
        Option<SequenceSpawnParams>,
        Option<ParticleSpawnParams>,
        Option<FieldSpawnParams>,
    ) {
        let tab = self.active_tab;
        let mode = self.mode;

        egui::TopBottomPanel::bottom("right_save_buttons")
            .exact_height(36.0)
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(6, 4)))
            .show_inside(ui, |ui| {
                ui.horizontal_centered(|ui| {
                    if ui.button("Save").clicked() {
                        if tab == Tab::Fields {
                            self.field.action_save(out_dir, selected_path);
                        }
                    }
                    if ui.button("Save As\u{2026}").clicked() {
                        if tab == Tab::Fields {
                            self.field.action_save_as(selected_path);
                        }
                    }
                });
            });

        let mut subset_spawn: Option<SubsetSpawnParams> = None;
        let mut mesh_spawn: Option<MeshSpawnParams> = None;
        let mut sequence_spawn: Option<SequenceSpawnParams> = None;
        let mut particle_spawn: Option<ParticleSpawnParams> = None;
        let mut field_spawn: Option<FieldSpawnParams> = None;

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(8.0);
                match (tab, mode, selected_path) {
                    (Tab::Images, _, Some(path)) => {
                        self.show_right_image_meta(ui, path, selected_index);
                    }
                    (Tab::Images, _, None) => {
                        ui.label(
                            egui::RichText::new("No image selected")
                                .size(13.0)
                                .color(ui.visuals().weak_text_color()),
                        );
                    }
                    (Tab::Templates, PaneMode::View, _) => {
                        self.show_right_template_view(ui, selected_path);
                    }
                    (Tab::Templates, PaneMode::New, _) => {
                        self.show_right_template_new(ui, templates_dir);
                    }
                    (Tab::Subsets, PaneMode::View, _) => {
                        self.subset.show_right_view(ui, selected_path);
                    }
                    (Tab::Subsets, PaneMode::New, _) => {
                        subset_spawn = self.subset.show_right_new(ui, images, templates);
                    }
                    (Tab::Meshes, PaneMode::View, _) => {
                        self.mesh.show_right_view(ui, selected_path);
                    }
                    (Tab::Meshes, PaneMode::New, _) => {
                        mesh_spawn = self.mesh.show_right_new(ui, images, templates);
                    }
                    (Tab::Sequences, PaneMode::View, _) => {
                        self.sequence.show_right_view(ui, selected_path);
                    }
                    (Tab::Sequences, PaneMode::New, _) => {
                        sequence_spawn = self.sequence.show_right_new(ui, images, templates);
                    }
                    (Tab::Particles, PaneMode::View, _) => {
                        self.particle.show_right_view(ui, selected_path);
                    }
                    (Tab::Particles, PaneMode::New, _) => {
                        particle_spawn = self.particle.show_right_new(ui, images, sequences);
                    }
                    (Tab::Fields, PaneMode::View, _) => {
                        self.field.show_right_view(ui, selected_path, out_dir);
                    }
                    (Tab::Fields, PaneMode::New, _) => {
                        field_spawn = self.field.show_right_new(ui, images, sequences);
                    }
                }
            });

        (subset_spawn, mesh_spawn, sequence_spawn, particle_spawn, field_spawn)
    }

    fn show_right_image_meta(
        &self,
        ui: &mut egui::Ui,
        path: &std::path::Path,
        index: Option<usize>,
    ) {
        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_uppercase())
            .unwrap_or_default();
        let file_size = std::fs::metadata(path)
            .map(|m| format_file_size(m.len()))
            .unwrap_or_else(|_| "—".to_string());
        let dims = self.middle.image_viewer.dimensions();

        meta_row(ui, "File", &filename);
        meta_row(
            ui,
            "Size",
            &dims
                .map(|(w, h)| format!("{w} \u{d7} {h} px"))
                .unwrap_or_else(|| "\u{2014}".to_string()),
        );
        meta_row(ui, "File size", &file_size);
        meta_row(
            ui,
            "Index",
            &index
                .map(|i| format!("{}", i + 1))
                .unwrap_or_else(|| "\u{2014}".to_string()),
        );
        meta_row(ui, "Format", &extension);
    }

    // -----------------------------------------------------------------------
    // Templates tab — view mode
    // -----------------------------------------------------------------------

    fn show_right_template_view(
        &mut self,
        ui: &mut egui::Ui,
        selected_path: Option<&std::path::Path>,
    ) {
        // Lazy-load (or clear) the template when the selected path changes.
        let cached_path = self.right.template.loaded.as_ref().map(|(p, _)| p.as_path());
        if cached_path != selected_path {
            self.right.template.loaded = selected_path.and_then(|path| {
                TemplateConfig::load(path).ok().map(|t| (path.to_path_buf(), t))
            });
        }

        if let Some((_, tmpl)) = &self.right.template.loaded {
            let n_px = tmpl.n_px();
            let shape_str = shape_label(&tmpl.shape).to_string();
            let size_str = format!("{} px", tmpl.size);
            let n_px_str = format!("{n_px}");
            let name = tmpl.name.clone();
            meta_row(ui, "Name", &name);
            meta_row(ui, "Shape", &shape_str);
            meta_row(ui, "Size", &size_str);
            meta_row(ui, "n_px", &n_px_str);
        } else {
            ui.label(
                egui::RichText::new("No template selected")
                    .size(13.0)
                    .color(ui.visuals().weak_text_color()),
            );
        }
    }

    // -----------------------------------------------------------------------
    // Templates tab — new mode
    // -----------------------------------------------------------------------

    fn show_right_template_new(
        &mut self,
        ui: &mut egui::Ui,
        templates_dir: &std::path::Path,
    ) {
        // Collect any save intent within a block to release the form borrow
        // before we mutate other self fields.
        let save_intent: Option<TemplateConfig> = {
            let form = &mut self.right.template.new_form;

            egui::Grid::new("template_new_grid")
                .num_columns(2)
                .spacing([8.0, 6.0])
                .min_col_width(60.0)
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("Name:")
                            .size(12.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut form.name)
                            .desired_width(150.0),
                    );
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Shape:")
                            .size(12.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut form.shape, TemplateShape::Circle, "Circle");
                        ui.radio_value(&mut form.shape, TemplateShape::Square, "Square");
                    });
                    ui.end_row();

                    ui.label(
                        egui::RichText::new("Size (px):")
                            .size(12.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut form.size_text)
                            .desired_width(80.0),
                    );
                    if resp.changed() {
                        let s = form.size_text.trim().to_string();
                        if s.is_empty() {
                            form.error = None;
                        } else {
                            match s.parse::<u32>() {
                                Ok(0) => form.error = Some("Size must be ≥ 1".to_string()),
                                Ok(v) => {
                                    form.size = v;
                                    form.error = None;
                                }
                                Err(_) => {
                                    form.error = Some("Enter a positive integer".to_string())
                                }
                            }
                        }
                    }
                    ui.end_row();
                });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);

            if let Some(err) = &form.error.clone() {
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(err)
                        .size(12.0)
                        .color(ui.visuals().error_fg_color),
                );
                ui.add_space(4.0);
            }

            let name = form.name.trim().to_string();
            let name_ok = !name.is_empty()
                && !name.contains('/')
                && !name.contains('\\')
                && !name.contains('\0');
            let size_ok = !form.size_text.trim().is_empty()
                && form.size > 0
                && form.error.is_none();
            let can_save = name_ok && size_ok;

            ui.horizontal(|ui| {
                ui.add_space(6.0);
                if ui
                    .add_enabled(can_save, egui::Button::new("Save"))
                    .clicked()
                {
                    Some(TemplateConfig {
                        name,
                        shape: form.shape.clone(),
                        size: form.size,
                    })
                } else {
                    None
                }
            })
            .inner
        }; // form borrow ends here

        if let Some(config) = save_intent {
            let dest_path = templates_dir.join(format!("{}.json", config.name));
            match config.save(templates_dir) {
                Ok(_) => {
                    self.right.template.loaded = Some((dest_path, config));
                    self.right.template.new_form = NewTemplateForm::default();
                    self.mode = PaneMode::View;
                    self.pending_refresh = true;
                }
                Err(e) => {
                    self.right.template.new_form.error = Some(e);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Import action (Images tab)
    // -----------------------------------------------------------------------

    fn action_import(&mut self, project: &mut Project) {
        let Some(paths) = rfd::FileDialog::new()
            .set_title("Import images")
            .add_filter("Images", &["jpg", "jpeg", "png", "tif", "tiff"])
            .pick_files()
        else {
            return;
        };

        let dest = project.images_data_dir();
        for src in &paths {
            let Some(fname) = src.file_name() else {
                continue;
            };
            let _ = std::fs::copy(src, dest.join(fname));
        }

        self.left.refresh(Tab::Images, project, true);
    }

    /// Force file-list refresh after an external save.
    pub fn refresh_file_list(&mut self, project: &Project) {
        self.left.refresh(self.active_tab, project, true);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn file_entry_stem(path: PathBuf) -> FileEntry {
    let display = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    FileEntry {
        age: file_age(&path),
        is_dir: false,
        display,
        path,
    }
}

fn file_age(path: &std::path::Path) -> String {
    let Ok(meta) = std::fs::metadata(path) else {
        return String::new();
    };
    let Ok(modified) = meta.modified() else {
        return String::new();
    };
    let Ok(elapsed) = SystemTime::now().duration_since(modified) else {
        return "now".to_string();
    };
    let secs = elapsed.as_secs();
    if secs < 60 {
        return "now".to_string();
    }
    if secs < 3600 {
        return format!("{}m", secs / 60);
    }
    if secs < 86400 {
        return format!("{}h", secs / 3600);
    }
    let days = secs / 86400;
    if days < 7 {
        return format!("{days}d");
    }
    if days < 30 {
        return format!("{}w", days / 7);
    }
    if days < 365 {
        return format!("{}mo", days / 30);
    }
    format!("{}y", days / 365)
}

fn format_file_size(bytes: u64) -> String {
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    if bytes < 1024 * 1024 {
        return format!("{:.1} KB", bytes as f64 / 1024.0);
    }
    if bytes < 1024 * 1024 * 1024 {
        return format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0));
    }
    format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
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

