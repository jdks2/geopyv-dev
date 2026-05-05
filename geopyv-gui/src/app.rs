use eframe::egui;

use crate::main_window::MainWindow;
use crate::project::Project;
use crate::recent::RecentProjects;

// ---------------------------------------------------------------------------
// Top-level application state
// ---------------------------------------------------------------------------

pub struct GeopyvApp {
    state: AppState,
    recent: RecentProjects,
    error: Option<String>,
}

enum AppState {
    Startup,
    ProjectOpen {
        project: Project,
        window: MainWindow,
    },
}

impl GeopyvApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_visuals(&cc.egui_ctx);
        Self {
            state: AppState::Startup,
            recent: RecentProjects::load(),
            error: None,
        }
    }

    // -----------------------------------------------------------------------
    // Startup window
    // -----------------------------------------------------------------------

    fn show_startup(&mut self, ctx: &egui::Context) {
        // Constrain the startup window to a compact size.
        ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(400.0, 260.0)));
        ctx.send_viewport_cmd(egui::ViewportCommand::MaxInnerSize(egui::vec2(560.0, 600.0)));

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);

                ui.heading(egui::RichText::new("geopyv").size(32.0).strong());
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("Digital Image Correlation")
                        .size(16.0)
                        .color(ui.visuals().weak_text_color()),
                );

                ui.add_space(32.0);

                let btn_size = egui::vec2(180.0, 36.0);

                if ui
                    .add_sized(btn_size, egui::Button::new("New Project"))
                    .clicked()
                {
                    self.action_new_project();
                }

                ui.add_space(8.0);

                if ui
                    .add_sized(btn_size, egui::Button::new("Open Project"))
                    .clicked()
                {
                    self.action_open_project();
                }

                self.show_recent_list(ui);
            });
        });
    }

    fn show_recent_list(&mut self, ui: &mut egui::Ui) {
        if self.recent.paths.is_empty() {
            return;
        }

        ui.add_space(28.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Recent")
                .size(14.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(4.0);

        let mut open_path: Option<std::path::PathBuf> = None;
        let mut remove_path: Option<std::path::PathBuf> = None;

        for path in &self.recent.paths {
            let label = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string_lossy().into_owned());

            let exists = path.is_dir() && path.join("project.json").exists();

            ui.horizontal(|ui| {
                let resp = ui.add(
                    egui::Button::new(
                        egui::RichText::new(&label).size(15.0).color(if exists {
                            ui.visuals().text_color()
                        } else {
                            ui.visuals().weak_text_color()
                        }),
                    )
                    .frame(false),
                );

                if resp.clicked() {
                    if exists {
                        open_path = Some(path.clone());
                    } else {
                        remove_path = Some(path.clone());
                    }
                }

                resp.on_hover_text(path.display().to_string());

                if !exists {
                    ui.label(
                        egui::RichText::new("(missing)")
                            .size(13.0)
                            .color(ui.visuals().error_fg_color),
                    );
                }
            });
        }

        if let Some(p) = open_path {
            self.open_project_at(p);
        }
        if let Some(p) = remove_path {
            self.recent.remove(&p);
        }
    }

    // -----------------------------------------------------------------------
    // Project open / create actions
    // -----------------------------------------------------------------------

    fn action_new_project(&mut self) {
        let Some(folder) = rfd::FileDialog::new()
            .set_title("Choose location for new project")
            .pick_folder()
        else {
            return;
        };

        match Project::create(&folder) {
            Ok(project) => self.load_project(project),
            Err(e) => self.error = Some(format!("Could not create project: {e}")),
        }
    }

    fn action_open_project(&mut self) {
        let Some(folder) = rfd::FileDialog::new()
            .set_title("Open project folder")
            .pick_folder()
        else {
            return;
        };

        self.open_project_at(folder);
    }

    fn open_project_at(&mut self, path: std::path::PathBuf) {
        match Project::open(&path) {
            Ok(project) => self.load_project(project),
            Err(e) => self.error = Some(format!("Could not open project: {e}")),
        }
    }

    fn load_project(&mut self, project: Project) {
        self.recent.push(&project.root);
        // Restore the main window to full size.
        // (MaxInnerSize from startup mode must be lifted.)
        self.state = AppState::ProjectOpen {
            project,
            window: MainWindow::new(),
        };
    }

    // -----------------------------------------------------------------------
    // Main window
    // -----------------------------------------------------------------------

    fn show_main(&mut self, ctx: &egui::Context) {
        // Lift the startup size constraints.
        ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(800.0, 500.0)));
        ctx.send_viewport_cmd(egui::ViewportCommand::MaxInnerSize(egui::vec2(f32::INFINITY, f32::INFINITY)));

        let AppState::ProjectOpen { project, window } = &mut self.state else {
            return;
        };

        ctx.send_viewport_cmd(egui::ViewportCommand::Title(
            format!("geopyv \u{2014} {}", project.meta.name),
        ));

        window.show(ctx, project);
    }

    // -----------------------------------------------------------------------
    // Error modal
    // -----------------------------------------------------------------------

    fn show_error_modal(&mut self, ctx: &egui::Context) {
        let Some(msg) = self.error.clone() else {
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
                    self.error = None;
                }
            });

        if !open {
            self.error = None;
        }
    }
}

impl eframe::App for GeopyvApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        match &self.state {
            AppState::Startup => self.show_startup(ctx),
            AppState::ProjectOpen { .. } => self.show_main(ctx),
        }

        self.show_error_modal(ctx);
    }
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

fn configure_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();

    visuals.window_fill = egui::Color32::from_rgb(22, 22, 24);
    visuals.panel_fill = egui::Color32::from_rgb(22, 22, 24);
    visuals.faint_bg_color = egui::Color32::from_rgb(30, 30, 32);

    visuals.window_corner_radius = egui::CornerRadius::same(6);
    visuals.menu_corner_radius = egui::CornerRadius::same(4);

    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(42, 42, 46);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(58, 58, 64);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(70, 70, 80);

    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(16.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(16.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(22.0, egui::FontFamily::Proportional),
    );
    ctx.set_style(style);
}
