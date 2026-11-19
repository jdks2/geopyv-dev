//! Calibration tab: ChArUco camera calibrations — solved from the project's
//! board images (ChArUco detection and the camera fit run in OpenCV, through
//! the Python package; see [`CalibrationBackend`]) or imported as `.pyv` — and
//! their diagnostic plots. The Mesh, Sequence and Field tabs use a saved
//! calibration for displacements in board units.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use eframe::egui;
use geopyv_dev::calibration::{CalibrationParams, CalibrationSolution};
use geopyv_dev::io::GeopyvObject;

use crate::colormap::{self, ColormapType};
use crate::draw::ImageCoord;
use crate::image_viewer::{HoverInfo, ImageViewer, TextureCache};
use crate::main_window::PaneMode;
use crate::overlay::{self, format_sci, nice_range};

// ---------------------------------------------------------------------------
// Solve hook
// ---------------------------------------------------------------------------

/// Everything a calibration solve needs: the board images and the printed
/// board (`gp.Calibration`'s arguments).
#[derive(Debug, Clone)]
pub struct CalibrationRequest {
    /// Folder holding the board images.
    pub dir: PathBuf,
    /// Shared file-name prefix of the images to use ("" = all).
    pub common_name: String,
    /// Image extension, with the dot (e.g. ".jpg").
    pub file_format: String,
    /// Chessboard columns and rows, square and marker side lengths (board units).
    pub columns: usize,
    pub rows: usize,
    pub square_length: f64,
    pub marker_length: f64,
    /// Numeric file-name suffix of the image that sets the object frame
    /// (`None`: the last accepted image).
    pub ext_id: Option<i64>,
    /// Binarisation threshold before detection (`None`: none).
    pub binary_threshold: Option<f64>,
    /// Minimum detected corners for an image to be used.
    pub acceptance_threshold: usize,
}

/// How the GUI solves calibrations. The Python package supplies one when it
/// launches the GUI; without it calibrations can only be imported.
pub struct CalibrationBackend {
    /// `Err(reason)` when calibrations cannot be solved (e.g. OpenCV missing).
    pub available: Box<dyn Fn() -> Result<(), String> + Send + Sync>,
    pub solve: Box<dyn Fn(&CalibrationRequest) -> Result<CalibrationSolution, String> + Send + Sync>,
}

// ---------------------------------------------------------------------------
// View state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CalibrationPane {
    /// One accepted image and its detected corners.
    #[default]
    Inspect,
    /// Detected corners of every accepted image.
    Coverage,
    /// How far the lens model moves each corner.
    Undistortion,
    /// Reprojection error of each corner.
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CalibrationQuantity {
    #[default]
    R,
    U,
    V,
}

impl CalibrationQuantity {
    fn of(self, d: [f64; 2]) -> f64 {
        match self {
            Self::R => d[0].hypot(d[1]),
            Self::U => d[0],
            Self::V => d[1],
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::R => "R (px)",
            Self::U => "u (px)",
            Self::V => "v (px)",
        }
    }
}

pub struct CalibrationViewState {
    pub loaded_path: Option<PathBuf>,
    pub solution: Option<CalibrationSolution>,
    pub load_error: Option<String>,
    pub pane: CalibrationPane,
    pub image_index: usize,
    pub quantity: CalibrationQuantity,
    pub colormap: ColormapType,
}

impl CalibrationViewState {
    fn new() -> Self {
        Self {
            loaded_path: None,
            solution: None,
            load_error: None,
            pane: CalibrationPane::default(),
            image_index: 0,
            quantity: CalibrationQuantity::default(),
            colormap: ColormapType::default(),
        }
    }

    pub fn set_solution(&mut self, path: Option<PathBuf>, solution: Option<CalibrationSolution>) {
        self.loaded_path = path;
        self.solution = solution;
        self.load_error = None;
        self.image_index = 0;
    }

    fn load(&mut self, selected: Option<&Path>) {
        if self.loaded_path.as_deref() == selected {
            return;
        }
        self.set_solution(selected.map(Path::to_path_buf), None);
        if let Some(p) = selected {
            match load_calibration(p) {
                Ok(c) => self.solution = Some(c),
                Err(e) => self.load_error = Some(e),
            }
        }
    }
}

/// A saved calibration, or why it could not be read.
pub fn load_calibration(path: &Path) -> Result<CalibrationSolution, String> {
    match geopyv_dev::io::load(path) {
        Ok(GeopyvObject::Calibration(c)) => Ok(c),
        Ok(_) => Err(format!("{} is not a calibration", path.display())),
        Err(e) => Err(e.to_string()),
    }
}

/// "Units:" choice shared by the Mesh and Sequence views and the Field form:
/// pixels, or one of the project's calibrations (by file stem).
pub fn calibration_combo(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    none_label: &str,
    calibrations: &[PathBuf],
    chosen: &mut Option<(PathBuf, CalibrationParams)>,
) -> bool {
    let stem = |p: &Path| p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let current = chosen.as_ref().map_or(none_label.to_string(), |(p, _)| stem(p));
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(16.0).color(ui.visuals().weak_text_color()));
        egui::ComboBox::from_id_salt(id).selected_text(current).width(130.0).show_ui(ui, |ui| {
            if ui.selectable_label(chosen.is_none(), none_label).clicked() && chosen.is_some() {
                *chosen = None;
                changed = true;
            }
            for p in calibrations {
                let on = chosen.as_ref().is_some_and(|(c, _)| c == p);
                if ui.selectable_label(on, stem(p)).clicked() && !on {
                    match load_calibration(p).and_then(|c| c.params().map_err(|e| e.to_string())) {
                        Ok(params) => {
                            *chosen = Some((p.clone(), params));
                            changed = true;
                        }
                        Err(e) => eprintln!("geopyv-gui: {e}"),
                    }
                }
            }
        });
    });
    changed
}

// ---------------------------------------------------------------------------
// New-calibration form
// ---------------------------------------------------------------------------

pub struct NewCalibrationForm {
    pub name: String,
    pub columns: String,
    pub rows: String,
    pub square_length: String,
    pub marker_length: String,
    pub common_name: String,
    pub ext_id: String,
    pub binary_threshold: String,
    pub acceptance_threshold: String,
    /// Which board image the central pane previews.
    pub preview: usize,
    pub error: Option<String>,
}

impl Default for NewCalibrationForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            columns: String::new(),
            rows: String::new(),
            square_length: String::new(),
            marker_length: String::new(),
            common_name: String::new(),
            ext_id: String::new(),
            binary_threshold: String::new(),
            acceptance_threshold: "50".to_string(),
            preview: 0,
            error: None,
        }
    }
}

impl NewCalibrationForm {
    /// The request, or what is wrong with the form.
    fn request(&self, dir: &Path, images: &[PathBuf]) -> Result<CalibrationRequest, String> {
        let name = self.name.trim();
        if name.is_empty() || name.contains(['/', '\\']) {
            return Err("Enter a name (no slashes)".into());
        }
        let count = |s: &str, what: &str| s.trim().parse::<usize>().ok().filter(|&v| v >= 2).ok_or(format!("{what}: a whole number >= 2"));
        let length = |s: &str, what: &str| s.trim().parse::<f64>().ok().filter(|v| *v > 0.0).ok_or(format!("{what}: a number > 0"));
        let (columns, rows) = (count(&self.columns, "Columns")?, count(&self.rows, "Rows")?);
        let (square_length, marker_length) = (length(&self.square_length, "Square length")?, length(&self.marker_length, "Marker length")?);
        if marker_length >= square_length {
            return Err("The marker must be smaller than the square".into());
        }
        let blank_or = |s: &str| (!s.trim().is_empty()).then(|| s.trim().to_string());
        let ext_id = blank_or(&self.ext_id)
            .map(|v| v.parse::<i64>())
            .transpose()
            .map_err(|_| "Object-frame image: its number, or blank".to_string())?;
        let binary_threshold = blank_or(&self.binary_threshold)
            .map(|v| v.parse::<f64>())
            .transpose()
            .map_err(|_| "Threshold: a grey level, or blank".to_string())?;
        let acceptance_threshold = self.acceptance_threshold.trim().parse::<usize>().map_err(|_| "Minimum corners: a whole number".to_string())?;
        let prefix = self.common_name.trim();
        let chosen: Vec<&PathBuf> = images
            .iter()
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(prefix)))
            .collect();
        if chosen.is_empty() {
            return Err("No board images (with that prefix) in Images/Calibration".into());
        }
        // the most common extension among them, as gp.Calibration takes one
        let mut exts: Vec<String> = chosen.iter().filter_map(|p| p.extension()).map(|e| e.to_string_lossy().to_string()).collect();
        exts.sort();
        let file_format = exts
            .chunk_by(|a, b| a == b)
            .max_by_key(|g| g.len())
            .map(|g| format!(".{}", g[0]))
            .unwrap_or_else(|| ".jpg".into());
        Ok(CalibrationRequest {
            dir: dir.to_path_buf(),
            common_name: prefix.to_string(),
            file_format,
            columns,
            rows,
            square_length,
            marker_length,
            ext_id,
            binary_threshold,
            acceptance_threshold,
        })
    }
}

// ---------------------------------------------------------------------------
// Tab state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SolveState {
    running: bool,
    result: Option<Result<CalibrationSolution, String>>,
}

pub struct CalibrationTabState {
    backend: Option<Arc<CalibrationBackend>>,
    /// `backend.available()`, asked once.
    availability: Option<Result<(), String>>,
    pub view: CalibrationViewState,
    pub new_form: NewCalibrationForm,
    viewer: ImageViewer,
    solve_state: Arc<Mutex<SolveState>>,
    pending_name: Option<String>,
}

impl CalibrationTabState {
    pub fn new(backend: Option<Arc<CalibrationBackend>>) -> Self {
        Self {
            backend,
            availability: None,
            view: CalibrationViewState::new(),
            new_form: NewCalibrationForm::default(),
            viewer: ImageViewer::new(),
            solve_state: Arc::new(Mutex::new(SolveState::default())),
            pending_name: None,
        }
    }

    pub fn is_solving(&self) -> bool {
        self.solve_state.lock().map(|s| s.running).unwrap_or(false)
    }

    /// The finished solve and its name, once.
    pub fn check_solve_complete(&mut self) -> Option<Result<(CalibrationSolution, String), String>> {
        let mut s = self.solve_state.lock().unwrap();
        if s.running {
            return None;
        }
        let result = s.result.take()?;
        let name = self.pending_name.take().unwrap_or_default();
        Some(result.map(|c| (c, name)))
    }

    fn availability(&mut self) -> Result<(), String> {
        let backend = self.backend.clone();
        self.availability
            .get_or_insert_with(|| match &backend {
                Some(b) => (b.available)(),
                None => Err("Solving calibrations needs the GUI started from Python (the geopyv-gui \
                             command); import a calibration saved from a script instead."
                    .into()),
            })
            .clone()
    }

    fn spawn_solve(&mut self, name: String, request: CalibrationRequest) {
        let Some(backend) = self.backend.clone() else { return };
        self.pending_name = Some(name);
        let state = Arc::clone(&self.solve_state);
        {
            let mut s = state.lock().unwrap();
            s.running = true;
            s.result = None;
        }
        std::thread::spawn(move || {
            let result = (backend.solve)(&request);
            if let Ok(mut s) = state.lock() {
                s.result = Some(result);
                s.running = false;
            }
        });
    }

    // -----------------------------------------------------------------------
    // Central pane
    // -----------------------------------------------------------------------

    pub fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        viewer_rect: egui::Rect,
        mode: PaneMode,
        selected_path: Option<&Path>,
        board_images: &[PathBuf],
        cache: &mut TextureCache,
    ) -> Option<HoverInfo> {
        if mode == PaneMode::New {
            let path = board_images.get(self.new_form.preview.min(board_images.len().saturating_sub(1))).cloned();
            let hover = match path {
                Some(p) => ui
                    .allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| self.viewer.show(ui, &p, cache, None))
                    .inner,
                None => {
                    message(ui, viewer_rect, "Import photographs of the ChArUco board (right) to begin");
                    None
                }
            };
            if self.is_solving() {
                let painter = ui.painter_at(viewer_rect);
                painter.rect_filled(viewer_rect, 0.0, egui::Color32::from_black_alpha(140));
                painter.text(
                    viewer_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Solving the calibration\u{2026}",
                    egui::FontId::proportional(18.0),
                    egui::Color32::WHITE,
                );
                ui.ctx().request_repaint();
            }
            return hover;
        }

        self.view.load(selected_path);
        let Some(sol) = self.view.solution.as_ref() else {
            let text = self.view.load_error.clone().unwrap_or_else(|| "Select a calibration from the list".into());
            message(ui, viewer_rect, &text);
            return None;
        };
        let n_images = sol.accepted_images.len();
        if n_images == 0 {
            message(ui, viewer_rect, "This calibration has no accepted images");
            return None;
        }
        let index = if self.view.pane == CalibrationPane::Inspect { self.view.image_index.min(n_images - 1) } else { 0 };
        let image = sol.accepted_images[index].clone();
        let mut hover = None;
        let mut coord = None;
        if image.exists() {
            hover = ui
                .allocate_new_ui(egui::UiBuilder::new().max_rect(viewer_rect), |ui| self.viewer.show(ui, &image, cache, None))
                .inner;
            coord = self.viewer.last_coord();
        }
        let sol = self.view.solution.as_ref().expect("checked above");
        let coord = coord.unwrap_or_else(|| {
            ui.painter().rect_filled(viewer_rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
            let (h, w) = sol.image_size;
            overlay::coord_from_points([[0.0, 0.0], [w as f64, h as f64]].into_iter(), viewer_rect)
        });
        let painter = ui.painter_at(viewer_rect);
        paint_pane(&painter, viewer_rect, &coord, sol, self.view.pane, index, self.view.quantity, self.view.colormap);
        hover
    }

    // -----------------------------------------------------------------------
    // Right pane — view
    // -----------------------------------------------------------------------

    pub fn show_right_view(&mut self, ui: &mut egui::Ui, selected_path: Option<&Path>) {
        self.view.load(selected_path);
        let Some(sol) = self.view.solution.as_ref() else {
            let text = self.view.load_error.clone().unwrap_or_else(|| "No calibration selected".into());
            ui.label(egui::RichText::new(text).size(15.0).color(ui.visuals().weak_text_color()));
            return;
        };
        let k = &sol.intmat;
        let err = sol.reprojection_error();
        let r: Vec<f64> = err.rows().into_iter().map(|e| e[0].hypot(e[1])).collect();
        let mean = r.iter().sum::<f64>() / r.len().max(1) as f64;
        let max = r.iter().cloned().fold(0.0, f64::max);
        let n_images = sol.accepted_images.len();

        header(ui, "Camera");
        egui::Grid::new("cal_params").num_columns(2).spacing([10.0, 3.0]).show(ui, |ui| {
            let mut row = |name: &str, v: String| {
                ui.label(egui::RichText::new(name).color(ui.visuals().weak_text_color()));
                ui.label(v);
                ui.end_row();
            };
            row("Focal length (px)", format!("{:.1}, {:.1}", k[[0, 0]], k[[1, 1]]));
            row("Principal point (px)", format!("{:.1}, {:.1}", k[[0, 2]], k[[1, 2]]));
            let d = sol.dist;
            row("Radial k1, k2, k3", format!("{}, {}, {}", format_sci(d[0]), format_sci(d[1]), format_sci(d[4])));
            row("Tangential p1, p2", format!("{}, {}", format_sci(d[2]), format_sci(d[3])));
            row("Images used", n_images.to_string());
            row("Image size (px)", format!("{} \u{d7} {}", sol.image_size.1, sol.image_size.0));
            row("Corners", r.len().to_string());
            row("Reprojection error (px)", format!("mean {mean:.3}, max {max:.3}"));
        });

        ui.add_space(6.0);
        header(ui, "Plot");
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.view.pane, CalibrationPane::Inspect, "Inspect")
                .on_hover_text("One image and the board corners found in it");
            ui.radio_value(&mut self.view.pane, CalibrationPane::Coverage, "Coverage")
                .on_hover_text("Corners found in every image: the board should cover the whole frame");
            ui.radio_value(&mut self.view.pane, CalibrationPane::Undistortion, "Lens distortion")
                .on_hover_text("How far the lens model moves each corner");
            ui.radio_value(&mut self.view.pane, CalibrationPane::Error, "Reprojection error")
                .on_hover_text("Found corner vs. where the solved camera puts it: small and patternless is good");
        });
        match self.view.pane {
            CalibrationPane::Inspect if n_images > 1 => {
                ui.add(egui::Slider::new(&mut self.view.image_index, 0..=n_images - 1).text("image"));
                if let Some(p) = sol.accepted_images.get(self.view.image_index) {
                    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    ui.label(egui::RichText::new(name).size(13.0).color(ui.visuals().weak_text_color()));
                }
            }
            CalibrationPane::Undistortion | CalibrationPane::Error => {
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.view.quantity, CalibrationQuantity::R, "R");
                    ui.radio_value(&mut self.view.quantity, CalibrationQuantity::U, "u");
                    ui.radio_value(&mut self.view.quantity, CalibrationQuantity::V, "v");
                });
                egui::ComboBox::from_id_salt("cal_colormap")
                    .selected_text(self.view.colormap.label())
                    .width(120.0)
                    .show_ui(ui, |ui| {
                        for &c in ColormapType::ALL {
                            ui.selectable_value(&mut self.view.colormap, c, c.label());
                        }
                    });
            }
            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // Right pane — new
    // -----------------------------------------------------------------------

    pub fn show_right_new(&mut self, ui: &mut egui::Ui, board_images: &[PathBuf], board_dir: &Path, save_dir: &Path) {
        header(ui, "Board images");
        ui.label(format!("{} in Images/Calibration", board_images.len()));
        if ui.button("Import board images\u{2026}").clicked() {
            if let Some(paths) = rfd::FileDialog::new()
                .set_title("Import ChArUco board images")
                .add_filter("Images", &["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
                .pick_files()
            {
                for src in &paths {
                    if let Some(name) = src.file_name() {
                        let _ = std::fs::copy(src, board_dir.join(name));
                    }
                }
            }
        }
        if board_images.len() > 1 {
            ui.add(egui::Slider::new(&mut self.new_form.preview, 0..=board_images.len() - 1).text("preview"));
        }
        ui.label(
            egui::RichText::new(
                "Photograph the board at least 10 times, tilted and moved about, through the viewing \
                 window at the test's focus, aperture and zoom.",
            )
            .size(13.0)
            .color(ui.visuals().weak_text_color()),
        );

        ui.add_space(6.0);
        header(ui, "Board");
        let f = &mut self.new_form;
        egui::Grid::new("cal_board").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            let mut field = |label: &str, text: &mut String, hint: &str| {
                ui.label(egui::RichText::new(label).color(ui.visuals().weak_text_color())).on_hover_text(hint);
                ui.add(egui::TextEdit::singleline(text).desired_width(90.0));
                ui.end_row();
            };
            field("Name:", &mut f.name, "Saved as Calibrations/<name>.pyv");
            field("Columns:", &mut f.columns, "Chessboard squares across");
            field("Rows:", &mut f.rows, "Chessboard squares down");
            field("Square length:", &mut f.square_length, "Side of one square, in the units results should use (e.g. mm)");
            field("Marker length:", &mut f.marker_length, "Side of one ArUco marker, same units");
        });

        ui.add_space(6.0);
        header(ui, "Options");
        egui::Grid::new("cal_options").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            let mut field = |label: &str, text: &mut String, hint: &str| {
                ui.label(egui::RichText::new(label).color(ui.visuals().weak_text_color())).on_hover_text(hint);
                ui.add(egui::TextEdit::singleline(text).desired_width(90.0));
                ui.end_row();
            };
            field("File prefix:", &mut f.common_name, "Use only images whose names start with this (blank: all)");
            field("Object-frame image:", &mut f.ext_id,
                  "Number in the file name of the image taken with the board on the specimen face; \
                   it sets the object coordinates (blank: the last image used)");
            field("Threshold:", &mut f.binary_threshold, "Grey level to binarise at before detection (blank: none)");
            field("Minimum corners:", &mut f.acceptance_threshold, "Images with fewer corners found are not used");
        });

        ui.add_space(8.0);
        let availability = self.availability();
        let request = self.new_form.request(board_dir, board_images).and_then(|r| {
            let name = self.new_form.name.trim();
            if save_dir.join(format!("{name}.pyv")).exists() {
                Err(format!("\"{name}\" already exists \u{2014} choose a different name"))
            } else {
                Ok(r)
            }
        });
        let running = self.is_solving();
        let can_solve = availability.is_ok() && request.is_ok() && !running;
        let response = ui.add_enabled(can_solve, egui::Button::new("Solve").min_size(egui::vec2(80.0, 28.0)));
        match (&availability, &request) {
            (Err(why), _) => warn(ui, why),
            (_, Err(why)) => warn(ui, why),
            _ => {}
        }
        if let Some(e) = &self.new_form.error {
            warn(ui, e);
        }
        if response.clicked() {
            if let Ok(req) = request {
                self.new_form.error = None;
                let name = self.new_form.name.trim().to_string();
                self.spawn_solve(name, req);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn paint_pane(
    painter: &egui::Painter,
    viewer_rect: egui::Rect,
    coord: &ImageCoord,
    sol: &CalibrationSolution,
    pane: CalibrationPane,
    index: usize,
    quantity: CalibrationQuantity,
    cmap: ColormapType,
) {
    let dot = |p: [f64; 2], r: f32, c: egui::Color32| {
        painter.circle_filled(coord.to_screen(egui::pos2(p[0] as f32, p[1] as f32)), r, c);
    };
    let red = egui::Color32::from_rgb(230, 40, 40);
    match pane {
        CalibrationPane::Inspect => {
            for c in sol.corners[index].rows() {
                dot([c[0], c[1]], 3.0, red);
            }
        }
        CalibrationPane::Coverage => {
            for c in sol.all_corners().rows() {
                dot([c[0], c[1]], 2.0, egui::Color32::from_rgba_unmultiplied(230, 40, 40, 170));
            }
        }
        CalibrationPane::Undistortion | CalibrationPane::Error => {
            let pts = sol.all_corners();
            let d = if pane == CalibrationPane::Undistortion { sol.undistortion() } else { sol.reprojection_error() };
            let vals: Vec<f64> = d.rows().into_iter().map(|r| quantity.of([r[0], r[1]])).collect();
            let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let (lo, hi) = if quantity == CalibrationQuantity::R { nice_range(0.0, hi) } else {
                let m = lo.abs().max(hi.abs());
                nice_range(-m, m)
            };
            for (p, &v) in pts.rows().into_iter().zip(&vals) {
                dot([p[0], p[1]], 3.5, colormap::map_value(v, lo, hi, cmap));
            }
            let title = if pane == CalibrationPane::Undistortion { "Distortion" } else { "Error" };
            overlay::render_colorbar(painter, viewer_rect, lo, hi, cmap, &format!("{title} {}", quantity.label()));
        }
    }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

fn message(ui: &mut egui::Ui, rect: egui::Rect, text: &str) {
    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(rect), |ui| {
        ui.painter().rect_filled(rect, 0.0, egui::Color32::from_rgb(15, 15, 15));
        ui.centered_and_justified(|ui| {
            ui.label(egui::RichText::new(text).size(16.0).color(ui.visuals().weak_text_color()));
        });
    });
}

fn header(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(format!("\u{2500}\u{2500} {text} \u{2500}\u{2500}")).size(15.0).color(ui.visuals().weak_text_color()));
}

fn warn(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).size(13.0).color(egui::Color32::from_rgb(230, 150, 60)));
}
