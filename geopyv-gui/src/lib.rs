//! geopyv desktop GUI. A thin egui front-end over the `geopyv-dev` core:
//! all solving, defaults and derived quantities come from the core; this
//! crate only gathers inputs, runs core solves on worker threads and draws
//! results.
//!
//! [`run`] is the entry point of the `geopyv-gui` binary; the Python
//! package's `geopyv-gui` console script uses [`run_with`] to also hand over
//! the calibration solve (ChArUco detection and the camera fit run in OpenCV,
//! through Python).

mod app;
mod calibration_tab;
mod colormap;
mod draw;
mod field_tab;
mod field_view;
mod image_viewer;
mod main_window;
mod mesh_tab;
mod overlay;
mod particle_tab;
mod progress;
mod project;
mod recent;
mod sequence_tab;
mod subset_tab;
mod time_input;

pub use calibration_tab::{CalibrationBackend, CalibrationRequest};

/// Open the geopyv window and block until it is closed. Calibrations can be
/// loaded and used but not solved (see [`run_with`]).
///
/// Must be called from the process's main thread, and at most once per
/// process (the windowing event loop cannot be re-created).
pub fn run() -> eframe::Result {
    run_with(None)
}

/// [`run`], with `calibration` solving calibrations from the project's
/// calibration images.
pub fn run_with(calibration: Option<CalibrationBackend>) -> eframe::Result {
    let calibration = calibration.map(std::sync::Arc::new);
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("geopyv")
            .with_inner_size([500.0, 320.0])
            .with_min_inner_size([400.0, 260.0])
            .with_app_id("geopyv-gui"),
        ..Default::default()
    };

    eframe::run_native(
        "geopyv",
        native_options,
        Box::new(move |cc| Ok(Box::new(app::GeopyvApp::new(cc, calibration)))),
    )
}
