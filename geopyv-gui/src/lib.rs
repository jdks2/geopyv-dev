//! geopyv desktop GUI. A thin egui front-end over the `geopyv-dev` core:
//! all solving, defaults and derived quantities come from the core; this
//! crate only gathers inputs, runs core solves on worker threads and draws
//! results.
//!
//! [`run`] is the single entry point, shared by the `geopyv-gui` binary and
//! the Python package's `geopyv-gui` console script.

mod app;
mod colormap;
mod draw;
mod field_tab;
mod image_viewer;
mod main_window;
mod mesh_tab;
mod particle_tab;
mod progress;
mod project;
mod recent;
mod sequence_tab;
mod subset_tab;

/// Open the geopyv window and block until it is closed.
///
/// Must be called from the process's main thread, and at most once per
/// process (the windowing event loop cannot be re-created).
pub fn run() -> eframe::Result {
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
        Box::new(|cc| Ok(Box::new(app::GeopyvApp::new(cc)))),
    )
}
