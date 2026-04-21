mod app;
mod draw;
mod image_viewer;
mod main_window;
mod project;
mod recent;
mod template;

fn main() -> eframe::Result {
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
