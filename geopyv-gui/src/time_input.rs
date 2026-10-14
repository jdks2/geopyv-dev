//! The `time=` input of the Mesh/Sequence forms: a time increment entered
//! as either `Δt (s)` or `Rate (Hz)` — editing one updates the other — or
//! both blank to detect timestamps from the image file names
//! (`geopyv_dev::timing`).

use std::path::{Path, PathBuf};

use eframe::egui;
use geopyv_dev::timing::{parse_filename_times, TimeInput};

#[derive(Default)]
pub struct TimeInputForm {
    pub dt_text: String,
    pub hz_text: String,
}

impl TimeInputForm {
    /// The parsed input: `Ok(None)` when blank (detect from file names).
    pub fn input(&self) -> Result<Option<TimeInput>, String> {
        let text = self.dt_text.trim();
        if text.is_empty() {
            return Ok(None);
        }
        match text.parse::<f64>() {
            Ok(dt) if dt.is_finite() && dt > 0.0 => Ok(Some(TimeInput::Increment(dt))),
            _ => Err("Δt must be a positive number of seconds".to_string()),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.input().is_ok()
    }

    /// The linked Δt / Hz fields plus what file-name detection finds for
    /// `paths` (the images the solve will use).
    pub fn show(&mut self, ui: &mut egui::Ui, id: &str, paths: &[PathBuf]) {
        egui::Grid::new(id)
            .num_columns(2)
            .spacing([8.0, 4.0])
            .min_col_width(72.0)
            .show(ui, |ui| {
                ui.label(label(ui, "Δt (s):"));
                if ui.add(egui::TextEdit::singleline(&mut self.dt_text).desired_width(80.0)).changed() {
                    self.hz_text = reciprocal_text(&self.dt_text);
                }
                ui.end_row();
                ui.label(label(ui, "Rate (Hz):"));
                if ui.add(egui::TextEdit::singleline(&mut self.hz_text).desired_width(80.0)).changed() {
                    self.dt_text = reciprocal_text(&self.hz_text);
                }
                ui.end_row();
            });
        let (text, colour) = match self.input() {
            Err(e) => (e, ui.visuals().error_fg_color),
            Ok(Some(_)) => ("Times = image index × Δt".to_string(), ui.visuals().weak_text_color()),
            Ok(None) => (
                format!("Blank: from file names — {}", detected(paths)),
                ui.visuals().weak_text_color(),
            ),
        };
        ui.label(egui::RichText::new(text).size(14.0).color(colour));
    }
}

fn label(ui: &egui::Ui, s: &str) -> egui::RichText {
    egui::RichText::new(s).size(16.0).color(ui.visuals().weak_text_color())
}

/// `1 / x` as text; blank for blank or unusable input.
fn reciprocal_text(text: &str) -> String {
    match text.trim().parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 => format!("{}", 1.0 / v),
        _ => String::new(),
    }
}

/// The timestamp format found in `paths`' names, or "none found".
fn detected(paths: &[PathBuf]) -> String {
    let stems: Vec<String> = paths
        .iter()
        .map(|p| Path::new(p).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default())
        .collect();
    let refs: Vec<&str> = stems.iter().map(String::as_str).collect();
    match (paths.len(), parse_filename_times(&refs)) {
        (0 | 1, _) => "select the images".to_string(),
        (_, Some(ft)) => ft.format.describe().to_string(),
        (_, None) => "no timestamp found".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dt_and_hz_are_reciprocal_and_blank_detects() {
        let mut f = TimeInputForm::default();
        assert_eq!(f.input(), Ok(None));
        f.dt_text = "0.1".into();
        assert_eq!(reciprocal_text(&f.dt_text), "10");
        assert_eq!(f.input(), Ok(Some(TimeInput::Increment(0.1))));
        f.dt_text = "-1".into();
        assert!(f.input().is_err());
        assert_eq!(reciprocal_text("abc"), "");
    }
}
