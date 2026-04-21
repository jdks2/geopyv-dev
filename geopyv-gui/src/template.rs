use std::path::{Path, PathBuf};

use eframe::egui;
use serde::{Deserialize, Serialize};

pub use geopyv_dev::templates::TemplateShape;

// ---------------------------------------------------------------------------
// TemplateConfig — lightweight GUI representation saved to /Templates/*.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    pub name: String,
    pub shape: TemplateShape,
    /// Radius (Circle) or half-side (Square), in pixels.
    pub size: u32,
}

impl TemplateConfig {
    /// Count of pixels in the template mask.
    pub fn n_px(&self) -> u32 {
        match self.shape {
            TemplateShape::Circle => {
                let r = self.size as i32;
                let mut n = 0u32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx * dx + dy * dy <= r * r {
                            n += 1;
                        }
                    }
                }
                n
            }
            TemplateShape::Square => {
                let s = 2 * self.size + 1;
                s * s
            }
        }
    }

    pub fn save(&self, dir: &Path) -> Result<PathBuf, String> {
        let path = dir.join(format!("{}.json", self.name));
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| format!("Serialisation error: {e}"))?;
        std::fs::write(&path, &json).map_err(|e| format!("Write error: {e}"))?;
        Ok(path)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Read error: {e}"))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("Parse error: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Label helper
// ---------------------------------------------------------------------------

pub fn shape_label(shape: &TemplateShape) -> &'static str {
    match shape {
        TemplateShape::Circle => "Circle",
        TemplateShape::Square => "Square",
    }
}

// ---------------------------------------------------------------------------
// Template mask preview renderer
// ---------------------------------------------------------------------------

/// Renders a template mask shape with a crosshair into `rect`.
pub fn render_template_preview(
    painter: &egui::Painter,
    rect: egui::Rect,
    shape: &TemplateShape,
    size: u32,
) {
    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(15, 15, 15));

    if size == 0 {
        return;
    }

    let center = rect.center();
    let draw_radius = rect.width().min(rect.height()) * 0.40;

    let mask_color = egui::Color32::from_rgb(210, 210, 210);
    match shape {
        TemplateShape::Circle => {
            painter.circle_filled(center, draw_radius, mask_color);
        }
        TemplateShape::Square => {
            let sq = egui::Rect::from_center_size(
                center,
                egui::vec2(draw_radius * 2.0, draw_radius * 2.0),
            );
            painter.rect_filled(sq, 0.0, mask_color);
        }
    }

    let arm = rect.width().min(rect.height()) * 0.08;
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(80, 200, 100));
    painter.line_segment(
        [center - egui::vec2(arm, 0.0), center + egui::vec2(arm, 0.0)],
        stroke,
    );
    painter.line_segment(
        [center - egui::vec2(0.0, arm), center + egui::vec2(0.0, arm)],
        stroke,
    );
}
