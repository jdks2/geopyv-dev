use std::collections::HashMap;
use std::path::{Path, PathBuf};

use eframe::egui;

use crate::draw::{DrawState, ImageCoord};

// ---------------------------------------------------------------------------
// Texture cache
// ---------------------------------------------------------------------------

pub struct TextureCache {
    map: HashMap<PathBuf, egui::TextureHandle>,
}

impl TextureCache {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn get_or_load(
        &mut self,
        path: &Path,
        ctx: &egui::Context,
    ) -> Option<&egui::TextureHandle> {
        if !self.map.contains_key(path) {
            let texture = load_texture(path, ctx)?;
            self.map.insert(path.to_path_buf(), texture);
        }
        self.map.get(path)
    }
}

fn load_texture(path: &Path, ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let img = image::open(path).ok()?;
    let grey = img.to_luma8();
    let (w, h) = grey.dimensions();

    let rgba: Vec<u8> = grey
        .pixels()
        .flat_map(|p| [p[0], p[0], p[0], 255u8])
        .collect();

    let color_image =
        egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);

    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    Some(ctx.load_texture(name, color_image, egui::TextureOptions::LINEAR))
}

// ---------------------------------------------------------------------------
// Pixel lookup data (kept separately from GPU texture)
// ---------------------------------------------------------------------------

struct GreyPixels {
    path: PathBuf,
    width: u32,
    height: u32,
    data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Hover info returned by show()
// ---------------------------------------------------------------------------

pub struct HoverInfo {
    pub pixel_x: u32,
    pub pixel_y: u32,
    pub intensity: u8,
}

// ---------------------------------------------------------------------------
// Image viewer
// ---------------------------------------------------------------------------

pub struct ImageViewer {
    zoom: f32,
    offset: egui::Vec2,
    /// Set true when path changes so first show() fits the image to the rect.
    needs_fit: bool,
    /// Path for which zoom/offset currently apply.
    last_path: Option<PathBuf>,
    grey: Option<GreyPixels>,
    /// The coordinate converter from the most recent show() call.
    last_coord: Option<ImageCoord>,
}

impl ImageViewer {
    pub fn new() -> Self {
        Self {
            zoom: 1.0,
            offset: egui::Vec2::ZERO,
            needs_fit: true,
            last_path: None,
            grey: None,
            last_coord: None,
        }
    }

    /// Renders the image inside `ui`'s available rect.
    ///
    /// `draw` — when `Some`, draw mode interactions and overlays are active and
    /// left-drag pan is suppressed so clicks register as polygon vertices.
    ///
    /// Returns hover pixel info when the cursor is over the image.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        path: &Path,
        cache: &mut TextureCache,
        draw: Option<&mut DrawState>,
    ) -> Option<HoverInfo> {
        // Detect path change.
        if self.last_path.as_deref() != Some(path) {
            self.last_path = Some(path.to_path_buf());
            self.needs_fit = true;
            self.offset = egui::Vec2::ZERO;
        }

        self.ensure_grey(path);

        let texture = cache.get_or_load(path, ui.ctx())?;
        let img_natural = texture.size_vec2();
        let tex_id = texture.id();

        let available = ui.available_rect_before_wrap();

        if self.needs_fit {
            let fit_x = available.width() / img_natural.x;
            let fit_y = available.height() / img_natural.y;
            self.zoom = fit_x.min(fit_y) * 0.95;
            self.offset = egui::Vec2::ZERO;
            self.needs_fit = false;
        }

        let response = ui.allocate_rect(available, egui::Sense::click_and_drag());
        let canvas_center = available.center();

        // Scroll-wheel zoom (always active, even in draw mode).
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let old_zoom = self.zoom;
                let factor = (1.0 + scroll * 0.004).clamp(0.8, 1.25);
                self.zoom = (self.zoom * factor).clamp(0.02, 80.0);
                let real_factor = self.zoom / old_zoom;
                if let Some(cursor) = ui.input(|i| i.pointer.hover_pos()) {
                    let q = cursor.to_vec2() - canvas_center.to_vec2();
                    self.offset = self.offset * real_factor + q * (1.0 - real_factor);
                }
            }
        }

        // Left-drag pan — suppressed when draw mode is active.
        let draw_active = draw
            .as_ref()
            .map(|d| d.mode.is_some())
            .unwrap_or(false);

        if !draw_active && response.dragged_by(egui::PointerButton::Primary) {
            self.offset += response.drag_delta();
        }

        // Double-click reset.
        if response.double_clicked() && !draw_active {
            self.needs_fit = true;
        }

        // Image display rect.
        let img_size = img_natural * self.zoom;
        let img_origin = (canvas_center + self.offset).to_vec2() - img_size * 0.5;
        let img_rect = egui::Rect::from_min_size(img_origin.to_pos2(), img_size);

        // Build coordinate converter (used by draw overlay and external callers).
        let coord = ImageCoord {
            canvas_center,
            offset: self.offset,
            zoom: self.zoom,
            img_size: img_natural,
        };
        self.last_coord = Some(coord);

        // Paint dark background + image.
        ui.painter()
            .rect_filled(available, 0.0, egui::Color32::from_rgb(15, 15, 15));
        ui.painter().image(
            tex_id,
            img_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // Draw overlay.
        if let Some(draw) = draw {
            draw.handle(ui, &response, &coord);
            draw.render(ui.painter(), &coord);
        }

        // Hover pixel info.
        if let Some(cursor) = response.hover_pos() {
            if img_rect.contains(cursor) {
                let frac_x = (cursor.x - img_rect.min.x) / img_rect.width();
                let frac_y = (cursor.y - img_rect.min.y) / img_rect.height();
                if let Some(grey) = &self.grey {
                    let px = (frac_x * grey.width as f32) as u32;
                    let py = (frac_y * grey.height as f32) as u32;
                    if px < grey.width && py < grey.height {
                        let intensity = grey.data[(py * grey.width + px) as usize];
                        return Some(HoverInfo {
                            pixel_x: px,
                            pixel_y: py,
                            intensity,
                        });
                    }
                }
            }
        }

        None
    }

    /// Returns the coordinate converter from the most recent show() call.
    pub fn last_coord(&self) -> Option<ImageCoord> {
        self.last_coord
    }

    /// Natural image dimensions if loaded.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.grey.as_ref().map(|g| (g.width, g.height))
    }

    fn ensure_grey(&mut self, path: &Path) {
        if self.grey.as_ref().map(|g| g.path.as_path()) == Some(path) {
            return;
        }
        let Ok(img) = image::open(path) else {
            return;
        };
        let grey = img.to_luma8();
        let (w, h) = grey.dimensions();
        self.grey = Some(GreyPixels {
            path: path.to_path_buf(),
            width: w,
            height: h,
            data: grey.into_raw(),
        });
    }
}
