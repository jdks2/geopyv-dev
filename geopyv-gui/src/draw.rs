use eframe::egui;

// ---------------------------------------------------------------------------
// Coordinate converter (image ↔ screen)
// ---------------------------------------------------------------------------

/// Converts between image-pixel space and screen space, given the current
/// pan offset and zoom of the image viewer.
#[derive(Clone, Copy)]
pub struct ImageCoord {
    pub canvas_center: egui::Pos2,
    pub offset: egui::Vec2,
    pub zoom: f32,
    pub img_size: egui::Vec2,
}

impl ImageCoord {
    pub fn to_screen(&self, img: egui::Pos2) -> egui::Pos2 {
        (self.canvas_center.to_vec2()
            + self.offset
            + img.to_vec2() * self.zoom
            - self.img_size * self.zoom * 0.5)
            .to_pos2()
    }

    pub fn to_image(&self, screen: egui::Pos2) -> egui::Pos2 {
        let rel = screen.to_vec2()
            - self.canvas_center.to_vec2()
            - self.offset
            + self.img_size * self.zoom * 0.5;
        (rel / self.zoom).to_pos2()
    }
}

// ---------------------------------------------------------------------------
// Draw mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveDrawMode {
    Boundary,
    Exclusion,
    Seed,
    /// Single coordinate (Subset / Particle).
    Point,
}

// ---------------------------------------------------------------------------
// Committed results
// ---------------------------------------------------------------------------

/// A committed polygon stored in image-pixel coordinates.
#[derive(Debug, Clone)]
pub struct DrawnPolygon {
    /// Closed polygon vertices in image pixel coordinates.
    pub vertices: Vec<egui::Pos2>,
}

// ---------------------------------------------------------------------------
// Draw state
// ---------------------------------------------------------------------------

pub struct DrawState {
    /// Active interaction mode; `None` means display-only (no input handling).
    pub mode: Option<ActiveDrawMode>,

    // Committed results accumulated during a form session.
    pub boundary: Option<DrawnPolygon>,
    pub exclusions: Vec<DrawnPolygon>,
    /// Seed point in image coords.
    pub seed: Option<egui::Pos2>,
    /// Single coordinate in image coords (Subset / Particle).
    pub point: Option<egui::Pos2>,

    // In-progress state.
    in_progress: Vec<egui::Pos2>,
    self_intersects: bool,
    /// Last cursor position in screen space (for snap indicator).
    cursor_screen: Option<egui::Pos2>,
}

/// Screen-space snap distance in pixels.
const SNAP_PX: f32 = 8.0;
/// Crosshair arm length in pixels.
const CROSSHAIR_ARM: f32 = 10.0;

impl DrawState {
    pub fn new() -> Self {
        Self {
            mode: None,
            boundary: None,
            exclusions: Vec::new(),
            seed: None,
            point: None,
            in_progress: Vec::new(),
            self_intersects: false,
            cursor_screen: None,
        }
    }

    /// Activate a draw mode (clears the in-progress buffer).
    pub fn start(&mut self, mode: ActiveDrawMode) {
        self.mode = Some(mode);
        self.in_progress.clear();
        self.self_intersects = false;
        self.cursor_screen = None;
    }

    /// Deactivate drawing without committing in-progress work.
    pub fn cancel(&mut self) {
        self.mode = None;
        self.in_progress.clear();
        self.self_intersects = false;
    }

    /// Remove the last in-progress vertex (Ctrl+Z / right-click).
    pub fn undo_vertex(&mut self) {
        self.in_progress.pop();
        self.update_intersection();
    }

    /// Whether the required data for Run is present given the active form context.
    /// Callers check their own required fields; this just reports drawing completeness.
    pub fn boundary_ok(&self) -> bool {
        self.boundary.is_some()
    }

    pub fn seed_ok(&self) -> bool {
        self.seed.is_some()
    }

    pub fn point_ok(&self) -> bool {
        self.point.is_some()
    }

    /// True when the in-progress polygon is self-intersecting (Run should be disabled).
    pub fn has_self_intersection(&self) -> bool {
        self.self_intersects
    }

    // -----------------------------------------------------------------------
    // Input handling — called from ImageViewer::show
    // -----------------------------------------------------------------------

    /// Process mouse and keyboard input for the active draw mode.
    /// Should be called once per frame while drawing is active.
    pub fn handle(
        &mut self,
        ui: &mut egui::Ui,
        response: &egui::Response,
        coord: &ImageCoord,
    ) {
        let Some(mode) = self.mode else {
            return;
        };

        // Global keyboard shortcuts.
        let (ctrl_z, escape) = ui.input(|i| {
            (
                i.modifiers.ctrl && i.key_pressed(egui::Key::Z),
                i.key_pressed(egui::Key::Escape),
            )
        });

        if escape {
            self.cancel();
            return;
        }
        if ctrl_z {
            self.undo_vertex();
        }

        // Update cursor position for snap indicator.
        self.cursor_screen = response.hover_pos();

        match mode {
            ActiveDrawMode::Boundary | ActiveDrawMode::Exclusion => {
                self.handle_polygon(ui, response, coord, mode);
            }
            ActiveDrawMode::Seed => {
                self.handle_seed(response, coord);
            }
            ActiveDrawMode::Point => {
                self.handle_point(response, coord);
            }
        }
    }

    fn handle_polygon(
        &mut self,
        _ui: &mut egui::Ui,
        response: &egui::Response,
        coord: &ImageCoord,
        mode: ActiveDrawMode,
    ) {
        // Right-click: remove last vertex.
        if response.clicked_by(egui::PointerButton::Secondary) {
            self.undo_vertex();
            return;
        }

        if !response.clicked_by(egui::PointerButton::Primary) {
            return;
        }

        let Some(cursor) = response.interact_pointer_pos() else {
            return;
        };

        // Snap-to-close: check if cursor is near first vertex.
        if self.in_progress.len() >= 3 {
            let first_screen = coord.to_screen(self.in_progress[0]);
            if cursor.distance(first_screen) < SNAP_PX {
                self.commit_polygon(mode);
                return;
            }
        }

        let img_pos = coord.to_image(cursor);

        // Exclusion vertices must be inside the boundary.
        if mode == ActiveDrawMode::Exclusion {
            if let Some(b) = &self.boundary {
                if !point_in_polygon(img_pos, &b.vertices) {
                    return;
                }
            }
        }

        self.in_progress.push(img_pos);
        self.update_intersection();
    }

    fn handle_seed(&mut self, response: &egui::Response, coord: &ImageCoord) {
        if !response.clicked_by(egui::PointerButton::Primary) {
            return;
        }
        let Some(cursor) = response.interact_pointer_pos() else {
            return;
        };
        let img_pos = coord.to_image(cursor);
        if let Some(b) = &self.boundary {
            if !point_in_polygon(img_pos, &b.vertices) {
                return;
            }
        }
        self.seed = Some(img_pos);
        self.mode = None; // Auto-deactivate after placement.
    }

    fn handle_point(&mut self, response: &egui::Response, coord: &ImageCoord) {
        if !response.clicked_by(egui::PointerButton::Primary) {
            return;
        }
        let Some(cursor) = response.interact_pointer_pos() else {
            return;
        };
        self.point = Some(coord.to_image(cursor));
        self.mode = None;
    }

    fn commit_polygon(&mut self, mode: ActiveDrawMode) {
        if self.in_progress.len() < 3 || self.self_intersects {
            return;
        }
        let poly = DrawnPolygon {
            vertices: self.in_progress.drain(..).collect(),
        };
        match mode {
            ActiveDrawMode::Boundary => self.boundary = Some(poly),
            ActiveDrawMode::Exclusion => self.exclusions.push(poly),
            _ => {}
        }
        self.self_intersects = false;
        self.mode = None;
    }

    fn update_intersection(&mut self) {
        self.self_intersects = polygon_self_intersects(&self.in_progress);
    }

    // -----------------------------------------------------------------------
    // Rendering — called from ImageViewer::show
    // -----------------------------------------------------------------------

    /// Paint all committed results and the in-progress overlay.
    pub fn render(&self, painter: &egui::Painter, coord: &ImageCoord) {
        // Committed boundary (green).
        if let Some(b) = &self.boundary {
            paint_polygon(painter, coord, &b.vertices, BOUNDARY_FILL, BOUNDARY_STROKE, 2.0);
        }

        // Committed exclusions (red).
        for ex in &self.exclusions {
            paint_polygon(painter, coord, &ex.vertices, EXCLUSION_FILL, EXCLUSION_STROKE, 2.0);
        }

        // Committed seed.
        if let Some(s) = self.seed {
            paint_crosshair_plus(painter, coord.to_screen(s), SEED_COLOR);
        }

        // Committed point.
        if let Some(p) = self.point {
            paint_crosshair_x(painter, coord.to_screen(p), SEED_COLOR);
        }

        // In-progress polygon.
        if !self.in_progress.is_empty() {
            self.render_in_progress(painter, coord);
        }
    }

    fn render_in_progress(&self, painter: &egui::Painter, coord: &ImageCoord) {
        let Some(mode) = self.mode else {
            return;
        };

        let screen_pts: Vec<egui::Pos2> = self
            .in_progress
            .iter()
            .map(|&p| coord.to_screen(p))
            .collect();

        let (fill_color, stroke_color) = if self.self_intersects {
            (INVALID_FILL, INVALID_STROKE)
        } else {
            match mode {
                ActiveDrawMode::Boundary => (BOUNDARY_FILL, BOUNDARY_STROKE),
                ActiveDrawMode::Exclusion => (EXCLUSION_FILL, EXCLUSION_STROKE),
                _ => (BOUNDARY_FILL, BOUNDARY_STROKE),
            }
        };

        // Draw line through in-progress vertices.
        if screen_pts.len() >= 2 {
            painter.add(egui::Shape::Path(egui::epaint::PathShape::line(
                screen_pts.clone(),
                egui::Stroke::new(2.0, stroke_color),
            )));
        }

        // Translucent fill if ≥3 vertices.
        if screen_pts.len() >= 3 {
            painter.add(egui::Shape::Path(egui::epaint::PathShape {
                points: screen_pts.clone(),
                closed: true,
                fill: fill_color,
                stroke: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT).into(),
            }));
        }

        // Vertex dots.
        for &pt in &screen_pts {
            painter.circle_filled(pt, 3.0, stroke_color);
        }

        // Snap indicator: circle around first vertex when cursor is close enough.
        if screen_pts.len() >= 3 {
            if let Some(cursor) = self.cursor_screen {
                if cursor.distance(screen_pts[0]) < SNAP_PX * 2.0 {
                    painter.circle_stroke(
                        screen_pts[0],
                        SNAP_PX,
                        egui::Stroke::new(1.5, stroke_color),
                    );
                }
            }
        }

        // Preview edge from last vertex to cursor.
        if let (Some(last), Some(cursor)) = (screen_pts.last(), self.cursor_screen) {
            painter.add(egui::Shape::dashed_line(
                &[*last, cursor],
                egui::Stroke::new(1.0, stroke_color.linear_multiply(0.6)),
                6.0,
                3.0,
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Painting helpers
// ---------------------------------------------------------------------------

fn paint_polygon(
    painter: &egui::Painter,
    coord: &ImageCoord,
    verts: &[egui::Pos2],
    fill: egui::Color32,
    stroke_color: egui::Color32,
    stroke_width: f32,
) {
    let screen_pts: Vec<egui::Pos2> = verts.iter().map(|&p| coord.to_screen(p)).collect();
    if screen_pts.len() < 3 {
        return;
    }
    painter.add(egui::Shape::Path(egui::epaint::PathShape {
        points: screen_pts.clone(),
        closed: true,
        fill,
        stroke: egui::Stroke::new(stroke_width, stroke_color).into(),
    }));
}

fn paint_crosshair_plus(painter: &egui::Painter, center: egui::Pos2, color: egui::Color32) {
    let s = egui::Stroke::new(2.0, color);
    painter.line_segment(
        [center - egui::vec2(CROSSHAIR_ARM, 0.0), center + egui::vec2(CROSSHAIR_ARM, 0.0)],
        s,
    );
    painter.line_segment(
        [center - egui::vec2(0.0, CROSSHAIR_ARM), center + egui::vec2(0.0, CROSSHAIR_ARM)],
        s,
    );
}

fn paint_crosshair_x(painter: &egui::Painter, center: egui::Pos2, color: egui::Color32) {
    let d = CROSSHAIR_ARM / std::f32::consts::SQRT_2;
    let s = egui::Stroke::new(2.0, color);
    painter.line_segment(
        [center - egui::vec2(d, d), center + egui::vec2(d, d)],
        s,
    );
    painter.line_segment(
        [center - egui::vec2(d, -d), center + egui::vec2(d, -d)],
        s,
    );
}

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------

const BOUNDARY_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(0, 180, 60, 35);
const BOUNDARY_STROKE: egui::Color32 = egui::Color32::from_rgb(0, 210, 70);
const EXCLUSION_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(200, 40, 0, 35);
const EXCLUSION_STROKE: egui::Color32 = egui::Color32::from_rgb(220, 60, 0);
const INVALID_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(255, 0, 0, 40);
const INVALID_STROKE: egui::Color32 = egui::Color32::from_rgb(255, 50, 50);
const SEED_COLOR: egui::Color32 = egui::Color32::WHITE;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Ray-casting point-in-polygon test.
pub fn point_in_polygon(p: egui::Pos2, polygon: &[egui::Pos2]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let vi = polygon[i];
        let vj = polygon[j];
        if ((vi.y > p.y) != (vj.y > p.y))
            && (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// O(n²) self-intersection test for an open polygon (not yet closed).
fn polygon_self_intersects(verts: &[egui::Pos2]) -> bool {
    let n = verts.len();
    if n < 4 {
        return false;
    }
    for i in 0..n - 1 {
        let p1 = verts[i];
        let p2 = verts[i + 1];
        // Skip adjacent edges; the closing edge wraps so skip i+2 up to n-2.
        let start = if i == 0 { 2 } else { i + 2 };
        for j in start..n - 1 {
            let p3 = verts[j];
            let p4 = verts[j + 1];
            if segments_intersect(p1, p2, p3, p4) {
                return true;
            }
        }
    }
    false
}

/// Cross product of 2D vectors (p1→p2) and (p1→p3).
fn cross2d(p1: egui::Pos2, p2: egui::Pos2, p3: egui::Pos2) -> f32 {
    (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
}

fn on_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> bool {
    p.x.min(a.x).min(b.x) <= p.x + f32::EPSILON
        && p.x <= a.x.max(b.x) + f32::EPSILON
        && p.y.min(a.y).min(b.y) <= p.y + f32::EPSILON
        && p.y <= a.y.max(b.y) + f32::EPSILON
}

fn segments_intersect(p1: egui::Pos2, p2: egui::Pos2, p3: egui::Pos2, p4: egui::Pos2) -> bool {
    let d1 = cross2d(p3, p4, p1);
    let d2 = cross2d(p3, p4, p2);
    let d3 = cross2d(p1, p2, p3);
    let d4 = cross2d(p1, p2, p4);

    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }

    if d1 == 0.0 && on_segment(p1, p3, p4) {
        return true;
    }
    if d2 == 0.0 && on_segment(p2, p3, p4) {
        return true;
    }
    if d3 == 0.0 && on_segment(p3, p1, p2) {
        return true;
    }
    if d4 == 0.0 && on_segment(p4, p1, p2) {
        return true;
    }

    false
}
