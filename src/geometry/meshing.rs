//! Mesh region-of-interest preparation utilities.
//!
//! Translates the non-gmsh parts of `geopyv/geometry/meshing.py`.
//!
//! `_mask_image` and `_define_RoI` are translated as `mask_image` and
//! `define_roi` respectively.
//!
//! `_gmsh_initializer` is **not** translated here because it requires the gmsh
//! C library.  Gmsh integration is deferred to the Session 5 mesh module, where
//! the full mesh generation pipeline will be assembled.  Call gmsh Python
//! bindings from the PyO3 side for now.
//!
//! # Polygon-fill algorithm
//!
//! PIL's `ImageDraw.polygon` is replaced by a winding-number point-in-polygon
//! test applied per pixel.  Interior pixels (winding ≠ 0) are set to 1.
//! Boundary pixels may differ from PIL at sub-pixel precision; test designs
//! avoid relying on exact boundary values.

use ndarray::{Array2, ArrayView2};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Output of `define_roi`.
pub struct RoiData {
    /// All boundary + exclusion node coordinates concatenated, shape `(M, 2)`.
    pub borders: Array2<f64>,
    /// Segment connectivity array (each row = `[start_node, end_node]`), shape `(M, 2)`.
    pub segments: Array2<i32>,
    /// Curve lists: each entry is a list of start-node indices for one closed loop.
    pub curves: Vec<Vec<i32>>,
    /// Binary mask if an image shape was supplied to `define_roi`, else `None`.
    pub mask: Option<Array2<u8>>,
}

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Build a binary mask for the region of interest.
///
/// Translates `geopyv.geometry.meshing._mask_image`.
///
/// # Arguments
/// * `img_shape`        - `(height, width)` of the output mask.
/// * `boundary_nodes`   - `(N, 2)` array of `[x, y]` boundary polygon vertices.
/// * `boundary_hard`    - If `true`, fill only inside the boundary polygon;
///                        if `false`, fill the entire image.
/// * `exclusion_nodes`  - Slice of exclusion polygon vertex arrays; each is `(K, 2)`.
///
/// Returns a `(height, width)` array with values 0 (masked) or 1 (active).
pub fn mask_image(
    img_shape: (usize, usize),
    boundary_nodes: ArrayView2<f64>,
    boundary_hard: bool,
    exclusion_nodes: &[ArrayView2<f64>],
) -> Array2<u8> {
    let (height, width) = img_shape;
    let mut mask = Array2::<u8>::zeros((height, width));

    // Collect boundary polygon as a Vec<[f64; 2]> of (x, y) pairs.
    let boundary_poly: Vec<[f64; 2]> = boundary_nodes
        .outer_iter()
        .map(|r| [r[0], r[1]])
        .collect();

    // Fill boundary region.
    for row in 0..height {
        for col in 0..width {
            let px = col as f64 + 0.5; // pixel centre x
            let py = row as f64 + 0.5; // pixel centre y
            let inside = if boundary_hard {
                point_in_polygon(px, py, &boundary_poly)
            } else {
                true
            };
            if inside {
                mask[[row, col]] = 1;
            }
        }
    }

    // Zero out exclusion regions.
    for exc in exclusion_nodes {
        let exc_poly: Vec<[f64; 2]> = exc.outer_iter().map(|r| [r[0], r[1]]).collect();
        for row in 0..height {
            for col in 0..width {
                if mask[[row, col]] == 1 {
                    let px = col as f64 + 0.5;
                    let py = row as f64 + 0.5;
                    if point_in_polygon(px, py, &exc_poly) {
                        mask[[row, col]] = 0;
                    }
                }
            }
        }
    }

    mask
}

/// Build segment and curve data for the mesh generator from boundary + exclusion nodes.
///
/// Translates `geopyv.geometry.meshing._define_RoI`.
///
/// When `img_shape` is `Some((h, w))`, also computes the binary mask via
/// `mask_image` and stores it in the returned `RoiData`.
///
/// # Arguments
/// * `boundary_nodes`   - `(N, 2)` boundary polygon vertices `[x, y]`.
/// * `boundary_hard`    - Passed through to `mask_image` if `img_shape` is supplied.
/// * `exclusion_nodes`  - Exclusion polygon arrays.
/// * `img_shape`        - Optional image `(height, width)` for mask generation.
pub fn define_roi(
    boundary_nodes: ArrayView2<f64>,
    boundary_hard: bool,
    exclusion_nodes: &[ArrayView2<f64>],
    img_shape: Option<(usize, usize)>,
) -> RoiData {
    let n_boundary = boundary_nodes.nrows();

    // Build initial segment array for the boundary.
    let mut borders_rows: Vec<[f64; 2]> = boundary_nodes
        .outer_iter()
        .map(|r| [r[0], r[1]])
        .collect();

    let mut seg_rows: Vec<[i32; 2]> = (0..n_boundary as i32)
        .map(|i| [i, (i + 1) % n_boundary as i32])
        .collect();

    let mut curves: Vec<Vec<i32>> = vec![(0..n_boundary as i32).collect()];

    // Append each exclusion.
    for exc in exclusion_nodes {
        let cur_max = seg_rows
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .max()
            .unwrap_or(-1);
        let n_exc = exc.nrows() as i32;
        let start = cur_max + 1;

        for r in exc.outer_iter() {
            borders_rows.push([r[0], r[1]]);
        }
        let exc_seg: Vec<[i32; 2]> = (0..n_exc)
            .map(|i| [start + i, start + (i + 1) % n_exc])
            .collect();
        let exc_curve: Vec<i32> = (0..n_exc).map(|i| start + i).collect();
        seg_rows.extend_from_slice(&exc_seg);
        curves.push(exc_curve);
    }

    let n_total = borders_rows.len();
    let borders = Array2::from_shape_fn((n_total, 2), |(i, j)| borders_rows[i][j]);

    let n_seg = seg_rows.len();
    let segments = Array2::from_shape_fn((n_seg, 2), |(i, j)| seg_rows[i][j]);

    let mask = img_shape.map(|shape| {
        mask_image(shape, boundary_nodes, boundary_hard, exclusion_nodes)
    });

    RoiData { borders, segments, curves, mask }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Winding-number point-in-polygon test.
///
/// Returns `true` if `(px, py)` is strictly inside the polygon defined by
/// the vertex sequence `poly`.  The polygon is implicitly closed
/// (last vertex connects to first).
///
/// Uses the winding-number algorithm which correctly handles self-intersecting
/// polygons.
fn point_in_polygon(px: f64, py: f64, poly: &[[f64; 2]]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut winding = 0i32;
    for i in 0..n {
        let j = (i + 1) % n;
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        if yi <= py {
            if yj > py {
                // Upward crossing.
                let cross = (xj - xi) * (py - yi) - (px - xi) * (yj - yi);
                if cross > 0.0 {
                    winding += 1;
                }
            }
        } else if yj <= py {
            // Downward crossing.
            let cross = (xj - xi) * (py - yi) - (px - xi) * (yj - yi);
            if cross < 0.0 {
                winding -= 1;
            }
        }
    }
    winding != 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn square_poly() -> Array2<f64> {
        // A 10×10 square from (10,10) to (20,20).
        array![[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
    }

    #[test]
    fn mask_image_interior_is_one() {
        let poly = square_poly();
        let mask = mask_image((30, 30), poly.view(), true, &[]);
        // Centre of the square (row=15, col=15) should be 1.
        assert_eq!(mask[[15, 15]], 1);
    }

    #[test]
    fn mask_image_exterior_is_zero() {
        let poly = square_poly();
        let mask = mask_image((30, 30), poly.view(), true, &[]);
        // Outside the square, e.g. (0, 0).
        assert_eq!(mask[[0, 0]], 0);
    }

    #[test]
    fn mask_image_soft_boundary_fills_all() {
        let poly = square_poly();
        let mask = mask_image((30, 30), poly.view(), false, &[]);
        // Soft boundary: entire image is 1.
        assert!(mask.iter().all(|&v| v == 1));
    }

    #[test]
    fn mask_image_exclusion_zeros_interior() {
        // Boundary: 30×30 image. Exclusion: 5×5 square at (12..17, 12..17).
        let boundary = array![[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]];
        let exclusion = array![[12.0, 12.0], [17.0, 12.0], [17.0, 17.0], [12.0, 17.0]];
        let mask = mask_image((30, 30), boundary.view(), true, &[exclusion.view()]);
        // Well inside exclusion → 0.
        assert_eq!(mask[[14, 14]], 0);
        // Well outside exclusion → 1.
        assert_eq!(mask[[2, 2]], 1);
    }

    #[test]
    fn define_roi_segment_count() {
        // Boundary with N nodes should produce N segments.
        let boundary = square_poly(); // 4 nodes
        let roi = define_roi(boundary.view(), true, &[], None);
        assert_eq!(roi.segments.nrows(), 4);
        assert_eq!(roi.curves.len(), 1);
        assert_eq!(roi.curves[0].len(), 4);
    }

    #[test]
    fn define_roi_exclusion_appended() {
        let boundary = square_poly(); // 4 nodes
        let exclusion = array![[12.0, 12.0], [17.0, 12.0], [17.0, 17.0], [12.0, 17.0]];
        let roi = define_roi(boundary.view(), true, &[exclusion.view()], None);
        assert_eq!(roi.borders.nrows(), 8); // 4 + 4
        assert_eq!(roi.segments.nrows(), 8); // 4 + 4
        assert_eq!(roi.curves.len(), 2);
    }

    #[test]
    fn define_roi_with_mask() {
        let boundary = array![[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]];
        let roi = define_roi(boundary.view(), true, &[], Some((20, 20)));
        assert!(roi.mask.is_some());
        let mask = roi.mask.unwrap();
        assert_eq!(mask.dim(), (20, 20));
    }

    #[test]
    fn point_in_polygon_inside() {
        let poly = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(point_in_polygon(5.0, 5.0, &poly));
    }

    #[test]
    fn point_in_polygon_outside() {
        let poly = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(!point_in_polygon(15.0, 15.0, &poly));
    }
}
