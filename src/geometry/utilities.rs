//! Geometric utility functions.
//!
//! Translates `geopyv/geometry/utilities.py`.

use ndarray::{Array2, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Return a characteristic element length given an element area, based on an
/// equilateral triangle.
///
/// `length = sqrt(4 * |area| / sqrt(3))`
pub fn area_to_length(area: f64) -> f64 {
    (4.0 * area.abs() / 3.0_f64.sqrt()).sqrt()
}

/// Compute a first-order triangulation from a mesh element connectivity array.
///
/// Translates `geopyv.geometry.utilities.plot_triangulation`.
///
/// For `mesh_order = 1` (3-node triangles): returns the input connectivity
/// with a repeated first node appended to close each triangle, plus gathered
/// x/y coordinate paths.
///
/// For `mesh_order = 2` (6-node triangles): splits each element into four
/// sub-triangles and returns the resulting connectivity.
///
/// # Returns
/// `(triangulation, x_path, y_path)`
pub fn plot_triangulation(
    elements: ArrayView2<u32>,
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    mesh_order: u32,
) -> (Array2<u32>, Array2<f64>, Array2<f64>) {
    let n_elem = elements.nrows();

    match mesh_order {
        1 => {
            // x_p = x[elements[:, [0, 1, 2, 0]]]  shape (N, 4)
            let idx = [0usize, 1, 2, 0];
            let x_p = Array2::from_shape_fn((n_elem, 4), |(i, j)| {
                x[elements[[i, idx[j]]] as usize]
            });
            let y_p = Array2::from_shape_fn((n_elem, 4), |(i, j)| {
                y[elements[[i, idx[j]]] as usize]
            });
            let tri = elements.to_owned();
            (tri, x_p, y_p)
        }
        2 => {
            // x_p = x[elements[:, [0, 3, 1, 4, 2, 5, 0]]]  shape (N, 7)
            let path_idx = [0usize, 3, 1, 4, 2, 5, 0];
            let x_p = Array2::from_shape_fn((n_elem, 7), |(i, j)| {
                x[elements[[i, path_idx[j]]] as usize]
            });
            let y_p = Array2::from_shape_fn((n_elem, 7), |(i, j)| {
                y[elements[[i, path_idx[j]]] as usize]
            });
            // Each 6-node element becomes 4 sub-triangles:
            // [[0,3,5], [1,3,4], [2,4,5], [3,4,5]]
            let sub: [[usize; 3]; 4] = [[0, 3, 5], [1, 3, 4], [2, 4, 5], [3, 4, 5]];
            let tri =
                Array2::from_shape_fn((n_elem * 4, 3), |(i, j)| {
                    let elem = i / 4;
                    let sub_idx = i % 4;
                    elements[[elem, sub[sub_idx][j]]]
                });
            (tri, x_p, y_p)
        }
        _ => panic!("mesh_order must be 1 or 2, got {mesh_order}"),
    }
}

/// Compute the area of a polygon using the Shoelace formula.
///
/// Replicates `geopyv.geometry.utilities.PolyArea` exactly, including the
/// double-`abs` that appears in the Python source.
///
/// `pts` must have shape `(N, 2)` with columns `[x, y]`.
pub fn poly_area(pts: ArrayView2<f64>) -> f64 {
    let n = pts.nrows();
    let x = pts.column(0);
    let y = pts.column(1);
    // np.dot(x, np.roll(y, 1)) = sum(x[i] * y[(i + n - 1) % n])
    let sum1: f64 = (0..n).map(|i| x[i] * y[(i + n - 1) % n]).sum();
    // np.dot(y, np.roll(x, 1)) = sum(y[i] * x[(i + n - 1) % n])
    let sum2: f64 = (0..n).map(|i| y[i] * x[(i + n - 1) % n]).sum();
    // Python: abs(0.5 * np.abs(sum1 - sum2))  (double abs — replicated faithfully)
    (0.5 * (sum1 - sum2).abs()).abs()
}

/// Even-odd (ray-casting) point-in-polygon test. `poly` has shape `(N, 2)`
/// with columns `[x, y]` and is treated as closed; fewer than 3 vertices
/// contain nothing.
pub fn point_in_polygon(p: [f64; 2], poly: ArrayView2<f64>) -> bool {
    let n = poly.nrows();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[[i, 0]], poly[[i, 1]]);
        let (xj, yj) = (poly[[j, 0]], poly[[j, 1]]);
        if (yi > p[1]) != (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Check whether point C is counter-clockwise from A→B.
///
/// Replicates `geopyv.geometry.utilities.ccw`.
#[inline]
pub fn ccw(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
}

/// Test whether line segment AB intersects line segment CD.
///
/// Replicates `geopyv.geometry.utilities.intersect`.
#[inline]
pub fn intersect(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    ccw(a, c, d) != ccw(b, c, d) && ccw(a, b, c) != ccw(a, b, d)
}

/// Check whether a 6-node polygon self-intersects (non-adjacent segment pairs).
///
/// Replicates `geopyv.geometry.utilities.polysect`.
///
/// `n` must be a slice of exactly 6 `[x, y]` points forming a closed polygon.
///
/// Returns `Some([i, j])` (the indices of the first intersecting segment pair)
/// or `None` if no intersection is found.
pub fn polysect(n: &[[f64; 2]]) -> Option<[usize; 2]> {
    // Non-adjacent segment pair indices (matching Python's co array).
    const CO: [[usize; 2]; 9] = [
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 3],
        [1, 4],
        [1, 5],
        [2, 4],
        [2, 5],
        [3, 5],
    ];
    for pair in &CO {
        let a = n[pair[0] % 6];
        let b = n[(pair[0] + 1) % 6];
        let c = n[pair[1] % 6];
        let d = n[(pair[1] + 1) % 6];
        if intersect(a, b, c, d) {
            return Some(*pair);
        }
    }
    None
}

/// Compute the centroid of a polygon.
///
/// Replicates `geopyv.geometry.utilities.polycentroid` exactly, including a
/// typo in the Python source where the y-centroid accumulation uses
/// `coords[i, 0]` (x-component) instead of `coords[i, 1]` (y-component).
/// This is preserved for numerical equivalence.
///
/// `coords` must have shape `(N, 2)` with columns `[x, y]`.
pub fn polycentroid(coords: ArrayView2<f64>) -> [f64; 2] {
    let n = coords.nrows();
    let mut centroid = [0.0f64; 2];
    let mut sa = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let a = coords[[i, 0]] * coords[[j, 1]] - coords[[j, 0]] * coords[[i, 1]];
        sa += a;
        centroid[0] += (coords[[i, 0]] + coords[[j, 0]]) * a;
        // NOTE: Python source has coords[i, 0] here (x-component), not coords[i, 1].
        // This is a typo in the original — replicated faithfully for equivalence.
        centroid[1] += (coords[[i, 0]] + coords[[j, 1]]) * a;
    }
    centroid[0] /= 3.0 * sa;
    centroid[1] /= 3.0 * sa;
    centroid
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn area_to_length_unit() {
        // equilateral triangle of area 1: l = sqrt(4/sqrt(3))
        let l = area_to_length(1.0);
        let expected = (4.0_f64 / 3.0_f64.sqrt()).sqrt();
        assert!((l - expected).abs() < 1e-12, "got {l}, expected {expected}");
    }

    #[test]
    fn area_to_length_negative() {
        // Sign of area should not affect result.
        assert!((area_to_length(-4.0) - area_to_length(4.0)).abs() < 1e-12);
    }

    #[test]
    fn poly_area_unit_square() {
        // Vertices in CCW order: (0,0), (1,0), (1,1), (0,1) → area = 1.
        let pts = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!((poly_area(pts.view()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn poly_area_known_triangle() {
        // Right triangle with legs 3, 4 → area = 6.
        let pts = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]];
        assert!((poly_area(pts.view()) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn ccw_counter_clockwise() {
        assert!(ccw([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]));
    }

    #[test]
    fn ccw_clockwise() {
        assert!(!ccw([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]));
    }

    #[test]
    fn intersect_crossing_diagonals() {
        // Diagonals of a unit square cross.
        assert!(intersect([0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]));
    }

    #[test]
    fn intersect_parallel_sides() {
        // Top and bottom edges of a unit square do not intersect.
        assert!(!intersect([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]));
    }

    #[test]
    fn polysect_convex_hexagon_no_intersect() {
        // A regular convex hexagon should have no self-intersections.
        use std::f64::consts::PI;
        let n: [[f64; 2]; 6] = std::array::from_fn(|i| {
            let theta = 2.0 * PI * i as f64 / 6.0;
            [theta.cos(), theta.sin()]
        });
        assert!(polysect(&n).is_none());
    }

    #[test]
    fn polysect_bowtie_intersects() {
        // A "bowtie" quad (crossed) should detect intersection.
        // Use 4 of 6 points with a crossing; fill the rest with non-crossing.
        let n: [[f64; 2]; 6] = [
            [0.0, 0.0],
            [2.0, 2.0], // segment 0→1 crosses 2→3
            [2.0, 0.0],
            [0.0, 2.0], // segment 2→3 crosses 0→1
            [1.0, 3.0],
            [3.0, 3.0],
        ];
        // Segments: (0,1),(1,2),(2,3),(3,4),(4,5),(5,0)
        // CO[0] = [0,2]: seg(0→1) vs seg(2→3): (0,0)→(2,2) vs (2,0)→(0,2) — these cross!
        assert!(polysect(&n).is_some());
    }

    #[test]
    fn polycentroid_square_x_component() {
        // For a 4x4 square centred at (2, 2), x-centroid should be 2.0.
        let coords = array![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let c = polycentroid(coords.view());
        assert!((c[0] - 2.0).abs() < 1e-10, "x-centroid = {}", c[0]);
    }

    #[test]
    fn point_in_polygon_square_and_concave() {
        let sq = array![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        assert!(point_in_polygon([2.0, 2.0], sq.view()));
        assert!(!point_in_polygon([5.0, 2.0], sq.view()));
        // L-shape: the notch at (3, 3) is outside.
        let l = array![[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0]];
        assert!(point_in_polygon([1.0, 3.0], l.view()));
        assert!(!point_in_polygon([3.0, 3.0], l.view()));
        assert!(!point_in_polygon([0.5, 0.5], array![[0.0, 0.0], [1.0, 1.0]].view()));
    }
}
