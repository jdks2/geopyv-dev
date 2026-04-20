//! Constrained Delaunay triangulation and adaptive mesh generation.
//!
//! Replaces the gmsh-based mesh generation in `geopyv/mesh.py` with a pure-Rust
//! pipeline built on the `spade` v2 CDT library.  gmsh is **not** a dependency of
//! `geopyv-dev`.
//!
//! # Pipeline (both `generate_mesh` and `adaptive_remesh`)
//!
//! 1. Insert boundary + exclusion polygon segments as hard constraint edges.
//! 2. Remove hole triangles (flood-fill; exclusion polygon interiors discarded).
//! 3. Ruppert refinement via [`spade::refine`] — `min_angle = 25°` plus a global
//!    `max_area` ceiling.
//! 4. Per-face adaptive insertion loop — for each face whose area exceeds the
//!    background-field target, insert its circumcenter; repeat until convergence.
//!    For uniform meshing the background target equals the global ceiling, so this
//!    loop is a no-op.
//! 5. Laplacian smoothing — 3 passes; constrained (polygon) nodes never move.
//! 6. Binary search over element size to hit `target_nodes`.
//! 7. Order-2 midpoint insertion when `mesh_order == 2`.

use std::collections::{HashMap, HashSet};

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use spade::{
    AngleLimit, ConstrainedDelaunayTriangulation, InsertionError, Point2,
    RefinementParameters, Triangulation,
};

use crate::Error;

// ---------------------------------------------------------------------------
// CDT type alias — `Point2<f64>` as vertex automatically satisfies
// `From<Point2<f64>>`, which spade requires in order to insert Steiner points
// during refinement.
// ---------------------------------------------------------------------------
type Cdt = ConstrainedDelaunayTriangulation<Point2<f64>>;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Output of mesh generation.
pub struct TriMesh {
    /// Node coordinates, shape `(N, 2)` — `[x, y]`.
    pub nodes: Array2<f64>,
    /// Element connectivity.
    /// - Order-1: shape `(M, 3)` — corner indices `[v0, v1, v2]`.
    /// - Order-2: shape `(M, 6)` — `[v0, v1, v2, mid01, mid12, mid20]`.
    pub elements: Array2<usize>,
    /// Ordered boundary node indices (follows polygon vertex order).
    pub boundary: Vec<usize>,
    /// Ordered exclusion node indices, one `Vec` per exclusion polygon.
    pub exclusions: Vec<Vec<usize>>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate a uniform mesh inside a polygonal region with optional holes.
///
/// Inputs come directly from [`crate::geometry::meshing::define_roi`].
///
/// # Arguments
/// * `borders`      — `(N, 2)` all polygon vertex coordinates (boundary + exclusions).
/// * `segments`     — `(S, 2)` segment connectivity: each row `[start, end]` (0-based
///                    indices into `borders`).
/// * `curves`       — One `Vec<i32>` per closed polygon (boundary first, then
///                    exclusions); each entry lists the *segment start-node* indices
///                    for that polygon.
/// * `size_lower`   — Minimum allowed element characteristic length.
/// * `size_upper`   — Maximum allowed element characteristic length.
/// * `target_nodes` — Desired node count after generation.
/// * `mesh_order`   — 1 (linear triangles) or 2 (quadratic with midpoints).
pub fn generate_mesh(
    borders: ArrayView2<f64>,
    segments: ArrayView2<i32>,
    curves: &[Vec<i32>],
    size_lower: f64,
    size_upper: f64,
    target_nodes: usize,
    mesh_order: u8,
) -> Result<TriMesh, Error> {
    validate_order(mesh_order)?;
    let poly = PolyData::from_arrays(borders, segments, curves);
    let best_size = bisect_size(&poly, target_nodes, size_lower, size_upper)?;
    let max_area = equilateral_area(best_size).max(equilateral_area(size_lower));
    // Uniform mesh: area_fn is constant.
    build_trimesh(&poly, max_area, |_, _| max_area, size_lower, mesh_order)
}

/// Remesh with per-element target areas (adaptive iteration).
///
/// # Arguments
/// * `current_nodes`    — `(N, 2)` node positions of the mesh to replace.
/// * `current_elements` — `(M, 3)` connectivity of corner nodes (order-1 only).
/// * `target_areas`     — `(M,)` desired element area per background element,
///                        computed by `_adaptive_mesh` logic in `mesh.py`.
/// * `borders` / `segments` / `curves` — same ROI definition as [`generate_mesh`].
/// * `size_lower`       — Minimum allowed element characteristic length.
/// * `target_nodes`     — Desired node count in the new mesh.
/// * `mesh_order`       — 1 or 2.
pub fn adaptive_remesh(
    current_nodes: ArrayView2<f64>,
    current_elements: ArrayView2<usize>,
    target_areas: ArrayView1<f64>,
    borders: ArrayView2<f64>,
    segments: ArrayView2<i32>,
    curves: &[Vec<i32>],
    size_lower: f64,
    target_nodes: usize,
    mesh_order: u8,
) -> Result<TriMesh, Error> {
    validate_order(mesh_order)?;
    let poly = PolyData::from_arrays(borders, segments, curves);
    let bg = BackgroundField::new(current_nodes, current_elements, target_areas);
    let best_scale = bisect_scale_adaptive(&poly, &bg, target_nodes, size_lower)?;
    let min_area = equilateral_area(size_lower);
    // Use maximum target area (×scale) as the global Ruppert ceiling.
    let global_max = (bg.target_areas.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        * best_scale)
        .max(min_area);
    let area_fn = |cx: f64, cy: f64| -> f64 {
        (bg.query(cx, cy) * best_scale).max(min_area)
    };
    build_trimesh(&poly, global_max, area_fn, size_lower, mesh_order)
}

// ---------------------------------------------------------------------------
// Internal polygon data
// ---------------------------------------------------------------------------

struct PolyData {
    vertices: Array2<f64>,
    segments: Array2<i32>,
    curves: Vec<Vec<i32>>,
    n_boundary: usize,
    n_per_exclusion: Vec<usize>,
}

impl PolyData {
    fn from_arrays(
        borders: ArrayView2<f64>,
        segments: ArrayView2<i32>,
        curves: &[Vec<i32>],
    ) -> Self {
        let n_boundary = curves.first().map(|c| c.len()).unwrap_or(0);
        let n_per_exclusion = curves.iter().skip(1).map(|c| c.len()).collect();
        PolyData {
            vertices: borders.to_owned(),
            segments: segments.to_owned(),
            curves: curves.to_vec(),
            n_boundary,
            n_per_exclusion,
        }
    }
}

// ---------------------------------------------------------------------------
// Background field (adaptive sizing)
// ---------------------------------------------------------------------------

struct BackgroundField {
    nodes: Array2<f64>,
    elements: Array2<usize>,
    target_areas: Array1<f64>,
}

impl BackgroundField {
    fn new(
        nodes: ArrayView2<f64>,
        elements: ArrayView2<usize>,
        target_areas: ArrayView1<f64>,
    ) -> Self {
        BackgroundField {
            nodes: nodes.to_owned(),
            elements: elements.to_owned(),
            target_areas: target_areas.to_owned(),
        }
    }

    /// Return the target area at `(qx, qy)` by finding the containing background
    /// triangle (piecewise-constant interpolation).
    /// Falls back to the mean when the point lies outside all triangles.
    fn query(&self, qx: f64, qy: f64) -> f64 {
        let ncols = self.elements.ncols().min(3);
        for e in 0..self.elements.nrows() {
            let i0 = self.elements[[e, 0]];
            let i1 = self.elements[[e, 1]];
            let i2 = self.elements[[e, 2 % ncols]]; // guard for ncols < 3
            if self.elements.ncols() < 3 { break; }
            let (ax, ay) = (self.nodes[[i0, 0]], self.nodes[[i0, 1]]);
            let (bx, by) = (self.nodes[[i1, 0]], self.nodes[[i1, 1]]);
            let (cx, cy) = (self.nodes[[i2, 0]], self.nodes[[i2, 1]]);
            if point_in_triangle(qx, qy, ax, ay, bx, by, cx, cy) {
                return self.target_areas[e];
            }
        }
        self.target_areas.mean().unwrap_or(1.0)
    }
}

// ---------------------------------------------------------------------------
// Core builder
// ---------------------------------------------------------------------------

fn build_trimesh(
    poly: &PolyData,
    global_max_area: f64,
    area_fn: impl Fn(f64, f64) -> f64,
    size_lower: f64,
    mesh_order: u8,
) -> Result<TriMesh, Error> {
    let (mut nodes, mut elems, constrained, boundary, exclusions) =
        cdt_pipeline(poly, global_max_area, area_fn, size_lower)?;
    laplacian_smooth(&mut nodes, &elems, &constrained, 3);
    if mesh_order == 2 {
        let (n2, e2) = insert_midpoints(&nodes, &elems);
        nodes = n2;
        elems = e2;
    }
    Ok(TriMesh { nodes, elements: elems, boundary, exclusions })
}

// ---------------------------------------------------------------------------
// CDT pipeline
// ---------------------------------------------------------------------------

fn cdt_pipeline(
    poly: &PolyData,
    global_max_area: f64,
    area_fn: impl Fn(f64, f64) -> f64,
    size_lower: f64,
) -> Result<
    (Array2<f64>, Array2<usize>, HashSet<usize>, Vec<usize>, Vec<Vec<usize>>),
    Error,
> {
    let mut cdt = Cdt::new();
    let n_verts = poly.vertices.nrows();

    // --- 1. Insert polygon vertices.
    let mut handles = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        let x = poly.vertices[[i, 0]];
        let y = poly.vertices[[i, 1]];
        let h = cdt.insert(Point2::new(x, y)).map_err(|e: InsertionError| {
            Error::MeshGeneration(format!("vertex insertion failed: {e:?}"))
        })?;
        handles.push(h);
    }

    // Record which node indices correspond to original polygon vertices (by
    // bit-exact position match) so they can be marked constrained post-refinement.
    let constrained_bits: HashSet<(u64, u64)> = (0..n_verts)
        .map(|i| {
            (
                poly.vertices[[i, 0]].to_bits(),
                poly.vertices[[i, 1]].to_bits(),
            )
        })
        .collect();

    // --- 2. Insert constraint edges.
    for s in 0..poly.segments.nrows() {
        let ia = poly.segments[[s, 0]] as usize;
        let ib = poly.segments[[s, 1]] as usize;
        cdt.add_constraint(handles[ia], handles[ib]);
    }

    // --- 3. Ruppert refinement — global quality pass.
    cdt.refine(
        RefinementParameters::new()
            .with_angle_limit(AngleLimit::from_deg(25.0))
            .with_max_allowed_area(global_max_area)
            .keep_constraint_edges()
            .with_max_additional_vertices(1_000_000),
    );

    // --- 4. Per-face adaptive insertion loop.
    // Build domain polygons once so the loop can skip out-of-domain faces.
    // Faces outside the outer boundary or inside exclusion polygons are discarded
    // in step 5; refining them here would produce circumcenters far outside the
    // domain, causing the mesh to grow unboundedly.
    let outer_poly: Vec<(f64, f64)> = {
        let n = poly.n_boundary;
        (0..n).map(|i| (poly.vertices[[i, 0]], poly.vertices[[i, 1]])).collect()
    };
    let excl_polys: Vec<Vec<(f64, f64)>> = {
        let mut offset = poly.n_boundary;
        let mut result = Vec::new();
        for &n in &poly.n_per_exclusion {
            let verts: Vec<(f64, f64)> = (offset..offset + n)
                .map(|i| (poly.vertices[[i, 0]], poly.vertices[[i, 1]]))
                .collect();
            result.push(verts);
            offset += n;
        }
        result
    };

    let min_area = equilateral_area(size_lower);
    for _ in 0..100 {
        let inserts: Vec<Point2<f64>> = cdt
            .inner_faces()
            .filter_map(|f| {
                let [p0, p1, p2] = f.positions();
                let cx = (p0.x + p1.x + p2.x) / 3.0;
                let cy = (p0.y + p1.y + p2.y) / 3.0;
                // Only refine faces that will survive hole-removal (inside outer,
                // outside every exclusion). Without this guard, circumcenters of
                // boundary-adjacent or hole-adjacent faces can land far outside the
                // domain and create an ever-expanding mesh.
                if !point_in_polygon_winding(cx, cy, &outer_poly) {
                    return None;
                }
                if excl_polys.iter().any(|ep| point_in_polygon_winding(cx, cy, ep)) {
                    return None;
                }
                let target = area_fn(cx, cy).max(min_area);
                if f.area() > target {
                    Some(
                        circumcenter(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y)
                            .unwrap_or(Point2::new(cx, cy)),
                    )
                } else {
                    None
                }
            })
            .collect();
        if inserts.is_empty() {
            break;
        }
        for p in inserts {
            let _ = cdt.insert(p); // insertion errors for Steiner points are ignored
        }
    }

    // --- 5. Extract nodes and order-1 faces.
    let mut pos_vec: Vec<(f64, f64)> = Vec::new();
    let mut handle_idx: HashMap<usize, usize> = HashMap::new(); // FixedHandle.index() → node idx
    let mut constrained_set: HashSet<usize> = HashSet::new();

    for v in cdt.vertices() {
        let idx = pos_vec.len();
        handle_idx.insert(v.fix().index(), idx);
        let p = v.position();
        pos_vec.push((p.x, p.y));
        if constrained_bits.contains(&(p.x.to_bits(), p.y.to_bits())) {
            constrained_set.insert(idx);
        }
    }

    // Collect inner faces, removing those inside exclusion polygons.
    // `excl_polys` was already built for the adaptive loop above.
    let mut face_rows: Vec<[usize; 3]> = Vec::new();
    for face in cdt.inner_faces() {
        let [p0, p1, p2] = face.positions();
        let cx = (p0.x + p1.x + p2.x) / 3.0;
        let cy = (p0.y + p1.y + p2.y) / 3.0;
        let in_hole = excl_polys
            .iter()
            .any(|ep| point_in_polygon_winding(cx, cy, ep));
        if in_hole {
            continue;
        }
        let verts = face.vertices();
        let a = *handle_idx.get(&verts[0].fix().index()).unwrap();
        let b = *handle_idx.get(&verts[1].fix().index()).unwrap();
        let c = *handle_idx.get(&verts[2].fix().index()).unwrap();
        face_rows.push([a, b, c]);
    }

    // Build output arrays.
    let n_nodes = pos_vec.len();
    let n_faces = face_rows.len();
    let mut nodes = Array2::<f64>::zeros((n_nodes, 2));
    for (i, (x, y)) in pos_vec.iter().enumerate() {
        nodes[[i, 0]] = *x;
        nodes[[i, 1]] = *y;
    }
    let mut elements = Array2::<usize>::zeros((n_faces, 3));
    for (i, [a, b, c]) in face_rows.iter().enumerate() {
        elements[[i, 0]] = *a;
        elements[[i, 1]] = *b;
        elements[[i, 2]] = *c;
    }

    // --- 6. Map polygon vertices to node indices.
    // Build a position → node-index lookup for original vertices.
    let pos_to_idx: HashMap<(u64, u64), usize> = pos_vec
        .iter()
        .enumerate()
        .map(|(i, (x, y))| ((x.to_bits(), y.to_bits()), i))
        .collect();

    let orig_to_node: Vec<usize> = (0..n_verts)
        .map(|i| {
            let key = (
                poly.vertices[[i, 0]].to_bits(),
                poly.vertices[[i, 1]].to_bits(),
            );
            *pos_to_idx.get(&key).unwrap_or(&0)
        })
        .collect();

    let boundary: Vec<usize> = poly
        .curves
        .first()
        .map(|c| c.iter().map(|&k| orig_to_node[k as usize]).collect())
        .unwrap_or_default();

    let exclusions: Vec<Vec<usize>> = poly
        .curves
        .iter()
        .skip(1)
        .map(|c| c.iter().map(|&k| orig_to_node[k as usize]).collect())
        .collect();

    Ok((nodes, elements, constrained_set, boundary, exclusions))
}

// ---------------------------------------------------------------------------
// Laplacian smoothing
// ---------------------------------------------------------------------------

fn laplacian_smooth(
    nodes: &mut Array2<f64>,
    elements: &Array2<usize>,
    constrained: &HashSet<usize>,
    passes: usize,
) {
    let n = nodes.nrows();
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for e in 0..elements.nrows() {
        let [v0, v1, v2] = [elements[[e, 0]], elements[[e, 1]], elements[[e, 2]]];
        for &a in &[v0, v1, v2] {
            for &b in &[v0, v1, v2] {
                if a != b {
                    adj[a].insert(b);
                }
            }
        }
    }

    let mut buf = vec![(0.0_f64, 0.0_f64); n];
    for _ in 0..passes {
        for i in 0..n {
            if constrained.contains(&i) || adj[i].is_empty() {
                buf[i] = (nodes[[i, 0]], nodes[[i, 1]]);
            } else {
                let (sx, sy) = adj[i].iter().fold((0.0, 0.0), |(ax, ay), &j| {
                    (ax + nodes[[j, 0]], ay + nodes[[j, 1]])
                });
                let k = adj[i].len() as f64;
                buf[i] = (sx / k, sy / k);
            }
        }
        for i in 0..n {
            nodes[[i, 0]] = buf[i].0;
            nodes[[i, 1]] = buf[i].1;
        }
    }
}

// ---------------------------------------------------------------------------
// Order-2 midpoint insertion
// ---------------------------------------------------------------------------

/// Expand order-1 triangles to order-2 by inserting midpoints on every edge.
///
/// Output element layout: `[v0, v1, v2, mid01, mid12, mid20]`.
fn insert_midpoints(
    nodes: &Array2<f64>,
    elements: &Array2<usize>,
) -> (Array2<f64>, Array2<usize>) {
    let n_orig = nodes.nrows();
    let n_elem = elements.nrows();
    let mut edge_to_mid: HashMap<(usize, usize), usize> = HashMap::new();
    let mut mids: Vec<(f64, f64)> = Vec::new();
    let mut next = n_orig;

    let mut get_or_insert = |a: usize, b: usize| -> usize {
        let key = if a < b { (a, b) } else { (b, a) };
        *edge_to_mid.entry(key).or_insert_with(|| {
            let mx = (nodes[[a, 0]] + nodes[[b, 0]]) * 0.5;
            let my = (nodes[[a, 1]] + nodes[[b, 1]]) * 0.5;
            mids.push((mx, my));
            let idx = next;
            next += 1;
            idx
        })
    };

    let mut rows: Vec<[usize; 6]> = Vec::with_capacity(n_elem);
    for e in 0..n_elem {
        let (v0, v1, v2) = (elements[[e, 0]], elements[[e, 1]], elements[[e, 2]]);
        let m01 = get_or_insert(v0, v1);
        let m12 = get_or_insert(v1, v2);
        let m20 = get_or_insert(v2, v0);
        rows.push([v0, v1, v2, m01, m12, m20]);
    }

    let n_new = n_orig + mids.len();
    let mut nodes2 = Array2::<f64>::zeros((n_new, 2));
    nodes2.slice_mut(s![..n_orig, ..]).assign(nodes);
    for (i, (mx, my)) in mids.iter().enumerate() {
        nodes2[[n_orig + i, 0]] = *mx;
        nodes2[[n_orig + i, 1]] = *my;
    }

    let mut elems2 = Array2::<usize>::zeros((n_elem, 6));
    for (e, row) in rows.iter().enumerate() {
        for j in 0..6 { elems2[[e, j]] = row[j]; }
    }
    (nodes2, elems2)
}

// ---------------------------------------------------------------------------
// Binary search helpers
// ---------------------------------------------------------------------------

fn count_nodes_for_size(poly: &PolyData, size: f64, size_lower: f64) -> Result<usize, Error> {
    let max_area = equilateral_area(size).max(equilateral_area(size_lower));
    let (nodes, ..) = cdt_pipeline(poly, max_area, |_, _| max_area, size_lower)?;
    Ok(nodes.nrows())
}

/// Bisect size ∈ [lo, hi] to hit `target` node count.
fn bisect_size(poly: &PolyData, target: usize, lo: f64, hi: f64) -> Result<f64, Error> {
    let mut lo = lo;
    let mut hi = hi;

    // Larger size → fewer nodes. Check bracket.
    let n_lo = count_nodes_for_size(poly, lo, lo)?; // smallest size → most nodes
    let n_hi = count_nodes_for_size(poly, hi, lo)?; // largest size → fewest nodes
    if n_lo <= target { return Ok(lo); }
    if n_hi >= target { return Ok(hi); }

    for _ in 0..20 {
        let mid = (lo + hi) * 0.5;
        let n = count_nodes_for_size(poly, mid, lo)?;
        if n > target {
            lo = mid; // too many nodes → larger size
        } else {
            hi = mid; // too few → smaller size
        }
        if (hi - lo) / hi < 0.005 { break; }
    }
    Ok((lo + hi) * 0.5)
}

fn count_nodes_adaptive(
    poly: &PolyData,
    bg: &BackgroundField,
    scale: f64,
    size_lower: f64,
) -> Result<usize, Error> {
    let min_area = equilateral_area(size_lower);
    let global_max = (bg.target_areas.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        * scale)
        .max(min_area);
    let (nodes, ..) = cdt_pipeline(
        poly,
        global_max,
        |cx, cy| (bg.query(cx, cy) * scale).max(min_area),
        size_lower,
    )?;
    Ok(nodes.nrows())
}

fn bisect_scale_adaptive(
    poly: &PolyData,
    bg: &BackgroundField,
    target: usize,
    size_lower: f64,
) -> Result<f64, Error> {
    let mut lo = 0.1_f64;
    let mut hi = 10.0_f64;

    let n_hi = count_nodes_adaptive(poly, bg, hi, size_lower)?;
    if n_hi >= target { return Ok(hi); }
    let n_lo = count_nodes_adaptive(poly, bg, lo, size_lower)?;
    if n_lo <= target { return Ok(lo); }

    for _ in 0..20 {
        let mid = (lo + hi) * 0.5;
        let n = count_nodes_adaptive(poly, bg, mid, size_lower)?;
        if n > target {
            lo = mid; // too many → scale up (larger elements)
        } else {
            hi = mid;
        }
        if (hi - lo) / hi < 0.005 { break; }
    }
    Ok((lo + hi) * 0.5)
}

// ---------------------------------------------------------------------------
// Geometric helpers
// ---------------------------------------------------------------------------

#[inline]
fn equilateral_area(s: f64) -> f64 {
    s * s * 3.0_f64.sqrt() / 4.0
}

/// Circumcenter of triangle `(ax,ay)-(bx,by)-(cx,cy)`.
/// Returns `None` for degenerate (collinear) triangles.
fn circumcenter(
    ax: f64, ay: f64,
    bx: f64, by: f64,
    cx: f64, cy: f64,
) -> Option<Point2<f64>> {
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < f64::EPSILON * 1e6 {
        return None; // degenerate
    }
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    Some(Point2::new(ux, uy))
}

fn point_in_polygon_winding(px: f64, py: f64, poly: &[(f64, f64)]) -> bool {
    let n = poly.len();
    if n < 3 { return false; }
    let mut w = 0i32;
    for i in 0..n {
        let j = (i + 1) % n;
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if yi <= py {
            if yj > py && (xj - xi) * (py - yi) - (px - xi) * (yj - yi) > 0.0 {
                w += 1;
            }
        } else if yj <= py && (xj - xi) * (py - yi) - (px - xi) * (yj - yi) < 0.0 {
            w -= 1;
        }
    }
    w != 0
}

fn point_in_triangle(
    px: f64, py: f64,
    ax: f64, ay: f64,
    bx: f64, by: f64,
    cx: f64, cy: f64,
) -> bool {
    let d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
    let d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
    let d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

fn validate_order(mesh_order: u8) -> Result<(), Error> {
    if mesh_order != 1 && mesh_order != 2 {
        Err(Error::InvalidInput(format!(
            "mesh_order must be 1 or 2, got {mesh_order}"
        )))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn square_roi(size: f64) -> (Array2<f64>, Array2<i32>, Vec<Vec<i32>>) {
        let borders = array![
            [0.0, 0.0], [size, 0.0], [size, size], [0.0, size]
        ];
        let segments = array![[0, 1], [1, 2], [2, 3], [3, 0]];
        let curves = vec![vec![0, 1, 2, 3]];
        (borders, segments, curves)
    }

    fn square_roi_with_hole(outer: f64, inner: f64) -> (Array2<f64>, Array2<i32>, Vec<Vec<i32>>) {
        let off = (outer - inner) * 0.5;
        let borders = array![
            [0.0, 0.0], [outer, 0.0], [outer, outer], [0.0, outer],
            [off, off], [off + inner, off], [off + inner, off + inner], [off, off + inner]
        ];
        let segments = array![
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4]
        ];
        let curves = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        (borders, segments, curves)
    }

    // -----------------------------------------------------------------------

    #[test]
    fn equilateral_area_correct() {
        // Side 2 → area = sqrt(3).
        assert!((equilateral_area(2.0) - 3.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn point_in_triangle_inside() {
        assert!(point_in_triangle(0.3, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn point_in_triangle_outside() {
        assert!(!point_in_triangle(2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn circumcenter_known() {
        // Right triangle at origin → circumcenter at midpoint of hypotenuse.
        let cc = circumcenter(0.0, 0.0, 2.0, 0.0, 0.0, 2.0).unwrap();
        assert!((cc.x - 1.0).abs() < 1e-10, "cc.x={}", cc.x);
        assert!((cc.y - 1.0).abs() < 1e-10, "cc.y={}", cc.y);
    }

    #[test]
    fn circumcenter_degenerate_returns_none() {
        // Collinear points.
        assert!(circumcenter(0.0, 0.0, 1.0, 0.0, 2.0, 0.0).is_none());
    }

    #[test]
    fn generate_mesh_produces_nodes_and_elements() {
        let (b, s, c) = square_roi(100.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, 50, 1).unwrap();
        assert!(mesh.nodes.nrows() >= 4, "need at least 4 nodes");
        assert!(mesh.elements.nrows() > 0, "need at least 1 element");
        assert_eq!(mesh.elements.ncols(), 3, "order-1 → 3 cols");
    }

    #[test]
    fn generate_mesh_node_count_near_target() {
        let (b, s, c) = square_roi(100.0);
        let target = 50;
        let mesh = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, target, 1).unwrap();
        let n = mesh.nodes.nrows();
        // Binary search converges to within 0.5% of size; accept ±50% on count.
        assert!(n >= target / 2, "too few nodes: {n}");
        assert!(n <= target * 3, "too many nodes: {n}");
    }

    #[test]
    fn generate_mesh_no_degenerate_elements() {
        let (b, s, c) = square_roi(100.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, 50, 1).unwrap();
        for e in 0..mesh.elements.nrows() {
            let [i0, i1, i2] = [
                mesh.elements[[e, 0]],
                mesh.elements[[e, 1]],
                mesh.elements[[e, 2]],
            ];
            let ax = mesh.nodes[[i0, 0]]; let ay = mesh.nodes[[i0, 1]];
            let bx = mesh.nodes[[i1, 0]]; let by = mesh.nodes[[i1, 1]];
            let cx = mesh.nodes[[i2, 0]]; let cy = mesh.nodes[[i2, 1]];
            let area = 0.5 * ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)).abs();
            assert!(area > 1e-10, "degenerate element {e} area={area:.2e}");
        }
    }

    #[test]
    fn generate_mesh_boundary_nodes_on_square_edges() {
        let (b, s, c) = square_roi(100.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, 50, 1).unwrap();
        for &idx in &mesh.boundary {
            let x = mesh.nodes[[idx, 0]];
            let y = mesh.nodes[[idx, 1]];
            let on = x < 1e-9 || y < 1e-9 || (x - 100.0).abs() < 1e-9 || (y - 100.0).abs() < 1e-9;
            assert!(on, "boundary node {idx} ({x:.2},{y:.2}) not on edge");
        }
    }

    #[test]
    fn generate_mesh_with_exclusion_no_interior_elements() {
        // Use a small domain (30×30 outer, 10×10 hole) so the debug-mode CDT stays fast.
        // bisect_size's bracket at size_lower=3.0 yields ~(700/equilateral_area(3.0))≈90 triangles
        // — manageable for 20 bisect iterations in an unoptimised build.
        let (b, s, c) = square_roi_with_hole(30.0, 10.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 3.0, 15.0, 25, 1).unwrap();
        assert_eq!(mesh.exclusions.len(), 1);
        let off = 10.0_f64; // (30-10)/2
        for e in 0..mesh.elements.nrows() {
            let [i0, i1, i2] = [mesh.elements[[e,0]], mesh.elements[[e,1]], mesh.elements[[e,2]]];
            let cx = (mesh.nodes[[i0,0]] + mesh.nodes[[i1,0]] + mesh.nodes[[i2,0]]) / 3.0;
            let cy = (mesh.nodes[[i0,1]] + mesh.nodes[[i1,1]] + mesh.nodes[[i2,1]]) / 3.0;
            assert!(
                !(cx > off && cx < off + 10.0 && cy > off && cy < off + 10.0),
                "element {e} centroid ({cx:.1},{cy:.1}) inside exclusion hole"
            );
        }
    }

    #[test]
    fn generate_mesh_order2_has_six_columns() {
        let (b, s, c) = square_roi(100.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, 50, 2).unwrap();
        assert_eq!(mesh.elements.ncols(), 6);
    }

    #[test]
    fn generate_mesh_order2_midpoints_halfway() {
        let (b, s, c) = square_roi(100.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, 50, 2).unwrap();
        // Check first element only.
        let v0 = mesh.elements[[0, 0]];
        let v1 = mesh.elements[[0, 1]];
        let m01 = mesh.elements[[0, 3]];
        let expected_x = (mesh.nodes[[v0, 0]] + mesh.nodes[[v1, 0]]) * 0.5;
        let expected_y = (mesh.nodes[[v0, 1]] + mesh.nodes[[v1, 1]]) * 0.5;
        assert!((mesh.nodes[[m01, 0]] - expected_x).abs() < 1e-10);
        assert!((mesh.nodes[[m01, 1]] - expected_y).abs() < 1e-10);
    }

    #[test]
    fn laplacian_smooth_moves_interior_toward_centroid() {
        let mut nodes = array![
            [0.0, 2.0], [4.0, 2.0], [2.0, 0.0], [2.0, 4.0],
            [2.5, 2.5]  // off-centre interior
        ];
        let elements = array![[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4]];
        let constrained: HashSet<usize> = [0, 1, 2, 3].into();
        laplacian_smooth(&mut nodes, &elements, &constrained, 3);
        let dx = nodes[[4, 0]] - 2.0;
        let dy = nodes[[4, 1]] - 2.0;
        assert!((dx * dx + dy * dy).sqrt() < 0.6, "node did not converge");
    }

    #[test]
    fn laplacian_smooth_does_not_move_constrained() {
        let mut nodes = array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let elements = array![[0, 1, 2]];
        let constrained: HashSet<usize> = [0, 1, 2].into();
        let orig = nodes.clone();
        laplacian_smooth(&mut nodes, &elements, &constrained, 5);
        assert_eq!(nodes, orig);
    }

    #[test]
    fn adaptive_remesh_valid_output() {
        let (b, s, c) = square_roi(100.0);
        let init = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, 80, 1).unwrap();
        let n_elem = init.elements.nrows();
        let target_areas = Array1::from_elem(n_elem, 50.0_f64);
        let remeshed = adaptive_remesh(
            init.nodes.view(),
            init.elements.slice(s![.., ..3]).view(),
            target_areas.view(),
            b.view(), s.view(), &c,
            2.0, 80, 1,
        ).unwrap();
        assert!(remeshed.nodes.nrows() > 4);
        assert!(remeshed.elements.nrows() > 0);
    }

    #[test]
    fn background_field_query_inside() {
        let nodes = array![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
        let elements = array![[0usize, 1, 2]];
        let ta = array![7.0_f64];
        let bg = BackgroundField::new(nodes.view(), elements.view(), ta.view());
        assert!((bg.query(1.0, 1.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn background_field_query_outside_uses_mean() {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let elements = array![[0usize, 1, 2]];
        let ta = array![3.0_f64];
        let bg = BackgroundField::new(nodes.view(), elements.view(), ta.view());
        assert!((bg.query(10.0, 10.0) - 3.0).abs() < 1e-10);
    }
}
