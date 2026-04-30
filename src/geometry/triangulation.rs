use std::collections::{HashMap, HashSet};

use ndarray::{Array2, ArrayView2};
use spade::{
    AngleLimit, ConstrainedDelaunayTriangulation, InsertionError, Point2,
    RefinementParameters, Triangulation,
};

use crate::Error;

type Cdt = ConstrainedDelaunayTriangulation<Point2<f64>>;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

pub struct TriMesh {
    /// Node coordinates, shape `(N, 2)`.
    pub nodes: Array2<f64>,
    /// Element connectivity — order-1: `(M, 3)`; order-2: `(M, 6)`.
    pub elements: Array2<usize>,
    /// Ordered boundary node indices.
    pub boundary: Vec<usize>,
    /// Ordered exclusion node indices, one `Vec` per exclusion polygon.
    pub exclusions: Vec<Vec<usize>>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate a uniform mesh inside a polygonal region with optional holes.
///
/// # Arguments
/// * `borders`      — `(N, 2)` all polygon vertex coordinates.
/// * `segments`     — `(S, 2)` segment connectivity `[start, end]` (0-based indices into `borders`).
/// * `curves`       — One `Vec<i32>` per closed polygon (boundary first, then exclusions).
/// * `size_lower`   — Minimum element characteristic length.
/// * `size_upper`   — Maximum element characteristic length.
/// * `target_nodes` — Desired node count.
/// * `mesh_order`   — 1 (linear) or 2 (quadratic with midpoints).
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
    build_trimesh(&poly, max_area, size_lower, mesh_order)
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
// Core builder
// ---------------------------------------------------------------------------

fn build_trimesh(
    poly: &PolyData,
    max_area: f64,
    size_lower: f64,
    mesh_order: u8,
) -> Result<TriMesh, Error> {
    let (mut nodes, mut elems, constrained, boundary, exclusions) =
        cdt_pipeline(poly, max_area, size_lower)?;

    let outer_poly: Vec<(f64, f64)> = (0..poly.n_boundary)
        .map(|i| (poly.vertices[[i, 0]], poly.vertices[[i, 1]]))
        .collect();
    let excl_polys: Vec<Vec<(f64, f64)>> = {
        let mut offset = poly.n_boundary;
        let mut result = Vec::new();
        for &n in &poly.n_per_exclusion {
            let verts = (offset..offset + n)
                .map(|i| (poly.vertices[[i, 0]], poly.vertices[[i, 1]]))
                .collect();
            result.push(verts);
            offset += n;
        }
        result
    };

    laplacian_smooth(&mut nodes, &elems, &constrained, 3, &outer_poly, &excl_polys);

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
    size_lower: f64,
) -> Result<(Array2<f64>, Array2<usize>, HashSet<usize>, Vec<usize>, Vec<Vec<usize>>), Error> {
    let n_verts = poly.vertices.nrows();

    // --- 1. Insert polygon vertices.
    let mut cdt = Cdt::new();
    let mut handles = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        let x = poly.vertices[[i, 0]];
        let y = poly.vertices[[i, 1]];
        let h = cdt.insert(Point2::new(x, y)).map_err(|e: InsertionError| {
            Error::MeshGeneration(format!("vertex insertion failed: {e:?}"))
        })?;
        handles.push(h);
    }

    // --- 2. Insert constraint edges.
    for s in 0..poly.segments.nrows() {
        let ia = poly.segments[[s, 0]] as usize;
        let ib = poly.segments[[s, 1]] as usize;
        cdt.add_constraint(handles[ia], handles[ib]);
    }

    // --- 3. Ruppert refinement.
    // exclude_outer_faces(true) prevents refinement (and Steiner insertion) outside
    // the constraint polygon boundary, so no post-refinement cleanup is needed.
    let global_max_area = global_max_area.max(equilateral_area(size_lower));
    cdt.refine(
        RefinementParameters::new()
            .with_angle_limit(AngleLimit::from_deg(25.0))
            .with_max_allowed_area(global_max_area)
            .exclude_outer_faces(true)
            .with_max_additional_vertices(1_000_000),
    );

    // --- 4. Enumerate post-refinement vertices.
    let mut handle_idx: HashMap<usize, usize> = HashMap::new();
    let mut pos_vec: Vec<(f64, f64)> = Vec::new();
    for v in cdt.vertices() {
        let idx = pos_vec.len();
        handle_idx.insert(v.fix().index(), idx);
        let p = v.position();
        pos_vec.push((p.x, p.y));
    }

    // --- 5. Build domain polygons for face classification.
    let outer_poly: Vec<(f64, f64)> = (0..poly.n_boundary)
        .map(|i| (poly.vertices[[i, 0]], poly.vertices[[i, 1]]))
        .collect();
    let excl_polys: Vec<Vec<(f64, f64)>> = {
        let mut offset = poly.n_boundary;
        let mut result = Vec::new();
        for &n in &poly.n_per_exclusion {
            let verts = (offset..offset + n)
                .map(|i| (poly.vertices[[i, 0]], poly.vertices[[i, 1]]))
                .collect();
            result.push(verts);
            offset += n;
        }
        result
    };

    // --- 6. Direct centroid test — mark excluded faces.
    // In a CDT with constraint edges, no face straddles a constraint boundary,
    // so a face whose centroid is outside the domain has all vertices outside.
    // This guarantees that excluded faces orphan all their outside-domain nodes
    // after compaction — no second pass required.
    let mut excluded: HashSet<_> = HashSet::new();
    for face in cdt.inner_faces() {
        let [p0, p1, p2] = face.positions();
        let cx = (p0.x + p1.x + p2.x) / 3.0;
        let cy = (p0.y + p1.y + p2.y) / 3.0;
        if !point_in_polygon_winding(cx, cy, &outer_poly)
            || excl_polys.iter().any(|ep| point_in_polygon_winding(cx, cy, ep))
        {
            excluded.insert(face.fix());
        }
    }

    // --- 7. Collect non-excluded faces.
    let mut face_rows: Vec<[usize; 3]> = Vec::new();
    for face in cdt.inner_faces() {
        if excluded.contains(&face.fix()) {
            continue;
        }
        let verts = face.vertices();
        let a = handle_idx[&verts[0].fix().index()];
        let b = handle_idx[&verts[1].fix().index()];
        let c = handle_idx[&verts[2].fix().index()];
        face_rows.push([a, b, c]);
    }

    // --- 8. Compact orphaned nodes.
    let used: HashSet<usize> = face_rows.iter().flatten().copied().collect();
    let mut old_to_new = vec![usize::MAX; pos_vec.len()];
    let mut new_pos: Vec<(f64, f64)> = Vec::new();
    for (old, &pos) in pos_vec.iter().enumerate() {
        if used.contains(&old) {
            old_to_new[old] = new_pos.len();
            new_pos.push(pos);
        }
    }
    for row in &mut face_rows {
        for v in row.iter_mut() {
            *v = old_to_new[*v];
        }
    }
    let pos_vec = new_pos;

    // --- 9. Constrained node detection.
    // A node is constrained if it is an original polygon vertex (bit-exact match)
    // or a Steiner point inserted on a constraint edge by Ruppert refinement.
    // Constrained nodes are pinned during Laplacian smoothing.
    let poly_coords: HashSet<(u64, u64)> = (0..n_verts)
        .map(|i| {
            (
                poly.vertices[[i, 0]].to_bits(),
                poly.vertices[[i, 1]].to_bits(),
            )
        })
        .collect();
    let constrained_set: HashSet<usize> = pos_vec
        .iter()
        .enumerate()
        .filter(|&(_, &(px, py))| {
            poly_coords.contains(&(px.to_bits(), py.to_bits()))
                || (0..poly.segments.nrows()).any(|s| {
                    let ia = poly.segments[[s, 0]] as usize;
                    let ib = poly.segments[[s, 1]] as usize;
                    let (ax, ay) = (poly.vertices[[ia, 0]], poly.vertices[[ia, 1]]);
                    let (bx, by) = (poly.vertices[[ib, 0]], poly.vertices[[ib, 1]]);
                    on_segment(px, py, ax, ay, bx, by)
                })
        })
        .map(|(i, _)| i)
        .collect();

    // --- 10. Build output arrays.
    let n_nodes = pos_vec.len();
    let n_faces = face_rows.len();
    let mut nodes = Array2::<f64>::zeros((n_nodes, 2));
    for (i, &(x, y)) in pos_vec.iter().enumerate() {
        nodes[[i, 0]] = x;
        nodes[[i, 1]] = y;
    }
    let mut elements = Array2::<usize>::zeros((n_faces, 3));
    for (i, &[a, b, c]) in face_rows.iter().enumerate() {
        elements[[i, 0]] = a;
        elements[[i, 1]] = b;
        elements[[i, 2]] = c;
    }

    // --- 11. Map original polygon vertices to compacted node indices.
    let pos_to_idx: HashMap<(u64, u64), usize> = pos_vec
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| ((x.to_bits(), y.to_bits()), i))
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
// Binary search for target node count
// ---------------------------------------------------------------------------

fn count_nodes_for_size(poly: &PolyData, size: f64, size_lower: f64) -> Result<usize, Error> {
    let max_area = equilateral_area(size).max(equilateral_area(size_lower));
    let (nodes, ..) = cdt_pipeline(poly, max_area, size_lower)?;
    Ok(nodes.nrows())
}

/// Bisect `size ∈ [lo, hi]` to hit `target` node count.
/// Larger size → coarser mesh → fewer nodes.
fn bisect_size(poly: &PolyData, target: usize, lo: f64, hi: f64) -> Result<f64, Error> {
    let mut lo = lo;
    let mut hi = hi;

    let n_lo = count_nodes_for_size(poly, lo, lo)?;
    let n_hi = count_nodes_for_size(poly, hi, lo)?;
    if n_lo <= target {
        return Ok(lo);
    }
    if n_hi >= target {
        return Ok(hi);
    }

    for _ in 0..20 {
        let mid = (lo + hi) * 0.5;
        let n = count_nodes_for_size(poly, mid, lo)?;
        if n > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) / hi < 0.005 {
            break;
        }
    }
    Ok((lo + hi) * 0.5)
}

// ---------------------------------------------------------------------------
// Laplacian smoothing
// ---------------------------------------------------------------------------

fn laplacian_smooth(
    nodes: &mut Array2<f64>,
    elements: &Array2<usize>,
    constrained: &HashSet<usize>,
    passes: usize,
    outer_poly: &[(f64, f64)],
    excl_polys: &[Vec<(f64, f64)>],
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

    let has_domain = !outer_poly.is_empty();
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
                let (nx, ny) = (sx / k, sy / k);
                let in_domain = !has_domain
                    || (point_in_polygon_winding(nx, ny, outer_poly)
                        && !excl_polys.iter().any(|ep| point_in_polygon_winding(nx, ny, ep)));
                buf[i] = if in_domain {
                    (nx, ny)
                } else {
                    (nodes[[i, 0]], nodes[[i, 1]])
                };
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

/// Expand order-1 triangles to order-2 by inserting edge midpoints.
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
    nodes2.slice_mut(ndarray::s![..n_orig, ..]).assign(nodes);
    for (i, &(mx, my)) in mids.iter().enumerate() {
        nodes2[[n_orig + i, 0]] = mx;
        nodes2[[n_orig + i, 1]] = my;
    }

    let mut elems2 = Array2::<usize>::zeros((n_elem, 6));
    for (e, row) in rows.iter().enumerate() {
        for j in 0..6 {
            elems2[[e, j]] = row[j];
        }
    }
    (nodes2, elems2)
}

// ---------------------------------------------------------------------------
// Geometric helpers
// ---------------------------------------------------------------------------

#[inline]
fn equilateral_area(s: f64) -> f64 {
    s * s * 3.0_f64.sqrt() / 4.0
}

/// Returns `true` if `(px, py)` lies on segment `(ax, ay)–(bx, by)`.
fn on_segment(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
    let len = (bx - ax).hypot(by - ay);
    if len < 1e-12 {
        return false;
    }
    let cross = (bx - ax) * (py - ay) - (px - ax) * (by - ay);
    if cross.abs() > 1e-9 * len {
        return false;
    }
    let dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay);
    dot >= -1e-9 * len && dot <= len * len + 1e-9 * len
}

fn point_in_polygon_winding(px: f64, py: f64, poly: &[(f64, f64)]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
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
        assert!((equilateral_area(2.0) - 3.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn generate_mesh_produces_nodes_and_elements() {
        let (b, s, c) = square_roi(100.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, 50, 1).unwrap();
        assert!(mesh.nodes.nrows() >= 4);
        assert!(mesh.elements.nrows() > 0);
        assert_eq!(mesh.elements.ncols(), 3);
    }

    #[test]
    fn generate_mesh_node_count_near_target() {
        let (b, s, c) = square_roi(100.0);
        let target = 50;
        let mesh = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, target, 1).unwrap();
        let n = mesh.nodes.nrows();
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
            assert!(area > 1e-10, "degenerate element {e}: area={area:.2e}");
        }
    }

    #[test]
    fn generate_mesh_boundary_nodes_on_square_edges() {
        let (b, s, c) = square_roi(100.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, 50, 1).unwrap();
        for &idx in &mesh.boundary {
            let x = mesh.nodes[[idx, 0]];
            let y = mesh.nodes[[idx, 1]];
            let on = x < 1e-9
                || y < 1e-9
                || (x - 100.0).abs() < 1e-9
                || (y - 100.0).abs() < 1e-9;
            assert!(on, "boundary node {idx} ({x:.2},{y:.2}) not on edge");
        }
    }

    #[test]
    fn generate_mesh_with_exclusion_no_interior_elements() {
        let (b, s, c) = square_roi_with_hole(30.0, 10.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 3.0, 15.0, 25, 1).unwrap();
        assert_eq!(mesh.exclusions.len(), 1);
        let off = 10.0_f64;
        for e in 0..mesh.elements.nrows() {
            let [i0, i1, i2] = [mesh.elements[[e, 0]], mesh.elements[[e, 1]], mesh.elements[[e, 2]]];
            let cx = (mesh.nodes[[i0, 0]] + mesh.nodes[[i1, 0]] + mesh.nodes[[i2, 0]]) / 3.0;
            let cy = (mesh.nodes[[i0, 1]] + mesh.nodes[[i1, 1]] + mesh.nodes[[i2, 1]]) / 3.0;
            assert!(
                !(cx > off && cx < off + 10.0 && cy > off && cy < off + 10.0),
                "element {e} centroid ({cx:.1},{cy:.1}) inside exclusion"
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
        let v0 = mesh.elements[[0, 0]];
        let v1 = mesh.elements[[0, 1]];
        let m01 = mesh.elements[[0, 3]];
        let expected_x = (mesh.nodes[[v0, 0]] + mesh.nodes[[v1, 0]]) * 0.5;
        let expected_y = (mesh.nodes[[v0, 1]] + mesh.nodes[[v1, 1]]) * 0.5;
        assert!((mesh.nodes[[m01, 0]] - expected_x).abs() < 1e-10);
        assert!((mesh.nodes[[m01, 1]] - expected_y).abs() < 1e-10);
    }

    #[test]
    fn generate_mesh_all_faces_inside_boundary() {
        let (b, s, c) = square_roi_with_hole(30.0, 10.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 3.0, 15.0, 30, 1).unwrap();
        let outer: Vec<(f64, f64)> = vec![(0., 0.), (30., 0.), (30., 30.), (0., 30.)];
        for i in 0..mesh.nodes.nrows() {
            let x = mesh.nodes[[i, 0]];
            let y = mesh.nodes[[i, 1]];
            let on_boundary = x < 1e-6
                || y < 1e-6
                || (x - 30.0).abs() < 1e-6
                || (y - 30.0).abs() < 1e-6;
            assert!(
                point_in_polygon_winding(x, y, &outer) || on_boundary,
                "node {i} ({x:.4},{y:.4}) outside outer boundary"
            );
        }
    }

    #[test]
    fn generate_mesh_no_nodes_outside_boundary() {
        // Node-level check (centroid check is insufficient — this verifies individual nodes).
        let (b, s, c) = square_roi(100.0);
        let mesh = generate_mesh(b.view(), s.view(), &c, 2.0, 50.0, 80, 1).unwrap();
        for i in 0..mesh.nodes.nrows() {
            let x = mesh.nodes[[i, 0]];
            let y = mesh.nodes[[i, 1]];
            assert!(
                x >= -1e-9 && x <= 100.0 + 1e-9 && y >= -1e-9 && y <= 100.0 + 1e-9,
                "node {i} ({x:.4},{y:.4}) outside [0,100]²"
            );
        }
    }

    #[test]
    fn generate_mesh_original_boundary_vertices_preserved() {
        // All original polygon vertices must appear in the mesh (constrained — never dropped).
        let boundary = array![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
        let segments = array![[0, 1], [1, 2], [2, 3], [3, 0]];
        let curves = vec![vec![0, 1, 2, 3]];
        let mesh = generate_mesh(boundary.view(), segments.view(), &curves, 2.0, 50.0, 50, 1).unwrap();
        for v in boundary.outer_iter() {
            let (ox, oy) = (v[0], v[1]);
            assert!(
                mesh.nodes.outer_iter().any(|n| (n[0] - ox).abs() < 1e-12 && (n[1] - oy).abs() < 1e-12),
                "original vertex ({ox},{oy}) missing from mesh nodes"
            );
        }
    }

    #[test]
    fn laplacian_smooth_moves_interior_toward_centroid() {
        let mut nodes = array![
            [0.0, 2.0], [4.0, 2.0], [2.0, 0.0], [2.0, 4.0],
            [2.5, 2.5]
        ];
        let elements = array![[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4]];
        let constrained: HashSet<usize> = [0, 1, 2, 3].into();
        laplacian_smooth(&mut nodes, &elements, &constrained, 3, &[], &[]);
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
        laplacian_smooth(&mut nodes, &elements, &constrained, 5, &[], &[]);
        assert_eq!(nodes, orig);
    }
}
