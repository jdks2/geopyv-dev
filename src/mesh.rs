//! DIC mesh solver and mesh geometry.
//!
//! Translates `geopyv/src/geopyv/mesh.py` (`MeshBase` + `Mesh` classes).
//! gmsh is **not** used; mesh generation delegates to
//! [`crate::geometry::triangulation`].
//!
//! # Architecture
//!
//! - [`Mesh`] is constructed from pre-computed geometry (nodes/elements) plus
//!   the reference/target QCQT arrays.
//! - [`Mesh::solve`] runs the reliability-guided DIC algorithm, stores per-node
//!   results in [`SolveState`], computes element areas / strains, and checks
//!   compatibility.
//! - Pure-math helpers (`element_area`, `element_strains`, etc.) are free
//!   functions so they can be called by `adaptive_remesh` logic without a
//!   `Mesh` borrow.

use std::collections::HashSet;

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::{
    geometry::triangulation::{self, TriMesh},
    image::Image,
    subset::{Subset, SolveResult},
    Error,
};

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Per-node DIC solve output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResult {
    pub p: Vec<f64>,
    pub c_zncc: f64,
    pub u: f64,
    pub v: f64,
}

/// Output of [`Mesh::solve`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshSolution {
    /// Node coordinates `(N, 2)`.
    pub nodes: Array2<f64>,
    /// Element connectivity `(M, 3)` or `(M, 6)`.
    pub elements: Array2<usize>,
    /// Boundary node indices.
    pub boundary: Vec<usize>,
    /// Exclusion node index groups.
    pub exclusions: Vec<Vec<usize>>,
    /// Signed element areas `(M,)`.
    pub areas: Array1<f64>,
    /// Element warp vectors `(M, 12)`.
    pub warps: Array2<f64>,
    /// Per-node displacements `(N, 2)`.
    pub displacements: Array2<f64>,
    /// Per-node ZNCC scores `(N,)`.
    pub c_zncc: Array1<f64>,
    /// Per-node warp parameters `(N, 6 or 12)`.
    pub p: Array2<f64>,
    /// Index of the seed node.
    pub seed_node: usize,
    pub mesh_order: u8,
    pub subset_order: u8,
    /// Per-node iteration counts `(N,)`.
    pub iterations: Array1<u32>,
    /// Per-node final ∆norm values `(N,)`.
    pub norms: Array1<f64>,
}

// ---------------------------------------------------------------------------
// Solve configuration
// ---------------------------------------------------------------------------

/// Parameters forwarded to each subset solver.
#[derive(Debug, Clone)]
pub struct SolveConfig {
    pub max_norm: f64,
    pub max_iterations: usize,
    pub subset_order: usize,
    pub tolerance: f64,
    pub method: SolveMethod,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            max_norm: 1e-5,
            max_iterations: 50,
            subset_order: 1,
            tolerance: 0.75,
            method: SolveMethod::Icgn,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveMethod {
    Icgn,
    Fagn,
}

// ---------------------------------------------------------------------------
// Mesh struct
// ---------------------------------------------------------------------------

/// Mesh-level DIC solver.
///
/// Construct via [`Mesh::from_trimesh`].
pub struct Mesh {
    nodes: Array2<f64>,
    elements: Array2<usize>,
    boundary: Vec<usize>,
    exclusions: Vec<Vec<usize>>,
    mesh_order: u8,
}

impl Mesh {
    /// Construct a `Mesh` from a [`TriMesh`] (output of
    /// [`triangulation::generate_mesh`]).
    pub fn from_trimesh(tm: TriMesh) -> Self {
        Mesh {
            nodes: tm.nodes,
            elements: tm.elements,
            boundary: tm.boundary,
            exclusions: tm.exclusions,
            mesh_order: 1, // TriMesh order-1 layout; order-2 is implicit via ncols==6
        }
    }

    /// Reconstruct a `Mesh` topology from a previous [`MeshSolution`].
    ///
    /// Used by the sequence solver in `sync` mode to reuse the node/element
    /// layout of the previous solved mesh rather than regenerating via CDT.
    pub fn from_solution(sol: &MeshSolution) -> Self {
        Mesh {
            nodes: sol.nodes.clone(),
            elements: sol.elements.clone(),
            boundary: sol.boundary.clone(),
            exclusions: sol.exclusions.clone(),
            mesh_order: sol.mesh_order,
        }
    }

    /// Create a mesh by running [`triangulation::generate_mesh`] and wrapping
    /// the result.
    pub fn generate(
        borders: ArrayView2<f64>,
        segments: ArrayView2<i32>,
        curves: &[Vec<i32>],
        size_lower: f64,
        size_upper: f64,
        target_nodes: usize,
        mesh_order: u8,
    ) -> Result<Self, Error> {
        let tm = triangulation::generate_mesh(
            borders,
            segments,
            curves,
            size_lower,
            size_upper,
            target_nodes,
            mesh_order,
        )?;
        let order = if tm.elements.ncols() == 6 { 2 } else { 1 };
        Ok(Mesh {
            nodes: tm.nodes,
            elements: tm.elements,
            boundary: tm.boundary,
            exclusions: tm.exclusions,
            mesh_order: order,
        })
    }

    /// Run the reliability-guided DIC solver.
    ///
    /// # Arguments
    /// * `f_img` — Reference [`Image`].
    /// * `g_img` — Target [`Image`].
    /// * `template_coords` — Subset template pixel coordinates (from
    ///   `templates::Circle` or `templates::Square`).
    /// * `seed_coord`  — Image coordinate near region of low deformation.
    /// * `seed_warp`   — Initial warp vector for the seed subset.
    /// * `cfg`         — Solver configuration.
    pub fn solve(
        &self,
        f_img: &Image,
        g_img: &Image,
        template_coords: &Array2<f64>,
        seed_coord: [f64; 2],
        seed_warp: &[f64],
        cfg: &SolveConfig,
    ) -> Result<MeshSolution, Error> {
        let n_nodes = self.nodes.nrows();
        let p_len = 6 * cfg.subset_order;

        // Normalise seed_warp to exactly p_len elements: truncate if too long,
        // zero-pad if too short.  Prevents solve_icgn from inferring the wrong
        // warp order when the caller passes a full 12-element warp with order-1.
        let seed_warp_norm: Vec<f64> = {
            let mut w = seed_warp.to_vec();
            w.resize(p_len, 0.0);
            w
        };

        let mut solved = vec![false; n_nodes]; // -1 (stored) ↔ true, 0 ↔ false
        let mut c_zncc = Array1::<f64>::zeros(n_nodes);
        let mut p = Array2::<f64>::zeros((n_nodes, p_len));
        let mut displacements = Array2::<f64>::zeros((n_nodes, 2));
        let mut iterations = Array1::<u32>::zeros(n_nodes);
        let mut norms = Array1::<f64>::zeros(n_nodes);

        // Build per-node subsets (no masking in pure-Rust path).
        let subsets: Vec<Subset> = (0..n_nodes)
            .map(|i| {
                let coord = [self.nodes[[i, 0]], self.nodes[[i, 1]]];
                Subset::new(coord, template_coords, &f_img.qcqt)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // --- Seed node.
        let seed_node = find_seed_node(&self.nodes, seed_coord);
        let seed_result = self.solve_one(
            &subsets[seed_node],
            g_img,
            &seed_warp_norm,
            cfg,
        )?;
        store_result(seed_node, &seed_result, &mut c_zncc, &mut p, &mut displacements, &mut iterations, &mut norms);
        solved[seed_node] = true;

        // --- Seed neighbours.
        let seed_p: Vec<f64> = p.row(seed_node).to_vec();
        self.solve_neighbours_from(
            seed_node,
            &seed_p,
            &subsets,
            g_img,
            cfg,
            &mut solved,
            &mut c_zncc,
            &mut p,
            &mut displacements,
            &mut iterations,
            &mut norms,
        )?;

        // --- Reliability-guided queue: highest-C_ZNCC first.
        // Priority queue of (solved, C_ZNCC×scale, node_idx) — max-heap.
        while let Some(cur_idx) = next_queue_node(&solved, c_zncc.view()) {
            solved[cur_idx] = true; // mark as propagated
            let p_0: Vec<f64> = p.row(cur_idx).to_vec();
            self.solve_neighbours_from(
                cur_idx,
                &p_0,
                &subsets,
                g_img,
                cfg,
                &mut solved,
                &mut c_zncc,
                &mut p,
                &mut displacements,
                &mut iterations,
                &mut norms,
            )?;
        }

        // --- Corrections (outlier re-solve).
        self.corrections(
            &subsets,
            g_img,
            cfg,
            &mut solved,
            &mut c_zncc,
            &mut p,
            &mut displacements,
            &mut iterations,
            &mut norms,
        )?;

        // --- Element areas and strains.
        let areas = element_area(&self.nodes, &self.elements);
        let warps = element_strains(
            &self.nodes,
            &self.elements,
            &displacements,
            self.mesh_order,
        )?;

        // --- Compatibility check.
        self.check_compatibility(&warps)?;

        Ok(MeshSolution {
            nodes: self.nodes.clone(),
            elements: self.elements.clone(),
            boundary: self.boundary.clone(),
            exclusions: self.exclusions.clone(),
            areas,
            warps,
            displacements,
            c_zncc,
            p,
            seed_node,
            mesh_order: self.mesh_order,
            subset_order: cfg.subset_order as u8,
            iterations,
            norms,
        })
    }

    // -----------------------------------------------------------------------
    // Internal solver helpers
    // -----------------------------------------------------------------------

    fn solve_one(
        &self,
        subset: &Subset,
        g_img: &Image,
        warp_0: &[f64],
        cfg: &SolveConfig,
    ) -> Result<SolveResult, Error> {
        match cfg.method {
            SolveMethod::Icgn => {
                subset.solve_icgn(&g_img.qcqt, warp_0, cfg.max_norm, cfg.max_iterations)
            }
            SolveMethod::Fagn => {
                subset.solve_fagn(&g_img.qcqt, warp_0, cfg.max_norm, cfg.max_iterations)
            }
        }
    }

    /// Solve all unsolved neighbours of `cur_idx`, applying warp extrapolation
    /// as pre-conditioning (three-attempt strategy: NN → projected → zero).
    fn solve_neighbours_from(
        &self,
        cur_idx: usize,
        p_0: &[f64],
        subsets: &[Subset],
        g_img: &Image,
        cfg: &SolveConfig,
        solved: &mut Vec<bool>,
        c_zncc: &mut Array1<f64>,
        p: &mut Array2<f64>,
        displacements: &mut Array2<f64>,
        iterations: &mut Array1<u32>,
        norms: &mut Array1<f64>,
    ) -> Result<(), Error> {
        let neighbours = connectivity(&self.elements, self.mesh_order, cur_idx, false);
        for &nb_idx in &neighbours {
            if solved[nb_idx] {
                continue;
            }
            // Attempt 1: use current p_0 as-is (nearest-neighbour preconditioning).
            let r1 = self.solve_one(&subsets[nb_idx], g_img, p_0, cfg)?;
            if r1.c_zncc >= cfg.tolerance {
                store_result(nb_idx, &r1, c_zncc, p, displacements, iterations, norms);
                solved[nb_idx] = true;
                continue;
            }
            // Attempt 2: projected preconditioning (Taylor expansion from cur_idx).
            let p_proj = project_warp(p_0, &self.nodes, cur_idx, nb_idx);
            let r2 = self.solve_one(&subsets[nb_idx], g_img, &p_proj, cfg)?;
            if r2.c_zncc >= cfg.tolerance {
                store_result(nb_idx, &r2, c_zncc, p, displacements, iterations, norms);
                solved[nb_idx] = true;
                continue;
            }
            // Attempt 3: zero initial guess.
            let zeros = vec![0.0f64; p_0.len()];
            let r3 = self.solve_one(&subsets[nb_idx], g_img, &zeros, cfg)?;
            store_result(nb_idx, &r3, c_zncc, p, displacements, iterations, norms);
            solved[nb_idx] = true;
        }
        Ok(())
    }

    /// Outlier correction: re-solve nodes flagged by `_corr`, `_flow`, `_R`.
    fn corrections(
        &self,
        subsets: &[Subset],
        g_img: &Image,
        cfg: &SolveConfig,
        solved: &mut Vec<bool>,
        c_zncc: &mut Array1<f64>,
        p: &mut Array2<f64>,
        displacements: &mut Array2<f64>,
        iterations: &mut Array1<u32>,
        norms: &mut Array1<f64>,
    ) -> Result<(), Error> {
        let corr_ids = corr(c_zncc.view());
        let (_, flow_ids, flow_lq, flow_iqr) =
            flow_stats(&self.elements, self.mesh_order, displacements.view());
        let (r_vals, r_ids) =
            r_stats(&self.elements, self.mesh_order, &self.nodes, displacements.view());

        // Merge and sort ascending by R (smallest R corrected first).
        let mut full_id: Vec<usize> = {
            let s: HashSet<usize> =
                corr_ids.iter().chain(flow_ids.iter()).chain(r_ids.iter()).copied().collect();
            s.into_iter().collect()
        };
        full_id.sort_by(|&a, &b| r_vals[a].partial_cmp(&r_vals[b]).unwrap());

        // Exempt nodes whose flow is within normal bounds and not in R outliers.
        let full_id_set: HashSet<usize> = full_id.iter().copied().collect();
        let exempt: HashSet<usize> = full_id
            .iter()
            .copied()
            .filter(|&j| {
                if r_ids.contains(&j) {
                    return false;
                }
                let f = flow_calc_excluding(
                    j,
                    &full_id_set,
                    &self.elements,
                    self.mesh_order,
                    displacements.view(),
                    None,
                );
                f > flow_lq - 2.5 * flow_iqr
            })
            .collect();

        let active_ids: Vec<usize> =
            full_id.into_iter().filter(|id| !exempt.contains(id)).collect();

        for i in 0..active_ids.len() {
            let j = active_ids[i];
            // Build collective preconditioning warp from good neighbours.
            let full_neighbours =
                connectivity(&self.elements, self.mesh_order, j, true);
            // Exclude current and later outliers from preconditioning.
            let later: HashSet<usize> = active_ids[i..].iter().copied().collect();
            let good_nb: Vec<usize> = full_neighbours
                .iter()
                .copied()
                .filter(|&nb| !later.contains(&nb) && c_zncc[nb] > cfg.tolerance)
                .collect();

            let warp: Vec<f64> = if !good_nb.is_empty() {
                let n = good_nb.len() as f64;
                let sum: Vec<f64> = (0..p.ncols())
                    .map(|k| good_nb.iter().map(|&nb| p[[nb, k]]).sum::<f64>() / n)
                    .collect();
                sum
            } else {
                // Fall back to single best neighbour by C_ZNCC.
                let nb_all = connectivity(&self.elements, self.mesh_order, j, true);
                let best = nb_all.iter().copied().max_by(|&a, &b| {
                    c_zncc[a].partial_cmp(&c_zncc[b]).unwrap()
                });
                match best {
                    Some(b) => p.row(b).to_vec(),
                    None => vec![0.0; p.ncols()],
                }
            };

            let result = self.solve_one(&subsets[j], g_img, &warp, cfg)?;
            // Force the warp regardless of convergence (matches Python behaviour).
            let forced = SolveResult {
                p: warp.clone(),
                c_zncc: result.c_zncc,
                c_znssd: result.c_znssd,
                iterations: result.iterations,
                converged: result.converged,
                history: result.history,
            };
            store_result(j, &forced, c_zncc, p, displacements, iterations, norms);
            solved[j] = true;
        }
        Ok(())
    }

    /// Compatibility check: reject mesh if any element has det(J) ≤ −1 (fold-over).
    fn check_compatibility(&self, warps: &Array2<f64>) -> Result<(), Error> {
        for e in 0..warps.nrows() {
            // du/dx = warps[e,2]; dv/dy = warps[e,5]
            if warps[[e, 2]] < -1.0 || warps[[e, 5]] < -1.0 {
                let cx = self.nodes.slice(s![.., 0]).mean().unwrap_or(0.0);
                let cy = self.nodes.slice(s![.., 1]).mean().unwrap_or(0.0);
                return Err(Error::InvalidInput(format!(
                    "mesh compatibility violated at element {e} near ({cx:.1},{cy:.1})"
                )));
            }
        }
        Ok(())
    }

    /// Adaptive remesh using element strain-area products.
    ///
    /// Mirrors `Mesh._adaptive_mesh`:
    ///   `D[e] = |du/dx + dv/dy| * |area[e]|`
    ///   `areas_new[e] = areas[e] * clip(D[e]/mean(D), α, 1/α)^-2`
    pub fn adaptive_target_areas(
        warps: &Array2<f64>,
        areas: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        adaptive_target_areas(warps, areas, alpha)
    }

    // -----------------------------------------------------------------------
    // Field accessors (used by the PyO3 wrapper)
    // -----------------------------------------------------------------------

    /// Node coordinates `(N, 2)`.
    pub fn nodes(&self) -> &Array2<f64> { &self.nodes }

    /// Element connectivity `(M, 3)` or `(M, 6)`.
    pub fn elements(&self) -> &Array2<usize> { &self.elements }

    /// Boundary node indices.
    pub fn boundary(&self) -> &[usize] { &self.boundary }

    /// Exclusion node index groups.
    pub fn exclusions(&self) -> &[Vec<usize>] { &self.exclusions }

    /// Mesh element order (1 or 2).
    pub fn mesh_order(&self) -> u8 { self.mesh_order }
}

// ---------------------------------------------------------------------------
// Free functions — pure math, no Image dependency.
// Used in unit tests and externally (e.g. sequence.rs).
// ---------------------------------------------------------------------------

/// Signed element areas: `0.5 * det(M)` where M has ones in col 0,
/// x-coords in col 1, y-coords in col 2 (rows = corner nodes).
pub fn element_area(nodes: &Array2<f64>, elements: &Array2<usize>) -> Array1<f64> {
    let n_elems = elements.nrows();
    let mut areas = Array1::<f64>::zeros(n_elems);
    for e in 0..n_elems {
        let (i0, i1, i2) = (elements[[e, 0]], elements[[e, 1]], elements[[e, 2]]);
        let (x0, y0) = (nodes[[i0, 0]], nodes[[i0, 1]]);
        let (x1, y1) = (nodes[[i1, 0]], nodes[[i1, 1]]);
        let (x2, y2) = (nodes[[i2, 0]], nodes[[i2, 1]]);
        // det([[1,x0,y0],[1,x1,y1],[1,x2,y2]])
        areas[e] = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    }
    areas
}

/// Shape functions at the element centroid for the given mesh order.
///
/// Returns `(N, dN, d2N)`:
/// - Order-1: `N` (3,), `dN` (2,3), `d2N = None`
/// - Order-2: `N` (6,), `dN` (2,6), `d2N = Some` (3,6)
pub fn shape_function(
    mesh_order: u8,
) -> (Vec<f64>, Vec<Vec<f64>>, Option<Vec<Vec<f64>>>) {
    match mesh_order {
        1 => {
            let n = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
            let dn = vec![
                vec![1.0, 0.0, -1.0],
                vec![0.0, 1.0, -1.0],
            ];
            (n, dn, None)
        }
        2 => {
            let n = vec![-1.0 / 9.0, -1.0 / 9.0, -1.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0];
            let dn = vec![
                vec![1.0/3.0, 0.0, -1.0/3.0, 4.0/3.0, -4.0/3.0, 0.0],
                vec![0.0, 1.0/3.0, -1.0/3.0, 4.0/3.0, 0.0, -4.0/3.0],
            ];
            let d2n = vec![
                vec![4.0, 0.0, 4.0, 0.0, 0.0, -8.0],
                vec![0.0, 0.0, 4.0, 4.0, -4.0, -4.0],
                vec![0.0, 4.0, 4.0, 0.0, -8.0, 0.0],
            ];
            (n, dn, Some(d2n))
        }
        _ => panic!("mesh_order must be 1 or 2"),
    }
}

/// Element centroid local coordinates. Returns `A` with shape `(n_elems, 3, 4)`:
/// row 0 = all ones; rows 1-2 = `[centroid, x-coords, y-coords]`.
pub fn local_coordinates(nodes: &Array2<f64>, elements: &Array2<usize>) -> Vec<[[f64; 4]; 3]> {
    let n_elems = elements.nrows();
    let mut a = vec![[[1.0f64; 4]; 3]; n_elems];
    for e in 0..n_elems {
        let n_verts = elements.ncols().min(3); // only first 3 (corner) nodes
        // Row 0 stays all ones.
        // Rows 1-2, col 0 = centroid x, y.
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mean_n = elements.ncols() as f64;
        for k in 0..elements.ncols() {
            let ni = elements[[e, k]];
            cx += nodes[[ni, 0]];
            cy += nodes[[ni, 1]];
        }
        a[e][1][0] = cx / mean_n;
        a[e][2][0] = cy / mean_n;
        // Rows 1-2, cols 1-3 = transposed corner node coordinates.
        for k in 0..n_verts {
            let ni = elements[[e, k]];
            a[e][1][k + 1] = nodes[[ni, 0]]; // x-row
            a[e][2][k + 1] = nodes[[ni, 1]]; // y-row
        }
    }
    a
}

/// Element warp vectors `(n_elems, 12)`, translating `MeshBase._element_strains`.
///
/// Indices 0-1: mean displacement; 2-5: 1st-order strain; 6-11: 2nd-order strain
/// (always zero for mesh_order == 1).
///
/// The 2×2 Jacobian `J_x_T = dN @ x` is inverted analytically for order-1 and
/// numerically for order-2.
pub fn element_strains(
    nodes: &Array2<f64>,
    elements: &Array2<usize>,
    displacements: &Array2<f64>,
    mesh_order: u8,
) -> Result<Array2<f64>, Error> {
    let n_elems = elements.nrows();
    let (n_sf, dn, d2n) = shape_function(mesh_order);
    let n_sf_len = n_sf.len(); // 3 or 6
    let mut warps = Array2::<f64>::zeros((n_elems, 12));
    let a = local_coordinates(nodes, elements);

    for e in 0..n_elems {
        // Gather node positions and displacements.
        let mut x = vec![[0.0f64; 2]; n_sf_len]; // x[k] = [xi, yi]
        let mut u = vec![[0.0f64; 2]; n_sf_len]; // u[k] = [uk, vk]
        for k in 0..n_sf_len {
            let ni = elements[[e, k]];
            x[k] = [nodes[[ni, 0]], nodes[[ni, 1]]];
            u[k] = [displacements[[ni, 0]], displacements[[ni, 1]]];
        }

        // Displacement centroid: N @ u (2 components).
        warps[[e, 0]] = n_sf.iter().zip(&u).map(|(&nk, &[uk, _])| nk * uk).sum();
        warps[[e, 1]] = n_sf.iter().zip(&u).map(|(&nk, &[_, vk])| nk * vk).sum();

        // 1st-order strains: J_u_T = dN @ u; J_x_T = dN @ x; strain = inv(J_x_T) @ J_u_T.
        // Both are 2×2 when dN is (2, n_sf_len) and x/u are (n_sf_len, 2).
        let mut j_x = [[0.0f64; 2]; 2]; // J_x_T[row][col]
        let mut j_u = [[0.0f64; 2]; 2];
        for row in 0..2 {
            for col in 0..2 {
                j_x[row][col] =
                    dn[row].iter().zip(&x).map(|(&d, xi)| d * xi[col]).sum();
                j_u[row][col] =
                    dn[row].iter().zip(&u).map(|(&d, ui)| d * ui[col]).sum();
            }
        }
        // 2×2 analytic inverse.
        let det = j_x[0][0] * j_x[1][1] - j_x[0][1] * j_x[1][0];
        if det.abs() < f64::EPSILON * 1e6 {
            return Err(Error::MeshGeneration(format!(
                "degenerate Jacobian at element {e}: det = {det:.2e}"
            )));
        }
        let inv = [
            [j_x[1][1] / det, -j_x[0][1] / det],
            [-j_x[1][0] / det, j_x[0][0] / det],
        ];
        // strain = inv(J_x_T) @ J_u_T → (2×2) @ (2×2) = (2×2), flatten row-major.
        // warps[e, 2..6] = [strain[0][0], strain[0][1], strain[1][0], strain[1][1]]
        // i.e. [du/dx, dv/dx, du/dy, dv/dy]
        for row in 0..2 {
            for col in 0..2 {
                warps[[e, 2 + row * 2 + col]] =
                    inv[row][0] * j_u[0][col] + inv[row][1] * j_u[1][col];
            }
        }

        // 2nd-order strains (mesh_order == 2 only).
        if let Some(ref d2n_mat) = d2n {
            // K_u = d2N @ u → (3, 2)
            let mut k_u = [[0.0f64; 2]; 3];
            for row in 0..3 {
                for col in 0..2 {
                    k_u[row][col] =
                        d2n_mat[row].iter().zip(&u).map(|(&d, ui)| d * ui[col]).sum();
                }
            }
            // dz = 2×2 matrix from corner node coords.
            let (x0, y0) = (x[0][0], x[0][1]);
            let (x1, y1) = (x[1][0], x[1][1]);
            let (x2, y2) = (x[2][0], x[2][1]);
            // det(A[e, :, [1,2,3]]) — the 3×3 submatrix of local_coordinates cols 1-3.
            // a[e][row][col+1] for row=0,1,2 and col=0,1,2.
            let ae = a[e];
            let det_a = ae[0][1] * (ae[1][2] * ae[2][3] - ae[1][3] * ae[2][2])
                - ae[0][2] * (ae[1][1] * ae[2][3] - ae[1][3] * ae[2][1])
                + ae[0][3] * (ae[1][1] * ae[2][2] - ae[1][2] * ae[2][1]);
            let dz = [
                [(y1 - y2) / det_a, (y2 - y0) / det_a],
                [(x2 - x1) / det_a, (x0 - x2) / det_a],
            ];
            // K_x_inv (3×3).
            let kx = [
                [
                    dz[0][0] * dz[0][0],
                    2.0 * dz[0][0] * dz[0][1],
                    dz[0][1] * dz[0][1],
                ],
                [
                    dz[0][0] * dz[1][0],
                    dz[0][0] * dz[1][1] + dz[0][1] * dz[1][0],
                    dz[0][1] * dz[1][1],
                ],
                [
                    dz[1][0] * dz[1][0],
                    2.0 * dz[1][0] * dz[1][1],
                    dz[1][1] * dz[1][1],
                ],
            ];
            // K_x_inv @ K_u → (3,2), flatten.
            for row in 0..3 {
                for col in 0..2 {
                    warps[[e, 6 + row * 2 + col]] =
                        kx[row][0] * k_u[0][col]
                            + kx[row][1] * k_u[1][col]
                            + kx[row][2] * k_u[2][col];
                }
            }
        }
    }
    Ok(warps)
}

/// Find the node index closest to `seed_coord`.
pub fn find_seed_node(nodes: &Array2<f64>, seed_coord: [f64; 2]) -> usize {
    (0..nodes.nrows())
        .min_by(|&a, &b| {
            let da = dist2(nodes, a, seed_coord);
            let db = dist2(nodes, b, seed_coord);
            da.partial_cmp(&db).unwrap()
        })
        .unwrap_or(0)
}

#[inline]
fn dist2(nodes: &Array2<f64>, idx: usize, coord: [f64; 2]) -> f64 {
    let dx = nodes[[idx, 0]] - coord[0];
    let dy = nodes[[idx, 1]] - coord[1];
    dx * dx + dy * dy
}

/// Return node indices connected to `idx`.
///
/// - Order-1 or `full=true`: all unique nodes sharing an element with `idx`,
///   excluding `idx` itself.
/// - Order-2, `full=false`: midpoint neighbours if `idx` is a corner; corner
///   neighbours if `idx` is a midpoint (based on column position in element).
pub fn connectivity(
    elements: &Array2<usize>,
    mesh_order: u8,
    idx: usize,
    full: bool,
) -> Vec<usize> {
    if mesh_order == 1 || full {
        let mut result: HashSet<usize> = HashSet::new();
        for e in 0..elements.nrows() {
            let n_cols = elements.ncols();
            let mut found = false;
            for k in 0..n_cols {
                if elements[[e, k]] == idx {
                    found = true;
                    break;
                }
            }
            if found {
                for k in 0..n_cols {
                    let nb = elements[[e, k]];
                    if nb != idx {
                        result.insert(nb);
                    }
                }
            }
        }
        let mut v: Vec<usize> = result.into_iter().collect();
        v.sort_unstable();
        v
    } else {
        // Order-2, full=false: edge-adjacent connectivity only.
        let mut result: HashSet<usize> = HashSet::new();
        for e in 0..elements.nrows() {
            let ncols = elements.ncols();
            for col in 0..ncols {
                if elements[[e, col]] != idx {
                    continue;
                }
                // Append based on which column position idx occupies.
                let row = &elements.row(e);
                let adds: &[usize] = match col {
                    0 => &[row[3], row[5]],     // corner 0 → mid01, mid20
                    1 => &[row[3], row[4]],     // corner 1 → mid01, mid12
                    2 => &[row[4], row[5]],     // corner 2 → mid12, mid20
                    3 => &[row[0], row[1]],     // mid01 → corner 0, corner 1
                    4 => &[row[1], row[2]],     // mid12 → corner 1, corner 2
                    5 => &[row[0], row[2]],     // mid20 → corner 0, corner 2
                    _ => &[],
                };
                for &nb in adds {
                    if nb != idx {
                        result.insert(nb);
                    }
                }
            }
        }
        let mut v: Vec<usize> = result.into_iter().collect();
        v.sort_unstable();
        v
    }
}

/// IQR-based outlier detection on C_ZNCC scores.
/// Returns indices where `C_ZNCC < LQ - 2.5 * IQR`.
pub fn corr(c_zncc: ArrayView1<f64>) -> Vec<usize> {
    let vals: Vec<f64> = c_zncc.iter().copied().collect();
    corr_1d(&vals)
}

pub fn corr_1d(vals: &[f64]) -> Vec<usize> {
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lq = percentile(&sorted, 25.0);
    let uq = percentile(&sorted, 75.0);
    let iqr = uq - lq;
    let fence = lq - 2.5 * iqr;
    vals.iter().enumerate().filter(|(_, v)| **v < fence).map(|(i, _)| i).collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let pos = p / 100.0 * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Flow for node `i`: dot(unit(disp_i), unit(mean_neighbour_disp)).
/// Returns `-1` if there are no valid neighbours.
pub fn flow_calc(
    idx: usize,
    elements: &Array2<usize>,
    mesh_order: u8,
    displacements: ArrayView2<f64>,
    exclude: &HashSet<usize>,
    displacement_override: Option<[f64; 2]>,
) -> f64 {
    let disp = displacement_override
        .unwrap_or([displacements[[idx, 0]], displacements[[idx, 1]]]);
    let neighbours = connectivity(elements, mesh_order, idx, true);
    let valid: Vec<usize> = neighbours
        .into_iter()
        .filter(|nb| !exclude.contains(nb))
        .collect();
    if valid.is_empty() {
        return -1.0;
    }
    let n = valid.len() as f64;
    let vx: f64 = valid.iter().map(|&nb| displacements[[nb, 0]]).sum::<f64>() / n;
    let vy: f64 = valid.iter().map(|&nb| displacements[[nb, 1]]).sum::<f64>() / n;
    let v_mag = (vx * vx + vy * vy).sqrt();
    let d_mag = (disp[0] * disp[0] + disp[1] * disp[1]).sqrt();
    if v_mag < f64::EPSILON || d_mag < f64::EPSILON {
        return -1.0;
    }
    (disp[0] * vx + disp[1] * vy) / (d_mag * v_mag)
}

fn flow_calc_excluding(
    idx: usize,
    exclude: &HashSet<usize>,
    elements: &Array2<usize>,
    mesh_order: u8,
    displacements: ArrayView2<f64>,
    displacement_override: Option<[f64; 2]>,
) -> f64 {
    flow_calc(idx, elements, mesh_order, displacements, exclude, displacement_override)
}

/// Absolute displacement magnitude.
pub fn r_calc(displacement: [f64; 2]) -> f64 {
    (displacement[0] * displacement[0] + displacement[1] * displacement[1]).sqrt()
}

/// Flow outlier IDs, LQ, IQR for the full mesh (no exclusions).
fn flow_stats(
    elements: &Array2<usize>,
    mesh_order: u8,
    displacements: ArrayView2<f64>,
) -> (Vec<f64>, Vec<usize>, f64, f64) {
    let n = displacements.nrows();
    let empty = HashSet::new();
    let flow: Vec<f64> = (0..n)
        .map(|i| flow_calc(i, elements, mesh_order, displacements, &empty, None))
        .collect();
    let mut sorted = flow.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lq = percentile(&sorted, 25.0);
    let uq = percentile(&sorted, 75.0);
    let iqr = uq - lq;
    let fence = lq - 1.5 * iqr;
    let ids: Vec<usize> = flow.iter().enumerate().filter(|(_, f)| **f < fence).map(|(i, _)| i).collect();
    (flow, ids, lq, iqr)
}

/// R (displacement magnitude) outlier IDs using element-local IQR.
fn r_stats(
    elements: &Array2<usize>,
    mesh_order: u8,
    _nodes: &Array2<f64>,
    displacements: ArrayView2<f64>,
) -> (Vec<f64>, HashSet<usize>) {
    let n = displacements.nrows();
    let r: Vec<f64> = (0..n)
        .map(|i| r_calc([displacements[[i, 0]], displacements[[i, 1]]]))
        .collect();
    let mut r_ids: HashSet<usize> = HashSet::new();
    let step = if mesh_order == 2 { 2 } else { 1 };
    for e in 0..elements.nrows() {
        // Collect all neighbourhood nodes from corners of this element.
        let mut local: HashSet<usize> = HashSet::new();
        for k in (0..3).step_by(1) {
            let corner = elements[[e, k * step.min(elements.ncols() - 1)]];
            for nb in connectivity(elements, mesh_order, corner, true) {
                local.insert(nb);
            }
        }
        let local_r: Vec<f64> = local.iter().map(|&nb| r[nb]).collect();
        if local_r.is_empty() {
            continue;
        }
        let mut sr = local_r.clone();
        sr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lq = percentile(&sr, 25.0);
        let uq = percentile(&sr, 75.0);
        let iqr = uq - lq;
        let fence = uq + 4.0 * iqr;
        for &nb in &local {
            if r[nb] > fence {
                r_ids.insert(nb);
            }
        }
    }
    (r, r_ids)
}

/// Adaptive target areas from shear-strain × area products.
///
/// `D[e] = |warps[e,3] + warps[e,4]| * |areas[e]|`  (warps[3]=dv/dx, warps[4]=du/dy)
/// `target[e] = areas[e] * clip(D[e] / mean(D), alpha, 1/alpha)^-2`
pub fn adaptive_target_areas(
    warps: &Array2<f64>,
    areas: &Array1<f64>,
    alpha: f64,
) -> Array1<f64> {
    let n = warps.nrows();
    let d: Vec<f64> = (0..n)
        .map(|e| (warps[[e, 3]] + warps[[e, 4]]).abs() * areas[e].abs())
        .collect();
    let d_mean: f64 = d.iter().sum::<f64>() / n as f64;
    let inv_alpha = 1.0 / alpha;
    (0..n)
        .map(|e| {
            let ratio = if d_mean > 0.0 {
                (d[e] / d_mean).clamp(alpha, inv_alpha)
            } else {
                1.0
            };
            areas[e] * ratio.powi(-2)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helper — warp propagation
// ---------------------------------------------------------------------------

/// Project warp from `from_idx` to `to_idx` using a Taylor expansion
/// (mirrors `Mesh._neighbours` attempt 2).
fn project_warp(
    p: &[f64],
    nodes: &Array2<f64>,
    from_idx: usize,
    to_idx: usize,
) -> Vec<f64> {
    let dx = nodes[[to_idx, 0]] - nodes[[from_idx, 0]];
    let dy = nodes[[to_idx, 1]] - nodes[[from_idx, 1]];
    let mut p_proj = p.to_vec();
    if p.len() >= 6 {
        p_proj[0] = p[0] + p[2] * dx + p[4] * dy;
        p_proj[1] = p[1] + p[3] * dx + p[5] * dy;
    }
    if p.len() >= 12 {
        p_proj[0] += 0.5 * p[6] * dx * dx + p[8] * dx * dy + 0.5 * p[10] * dy * dy;
        p_proj[1] += 0.5 * p[7] * dx * dx + p[9] * dx * dy + 0.5 * p[11] * dy * dy;
        p_proj[2] = p[2] + p[6] * dx + p[8] * dy;
        p_proj[3] = p[3] + p[7] * dx + p[9] * dy;
        p_proj[4] = p[4] + p[8] * dy + p[10] * dy;
        p_proj[5] = p[5] + p[9] * dy + p[11] * dy;
    }
    p_proj
}

/// Store a `SolveResult` into the per-node arrays.
fn store_result(
    idx: usize,
    result: &SolveResult,
    c_zncc: &mut Array1<f64>,
    p: &mut Array2<f64>,
    displacements: &mut Array2<f64>,
    iterations: &mut Array1<u32>,
    norms: &mut Array1<f64>,
) {
    c_zncc[idx] = result.c_zncc.max(0.0);
    for (k, &v) in result.p.iter().enumerate() {
        p[[idx, k]] = v;
    }
    displacements[[idx, 0]] = result.p[0];
    displacements[[idx, 1]] = result.p[1];
    iterations[idx] = result.iterations as u32;
    norms[idx] = result.history.last().map(|h| h.1).unwrap_or(0.0);
}

/// Pick the next node to propagate from: highest `C_ZNCC × solved_flag`.
/// Returns `None` when all nodes have been propagated (`solved[i] = true`
/// after marking, or not yet solved but also not yet queued).
///
/// In practice we want the max over nodes that have been solved-but-not-propagated
/// (i.e. `solved[i] == false` but have been stored by `solve_neighbours_from`).
/// This mirrors `np.argmax(solved * C_ZNCC)` in the Python where `solved` starts at
/// 0 and flips to 1 when stored.
/// Pick the next node to propagate from: the unsolved node with highest C_ZNCC.
/// A node is "ready to propagate" if it has c_zncc > 0 but hasn't been marked
/// as propagated yet. Returns `None` when no such node exists.
fn next_queue_node(solved: &[bool], c_zncc: ArrayView1<f64>) -> Option<usize> {
    let mut best_idx = None;
    let mut best_val = 0.0_f64; // strictly > 0 required to be "stored"
    for (i, &is_propagated) in solved.iter().enumerate() {
        if is_propagated {
            continue;
        }
        let cv = c_zncc[i];
        if cv > best_val {
            best_val = cv;
            best_idx = Some(i);
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn unit_triangle_mesh() -> (Array2<f64>, Array2<usize>) {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        (nodes, elements)
    }

    // -----------------------------------------------------------------------
    // element_area
    // -----------------------------------------------------------------------

    #[test]
    fn element_area_unit_triangle() {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let elements = array![[0usize, 1, 2]];
        let areas = element_area(&nodes, &elements);
        assert!((areas[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn element_area_two_triangles() {
        let (nodes, elements) = unit_triangle_mesh();
        let areas = element_area(&nodes, &elements);
        assert_eq!(areas.len(), 2);
        for a in areas.iter() {
            assert!((a.abs() - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn element_area_scaled_triangle() {
        let nodes = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let elements = array![[0usize, 1, 2]];
        let areas = element_area(&nodes, &elements);
        assert!((areas[0] - 2.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // shape_function
    // -----------------------------------------------------------------------

    #[test]
    fn shape_function_order1_n_sums_to_one() {
        let (n, _, _) = shape_function(1);
        assert!((n.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn shape_function_order1_d2n_none() {
        let (_, _, d2n) = shape_function(1);
        assert!(d2n.is_none());
    }

    #[test]
    fn shape_function_order2_n_values() {
        let (n, _, _) = shape_function(2);
        let expected = [-1.0/9.0, -1.0/9.0, -1.0/9.0, 4.0/9.0, 4.0/9.0, 4.0/9.0];
        for (a, b) in n.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn shape_function_order2_has_d2n() {
        let (_, _, d2n) = shape_function(2);
        assert!(d2n.is_some());
        assert_eq!(d2n.unwrap().len(), 3);
    }

    // -----------------------------------------------------------------------
    // element_strains
    // -----------------------------------------------------------------------

    #[test]
    fn element_strains_pure_translation_order1() {
        let (nodes, elements) = unit_triangle_mesh();
        let disps = array![[3.5, -2.1], [3.5, -2.1], [3.5, -2.1], [3.5, -2.1]];
        let warps = element_strains(&nodes, &elements, &disps, 1).unwrap();
        // Displacement centroid = (3.5, -2.1), strains = 0.
        for e in 0..2 {
            assert!((warps[[e, 0]] - 3.5).abs() < 1e-10);
            assert!((warps[[e, 1]] - (-2.1)).abs() < 1e-10);
            for k in 2..6 {
                assert!(warps[[e, k]].abs() < 1e-10, "warps[{e},{k}] = {:.2e}", warps[[e,k]]);
            }
        }
    }

    #[test]
    fn element_strains_x_shear_order1() {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let elements = array![[0usize, 1, 2]];
        let disps = array![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]];
        let warps = element_strains(&nodes, &elements, &disps, 1).unwrap();
        // Centroid u = 1/3.
        assert!((warps[[0, 0]] - 1.0 / 3.0).abs() < 1e-12);
        assert!(warps[[0, 1]].abs() < 1e-12);
        // du/dx = 1, rest = 0.
        assert!((warps[[0, 2]] - 1.0).abs() < 1e-10);
        assert!(warps[[0, 3]].abs() < 1e-10);
        assert!(warps[[0, 4]].abs() < 1e-10);
        assert!(warps[[0, 5]].abs() < 1e-10);
    }

    #[test]
    fn element_strains_shape_order1() {
        let (nodes, elements) = unit_triangle_mesh();
        let disps = Array2::<f64>::zeros((4, 2));
        let warps = element_strains(&nodes, &elements, &disps, 1).unwrap();
        assert_eq!(warps.shape(), &[2, 12]);
    }

    #[test]
    fn element_strains_pure_translation_order2() {
        // 9-node mesh: 4 corners + 5 midpoints
        let nodes = array![
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1.0, 0.5], [0.5, 1.0]
        ];
        let elements = array![
            [0usize, 1, 2, 4, 5, 6],
            [1, 3, 2, 7, 8, 5]
        ];
        let disps = array![
            [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0],
            [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]
        ];
        let warps = element_strains(&nodes, &elements, &disps, 2).unwrap();
        // Pure translation: centroid disp = (1,2), all strains = 0.
        for e in 0..2 {
            assert!((warps[[e, 0]] - 1.0).abs() < 1e-10);
            assert!((warps[[e, 1]] - 2.0).abs() < 1e-10);
            for k in 2..12 {
                assert!(
                    warps[[e, k]].abs() < 1e-8,
                    "warps[{e},{k}] = {:.2e}",
                    warps[[e, k]]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // connectivity
    // -----------------------------------------------------------------------

    #[test]
    fn connectivity_order1_corner() {
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let result = connectivity(&elements, 1, 0, false);
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn connectivity_order1_shared_node() {
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let result = connectivity(&elements, 1, 1, false);
        assert_eq!(result, vec![0, 2, 3]);
    }

    #[test]
    fn connectivity_order2_corner_gets_midpoints() {
        // elem 0 = [0,1,2,4,5,6] — corner 0 at position 0 → mids at positions 3 and 5: [4,6]
        let elements = array![[0usize, 1, 2, 4, 5, 6], [1, 3, 2, 7, 8, 5]];
        let result = connectivity(&elements, 2, 0, false);
        assert_eq!(result, vec![4, 6]);
    }

    #[test]
    fn connectivity_order2_midpoint_gets_corners() {
        // elem 0 = [0,1,2,4,5,6] — midpoint 4 at position 3 → corners [:2] = [0,1]
        let elements = array![[0usize, 1, 2, 4, 5, 6], [1, 3, 2, 7, 8, 5]];
        let result = connectivity(&elements, 2, 4, false);
        assert_eq!(result, vec![0, 1]);
    }

    // -----------------------------------------------------------------------
    // find_seed_node
    // -----------------------------------------------------------------------

    #[test]
    fn find_seed_node_exact() {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert_eq!(find_seed_node(&nodes, [1.0, 0.0]), 1);
    }

    #[test]
    fn find_seed_node_nearest() {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        // (0.4, 0.4): d0=0.566, d1=0.721, d2=0.721, d3=0.849 → node 0
        assert_eq!(find_seed_node(&nodes, [0.4, 0.4]), 0);
    }

    // -----------------------------------------------------------------------
    // corr_1d (IQR outlier detection)
    // -----------------------------------------------------------------------

    #[test]
    fn corr_no_outliers() {
        let vals = vec![0.9f64; 10];
        assert!(corr_1d(&vals).is_empty());
    }

    #[test]
    fn corr_detects_single_outlier() {
        let mut vals = vec![0.9f64; 19];
        vals.push(-10.0);
        let result = corr_1d(&vals);
        assert!(result.contains(&19));
    }

    // -----------------------------------------------------------------------
    // flow_calc
    // -----------------------------------------------------------------------

    #[test]
    fn flow_calc_aligned_returns_one() {
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let disps = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
        let result = flow_calc(0, &elements, 1, disps.view(), &HashSet::new(), None);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn flow_calc_anti_aligned_returns_minus_one() {
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let disps = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
        let result =
            flow_calc(0, &elements, 1, disps.view(), &HashSet::new(), Some([-1.0, 0.0]));
        assert!((result - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn flow_calc_no_neighbours_returns_minus_one() {
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let disps = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
        // Exclude all of node 0's neighbours.
        let exclude: HashSet<usize> = [1, 2].iter().copied().collect();
        let result = flow_calc(0, &elements, 1, disps.view(), &exclude, None);
        assert_eq!(result, -1.0);
    }

    // -----------------------------------------------------------------------
    // r_calc
    // -----------------------------------------------------------------------

    #[test]
    fn r_calc_unit() {
        assert!((r_calc([1.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn r_calc_pythagorean() {
        assert!((r_calc([3.0, 4.0]) - 5.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // adaptive_target_areas
    // -----------------------------------------------------------------------

    #[test]
    fn adaptive_target_areas_uniform_shear_unchanged() {
        // Uniform shear strain → D/mean(D) = 1 for all elements → clamp(1, α, 1/α) = 1 → target = area.
        let warps = array![[0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let areas: Array1<f64> = array![2.0, 3.0];
        let target = adaptive_target_areas(&warps, &areas, 0.5);
        // D[0] = |0.5+0.5| * 2 = 2; D[1] = |0.5+0.5| * 3 = 3; mean = 2.5
        // ratio[0] = clamp(2/2.5=0.8, 0.5, 2) = 0.8; target[0] = 2 * 0.8^-2 = 3.125
        // ratio[1] = clamp(3/2.5=1.2, 0.5, 2) = 1.2; target[1] = 3 * 1.2^-2 = 2.083...
        assert!((target[0] - 2.0 * 0.8f64.powi(-2)).abs() < 1e-10);
        assert!((target[1] - 3.0 * 1.2f64.powi(-2)).abs() < 1e-10);
    }

    #[test]
    fn adaptive_target_areas_zero_shear_stays_original() {
        // Zero shear → D_mean = 0 → ratio = 1 → target = area.
        let warps = Array2::<f64>::zeros((3, 12));
        let areas: Array1<f64> = array![1.0, 2.0, 3.0];
        let target = adaptive_target_areas(&warps, &areas, 0.5);
        for (t, a) in target.iter().zip(areas.iter()) {
            assert!((t - a).abs() < 1e-12);
        }
    }
}
