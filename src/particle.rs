//! Particle tracking and strain-path computation.
//!
//! Translates `geopyv/src/geopyv/particle.py` (Particle class; all `geomat`
//! sections are excluded — no stress path, no friction work).
//!
//! # Architecture
//!
//! - Pure-math free functions (`local_coordinates`, `shape_function`,
//!   `warp_increment`, `element_locator`, `strain_def`) are exposed `pub` so
//!   they can be called by [`crate::field`] and tested in unit tests.
//! - [`Particle`] owns the accumulated state arrays; [`ParticleConfig`] holds
//!   solve parameters.
//! - [`Particle::solve_increment`] is the per-step method called by
//!   [`crate::field::Field`]; [`Particle::solve`] loops over a slice of mesh
//!   data.
//!
//! # Excluded from translation
//!
//! - `_stress_path`, stress-variant `_strain_path_full/inc` — require `geomat`
//! - `_stresses`, `_ps`, `_qs`, `_friction_works` data fields
//! - `model`, `state`, `parameters` arguments to `solve()`
//! - `_plotting_coordinates` — plotting only
//! - `_coord_map` — calibrated-space mapping (skip for uncalibrated)
//! - `ParticleBase.trace`, `history` — plotting
//! - `_check_update` filename comparison — replaced by explicit `ref_update`
//!   flag in `solve_increment`

use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::Error;

// ---------------------------------------------------------------------------
// MeshData — thin view of the mesh arrays needed by a particle
// ---------------------------------------------------------------------------

/// Minimal mesh data required for particle strain-path computation.
///
/// [`crate::field::Field`] constructs this from a [`crate::mesh::MeshSolution`].
pub struct MeshData<'a> {
    pub nodes: &'a Array2<f64>,
    pub elements: &'a Array2<usize>,
    pub displacements: &'a Array2<f64>,
    pub mesh_order: u8,
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute barycentric coordinates of `coordinate` inside a triangle.
///
/// Replicates `Particle._local_coordinates`.
///
/// # Arguments
/// * `coordinate`     — particle position [x, y]
/// * `element_nodes`  — node coordinates; at least 3 rows, 2 columns.
///                       Only the first 3 rows (corner nodes) are used.
///
/// # Returns
/// `(zeta, eta, theta, det_denom)`
/// where `zeta + eta + theta == 1` and `det_denom` is the denominator
/// determinant (= 2 × signed area of the triangle).  `zeta >= 0 && eta >= 0
/// && theta >= 0` iff `coordinate` is inside (or on the boundary of) the
/// triangle.
pub fn local_coordinates(
    coordinate: [f64; 2],
    element_nodes: ArrayView2<f64>,
) -> (f64, f64, f64, f64) {
    // Python:
    //   A[1:, 0] = coordinate
    //   A[1:, 1:] = element_nodes[:3, :2].T
    // → A = [[1,    1,    1,    1   ],
    //        [px, n0.x, n1.x, n2.x ],
    //        [py, n0.y, n1.y, n2.y ]]
    let px = coordinate[0];
    let py = coordinate[1];
    let (n0x, n0y) = (element_nodes[[0, 0]], element_nodes[[0, 1]]);
    let (n1x, n1y) = (element_nodes[[1, 0]], element_nodes[[1, 1]]);
    let (n2x, n2y) = (element_nodes[[2, 0]], element_nodes[[2, 1]]);

    // det(A[:, [1,2,3]])
    let det_denom = det3(
        1.0, 1.0, 1.0,
        n0x, n1x, n2x,
        n0y, n1y, n2y,
    );
    // det(A[:, [0,2,3]])
    let det_zeta = det3(
        1.0, 1.0, 1.0,
        px, n1x, n2x,
        py, n1y, n2y,
    );
    // det(A[:, [0,3,1]])
    let det_eta = det3(
        1.0, 1.0, 1.0,
        px, n2x, n0x,
        py, n2y, n0y,
    );
    let zeta  = det_zeta / det_denom;
    let eta   = det_eta  / det_denom;
    let theta = 1.0 - zeta - eta;
    (zeta, eta, theta, det_denom)
}

#[inline]
fn det3(
    a00: f64, a01: f64, a02: f64,
    a10: f64, a11: f64, a12: f64,
    a20: f64, a21: f64, a22: f64,
) -> f64 {
    a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
}

/// Evaluate element shape functions and their derivatives.
///
/// Replicates `Particle._shape_function`.
///
/// # Returns
/// * `n`   — shape values, length 3 (order-1) or 6 (order-2)
/// * `dn`  — `(2, 3 or 6)` first derivatives w.r.t. (zeta, eta)
/// * `d2n` — `Some((3, 6))` second derivatives (order-2 only), else `None`
pub fn shape_function(
    mesh_order: u8,
    zeta: f64,
    eta: f64,
    theta: f64,
) -> (Array1<f64>, Array2<f64>, Option<Array2<f64>>) {
    if mesh_order == 1 {
        let n = Array1::from_vec(vec![zeta, eta, theta]);
        #[rustfmt::skip]
        let dn = Array2::from_shape_vec(
            (2, 3),
            vec![1.0, 0.0, -1.0, 0.0, 1.0, -1.0],
        ).unwrap();
        (n, dn, None)
    } else {
        let n = Array1::from_vec(vec![
            zeta  * (2.0 * zeta  - 1.0),
            eta   * (2.0 * eta   - 1.0),
            theta * (2.0 * theta - 1.0),
            4.0 * zeta * eta,
            4.0 * eta  * theta,
            4.0 * theta * zeta,
        ]);
        #[rustfmt::skip]
        let dn = Array2::from_shape_vec(
            (2, 6),
            vec![
                4.0*zeta - 1.0,  0.0,             1.0 - 4.0*theta,
                4.0*eta,         -4.0*eta,         4.0*(theta - zeta),
                0.0,             4.0*eta - 1.0,   1.0 - 4.0*theta,
                4.0*zeta,        4.0*(theta - eta), -4.0*zeta,
            ],
        ).unwrap();
        #[rustfmt::skip]
        let d2n = Array2::from_shape_vec(
            (3, 6),
            vec![
                4.0, 0.0, 4.0, 0.0,  0.0, -8.0,
                0.0, 0.0, 4.0, 4.0, -4.0, -4.0,
                0.0, 4.0, 4.0, 0.0, -8.0,  0.0,
            ],
        ).unwrap();
        (n, dn, Some(d2n))
    }
}

/// Compute the warp increment for a particle at `coordinate`.
///
/// Replicates `Particle._warp_increment` (without the plotting-coordinates
/// side-effect).
///
/// # Arguments
/// * `coordinate`     — current particle position [x, y]
/// * `element_nodes`  — `(3 or 6, 2)` node positions for the element
/// * `element_disps`  — `(3 or 6, 2)` node displacements for the element
/// * `mesh_order`     — 1 or 2
///
/// # Returns
/// Warp increment vector of length `6 * mesh_order`:
/// `[u, v, du/dx, dv/dx, du/dy, dv/dy, (6 higher-order terms for order-2)]`
pub fn warp_increment(
    coordinate: [f64; 2],
    element_nodes: ArrayView2<f64>,
    element_disps: ArrayView2<f64>,
    mesh_order: u8,
) -> Array1<f64> {
    let p_len = 6 * mesh_order as usize;
    let mut warp_inc = Array1::<f64>::zeros(p_len);

    let (zeta, eta, theta, det_denom) = local_coordinates(coordinate, element_nodes);
    let (n, dn, d2n_opt) = shape_function(mesh_order, zeta, eta, theta);

    // Interpolated displacement: warp_inc[0:2] = N @ u
    let disp = n.dot(&element_disps);   // (6or3,) dot (6or3, 2) → (2,)
    warp_inc[0] = disp[0];
    warp_inc[1] = disp[1];

    // 1st-order strain: (inv(J_x_T) @ J_u_T).flatten()
    // J_x_T = dN @ x  →  (2, 3or6) @ (3or6, 2) = (2, 2)
    // J_u_T = dN @ u  →  same shape
    let j_x_t = dn.dot(&element_nodes);    // (2, 2)
    let j_u_t = dn.dot(&element_disps);    // (2, 2)
    let inv_j = inv2x2(j_x_t.view());
    let grad_u = inv_j.dot(&j_u_t);        // (2, 2)
    warp_inc[2] = grad_u[[0, 0]]; // du/dx
    warp_inc[3] = grad_u[[1, 0]]; // dv/dx
    warp_inc[4] = grad_u[[0, 1]]; // du/dy
    warp_inc[5] = grad_u[[1, 1]]; // dv/dy

    // 2nd-order strain (order-2 only)
    if mesh_order == 2 {
        let d2n = d2n_opt.unwrap();
        let k_u = d2n.dot(&element_disps); // (3, 2)

        let x = element_nodes;
        // dz = [[x[1,1]-x[2,1], x[2,1]-x[0,1]],
        //        [x[2,0]-x[1,0], x[0,0]-x[2,0]]] / det_denom
        let mut dz = Array2::<f64>::zeros((2, 2));
        dz[[0, 0]] = x[[1, 1]] - x[[2, 1]];
        dz[[0, 1]] = x[[2, 1]] - x[[0, 1]];
        dz[[1, 0]] = x[[2, 0]] - x[[1, 0]];
        dz[[1, 1]] = x[[0, 0]] - x[[2, 0]];
        let dz = dz / det_denom;

        // K_x_inv (3×3): second-order metric tensor
        let k_x_inv = build_k_x_inv(dz.view());

        let higher = k_x_inv.dot(&k_u); // (3, 2)
        // Flatten row-major into warp_inc[6..12]
        warp_inc[6]  = higher[[0, 0]];
        warp_inc[7]  = higher[[0, 1]];
        warp_inc[8]  = higher[[1, 0]];
        warp_inc[9]  = higher[[1, 1]];
        warp_inc[10] = higher[[2, 0]];
        warp_inc[11] = higher[[2, 1]];
    }

    warp_inc
}

/// Invert a 2×2 matrix analytically.
fn inv2x2(m: ArrayView2<f64>) -> Array2<f64> {
    let det = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    let inv_det = 1.0 / det;
    Array2::from_shape_vec(
        (2, 2),
        vec![
             m[[1, 1]] * inv_det, -m[[0, 1]] * inv_det,
            -m[[1, 0]] * inv_det,  m[[0, 0]] * inv_det,
        ],
    ).unwrap()
}

/// Build the 3×3 second-order metric tensor K_x_inv from the 2×2 dz matrix.
fn build_k_x_inv(dz: ArrayView2<f64>) -> Array2<f64> {
    let (d00, d01) = (dz[[0, 0]], dz[[0, 1]]);
    let (d10, d11) = (dz[[1, 0]], dz[[1, 1]]);
    Array2::from_shape_vec(
        (3, 3),
        vec![
            d00 * d00,  2.0 * d00 * d01,  d01 * d01,
            d00 * d10,  d00 * d11 + d01 * d10,  d01 * d11,
            d10 * d10,  2.0 * d10 * d11,  d11 * d11,
        ],
    ).unwrap()
}

/// Compute element centroids (mean of corner node positions).
pub fn compute_centroids(nodes: &Array2<f64>, elements: &Array2<usize>) -> Array2<f64> {
    let n_elem = elements.nrows();
    let mut centroids = Array2::zeros((n_elem, 2));
    for i in 0..n_elem {
        let n0 = elements[[i, 0]];
        let n1 = elements[[i, 1]];
        let n2 = elements[[i, 2]];
        centroids[[i, 0]] = (nodes[[n0, 0]] + nodes[[n1, 0]] + nodes[[n2, 0]]) / 3.0;
        centroids[[i, 1]] = (nodes[[n0, 1]] + nodes[[n1, 1]] + nodes[[n2, 1]]) / 3.0;
    }
    centroids
}

/// Find the index of the element containing `coordinate`.
///
/// Replicates `Particle._element_locator`.
///
/// Searches nearest centroids first, testing each with the barycentric
/// sign test.  Falls back to the nearest centroid if no element contains
/// the point (adrift case).
pub fn element_locator(
    coordinate: [f64; 2],
    nodes: &Array2<f64>,
    elements: &Array2<usize>,
    centroids: &Array2<f64>,
) -> usize {
    let n_elem = centroids.nrows();
    // Squared centroid distances → sorted indices
    let mut dists: Vec<(f64, usize)> = (0..n_elem)
        .map(|i| {
            let dx = centroids[[i, 0]] - coordinate[0];
            let dy = centroids[[i, 1]] - coordinate[1];
            (dx * dx + dy * dy, i)
        })
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Check sorted elements; Python checks min(10, ceil(0.05*n_elem)) first
    // via argpartition then the whole array. Sorting and iterating is equivalent.
    let n_check = (n_elem * 5 / 100).max(1).min(10);
    for &(_, idx) in dists.iter().take(n_check.max(n_elem)) {
        if point_in_triangle(coordinate, nodes, elements, idx) {
            return idx;
        }
    }
    // Adrift fallback: nearest centroid
    dists[0].1
}

/// Returns true if `coordinate` is inside (or on the boundary of) triangle `idx`.
fn point_in_triangle(
    coordinate: [f64; 2],
    nodes: &Array2<f64>,
    elements: &Array2<usize>,
    idx: usize,
) -> bool {
    // Extract corner-node rows into a small owned view
    let i0 = elements[[idx, 0]];
    let i1 = elements[[idx, 1]];
    let i2 = elements[[idx, 2]];
    let corners = Array2::from_shape_vec(
        (3, 2),
        vec![
            nodes[[i0, 0]], nodes[[i0, 1]],
            nodes[[i1, 0]], nodes[[i1, 1]],
            nodes[[i2, 0]], nodes[[i2, 1]],
        ],
    ).unwrap();
    let (z, e, t, _) = local_coordinates(coordinate, corners.view());
    z >= 0.0 && e >= 0.0 && t >= 0.0
}

/// Compute strains and strain increments from accumulated warp vectors.
///
/// Replicates `Particle._strain_def`.
///
/// # Arguments
/// * `warps`      — `(inc_no, 6*mesh_order)` accumulated warp state
/// * `factor`     — volumetric correction: `1.0` removes full isotropic part,
///                  `0.0` leaves increments unchanged
/// * `true_incs`  — if `false`, linearise the first two strain increments
///
/// # Returns
/// `(strains, strain_incs)` where
/// * `strains`     has shape `(inc_no, 6)`
/// * `strain_incs` has shape `(inc_no-1, 6)`
pub fn strain_def(
    warps: &Array2<f64>,
    factor: f64,
    true_incs: bool,
) -> (Array2<f64>, Array2<f64>) {
    let inc_no = warps.nrows();
    let mut strains = Array2::<f64>::zeros((inc_no, 6));

    // Sign convention: compression positive (opposite to DIC warp sign)
    for i in 0..inc_no {
        strains[[i, 5]] = -(warps[[i, 3]] + warps[[i, 4]]) / 2.0; // shear
        strains[[i, 0]] = -warps[[i, 2]];                          // eps_xx
        strains[[i, 1]] = -warps[[i, 5]];                          // eps_yy
    }

    // strain_incs = diff(strains, axis=0)
    let n_inc = inc_no - 1;
    let mut strain_incs = Array2::<f64>::zeros((n_inc, 6));
    for i in 0..n_inc {
        for j in 0..6 {
            strain_incs[[i, j]] = strains[[i + 1, j]] - strains[[i, j]];
        }
    }

    if !true_incs {
        // Linearise first two increment rows (Python: strain_incs[[0,1]])
        let lim = n_inc.min(2);
        for i in 0..lim {
            strain_incs[[i, 0]] = -((-strain_incs[[i, 0]]).exp() - 1.0);
            strain_incs[[i, 1]] = -((-strain_incs[[i, 1]]).exp() - 1.0);
        }
    }

    // Remove volumetric part from eps_xx and eps_yy increments
    if factor.abs() > f64::EPSILON {
        for i in 0..n_inc {
            let mean_vol = (strain_incs[[i, 0]] + strain_incs[[i, 1]]) / 2.0;
            strain_incs[[i, 0]] -= factor * mean_vol;
            strain_incs[[i, 1]] -= factor * mean_vol;
        }
    }

    // Recompute strains[1:, [0,1]] = cumsum of corrected increments
    let mut cumsum_xx = 0.0f64;
    let mut cumsum_yy = 0.0f64;
    for i in 0..n_inc {
        cumsum_xx += strain_incs[[i, 0]];
        cumsum_yy += strain_incs[[i, 1]];
        strains[[i + 1, 0]] = cumsum_xx;
        strains[[i + 1, 1]] = cumsum_yy;
    }

    (strains, strain_incs)
}

/// Compute volumetric strains from a volumes array.
///
/// Replicates `self._vol_strains = (self._volumes - self._volumes[0]) / self._volumes[0]`.
pub fn vol_strains(volumes: &Array1<f64>) -> Array1<f64> {
    let v0 = volumes[0];
    volumes.mapv(|v| (v - v0) / v0)
}

// ---------------------------------------------------------------------------
// Particle struct
// ---------------------------------------------------------------------------

/// Configuration for [`Particle::solve`].
#[derive(Debug, Clone)]
pub struct ParticleConfig {
    /// Volumetric correction factor applied in `strain_def`:
    /// `1.0` → full isotropic removal, `0.0` → no correction.
    pub factor: f64,
    /// Use true (logarithmic) strain increments.
    pub true_incs: bool,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self { factor: 1.0, true_incs: true }
    }
}

/// Output of [`Particle::finalize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleSolution {
    /// Particle positions `(inc_no, 2)`.
    pub coordinates: Array2<f64>,
    /// Accumulated warp vectors `(inc_no, 6*mesh_order)`.
    pub warps: Array2<f64>,
    /// Warp increments `(inc_no, 6*mesh_order)`.
    pub incs: Array2<f64>,
    /// Volumes `(inc_no,)`.
    pub volumes: Array1<f64>,
    /// Strains `(inc_no, 6)`.
    pub strains: Array2<f64>,
    /// Strain increments `(inc_no-1, 6)`.
    pub strain_incs: Array2<f64>,
    /// Volumetric strains `(inc_no,)`.
    pub vol_strains: Array1<f64>,
    /// Increment indices at which the reference mesh was updated.
    pub reference_update_register: Vec<usize>,
}

/// Lagrangian / Eulerian particle tracking and strain-path computation.
///
/// Construct with [`Particle::new`]; then either:
/// - call [`Particle::solve`] for a standalone sequence solve, or
/// - call [`Particle::solve_increment`] per step from [`crate::field::Field`].
pub struct Particle {
    // --- Configuration ---
    pub mesh_order: u8,
    pub track: bool,   // true → Lagrangian (coordinate moves with material)

    // --- Size ---
    inc_no: usize,

    // --- Accumulated state arrays ---
    /// Particle coordinates `(inc_no, 2)`.
    pub coordinates: Array2<f64>,
    /// Accumulated warp vectors `(inc_no, 6*mesh_order)`.
    pub warps: Array2<f64>,
    /// Warp increments `(inc_no, 6*mesh_order)`.
    pub incs: Array2<f64>,
    /// Particle volumes `(inc_no,)`.
    pub volumes: Array1<f64>,

    // --- Reference tracking ---
    reference_index: usize,
    pub reference_update_register: Vec<usize>,
    _adrift: bool,

    // --- Solve bookkeeping ---
    pub current_step: usize,
    pub solved: bool,
}

impl Particle {
    /// Construct a new particle.
    ///
    /// # Arguments
    /// * `coordinate`     — initial position [x, y]
    /// * `initial_warp`   — initial warp vector (length 6 or 12); zero-padded
    ///                       if shorter than `6 * mesh_order`
    /// * `initial_volume` — initial volume
    /// * `inc_no`         — total number of increments (= number of meshes + 1)
    /// * `mesh_order`     — 1 or 2
    /// * `track`          — `true` for Lagrangian tracking
    pub fn new(
        coordinate: [f64; 2],
        initial_warp: &[f64],
        initial_volume: f64,
        inc_no: usize,
        mesh_order: u8,
        track: bool,
    ) -> Result<Self, Error> {
        if initial_volume <= 0.0 {
            return Err(Error::InvalidInput(
                "initial_volume must be > 0".to_string(),
            ));
        }
        let p_len = 6 * mesh_order as usize;
        let mut warps = Array2::<f64>::zeros((inc_no, p_len));
        let copy_len = initial_warp.len().min(p_len);
        for i in 0..copy_len {
            warps[[0, i]] = initial_warp[i];
        }
        let mut coordinates = Array2::<f64>::zeros((inc_no, 2));
        coordinates[[0, 0]] = coordinate[0];
        coordinates[[0, 1]] = coordinate[1];
        let mut volumes = Array1::<f64>::zeros(inc_no);
        volumes[0] = initial_volume;

        Ok(Particle {
            mesh_order,
            track,
            inc_no,
            coordinates,
            warps,
            incs: Array2::zeros((inc_no, p_len)),
            volumes,
            reference_index: 0,
            reference_update_register: Vec::new(),
            _adrift: false,
            current_step: 0,
            solved: false,
        })
    }

    /// Solve a single increment.
    ///
    /// Replicates `Particle._strain_path_inc` (minus stress path).
    ///
    /// # Arguments
    /// * `m`          — increment index (0-based; result stored at `m+1`)
    /// * `cm`         — current mesh data
    /// * `ref_update` — `true` if the reference mesh changed at step `m`
    ///                   (replaces Python's filename-based `_check_update`)
    ///
    /// Returns `true` on success.
    pub fn solve_increment(
        &mut self,
        m: usize,
        cm: &MeshData<'_>,
        ref_update: bool,
    ) -> bool {
        if ref_update {
            self.reference_index = m;
            self.reference_update_register.push(m);
        }

        let centroids = compute_centroids(cm.nodes, cm.elements);
        let coord = [
            self.coordinates[[self.reference_index, 0]],
            self.coordinates[[self.reference_index, 1]],
        ];
        let tri_idx = element_locator(coord, cm.nodes, cm.elements, &centroids);

        let elem = cm.elements.row(tri_idx);
        // Gather element node coordinates and displacements
        let ncols = cm.elements.ncols(); // 3 or 6
        let mut e_nodes = Array2::<f64>::zeros((ncols, 2));
        let mut e_disps = Array2::<f64>::zeros((ncols, 2));
        for (k, &ni) in elem.iter().enumerate() {
            e_nodes[[k, 0]] = cm.nodes[[ni, 0]];
            e_nodes[[k, 1]] = cm.nodes[[ni, 1]];
            e_disps[[k, 0]] = cm.displacements[[ni, 0]];
            e_disps[[k, 1]] = cm.displacements[[ni, 1]];
        }

        let mut warp_inc =
            warp_increment(coord, e_nodes.view(), e_disps.view(), cm.mesh_order);

        // Clip strain components to [-0.99, 0.99] (matches Python)
        for i in 2..warp_inc.len() {
            warp_inc[i] = warp_inc[i].clamp(-0.99, 0.99);
        }

        // Store increment
        for j in 0..self.incs.ncols() {
            self.incs[[m + 1, j]] = warp_inc[j];
        }

        // --- Update coordinate ---
        let dx = warp_inc[0] * if self.track { 1.0 } else { 0.0 };
        let dy = warp_inc[1] * if self.track { 1.0 } else { 0.0 };
        self.coordinates[[m + 1, 0]] =
            self.coordinates[[self.reference_index, 0]] + dx;
        self.coordinates[[m + 1, 1]] =
            self.coordinates[[self.reference_index, 1]] + dy;

        // --- Update warps ---
        let ri = self.reference_index;
        self.warps[[m + 1, 0]] = self.warps[[ri, 0]] + warp_inc[0];
        self.warps[[m + 1, 1]] = self.warps[[ri, 1]] + warp_inc[1];
        // Logarithmic accumulation for normal strains (du/dx, dv/dy)
        self.warps[[m + 1, 2]] =
            self.warps[[ri, 2]] + (1.0 + warp_inc[2]).ln();
        self.warps[[m + 1, 3]] = self.warps[[ri, 3]] + warp_inc[3];
        self.warps[[m + 1, 4]] = self.warps[[ri, 4]] + warp_inc[4];
        self.warps[[m + 1, 5]] =
            self.warps[[ri, 5]] + (1.0 + warp_inc[5]).ln();
        // Higher-order terms (linear accumulation)
        let p_len = self.warps.ncols();
        for j in 6..p_len {
            self.warps[[m + 1, j]] = self.warps[[ri, j]] + warp_inc[j];
        }

        // --- Update volume: J = (1+du/dx)(1+dv/dy) - (dv/dx)(du/dy) ---
        self.volumes[m + 1] = self.volumes[ri]
            * ((1.0 + warp_inc[2]) * (1.0 + warp_inc[5])
                - warp_inc[3] * warp_inc[4]);

        true
    }

    /// Solve for all increments over a sequence of meshes.
    ///
    /// Replicates `Particle._strain_path_full` (minus stress path, minus alive_bar).
    ///
    /// # Arguments
    /// * `meshes` — one entry per increment (not per frame);
    ///              `meshes[m]` supplies the DIC solve between frame `m` and
    ///              frame `m+1`.  Length must equal `inc_no - 1`.
    /// * `cfg`    — solve configuration
    pub fn solve(
        &mut self,
        meshes: &[MeshData<'_>],
        cfg: &ParticleConfig,
    ) -> Result<ParticleSolution, Error> {
        let expected = self.inc_no - 1;
        if meshes.len() != expected {
            return Err(Error::InvalidInput(format!(
                "expected {} meshes for {} increments, got {}",
                expected,
                self.inc_no,
                meshes.len()
            )));
        }

        for m in 0..expected {
            self.solve_increment(m, &meshes[m], false);
        }

        self.solved = true;
        Ok(self.finalize(cfg))
    }

    /// Compute strains from accumulated warps and return a [`ParticleSolution`].
    pub fn finalize(&self, cfg: &ParticleConfig) -> ParticleSolution {
        let (strains, strain_incs) =
            strain_def(&self.warps, cfg.factor, cfg.true_incs);
        let vs = vol_strains(&self.volumes);

        ParticleSolution {
            coordinates: self.coordinates.clone(),
            warps: self.warps.clone(),
            incs: self.incs.clone(),
            volumes: self.volumes.clone(),
            strains,
            strain_incs,
            vol_strains: vs,
            reference_update_register: self.reference_update_register.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests (Phase 2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Unit right triangle: (0,0), (1,0), (0,1)
    fn tri_nodes() -> Array2<f64> {
        array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    }

    // -----------------------------------------------------------------------
    // local_coordinates — Tier A (atol = 1e-12)
    // -----------------------------------------------------------------------

    #[test]
    fn test_local_coordinates_centroid() {
        let coord = [1.0 / 3.0, 1.0 / 3.0];
        let nodes = tri_nodes();
        let (z, e, t, _) = local_coordinates(coord, nodes.view());
        assert!((z - 1.0 / 3.0).abs() < 1e-12, "zeta={z}");
        assert!((e - 1.0 / 3.0).abs() < 1e-12, "eta={e}");
        assert!((t - 1.0 / 3.0).abs() < 1e-12, "theta={t}");
    }

    #[test]
    fn test_local_coordinates_vertex0() {
        let nodes = tri_nodes();
        let (z, e, t, _) = local_coordinates([0.0, 0.0], nodes.view());
        assert!((z - 1.0).abs() < 1e-12);
        assert!(e.abs() < 1e-12);
        assert!(t.abs() < 1e-12);
    }

    #[test]
    fn test_local_coordinates_vertex1() {
        let nodes = tri_nodes();
        let (z, e, t, _) = local_coordinates([1.0, 0.0], nodes.view());
        assert!(z.abs() < 1e-12);
        assert!((e - 1.0).abs() < 1e-12);
        assert!(t.abs() < 1e-12);
    }

    #[test]
    fn test_local_coordinates_partition_of_unity() {
        let nodes = tri_nodes();
        let (z, e, t, _) = local_coordinates([0.3, 0.25], nodes.view());
        assert!((z + e + t - 1.0).abs() < 1e-12, "sum={}", z + e + t);
    }

    #[test]
    fn test_local_coordinates_interior_point() {
        // (0.25, 0.25) → zeta=0.5, eta=0.25, theta=0.25 (analytically)
        let nodes = tri_nodes();
        let (z, e, t, _) = local_coordinates([0.25, 0.25], nodes.view());
        assert!((z - 0.5).abs() < 1e-12,  "zeta={z}");
        assert!((e - 0.25).abs() < 1e-12, "eta={e}");
        assert!((t - 0.25).abs() < 1e-12, "theta={t}");
    }

    // -----------------------------------------------------------------------
    // shape_function — Tier A (atol = 1e-12)
    // -----------------------------------------------------------------------

    #[test]
    fn test_shape_function_order1_centroid() {
        let (n, dn, d2n) = shape_function(1, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        assert!((n[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((n[1] - 1.0 / 3.0).abs() < 1e-12);
        assert!((n[2] - 1.0 / 3.0).abs() < 1e-12);
        assert!(d2n.is_none());
        // dN row-0: [1, 0, -1]
        assert!((dn[[0, 0]] - 1.0).abs() < 1e-12);
        assert!(dn[[0, 1]].abs() < 1e-12);
        assert!((dn[[0, 2]] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_shape_function_order1_vertex() {
        let (n, _, _) = shape_function(1, 1.0, 0.0, 0.0);
        assert!((n[0] - 1.0).abs() < 1e-12);
        assert!(n[1].abs() < 1e-12);
        assert!(n[2].abs() < 1e-12);
    }

    #[test]
    fn test_shape_function_order2_centroid() {
        let (n, _, _) = shape_function(2, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        let expected = [-1.0 / 9.0, -1.0 / 9.0, -1.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((n[i] - exp).abs() < 1e-12, "N[{i}]={}, expected={exp}", n[i]);
        }
    }

    #[test]
    fn test_shape_function_order2_partition_of_unity() {
        let (n, _, _) = shape_function(2, 0.4, 0.35, 0.25);
        assert!((n.sum() - 1.0).abs() < 1e-12, "sum={}", n.sum());
    }

    #[test]
    fn test_shape_function_order2_vertex() {
        // Corner node 0: zeta=1, eta=0, theta=0 → N=[1,0,0,0,0,0]
        let (n, _, _) = shape_function(2, 1.0, 0.0, 0.0);
        assert!((n[0] - 1.0).abs() < 1e-12);
        for i in 1..6 {
            assert!(n[i].abs() < 1e-12, "N[{i}]={}", n[i]);
        }
    }

    #[test]
    fn test_shape_function_order2_midpoint() {
        // Midpoint of edge 0-1: zeta=0.5, eta=0.5, theta=0 → N=[0,0,0,1,0,0]
        let (n, _, _) = shape_function(2, 0.5, 0.5, 0.0);
        let expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((n[i] - exp).abs() < 1e-12, "N[{i}]={}, expected={exp}", n[i]);
        }
    }

    // -----------------------------------------------------------------------
    // warp_increment — Tier A (atol = 1e-12)
    // -----------------------------------------------------------------------

    #[test]
    fn test_warp_increment_pure_translation_order1() {
        // Uniform (u=3.5, v=-2.1): displacement=(u,v), all strain=0
        let nodes = tri_nodes();
        let disps = array![[3.5, -2.1], [3.5, -2.1], [3.5, -2.1]];
        let w = warp_increment([1.0/3.0, 1.0/3.0], nodes.view(), disps.view(), 1);
        assert!((w[0] - 3.5).abs() < 1e-12,  "w[0]={}", w[0]);
        assert!((w[1] + 2.1).abs() < 1e-12,  "w[1]={}", w[1]);
        for i in 2..6 { assert!(w[i].abs() < 1e-12, "w[{i}]={}", w[i]); }
    }

    #[test]
    fn test_warp_increment_x_stretch_order1() {
        // Node 1 displaced +1 in x → warp=[1/3, 0, 1, 0, 0, 0]
        let nodes = tri_nodes();
        let disps = array![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]];
        let w = warp_increment([1.0/3.0, 1.0/3.0], nodes.view(), disps.view(), 1);
        assert!((w[0] - 1.0/3.0).abs() < 1e-12, "w[0]={}", w[0]);
        assert!(w[1].abs() < 1e-12);
        assert!((w[2] - 1.0).abs() < 1e-12,     "w[2]={}", w[2]);  // du/dx
        for i in 3..6 { assert!(w[i].abs() < 1e-12, "w[{i}]={}", w[i]); }
    }

    #[test]
    fn test_warp_increment_pure_translation_order2() {
        // Uniform translation on all 6 nodes of element 0
        let nodes = array![
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        ];
        // All nodes same displacement → strain = 0
        let disps = array![
            [1.5, 2.5], [1.5, 2.5], [1.5, 2.5],
            [1.5, 2.5], [1.5, 2.5], [1.5, 2.5],
        ];
        let w = warp_increment([1.0/3.0, 1.0/3.0], nodes.view(), disps.view(), 2);
        assert!((w[0] - 1.5).abs() < 1e-11, "w[0]={}", w[0]);
        assert!((w[1] - 2.5).abs() < 1e-11, "w[1]={}", w[1]);
        for i in 2..12 { assert!(w[i].abs() < 1e-11, "w[{i}]={}", w[i]); }
    }

    // -----------------------------------------------------------------------
    // element_locator
    // -----------------------------------------------------------------------

    fn make_mesh_o1() -> (Array2<f64>, Array2<usize>, Array2<f64>) {
        let nodes = array![[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]];
        let elems = array![[0usize,1,2],[1,3,2]];
        let centroids = compute_centroids(&nodes, &elems);
        (nodes, elems, centroids)
    }

    #[test]
    fn test_element_locator_centroid_elem0() {
        let (nodes, elems, centroids) = make_mesh_o1();
        let coord = [centroids[[0, 0]], centroids[[0, 1]]];
        assert_eq!(element_locator(coord, &nodes, &elems, &centroids), 0);
    }

    #[test]
    fn test_element_locator_centroid_elem1() {
        let (nodes, elems, centroids) = make_mesh_o1();
        let coord = [centroids[[1, 0]], centroids[[1, 1]]];
        assert_eq!(element_locator(coord, &nodes, &elems, &centroids), 1);
    }

    #[test]
    fn test_element_locator_interior_point() {
        let (nodes, elems, centroids) = make_mesh_o1();
        assert_eq!(element_locator([0.2, 0.2], &nodes, &elems, &centroids), 0);
        assert_eq!(element_locator([0.8, 0.8], &nodes, &elems, &centroids), 1);
    }

    #[test]
    fn test_element_locator_exterior_fallback() {
        // (2,2) outside mesh → nearest centroid is element 1
        let (nodes, elems, centroids) = make_mesh_o1();
        assert_eq!(element_locator([2.0, 2.0], &nodes, &elems, &centroids), 1);
    }

    // -----------------------------------------------------------------------
    // strain_def — Tier A (atol=1e-12) for zero cases, Tier B otherwise
    // -----------------------------------------------------------------------

    #[test]
    fn test_strain_def_all_zeros() {
        let warps = Array2::<f64>::zeros((3, 6));
        let (strains, incs) = strain_def(&warps, 1.0, true);
        assert!(strains.iter().all(|&x| x == 0.0));
        assert!(incs.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_strain_def_eps_xx() {
        // warps[:,2] = [0, 0.1, 0.2] → strains[:,0] = [0, -0.1, -0.2] before correction
        let mut warps = Array2::<f64>::zeros((3, 6));
        warps[[1, 2]] = 0.1;
        warps[[2, 2]] = 0.2;
        let (strains, _) = strain_def(&warps, 0.0, true);
        assert!((strains[[1, 0]] + 0.1).abs() < 1e-12, "strains[1,0]={}", strains[[1,0]]);
        assert!((strains[[2, 0]] + 0.2).abs() < 1e-12, "strains[2,0]={}", strains[[2,0]]);
    }

    #[test]
    fn test_strain_def_factor_one_removes_mean() {
        // eps_xx_inc = -0.1, eps_yy_inc = -0.05 per step; mean = -0.075
        // After factor=1 correction: eps_xx_inc = -0.025, eps_yy_inc = +0.025
        let mut warps = Array2::<f64>::zeros((3, 6));
        warps[[1, 2]] = 0.1;  warps[[2, 2]] = 0.2;
        warps[[1, 5]] = 0.05; warps[[2, 5]] = 0.10;
        let (_, incs) = strain_def(&warps, 1.0, true);
        assert!((incs[[0, 0]] + 0.025).abs() < 1e-10, "incs[0,0]={}", incs[[0,0]]);
        assert!((incs[[0, 1]] - 0.025).abs() < 1e-10, "incs[0,1]={}", incs[[0,1]]);
    }

    #[test]
    fn test_strain_def_cumsum_consistency() {
        // strains[1:,0] must equal cumsum of strain_incs[:,0]
        let mut warps = Array2::<f64>::zeros((4, 6));
        warps[[1, 2]] = 0.05; warps[[2, 2]] = 0.12; warps[[3, 2]] = 0.20;
        let (strains, incs) = strain_def(&warps, 0.0, true);
        let cumsum: f64 = incs.column(0).iter().take(1).sum();
        assert!((strains[[1, 0]] - cumsum).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Particle struct — integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_particle_new() {
        let p = Particle::new([10.0, 20.0], &[0.0; 6], 1e9, 3, 1, true).unwrap();
        assert_eq!(p.inc_no, 3);
        assert_eq!(p.mesh_order, 1);
        assert!(p.track);
        assert_eq!(p.coordinates.nrows(), 3);
        assert!((p.coordinates[[0, 0]] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_particle_solve_increment_pure_translation() {
        // Mesh with uniform displacement (u=0.5, v=0.1) on all 4 nodes.
        // Particle at centroid of elem 0 should acquire warp_inc = [0.5, 0.1, 0,0,0,0].
        let nodes = array![[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]];
        let elements = array![[0usize,1,2],[1,3,2]];
        let disps = array![[0.5,0.1],[0.5,0.1],[0.5,0.1],[0.5,0.1]];
        let cm = MeshData {
            nodes: &nodes,
            elements: &elements,
            displacements: &disps,
            mesh_order: 1,
        };

        let mut p = Particle::new(
            [1.0/3.0, 1.0/3.0],
            &[0.0; 6],
            1e9,
            2,   // 2 frames → 1 increment
            1,
            true,
        ).unwrap();
        assert!(p.solve_increment(0, &cm, false));

        // Coordinate should have moved by (0.5, 0.1) in Lagrangian mode
        assert!((p.coordinates[[1, 0]] - (1.0/3.0 + 0.5)).abs() < 1e-10,
            "x={}", p.coordinates[[1,0]]);
        assert!((p.coordinates[[1, 1]] - (1.0/3.0 + 0.1)).abs() < 1e-10,
            "y={}", p.coordinates[[1,1]]);
        // Strain components should be ~0 (pure translation)
        for i in 2..6 {
            assert!(p.incs[[1, i]].abs() < 1e-10, "incs[1,{i}]={}", p.incs[[1,i]]);
        }
    }

    #[test]
    fn test_particle_solve_sequence() {
        // Two increments of uniform translation → warps accumulate correctly.
        let nodes = array![[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]];
        let elements = array![[0usize,1,2],[1,3,2]];
        let disps = array![[0.1,0.0],[0.1,0.0],[0.1,0.0],[0.1,0.0]];
        let m = MeshData {
            nodes: &nodes, elements: &elements,
            displacements: &disps, mesh_order: 1,
        };

        let mut p = Particle::new([0.3, 0.3], &[0.0;6], 1e6, 3, 1, true).unwrap();
        let sol = p.solve(&[m], &ParticleConfig::default());
        // solve expects 2 meshes for inc_no=3; we only supplied 1 → should error
        assert!(sol.is_err());
    }
}
