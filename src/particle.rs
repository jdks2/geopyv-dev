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

use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::{
    calibration::CalibrationParams,
    mesh::MeshSolution,
    sequence::SequenceSolution,
    Error,
};

// ---------------------------------------------------------------------------
// ParticleSource — owned reference to mesh data
// ---------------------------------------------------------------------------

/// Source of mesh data for particle strain-path computation.
pub enum ParticleSource {
    Mesh(Arc<MeshSolution>),
    Sequence(Arc<SequenceSolution>),
    /// Reconstructed from a saved solution; mesh data is not available.
    Loaded { inc_no: usize, mesh_order: u8 },
}

impl ParticleSource {
    pub fn inc_no(&self) -> usize {
        match self {
            Self::Mesh(_) => 2,
            Self::Sequence(s) => s.n_meshes() + 1,
            Self::Loaded { inc_no, .. } => *inc_no,
        }
    }
    pub fn mesh_order(&self) -> u8 {
        match self {
            Self::Mesh(m) => m.mesh_order,
            Self::Sequence(s) => s.mesh_order,
            Self::Loaded { mesh_order, .. } => *mesh_order,
        }
    }
    pub fn image_0_path(&self) -> Option<&PathBuf> {
        match self {
            Self::Mesh(m) => Some(&m.f_img_path),
            Self::Sequence(s) => s.first_f_img_path.as_ref(),
            Self::Loaded { .. } => None,
        }
    }
    /// Load the mesh for increment `m`.
    ///
    /// For `Mesh` sources this clones the single in-memory solution.
    /// For `Sequence` sources this delegates to [`SequenceSolution::load_mesh_at`],
    /// which either clones from memory or deserialises from disk.
    pub fn load_mesh_at(&self, m: usize) -> Result<Arc<MeshSolution>, Error> {
        match self {
            Self::Mesh(ms) => Ok(Arc::clone(ms)),
            Self::Sequence(s) => s.load_mesh_at(m),
            Self::Loaded { .. } => Err(Error::InvalidInput(
                "cannot load mesh increments from a loaded Particle".to_string(),
            )),
        }
    }
    /// Number of mesh increments (1 for a single `Mesh` source; `n_pairs` for a sequence).
    pub fn n_meshes(&self) -> usize {
        match self {
            Self::Mesh(_) => 1,
            Self::Sequence(s) => s.n_meshes(),
            Self::Loaded { inc_no, .. } => inc_no.saturating_sub(1),
        }
    }
    pub fn ref_update_at(&self, m: usize) -> bool {
        match self {
            Self::Mesh(_) => false,
            Self::Sequence(s) => s.reference_updates.get(m).copied().unwrap_or(false),
            Self::Loaded { .. } => false,
        }
    }
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

/// Find the index of the element containing `coordinate`.
///
/// Replicates `Particle._element_locator`.
///
/// Searches nearest centroids first, testing each with the barycentric
/// sign test.  Falls back to the nearest centroid if no element contains
/// the point (adrift case).
/// Returns `(element_index, adrift)`.  `adrift` is `true` when the coordinate
/// falls outside all mesh elements and the nearest centroid was used as fallback.
pub fn element_locator(
    coordinate: [f64; 2],
    nodes: &Array2<f64>,
    elements: &Array2<usize>,
    centroids: &Array2<f64>,
) -> (usize, bool) {
    let n_elem = centroids.nrows();
    let mut dists: Vec<(f64, usize)> = (0..n_elem)
        .map(|i| {
            let dx = centroids[[i, 0]] - coordinate[0];
            let dy = centroids[[i, 1]] - coordinate[1];
            (dx * dx + dy * dy, i)
        })
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for &(_, idx) in dists.iter() {
        if point_in_triangle(coordinate, nodes, elements, idx) {
            return (idx, false);
        }
    }
    (dists[0].1, true)
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleConfig {
    /// Volumetric correction factor applied in `strain_def`:
    /// `1.0` → full isotropic removal, `0.0` → no correction.
    pub factor: f64,
    /// Use true (logarithmic) strain increments.
    pub true_incs: bool,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self { factor: 0.0, true_incs: true }
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
    /// Path of the initial (reference) image.
    #[serde(default)]
    pub image_0_path: Option<PathBuf>,
    /// Whether calibration was applied during solve.
    #[serde(default)]
    pub calibrated: bool,
    /// Strain-computation settings used to produce this solution.
    #[serde(default)]
    pub config: Option<ParticleConfig>,
}

/// Lagrangian / Eulerian particle tracking and strain-path computation.
///
/// Construct with [`Particle::new`]; then call [`Particle::solve`].
pub struct Particle {
    pub source: ParticleSource,
    pub track: bool,

    // --- Accumulated state arrays ---
    pub coordinates: Array2<f64>,
    pub warps: Array2<f64>,
    pub incs: Array2<f64>,
    pub volumes: Array1<f64>,

    // --- Reference tracking ---
    reference_index: usize,
    pub reference_update_register: Vec<usize>,
    pub _adrift: bool,

    pub(crate) solution: Option<Arc<ParticleSolution>>,
}

impl Particle {
    /// Construct a new particle.
    ///
    /// All metadata (`inc_no`, `mesh_order`, `image_0_path`) are derived from `source`.
    pub fn new(
        source: ParticleSource,
        coordinate: [f64; 2],
        initial_warp: &[f64],
        initial_volume: f64,
        track: bool,
    ) -> Result<Self, Error> {
        if initial_volume <= 0.0 {
            return Err(Error::InvalidInput(
                "initial_volume must be > 0".to_string(),
            ));
        }
        let inc_no = source.inc_no();
        let mesh_order = source.mesh_order();
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
            source,
            track,
            coordinates,
            warps,
            incs: Array2::zeros((inc_no, p_len)),
            volumes,
            reference_index: 0,
            reference_update_register: Vec::new(),
            _adrift: false,
            solution: None,
        })
    }

    pub fn inc_no(&self) -> usize { self.source.inc_no() }
    pub fn mesh_order(&self) -> u8 { self.source.mesh_order() }
    pub fn image_0_path(&self) -> Option<&PathBuf> {
        if let Some(sol) = &self.solution {
            if sol.image_0_path.is_some() { return sol.image_0_path.as_ref(); }
        }
        self.source.image_0_path()
    }
    pub fn solved(&self) -> bool { self.solution.is_some() }
    pub fn solution(&self) -> Option<&Arc<ParticleSolution>> { self.solution.as_ref() }

    /// Reconstruct a `Particle` shell from a saved [`ParticleSolution`].
    ///
    /// The resulting particle is marked as solved and all data arrays are
    /// populated from the solution.  Mesh data is not available (`source` is
    /// set to [`ParticleSource::Loaded`]).
    pub fn from_solution(sol: ParticleSolution) -> Self {
        let inc_no = sol.coordinates.nrows();
        let mesh_order = if sol.warps.ncols() > 0 { (sol.warps.ncols() / 6) as u8 } else { 1 };
        Particle {
            source: ParticleSource::Loaded { inc_no, mesh_order },
            track: true,
            coordinates: sol.coordinates.clone(),
            warps: sol.warps.clone(),
            incs: sol.incs.clone(),
            volumes: sol.volumes.clone(),
            reference_index: 0,
            reference_update_register: sol.reference_update_register.clone(),
            _adrift: false,
            solution: Some(Arc::new(sol)),
        }
    }

    /// Solve a single increment.  Reads mesh data and ref_update from `self.source`.
    ///
    /// # Arguments
    /// * `m` — increment index (0-based; result stored at `m+1`)
    ///
    /// Returns `true` on success.
    /// Advance the particle by one increment using the supplied mesh data.
    ///
    /// The caller is responsible for loading (or borrowing) the correct mesh for
    /// increment `m` — this decouples loading from solving and enables the
    /// saved-by-reference path where only one mesh file is live at a time.
    pub fn solve_increment(&mut self, m: usize, mesh: &MeshSolution, calibration: Option<&CalibrationParams>) -> bool {
        let ref_update = self.source.ref_update_at(m);
        if ref_update {
            self.reference_index = m;
            self.reference_update_register.push(m);
        }

        let mut warp_inc = if let Some(params) = calibration {
            // coord is in object space (maintained throughout calibrated solve).
            let coord_obj = [
                self.coordinates[[self.reference_index, 0]],
                self.coordinates[[self.reference_index, 1]],
            ];

            // Map back to image space for element_locator (mesh data is image-space).
            let coord_img_arr = params.o2i(ndarray::array![[coord_obj[0], coord_obj[1]]].view());
            let coord_img = [coord_img_arr[[0, 0]], coord_img_arr[[0, 1]]];

            let (tri_idx, adrift) = element_locator(coord_img, &mesh.nodes, &mesh.elements, &mesh.centroids);
            self._adrift = adrift;

            let elem = mesh.elements.row(tri_idx);
            let ncols = mesh.elements.ncols();
            let mut e_nodes_img = Array2::<f64>::zeros((ncols, 2));
            let mut e_displaced_img = Array2::<f64>::zeros((ncols, 2));
            for (k, &ni) in elem.iter().enumerate() {
                e_nodes_img[[k, 0]] = mesh.nodes[[ni, 0]];
                e_nodes_img[[k, 1]] = mesh.nodes[[ni, 1]];
                e_displaced_img[[k, 0]] = mesh.nodes[[ni, 0]] + mesh.displacements[[ni, 0]];
                e_displaced_img[[k, 1]] = mesh.nodes[[ni, 1]] + mesh.displacements[[ni, 1]];
            }

            // Convert element nodes and displaced counterparts to object space (6–12 i2o calls).
            let e_nodes_obj = params.i2o(e_nodes_img.view());
            let e_displaced_obj = params.i2o(e_displaced_img.view());
            let e_disps_obj = e_displaced_obj - &e_nodes_obj;

            warp_increment(coord_obj, e_nodes_obj.view(), e_disps_obj.view(), mesh.mesh_order)
        } else {
            let coord = [
                self.coordinates[[self.reference_index, 0]],
                self.coordinates[[self.reference_index, 1]],
            ];
            let (tri_idx, adrift) = element_locator(coord, &mesh.nodes, &mesh.elements, &mesh.centroids);
            self._adrift = adrift;

            let elem = mesh.elements.row(tri_idx);
            let ncols = mesh.elements.ncols();
            let mut e_nodes = Array2::<f64>::zeros((ncols, 2));
            let mut e_disps = Array2::<f64>::zeros((ncols, 2));
            for (k, &ni) in elem.iter().enumerate() {
                e_nodes[[k, 0]] = mesh.nodes[[ni, 0]];
                e_nodes[[k, 1]] = mesh.nodes[[ni, 1]];
                e_disps[[k, 0]] = mesh.displacements[[ni, 0]];
                e_disps[[k, 1]] = mesh.displacements[[ni, 1]];
            }

            warp_increment(coord, e_nodes.view(), e_disps.view(), mesh.mesh_order)
        };

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

    /// Solve for all increments using mesh data from `self.source`.
    ///
    /// Meshes are loaded one at a time; for saved-by-reference sequences each
    /// mesh file is read, used for its single increment, then dropped before
    /// the next file is opened.
    ///
    /// On completion stores the result internally.  Access via `self.solution()`.
    pub fn solve(&mut self, cfg: &ParticleConfig, calibration: Option<&CalibrationParams>) -> Result<(), Error> {
        if let Some(params) = calibration {
            let img = ndarray::array![[self.coordinates[[0, 0]], self.coordinates[[0, 1]]]];
            let obj = params.i2o(img.view());
            self.coordinates[[0, 0]] = obj[[0, 0]];
            self.coordinates[[0, 1]] = obj[[0, 1]];
        }
        let n = self.source.n_meshes();
        for m in 0..n {
            let mesh = self.source.load_mesh_at(m)?;
            self.solve_increment(m, &mesh, calibration);
        }
        let mut sol = self.finalize(cfg);
        sol.calibrated = calibration.is_some();
        self.solution = Some(Arc::new(sol));
        Ok(())
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
            image_0_path: self.image_0_path().cloned(),
            calibrated: false,
            config: Some(cfg.clone()),
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
        let centroids = crate::mesh::compute_centroids(&nodes, &elems);
        (nodes, elems, centroids)
    }

    #[test]
    fn test_element_locator_centroid_elem0() {
        let (nodes, elems, centroids) = make_mesh_o1();
        let coord = [centroids[[0, 0]], centroids[[0, 1]]];
        assert_eq!(element_locator(coord, &nodes, &elems, &centroids), (0, false));
    }

    #[test]
    fn test_element_locator_centroid_elem1() {
        let (nodes, elems, centroids) = make_mesh_o1();
        let coord = [centroids[[1, 0]], centroids[[1, 1]]];
        assert_eq!(element_locator(coord, &nodes, &elems, &centroids), (1, false));
    }

    #[test]
    fn test_element_locator_interior_point() {
        let (nodes, elems, centroids) = make_mesh_o1();
        assert_eq!(element_locator([0.2, 0.2], &nodes, &elems, &centroids), (0, false));
        assert_eq!(element_locator([0.8, 0.8], &nodes, &elems, &centroids), (1, false));
    }

    #[test]
    fn test_element_locator_exterior_fallback() {
        // (2,2) outside mesh → nearest centroid is element 1, adrift=true
        let (nodes, elems, centroids) = make_mesh_o1();
        assert_eq!(element_locator([2.0, 2.0], &nodes, &elems, &centroids), (1, true));
    }

    #[test]
    fn test_element_locator_adrift_flag() {
        let (nodes, elems, centroids) = make_mesh_o1();
        let (_, adrift_inside) = element_locator([0.2, 0.2], &nodes, &elems, &centroids);
        let (_, adrift_outside) = element_locator([5.0, 5.0], &nodes, &elems, &centroids);
        assert!(!adrift_inside);
        assert!(adrift_outside);
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

    fn make_mesh_sol_o1(disps: Array2<f64>) -> crate::mesh::MeshSolution {
        use ndarray::{Array1, Array2};
        use std::path::PathBuf;
        let nodes = array![[0.0f64,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]];
        let elements = array![[0usize,1,2],[1,3,2]];
        let centroids = crate::mesh::compute_centroids(&nodes, &elements);
        let n = 4;
        crate::mesh::MeshSolution {
            nodes,
            elements,
            boundary: vec![0, 1, 3, 2],
            exclusions: vec![],
            centroids,
            areas: Array1::from_vec(vec![0.5, 0.5]),
            warps: Array2::zeros((2, 6)),
            displacements: disps,
            c_zncc: Array1::ones(n),
            p: Array2::zeros((n, 6)),
            seed_node: 0,
            mesh_order: 1,
            subset_order: 1,
            iterations: Array1::zeros(n),
            norms: Array1::zeros(n),
            f_img_path: PathBuf::new(),
            g_img_path: PathBuf::new(),
            solve_config: None,
            seed: None,
        }
    }

    fn make_seq_o1(disps_list: Vec<Array2<f64>>) -> Arc<crate::sequence::SequenceSolution> {
        let n_pairs = disps_list.len();
        let mesh_solutions: Vec<_> = disps_list.into_iter().map(|d| Arc::new(make_mesh_sol_o1(d))).collect();
        Arc::new(crate::sequence::SequenceSolution {
            mesh_solutions,
            mesh_paths: vec![],
            all_converged: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: vec![false; n_pairs],
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: crate::sequence::default_boundary_region(),
            exclusion_regions: vec![],
            options: None,
            border: 0,
        })
    }

    #[test]
    fn test_particle_new() {
        let seq = make_seq_o1(vec![ndarray::Array2::<f64>::zeros((4, 2))]);
        let p = Particle::new(
            ParticleSource::Sequence(seq),
            [10.0, 20.0],
            &[0.0; 6],
            1e9,
            true,
        ).unwrap();
        assert_eq!(p.inc_no(), 2);
        assert_eq!(p.mesh_order(), 1);
        assert!(p.track);
        assert_eq!(p.coordinates.nrows(), 2);
        assert!((p.coordinates[[0, 0]] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_particle_solve_increment_pure_translation() {
        // Mesh with uniform displacement (u=0.5, v=0.1) on all 4 nodes.
        // Particle at centroid of elem 0 should acquire warp_inc = [0.5, 0.1, 0,0,0,0].
        let disps = array![[0.5,0.1],[0.5,0.1],[0.5,0.1],[0.5,0.1]];
        let mesh_sol = make_mesh_sol_o1(disps);
        let source = ParticleSource::Mesh(Arc::new(mesh_sol.clone()));

        let mut p = Particle::new(
            source,
            [1.0/3.0, 1.0/3.0],
            &[0.0; 6],
            1e9,
            true,
        ).unwrap();
        assert!(p.solve_increment(0, &mesh_sol, None));

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
        // With a fixed reference (no reference updates), each step queries the mesh
        // at reference position [0.3, 0.3], so coords[m+1] = ref + disp = 0.4 both times.
        // The second step with reference_updates[1]=true advances the reference to step 1,
        // so coords[2] = coords[1] + disp = 0.5.
        let disps = array![[0.1f64,0.0],[0.1,0.0],[0.1,0.0],[0.1,0.0]];
        let mesh_sol_0 = make_mesh_sol_o1(disps.clone());
        let mesh_sol_1 = make_mesh_sol_o1(disps);
        let seq = Arc::new(crate::sequence::SequenceSolution {
            mesh_solutions: vec![Arc::new(mesh_sol_0), Arc::new(mesh_sol_1)],
            mesh_paths: vec![],
            all_converged: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: vec![false, true],  // ref update at step 1
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: crate::sequence::default_boundary_region(),
            exclusion_regions: vec![],
            options: None,
            border: 0,
        });
        let mut p = Particle::new(
            ParticleSource::Sequence(seq),
            [0.3, 0.3],
            &[0.0; 6],
            1e6,
            true,
        ).unwrap();
        p.solve(&ParticleConfig::default(), None).unwrap();
        let sol = p.solution().unwrap();
        // Step 0: ref=0, coord=[0.4,0.3]; step 1: ref advances to 1, coord=[0.5,0.3]
        assert!((sol.coordinates[[1, 0]] - 0.4).abs() < 1e-9,
            "x1={}", sol.coordinates[[1,0]]);
        assert!((sol.coordinates[[2, 0]] - 0.5).abs() < 1e-9,
            "x2={}", sol.coordinates[[2,0]]);
    }

    // -----------------------------------------------------------------------
    // Saved-by-reference: solve_increment and solve load from disk
    // -----------------------------------------------------------------------

    fn make_saved_by_ref_seq(tag: &str, disps_list: Vec<Array2<f64>>) -> (Arc<crate::sequence::SequenceSolution>, Vec<std::path::PathBuf>) {
        let paths: Vec<_> = disps_list.iter().enumerate().map(|(i, _)| {
            std::env::temp_dir().join(format!("geopyv_test_particle_{tag}_{i}.pyv"))
        }).collect();
        for (i, disps) in disps_list.iter().enumerate() {
            let ms = make_mesh_sol_o1(disps.clone());
            crate::io::save(&paths[i], &crate::io::GeopyvObject::Mesh(ms)).unwrap();
        }
        let sol = Arc::new(crate::sequence::SequenceSolution {
            mesh_solutions: vec![],
            mesh_paths: paths.clone(),
            all_converged: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: vec![false; disps_list.len()],
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: crate::sequence::default_boundary_region(),
            exclusion_regions: vec![],
            options: None,
            border: 0,
        });
        (sol, paths)
    }

    #[test]
    fn test_particle_source_inc_no_saved_by_reference() {
        let (seq, paths) = make_saved_by_ref_seq("inc_no", vec![
            Array2::<f64>::zeros((4, 2)),
            Array2::<f64>::zeros((4, 2)),
        ]);
        let src = ParticleSource::Sequence(seq);
        assert_eq!(src.inc_no(), 3);
        for p in paths { let _ = std::fs::remove_file(p); }
    }

    #[test]
    fn test_particle_source_mesh_order_saved_by_reference() {
        let (seq, paths) = make_saved_by_ref_seq("mesh_order", vec![Array2::<f64>::zeros((4, 2))]);
        let src = ParticleSource::Sequence(seq);
        assert_eq!(src.mesh_order(), 1);
        for p in paths { let _ = std::fs::remove_file(p); }
    }

    #[test]
    fn test_particle_solve_increment_saved_by_reference() {
        let disps = array![[0.3f64, 0.0], [0.3, 0.0], [0.3, 0.0], [0.3, 0.0]];
        let (seq, paths) = make_saved_by_ref_seq("si_sbr", vec![disps]);

        let mut p = Particle::new(
            ParticleSource::Sequence(seq),
            [1.0/3.0, 1.0/3.0],
            &[0.0; 6],
            1e9,
            true,
        ).unwrap();
        assert_eq!(p.inc_no(), 2);

        let mesh = p.source.load_mesh_at(0).unwrap();
        assert!(p.solve_increment(0, &mesh, None));
        assert!((p.coordinates[[1, 0]] - (1.0/3.0 + 0.3)).abs() < 1e-9,
            "x={}", p.coordinates[[1, 0]]);

        for path in paths { let _ = std::fs::remove_file(path); }
    }

    #[test]
    fn test_particle_solve_saved_by_reference() {
        // Same geometry as test_particle_solve_sequence but meshes are on disk.
        // reference_updates[1]=true advances the reference after step 0, so
        // coords[2] = coords[1] + disp = 0.5 (not 0.4).
        let disps_0 = array![[0.1f64, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0]];
        let disps_1 = array![[0.1f64, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0]];
        let (seq_base, paths) = make_saved_by_ref_seq("solve_sbr", vec![disps_0, disps_1]);

        // Reconstruct with reference update at step 1.
        let seq = Arc::new(crate::sequence::SequenceSolution {
            mesh_solutions: vec![],
            mesh_paths: seq_base.mesh_paths.clone(),
            all_converged: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: vec![false, true],
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: crate::sequence::default_boundary_region(),
            exclusion_regions: vec![],
            options: None,
            border: 0,
        });

        let mut p = Particle::new(
            ParticleSource::Sequence(seq),
            [0.3, 0.3],
            &[0.0; 6],
            1e6,
            true,
        ).unwrap();
        assert_eq!(p.inc_no(), 3);
        p.solve(&ParticleConfig::default(), None).unwrap();

        let sol = p.solution().unwrap();
        assert_eq!(sol.coordinates.nrows(), 3);
        assert!((sol.coordinates[[1, 0]] - 0.4).abs() < 1e-9,
            "x1={}", sol.coordinates[[1, 0]]);
        assert!((sol.coordinates[[2, 0]] - 0.5).abs() < 1e-9,
            "x2={}", sol.coordinates[[2, 0]]);

        for path in paths { let _ = std::fs::remove_file(path); }
    }

    // -----------------------------------------------------------------------
    // Calibration integration
    // -----------------------------------------------------------------------

    /// A camera model whose net image<->object mapping is exactly the identity
    /// (intmat's focal length equals extmat's depth, zero principal point,
    /// zero rotation/lateral-translation/distortion). `extmat` cannot literally
    /// be the identity matrix — that puts the camera exactly on the z=0 object
    /// plane (division by zero) — so the depth (extmat[2,3]) and focal length
    /// must match instead.
    fn identity_calibration() -> CalibrationParams {
        let intmat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut extmat = ndarray::Array2::<f64>::eye(4);
        extmat[[2, 3]] = 1.0;
        CalibrationParams::new(intmat, extmat, [0.0; 5]).unwrap()
    }

    /// Same as `identity_calibration` but with the focal length doubled, so
    /// i2o(imgpt) == 0.5 * imgpt for every point (a pure uniform scale, not an
    /// identity map).
    fn half_scale_calibration() -> CalibrationParams {
        let intmat = array![[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let mut extmat = ndarray::Array2::<f64>::eye(4);
        extmat[[2, 3]] = 1.0;
        CalibrationParams::new(intmat, extmat, [0.0; 5]).unwrap()
    }

    #[test]
    fn test_particle_calibrated_flag_true() {
        let disps = array![[0.5f64, 0.1], [0.5, 0.1], [0.5, 0.1], [0.5, 0.1]];
        let seq = make_seq_o1(vec![disps]);
        let cal = identity_calibration();
        let mut p = Particle::new(ParticleSource::Sequence(seq), [1.0/3.0, 1.0/3.0], &[0.0; 6], 1e9, true).unwrap();
        p.solve(&ParticleConfig::default(), Some(&cal)).unwrap();
        assert!(p.solution().unwrap().calibrated);
    }

    #[test]
    fn test_particle_calibrated_flag_false() {
        let disps = array![[0.5f64, 0.1], [0.5, 0.1], [0.5, 0.1], [0.5, 0.1]];
        let seq = make_seq_o1(vec![disps]);
        let mut p = Particle::new(ParticleSource::Sequence(seq), [1.0/3.0, 1.0/3.0], &[0.0; 6], 1e9, true).unwrap();
        p.solve(&ParticleConfig::default(), None).unwrap();
        assert!(!p.solution().unwrap().calibrated);
    }

    #[test]
    fn test_particle_calibrate_identity() {
        // An identity-mapping camera should reproduce the uncalibrated solve
        // exactly — coordinates and warps in "object space" are numerically
        // identical to pixel space.
        let disps = array![[0.5f64, 0.1], [0.5, 0.1], [0.5, 0.1], [0.5, 0.1]];

        let seq_a = make_seq_o1(vec![disps.clone()]);
        let mut p_uncal = Particle::new(ParticleSource::Sequence(seq_a), [1.0/3.0, 1.0/3.0], &[0.0; 6], 1e9, true).unwrap();
        p_uncal.solve(&ParticleConfig::default(), None).unwrap();
        let sol_uncal = p_uncal.solution().unwrap();

        let seq_b = make_seq_o1(vec![disps]);
        let cal = identity_calibration();
        let mut p_cal = Particle::new(ParticleSource::Sequence(seq_b), [1.0/3.0, 1.0/3.0], &[0.0; 6], 1e9, true).unwrap();
        p_cal.solve(&ParticleConfig::default(), Some(&cal)).unwrap();
        let sol_cal = p_cal.solution().unwrap();

        for i in 0..sol_uncal.coordinates.nrows() {
            for j in 0..2 {
                assert!(
                    (sol_cal.coordinates[[i, j]] - sol_uncal.coordinates[[i, j]]).abs() < 1e-9,
                    "coord[{i},{j}]: {} vs {}", sol_cal.coordinates[[i, j]], sol_uncal.coordinates[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_particle_calibrate_pure_scale() {
        // A pure-uniform-scale camera should scale the particle's *displacement
        // increment* from the reference coordinate by the same factor — the
        // reference coordinate itself is also converted to object space at
        // solve start, so absolute coordinates scale too, but by the same 0.5
        // factor relative to the (also-scaled) reference.
        let disps = array![[0.5f64, 0.1], [0.5, 0.1], [0.5, 0.1], [0.5, 0.1]];
        let coord = [1.0 / 3.0, 1.0 / 3.0];

        let seq_a = make_seq_o1(vec![disps.clone()]);
        let mut p_uncal = Particle::new(ParticleSource::Sequence(seq_a), coord, &[0.0; 6], 1e9, true).unwrap();
        p_uncal.solve(&ParticleConfig::default(), None).unwrap();
        let sol_uncal = p_uncal.solution().unwrap();
        let uncal_disp = [
            sol_uncal.coordinates[[1, 0]] - sol_uncal.coordinates[[0, 0]],
            sol_uncal.coordinates[[1, 1]] - sol_uncal.coordinates[[0, 1]],
        ];

        let seq_b = make_seq_o1(vec![disps]);
        let cal = half_scale_calibration();
        let mut p_cal = Particle::new(ParticleSource::Sequence(seq_b), coord, &[0.0; 6], 1e9, true).unwrap();
        p_cal.solve(&ParticleConfig::default(), Some(&cal)).unwrap();
        let sol_cal = p_cal.solution().unwrap();
        let cal_disp = [
            sol_cal.coordinates[[1, 0]] - sol_cal.coordinates[[0, 0]],
            sol_cal.coordinates[[1, 1]] - sol_cal.coordinates[[0, 1]],
        ];

        assert!((cal_disp[0] - 0.5 * uncal_disp[0]).abs() < 1e-9, "{} vs {}", cal_disp[0], 0.5 * uncal_disp[0]);
        assert!((cal_disp[1] - 0.5 * uncal_disp[1]).abs() < 1e-9, "{} vs {}", cal_disp[1], 0.5 * uncal_disp[1]);
    }
}
