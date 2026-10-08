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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::{
    calibration::CalibrationParams,
    masks::zone_at,
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

// ---------------------------------------------------------------------------
// Meshless strain estimation
// ---------------------------------------------------------------------------

/// Fit-quality / neighbourhood-selection parameters for the meshless strain
/// estimator (see [`meshless_warp_increment`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshlessParams {
    /// Neighbourhood search radius (pixels).
    pub radius: f64,
    /// Minimum node `c_zncc` to include as a neighbour; `None` disables the
    /// quality gate.
    pub quality_gate: Option<f64>,
    /// Minimum accepted neighbours before falling back to the single
    /// nearest node's own warp.
    pub min_neighbours: usize,
    /// Number of IRLS reweighting iterations.
    pub max_iterations: usize,
    /// Tukey biweight tuning constant.
    pub tukey_c: f64,
    /// When `true` and the source mesh carries a `ZonalMaskingRecord`
    /// (`masking="zonal"` solve), restrict the meshless neighbourhood to
    /// nodes sharing the query point's own zone -- the same exclusion
    /// principle zonal masking already applies to a subset's own pixels,
    /// applied instead to which neighbour subsets a particle's meshless
    /// fit may draw on. A graceful no-op (identical to `false`) when the
    /// mesh has no zonal record at all. Default `false` -- existing
    /// callers are unaffected.
    pub zone_aware: bool,
}

impl Default for MeshlessParams {
    fn default() -> Self {
        Self {
            radius: 50.0,
            quality_gate: None,
            min_neighbours: 6,
            max_iterations: 5,
            tukey_c: 4.685,
            zone_aware: false,
        }
    }
}

/// Strain-computation method for [`Particle::solve_increment`]. Defaults to
/// `Meshless(MeshlessParams::default())` -- the same default the Python
/// boundary (`strain_method=None`) resolves to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrainMethod {
    /// Locate the containing mesh element and differentiate its shape
    /// functions ([`warp_increment`]). Kept for comparison purposes.
    Mesh,
    /// Gather nearby mesh nodes (subsets) directly and fit a robust local
    /// affine displacement field ([`meshless_warp_increment`]), ignoring
    /// mesh element connectivity entirely.
    Meshless(MeshlessParams),
}

impl Default for StrainMethod {
    fn default() -> Self {
        StrainMethod::Meshless(MeshlessParams::default())
    }
}

/// Compute the warp increment for a particle at `coordinate` from nearby
/// mesh-node subsets directly, without mesh element connectivity.
///
/// Gathers nodes within `params.radius` (optionally gated by
/// `params.quality_gate` on `c_zncc`) and fits a local affine displacement
/// field `u ≈ a0 + a1·dx + a2·dy`, `v ≈ b0 + b1·dx + b2·dy` by iteratively
/// reweighted least squares — a Gaussian spatial kernel combined with a
/// Tukey-biweight robustness weight on fit residuals — so neighbours
/// belonging to a different strain regime are progressively downweighted
/// rather than requiring an explicit discontinuity test or cluster count.
/// Degrades to a plain Gaussian-weighted fit when the neighbourhood is
/// homogeneous (residual scale ≈ 0).
///
/// Falls back to the single nearest (quality-gated) node's own `p` row
/// when fewer than `params.min_neighbours` remain, or when a fit is
/// singular (e.g. collinear neighbours).
///
/// `zonal`, when `Some((node_zone, zone_image))`, restricts the candidate
/// neighbourhood to nodes sharing the query point's own zone (looked up via
/// [`zone_at`]) -- see [`MeshlessParams::zone_aware`]. Ignored when `None`
/// (the mesh has no `ZonalMaskingRecord`, or the caller didn't ask).
///
/// # Returns
/// Warp increment vector of length `out_len` (`6 * mesh_order`), in the
/// same layout as [`warp_increment`]: `[u, v, du/dx, dv/dx, du/dy, dv/dy,
/// 0, ..., 0]` — higher-order (order-2) terms are always zero, since the
/// local fit is affine-only.
pub fn meshless_warp_increment(
    coordinate: [f64; 2],
    nodes: ArrayView2<f64>,
    p: ArrayView2<f64>,
    c_zncc: ArrayView1<f64>,
    params: &MeshlessParams,
    out_len: usize,
    zonal: Option<(ArrayView1<u32>, ArrayView2<u8>)>,
) -> Array1<f64> {
    let n_nodes = nodes.nrows();
    let r2 = params.radius * params.radius;

    let mut candidates: Vec<usize> = (0..n_nodes)
        .filter(|&i| {
            let dx = nodes[[i, 0]] - coordinate[0];
            let dy = nodes[[i, 1]] - coordinate[1];
            dx * dx + dy * dy <= r2
        })
        .collect();
    if let Some(gate) = params.quality_gate {
        candidates.retain(|&i| c_zncc[i] >= gate);
    }
    if let Some((node_zone, zone_image)) = &zonal {
        let query_zone = zone_at(zone_image.view(), coordinate) as u32;
        candidates.retain(|&i| node_zone[i] == query_zone);
    }

    if candidates.len() < params.min_neighbours {
        return nearest_node_warp(coordinate, nodes, p, c_zncc, params.quality_gate, out_len, zonal);
    }

    let dx: Vec<f64> = candidates.iter().map(|&i| nodes[[i, 0]] - coordinate[0]).collect();
    let dy: Vec<f64> = candidates.iter().map(|&i| nodes[[i, 1]] - coordinate[1]).collect();
    let u: Vec<f64> = candidates.iter().map(|&i| p[[i, 0]]).collect();
    let v: Vec<f64> = candidates.iter().map(|&i| p[[i, 1]]).collect();
    let dist: Vec<f64> = dx.iter().zip(&dy).map(|(&x, &y)| (x * x + y * y).sqrt()).collect();

    // Gaussian spatial kernel; bandwidth tied to the search radius.
    let bandwidth = (params.radius / 2.0).max(f64::EPSILON);
    let spatial_weight = |d: f64| (-0.5 * (d / bandwidth).powi(2)).exp();
    let mut weights: Vec<f64> = dist.iter().map(|&d| spatial_weight(d)).collect();

    let mut fit = match weighted_affine_fit(&dx, &dy, &u, &v, &weights) {
        Some(f) => f,
        None => return nearest_node_warp(coordinate, nodes, p, c_zncc, params.quality_gate, out_len, zonal),
    };

    for _ in 0..params.max_iterations {
        let (cu, cv) = fit;
        let resid: Vec<f64> = (0..candidates.len())
            .map(|k| {
                let ru = u[k] - (cu[0] + cu[1] * dx[k] + cu[2] * dy[k]);
                let rv = v[k] - (cv[0] + cv[1] * dx[k] + cv[2] * dy[k]);
                (ru * ru + rv * rv).sqrt()
            })
            .collect();
        let scale = mad_scale(&resid);
        if scale <= f64::EPSILON {
            break; // residuals already ~0: neighbourhood is homogeneous
        }
        weights = dist.iter().zip(&resid).map(|(&d, &r)| {
            let t = r / (params.tukey_c * scale);
            let robust = if t.abs() < 1.0 { (1.0 - t * t).powi(2) } else { 0.0 };
            spatial_weight(d) * robust
        }).collect();
        match weighted_affine_fit(&dx, &dy, &u, &v, &weights) {
            Some(next) => fit = next,
            None => break, // keep the last well-conditioned fit
        }
    }

    let (cu, cv) = fit;
    let mut warp = Array1::<f64>::zeros(out_len);
    warp[0] = cu[0]; // u
    warp[1] = cv[0]; // v
    warp[2] = cu[1]; // du/dx
    warp[3] = cv[1]; // dv/dx
    warp[4] = cu[2]; // du/dy
    warp[5] = cv[2]; // dv/dy
    warp
}

/// Robust local-affine estimate of a scalar field's value and spatial
/// gradient at `coordinate`, from scattered `(points, values)` pairs --
/// the single-channel analogue of [`meshless_warp_increment`]'s two-channel
/// `(u, v)` fit, reusing the identical Gaussian-kernel-weighted, IRLS
/// Tukey-biweight-reweighted mechanism (same `params.radius` bandwidth
/// convention, same `params.max_iterations`/`params.tukey_c`). Used to
/// estimate the spatial gradient of a per-particle scalar (e.g. `gamma_max`)
/// directly from *other particles'* own positions and already-computed
/// values -- no new query grid, no pixel data.
///
/// Unlike [`meshless_warp_increment`], there is no `quality_gate`
/// filtering here: a scalar field sample (e.g. a particle's own `gamma_max`)
/// doesn't carry a `c_zncc` of its own the way a mesh node's warp does, so
/// that concept doesn't apply to this fit.
///
/// Returns `[value_fit, d(value)/dx, d(value)/dy]`. Falls back to the
/// nearest point's own value (gradient `[0.0, 0.0]`) when fewer than
/// `params.min_neighbours` candidates fall within `params.radius`, or the
/// weighted design matrix is singular -- mirrors
/// [`meshless_warp_increment`]'s own fallback policy exactly.
pub fn meshless_scalar_gradient(
    coordinate: [f64; 2],
    points: ArrayView2<f64>,
    values: ArrayView1<f64>,
    params: &MeshlessParams,
) -> [f64; 3] {
    let n_points = points.nrows();
    let r2 = params.radius * params.radius;

    let candidates: Vec<usize> = (0..n_points)
        .filter(|&i| {
            let dx = points[[i, 0]] - coordinate[0];
            let dy = points[[i, 1]] - coordinate[1];
            dx * dx + dy * dy <= r2
        })
        .collect();

    if candidates.len() < params.min_neighbours {
        return nearest_point_value(coordinate, points, values);
    }

    let dx: Vec<f64> = candidates.iter().map(|&i| points[[i, 0]] - coordinate[0]).collect();
    let dy: Vec<f64> = candidates.iter().map(|&i| points[[i, 1]] - coordinate[1]).collect();
    let val: Vec<f64> = candidates.iter().map(|&i| values[i]).collect();
    let dist: Vec<f64> = dx.iter().zip(&dy).map(|(&x, &y)| (x * x + y * y).sqrt()).collect();

    let bandwidth = (params.radius / 2.0).max(f64::EPSILON);
    let spatial_weight = |d: f64| (-0.5 * (d / bandwidth).powi(2)).exp();
    let mut weights: Vec<f64> = dist.iter().map(|&d| spatial_weight(d)).collect();

    let mut fit = match weighted_affine_fit_scalar(&dx, &dy, &val, &weights) {
        Some(f) => f,
        None => return nearest_point_value(coordinate, points, values),
    };

    for _ in 0..params.max_iterations {
        let resid: Vec<f64> = (0..candidates.len())
            .map(|k| (val[k] - (fit[0] + fit[1] * dx[k] + fit[2] * dy[k])).abs())
            .collect();
        let scale = mad_scale(&resid);
        if scale <= f64::EPSILON {
            break; // residuals already ~0: neighbourhood is homogeneous
        }
        weights = dist.iter().zip(&resid).map(|(&d, &r)| {
            let t = r / (params.tukey_c * scale);
            let robust = if t.abs() < 1.0 { (1.0 - t * t).powi(2) } else { 0.0 };
            spatial_weight(d) * robust
        }).collect();
        match weighted_affine_fit_scalar(&dx, &dy, &val, &weights) {
            Some(next) => fit = next,
            None => break, // keep the last well-conditioned fit
        }
    }

    fit
}

/// Estimate [`meshless_scalar_gradient`] at every row of `query_points`
/// (typically the same points as `points` itself -- every particle's own
/// position, each evaluating its neighbours' values). One Rust-side batch
/// call so the Python caller doesn't pay a per-point FFI round-trip.
pub fn meshless_scalar_gradient_batch(
    query_points: ArrayView2<f64>,
    points: ArrayView2<f64>,
    values: ArrayView1<f64>,
    params: &MeshlessParams,
) -> Array2<f64> {
    let n = query_points.nrows();
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let coordinate = [query_points[[i, 0]], query_points[[i, 1]]];
        let [_value, dfdx, dfdy] = meshless_scalar_gradient(coordinate, points, values, params);
        out[[i, 0]] = dfdx;
        out[[i, 1]] = dfdy;
    }
    out
}

/// Single-channel analogue of [`weighted_affine_fit`]: weighted local
/// affine fit `value ≈ a0 + a1·dx + a2·dy` via the weighted normal
/// equations. A genuine one-channel fit (not the two-channel
/// `weighted_affine_fit` called with duplicated data) -- IRLS residual
/// scaling is channel-count-sensitive (`meshless_warp_increment` combines
/// `(ru, rv)` as `sqrt(ru²+rv²)`), so reusing the two-channel fit here would
/// silently distort the Tukey cutoff by a constant `sqrt(2)` factor rather
/// than being a correct generalisation.
///
/// Returns `None` if the weighted design matrix is singular (e.g. all
/// neighbours collinear, or all weights collapsed to zero).
fn weighted_affine_fit_scalar(dx: &[f64], dy: &[f64], val: &[f64], w: &[f64]) -> Option<[f64; 3]> {
    let n = dx.len();
    let sw: Vec<f64> = w.iter().map(|x| x.max(0.0).sqrt()).collect();
    let a = DMatrix::from_fn(n, 3, |r, c| {
        sw[r] * match c { 0 => 1.0, 1 => dx[r], _ => dy[r] }
    });
    let y = DVector::from_fn(n, |r, _| sw[r] * val[r]);
    let at = a.transpose();
    let ata = &at * &a;
    let inv = ata.try_inverse()?;
    let c = &inv * (&at * &y);
    Some([c[0], c[1], c[2]])
}

/// Fallback for [`meshless_scalar_gradient`]: the nearest point's own
/// value, with zero gradient (mirrors [`nearest_node_warp`]'s "no local
/// slope information available" convention).
fn nearest_point_value(
    coordinate: [f64; 2],
    points: ArrayView2<f64>,
    values: ArrayView1<f64>,
) -> [f64; 3] {
    let n = points.nrows();
    let dist2 = |i: usize| {
        let dx = points[[i, 0]] - coordinate[0];
        let dy = points[[i, 1]] - coordinate[1];
        dx * dx + dy * dy
    };
    let idx = (0..n)
        .map(|i| (dist2(i), i))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, i)| i)
        .unwrap_or(0);
    [values.get(idx).copied().unwrap_or(0.0), 0.0, 0.0]
}

/// Weighted local affine fit `u ≈ a0 + a1·dx + a2·dy`,
/// `v ≈ b0 + b1·dx + b2·dy` via the weighted normal equations
/// (mirrors `mesh::ols_solve`, extended with per-row weights and a shared
/// design matrix for both components).
///
/// Returns `None` if the weighted design matrix is singular (e.g. all
/// neighbours collinear, or all weights collapsed to zero).
fn weighted_affine_fit(
    dx: &[f64],
    dy: &[f64],
    u: &[f64],
    v: &[f64],
    w: &[f64],
) -> Option<([f64; 3], [f64; 3])> {
    let n = dx.len();
    let sw: Vec<f64> = w.iter().map(|x| x.max(0.0).sqrt()).collect();
    let a = DMatrix::from_fn(n, 3, |r, c| {
        sw[r] * match c { 0 => 1.0, 1 => dx[r], _ => dy[r] }
    });
    let yu = DVector::from_fn(n, |r, _| sw[r] * u[r]);
    let yv = DVector::from_fn(n, |r, _| sw[r] * v[r]);
    let at = a.transpose();
    let ata = &at * &a;
    let inv = ata.try_inverse()?;
    let cu = &inv * (&at * &yu);
    let cv = &inv * (&at * &yv);
    Some(([cu[0], cu[1], cu[2]], [cv[0], cv[1], cv[2]]))
}

/// Median absolute deviation, scaled to be a consistent estimator of the
/// standard deviation under normality (`1.4826 * MAD`).
fn mad_scale(values: &[f64]) -> f64 {
    let med = median(values);
    let abs_dev: Vec<f64> = values.iter().map(|&v| (v - med).abs()).collect();
    1.4826 * median(&abs_dev)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Fallback for [`meshless_warp_increment`]: the nearest node's own `p` row,
/// copied directly (zero-padded to `out_len`). Prefers quality-gated nodes;
/// if the gate rejects every node, falls back to the plain nearest node.
///
/// `zonal`, when `Some((node_zone, zone_image))`, keeps the "never blends
/// cross-zone data" guarantee unconditional: the search prefers a
/// quality-gated same-zone node, then any same-zone node, and only falls
/// through to the original zone-agnostic search if literally no node
/// anywhere shares the query's own zone (degenerate -- shouldn't happen
/// given a zone always contains at least the nodes that defined it).
fn nearest_node_warp(
    coordinate: [f64; 2],
    nodes: ArrayView2<f64>,
    p: ArrayView2<f64>,
    c_zncc: ArrayView1<f64>,
    quality_gate: Option<f64>,
    out_len: usize,
    zonal: Option<(ArrayView1<u32>, ArrayView2<u8>)>,
) -> Array1<f64> {
    let n_nodes = nodes.nrows();
    let dist2 = |i: usize| {
        let dx = nodes[[i, 0]] - coordinate[0];
        let dy = nodes[[i, 1]] - coordinate[1];
        dx * dx + dy * dy
    };
    let nearest = |pred: &dyn Fn(usize) -> bool| -> Option<usize> {
        (0..n_nodes)
            .filter(|&i| pred(i))
            .map(|i| (dist2(i), i))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, i)| i)
    };
    let gated = |i: usize| quality_gate.map_or(true, |gate| c_zncc[i] >= gate);

    let idx = if let Some((node_zone, zone_image)) = &zonal {
        let query_zone = zone_at(zone_image.view(), coordinate) as u32;
        let in_zone = |i: usize| node_zone[i] == query_zone;
        nearest(&|i| gated(i) && in_zone(i))
            .or_else(|| nearest(&in_zone))
            .or_else(|| nearest(&gated))
            .or_else(|| nearest(&|_| true))
    } else {
        nearest(&gated).or_else(|| nearest(&|_| true))
    }
    .unwrap_or(0);

    let mut warp = Array1::<f64>::zeros(out_len);
    let cols = p.ncols().min(out_len);
    for j in 0..cols {
        warp[j] = p[[idx, j]];
    }
    warp
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

/// Eigenvalues and principal-direction angle of the symmetric 2x2 tensor
/// `[[exx, exy], [exy, eyy]]`, where `exy` is the *tensorial* (already
/// halved) shear component -- not engineering shear. Returns `(ep1, ep2,
/// theta_p)` with `ep1 >= ep2` and `theta_p` (radians, in `(-pi/2, pi/2]`)
/// the angle of the `ep1` eigenvector from the x-axis. Used by
/// [`principal_strains`].
fn principal_2x2(exx: f64, eyy: f64, exy: f64) -> (f64, f64, f64) {
    let mean = (exx + eyy) / 2.0;
    let radius = ((exx - eyy) / 2.0).hypot(exy);
    let theta_p = 0.5 * (2.0 * exy).atan2(exx - eyy);
    (mean + radius, mean - radius, theta_p)
}

/// Principal strains, max shear magnitude, and principal-plane orientation
/// derived from `strains`' populated tensor columns (`0` = eps_xx, `1` =
/// eps_yy, `5` = tensorial mean shear -- see [`strain_def`]'s sign
/// convention comment, which this inherits unchanged since it operates on
/// the already-computed `strains` array, not on raw warps).
///
/// # Returns
/// `(inc_no, 4)`: columns `[ep1, ep2, gamma_max, theta_p]`, where `ep1 >=
/// ep2` are the eigenvalues (principal strains) and `gamma_max = ep1 - ep2`
/// is the *engineering* max shear strain (the Mohr's-circle diameter, i.e.
/// twice the Mohr's-circle radius -- consistent with `gamma_xy` elsewhere
/// in this codebase being engineering, not tensorial, shear).
pub fn principal_strains(strains: &Array2<f64>) -> Array2<f64> {
    let inc_no = strains.nrows();
    let mut out = Array2::<f64>::zeros((inc_no, 4));
    for i in 0..inc_no {
        let (ep1, ep2, theta_p) =
            principal_2x2(strains[[i, 0]], strains[[i, 1]], strains[[i, 5]]);
        out[[i, 0]] = ep1;
        out[[i, 1]] = ep2;
        out[[i, 2]] = ep1 - ep2;
        out[[i, 3]] = theta_p;
    }
    out
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
    /// How `solve_increment` computes each increment's warp: the meshless
    /// local fit (default) or mesh-element interpolation.
    #[serde(default)]
    pub strain_method: StrainMethod,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self { factor: 0.0, true_incs: true, strain_method: StrainMethod::default() }
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
    /// Mohr/principal strains per increment: `[ep1, ep2, gamma_max, theta_p]`,
    /// shape `(inc_no, 4)` -- see [`principal_strains`]. Computed once here,
    /// at solve time, rather than on demand at plot time (that used to mean
    /// re-deriving it from `strains` every call). `None` only for solutions
    /// predating this field.
    #[serde(default)]
    pub principal_strains: Option<Array2<f64>>,
    /// Spatial gradient of `gamma_max` per increment: `[dgamma/dx,
    /// dgamma/dy]`, shape `(inc_no, 2)` -- see [`crate::field::Field::solve`]'s
    /// post-finalize gradient pass. `None` for solutions predating this
    /// field, for a `Particle` solved standalone (not via `Field`, so it has
    /// no neighbours to estimate a spatial gradient from), or when solved
    /// with an explicit `StrainMethod::Mesh` -- a spatial gradient has no
    /// meshless neighbourhood radius to be driven by in that case, and this
    /// deliberately does NOT fall back to an independent default radius
    /// (that was exactly the earlier bug: two silently-mismatchable radii
    /// for what should be one signal).
    #[serde(default)]
    pub gamma_max_grad: Option<Array2<f64>>,
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
    pub fn solve_increment(&mut self, m: usize, mesh: &MeshSolution, cfg: &ParticleConfig, calibration: Option<&CalibrationParams>) -> bool {
        let ref_update = self.source.ref_update_at(m);
        self.update_reference_index(m, ref_update);

        let warp_inc = if let StrainMethod::Meshless(params) = &cfg.strain_method {
            // Calibration is not supported on the meshless path yet (v1):
            // it always operates on image-space coordinates directly.
            let coord = [
                self.coordinates[[self.reference_index, 0]],
                self.coordinates[[self.reference_index, 1]],
            ];
            let zonal = mesh
                .zonal_masking
                .as_ref()
                .filter(|_| params.zone_aware)
                .map(|z| (z.node_zone.view(), z.zone_image.view()));
            meshless_warp_increment(
                coord,
                mesh.nodes.view(),
                mesh.p.view(),
                mesh.c_zncc.view(),
                params,
                self.incs.ncols(),
                zonal,
            )
        } else if let Some(params) = calibration {
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

        self.apply_warp_increment(m, warp_inc)
    }

    /// Updates `self.reference_index` (and records it) when the source
    /// reports the reference advanced at increment `m`. Must run *before*
    /// [`Particle::solve_increment`] reads
    /// `self.coordinates[[self.reference_index, ..]]` to know where to
    /// locate from.
    fn update_reference_index(&mut self, m: usize, ref_update: bool) {
        if ref_update {
            self.reference_index = m;
            self.reference_update_register.push(m);
        }
    }

    /// Everything downstream of having a `warp_inc` vector
    /// `[u, v, du/dx, dv/dx, du/dy, dv/dy, ...]`: strain clipping, storing
    /// the increment, updating the tracked coordinate, logarithmic warp
    /// accumulation, and the Jacobian-based volume update. Pure refactor of
    /// what was previously the tail of `solve_increment` — no behavioural
    /// change (verified by the existing `solve_increment` tests still
    /// passing unchanged).
    fn apply_warp_increment(&mut self, m: usize, mut warp_inc: Array1<f64>) -> bool {
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
            self.solve_increment(m, &mesh, cfg, calibration);
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
        let principal = principal_strains(&strains);

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
            principal_strains: Some(principal),
            // Populated afterward by Field::solve's cross-particle pass when
            // solved meshless; stays None for a standalone Particle::solve
            // (no neighbours) or an explicit StrainMethod::Mesh solve.
            gamma_max_grad: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests (Phase 2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `ParticleConfig` pinned to mesh-element interpolation -- the tests
    /// below check shape-function strain, not the meshless default.
    fn mesh_cfg() -> ParticleConfig {
        ParticleConfig { strain_method: StrainMethod::Mesh, ..Default::default() }
    }

    #[test]
    fn strain_method_default_is_meshless_with_default_params() {
        // Must match the Python boundary's `strain_method=None`.
        match ParticleConfig::default().strain_method {
            StrainMethod::Meshless(p) => {
                let d = MeshlessParams::default();
                assert_eq!(p.radius, d.radius);
                assert_eq!(p.min_neighbours, d.min_neighbours);
                assert!(!p.zone_aware);
            }
            StrainMethod::Mesh => panic!("core default must be meshless"),
        }
    }
    use ndarray::{array, s};

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
    // meshless_warp_increment — Tier B (rtol = 1e-8) / robustness checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_meshless_homogeneous_matches_affine_field() {
        // Exact affine field: u = 1.0 + 0.2*dx + 0.1*dy, v = -0.5 + 0.05*dx - 0.3*dy.
        // A homogeneous neighbourhood should be fit essentially exactly, with
        // the IRLS reweighting converging immediately (residual scale ~ 0).
        let coordinate = [0.0, 0.0];
        let offsets: Vec<(f64, f64)> = vec![
            (-10.0, -10.0), (0.0, -10.0), (10.0, -10.0),
            (-10.0, 0.0),                  (10.0, 0.0),
            (-10.0, 10.0),  (0.0, 10.0),   (10.0, 10.0),
        ];
        let n = offsets.len();
        let mut nodes = Array2::<f64>::zeros((n, 2));
        let mut p = Array2::<f64>::zeros((n, 6));
        let c_zncc = Array1::<f64>::from_elem(n, 1.0);
        for (i, &(dx, dy)) in offsets.iter().enumerate() {
            nodes[[i, 0]] = dx;
            nodes[[i, 1]] = dy;
            p[[i, 0]] = 1.0 + 0.2 * dx + 0.1 * dy;
            p[[i, 1]] = -0.5 + 0.05 * dx - 0.3 * dy;
        }
        let params = MeshlessParams { radius: 20.0, quality_gate: None, min_neighbours: 4, max_iterations: 5, tukey_c: 4.685, zone_aware: false };
        let w = meshless_warp_increment(coordinate, nodes.view(), p.view(), c_zncc.view(), &params, 6, None);
        assert!((w[0] - 1.0).abs() < 1e-8, "u={}", w[0]);
        assert!((w[1] - (-0.5)).abs() < 1e-8, "v={}", w[1]);
        assert!((w[2] - 0.2).abs() < 1e-8, "du/dx={}", w[2]);
        assert!((w[3] - 0.05).abs() < 1e-8, "dv/dx={}", w[3]);
        assert!((w[4] - 0.1).abs() < 1e-8, "du/dy={}", w[4]);
        assert!((w[5] - (-0.3)).abs() < 1e-8, "dv/dy={}", w[5]);
    }

    #[test]
    fn test_meshless_two_cluster_recovers_near_side() {
        // Near-side region: 5x5 grid (y in {-25,-20,-15,-10,-5}, x in
        // {-20,..,20}), u=3.5,v=2.1 (zero gradient) — 25 points, the clear
        // majority. Far-side intrusion: a single row of 5 points at y=10,
        // u=5.0,v=2.6 — a minority patch belonging to a different strain
        // regime, mimicking a particle whose neighbourhood is mostly
        // homogeneous but partially overlaps a shear-band-like boundary
        // (cf. Case B in mds/trial_sb.md). IRLS should treat the minority
        // patch as outliers and recover the majority (near-side) value;
        // a non-robust weighted average would be pulled noticeably toward
        // the far-side value.
        let coordinate = [0.0, -3.0];
        let near_y = [-25.0, -20.0, -15.0, -10.0, -5.0];
        let xs = [-20.0, -10.0, 0.0, 10.0, 20.0];
        let mut rows: Vec<([f64; 2], [f64; 2])> = Vec::new();
        for &y in &near_y { for &x in &xs { rows.push(([x, y], [3.5, 2.1])); } }
        for &x in &xs { rows.push(([x, 10.0], [5.0, 2.6])); }
        let n = rows.len();
        let mut nodes = Array2::<f64>::zeros((n, 2));
        let mut p = Array2::<f64>::zeros((n, 6));
        let c_zncc = Array1::<f64>::from_elem(n, 1.0);
        for (i, (xy, uv)) in rows.iter().enumerate() {
            nodes[[i, 0]] = xy[0];
            nodes[[i, 1]] = xy[1];
            p[[i, 0]] = uv[0];
            p[[i, 1]] = uv[1];
        }
        let params = MeshlessParams { radius: 35.0, quality_gate: None, min_neighbours: 6, max_iterations: 10, tukey_c: 4.685, zone_aware: false };
        let w = meshless_warp_increment(coordinate, nodes.view(), p.view(), c_zncc.view(), &params, 6, None);
        assert!((w[0] - 3.5).abs() < 0.3, "u0={} (expected close to the near-side 3.5)", w[0]);
        assert!((w[0] - 3.5).abs() < (w[0] - 5.0).abs(), "u0={} ended up closer to the far side than the near side", w[0]);
        assert!((w[1] - 2.1).abs() < 0.2, "v0={} (expected close to 2.1)", w[1]);
    }

    #[test]
    fn test_meshless_fallback_below_min_neighbours() {
        // Only 2 nodes within radius but min_neighbours=6 → fall back to the
        // nearest node's own p row, unchanged.
        let coordinate = [0.0, 0.0];
        let nodes = array![[1.0, 0.0], [-2.0, 0.0], [100.0, 100.0]];
        let p = array![
            [1.0, 2.0, 0.1, 0.2, 0.3, 0.4],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ];
        let c_zncc = array![1.0, 1.0, 1.0];
        let params = MeshlessParams { radius: 5.0, quality_gate: None, min_neighbours: 6, max_iterations: 5, tukey_c: 4.685, zone_aware: false };
        let w = meshless_warp_increment(coordinate, nodes.view(), p.view(), c_zncc.view(), &params, 6, None);
        // Nearest node (index 0, distance 1.0) wins over index 1 (distance 2.0).
        for j in 0..6 {
            assert!((w[j] - p[[0, j]]).abs() < 1e-12, "w[{j}]={}, expected {}", w[j], p[[0, j]]);
        }
    }

    #[test]
    fn test_meshless_quality_gate_excludes_low_zncc() {
        // 8 good-quality neighbours on a constant field plus 4 low-quality
        // neighbours (close to the target, wildly different value); the
        // quality gate should exclude the latter entirely, leaving the fit
        // equivalent to the homogeneous case.
        let coordinate = [0.0, 0.0];
        let good_dx = [-10.0, -5.0, 5.0, 10.0, -10.0, -5.0, 5.0, 10.0];
        let good_dy = [-10.0, -10.0, -10.0, -10.0, 10.0, 10.0, 10.0, 10.0];
        let n_good = good_dx.len();
        let n_bad = 4;
        let n = n_good + n_bad;
        let mut nodes = Array2::<f64>::zeros((n, 2));
        let mut p = Array2::<f64>::zeros((n, 6));
        let mut c_zncc = Array1::<f64>::zeros(n);
        for i in 0..n_good {
            nodes[[i, 0]] = good_dx[i];
            nodes[[i, 1]] = good_dy[i];
            p[[i, 0]] = 2.0;
            p[[i, 1]] = -1.0;
            c_zncc[i] = 0.95;
        }
        for k in 0..n_bad {
            let i = n_good + k;
            nodes[[i, 0]] = 1.0 + k as f64;
            nodes[[i, 1]] = 0.0;
            p[[i, 0]] = 50.0;
            p[[i, 1]] = 50.0;
            c_zncc[i] = 0.2;
        }
        let params = MeshlessParams { radius: 20.0, quality_gate: Some(0.5), min_neighbours: 4, max_iterations: 5, tukey_c: 4.685, zone_aware: false };
        let w = meshless_warp_increment(coordinate, nodes.view(), p.view(), c_zncc.view(), &params, 6, None);
        assert!((w[0] - 2.0).abs() < 1e-6, "u0={} (low-quality outliers should be excluded)", w[0]);
        assert!((w[1] - (-1.0)).abs() < 1e-6, "v0={}", w[1]);
    }

    // -----------------------------------------------------------------------
    // zone-aware meshless filtering
    // -----------------------------------------------------------------------

    #[test]
    fn zone_at_matches_zone_mask_update_lookup() {
        // Same coordinate, same zone_image -- zone_at() must return exactly
        // what zone_mask_update's own centre-label lookup uses internally
        // (extracted-helper equivalence, not a coincidence).
        let mut zone_image = Array2::<u8>::zeros((20, 20));
        zone_image.slice_mut(s![.., 10..]).fill(3);
        let coord = [12.0, 5.0];
        assert_eq!(zone_at(zone_image.view(), coord), 3);

        let mut mask = crate::masks::LocalMask::circle(3).unwrap();
        let before: std::collections::HashSet<(i64, i64)> =
            mask.coords.outer_iter().map(|r| (r[0] as i64, r[1] as i64)).collect();
        mask.zone_mask_update(coord, zone_image.view());
        let after: std::collections::HashSet<(i64, i64)> =
            mask.coords.outer_iter().map(|r| (r[0] as i64, r[1] as i64)).collect();
        // Every kept offset's absolute pixel must share zone_at's own zone
        // (3, since the whole neighbourhood here sits at x>=10).
        for &(dx, dy) in &after {
            let px = [coord[0] + dx as f64, coord[1] + dy as f64];
            assert_eq!(zone_at(zone_image.view(), px), 3);
        }
        assert!(after.len() <= before.len());
    }

    #[test]
    fn meshless_warp_increment_zone_aware_excludes_cross_zone_neighbours() {
        // Two-zone step case, more extreme than the plain IRLS two-cluster
        // test above: the "wrong side" is the numerical MAJORITY within
        // radius (15 far-zone points vs 9 near-zone points), so an
        // IRLS-only fit (no zone filter) is pulled measurably toward the
        // wrong value; zone-aware filtering must still recover the exact
        // near-side value regardless, because the far-zone points are
        // never candidates at all.
        let coordinate = [0.0, -2.0];
        let mut rows: Vec<([f64; 2], [f64; 2], u32)> = Vec::new();
        for &y in &[-15.0, -10.0, -5.0] {
            for &x in &[-10.0, 0.0, 10.0] {
                rows.push(([x, y], [3.5, 2.1], 1)); // near zone (query's own)
            }
        }
        for &y in &[5.0, 10.0, 15.0, 20.0, 25.0] {
            for &x in &[-10.0, 0.0, 10.0] {
                rows.push(([x, y], [9.0, -4.0], 2)); // far zone -- numerical majority
            }
        }
        let n = rows.len();
        let mut nodes = Array2::<f64>::zeros((n, 2));
        let mut p = Array2::<f64>::zeros((n, 6));
        let mut node_zone = Array1::<u32>::zeros(n);
        let c_zncc = Array1::<f64>::from_elem(n, 1.0);
        for (i, &(coord, warp, z)) in rows.iter().enumerate() {
            nodes[[i, 0]] = coord[0];
            nodes[[i, 1]] = coord[1];
            p[[i, 0]] = warp[0];
            p[[i, 1]] = warp[1];
            node_zone[i] = z;
        }
        // query_zone lookup: zone_image is all-1 except y>=0 which is zone 2
        // (matches the near/far split above; coordinate itself is at y=-2,
        // zone 1).
        let mut zone_image = Array2::<u8>::ones((60, 60));
        zone_image.slice_mut(s![30.., ..]).fill(2); // rows >= 30 (image y=30 <-> coord y=0)
        let base_params =
            MeshlessParams { radius: 40.0, quality_gate: None, min_neighbours: 4, max_iterations: 5, tukey_c: 4.685, zone_aware: false };

        let w_unaware =
            meshless_warp_increment(coordinate, nodes.view(), p.view(), c_zncc.view(), &base_params, 6, None);
        let zonal = Some((node_zone.view(), zone_image.view()));
        let w_aware =
            meshless_warp_increment(coordinate, nodes.view(), p.view(), c_zncc.view(), &base_params, 6, zonal);

        assert!((w_aware[0] - 3.5).abs() < 1e-6, "zone-aware u={} (expected exactly 3.5)", w_aware[0]);
        assert!((w_aware[1] - 2.1).abs() < 1e-6, "zone-aware v={}", w_aware[1]);
        assert!(
            (w_unaware[0] - 3.5).abs() > 0.5,
            "unfiltered fit should be pulled well away from 3.5 by the majority far-zone \
             cluster (got u={}), otherwise this test isn't exercising the filter",
            w_unaware[0]
        );
    }

    #[test]
    fn nearest_node_warp_fallback_respects_zone_when_zone_aware() {
        // Tiny radius forces the min_neighbours fallback. Nodes: one
        // zone-2 node very close to the query, one zone-1 (query's own
        // zone) node further away. zone_aware=false's fallback should pick
        // the geometrically nearest (zone 2); zone_aware=true's fallback
        // must pick the nearest SAME-ZONE node instead (zone 1), even
        // though it's farther away.
        let coordinate = [0.0, 0.0];
        let nodes = array![[1.0, 0.0], [5.0, 0.0]];
        let mut p = Array2::<f64>::zeros((2, 6));
        p[[0, 0]] = 99.0; // zone 2 (near)
        p[[1, 0]] = 7.0;  // zone 1 (query's own, farther)
        let c_zncc = Array1::<f64>::from_elem(2, 1.0);
        let node_zone = array![2u32, 1u32];
        let mut zone_image = Array2::<u8>::ones((10, 10));
        zone_image[[0, 0]] = 1; // query's own coordinate sits in zone 1

        let params_unaware =
            MeshlessParams { radius: 0.1, quality_gate: None, min_neighbours: 4, max_iterations: 5, tukey_c: 4.685, zone_aware: false };
        let w_unaware =
            meshless_warp_increment(coordinate, nodes.view(), p.view(), c_zncc.view(), &params_unaware, 6, None);
        assert!((w_unaware[0] - 99.0).abs() < 1e-9, "zone-agnostic fallback should pick the nearest node overall (zone 2): got {}", w_unaware[0]);

        let params_aware =
            MeshlessParams { radius: 0.1, quality_gate: None, min_neighbours: 4, max_iterations: 5, tukey_c: 4.685, zone_aware: true };
        let zonal = Some((node_zone.view(), zone_image.view()));
        let w_aware =
            meshless_warp_increment(coordinate, nodes.view(), p.view(), c_zncc.view(), &params_aware, 6, zonal);
        assert!((w_aware[0] - 7.0).abs() < 1e-9, "zone-aware fallback should skip the cross-zone node and pick the nearest same-zone one: got {}", w_aware[0]);
    }

    // -----------------------------------------------------------------------
    // meshless_scalar_gradient — Tier C (geometric/statistical)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_gradient_recovers_exact_linear_field() {
        // value = 2.0 + 0.3*x - 0.7*y exactly at every scattered point ->
        // the fit should recover [b, c] = [0.3, -0.7] (and the intercept
        // value at `coordinate`) to tight tolerance, regardless of query
        // position.
        let a = 2.0;
        let b = 0.3;
        let c = -0.7;
        let coords: Vec<(f64, f64)> = vec![
            (-10.0, -10.0), (0.0, -10.0), (10.0, -10.0),
            (-10.0, 0.0), (5.0, 3.0), (10.0, 0.0),
            (-10.0, 10.0), (0.0, 10.0), (10.0, 10.0),
        ];
        let n = coords.len();
        let mut points = Array2::<f64>::zeros((n, 2));
        let mut values = Array1::<f64>::zeros(n);
        for (i, &(x, y)) in coords.iter().enumerate() {
            points[[i, 0]] = x;
            points[[i, 1]] = y;
            values[i] = a + b * x + c * y;
        }
        let params = MeshlessParams { radius: 20.0, quality_gate: None, min_neighbours: 4, max_iterations: 5, tukey_c: 4.685, zone_aware: false };
        let query = [1.0, 2.0];
        let [value, dfdx, dfdy] = meshless_scalar_gradient(query, points.view(), values.view(), &params);
        assert!((dfdx - b).abs() < 1e-8, "d/dx={dfdx} (expected {b})");
        assert!((dfdy - c).abs() < 1e-8, "d/dy={dfdy} (expected {c})");
        let expected_value = a + b * query[0] + c * query[1];
        assert!((value - expected_value).abs() < 1e-8, "value={value} (expected {expected_value})");
    }

    #[test]
    fn test_scalar_gradient_uniform_field_gives_zero_gradient() {
        let coords: Vec<(f64, f64)> = vec![
            (-10.0, -10.0), (0.0, -10.0), (10.0, -10.0),
            (-10.0, 0.0), (10.0, 0.0),
            (-10.0, 10.0), (0.0, 10.0), (10.0, 10.0),
        ];
        let n = coords.len();
        let mut points = Array2::<f64>::zeros((n, 2));
        let values = Array1::<f64>::from_elem(n, 4.2);
        for (i, &(x, y)) in coords.iter().enumerate() {
            points[[i, 0]] = x;
            points[[i, 1]] = y;
        }
        let params = MeshlessParams { radius: 20.0, quality_gate: None, min_neighbours: 4, max_iterations: 5, tukey_c: 4.685, zone_aware: false };
        let [value, dfdx, dfdy] = meshless_scalar_gradient([0.0, 0.0], points.view(), values.view(), &params);
        assert!((value - 4.2).abs() < 1e-8, "value={value}");
        assert!(dfdx.abs() < 1e-8, "d/dx={dfdx} (expected 0)");
        assert!(dfdy.abs() < 1e-8, "d/dy={dfdy} (expected 0)");
    }

    #[test]
    fn test_scalar_gradient_fallback_below_min_neighbours() {
        // Only 2 points within radius but min_neighbours=6 -> falls back to
        // the nearest point's own value, zero gradient.
        let points = array![[1.0, 0.0], [-2.0, 0.0], [100.0, 100.0]];
        let values = array![7.0, 9.0, -1.0];
        let params = MeshlessParams { radius: 5.0, quality_gate: None, min_neighbours: 6, max_iterations: 5, tukey_c: 4.685, zone_aware: false };
        let [value, dfdx, dfdy] = meshless_scalar_gradient([0.0, 0.0], points.view(), values.view(), &params);
        assert!((value - 7.0).abs() < 1e-12, "value={value} (expected nearest point's own 7.0)");
        assert_eq!(dfdx, 0.0);
        assert_eq!(dfdy, 0.0);
    }

    #[test]
    fn test_scalar_gradient_batch_matches_single_point_calls() {
        let coords: Vec<(f64, f64)> = vec![
            (-10.0, -10.0), (0.0, -10.0), (10.0, -10.0),
            (-10.0, 0.0), (5.0, 3.0), (10.0, 0.0),
            (-10.0, 10.0), (0.0, 10.0), (10.0, 10.0),
        ];
        let n = coords.len();
        let mut points = Array2::<f64>::zeros((n, 2));
        let mut values = Array1::<f64>::zeros(n);
        for (i, &(x, y)) in coords.iter().enumerate() {
            points[[i, 0]] = x;
            points[[i, 1]] = y;
            values[i] = 1.0 + 0.4 * x - 0.2 * y;
        }
        let params = MeshlessParams { radius: 20.0, quality_gate: None, min_neighbours: 4, max_iterations: 5, tukey_c: 4.685, zone_aware: false };
        let batch = meshless_scalar_gradient_batch(points.view(), points.view(), values.view(), &params);
        assert_eq!(batch.dim(), (n, 2));
        for i in 0..n {
            let coordinate = [points[[i, 0]], points[[i, 1]]];
            let [_v, dfdx, dfdy] = meshless_scalar_gradient(coordinate, points.view(), values.view(), &params);
            assert!((batch[[i, 0]] - dfdx).abs() < 1e-12);
            assert!((batch[[i, 1]] - dfdy).abs() < 1e-12);
        }
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
    // principal_2x2 / principal_strains — Tier A (atol=1e-10)
    // -----------------------------------------------------------------------

    #[test]
    fn test_principal_2x2_pure_normal_strain() {
        // No shear: eigenvalues are just exx, eyy themselves; principal
        // direction aligned with the x-axis (theta_p = 0).
        let (ep1, ep2, theta_p) = principal_2x2(0.1, -0.05, 0.0);
        assert!((ep1 - 0.1).abs() < 1e-10, "ep1={ep1}");
        assert!((ep2 + 0.05).abs() < 1e-10, "ep2={ep2}");
        assert!(theta_p.abs() < 1e-10, "theta_p={theta_p}");
    }

    #[test]
    fn test_principal_2x2_pure_shear_45deg() {
        // Pure (tensorial) shear, no normal strain: principal strains are
        // +-exy at 45 degrees, the textbook Mohr's-circle result.
        let (ep1, ep2, theta_p) = principal_2x2(0.0, 0.0, 0.05);
        assert!((ep1 - 0.05).abs() < 1e-10, "ep1={ep1}");
        assert!((ep2 + 0.05).abs() < 1e-10, "ep2={ep2}");
        assert!((theta_p - std::f64::consts::FRAC_PI_4).abs() < 1e-10, "theta_p={theta_p}");
    }

    #[test]
    fn test_principal_2x2_pure_rotation_has_zero_strain() {
        // A pure rigid rotation's *symmetric* part is zero -- this is the
        // exact property the circular_shear_band_speckle.md investigation
        // relied on (WarpMode::CircularShear's core has nonzero du/dy,
        // dv/dx individually but is strain-free). Antisymmetric gradient
        // (dudy = a, dvdx = -a) must symmetrize to exy_tensor = 0.
        let a = 0.2;
        let exy_tensor = ((-a) + a) / 2.0; // (dvdx + dudy) / 2
        let (ep1, ep2, theta_p) = principal_2x2(0.0, 0.0, exy_tensor);
        assert!(ep1.abs() < 1e-12, "ep1={ep1}");
        assert!(ep2.abs() < 1e-12, "ep2={ep2}");
        assert!(theta_p.abs() < 1e-12, "theta_p={theta_p}");
    }

    #[test]
    fn test_principal_2x2_matches_trace_and_determinant_identities() {
        // Cross-check against the two eigenvalue invariants (trace, det)
        // independently of the closed-form formula under test -- catches
        // sign/factor-of-2 errors that a self-consistent re-derivation of
        // the same formula wouldn't.
        let (exx, eyy, exy) = (0.03, 0.01, 0.02);
        let (ep1, ep2, _) = principal_2x2(exx, eyy, exy);
        assert!(((ep1 + ep2) - (exx + eyy)).abs() < 1e-12, "trace mismatch: {} vs {}", ep1 + ep2, exx + eyy);
        assert!(((ep1 * ep2) - (exx * eyy - exy * exy)).abs() < 1e-12, "det mismatch: {} vs {}", ep1 * ep2, exx * eyy - exy * exy);
        assert!(ep1 >= ep2, "ep1={ep1} should be >= ep2={ep2}");
    }

    #[test]
    fn test_principal_strains_gamma_max_and_shape() {
        let mut strains = Array2::<f64>::zeros((2, 6));
        strains[[1, 0]] = 0.03; // eps_xx
        strains[[1, 1]] = 0.01; // eps_yy
        strains[[1, 5]] = 0.02; // tensorial shear
        let out = principal_strains(&strains);
        assert_eq!(out.shape(), &[2, 4]);
        assert!(out.row(0).iter().all(|&x| x == 0.0), "zero-strain row must stay zero");
        let (ep1, ep2, _) = principal_2x2(0.03, 0.01, 0.02);
        assert!((out[[1, 0]] - ep1).abs() < 1e-12);
        assert!((out[[1, 1]] - ep2).abs() < 1e-12);
        assert!((out[[1, 2]] - (ep1 - ep2)).abs() < 1e-12, "gamma_max must equal ep1-ep2 (engineering, not tensorial)");
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
            template_shape: None,
            template_sizes: None,
            zonal_masking: None,
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
        assert!(p.solve_increment(0, &mesh_sol, &mesh_cfg(), None));

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
        p.solve(&mesh_cfg(), None).unwrap();
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
        assert!(p.solve_increment(0, &mesh, &mesh_cfg(), None));
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
        p.solve(&mesh_cfg(), None).unwrap();

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
        p.solve(&mesh_cfg(), Some(&cal)).unwrap();
        assert!(p.solution().unwrap().calibrated);
    }

    #[test]
    fn test_particle_calibrated_flag_false() {
        let disps = array![[0.5f64, 0.1], [0.5, 0.1], [0.5, 0.1], [0.5, 0.1]];
        let seq = make_seq_o1(vec![disps]);
        let mut p = Particle::new(ParticleSource::Sequence(seq), [1.0/3.0, 1.0/3.0], &[0.0; 6], 1e9, true).unwrap();
        p.solve(&mesh_cfg(), None).unwrap();
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
        p_uncal.solve(&mesh_cfg(), None).unwrap();
        let sol_uncal = p_uncal.solution().unwrap();

        let seq_b = make_seq_o1(vec![disps]);
        let cal = identity_calibration();
        let mut p_cal = Particle::new(ParticleSource::Sequence(seq_b), [1.0/3.0, 1.0/3.0], &[0.0; 6], 1e9, true).unwrap();
        p_cal.solve(&mesh_cfg(), Some(&cal)).unwrap();
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
        p_uncal.solve(&mesh_cfg(), None).unwrap();
        let sol_uncal = p_uncal.solution().unwrap();
        let uncal_disp = [
            sol_uncal.coordinates[[1, 0]] - sol_uncal.coordinates[[0, 0]],
            sol_uncal.coordinates[[1, 1]] - sol_uncal.coordinates[[0, 1]],
        ];

        let seq_b = make_seq_o1(vec![disps]);
        let cal = half_scale_calibration();
        let mut p_cal = Particle::new(ParticleSource::Sequence(seq_b), coord, &[0.0; 6], 1e9, true).unwrap();
        p_cal.solve(&mesh_cfg(), Some(&cal)).unwrap();
        let sol_cal = p_cal.solution().unwrap();
        let cal_disp = [
            sol_cal.coordinates[[1, 0]] - sol_cal.coordinates[[0, 0]],
            sol_cal.coordinates[[1, 1]] - sol_cal.coordinates[[0, 1]],
        ];

        assert!((cal_disp[0] - 0.5 * uncal_disp[0]).abs() < 1e-9, "{} vs {}", cal_disp[0], 0.5 * uncal_disp[0]);
        assert!((cal_disp[1] - 0.5 * uncal_disp[1]).abs() < 1e-9, "{} vs {}", cal_disp[1], 0.5 * uncal_disp[1]);
    }
}
