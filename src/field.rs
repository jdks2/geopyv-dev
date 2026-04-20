//! Field: distributed Lagrangian/Eulerian particle tracking and strain-path computation.
//!
//! Translates `geopyv/src/geopyv/field.py` (Field class; all `geomat`
//! sections are excluded — no stress path, no friction work).
//!
//! # Architecture
//!
//! - `distribute_particles` — free function; computes particle positions and
//!   volumes from mesh element data (replicates `Field._distribute_particles`).
//! - [`Field`] — stores initial particle positions, volumes and solve config.
//! - [`Field::solve`] — drives [`crate::particle::Particle`] instances in
//!   parallel (rayon) across all mesh increments.
//! - [`FieldSolution`] — collects all per-particle solutions and derived
//!   aggregate quantities.
//!
//! # Excluded from translation
//!
//! - `_stress_state`, `_stresses`, `_original_stresses` — geomat
//! - `stress()`, `model`/`state`/`parameters` args — geomat
//! - `_works`, `_friction_works` — geomat
//! - `FieldResults.regenerate` — high-level serialisation; deferred to io.rs
//! - `FieldBase` plotting methods — out of scope
//! - `_initial_mesh`, `_update_mesh`, gmsh calls — replaced by spade (triangulation.rs)
//!   at the mesh level; `Field` accepts pre-distributed coordinates or calls
//!   `distribute_particles` after building a triangulation mesh.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    particle::{MeshData, Particle, ParticleConfig, ParticleSolution},
    Error,
};

// ---------------------------------------------------------------------------
// distribute_particles — free function
// ---------------------------------------------------------------------------

/// Distribute particles at element centroids and compute representative volumes.
///
/// Replicates `Field._distribute_particles`.
///
/// # Arguments
/// * `nodes`    — `(N, 2)` mesh node coordinates
/// * `elements` — `(M, 3)` or `(M, 6)` element connectivity; only the first
///                3 columns (corner nodes) are used
/// * `depth`    — depth multiplier for volume (default 1.0 in Python)
///
/// # Returns
/// `(coordinates, volumes)` where
/// * `coordinates` has shape `(M, 2)` — element centroids
/// * `volumes`     has shape `(M,)`   — `|element area| × depth`
pub fn distribute_particles(
    nodes: &Array2<f64>,
    elements: &Array2<usize>,
    depth: f64,
) -> (Array2<f64>, Array1<f64>) {
    let m = elements.nrows();
    let mut coordinates = Array2::<f64>::zeros((m, 2));
    let mut volumes = Array1::<f64>::zeros(m);

    for i in 0..m {
        let n0 = elements[[i, 0]];
        let n1 = elements[[i, 1]];
        let n2 = elements[[i, 2]];

        // Centroid = mean of 3 corner nodes
        coordinates[[i, 0]] = (nodes[[n0, 0]] + nodes[[n1, 0]] + nodes[[n2, 0]]) / 3.0;
        coordinates[[i, 1]] = (nodes[[n0, 1]] + nodes[[n1, 1]] + nodes[[n2, 1]]) / 3.0;

        // Volume = |det(M)| / 2 * depth, where
        // M = [[1, x0, y0], [1, x1, y1], [1, x2, y2]]
        // det = 1*(x1*y2 - x2*y1) - x0*(y2-y1) + y0*(x2-x1)  →  2×signed area
        let (x0, y0) = (nodes[[n0, 0]], nodes[[n0, 1]]);
        let (x1, y1) = (nodes[[n1, 0]], nodes[[n1, 1]]);
        let (x2, y2) = (nodes[[n2, 0]], nodes[[n2, 1]]);
        let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        volumes[i] = det.abs() * 0.5 * depth;
    }

    (coordinates, volumes)
}

// ---------------------------------------------------------------------------
// FieldSolution
// ---------------------------------------------------------------------------

/// Result of [`Field::solve`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSolution {
    /// Per-particle strain-path solutions.
    pub particles: Vec<ParticleSolution>,
    /// Sum of volumes across all particles at each increment `(inc_no,)`.
    pub vol_totals: Array1<f64>,
    /// Increment indices at which the reference mesh was updated.
    /// Derived from the `ref_updates` slice passed to `solve`.
    pub reference_update_register: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Field struct
// ---------------------------------------------------------------------------

/// Distributed particle field for strain-path tracking.
///
/// Construct with [`Field::new`] supplying per-particle initial positions and
/// volumes; then call [`Field::solve`].
pub struct Field {
    /// Initial particle coordinates `(n_particles, 2)`.
    pub coordinates: Array2<f64>,
    /// Initial particle volumes `(n_particles,)`.
    pub volumes: Array1<f64>,
    /// `true` for Lagrangian tracking (coordinates move with material).
    pub track: bool,
    /// Depth multiplier (used when computing distributed volumes).
    pub depth: f64,
    /// Total number of frames, including the initial state.
    pub inc_no: usize,
    /// Set to `true` after a successful [`Field::solve`].
    pub solved: bool,
}

impl Field {
    /// Construct a new Field.
    ///
    /// # Arguments
    /// * `coordinates` — `(n_particles, 2)` initial positions
    /// * `volumes`     — `(n_particles,)` initial volumes; all must be > 0
    /// * `track`       — Lagrangian (`true`) or Eulerian (`false`)
    /// * `depth`       — depth multiplier; must be > 0
    /// * `inc_no`      — number of frames (≥ 2; `inc_no - 1` mesh increments)
    pub fn new(
        coordinates: Array2<f64>,
        volumes: Array1<f64>,
        track: bool,
        depth: f64,
        inc_no: usize,
    ) -> Result<Self, Error> {
        if coordinates.ncols() != 2 {
            return Err(Error::InvalidInput(
                "coordinates must have 2 columns".to_string(),
            ));
        }
        if volumes.len() != coordinates.nrows() {
            return Err(Error::InvalidInput(format!(
                "volumes length {} does not match coordinates rows {}",
                volumes.len(),
                coordinates.nrows()
            )));
        }
        if volumes.iter().any(|&v| v <= 0.0) {
            return Err(Error::InvalidInput(
                "all volumes must be > 0".to_string(),
            ));
        }
        if depth <= 0.0 {
            return Err(Error::InvalidInput("depth must be > 0".to_string()));
        }
        if inc_no < 2 {
            return Err(Error::InvalidInput(
                "inc_no must be >= 2".to_string(),
            ));
        }
        Ok(Field {
            coordinates,
            volumes,
            track,
            depth,
            inc_no,
            solved: false,
        })
    }

    /// Number of particles.
    pub fn n_particles(&self) -> usize {
        self.coordinates.nrows()
    }

    /// Solve strain paths for all particles over a sequence of mesh increments.
    ///
    /// Replicates `Field.solve` (minus geomat sections, minus alive_bar).
    ///
    /// # Arguments
    /// * `meshes`      — one entry per increment; `meshes[m]` supplies the DIC
    ///                   solve between frame `m` and `m+1`.  Length must equal
    ///                   `inc_no - 1`.
    /// * `ref_updates` — one `bool` per increment; `true` if the reference mesh
    ///                   changed at step `m` (replaces `_check_update`).
    ///                   May be empty (treated as all-`false`).
    /// * `factor`      — volumetric correction factor passed to `strain_def`
    /// * `true_incs`   — logarithmic strain increments
    pub fn solve(
        &mut self,
        meshes: &[MeshData<'_>],
        ref_updates: &[bool],
        factor: f64,
        true_incs: bool,
    ) -> Result<FieldSolution, Error> {
        let expected = self.inc_no - 1;
        if meshes.len() != expected {
            return Err(Error::InvalidInput(format!(
                "expected {} meshes for {} increments, got {}",
                expected,
                self.inc_no,
                meshes.len()
            )));
        }
        if !ref_updates.is_empty() && ref_updates.len() != expected {
            return Err(Error::InvalidInput(format!(
                "ref_updates length {} does not match {} increments",
                ref_updates.len(),
                expected
            )));
        }

        let n = self.n_particles();
        let cfg = ParticleConfig { factor, true_incs };

        // Build per-particle initial warps (all zeros) and solve in parallel.
        let initial_warps: Vec<[f64; 6]> = vec![[0.0; 6]; n];

        let results: Vec<Result<ParticleSolution, Error>> = (0..n)
            .into_par_iter()
            .map(|pi| {
                let coord = [self.coordinates[[pi, 0]], self.coordinates[[pi, 1]]];
                let vol = self.volumes[pi];
                let mesh_order = if meshes.is_empty() { 1 } else { meshes[0].mesh_order };

                let mut particle = Particle::new(
                    coord,
                    &initial_warps[pi],
                    vol,
                    self.inc_no,
                    mesh_order,
                    self.track,
                )?;

                // Solve increment by increment so ref_update can vary per step.
                for m in 0..expected {
                    let ref_update = ref_updates.get(m).copied().unwrap_or(false);
                    particle.solve_increment(m, &meshes[m], ref_update);
                }
                particle.solved = true;
                Ok(particle.finalize(&cfg))
            })
            .collect();

        // Propagate first error, if any.
        let particle_solutions: Vec<ParticleSolution> = results
            .into_iter()
            .collect::<Result<_, _>>()?;

        // Sum volumes across particles at each increment.
        let mut vol_totals = Array1::<f64>::zeros(self.inc_no);
        for sol in &particle_solutions {
            for (i, &v) in sol.volumes.iter().enumerate() {
                vol_totals[i] += v;
            }
        }

        // Derive reference_update_register from ref_updates slice.
        let reference_update_register: Vec<usize> = (0..expected)
            .filter(|&m| ref_updates.get(m).copied().unwrap_or(false))
            .collect();

        self.solved = true;
        Ok(FieldSolution {
            particles: particle_solutions,
            vol_totals,
            reference_update_register,
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests (Phase 2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Shared test mesh: 4 nodes, 2 right-triangle elements (unit square)
    fn unit_square_mesh() -> (Array2<f64>, Array2<usize>) {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let elems = array![[0usize, 1, 2], [1, 3, 2]];
        (nodes, elems)
    }

    fn pure_translation_disps(u: f64, v: f64) -> Array2<f64> {
        array![[u, v], [u, v], [u, v], [u, v]]
    }

    // -----------------------------------------------------------------------
    // distribute_particles — Tier A (atol = 1e-12)
    // -----------------------------------------------------------------------

    #[test]
    fn test_distribute_particles_centroids() {
        let (nodes, elems) = unit_square_mesh();
        let (coords, _) = distribute_particles(&nodes, &elems, 1.0);
        let c0 = [
            (nodes[[0, 0]] + nodes[[1, 0]] + nodes[[2, 0]]) / 3.0,
            (nodes[[0, 1]] + nodes[[1, 1]] + nodes[[2, 1]]) / 3.0,
        ];
        assert!((coords[[0, 0]] - c0[0]).abs() < 1e-12);
        assert!((coords[[0, 1]] - c0[1]).abs() < 1e-12);
    }

    #[test]
    fn test_distribute_particles_volumes_unit_square() {
        let (nodes, elems) = unit_square_mesh();
        let (_, vols) = distribute_particles(&nodes, &elems, 1.0);
        // Two right triangles tiling unit square → each area = 0.5
        assert!((vols[0] - 0.5).abs() < 1e-12, "vol[0]={}", vols[0]);
        assert!((vols[1] - 0.5).abs() < 1e-12, "vol[1]={}", vols[1]);
    }

    #[test]
    fn test_distribute_particles_total_area_unit_square() {
        let (nodes, elems) = unit_square_mesh();
        let (_, vols) = distribute_particles(&nodes, &elems, 1.0);
        assert!((vols.sum() - 1.0).abs() < 1e-12, "sum={}", vols.sum());
    }

    #[test]
    fn test_distribute_particles_depth_scaling() {
        let (nodes, elems) = unit_square_mesh();
        let (_, vols_1) = distribute_particles(&nodes, &elems, 1.0);
        let (_, vols_3) = distribute_particles(&nodes, &elems, 3.0);
        for i in 0..2 {
            assert!((vols_3[i] - vols_1[i] * 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_distribute_particles_single_triangle() {
        // Equilateral-ish: (0,0),(2,0),(1,1) → area = 1.0
        let nodes = array![[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]];
        let elems = array![[0usize, 1, 2]];
        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        let expected_centroid = [1.0, 1.0 / 3.0];
        assert!((coords[[0, 0]] - expected_centroid[0]).abs() < 1e-12);
        assert!((coords[[0, 1]] - expected_centroid[1]).abs() < 1e-12);
        assert!((vols[0] - 1.0).abs() < 1e-12, "area={}", vols[0]);
    }

    #[test]
    fn test_distribute_particles_uses_only_corner_nodes() {
        // Order-2 element columns: only first 3 used
        let (nodes, _) = unit_square_mesh();
        let elems_o1 = array![[0usize, 1, 2], [1, 3, 2]];
        // Fabricate order-2 elements with dummy midpoint columns
        let elems_o2 = array![[0usize, 1, 2, 0, 0, 0], [1, 3, 2, 0, 0, 0]];
        let (coords_o1, vols_o1) = distribute_particles(&nodes, &elems_o1, 1.0);
        let (coords_o2, vols_o2) = distribute_particles(&nodes, &elems_o2, 1.0);
        for i in 0..2 {
            assert!((coords_o1[[i, 0]] - coords_o2[[i, 0]]).abs() < 1e-12);
            assert!((vols_o1[i] - vols_o2[i]).abs() < 1e-12);
        }
    }

    // -----------------------------------------------------------------------
    // Field::new validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_field_new_valid() {
        let (nodes, elems) = unit_square_mesh();
        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        let f = Field::new(coords, vols, true, 1.0, 3);
        assert!(f.is_ok());
        let f = f.unwrap();
        assert_eq!(f.n_particles(), 2);
        assert!(!f.solved);
    }

    #[test]
    fn test_field_new_wrong_volumes_length() {
        let (nodes, elems) = unit_square_mesh();
        let (coords, _) = distribute_particles(&nodes, &elems, 1.0);
        let vols = array![0.5]; // 1 volume for 2 particles
        assert!(Field::new(coords, vols, true, 1.0, 3).is_err());
    }

    #[test]
    fn test_field_new_zero_volume() {
        let (nodes, elems) = unit_square_mesh();
        let (coords, _) = distribute_particles(&nodes, &elems, 1.0);
        let vols = array![0.0, 0.5]; // zero volume
        assert!(Field::new(coords, vols, true, 1.0, 3).is_err());
    }

    #[test]
    fn test_field_new_inc_no_too_small() {
        let (nodes, elems) = unit_square_mesh();
        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        assert!(Field::new(coords, vols, true, 1.0, 1).is_err()); // inc_no = 1 < 2
    }

    // -----------------------------------------------------------------------
    // Field::solve integration tests — Tier B (rtol = 1e-8)
    // -----------------------------------------------------------------------

    fn make_field(inc_no: usize) -> Field {
        let (nodes, elems) = unit_square_mesh();
        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        Field::new(coords, vols, true, 1.0, inc_no).unwrap()
    }

    fn make_mesh_data<'a>(
        nodes: &'a Array2<f64>,
        elements: &'a Array2<usize>,
        disps: &'a Array2<f64>,
    ) -> MeshData<'a> {
        MeshData {
            nodes,
            elements,
            displacements: disps,
            mesh_order: 1,
        }
    }

    #[test]
    fn test_field_solve_pure_translation_no_strain() {
        let (nodes, elems) = unit_square_mesh();
        let disps = pure_translation_disps(0.3, 0.1);
        let cm = make_mesh_data(&nodes, &elems, &disps);
        let meshes = vec![cm];

        let mut field = make_field(2); // 2 frames → 1 increment
        let sol = field.solve(&meshes, &[], 1.0, true).unwrap();

        assert_eq!(sol.particles.len(), 2);
        for p in &sol.particles {
            // Pure translation → all strain components ≈ 0
            for j in 2..6 {
                assert!(
                    p.warps[[1, j]].abs() < 1e-10,
                    "warp[1,{j}]={}", p.warps[[1, j]]
                );
            }
        }
    }

    #[test]
    fn test_field_solve_lagrangian_coordinate_shift() {
        let (nodes, elems) = unit_square_mesh();
        let u = 0.25_f64;
        let disps = pure_translation_disps(u, 0.0);
        let cm = make_mesh_data(&nodes, &elems, &disps);

        // Track = true (Lagrangian) → coordinates move
        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        let initial_coords = coords.clone();
        let mut field = Field::new(coords, vols, true, 1.0, 2).unwrap();
        let sol = field.solve(&[make_mesh_data(&nodes, &elems, &disps)], &[], 1.0, true).unwrap();

        for (pi, p) in sol.particles.iter().enumerate() {
            assert!(
                (p.coordinates[[1, 0]] - (initial_coords[[pi, 0]] + u)).abs() < 1e-10,
                "particle {pi} x={}, expected {}",
                p.coordinates[[1, 0]],
                initial_coords[[pi, 0]] + u
            );
        }
    }

    #[test]
    fn test_field_solve_eulerian_coordinates_fixed() {
        let (nodes, elems) = unit_square_mesh();
        let disps = pure_translation_disps(0.3, -0.1);

        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        let initial_coords = coords.clone();
        let mut field = Field::new(coords, vols, false, 1.0, 2).unwrap(); // track = false
        let sol = field.solve(&[make_mesh_data(&nodes, &elems, &disps)], &[], 1.0, true).unwrap();

        for (pi, p) in sol.particles.iter().enumerate() {
            assert!(
                (p.coordinates[[1, 0]] - initial_coords[[pi, 0]]).abs() < 1e-12,
                "particle {pi} should not move"
            );
        }
    }

    #[test]
    fn test_field_solve_vol_totals_shape() {
        let (nodes, elems) = unit_square_mesh();
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field(3); // 3 frames → 2 increments
        let meshes = vec![
            make_mesh_data(&nodes, &elems, &disps),
            make_mesh_data(&nodes, &elems, &disps),
        ];
        let sol = field.solve(&meshes, &[], 0.0, true).unwrap();
        assert_eq!(sol.vol_totals.len(), 3);
    }

    #[test]
    fn test_field_solve_vol_totals_pure_translation() {
        // Pure translation → volume unchanged per particle → vol_totals constant
        let (nodes, elems) = unit_square_mesh();
        let disps = pure_translation_disps(0.2, 0.0);
        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        let total_initial = vols.sum();

        let mut field = Field::new(coords, vols, true, 1.0, 2).unwrap();
        let sol = field.solve(&[make_mesh_data(&nodes, &elems, &disps)], &[], 0.0, true).unwrap();

        // vol_totals[0] and [1] should both ≈ total_initial
        assert!((sol.vol_totals[0] - total_initial).abs() < 1e-10, "vt[0]={}", sol.vol_totals[0]);
        assert!((sol.vol_totals[1] - total_initial).abs() < 1e-10, "vt[1]={}", sol.vol_totals[1]);
    }

    #[test]
    fn test_field_solve_sets_solved_flag() {
        let (nodes, elems) = unit_square_mesh();
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field(2);
        assert!(!field.solved);
        field.solve(&[make_mesh_data(&nodes, &elems, &disps)], &[], 1.0, true).unwrap();
        assert!(field.solved);
    }

    #[test]
    fn test_field_solve_wrong_mesh_count() {
        let (nodes, elems) = unit_square_mesh();
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field(3); // needs 2 meshes
        // Supply only 1 mesh → error
        let result = field.solve(&[make_mesh_data(&nodes, &elems, &disps)], &[], 1.0, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_field_solve_ref_update_register() {
        let (nodes, elems) = unit_square_mesh();
        let disps = Array2::<f64>::zeros((4, 2));
        let meshes = vec![
            make_mesh_data(&nodes, &elems, &disps),
            make_mesh_data(&nodes, &elems, &disps),
        ];
        let mut field = make_field(3);
        // Mark step 1 as a reference update
        let sol = field.solve(&meshes, &[false, true], 1.0, true).unwrap();
        assert_eq!(sol.reference_update_register, vec![1]);
    }

    #[test]
    fn test_field_solve_particle_count() {
        let (nodes, elems) = unit_square_mesh();
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field(2);
        let sol = field.solve(&[make_mesh_data(&nodes, &elems, &disps)], &[], 1.0, true).unwrap();
        assert_eq!(sol.particles.len(), 2);
    }

    #[test]
    fn test_field_solve_x_stretch_eps_xx() {
        // 10% x-stretch: node displacement proportional to x coordinate
        let nodes = array![[0.0, 0.0_f64], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let elems = array![[0usize, 1, 2], [1, 3, 2]];
        let disps = array![[0.0, 0.0_f64], [0.1, 0.0], [0.0, 0.0], [0.1, 0.0]];

        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        let mut field = Field::new(coords, vols, true, 1.0, 2).unwrap();
        let sol = field.solve(&[make_mesh_data(&nodes, &elems, &disps)], &[], 0.0, true).unwrap();

        // For a uniform x-stretch of 10%, warp_inc[2] = du/dx ≈ 0.1 for each particle
        for p in &sol.particles {
            assert!(
                (p.incs[[1, 2]] - 0.1).abs() < 1e-8,
                "incs[1,2]={}", p.incs[[1, 2]]
            );
        }
    }

    #[test]
    fn test_field_solve_two_increments() {
        let (nodes, elems) = unit_square_mesh();
        let disps = pure_translation_disps(0.1, 0.0);
        let meshes = vec![
            make_mesh_data(&nodes, &elems, &disps),
            make_mesh_data(&nodes, &elems, &disps),
        ];
        let mut field = make_field(3); // 3 frames → 2 increments
        let sol = field.solve(&meshes, &[], 1.0, true).unwrap();
        // Each particle: 3 frames → coordinates (3, 2)
        for p in &sol.particles {
            assert_eq!(p.coordinates.nrows(), 3);
        }
    }
}
