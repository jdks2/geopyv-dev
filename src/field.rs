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

use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    calibration::CalibrationParams,
    particle::{Particle, ParticleConfig, ParticleSource, ParticleSolution},
    sequence::SequenceSolution,
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
// FieldDistribution
// ---------------------------------------------------------------------------

/// How initial particle positions and volumes are determined.
pub enum FieldDistribution {
    /// Explicit initial positions and volumes — used as-is.
    Explicit {
        coordinates: Array2<f64>,
        volumes: Array1<f64>,
    },
    /// Generate by triangulating the supplied boundary/exclusion polygons.
    FromBoundary {
        boundary_nodes: Array2<f64>,
        exclusion_nodes: Vec<Array2<f64>>,
        target_particles: usize,
    },
    /// Derive from the SequenceSolution: place particles at element centroids
    /// of the first mesh in the sequence.
    FromSequence,
}

// ---------------------------------------------------------------------------
// FieldSolution
// ---------------------------------------------------------------------------

/// Result of [`Field::solve`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSolution {
    /// Per-particle strain-path solutions.
    pub particles: Vec<Arc<ParticleSolution>>,
    /// Initial coordinates of all particles `(N, 2)`.
    pub initial_coordinates: Array2<f64>,
    /// Sum of volumes across all particles at each increment `(inc_no,)`.
    pub vol_totals: Array1<f64>,
    /// Increment indices at which the reference mesh was updated.
    pub reference_update_register: Vec<usize>,
    /// Path of the initial (reference) image.
    #[serde(default)]
    pub image_0_path: Option<PathBuf>,
    /// Whether calibration was applied during solve.
    #[serde(default)]
    pub calibrated: bool,
}

// ---------------------------------------------------------------------------
// Field struct
// ---------------------------------------------------------------------------

/// Distributed particle field for strain-path tracking.
///
/// Construct with [`Field::new`]; then call [`Field::solve`].
pub struct Field {
    pub source: Arc<SequenceSolution>,
    /// Initial particle coordinates `(n_particles, 2)`.
    pub coordinates: Array2<f64>,
    /// Initial particle volumes `(n_particles,)`.
    pub volumes: Array1<f64>,
    pub track: bool,
    pub depth: f64,
    solution: Option<FieldSolution>,
}

impl Field {
    /// Construct a new Field.
    pub fn new(
        source: Arc<SequenceSolution>,
        distribution: FieldDistribution,
        track: bool,
        depth: f64,
    ) -> Result<Self, Error> {
        if depth <= 0.0 {
            return Err(Error::InvalidInput("depth must be > 0".to_string()));
        }
        if source.n_meshes() == 0 {
            return Err(Error::InvalidInput(
                "source sequence has no mesh solutions".to_string(),
            ));
        }

        let (coordinates, volumes) = match distribution {
            FieldDistribution::Explicit { coordinates, volumes } => {
                if coordinates.ncols() != 2 {
                    return Err(Error::InvalidInput(
                        "coordinates must have 2 columns".to_string(),
                    ));
                }
                if volumes.len() != coordinates.nrows() {
                    return Err(Error::InvalidInput(format!(
                        "volumes length {} does not match coordinates rows {}",
                        volumes.len(), coordinates.nrows()
                    )));
                }
                if volumes.iter().any(|&v| v <= 0.0) {
                    return Err(Error::InvalidInput(
                        "all volumes must be > 0".to_string(),
                    ));
                }
                (coordinates, volumes)
            }
            FieldDistribution::FromBoundary { .. } => {
                return Err(Error::InvalidInput(
                    "FromBoundary distribution requires triangulation (not yet implemented at this level)".to_string(),
                ));
            }
            FieldDistribution::FromSequence => {
                let first = source.load_mesh_at(0)
                    .map_err(|e| Error::InvalidInput(
                        format!("failed to load first mesh: {e}"),
                    ))?;
                distribute_particles(&first.nodes, &first.elements, depth)
            }
        };

        Ok(Field { source, coordinates, volumes, track, depth, solution: None })
    }

    pub fn n_particles(&self) -> usize { self.coordinates.nrows() }
    pub fn inc_no(&self) -> usize {
        if let Some(sol) = &self.solution { return sol.vol_totals.len(); }
        self.source.n_meshes() + 1
    }
    pub fn image_0_path(&self) -> Option<&PathBuf> {
        if let Some(sol) = &self.solution {
            if sol.image_0_path.is_some() { return sol.image_0_path.as_ref(); }
        }
        self.source.first_f_img_path.as_ref()
    }
    pub fn solved(&self) -> bool { self.solution.is_some() }
    pub fn solution(&self) -> Option<&FieldSolution> { self.solution.as_ref() }

    /// Reconstruct a `Field` shell from a saved [`FieldSolution`].
    ///
    /// The resulting field is marked as solved.  The source is set to a dummy
    /// `SequenceSolution` (no mesh data); only solution getters work.
    pub fn from_solution(sol: FieldSolution) -> Self {
        let dummy_source = Arc::new(SequenceSolution {
            mesh_solutions: Vec::new(),
            mesh_paths: Vec::new(),
            all_converged: true,
            unsolvable: false,
            override_log: Vec::new(),
            reference_updates: Vec::new(),
            mesh_order: 1,
            first_f_img_path: sol.image_0_path.clone(),
            boundary_region: crate::sequence::default_boundary_region(),
            exclusion_regions: Vec::new(),
        });
        let n = sol.initial_coordinates.nrows();
        let mut volumes = Array1::<f64>::zeros(n);
        for (i, p) in sol.particles.iter().enumerate().take(n) {
            if !p.volumes.is_empty() { volumes[i] = p.volumes[0]; }
        }
        let coordinates = sol.initial_coordinates.clone();
        Field { source: dummy_source, coordinates, volumes, track: true, depth: 1.0,
                solution: Some(sol) }
    }

    /// Solve strain paths for all particles.
    ///
    /// Iterates over increments sequentially, loading one mesh at a time.  For
    /// saved-by-reference sequences each mesh file is read exactly once; all
    /// particles advance their increment in parallel (rayon), then the mesh is
    /// dropped before the next file is opened.
    pub fn solve(&mut self, factor: f64, true_incs: bool, calibration: Option<&CalibrationParams>) -> Result<(), Error> {
        let n = self.n_particles();
        let n_meshes = self.source.n_meshes();
        let cfg = ParticleConfig { factor, true_incs };
        let mesh_order = self.source.mesh_order;
        let initial_warp = vec![0.0f64; 6 * mesh_order as usize];
        let source = Arc::clone(&self.source);

        // Phase 1: create all particles with correctly-sized state arrays.
        let mut particles: Vec<Particle> = (0..n)
            .map(|pi| {
                Particle::new(
                    ParticleSource::Sequence(Arc::clone(&source)),
                    [self.coordinates[[pi, 0]], self.coordinates[[pi, 1]]],
                    &initial_warp,
                    self.volumes[pi],
                    self.track,
                )
            })
            .collect::<Result<_, _>>()?;

        // Convert initial coordinates to object space when calibration is active.
        if let Some(params) = calibration {
            for p in &mut particles {
                let img = ndarray::array![[p.coordinates[[0, 0]], p.coordinates[[0, 1]]]];
                let obj = params.i2o(img.view());
                p.coordinates[[0, 0]] = obj[[0, 0]];
                p.coordinates[[0, 1]] = obj[[0, 1]];
            }
        }

        // Phase 2: one mesh at a time — load, solve all particles, drop.
        for m in 0..n_meshes {
            let mesh = source.load_mesh_at(m)?;
            particles.par_iter_mut().for_each(|p| {
                p.solve_increment(m, &mesh, calibration);
            });
        }

        // Phase 3: finalize strain paths.
        let calibrated = calibration.is_some();
        let particle_solutions: Vec<Arc<ParticleSolution>> = particles.iter()
            .map(|p| {
                let mut sol = p.finalize(&cfg);
                sol.calibrated = calibrated;
                Arc::new(sol)
            })
            .collect();

        let mut vol_totals = Array1::<f64>::zeros(n_meshes + 1);
        for sol in &particle_solutions {
            for (i, &v) in sol.volumes.iter().enumerate() {
                vol_totals[i] += v;
            }
        }

        let reference_update_register: Vec<usize> = source
            .reference_updates
            .iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .map(|(i, _)| i)
            .collect();

        self.solution = Some(FieldSolution {
            particles: particle_solutions,
            initial_coordinates: self.coordinates.clone(),
            vol_totals,
            reference_update_register,
            image_0_path: self.image_0_path().cloned(),
            calibrated,
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests (Phase 6 — updated for new API)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mesh::compute_centroids, sequence::SequenceSolution};
    use ndarray::array;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn unit_square_mesh() -> (Array2<f64>, Array2<usize>) {
        let nodes = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let elems = array![[0usize, 1, 2], [1, 3, 2]];
        (nodes, elems)
    }

    fn pure_translation_disps(u: f64, v: f64) -> Array2<f64> {
        array![[u, v], [u, v], [u, v], [u, v]]
    }

    fn make_mesh_sol(nodes: &Array2<f64>, elems: &Array2<usize>, disps: &Array2<f64>)
        -> crate::mesh::MeshSolution
    {
        let centroids = compute_centroids(nodes, elems);
        let n = nodes.nrows();
        crate::mesh::MeshSolution {
            nodes: nodes.clone(),
            elements: elems.clone(),
            boundary: (0..n).collect(),
            exclusions: vec![],
            centroids,
            areas: Array1::from_vec(vec![0.5; elems.nrows()]),
            warps: Array2::zeros((elems.nrows(), 12)),
            displacements: disps.clone(),
            c_zncc: Array1::ones(n),
            p: Array2::zeros((n, 6)),
            seed_node: 0,
            mesh_order: 1,
            subset_order: 1,
            iterations: Array1::zeros(n),
            norms: Array1::zeros(n),
            f_img_path: PathBuf::new(),
            g_img_path: PathBuf::new(),
        }
    }

    fn make_seq(mesh_sols: Vec<crate::mesh::MeshSolution>, ref_updates: Vec<bool>)
        -> Arc<SequenceSolution>
    {
        let mesh_solutions: Vec<Arc<crate::mesh::MeshSolution>> =
            mesh_sols.into_iter().map(Arc::new).collect();
        Arc::new(SequenceSolution {
            mesh_solutions,
            mesh_paths: vec![],
            all_converged: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: ref_updates,
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: crate::sequence::default_boundary_region(),
            exclusion_regions: vec![],
        })
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
        let nodes = array![[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]];
        let elems = array![[0usize, 1, 2]];
        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        assert!((coords[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((coords[[0, 1]] - 1.0/3.0).abs() < 1e-12);
        assert!((vols[0] - 1.0).abs() < 1e-12, "area={}", vols[0]);
    }

    #[test]
    fn test_distribute_particles_uses_only_corner_nodes() {
        let (nodes, _) = unit_square_mesh();
        let elems_o1 = array![[0usize, 1, 2], [1, 3, 2]];
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

    fn single_mesh_seq(disps: &Array2<f64>) -> Arc<SequenceSolution> {
        let (nodes, elems) = unit_square_mesh();
        make_seq(vec![make_mesh_sol(&nodes, &elems, disps)], vec![false])
    }

    fn two_mesh_seq(disps: &Array2<f64>, ref_updates: Vec<bool>) -> Arc<SequenceSolution> {
        let (nodes, elems) = unit_square_mesh();
        make_seq(
            vec![make_mesh_sol(&nodes, &elems, disps), make_mesh_sol(&nodes, &elems, disps)],
            ref_updates,
        )
    }

    #[test]
    fn test_field_new_from_sequence_valid() {
        let seq = single_mesh_seq(&Array2::<f64>::zeros((4, 2)));
        let f = Field::new(Arc::clone(&seq), FieldDistribution::FromSequence, true, 1.0);
        assert!(f.is_ok());
        let f = f.unwrap();
        assert_eq!(f.n_particles(), 2);
        assert!(!f.solved());
    }

    #[test]
    fn test_field_new_explicit_wrong_volumes_length() {
        let seq = single_mesh_seq(&Array2::<f64>::zeros((4, 2)));
        let (nodes, elems) = unit_square_mesh();
        let (coords, _) = distribute_particles(&nodes, &elems, 1.0);
        let vols = array![0.5f64];
        let dist = FieldDistribution::Explicit { coordinates: coords, volumes: vols };
        assert!(Field::new(seq, dist, true, 1.0).is_err());
    }

    #[test]
    fn test_field_new_explicit_zero_volume() {
        let seq = single_mesh_seq(&Array2::<f64>::zeros((4, 2)));
        let (nodes, elems) = unit_square_mesh();
        let (coords, _) = distribute_particles(&nodes, &elems, 1.0);
        let vols = array![0.0f64, 0.5];
        let dist = FieldDistribution::Explicit { coordinates: coords, volumes: vols };
        assert!(Field::new(seq, dist, true, 1.0).is_err());
    }

    #[test]
    fn test_field_new_bad_depth() {
        let seq = single_mesh_seq(&Array2::<f64>::zeros((4, 2)));
        assert!(Field::new(seq, FieldDistribution::FromSequence, true, 0.0).is_err());
    }

    // -----------------------------------------------------------------------
    // Field::solve integration tests — Tier B (rtol = 1e-8)
    // -----------------------------------------------------------------------

    fn make_field_from_seq(seq: Arc<SequenceSolution>, track: bool) -> Field {
        Field::new(seq, FieldDistribution::FromSequence, track, 1.0).unwrap()
    }

    #[test]
    fn test_field_solve_pure_translation_no_strain() {
        let disps = pure_translation_disps(0.3, 0.1);
        let mut field = make_field_from_seq(single_mesh_seq(&disps), true);
        field.solve(1.0, true, None).unwrap();
        let sol = field.solution().unwrap();
        assert_eq!(sol.particles.len(), 2);
        for p in &sol.particles {
            for j in 2..6 {
                assert!(p.warps[[1, j]].abs() < 1e-10, "warp[1,{j}]={}", p.warps[[1, j]]);
            }
        }
    }

    #[test]
    fn test_field_solve_lagrangian_coordinate_shift() {
        let (nodes, elems) = unit_square_mesh();
        let u = 0.25_f64;
        let disps = pure_translation_disps(u, 0.0);
        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        let initial_coords = coords.clone();
        let seq = single_mesh_seq(&disps);
        let dist = FieldDistribution::Explicit { coordinates: coords, volumes: vols };
        let mut field = Field::new(Arc::clone(&seq), dist, true, 1.0).unwrap();
        field.solve(1.0, true, None).unwrap();
        let sol = field.solution().unwrap();
        for (pi, p) in sol.particles.iter().enumerate() {
            assert!(
                (p.coordinates[[1, 0]] - (initial_coords[[pi, 0]] + u)).abs() < 1e-10,
                "particle {pi} x={}", p.coordinates[[1, 0]]
            );
        }
    }

    #[test]
    fn test_field_solve_eulerian_coordinates_fixed() {
        let (nodes, elems) = unit_square_mesh();
        let disps = pure_translation_disps(0.3, -0.1);
        let (coords, vols) = distribute_particles(&nodes, &elems, 1.0);
        let initial_coords = coords.clone();
        let seq = single_mesh_seq(&disps);
        let dist = FieldDistribution::Explicit { coordinates: coords, volumes: vols };
        let mut field = Field::new(seq, dist, false, 1.0).unwrap();
        field.solve(1.0, true, None).unwrap();
        let sol = field.solution().unwrap();
        for (pi, p) in sol.particles.iter().enumerate() {
            assert!(
                (p.coordinates[[1, 0]] - initial_coords[[pi, 0]]).abs() < 1e-12,
                "particle {pi} should not move"
            );
        }
    }

    #[test]
    fn test_field_solve_vol_totals_shape() {
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field_from_seq(two_mesh_seq(&disps, vec![false, false]), true);
        field.solve(1.0, true, None).unwrap();
        assert_eq!(field.solution().unwrap().vol_totals.len(), 3);
    }

    #[test]
    fn test_field_solve_vol_totals_pure_translation() {
        let (nodes, elems) = unit_square_mesh();
        let disps = pure_translation_disps(0.2, 0.0);
        let (_, vols) = distribute_particles(&nodes, &elems, 1.0);
        let total_initial = vols.sum();
        let mut field = make_field_from_seq(single_mesh_seq(&disps), true);
        field.solve(1.0, true, None).unwrap();
        let sol = field.solution().unwrap();
        assert!((sol.vol_totals[0] - total_initial).abs() < 1e-10, "vt[0]={}", sol.vol_totals[0]);
        assert!((sol.vol_totals[1] - total_initial).abs() < 1e-10, "vt[1]={}", sol.vol_totals[1]);
    }

    #[test]
    fn test_field_solve_sets_solved_flag() {
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field_from_seq(single_mesh_seq(&disps), true);
        assert!(!field.solved());
        field.solve(1.0, true, None).unwrap();
        assert!(field.solved());
    }

    #[test]
    fn test_field_solve_ref_update_register() {
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field_from_seq(two_mesh_seq(&disps, vec![false, true]), true);
        field.solve(1.0, true, None).unwrap();
        assert_eq!(field.solution().unwrap().reference_update_register, vec![1]);
    }

    #[test]
    fn test_field_solve_particle_count() {
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field_from_seq(single_mesh_seq(&disps), true);
        field.solve(1.0, true, None).unwrap();
        assert_eq!(field.solution().unwrap().particles.len(), 2);
    }

    #[test]
    fn test_field_solve_x_stretch_eps_xx() {
        let nodes = array![[0.0, 0.0_f64], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let elems = array![[0usize, 1, 2], [1, 3, 2]];
        let disps = array![[0.0, 0.0_f64], [0.1, 0.0], [0.0, 0.0], [0.1, 0.0]];
        let seq = make_seq(vec![make_mesh_sol(&nodes, &elems, &disps)], vec![false]);
        let mut field = make_field_from_seq(seq, true);
        field.solve(1.0, true, None).unwrap();
        let sol = field.solution().unwrap();
        for p in &sol.particles {
            assert!((p.incs[[1, 2]] - 0.1).abs() < 1e-8, "incs[1,2]={}", p.incs[[1, 2]]);
        }
    }

    #[test]
    fn test_field_solve_two_increments() {
        let disps = pure_translation_disps(0.1, 0.0);
        let mut field = make_field_from_seq(two_mesh_seq(&disps, vec![false, false]), true);
        field.solve(1.0, true, None).unwrap();
        let sol = field.solution().unwrap();
        for p in &sol.particles {
            assert_eq!(p.coordinates.nrows(), 3);
        }
    }

    #[test]
    fn test_field_solution_initial_coordinates() {
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field_from_seq(single_mesh_seq(&disps), true);
        let initial = field.coordinates.clone();
        field.solve(1.0, true, None).unwrap();
        assert_eq!(field.solution().unwrap().initial_coordinates, initial);
    }

    // -----------------------------------------------------------------------
    // Saved-by-reference: Field::new and Field::solve load from disk
    // -----------------------------------------------------------------------

    fn make_saved_by_ref_field_seq(
        tag: &str,
        mesh_sols: Vec<crate::mesh::MeshSolution>,
        ref_updates: Vec<bool>,
    ) -> (Arc<SequenceSolution>, Vec<std::path::PathBuf>) {
        let paths: Vec<_> = mesh_sols.iter().enumerate().map(|(i, _)| {
            std::env::temp_dir().join(format!("geopyv_test_field_{tag}_{i}.pyv"))
        }).collect();
        for (i, ms) in mesh_sols.iter().enumerate() {
            crate::io::save(&paths[i], &crate::io::GeopyvObject::Mesh(ms.clone())).unwrap();
        }
        let sol = Arc::new(SequenceSolution {
            mesh_solutions: vec![],
            mesh_paths: paths.clone(),
            all_converged: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: ref_updates,
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: crate::sequence::default_boundary_region(),
            exclusion_regions: vec![],
        });
        (sol, paths)
    }

    #[test]
    fn test_field_new_saved_by_reference() {
        let (nodes, elems) = unit_square_mesh();
        let disps = pure_translation_disps(0.0, 0.0);
        let ms = make_mesh_sol(&nodes, &elems, &disps);
        let (seq, paths) = make_saved_by_ref_field_seq("new_sbr", vec![ms], vec![false]);

        let f = Field::new(Arc::clone(&seq), FieldDistribution::FromSequence, true, 1.0);
        assert!(f.is_ok(), "Field::new should succeed for saved-by-reference sequence");
        assert_eq!(f.unwrap().n_particles(), 2);

        for p in paths { let _ = std::fs::remove_file(p); }
    }

    #[test]
    fn test_field_new_empty_saved_by_reference_returns_err() {
        let seq = Arc::new(SequenceSolution {
            mesh_solutions: vec![],
            mesh_paths: vec![],
            all_converged: false,
            unsolvable: true,
            override_log: vec![],
            reference_updates: vec![],
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: crate::sequence::default_boundary_region(),
            exclusion_regions: vec![],
        });
        assert!(Field::new(seq, FieldDistribution::FromSequence, true, 1.0).is_err());
    }

    #[test]
    fn test_field_solve_saved_by_reference_pure_translation() {
        let (nodes, elems) = unit_square_mesh();
        let u = 0.25_f64;
        let disps = pure_translation_disps(u, 0.0);
        let ms = make_mesh_sol(&nodes, &elems, &disps);
        let (seq, paths) = make_saved_by_ref_field_seq("pt_sbr", vec![ms], vec![false]);

        let mut field = make_field_from_seq(Arc::clone(&seq), true);
        field.solve(1.0, true, None).unwrap();

        let sol = field.solution().unwrap();
        assert_eq!(sol.particles.len(), 2);
        for p in &sol.particles {
            for j in 2..6 {
                assert!(p.warps[[1, j]].abs() < 1e-9, "warp[1,{j}]={}", p.warps[[1, j]]);
            }
        }

        for path in paths { let _ = std::fs::remove_file(path); }
    }

    #[test]
    fn test_field_solve_saved_by_reference_two_increments() {
        let (nodes, elems) = unit_square_mesh();
        let disps = pure_translation_disps(0.1, 0.0);
        let ms0 = make_mesh_sol(&nodes, &elems, &disps);
        let ms1 = make_mesh_sol(&nodes, &elems, &disps);
        let (seq, paths) =
            make_saved_by_ref_field_seq("two_incr", vec![ms0, ms1], vec![false, false]);

        let mut field = make_field_from_seq(Arc::clone(&seq), true);
        field.solve(1.0, true, None).unwrap();

        let sol = field.solution().unwrap();
        for p in &sol.particles {
            assert_eq!(p.coordinates.nrows(), 3);
        }

        for path in paths { let _ = std::fs::remove_file(path); }
    }

    #[test]
    fn test_field_solve_matches_in_memory_and_saved_by_reference() {
        // Both modes must produce identical strain increments.
        let (nodes, elems) = unit_square_mesh();
        let disps = array![[0.0, 0.0_f64], [0.1, 0.0], [0.0, 0.0], [0.1, 0.0]];
        let ms = make_mesh_sol(&nodes, &elems, &disps);

        // In-memory solve
        let seq_mem = make_seq(vec![ms.clone()], vec![false]);
        let mut field_mem = make_field_from_seq(seq_mem, true);
        field_mem.solve(0.0, true, None).unwrap();
        let sol_mem = field_mem.solution().unwrap();

        // Saved-by-reference solve
        let (seq_sbr, paths) =
            make_saved_by_ref_field_seq("match_sbr", vec![ms], vec![false]);
        let mut field_sbr = make_field_from_seq(Arc::clone(&seq_sbr), true);
        field_sbr.solve(0.0, true, None).unwrap();
        let sol_sbr = field_sbr.solution().unwrap();

        for (pm, ps) in sol_mem.particles.iter().zip(sol_sbr.particles.iter()) {
            for j in 0..6 {
                assert!((pm.incs[[1, j]] - ps.incs[[1, j]]).abs() < 1e-12,
                    "particle incs mismatch at j={j}: mem={} sbr={}",
                    pm.incs[[1, j]], ps.incs[[1, j]]);
            }
        }

        for path in paths { let _ = std::fs::remove_file(path); }
    }
}
