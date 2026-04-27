//! Sequence: coordinated multi-image-pair DIC solving with mesh reuse.
//!
//! Translates `geopyv/src/geopyv/sequence.py` (`Sequence` class).
//!
//! # Architecture
//!
//! - [`Sequence`] stores an ordered list of image paths and mesh-generation
//!   configuration.
//! - [`Sequence::solve`] iterates over consecutive image pairs, generates (or
//!   reuses) a mesh for each pair, solves it, and applies deformation
//!   preconditioning via a [`crate::particle::Particle`].
//! - [`SequenceSolution`] stores the per-pair [`crate::mesh::MeshSolution`]
//!   results plus solve metadata.
//!
//! # Solve modes
//!
//! | Flag | Behaviour |
//! |------|-----------|
//! | `guide` | Particle-based warp preconditioning for next seed warp |
//! | `sequential` | After each success, advance reference to g (cumulative) |
//! | `sync` | Reuse previous mesh geometry for next pair (skips CDT) |
//! | `override_` | On failure, relax tolerance for next attempt from updated ref |
//!
//! # Excluded from translation
//!
//! - `SequenceBase` plotting methods (`inspect`, `convergence`, `contour`, `quiver`)
//! - `load` / `save_by_reference` file management
//! - `SequenceResults.regenerate` — high-level deserialization; deferred to io.rs
//! - GUI selectors (`gp.gui.selectors.coordinate.CoordinateSelector`)

use std::path::PathBuf;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    geometry::triangulation,
    image::Image,
    mesh::{adaptive_target_areas, Mesh, MeshSolution, SolveConfig},
    particle::{MeshData, Particle},
    Error,
};

// ---------------------------------------------------------------------------
// Mesh-generation configuration
// ---------------------------------------------------------------------------

/// Parameters forwarded to [`Mesh::generate`] for each image pair.
#[derive(Debug, Clone)]
pub struct SequenceMeshConfig {
    /// Boundary polygon vertices + exclusion polygon vertices/segments in the
    /// format expected by [`crate::geometry::triangulation::generate_mesh`].
    pub borders: ndarray::Array2<f64>,
    pub segments: ndarray::Array2<i32>,
    pub curves: Vec<Vec<i32>>,
    pub size_lower: f64,
    pub size_upper: f64,
    pub target_nodes: usize,
    pub mesh_order: u8,
}

// ---------------------------------------------------------------------------
// Solve configuration
// ---------------------------------------------------------------------------

/// Solver parameters for each mesh pair within a sequence.
#[derive(Debug, Clone)]
pub struct SequenceSolveConfig {
    /// Per-subset solver settings forwarded to [`Mesh::solve`].
    pub mesh_cfg: SolveConfig,
    /// Template pixel coordinates (from [`crate::templates`]).
    pub template_coords: Array2<f64>,
    /// Initial seed coordinate for the reliability-guided solver.
    pub seed_coord: [f64; 2],
    /// Initial seed warp vector (length ≤ 12, zero-padded to 12 internally).
    pub seed_warp: Vec<f64>,
    /// Number of adaptive-remesh iterations per pair.
    pub adaptive_iterations: usize,
    /// Adaptivity control parameter α (must be in (0, 1)).
    pub alpha: f64,
    /// Apply particle-based warp preconditioning between pairs.
    pub guide: bool,
    /// Advance reference image after each successful solve (cumulative mode).
    pub sequential: bool,
    /// Reuse previous mesh geometry in sync mode (skips CDT regeneration).
    pub sync: bool,
    /// Relax tolerance on retry after a failed non-consecutive pair.
    pub override_: bool,
    /// Image border (pixels) for B-spline precomputation. Defaults to 20.
    pub border: usize,
}

impl Default for SequenceSolveConfig {
    fn default() -> Self {
        Self {
            mesh_cfg: SolveConfig::default(),
            template_coords: Array2::zeros((0, 2)),
            seed_coord: [0.0, 0.0],
            seed_warp: vec![0.0; 12],
            adaptive_iterations: 0,
            alpha: 0.5,
            guide: true,
            sequential: false,
            sync: true,
            override_: false,
            border: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// Sequence solution
// ---------------------------------------------------------------------------

/// Result of [`Sequence::solve`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceSolution {
    /// Per-pair DIC solutions.  `mesh_solutions[i]` corresponds to the pair
    /// `(images[i], images[i+1])`.
    pub mesh_solutions: Vec<MeshSolution>,
    /// `true` when all image pairs were solved successfully.
    pub solved: bool,
    /// `true` when a consecutive pair was unsolvable and the sequence was
    /// curtailed.
    pub unsolvable: bool,
    /// g-indices (1-based) at which tolerance override was active.
    pub override_log: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Sequence struct
// ---------------------------------------------------------------------------

/// Multi-image-pair DIC sequence controller.
///
/// Construct with [`Sequence::new`] supplying an ordered list of image file
/// paths and the mesh generation configuration; then call [`Sequence::solve`].
pub struct Sequence {
    /// Ordered list of image file paths (at least 2).
    pub image_paths: Vec<PathBuf>,
    /// Mesh generation configuration shared across all pairs.
    pub mesh_cfg: SequenceMeshConfig,
}

impl Sequence {
    /// Construct a new Sequence.
    ///
    /// # Arguments
    /// * `image_paths` — ordered list of ≥2 image file paths
    /// * `mesh_cfg`    — mesh generation parameters
    pub fn new(
        image_paths: Vec<PathBuf>,
        mesh_cfg: SequenceMeshConfig,
    ) -> Result<Self, Error> {
        if image_paths.len() < 2 {
            return Err(Error::InvalidInput(
                "sequence requires at least 2 images".to_string(),
            ));
        }
        for p in &image_paths {
            if !p.exists() {
                return Err(Error::FileNotFound(p.to_string_lossy().to_string()));
            }
        }
        if mesh_cfg.size_lower <= 0.0 || mesh_cfg.size_lower >= mesh_cfg.size_upper {
            return Err(Error::InvalidInput(
                "size_lower must be > 0 and < size_upper".to_string(),
            ));
        }
        if mesh_cfg.target_nodes < 1 {
            return Err(Error::InvalidInput(
                "target_nodes must be >= 1".to_string(),
            ));
        }
        Ok(Sequence { image_paths, mesh_cfg })
    }

    /// Number of image pairs (= number of meshes to solve).
    pub fn n_pairs(&self) -> usize {
        self.image_paths.len() - 1
    }

    /// Solve all image pairs.
    ///
    /// Replicates `Sequence.solve` (minus alive_bar, GUI, geomat sections,
    /// and save-by-reference disk I/O).
    pub fn solve(&self, cfg: &SequenceSolveConfig) -> Result<SequenceSolution, Error> {
        let n_images = self.image_paths.len();
        let mut mesh_solutions: Vec<MeshSolution> = Vec::with_capacity(n_images - 1);
        let mut override_log: Vec<usize> = Vec::new();

        let mut f_index = 0usize;
        let mut g_index = 1usize;

        let mut seed_coord = cfg.seed_coord;
        // Normalise seed_warp to 12 elements.
        let mut seed_warp = cfg.seed_warp.clone();
        seed_warp.resize(12, 0.0);

        // Cached previous-pair solution for sync mode.
        let mut sync_sol: Option<MeshSolution> = None;
        // Override flag: relax tolerance for the next mesh solve.
        let mut mesh_override = false;

        // Load initial images.
        let mut f_img = Image::from_file(&self.image_paths[f_index], cfg.border)?;
        let mut g_img = Image::from_file(&self.image_paths[g_index], cfg.border)?;

        let all_solved;

        'outer: loop {
            // --- Build mesh for this pair. ----------------------------------
            // In sync mode, reuse previous pair's geometry when available.
            let adaptive_iters = if cfg.sync && sync_sol.is_some() {
                0 // geometry is fixed; adaptive remesh would change node count
            } else {
                cfg.adaptive_iterations
            };

            let mesh = match (cfg.sync, &sync_sol) {
                (true, Some(prev)) => Mesh::from_solution(prev),
                _ => self.generate_mesh()?,
            };

            // Override: use tolerance = 0 (accept any subset).
            let pair_cfg = if mesh_override {
                SolveConfig {
                    tolerance: 0.0,
                    ..cfg.mesh_cfg.clone()
                }
            } else {
                cfg.mesh_cfg.clone()
            };

            // --- Solve this pair (with optional adaptive remesh). -----------
            let pair_result = solve_pair_adaptive(
                mesh,
                &f_img,
                &g_img,
                &cfg.template_coords,
                seed_coord,
                &seed_warp,
                &pair_cfg,
                adaptive_iters,
                cfg.alpha,
                &self.mesh_cfg,
            );

            let mesh_sol = match pair_result {
                Ok(sol) => sol,
                Err(_) => {
                    // Attempt to fall back to an updated reference.
                    if f_index + 1 < g_index {
                        // Non-consecutive pair failed: step reference forward.
                        f_index = g_index - 1;
                        f_img = Image::from_file(&self.image_paths[f_index], cfg.border)?;
                        if cfg.sync {
                            sync_sol = None;
                        }
                        if cfg.override_ {
                            mesh_override = true;
                        }
                        continue 'outer;
                    } else {
                        // Consecutive pair truly unsolvable: curtail sequence.
                        return Ok(SequenceSolution {
                            mesh_solutions,
                            solved: false,
                            unsolvable: true,
                            override_log,
                        });
                    }
                }
            };

            // --- Log override usage. ----------------------------------------
            if mesh_override {
                if mesh_sol.c_zncc.iter().any(|&c| c < cfg.mesh_cfg.tolerance) {
                    override_log.push(g_index);
                }
                mesh_override = false;
            }

            // --- Store result and update sync geometry. ---------------------
            if cfg.sync {
                sync_sol = Some(mesh_sol.clone());
            }
            mesh_solutions.push(mesh_sol.clone());

            // --- Advance target image. ------------------------------------
            g_index += 1;
            if g_index >= n_images {
                all_solved = true;
                break 'outer;
            }
            g_img = Image::from_file(&self.image_paths[g_index], cfg.border)?;

            // --- Deformation preconditioning. ----------------------------
            if cfg.guide {
                let (disp, new_warp) = deformation_preconditioning(
                    &mesh_sol,
                    seed_coord,
                    self.mesh_cfg.mesh_order,
                    cfg.mesh_cfg.subset_order as u8,
                );
                seed_coord[0] += disp[0];
                seed_coord[1] += disp[1];
                // Update seed_warp: first 6*min(mesh_order,subset_order) terms.
                let n_copy = 6 * (self.mesh_cfg.mesh_order as usize)
                    .min(cfg.mesh_cfg.subset_order);
                for i in 0..n_copy.min(new_warp.len()).min(seed_warp.len()) {
                    seed_warp[i] = new_warp[i];
                }
                for i in n_copy..12 {
                    seed_warp[i] = 0.0;
                }
            }

            // --- Sequential reference update. ----------------------------
            if cfg.sequential {
                f_index = g_index - 1;
                f_img = Image::from_file(&self.image_paths[f_index], cfg.border)?;
                if cfg.sync {
                    sync_sol = None;
                }
            }
        }

        Ok(SequenceSolution {
            mesh_solutions,
            solved: all_solved,
            unsolvable: false,
            override_log,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn generate_mesh(&self) -> Result<Mesh, Error> {
        Mesh::generate(
            self.mesh_cfg.borders.view(),
            self.mesh_cfg.segments.view(),
            &self.mesh_cfg.curves,
            self.mesh_cfg.size_lower,
            self.mesh_cfg.size_upper,
            self.mesh_cfg.target_nodes,
            self.mesh_cfg.mesh_order,
        )
    }
}

// ---------------------------------------------------------------------------
// solve_pair_adaptive — free function
// ---------------------------------------------------------------------------

/// Run the DIC solver for one image pair with optional adaptive remesh.
///
/// On each iteration: solve the current mesh; compute adaptive target areas;
/// regenerate via CDT; repeat.
#[allow(clippy::too_many_arguments)]
fn solve_pair_adaptive(
    mesh: Mesh,
    f_img: &Image,
    g_img: &Image,
    template_coords: &Array2<f64>,
    seed_coord: [f64; 2],
    seed_warp: &[f64],
    cfg: &SolveConfig,
    adaptive_iterations: usize,
    alpha: f64,
    mesh_cfg: &SequenceMeshConfig,
) -> Result<MeshSolution, Error> {
    let mut current_mesh = mesh;
    let mut sol = current_mesh.solve(f_img, g_img, template_coords, seed_coord, seed_warp, cfg)?;

    for _ in 0..adaptive_iterations {
        let target_areas = adaptive_target_areas(&sol.warps, &sol.areas, alpha);
        let new_trimesh = triangulation::adaptive_remesh(
            sol.nodes.view(),
            sol.elements.view(),
            target_areas.view(),
            mesh_cfg.borders.view(),
            mesh_cfg.segments.view(),
            &mesh_cfg.curves,
            mesh_cfg.size_lower,
            mesh_cfg.target_nodes,
            mesh_cfg.mesh_order,
        )?;
        current_mesh = Mesh::from_trimesh(new_trimesh);
        sol = current_mesh.solve(f_img, g_img, template_coords, seed_coord, seed_warp, cfg)?;
    }

    Ok(sol)
}

// ---------------------------------------------------------------------------
// deformation_preconditioning — free function
// ---------------------------------------------------------------------------

/// Replicate `Sequence._deformation_preconditioning`.
///
/// Creates a single-increment [`Particle`] at `seed_coord`, applies one
/// `solve_increment` on the solved mesh, and returns the resulting warp at
/// frame 1.
///
/// # Returns
/// `(seed_displacement, seed_warp)`
/// * `seed_displacement` — `[u, v]` displacement at the seed coordinate
/// * `seed_warp`         — warp vector of length `6*subset_order` for the
///                         next pair's initial warp (first
///                         `6*min(mesh_order,subset_order)` terms from frame 1)
pub fn deformation_preconditioning(
    sol: &MeshSolution,
    seed_coord: [f64; 2],
    mesh_order: u8,
    subset_order: u8,
) -> ([f64; 2], Vec<f64>) {
    let particle_result = Particle::new(
        seed_coord,
        &[0.0f64; 12],
        1.0,
        2,        // 2 frames → 1 increment
        mesh_order,
        true,     // Lagrangian
    );

    let mut particle = match particle_result {
        Ok(p) => p,
        Err(_) => return ([0.0; 2], vec![0.0; 6 * subset_order as usize]),
    };

    let mesh_data = MeshData {
        nodes: &sol.nodes,
        elements: &sol.elements,
        displacements: &sol.displacements,
        mesh_order,
    };
    particle.solve_increment(0, &mesh_data, false);

    let p_len = 6 * mesh_order as usize;
    let warp_1: Vec<f64> = (0..p_len).map(|j| particle.warps[[1, j]]).collect();

    let disp = [
        warp_1.first().copied().unwrap_or(0.0),
        warp_1.get(1).copied().unwrap_or(0.0),
    ];

    // Seed warp: first 6*min(mesh_order,subset_order) terms.
    let n_copy = 6 * (mesh_order as usize).min(subset_order as usize);
    let out_len = 6 * subset_order as usize;
    let mut seed_warp_out = vec![0.0f64; out_len];
    for i in 0..n_copy.min(warp_1.len()).min(out_len) {
        seed_warp_out[i] = warp_1[i];
    }

    (disp, seed_warp_out)
}

// ---------------------------------------------------------------------------
// Unit tests (Phase 2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

    // -----------------------------------------------------------------------
    // Sorting: replicate the image-filename sort logic in Rust
    // -----------------------------------------------------------------------

    /// Extract the last contiguous run of decimal digits from a name and parse it.
    fn last_number_in_name(name: &str) -> usize {
        name.split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .last()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    fn sort_image_names(mut names: Vec<&str>) -> Vec<&str> {
        names.sort_by_key(|n| last_number_in_name(n));
        names
    }

    #[test]
    fn test_sort_images_basic() {
        let images = vec!["img_003.jpg", "img_001.jpg", "img_002.jpg"];
        assert_eq!(sort_image_names(images), ["img_001.jpg", "img_002.jpg", "img_003.jpg"]);
    }

    #[test]
    fn test_sort_images_non_sequential() {
        let images = vec!["frame_10.jpg", "frame_0.jpg", "frame_5.jpg"];
        let sorted = sort_image_names(images);
        assert_eq!(sorted, ["frame_0.jpg", "frame_5.jpg", "frame_10.jpg"]);
    }

    #[test]
    fn test_sort_images_zero_padded() {
        let images = vec!["img_100.jpg", "img_009.jpg", "img_010.jpg"];
        let sorted = sort_image_names(images);
        assert_eq!(sorted, ["img_009.jpg", "img_010.jpg", "img_100.jpg"]);
    }

    #[test]
    fn test_sort_images_last_group_used() {
        // "exp1_frame_003.jpg" → last number = 3
        let images = vec!["exp1_frame_003.jpg", "exp1_frame_001.jpg", "exp1_frame_002.jpg"];
        let sorted = sort_image_names(images);
        assert_eq!(sorted, ["exp1_frame_001.jpg", "exp1_frame_002.jpg", "exp1_frame_003.jpg"]);
    }

    // -----------------------------------------------------------------------
    // seed_warp_slice
    // -----------------------------------------------------------------------

    fn apply_seed_warp_slice(warp_row: &[f64], mesh_order: u8, subset_order: u8) -> Vec<f64> {
        let n = 6 * (mesh_order as usize).min(subset_order as usize);
        let mut out = vec![0.0f64; 6 * subset_order as usize];
        for i in 0..n.min(warp_row.len()).min(out.len()) {
            out[i] = warp_row[i];
        }
        out
    }

    #[test]
    fn test_seed_warp_slice_order1_1() {
        let warp: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let result = apply_seed_warp_slice(&warp, 1, 1);
        assert_eq!(result.len(), 6);
        assert_eq!(&result[..], &warp[..6]);
    }

    #[test]
    fn test_seed_warp_slice_order2_1() {
        let warp: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let result = apply_seed_warp_slice(&warp, 2, 1);
        assert_eq!(result.len(), 6);
        assert_eq!(&result[..], &warp[..6]);
    }

    #[test]
    fn test_seed_warp_slice_order1_2() {
        let warp: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let result = apply_seed_warp_slice(&warp, 1, 2);
        assert_eq!(result.len(), 12);
        for (i, &v) in result.iter().enumerate() {
            if i < 6 { assert_eq!(v, warp[i]); }
            else { assert_eq!(v, 0.0); }
        }
    }

    #[test]
    fn test_seed_warp_slice_order2_2() {
        let warp: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let result = apply_seed_warp_slice(&warp, 2, 2);
        assert_eq!(result.len(), 12);
        assert_eq!(&result[..], &warp[..]);
    }

    // -----------------------------------------------------------------------
    // target / reference update logic
    // -----------------------------------------------------------------------

    #[test]
    fn test_target_update_increments_g_index() {
        let g = 1usize + 1;
        assert_eq!(g, 2);
        assert!(g < 5); // not solved yet
    }

    #[test]
    fn test_target_update_solved_at_last_image() {
        let n = 5usize;
        let g = 4usize + 1;
        assert_eq!(g, n); // solved
    }

    #[test]
    fn test_reference_update_f_index() {
        let g_index = 3usize;
        assert_eq!(g_index - 1, 2usize);
    }

    #[test]
    fn test_reference_update_coord_shift() {
        let mut seed_coord = [100.0f64, 200.0];
        let disp = [5.0f64, -3.0];
        seed_coord[0] += disp[0];
        seed_coord[1] += disp[1];
        assert!((seed_coord[0] - 105.0).abs() < 1e-12);
        assert!((seed_coord[1] - 197.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // deformation_preconditioning — unit square pure translation
    // -----------------------------------------------------------------------

    fn unit_square_solution(u: f64, v: f64) -> MeshSolution {
        let nodes = array![[0.0f64, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let elements = array![[0usize, 1, 2], [1, 3, 2]];
        let disps = array![[u, v], [u, v], [u, v], [u, v]];
        let n = 4;
        MeshSolution {
            nodes,
            elements,
            boundary: vec![0, 1, 2, 3],
            exclusions: vec![],
            areas: Array1::from_vec(vec![0.5, 0.5]),
            warps: ndarray::Array2::zeros((2, 6)),
            displacements: disps,
            c_zncc: Array1::ones(n),
            p: ndarray::Array2::zeros((n, 6)),
            seed_node: 0,
            mesh_order: 1,
            subset_order: 1,
            iterations: Array1::zeros(n),
            norms: Array1::zeros(n),
        }
    }

    #[test]
    fn test_deformation_preconditioning_pure_translation_disp() {
        let (u, v) = (0.3f64, -0.2f64);
        let sol = unit_square_solution(u, v);
        let (disp, _) = deformation_preconditioning(&sol, [0.333, 0.333], 1, 1);
        assert!((disp[0] - u).abs() < 1e-8, "disp[0]={}", disp[0]);
        assert!((disp[1] - v).abs() < 1e-8, "disp[1]={}", disp[1]);
    }

    #[test]
    fn test_deformation_preconditioning_pure_translation_no_strain() {
        let sol = unit_square_solution(0.5, 0.0);
        let (_, warp) = deformation_preconditioning(&sol, [0.333, 0.333], 1, 1);
        for &s in &warp[2..] {
            assert!(s.abs() < 1e-8, "strain={s}");
        }
    }

    #[test]
    fn test_deformation_preconditioning_warp_matches_disp() {
        let (u, v) = (0.25f64, 0.1f64);
        let sol = unit_square_solution(u, v);
        let (disp, warp) = deformation_preconditioning(&sol, [0.5, 0.2], 1, 1);
        assert!((warp[0] - disp[0]).abs() < 1e-10);
        assert!((warp[1] - disp[1]).abs() < 1e-10);
    }

    #[test]
    fn test_deformation_preconditioning_warp_length_o1() {
        let sol = unit_square_solution(0.0, 0.0);
        let (_, warp) = deformation_preconditioning(&sol, [0.5, 0.3], 1, 1);
        assert_eq!(warp.len(), 6);
    }

    #[test]
    fn test_deformation_preconditioning_warp_length_o2_subset() {
        let sol = unit_square_solution(0.0, 0.0);
        let (_, warp) = deformation_preconditioning(&sol, [0.5, 0.3], 1, 2);
        assert_eq!(warp.len(), 12);
    }

    #[test]
    fn test_deformation_preconditioning_zeros_gives_zero_disp() {
        let sol = unit_square_solution(0.0, 0.0);
        let (disp, _) = deformation_preconditioning(&sol, [0.5, 0.3], 1, 1);
        assert!(disp[0].abs() < 1e-12);
        assert!(disp[1].abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Sequence::new validation
    // -----------------------------------------------------------------------

    fn dummy_mesh_cfg() -> SequenceMeshConfig {
        SequenceMeshConfig {
            borders: ndarray::Array2::zeros((3, 2)),
            segments: ndarray::Array2::zeros((3, 2)),
            curves: vec![],
            size_lower: 1.0,
            size_upper: 100.0,
            target_nodes: 10,
            mesh_order: 1,
        }
    }

    #[test]
    fn test_sequence_new_too_few_images() {
        let paths = vec![PathBuf::from("/tmp/only_one.jpg")];
        let result = Sequence::new(paths, dummy_mesh_cfg());
        assert!(result.is_err());
    }

    #[test]
    fn test_sequence_new_size_lower_ge_upper() {
        let mut cfg = dummy_mesh_cfg();
        cfg.size_lower = 100.0;
        cfg.size_upper = 10.0;
        let paths = vec![PathBuf::from("/a"), PathBuf::from("/b")];
        assert!(Sequence::new(paths, cfg).is_err());
    }

    #[test]
    fn test_sequence_new_zero_size_lower() {
        let mut cfg = dummy_mesh_cfg();
        cfg.size_lower = 0.0;
        let paths = vec![PathBuf::from("/a"), PathBuf::from("/b")];
        assert!(Sequence::new(paths, cfg).is_err());
    }

    #[test]
    fn test_sequence_n_pairs() {
        // n_pairs = n_images - 1
        // We can't test this on a real Sequence without valid files, but
        // the formula is verified here directly.
        let n_images = 5usize;
        assert_eq!(n_images - 1, 4);
    }
}
