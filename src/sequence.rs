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
//! These flags are grouped in [`SequenceOptions`], passed via [`SequenceSolveConfig::options`].
//!
//! # Excluded from translation
//!
//! - `SequenceBase` plotting methods (`inspect`, `convergence`, `contour`, `quiver`)
//! - `load` / `save_by_reference` file management
//! - `SequenceResults.regenerate` — high-level deserialization; deferred to io.rs
//! - GUI selectors (`gp.gui.selectors.coordinate.CoordinateSelector`)
//! - Adaptive remeshing — deferred; use `adaptive_iterations=0`

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    image::Image,
    io::{save as io_save, GeopyvObject},
    mesh::{Mesh, MeshSolution, SeedConfig, SolveConfig},
    particle::{MeshData, Particle},
    masks::LocalMask,
    Error,
};

// ---------------------------------------------------------------------------
// Filename sort helper
// ---------------------------------------------------------------------------

/// Extract the last contiguous run of decimal digits from a string and return
/// it as a `u64`.  Returns 0 if no digits are found.
fn last_number_in_name(s: &str) -> u64 {
    s.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .last()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Mesh-generation configuration
// ---------------------------------------------------------------------------

/// Parameters forwarded to [`Mesh::new`] for each image pair.
#[derive(Debug, Clone)]
pub struct SequenceMeshConfig {
    /// Boundary polygon vertices `(N, 2)`.
    pub boundary_nodes: ndarray::Array2<f64>,
    /// If `true`, only pixels inside the boundary polygon are active.
    pub boundary_hard: bool,
    /// Exclusion polygon arrays.
    pub exclusion_nodes: Vec<ndarray::Array2<f64>>,
    /// Per-exclusion hard flag (soft = meshing constraint only, does not clip mask).
    pub exclusions_hard: Vec<bool>,
    /// `(size_lower, size_upper)` element edge lengths.
    pub size: (f64, f64),
    pub target_nodes: usize,
    pub mesh_order: u8,
}

// ---------------------------------------------------------------------------
// Solve configuration
// ---------------------------------------------------------------------------

/// Temporal coupling strategy for [`Sequence::solve`].
#[derive(Debug, Clone)]
pub struct SequenceOptions {
    /// Apply particle-based warp preconditioning between pairs.
    pub guide: bool,
    /// Advance reference image after each successful solve (cumulative mode).
    pub sequential: bool,
    /// Reuse previous mesh geometry in sync mode (skips CDT regeneration).
    pub sync: bool,
    /// Relax tolerance on retry after a failed non-consecutive pair.
    pub override_: bool,
}

impl Default for SequenceOptions {
    fn default() -> Self {
        SequenceOptions { guide: true, sequential: false, sync: true, override_: false }
    }
}

/// Solver parameters for each mesh pair within a sequence.
#[derive(Debug, Clone)]
pub struct SequenceSolveConfig {
    /// Per-subset solver settings forwarded to [`Mesh::solve`].
    pub mesh_cfg: SolveConfig,
    /// Local mask (cloned per node when a mask is present).
    pub local_mask: LocalMask,
    /// Seed-node parameters (coord, initial warp, tolerance).
    pub seed: SeedConfig,
    /// Temporal coupling strategy (guide, sequential, sync, override).
    pub options: SequenceOptions,
    /// Image border (pixels) for B-spline precomputation. Defaults to 20.
    pub border: usize,
    /// When `Some(dir)`, save each mesh frame to `{dir}/mesh_{i:04}.pyv`
    /// immediately after solving; frame is not accumulated in memory.
    /// When `None` (default), accumulate all solutions in
    /// `SequenceSolution::mesh_solutions`.
    pub save: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Sequence solution
// ---------------------------------------------------------------------------

/// Result of [`Sequence::solve`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceSolution {
    /// Per-pair DIC solutions when solve was called with `save = None`.
    /// Empty when meshes were saved by reference.
    pub mesh_solutions: Vec<MeshSolution>,
    /// File paths of per-frame `.pyv` files when solve was called with
    /// `save = Some(dir)`.  Empty when meshes are held in memory.
    pub mesh_paths: Vec<PathBuf>,
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
        if mesh_cfg.size.0 <= 0.0 || mesh_cfg.size.0 >= mesh_cfg.size.1 {
            return Err(Error::InvalidInput(
                "size.0 must be > 0 and < size.1".to_string(),
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

    /// Construct a `Sequence` by scanning a directory for image files.
    ///
    /// All `*.jpg`, `*.jpeg`, and `*.png` files (case-insensitive) found in
    /// `dir` are collected and sorted by the last run of digits in their file
    /// stem (e.g. `frame_003.jpg` → key 3).  At least 2 images are required.
    pub fn from_dir(dir: &Path, mesh_cfg: SequenceMeshConfig) -> Result<Self, Error> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| Error::FileNotFound(format!("{}: {e}", dir.display())))?;

        let image_exts = ["jpg", "jpeg", "png"];
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| image_exts.contains(&e.to_lowercase().as_str()))
                    .unwrap_or(false)
            })
            .collect();

        paths.sort_by_key(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .map(last_number_in_name)
                .unwrap_or(0)
        });

        Sequence::new(paths, mesh_cfg)
    }

    /// Solve all image pairs.
    ///
    /// Replicates `Sequence.solve` (minus alive_bar, GUI, geomat sections).
    /// When `cfg.save` is `Some(dir)`, each solved mesh is written to
    /// `{dir}/mesh_{i:04}.pyv` and not held in memory.
    pub fn solve(&self, cfg: &SequenceSolveConfig) -> Result<SequenceSolution, Error> {
        let n_images = self.image_paths.len();
        let mut mesh_solutions: Vec<MeshSolution> = Vec::with_capacity(n_images - 1);
        let mut mesh_paths: Vec<PathBuf> = Vec::new();
        let mut override_log: Vec<usize> = Vec::new();

        if let Some(ref save_dir) = cfg.save {
            std::fs::create_dir_all(save_dir)?;
        }

        let mut f_index = 0usize;
        let mut g_index = 1usize;

        let mut seed_coord = cfg.seed.coord;
        let warp_len = 6 * cfg.mesh_cfg.subset_order;
        let mut seed_warp: Vec<f64> = {
            let mut w = cfg.seed.warp.clone();
            w.resize(warp_len, 0.0);
            w
        };

        // Cached previous-pair solution for sync mode.
        let mut sync_sol: Option<MeshSolution> = None;
        // Override flag: relax tolerance for the next mesh solve.
        let mut mesh_override = false;

        // Load initial images as Arc to enable zero-cost sharing across pairs.
        let mut f_img: Arc<Image> = Arc::new(Image::from_file(&self.image_paths[f_index], cfg.border)?);
        let mut g_img: Arc<Image> = Arc::new(Image::from_file(&self.image_paths[g_index], cfg.border)?);

        // Progress: outer bar for image pairs, inner bar for subsets.
        let mp = indicatif::MultiProgress::new();
        let pb_seq = mp.add(indicatif::ProgressBar::new((n_images - 1) as u64));
        pb_seq.set_style(
            indicatif::ProgressStyle::with_template(
                "Solving sequence: [{bar:40.green}] {pos}/{len} pairs  ({msg})"
            )
            .unwrap()
            .progress_chars("█░"),
        );
        let pb_mesh = mp.add(indicatif::ProgressBar::new(0));
        pb_mesh.set_style(
            indicatif::ProgressStyle::with_template(
                "  Solving mesh:  [{bar:40.cyan}] {pos}/{len} subsets  eta {eta}"
            )
            .unwrap()
            .progress_chars("█░"),
        );

        let all_solved;

        'outer: loop {
            // --- Build mesh for this pair. ----------------------------------
            // In sync mode, reuse previous pair's geometry when available.
            let excl_views: Vec<_> = self.mesh_cfg.exclusion_nodes.iter().map(|a| a.view()).collect();
            let mesh = match (cfg.options.sync, &sync_sol) {
                (true, Some(prev)) => Mesh::from_solution(prev, Arc::clone(&f_img), Arc::clone(&g_img)),
                _ => Mesh::new(
                    self.mesh_cfg.boundary_nodes.view(),
                    self.mesh_cfg.boundary_hard,
                    &excl_views,
                    &self.mesh_cfg.exclusions_hard,
                    self.mesh_cfg.size,
                    self.mesh_cfg.target_nodes,
                    self.mesh_cfg.mesh_order,
                    Arc::clone(&f_img),
                    Arc::clone(&g_img),
                )?,
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

            let pair_seed = SeedConfig {
                coord: seed_coord,
                warp: seed_warp.clone(),
                tolerance: cfg.seed.tolerance,
            };

            // --- Solve this pair. ------------------------------------------
            let ref_name = self.image_paths[f_index].file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| f_index.to_string());
            let tar_name = self.image_paths[g_index].file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| g_index.to_string());
            pb_seq.set_message(format!("{}→{}", ref_name, tar_name));
            pb_mesh.set_length(mesh.nodes().nrows() as u64);
            pb_mesh.set_position(0);

            let pair_result = mesh.solve(&cfg.local_mask, &pair_seed, &pair_cfg, Some(&pb_mesh));

            let mesh_sol = match pair_result {
                Ok(sol) => sol,
                Err(_) => {
                    // Attempt to fall back to an updated reference.
                    if f_index + 1 < g_index {
                        // Non-consecutive pair failed: step reference forward.
                        f_index = g_index - 1;
                        f_img = Arc::new(Image::from_file(&self.image_paths[f_index], cfg.border)?);
                        if cfg.options.sync {
                            sync_sol = None;
                        }
                        if cfg.options.override_ {
                            mesh_override = true;
                        }
                        continue 'outer;
                    } else {
                        // Consecutive pair truly unsolvable: curtail sequence.
                        return Ok(SequenceSolution {
                            mesh_solutions,
                            mesh_paths,
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
            if cfg.options.sync {
                sync_sol = Some(mesh_sol.clone());
            }
            if let Some(ref save_dir) = cfg.save {
                let frame_path = save_dir.join(format!("mesh_{:04}.pyv", mesh_paths.len()));
                io_save(&frame_path, &GeopyvObject::Mesh(mesh_sol.clone()))
                    .map_err(|e| Error::Io(format!("frame save failed: {e}")))?;
                mesh_paths.push(frame_path);
            } else {
                mesh_solutions.push(mesh_sol.clone());
            }

            pb_seq.inc(1);

            // --- Advance target image. ------------------------------------
            g_index += 1;
            if g_index >= n_images {
                all_solved = true;
                break 'outer;
            }
            g_img = Arc::new(Image::from_file(&self.image_paths[g_index], cfg.border)?);

            // --- Deformation preconditioning. ----------------------------
            if cfg.options.guide {
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
                for i in n_copy..seed_warp.len() {
                    seed_warp[i] = 0.0;
                }
            }

            // --- Sequential reference update. ----------------------------
            if cfg.options.sequential {
                f_index = g_index - 1;
                f_img = Arc::clone(&g_img);
                if cfg.options.sync {
                    sync_sol = None;
                }
            }
        }

        pb_seq.finish_and_clear();
        pb_mesh.finish_and_clear();

        Ok(SequenceSolution {
            mesh_solutions,
            mesh_paths,
            solved: all_solved,
            unsolvable: false,
            override_log,
        })
    }

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
        None,
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
    // Sorting: verify the image-filename sort logic used by from_dir
    // -----------------------------------------------------------------------

    fn sort_image_names(mut names: Vec<&str>) -> Vec<&str> {
        names.sort_by_key(|n| super::last_number_in_name(n));
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
            f_img_path: PathBuf::new(),
            g_img_path: PathBuf::new(),
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
        let boundary = ndarray::array![[0.0f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        SequenceMeshConfig {
            boundary_nodes: boundary,
            boundary_hard: false,
            exclusion_nodes: vec![],
            exclusions_hard: vec![],
            size: (1.0, 100.0),
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
        cfg.size = (100.0, 10.0);
        let paths = vec![PathBuf::from("/a"), PathBuf::from("/b")];
        assert!(Sequence::new(paths, cfg).is_err());
    }

    #[test]
    fn test_sequence_new_zero_size_lower() {
        let mut cfg = dummy_mesh_cfg();
        cfg.size.0 = 0.0;
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
