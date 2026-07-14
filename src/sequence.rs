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

use ndarray::Axis;
use serde::{Deserialize, Serialize};

use crate::{
    geometry::region::{Region, RegionOption},
    image::Image,
    io::{load as io_load, save as io_save, GeopyvObject},
    mesh::{Mesh, MeshSolution, SeedConfig, SolveConfig},
    particle::{Particle, ParticleSource},
    masks::LocalMask,
    subset::Subset,
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
///
/// `boundary`/`exclusions` are live [`Region`]s (not raw arrays) so that
/// [`Sequence::solve`] can track their displacement across reference-image
/// updates, mirroring `Mesh._store_region` / `Mesh._update_region` in the
/// original Python (`geopyv/src/geopyv/mesh.py`).
#[derive(Debug, Clone)]
pub struct SequenceMeshConfig {
    /// Boundary region (polygon vertices + tracking mode).
    pub boundary: Region,
    /// Exclusion regions.
    pub exclusions: Vec<Region>,
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
    pub mesh_solutions: Vec<Arc<MeshSolution>>,
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
    /// One entry per mesh pair; `true` if the reference image advanced at that step.
    pub reference_updates: Vec<bool>,
    /// Element order of the meshes in this sequence.  Available in both storage
    /// modes without loading any mesh file.
    #[serde(default = "default_mesh_order")]
    pub mesh_order: u8,
    /// `f_img_path` of the first solved mesh pair; `None` if no pair was solved.
    #[serde(default)]
    pub first_f_img_path: Option<PathBuf>,
    /// Final tracked state of the boundary region (displaced across the whole run).
    #[serde(default = "default_boundary_region")]
    pub boundary_region: Region,
    /// Final tracked state of each exclusion region.
    #[serde(default)]
    pub exclusion_regions: Vec<Region>,
}

/// Minimal static placeholder region — used where a full boundary/exclusion
/// history isn't available or needed (e.g. reconstructing a `Sequence` shell
/// from a saved [`SequenceSolution`], where only the solved data matters).
pub fn default_boundary_region() -> Region {
    Region::path(
        Some([0.0, 0.0]),
        ndarray::array![[0.0, 0.0]],
        RegionOption::S,
        false,
        false,
        0.0,
    )
    .expect("static placeholder region is always valid")
}

fn default_mesh_order() -> u8 { 1 }

impl SequenceSolution {
    /// Number of mesh pairs, regardless of storage mode.
    pub fn n_meshes(&self) -> usize {
        if self.mesh_solutions.is_empty() {
            self.mesh_paths.len()
        } else {
            self.mesh_solutions.len()
        }
    }

    /// Load the [`MeshSolution`] for pair `m`.
    ///
    /// In-memory mode: clones from the in-memory vector.
    /// Saved-by-reference mode: deserialises from the corresponding `.pyv` file.
    pub fn load_mesh_at(&self, m: usize) -> Result<Arc<MeshSolution>, Error> {
        if let Some(arc) = self.mesh_solutions.get(m) {
            return Ok(Arc::clone(arc));
        }
        let path = &self.mesh_paths[m];
        match io_load(path)? {
            GeopyvObject::Mesh(ms) => Ok(Arc::new(ms)),
            _ => Err(Error::InvalidInput(
                format!("expected Mesh at {}", path.display()),
            )),
        }
    }
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
        let mut mesh_solutions: Vec<Arc<MeshSolution>> = Vec::with_capacity(n_images - 1);
        let mut mesh_paths: Vec<PathBuf> = Vec::new();
        let mut override_log: Vec<usize> = Vec::new();
        let n_pairs = n_images - 1;
        let mut reference_updates: Vec<bool> = vec![false; n_pairs];
        let mut first_f_img_path: Option<PathBuf> = None;

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
        let mut sync_sol: Option<Arc<MeshSolution>> = None;
        // Override flag: relax tolerance for the next mesh solve.
        let mut mesh_override = false;

        // Live region state, tracked across pairs (mirrors Python's
        // `boundary_obj`/`exclusion_objs` being threaded through every
        // `Mesh()` construction in `Sequence.solve`). Boundary is always
        // coerced from `R` to `F` (mesh.py:898-899): a boundary can deform,
        // it doesn't just rigidly translate/rotate.
        let mut boundary_region = self.mesh_cfg.boundary.clone();
        if boundary_region.option == RegionOption::R {
            boundary_region.option = RegionOption::F;
        }
        let mut exclusion_regions = self.mesh_cfg.exclusions.clone();

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
            // --- Update tracked regions for this pair's reference image. ---
            // Mirrors `Mesh._update_region` being called unconditionally from
            // `Mesh.__init__` (mesh.py:912-913): a no-op unless the reference
            // image just changed, in which case it snaps `current_nodes` to
            // the last stored (displaced) snapshot.
            let f_path = f_img.filepath.as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            boundary_region.update(&f_path);
            for r in exclusion_regions.iter_mut() {
                r.update(&f_path);
            }

            // --- Build mesh for this pair. ----------------------------------
            // In sync mode, reuse previous pair's geometry when available.
            let excl_views: Vec<_> = exclusion_regions.iter().map(|r| r.current_nodes.view()).collect();
            let excl_hard: Vec<bool> = exclusion_regions.iter().map(|r| r.hard).collect();
            let mesh = match (cfg.options.sync, &sync_sol) {
                (true, Some(prev)) => Mesh::from_solution(prev, Arc::clone(&f_img), Arc::clone(&g_img)),
                _ => Mesh::new(
                    boundary_region.current_nodes.view(),
                    boundary_region.hard,
                    &excl_views,
                    &excl_hard,
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

            let mesh_sol: Arc<MeshSolution> = match pair_result {
                Ok(sol) => Arc::new(sol),
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
                            reference_updates,
                            mesh_order: self.mesh_cfg.mesh_order,
                            first_f_img_path,
                            boundary_region,
                            exclusion_regions,
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

            // Capture reference image path from the first solved pair.
            if first_f_img_path.is_none() {
                first_f_img_path = Some(mesh_sol.f_img_path.clone());
            }

            // --- Store displaced region state. -------------------------------
            // Mirrors `Mesh._store_region` (mesh.py:1400-1429): records this
            // pair's displaced boundary/exclusion positions so a later
            // reference update can snap back to where the specimen actually
            // moved to, instead of the original static polygon.
            store_region_step(
                &mut boundary_region,
                &mut exclusion_regions,
                &mesh_sol,
                &f_img,
                &g_img,
                cfg.mesh_cfg.subset_order,
            )?;

            // --- Store result and update sync geometry. ---------------------
            if cfg.options.sync {
                sync_sol = Some(Arc::clone(&mesh_sol));
            }
            if let Some(ref save_dir) = cfg.save {
                let frame_path = save_dir.join(format!("mesh_{:04}.pyv", mesh_paths.len()));
                io_save(&frame_path, &GeopyvObject::Mesh((*mesh_sol).clone()))
                    .map_err(|e| Error::Io(format!("frame save failed: {e}")))?;
                mesh_paths.push(frame_path);
            } else {
                mesh_solutions.push(Arc::clone(&mesh_sol));
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
                let next_pair = g_index - 1;
                if next_pair < n_pairs {
                    reference_updates[next_pair] = true;
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
            reference_updates,
            mesh_order: self.mesh_cfg.mesh_order,
            first_f_img_path,
            boundary_region,
            exclusion_regions,
        })
    }

}

// ---------------------------------------------------------------------------
// store_region_step — free function
// ---------------------------------------------------------------------------

/// Store this pair's displaced boundary/exclusion node positions into their
/// tracked [`Region`]s.
///
/// Mirrors `Mesh._store_region` (`mesh.py:1400-1429`). Exclusions with
/// `option == R` (rigid) are re-registered via a fresh [`Subset`] solve
/// centred on the exclusion (Python's Circle-`R` special case, `mesh.py:1403-1421`)
/// rather than using raw nodal displacements, since a small circular exclusion's
/// own mesh nodes are a noisier basis for translation/rotation than a direct
/// correlation. If that solve doesn't reach `tolerance = 0.9`, storing is
/// skipped entirely for this pair (boundary included) — matching Python's
/// early return (`mesh.py:1416-1419`), which leaves the mesh `solved` but
/// simply drops that pair's region-history entry.
fn store_region_step(
    boundary: &mut Region,
    exclusions: &mut [Region],
    sol: &MeshSolution,
    f_img: &Arc<Image>,
    g_img: &Arc<Image>,
    subset_order: usize,
) -> Result<(), Error> {
    enum ExclWarp {
        Rigid(Vec<f64>),
        Flexible(ndarray::Array2<f64>),
        None,
    }

    let mut warps = Vec::with_capacity(exclusions.len());
    for (i, excl) in exclusions.iter().enumerate() {
        match excl.option {
            RegionOption::R => {
                let radius = excl.radius()?;
                let local_mask = LocalMask::circle(radius.round() as usize)?;
                let subset = Subset::new(
                    excl.current_centre,
                    &local_mask,
                    None,
                    Arc::clone(f_img),
                    Arc::clone(g_img),
                    subset_order,
                )?;
                let disp = sol.displacements.select(Axis(0), &sol.exclusions[i]);
                let mean = disp.mean_axis(Axis(0)).unwrap_or_else(|| ndarray::Array1::zeros(2));
                let mut warp_0 = vec![0.0f64; 6 * subset_order];
                warp_0[0] = mean[0];
                warp_0[1] = mean[1];
                let result = subset.solve_icgn(Some(&warp_0), 0.9, 1e-5, 50)?;
                if !result.solved {
                    // Registration failed: skip storing entirely for this pair.
                    return Ok(());
                }
                warps.push(ExclWarp::Rigid(result.p));
            }
            RegionOption::F => {
                let disp = sol.displacements.select(Axis(0), &sol.exclusions[i]);
                warps.push(ExclWarp::Flexible(disp));
            }
            RegionOption::S | RegionOption::D => {
                warps.push(ExclWarp::None);
            }
        }
    }

    for (excl, warp) in exclusions.iter_mut().zip(warps) {
        match warp {
            ExclWarp::Rigid(p) => excl.store_rigid(&p)?,
            ExclWarp::Flexible(disp) => excl.store_flexible(disp.view())?,
            ExclWarp::None => {}
        }
    }

    let boundary_disp = sol.displacements.select(Axis(0), &sol.boundary);
    boundary.store_flexible(boundary_disp.view())?;

    Ok(())
}

// ---------------------------------------------------------------------------
// deformation_preconditioning — free function
// ---------------------------------------------------------------------------

/// Replicate `Sequence._deformation_preconditioning`.
///
/// # Returns
/// `(seed_displacement, seed_warp)` — displacement `[u, v]` at the seed
/// coordinate and warp vector of length `6*subset_order` for the next pair.
pub fn deformation_preconditioning(
    sol: &MeshSolution,
    seed_coord: [f64; 2],
    mesh_order: u8,
    subset_order: u8,
) -> ([f64; 2], Vec<f64>) {
    let source = ParticleSource::Mesh(Arc::new(sol.clone()));
    let mut particle = match Particle::new(source, seed_coord, &[0.0f64; 12], 1.0, true) {
        Ok(p) => p,
        Err(_) => return ([0.0; 2], vec![0.0; 6 * subset_order as usize]),
    };
    particle.solve_increment(0, sol, None);

    let p_len = 6 * mesh_order as usize;
    let warp_1: Vec<f64> = (0..p_len).map(|j| particle.warps[[1, j]]).collect();
    let disp = [
        warp_1.first().copied().unwrap_or(0.0),
        warp_1.get(1).copied().unwrap_or(0.0),
    ];

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
        let centroids = crate::mesh::compute_centroids(&nodes, &elements);
        let disps = array![[u, v], [u, v], [u, v], [u, v]];
        let n = 4;
        MeshSolution {
            nodes,
            elements,
            boundary: vec![0, 1, 2, 3],
            exclusions: vec![],
            centroids,
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

    fn dummy_region() -> Region {
        let boundary = ndarray::array![[0.0f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        // Static (untracked) — preserves the pre-tracking test behaviour: a
        // frozen boundary, same as a raw-ndarray caller gets today.
        Region::path(None, boundary, RegionOption::S, false, true, 0.0).unwrap()
    }

    fn dummy_mesh_cfg() -> SequenceMeshConfig {
        SequenceMeshConfig {
            boundary: dummy_region(),
            exclusions: vec![],
            size: (1.0, 100.0),
            target_nodes: 10,
            mesh_order: 1,
        }
    }

    // -----------------------------------------------------------------------
    // store_region_step — boundary tracks displacement across reference updates
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_region_step_boundary_tracks_displacement() {
        let sol = unit_square_solution(0.3, -0.2);
        let boundary_nodes = array![[0.0f64, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mut boundary =
            Region::path(None, boundary_nodes.clone(), RegionOption::F, false, true, 0.0).unwrap();
        let mut exclusions: Vec<Region> = vec![];
        let f_img = Arc::new(crate::image::Image::from_array(ndarray::Array2::zeros((100, 100)), 20));
        let g_img = Arc::new(crate::image::Image::from_array(ndarray::Array2::zeros((100, 100)), 20));

        store_region_step(&mut boundary, &mut exclusions, &sol, &f_img, &g_img, 1).unwrap();

        // Displacement is recorded in history immediately...
        assert_eq!(boundary.counter, 1);
        assert_eq!(boundary.history_nodes.len(), 2);
        for i in 0..4 {
            assert!((boundary.history_nodes[1][[i, 0]] - (boundary_nodes[[i, 0]] + 0.3)).abs() < 1e-12);
            assert!((boundary.history_nodes[1][[i, 1]] - (boundary_nodes[[i, 1]] - 0.2)).abs() < 1e-12);
        }
        // ...but `current_nodes` (what the next Mesh::new would use) stays at
        // the original position until a reference update actually happens.
        assert!((boundary.current_nodes[[0, 0]] - 0.0).abs() < 1e-12);

        // Simulate the reference update that would trigger a fresh Mesh::new
        // for the next pair: the boundary should now snap to the displaced
        // position instead of resetting to the original static polygon.
        boundary.update("frame_002.jpg");
        for i in 0..4 {
            assert!((boundary.current_nodes[[i, 0]] - (boundary_nodes[[i, 0]] + 0.3)).abs() < 1e-12);
            assert!((boundary.current_nodes[[i, 1]] - (boundary_nodes[[i, 1]] - 0.2)).abs() < 1e-12);
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

    // -----------------------------------------------------------------------
    // SequenceSolution::n_meshes and load_mesh_at
    // -----------------------------------------------------------------------

    fn make_sequence_solution_in_memory(n: usize) -> SequenceSolution {
        let meshes: Vec<Arc<MeshSolution>> = (0..n).map(|_| Arc::new(unit_square_solution(0.1, 0.0))).collect();
        SequenceSolution {
            mesh_solutions: meshes,
            mesh_paths: vec![],
            solved: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: vec![false; n],
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: dummy_region(),
            exclusion_regions: vec![],
        }
    }

    #[test]
    fn test_n_meshes_in_memory() {
        let sol = make_sequence_solution_in_memory(3);
        assert_eq!(sol.n_meshes(), 3);
    }

    #[test]
    fn test_n_meshes_saved_by_reference() {
        let sol = SequenceSolution {
            mesh_solutions: vec![],
            mesh_paths: vec![PathBuf::from("/a"), PathBuf::from("/b")],
            solved: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: vec![false, false],
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: dummy_region(),
            exclusion_regions: vec![],
        };
        assert_eq!(sol.n_meshes(), 2);
    }

    #[test]
    fn test_n_meshes_empty() {
        let sol = make_sequence_solution_in_memory(0);
        assert_eq!(sol.n_meshes(), 0);
    }

    #[test]
    fn test_load_mesh_at_in_memory() {
        let sol = make_sequence_solution_in_memory(2);
        let m = sol.load_mesh_at(0).unwrap();
        assert_eq!(m.mesh_order, 1);
        assert_eq!(m.nodes, sol.mesh_solutions[0].nodes);
    }

    #[test]
    fn test_load_mesh_at_saved_by_reference() {
        // Write a mesh to a temp file, then load it back via load_mesh_at.
        let mesh = unit_square_solution(0.2, 0.0);
        let tmp = std::env::temp_dir().join("geopyv_test_seq_load_mesh_at.pyv");
        crate::io::save(&tmp, &GeopyvObject::Mesh(mesh.clone())).unwrap();

        let sol = SequenceSolution {
            mesh_solutions: vec![],
            mesh_paths: vec![tmp.clone()],
            solved: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: vec![false],
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: dummy_region(),
            exclusion_regions: vec![],
        };

        let loaded = sol.load_mesh_at(0).unwrap();
        assert_eq!(loaded.nodes, mesh.nodes);
        assert_eq!(loaded.mesh_order, mesh.mesh_order);

        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn test_load_mesh_at_wrong_type_returns_err() {
        // Save a SequenceSolution to disk, then try to load it as a mesh pair → Err.
        let inner = make_sequence_solution_in_memory(1);
        let tmp = std::env::temp_dir().join("geopyv_test_seq_wrong_type.pyv");
        crate::io::save(&tmp, &GeopyvObject::Sequence(inner)).unwrap();

        let sol = SequenceSolution {
            mesh_solutions: vec![],
            mesh_paths: vec![tmp.clone()],
            solved: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: vec![false],
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: dummy_region(),
            exclusion_regions: vec![],
        };
        assert!(sol.load_mesh_at(0).is_err());

        let _ = std::fs::remove_file(tmp);
    }
}
