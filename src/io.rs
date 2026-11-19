//! Serialisation / deserialisation for geopyv-dev result objects.
//!
//! Translates `geopyv/src/geopyv/io.py` (save/load functions).
//!
//! # Format
//!
//! Every `.pyv` file written by this module has the following layout:
//!
//! ```text
//! [0..4]   magic  b"GPYV"
//! [4]      version  0x01 (uncompressed, schema v0) | 0x02 (zstd, schema v0)
//!                    | 0x03 (zstd, schema v1) | 0x07 (zstd, schema v2)
//!                    | 0x08 (zstd, schema v3) | 0x09 (zstd, schema v4)
//!                    | 0x0A (zstd, schema v5) | 0x0B (zstd, schema v6)
//!                    | 0x0C (zstd, schema v7) | 0x0D (zstd, schema v8)
//!                    | 0x0E (zstd, schema v9) | 0x0F (zstd, schema v10)
//!                    | 0x10 (zstd, schema v11) | 0x11 (zstd, schema v12)
//!                    | 0x12 (zstd, schema v13) | 0x13 (zstd, schema v14)
//!                    | 0x14 (zstd, schema v15) | 0x15 (zstd, schema v16)
//!                    | 0x16 (zstd, schema v17) | 0x17 (zstd, schema v18 — current)
//! [5..]    bincode v2 (standard config) encoded GeopyvObject (schema-version-
//!          dependent shape — see "Schema versioning" below), zstd-compressed
//!          (level 3) for every version except 0x01
//! ```
//!
//! New files are always written with version 0x17 (zstd + current schema).
//! Bytes 0x04-0x06 were used only by unreleased development builds and are
//! deliberately rejected as unsupported.
//!
//! The type tag is embedded in the bincode stream via serde's enum encoding.
//!
//! # Schema versioning
//!
//! bincode is a positional, non-self-describing format: unlike JSON, a struct
//! field that's missing from the byte stream does **not** fall back to
//! `#[serde(default)]` — decoding just fails (or worse, silently misaligns and
//! decodes garbage) partway through. Relying on `#[serde(default)]` alone for
//! forward/backward compatibility does not work with this format, no matter
//! how many fields carry that attribute.
//!
//! The version byte above is therefore the *only* thing that determines how
//! the payload is decoded, and it must be bumped every time any type reachable
//! from [`GeopyvObject`] gains, removes, or reorders a field:
//!
//! 1. Freeze the current shapes of every changed type as `Legacy*` structs
//!    below (copy-paste the struct as it stood *before* your change).
//! 2. Add `From<Legacy*> for *` conversions, filling new fields with sensible
//!    defaults (`None`, `0`, etc. — whatever the type used to implicitly mean).
//! 3. Bump [`CURRENT_VERSION`] to a new, previously-unused byte value.
//! 4. In [`load`], the new version byte decodes directly via the current
//!    types; every older byte value decodes via [`LegacyGeopyvObject`] (or a
//!    chain of them, if there are multiple historical shapes still in use)
//!    and is converted with `.into()`.
//!
//! This is deliberately manual rather than automatic: it guarantees files
//! saved by any given release keep loading in every later release, without
//! depending on bincode behaviour that doesn't actually provide that guarantee.
//!
//! # Breaking change from geopyv Python
//!
//! The Python `io.py` uses `pickle` with a `dict`-based data model. The Rust
//! format is intentionally incompatible: old `.pyv` files cannot be read here,
//! and files written here cannot be read by the Python package. Users who need
//! files from an older version should use the corresponding Python release.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    calibration::CalibrationSolution,
    field::{FieldRegion, FieldSolution},
    geometry::region::Region,
    masks::MaskShape,
    mesh::{MeshSolution, SeedConfig, SolveConfig, SolveMethod, Texture},
    particle::{principal_strains, ParticleConfig, ParticleSolution, StrainMethod},
    sequence::{SequenceOptions, SequenceSolution},
    speckle::{Progression, ScaleType, Speckle, WarpMode},
    subset::{MaskSummary, SolveResult, SubsetSolution},
    Error,
};

// ---------------------------------------------------------------------------
// Magic header
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"GPYV";
/// Legacy uncompressed format; schema v0. Still readable on load.
const VERSION_UNCOMPRESSED: u8 = 0x01;
/// zstd-compressed, schema v0 (pre-2026-07 solve-settings/mask fields). Still
/// readable on load, but no longer written.
const VERSION_ZSTD_V0: u8 = 0x02;
/// zstd-compressed, schema v1: the format written by the released
/// package (GitHub `main` at `df92b03`). Still readable on load, but no
/// longer written.
const VERSION_ZSTD_V1: u8 = 0x03;
/// zstd-compressed, schema v2. Relative to v1: `MeshSolution`
/// gains `template_shape` / `template_sizes` / `zonal_masking`;
/// `SolveConfig` gains `preconditioning` / `layer_rg`; `SolveResult` gains
/// `eta_omitted`; `ParticleSolution` gains `principal_strains` /
/// `gamma_max_grad`; `ParticleConfig` gains `strain_method`; `Speckle` gains
/// `speckle_limit`. Still readable on load, but no longer written. (0x04-0x06
/// skipped: used by unreleased development builds with incompatible shapes.)
const VERSION_ZSTD_V2: u8 = 0x07;
/// zstd-compressed, schema v3. Relative to v2: `FieldSolution` gains
/// `region`. Still readable on load, but no longer written.
const VERSION_ZSTD_V3: u8 = 0x08;
/// zstd-compressed, schema v4. Relative to v3: `SubsetSolution`,
/// `MeshSolution`, `SequenceSolution` (+ `time_source`), `ParticleSolution`
/// and `FieldSolution` gain `time`. Still readable on load, but no longer
/// written.
const VERSION_ZSTD_V4: u8 = 0x09;
/// zstd-compressed, schema v5. Relative to v4: `MeshSolution` loses
/// `zonal_masking` (zonal masking removed; a stored record is dropped on
/// load). Still readable on load, but no longer written.
const VERSION_ZSTD_V5: u8 = 0x0A;
/// zstd-compressed, schema v6. Relative to v5: `SubsetSolution` gains
/// `seams`; `MeshSolution` gains `seams`; `SolveConfig` gains `masking`. Still
/// readable on load, but no longer written.
const VERSION_ZSTD_V6: u8 = 0x0B;
/// zstd-compressed, schema v7. Relative to v6: `MeshSolution` gains
/// `tracks` (tracked sequences); `StrainMethod` gains `ElementWarps` (appended,
/// so older tags are unchanged). Still readable on load, but no longer
/// written.
const VERSION_ZSTD_V7: u8 = 0x0C;
/// zstd-compressed, schema v8. Relative to v7: `SolveResult` and
/// `SolveConfig` gain `texture`. Still readable on load, but no longer
/// written.
const VERSION_ZSTD_V8: u8 = 0x0D;
/// zstd-compressed, schema v9. Relative to v8: `Speckle` gains
/// `grain_shape` and `grain_rotation`. Still readable on load, but no longer
/// written.
const VERSION_ZSTD_V9: u8 = 0x0E;
/// zstd-compressed, schema v10. Relative to v9: `Speckle` gains
/// `grain_overlap`. Still readable on load, but no longer written.
const VERSION_ZSTD_V10: u8 = 0x0F;
/// zstd-compressed, schema v11. Relative to v10: `Speckle` gains
/// `no_overlap`. Still readable on load, but no longer written.
const VERSION_ZSTD_V11: u8 = 0x10;
/// zstd-compressed, schema v12. Relative to v11: `Speckle` gains `soil`.
/// Still readable on load, but no longer written.
const VERSION_ZSTD_V12: u8 = 0x11;
/// zstd-compressed, schema v13. Relative to v12: `SoilConfig` is specified
/// by matrix and seeding brightness and spread and the artificial seeding
/// ratio (was grain brightness, grain spread, coverage, pattern strength).
/// Still readable on load, but no longer written.
const VERSION_ZSTD_V13: u8 = 0x12;
/// zstd-compressed, schema v14. Relative to v13: `SoilConfig` loses
/// `depth_dimming`. Still readable on load, but no longer written.
const VERSION_ZSTD_V14: u8 = 0x13;
/// zstd-compressed, schema v15. Relative to v14: `Speckle` gains `seed`.
/// Still readable on load, but no longer written.
const VERSION_ZSTD_V15: u8 = 0x14;
/// zstd-compressed, schema v16. Relative to v15: `SoilConfig`'s grain size
/// and shape (`grain_size`, `grain_size_spread`, `grain`) become a
/// characterisation (`psd`, `d_p`, aspect ratio and sphericity with their
/// spreads). Still readable on load, but no longer written.
const VERSION_ZSTD_V16: u8 = 0x15;
/// zstd-compressed, schema v17. Relative to v16: `SoilConfig` loses
/// `sphericity` / `sphericity_sd` and gains `outlines` (outline statistics).
/// Still readable on load, but no longer written.
const VERSION_ZSTD_V17: u8 = 0x16;
/// zstd-compressed, schema v18 (current). Relative to v17: `SoilConfig` gains
/// brightness variation inside grains (`grain_tilt`, `grain_dome`,
/// `grain_rim`, `grain_texture`, `texture_size`). Always written by [`save`].
const CURRENT_VERSION: u8 = 0x17;

// ---------------------------------------------------------------------------
// Tagged union for all serialisable object types
// ---------------------------------------------------------------------------

/// Discriminated union of all geopyv-dev result types that can be saved to a
/// `.pyv` file.  The variant tag is embedded in the bincode stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeopyvObject {
    Subset(SubsetSolution),
    Mesh(MeshSolution),
    Field(FieldSolution),
    Sequence(SequenceSolution),
    Particle(ParticleSolution),
    Speckle(Speckle),
    // Appended after the original five variants + Speckle — a new variant at
    // the end doesn't disturb the existing variants' bincode tag indices, so
    // no schema-version bump is needed for this addition (see module docs).
    Calibration(CalibrationSolution),
}

// ---------------------------------------------------------------------------
// Legacy shapes (schema v0 — everything before the 2026-07 solve-settings
// addition). Frozen: do not edit these to match future changes — freeze a
// new `Legacy*V1` set instead and chain it in `load`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySolveResult {
    p: Vec<f64>,
    c_zncc: f64,
    c_znssd: f64,
    iterations: usize,
    converged: bool,
    #[serde(default, alias = "solved")]
    quality_ok: bool,
    history: Vec<(usize, f64, f64, f64)>,
    #[serde(default)]
    max_norm: f64,
    #[serde(default)]
    tolerance: f64,
}

impl From<LegacySolveResult> for SolveResult {
    fn from(l: LegacySolveResult) -> Self {
        let subset_order = l.p.len() / 6;
        SolveResult {
            p: l.p,
            c_zncc: l.c_zncc,
            c_znssd: l.c_znssd,
            iterations: l.iterations,
            converged: l.converged,
            quality_ok: l.quality_ok,
            history: l.history,
            max_norm: l.max_norm,
            tolerance: l.tolerance,
            subset_order,
            max_iterations: 0,
            method: SolveMethod::Icgn,
            eta_omitted: None,
            texture: Texture::Deformable,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySubsetSolution {
    coord: [f64; 2],
    #[serde(alias = "template")]
    mask: MaskSummary,
    ref_image: PathBuf,
    target_image: PathBuf,
    result: LegacySolveResult,
    #[serde(default)]
    std_dev: f64,
    #[serde(default)]
    sssig: f64,
    #[serde(default)]
    delta_f: f64,
}

impl From<LegacySubsetSolution> for SubsetSolution {
    fn from(l: LegacySubsetSolution) -> Self {
        SubsetSolution {
            coord: l.coord,
            mask: l.mask,
            ref_image: l.ref_image,
            target_image: l.target_image,
            result: l.result.into(),
            std_dev: l.std_dev,
            sssig: l.sssig,
            delta_f: l.delta_f,
            time: None,
            seams: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMeshSolution {
    nodes: ndarray::Array2<f64>,
    elements: ndarray::Array2<usize>,
    boundary: Vec<usize>,
    exclusions: Vec<Vec<usize>>,
    centroids: ndarray::Array2<f64>,
    areas: ndarray::Array1<f64>,
    warps: ndarray::Array2<f64>,
    displacements: ndarray::Array2<f64>,
    c_zncc: ndarray::Array1<f64>,
    p: ndarray::Array2<f64>,
    seed_node: usize,
    mesh_order: u8,
    subset_order: u8,
    iterations: ndarray::Array1<u32>,
    norms: ndarray::Array1<f64>,
    f_img_path: PathBuf,
    g_img_path: PathBuf,
}

impl From<LegacyMeshSolution> for MeshSolution {
    fn from(l: LegacyMeshSolution) -> Self {
        MeshSolution {
            nodes: l.nodes,
            elements: l.elements,
            boundary: l.boundary,
            exclusions: l.exclusions,
            centroids: l.centroids,
            areas: l.areas,
            warps: l.warps,
            displacements: l.displacements,
            c_zncc: l.c_zncc,
            p: l.p,
            seed_node: l.seed_node,
            mesh_order: l.mesh_order,
            subset_order: l.subset_order,
            iterations: l.iterations,
            norms: l.norms,
            f_img_path: l.f_img_path,
            g_img_path: l.g_img_path,
            solve_config: None,
            seed: None,
            template_shape: None,
            template_sizes: None,
            time: None,
            seams: None,
            tracks: None,
        }
    }
}

fn legacy_default_mesh_order() -> u8 { 1 }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySequenceSolution {
    mesh_solutions: Vec<Arc<LegacyMeshSolution>>,
    mesh_paths: Vec<PathBuf>,
    #[serde(alias = "solved")]
    all_converged: bool,
    unsolvable: bool,
    override_log: Vec<usize>,
    reference_updates: Vec<bool>,
    #[serde(default = "legacy_default_mesh_order")]
    mesh_order: u8,
    #[serde(default)]
    first_f_img_path: Option<PathBuf>,
    #[serde(default = "crate::sequence::default_boundary_region")]
    boundary_region: Region,
    #[serde(default)]
    exclusion_regions: Vec<Region>,
}

impl From<LegacySequenceSolution> for SequenceSolution {
    fn from(l: LegacySequenceSolution) -> Self {
        SequenceSolution {
            mesh_solutions: l.mesh_solutions.into_iter()
                .map(|arc| Arc::new((*arc).clone().into()))
                .collect(),
            mesh_paths: l.mesh_paths,
            all_converged: l.all_converged,
            unsolvable: l.unsolvable,
            override_log: l.override_log,
            reference_updates: l.reference_updates,
            mesh_order: l.mesh_order,
            first_f_img_path: l.first_f_img_path,
            boundary_region: l.boundary_region,
            exclusion_regions: l.exclusion_regions,
            options: None,
            border: 0,
            time: None,
            time_source: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyParticleSolution {
    coordinates: ndarray::Array2<f64>,
    warps: ndarray::Array2<f64>,
    incs: ndarray::Array2<f64>,
    volumes: ndarray::Array1<f64>,
    strains: ndarray::Array2<f64>,
    strain_incs: ndarray::Array2<f64>,
    vol_strains: ndarray::Array1<f64>,
    reference_update_register: Vec<usize>,
    #[serde(default)]
    image_0_path: Option<PathBuf>,
    #[serde(default)]
    calibrated: bool,
}

impl From<LegacyParticleSolution> for ParticleSolution {
    fn from(l: LegacyParticleSolution) -> Self {
        let principal = principal_strains(&l.strains);
        ParticleSolution {
            coordinates: l.coordinates,
            warps: l.warps,
            incs: l.incs,
            volumes: l.volumes,
            strains: l.strains,
            strain_incs: l.strain_incs,
            vol_strains: l.vol_strains,
            reference_update_register: l.reference_update_register,
            image_0_path: l.image_0_path,
            calibrated: l.calibrated,
            config: None,
            principal_strains: Some(principal),
            gamma_max_grad: None,
            time: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyFieldSolution {
    particles: Vec<Arc<LegacyParticleSolution>>,
    initial_coordinates: ndarray::Array2<f64>,
    vol_totals: ndarray::Array1<f64>,
    reference_update_register: Vec<usize>,
    #[serde(default)]
    image_0_path: Option<PathBuf>,
    #[serde(default)]
    calibrated: bool,
}

impl From<LegacyFieldSolution> for FieldSolution {
    fn from(l: LegacyFieldSolution) -> Self {
        FieldSolution {
            particles: l.particles.into_iter()
                .map(|arc| Arc::new((*arc).clone().into()))
                .collect(),
            initial_coordinates: l.initial_coordinates,
            vol_totals: l.vol_totals,
            reference_update_register: l.reference_update_register,
            image_0_path: l.image_0_path,
            calibrated: l.calibrated,
            depth: 0.0,
            track: false,
            region: None,
            time: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy shapes (schema v1 — the released package's 0x03 format). Frozen:
// do not edit these to match future changes — freeze a new `Legacy*V2` set
// instead and chain it in `load`. Types not listed here (`SeedConfig`,
// `SubsetSolution`'s `MaskSummary`, `SequenceOptions`, `Region`,
// `CalibrationSolution`, ...) are unchanged since v1 and reused directly;
// enums that only gained trailing variants (`MaskShape`, `WarpMode`,
// `ShearBandOption`) decode unchanged too.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySolveResultV1 {
    p: Vec<f64>,
    c_zncc: f64,
    c_znssd: f64,
    iterations: usize,
    converged: bool,
    #[serde(default, alias = "solved")]
    quality_ok: bool,
    history: Vec<(usize, f64, f64, f64)>,
    #[serde(default)]
    max_norm: f64,
    #[serde(default)]
    tolerance: f64,
    #[serde(default)]
    subset_order: usize,
    #[serde(default)]
    max_iterations: usize,
    #[serde(default = "default_solve_method")]
    method: SolveMethod,
}

fn default_solve_method() -> SolveMethod {
    SolveMethod::Icgn
}

impl From<LegacySolveResultV1> for SolveResult {
    fn from(l: LegacySolveResultV1) -> Self {
        SolveResult {
            p: l.p,
            c_zncc: l.c_zncc,
            c_znssd: l.c_znssd,
            iterations: l.iterations,
            converged: l.converged,
            quality_ok: l.quality_ok,
            history: l.history,
            max_norm: l.max_norm,
            tolerance: l.tolerance,
            subset_order: l.subset_order,
            max_iterations: l.max_iterations,
            method: l.method,
            eta_omitted: None,
            texture: Texture::Deformable,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySubsetSolutionV1 {
    coord: [f64; 2],
    #[serde(alias = "template")]
    mask: MaskSummary,
    ref_image: PathBuf,
    target_image: PathBuf,
    result: LegacySolveResultV1,
    #[serde(default)]
    std_dev: f64,
    #[serde(default)]
    sssig: f64,
    #[serde(default)]
    delta_f: f64,
}

impl From<LegacySubsetSolutionV1> for SubsetSolution {
    fn from(l: LegacySubsetSolutionV1) -> Self {
        SubsetSolution {
            coord: l.coord,
            mask: l.mask,
            ref_image: l.ref_image,
            target_image: l.target_image,
            result: l.result.into(),
            std_dev: l.std_dev,
            sssig: l.sssig,
            delta_f: l.delta_f,
            time: None,
            seams: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySolveConfigV1 {
    max_norm: f64,
    max_iterations: usize,
    subset_order: usize,
    tolerance: f64,
    method: SolveMethod,
    #[serde(default)]
    override_active: bool,
}

impl From<LegacySolveConfigV1> for SolveConfig {
    fn from(l: LegacySolveConfigV1) -> Self {
        SolveConfig {
            max_norm: l.max_norm,
            max_iterations: l.max_iterations,
            subset_order: l.subset_order,
            tolerance: l.tolerance,
            method: l.method,
            override_active: l.override_active,
            ..SolveConfig::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMeshSolutionV1 {
    nodes: ndarray::Array2<f64>,
    elements: ndarray::Array2<usize>,
    boundary: Vec<usize>,
    exclusions: Vec<Vec<usize>>,
    centroids: ndarray::Array2<f64>,
    areas: ndarray::Array1<f64>,
    warps: ndarray::Array2<f64>,
    displacements: ndarray::Array2<f64>,
    c_zncc: ndarray::Array1<f64>,
    p: ndarray::Array2<f64>,
    seed_node: usize,
    mesh_order: u8,
    subset_order: u8,
    iterations: ndarray::Array1<u32>,
    norms: ndarray::Array1<f64>,
    f_img_path: PathBuf,
    g_img_path: PathBuf,
    #[serde(default)]
    solve_config: Option<LegacySolveConfigV1>,
    #[serde(default)]
    seed: Option<SeedConfig>,
}

impl From<LegacyMeshSolutionV1> for MeshSolution {
    fn from(l: LegacyMeshSolutionV1) -> Self {
        MeshSolution {
            nodes: l.nodes,
            elements: l.elements,
            boundary: l.boundary,
            exclusions: l.exclusions,
            centroids: l.centroids,
            areas: l.areas,
            warps: l.warps,
            displacements: l.displacements,
            c_zncc: l.c_zncc,
            p: l.p,
            seed_node: l.seed_node,
            mesh_order: l.mesh_order,
            subset_order: l.subset_order,
            iterations: l.iterations,
            norms: l.norms,
            f_img_path: l.f_img_path,
            g_img_path: l.g_img_path,
            solve_config: l.solve_config.map(Into::into),
            seed: l.seed,
            template_shape: None,
            template_sizes: None,
            time: None,
            seams: None,
            tracks: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySequenceSolutionV1 {
    mesh_solutions: Vec<Arc<LegacyMeshSolutionV1>>,
    mesh_paths: Vec<PathBuf>,
    #[serde(alias = "solved")]
    all_converged: bool,
    unsolvable: bool,
    override_log: Vec<usize>,
    reference_updates: Vec<bool>,
    #[serde(default = "legacy_default_mesh_order")]
    mesh_order: u8,
    #[serde(default)]
    first_f_img_path: Option<PathBuf>,
    #[serde(default = "crate::sequence::default_boundary_region")]
    boundary_region: Region,
    #[serde(default)]
    exclusion_regions: Vec<Region>,
    #[serde(default)]
    options: Option<SequenceOptions>,
    #[serde(default)]
    border: usize,
}

impl From<LegacySequenceSolutionV1> for SequenceSolution {
    fn from(l: LegacySequenceSolutionV1) -> Self {
        SequenceSolution {
            mesh_solutions: l.mesh_solutions.into_iter()
                .map(|arc| Arc::new((*arc).clone().into()))
                .collect(),
            mesh_paths: l.mesh_paths,
            all_converged: l.all_converged,
            unsolvable: l.unsolvable,
            override_log: l.override_log,
            reference_updates: l.reference_updates,
            mesh_order: l.mesh_order,
            first_f_img_path: l.first_f_img_path,
            boundary_region: l.boundary_region,
            exclusion_regions: l.exclusion_regions,
            options: l.options,
            border: l.border,
            time: None,
            time_source: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyParticleConfigV1 {
    factor: f64,
    true_incs: bool,
}

impl From<LegacyParticleConfigV1> for ParticleConfig {
    fn from(l: LegacyParticleConfigV1) -> Self {
        // v1 had only mesh-element interpolation -- record what was used.
        ParticleConfig { factor: l.factor, true_incs: l.true_incs, strain_method: StrainMethod::Mesh }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyParticleSolutionV1 {
    coordinates: ndarray::Array2<f64>,
    warps: ndarray::Array2<f64>,
    incs: ndarray::Array2<f64>,
    volumes: ndarray::Array1<f64>,
    strains: ndarray::Array2<f64>,
    strain_incs: ndarray::Array2<f64>,
    vol_strains: ndarray::Array1<f64>,
    reference_update_register: Vec<usize>,
    #[serde(default)]
    image_0_path: Option<PathBuf>,
    #[serde(default)]
    calibrated: bool,
    #[serde(default)]
    config: Option<LegacyParticleConfigV1>,
}

impl From<LegacyParticleSolutionV1> for ParticleSolution {
    fn from(l: LegacyParticleSolutionV1) -> Self {
        ParticleSolution {
            principal_strains: Some(principal_strains(&l.strains)),
            coordinates: l.coordinates,
            warps: l.warps,
            incs: l.incs,
            volumes: l.volumes,
            strains: l.strains,
            strain_incs: l.strain_incs,
            vol_strains: l.vol_strains,
            reference_update_register: l.reference_update_register,
            image_0_path: l.image_0_path,
            calibrated: l.calibrated,
            config: l.config.map(Into::into),
            gamma_max_grad: None,
            time: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyFieldSolutionV1 {
    particles: Vec<Arc<LegacyParticleSolutionV1>>,
    initial_coordinates: ndarray::Array2<f64>,
    vol_totals: ndarray::Array1<f64>,
    reference_update_register: Vec<usize>,
    #[serde(default)]
    image_0_path: Option<PathBuf>,
    #[serde(default)]
    calibrated: bool,
    #[serde(default)]
    depth: f64,
    #[serde(default)]
    track: bool,
}

impl From<LegacyFieldSolutionV1> for FieldSolution {
    fn from(l: LegacyFieldSolutionV1) -> Self {
        FieldSolution {
            particles: l.particles.into_iter()
                .map(|arc| Arc::new((*arc).clone().into()))
                .collect(),
            initial_coordinates: l.initial_coordinates,
            vol_totals: l.vol_totals,
            reference_update_register: l.reference_update_register,
            image_0_path: l.image_0_path,
            calibrated: l.calibrated,
            depth: l.depth,
            track: l.track,
            region: None,
            time: None,
        }
    }
}

/// `Speckle` as of v1 (and v0): no `speckle_limit`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV1 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
}

impl From<LegacySpeckleV1> for Speckle {
    fn from(l: LegacySpeckleV1) -> Self {
        Speckle {
            image_dir: l.image_dir,
            name: l.name,
            file_format: l.file_format,
            image_size: l.image_size,
            speckle_size: l.speckle_size,
            speckle_number: l.speckle_number,
            speckle_limit: false,
            progression: l.progression,
            comp: l.comp,
            origin: l.origin,
            mode: l.mode,
            noise_pos: l.noise_pos,
            noise_int: l.noise_int,
            image_no: l.image_no,
            scale: l.scale,
            speckle_positions: l.speckle_positions,
            solved: l.solved,
            grain_shape: crate::speckle::GrainShape::Round,
            grain_rotation: crate::speckle::GrainRotation::Material,
            grain_overlap: crate::speckle::GrainOverlap::Add,
            no_overlap: None,
            soil: None,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LegacyGeopyvObjectV1 {
    Subset(LegacySubsetSolutionV1),
    Mesh(LegacyMeshSolutionV1),
    Field(LegacyFieldSolutionV1),
    Sequence(LegacySequenceSolutionV1),
    Particle(LegacyParticleSolutionV1),
    Speckle(LegacySpeckleV1),
    Calibration(CalibrationSolution),
}

impl From<LegacyGeopyvObjectV1> for GeopyvObject {
    fn from(l: LegacyGeopyvObjectV1) -> Self {
        match l {
            LegacyGeopyvObjectV1::Subset(s) => GeopyvObject::Subset(s.into()),
            LegacyGeopyvObjectV1::Mesh(m) => GeopyvObject::Mesh(m.into()),
            LegacyGeopyvObjectV1::Field(f) => GeopyvObject::Field(f.into()),
            LegacyGeopyvObjectV1::Sequence(s) => GeopyvObject::Sequence(s.into()),
            LegacyGeopyvObjectV1::Particle(p) => GeopyvObject::Particle(p.into()),
            LegacyGeopyvObjectV1::Speckle(s) => GeopyvObject::Speckle(s.into()),
            LegacyGeopyvObjectV1::Calibration(c) => GeopyvObject::Calibration(c),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LegacyGeopyvObject {
    Subset(LegacySubsetSolution),
    Mesh(LegacyMeshSolution),
    Field(LegacyFieldSolution),
    Sequence(LegacySequenceSolution),
    Particle(LegacyParticleSolution),
    Speckle(LegacySpeckleV1),
}

impl From<LegacyGeopyvObject> for GeopyvObject {
    fn from(l: LegacyGeopyvObject) -> Self {
        match l {
            LegacyGeopyvObject::Subset(s) => GeopyvObject::Subset(s.into()),
            LegacyGeopyvObject::Mesh(m) => GeopyvObject::Mesh(m.into()),
            LegacyGeopyvObject::Field(f) => GeopyvObject::Field(f.into()),
            LegacyGeopyvObject::Sequence(s) => GeopyvObject::Sequence(s.into()),
            LegacyGeopyvObject::Particle(p) => GeopyvObject::Particle(p.into()),
            LegacyGeopyvObject::Speckle(s) => GeopyvObject::Speckle(s.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy shapes (schema v2 — the 0x07 format). Frozen: do not edit these to
// match future changes — freeze a new `Legacy*V3` set instead and chain it
// in `load`. Only `FieldSolution` changed in v3; every other variant reuses
// the v3 shape.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyFieldSolutionV2 {
    particles: Vec<Arc<LegacyParticleSolutionV3>>,
    initial_coordinates: ndarray::Array2<f64>,
    vol_totals: ndarray::Array1<f64>,
    reference_update_register: Vec<usize>,
    #[serde(default)]
    image_0_path: Option<PathBuf>,
    #[serde(default)]
    calibrated: bool,
    #[serde(default)]
    depth: f64,
    #[serde(default)]
    track: bool,
}

impl From<LegacyFieldSolutionV2> for FieldSolution {
    fn from(l: LegacyFieldSolutionV2) -> Self {
        FieldSolution {
            particles: l.particles.into_iter().map(arc_into).collect(),
            initial_coordinates: l.initial_coordinates,
            vol_totals: l.vol_totals,
            reference_update_register: l.reference_update_register,
            image_0_path: l.image_0_path,
            calibrated: l.calibrated,
            depth: l.depth,
            track: l.track,
            region: None,
            time: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LegacyGeopyvObjectV2 {
    Subset(LegacySubsetSolutionV3),
    Mesh(LegacyMeshSolutionV3),
    Field(LegacyFieldSolutionV2),
    Sequence(LegacySequenceSolutionV3),
    Particle(LegacyParticleSolutionV3),
    Speckle(LegacySpeckleV8),
    Calibration(CalibrationSolution),
}

impl From<LegacyGeopyvObjectV2> for GeopyvObject {
    fn from(l: LegacyGeopyvObjectV2) -> Self {
        match l {
            LegacyGeopyvObjectV2::Subset(s) => GeopyvObject::Subset(s.into()),
            LegacyGeopyvObjectV2::Mesh(m) => GeopyvObject::Mesh(m.into()),
            LegacyGeopyvObjectV2::Field(f) => GeopyvObject::Field(f.into()),
            LegacyGeopyvObjectV2::Sequence(s) => GeopyvObject::Sequence(s.into()),
            LegacyGeopyvObjectV2::Particle(p) => GeopyvObject::Particle(p.into()),
            LegacyGeopyvObjectV2::Speckle(s) => GeopyvObject::Speckle(s.into()),
            LegacyGeopyvObjectV2::Calibration(c) => GeopyvObject::Calibration(c),
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy shapes (schema v3 — the 0x08 format). Frozen: do not edit these to
// match future changes — freeze a new `Legacy*V4` set instead and chain it
// in `load`. v4 added `time` to every solution type (and `time_source` to
// `SequenceSolution`).
// ---------------------------------------------------------------------------

fn arc_into<L: Clone + Into<T>, T>(l: Arc<L>) -> Arc<T> {
    Arc::new(Arc::unwrap_or_clone(l).into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySubsetSolutionV3 {
    coord: [f64; 2],
    mask: MaskSummary,
    ref_image: PathBuf,
    target_image: PathBuf,
    result: LegacySolveResultV7,
    std_dev: f64,
    sssig: f64,
    delta_f: f64,
}

impl From<LegacySubsetSolutionV3> for SubsetSolution {
    fn from(l: LegacySubsetSolutionV3) -> Self {
        SubsetSolution {
            coord: l.coord,
            mask: l.mask,
            ref_image: l.ref_image,
            target_image: l.target_image,
            result: l.result.into(),
            std_dev: l.std_dev,
            sssig: l.sssig,
            delta_f: l.delta_f,
            time: None,
            seams: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMeshSolutionV3 {
    nodes: ndarray::Array2<f64>,
    elements: ndarray::Array2<usize>,
    boundary: Vec<usize>,
    exclusions: Vec<Vec<usize>>,
    centroids: ndarray::Array2<f64>,
    areas: ndarray::Array1<f64>,
    warps: ndarray::Array2<f64>,
    displacements: ndarray::Array2<f64>,
    c_zncc: ndarray::Array1<f64>,
    p: ndarray::Array2<f64>,
    seed_node: usize,
    mesh_order: u8,
    subset_order: u8,
    iterations: ndarray::Array1<u32>,
    norms: ndarray::Array1<f64>,
    f_img_path: PathBuf,
    g_img_path: PathBuf,
    solve_config: Option<LegacySolveConfigV5>,
    seed: Option<SeedConfig>,
    template_shape: Option<MaskShape>,
    template_sizes: Option<ndarray::Array1<u32>>,
    zonal_masking: Option<LegacyZonalMaskingRecord>,
}

impl From<LegacyMeshSolutionV3> for MeshSolution {
    fn from(l: LegacyMeshSolutionV3) -> Self {
        MeshSolution {
            nodes: l.nodes,
            elements: l.elements,
            boundary: l.boundary,
            exclusions: l.exclusions,
            centroids: l.centroids,
            areas: l.areas,
            warps: l.warps,
            displacements: l.displacements,
            c_zncc: l.c_zncc,
            p: l.p,
            seed_node: l.seed_node,
            mesh_order: l.mesh_order,
            subset_order: l.subset_order,
            iterations: l.iterations,
            norms: l.norms,
            f_img_path: l.f_img_path,
            g_img_path: l.g_img_path,
            solve_config: l.solve_config.map(Into::into),
            seed: l.seed,
            template_shape: l.template_shape,
            template_sizes: l.template_sizes,
            time: None,
            seams: None,
            tracks: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySequenceSolutionV3 {
    mesh_solutions: Vec<Arc<LegacyMeshSolutionV3>>,
    mesh_paths: Vec<PathBuf>,
    all_converged: bool,
    unsolvable: bool,
    override_log: Vec<usize>,
    reference_updates: Vec<bool>,
    mesh_order: u8,
    first_f_img_path: Option<PathBuf>,
    boundary_region: Region,
    exclusion_regions: Vec<Region>,
    options: Option<SequenceOptions>,
    border: usize,
}

impl From<LegacySequenceSolutionV3> for SequenceSolution {
    fn from(l: LegacySequenceSolutionV3) -> Self {
        SequenceSolution {
            mesh_solutions: l.mesh_solutions.into_iter().map(arc_into).collect(),
            mesh_paths: l.mesh_paths,
            all_converged: l.all_converged,
            unsolvable: l.unsolvable,
            override_log: l.override_log,
            reference_updates: l.reference_updates,
            mesh_order: l.mesh_order,
            first_f_img_path: l.first_f_img_path,
            boundary_region: l.boundary_region,
            exclusion_regions: l.exclusion_regions,
            options: l.options,
            border: l.border,
            time: None,
            time_source: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyParticleSolutionV3 {
    coordinates: ndarray::Array2<f64>,
    warps: ndarray::Array2<f64>,
    incs: ndarray::Array2<f64>,
    volumes: ndarray::Array1<f64>,
    strains: ndarray::Array2<f64>,
    strain_incs: ndarray::Array2<f64>,
    vol_strains: ndarray::Array1<f64>,
    reference_update_register: Vec<usize>,
    image_0_path: Option<PathBuf>,
    calibrated: bool,
    config: Option<ParticleConfig>,
    principal_strains: Option<ndarray::Array2<f64>>,
    gamma_max_grad: Option<ndarray::Array2<f64>>,
}

impl From<LegacyParticleSolutionV3> for ParticleSolution {
    fn from(l: LegacyParticleSolutionV3) -> Self {
        ParticleSolution {
            coordinates: l.coordinates,
            warps: l.warps,
            incs: l.incs,
            volumes: l.volumes,
            strains: l.strains,
            strain_incs: l.strain_incs,
            vol_strains: l.vol_strains,
            reference_update_register: l.reference_update_register,
            image_0_path: l.image_0_path,
            calibrated: l.calibrated,
            config: l.config,
            principal_strains: l.principal_strains,
            gamma_max_grad: l.gamma_max_grad,
            time: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyFieldSolutionV3 {
    particles: Vec<Arc<LegacyParticleSolutionV3>>,
    initial_coordinates: ndarray::Array2<f64>,
    vol_totals: ndarray::Array1<f64>,
    reference_update_register: Vec<usize>,
    image_0_path: Option<PathBuf>,
    calibrated: bool,
    depth: f64,
    track: bool,
    region: Option<FieldRegion>,
}

impl From<LegacyFieldSolutionV3> for FieldSolution {
    fn from(l: LegacyFieldSolutionV3) -> Self {
        FieldSolution {
            particles: l.particles.into_iter().map(arc_into).collect(),
            initial_coordinates: l.initial_coordinates,
            vol_totals: l.vol_totals,
            reference_update_register: l.reference_update_register,
            image_0_path: l.image_0_path,
            calibrated: l.calibrated,
            depth: l.depth,
            track: l.track,
            region: l.region,
            time: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LegacyGeopyvObjectV3 {
    Subset(LegacySubsetSolutionV3),
    Mesh(LegacyMeshSolutionV3),
    Field(LegacyFieldSolutionV3),
    Sequence(LegacySequenceSolutionV3),
    Particle(LegacyParticleSolutionV3),
    Speckle(LegacySpeckleV8),
    Calibration(CalibrationSolution),
}

impl From<LegacyGeopyvObjectV3> for GeopyvObject {
    fn from(l: LegacyGeopyvObjectV3) -> Self {
        match l {
            LegacyGeopyvObjectV3::Subset(s) => GeopyvObject::Subset(s.into()),
            LegacyGeopyvObjectV3::Mesh(m) => GeopyvObject::Mesh(m.into()),
            LegacyGeopyvObjectV3::Field(f) => GeopyvObject::Field(f.into()),
            LegacyGeopyvObjectV3::Sequence(s) => GeopyvObject::Sequence(s.into()),
            LegacyGeopyvObjectV3::Particle(p) => GeopyvObject::Particle(p.into()),
            LegacyGeopyvObjectV3::Speckle(s) => GeopyvObject::Speckle(s.into()),
            LegacyGeopyvObjectV3::Calibration(c) => GeopyvObject::Calibration(c),
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy shapes (schema v4 — the 0x09 format). Frozen: do not edit these to
// match future changes — freeze a new `Legacy*V5` set instead and chain it
// in `load`. v5 removed `MeshSolution::zonal_masking` (zonal masking was
// removed); the record is read and dropped. `LegacyZonalMaskingRecord` is
// that record's shape from v2 (0x07) to v4.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyZonalMaskingRecord {
    node_gamma_max: ndarray::Array1<f64>,
    node_gamma_max_grad: ndarray::Array1<f64>,
    node_boundary: Vec<bool>,
    node_zone: ndarray::Array1<u32>,
    node_pre_px: ndarray::Array1<u32>,
    node_post_px: ndarray::Array1<u32>,
    node_guard_fallback: Vec<bool>,
    k: f64,
    grad_median: f64,
    grad_mad: f64,
    grad_cutoff: f64,
    zone_image: ndarray::Array2<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMeshSolutionV4 {
    nodes: ndarray::Array2<f64>,
    elements: ndarray::Array2<usize>,
    boundary: Vec<usize>,
    exclusions: Vec<Vec<usize>>,
    centroids: ndarray::Array2<f64>,
    areas: ndarray::Array1<f64>,
    warps: ndarray::Array2<f64>,
    displacements: ndarray::Array2<f64>,
    c_zncc: ndarray::Array1<f64>,
    p: ndarray::Array2<f64>,
    seed_node: usize,
    mesh_order: u8,
    subset_order: u8,
    iterations: ndarray::Array1<u32>,
    norms: ndarray::Array1<f64>,
    f_img_path: PathBuf,
    g_img_path: PathBuf,
    solve_config: Option<LegacySolveConfigV5>,
    seed: Option<SeedConfig>,
    template_shape: Option<MaskShape>,
    template_sizes: Option<ndarray::Array1<u32>>,
    zonal_masking: Option<LegacyZonalMaskingRecord>,
    time: Option<(f64, f64)>,
}

impl From<LegacyMeshSolutionV4> for MeshSolution {
    fn from(l: LegacyMeshSolutionV4) -> Self {
        MeshSolution {
            nodes: l.nodes,
            elements: l.elements,
            boundary: l.boundary,
            exclusions: l.exclusions,
            centroids: l.centroids,
            areas: l.areas,
            warps: l.warps,
            displacements: l.displacements,
            c_zncc: l.c_zncc,
            p: l.p,
            seed_node: l.seed_node,
            mesh_order: l.mesh_order,
            subset_order: l.subset_order,
            iterations: l.iterations,
            norms: l.norms,
            f_img_path: l.f_img_path,
            g_img_path: l.g_img_path,
            solve_config: l.solve_config.map(Into::into),
            seed: l.seed,
            template_shape: l.template_shape,
            template_sizes: l.template_sizes,
            time: l.time,
            seams: None,
            tracks: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySequenceSolutionV4 {
    mesh_solutions: Vec<Arc<LegacyMeshSolutionV4>>,
    mesh_paths: Vec<PathBuf>,
    all_converged: bool,
    unsolvable: bool,
    override_log: Vec<usize>,
    reference_updates: Vec<bool>,
    mesh_order: u8,
    first_f_img_path: Option<PathBuf>,
    boundary_region: Region,
    exclusion_regions: Vec<Region>,
    options: Option<SequenceOptions>,
    border: usize,
    time: Option<Vec<f64>>,
    time_source: crate::timing::TimeSource,
}

impl From<LegacySequenceSolutionV4> for SequenceSolution {
    fn from(l: LegacySequenceSolutionV4) -> Self {
        SequenceSolution {
            mesh_solutions: l.mesh_solutions.into_iter().map(arc_into).collect(),
            mesh_paths: l.mesh_paths,
            all_converged: l.all_converged,
            unsolvable: l.unsolvable,
            override_log: l.override_log,
            reference_updates: l.reference_updates,
            mesh_order: l.mesh_order,
            first_f_img_path: l.first_f_img_path,
            boundary_region: l.boundary_region,
            exclusion_regions: l.exclusion_regions,
            options: l.options,
            border: l.border,
            time: l.time,
            time_source: l.time_source,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LegacyGeopyvObjectV4 {
    Subset(LegacySubsetSolutionV5),
    Mesh(LegacyMeshSolutionV4),
    Field(FieldSolution),
    Sequence(LegacySequenceSolutionV4),
    Particle(ParticleSolution),
    Speckle(LegacySpeckleV8),
    Calibration(CalibrationSolution),
}

impl From<LegacyGeopyvObjectV4> for GeopyvObject {
    fn from(l: LegacyGeopyvObjectV4) -> Self {
        match l {
            LegacyGeopyvObjectV4::Subset(s) => GeopyvObject::Subset(s.into()),
            LegacyGeopyvObjectV4::Mesh(m) => GeopyvObject::Mesh(m.into()),
            LegacyGeopyvObjectV4::Field(f) => GeopyvObject::Field(f),
            LegacyGeopyvObjectV4::Sequence(s) => GeopyvObject::Sequence(s.into()),
            LegacyGeopyvObjectV4::Particle(p) => GeopyvObject::Particle(p),
            LegacyGeopyvObjectV4::Speckle(s) => GeopyvObject::Speckle(s.into()),
            LegacyGeopyvObjectV4::Calibration(c) => GeopyvObject::Calibration(c),
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy shapes (schema v5 — the 0x0A format). Frozen: do not edit these to
// match future changes — freeze a new `Legacy*V6` set instead and chain it
// in `load`. v6 added `SubsetSolution::seams`, `MeshSolution::seams` and
// `SolveConfig::masking`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySubsetSolutionV5 {
    coord: [f64; 2],
    mask: MaskSummary,
    ref_image: PathBuf,
    target_image: PathBuf,
    result: LegacySolveResultV7,
    std_dev: f64,
    sssig: f64,
    delta_f: f64,
    time: Option<(f64, f64)>,
}

impl From<LegacySubsetSolutionV5> for SubsetSolution {
    fn from(l: LegacySubsetSolutionV5) -> Self {
        SubsetSolution {
            coord: l.coord,
            mask: l.mask,
            ref_image: l.ref_image,
            target_image: l.target_image,
            result: l.result.into(),
            std_dev: l.std_dev,
            sssig: l.sssig,
            delta_f: l.delta_f,
            time: l.time,
            seams: None,
        }
    }
}

/// `SolveConfig` before `masking` (v6). Also the shape stored by v3–v5 meshes.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySolveConfigV5 {
    max_norm: f64,
    max_iterations: usize,
    subset_order: usize,
    tolerance: f64,
    method: SolveMethod,
    override_active: bool,
    preconditioning: crate::mesh::Preconditioning,
    layer_rg: crate::mesh::LayerRgConfig,
}

#[cfg(test)]
impl LegacySolveConfigV5 {
    /// The v5 shape of a current config (tests writing old-format files).
    fn from_current(c: SolveConfig) -> Self {
        LegacySolveConfigV5 {
            max_norm: c.max_norm,
            max_iterations: c.max_iterations,
            subset_order: c.subset_order,
            tolerance: c.tolerance,
            method: c.method,
            override_active: c.override_active,
            preconditioning: c.preconditioning,
            layer_rg: c.layer_rg,
        }
    }
}

impl From<LegacySolveConfigV5> for SolveConfig {
    fn from(l: LegacySolveConfigV5) -> Self {
        SolveConfig {
            max_norm: l.max_norm,
            max_iterations: l.max_iterations,
            subset_order: l.subset_order,
            tolerance: l.tolerance,
            method: l.method,
            override_active: l.override_active,
            preconditioning: l.preconditioning,
            layer_rg: l.layer_rg,
            masking: crate::mesh::Masking::Uniform,
            texture: Texture::Deformable,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMeshSolutionV5 {
    nodes: ndarray::Array2<f64>,
    elements: ndarray::Array2<usize>,
    boundary: Vec<usize>,
    exclusions: Vec<Vec<usize>>,
    centroids: ndarray::Array2<f64>,
    areas: ndarray::Array1<f64>,
    warps: ndarray::Array2<f64>,
    displacements: ndarray::Array2<f64>,
    c_zncc: ndarray::Array1<f64>,
    p: ndarray::Array2<f64>,
    seed_node: usize,
    mesh_order: u8,
    subset_order: u8,
    iterations: ndarray::Array1<u32>,
    norms: ndarray::Array1<f64>,
    f_img_path: PathBuf,
    g_img_path: PathBuf,
    solve_config: Option<LegacySolveConfigV5>,
    seed: Option<SeedConfig>,
    template_shape: Option<MaskShape>,
    template_sizes: Option<ndarray::Array1<u32>>,
    time: Option<(f64, f64)>,
}

impl From<LegacyMeshSolutionV5> for MeshSolution {
    fn from(l: LegacyMeshSolutionV5) -> Self {
        MeshSolution {
            nodes: l.nodes,
            elements: l.elements,
            boundary: l.boundary,
            exclusions: l.exclusions,
            centroids: l.centroids,
            areas: l.areas,
            warps: l.warps,
            displacements: l.displacements,
            c_zncc: l.c_zncc,
            p: l.p,
            seed_node: l.seed_node,
            mesh_order: l.mesh_order,
            subset_order: l.subset_order,
            iterations: l.iterations,
            norms: l.norms,
            f_img_path: l.f_img_path,
            g_img_path: l.g_img_path,
            solve_config: l.solve_config.map(Into::into),
            seed: l.seed,
            template_shape: l.template_shape,
            template_sizes: l.template_sizes,
            time: l.time,
            seams: None,
            tracks: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySequenceSolutionV5 {
    mesh_solutions: Vec<Arc<LegacyMeshSolutionV5>>,
    mesh_paths: Vec<PathBuf>,
    all_converged: bool,
    unsolvable: bool,
    override_log: Vec<usize>,
    reference_updates: Vec<bool>,
    mesh_order: u8,
    first_f_img_path: Option<PathBuf>,
    boundary_region: Region,
    exclusion_regions: Vec<Region>,
    options: Option<SequenceOptions>,
    border: usize,
    time: Option<Vec<f64>>,
    time_source: crate::timing::TimeSource,
}

impl From<LegacySequenceSolutionV5> for SequenceSolution {
    fn from(l: LegacySequenceSolutionV5) -> Self {
        SequenceSolution {
            mesh_solutions: l.mesh_solutions.into_iter().map(arc_into).collect(),
            mesh_paths: l.mesh_paths,
            all_converged: l.all_converged,
            unsolvable: l.unsolvable,
            override_log: l.override_log,
            reference_updates: l.reference_updates,
            mesh_order: l.mesh_order,
            first_f_img_path: l.first_f_img_path,
            boundary_region: l.boundary_region,
            exclusion_regions: l.exclusion_regions,
            options: l.options,
            border: l.border,
            time: l.time,
            time_source: l.time_source,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LegacyGeopyvObjectV5 {
    Subset(LegacySubsetSolutionV5),
    Mesh(LegacyMeshSolutionV5),
    Field(FieldSolution),
    Sequence(LegacySequenceSolutionV5),
    Particle(ParticleSolution),
    Speckle(LegacySpeckleV8),
    Calibration(CalibrationSolution),
}

impl From<LegacyGeopyvObjectV5> for GeopyvObject {
    fn from(l: LegacyGeopyvObjectV5) -> Self {
        match l {
            LegacyGeopyvObjectV5::Subset(s) => GeopyvObject::Subset(s.into()),
            LegacyGeopyvObjectV5::Mesh(m) => GeopyvObject::Mesh(m.into()),
            LegacyGeopyvObjectV5::Field(f) => GeopyvObject::Field(f),
            LegacyGeopyvObjectV5::Sequence(s) => GeopyvObject::Sequence(s.into()),
            LegacyGeopyvObjectV5::Particle(p) => GeopyvObject::Particle(p),
            LegacyGeopyvObjectV5::Speckle(s) => GeopyvObject::Speckle(s.into()),
            LegacyGeopyvObjectV5::Calibration(c) => GeopyvObject::Calibration(c),
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy shapes (schema v6, 0x0B) — before `MeshSolution` gained `tracks`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMeshSolutionV6 {
    nodes: ndarray::Array2<f64>,
    elements: ndarray::Array2<usize>,
    boundary: Vec<usize>,
    exclusions: Vec<Vec<usize>>,
    centroids: ndarray::Array2<f64>,
    areas: ndarray::Array1<f64>,
    warps: ndarray::Array2<f64>,
    displacements: ndarray::Array2<f64>,
    c_zncc: ndarray::Array1<f64>,
    p: ndarray::Array2<f64>,
    seed_node: usize,
    mesh_order: u8,
    subset_order: u8,
    iterations: ndarray::Array1<u32>,
    norms: ndarray::Array1<f64>,
    f_img_path: PathBuf,
    g_img_path: PathBuf,
    solve_config: Option<LegacySolveConfigV7>,
    seed: Option<SeedConfig>,
    template_shape: Option<MaskShape>,
    template_sizes: Option<ndarray::Array1<u32>>,
    time: Option<(f64, f64)>,
    seams: Option<crate::mesh::MeshSeams>,
}

impl From<LegacyMeshSolutionV6> for MeshSolution {
    fn from(l: LegacyMeshSolutionV6) -> Self {
        MeshSolution {
            nodes: l.nodes,
            elements: l.elements,
            boundary: l.boundary,
            exclusions: l.exclusions,
            centroids: l.centroids,
            areas: l.areas,
            warps: l.warps,
            displacements: l.displacements,
            c_zncc: l.c_zncc,
            p: l.p,
            seed_node: l.seed_node,
            mesh_order: l.mesh_order,
            subset_order: l.subset_order,
            iterations: l.iterations,
            norms: l.norms,
            f_img_path: l.f_img_path,
            g_img_path: l.g_img_path,
            solve_config: l.solve_config.map(Into::into),
            seed: l.seed,
            template_shape: l.template_shape,
            template_sizes: l.template_sizes,
            time: l.time,
            seams: l.seams,
            tracks: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySequenceSolutionV6 {
    mesh_solutions: Vec<Arc<LegacyMeshSolutionV6>>,
    mesh_paths: Vec<PathBuf>,
    all_converged: bool,
    unsolvable: bool,
    override_log: Vec<usize>,
    reference_updates: Vec<bool>,
    mesh_order: u8,
    first_f_img_path: Option<PathBuf>,
    boundary_region: Region,
    exclusion_regions: Vec<Region>,
    options: Option<SequenceOptions>,
    border: usize,
    time: Option<Vec<f64>>,
    time_source: crate::timing::TimeSource,
}

impl From<LegacySequenceSolutionV6> for SequenceSolution {
    fn from(l: LegacySequenceSolutionV6) -> Self {
        SequenceSolution {
            mesh_solutions: l.mesh_solutions.into_iter().map(arc_into).collect(),
            mesh_paths: l.mesh_paths,
            all_converged: l.all_converged,
            unsolvable: l.unsolvable,
            override_log: l.override_log,
            reference_updates: l.reference_updates,
            mesh_order: l.mesh_order,
            first_f_img_path: l.first_f_img_path,
            boundary_region: l.boundary_region,
            exclusion_regions: l.exclusion_regions,
            options: l.options,
            border: l.border,
            time: l.time,
            time_source: l.time_source,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LegacyGeopyvObjectV6 {
    Subset(LegacySubsetSolutionV7),
    Mesh(LegacyMeshSolutionV6),
    Field(FieldSolution),
    Sequence(LegacySequenceSolutionV6),
    Particle(ParticleSolution),
    Speckle(LegacySpeckleV8),
    Calibration(CalibrationSolution),
}

impl From<LegacyGeopyvObjectV6> for GeopyvObject {
    fn from(l: LegacyGeopyvObjectV6) -> Self {
        match l {
            LegacyGeopyvObjectV6::Subset(s) => GeopyvObject::Subset(s.into()),
            LegacyGeopyvObjectV6::Mesh(m) => GeopyvObject::Mesh(m.into()),
            LegacyGeopyvObjectV6::Field(f) => GeopyvObject::Field(f),
            LegacyGeopyvObjectV6::Sequence(s) => GeopyvObject::Sequence(s.into()),
            LegacyGeopyvObjectV6::Particle(p) => GeopyvObject::Particle(p),
            LegacyGeopyvObjectV6::Speckle(s) => GeopyvObject::Speckle(s.into()),
            LegacyGeopyvObjectV6::Calibration(c) => GeopyvObject::Calibration(c),
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy shapes (schema v7, 0x0C) — before `SolveResult` and `SolveConfig`
// gained `texture`. Also the `SolveResult` shape of v3–v6 subsets and the
// `SolveConfig` shape of v6 meshes.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySolveResultV7 {
    p: Vec<f64>,
    c_zncc: f64,
    c_znssd: f64,
    iterations: usize,
    converged: bool,
    quality_ok: bool,
    history: Vec<(usize, f64, f64, f64)>,
    max_norm: f64,
    tolerance: f64,
    subset_order: usize,
    max_iterations: usize,
    method: SolveMethod,
    eta_omitted: Option<(f64, f64)>,
}

impl From<LegacySolveResultV7> for SolveResult {
    fn from(l: LegacySolveResultV7) -> Self {
        SolveResult {
            p: l.p,
            c_zncc: l.c_zncc,
            c_znssd: l.c_znssd,
            iterations: l.iterations,
            converged: l.converged,
            quality_ok: l.quality_ok,
            history: l.history,
            max_norm: l.max_norm,
            tolerance: l.tolerance,
            subset_order: l.subset_order,
            max_iterations: l.max_iterations,
            method: l.method,
            eta_omitted: l.eta_omitted,
            texture: Texture::Deformable,
        }
    }
}

#[cfg(test)]
impl LegacySolveResultV7 {
    /// The v7 shape of a current result (tests writing old-format files).
    fn from_current(r: SolveResult) -> Self {
        LegacySolveResultV7 {
            p: r.p,
            c_zncc: r.c_zncc,
            c_znssd: r.c_znssd,
            iterations: r.iterations,
            converged: r.converged,
            quality_ok: r.quality_ok,
            history: r.history,
            max_norm: r.max_norm,
            tolerance: r.tolerance,
            subset_order: r.subset_order,
            max_iterations: r.max_iterations,
            method: r.method,
            eta_omitted: r.eta_omitted,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySubsetSolutionV7 {
    coord: [f64; 2],
    mask: MaskSummary,
    ref_image: PathBuf,
    target_image: PathBuf,
    result: LegacySolveResultV7,
    std_dev: f64,
    sssig: f64,
    delta_f: f64,
    time: Option<(f64, f64)>,
    seams: Option<crate::seam::SeamRecord>,
}

impl From<LegacySubsetSolutionV7> for SubsetSolution {
    fn from(l: LegacySubsetSolutionV7) -> Self {
        SubsetSolution {
            coord: l.coord,
            mask: l.mask,
            ref_image: l.ref_image,
            target_image: l.target_image,
            result: l.result.into(),
            std_dev: l.std_dev,
            sssig: l.sssig,
            delta_f: l.delta_f,
            time: l.time,
            seams: l.seams,
        }
    }
}

#[cfg(test)]
impl LegacySubsetSolutionV7 {
    fn from_current(s: SubsetSolution) -> Self {
        LegacySubsetSolutionV7 {
            coord: s.coord,
            mask: s.mask,
            ref_image: s.ref_image,
            target_image: s.target_image,
            result: LegacySolveResultV7::from_current(s.result),
            std_dev: s.std_dev,
            sssig: s.sssig,
            delta_f: s.delta_f,
            time: s.time,
            seams: s.seams,
        }
    }
}

/// `SolveConfig` before `texture` (v8).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySolveConfigV7 {
    max_norm: f64,
    max_iterations: usize,
    subset_order: usize,
    tolerance: f64,
    method: SolveMethod,
    override_active: bool,
    preconditioning: crate::mesh::Preconditioning,
    layer_rg: crate::mesh::LayerRgConfig,
    masking: crate::mesh::Masking,
}

impl From<LegacySolveConfigV7> for SolveConfig {
    fn from(l: LegacySolveConfigV7) -> Self {
        SolveConfig {
            max_norm: l.max_norm,
            max_iterations: l.max_iterations,
            subset_order: l.subset_order,
            tolerance: l.tolerance,
            method: l.method,
            override_active: l.override_active,
            preconditioning: l.preconditioning,
            layer_rg: l.layer_rg,
            masking: l.masking,
            texture: Texture::Deformable,
        }
    }
}

#[cfg(test)]
impl LegacySolveConfigV7 {
    fn from_current(c: SolveConfig) -> Self {
        LegacySolveConfigV7 {
            max_norm: c.max_norm,
            max_iterations: c.max_iterations,
            subset_order: c.subset_order,
            tolerance: c.tolerance,
            method: c.method,
            override_active: c.override_active,
            preconditioning: c.preconditioning,
            layer_rg: c.layer_rg,
            masking: c.masking,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMeshSolutionV7 {
    nodes: ndarray::Array2<f64>,
    elements: ndarray::Array2<usize>,
    boundary: Vec<usize>,
    exclusions: Vec<Vec<usize>>,
    centroids: ndarray::Array2<f64>,
    areas: ndarray::Array1<f64>,
    warps: ndarray::Array2<f64>,
    displacements: ndarray::Array2<f64>,
    c_zncc: ndarray::Array1<f64>,
    p: ndarray::Array2<f64>,
    seed_node: usize,
    mesh_order: u8,
    subset_order: u8,
    iterations: ndarray::Array1<u32>,
    norms: ndarray::Array1<f64>,
    f_img_path: PathBuf,
    g_img_path: PathBuf,
    solve_config: Option<LegacySolveConfigV7>,
    seed: Option<SeedConfig>,
    template_shape: Option<MaskShape>,
    template_sizes: Option<ndarray::Array1<u32>>,
    time: Option<(f64, f64)>,
    seams: Option<crate::mesh::MeshSeams>,
    tracks: Option<crate::mesh::NodeTracks>,
}

impl From<LegacyMeshSolutionV7> for MeshSolution {
    fn from(l: LegacyMeshSolutionV7) -> Self {
        MeshSolution {
            nodes: l.nodes,
            elements: l.elements,
            boundary: l.boundary,
            exclusions: l.exclusions,
            centroids: l.centroids,
            areas: l.areas,
            warps: l.warps,
            displacements: l.displacements,
            c_zncc: l.c_zncc,
            p: l.p,
            seed_node: l.seed_node,
            mesh_order: l.mesh_order,
            subset_order: l.subset_order,
            iterations: l.iterations,
            norms: l.norms,
            f_img_path: l.f_img_path,
            g_img_path: l.g_img_path,
            solve_config: l.solve_config.map(Into::into),
            seed: l.seed,
            template_shape: l.template_shape,
            template_sizes: l.template_sizes,
            time: l.time,
            seams: l.seams,
            tracks: l.tracks,
        }
    }
}

#[cfg(test)]
impl LegacyMeshSolutionV7 {
    fn from_current(m: MeshSolution) -> Self {
        LegacyMeshSolutionV7 {
            nodes: m.nodes,
            elements: m.elements,
            boundary: m.boundary,
            exclusions: m.exclusions,
            centroids: m.centroids,
            areas: m.areas,
            warps: m.warps,
            displacements: m.displacements,
            c_zncc: m.c_zncc,
            p: m.p,
            seed_node: m.seed_node,
            mesh_order: m.mesh_order,
            subset_order: m.subset_order,
            iterations: m.iterations,
            norms: m.norms,
            f_img_path: m.f_img_path,
            g_img_path: m.g_img_path,
            solve_config: m.solve_config.map(LegacySolveConfigV7::from_current),
            seed: m.seed,
            template_shape: m.template_shape,
            template_sizes: m.template_sizes,
            time: m.time,
            seams: m.seams,
            tracks: m.tracks,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySequenceSolutionV7 {
    mesh_solutions: Vec<Arc<LegacyMeshSolutionV7>>,
    mesh_paths: Vec<PathBuf>,
    all_converged: bool,
    unsolvable: bool,
    override_log: Vec<usize>,
    reference_updates: Vec<bool>,
    mesh_order: u8,
    first_f_img_path: Option<PathBuf>,
    boundary_region: Region,
    exclusion_regions: Vec<Region>,
    options: Option<SequenceOptions>,
    border: usize,
    time: Option<Vec<f64>>,
    time_source: crate::timing::TimeSource,
}

impl From<LegacySequenceSolutionV7> for SequenceSolution {
    fn from(l: LegacySequenceSolutionV7) -> Self {
        SequenceSolution {
            mesh_solutions: l.mesh_solutions.into_iter().map(arc_into).collect(),
            mesh_paths: l.mesh_paths,
            all_converged: l.all_converged,
            unsolvable: l.unsolvable,
            override_log: l.override_log,
            reference_updates: l.reference_updates,
            mesh_order: l.mesh_order,
            first_f_img_path: l.first_f_img_path,
            boundary_region: l.boundary_region,
            exclusion_regions: l.exclusion_regions,
            options: l.options,
            border: l.border,
            time: l.time,
            time_source: l.time_source,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LegacyGeopyvObjectV7 {
    Subset(LegacySubsetSolutionV7),
    Mesh(LegacyMeshSolutionV7),
    Field(FieldSolution),
    Sequence(LegacySequenceSolutionV7),
    Particle(ParticleSolution),
    Speckle(LegacySpeckleV8),
    Calibration(CalibrationSolution),
}

impl From<LegacyGeopyvObjectV7> for GeopyvObject {
    fn from(l: LegacyGeopyvObjectV7) -> Self {
        match l {
            LegacyGeopyvObjectV7::Subset(s) => GeopyvObject::Subset(s.into()),
            LegacyGeopyvObjectV7::Mesh(m) => GeopyvObject::Mesh(m.into()),
            LegacyGeopyvObjectV7::Field(f) => GeopyvObject::Field(f),
            LegacyGeopyvObjectV7::Sequence(s) => GeopyvObject::Sequence(s.into()),
            LegacyGeopyvObjectV7::Particle(p) => GeopyvObject::Particle(p),
            LegacyGeopyvObjectV7::Speckle(s) => GeopyvObject::Speckle(s.into()),
            LegacyGeopyvObjectV7::Calibration(c) => GeopyvObject::Calibration(c),
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy shapes (schema v8, 0x0D) — before `Speckle` gained grain shapes.
// Also the `Speckle` shape of v2–v7 files.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV8 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    speckle_limit: bool,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
}

impl From<LegacySpeckleV8> for Speckle {
    fn from(l: LegacySpeckleV8) -> Self {
        Speckle {
            image_dir: l.image_dir,
            name: l.name,
            file_format: l.file_format,
            image_size: l.image_size,
            speckle_size: l.speckle_size,
            speckle_number: l.speckle_number,
            speckle_limit: l.speckle_limit,
            progression: l.progression,
            comp: l.comp,
            origin: l.origin,
            mode: l.mode,
            noise_pos: l.noise_pos,
            noise_int: l.noise_int,
            image_no: l.image_no,
            scale: l.scale,
            speckle_positions: l.speckle_positions,
            solved: l.solved,
            grain_shape: crate::speckle::GrainShape::Round,
            grain_rotation: crate::speckle::GrainRotation::Material,
            grain_overlap: crate::speckle::GrainOverlap::Add,
            no_overlap: None,
            soil: None,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LegacyGeopyvObjectV8 {
    Subset(SubsetSolution),
    Mesh(MeshSolution),
    Field(FieldSolution),
    Sequence(SequenceSolution),
    Particle(ParticleSolution),
    Speckle(LegacySpeckleV8),
    Calibration(CalibrationSolution),
}

impl From<LegacyGeopyvObjectV8> for GeopyvObject {
    fn from(l: LegacyGeopyvObjectV8) -> Self {
        match l {
            LegacyGeopyvObjectV8::Subset(s) => GeopyvObject::Subset(s),
            LegacyGeopyvObjectV8::Mesh(m) => GeopyvObject::Mesh(m),
            LegacyGeopyvObjectV8::Field(f) => GeopyvObject::Field(f),
            LegacyGeopyvObjectV8::Sequence(s) => GeopyvObject::Sequence(s),
            LegacyGeopyvObjectV8::Particle(p) => GeopyvObject::Particle(p),
            LegacyGeopyvObjectV8::Speckle(s) => GeopyvObject::Speckle(s.into()),
            LegacyGeopyvObjectV8::Calibration(c) => GeopyvObject::Calibration(c),
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy shapes (schema v9, 0x0E, and v10, 0x0F) — `Speckle` before
// `grain_overlap` (v9) and before `no_overlap` (v10).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV10 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    speckle_limit: bool,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
    grain_shape: crate::speckle::GrainShape,
    grain_rotation: crate::speckle::GrainRotation,
    grain_overlap: crate::speckle::GrainOverlap,
}

impl From<LegacySpeckleV10> for Speckle {
    fn from(l: LegacySpeckleV10) -> Self {
        Speckle {
            image_dir: l.image_dir,
            name: l.name,
            file_format: l.file_format,
            image_size: l.image_size,
            speckle_size: l.speckle_size,
            speckle_number: l.speckle_number,
            speckle_limit: l.speckle_limit,
            progression: l.progression,
            comp: l.comp,
            origin: l.origin,
            mode: l.mode,
            noise_pos: l.noise_pos,
            noise_int: l.noise_int,
            image_no: l.image_no,
            scale: l.scale,
            speckle_positions: l.speckle_positions,
            solved: l.solved,
            grain_shape: l.grain_shape,
            grain_rotation: l.grain_rotation,
            grain_overlap: l.grain_overlap,
            no_overlap: None,
            soil: None,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV9 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    speckle_limit: bool,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
    grain_shape: crate::speckle::GrainShape,
    grain_rotation: crate::speckle::GrainRotation,
}

impl From<LegacySpeckleV9> for Speckle {
    fn from(l: LegacySpeckleV9) -> Self {
        LegacySpeckleV10 {
            image_dir: l.image_dir, name: l.name, file_format: l.file_format, image_size: l.image_size,
            speckle_size: l.speckle_size, speckle_number: l.speckle_number, speckle_limit: l.speckle_limit,
            progression: l.progression, comp: l.comp, origin: l.origin, mode: l.mode,
            noise_pos: l.noise_pos, noise_int: l.noise_int, image_no: l.image_no, scale: l.scale,
            speckle_positions: l.speckle_positions, solved: l.solved,
            grain_shape: l.grain_shape, grain_rotation: l.grain_rotation,
            grain_overlap: crate::speckle::GrainOverlap::Add,
        }
        .into()
    }
}

macro_rules! legacy_object_with_speckle {
    ($name:ident, $speckle:ty) => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        enum $name {
            Subset(SubsetSolution),
            Mesh(MeshSolution),
            Field(FieldSolution),
            Sequence(SequenceSolution),
            Particle(ParticleSolution),
            Speckle($speckle),
            Calibration(CalibrationSolution),
        }

        impl From<$name> for GeopyvObject {
            fn from(l: $name) -> Self {
                match l {
                    $name::Subset(s) => GeopyvObject::Subset(s),
                    $name::Mesh(m) => GeopyvObject::Mesh(m),
                    $name::Field(f) => GeopyvObject::Field(f),
                    $name::Sequence(s) => GeopyvObject::Sequence(s),
                    $name::Particle(p) => GeopyvObject::Particle(p),
                    $name::Speckle(s) => GeopyvObject::Speckle(s.into()),
                    $name::Calibration(c) => GeopyvObject::Calibration(c),
                }
            }
        }
    };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV11 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    speckle_limit: bool,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
    grain_shape: crate::speckle::GrainShape,
    grain_rotation: crate::speckle::GrainRotation,
    grain_overlap: crate::speckle::GrainOverlap,
    no_overlap: Option<f64>,
}

impl From<LegacySpeckleV11> for Speckle {
    fn from(l: LegacySpeckleV11) -> Self {
        Speckle {
            image_dir: l.image_dir,
            name: l.name,
            file_format: l.file_format,
            image_size: l.image_size,
            speckle_size: l.speckle_size,
            speckle_number: l.speckle_number,
            speckle_limit: l.speckle_limit,
            progression: l.progression,
            comp: l.comp,
            origin: l.origin,
            mode: l.mode,
            noise_pos: l.noise_pos,
            noise_int: l.noise_int,
            image_no: l.image_no,
            scale: l.scale,
            speckle_positions: l.speckle_positions,
            solved: l.solved,
            grain_shape: l.grain_shape,
            grain_rotation: l.grain_rotation,
            grain_overlap: l.grain_overlap,
            no_overlap: l.no_overlap,
            soil: None,
            seed: None,
        }
    }
}

/// `SoilConfig` as of v12 (0x11).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySoilConfigV12 {
    kind: crate::soil::SoilKind,
    matrix_brightness: f64,
    grain_brightness: f64,
    grain_spread: f64,
    grain_size: f64,
    grain_size_spread: f64,
    grain: LegacySoilGrain,
    coverage: f64,
    depth_layers: usize,
    depth_dimming: f64,
    pattern_strength: f64,
    pattern_size: f64,
    blur: f64,
}

impl From<LegacySoilConfigV12> for LegacySoilConfigV15 {
    /// Clay converts exactly. Sand had no dyed grains: its grains become the
    /// matrix (no seeding); the brightness behind all layers is no longer
    /// stored separately and follows from the matrix brightness. Depth
    /// dimming is dropped (removed in v14).
    fn from(l: LegacySoilConfigV12) -> Self {
        let (matrix_brightness, matrix_spread, asr) = match l.kind {
            crate::soil::SoilKind::Clay => (l.matrix_brightness, l.pattern_strength, l.coverage),
            crate::soil::SoilKind::Sand => (l.grain_brightness, l.grain_spread, 0.0),
        };
        LegacySoilConfigV15 {
            kind: l.kind,
            matrix_brightness,
            matrix_spread,
            seed_brightness: l.grain_brightness,
            seed_spread: l.grain_spread,
            asr,
            grain_size: l.grain_size,
            grain_size_spread: l.grain_size_spread,
            grain: l.grain,
            depth_layers: l.depth_layers,
            pattern_size: l.pattern_size,
            blur: l.blur,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV12 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    speckle_limit: bool,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
    grain_shape: crate::speckle::GrainShape,
    grain_rotation: crate::speckle::GrainRotation,
    grain_overlap: crate::speckle::GrainOverlap,
    no_overlap: Option<f64>,
    soil: Option<LegacySoilConfigV12>,
}

impl From<LegacySpeckleV12> for Speckle {
    fn from(l: LegacySpeckleV12) -> Self {
        Speckle {
            image_dir: l.image_dir,
            name: l.name,
            file_format: l.file_format,
            image_size: l.image_size,
            speckle_size: l.speckle_size,
            speckle_number: l.speckle_number,
            speckle_limit: l.speckle_limit,
            progression: l.progression,
            comp: l.comp,
            origin: l.origin,
            mode: l.mode,
            noise_pos: l.noise_pos,
            noise_int: l.noise_int,
            image_no: l.image_no,
            scale: l.scale,
            speckle_positions: l.speckle_positions,
            solved: l.solved,
            grain_shape: l.grain_shape,
            grain_rotation: l.grain_rotation,
            grain_overlap: l.grain_overlap,
            no_overlap: l.no_overlap,
            soil: l.soil.map(|c| LegacySoilConfigV15::from(c).into()),
            seed: None,
        }
    }
}

/// `SoilConfig` as of v13 (0x12).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySoilConfigV13 {
    kind: crate::soil::SoilKind,
    matrix_brightness: f64,
    matrix_spread: f64,
    seed_brightness: f64,
    seed_spread: f64,
    asr: f64,
    grain_size: f64,
    grain_size_spread: f64,
    grain: LegacySoilGrain,
    depth_layers: usize,
    depth_dimming: f64,
    pattern_size: f64,
    blur: f64,
}

impl From<LegacySoilConfigV13> for LegacySoilConfigV15 {
    /// Depth dimming is dropped (removed in v14); everything else is kept.
    fn from(l: LegacySoilConfigV13) -> Self {
        LegacySoilConfigV15 {
            kind: l.kind,
            matrix_brightness: l.matrix_brightness,
            matrix_spread: l.matrix_spread,
            seed_brightness: l.seed_brightness,
            seed_spread: l.seed_spread,
            asr: l.asr,
            grain_size: l.grain_size,
            grain_size_spread: l.grain_size_spread,
            grain: l.grain,
            depth_layers: l.depth_layers,
            pattern_size: l.pattern_size,
            blur: l.blur,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV13 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    speckle_limit: bool,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
    grain_shape: crate::speckle::GrainShape,
    grain_rotation: crate::speckle::GrainRotation,
    grain_overlap: crate::speckle::GrainOverlap,
    no_overlap: Option<f64>,
    soil: Option<LegacySoilConfigV13>,
}

impl From<LegacySpeckleV13> for Speckle {
    fn from(l: LegacySpeckleV13) -> Self {
        Speckle {
            image_dir: l.image_dir,
            name: l.name,
            file_format: l.file_format,
            image_size: l.image_size,
            speckle_size: l.speckle_size,
            speckle_number: l.speckle_number,
            speckle_limit: l.speckle_limit,
            progression: l.progression,
            comp: l.comp,
            origin: l.origin,
            mode: l.mode,
            noise_pos: l.noise_pos,
            noise_int: l.noise_int,
            image_no: l.image_no,
            scale: l.scale,
            speckle_positions: l.speckle_positions,
            solved: l.solved,
            grain_shape: l.grain_shape,
            grain_rotation: l.grain_rotation,
            grain_overlap: l.grain_overlap,
            no_overlap: l.no_overlap,
            soil: l.soil.map(|c| LegacySoilConfigV15::from(c).into()),
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV14 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    speckle_limit: bool,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
    grain_shape: crate::speckle::GrainShape,
    grain_rotation: crate::speckle::GrainRotation,
    grain_overlap: crate::speckle::GrainOverlap,
    no_overlap: Option<f64>,
    soil: Option<LegacySoilConfigV15>,
}

impl From<LegacySpeckleV14> for Speckle {
    fn from(l: LegacySpeckleV14) -> Self {
        Speckle {
            image_dir: l.image_dir,
            name: l.name,
            file_format: l.file_format,
            image_size: l.image_size,
            speckle_size: l.speckle_size,
            speckle_number: l.speckle_number,
            speckle_limit: l.speckle_limit,
            progression: l.progression,
            comp: l.comp,
            origin: l.origin,
            mode: l.mode,
            noise_pos: l.noise_pos,
            noise_int: l.noise_int,
            image_no: l.image_no,
            scale: l.scale,
            speckle_positions: l.speckle_positions,
            solved: l.solved,
            grain_shape: l.grain_shape,
            grain_rotation: l.grain_rotation,
            grain_overlap: l.grain_overlap,
            no_overlap: l.no_overlap,
            soil: l.soil.map(Into::into),
            seed: None,
        }
    }
}

/// Grain outline of `SoilConfig` up to v15 (0x14).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum LegacySoilGrain {
    Sphere,
    Elongated { aspect: f64 },
    Cluster { roundness: f64 },
}

/// `SoilConfig` as of v14-v15 (0x13-0x14).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySoilConfigV15 {
    kind: crate::soil::SoilKind,
    matrix_brightness: f64,
    matrix_spread: f64,
    seed_brightness: f64,
    seed_spread: f64,
    asr: f64,
    grain_size: f64,
    grain_size_spread: f64,
    grain: LegacySoilGrain,
    depth_layers: usize,
    pattern_size: f64,
    blur: f64,
}

impl From<LegacySoilConfigV15> for crate::soil::SoilConfig {
    /// Grain size and shape become a characterisation. The log-normal sizes
    /// (by count: median `grain_size`, 90th percentile `grain_size_spread` x
    /// median) become a size curve in px whose median by area is `d_p`.
    /// Spheres and clusters: aspect ratio 1; ellipses of long/short `aspect`:
    /// aspect ratio 1 / aspect; default outline statistics (the images were
    /// drawn with older outlines). Brightness, asr, layers, pattern and blur
    /// are kept.
    fn from(l: LegacySoilConfigV15) -> Self {
        let sigma = l.grain_size_spread.ln() / 1.2816;
        let x50 = l.grain_size * (2.0 * sigma * sigma).exp(); // area-weighted median
        let psd = if sigma > 0.0 {
            (-6..=6)
                .map(|k| {
                    let z = 0.5 * k as f64;
                    [x50 * (sigma * z).exp(), 50.0 * libm::erfc(-z / std::f64::consts::SQRT_2)]
                })
                .collect()
        } else {
            Vec::new()
        };
        let aspect_ratio = match l.grain {
            LegacySoilGrain::Sphere | LegacySoilGrain::Cluster { .. } => 1.0,
            LegacySoilGrain::Elongated { aspect } => 1.0 / aspect,
        };
        crate::soil::SoilConfig {
            kind: l.kind,
            matrix_brightness: l.matrix_brightness,
            matrix_spread: l.matrix_spread,
            seed_brightness: l.seed_brightness,
            seed_spread: l.seed_spread,
            asr: l.asr,
            psd,
            d_p: x50,
            aspect_ratio,
            aspect_ratio_sd: 0.0,
            outlines: None,
            depth_layers: l.depth_layers,
            pattern_size: l.pattern_size,
            blur: l.blur,
            // drawn before brightness inside grains existed: flat grains
            grain_tilt: 0.0,
            grain_dome: 0.0,
            grain_rim: 0.0,
            grain_texture: 0.0,
            texture_size: 0.10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV15 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    speckle_limit: bool,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
    grain_shape: crate::speckle::GrainShape,
    grain_rotation: crate::speckle::GrainRotation,
    grain_overlap: crate::speckle::GrainOverlap,
    no_overlap: Option<f64>,
    soil: Option<LegacySoilConfigV15>,
    seed: Option<u64>,
}

impl From<LegacySpeckleV15> for Speckle {
    fn from(l: LegacySpeckleV15) -> Self {
        Speckle {
            image_dir: l.image_dir,
            name: l.name,
            file_format: l.file_format,
            image_size: l.image_size,
            speckle_size: l.speckle_size,
            speckle_number: l.speckle_number,
            speckle_limit: l.speckle_limit,
            progression: l.progression,
            comp: l.comp,
            origin: l.origin,
            mode: l.mode,
            noise_pos: l.noise_pos,
            noise_int: l.noise_int,
            image_no: l.image_no,
            scale: l.scale,
            speckle_positions: l.speckle_positions,
            solved: l.solved,
            grain_shape: l.grain_shape,
            grain_rotation: l.grain_rotation,
            grain_overlap: l.grain_overlap,
            no_overlap: l.no_overlap,
            soil: l.soil.map(Into::into),
            seed: l.seed,
        }
    }
}

/// `SoilConfig` as of v16 (0x15).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySoilConfigV16 {
    kind: crate::soil::SoilKind,
    matrix_brightness: f64,
    matrix_spread: f64,
    seed_brightness: f64,
    seed_spread: f64,
    asr: f64,
    psd: Vec<[f64; 2]>,
    d_p: f64,
    aspect_ratio: f64,
    aspect_ratio_sd: f64,
    sphericity: f64,
    sphericity_sd: f64,
    depth_layers: usize,
    pattern_size: f64,
    blur: f64,
}

impl From<LegacySoilConfigV16> for crate::soil::SoilConfig {
    /// Sphericity is dropped (no longer an input); default outline statistics
    /// (the images were drawn with older outlines).
    fn from(l: LegacySoilConfigV16) -> Self {
        crate::soil::SoilConfig {
            kind: l.kind,
            matrix_brightness: l.matrix_brightness,
            matrix_spread: l.matrix_spread,
            seed_brightness: l.seed_brightness,
            seed_spread: l.seed_spread,
            asr: l.asr,
            psd: l.psd,
            d_p: l.d_p,
            aspect_ratio: l.aspect_ratio,
            aspect_ratio_sd: l.aspect_ratio_sd,
            outlines: None,
            depth_layers: l.depth_layers,
            pattern_size: l.pattern_size,
            blur: l.blur,
            // drawn before brightness inside grains existed: flat grains
            grain_tilt: 0.0,
            grain_dome: 0.0,
            grain_rim: 0.0,
            grain_texture: 0.0,
            texture_size: 0.10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV16 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    speckle_limit: bool,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
    grain_shape: crate::speckle::GrainShape,
    grain_rotation: crate::speckle::GrainRotation,
    grain_overlap: crate::speckle::GrainOverlap,
    no_overlap: Option<f64>,
    soil: Option<LegacySoilConfigV16>,
    seed: Option<u64>,
}

impl From<LegacySpeckleV16> for Speckle {
    fn from(l: LegacySpeckleV16) -> Self {
        Speckle {
            image_dir: l.image_dir,
            name: l.name,
            file_format: l.file_format,
            image_size: l.image_size,
            speckle_size: l.speckle_size,
            speckle_number: l.speckle_number,
            speckle_limit: l.speckle_limit,
            progression: l.progression,
            comp: l.comp,
            origin: l.origin,
            mode: l.mode,
            noise_pos: l.noise_pos,
            noise_int: l.noise_int,
            image_no: l.image_no,
            scale: l.scale,
            speckle_positions: l.speckle_positions,
            solved: l.solved,
            grain_shape: l.grain_shape,
            grain_rotation: l.grain_rotation,
            grain_overlap: l.grain_overlap,
            no_overlap: l.no_overlap,
            soil: l.soil.map(Into::into),
            seed: l.seed,
        }
    }
}

/// `SoilConfig` as of v17 (0x16).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySoilConfigV17 {
    kind: crate::soil::SoilKind,
    matrix_brightness: f64,
    matrix_spread: f64,
    seed_brightness: f64,
    seed_spread: f64,
    asr: f64,
    psd: Vec<[f64; 2]>,
    d_p: f64,
    aspect_ratio: f64,
    aspect_ratio_sd: f64,
    outlines: Option<crate::soil::OutlineStats>,
    depth_layers: usize,
    pattern_size: f64,
    blur: f64,
}

impl From<LegacySoilConfigV17> for crate::soil::SoilConfig {
    /// No brightness variation inside grains (the images were drawn without).
    fn from(l: LegacySoilConfigV17) -> Self {
        crate::soil::SoilConfig {
            kind: l.kind,
            matrix_brightness: l.matrix_brightness,
            matrix_spread: l.matrix_spread,
            seed_brightness: l.seed_brightness,
            seed_spread: l.seed_spread,
            asr: l.asr,
            psd: l.psd,
            d_p: l.d_p,
            aspect_ratio: l.aspect_ratio,
            aspect_ratio_sd: l.aspect_ratio_sd,
            outlines: l.outlines,
            depth_layers: l.depth_layers,
            pattern_size: l.pattern_size,
            blur: l.blur,
            // drawn before brightness inside grains existed: flat grains
            grain_tilt: 0.0,
            grain_dome: 0.0,
            grain_rim: 0.0,
            grain_texture: 0.0,
            texture_size: 0.10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySpeckleV17 {
    image_dir: String,
    name: String,
    file_format: String,
    image_size: (usize, usize),
    speckle_size: f64,
    speckle_number: usize,
    speckle_limit: bool,
    progression: Progression,
    comp: [f64; 12],
    origin: [f64; 2],
    mode: WarpMode,
    noise_pos: f64,
    noise_int: f64,
    image_no: usize,
    scale: ScaleType,
    speckle_positions: Option<ndarray::Array2<f64>>,
    solved: bool,
    grain_shape: crate::speckle::GrainShape,
    grain_rotation: crate::speckle::GrainRotation,
    grain_overlap: crate::speckle::GrainOverlap,
    no_overlap: Option<f64>,
    soil: Option<LegacySoilConfigV17>,
    seed: Option<u64>,
}

impl From<LegacySpeckleV17> for Speckle {
    fn from(l: LegacySpeckleV17) -> Self {
        Speckle {
            image_dir: l.image_dir,
            name: l.name,
            file_format: l.file_format,
            image_size: l.image_size,
            speckle_size: l.speckle_size,
            speckle_number: l.speckle_number,
            speckle_limit: l.speckle_limit,
            progression: l.progression,
            comp: l.comp,
            origin: l.origin,
            mode: l.mode,
            noise_pos: l.noise_pos,
            noise_int: l.noise_int,
            image_no: l.image_no,
            scale: l.scale,
            speckle_positions: l.speckle_positions,
            solved: l.solved,
            grain_shape: l.grain_shape,
            grain_rotation: l.grain_rotation,
            grain_overlap: l.grain_overlap,
            no_overlap: l.no_overlap,
            soil: l.soil.map(Into::into),
            seed: l.seed,
        }
    }
}

legacy_object_with_speckle!(LegacyGeopyvObjectV9, LegacySpeckleV9);
legacy_object_with_speckle!(LegacyGeopyvObjectV10, LegacySpeckleV10);
legacy_object_with_speckle!(LegacyGeopyvObjectV11, LegacySpeckleV11);
legacy_object_with_speckle!(LegacyGeopyvObjectV12, LegacySpeckleV12);
legacy_object_with_speckle!(LegacyGeopyvObjectV13, LegacySpeckleV13);
legacy_object_with_speckle!(LegacyGeopyvObjectV14, LegacySpeckleV14);
legacy_object_with_speckle!(LegacyGeopyvObjectV15, LegacySpeckleV15);
legacy_object_with_speckle!(LegacyGeopyvObjectV16, LegacySpeckleV16);
legacy_object_with_speckle!(LegacyGeopyvObjectV17, LegacySpeckleV17);

// ---------------------------------------------------------------------------
// save / load
// ---------------------------------------------------------------------------

/// Serialise a [`GeopyvObject`] to a `.pyv` file at `path`.
///
/// The file is created (or truncated if it already exists).
/// Format: 4-byte magic `b"GPYV"` + 1-byte version [`CURRENT_VERSION`] +
/// zstd-compressed bincode payload (current schema).
pub fn save<P: AsRef<Path>>(path: P, object: &GeopyvObject) -> Result<(), Error> {
    let encoded =
        bincode::serde::encode_to_vec(object, bincode::config::standard())
            .map_err(|e| Error::Io(e.to_string()))?;
    let compressed = zstd::encode_all(encoded.as_slice(), 3)
        .map_err(|e| Error::Io(e.to_string()))?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&[CURRENT_VERSION])?;
    file.write_all(&compressed)?;
    Ok(())
}

/// Load a [`GeopyvObject`] from a `.pyv` file at `path`.
///
/// Returns [`Error::InvalidMagic`] if the file does not start with `b"GPYV"`.
/// Returns [`Error::UnsupportedVersion`] for any unrecognised version byte.
///
/// Files with version `0x01`/`0x02` (schema v0), `0x03` (schema v1, the
/// released package's format), `0x07` (schema v2), `0x08` (schema v3), `0x09` (schema v4), `0x0A` (schema v5) or `0x0B` (schema v6) are decoded via the frozen `Legacy*` /
/// `Legacy*V1` types and migrated into the current shapes — see "Schema
/// versioning" in the module docs.
pub fn load<P: AsRef<Path>>(path: P) -> Result<GeopyvObject, Error> {
    let mut file = std::fs::File::open(path.as_ref()).map_err(|e| {
        Error::FileNotFound(format!("{}: {}", path.as_ref().display(), e))
    })?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::InvalidMagic);
    }

    let mut ver_buf = [0u8; 1];
    file.read_exact(&mut ver_buf)?;

    let mut raw = Vec::new();
    file.read_to_end(&mut raw)?;

    let data = match ver_buf[0] {
        VERSION_UNCOMPRESSED => raw,
        VERSION_ZSTD_V0 | VERSION_ZSTD_V1 | VERSION_ZSTD_V2 | VERSION_ZSTD_V3 | VERSION_ZSTD_V4
        | VERSION_ZSTD_V5 | VERSION_ZSTD_V6 | VERSION_ZSTD_V7 | VERSION_ZSTD_V8
        | VERSION_ZSTD_V9 | VERSION_ZSTD_V10 | VERSION_ZSTD_V11 | VERSION_ZSTD_V12
        | VERSION_ZSTD_V13 | VERSION_ZSTD_V14 | VERSION_ZSTD_V15 | VERSION_ZSTD_V16 | VERSION_ZSTD_V17 | CURRENT_VERSION => {
            zstd::decode_all(raw.as_slice()).map_err(|e| Error::Io(e.to_string()))?
        }
        v => return Err(Error::UnsupportedVersion(v)),
    };

    match ver_buf[0] {
        CURRENT_VERSION => {
            let (object, _) =
                bincode::serde::decode_from_slice::<GeopyvObject, _>(
                    &data,
                    bincode::config::standard(),
                )
                .map_err(|e| Error::Io(e.to_string()))?;
            Ok(object)
        }
        VERSION_ZSTD_V17 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV17, _>(&data, bincode::config::standard())
                    .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V16 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV16, _>(&data, bincode::config::standard())
                    .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V15 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV15, _>(&data, bincode::config::standard())
                    .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V14 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV14, _>(&data, bincode::config::standard())
                    .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V13 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV13, _>(&data, bincode::config::standard())
                    .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V12 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV12, _>(&data, bincode::config::standard())
                    .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V11 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV11, _>(&data, bincode::config::standard())
                    .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V10 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV10, _>(&data, bincode::config::standard())
                    .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V9 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV9, _>(&data, bincode::config::standard())
                    .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V8 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV8, _>(
                    &data,
                    bincode::config::standard(),
                )
                .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V7 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV7, _>(
                    &data,
                    bincode::config::standard(),
                )
                .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V6 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV6, _>(
                    &data,
                    bincode::config::standard(),
                )
                .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V5 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV5, _>(
                    &data,
                    bincode::config::standard(),
                )
                .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V4 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV4, _>(
                    &data,
                    bincode::config::standard(),
                )
                .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V3 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV3, _>(
                    &data,
                    bincode::config::standard(),
                )
                .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V2 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV2, _>(
                    &data,
                    bincode::config::standard(),
                )
                .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        VERSION_ZSTD_V1 => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObjectV1, _>(
                    &data,
                    bincode::config::standard(),
                )
                .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
        _ => {
            let (legacy, _) =
                bincode::serde::decode_from_slice::<LegacyGeopyvObject, _>(
                    &data,
                    bincode::config::standard(),
                )
                .map_err(|e| Error::Io(e.to_string()))?;
            Ok(legacy.into())
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests (Phase 2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use ndarray::{array, Array1, Array2};

    use std::sync::Arc;
    use crate::{
        mesh::{MeshSolution, NodeResult},
        sequence::SequenceSolution,
    };

    /// Build a minimal MeshSolution with known values for round-trip tests.
    fn make_mesh_solution() -> MeshSolution {
        let nodes = array![[0.0_f64, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let elements = array![[0usize, 1, 2]];
        let centroids = crate::mesh::compute_centroids(&nodes, &elements);
        MeshSolution {
            nodes,
            elements,
            boundary: vec![0, 1, 2],
            exclusions: vec![],
            centroids,
            areas: array![0.5_f64],
            warps: array![[0.1_f64, 0.2, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            displacements: array![[0.1_f64, 0.2], [0.3, 0.4], [0.5, 0.6]],
            c_zncc: array![0.999_f64, 0.998, 0.997],
            p: array![[0.1_f64, 0.2, 0.0, 0.0, 0.0, 0.0],
                      [0.3, 0.4, 0.0, 0.0, 0.0, 0.0],
                      [0.5, 0.6, 0.0, 0.0, 0.0, 0.0]],
            seed_node: 0,
            mesh_order: 1,
            subset_order: 1,
            iterations: ndarray::array![0u32, 0, 0],
            norms: ndarray::array![0.0_f64, 0.0, 0.0],
            f_img_path: PathBuf::new(),
            g_img_path: PathBuf::new(),
            solve_config: Some(crate::mesh::SolveConfig {
                max_norm: 1e-5,
                max_iterations: 50,
                subset_order: 1,
                tolerance: 0.75,
                method: crate::mesh::SolveMethod::Icgn,
                override_active: false,
                ..Default::default()
            }),
            seed: Some(crate::mesh::SeedConfig {
                coord: [0.5, 0.5],
                warp: vec![0.0; 6],
                tolerance: 0.9,
            }),
            template_shape: Some(crate::masks::MaskShape::Circle),
            template_sizes: Some(ndarray::array![8u32, 8, 8]),
            time: None,
            seams: None,
            tracks: None,
        }
    }

    /// Round-trip MeshSolution through save/load.
    #[test]
    fn test_mesh_round_trip() {
        let original = make_mesh_solution();
        let tmp = std::env::temp_dir().join("geopyv_test_mesh.pyv");
        save(&tmp, &GeopyvObject::Mesh(original.clone())).unwrap();

        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Mesh(m) => {
                assert_eq!(m.nodes, original.nodes);
                assert_eq!(m.elements, original.elements);
                assert_eq!(m.boundary, original.boundary);
                assert!((m.areas[0] - original.areas[0]).abs() < 1e-15);
                assert_eq!(m.seed_node, original.seed_node);
                assert_eq!(m.mesh_order, original.mesh_order);
                let cfg = m.solve_config.as_ref().expect("solve_config should round-trip");
                let orig_cfg = original.solve_config.as_ref().unwrap();
                assert_eq!(cfg.max_iterations, orig_cfg.max_iterations);
                assert_eq!(cfg.method, orig_cfg.method);
                let seed = m.seed.as_ref().expect("seed should round-trip");
                assert_eq!(seed.coord, original.seed.as_ref().unwrap().coord);
                assert_eq!(m.template_shape, original.template_shape);
                assert_eq!(m.template_sizes, original.template_sizes);
            }
            _ => panic!("expected Mesh variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v4 (0x09) mesh file carrying a zonal-masking record loads,
    /// with the record dropped and everything else (incl. `time`) kept.
    #[test]
    fn test_schema_v4_zonal_mesh_file_loads_without_record() {
        let m = make_mesh_solution();
        let mut zone_image = ndarray::Array2::<u8>::ones((6, 8));
        zone_image.slice_mut(ndarray::s![.., 4..]).fill(2);
        let legacy = LegacyGeopyvObjectV4::Mesh(LegacyMeshSolutionV4 {
            nodes: m.nodes.clone(),
            elements: m.elements.clone(),
            boundary: m.boundary.clone(),
            exclusions: m.exclusions.clone(),
            centroids: m.centroids.clone(),
            areas: m.areas.clone(),
            warps: m.warps.clone(),
            displacements: m.displacements.clone(),
            c_zncc: m.c_zncc.clone(),
            p: m.p.clone(),
            seed_node: m.seed_node,
            mesh_order: m.mesh_order,
            subset_order: m.subset_order,
            iterations: m.iterations.clone(),
            norms: m.norms.clone(),
            f_img_path: m.f_img_path.clone(),
            g_img_path: m.g_img_path.clone(),
            solve_config: m.solve_config.clone().map(LegacySolveConfigV5::from_current),
            seed: m.seed.clone(),
            template_shape: m.template_shape.clone(),
            template_sizes: m.template_sizes.clone(),
            zonal_masking: Some(LegacyZonalMaskingRecord {
                node_gamma_max: array![0.01_f64, 0.4, 0.02],
                node_gamma_max_grad: array![0.002_f64, 0.05, 0.003],
                node_boundary: vec![false, true, false],
                node_zone: array![1_u32, 2, 1],
                node_pre_px: array![120_u32, 120, 120],
                node_post_px: array![118_u32, 61, 119],
                node_guard_fallback: vec![false, false, true],
                k: 2.0,
                grad_median: 0.003,
                grad_mad: 0.001,
                grad_cutoff: 0.003 + 2.0 * 1.4826 * 0.001,
                zone_image,
            }),
            time: Some((0.0, 0.5)),
        });
        let tmp = write_versioned("geopyv_test_v4_zonal_mesh.pyv", VERSION_ZSTD_V4, &legacy);
        match load(&tmp).unwrap() {
            GeopyvObject::Mesh(l) => {
                assert_eq!(l.displacements, m.displacements);
                assert_eq!(l.template_sizes, m.template_sizes);
                assert_eq!(l.time, Some((0.0, 0.5)));
            }
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A released-format (schema v1, `0x03`) `Mesh` file loads, with the new
    /// `MeshSolution`/`SolveConfig` fields defaulted.
    #[test]
    fn test_schema_v1_mesh_file_loads_via_legacy_path() {
        let nodes = array![[0.0_f64, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let elements = array![[0usize, 1, 2]];
        let centroids = crate::mesh::compute_centroids(&nodes, &elements);
        let legacy = LegacyGeopyvObjectV1::Mesh(LegacyMeshSolutionV1 {
            nodes,
            elements,
            boundary: vec![0, 1, 2],
            exclusions: vec![],
            centroids,
            areas: array![0.5_f64],
            warps: ndarray::Array2::zeros((1, 12)),
            displacements: array![[0.0_f64, 0.0], [0.0, 0.0], [0.0, 0.0]],
            c_zncc: array![0.9_f64, 0.9, 0.9],
            p: ndarray::Array2::zeros((3, 6)),
            seed_node: 0,
            mesh_order: 1,
            subset_order: 1,
            iterations: ndarray::array![0u32, 0, 0],
            norms: ndarray::array![0.0_f64, 0.0, 0.0],
            f_img_path: PathBuf::new(),
            g_img_path: PathBuf::new(),
            solve_config: Some(LegacySolveConfigV1 {
                max_norm: 1e-5,
                max_iterations: 50,
                subset_order: 1,
                tolerance: 0.75,
                method: SolveMethod::Fagn,
                override_active: true,
            }),
            seed: Some(crate::mesh::SeedConfig {
                coord: [0.5, 0.5],
                warp: vec![0.0; 6],
                tolerance: 0.9,
            }),
        });
        let encoded =
            bincode::serde::encode_to_vec(&legacy, bincode::config::standard()).unwrap();
        let compressed = zstd::encode_all(encoded.as_slice(), 3).unwrap();
        let tmp = std::env::temp_dir().join("geopyv_test_schema_v1_mesh.pyv");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(MAGIC).unwrap();
            f.write_all(&[VERSION_ZSTD_V1]).unwrap();
            f.write_all(&compressed).unwrap();
        }
        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Mesh(m) => {
                assert_eq!(m.boundary, vec![0, 1, 2]);
                assert!(m.template_shape.is_none());
                assert!(m.template_sizes.is_none());
                assert_eq!(m.seed.as_ref().unwrap().coord, [0.5, 0.5]);
                let cfg = m.solve_config.as_ref().unwrap();
                assert!(matches!(cfg.method, SolveMethod::Fagn));
                assert!(cfg.override_active);
                assert_eq!(cfg.preconditioning, crate::mesh::Preconditioning::Rg);
            }
            _ => panic!("expected Mesh variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// Round-trip SequenceSolution through save/load.
    #[test]
    fn test_sequence_round_trip() {
        let sol = SequenceSolution {
            mesh_solutions: vec![Arc::new(make_mesh_solution()), Arc::new(make_mesh_solution())],
            mesh_paths: vec![],
            all_converged: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: vec![false, false],
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: crate::sequence::default_boundary_region(),
            exclusion_regions: vec![],
            options: Some(crate::sequence::SequenceOptions {
                guide: true,
                sequential: false,
                sync: true,
                override_: false,
            }),
            border: 20,
            time: None,
            time_source: Default::default(),
        };
        let tmp = std::env::temp_dir().join("geopyv_test_seq.pyv");
        save(&tmp, &GeopyvObject::Sequence(sol.clone())).unwrap();

        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Sequence(s) => {
                assert_eq!(s.mesh_solutions.len(), 2);
                assert!(s.all_converged);
                assert!(!s.unsolvable);
                let opts = s.options.as_ref().expect("options should round-trip");
                assert!(opts.guide);
                assert!(opts.sync);
                assert_eq!(s.border, 20);
            }
            _ => panic!("expected Sequence variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// f64 values round-trip with bit-exact precision (Tier A).
    #[test]
    fn test_f64_precision() {
        let val = std::f64::consts::PI;
        let mut sol = make_mesh_solution();
        sol.displacements[[0, 0]] = val;
        let tmp = std::env::temp_dir().join("geopyv_test_f64.pyv");
        save(&tmp, &GeopyvObject::Mesh(sol)).unwrap();
        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Mesh(m) => {
                assert_eq!(m.displacements[[0, 0]], val);
            }
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// Invalid magic bytes return `Error::InvalidMagic`.
    #[test]
    fn test_invalid_magic() {
        let tmp = std::env::temp_dir().join("geopyv_test_bad_magic.pyv");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(b"XXXX\x01\x00\x00\x00").unwrap();
        }
        let result = load(&tmp);
        assert!(matches!(result, Err(Error::InvalidMagic)));
        let _ = std::fs::remove_file(tmp);
    }

    /// Wrong version byte returns `Error::UnsupportedVersion`.
    #[test]
    fn test_unsupported_version() {
        let tmp = std::env::temp_dir().join("geopyv_test_bad_ver.pyv");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(b"GPYV\x7f\x00\x00\x00").unwrap();
        }
        let result = load(&tmp);
        assert!(matches!(result, Err(Error::UnsupportedVersion(0x7f))));
        let _ = std::fs::remove_file(tmp);
    }

    /// Bytes 0x04-0x06 were only ever written by unreleased development
    /// builds -- rejected outright, never misread as the current schema.
    #[test]
    fn test_interim_dev_versions_rejected() {
        for v in [0x04u8, 0x05, 0x06] {
            let tmp = std::env::temp_dir().join(format!("geopyv_test_interim_{v}.pyv"));
            {
                let mut f = std::fs::File::create(&tmp).unwrap();
                f.write_all(MAGIC).unwrap();
                f.write_all(&[v, 0, 0, 0]).unwrap();
            }
            assert!(matches!(load(&tmp), Err(Error::UnsupportedVersion(x)) if x == v));
            let _ = std::fs::remove_file(tmp);
        }
    }

    fn make_particle_solution() -> ParticleSolution {
        ParticleSolution {
            coordinates: array![[1.0_f64, 2.0], [1.5, 2.5]],
            warps: Array2::zeros((2, 6)),
            incs: Array2::zeros((2, 6)),
            volumes: array![1.0_f64, 1.0],
            strains: Array2::zeros((2, 6)),
            strain_incs: Array2::zeros((1, 6)),
            vol_strains: array![0.0_f64, 0.0],
            reference_update_register: vec![],
            image_0_path: None,
            calibrated: false,
            config: None,
            principal_strains: Some(Array2::zeros((2, 4))),
            gamma_max_grad: None,
            time: None,
        }
    }

    /// FieldSolution (with its region) round-trips through the current schema.
    #[test]
    fn test_field_region_round_trip() {
        let region = crate::field::FieldRegion {
            boundary: array![[0.0_f64, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            exclusions: vec![array![[1.0_f64, 1.0], [2.0, 1.0], [2.0, 2.0]]],
        };
        let sol = FieldSolution {
            particles: vec![Arc::new(make_particle_solution())],
            initial_coordinates: array![[1.0_f64, 2.0]],
            vol_totals: array![1.0_f64, 1.0],
            reference_update_register: vec![],
            image_0_path: None,
            calibrated: false,
            depth: 1.0,
            track: true,
            region: Some(region.clone()),
            time: None,
        };
        let tmp = std::env::temp_dir().join("geopyv_test_field_region.pyv");
        save(&tmp, &GeopyvObject::Field(sol)).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Field(f) => {
                let r = f.region.expect("region survives round trip");
                assert_eq!(r.boundary, region.boundary);
                assert_eq!(r.exclusions, region.exclusions);
            }
            _ => panic!("expected Field"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v2 (0x07) field file loads via the legacy path with no region.
    #[test]
    fn test_schema_v2_field_file_loads_via_legacy_path() {
        let legacy = LegacyGeopyvObjectV2::Field(LegacyFieldSolutionV2 {
            particles: vec![Arc::new(v3_particle(make_particle_solution()))],
            initial_coordinates: array![[1.0_f64, 2.0]],
            vol_totals: array![1.0_f64, 1.0],
            reference_update_register: vec![1],
            image_0_path: None,
            calibrated: false,
            depth: 2.0,
            track: true,
        });
        let encoded = bincode::serde::encode_to_vec(&legacy, bincode::config::standard()).unwrap();
        let compressed = zstd::encode_all(encoded.as_slice(), 3).unwrap();
        let tmp = std::env::temp_dir().join("geopyv_test_v2_field.pyv");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(MAGIC).unwrap();
            f.write_all(&[VERSION_ZSTD_V2]).unwrap();
            f.write_all(&compressed).unwrap();
        }
        match load(&tmp).unwrap() {
            GeopyvObject::Field(f) => {
                assert!(f.region.is_none());
                assert_eq!(f.depth, 2.0);
                assert_eq!(f.reference_update_register, vec![1]);
                assert_eq!(f.particles[0].coordinates, array![[1.0_f64, 2.0], [1.5, 2.5]]);
            }
            _ => panic!("expected Field"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// The schema-v3 shape of a current `ParticleSolution` (no `time`).
    fn v3_particle(p: ParticleSolution) -> LegacyParticleSolutionV3 {
        LegacyParticleSolutionV3 {
            coordinates: p.coordinates,
            warps: p.warps,
            incs: p.incs,
            volumes: p.volumes,
            strains: p.strains,
            strain_incs: p.strain_incs,
            vol_strains: p.vol_strains,
            reference_update_register: p.reference_update_register,
            image_0_path: p.image_0_path,
            calibrated: p.calibrated,
            config: p.config,
            principal_strains: p.principal_strains,
            gamma_max_grad: p.gamma_max_grad,
        }
    }

    fn write_versioned<T: Serialize>(name: &str, version: u8, obj: &T) -> PathBuf {
        let encoded = bincode::serde::encode_to_vec(obj, bincode::config::standard()).unwrap();
        let compressed = zstd::encode_all(encoded.as_slice(), 3).unwrap();
        let tmp = std::env::temp_dir().join(name);
        let mut f = std::fs::File::create(&tmp).unwrap();
        f.write_all(MAGIC).unwrap();
        f.write_all(&[version]).unwrap();
        f.write_all(&compressed).unwrap();
        tmp
    }

    /// Schema v3 (0x08) field and mesh files load via the legacy path with
    /// no `time`.
    #[test]
    fn test_schema_v3_files_load_without_time() {
        let legacy = LegacyGeopyvObjectV3::Field(LegacyFieldSolutionV3 {
            particles: vec![Arc::new(v3_particle(make_particle_solution()))],
            initial_coordinates: array![[1.0_f64, 2.0]],
            vol_totals: array![1.0_f64, 1.0],
            reference_update_register: vec![],
            image_0_path: None,
            calibrated: false,
            depth: 3.0,
            track: true,
            region: None,
        });
        let tmp = write_versioned("geopyv_test_v3_field.pyv", VERSION_ZSTD_V3, &legacy);
        match load(&tmp).unwrap() {
            GeopyvObject::Field(f) => {
                assert!(f.time.is_none());
                assert!(f.particles[0].time.is_none());
                assert_eq!(f.depth, 3.0);
            }
            _ => panic!("expected Field"),
        }
        let _ = std::fs::remove_file(tmp);

        let m = make_mesh_solution();
        let legacy = LegacyGeopyvObjectV3::Mesh(LegacyMeshSolutionV3 {
            nodes: m.nodes.clone(),
            elements: m.elements.clone(),
            boundary: m.boundary.clone(),
            exclusions: m.exclusions.clone(),
            centroids: m.centroids.clone(),
            areas: m.areas.clone(),
            warps: m.warps.clone(),
            displacements: m.displacements.clone(),
            c_zncc: m.c_zncc.clone(),
            p: m.p.clone(),
            seed_node: m.seed_node,
            mesh_order: m.mesh_order,
            subset_order: m.subset_order,
            iterations: m.iterations.clone(),
            norms: m.norms.clone(),
            f_img_path: m.f_img_path.clone(),
            g_img_path: m.g_img_path.clone(),
            solve_config: m.solve_config.clone().map(LegacySolveConfigV5::from_current),
            seed: m.seed.clone(),
            template_shape: None,
            template_sizes: None,
            zonal_masking: None,
        });
        let tmp = write_versioned("geopyv_test_v3_mesh.pyv", VERSION_ZSTD_V3, &legacy);
        match load(&tmp).unwrap() {
            GeopyvObject::Mesh(l) => {
                assert!(l.time.is_none());
                assert_eq!(l.displacements, m.displacements);
            }
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// `time` survives a current-format round trip on every solution type.
    #[test]
    fn test_time_round_trip() {
        let mut m = make_mesh_solution();
        m.time = Some((0.5, 0.75));
        let tmp = std::env::temp_dir().join("geopyv_test_time_mesh.pyv");
        save(&tmp, &GeopyvObject::Mesh(m.clone())).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Mesh(l) => assert_eq!(l.time, Some((0.5, 0.75))),
            _ => panic!("expected Mesh"),
        }
        let mut seq = SequenceSolution::from_mesh_solution(Arc::new(m));
        seq.time_source = crate::timing::TimeSource::Filename("date-time".into());
        save(&tmp, &GeopyvObject::Sequence(seq)).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Sequence(s) => {
                assert_eq!(s.time, Some(vec![0.5, 0.75]));
                assert_eq!(s.time_source, crate::timing::TimeSource::Filename("date-time".into()));
                assert_eq!(s.mesh_solutions[0].time, Some((0.5, 0.75)));
            }
            _ => panic!("expected Sequence"),
        }
        let mut p = make_particle_solution();
        p.time = Some(vec![0.0, 0.1]);
        save(&tmp, &GeopyvObject::Particle(p)).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Particle(l) => assert_eq!(l.time, Some(vec![0.0, 0.1])),
            _ => panic!("expected Particle"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A version 0x01 (legacy uncompressed, schema v0) file is still readable
    /// — routed through the legacy decode path, so `solve_config`/`seed` are
    /// dropped (not present in schema v0) but the rest of the shape survives.
    #[test]
    fn test_v1_backwards_compat() {
        let sol = make_mesh_solution();
        let encoded = bincode::serde::encode_to_vec(
            &GeopyvObject::Mesh(sol.clone()),
            bincode::config::standard(),
        )
        .unwrap();
        let tmp = std::env::temp_dir().join("geopyv_test_v1_compat.pyv");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(b"GPYV\x01").unwrap();
            f.write_all(&encoded).unwrap();
        }
        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Mesh(m) => {
                assert_eq!(m.nodes, sol.nodes);
            }
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// Loading a nonexistent file returns `Error::FileNotFound`.
    #[test]
    fn test_file_not_found() {
        let result = load("/nonexistent/path/file.pyv");
        assert!(matches!(result, Err(Error::FileNotFound(_))));
    }

    /// A genuine schema-v0 file (version byte `0x02`, pre-solve-settings shape,
    /// written by hand-constructing the frozen `Legacy*` types directly) loads
    /// through the real [`load`] entry point, with `solve_config`/`seed`
    /// defaulting to `None` via the `Legacy* -> *` conversions.
    #[test]
    fn test_schema_v0_mesh_file_loads_via_legacy_path() {
        let nodes = array![[0.0_f64, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let elements = array![[0usize, 1, 2]];
        let centroids = crate::mesh::compute_centroids(&nodes, &elements);
        let legacy = LegacyGeopyvObject::Mesh(LegacyMeshSolution {
            nodes,
            elements,
            boundary: vec![0, 1, 2],
            exclusions: vec![],
            centroids,
            areas: array![0.5_f64],
            warps: ndarray::Array2::zeros((1, 12)),
            displacements: array![[0.0_f64, 0.0], [0.0, 0.0], [0.0, 0.0]],
            c_zncc: array![0.9_f64, 0.9, 0.9],
            p: ndarray::Array2::zeros((3, 6)),
            seed_node: 0,
            mesh_order: 1,
            subset_order: 1,
            iterations: ndarray::array![0u32, 0, 0],
            norms: ndarray::array![0.0_f64, 0.0, 0.0],
            f_img_path: PathBuf::new(),
            g_img_path: PathBuf::new(),
        });
        let encoded =
            bincode::serde::encode_to_vec(&legacy, bincode::config::standard()).unwrap();
        let compressed = zstd::encode_all(encoded.as_slice(), 3).unwrap();
        let tmp = std::env::temp_dir().join("geopyv_test_schema_v0_mesh.pyv");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(MAGIC).unwrap();
            f.write_all(&[VERSION_ZSTD_V0]).unwrap();
            f.write_all(&compressed).unwrap();
        }

        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Mesh(m) => {
                assert!(m.solve_config.is_none());
                assert!(m.seed.is_none());
                assert_eq!(m.boundary, vec![0, 1, 2]);
                assert_eq!(m.seed_node, 0);
            }
            _ => panic!("expected Mesh variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A released-format (schema v1, `0x03`) `Subset` file loads through the
    /// real [`load`] entry point, with `eta_omitted` defaulting to `None`.
    #[test]
    fn test_schema_v1_subset_file_loads_via_legacy_path() {
        let legacy = LegacyGeopyvObjectV1::Subset(LegacySubsetSolutionV1 {
            coord: [10.0, 20.0],
            mask: MaskSummary {
                shape: crate::masks::MaskShape::Circle,
                size: 25,
                n_px: 1961,
            },
            ref_image: PathBuf::from("ref.jpg"),
            target_image: PathBuf::from("tar.jpg"),
            result: LegacySolveResultV1 {
                p: vec![0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
                c_zncc: 0.999,
                c_znssd: 0.002,
                iterations: 4,
                converged: true,
                quality_ok: true,
                history: vec![(1, 0.5, 0.9, 0.2), (2, 1e-6, 0.999, 0.002)],
                max_norm: 1e-5,
                tolerance: 0.75,
                subset_order: 1,
                max_iterations: 50,
                method: crate::mesh::SolveMethod::Icgn,
            },
            std_dev: 71.5,
            sssig: 430193.5,
            delta_f: 3166.75,
        });
        let encoded =
            bincode::serde::encode_to_vec(&legacy, bincode::config::standard()).unwrap();
        let compressed = zstd::encode_all(encoded.as_slice(), 3).unwrap();
        let tmp = std::env::temp_dir().join("geopyv_test_schema_v1_subset.pyv");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(MAGIC).unwrap();
            f.write_all(&[VERSION_ZSTD_V1]).unwrap();
            f.write_all(&compressed).unwrap();
        }

        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Subset(s) => {
                assert_eq!(s.coord, [10.0, 20.0]);
                assert_eq!(s.result.iterations, 4);
                assert_eq!(s.result.c_zncc, 0.999);
                assert!(s.result.eta_omitted.is_none());
            }
            _ => panic!("expected Subset variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// Round-trip through the current (`CURRENT_VERSION`) format: a `Subset`
    /// with `eta_omitted = Some(...)` survives `save`/`load` unchanged.
    #[test]
    fn test_subset_eta_omitted_round_trips() {
        let sol = SubsetSolution {
            coord: [5.0, 5.0],
            mask: MaskSummary { shape: crate::masks::MaskShape::Circle, size: 25, n_px: 1961 },
            ref_image: PathBuf::from("ref.jpg"),
            target_image: PathBuf::from("tar.jpg"),
            result: SolveResult {
                p: vec![0.0; 6],
                c_zncc: 0.99,
                c_znssd: 0.02,
                iterations: 3,
                converged: true,
                quality_ok: true,
                history: vec![],
                max_norm: 1e-5,
                tolerance: 0.75,
                subset_order: 1,
                max_iterations: 50,
                method: crate::mesh::SolveMethod::Icgn,
                eta_omitted: Some((0.0123, 0.0456)),
                texture: Texture::Deformable,
            },
            std_dev: 71.5,
            sssig: 430193.5,
            delta_f: 3166.75,
            time: None,
            seams: None,
        };
        let tmp = std::env::temp_dir().join("geopyv_test_eta_omitted_round_trip.pyv");
        save(&tmp, &GeopyvObject::Subset(sol.clone())).unwrap();
        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Subset(s) => {
                assert_eq!(s.result.eta_omitted, Some((0.0123, 0.0456)));
            }
            _ => panic!("expected Subset variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    fn seam_subset_solution() -> SubsetSolution {
        use crate::seam::*;
        let line = SeamLine { kind: EdgeKind::Hinge, theta: 0.3, c: 7.5, t: 120.0 };
        let pass = SeamPass { stage: SeamStage::Pass1, t_bend: 12.0, hinge: Some(line), jump: None,
                              decision: SeamDecision::Cut(EdgeKind::Hinge) };
        SubsetSolution {
            coord: [101.5, 99.0],
            mask: MaskSummary { shape: crate::masks::MaskShape::Circle, size: 40, n_px: 3000 },
            ref_image: PathBuf::from("ref.png"),
            target_image: PathBuf::from("tar.png"),
            result: SolveResult {
                p: vec![0.5, 0.1, 0.01, 0.0, 0.0, 0.02], c_zncc: 0.99, c_znssd: 0.02, iterations: 4,
                converged: true, quality_ok: true, history: vec![], max_norm: 1e-5, tolerance: 0.0,
                subset_order: 1, max_iterations: 80, method: crate::mesh::SolveMethod::Icgn,
                eta_omitted: None,
                texture: Texture::Deformable,
            },
            std_dev: 50.0,
            sssig: 1.0e5,
            delta_f: 3000.0,
            time: Some((0.0, 1.0)),
            seams: Some(SeamRecord {
                plain_coord: [100.0, 99.0],
                plain_p: vec![0.49, 0.1, 0.012, 0.0, 0.0, 0.02],
                moved: Some([1.5, 0.0]),
                action: SeamAction::Cut,
                lines: vec![line],
                gap: 3.0,
                passes: vec![pass],
                h: 6.0,
                levels: SeamLevels::image_free(5025, 6.0),
                flags: vec![SeamFlag::SecondLineNearCentre],
            }),
        }
    }

    /// A subset carrying a seam record survives `save`/`load` unchanged.
    #[test]
    fn test_subset_seams_round_trip() {
        let sol = seam_subset_solution();
        let tmp = std::env::temp_dir().join("geopyv_test_seams_round_trip.pyv");
        save(&tmp, &GeopyvObject::Subset(sol.clone())).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Subset(s) => assert_eq!(s.seams, sol.seams),
            _ => panic!("expected Subset variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v5 (0x0A) subset file loads with no seam record.
    #[test]
    fn test_schema_v5_subset_file_loads_via_legacy_path() {
        let sol = seam_subset_solution();
        let legacy = LegacyGeopyvObjectV5::Subset(LegacySubsetSolutionV5 {
            coord: sol.coord,
            mask: sol.mask.clone(),
            ref_image: sol.ref_image.clone(),
            target_image: sol.target_image.clone(),
            result: LegacySolveResultV7::from_current(sol.result.clone()),
            std_dev: sol.std_dev,
            sssig: sol.sssig,
            delta_f: sol.delta_f,
            time: sol.time,
        });
        let tmp = write_versioned("geopyv_test_v5_subset.pyv", VERSION_ZSTD_V5, &legacy);
        match load(&tmp).unwrap() {
            GeopyvObject::Subset(s) => {
                assert_eq!(s.result.p, sol.result.p);
                assert_eq!(s.time, Some((0.0, 1.0)));
                assert!(s.seams.is_none());
            }
            _ => panic!("expected Subset variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A mesh carrying an adaptive-masking record survives `save`/`load`.
    #[test]
    fn test_mesh_seams_round_trip() {
        let mut m = make_mesh_solution();
        let n = m.nodes.nrows();
        m.seams = Some(crate::mesh::MeshSeams {
            plain_nodes: m.nodes.clone(),
            plain_p: ndarray::Array2::zeros((n, 6)),
            flagged: vec![true, false, false],
            sent: vec![true, true, false],
            rounds: 2,
            noise_source: crate::screen::NoiseSource::Default { sigma: 2.0 },
            noise_scale: ndarray::array![0.007, 0.006, 0.008],
            still_mismatch: None,
            h: 6.1,
            records: vec![seam_subset_solution().seams, None, None],
        });
        let tmp = std::env::temp_dir().join("geopyv_test_mesh_seams_round_trip.pyv");
        save(&tmp, &GeopyvObject::Mesh(m.clone())).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Mesh(l) => assert_eq!(l.seams, m.seams),
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v5 (0x0A) mesh file loads with uniform masking and no record.
    #[test]
    fn test_schema_v5_mesh_file_loads_via_legacy_path() {
        let m = make_mesh_solution();
        let legacy = LegacyGeopyvObjectV5::Mesh(LegacyMeshSolutionV5 {
            nodes: m.nodes.clone(),
            elements: m.elements.clone(),
            boundary: m.boundary.clone(),
            exclusions: m.exclusions.clone(),
            centroids: m.centroids.clone(),
            areas: m.areas.clone(),
            warps: m.warps.clone(),
            displacements: m.displacements.clone(),
            c_zncc: m.c_zncc.clone(),
            p: m.p.clone(),
            seed_node: m.seed_node,
            mesh_order: m.mesh_order,
            subset_order: m.subset_order,
            iterations: m.iterations.clone(),
            norms: m.norms.clone(),
            f_img_path: m.f_img_path.clone(),
            g_img_path: m.g_img_path.clone(),
            solve_config: m.solve_config.clone().map(LegacySolveConfigV5::from_current),
            seed: m.seed.clone(),
            template_shape: m.template_shape.clone(),
            template_sizes: m.template_sizes.clone(),
            time: Some((0.0, 2.0)),
        });
        let tmp = write_versioned("geopyv_test_v5_mesh.pyv", VERSION_ZSTD_V5, &legacy);
        match load(&tmp).unwrap() {
            GeopyvObject::Mesh(l) => {
                assert_eq!(l.displacements, m.displacements);
                assert_eq!(l.time, Some((0.0, 2.0)));
                assert!(l.seams.is_none());
                assert_eq!(l.solve_config.map(|c| c.masking), m.solve_config.as_ref().map(|_| crate::mesh::Masking::Uniform));
            }
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v6 (0x0B) mesh file loads with no tracks.
    #[test]
    fn test_schema_v6_mesh_file_loads_via_legacy_path() {
        let m = make_mesh_solution();
        let legacy = LegacyGeopyvObjectV6::Mesh(LegacyMeshSolutionV6 {
            nodes: m.nodes.clone(),
            elements: m.elements.clone(),
            boundary: m.boundary.clone(),
            exclusions: m.exclusions.clone(),
            centroids: m.centroids.clone(),
            areas: m.areas.clone(),
            warps: m.warps.clone(),
            displacements: m.displacements.clone(),
            c_zncc: m.c_zncc.clone(),
            p: m.p.clone(),
            seed_node: m.seed_node,
            mesh_order: m.mesh_order,
            subset_order: m.subset_order,
            iterations: m.iterations.clone(),
            norms: m.norms.clone(),
            f_img_path: m.f_img_path.clone(),
            g_img_path: m.g_img_path.clone(),
            solve_config: m.solve_config.clone().map(LegacySolveConfigV7::from_current),
            seed: m.seed.clone(),
            template_shape: m.template_shape.clone(),
            template_sizes: m.template_sizes.clone(),
            time: Some((0.0, 2.0)),
            seams: None,
        });
        let tmp = write_versioned("geopyv_test_v6_mesh.pyv", VERSION_ZSTD_V6, &legacy);
        match load(&tmp).unwrap() {
            GeopyvObject::Mesh(l) => {
                assert_eq!(l.displacements, m.displacements);
                assert_eq!(l.time, Some((0.0, 2.0)));
                assert!(l.tracks.is_none());
            }
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A tracked mesh's per-subset history survives a save/load round trip.
    #[test]
    fn test_mesh_tracks_round_trip() {
        let mut m = make_mesh_solution();
        let n = m.nodes.nrows();
        let w = ndarray::Array2::from_shape_fn((n, 6), |(i, k)| 0.01 * (i * 6 + k) as f64);
        m.tracks = Some(crate::mesh::NodeTracks {
            id: (0..n as u64).map(|i| 10 + i).collect(),
            origin: m.nodes.clone(),
            position: m.nodes.clone() + 0.5,
            reference_warp: w.clone(),
            increment: w.clone() * 2.0,
            warp: w * 3.0,
        });
        let tmp = std::env::temp_dir().join("geopyv_test_tracks.pyv");
        save(&tmp, &GeopyvObject::Mesh(m.clone())).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Mesh(l) => assert_eq!(l.tracks, m.tracks),
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v7 (0x0C) subset file loads with the deformable texture.
    #[test]
    fn test_schema_v7_subset_file_loads_via_legacy_path() {
        let sol = seam_subset_solution();
        let legacy = LegacyGeopyvObjectV7::Subset(LegacySubsetSolutionV7::from_current(sol.clone()));
        let tmp = write_versioned("geopyv_test_v7_subset.pyv", VERSION_ZSTD_V7, &legacy);
        match load(&tmp).unwrap() {
            GeopyvObject::Subset(s) => {
                assert_eq!(s.result.p, sol.result.p);
                assert_eq!(s.result.texture, Texture::Deformable);
                assert_eq!(s.seams, sol.seams);
            }
            _ => panic!("expected Subset variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v7 (0x0C) mesh file loads with the deformable texture and
    /// keeps its tracks.
    #[test]
    fn test_schema_v7_mesh_file_loads_via_legacy_path() {
        let mut m = make_mesh_solution();
        let n = m.nodes.nrows();
        let w = ndarray::Array2::from_shape_fn((n, 6), |(i, k)| 0.01 * (i * 6 + k) as f64);
        m.tracks = Some(crate::mesh::NodeTracks {
            id: (0..n as u64).collect(),
            origin: m.nodes.clone(),
            position: m.nodes.clone(),
            reference_warp: w.clone(),
            increment: w.clone(),
            warp: w,
        });
        assert!(m.solve_config.is_some());
        let legacy = LegacyGeopyvObjectV7::Mesh(LegacyMeshSolutionV7::from_current(m.clone()));
        let tmp = write_versioned("geopyv_test_v7_mesh.pyv", VERSION_ZSTD_V7, &legacy);
        match load(&tmp).unwrap() {
            GeopyvObject::Mesh(l) => {
                assert_eq!(l.displacements, m.displacements);
                assert_eq!(l.tracks, m.tracks);
                assert_eq!(l.solve_config.map(|c| c.texture), Some(Texture::Deformable));
            }
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// The rigid texture survives a current-format round trip on a subset
    /// result and a mesh config.
    #[test]
    fn test_texture_round_trip() {
        let mut sol = seam_subset_solution();
        sol.result.texture = Texture::RigidSphere;
        let tmp = std::env::temp_dir().join("geopyv_test_texture_subset.pyv");
        save(&tmp, &GeopyvObject::Subset(sol)).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Subset(s) => assert_eq!(s.result.texture, Texture::RigidSphere),
            _ => panic!("expected Subset variant"),
        }
        let _ = std::fs::remove_file(tmp);
        let mut m = make_mesh_solution();
        m.solve_config.as_mut().unwrap().texture = Texture::RigidSphere;
        let tmp = std::env::temp_dir().join("geopyv_test_texture_mesh.pyv");
        save(&tmp, &GeopyvObject::Mesh(m)).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Mesh(l) => assert_eq!(l.solve_config.map(|c| c.texture), Some(Texture::RigidSphere)),
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    fn a_speckle() -> Speckle {
        Speckle::new(
            "/tmp".into(), "s".into(), ".png".into(), (64, 48), 10.0, 30, true,
            crate::speckle::Progression::Deformation, [0.0; 12], [32.0, 24.0], 0.0, 0.0,
            WarpMode::None, 2, crate::speckle::ScaleType::Lin,
        ).unwrap()
    }

    /// A schema v8 (0x0D) speckle file loads as round grains.
    #[test]
    fn test_schema_v8_speckle_loads_round() {
        let sp = a_speckle();
        let legacy = LegacyGeopyvObjectV8::Speckle(LegacySpeckleV8 {
            image_dir: sp.image_dir.clone(), name: sp.name.clone(), file_format: sp.file_format.clone(),
            image_size: sp.image_size, speckle_size: sp.speckle_size, speckle_number: sp.speckle_number,
            speckle_limit: sp.speckle_limit, progression: sp.progression.clone(), comp: sp.comp,
            origin: sp.origin, mode: sp.mode.clone(), noise_pos: sp.noise_pos, noise_int: sp.noise_int,
            image_no: sp.image_no, scale: sp.scale.clone(), speckle_positions: None, solved: false,
        });
        let tmp = write_versioned("geopyv_test_v8_speckle.pyv", VERSION_ZSTD_V8, &legacy);
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                assert_eq!(s.grain_shape, crate::speckle::GrainShape::Round);
                assert_eq!(s.speckle_number, 30);
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// Grain settings survive a current-format round trip.
    #[test]
    fn test_speckle_grains_round_trip() {
        use crate::speckle::{GrainRotation, GrainShape};
        let sp = a_speckle().with_grains(GrainShape::Ellipse { aspect: 4.0 }, GrainRotation::None).unwrap()
            .with_overlap(crate::speckle::GrainOverlap::Opaque).with_no_overlap(0.4).unwrap();
        let tmp = std::env::temp_dir().join("geopyv_test_speckle_grains.pyv");
        save(&tmp, &GeopyvObject::Speckle(sp)).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                assert_eq!(s.grain_shape, GrainShape::Ellipse { aspect: 4.0 });
                assert_eq!(s.grain_rotation, GrainRotation::None);
                assert_eq!(s.grain_overlap, crate::speckle::GrainOverlap::Opaque);
                assert_eq!(s.no_overlap, Some(0.4));
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// Schema v9 (0x0E) and v10 (0x0F) speckle files load with their grain
    /// settings and the later fields at their defaults.
    #[test]
    fn test_schema_v9_v10_speckle_load() {
        use crate::speckle::{GrainOverlap, GrainRotation, GrainShape};
        let sp = a_speckle();
        let v10 = LegacySpeckleV10 {
            image_dir: sp.image_dir.clone(), name: sp.name.clone(), file_format: sp.file_format.clone(),
            image_size: sp.image_size, speckle_size: sp.speckle_size, speckle_number: sp.speckle_number,
            speckle_limit: sp.speckle_limit, progression: sp.progression.clone(), comp: sp.comp,
            origin: sp.origin, mode: sp.mode.clone(), noise_pos: sp.noise_pos, noise_int: sp.noise_int,
            image_no: sp.image_no, scale: sp.scale.clone(), speckle_positions: None, solved: false,
            grain_shape: GrainShape::Ellipse { aspect: 2.0 }, grain_rotation: GrainRotation::None,
            grain_overlap: GrainOverlap::Opaque,
        };
        let tmp = write_versioned("geopyv_test_v10_speckle.pyv", VERSION_ZSTD_V10, &LegacyGeopyvObjectV10::Speckle(v10));
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                assert_eq!(s.grain_overlap, GrainOverlap::Opaque);
                assert_eq!(s.no_overlap, None);
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
        let v9 = LegacySpeckleV9 {
            image_dir: sp.image_dir.clone(), name: sp.name.clone(), file_format: sp.file_format.clone(),
            image_size: sp.image_size, speckle_size: sp.speckle_size, speckle_number: sp.speckle_number,
            speckle_limit: sp.speckle_limit, progression: sp.progression.clone(), comp: sp.comp,
            origin: sp.origin, mode: sp.mode.clone(), noise_pos: sp.noise_pos, noise_int: sp.noise_int,
            image_no: sp.image_no, scale: sp.scale.clone(), speckle_positions: None, solved: false,
            grain_shape: GrainShape::Cluster { n: 4 }, grain_rotation: GrainRotation::Material,
        };
        let tmp = write_versioned("geopyv_test_v9_speckle.pyv", VERSION_ZSTD_V9, &LegacyGeopyvObjectV9::Speckle(v9));
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                assert_eq!(s.grain_shape, GrainShape::Cluster { n: 4 });
                assert_eq!(s.grain_overlap, GrainOverlap::Add);
                assert_eq!(s.no_overlap, None);
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v11 (0x10) speckle file loads with no soil; a soil setting
    /// survives a current-format round trip.
    #[test]
    fn test_speckle_soil_versions() {
        let sp = a_speckle().with_no_overlap(0.3).unwrap();
        let v11 = LegacySpeckleV11 {
            image_dir: sp.image_dir.clone(), name: sp.name.clone(), file_format: sp.file_format.clone(),
            image_size: sp.image_size, speckle_size: sp.speckle_size, speckle_number: sp.speckle_number,
            speckle_limit: sp.speckle_limit, progression: sp.progression.clone(), comp: sp.comp,
            origin: sp.origin, mode: sp.mode.clone(), noise_pos: sp.noise_pos, noise_int: sp.noise_int,
            image_no: sp.image_no, scale: sp.scale.clone(), speckle_positions: None, solved: false,
            grain_shape: sp.grain_shape, grain_rotation: sp.grain_rotation, grain_overlap: sp.grain_overlap,
            no_overlap: sp.no_overlap,
        };
        let tmp = write_versioned("geopyv_test_v11_speckle.pyv", VERSION_ZSTD_V11, &LegacyGeopyvObjectV11::Speckle(v11));
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                assert_eq!(s.no_overlap, Some(0.3));
                assert!(s.soil.is_none());
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
        let soil = crate::soil::SoilConfig::clay(180.0, 135.0, 0.2);
        let sp = a_speckle().with_soil(soil.clone()).unwrap();
        let tmp = std::env::temp_dir().join("geopyv_test_speckle_soil.pyv");
        save(&tmp, &GeopyvObject::Speckle(sp)).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => assert_eq!(s.soil, Some(soil)),
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v12 (0x11) soil file loads with its settings converted.
    #[test]
    fn test_schema_v12_soil_converts() {
        let sp = a_speckle();
        let old = LegacySoilConfigV12 {
            kind: crate::soil::SoilKind::Clay, matrix_brightness: 180.0, grain_brightness: 135.0, grain_spread: 0.04,
            grain_size: 6.0, grain_size_spread: 1.8, grain: LegacySoilGrain::Sphere, coverage: 0.15,
            depth_layers: 1, depth_dimming: 1.0, pattern_strength: 0.05, pattern_size: 10.0, blur: 1.0,
        };
        let v12 = LegacySpeckleV12 {
            image_dir: sp.image_dir.clone(), name: sp.name.clone(), file_format: sp.file_format.clone(),
            image_size: sp.image_size, speckle_size: sp.speckle_size, speckle_number: sp.speckle_number,
            speckle_limit: sp.speckle_limit, progression: sp.progression.clone(), comp: sp.comp,
            origin: sp.origin, mode: sp.mode.clone(), noise_pos: sp.noise_pos, noise_int: sp.noise_int,
            image_no: sp.image_no, scale: sp.scale.clone(), speckle_positions: None, solved: false,
            grain_shape: sp.grain_shape, grain_rotation: sp.grain_rotation, grain_overlap: sp.grain_overlap,
            no_overlap: None, soil: Some(old),
        };
        let tmp = write_versioned("geopyv_test_v12_soil.pyv", VERSION_ZSTD_V12, &LegacyGeopyvObjectV12::Speckle(v12));
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                let c = s.soil.unwrap();
                assert_eq!((c.matrix_brightness, c.matrix_spread, c.seed_brightness, c.seed_spread, c.asr), (180.0, 0.05, 135.0, 0.04, 0.15));
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v13 (0x12) soil file loads with depth dimming dropped.
    #[test]
    fn test_schema_v13_soil_drops_dimming() {
        let sp = a_speckle();
        let old = LegacySoilConfigV13 {
            kind: crate::soil::SoilKind::Sand, matrix_brightness: 107.0, matrix_spread: 0.09, seed_brightness: 68.0,
            seed_spread: 0.09, asr: 0.26, grain_size: 6.0, grain_size_spread: 1.8, grain: LegacySoilGrain::Sphere,
            depth_layers: 4, depth_dimming: 0.8, pattern_size: 10.0, blur: 1.0,
        };
        let v13 = LegacySpeckleV13 {
            image_dir: sp.image_dir.clone(), name: sp.name.clone(), file_format: sp.file_format.clone(),
            image_size: sp.image_size, speckle_size: sp.speckle_size, speckle_number: sp.speckle_number,
            speckle_limit: sp.speckle_limit, progression: sp.progression.clone(), comp: sp.comp,
            origin: sp.origin, mode: sp.mode.clone(), noise_pos: sp.noise_pos, noise_int: sp.noise_int,
            image_no: sp.image_no, scale: sp.scale.clone(), speckle_positions: None, solved: false,
            grain_shape: sp.grain_shape, grain_rotation: sp.grain_rotation, grain_overlap: sp.grain_overlap,
            no_overlap: None, soil: Some(old),
        };
        let tmp = write_versioned("geopyv_test_v13_soil.pyv", VERSION_ZSTD_V13, &LegacyGeopyvObjectV13::Speckle(v13));
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                let c = s.soil.unwrap();
                assert_eq!((c.matrix_brightness, c.matrix_spread, c.asr, c.depth_layers, c.blur), (107.0, 0.09, 0.26, 4, 1.0));
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v14 (0x13) soil file loads with no seed kept; a current file
    /// keeps the seed of its solve.
    #[test]
    fn test_schema_v14_speckle_seed() {
        let sp = a_speckle();
        let v14 = LegacySpeckleV14 {
            image_dir: sp.image_dir.clone(), name: sp.name.clone(), file_format: sp.file_format.clone(),
            image_size: sp.image_size, speckle_size: sp.speckle_size, speckle_number: sp.speckle_number,
            speckle_limit: sp.speckle_limit, progression: sp.progression.clone(), comp: sp.comp,
            origin: sp.origin, mode: sp.mode.clone(), noise_pos: sp.noise_pos, noise_int: sp.noise_int,
            image_no: sp.image_no, scale: sp.scale.clone(), speckle_positions: None, solved: false,
            grain_shape: sp.grain_shape, grain_rotation: sp.grain_rotation, grain_overlap: sp.grain_overlap,
            no_overlap: None, soil: Some(old_soil()),
        };
        let tmp = write_versioned("geopyv_test_v14_seed.pyv", VERSION_ZSTD_V14, &LegacyGeopyvObjectV14::Speckle(v14));
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => assert!(s.seed.is_none() && s.soil.unwrap().d_p > 6.0),
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
        let mut sp = a_speckle();
        sp.seed = Some(17);
        let tmp = std::env::temp_dir().join("geopyv_test_v15_seed.pyv");
        save(&tmp, &GeopyvObject::Speckle(sp)).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => assert_eq!(s.seed, Some(17)),
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    fn old_soil() -> LegacySoilConfigV15 {
        LegacySoilConfigV15 {
            kind: crate::soil::SoilKind::Clay, matrix_brightness: 180.0, matrix_spread: 0.05, seed_brightness: 105.0,
            seed_spread: 0.09, asr: 0.1, grain_size: 6.0, grain_size_spread: 1.8,
            grain: LegacySoilGrain::Elongated { aspect: 2.0 }, depth_layers: 1, pattern_size: 10.0, blur: 0.0,
        }
    }

    /// A schema v15 (0x14) soil file loads with its grain size and shape
    /// turned into a characterisation; the seed is kept.
    #[test]
    fn test_schema_v15_soil_characterisation() {
        let sp = a_speckle();
        let v15 = LegacySpeckleV15 {
            image_dir: sp.image_dir.clone(), name: sp.name.clone(), file_format: sp.file_format.clone(),
            image_size: sp.image_size, speckle_size: sp.speckle_size, speckle_number: sp.speckle_number,
            speckle_limit: sp.speckle_limit, progression: sp.progression.clone(), comp: sp.comp,
            origin: sp.origin, mode: sp.mode.clone(), noise_pos: sp.noise_pos, noise_int: sp.noise_int,
            image_no: sp.image_no, scale: sp.scale.clone(), speckle_positions: None, solved: false,
            grain_shape: sp.grain_shape, grain_rotation: sp.grain_rotation, grain_overlap: sp.grain_overlap,
            no_overlap: None, soil: Some(old_soil()), seed: Some(5),
        };
        let tmp = write_versioned("geopyv_test_v15_soil.pyv", VERSION_ZSTD_V15, &LegacyGeopyvObjectV15::Speckle(v15));
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                assert_eq!(s.seed, Some(5));
                let c = s.soil.unwrap();
                let sigma = 1.8f64.ln() / 1.2816;
                assert!((c.d_p - 6.0 * (2.0 * sigma * sigma).exp()).abs() < 1e-9);
                assert!((c.aspect_ratio - 0.5).abs() < 1e-12 && c.outlines.is_none());
                assert!(c.validate().is_ok());
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v16 (0x15) soil file loads with its sphericity dropped and the
    /// default outline statistics; a current file keeps given statistics.
    #[test]
    fn test_schema_v16_soil_drops_sphericity() {
        let sp = a_speckle();
        let v16 = LegacySpeckleV16 {
            image_dir: sp.image_dir.clone(), name: sp.name.clone(), file_format: sp.file_format.clone(),
            image_size: sp.image_size, speckle_size: sp.speckle_size, speckle_number: sp.speckle_number,
            speckle_limit: sp.speckle_limit, progression: sp.progression.clone(), comp: sp.comp,
            origin: sp.origin, mode: sp.mode.clone(), noise_pos: sp.noise_pos, noise_int: sp.noise_int,
            image_no: sp.image_no, scale: sp.scale.clone(), speckle_positions: None, solved: false,
            grain_shape: sp.grain_shape, grain_rotation: sp.grain_rotation, grain_overlap: sp.grain_overlap,
            no_overlap: None, seed: Some(5),
            soil: Some(LegacySoilConfigV16 {
                kind: crate::soil::SoilKind::Sand, matrix_brightness: 106.0, matrix_spread: 0.15,
                seed_brightness: 62.0, seed_spread: 0.09, asr: 0.18, psd: Vec::new(), d_p: 6.0,
                aspect_ratio: 0.7, aspect_ratio_sd: 0.1, sphericity: 0.84, sphericity_sd: 0.06,
                depth_layers: 6, pattern_size: 10.0, blur: 0.0,
            }),
        };
        let tmp = write_versioned("geopyv_test_v16_soil.pyv", VERSION_ZSTD_V16, &LegacyGeopyvObjectV16::Speckle(v16));
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                let c = s.soil.unwrap();
                assert!((c.aspect_ratio - 0.7).abs() < 1e-12 && (c.aspect_ratio_sd - 0.1).abs() < 1e-12);
                assert!(c.outlines.is_none() && c.validate().is_ok());
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
        let mut sp = a_speckle();
        let mut soil = crate::soil::SoilConfig::sand(106.0, 62.0, 0.18);
        let mut stats = crate::soil::OutlineStats::leighton_buzzard();
        stats.irregularity = 0.05;
        soil.outlines = Some(stats.clone());
        sp.soil = Some(soil);
        let tmp = std::env::temp_dir().join("geopyv_test_v17_outlines.pyv");
        save(&tmp, &GeopyvObject::Speckle(sp)).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => assert_eq!(s.soil.unwrap().outlines, Some(stats)),
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A schema v17 (0x16) soil file loads with its outline statistics and no
    /// brightness variation inside grains; a current file keeps the settings.
    #[test]
    fn test_schema_v17_soil_has_no_grain_texture() {
        let sp = a_speckle();
        let stats = crate::soil::OutlineStats::leighton_buzzard();
        let v17 = LegacySpeckleV17 {
            image_dir: sp.image_dir.clone(), name: sp.name.clone(), file_format: sp.file_format.clone(),
            image_size: sp.image_size, speckle_size: sp.speckle_size, speckle_number: sp.speckle_number,
            speckle_limit: sp.speckle_limit, progression: sp.progression.clone(), comp: sp.comp,
            origin: sp.origin, mode: sp.mode.clone(), noise_pos: sp.noise_pos, noise_int: sp.noise_int,
            image_no: sp.image_no, scale: sp.scale.clone(), speckle_positions: None, solved: false,
            grain_shape: sp.grain_shape, grain_rotation: sp.grain_rotation, grain_overlap: sp.grain_overlap,
            no_overlap: None, seed: Some(5),
            soil: Some(LegacySoilConfigV17 {
                kind: crate::soil::SoilKind::Sand, matrix_brightness: 106.0, matrix_spread: 0.15,
                seed_brightness: 62.0, seed_spread: 0.09, asr: 0.18, psd: Vec::new(), d_p: 6.0,
                aspect_ratio: 0.7, aspect_ratio_sd: 0.1, outlines: Some(stats.clone()),
                depth_layers: 6, pattern_size: 10.0, blur: 0.8,
            }),
        };
        let tmp = write_versioned("geopyv_test_v17_soil.pyv", VERSION_ZSTD_V17, &LegacyGeopyvObjectV17::Speckle(v17));
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                let c = s.soil.unwrap();
                assert_eq!(c.outlines, Some(stats));
                assert!(c.grain_tilt == 0.0 && c.grain_dome == 0.0 && c.grain_rim == 0.0 && c.grain_texture == 0.0);
                assert!((c.blur - 0.8).abs() < 1e-12 && c.validate().is_ok());
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
        let mut sp = a_speckle();
        sp.soil = Some(crate::soil::SoilConfig {
            grain_tilt: 0.16, grain_dome: 0.15, grain_rim: 0.05, grain_texture: 0.06, texture_size: 0.08,
            ..crate::soil::SoilConfig::sand(106.0, 62.0, 0.18)
        });
        let tmp = std::env::temp_dir().join("geopyv_test_v18_texture.pyv");
        save(&tmp, &GeopyvObject::Speckle(sp)).unwrap();
        match load(&tmp).unwrap() {
            GeopyvObject::Speckle(s) => {
                let c = s.soil.unwrap();
                assert_eq!((c.grain_tilt, c.grain_dome, c.grain_rim, c.grain_texture, c.texture_size),
                           (0.16, 0.15, 0.05, 0.06, 0.08));
            }
            _ => panic!("expected Speckle"),
        }
        let _ = std::fs::remove_file(tmp);
    }
}
