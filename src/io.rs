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
//!                    | 0x03 (zstd, schema v1) | 0x07 (zstd, schema v2 — current)
//! [5..]    bincode v2 (standard config) encoded GeopyvObject (schema-version-
//!          dependent shape — see "Schema versioning" below), zstd-compressed
//!          (level 3) for every version except 0x01
//! ```
//!
//! New files are always written with version 0x07 (zstd + current schema).
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
    field::FieldSolution,
    geometry::region::Region,
    mesh::{MeshSolution, SeedConfig, SolveConfig, SolveMethod},
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
/// zstd-compressed, schema v2 (current). Relative to v1: `MeshSolution`
/// gains `template_shape` / `template_sizes` / `zonal_masking`;
/// `SolveConfig` gains `preconditioning` / `layer_rg`; `SolveResult` gains
/// `eta_omitted`; `ParticleSolution` gains `principal_strains` /
/// `gamma_max_grad`; `ParticleConfig` gains `strain_method`; `Speckle` gains
/// `speckle_limit`. Always written by [`save`]. (0x04-0x06 skipped: used by
/// unreleased development builds with incompatible shapes.)
const CURRENT_VERSION: u8 = 0x07;

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
            zonal_masking: None,
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
            zonal_masking: None,
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
/// Files with version `0x01`/`0x02` (schema v0) or `0x03` (schema v1, the
/// released package's format) are decoded via the frozen `Legacy*` /
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
        VERSION_ZSTD_V0 | VERSION_ZSTD_V1 | CURRENT_VERSION => {
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
            zonal_masking: None,
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
                assert!(m.zonal_masking.is_none());
            }
            _ => panic!("expected Mesh variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// A populated `ZonalMaskingRecord` survives save/load at schema v3.
    #[test]
    fn test_zonal_masking_record_round_trips() {
        use crate::mesh::ZonalMaskingRecord;
        let mut original = make_mesh_solution();
        let mut zone_image = ndarray::Array2::<u8>::ones((6, 8));
        zone_image.slice_mut(ndarray::s![.., 4..]).fill(2);
        original.zonal_masking = Some(ZonalMaskingRecord {
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
            zone_image: zone_image.clone(),
        });
        let tmp = std::env::temp_dir().join("geopyv_test_zonal_record.pyv");
        save(&tmp, &GeopyvObject::Mesh(original.clone())).unwrap();
        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Mesh(m) => {
                let rec = m.zonal_masking.as_ref().expect("zonal record round-trips");
                assert_eq!(rec.node_boundary, vec![false, true, false]);
                assert_eq!(rec.node_gamma_max_grad, array![0.002_f64, 0.05, 0.003]);
                assert_eq!(rec.node_zone, array![1_u32, 2, 1]);
                assert_eq!(rec.node_post_px, array![118_u32, 61, 119]);
                assert_eq!(rec.node_guard_fallback, vec![false, false, true]);
                assert_eq!(rec.zone_image, zone_image);
                assert!((rec.k - 2.0).abs() < 1e-15);
            }
            _ => panic!("expected Mesh variant"),
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
                assert!(m.zonal_masking.is_none());
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
            },
            std_dev: 71.5,
            sssig: 430193.5,
            delta_f: 3166.75,
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
}
