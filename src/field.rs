//! Field: distributed Lagrangian/Eulerian particle tracking and strain-path computation.
//!
//! Translates `geopyv/src/geopyv/field.py` (Field class; all `geomat`
//! sections are excluded — no stress path, no friction work).
//!
//! # Architecture
//!
//! - `grid_particles` — free function; regular-grid placement inside a
//!   boundary polygon (also used by front-ends to preview it).
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

use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    calibration::CalibrationParams,
    geometry::utilities::point_in_polygon,
    particle::{
        meshless_scalar_gradient_batch, Particle, ParticleConfig, ParticleSource,
        ParticleSolution, StrainMethod,
    },
    sequence::SequenceSolution,
    Error,
};

// ---------------------------------------------------------------------------
// grid_particles — free function
// ---------------------------------------------------------------------------

/// Place particles on a regular square grid of pitch `spacing` inside
/// `boundary` and outside every polygon in `exclusions`.
///
/// Grid points sit at cell centres, starting half a pitch in from the
/// boundary's bounding-box minimum. Each particle represents one grid cell,
/// so its volume is `spacing² × depth` (the same area-times-depth convention
/// as [`distribute_particles`]). Front-ends call this directly to preview the
/// placement [`FieldDistribution::Grid`] will produce.
///
/// # Returns
/// `(coordinates, volumes)` — `(P, 2)` and `(P,)`. Empty when `boundary` has
/// fewer than 3 vertices or `spacing` is not positive.
pub fn grid_particles(
    boundary: ArrayView2<f64>,
    exclusions: &[ArrayView2<f64>],
    spacing: f64,
    depth: f64,
) -> (Array2<f64>, Array1<f64>) {
    let empty = || (Array2::zeros((0, 2)), Array1::zeros(0));
    if boundary.nrows() < 3 || !(spacing > 0.0) {
        return empty();
    }
    let fold = |col: usize, f: fn(f64, f64) -> f64, init: f64| {
        boundary.column(col).iter().copied().fold(init, f)
    };
    let (min_x, max_x) = (fold(0, f64::min, f64::INFINITY), fold(0, f64::max, f64::NEG_INFINITY));
    let (min_y, max_y) = (fold(1, f64::min, f64::INFINITY), fold(1, f64::max, f64::NEG_INFINITY));

    let nx = ((max_x - min_x) / spacing).floor() as usize;
    let ny = ((max_y - min_y) / spacing).floor() as usize;
    let mut points: Vec<f64> = Vec::new();
    for i in 0..=nx {
        let x = min_x + spacing * (i as f64 + 0.5);
        if x > max_x { break; }
        for j in 0..=ny {
            let y = min_y + spacing * (j as f64 + 0.5);
            if y > max_y { break; }
            if point_in_polygon([x, y], boundary)
                && !exclusions.iter().any(|ex| point_in_polygon([x, y], *ex))
            {
                points.extend_from_slice(&[x, y]);
            }
        }
    }
    let n = points.len() / 2;
    let coordinates = Array2::from_shape_vec((n, 2), points).expect("2 values pushed per point");
    (coordinates, Array1::from_elem(n, spacing * spacing * depth))
}

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
    /// Regular square grid of pitch `spacing` inside `boundary_nodes`,
    /// outside every `exclusion_nodes` polygon — see [`grid_particles`].
    Grid {
        boundary_nodes: Array2<f64>,
        exclusion_nodes: Vec<Array2<f64>>,
        spacing: f64,
    },
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
    /// Assumed out-of-plane depth used for volumetric strain calculations.
    #[serde(default)]
    pub depth: f64,
    /// Whether per-particle tracking (`Particle.track`) was enabled.
    #[serde(default)]
    pub track: bool,
    /// The region the particles were placed in, in the same (reference)
    /// space as `initial_coordinates` — used to drop contour triangles that
    /// bridge exclusions or concave boundary sections. `None` only for
    /// solutions predating this field (.pyv schema < 0x08).
    #[serde(default)]
    pub region: Option<FieldRegion>,
}

/// Boundary and exclusion polygons of a [`Field`], each `(N, 2)` `[x, y]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldRegion {
    pub boundary: Array2<f64>,
    pub exclusions: Vec<Array2<f64>>,
}

impl FieldRegion {
    /// Inside the boundary and outside every exclusion.
    pub fn contains(&self, p: [f64; 2]) -> bool {
        point_in_polygon(p, self.boundary.view())
            && !self.exclusions.iter().any(|ex| point_in_polygon(p, ex.view()))
    }

    /// The reference-mesh boundary/exclusion polygons of `mesh`.
    fn from_mesh(mesh: &crate::mesh::MeshSolution) -> Self {
        let poly = |idx: &[usize]| {
            Array2::from_shape_fn((idx.len(), 2), |(i, j)| mesh.nodes[[idx[i], j]])
        };
        FieldRegion {
            boundary: poly(&mesh.boundary),
            exclusions: mesh.exclusions.iter().map(|e| poly(e)).collect(),
        }
    }

    /// Map every polygon from image to object space.
    fn i2o(&self, params: &CalibrationParams) -> Self {
        FieldRegion {
            boundary: params.i2o(self.boundary.view()),
            exclusions: self.exclusions.iter().map(|e| params.i2o(e.view())).collect(),
        }
    }
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
    /// Region the particles were placed in. Set by [`FieldDistribution::Grid`];
    /// otherwise taken from the first mesh's boundary/exclusions at solve time.
    pub region: Option<FieldRegion>,
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

        let mut region = None;
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
            FieldDistribution::Grid { boundary_nodes, exclusion_nodes, spacing } => {
                let views: Vec<_> = exclusion_nodes.iter().map(|e| e.view()).collect();
                let (coordinates, volumes) =
                    grid_particles(boundary_nodes.view(), &views, spacing, depth);
                if coordinates.nrows() == 0 {
                    return Err(Error::InvalidInput(
                        "grid distribution placed no particles: check boundary and spacing".to_string(),
                    ));
                }
                region = Some(FieldRegion { boundary: boundary_nodes, exclusions: exclusion_nodes });
                (coordinates, volumes)
            }
        };

        Ok(Field { source, coordinates, volumes, track, depth, region, solution: None })
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
            options: None,
            border: 0,
        });
        let n = sol.initial_coordinates.nrows();
        let mut volumes = Array1::<f64>::zeros(n);
        for (i, p) in sol.particles.iter().enumerate().take(n) {
            if !p.volumes.is_empty() { volumes[i] = p.volumes[0]; }
        }
        let coordinates = sol.initial_coordinates.clone();
        let region = sol.region.clone();
        Field { source: dummy_source, coordinates, volumes, track: true, depth: 1.0,
                region, solution: Some(sol) }
    }

    /// Solve strain paths for all particles.
    ///
    /// Iterates over increments sequentially, loading one mesh at a time.  For
    /// saved-by-reference sequences each mesh file is read exactly once; all
    /// particles advance their increment in parallel (rayon), then the mesh is
    /// dropped before the next file is opened.
    pub fn solve(&mut self, factor: f64, true_incs: bool, calibration: Option<&CalibrationParams>, strain_method: StrainMethod) -> Result<(), Error> {
        let n = self.n_particles();
        let n_meshes = self.source.n_meshes();
        let cfg = ParticleConfig { factor, true_incs, strain_method };
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
        let mut region = self.region.clone();
        for m in 0..n_meshes {
            let mesh = source.load_mesh_at(m)?;
            if m == 0 && region.is_none() {
                region = Some(FieldRegion::from_mesh(&mesh));
            }
            particles.par_iter_mut().for_each(|p| {
                p.solve_increment(m, &mesh, &cfg, calibration);
            });
        }

        // Phase 3: finalize strain paths.
        let calibrated = calibration.is_some();
        let mut solutions: Vec<ParticleSolution> = particles.iter()
            .map(|p| {
                let mut sol = p.finalize(&cfg);
                sol.calibrated = calibrated;
                sol
            })
            .collect();

        // Phase 3b: cross-particle meshless spatial gradient of gamma_max,
        // one full pass per increment across every particle's already-
        // finalized principal_strains history -- see ParticleSolution::
        // gamma_max_grad's doc comment for why this only runs under
        // StrainMethod::Meshless (no independent fallback radius). Done
        // here, after every particle's full strain history is already
        // finalized, rather than interleaved into Phase 2's per-increment
        // loop above -- strain_def/principal_strains are both whole-history
        // functions (operate on the full accumulated `warps` array in one
        // shot), not incremental, so there is no per-increment gamma_max
        // available until finalize() has already run for every particle.
        if let StrainMethod::Meshless(params) = &cfg.strain_method {
            if let Some(inc_no) = solutions.first().map(|s| s.coordinates.nrows()) {
                let n_particles = solutions.len();
                let mut grads: Vec<Array2<f64>> = (0..n_particles)
                    .map(|_| Array2::<f64>::from_elem((inc_no, 2), f64::NAN))
                    .collect();

                for m in 0..inc_no {
                    let mut coords = Array2::<f64>::zeros((n_particles, 2));
                    let mut values = Array1::<f64>::zeros(n_particles);
                    for (i, sol) in solutions.iter().enumerate() {
                        coords[[i, 0]] = sol.coordinates[[m, 0]];
                        coords[[i, 1]] = sol.coordinates[[m, 1]];
                        values[i] = sol
                            .principal_strains
                            .as_ref()
                            .map(|p| p[[m, 2]])
                            .unwrap_or(f64::NAN);
                    }
                    let grad = meshless_scalar_gradient_batch(
                        coords.view(), coords.view(), values.view(), params,
                    );
                    for i in 0..n_particles {
                        grads[i][[m, 0]] = grad[[i, 0]];
                        grads[i][[m, 1]] = grad[[i, 1]];
                    }
                }

                for (sol, g) in solutions.iter_mut().zip(grads.into_iter()) {
                    sol.gamma_max_grad = Some(g);
                }
            }
        }

        let particle_solutions: Vec<Arc<ParticleSolution>> =
            solutions.into_iter().map(Arc::new).collect();

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
            depth: self.depth,
            track: self.track,
            region: region.map(|r| match calibration {
                Some(params) => r.i2o(params),
                None => r,
            }),
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Contour data — shared by the Python `Field.contour()` and the GUI
// ---------------------------------------------------------------------------

/// A per-particle scalar that [`FieldSolution::contour_values`] can plot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldQuantity {
    U,
    V,
    R,
    EpXx,
    EpYy,
    EpXy,
    EpVol,
    Ep1,
    Ep2,
    GammaMax,
    ThetaP,
    GammaMaxGrad,
}

impl FieldQuantity {
    pub const ALL: [FieldQuantity; 12] = [
        Self::U, Self::V, Self::R, Self::EpXx, Self::EpYy, Self::EpXy, Self::EpVol,
        Self::Ep1, Self::Ep2, Self::GammaMax, Self::ThetaP, Self::GammaMaxGrad,
    ];

    /// The Python-facing quantity name (`"u"`, `"ep_xx"`, `"gamma_max_grad"`, ...).
    pub fn name(self) -> &'static str {
        match self {
            Self::U => "u",
            Self::V => "v",
            Self::R => "R",
            Self::EpXx => "ep_xx",
            Self::EpYy => "ep_yy",
            Self::EpXy => "ep_xy",
            Self::EpVol => "ep_vol",
            Self::Ep1 => "ep1",
            Self::Ep2 => "ep2",
            Self::GammaMax => "gamma_max",
            Self::ThetaP => "theta_p",
            Self::GammaMaxGrad => "gamma_max_grad",
        }
    }
}

impl std::str::FromStr for FieldQuantity {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        Self::ALL.into_iter().find(|q| q.name() == s).ok_or_else(|| {
            let names: Vec<_> = Self::ALL.iter().map(|q| q.name()).collect();
            Error::InvalidInput(format!("quantity must be one of {names:?}, got {s:?}"))
        })
    }
}

/// How a per-increment series is collapsed to one contour value.
///
/// `window` is a half-open increment range `[start, stop)`; `None` means the
/// whole series. With `dt` unset the value is last-minus-first over the
/// window (or the sum of |increment deltas| if `absolute`); with `dt` set it
/// is the mean rate `(last - first) / (len * dt)`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ContourReduction {
    pub window: Option<(usize, usize)>,
    pub dt: Option<f64>,
    pub absolute: bool,
}

impl ContourReduction {
    /// The concrete `[start, stop)` range for a series of `inc_no` values.
    fn range(&self, inc_no: usize) -> Result<(usize, usize), Error> {
        let (start, stop) = self.window.unwrap_or((0, inc_no));
        if start >= stop || stop > inc_no {
            return Err(Error::InvalidInput(format!(
                "window [{start}, {stop}) is empty or exceeds the {inc_no} increments"
            )));
        }
        Ok((start, stop))
    }

    fn reduce(&self, v: &[f64]) -> f64 {
        let n = v.len();
        match self.dt {
            None if self.absolute => v.windows(2).map(|w| (w[1] - w[0]).abs()).sum(),
            None if n > 1 => v[n - 1] - v[0],
            None => v[n - 1],
            Some(dt) if n > 1 => (v[n - 1] - v[0]) / (n as f64 * dt),
            Some(_) => 0.0,
        }
    }
}

/// One per-increment series of a [`ParticleSolution`], as plotted by
/// `history()` / `trace()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesQuantity {
    /// Column of `warps`.
    Warp(usize),
    /// Column of `strains`.
    Strain(usize),
    VolStrain,
}

impl ParticleSolution {
    /// The per-increment values of `q`, length `inc_no`.
    pub fn series(&self, q: SeriesQuantity) -> Array1<f64> {
        match q {
            SeriesQuantity::Warp(c) => self.warps.column(c).to_owned(),
            SeriesQuantity::Strain(c) => self.strains.column(c).to_owned(),
            SeriesQuantity::VolStrain => self.vol_strains.clone(),
        }
    }
}

impl FieldSolution {
    /// Number of increments (rows) in every particle's history.
    pub fn inc_no(&self) -> usize {
        self.particles.first().map_or(0, |p| p.coordinates.nrows())
    }

    /// One reduced value of `q` per particle — see [`ContourReduction`].
    ///
    /// `GammaMaxGrad` reduces each gradient component separately and returns
    /// the magnitude of the result. Errors if `q` needs a stored array this
    /// solution lacks (`principal_strains` on pre-v2 files, `gamma_max_grad`
    /// unless solved meshless via `Field::solve`).
    pub fn contour_values(&self, q: FieldQuantity, red: &ContourReduction) -> Result<Array1<f64>, Error> {
        let (start, stop) = red.range(self.inc_no())?;
        let mut out = Array1::<f64>::zeros(self.particles.len());
        let mut buf = Vec::with_capacity(stop - start);
        for (i, p) in self.particles.iter().enumerate() {
            let col = |a: &Array2<f64>, c: usize, buf: &mut Vec<f64>| {
                buf.clear();
                buf.extend((start..stop).map(|m| a[[m, c]]));
            };
            match q {
                FieldQuantity::U => col(&p.warps, 0, &mut buf),
                FieldQuantity::V => col(&p.warps, 1, &mut buf),
                FieldQuantity::R => {
                    buf.clear();
                    buf.extend((start..stop).map(|m| p.warps[[m, 0]].hypot(p.warps[[m, 1]])));
                }
                FieldQuantity::EpXx => col(&p.strains, 0, &mut buf),
                FieldQuantity::EpYy => col(&p.strains, 1, &mut buf),
                FieldQuantity::EpXy => col(&p.strains, 5, &mut buf),
                FieldQuantity::EpVol => {
                    buf.clear();
                    buf.extend((start..stop).map(|m| p.vol_strains[m]));
                }
                FieldQuantity::Ep1 | FieldQuantity::Ep2 | FieldQuantity::GammaMax | FieldQuantity::ThetaP => {
                    let ps = p.principal_strains.as_ref().ok_or_else(|| Error::InvalidInput(
                        "principal_strains is not available on this solution (it predates \
                         this field -- re-solve to populate it)".to_string(),
                    ))?;
                    let c = match q {
                        FieldQuantity::Ep1 => 0,
                        FieldQuantity::Ep2 => 1,
                        FieldQuantity::GammaMax => 2,
                        _ => 3,
                    };
                    col(ps, c, &mut buf);
                }
                FieldQuantity::GammaMaxGrad => {
                    let g = p.gamma_max_grad.as_ref().ok_or_else(|| Error::InvalidInput(
                        "gamma_max_grad is not available -- it requires solving via \
                         Field.solve() with strain_method left as meshless (the default)"
                            .to_string(),
                    ))?;
                    col(g, 0, &mut buf);
                    let gx = red.reduce(&buf);
                    col(g, 1, &mut buf);
                    let gy = red.reduce(&buf);
                    out[i] = gx.hypot(gy);
                    continue;
                }
            }
            out[i] = red.reduce(&buf);
        }
        Ok(out)
    }

    /// Particle positions to draw a contour at, `(N, 2)`: the initial
    /// coordinates, or (`deformed`) the positions at the last increment of
    /// `red`'s window.
    pub fn contour_coordinates(&self, deformed: bool, red: &ContourReduction) -> Result<Array2<f64>, Error> {
        if !deformed {
            return Ok(self.initial_coordinates.clone());
        }
        let (_, stop) = red.range(self.inc_no())?;
        let mut out = Array2::<f64>::zeros((self.particles.len(), 2));
        for (i, p) in self.particles.iter().enumerate() {
            out[[i, 0]] = p.coordinates[[stop - 1, 0]];
            out[[i, 1]] = p.coordinates[[stop - 1, 1]];
        }
        Ok(out)
    }

    /// Delaunay triangulation of the initial coordinates, `(M, 3)` particle
    /// indices, with every triangle that leaves the field's [`FieldRegion`]
    /// removed (its centroid or any edge midpoint lies outside the boundary
    /// or inside an exclusion). Without a stored region (pre-0x08 files) the
    /// full convex hull is returned.
    ///
    /// Always built in the reference configuration, so the same triangles
    /// stay valid when drawn at deformed positions.
    pub fn contour_triangles(&self) -> Array2<usize> {
        let tris = delaunay(self.initial_coordinates.view());
        let Some(region) = &self.region else { return tris };
        let c = &self.initial_coordinates;
        let pt = |i: usize| [c[[i, 0]], c[[i, 1]]];
        let mid = |a: [f64; 2], b: [f64; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let kept: Vec<usize> = tris
            .rows()
            .into_iter()
            .filter(|t| {
                let (a, b, d) = (pt(t[0]), pt(t[1]), pt(t[2]));
                let centroid = [(a[0] + b[0] + d[0]) / 3.0, (a[1] + b[1] + d[1]) / 3.0];
                [centroid, mid(a, b), mid(b, d), mid(d, a)].iter().all(|&p| region.contains(p))
            })
            .flat_map(|t| [t[0], t[1], t[2]])
            .collect();
        Array2::from_shape_vec((kept.len() / 3, 3), kept).expect("3 indices pushed per triangle")
    }
}

/// Unconstrained Delaunay triangulation of `points` `(N, 2)`, as `(M, 3)`
/// indices into `points`. Duplicate points share the first one's index;
/// non-finite points are skipped.
pub fn delaunay(points: ArrayView2<f64>) -> Array2<usize> {
    use spade::{DelaunayTriangulation, Point2, Triangulation};
    let mut dt = DelaunayTriangulation::<Point2<f64>>::new();
    // spade vertex index -> input row (duplicates are merged into one vertex).
    let mut original = Vec::with_capacity(points.nrows());
    for i in 0..points.nrows() {
        let (x, y) = (points[[i, 0]], points[[i, 1]]);
        if !(x.is_finite() && y.is_finite()) {
            continue;
        }
        if let Ok(h) = dt.insert(Point2::new(x, y)) {
            if h.index() == original.len() {
                original.push(i);
            }
        }
    }
    let idx: Vec<usize> = dt
        .inner_faces()
        .flat_map(|f| f.vertices().map(|v| original[v.fix().index()]))
        .collect();
    Array2::from_shape_vec((idx.len() / 3, 3), idx).expect("3 vertices per face")
}

// ---------------------------------------------------------------------------
// Unit tests (Phase 6 — updated for new API)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mesh::compute_centroids, particle::MeshlessParams, sequence::SequenceSolution,
    };
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
            solve_config: None,
            seed: None,
            template_shape: None,
            template_sizes: None,
            zonal_masking: None,
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
            options: None,
            border: 0,
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
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
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
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
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
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
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
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
        assert_eq!(field.solution().unwrap().vol_totals.len(), 3);
    }

    #[test]
    fn test_field_solve_vol_totals_pure_translation() {
        let (nodes, elems) = unit_square_mesh();
        let disps = pure_translation_disps(0.2, 0.0);
        let (_, vols) = distribute_particles(&nodes, &elems, 1.0);
        let total_initial = vols.sum();
        let mut field = make_field_from_seq(single_mesh_seq(&disps), true);
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
        let sol = field.solution().unwrap();
        assert!((sol.vol_totals[0] - total_initial).abs() < 1e-10, "vt[0]={}", sol.vol_totals[0]);
        assert!((sol.vol_totals[1] - total_initial).abs() < 1e-10, "vt[1]={}", sol.vol_totals[1]);
    }

    #[test]
    fn test_field_solve_sets_solved_flag() {
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field_from_seq(single_mesh_seq(&disps), true);
        assert!(!field.solved());
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
        assert!(field.solved());
    }

    #[test]
    fn test_field_solve_ref_update_register() {
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field_from_seq(two_mesh_seq(&disps, vec![false, true]), true);
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
        assert_eq!(field.solution().unwrap().reference_update_register, vec![1]);
    }

    #[test]
    fn test_field_solve_particle_count() {
        let disps = Array2::<f64>::zeros((4, 2));
        let mut field = make_field_from_seq(single_mesh_seq(&disps), true);
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
        assert_eq!(field.solution().unwrap().particles.len(), 2);
    }

    #[test]
    fn test_field_solve_x_stretch_eps_xx() {
        let nodes = array![[0.0, 0.0_f64], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let elems = array![[0usize, 1, 2], [1, 3, 2]];
        let disps = array![[0.0, 0.0_f64], [0.1, 0.0], [0.0, 0.0], [0.1, 0.0]];
        let seq = make_seq(vec![make_mesh_sol(&nodes, &elems, &disps)], vec![false]);
        let mut field = make_field_from_seq(seq, true);
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
        let sol = field.solution().unwrap();
        for p in &sol.particles {
            assert!((p.incs[[1, 2]] - 0.1).abs() < 1e-8, "incs[1,2]={}", p.incs[[1, 2]]);
        }
    }

    #[test]
    fn test_field_solve_two_increments() {
        let disps = pure_translation_disps(0.1, 0.0);
        let mut field = make_field_from_seq(two_mesh_seq(&disps, vec![false, false]), true);
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
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
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
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
            options: None,
            border: 0,
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
            options: None,
            border: 0,
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
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();

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
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();

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
        field_mem.solve(0.0, true, None, StrainMethod::Mesh).unwrap();
        let sol_mem = field_mem.solution().unwrap();

        // Saved-by-reference solve
        let (seq_sbr, paths) =
            make_saved_by_ref_field_seq("match_sbr", vec![ms], vec![false]);
        let mut field_sbr = make_field_from_seq(Arc::clone(&seq_sbr), true);
        field_sbr.solve(0.0, true, None, StrainMethod::Mesh).unwrap();
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

    // -----------------------------------------------------------------------
    // Calibration integration
    // -----------------------------------------------------------------------

    /// A camera model whose net image<->object mapping is exactly the identity.
    /// See the identical helper in `particle.rs`'s tests for the derivation —
    /// `extmat` cannot literally be the identity matrix (divides by zero at
    /// z=0), so the depth (extmat[2,3]) and focal length must match instead.
    fn identity_calibration() -> CalibrationParams {
        let intmat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut extmat = Array2::<f64>::eye(4);
        extmat[[2, 3]] = 1.0;
        CalibrationParams::new(intmat, extmat, [0.0; 5]).unwrap()
    }

    /// Same as `identity_calibration` but with the focal length doubled, so
    /// i2o(imgpt) == 0.5 * imgpt for every point.
    fn half_scale_calibration() -> CalibrationParams {
        let intmat = array![[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let mut extmat = Array2::<f64>::eye(4);
        extmat[[2, 3]] = 1.0;
        CalibrationParams::new(intmat, extmat, [0.0; 5]).unwrap()
    }

    #[test]
    fn test_field_calibrated_flag_true() {
        let disps = pure_translation_disps(0.5, 0.1);
        let mut field = make_field_from_seq(single_mesh_seq(&disps), true);
        let cal = identity_calibration();
        field.solve(0.0, true, Some(&cal), StrainMethod::Mesh).unwrap();
        assert!(field.solution().unwrap().calibrated);
    }

    #[test]
    fn test_field_calibrated_flag_false() {
        let disps = pure_translation_disps(0.5, 0.1);
        let mut field = make_field_from_seq(single_mesh_seq(&disps), true);
        field.solve(0.0, true, None, StrainMethod::Mesh).unwrap();
        assert!(!field.solution().unwrap().calibrated);
    }

    #[test]
    fn test_field_calibrate_pure_scale() {
        // Same reasoning as Particle's pure_scale test: each particle's
        // displacement increment from its (also-scaled) reference should be
        // exactly 0.5x the uncalibrated pixel-space increment.
        let disps = pure_translation_disps(0.5, 0.1);

        let mut field_uncal = make_field_from_seq(single_mesh_seq(&disps), true);
        field_uncal.solve(0.0, true, None, StrainMethod::Mesh).unwrap();
        let sol_uncal = field_uncal.solution().unwrap();

        let cal = half_scale_calibration();
        let mut field_cal = make_field_from_seq(single_mesh_seq(&disps), true);
        field_cal.solve(0.0, true, Some(&cal), StrainMethod::Mesh).unwrap();
        let sol_cal = field_cal.solution().unwrap();

        for (p_uncal, p_cal) in sol_uncal.particles.iter().zip(sol_cal.particles.iter()) {
            let uncal_disp_x = p_uncal.coordinates[[1, 0]] - p_uncal.coordinates[[0, 0]];
            let cal_disp_x = p_cal.coordinates[[1, 0]] - p_cal.coordinates[[0, 0]];
            assert!(
                (cal_disp_x - 0.5 * uncal_disp_x).abs() < 1e-9,
                "{} vs {}", cal_disp_x, 0.5 * uncal_disp_x
            );
        }
    }

    // -----------------------------------------------------------------------
    // principal_strains / gamma_max_grad -- stored at solve time, not
    // recomputed at plot time.
    // -----------------------------------------------------------------------

    /// 4 triangles meeting at a centre node, corners of a 2x2 square -- same
    /// "cross" topology as tests/python/field/test_field.py's own
    /// NODES_GRID/ELEMS_GRID fixture. Gives 4 non-collinear particle
    /// centroids, the minimum needed for a well-conditioned 2D affine fit
    /// (unlike unit_square_mesh's 2 particles, which can only determine a
    /// 1D directional derivative).
    fn cross_mesh() -> (Array2<f64>, Array2<usize>) {
        let nodes = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]];
        let elems = array![[0usize, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]];
        (nodes, elems)
    }

    #[test]
    fn test_field_solve_principal_strains_matches_direct_call() {
        let (nodes, elems) = unit_square_mesh();
        let disps = array![[0.0, 0.0_f64], [0.1, 0.0], [0.0, 0.05], [0.1, 0.05]];
        let seq = make_seq(vec![make_mesh_sol(&nodes, &elems, &disps)], vec![false]);
        let mut field = make_field_from_seq(seq, true);
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
        let sol = field.solution().unwrap();
        for p in &sol.particles {
            let expected = crate::particle::principal_strains(&p.strains);
            let got = p.principal_strains.as_ref().expect("principal_strains should be populated");
            assert_eq!(got.shape(), expected.shape());
            for i in 0..expected.nrows() {
                for j in 0..4 {
                    assert!(
                        (got[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                        "row {i} col {j}: got {} expected {}", got[[i, j]], expected[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_field_solve_gamma_max_grad_none_when_mesh() {
        let (nodes, elems) = cross_mesh();
        let n = nodes.nrows();
        let disps = Array2::<f64>::zeros((n, 2));
        let seq = make_seq(vec![make_mesh_sol(&nodes, &elems, &disps)], vec![false]);
        let mut field = make_field_from_seq(seq, true);
        field.solve(1.0, true, None, StrainMethod::Mesh).unwrap();
        let sol = field.solution().unwrap();
        assert!(!sol.particles.is_empty());
        for p in &sol.particles {
            assert!(
                p.gamma_max_grad.is_none(),
                "gamma_max_grad must stay None under StrainMethod::Mesh -- no independent \
                 fallback radius to drive it"
            );
        }
    }

    #[test]
    fn test_field_solve_gamma_max_grad_populated_and_nontrivial_when_meshless() {
        let (nodes, elems) = cross_mesh();
        let centroids = compute_centroids(&nodes, &elems);
        let n = nodes.nrows();
        // v = 0.01 * x^2 at each node: NOT affine, so a robust *local*
        // affine fit (StrainMethod::Meshless) picks up a position-dependent
        // dv/dx (steeper toward +x), hence a position-dependent gamma_max,
        // hence a genuinely non-zero spatial gradient to recover. A
        // globally-affine field would give every local fit the same slope
        // and the gradient would be exactly zero everywhere -- not a useful
        // wiring check.
        let mut p = Array2::<f64>::zeros((n, 6));
        for i in 0..n {
            let x = nodes[[i, 0]];
            p[[i, 1]] = 0.01 * x * x;
        }
        let mesh_sol = crate::mesh::MeshSolution {
            nodes: nodes.clone(),
            elements: elems.clone(),
            boundary: (0..n).collect(),
            exclusions: vec![],
            centroids,
            areas: Array1::from_vec(vec![0.5; elems.nrows()]),
            warps: Array2::zeros((elems.nrows(), 12)),
            displacements: Array2::zeros((n, 2)),
            c_zncc: Array1::ones(n),
            p,
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
        };
        let seq = make_seq(vec![mesh_sol], vec![false]);
        let mut field = make_field_from_seq(seq, true);
        let params = MeshlessParams {
            radius: 10.0, quality_gate: None, min_neighbours: 3, max_iterations: 5, tukey_c: 4.685,
            zone_aware: false,
        };
        field.solve(0.0, true, None, StrainMethod::Meshless(params)).unwrap();
        let sol = field.solution().unwrap();
        assert_eq!(sol.particles.len(), 4);

        let mut any_nonzero = false;
        for p in &sol.particles {
            let g = p.gamma_max_grad.as_ref()
                .expect("gamma_max_grad should be populated under StrainMethod::Meshless");
            assert_eq!(g.nrows(), p.coordinates.nrows());
            assert_eq!(g.ncols(), 2);
            if g[[1, 0]].abs() > 1e-8 || g[[1, 1]].abs() > 1e-8 {
                any_nonzero = true;
            }
        }
        assert!(
            any_nonzero,
            "expected at least one particle to see a non-trivial gamma_max gradient from a \
             spatially-varying field"
        );
    }

    #[test]
    fn grid_particles_cell_centres_inside_boundary_outside_exclusions() {
        let boundary = ndarray::array![[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]];
        let hole = ndarray::array![[15.0, 15.0], [25.0, 15.0], [25.0, 25.0], [15.0, 25.0]];
        let (c, v) = grid_particles(boundary.view(), &[], 10.0, 2.0);
        assert_eq!(c.nrows(), 16);
        assert_eq!((c[[0, 0]], c[[0, 1]]), (5.0, 5.0));
        assert!(v.iter().all(|&x| x == 200.0));
        // The 10x10 hole contains no cell centre (centres at 5, 15, 25, 35)
        // except exactly on its edge — shrink it to remove one interior centre.
        let (c2, _) = grid_particles(boundary.view(), &[hole.view()], 20.0, 1.0);
        let (c3, _) = grid_particles(boundary.view(), &[], 20.0, 1.0);
        assert_eq!(c3.nrows(), 4);
        assert_eq!(c2.nrows(), 4, "20 px grid centres (10, 30) all lie outside the hole");
        let small_hole = ndarray::array![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let (c4, _) = grid_particles(boundary.view(), &[small_hole.view()], 10.0, 1.0);
        assert_eq!(c4.nrows(), 15, "the (5, 5) centre is excluded");
        let (c5, _) = grid_particles(boundary.view(), &[], 0.0, 1.0);
        assert_eq!(c5.nrows(), 0);
    }

    // -----------------------------------------------------------------------
    // Contour data
    // -----------------------------------------------------------------------

    /// One particle whose `warps[:, 0]`, `strains[:, 0]` and both
    /// `gamma_max_grad` columns follow `series`, at fixed position `xy`
    /// moving by `+1` in x per increment.
    fn series_particle(series: &[f64], xy: [f64; 2]) -> Arc<ParticleSolution> {
        let n = series.len();
        let col = Array1::from_vec(series.to_vec());
        let mut warps = Array2::<f64>::zeros((n, 6));
        warps.column_mut(0).assign(&col);
        warps.column_mut(1).assign(&col);
        let mut strains = Array2::<f64>::zeros((n, 6));
        strains.column_mut(0).assign(&col);
        let mut grad = Array2::<f64>::zeros((n, 2));
        grad.column_mut(0).assign(&col);
        grad.column_mut(1).assign(&col);
        let coordinates = Array2::from_shape_fn((n, 2), |(m, j)| xy[j] + if j == 0 { m as f64 } else { 0.0 });
        Arc::new(ParticleSolution {
            coordinates,
            warps,
            incs: Array2::zeros((n, 6)),
            volumes: Array1::ones(n),
            principal_strains: Some(crate::particle::principal_strains(&strains)),
            strains,
            strain_incs: Array2::zeros((n - 1, 6)),
            vol_strains: col,
            reference_update_register: vec![],
            image_0_path: None,
            calibrated: false,
            config: None,
            gamma_max_grad: Some(grad),
        })
    }

    fn field_sol(particles: Vec<Arc<ParticleSolution>>, region: Option<FieldRegion>) -> FieldSolution {
        let initial_coordinates = Array2::from_shape_fn((particles.len(), 2), |(i, j)| particles[i].coordinates[[0, j]]);
        let inc_no = particles[0].coordinates.nrows();
        FieldSolution {
            particles,
            initial_coordinates,
            vol_totals: Array1::ones(inc_no),
            reference_update_register: vec![],
            image_0_path: None,
            calibrated: false,
            depth: 1.0,
            track: true,
            region,
        }
    }

    fn red(window: Option<(usize, usize)>, dt: Option<f64>, absolute: bool) -> ContourReduction {
        ContourReduction { window, dt, absolute }
    }

    #[test]
    fn contour_reduction_matches_python_reduce_series() {
        let sol = field_sol(vec![series_particle(&[0.0, 2.0, 1.0, 4.0], [0.0, 0.0])], None);
        let v = |r: ContourReduction| sol.contour_values(FieldQuantity::U, &r).unwrap()[0];
        assert_eq!(v(red(None, None, false)), 4.0, "last - first");
        assert_eq!(v(red(None, None, true)), 2.0 + 1.0 + 3.0, "sum |diff|");
        assert_eq!(v(red(Some((1, 3)), None, false)), -1.0, "window is half-open");
        assert_eq!(v(red(Some((1, 2)), None, false)), 2.0, "single value -> value");
        assert_eq!(v(red(Some((1, 2)), None, true)), 0.0, "single value, absolute -> 0");
        assert!((v(red(None, Some(0.5), false)) - 4.0 / (4.0 * 0.5)).abs() < 1e-15, "mean rate");
        assert!((v(red(None, Some(0.5), true)) - 2.0).abs() < 1e-15, "dt ignores absolute");
        assert_eq!(v(red(Some((2, 3)), Some(0.5), false)), 0.0, "single value rate -> 0");
        for bad in [(2, 2), (3, 1), (0, 5)] {
            assert!(sol.contour_values(FieldQuantity::U, &red(Some(bad), None, false)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn contour_values_quantities() {
        let sol = field_sol(vec![series_particle(&[0.0, 3.0], [0.0, 0.0])], None);
        let all = red(None, None, false);
        let v = |q| sol.contour_values(q, &all).unwrap()[0];
        assert_eq!(v(FieldQuantity::U), 3.0);
        assert_eq!(v(FieldQuantity::V), 3.0);
        assert!((v(FieldQuantity::R) - 18f64.sqrt()).abs() < 1e-12);
        assert_eq!(v(FieldQuantity::EpXx), 3.0);
        assert_eq!(v(FieldQuantity::EpYy), 0.0);
        assert_eq!(v(FieldQuantity::EpVol), 3.0);
        // Uniaxial ep_xx = 3: ep1 = 3, ep2 = 0, gamma_max = 3.
        assert!((v(FieldQuantity::Ep1) - 3.0).abs() < 1e-12);
        assert!(v(FieldQuantity::Ep2).abs() < 1e-12);
        assert!((v(FieldQuantity::GammaMax) - 3.0).abs() < 1e-12);
        // Each gradient component reduces to 3 -> magnitude 3*sqrt(2).
        assert!((v(FieldQuantity::GammaMaxGrad) - 3.0 * 2f64.sqrt()).abs() < 1e-12);
        for q in FieldQuantity::ALL {
            assert_eq!(q.name().parse::<FieldQuantity>().unwrap(), q);
        }
        assert!("bogus".parse::<FieldQuantity>().is_err());
    }

    #[test]
    fn contour_values_missing_gradient_errors() {
        let mut p = (*series_particle(&[0.0, 1.0], [0.0, 0.0])).clone();
        p.gamma_max_grad = None;
        let sol = field_sol(vec![Arc::new(p)], None);
        assert!(sol.contour_values(FieldQuantity::GammaMaxGrad, &ContourReduction::default()).is_err());
        assert!(sol.contour_values(FieldQuantity::U, &ContourReduction::default()).is_ok());
    }

    #[test]
    fn contour_coordinates_reference_and_deformed() {
        let sol = field_sol(vec![series_particle(&[0.0, 0.0, 0.0], [5.0, 7.0])], None);
        let r = sol.contour_coordinates(false, &ContourReduction::default()).unwrap();
        assert_eq!((r[[0, 0]], r[[0, 1]]), (5.0, 7.0));
        let d = sol.contour_coordinates(true, &ContourReduction::default()).unwrap();
        assert_eq!((d[[0, 0]], d[[0, 1]]), (7.0, 7.0), "last increment");
        let d = sol.contour_coordinates(true, &red(Some((0, 2)), None, false)).unwrap();
        assert_eq!((d[[0, 0]], d[[0, 1]]), (6.0, 7.0), "window end is exclusive");
    }

    /// Grid of particles at `x, y in {0.5, 1.5, ..., n-0.5}`.
    fn grid_field(n: usize, region: Option<FieldRegion>) -> FieldSolution {
        let particles = (0..n * n)
            .map(|k| series_particle(&[0.0, 0.0], [(k / n) as f64 + 0.5, (k % n) as f64 + 0.5]))
            .collect();
        field_sol(particles, region)
    }

    fn square(lo: f64, hi: f64) -> Array2<f64> {
        ndarray::array![[lo, lo], [hi, lo], [hi, hi], [lo, hi]]
    }

    fn tri_centroids(sol: &FieldSolution, tris: &Array2<usize>) -> Vec<[f64; 2]> {
        let c = &sol.initial_coordinates;
        tris.rows().into_iter().map(|t| {
            [(0..3).map(|k| c[[t[k], 0]]).sum::<f64>() / 3.0, (0..3).map(|k| c[[t[k], 1]]).sum::<f64>() / 3.0]
        }).collect()
    }

    #[test]
    fn contour_triangles_without_region_cover_hull() {
        let sol = grid_field(4, None);
        assert_eq!(sol.contour_triangles().nrows(), 18, "3x3 cells, 2 triangles each");
    }

    #[test]
    fn contour_triangles_drop_exclusion() {
        let region = FieldRegion { boundary: square(0.0, 4.0), exclusions: vec![square(1.6, 2.4)] };
        let sol = grid_field(4, Some(region));
        let tris = sol.contour_triangles();
        assert_eq!(tris.nrows(), 16, "the centre cell's 2 triangles are removed");
        for c in tri_centroids(&sol, &tris) {
            assert!(!(1.5 < c[0] && c[0] < 2.5 && 1.5 < c[1] && c[1] < 2.5), "{c:?}");
        }
    }

    #[test]
    fn contour_triangles_drop_concave_notch() {
        // L-shape: [0,4]^2 minus the top-right [2,4]x[2,4] quadrant; only the
        // 12 grid points inside the L are particles, but their hull spans the notch.
        let boundary = ndarray::array![[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0]];
        let particles: Vec<_> = (0..16)
            .map(|k| [(k / 4) as f64 + 0.5, (k % 4) as f64 + 0.5])
            .filter(|p| !(p[0] > 2.0 && p[1] > 2.0))
            .map(|p| series_particle(&[0.0, 0.0], p))
            .collect();
        let unmasked = field_sol(particles.clone(), None).contour_triangles().nrows();
        let region = FieldRegion { boundary: boundary.clone(), exclusions: vec![] };
        let sol = field_sol(particles, Some(region.clone()));
        let tris = sol.contour_triangles();
        assert!(tris.nrows() < unmasked, "hull triangles across the notch are removed");
        assert_eq!(tris.nrows(), 10, "5 cells of the L, 2 triangles each");
        for c in tri_centroids(&sol, &tris) {
            assert!(region.contains(c), "{c:?}");
        }
    }

    #[test]
    fn delaunay_merges_duplicates_and_skips_nan() {
        let pts = ndarray::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [f64::NAN, 0.0]];
        let t = delaunay(pts.view());
        assert_eq!(t.nrows(), 1);
        let mut idx: Vec<_> = t.row(0).to_vec();
        idx.sort();
        assert_eq!(idx, vec![0, 1, 2]);
    }

    #[test]
    fn solve_stores_region() {
        let seq = single_mesh_seq(&Array2::<f64>::zeros((4, 2)));
        let mut f = make_field_from_seq(Arc::clone(&seq), true);
        f.solve(0.0, true, None, StrainMethod::Mesh).unwrap();
        let r = f.solution().unwrap().region.as_ref().expect("taken from the first mesh");
        assert_eq!(r.boundary.nrows(), 4);

        let dist = FieldDistribution::Grid {
            boundary_nodes: square(0.0, 1.0),
            exclusion_nodes: vec![],
            spacing: 0.5,
        };
        let mut g = Field::new(seq, dist, true, 1.0).unwrap();
        g.solve(0.0, true, None, StrainMethod::Mesh).unwrap();
        assert_eq!(g.solution().unwrap().region.as_ref().unwrap().boundary, square(0.0, 1.0));
    }
}
