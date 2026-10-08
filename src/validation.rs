//! DIC validation: compare observed warp fields against applied (ground-truth)
//! warp fields and remove outlier particles.

use std::sync::Arc;

use ndarray::{s, Array1, Array2, Array3, ArrayView3};
use serde::{Deserialize, Serialize};

use crate::{field::FieldSolution, speckle::Speckle, Error};

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub applied:  Array3<f64>,
    pub observed: Array3<f64>,
}

// ---------------------------------------------------------------------------
// ValidationFieldData / ValidationSolution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFieldData {
    pub result:       ValidationResult,
    pub coordinates:  Array3<f64>,         // (n_frames+1, n_particles, 2)
    pub image_0_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSolution {
    pub fields: Vec<ValidationFieldData>,
    pub labels: Vec<String>,
    pub pm:     Array2<f64>,   // (image_no, 12)
    pub mult:   Array1<f64>,   // (image_no,)
}

// ---------------------------------------------------------------------------
// Validation struct
// ---------------------------------------------------------------------------

pub struct Validation {
    speckle:         Arc<Speckle>,
    field_solutions: Vec<FieldSolution>,
    labels:          Vec<String>,
}

impl Validation {
    pub fn new(
        speckle: Arc<Speckle>,
        field_solutions: Vec<FieldSolution>,
        labels: Vec<String>,
    ) -> Result<Self, Error> {
        if field_solutions.len() != labels.len() {
            return Err(Error::InvalidInput(format!(
                "field_solutions length {} != labels length {}",
                field_solutions.len(), labels.len()
            )));
        }
        Ok(Self { speckle, field_solutions, labels })
    }

    pub fn solve(&self, cumulative: bool, skim: Option<usize>) -> Result<ValidationSolution, Error> {
        let n_images = self.speckle.image_no;
        let n_frames = n_images.saturating_sub(1);

        // Build pm (image_no, 12) and mult (image_no,)
        let mut pm   = Array2::<f64>::zeros((n_images, 12));
        let mut mult = Array1::<f64>::zeros(n_images);
        for i in 0..n_images {
            // mult tracks the progression variable (deformation scale or noise scale);
            // pm tracks the actual deformation state at each image.
            mult[i] = self.speckle.scale_mult_at(i);
            let dm = self.speckle.deform_mult_at(i);
            for k in 0..12usize {
                pm[[i, k]] = self.speckle.comp[k] * dm;
            }
        }

        let mut fields = Vec::with_capacity(self.field_solutions.len());

        for field_sol in &self.field_solutions {
            let n_particles = field_sol.particles.len();
            let n_warp = if n_particles > 0 {
                field_sol.particles[0].warps.ncols()
            } else {
                12
            };

            let mut applied  = Array3::<f64>::zeros((n_frames, n_particles, 12));
            let mut observed = Array3::<f64>::zeros((n_frames, n_particles, 12));

            // Initial particle coordinates for warp evaluation
            let init_coords = field_sol.initial_coordinates.view();

            // Applied: ground-truth warp from Speckle
            if cumulative {
                for i in 1..=n_frames {
                    let w = self.speckle.warp(i, init_coords)?;
                    applied.slice_mut(s![i - 1, .., ..]).assign(&w);
                }
            } else {
                // warp(0) = zero-warp (reference image)
                let mut prev = self.speckle.warp(0, init_coords)?;
                for i in 1..=n_frames {
                    let curr = self.speckle.warp(i, init_coords)?;
                    let diff = &curr - &prev;
                    applied.slice_mut(s![i - 1, .., ..]).assign(&diff);
                    prev = curr;
                }
            }

            // Observed: copy particle warp paths; zero-pad if order-1 (6-wide)
            // cumulative=true  → observed[m] = warps[m+1]             (cumulative from image 0)
            // cumulative=false → observed[m] = warps[m+1] - warps[m]  (frame-to-frame increment)
            let copy_cols = n_warp.min(12);
            for j in 0..n_particles {
                let all_warps = field_sol.particles[j].warps.view();
                let max_rows = all_warps.nrows().saturating_sub(1).min(n_frames);
                for row in 0..max_rows {
                    for col in 0..copy_cols {
                        observed[[row, j, col]] = if cumulative {
                            all_warps[[row + 1, col]]
                        } else {
                            all_warps[[row + 1, col]] - all_warps[[row, col]]
                        };
                    }
                }
            }

            // Skim outliers
            let result = if let Some(k) = skim {
                if k > 0 {
                    anomalies(applied.view(), observed.view(), k)?
                } else {
                    ValidationResult { applied, observed }
                }
            } else {
                ValidationResult { applied, observed }
            };

            // Coordinates: (n_frames+1, n_particles, 2)
            let mut coordinates = Array3::<f64>::zeros((n_frames + 1, n_particles, 2));
            for j in 0..n_particles {
                let p_coords = &field_sol.particles[j].coordinates;
                let rows = p_coords.nrows().min(n_frames + 1);
                for row in 0..rows {
                    coordinates[[row, j, 0]] = p_coords[[row, 0]];
                    coordinates[[row, j, 1]] = p_coords[[row, 1]];
                }
            }

            let image_0_path = field_sol.image_0_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned());

            fields.push(ValidationFieldData { result, coordinates, image_0_path });
        }

        Ok(ValidationSolution {
            fields,
            labels: self.labels.clone(),
            pm,
            mult,
        })
    }
}

// ---------------------------------------------------------------------------
// anomalies — pure-array outlier removal
// ---------------------------------------------------------------------------

pub fn anomalies(
    applied: ArrayView3<f64>,
    observed: ArrayView3<f64>,
    skim: usize,
) -> Result<ValidationResult, Error> {
    let (n_frames, n_particles, n_warp) = {
        let s = applied.shape();
        (s[0], s[1], s[2])
    };
    if observed.shape() != applied.shape() {
        return Err(Error::InvalidInput(format!(
            "applied shape {:?} != observed shape {:?}",
            applied.shape(),
            observed.shape()
        )));
    }
    if skim >= n_particles {
        return Err(Error::InvalidInput(format!(
            "skim ({skim}) must be less than n_particles ({n_particles})"
        )));
    }

    let n_kept = n_particles - skim;
    let mut out_applied  = Array3::<f64>::zeros((n_frames, n_kept, n_warp));
    let mut out_observed = Array3::<f64>::zeros((n_frames, n_kept, n_warp));

    for i in 0..n_frames {
        let mut errors: Vec<(f64, usize)> = (0..n_particles)
            .map(|j| {
                let du = applied[[i, j, 0]] - observed[[i, j, 0]];
                let dv = applied[[i, j, 1]] - observed[[i, j, 1]];
                ((du * du + dv * dv).sqrt(), j)
            })
            .collect();

        errors.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let outlier_indices: std::collections::HashSet<usize> =
            errors[..skim].iter().map(|&(_, j)| j).collect();

        let mut row = 0;
        for j in 0..n_particles {
            if !outlier_indices.contains(&j) {
                out_applied.slice_mut(ndarray::s![i, row, ..])
                    .assign(&applied.slice(ndarray::s![i, j, ..]));
                out_observed.slice_mut(ndarray::s![i, row, ..])
                    .assign(&observed.slice(ndarray::s![i, j, ..]));
                row += 1;
            }
        }
    }

    Ok(ValidationResult {
        applied:  out_applied,
        observed: out_observed,
    })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        field::FieldSolution,
        particle::ParticleSolution,
        speckle::{Progression, ScaleType, Speckle, WarpMode},
    };
    use ndarray::{array, Array1, Array2};

    // ---- anomalies tests (unchanged) ----

    #[test]
    fn test_anomalies_skim_zero() {
        let a = Array3::<f64>::zeros((2, 5, 12));
        let o = Array3::<f64>::zeros((2, 5, 12));
        let result = anomalies(a.view(), o.view(), 0).unwrap();
        assert_eq!(result.applied.shape(), &[2, 5, 12]);
        assert_eq!(result.observed.shape(), &[2, 5, 12]);
    }

    #[test]
    fn test_anomalies_removes_outlier() {
        let mut applied  = Array3::<f64>::zeros((1, 5, 12));
        let observed = Array3::<f64>::zeros((1, 5, 12));
        applied[[0, 3, 0]] = 10.0;
        let result = anomalies(applied.view(), observed.view(), 1).unwrap();
        assert_eq!(result.applied.shape(), &[1, 4, 12]);
        for j in 0..4 {
            assert_eq!(result.applied[[0, j, 0]], 0.0);
        }
    }

    #[test]
    fn test_anomalies_removes_top_k() {
        let mut applied  = Array3::<f64>::zeros((1, 5, 12));
        let observed = Array3::<f64>::zeros((1, 5, 12));
        applied[[0, 1, 0]] = 5.0;
        applied[[0, 4, 1]] = 8.0;
        let result = anomalies(applied.view(), observed.view(), 2).unwrap();
        assert_eq!(result.applied.shape(), &[1, 3, 12]);
        for j in 0..3 {
            assert_eq!(result.applied[[0, j, 0]], 0.0);
            assert_eq!(result.applied[[0, j, 1]], 0.0);
        }
    }

    #[test]
    fn test_anomalies_multiframe() {
        let mut applied  = Array3::<f64>::zeros((2, 4, 12));
        let observed = Array3::<f64>::zeros((2, 4, 12));
        applied[[0, 2, 0]] = 9.0;
        applied[[1, 0, 0]] = 9.0;
        let result = anomalies(applied.view(), observed.view(), 1).unwrap();
        assert_eq!(result.applied.shape(), &[2, 3, 12]);
        for j in 0..3 { assert_eq!(result.applied[[0, j, 0]], 0.0); }
        for j in 0..3 { assert_eq!(result.applied[[1, j, 0]], 0.0); }
    }

    #[test]
    fn test_anomalies_skim_too_large() {
        let a = Array3::<f64>::zeros((1, 3, 12));
        let o = Array3::<f64>::zeros((1, 3, 12));
        assert!(matches!(anomalies(a.view(), o.view(), 3), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_anomalies_shape_mismatch() {
        let a = Array3::<f64>::zeros((1, 3, 12));
        let o = Array3::<f64>::zeros((1, 4, 12));
        assert!(matches!(anomalies(a.view(), o.view(), 1), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_anomalies_values_exact() {
        let mut applied  = Array3::<f64>::zeros((1, 3, 12));
        let mut observed = Array3::<f64>::zeros((1, 3, 12));
        applied[[0, 0, 0]]  = 1.23456789012345;
        observed[[0, 0, 1]] = 9.87654321098765;
        applied[[0, 2, 0]]  = 100.0;
        let result = anomalies(applied.view(), observed.view(), 1).unwrap();
        assert!((result.applied[[0, 0, 0]] - 1.23456789012345).abs() < 1e-12);
        assert!((result.observed[[0, 0, 1]] - 9.87654321098765).abs() < 1e-12);
    }

    // ---- Validation::solve tests ----

    fn make_speckle_translation(u: f64, image_no: usize) -> Speckle {
        let mut comp = [0.0f64; 12];
        comp[0] = u;
        Speckle::new(
            "/tmp".into(), "vs".into(), ".jpg".into(),
            (200, 200),
            1.0, 10,
            false,
            Progression::Deformation,
            comp, [100.0, 100.0],
            0.0, 0.0,
            WarpMode::None,
            image_no, ScaleType::Lin,
        ).unwrap()
    }

    fn make_particle_sol(n_frames: usize, u: f64) -> ParticleSolution {
        // warps: (n_frames+1, 12); row 0 = reference (zeros), rows 1..= are cumulative
        let mut warps = Array2::<f64>::zeros((n_frames + 1, 12));
        let mut coords = Array2::<f64>::zeros((n_frames + 1, 2));
        coords[[0, 0]] = 100.0;
        coords[[0, 1]] = 100.0;
        for i in 1..=(n_frames) {
            warps[[i, 0]] = u * i as f64; // cumulative u = u*i
            coords[[i, 0]] = 100.0 + u * i as f64;
            coords[[i, 1]] = 100.0;
        }
        ParticleSolution {
            coordinates: coords,
            warps,
            incs: Array2::zeros((n_frames + 1, 12)),
            volumes: Array1::ones(n_frames + 1),
            strains: Array2::zeros((n_frames + 1, 6)),
            strain_incs: Array2::zeros((n_frames, 6)),
            vol_strains: Array1::zeros(n_frames + 1),
            reference_update_register: vec![],
            image_0_path: None,
            calibrated: false,
            config: None,
            principal_strains: None,
            gamma_max_grad: None,
        }
    }

    fn make_field_sol(n_frames: usize, u: f64, n_particles: usize) -> FieldSolution {
        let particles: Vec<Arc<ParticleSolution>> =
            (0..n_particles).map(|_| Arc::new(make_particle_sol(n_frames, u))).collect();
        let mut init_coords = Array2::<f64>::zeros((n_particles, 2));
        for j in 0..n_particles {
            init_coords[[j, 0]] = 100.0;
            init_coords[[j, 1]] = 100.0;
        }
        FieldSolution {
            particles,
            initial_coordinates: init_coords,
            vol_totals: Array1::ones(n_frames + 1),
            reference_update_register: vec![],
            image_0_path: None,
            calibrated: false,
            depth: 1.0,
            track: false,
            region: None,
        }
    }

    #[test]
    fn test_validation_solve_cumulative_applied_matches_observed() {
        // Speckle: pure u-translation, comp[0]=1.0, image_no=4 (3 frames), ScaleType::Lin
        // scale_mult_at(i) = i/3, so applied[i-1,j,0] = warp(i)[j,0] = 1.0 * (i/3)
        // Particle warps row i = cumulative u = i/3
        let image_no = 4usize;
        let n_frames = image_no - 1;
        let u = 1.0_f64;
        let speckle = make_speckle_translation(u, image_no);

        // Build ParticleSolution with warps[i,0] = u * scale_mult_at(i) for i in 1..=n_frames
        let mut particles: Vec<Arc<ParticleSolution>> = Vec::new();
        for _ in 0..2 {
            let mut warps = Array2::<f64>::zeros((n_frames + 1, 12));
            let mut coords = Array2::<f64>::zeros((n_frames + 1, 2));
            coords[[0, 0]] = 100.0; coords[[0, 1]] = 100.0;
            for i in 1..=n_frames {
                // ScaleType::Lin, image_no=4: scale_mult_at(i) = i/3
                let m = i as f64 / 3.0;
                warps[[i, 0]] = u * m;
                coords[[i, 0]] = 100.0 + u * m;
                coords[[i, 1]] = 100.0;
            }
            particles.push(Arc::new(ParticleSolution {
                coordinates: coords,
                warps,
                incs: Array2::zeros((n_frames + 1, 12)),
                volumes: Array1::ones(n_frames + 1),
                strains: Array2::zeros((n_frames + 1, 6)),
                strain_incs: Array2::zeros((n_frames, 6)),
                vol_strains: Array1::zeros(n_frames + 1),
                reference_update_register: vec![],
                image_0_path: None,
                calibrated: false,
                config: None,
                principal_strains: None,
                gamma_max_grad: None,
            }));
        }

        let mut init_coords = Array2::<f64>::zeros((2, 2));
        init_coords[[0, 0]] = 100.0; init_coords[[0, 1]] = 100.0;
        init_coords[[1, 0]] = 100.0; init_coords[[1, 1]] = 100.0;
        let field_sol = FieldSolution {
            particles,
            initial_coordinates: init_coords,
            vol_totals: Array1::ones(n_frames + 1),
            reference_update_register: vec![],
            image_0_path: None,
            calibrated: false,
            depth: 1.0,
            track: false,
            region: None,
        };

        let validation = Validation::new(
            Arc::new(speckle),
            vec![field_sol],
            vec!["test".to_string()],
        ).unwrap();

        let sol = validation.solve(true, None).unwrap();
        let fd = &sol.fields[0];

        // applied[i,j,0] should ≈ observed[i,j,0] (both = mult[i+1])
        for i in 0..n_frames {
            for j in 0..2 {
                let diff = (fd.result.applied[[i, j, 0]] - fd.result.observed[[i, j, 0]]).abs();
                assert!(diff < 1e-8, "frame {i} particle {j} diff={diff}");
            }
        }
    }

    #[test]
    fn test_validation_solve_noncumulative_applied() {
        // Non-cumulative: applied[i] = warp(i+1) - warp(i)
        // With linear mult 0..1 over 4 images: mult[i] = i/3
        // warp(i)[j,0] = comp[0] * i/3 = u * i/3
        // applied[i-1] = u*(i/3 - (i-1)/3) = u/3
        let image_no = 4usize;
        let n_frames = image_no - 1;
        let u = 3.0_f64;
        let speckle = make_speckle_translation(u, image_no);
        let field_sol = make_field_sol(n_frames, 0.0, 2);

        let validation = Validation::new(
            Arc::new(speckle),
            vec![field_sol],
            vec!["nc".to_string()],
        ).unwrap();

        let sol = validation.solve(false, None).unwrap();
        let fd = &sol.fields[0];

        // Each frame increment = u/3 = 1.0
        for i in 0..n_frames {
            for j in 0..2 {
                let a = fd.result.applied[[i, j, 0]];
                assert!((a - 1.0).abs() < 1e-8, "frame {i} applied={a}");
            }
        }
    }

    #[test]
    fn test_validation_solve_noncumulative_observed_is_incremental() {
        // cumulative=false: observed[m] = warps[m+1] - warps[m] (incremental),
        // not warps[m+1] (cumulative). For a perfect-DIC simulation, applied ≈ observed.
        // With linear speckle u=3, 4 images: applied[m,j,0] = u/3 = 1.0 per frame.
        // make_particle_sol sets warps[i,0] = u*i so observed[m,j,0] = u*1 = u (WRONG old code)
        // or u*(m+1) - u*m = u/n … wait — make_particle_sol uses u=1.0 → step=1.0 per frame.
        // Use u=1.0 to match speckle increment exactly: speckle step = 1/3, particle step = 1.
        // Instead, set up where particle step matches speckle step.
        let image_no = 4usize;
        let n_frames = image_no - 1; // 3
        let u = 3.0_f64; // speckle: total displacement = u, step = u/(image_no-1) = 1.0
        let speckle = make_speckle_translation(u, image_no);
        // particle warps[i,0] = 1.0 * i (cumulative 1.0 per frame, matching speckle)
        let field_sol = make_field_sol(n_frames, 1.0, 2);

        let validation = Validation::new(
            Arc::new(speckle),
            vec![field_sol],
            vec!["nc_obs".to_string()],
        ).unwrap();

        let sol = validation.solve(false, None).unwrap();
        let fd = &sol.fields[0];

        // observed[m,j,0] must be incremental: warps[m+1,0] - warps[m,0] = 1.0
        for i in 0..n_frames {
            for j in 0..2 {
                let obs = fd.result.observed[[i, j, 0]];
                assert!((obs - 1.0).abs() < 1e-8,
                    "frame {i} particle {j} observed={obs}, expected 1.0 (incremental)");
                // applied and observed should match (near-zero error)
                let app = fd.result.applied[[i, j, 0]];
                assert!((app - obs).abs() < 1e-8,
                    "frame {i} particle {j} applied={app} != observed={obs}");
            }
        }
    }

    #[test]
    fn test_validation_solve_solution_shapes() {
        let image_no = 3usize;
        let n_frames = image_no - 1;
        let speckle = make_speckle_translation(1.0, image_no);
        let field_sol = make_field_sol(n_frames, 0.5, 4);

        let validation = Validation::new(
            Arc::new(speckle),
            vec![field_sol],
            vec!["shapes".to_string()],
        ).unwrap();

        let sol = validation.solve(true, None).unwrap();
        assert_eq!(sol.pm.shape(), &[image_no, 12]);
        assert_eq!(sol.mult.shape(), &[image_no]);
        let fd = &sol.fields[0];
        assert_eq!(fd.result.applied.shape(), &[n_frames, 4, 12]);
        assert_eq!(fd.result.observed.shape(), &[n_frames, 4, 12]);
        assert_eq!(fd.coordinates.shape(), &[n_frames + 1, 4, 2]);
    }
}
