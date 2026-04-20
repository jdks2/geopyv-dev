//! DIC validation: compare observed warp fields against applied (ground-truth)
//! warp fields and remove outlier particles.
//!
//! Translates `geopyv/src/geopyv/validation.py` — the `Validation` procedure
//! class only.  Excluded:
//! - `ValidationBase` plotting methods (`standard_error`, `mean_error`, etc.)
//! - Input-checking logic (dissolved — replaced by `Result<_, Error>`)
//! - `Validation.solve()` warp-application logic — requires `Speckle` (DEFERRED)
//!
//! # Architecture
//!
//! The core computation that CAN be translated independently is `_anomalies`:
//! given `applied` and `observed` warp arrays, remove the `skim` particles with
//! the largest Euclidean displacement error per frame.

use ndarray::{Array3, ArrayView3};

use crate::Error;

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

/// Result of a DIC validation comparison.
///
/// In `geopyv`, this is `data["applied"]` and `data["observed"]` after
/// `Validation.solve()` and `_anomalies()`.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Applied (ground-truth) warp vectors `(n_frames, n_particles, 12)`.
    pub applied: Array3<f64>,
    /// Observed (DIC-measured) warp vectors `(n_frames, n_particles, 12)`.
    pub observed: Array3<f64>,
}

// ---------------------------------------------------------------------------
// anomalies — pure-array outlier removal
// ---------------------------------------------------------------------------

/// Remove the `skim` particles with the largest displacement error from each
/// frame in `applied` and `observed`.
///
/// Replicates `Validation._anomalies`.
///
/// # Arguments
/// * `applied`  — ground-truth warp array `(n_frames, n_particles, 12)`
/// * `observed` — DIC-measured warp array `(n_frames, n_particles, 12)`
/// * `skim`     — number of outlier particles to remove per frame
///
/// # Returns
/// A [`ValidationResult`] with `applied` and `observed` of shape
/// `(n_frames, n_particles - skim, 12)`.
///
/// # Errors
/// Returns [`Error::InvalidInput`] if `skim >= n_particles` or the shapes of
/// `applied` and `observed` do not match.
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
    let mut out_applied = Array3::<f64>::zeros((n_frames, n_kept, n_warp));
    let mut out_observed = Array3::<f64>::zeros((n_frames, n_kept, n_warp));

    for i in 0..n_frames {
        // L2 error on first 2 warp components (u, v displacement).
        let mut errors: Vec<(f64, usize)> = (0..n_particles)
            .map(|j| {
                let du = applied[[i, j, 0]] - observed[[i, j, 0]];
                let dv = applied[[i, j, 1]] - observed[[i, j, 1]];
                ((du * du + dv * dv).sqrt(), j)
            })
            .collect();

        // Sort descending by error; top `skim` are outliers.
        errors.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let outlier_indices: std::collections::HashSet<usize> =
            errors[..skim].iter().map(|&(_, j)| j).collect();

        // Collect kept rows in original order.
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
        applied: out_applied,
        observed: out_observed,
    })
}

// ---------------------------------------------------------------------------
// Unit tests (Phase 2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// With skim=0, output should equal input.
    #[test]
    fn test_anomalies_skim_zero() {
        let a = Array3::<f64>::zeros((2, 5, 12));
        let o = Array3::<f64>::zeros((2, 5, 12));
        let result = anomalies(a.view(), o.view(), 0).unwrap();
        assert_eq!(result.applied.shape(), &[2, 5, 12]);
        assert_eq!(result.observed.shape(), &[2, 5, 12]);
    }

    /// Known outlier is removed; remaining particles are preserved in order.
    #[test]
    fn test_anomalies_removes_outlier() {
        // 1 frame, 5 particles, 12 warp components.
        let mut applied = Array3::<f64>::zeros((1, 5, 12));
        let mut observed = Array3::<f64>::zeros((1, 5, 12));

        // Particle 3 has a large displacement error (u=10).
        applied[[0, 3, 0]] = 10.0;
        observed[[0, 3, 0]] = 0.0;

        let result = anomalies(applied.view(), observed.view(), 1).unwrap();
        assert_eq!(result.applied.shape(), &[1, 4, 12]);

        // Particle 3 should be gone; all other rows should be zero.
        for j in 0..4 {
            assert_eq!(result.applied[[0, j, 0]], 0.0,
                "expected 0.0 at row {j} (outlier removed)");
        }
    }

    /// The two largest errors are removed, not just the first.
    #[test]
    fn test_anomalies_removes_top_k() {
        let mut applied = Array3::<f64>::zeros((1, 5, 12));
        let observed = Array3::<f64>::zeros((1, 5, 12));

        // Particles 1 and 4 have the two largest errors.
        applied[[0, 1, 0]] = 5.0;
        applied[[0, 4, 1]] = 8.0;

        let result = anomalies(applied.view(), observed.view(), 2).unwrap();
        assert_eq!(result.applied.shape(), &[1, 3, 12]);

        // Remaining rows: 0, 2, 3 — all zero.
        for j in 0..3 {
            assert_eq!(result.applied[[0, j, 0]], 0.0);
            assert_eq!(result.applied[[0, j, 1]], 0.0);
        }
    }

    /// Multi-frame: each frame gets its own independent outlier removal.
    #[test]
    fn test_anomalies_multiframe() {
        let mut applied = Array3::<f64>::zeros((2, 4, 12));
        let observed = Array3::<f64>::zeros((2, 4, 12));

        // Frame 0: particle 2 is the outlier.
        applied[[0, 2, 0]] = 9.0;
        // Frame 1: particle 0 is the outlier.
        applied[[1, 0, 0]] = 9.0;

        let result = anomalies(applied.view(), observed.view(), 1).unwrap();
        assert_eq!(result.applied.shape(), &[2, 3, 12]);

        // Frame 0, 3 remaining particles: rows 0,1,3 of original → all zero.
        for j in 0..3 {
            assert_eq!(result.applied[[0, j, 0]], 0.0);
        }
        // Frame 1, 3 remaining particles: rows 1,2,3 → all zero.
        for j in 0..3 {
            assert_eq!(result.applied[[1, j, 0]], 0.0);
        }
    }

    /// skim >= n_particles should return an error.
    #[test]
    fn test_anomalies_skim_too_large() {
        let a = Array3::<f64>::zeros((1, 3, 12));
        let o = Array3::<f64>::zeros((1, 3, 12));
        assert!(matches!(anomalies(a.view(), o.view(), 3), Err(Error::InvalidInput(_))));
        assert!(matches!(anomalies(a.view(), o.view(), 5), Err(Error::InvalidInput(_))));
    }

    /// Mismatched shapes should return an error.
    #[test]
    fn test_anomalies_shape_mismatch() {
        let a = Array3::<f64>::zeros((1, 3, 12));
        let o = Array3::<f64>::zeros((1, 4, 12));
        assert!(matches!(anomalies(a.view(), o.view(), 1), Err(Error::InvalidInput(_))));
    }

    /// f64 values are preserved exactly (Tier A, atol = 1e-12).
    #[test]
    fn test_anomalies_values_exact() {
        let mut applied = Array3::<f64>::zeros((1, 3, 12));
        let mut observed = Array3::<f64>::zeros((1, 3, 12));

        applied[[0, 0, 0]] = 1.23456789012345;
        observed[[0, 0, 1]] = 9.87654321098765;
        // Particle 2 is outlier (large error).
        applied[[0, 2, 0]] = 100.0;

        let result = anomalies(applied.view(), observed.view(), 1).unwrap();
        // Particles 0 and 1 remain.
        assert!((result.applied[[0, 0, 0]] - 1.23456789012345).abs() < 1e-12);
        assert!((result.observed[[0, 0, 1]] - 9.87654321098765).abs() < 1e-12);
    }
}
