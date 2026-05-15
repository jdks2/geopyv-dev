//! Camera calibration: coordinate mapping between image space and object space.
//!
//! Translates the `o2i` / `i2o` math from `geopyv/src/geopyv/calibration.py`
//! (CalibrationBase class).  The cv2 ArUco solve loop stays in Python.

use ndarray::{Array2, ArrayView2};
use nalgebra::{Matrix3, Matrix4};

use crate::Error;

// ---------------------------------------------------------------------------
// CalibrationParams
// ---------------------------------------------------------------------------

/// Camera calibration parameters for a single image plane (Z = 0 in object space).
///
/// Build with [`CalibrationParams::new`].  Matrix inverses are precomputed at
/// construction time.
#[derive(Debug, Clone)]
pub struct CalibrationParams {
    pub intmat: Array2<f64>,
    pub extmat: Array2<f64>,
    pub dist: [f64; 5],
    inv_intmat: Array2<f64>,
    inv_extmat: Array2<f64>,
}

impl CalibrationParams {
    pub fn new(
        intmat: Array2<f64>,
        extmat: Array2<f64>,
        dist: [f64; 5],
    ) -> Result<Self, Error> {
        if intmat.shape() != [3, 3] {
            return Err(Error::InvalidInput("intmat must be 3×3".to_string()));
        }
        if extmat.shape() != [4, 4] {
            return Err(Error::InvalidInput("extmat must be 4×4".to_string()));
        }

        let inv_intmat = invert3(&intmat)
            .ok_or_else(|| Error::InvalidInput("intmat is singular".to_string()))?;
        let inv_extmat = invert4(&extmat)
            .ok_or_else(|| Error::InvalidInput("extmat is singular".to_string()))?;

        Ok(Self { intmat, extmat, dist, inv_intmat, inv_extmat })
    }

    /// Map object-space coordinates to image pixels.
    ///
    /// `objpnts` — `(N, 2)` columns `[x, y]`; z = 0 is assumed.
    /// Returns `(N, 2)` pixel coordinates `[u, v]`.
    pub fn o2i(&self, objpnts: ArrayView2<f64>) -> Array2<f64> {
        let n = objpnts.nrows();
        let mut out = Array2::<f64>::zeros((n, 2));
        let [k1, k2, p1, p2, k3] = self.dist;

        for i in 0..n {
            let x = objpnts[[i, 0]];
            let y = objpnts[[i, 1]];

            // Camera frame: extmat @ [x, y, 0, 1]^T (z=0 so z-column dropped)
            let raw0 = self.extmat[[0, 0]] * x + self.extmat[[0, 1]] * y + self.extmat[[0, 3]];
            let raw1 = self.extmat[[1, 0]] * x + self.extmat[[1, 1]] * y + self.extmat[[1, 3]];
            let raw2 = self.extmat[[2, 0]] * x + self.extmat[[2, 1]] * y + self.extmat[[2, 3]];

            // Normalise by z
            let xc = raw0 / raw2;
            let yc = raw1 / raw2;

            // Radial + tangential distortion
            let r2 = xc * xc + yc * yc;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let f = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            let xpp = xc * f + 2.0 * p1 * xc * yc + p2 * (r2 + 2.0 * xc * xc);
            let ypp = yc * f + p1 * (r2 + 2.0 * yc * yc) + 2.0 * p2 * xc * yc;

            // Project: intmat @ [xpp, ypp, 1]^T
            out[[i, 0]] = self.intmat[[0, 0]] * xpp + self.intmat[[0, 1]] * ypp + self.intmat[[0, 2]];
            out[[i, 1]] = self.intmat[[1, 0]] * xpp + self.intmat[[1, 1]] * ypp + self.intmat[[1, 2]];
        }
        out
    }

    /// Map image pixels to object-space coordinates (z = 0 plane).
    ///
    /// `imgpnts` — `(N, 2)` pixel coordinates `[u, v]`.
    /// Returns `(N, 2)` object-space coordinates `[x, y]`.
    pub fn i2o(&self, imgpnts: ArrayView2<f64>) -> Array2<f64> {
        let n = imgpnts.nrows();
        let mut out = Array2::<f64>::zeros((n, 2));

        for i in 0..n {
            let u = imgpnts[[i, 0]];
            let v = imgpnts[[i, 1]];

            // Unproject to normalised image plane: inv_intmat @ [u, v, 1]^T
            let xpp = self.inv_intmat[[0, 0]] * u
                + self.inv_intmat[[0, 1]] * v
                + self.inv_intmat[[0, 2]];
            let ypp = self.inv_intmat[[1, 0]] * u
                + self.inv_intmat[[1, 1]] * v
                + self.inv_intmat[[1, 2]];

            // Undistort via Newton iteration
            let [xc, yc] = undistort_point([xpp, ypp], &self.dist);

            // Scale to z=0 plane in object space
            let s = depth_recovery([xc, yc], &self.inv_extmat);
            let xcs = xc * s;
            let ycs = yc * s;
            // z component of camera ray = s * 1.0 = s

            // Back-project: inv_extmat @ [xcs, ycs, s, 1]^T
            out[[i, 0]] = self.inv_extmat[[0, 0]] * xcs
                + self.inv_extmat[[0, 1]] * ycs
                + self.inv_extmat[[0, 2]] * s
                + self.inv_extmat[[0, 3]];
            out[[i, 1]] = self.inv_extmat[[1, 0]] * xcs
                + self.inv_extmat[[1, 1]] * ycs
                + self.inv_extmat[[1, 2]] * s
                + self.inv_extmat[[1, 3]];
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Internal math
// ---------------------------------------------------------------------------

/// Newton iteration to remove lens distortion from a normalised image point.
///
/// Solves: `distort(X_c) = x_pp` for `X_c`.
/// Initial guess `X_c = x_pp`; at most 20 iterations; tolerance 1e-9 on residual norm.
fn undistort_point(x_pp: [f64; 2], dist: &[f64; 5]) -> [f64; 2] {
    let [k1, k2, p1, p2, k3] = *dist;
    let mut x = x_pp[0];
    let mut y = x_pp[1];

    for _ in 0..20 {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let f = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        // Radial derivative: d(radial)/d(r2)
        let fp = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

        // Forward-distortion residual
        let fx = x * f + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - x_pp[0];
        let fy = y * f + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y - x_pp[1];

        if (fx * fx + fy * fy).sqrt() < 1e-9 {
            break;
        }

        // Analytical 2×2 Jacobian
        let jxx = f + 2.0 * x * x * fp + 2.0 * p1 * y + 6.0 * p2 * x;
        let jxy = 2.0 * x * y * fp + 2.0 * p1 * x + 2.0 * p2 * y;
        let jyx = 2.0 * x * y * fp + 2.0 * p2 * y + 2.0 * p1 * x;
        let jyy = f + 2.0 * y * y * fp + 2.0 * p2 * x + 6.0 * p1 * y;

        let det = jxx * jyy - jxy * jyx;
        if det.abs() < 1e-15 {
            break;
        }
        x -= (jyy * fx - jxy * fy) / det;
        y -= (jxx * fy - jyx * fx) / det;
    }

    [x, y]
}

/// Compute the depth scale that places the camera ray `[xc, yc, 1]` on the z = 0 object plane.
///
/// Port of Python `_depth_recovery`.
fn depth_recovery(x_c: [f64; 2], inv_extmat: &Array2<f64>) -> f64 {
    let denom = inv_extmat[[2, 0]] * x_c[0]
        + inv_extmat[[2, 1]] * x_c[1]
        + inv_extmat[[2, 2]];
    -inv_extmat[[2, 3]] / denom
}

// ---------------------------------------------------------------------------
// Small-matrix inverse helpers (no ndarray-linalg needed)
// ---------------------------------------------------------------------------

fn invert3(a: &Array2<f64>) -> Option<Array2<f64>> {
    let m = Matrix3::from_fn(|i, j| a[[i, j]]);
    let inv = m.try_inverse()?;
    let mut out = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            out[[i, j]] = inv[(i, j)];
        }
    }
    Some(out)
}

fn invert4(a: &Array2<f64>) -> Option<Array2<f64>> {
    let m = Matrix4::from_fn(|i, j| a[[i, j]]);
    let inv = m.try_inverse()?;
    let mut out = Array2::<f64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            out[[i, j]] = inv[(i, j)];
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Build a realistic pinhole camera with mild k1 distortion.
    fn test_params() -> CalibrationParams {
        // Typical lab-camera intrinsics (focal ≈ 2000 px, 1280×960 sensor)
        let intmat = array![
            [2000.0, 0.0, 640.0],
            [0.0, 2000.0, 480.0],
            [0.0, 0.0, 1.0]
        ];
        // Extrinsic: small rotation + 500 mm translation in z
        use std::f64::consts::PI;
        let theta = 2.0 * PI / 180.0; // 2° tilt
        let ct = theta.cos();
        let st = theta.sin();
        let extmat = array![
            [ct, -st, 0.0, 0.0],
            [st,  ct, 0.0, 0.0],
            [0.0, 0.0, 1.0, 500.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        let dist = [-0.1, 0.02, 0.0, 0.0, 0.0];
        CalibrationParams::new(intmat, extmat, dist).unwrap()
    }

    #[test]
    fn test_o2i_roundtrip() {
        let p = test_params();
        // Grid of image points in a central region
        let imgpnts = array![
            [300.0, 200.0],
            [640.0, 480.0],
            [900.0, 700.0],
            [400.0, 600.0],
        ];
        let objpnts = p.i2o(imgpnts.view());
        let recovered = p.o2i(objpnts.view());
        for i in 0..imgpnts.nrows() {
            let eu = (recovered[[i, 0]] - imgpnts[[i, 0]]).abs();
            let ev = (recovered[[i, 1]] - imgpnts[[i, 1]]).abs();
            assert!(eu < 1e-4, "row {i}: u error {eu}");
            assert!(ev < 1e-4, "row {i}: v error {ev}");
        }
    }

    #[test]
    fn test_undistort_zero_distortion() {
        let x_pp = [0.3_f64, -0.2_f64];
        let dist = [0.0; 5];
        let result = undistort_point(x_pp, &dist);
        assert!((result[0] - x_pp[0]).abs() < 1e-12, "x: {:?}", result);
        assert!((result[1] - x_pp[1]).abs() < 1e-12, "y: {:?}", result);
    }

    #[test]
    fn test_undistort_small_k1() {
        // Forward-distort a known point then verify Newton recovers the original.
        let dist = [0.1, 0.0, 0.0, 0.0, 0.0_f64];
        let [k1, k2, p1, p2, k3] = dist;
        let xc = [0.2_f64, 0.15_f64];
        let r2 = xc[0] * xc[0] + xc[1] * xc[1];
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let f = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        // Forward distortion (no tangential)
        let x_pp = [
            xc[0] * f + 2.0 * p1 * xc[0] * xc[1] + p2 * (r2 + 2.0 * xc[0] * xc[0]),
            xc[1] * f + p1 * (r2 + 2.0 * xc[1] * xc[1]) + 2.0 * p2 * xc[0] * xc[1],
        ];
        let recovered = undistort_point(x_pp, &dist);
        assert!((recovered[0] - xc[0]).abs() < 1e-9, "x: {} vs {}", recovered[0], xc[0]);
        assert!((recovered[1] - xc[1]).abs() < 1e-9, "y: {} vs {}", recovered[1], xc[1]);
    }

    #[test]
    fn test_new_rejects_wrong_shape() {
        let bad = array![[1.0, 0.0], [0.0, 1.0]]; // 2×2
        let good4 = ndarray::Array2::<f64>::eye(4);
        let dist = [0.0; 5];
        assert!(CalibrationParams::new(bad.clone(), good4.clone(), dist).is_err());
        let good3 = ndarray::Array2::<f64>::eye(3);
        assert!(CalibrationParams::new(good3, bad, dist).is_err());
    }
}
