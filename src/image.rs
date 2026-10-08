//! Image module for geopyv-dev.
//! 
//! Image loading and bi-quintic B-spline precomputation.
//!
//! # Memory note (architectural candidate)
//!
//! `qcqt` is a `(rows*6) × (cols*6)` dense f64 array. For a 1001×1001 image
//! that is ≈ 288 MB. A lazy/chunked representation would reduce peak memory
//! significantly. Propose to the user before implementing.
//!
//! # Tolerance tiers
//! - `get_c`    Tier B  rtol = 1e-8   (FFT-based computation)
//! - `get_qcqt` Tier A  atol = 1e-12  (pure matrix algebra)

use std::path::{Path, PathBuf};

use nalgebra::SMatrix;
use ndarray::{s, Array2, Array3, ArrayView2};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::Error;

type Mat6 = SMatrix<f64, 6, 6>;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Pre-processed image ready for bi-quintic B-spline DIC interpolation.
pub struct Image {
    /// Grayscale pixel intensities (height × width), values 0.0–255.0.
    pub image_gs: Array2<f64>,
    /// Pre-computed Q·C_block·Qᵀ for every pixel, **block-contiguous**:
    /// shape `(rows, cols, 36)`, the block for pixel `(i, j)` is the
    /// contiguous slice `qcqt.slice(s![i, j, ..])` laid out row-major
    /// (`[r*6 + c]`). This is the `layer_rg_plan.md` §11.5 layout — the
    /// old `(rows*6, cols*6)` array scattered a block's 6 rows `cols*6*8`
    /// bytes apart, hostile to the `bspline_eval` gather that runs
    /// `n_px * iters * nodes` times.
    pub qcqt: Array3<f64>,
    /// Border used for padding during B-spline coefficient computation. Must be ≥ 3.
    pub border: usize,
    /// File path used to load the image; `None` when constructed from an array.
    pub filepath: Option<PathBuf>,
}

impl Image {
    /// Load from a JPEG/PNG/etc file, convert to greyscale, apply Gaussian
    /// pre-filter, then pre-compute the B-spline coefficient matrix and QCQT.
    ///
    /// Greyscale conversion uses OpenCV's BGR2GRAY weights
    /// (`0.299·R + 0.587·G + 0.114·B`) so results match `geopyv`'s `cv2` output.
    /// Gaussian blur uses a 5×5 kernel with σ = 1.1 and BORDER_REFLECT_101
    /// boundary handling, matching `cv2.GaussianBlur(…, sigmaX=1.1, sigmaY=1.1)`.
    pub fn from_file(path: &Path, border: usize) -> Result<Self, Error> {
        if !path.exists() {
            return Err(Error::FileNotFound(path.display().to_string()));
        }
        let image_gs = load_grayscale(path)?;
        let mut img = Self::from_array(image_gs, border);
        img.filepath = Some(path.to_path_buf());
        Ok(img)
    }

    /// Create directly from a pre-loaded greyscale array (values 0.0–255.0).
    ///
    /// Useful for testing and for the PyO3 wrapper when cv2 has already
    /// performed loading and Gaussian pre-filtering on the Python side.
    pub fn from_array(image_gs: Array2<f64>, border: usize) -> Self {
        let c = get_c(&image_gs, border);
        let qcqt = get_qcqt(&image_gs, &c, border);
        Self { image_gs, qcqt, border, filepath: None }
    }
}

// ---------------------------------------------------------------------------
// Internal computation (pub(crate) for Rust integration tests)
// ---------------------------------------------------------------------------

/// Compute the bi-quintic B-spline coefficient array C via FFT deconvolution.
///
/// Follows `Image._get_C` from `image.py`:
/// - Pad with `border` replicated pixels on all sides.
/// - Deconvolve row-by-row, then column-by-column, using the quintic kernel
///   `k = [1/120, 13/60, 11/20, 13/60, 1/120, 0]`.
///
/// DEVIATION FROM `geopyv`: the deconvolution kernel is laid out zero-phase
/// (see `build_kernel`), not with `geopyv`'s one-sample offset. `geopyv`'s
/// layout shifts every interpolated intensity by (1, 1) px; this does not.
/// Consequently `get_c` / `get_qcqt` golden values do not match the Python
/// package.
///
/// Returns an array of shape `(rows + 2·border, cols + 2·border)`.
///
/// Tolerance tier B: rtol = 1e-8.
pub(crate) fn get_c(image_gs: &Array2<f64>, border: usize) -> Array2<f64> {
    let (rows, cols) = image_gs.dim();
    let pad_rows = rows + 2 * border;
    let pad_cols = cols + 2 * border;

    let padded = pad_replicate(image_gs, border);

    // Quintic B-spline kernel coefficients.
    let k: [f64; 6] = [1.0 / 120.0, 13.0 / 60.0, 11.0 / 20.0, 13.0 / 60.0, 1.0 / 120.0, 0.0];

    // Initialise C as complex (imaginary parts track FFT round-trip residuals).
    let mut c_flat: Vec<Complex<f64>> = padded.iter().map(|&v| Complex::new(v, 0.0)).collect();

    let mut planner = FftPlanner::<f64>::new();

    // --- Row-by-row deconvolution ---
    // kernel_x: the symmetric 5-tap quintic sampling kernel laid out
    // zero-phase (centre tap at index 0) -- see build_kernel.
    let mut kernel_x = build_kernel(&k, pad_cols);
    let fft_cols = planner.plan_fft_forward(pad_cols);
    let ifft_cols = planner.plan_fft_inverse(pad_cols);
    fft_cols.process(&mut kernel_x);

    // Row pass — parallel: each row is an independent FFT deconvolution.
    // fft_cols / ifft_cols are Arc<dyn Fft> and are Send + Sync.
    let scale_cols = 1.0 / pad_cols as f64;
    c_flat.par_chunks_exact_mut(pad_cols).for_each(|row| {
        fft_cols.process(row);
        for (v, kv) in row.iter_mut().zip(kernel_x.iter()) {
            *v /= *kv;
        }
        ifft_cols.process(row);
        for v in row.iter_mut() {
            *v *= scale_cols;
        }
    });

    // Transpose c_flat from row-major (pad_rows × pad_cols) to column-major
    // (pad_cols × pad_rows) so each column becomes a contiguous slice.
    // This eliminates the stride-16KB cache misses in the column pass.
    let mut c_t = vec![Complex::new(0.0, 0.0); pad_rows * pad_cols];
    for i in 0..pad_rows {
        for j in 0..pad_cols {
            c_t[j * pad_rows + i] = c_flat[i * pad_cols + j];
        }
    }

    // Column pass — parallel: each "row" of c_t is one original column, now contiguous.
    let mut kernel_y = build_kernel(&k, pad_rows);
    let fft_rows  = planner.plan_fft_forward(pad_rows);
    let ifft_rows = planner.plan_fft_inverse(pad_rows);
    fft_rows.process(&mut kernel_y);

    let scale_rows = 1.0 / pad_rows as f64;
    c_t.par_chunks_exact_mut(pad_rows).for_each(|col| {
        fft_rows.process(col);
        for (v, kv) in col.iter_mut().zip(kernel_y.iter()) {
            *v /= *kv;
        }
        ifft_rows.process(col);
        for v in col.iter_mut() {
            *v *= scale_rows;
        }
    });

    // Transpose back to row-major.
    for i in 0..pad_rows {
        for j in 0..pad_cols {
            c_flat[i * pad_cols + j] = c_t[j * pad_rows + i];
        }
    }

    // Discard imaginary parts (match `np.real(C)`).
    let real: Vec<f64> = c_flat.into_iter().map(|v| v.re).collect();
    Array2::from_shape_vec((pad_rows, pad_cols), real).expect("shape mismatch in get_c")
}

/// Compute Q·C_block·Qᵀ for every pixel in parallel.
///
/// Replicates `Image._get_QCQT` + `_image.cpp:_QCQT`, replacing OpenMP with Rayon.
/// For pixel (i, j): C_block = C[i+border-2 .. i+border+4, j+border-2 .. j+border+4].
///
/// Returns a **block-contiguous** `(rows, cols, 36)` array — the 36 values
/// of block `(i, j)` are one contiguous run, laid out `[dr*6 + dc]`
/// (`layer_rg_plan.md` §11.5).
///
/// Tolerance tier A: atol = 1e-12.
pub(crate) fn get_qcqt(image_gs: &Array2<f64>, c: &Array2<f64>, border: usize) -> Array3<f64> {
    let (rows, cols) = image_gs.dim();
    let q = q_matrix();
    let qt = q.transpose();

    // One row-band (all 36-blocks for one image row) per rayon task,
    // written contiguously.
    let mut flat = vec![0.0f64; rows * cols * 36];
    flat.par_chunks_exact_mut(cols * 36)
        .enumerate()
        .for_each(|(i, band)| {
            for j in 0..cols {
                let ir = i + border - 2;
                let ic = j + border - 2;
                let c_view = c.slice(s![ir..ir + 6, ic..ic + 6]);
                let c_mat = view_to_mat6(c_view);
                let result = q * c_mat * qt;
                let block = &mut band[j * 36..j * 36 + 36];
                for dr in 0..6_usize {
                    for dc in 0..6_usize {
                        block[dr * 6 + dc] = result[(dr, dc)];
                    }
                }
            }
        });

    Array3::from_shape_vec((rows, cols, 36), flat).expect("shape mismatch in get_qcqt")
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Bi-quintic B-spline evaluation matrix Q (row 0 = value, rows 1–5 = derivatives).
///
/// Matches the Python literal in `image.py` / `_get_QCQT`.
fn q_matrix() -> Mat6 {
    #[rustfmt::skip]
    let m = Mat6::from_row_slice(&[
         1.0/120.0,  13.0/60.0,  11.0/20.0,  13.0/60.0,  1.0/120.0,  0.0,
        -1.0/24.0,  -5.0/12.0,   0.0,         5.0/12.0,   1.0/24.0,   0.0,
         1.0/12.0,   1.0/6.0,   -1.0/2.0,     1.0/6.0,    1.0/12.0,   0.0,
        -1.0/12.0,   1.0/6.0,    0.0,         -1.0/6.0,   1.0/12.0,   0.0,
         1.0/24.0,  -1.0/6.0,    1.0/4.0,    -1.0/6.0,   1.0/24.0,   0.0,
        -1.0/120.0,  1.0/24.0,  -1.0/12.0,   1.0/12.0,  -1.0/24.0,   1.0/120.0,
    ]);
    m
}

/// Copy a 6×6 ndarray view into a stack-allocated nalgebra matrix (row-major).
fn view_to_mat6(v: ArrayView2<f64>) -> Mat6 {
    let mut data = [0.0f64; 36];
    let mut k = 0;
    for r in 0..6 {
        for c in 0..6 {
            data[k] = v[[r, c]];
            k += 1;
        }
    }
    Mat6::from_row_slice(&data)
}

/// Build the zero-phase B-spline prefilter kernel vector of length `n`.
///
/// `k = [β(-2), β(-1), β(0), β(1), β(2), 0]` is the symmetric 5-tap quintic
/// sampling kernel (the trailing `0` is not a real tap). For an FFT
/// deconvolution to invert the sampling convolution *without* introducing a
/// spatial shift, the kernel must be centred on index 0:
/// ```text
/// kernel[0]   = β(0)  = k[2]
/// kernel[1]   = β(1)  = k[3]
/// kernel[2]   = β(2)  = k[4]
/// kernel[n-2] = β(-2) = k[0]
/// kernel[n-1] = β(-1) = k[1]
/// ```
///
/// NOTE: this deliberately differs from `geopyv`'s `image.py::_get_C`
/// (`kernel[0:3] = k[3:]`, `kernel[-3:] = k[0:3]`), which places the centre
/// tap `β(0)` at index `n-1` -- a one-sample phase error that shifts every
/// interpolated intensity by exactly (1, 1) px. That bias is invisible for
/// rigid motion (both frames shift equally) but corrupts any strain-gradient
/// region by `∇u · (1, 1)` px. geopyv_dev corrects it here; interpolation
/// golden values consequently do not match the Python package.
fn build_kernel(k: &[f64; 6], n: usize) -> Vec<Complex<f64>> {
    let mut kernel = vec![Complex::new(0.0, 0.0); n];
    kernel[0] = Complex::new(k[2], 0.0);
    kernel[1] = Complex::new(k[3], 0.0);
    kernel[2] = Complex::new(k[4], 0.0);
    kernel[n - 2] = Complex::new(k[0], 0.0);
    kernel[n - 1] = Complex::new(k[1], 0.0);
    kernel
}

/// Pad a 2-D array with `border` replicated pixels on all sides (BORDER_REPLICATE).
///
/// For position (i, j) in the padded output:
/// - i clamped to [0, rows-1], j clamped to [0, cols-1].
fn pad_replicate(arr: &Array2<f64>, border: usize) -> Array2<f64> {
    let (rows, cols) = arr.dim();
    Array2::from_shape_fn((rows + 2 * border, cols + 2 * border), |(i, j)| {
        let ri = i.saturating_sub(border).min(rows - 1);
        let ci = j.saturating_sub(border).min(cols - 1);
        arr[[ri, ci]]
    })
}

/// Load a colour image and return a greyscale array matching cv2's BGR2GRAY.
///
/// Conversion: `Y = (2126·R + 7152·G + 722·B + 5000) / 10000`  (integer division),
/// then 5×5 Gaussian blur (σ = 1.1, BORDER_REFLECT_101) and rounding to 0–255,
/// matching `cv2.GaussianBlur(gs, ksize=(5,5), sigmaX=1.1, sigmaY=1.1)`.
fn load_grayscale(path: &Path) -> Result<Array2<f64>, Error> {
    let img = image::open(path)
        .map_err(|e| Error::ImageLoad(e.to_string()))?
        .to_rgb8();
    let (width, height) = img.dimensions();
    let h = height as usize;
    let w = width as usize;

    // BGR2GRAY (OpenCV integer formula).
    let gs = Array2::from_shape_fn((h, w), |(y, x)| {
        let p = img.get_pixel(x as u32, y as u32);
        let r = p[0] as u32;
        let g = p[1] as u32;
        let b = p[2] as u32;
        ((2126 * r + 7152 * g + 722 * b + 5000) / 10000) as f64
    });

    Ok(gaussian_blur_5x5(&gs))
}

/// 5×5 separable Gaussian blur with σ = 1.1, BORDER_REFLECT_101.
///
/// After blurring the float values are rounded and clamped to 0–255,
/// matching cv2's uint8 output.
fn gaussian_blur_5x5(gs: &Array2<f64>) -> Array2<f64> {
    let sigma = 1.1_f64;
    // Compute normalised kernel (identical to `cv2.getGaussianKernel(5, 1.1)`).
    let raw: [f64; 5] = std::array::from_fn(|i| {
        let x = i as f64 - 2.0;
        (-x * x / (2.0 * sigma * sigma)).exp()
    });
    let sum: f64 = raw.iter().sum();
    let k: [f64; 5] = std::array::from_fn(|i| raw[i] / sum);

    let (rows, cols) = gs.dim();

    // Horizontal pass.
    let mut h = Array2::<f64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            let val: f64 = (0..5_usize)
                .map(|ki| {
                    let jj = reflect101(j as isize + ki as isize - 2, cols as isize) as usize;
                    k[ki] * gs[[i, jj]]
                })
                .sum();
            h[[i, j]] = val;
        }
    }

    // Vertical pass + round + clamp.
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let val: f64 = (0..5_usize)
            .map(|ki| {
                let ii = reflect101(i as isize + ki as isize - 2, rows as isize) as usize;
                k[ki] * h[[ii, j]]
            })
            .sum();
        val.round().clamp(0.0, 255.0)
    })
}

/// BORDER_REFLECT_101 index reflection (gfedcb|abcdefgh|gfedcba).
///
/// Maps out-of-bounds indices back into [0, len).
#[inline]
fn reflect101(idx: isize, len: isize) -> isize {
    if idx < 0 {
        -idx
    } else if idx >= len {
        2 * (len - 1) - idx
    } else {
        idx
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Tier B: constant image → constant C (B-spline kernel sums to 1).
    #[test]
    fn test_get_c_constant() {
        let border = 5;
        let val = 77.0_f64;
        let img = Array2::from_elem((20, 25), val);
        let c = get_c(&img, border);
        assert_eq!(c.dim(), (20 + 2 * border, 25 + 2 * border));
        for &v in c.iter() {
            let diff = (v - val).abs();
            assert!(diff < 1e-8 * val, "C value {v} differs from {val} by {diff}");
        }
    }

    /// Tier A: constant C → QCQT block = c·e₀₀ (only [0,0] entry is c, rest 0).
    #[test]
    fn test_get_qcqt_constant() {
        let border = 5;
        let val = 77.0_f64;
        let img = Array2::from_elem((20, 25), val);
        let c = get_c(&img, border);
        let qcqt = get_qcqt(&img, &c, border);
        assert_eq!(qcqt.dim(), (20, 25, 36));
        // Interior pixel (5, 5): 36-element block, [dr*6 + dc].
        assert!((qcqt[[5, 5, 0]] - val).abs() < 1e-10, "block[0] = {}", qcqt[[5, 5, 0]]);
        for k in 1..36 {
            assert!(qcqt[[5, 5, k]].abs() < 1e-10, "block[{k}] = {}", qcqt[[5, 5, k]]);
        }
    }

    /// pad_replicate: corner pixels should hold the corner value of the source.
    #[test]
    fn test_pad_replicate_corners() {
        let arr = Array2::from_shape_vec((3, 4), (0..12).map(|v| v as f64).collect()).unwrap();
        let border = 2;
        let padded = pad_replicate(&arr, border);
        assert_eq!(padded.dim(), (3 + 2 * border, 4 + 2 * border));
        // Top-left corner should be arr[0,0] = 0.
        assert_eq!(padded[[0, 0]], 0.0);
        // Bottom-right corner should be arr[2,3] = 11.
        assert_eq!(padded[[padded.dim().0 - 1, padded.dim().1 - 1]], 11.0);
        // Centre block equals original.
        assert_eq!(padded[[border, border]], arr[[0, 0]]);
    }

    /// Q matrix row-sums: row 0 = 1, all others = 0 (partition of unity).
    #[test]
    fn test_q_row_sums() {
        let q = q_matrix();
        for r in 0..6 {
            let row_sum: f64 = (0..6).map(|c| q[(r, c)]).sum();
            let expected = if r == 0 { 1.0 } else { 0.0 };
            assert!(
                (row_sum - expected).abs() < 1e-15,
                "Q row {r} sum = {row_sum}, expected {expected}"
            );
        }
    }
}
