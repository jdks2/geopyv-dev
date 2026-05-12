//! Subset mask shapes: Circle and Square.
//!
//! Translates `geopyv/templates.py`.  A `LocalMask` records which pixel offsets
//! relative to a subset centre are active, and an optional subset mask that
//! can zero out pixels covered by a binary image mask.
//!
//! # Coordinate convention
//!
//! Both `Circle` and `Square` use `coords[:, 0]` = x-offset, `coords[:, 1]` = y-offset,
//! matching the `bspline_eval(x, y, ...)` and `bspline_grad(x, y, ...)` call convention.
//! This is also preserved after `mask_update`.

use ndarray::{s, Array2, ArrayView2};

use crate::Error;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MaskShape {
    Circle,
    Square,
}

/// Subset local mask: describes which pixel offsets relative to a centre are
/// active for DIC intensity sampling.
#[derive(Debug, Clone)]
pub struct LocalMask {
    pub shape: MaskShape,
    /// Human-readable description of what `size` measures.
    pub dimension: String,
    /// Radius (Circle) or half-side-length (Square).
    pub size: usize,
    /// Number of pixels in the unmasked mask.
    pub n_px: usize,
    /// Pixel offset coordinates, shape `(n_px, 2)`.
    /// Column convention depends on shape — see module doc.
    pub coords: Array2<f64>,
    /// Binary mask of shape `((2*size+1), (2*size+1))`: 1 = active, 0 = masked.
    /// `ndarray::int32` to match Python's `np.intc`.
    pub subset_mask: Array2<i32>,
    /// Number of pixels after applying an image mask via `mask_update`.
    pub m_n_px: Option<usize>,
}

impl LocalMask {
    /// Create a circular local mask of the given radius.
    ///
    /// Replicates `geopyv.templates.Circle.__init__`.
    pub fn circle(radius: usize) -> Result<Self, Error> {
        if radius == 0 {
            return Err(Error::InvalidInput("radius must be > 0".to_string()));
        }
        let size = radius;
        let side = 2 * size + 1;

        // Compute coords and subset_mask in one pass.
        // Row-major scan:
        //   coords[:, 0] = x-offset (col - size)
        //   coords[:, 1] = y-offset (row - size)
        // Matches Square and bspline_eval(x, y) convention.
        let mut coord_rows: Vec<[f64; 2]> = Vec::new();
        let mut subset_mask = Array2::<i32>::ones((side, side));

        for row in 0..side {
            for col in 0..side {
                let x_val = col as i64 - size as i64; // x varies along cols
                let y_val = row as i64 - size as i64; // y varies along rows
                let dist = ((x_val * x_val + y_val * y_val) as f64).sqrt();
                if dist <= size as f64 {
                    coord_rows.push([x_val as f64, y_val as f64]);
                } else {
                    subset_mask[[row, col]] = 0;
                }
            }
        }

        let n_px = coord_rows.len();
        let coords = Array2::from_shape_fn((n_px, 2), |(i, j)| coord_rows[i][j]);

        Ok(LocalMask {
            shape: MaskShape::Circle,
            dimension: "radius".to_string(),
            size,
            n_px,
            coords,
            subset_mask,
            m_n_px: None,
        })
    }

    /// Create a square local mask with the given half-side-length.
    ///
    /// Replicates `geopyv.templates.Square.__init__`.
    pub fn square(length: usize) -> Result<Self, Error> {
        if length == 0 {
            return Err(Error::InvalidInput("length must be > 0".to_string()));
        }
        let size = length;
        let side = 2 * size + 1;

        // np.meshgrid(x_arr, y_arr) + np.ravel convention:
        //   x_s[i, j] = j - size  (x varies along cols)
        //   y_s[i, j] = i - size  (y varies along rows)
        //   ravel(x_s): inner loop is col → x-offset first
        //   coords[:, 0] = x-offset, coords[:, 1] = y-offset
        let n_px = side * side;
        let coords = Array2::from_shape_fn((n_px, 2), |(k, ax)| {
            let row = k / side;
            let col = k % side;
            let x_val = col as i64 - size as i64;
            let y_val = row as i64 - size as i64;
            if ax == 0 { x_val as f64 } else { y_val as f64 }
        });

        // Square subset_mask: all ones (no masking by shape).
        let subset_mask = Array2::<i32>::ones((side, side));

        Ok(LocalMask {
            shape: MaskShape::Square,
            dimension: "length".to_string(),
            size,
            n_px,
            coords,
            subset_mask,
            m_n_px: None,
        })
    }

    /// Apply a binary image mask to the local mask, updating `coords` and `m_n_px`.
    ///
    /// Replicates `geopyv.templates.Template.mask`.
    ///
    /// After this call `coords` is always in `[row-offset, col-offset]` order
    /// (matching `np.argwhere` row/col convention) regardless of mask shape.
    ///
    /// # Arguments
    /// * `centre` - `[x, y]` centre coordinates (integer pixel, 0-indexed).
    /// * `mask`   - Binary image mask: 0 = masked, 1 = active.
    pub fn mask_update(&mut self, centre: [f64; 2], mask: ArrayView2<u8>) {
        let mask_height = mask.nrows() as isize;
        let mask_width = mask.ncols() as isize;
        let size = self.size as isize;

        let center_x = centre[0] as isize;
        let center_y = centre[1] as isize;

        let mut ly = center_y - size;
        let mut uy = center_y + size + 1;
        let mut lx = center_x - size;
        let mut ux = center_x + size + 1;

        let mut lyp: usize = 0;
        let mut uyp: usize = 0;
        let mut lxp: usize = 0;
        let mut uxp: usize = 0;

        if lx < 0 {
            lxp = (-lx) as usize;
            lx = 0;
        }
        if ly < 0 {
            lyp = (-ly) as usize;
            ly = 0;
        }
        if ux >= mask_width {
            uxp = (ux + 1 - mask_width) as usize;
            ux = mask_width - 1;
        }
        if uy >= mask_height {
            uyp = (uy + 1 - mask_height) as usize;
            uy = mask_height - 1;
        }

        // Extract local region and embed in the padded frame.
        let side = (2 * self.size + 1) as usize;
        let mut local_padded = Array2::<u8>::zeros((side, side));
        let local_slice = mask.slice(s![ly as usize..uy as usize, lx as usize..ux as usize]);
        let target_row_end = side - uyp;
        let target_col_end = side - uxp;
        local_padded
            .slice_mut(s![lyp..target_row_end, lxp..target_col_end])
            .assign(&local_slice);

        // Clone subset_mask to avoid borrow conflict.
        let sm = self.subset_mask.clone();

        // Collect active coords: positions where local_padded AND subset_mask are both non-zero.
        // Output uses [x-offset, y-offset] convention matching bspline_eval(x, y).
        let mut new_coords: Vec<[f64; 2]> = Vec::new();
        for row in 0..side {
            for col in 0..side {
                if local_padded[[row, col]] != 0 && sm[[row, col]] != 0 {
                    let x_off = col as isize - self.size as isize;
                    let y_off = row as isize - self.size as isize;
                    new_coords.push([x_off as f64, y_off as f64]);
                }
            }
        }

        self.m_n_px = Some(new_coords.len());
        let n = new_coords.len();
        self.coords = Array2::from_shape_fn((n, 2), |(i, j)| new_coords[i][j]);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_5_n_px() {
        // 81 lattice points inside a radius-5 circle (computed analytically).
        let t = LocalMask::circle(5).unwrap();
        assert_eq!(t.n_px, 81);
    }

    #[test]
    fn circle_5_first_coord() {
        // Row-major scan: first included pixel is at row=0, col=5 → [x=0, y=-5].
        let t = LocalMask::circle(5).unwrap();
        assert_eq!(t.coords[[0, 0]], 0.0);  // x-offset
        assert_eq!(t.coords[[0, 1]], -5.0); // y-offset
    }

    #[test]
    fn circle_5_last_coord() {
        // Last included pixel is at row=10, col=5 → [x=0, y=5].
        let t = LocalMask::circle(5).unwrap();
        let n = t.n_px;
        assert_eq!(t.coords[[n - 1, 0]], 0.0);
        assert_eq!(t.coords[[n - 1, 1]], 5.0);
    }

    #[test]
    fn circle_5_subset_mask_shape() {
        let t = LocalMask::circle(5).unwrap();
        assert_eq!(t.subset_mask.dim(), (11, 11));
    }

    #[test]
    fn circle_5_subset_mask_values() {
        let t = LocalMask::circle(5).unwrap();
        // Centre should be active.
        assert_eq!(t.subset_mask[[5, 5]], 1);
        // Poles should be active (dist = 5 = size, boundary included).
        assert_eq!(t.subset_mask[[0, 5]], 1); // top pole: y=-5, x=0
        assert_eq!(t.subset_mask[[5, 0]], 1); // left pole: y=0, x=-5
        // Corners should be masked (dist = 5√2 > 5).
        assert_eq!(t.subset_mask[[0, 0]], 0);
        assert_eq!(t.subset_mask[[0, 10]], 0);
    }

    #[test]
    fn square_5_n_px() {
        let t = LocalMask::square(5).unwrap();
        assert_eq!(t.n_px, 121); // (2*5+1)^2 = 121
    }

    #[test]
    fn square_5_first_and_last_coord() {
        // coords[:, 0] = x-offset, coords[:, 1] = y-offset.
        // First (row=0, col=0): x=-5, y=-5.
        let t = LocalMask::square(5).unwrap();
        assert_eq!(t.coords[[0, 0]], -5.0); // x
        assert_eq!(t.coords[[0, 1]], -5.0); // y
        let n = t.n_px;
        assert_eq!(t.coords[[n - 1, 0]], 5.0); // x
        assert_eq!(t.coords[[n - 1, 1]], 5.0); // y
    }

    #[test]
    fn square_5_subset_mask_all_ones() {
        let t = LocalMask::square(5).unwrap();
        assert!(t.subset_mask.iter().all(|&v| v == 1));
    }

    #[test]
    fn mask_update_full_mask() {
        // Apply a full-ones mask to square(2): all coords should survive.
        let mut t = LocalMask::square(2).unwrap();
        let mask = Array2::<u8>::ones((20, 20));
        t.mask_update([5.0, 5.0], mask.view());
        assert_eq!(t.m_n_px, Some(25)); // (2*2+1)^2 = 25
    }

    #[test]
    fn mask_update_zero_mask() {
        // Apply a full-zeros mask: no coords survive.
        let mut t = LocalMask::square(2).unwrap();
        let mask = Array2::<u8>::zeros((20, 20));
        t.mask_update([5.0, 5.0], mask.view());
        assert_eq!(t.m_n_px, Some(0));
    }
}
