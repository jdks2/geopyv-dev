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
// Zone lookup
// ---------------------------------------------------------------------------

/// Zone id at `coord` in a whole-image per-pixel zone-label grid, clamping
/// out-of-bounds coordinates to the nearest edge pixel (same convention as
/// [`LocalMask::zone_mask_update`] and `Mesh::solve_zonal_masking_impl`'s own
/// `node_zone` construction) -- the single definition of "the zone at a
/// point" shared by masking, `node_zone`, and zone-aware meshless filtering.
pub fn zone_at(zone_image: ArrayView2<u8>, coord: [f64; 2]) -> u8 {
    let h = zone_image.nrows() as isize;
    let w = zone_image.ncols() as isize;
    let x = (coord[0] as isize).clamp(0, w - 1) as usize;
    let y = (coord[1] as isize).clamp(0, h - 1) as usize;
    zone_image[[y, x]]
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MaskShape {
    Circle,
    Square,
    /// Half of a `Circle`, keeping only the bottom half (`y_offset >= 0`,
    /// i.e. rows at or below the subset centre — row index increases
    /// downward, see module doc) relative to each subset's own centre. A
    /// fixed-template demonstration that changing the subset template
    /// shape alone can improve performance at a discontinuity.
    Semicircle,
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

    /// Create a semicircular local mask of the given radius: a `Circle`
    /// keeping only the bottom half (`y_offset >= 0`) relative to the
    /// subset's own centre.
    pub fn semicircle(radius: usize) -> Result<Self, Error> {
        if radius == 0 {
            return Err(Error::InvalidInput("radius must be > 0".to_string()));
        }
        let size = radius;
        let side = 2 * size + 1;

        let mut coord_rows: Vec<[f64; 2]> = Vec::new();
        let mut subset_mask = Array2::<i32>::ones((side, side));

        for row in 0..side {
            for col in 0..side {
                let x_val = col as i64 - size as i64;
                let y_val = row as i64 - size as i64;
                let dist = ((x_val * x_val + y_val * y_val) as f64).sqrt();
                if dist <= size as f64 && y_val >= 0 {
                    coord_rows.push([x_val as f64, y_val as f64]);
                } else {
                    subset_mask[[row, col]] = 0;
                }
            }
        }

        let n_px = coord_rows.len();
        let coords = Array2::from_shape_fn((n_px, 2), |(i, j)| coord_rows[i][j]);

        Ok(LocalMask {
            shape: MaskShape::Semicircle,
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

    /// Apply a whole-image ZONE label mask: keep only pixels whose label
    /// matches the label AT `centre` itself, discarding pixels belonging to
    /// any other zone. Sibling to [`mask_update`](Self::mask_update) --
    /// same window-extraction/clamping logic (kept in sync deliberately
    /// rather than refactored into a shared helper, to leave `mask_update`
    /// itself untouched) -- but the keep-rule is "pixel's label == centre's
    /// own label" instead of "pixel's raw value != 0".
    ///
    /// Lets a subset straddling a detected discontinuity (e.g. a shear-band
    /// edge) exclude pixels on the OTHER side of that discontinuity from its
    /// own centre, using an externally-computed zone map (e.g. thresholded/
    /// connected-component-labelled from a particle strain field) rather
    /// than any per-subset local detection.
    ///
    /// # Arguments
    /// * `centre`      - `[x, y]` centre coordinates (integer pixel, 0-indexed).
    /// * `zone_labels` - Whole-image per-pixel zone id (small integer labels;
    ///   any two different values are treated as different zones).
    ///
    /// # Caveat
    /// Pixels outside `zone_labels`' own bounds are padded with `0` (matching
    /// `mask_update`'s padding convention). If `0` is also a genuine zone id
    /// AND a subset's footprint extends past the image edge, that padding is
    /// indistinguishable from real zone-0 pixels and would be kept rather
    /// than excluded. Callers whose ROI margin is smaller than a subset
    /// radius should reserve `0` for "no zone" and start real zone ids at 1.
    pub fn zone_mask_update(&mut self, centre: [f64; 2], zone_labels: ArrayView2<u8>) {
        let mask_height = zone_labels.nrows() as isize;
        let mask_width = zone_labels.ncols() as isize;
        let size = self.size as isize;

        let center_x = (centre[0] as isize).clamp(0, mask_width - 1);
        let center_y = (centre[1] as isize).clamp(0, mask_height - 1);
        let centre_label = zone_at(zone_labels, centre);

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

        let side = (2 * self.size + 1) as usize;
        let mut local_padded = Array2::<u8>::zeros((side, side));
        let local_slice = zone_labels.slice(s![ly as usize..uy as usize, lx as usize..ux as usize]);
        let target_row_end = side - uyp;
        let target_col_end = side - uxp;
        local_padded
            .slice_mut(s![lyp..target_row_end, lxp..target_col_end])
            .assign(&local_slice);

        let sm = self.subset_mask.clone();

        let mut new_coords: Vec<[f64; 2]> = Vec::new();
        for row in 0..side {
            for col in 0..side {
                let keep = local_padded[[row, col]] == centre_label && sm[[row, col]] != 0;
                if keep {
                    let x_off = col as isize - self.size as isize;
                    let y_off = row as isize - self.size as isize;
                    new_coords.push([x_off as f64, y_off as f64]);
                } else {
                    // Also bake the exclusion into `subset_mask` itself, not
                    // just `coords`/`m_n_px`: `Subset::new`'s OWN global-mask
                    // (boundary/exclusion rasterisation) branch clones this
                    // LocalMask and rebuilds `coords` from scratch via
                    // `mask_update`, which reads `subset_mask` as its base
                    // shape -- it does NOT consult the `coords` this method
                    // just computed. Without also zeroing `subset_mask` here,
                    // a zone-masked LocalMask combined with any mesh that
                    // also has a boundary/exclusion mask would silently lose
                    // its zone exclusion entirely (confirmed: exact bit-for-
                    // bit identical solve output with zone masking "applied").
                    self.subset_mask[[row, col]] = 0;
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
    fn semicircle_5_n_px_is_half_of_circle_rounded_up() {
        // Circle(5) has 81 px; the y=0 row (11 px) is shared, everything
        // strictly above (y<0) is dropped -- so semicircle keeps the
        // bottom half (y>=0) inclusive of the centre row.
        let full = LocalMask::circle(5).unwrap();
        let half = LocalMask::semicircle(5).unwrap();
        assert_eq!(full.n_px, 81);
        assert_eq!(half.n_px, 46);
    }

    #[test]
    fn semicircle_5_all_coords_bottom_half() {
        let t = LocalMask::semicircle(5).unwrap();
        for i in 0..t.n_px {
            assert!(t.coords[[i, 1]] >= 0.0, "y-offset {} should be >= 0", t.coords[[i, 1]]);
        }
    }

    #[test]
    fn semicircle_5_subset_mask_values() {
        let t = LocalMask::semicircle(5).unwrap();
        // Centre active.
        assert_eq!(t.subset_mask[[5, 5]], 1);
        // Bottom pole (y=5, x=0) active; top pole (y=-5, x=0) masked out.
        assert_eq!(t.subset_mask[[10, 5]], 1);
        assert_eq!(t.subset_mask[[0, 5]], 0);
        // Any row strictly above centre is entirely masked.
        assert!(t.subset_mask.row(0).iter().all(|&v| v == 0));
        assert!(t.subset_mask.row(4).iter().all(|&v| v == 0));
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

    // ---- Zone masking ----

    #[test]
    fn zone_mask_update_uniform_label_keeps_everything() {
        // Every pixel shares one label -- identical to an all-ones mask_update.
        let mut t = LocalMask::square(2).unwrap();
        let labels = Array2::<u8>::from_elem((20, 20), 7);
        t.zone_mask_update([5.0, 5.0], labels.view());
        assert_eq!(t.m_n_px, Some(25)); // (2*2+1)^2 = 25
    }

    #[test]
    fn zone_mask_update_splits_on_boundary() {
        // Two zones split by a vertical line at column 5: label 0 for
        // col < 5, label 1 for col >= 5. A subset centred at (5, 5) (zone 1)
        // should keep only its own-zone half (including the centre column).
        let mut labels = Array2::<u8>::zeros((20, 20));
        labels.slice_mut(s![.., 5..]).fill(1);
        let mut t = LocalMask::square(2).unwrap();
        t.zone_mask_update([5.0, 5.0], labels.view());
        // side=5 (2*2+1), centre col index 2 -> image col 5; kept columns are
        // image cols 5..=7 (offsets 0..=2), i.e. 3 of the 5 columns, all 5 rows.
        assert_eq!(t.m_n_px, Some(3 * 5));
        for c in t.coords.rows() {
            assert!(c[0] >= 0.0, "x_off = {} should be on the label-1 side", c[0]);
        }
    }

    #[test]
    fn zone_mask_update_clamps_at_image_edge() {
        // Centre at (1, 1) with size=2: the window extends 1px past the
        // top/left image edges. Reserve label 0 for "no zone" (per the
        // caveat) and label 1 for the one real zone covering the whole
        // 20x20 image, so out-of-bounds zero-padding is distinguishable
        // from real data and gets correctly excluded rather than kept.
        let labels = Array2::<u8>::from_elem((20, 20), 1);
        let mut t = LocalMask::square(2).unwrap();
        t.zone_mask_update([1.0, 1.0], labels.view());
        // In-bounds region is rows/cols 0..=3 (4x4=16); the rest of the 5x5
        // window is out-of-bounds padding (label 0 != centre's label 1).
        assert_eq!(t.m_n_px, Some(16));
    }

    #[test]
    fn zone_mask_update_survives_a_later_mask_update() {
        // Regression test for a real bug: `Subset::new`'s `global_mask`
        // branch clones the LocalMask it's given and calls `mask_update`,
        // which rebuilds `coords` from `subset_mask` (the shape bitmap)
        // intersected with the boundary mask -- it does NOT consult
        // `coords`/`m_n_px` as they stood beforehand. A `zone_mask_update`
        // that only touched `coords` (not `subset_mask`) would therefore be
        // silently discarded by any mesh that also has a boundary/exclusion
        // mask (i.e. almost every real mesh) the moment `mask_update` runs
        // afterwards. Confirmed the hard way: a full solve with zone_mask
        // set produced bit-for-bit identical output to one without it.
        let mut labels = Array2::<u8>::zeros((20, 20));
        labels.slice_mut(s![.., 5..]).fill(1);
        let mut t = LocalMask::square(2).unwrap();
        t.zone_mask_update([5.0, 5.0], labels.view()); // keeps only the label-1 (right) half, 15 px
        assert_eq!(t.m_n_px, Some(15));

        // Now apply an all-ones (fully permissive) boundary mask, exactly
        // as `Subset::new` does when `global_mask` is `Some`. If the zone
        // exclusion only lived in `coords`, this would silently restore all
        // 25 pixels of the original square.
        let boundary_mask = Array2::<u8>::ones((20, 20));
        t.mask_update([5.0, 5.0], boundary_mask.view());
        assert_eq!(t.m_n_px, Some(15), "zone exclusion must survive a later mask_update");
        for c in t.coords.rows() {
            assert!(c[0] >= 0.0, "x_off = {} should still be on the label-1 side", c[0]);
        }
    }

}
