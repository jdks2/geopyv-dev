//! Region module — defines tracked/static geometric boundaries for mesh analysis.
//!
//! Translates `geopyv/geometry/region.py`.
//!
//! Python base-class boilerplate (`Object`, `RegionBase`, `_report` / `gp.check`
//! calls, GUI selectors) is removed:
//!   - Type checking: handled by PyO3 at the Python boundary.
//!   - Domain validation: `Result<T, Error>` constructors.
//!   - GUI fallback: centre is always required; passing `None` is an error.

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::Error;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegionOption {
    /// Defined: a pre-supplied controlled region (multi-snapshot boundary).
    D,
    /// Static: an untracked exclusion region.
    S,
    /// Rigid: tracked region that can translate and rotate but not deform.
    R,
    /// Flexible: tracked region that can fully deform.
    F,
}

impl std::fmt::Display for RegionOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            RegionOption::D => "D",
            RegionOption::S => "S",
            RegionOption::R => "R",
            RegionOption::F => "F",
        };
        f.write_str(s)
    }
}

impl std::str::FromStr for RegionOption {
    type Err = crate::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "D" => Ok(RegionOption::D),
            "S" => Ok(RegionOption::S),
            "R" => Ok(RegionOption::R),
            "F" => Ok(RegionOption::F),
            other => Err(crate::Error::InvalidInput(format!(
                "option must be 'D', 'S', 'R', or 'F'; got '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegionShape {
    Circle { radius: f64, size: f64 },
    Path { radius: f64 },
    Generic,
}

// ---------------------------------------------------------------------------
// Region struct
// ---------------------------------------------------------------------------

/// A geometric boundary region used to define the mesh analysis domain.
///
/// Closely mirrors the Python `Region` / `Circle` / `Path` class hierarchy,
/// collapsed into a single struct with a `shape` tag and option enum.
///
/// `history_nodes[k]` / `history_centres[k]` hold the state after `k`
/// `store_*` calls.  `current_nodes` / `current_centre` are the working
/// state, updated by `update()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    pub shape: RegionShape,
    pub option: RegionOption,
    pub hard: bool,
    pub compensate: bool,
    pub solved: bool,
    pub calibrated: bool,
    /// Working node set (updated by `update`).
    #[serde(with = "serde_ndarray_2d")]
    pub current_nodes: Array2<f64>,
    /// Working centre (updated by `update`).
    pub current_centre: [f64; 2],
    /// Accumulated node snapshots (grows with each `store_*` call).
    #[serde(with = "serde_vec_ndarray_2d")]
    pub history_nodes: Vec<Array2<f64>>,
    /// Accumulated centre snapshots (grows with each `store_*` call).
    pub history_centres: Vec<[f64; 2]>,
    pub counter: usize,
    pub ref_index: Option<usize>,
    pub reference_update_register: Vec<usize>,
}

impl Region {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a circular region.
    ///
    /// Replicates `geopyv.geometry.region.Circle.__init__`.
    ///
    /// # Arguments
    /// * `centre` - `[x, y]` centre of the circle.
    /// * `radius` - Radius of the boundary polygon.
    /// * `size`   - Approximate arc spacing between polygon vertices.
    /// * `option` - Tracking mode (`D`, `S`, `R`, `F`).
    /// * `hard`   - Include boundary in binary mask (`true`) or use full image (`false`).
    /// * `compensate` - Enable compensation (stored for later use).
    pub fn circle(
        centre: [f64; 2],
        radius: f64,
        size: f64,
        option: RegionOption,
        hard: bool,
        compensate: bool,
    ) -> Result<Self, Error> {
        if radius <= 0.0 {
            return Err(Error::InvalidInput(format!("radius must be > 0, got {radius}")));
        }
        if size <= 0.0 {
            return Err(Error::InvalidInput(format!("size must be > 0, got {size}")));
        }

        let number_points = {
            let computed = (2.0 * std::f64::consts::PI * radius / size) as usize;
            computed.max(6) // at least 6 points
        };
        let theta_step = 2.0 * std::f64::consts::PI / number_points as f64;
        let nodes: Array2<f64> = Array2::from_shape_fn((number_points, 2), |(i, j)| {
            let theta = theta_step * i as f64;
            if j == 0 { radius * theta.cos() + centre[0] } else { radius * theta.sin() + centre[1] }
        });

        Ok(Self::from_parts(
            RegionShape::Circle { radius, size },
            option,
            centre,
            nodes,
            hard,
            compensate,
        ))
    }

    /// Create a path (arbitrary polygon) region.
    ///
    /// Replicates `geopyv.geometry.region.Path.__init__`.
    ///
    /// If `centre` is `None`, it is computed as the mean of `nodes`.
    pub fn path(
        centre: Option<[f64; 2]>,
        nodes: Array2<f64>,
        option: RegionOption,
        hard: bool,
        compensate: bool,
        radius: f64,
    ) -> Result<Self, Error> {
        let n_nodes = nodes.nrows();
        if n_nodes == 0 {
            return Err(Error::InvalidInput("nodes must not be empty".to_string()));
        }

        let c: [f64; 2] = match centre {
            Some(pt) => pt,
            None => {
                let mx = nodes.column(0).mean().unwrap_or(0.0);
                let my = nodes.column(1).mean().unwrap_or(0.0);
                [mx, my]
            }
        };

        Ok(Self::from_parts(
            RegionShape::Path { radius },
            option,
            c,
            nodes,
            hard,
            compensate,
        ))
    }

    /// Low-level constructor used by `circle` and `path`.
    fn from_parts(
        shape: RegionShape,
        option: RegionOption,
        centre: [f64; 2],
        nodes: Array2<f64>,
        hard: bool,
        compensate: bool,
    ) -> Self {
        let history_nodes = vec![nodes.clone()];
        let history_centres = vec![centre];
        Region {
            shape,
            option,
            hard,
            compensate,
            solved: false,
            calibrated: false,
            current_nodes: nodes,
            current_centre: centre,
            history_nodes,
            history_centres,
            counter: 0,
            ref_index: None,
            reference_update_register: Vec::new(),
        }
    }

    /// Return the region's characteristic radius.
    ///
    /// `RegionShape::Circle { radius, .. }` and `RegionShape::Path { radius }`
    /// both carry one (used to size the rigid-registration subset template);
    /// `Generic` has none and is unreachable via the current constructors.
    pub fn radius(&self) -> Result<f64, Error> {
        match self.shape {
            RegionShape::Circle { radius, .. } => Ok(radius),
            RegionShape::Path { radius } => Ok(radius),
            RegionShape::Generic => Err(Error::InvalidInput(
                "region has no radius (RegionShape::Generic)".to_string(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Store methods (called per sequence step)
    // -----------------------------------------------------------------------

    /// Update region state for a Rigid (`R`) tracking step.
    ///
    /// Replicates `Region._store` for `option == "R"`.
    ///
    /// `warp` is a 1-D warp parameter vector with at least 5 elements:
    /// `[u, v, _, du/dx, du/dy, ...]` where `[u, v]` is the translation and
    /// `theta = (warp[3] - warp[4]) / 2` is the rotation angle.
    pub fn store_rigid(&mut self, warp: &[f64]) -> Result<(), Error> {
        if warp.len() < 5 {
            return Err(Error::InvalidInput(
                "rigid warp vector needs at least 5 elements".to_string(),
            ));
        }
        let n = self.current_nodes.nrows();
        // local_coordinates = nodes - centre  (broadcast subtract)
        let local: Array2<f64> = Array2::from_shape_fn((n, 2), |(i, j)| {
            self.current_nodes[[i, j]] - self.current_centre[j]
        });
        let theta = (warp[3] - warp[4]) / 2.0;
        let (c, s) = (theta.cos(), theta.sin());
        // rot = [[cos, sin], [-sin, cos]]
        let rot = [[c, s], [-s, c]];
        // rotated = local @ rot.T  (rot is already symmetric here? No—use dot product)
        let rotated: Array2<f64> = Array2::from_shape_fn((n, 2), |(i, out_j)| {
            rot[0][out_j] * local[[i, 0]] + rot[1][out_j] * local[[i, 1]]
        });
        // new_nodes = centre + warp[:2] + rotated
        let new_nodes: Array2<f64> = Array2::from_shape_fn((n, 2), |(i, j)| {
            self.current_centre[j] + warp[j] + rotated[[i, j]]
        });
        let new_centre: [f64; 2] = [
            self.current_centre[0] + warp[0],
            self.current_centre[1] + warp[1],
        ];
        self.history_nodes.push(new_nodes);
        self.history_centres.push(new_centre);
        self.solved = true;
        self.counter += 1;
        Ok(())
    }

    /// Update region state for a Flexible (`F`) tracking step.
    ///
    /// Replicates `Region._store` for `option == "F"`.
    ///
    /// `warp` is a 2-D array of shape `(N, 2)` with per-node displacement
    /// vectors `[du, dv]`, matching the shape of `current_nodes`.
    pub fn store_flexible(&mut self, warp: ArrayView2<f64>) -> Result<(), Error> {
        let n = self.current_nodes.nrows();
        if warp.nrows() != n || warp.ncols() != 2 {
            return Err(Error::InvalidInput(format!(
                "flexible warp must have shape ({n}, 2), got ({}, {})",
                warp.nrows(),
                warp.ncols()
            )));
        }
        let new_nodes = &self.current_nodes + &warp;
        // centre += mean(warp, axis=0)
        let warp_mean_x = warp.column(0).mean().unwrap_or(0.0);
        let warp_mean_y = warp.column(1).mean().unwrap_or(0.0);
        let new_centre: [f64; 2] = [
            self.current_centre[0] + warp_mean_x,
            self.current_centre[1] + warp_mean_y,
        ];
        self.history_nodes.push(new_nodes);
        self.history_centres.push(new_centre);
        self.solved = true;
        self.counter += 1;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Update method (called per sequence image)
    // -----------------------------------------------------------------------

    /// Update the working nodes/centre from the stored history using the index
    /// embedded in the image filename.
    ///
    /// Replicates `Region._update`.
    ///
    /// Extracts the last run of digits from `filepath` as the frame index.
    /// If the index differs from `ref_index`, the working state is set to
    /// `history_nodes[counter]` / `history_centres[counter]`.
    ///
    /// "S" (Static) regions are never updated.
    pub fn update(&mut self, filepath: &str) {
        if self.option == RegionOption::S {
            return;
        }
        if let Some(f_index) = last_integer_in_str(filepath) {
            if Some(f_index) != self.ref_index {
                self.ref_index = Some(f_index);
                self.reference_update_register.push(f_index);
                let idx = self.counter;
                if idx < self.history_nodes.len() {
                    self.current_nodes = self.history_nodes[idx].clone();
                    self.current_centre = self.history_centres[idx];
                }
            }
        }
        // If no integer found in filepath, silently skip (matches Python's try/except).
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Extract the last run of ASCII digits from a string and parse it as `usize`.
///
/// Replicates Python `int(re.findall(r"\d+", s)[-1])`.
fn last_integer_in_str(s: &str) -> Option<usize> {
    let end = s.rfind(|c: char| c.is_ascii_digit())?;
    let start = s[..=end]
        .rfind(|c: char| !c.is_ascii_digit())
        .map(|i| i + s[i..].chars().next().map_or(1, |c| c.len_utf8()))
        .unwrap_or(0);
    s[start..=end].parse().ok()
}

// ---------------------------------------------------------------------------
// Minimal serde helpers for ndarray (until a proper crate is added)
// ---------------------------------------------------------------------------
// These serialize Array2<f64> as Vec<Vec<f64>> and Array1<f64> as Vec<f64>.
// Sufficient for bincode round-trips within this module.

mod serde_ndarray_2d {
    use ndarray::Array2;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(arr: &Array2<f64>, s: S) -> Result<S::Ok, S::Error> {
        let v: Vec<Vec<f64>> = arr.outer_iter().map(|row| row.to_vec()).collect();
        v.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Array2<f64>, D::Error> {
        let v: Vec<Vec<f64>> = Vec::deserialize(d)?;
        if v.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }
        let cols = v[0].len();
        let flat: Vec<f64> = v.into_iter().flatten().collect();
        let rows = flat.len() / cols;
        Array2::from_shape_vec((rows, cols), flat).map_err(serde::de::Error::custom)
    }
}

mod serde_vec_ndarray_2d {
    use ndarray::Array2;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &Vec<Array2<f64>>, s: S) -> Result<S::Ok, S::Error> {
        let nested: Vec<Vec<Vec<f64>>> = v
            .iter()
            .map(|arr| arr.outer_iter().map(|row| row.to_vec()).collect())
            .collect();
        nested.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Array2<f64>>, D::Error> {
        let nested: Vec<Vec<Vec<f64>>> = Vec::deserialize(d)?;
        nested
            .into_iter()
            .map(|vv| {
                if vv.is_empty() {
                    return Ok(Array2::zeros((0, 0)));
                }
                let cols = vv[0].len();
                let flat: Vec<f64> = vv.into_iter().flatten().collect();
                let rows = flat.len() / cols;
                Array2::from_shape_vec((rows, cols), flat).map_err(serde::de::Error::custom)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_option_roundtrip() {
        assert_eq!("R".parse::<RegionOption>().unwrap(), RegionOption::R);
        assert!("X".parse::<RegionOption>().is_err());
    }

    #[test]
    fn circle_region_node_count() {
        // radius=50, size=20 → max(6, int(2π*50/20)) = max(6, 15) = 15 nodes
        let r = Region::circle([100.0, 100.0], 50.0, 20.0, RegionOption::F, true, true).unwrap();
        let expected = ((2.0 * std::f64::consts::PI * 50.0 / 20.0) as usize).max(6);
        assert_eq!(r.current_nodes.nrows(), expected);
    }

    #[test]
    fn circle_region_history_initialized() {
        let r = Region::circle([0.0, 0.0], 10.0, 5.0, RegionOption::F, true, true).unwrap();
        assert_eq!(r.history_nodes.len(), 1);
        assert_eq!(r.history_centres.len(), 1);
        assert_eq!(r.counter, 0);
    }

    #[test]
    fn path_region_auto_centre() {
        // Centre should be the mean of the four corners of a unit square.
        let nodes = ndarray::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let r = Region::path(None, nodes, RegionOption::F, true, true, 5.0).unwrap();
        assert!((r.current_centre[0] - 0.5).abs() < 1e-12);
        assert!((r.current_centre[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn store_flexible_updates_history() {
        let nodes = ndarray::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mut r = Region::path(None, nodes.clone(), RegionOption::F, true, true, 5.0).unwrap();
        let warp = ndarray::array![[1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5]];
        r.store_flexible(warp.view()).unwrap();
        assert_eq!(r.counter, 1);
        assert_eq!(r.history_nodes.len(), 2);
        // New nodes = original + warp.
        assert!((r.history_nodes[1][[0, 0]] - 1.0).abs() < 1e-12);
        assert!((r.history_nodes[1][[0, 1]] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn store_rigid_updates_history() {
        let nodes = ndarray::array![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0],
                                     [0.0, -1.0], [0.5, 0.5], [-0.5, -0.5]];
        let mut r = Region::path(
            Some([0.0, 0.0]),
            nodes,
            RegionOption::R,
            true,
            true,
            5.0,
        ).unwrap();
        // Pure translation, no rotation (warp[3]=warp[4]=0 → theta=0).
        let warp = [1.0, 2.0, 0.0, 0.0, 0.0];
        r.store_rigid(&warp).unwrap();
        assert_eq!(r.counter, 1);
        // Centre should move by (1, 2).
        assert!((r.history_centres[1][0] - 1.0).abs() < 1e-12);
        assert!((r.history_centres[1][1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn update_sets_current_nodes() {
        let nodes = ndarray::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mut r = Region::path(None, nodes.clone(), RegionOption::F, true, true, 5.0).unwrap();
        let warp = ndarray::array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        r.store_flexible(warp.view()).unwrap(); // counter=1, history has 2 entries
        // update with counter=1: sets current to history[1]
        r.update("image_001.jpg");
        assert!((r.current_nodes[[0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn last_integer_in_str_basic() {
        assert_eq!(last_integer_in_str("image_042.jpg"), Some(42));
        assert_eq!(last_integer_in_str("no_digits"), None);
        assert_eq!(last_integer_in_str("a1b2c3"), Some(3));
    }
}
