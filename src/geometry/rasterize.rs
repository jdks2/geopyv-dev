//! Image-grid primitives for the zonal-masking classifier
//! (`Mesh::solve_zonal_masking_impl`): meshless rasterisation of a
//! scattered node field, Gaussian blur, Sobel, connected components and a
//! marker-controlled watershed — see
//! `geopyv_dev_fresh/solver_options_restructure.md` §4's "Stage B".
//!
//! Pure geometry/numerics: no knowledge of meshes, subsets, or strain.
//! `nearest_node_values` deliberately does not use `mesh.elements()` or any
//! shape-function/triangle point-location — only nearest-point lookup
//! against `points`, matching this project's "meshless everywhere" rule
//! (`geopyv_dev_fresh/fresh_guidance.md` rule 6).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use ndarray::{Array2, ArrayView2};

/// Index (into `points`) of the nearest node to every pixel of an
/// `image_shape` grid, row-major.
///
/// Brute-force per-pixel nearest-neighbour is `O(H*W*N)` — too slow at
/// real scale (~2*10^10 for a 2000x2000 image and 5000 nodes). This
/// bucket-grids `points` into cells sized to their own mean spacing, then
/// for each pixel searches outward ring by ring from its own cell until a
/// candidate is found, expanding just far enough to guarantee correctness
/// (a point in a ring `r` cells out can never be closer than a point
/// already found within a strictly smaller search radius, so search stops
/// growing the ring once the best candidate found so far is provably
/// closer than anything a wider ring could contain).
///
/// Panics if `points` is empty. Backs [`nearest_node_values`].
fn nearest_node_indices(points: ArrayView2<f64>, image_shape: (usize, usize)) -> Array2<usize> {
    let n = points.nrows();
    assert!(n > 0, "nearest_node_indices: no points to rasterise against");
    let (h, w) = image_shape;

    let (x_min, x_max, y_min, y_max) = {
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for i in 0..n {
            let (x, y) = (points[[i, 0]], points[[i, 1]]);
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
        (x_min, x_max, y_min, y_max)
    };

    // Cell size ~ mean node spacing over the node bounding box (padded to
    // avoid a degenerate zero-area box with very few/collinear nodes).
    let area = ((x_max - x_min).max(1.0)) * ((y_max - y_min).max(1.0));
    let cell = (area / n as f64).sqrt().max(1.0);

    let n_cols = (((x_max - x_min) / cell).ceil() as isize + 1).max(1);
    let n_rows = (((y_max - y_min) / cell).ceil() as isize + 1).max(1);
    let cell_of = |x: f64, y: f64| -> (isize, isize) {
        (
            (((x - x_min) / cell) as isize).clamp(0, n_cols - 1),
            (((y - y_min) / cell) as isize).clamp(0, n_rows - 1),
        )
    };

    let mut buckets: std::collections::HashMap<(isize, isize), Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let c = cell_of(points[[i, 0]], points[[i, 1]]);
        buckets.entry(c).or_default().push(i);
    }

    let mut out = Array2::<usize>::zeros((h, w));
    for row in 0..h {
        let py = row as f64;
        for col in 0..w {
            let px = col as f64;
            let (cx, cy) = cell_of(px, py);
            let mut best_i: Option<usize> = None;
            let mut best_d2 = f64::INFINITY;
            let mut ring: isize = 0;
            loop {
                let mut any_cell_in_range = false;
                for gy in (cy - ring)..=(cy + ring) {
                    for gx in (cx - ring)..=(cx + ring) {
                        // Only the newly-added outer shell at this ring
                        // (interior cells were already scanned at smaller
                        // rings) -- skip cells strictly inside the previous
                        // ring's box.
                        if ring > 0 && gx > cx - ring && gx < cx + ring && gy > cy - ring && gy < cy + ring {
                            continue;
                        }
                        if gx < 0 || gy < 0 || gx >= n_cols || gy >= n_rows {
                            continue;
                        }
                        any_cell_in_range = true;
                        if let Some(idxs) = buckets.get(&(gx, gy)) {
                            for &i in idxs {
                                let dx = points[[i, 0]] - px;
                                let dy = points[[i, 1]] - py;
                                let d2 = dx * dx + dy * dy;
                                if d2 < best_d2 {
                                    best_d2 = d2;
                                    best_i = Some(i);
                                }
                            }
                        }
                    }
                }
                // Stop once the best candidate found is provably closer
                // than anything a wider ring could contain (the nearest
                // possible point in ring `ring+1` is at least `ring*cell`
                // away from the query cell's own boundary).
                if best_i.is_some() && (ring as f64) * cell >= best_d2.sqrt() {
                    break;
                }
                if !any_cell_in_range && ring > n_cols.max(n_rows) {
                    break; // exhausted the whole grid (degenerate/empty case)
                }
                ring += 1;
            }
            out[[row, col]] = best_i.unwrap_or(0);
        }
    }
    out
}

/// Nearest-node `f64` value at every pixel of an `image_shape` grid
/// (`(height, width)`, row = y, col = x — matches `Image::image_gs`), for
/// rasterising a scalar node field (e.g. `gamma_max`) to a dense image
/// before pixel-grid filtering.
///
/// Panics if `points.nrows() != values.len()` or `points` is empty.
pub fn nearest_node_values(
    points: ArrayView2<f64>,
    values: &[f64],
    image_shape: (usize, usize),
) -> Array2<f64> {
    assert_eq!(points.nrows(), values.len(), "points/values length mismatch");
    nearest_node_indices(points, image_shape).mapv(|i| values[i])
}

/// BORDER_REFLECT_101 index reflection (`gfedcb|abcdefgh|gfedcba`) — maps an
/// out-of-bounds index back into `[0, len)`. Iterates for indices more than
/// one image-width out of bounds (a wide Gaussian kernel near a small
/// image).
#[inline]
fn reflect101(mut idx: isize, len: isize) -> isize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    idx = idx.rem_euclid(period);
    if idx >= len {
        period - idx
    } else {
        idx
    }
}

/// Separable Gaussian blur of a float image, `σ = sigma` pixels,
/// BORDER_REFLECT_101 edges, kernel truncated at `⌈3σ⌉`. Returns a copy
/// unchanged when `sigma <= 0`. Values are not rounded or clamped (unlike
/// `image::gaussian_blur_5x5`, which targets a `u8` image).
pub fn gaussian_blur(img: ArrayView2<f64>, sigma: f64) -> Array2<f64> {
    let (h, w) = img.dim();
    if sigma <= 0.0 || h == 0 || w == 0 {
        return img.to_owned();
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = {
        let raw: Vec<f64> = (-radius..=radius)
            .map(|x| (-(x * x) as f64 / (2.0 * sigma * sigma)).exp())
            .collect();
        let sum: f64 = raw.iter().sum();
        raw.into_iter().map(|v| v / sum).collect()
    };

    // Horizontal pass.
    let mut tmp = Array2::<f64>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let jj = reflect101(j as isize + ki as isize - radius, w as isize) as usize;
                acc += kv * img[[i, jj]];
            }
            tmp[[i, j]] = acc;
        }
    }
    // Vertical pass.
    Array2::from_shape_fn((h, w), |(i, j)| {
        let mut acc = 0.0;
        for (ki, &kv) in kernel.iter().enumerate() {
            let ii = reflect101(i as isize + ki as isize - radius, h as isize) as usize;
            acc += kv * tmp[[ii, j]];
        }
        acc
    })
}

/// Sobel gradient magnitude of a float image, `sqrt(gx^2 + gy^2)` with the
/// standard 3×3 Sobel kernels and BORDER_REFLECT_101 edges. Used to turn a
/// smoothed scalar field (e.g. a dense `gamma_max` image) into a clean
/// dense `|∇·|` image whose ridges are the interfaces — far less noisy than
/// a meshless per-node gradient over scattered points.
pub fn sobel_magnitude(img: ArrayView2<f64>) -> Array2<f64> {
    let (h, w) = img.dim();
    if h == 0 || w == 0 {
        return img.to_owned();
    }
    let at = |i: isize, j: isize| -> f64 {
        img[[
            reflect101(i, h as isize) as usize,
            reflect101(j, w as isize) as usize,
        ]]
    };
    Array2::from_shape_fn((h, w), |(r, c)| {
        let (i, j) = (r as isize, c as isize);
        let gx = (at(i - 1, j + 1) + 2.0 * at(i, j + 1) + at(i + 1, j + 1))
            - (at(i - 1, j - 1) + 2.0 * at(i, j - 1) + at(i + 1, j - 1));
        let gy = (at(i + 1, j - 1) + 2.0 * at(i + 1, j) + at(i + 1, j + 1))
            - (at(i - 1, j - 1) + 2.0 * at(i - 1, j) + at(i - 1, j + 1));
        (gx * gx + gy * gy).sqrt()
    })
}

/// Marker-controlled watershed of `surface` from `markers` (Meyer's
/// flooding). Every `markers` pixel `> 0` is a seed keeping its label;
/// every `0` pixel is flooded and joins whichever marker's flood front,
/// rising through the lowest `surface` barrier, reaches it first — so the
/// seam between two regions settles on the `surface` ridge (crest of
/// `|∇γ|`) between their markers, whatever the ridge's width.
///
/// 4-connectivity. Ties in `surface` broken by insertion order (FIFO),
/// which keeps fronts advancing evenly. Pixels unreachable from any marker
/// (only when `markers` is all-`0`) stay `0`.
pub fn watershed_from_markers(surface: ArrayView2<f64>, markers: ArrayView2<u32>) -> Array2<u32> {
    let (h, w) = surface.dim();
    let mut labels = markers.to_owned();

    // Min-heap on (barrier, seq): `barrier` is the running max of `surface`
    // along the flood path to this pixel; `seq` is a FIFO tie-breaker.
    struct Item {
        barrier: f64,
        seq: u64,
        rc: (usize, usize),
    }
    impl PartialEq for Item {
        fn eq(&self, o: &Self) -> bool {
            self.barrier == o.barrier && self.seq == o.seq
        }
    }
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Item {
        fn cmp(&self, o: &Self) -> Ordering {
            // Reversed: BinaryHeap is a max-heap, we want min barrier then
            // min seq popped first.
            o.barrier
                .partial_cmp(&self.barrier)
                .unwrap_or(Ordering::Equal)
                .then(o.seq.cmp(&self.seq))
        }
    }

    let mut heap: BinaryHeap<Item> = BinaryHeap::new();
    let mut seq: u64 = 0;
    let neigh = |r: usize, c: usize| {
        let mut out: [Option<(usize, usize)>; 4] = [None; 4];
        if r > 0 {
            out[0] = Some((r - 1, c));
        }
        if r + 1 < h {
            out[1] = Some((r + 1, c));
        }
        if c > 0 {
            out[2] = Some((r, c - 1));
        }
        if c + 1 < w {
            out[3] = Some((r, c + 1));
        }
        out
    };

    for r in 0..h {
        for c in 0..w {
            if labels[[r, c]] != 0 {
                for nb in neigh(r, c).into_iter().flatten() {
                    if labels[nb] == 0 {
                        heap.push(Item {
                            barrier: surface[[r, c]].max(surface[nb]),
                            seq,
                            rc: nb,
                        });
                        seq += 1;
                    }
                }
            }
        }
    }

    while let Some(Item { barrier, rc, .. }) = heap.pop() {
        let (r, c) = rc;
        if labels[[r, c]] != 0 {
            continue;
        }
        // Adopt the label of an already-labelled neighbour.
        let mut lab = 0;
        for nb in neigh(r, c).into_iter().flatten() {
            if labels[nb] != 0 {
                lab = labels[nb];
                break;
            }
        }
        if lab == 0 {
            continue;
        }
        labels[[r, c]] = lab;
        for nb in neigh(r, c).into_iter().flatten() {
            if labels[nb] == 0 {
                heap.push(Item {
                    barrier: barrier.max(surface[nb]),
                    seq,
                    rc: nb,
                });
                seq += 1;
            }
        }
    }
    labels
}

/// Assign every `0` (unlabelled) pixel the label of its nearest labelled
/// pixel, by multi-source breadth-first flood from all labelled pixels at
/// once (8-connectivity, so the inter-region seam sits close to the true
/// Euclidean midline rather than the axis-biased L1 one). An all-`0` input
/// is returned unchanged.
///
/// Used to close the thin "on a `|∇γ|` ridge" band left unlabelled by
/// [`label_components`] on the low-gradient regime cores — the seam between
/// two filled regions lands mid-ridge for a symmetric band.
pub fn fill_from_nearest_label(labels: ArrayView2<u32>) -> Array2<u32> {
    let (h, w) = labels.dim();
    let mut out = labels.to_owned();
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for r in 0..h {
        for c in 0..w {
            if out[[r, c]] != 0 {
                queue.push_back((r, c));
            }
        }
    }
    while let Some((r, c)) = queue.pop_front() {
        let lab = out[[r, c]];
        for dr in -1_isize..=1 {
            for dc in -1_isize..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr < 0 || nc < 0 || nr >= h as isize || nc >= w as isize {
                    continue;
                }
                let (nr, nc) = (nr as usize, nc as usize);
                if out[[nr, nc]] == 0 {
                    out[[nr, nc]] = lab;
                    queue.push_back((nr, nc));
                }
            }
        }
    }
    out
}

/// Connected-component labelling of a binary image, 4-connectivity (a
/// diagonal-only touch does not join two regions — matches the intent of
/// separating "top" from "bottom" across a band that only touches
/// diagonally at a pixel corner, the stricter and more conservative
/// choice for this masking use case). Plain BFS flood-fill, `O(H*W)`.
///
/// Returns `(labels, n_components)`: `labels[[r,c]] == 0` for every
/// `false` (background) pixel; `true` pixels get `1..=n_components`,
/// one id per connected region.
pub fn label_components(binary: ArrayView2<bool>) -> (Array2<u32>, u32) {
    let (h, w) = binary.dim();
    let mut labels = Array2::<u32>::zeros((h, w));
    let mut next_label: u32 = 0;
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    for r0 in 0..h {
        for c0 in 0..w {
            if !binary[[r0, c0]] || labels[[r0, c0]] != 0 {
                continue;
            }
            next_label += 1;
            labels[[r0, c0]] = next_label;
            queue.push_back((r0, c0));
            while let Some((r, c)) = queue.pop_front() {
                let neighbours = [
                    (r.checked_sub(1), Some(c)),
                    (Some(r + 1), Some(c)),
                    (Some(r), c.checked_sub(1)),
                    (Some(r), Some(c + 1)),
                ];
                for (nr, nc) in neighbours {
                    let (Some(nr), Some(nc)) = (nr, nc) else { continue };
                    if nr >= h || nc >= w {
                        continue;
                    }
                    if binary[[nr, nc]] && labels[[nr, nc]] == 0 {
                        labels[[nr, nc]] = next_label;
                        queue.push_back((nr, nc));
                    }
                }
            }
        }
    }
    (labels, next_label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn nearest_node_indices_two_points_splits_image_at_midline() {
        // Two nodes, one at x=1 (label 1), one at x=8 (label 2), same y.
        // Every pixel should take the label of whichever node is nearer.
        let points = array![[1.0, 5.0], [8.0, 5.0]];
        let labels = [1u8, 2u8];
        let out = nearest_node_indices(points.view(), (10, 10)).mapv(|i| labels[i]);
        assert_eq!(out[[5, 0]], 1);
        assert_eq!(out[[5, 1]], 1);
        assert_eq!(out[[5, 4]], 1); // dist to node0=3, node1=4 -> node0 nearer
        assert_eq!(out[[5, 5]], 2); // dist to node0=4, node1=3 -> node1 nearer
        assert_eq!(out[[5, 9]], 2);
    }

    #[test]
    fn nearest_node_indices_every_pixel_gets_the_true_nearest_of_three() {
        // Brute-force cross-check against a direct O(N) scan per query
        // pixel, on a less trivial 3-node, non-axis-aligned layout.
        let points = array![[2.0, 2.0], [15.0, 3.0], [8.0, 14.0]];
        let labels = [10u8, 20u8, 30u8];
        let shape = (18usize, 18usize);
        let out = nearest_node_indices(points.view(), shape).mapv(|i| labels[i]);
        for r in 0..shape.0 {
            for c in 0..shape.1 {
                let (px, py) = (c as f64, r as f64);
                let mut best_d2 = f64::INFINITY;
                for i in 0..3 {
                    let dx = points[[i, 0]] - px;
                    let dy = points[[i, 1]] - py;
                    let d2 = dx * dx + dy * dy;
                    if d2 < best_d2 {
                        best_d2 = d2;
                    }
                }
                // Accept ANY point achieving the true minimum distance, not
                // just index 0's pick on a tie -- a genuine equidistant tie
                // is legitimately resolvable either way; what must hold is
                // that the returned label's own distance equals the true
                // minimum, not that it matches one arbitrary tie-break
                // convention.
                let got_label = out[[r, c]];
                let got_is_at_min_dist = (0..3).any(|i| {
                    labels[i] == got_label && {
                        let dx = points[[i, 0]] - px;
                        let dy = points[[i, 1]] - py;
                        ((dx * dx + dy * dy) - best_d2).abs() < 1e-9
                    }
                });
                assert!(got_is_at_min_dist, "mismatch at ({r},{c}): got label {got_label}, best_d2={best_d2}");
            }
        }
    }

    #[test]
    fn label_components_top_band_bottom_gives_three_disjoint_regions() {
        // A 10x10 image: rows 0-2 "top" (true), rows 4-5 "band" (true),
        // rows 7-9 "bottom" (true), separated by false rows -- matches the
        // real straight-band use case (three spatially disjoint regions
        // from what is, before this step, only a binary elevated/not
        // classification).
        let mut img = Array2::<bool>::from_elem((10, 10), false);
        for r in 0..3 {
            for c in 0..10 {
                img[[r, c]] = true;
            }
        }
        for r in 4..6 {
            for c in 0..10 {
                img[[r, c]] = true;
            }
        }
        for r in 7..10 {
            for c in 0..10 {
                img[[r, c]] = true;
            }
        }
        let (labels, n) = label_components(img.view());
        assert_eq!(n, 3);
        let top = labels[[1, 5]];
        let band = labels[[4, 5]];
        let bottom = labels[[8, 5]];
        assert_ne!(top, 0);
        assert_ne!(band, 0);
        assert_ne!(bottom, 0);
        assert_ne!(top, band);
        assert_ne!(band, bottom);
        assert_ne!(top, bottom);
        // Background stays 0.
        assert_eq!(labels[[3, 5]], 0);
        assert_eq!(labels[[6, 5]], 0);
    }

    #[test]
    fn label_components_diagonal_touch_stays_two_regions_4_connectivity() {
        let img = array![[true, false], [false, true]];
        let (_labels, n) = label_components(img.view());
        assert_eq!(n, 2, "diagonal-only touch must not join under 4-connectivity");
    }

    #[test]
    fn label_components_empty_image_gives_zero_components() {
        let img = Array2::<bool>::from_elem((5, 5), false);
        let (labels, n) = label_components(img.view());
        assert_eq!(n, 0);
        assert!(labels.iter().all(|&v| v == 0));
    }

    #[test]
    fn nearest_node_values_recovers_a_known_linear_field() {
        // f(x, y) = 2x + y sampled at a scatter of nodes; every pixel takes
        // its nearest node's value, so the max error over the image is
        // bounded by the field's Lipschitz constant times half the node
        // spacing -- here just check a few pixels sitting right on a node.
        let points = array![[1.0, 1.0], [8.0, 1.0], [1.0, 8.0], [8.0, 8.0], [4.0, 4.0]];
        let f = |x: f64, y: f64| 2.0 * x + y;
        let values: Vec<f64> = (0..points.nrows())
            .map(|i| f(points[[i, 0]], points[[i, 1]]))
            .collect();
        let out = nearest_node_values(points.view(), &values, (10, 10));
        assert!((out[[1, 1]] - f(1.0, 1.0)).abs() < 1e-9);
        assert!((out[[8, 8]] - f(8.0, 8.0)).abs() < 1e-9);
        assert!((out[[4, 4]] - f(4.0, 4.0)).abs() < 1e-9);
    }

    #[test]
    fn gaussian_blur_preserves_a_constant_field_and_total_mass() {
        let img = Array2::<f64>::from_elem((12, 15), 3.5);
        let blurred = gaussian_blur(img.view(), 2.0);
        for &v in blurred.iter() {
            assert!((v - 3.5).abs() < 1e-9, "constant field must survive a blur");
        }
        // A single spike: blurring conserves the sum (normalised kernel,
        // reflecting borders).
        let mut spike = Array2::<f64>::zeros((21, 21));
        spike[[10, 10]] = 100.0;
        let bs = gaussian_blur(spike.view(), 1.5);
        assert!((bs.sum() - 100.0).abs() < 1e-6);
        assert!(bs[[10, 10]] < 100.0 && bs[[10, 10]] > 0.0);
    }

    #[test]
    fn gaussian_blur_zero_sigma_is_identity() {
        let img = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(gaussian_blur(img.view(), 0.0), img);
    }

    #[test]
    fn fill_from_nearest_label_partitions_the_gap_between_two_seeds() {
        // Column 0 = label 1, column 9 = label 2, columns 1..9 unlabelled.
        // After the fill every column < ~4.5 should be 1, the rest 2.
        let mut labels = Array2::<u32>::zeros((3, 10));
        for r in 0..3 {
            labels[[r, 0]] = 1;
            labels[[r, 9]] = 2;
        }
        let filled = fill_from_nearest_label(labels.view());
        assert!(filled.iter().all(|&v| v != 0), "no pixel left unlabelled");
        assert_eq!(filled[[1, 1]], 1);
        assert_eq!(filled[[1, 3]], 1);
        assert_eq!(filled[[1, 6]], 2);
        assert_eq!(filled[[1, 8]], 2);
    }

    #[test]
    fn fill_from_nearest_label_all_zero_is_a_noop() {
        let labels = Array2::<u32>::zeros((4, 4));
        assert_eq!(fill_from_nearest_label(labels.view()), labels);
    }

    #[test]
    fn watershed_seam_lands_on_the_ridge_not_the_midline() {
        // A 3-row `|∇γ|` surface: low on the left, a sharp ridge at
        // column 7, low on the right -- but the two markers are placed
        // ASYMMETRICALLY (col 0 and col 9). A nearest-label fill would put
        // the seam near col 4-5 (the midline); the watershed must put it at
        // the ridge (col 7), because that is the lowest barrier separating
        // the two floods.
        let (h, w) = (3usize, 12usize);
        let surface = Array2::from_shape_fn((h, w), |(_r, c)| if c == 7 { 10.0 } else { 0.1 });
        let mut markers = Array2::<u32>::zeros((h, w));
        for r in 0..h {
            markers[[r, 0]] = 1;
            markers[[r, 9]] = 2;
        }
        let out = watershed_from_markers(surface.view(), markers.view());
        assert!(out.iter().all(|&v| v != 0), "every pixel flooded");
        // Everything left of the ridge is region 1, everything right is 2.
        assert_eq!(out[[1, 3]], 1);
        assert_eq!(out[[1, 6]], 1);
        assert_eq!(out[[1, 8]], 2);
        // The ridge pixel itself joins one side (either is acceptable).
        assert!(matches!(out[[1, 7]], 1 | 2));
    }

    #[test]
    fn watershed_all_zero_markers_leaves_all_zero() {
        let surface = Array2::<f64>::ones((4, 4));
        let markers = Array2::<u32>::zeros((4, 4));
        assert_eq!(watershed_from_markers(surface.view(), markers.view()), markers);
    }

    #[test]
    fn sobel_magnitude_zero_on_flat_peaks_on_a_step() {
        let mut img = Array2::<f64>::zeros((5, 8));
        for r in 0..5 {
            for c in 4..8 {
                img[[r, c]] = 1.0; // step between col 3 and col 4
            }
        }
        let g = sobel_magnitude(img.view());
        assert!(g[[2, 0]].abs() < 1e-9, "flat interior -> ~0 gradient");
        assert!(g[[2, 7]].abs() < 1e-9);
        assert!(g[[2, 3]] > 1.0, "the step edge -> large gradient");
    }

    #[test]
    fn fill_from_nearest_label_three_stripe_cores_stay_separated() {
        // Mirrors the straight-band case: three horizontal low-gradient
        // cores (rows 0-1 = zone 1, rows 4-5 = zone 2, rows 8-9 = zone 3)
        // with unlabelled ridge bands between them. The fill must not merge
        // any two cores.
        let mut labels = Array2::<u32>::zeros((10, 6));
        for c in 0..6 {
            for r in 0..2 {
                labels[[r, c]] = 1;
            }
            for r in 4..6 {
                labels[[r, c]] = 2;
            }
            for r in 8..10 {
                labels[[r, c]] = 3;
            }
        }
        let filled = fill_from_nearest_label(labels.view());
        assert_eq!(filled[[0, 3]], 1);
        assert_eq!(filled[[5, 3]], 2);
        assert_eq!(filled[[9, 3]], 3);
        // Ridge row 3 is nearer the rows-4-5 core than the rows-0-1 core.
        assert_eq!(filled[[3, 3]], 2);
        // Ridge row 2 is equidistant-ish; must be 1 or 2, never 3.
        assert!(matches!(filled[[2, 3]], 1 | 2));
    }
}
