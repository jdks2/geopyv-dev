//! Solve-path timing harness — `geopyv_dev_fresh/layer_rg_plan.md` §12.
//!
//! Ignored by default. Run:
//!
//! ```text
//! cargo test --release --test solve_timing -- --ignored --nocapture
//! ```
//!
//! Reproducible: a self-contained synthetic speckle-ish texture and an
//! analytic warp, no external images. Prints a table of
//! `(variant, N, preconditioning, workers) -> {total, mean_iters, max_iters}`
//! so a before/after diff attributes each change in §11 / §3.
//!
//! Sizes default small so an accidental non-`--ignored` run is cheap-ish;
//! override with `GEOPYV_BENCH_NODES=2000,8000,20000`.

use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;

use geopyv_dev::image::Image;
use geopyv_dev::masks::LocalMask;
use geopyv_dev::mesh::{LayerRgConfig, Mesh, Preconditioning, SeedConfig, SolveConfig};

const IMG: usize = 1200;

/// Deterministic broadband texture — several incommensurate sinusoids so
/// every subset has gradient content in both directions.
fn texture() -> Array2<f64> {
    Array2::from_shape_fn((IMG, IMG), |(y, x)| {
        let (xf, yf) = (x as f64, y as f64);
        let v = (0.63 * xf).sin() * (0.41 * yf).cos()
            + (0.27 * xf + 0.19 * yf).sin()
            + (0.11 * xf - 0.37 * yf).cos()
            + 0.5 * (0.91 * xf + 0.53 * yf).sin();
        128.0 + 35.0 * v
    })
}

/// Bilinear resample of `src` at `(x - warp_x, y - warp_y)`.
/// `warp`: `smooth` = uniform 0.6 px shift; `shearband` = that plus a
/// localised horizontal shear across a band at mid-height.
fn warped(src: &Array2<f64>, variant: &str) -> Array2<f64> {
    let (h, w) = src.dim();
    Array2::from_shape_fn((h, w), |(y, x)| {
        let (xf, yf) = (x as f64, y as f64);
        let mut ux = 0.6;
        let uy = 0.4;
        if variant == "shearband" {
            let band_c = h as f64 * 0.5;
            let band_hw = h as f64 * 0.08;
            let d = (yf - band_c) / band_hw;
            ux += 3.0 * d.clamp(-1.0, 1.0); // ±3 px across the band
        }
        let sx = xf - ux;
        let sy = yf - uy;
        let x0 = sx.floor().clamp(0.0, (w - 1) as f64) as usize;
        let y0 = sy.floor().clamp(0.0, (h - 1) as f64) as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fx = (sx - x0 as f64).clamp(0.0, 1.0);
        let fy = (sy - y0 as f64).clamp(0.0, 1.0);
        let top = src[[y0, x0]] * (1.0 - fx) + src[[y0, x1]] * fx;
        let bot = src[[y1, x0]] * (1.0 - fx) + src[[y1, x1]] * fx;
        top * (1.0 - fy) + bot * fy
    })
}

fn bench_sizes() -> Vec<usize> {
    std::env::var("GEOPYV_BENCH_NODES")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![800, 2500, 8000])
}

struct Row {
    variant: &'static str,
    n_req: usize,
    n_actual: usize,
    precond: &'static str,
    workers: usize,
    total_ms: f64,
    mean_iters: f64,
    max_iters: u32,
}

fn run_one(
    variant: &'static str,
    n_req: usize,
    precond: Preconditioning,
    workers: Option<usize>,
    ref_img: &Arc<Image>,
    tar_img: &Arc<Image>,
) -> Row {
    let margin = IMG as f64 * 0.08;
    let lo = margin;
    let hi = IMG as f64 - margin;
    let boundary = ndarray::array![[lo, lo], [hi, lo], [hi, hi], [lo, hi]];

    let mut mesh = Mesh::new(
        boundary.view(),
        false,
        &[],
        &[],
        (1.0, 5000.0),
        n_req,
        1,
        Arc::clone(ref_img),
        Arc::clone(tar_img),
    )
    .expect("mesh build");

    let cfg = SolveConfig {
        subset_order: 1,
        tolerance: 0.6,
        override_active: true,
        preconditioning: precond,
        layer_rg: LayerRgConfig { batch_factor: 4, root_rel_eps: 0.02, max_workers: workers },
        ..Default::default()
    };
    let seed = SeedConfig { coord: [IMG as f64 / 2.0, IMG as f64 / 2.0], warp: vec![0.0; 6], tolerance: 0.5 };
    let local_mask = LocalMask::circle(14).unwrap();

    let n_actual = mesh.nodes().nrows();
    let t0 = Instant::now();
    mesh.solve(&local_mask, &seed, &cfg, None).expect("solve");
    let total_ms = t0.elapsed().as_secs_f64() * 1e3;

    let sol = mesh.solution().unwrap();
    let iters: Vec<u32> = sol.iterations.iter().copied().collect();
    let mean_iters = iters.iter().map(|&i| i as f64).sum::<f64>() / iters.len() as f64;
    let max_iters = iters.iter().copied().max().unwrap_or(0);

    Row {
        variant,
        n_req,
        n_actual,
        precond: match precond {
            Preconditioning::Rg => "RG",
            Preconditioning::LayerRg => "layer-RG",
        },
        workers: workers.unwrap_or_else(rayon::current_num_threads),
        total_ms,
        mean_iters,
        max_iters,
    }
}

#[test]
#[ignore = "timing harness — run with --ignored --nocapture"]
fn solve_timing() {
    let base = texture();
    let variants: [&str; 2] = ["smooth", "shearband"];

    let mut rows: Vec<Row> = Vec::new();
    for &variant in &variants {
        let ref_img = Arc::new(Image::from_array(base.clone(), 20));
        let tar_img = Arc::new(Image::from_array(warped(&base, variant), 20));
        for &n in &bench_sizes() {
            // RG baseline.
            rows.push(run_one(variant, n, Preconditioning::Rg, None, &ref_img, &tar_img));
            // layer-RG at a few worker counts.
            for w in [Some(1usize), Some(4), None] {
                rows.push(run_one(variant, n, Preconditioning::LayerRg, w, &ref_img, &tar_img));
            }
        }
    }

    println!(
        "\n{:<10} {:>7} {:>7} {:>9} {:>7} {:>11} {:>10} {:>9}",
        "variant", "N_req", "N", "precond", "workers", "total_ms", "mean_it", "max_it"
    );
    for r in &rows {
        println!(
            "{:<10} {:>7} {:>7} {:>9} {:>7} {:>11.1} {:>10.2} {:>9}",
            r.variant, r.n_req, r.n_actual, r.precond, r.workers, r.total_ms, r.mean_iters, r.max_iters
        );
    }
    println!();
}
