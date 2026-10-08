//! Bridges the core's [`SolveProgress`] observer to a tab's progress bar and
//! Cancel button.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use geopyv_dev::progress::SolveProgress;

/// Reports a core solve's progress as a fraction within `[start, end]` of a
/// tab's progress bar, and cancels when `cancel` is set.
pub struct GuiProgress<'a, F: Fn(f32, &str) + Sync> {
    report: F,
    cancel: &'a AtomicBool,
    start: f32,
    end: f32,
    n_nodes: AtomicUsize,
    done: AtomicUsize,
    pair: AtomicUsize,
    n_pairs: AtomicUsize,
    label: Mutex<String>,
}

impl<'a, F: Fn(f32, &str) + Sync> GuiProgress<'a, F> {
    pub fn new(report: F, cancel: &'a AtomicBool, start: f32, end: f32) -> Self {
        Self {
            report,
            cancel,
            start,
            end,
            n_nodes: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            pair: AtomicUsize::new(0),
            n_pairs: AtomicUsize::new(1),
            label: Mutex::new(String::new()),
        }
    }

    fn update(&self) {
        let n = self.n_nodes.load(Ordering::Relaxed).max(1);
        let done = self.done.load(Ordering::Relaxed).min(n);
        let pair = self.pair.load(Ordering::Relaxed);
        let n_pairs = self.n_pairs.load(Ordering::Relaxed).max(1);
        let frac = (pair as f32 + done as f32 / n as f32) / n_pairs as f32;
        let label = self.label.lock().map(|l| l.clone()).unwrap_or_default();
        let msg = if label.is_empty() {
            format!("Solving subset {done}/{n}\u{2026}")
        } else {
            format!("Frame {}/{n_pairs} ({label}): subset {done}/{n}\u{2026}", pair + 1)
        };
        (self.report)(self.start + (self.end - self.start) * frac, &msg);
    }
}

impl<F: Fn(f32, &str) + Sync> SolveProgress for GuiProgress<'_, F> {
    fn mesh_start(&self, n_nodes: usize) {
        self.n_nodes.store(n_nodes, Ordering::Relaxed);
        self.done.store(0, Ordering::Relaxed);
        self.update();
    }
    fn node_done(&self) {
        self.done.fetch_add(1, Ordering::Relaxed);
        self.update();
    }
    fn pair_start(&self, pair: usize, n_pairs: usize, label: &str) {
        self.pair.store(pair, Ordering::Relaxed);
        self.n_pairs.store(n_pairs, Ordering::Relaxed);
        if let Ok(mut l) = self.label.lock() {
            *l = label.to_string();
        }
        self.done.store(0, Ordering::Relaxed);
        self.update();
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}
