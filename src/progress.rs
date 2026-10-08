//! Progress reporting and cancellation for long-running solves.
//!
//! [`Mesh::solve`](crate::mesh::Mesh::solve) and
//! [`Sequence::solve`](crate::sequence::Sequence::solve) report to a
//! [`SolveProgress`] observer. Passing `None` gives the terminal progress bars
//! ([`TerminalProgress`]); a front-end (the GUI) passes its own observer to
//! show progress and to cancel.
//!
//! Every callback is made from the solve's single coordinating thread — never
//! from inside the parallel subset workers — once per subset at most, so an
//! observer costs nothing measurable against a subset solve (tens of µs).

use crate::Error;

/// Observer for a running solve. All methods have no-op defaults.
pub trait SolveProgress: Sync {
    /// A mesh solve over `n_nodes` subsets is starting (called again for
    /// each pass of a multi-pass solve, e.g. zonal masking).
    fn mesh_start(&self, _n_nodes: usize) {}
    /// One more subset has been solved and stored.
    fn node_done(&self) {}
    /// Sequence pair `pair` (0-based, of `n_pairs`) is starting; `label`
    /// names the image pair, e.g. `"img_00→img_01"`.
    fn pair_start(&self, _pair: usize, _n_pairs: usize, _label: &str) {}
    /// Polled after every subset and before every sequence pair; returning
    /// `true` stops the solve with [`Error::Cancelled`].
    fn cancelled(&self) -> bool {
        false
    }
}

/// Silent observer: no output, never cancels.
impl SolveProgress for () {}

/// Record one finished subset and honour a cancellation request.
pub(crate) fn node_done(progress: &dyn SolveProgress) -> Result<(), Error> {
    progress.node_done();
    if progress.cancelled() { Err(Error::Cancelled) } else { Ok(()) }
}

/// Terminal progress bars (indicatif): a subset bar, plus a pair bar for a
/// sequence. Cleared when dropped.
pub struct TerminalProgress {
    pairs: Option<indicatif::ProgressBar>,
    subsets: indicatif::ProgressBar,
    _multi: indicatif::MultiProgress,
}

impl TerminalProgress {
    fn style(template: &str) -> indicatif::ProgressStyle {
        indicatif::ProgressStyle::with_template(template)
            .expect("static progress template is valid")
            .progress_chars("█░")
    }

    /// One "Solving mesh" subset bar.
    pub fn mesh() -> Self {
        let multi = indicatif::MultiProgress::new();
        let subsets = multi.add(indicatif::ProgressBar::new(0));
        subsets.set_style(Self::style("  Solving mesh:  [{bar:40.cyan}] {pos}/{len} subsets  eta {eta}"));
        TerminalProgress { pairs: None, subsets, _multi: multi }
    }

    /// A "Solving sequence" pair bar above the subset bar.
    pub fn sequence(n_pairs: usize) -> Self {
        let multi = indicatif::MultiProgress::new();
        let pairs = multi.add(indicatif::ProgressBar::new(n_pairs as u64));
        pairs.set_style(Self::style("Solving sequence: [{bar:40.green}] {pos}/{len} pairs  ({msg})"));
        let subsets = multi.add(indicatif::ProgressBar::new(0));
        subsets.set_style(Self::style("  Solving mesh:  [{bar:40.cyan}] {pos}/{len} subsets  eta {eta}"));
        TerminalProgress { pairs: Some(pairs), subsets, _multi: multi }
    }
}

impl SolveProgress for TerminalProgress {
    fn mesh_start(&self, n_nodes: usize) {
        self.subsets.set_length(n_nodes as u64);
        self.subsets.set_position(0);
    }
    fn node_done(&self) {
        self.subsets.inc(1);
    }
    fn pair_start(&self, pair: usize, _n_pairs: usize, label: &str) {
        if let Some(pairs) = &self.pairs {
            pairs.set_position(pair as u64);
            pairs.set_message(label.to_string());
        }
    }
}

impl Drop for TerminalProgress {
    fn drop(&mut self) {
        if let Some(pairs) = &self.pairs {
            pairs.finish_and_clear();
        }
        self.subsets.finish_and_clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct StopAfter(usize, AtomicUsize);
    impl SolveProgress for StopAfter {
        fn node_done(&self) { self.1.fetch_add(1, Ordering::Relaxed); }
        fn cancelled(&self) -> bool { self.1.load(Ordering::Relaxed) >= self.0 }
    }

    #[test]
    fn node_done_reports_then_cancels() {
        let p = StopAfter(2, AtomicUsize::new(0));
        assert!(node_done(&p).is_ok());
        assert!(matches!(node_done(&p), Err(Error::Cancelled)));
        assert_eq!(p.1.load(Ordering::Relaxed), 2);
    }
}
