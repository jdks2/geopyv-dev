//! Image timing: the `time` input of `Subset`/`Mesh`/`Sequence`, and
//! timestamp detection from image file names.
//!
//! Every time is in seconds. A user-supplied `time` is either a scalar
//! increment (the spacing between consecutive images) or one time per image.
//! With no `time`, [`parse_filename_times`] looks for a timestamp in each
//! file stem; it is deliberately conservative, since a frame counter taken
//! for a time would silently corrupt every rate:
//!
//! 1. **date-time** — `YYYY MM DD HH MM SS[.fff…]`, each separator optional
//!    and one of `-_.T :` (`2026-10-14T13-45-02.123`, `20261014_134502123`).
//! 2. **unix epoch** — a whole digit run of 10/13/16/19 digits (s/ms/µs/ns)
//!    between 2001 and 2100.
//! 3. **time of day** — `HH MM SS[.fff…]` with *mandatory* separators
//!    (`13-45-02.123`); a midnight crossing is unwrapped.
//!
//! A bare digit run is never read as a time (it is taken to be a frame
//! counter), and a format is accepted only if it matches every file at the
//! same digit-run position with valid fields and the times are not all equal.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Error;

const NS_PER_S: i128 = 1_000_000_000;
const DAY_NS: i128 = 86_400 * NS_PER_S;

/// Where a solution's `time` came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum TimeSource {
    /// Supplied by the user (`time=`).
    Given,
    /// Detected from the image file names; holds the format's description.
    Filename(String),
    /// No time information.
    #[default]
    None,
}

impl TimeSource {
    /// Human-readable description, e.g. `"filename (date-time)"`.
    pub fn describe(&self) -> String {
        match self {
            TimeSource::Given => "given".to_string(),
            TimeSource::Filename(f) => format!("filename ({f})"),
            TimeSource::None => "none".to_string(),
        }
    }
}

/// What an increment is measured from, for plots of a multi-pair solution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum IncrementMode {
    /// From the reference image the pair was solved against.
    #[default]
    Leapfrog,
    /// From the previous target image.
    Sequential,
}

impl IncrementMode {
    pub fn name(self) -> &'static str {
        match self {
            IncrementMode::Leapfrog => "leapfrog",
            IncrementMode::Sequential => "sequential",
        }
    }
}

impl std::str::FromStr for IncrementMode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        match s {
            "leapfrog" => Ok(IncrementMode::Leapfrog),
            "sequential" => Ok(IncrementMode::Sequential),
            _ => Err(Error::InvalidInput(format!("mode must be 'leapfrog' or 'sequential', got {s:?}"))),
        }
    }
}

/// The user's `time=` argument.
#[derive(Debug, Clone, PartialEq)]
pub enum TimeInput {
    /// Seconds between consecutive images: `t_i = i * dt`.
    Increment(f64),
    /// One time per image, in seconds.
    Times(Vec<f64>),
}

/// The timestamp formats [`parse_filename_times`] recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilenameFormat {
    DateTime,
    /// Epoch with `10^exp` units per second (0 = s, 3 = ms, 6 = µs, 9 = ns).
    Epoch(u32),
    TimeOfDay,
}

impl FilenameFormat {
    pub fn describe(self) -> &'static str {
        match self {
            FilenameFormat::DateTime => "date-time",
            FilenameFormat::Epoch(0) => "unix epoch, s",
            FilenameFormat::Epoch(3) => "unix epoch, ms",
            FilenameFormat::Epoch(6) => "unix epoch, us",
            FilenameFormat::Epoch(_) => "unix epoch, ns",
            FilenameFormat::TimeOfDay => "time of day",
        }
    }

    fn is_time_of_day(self) -> bool {
        self == FilenameFormat::TimeOfDay
    }
}

/// Raw timestamps parsed from a set of file stems, in nanoseconds (since the
/// epoch, or since midnight for [`FilenameFormat::TimeOfDay`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FilenameTimes {
    pub format: FilenameFormat,
    pub raw_ns: Vec<i128>,
}

impl FilenameTimes {
    /// Seconds relative to the first entry, in the given order. Time-of-day
    /// values gain a day each time they step backwards (midnight crossing).
    pub fn relative_seconds(&self) -> Vec<f64> {
        let mut day = 0i128;
        let mut prev: Option<i128> = None;
        let unwrapped: Vec<i128> = self
            .raw_ns
            .iter()
            .map(|&t| {
                if self.format.is_time_of_day() {
                    if let Some(p) = prev {
                        if t < p {
                            day += DAY_NS;
                        }
                    }
                    prev = Some(t);
                }
                t + day
            })
            .collect();
        let t0 = unwrapped.first().copied().unwrap_or(0);
        unwrapped.iter().map(|&t| (t - t0) as f64 / NS_PER_S as f64).collect()
    }

    /// The chronological order of the entries. Time-of-day values sit on a
    /// 24 h circle, so the series is taken to start just after the largest
    /// gap (correct for any recording shorter than about half a day).
    pub fn chronological_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.raw_ns.len()).collect();
        order.sort_by_key(|&i| self.raw_ns[i]);
        if self.format.is_time_of_day() && order.len() > 1 {
            let n = order.len();
            let gap = |k: usize| {
                let a = self.raw_ns[order[k]];
                let b = self.raw_ns[order[(k + 1) % n]];
                if k + 1 == n { b + DAY_NS - a } else { b - a }
            };
            let widest = (0..n).max_by_key(|&k| gap(k)).unwrap_or(n - 1);
            order.rotate_left((widest + 1) % n);
        }
        order
    }
}

/// Detect a timestamp common to every stem — see the module docs.
pub fn parse_filename_times(stems: &[&str]) -> Option<FilenameTimes> {
    if stems.is_empty() {
        return None;
    }
    // In precedence order: most specific first.
    let parsers: [fn(&[u8], usize, usize) -> Option<(FilenameFormat, i128)>; 3] =
        [parse_datetime_at, parse_epoch_at, parse_time_of_day_at];
    'formats: for parse in parsers {
        // Matches per stem: (digit-run index, format, ns).
        let matches: Vec<Vec<(usize, FilenameFormat, i128)>> = stems
            .iter()
            .map(|s| {
                let b = s.as_bytes();
                digit_runs(b)
                    .iter()
                    .enumerate()
                    .filter_map(|(r, &(start, end))| parse(b, start, end).map(|(f, t)| (r, f, t)))
                    .collect()
            })
            .collect();
        // The leftmost run index (and sub-format) valid in every stem.
        for &(run, fmt, _) in &matches[0] {
            let picked: Option<Vec<i128>> = matches
                .iter()
                .map(|m| m.iter().find(|&&(r, f, _)| r == run && f == fmt).map(|&(_, _, t)| t))
                .collect();
            if let Some(raw_ns) = picked {
                if raw_ns.len() > 1 && raw_ns.iter().all(|&t| t == raw_ns[0]) {
                    // A constant timestamp (e.g. a session start) carries no
                    // per-image timing — try the next format.
                    continue 'formats;
                }
                return Some(FilenameTimes { format: fmt, raw_ns });
            }
        }
    }
    None
}

/// Resolve a `time=` input (or its absence) for an ordered list of images.
///
/// Returns the per-image times (seconds) and their source. With `input`
/// `None`, the file names are parsed; finding nothing is not an error.
pub fn resolve_times(paths: &[PathBuf], input: Option<&TimeInput>) -> Result<(Option<Vec<f64>>, TimeSource), Error> {
    match input {
        Some(TimeInput::Increment(dt)) => {
            if !dt.is_finite() || *dt <= 0.0 {
                return Err(Error::InvalidInput(format!("time increment must be finite and > 0, got {dt}")));
            }
            Ok((Some((0..paths.len()).map(|i| i as f64 * dt).collect()), TimeSource::Given))
        }
        Some(TimeInput::Times(t)) => {
            if t.len() != paths.len() {
                return Err(Error::InvalidInput(format!(
                    "time must have one value per image ({}), got {}", paths.len(), t.len()
                )));
            }
            if t.iter().any(|v| !v.is_finite()) {
                return Err(Error::InvalidInput("time values must be finite".to_string()));
            }
            Ok((Some(t.clone()), TimeSource::Given))
        }
        None => {
            let stems: Vec<String> = paths.iter().map(|p| file_stem(p)).collect();
            let refs: Vec<&str> = stems.iter().map(String::as_str).collect();
            Ok(match parse_filename_times(&refs) {
                Some(ft) => (Some(ft.relative_seconds()), TimeSource::Filename(ft.format.describe().to_string())),
                None => (None, TimeSource::None),
            })
        }
    }
}

/// [`resolve_times`] for an image pair: `(f_time, g_time)`. Unknown paths
/// (e.g. in-memory images) give no filename times.
pub fn resolve_pair_time(
    f_path: Option<&Path>,
    g_path: Option<&Path>,
    input: Option<&TimeInput>,
) -> Result<(Option<(f64, f64)>, TimeSource), Error> {
    let paths = match (f_path, g_path) {
        (Some(f), Some(g)) => vec![f.to_path_buf(), g.to_path_buf()],
        _ if input.is_none() => return Ok((None, TimeSource::None)),
        _ => vec![PathBuf::new(), PathBuf::new()],
    };
    let (t, source) = resolve_times(&paths, input)?;
    Ok((t.map(|t| (t[0], t[1])), source))
}

/// Sort `paths` chronologically if their names carry a timestamp; returns
/// `false` (leaving `paths` untouched) otherwise.
pub fn sort_by_filename_time(paths: &mut Vec<PathBuf>) -> bool {
    let stems: Vec<String> = paths.iter().map(|p| file_stem(p)).collect();
    let refs: Vec<&str> = stems.iter().map(String::as_str).collect();
    match parse_filename_times(&refs) {
        Some(ft) => {
            let order = ft.chronological_order();
            *paths = order.into_iter().map(|i| paths[i].clone()).collect();
            true
        }
        None => false,
    }
}

fn file_stem(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Parsers. Each takes the stem bytes and one digit run `[start, end)` and
// returns the format and time (ns) of a timestamp starting at that run.
// ---------------------------------------------------------------------------

fn digit_runs(b: &[u8]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let s = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            runs.push((s, i));
        } else {
            i += 1;
        }
    }
    runs
}

fn num(b: &[u8], pos: usize, len: usize) -> Option<i128> {
    let d = b.get(pos..pos + len)?;
    if !d.iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(d.iter().fold(0i128, |acc, c| acc * 10 + (c - b'0') as i128))
}

/// Up to 9 fraction digits as nanoseconds (extra digits truncated).
fn frac_ns(d: &[u8]) -> i128 {
    let mut ns = 0i128;
    for k in 0..9 {
        ns = ns * 10 + d.get(k).map_or(0, |c| (c - b'0') as i128);
    }
    ns
}

/// Optional fractional seconds after the seconds field ending at `pos`:
/// digits continuing the same run, `.`/`,` + digits, or `_`/`-` + exactly
/// 3, 6 or 9 digits (so a trailing `_0001` frame counter is not taken).
fn fraction_at(b: &[u8], pos: usize) -> i128 {
    let run_end = |s: usize| {
        let mut e = s;
        while e < b.len() && b[e].is_ascii_digit() {
            e += 1;
        }
        e
    };
    match b.get(pos) {
        Some(c) if c.is_ascii_digit() => frac_ns(&b[pos..run_end(pos)]),
        Some(b'.') | Some(b',') if b.get(pos + 1).is_some_and(u8::is_ascii_digit) => {
            frac_ns(&b[pos + 1..run_end(pos + 1)])
        }
        Some(b'_') | Some(b'-') => {
            let e = run_end(pos + 1);
            if matches!(e - (pos + 1), 3 | 6 | 9) { frac_ns(&b[pos + 1..e]) } else { 0 }
        }
        _ => 0,
    }
}

/// Days since 1970-01-01 (Howard Hinnant's `days_from_civil`).
fn days_from_civil(y: i128, m: i128, d: i128) -> i128 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn days_in_month(y: i128, m: i128) -> i128 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        _ => 28,
    }
}

fn hms_ns(h: i128, m: i128, s: i128) -> Option<i128> {
    (h < 24 && m < 60 && s < 60).then_some(((h * 60 + m) * 60 + s) * NS_PER_S)
}

fn parse_datetime_at(b: &[u8], start: usize, _end: usize) -> Option<(FilenameFormat, i128)> {
    const SEPS: &[u8] = b"-_.T :";
    let mut pos = start;
    let year = num(b, pos, 4)?;
    pos += 4;
    let mut fields = [0i128; 5];
    for f in fields.iter_mut() {
        if b.get(pos).is_some_and(|c| SEPS.contains(c)) {
            pos += 1;
        }
        *f = num(b, pos, 2)?;
        pos += 2;
    }
    let [mo, d, h, mi, s] = fields;
    if !(1970..=2100).contains(&year) || !(1..=12).contains(&mo) || d < 1 || d > days_in_month(year, mo) {
        return None;
    }
    let t = days_from_civil(year, mo, d) * DAY_NS + hms_ns(h, mi, s)? + fraction_at(b, pos);
    Some((FilenameFormat::DateTime, t))
}

fn parse_epoch_at(b: &[u8], start: usize, end: usize) -> Option<(FilenameFormat, i128)> {
    let exp = match end - start {
        10 => 0,
        13 => 3,
        16 => 6,
        19 => 9,
        _ => return None,
    };
    let v = num(b, start, end - start)?;
    let scale = 10i128.pow(9 - exp);
    let mut t = v * scale;
    if exp == 0 && b.get(end) == Some(&b'.') && b.get(end + 1).is_some_and(u8::is_ascii_digit) {
        t += fraction_at(b, end);
    }
    let secs = t / NS_PER_S;
    // 2001-01-01 .. 2100-01-01
    (978_307_200..4_102_444_800).contains(&secs).then_some((FilenameFormat::Epoch(exp), t))
}

fn parse_time_of_day_at(b: &[u8], start: usize, end: usize) -> Option<(FilenameFormat, i128)> {
    const SEPS: &[u8] = b"-_:.";
    if end - start != 2 {
        return None;
    }
    let mut pos = start;
    let mut fields = [0i128; 3];
    for (k, f) in fields.iter_mut().enumerate() {
        if k > 0 {
            if !b.get(pos).is_some_and(|c| SEPS.contains(c)) {
                return None;
            }
            pos += 1;
        }
        *f = num(b, pos, 2)?;
        pos += 2;
        // Each field is a whole 2-digit run.
        if k < 2 && b.get(pos).is_some_and(u8::is_ascii_digit) {
            return None;
        }
    }
    let [h, m, s] = fields;
    Some((FilenameFormat::TimeOfDay, hms_ns(h, m, s)? + fraction_at(b, pos)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(stems: &[&str]) -> Option<(FilenameFormat, Vec<f64>)> {
        parse_filename_times(stems).map(|ft| (ft.format, ft.relative_seconds()))
    }

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn datetime_separated_with_fraction() {
        let (f, t) = rel(&["img_2026-10-14T13-45-02.123", "img_2026-10-14T13-45-02.623", "img_2026-10-14T13-45-04.000"]).unwrap();
        assert_eq!(f, FilenameFormat::DateTime);
        assert!(close(&t, &[0.0, 0.5, 1.877]), "{t:?}");
    }

    #[test]
    fn datetime_compact_with_ms_in_run_and_trailing_counter() {
        // Basler-style: HHMMSSmmm in one run, then a `_0001` frame counter.
        let (f, t) = rel(&["cam__20231004_103512345_0001", "cam__20231004_103512445_0002", "cam__20231004_103513045_0003"]).unwrap();
        assert_eq!(f, FilenameFormat::DateTime);
        assert!(close(&t, &[0.0, 0.1, 0.7]), "{t:?}");
    }

    #[test]
    fn datetime_crosses_midnight_and_month() {
        let (_, t) = rel(&["2026_01_31_23_59_59", "2026_02_01_00_00_01"]).unwrap();
        assert!(close(&t, &[0.0, 2.0]), "{t:?}");
    }

    #[test]
    fn datetime_irregular_spacing() {
        let (_, t) = rel(&["a_20261014_120000", "a_20261014_120001", "a_20261014_120011"]).unwrap();
        assert!(close(&t, &[0.0, 1.0, 11.0]));
    }

    #[test]
    fn epoch_units() {
        let (f, t) = rel(&["f_1697283902", "f_1697283904"]).unwrap();
        assert_eq!(f, FilenameFormat::Epoch(0));
        assert!(close(&t, &[0.0, 2.0]));
        let (f, t) = rel(&["f_1697283902123", "f_1697283902173"]).unwrap();
        assert_eq!(f, FilenameFormat::Epoch(3));
        assert!(close(&t, &[0.0, 0.05]));
        let (f, t) = rel(&["1697283902123456789", "1697283902123457789"]).unwrap();
        assert_eq!(f, FilenameFormat::Epoch(9));
        assert!(close(&t, &[0.0, 1e-6]), "{t:?}");
    }

    #[test]
    fn time_of_day_unwraps_midnight() {
        let (f, t) = rel(&["shot_23-59-59.500", "shot_00-00-00.500", "shot_00-00-02.000"]).unwrap();
        assert_eq!(f, FilenameFormat::TimeOfDay);
        assert!(close(&t, &[0.0, 1.0, 2.5]), "{t:?}");
    }

    #[test]
    fn frame_counters_are_not_times() {
        assert!(parse_filename_times(&["frame_000001", "frame_000002", "frame_000003"]).is_none());
        assert!(parse_filename_times(&["IMG_0001", "IMG_0002"]).is_none());
        assert!(parse_filename_times(&["0000000001", "0000000002"]).is_none());
        // A 6-digit HHMMSS-looking counter, no separators.
        assert!(parse_filename_times(&["img_134502", "img_134503"]).is_none());
    }

    #[test]
    fn mixed_or_invalid_is_rejected() {
        assert!(parse_filename_times(&["2026-10-14T13-45-02", "frame_0002"]).is_none());
        assert!(parse_filename_times(&["2026-10-14T25-45-02", "2026-10-14T25-45-03"]).is_none());
        assert!(parse_filename_times(&["t_25-00-00", "t_25-00-01"]).is_none());
    }

    #[test]
    fn constant_timestamp_is_rejected() {
        assert!(parse_filename_times(&["run_20261014_120000_a", "run_20261014_120000_b"]).is_none());
    }

    #[test]
    fn chronological_order_sorts_and_rotates_time_of_day() {
        let ft = parse_filename_times(&["x_00-00-01", "x_23-59-58", "x_23-59-59"]).unwrap();
        assert_eq!(ft.chronological_order(), vec![1, 2, 0]);
        let ft = parse_filename_times(&["c_20261014_120005", "c_20261014_120001"]).unwrap();
        assert_eq!(ft.chronological_order(), vec![1, 0]);
    }

    #[test]
    fn resolve_inputs() {
        let paths: Vec<PathBuf> = ["a.png", "b.png", "c.png"].iter().map(PathBuf::from).collect();
        let (t, s) = resolve_times(&paths, Some(&TimeInput::Increment(0.1))).unwrap();
        assert!(close(&t.unwrap(), &[0.0, 0.1, 0.2]));
        assert_eq!(s, TimeSource::Given);
        let (t, _) = resolve_times(&paths, Some(&TimeInput::Times(vec![1.0, 1.5, 4.0]))).unwrap();
        assert_eq!(t.unwrap(), vec![1.0, 1.5, 4.0]);
        assert!(resolve_times(&paths, Some(&TimeInput::Times(vec![1.0, 2.0]))).is_err());
        assert!(resolve_times(&paths, Some(&TimeInput::Increment(0.0))).is_err());
        assert_eq!(resolve_times(&paths, None).unwrap(), (None, TimeSource::None));
    }
}
