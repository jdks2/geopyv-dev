//! Desktop GUI launcher — exposes `geopyv_gui::run_with` so the Python package
//! can ship the `geopyv-gui` console script without a separate binary, and
//! hands the GUI the calibration solve (`geopyv_dev.calibration.Calibration`,
//! which runs ChArUco detection and the camera fit in OpenCV).

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use geopyv_dev::calibration::CalibrationSolution;
use geopyv_gui::{CalibrationBackend, CalibrationRequest};

use crate::py_calibration::PyCalibrationSolution;

/// Solve `req` with `gp.Calibration`, as a script would.
fn solve_calibration(py: Python<'_>, req: &CalibrationRequest) -> PyResult<CalibrationSolution> {
    let module = py.import_bound("geopyv_dev.calibration")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("calibration_dir", req.dir.to_string_lossy().into_owned())?;
    kwargs.set_item("common_name", &req.common_name)?;
    kwargs.set_item("file_format", &req.file_format)?;
    kwargs.set_item("board_parameters", (req.columns, req.rows, req.square_length, req.marker_length))?;
    let cal = module.getattr("Calibration")?.call((), Some(&kwargs))?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("ext_id", req.ext_id)?;
    kwargs.set_item("binary_threshold", req.binary_threshold)?;
    kwargs.set_item("acceptance_threshold", req.acceptance_threshold)?;
    cal.call_method("solve", (), Some(&kwargs))?;
    let solution = cal.call_method0("_solution")?;
    let solution = solution.downcast::<PyCalibrationSolution>()?;
    let inner = solution.borrow().inner.clone();
    Ok(inner)
}

fn backend() -> CalibrationBackend {
    CalibrationBackend {
        available: Box::new(|| {
            Python::with_gil(|py| py.import_bound("cv2").map(|_| ()))
                .map_err(|_| "Solving calibrations needs OpenCV: pip install \"geopyv_dev[calibration]\"".to_string())
        }),
        solve: Box::new(|req| Python::with_gil(|py| solve_calibration(py, req)).map_err(|e| e.to_string())),
    }
}

/// Open the geopyv desktop GUI and block until its window is closed.
///
/// Releases the GIL while the window is open. Must be called from the main
/// thread, at most once per process. Normally launched via the
/// ``geopyv-gui`` command rather than called directly.
#[pyfunction]
pub fn run_gui(py: Python<'_>) -> PyResult<()> {
    py.allow_threads(|| geopyv_gui::run_with(Some(backend())).map_err(|e| e.to_string()))
        .map_err(|e| PyRuntimeError::new_err(format!("geopyv-gui failed: {e}")))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_gui, m)?)?;
    Ok(())
}
