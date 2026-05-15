//! PyO3 wrapper for `geopyv_dev::calibration`.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use geopyv_dev::calibration::CalibrationParams;

use crate::Error;

// ---------------------------------------------------------------------------
// CalibrationParams class
// ---------------------------------------------------------------------------

#[pyclass(name = "CalibrationParams")]
pub struct PyCalibrationParams {
    pub(crate) inner: CalibrationParams,
}

#[pymethods]
impl PyCalibrationParams {
    #[new]
    fn new(
        intmat: PyReadonlyArray2<f64>,
        extmat: PyReadonlyArray2<f64>,
        dist: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let intmat = intmat.as_array().to_owned();
        let extmat = extmat.as_array().to_owned();
        let dist_slice = dist.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("dist must be a contiguous array")
        })?;
        if dist_slice.len() != 5 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dist must have exactly 5 elements [k1, k2, p1, p2, k3]",
            ));
        }
        let dist_arr: [f64; 5] = dist_slice.try_into().unwrap();
        let inner = CalibrationParams::new(intmat, extmat, dist_arr).map_err(Error::from)?;
        Ok(Self { inner })
    }

    fn o2i<'py>(&self, py: Python<'py>, objpnts: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
        self.inner.o2i(objpnts.as_array()).into_pyarray_bound(py)
    }

    fn i2o<'py>(&self, py: Python<'py>, imgpnts: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
        self.inner.i2o(imgpnts.as_array()).into_pyarray_bound(py)
    }

    #[getter]
    fn intmat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.intmat.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn extmat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.extmat.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn dist<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.inner.dist.to_vec()).into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrationParams(fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1})",
            self.inner.intmat[[0, 0]],
            self.inner.intmat[[1, 1]],
            self.inner.intmat[[0, 2]],
            self.inner.intmat[[1, 2]],
        )
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCalibrationParams>()?;
    Ok(())
}
