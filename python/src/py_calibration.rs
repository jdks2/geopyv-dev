//! PyO3 wrapper for `geopyv_dev::calibration`.

use std::path::PathBuf;

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use geopyv_dev::calibration::{CalibrationParams, CalibrationSolution};
use geopyv_dev::io::{save as io_save, GeopyvObject};

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

    /// Perturb the extrinsic matrix by an additional rotation and/or translation.
    /// Returns a new `CalibrationParams` — does not mutate `self`.
    #[pyo3(signature = (dangles=[0.0, 0.0, 0.0], centre=[0.0, 0.0]))]
    fn modify(&self, dangles: [f64; 3], centre: [f64; 2]) -> PyResult<Self> {
        let inner = self.inner.modify(dangles, centre).map_err(Error::from)?;
        Ok(Self { inner })
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
// CalibrationSolution class — serialisation boundary
// ---------------------------------------------------------------------------

/// Everything needed to reconstruct a solved `Calibration` from a `.pyv` file
/// — the numeric camera model plus the diagnostic data behind `inspect`/
/// `visualise`/`contour`/`error`. Constructed from Python (`geopyv_dev.calibration.
/// Calibration.save`) and returned by `gp.load()`; not meant to be built by hand.
#[pyclass(name = "CalibrationSolution")]
pub struct PyCalibrationSolution {
    pub(crate) inner: CalibrationSolution,
}

#[pymethods]
impl PyCalibrationSolution {
    #[new]
    fn new(
        intmat: PyReadonlyArray2<f64>,
        extmat: PyReadonlyArray2<f64>,
        dist: PyReadonlyArray1<f64>,
        corners: Vec<PyReadonlyArray2<f64>>,
        ids: Vec<PyReadonlyArray1<i32>>,
        accepted_images: Vec<String>,
        image_size: (usize, usize),
        reimgpnts: Vec<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        let dist_slice = dist.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("dist must be a contiguous array")
        })?;
        if dist_slice.len() != 5 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dist must have exactly 5 elements [k1, k2, p1, p2, k3]",
            ));
        }
        let dist_arr: [f64; 5] = dist_slice.try_into().unwrap();

        let inner = CalibrationSolution {
            intmat: intmat.as_array().to_owned(),
            extmat: extmat.as_array().to_owned(),
            dist: dist_arr,
            corners: corners.iter().map(|c| c.as_array().to_owned()).collect(),
            ids: ids.iter().map(|i| i.as_array().to_owned()).collect(),
            accepted_images: accepted_images.into_iter().map(PathBuf::from).collect(),
            image_size,
            reimgpnts: reimgpnts.iter().map(|r| r.as_array().to_owned()).collect(),
        };
        Ok(Self { inner })
    }

    /// Save to a `.pyv` file — same format/dispatch as Mesh/Sequence/Particle/Field.
    fn save(&self, path: &str) -> PyResult<()> {
        io_save(path, &GeopyvObject::Calibration(self.inner.clone())).map_err(Error::from)?;
        Ok(())
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

    #[getter]
    fn corners<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.inner.corners.iter().map(|c| c.clone().into_pyarray_bound(py)).collect()
    }

    #[getter]
    fn ids<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<i32>>> {
        self.inner.ids.iter().map(|i| i.clone().into_pyarray_bound(py)).collect()
    }

    #[getter]
    fn accepted_images(&self) -> Vec<String> {
        self.inner.accepted_images.iter().map(|p| p.display().to_string()).collect()
    }

    #[getter]
    fn image_size(&self) -> (usize, usize) {
        self.inner.image_size
    }

    #[getter]
    fn reimgpnts<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.inner.reimgpnts.iter().map(|r| r.clone().into_pyarray_bound(py)).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrationSolution(n_images={}, image_size={:?})",
            self.inner.accepted_images.len(),
            self.inner.image_size,
        )
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCalibrationParams>()?;
    m.add_class::<PyCalibrationSolution>()?;
    Ok(())
}
