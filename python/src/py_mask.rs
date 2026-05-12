//! PyO3 wrapper providing a unified `Mask` Python class for local and global masks.

use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use geopyv_dev::{
    geometry::meshing::mask_image,
    masks::{LocalMask, MaskShape},
};

use crate::{py_geometry::extract_region, py_image::PyImage, Error};

// ---------------------------------------------------------------------------
// Internal discriminator
// ---------------------------------------------------------------------------

pub(crate) enum MaskInner {
    Local(LocalMask),
    Global { binary: Array2<u8> },
}

// ---------------------------------------------------------------------------
// PyMask
// ---------------------------------------------------------------------------

/// Subset mask.
///
/// Parameters
/// ----------
/// mask_type : str
///     ``"local"`` or ``"global"``.
///
/// For ``mask_type="local"``:
///     shape : str
///         ``"circle"`` or ``"square"``.
///     size : int, optional
///         Radius (circle) or half-side-length (square) in pixels. Default 25.
///
/// For ``mask_type="global"``:
///     f_img : Image
///         Reference image used to compute the binary mask.
///     boundary : CircleRegion, PathRegion, or ndarray (N, 2)
///         Boundary region defining the active area.
///     exclusions : list[CircleRegion | PathRegion | ndarray], optional
///         Exclusion regions. Default ``[]``.
///
/// Examples
/// --------
/// >>> local_mask  = Mask(mask_type="local",  shape="circle", size=30)
/// >>> global_mask = Mask(mask_type="global", f_img=f_img, boundary=boundary)
#[pyclass(name = "Mask")]
pub struct PyMask {
    pub(crate) inner: MaskInner,
}

impl PyMask {
    pub(crate) fn from_local(mask: LocalMask) -> Self {
        PyMask { inner: MaskInner::Local(mask) }
    }

    /// Return a shared reference to the `LocalMask`, or a `PyErr` if this is a global mask.
    pub(crate) fn local_mask_ref(&self) -> Result<&LocalMask, PyErr> {
        match &self.inner {
            MaskInner::Local(lm) => Ok(lm),
            MaskInner::Global { .. } => Err(PyTypeError::new_err(
                "expected a local mask (mask_type='local'), got a global mask",
            )),
        }
    }

    /// Return a view of the global binary mask, or `None` if this is a local mask.
    pub(crate) fn global_view(&self) -> Option<ArrayView2<'_, u8>> {
        match &self.inner {
            MaskInner::Global { binary } => Some(binary.view()),
            MaskInner::Local(_) => None,
        }
    }
}

#[pymethods]
impl PyMask {
    #[new]
    #[pyo3(signature = (mask_type, shape=None, size=25, f_img=None, boundary=None, exclusions=None))]
    fn new(
        _py: Python<'_>,
        mask_type: &str,
        shape: Option<&str>,
        size: usize,
        f_img: Option<&Bound<'_, PyImage>>,
        boundary: Option<&Bound<'_, PyAny>>,
        exclusions: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Self> {
        match mask_type {
            "local" => {
                let shape_str = shape.ok_or_else(|| {
                    PyValueError::new_err("shape is required for mask_type='local'")
                })?;
                let lm = match shape_str {
                    "circle" => LocalMask::circle(size).map_err(Error::from)?,
                    "square" => LocalMask::square(size).map_err(Error::from)?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown shape '{}': expected 'circle' or 'square'",
                            other
                        )))
                    }
                };
                Ok(PyMask { inner: MaskInner::Local(lm) })
            }
            "global" => {
                let f_img_bound = f_img.ok_or_else(|| {
                    PyValueError::new_err("f_img is required for mask_type='global'")
                })?;
                let boundary_obj = boundary.ok_or_else(|| {
                    PyValueError::new_err("boundary is required for mask_type='global'")
                })?;
                let (boundary_nodes, boundary_hard) = extract_region(boundary_obj)?;
                let excl_owned: Vec<Array2<f64>> = exclusions
                    .unwrap_or_default()
                    .iter()
                    .map(|obj| extract_region(obj).map(|(nodes, _)| nodes))
                    .collect::<PyResult<Vec<_>>>()?;
                let excl_views: Vec<_> = excl_owned.iter().map(|a| a.view()).collect();
                let excl_hard: Vec<bool> = vec![true; excl_owned.len()];
                let f_ref = f_img_bound.borrow();
                let img_shape = f_ref.inner.image_gs.dim();
                let binary = mask_image(
                    img_shape,
                    boundary_nodes.view(),
                    boundary_hard,
                    &excl_views,
                    &excl_hard,
                );
                Ok(PyMask { inner: MaskInner::Global { binary } })
            }
            other => Err(PyValueError::new_err(format!(
                "unknown mask_type '{}': expected 'local' or 'global'",
                other
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // Discriminator
    // -----------------------------------------------------------------------

    /// ``"local"`` or ``"global"``.
    #[getter]
    fn mask_type(&self) -> &str {
        match &self.inner {
            MaskInner::Local(_) => "local",
            MaskInner::Global { .. } => "global",
        }
    }

    // -----------------------------------------------------------------------
    // Local-mask getters (None for global masks)
    // -----------------------------------------------------------------------

    /// ``"circle"`` or ``"square"`` for local masks; ``None`` for global.
    #[getter]
    fn shape(&self) -> Option<&str> {
        match &self.inner {
            MaskInner::Local(lm) => match lm.shape {
                MaskShape::Circle => Some("circle"),
                MaskShape::Square => Some("square"),
            },
            MaskInner::Global { .. } => None,
        }
    }

    /// ``"radius"`` (circle) or ``"length"`` (square) for local masks; ``None`` for global.
    #[getter]
    fn dimension(&self) -> Option<&str> {
        match &self.inner {
            MaskInner::Local(lm) => Some(&lm.dimension),
            MaskInner::Global { .. } => None,
        }
    }

    /// Radius / half-side-length in pixels for local masks; ``None`` for global.
    #[getter]
    fn size(&self) -> Option<usize> {
        match &self.inner {
            MaskInner::Local(lm) => Some(lm.size),
            MaskInner::Global { .. } => None,
        }
    }

    /// Number of active pixels (unmasked) for local masks; ``None`` for global.
    #[getter]
    fn n_px(&self) -> Option<usize> {
        match &self.inner {
            MaskInner::Local(lm) => Some(lm.n_px),
            MaskInner::Global { .. } => None,
        }
    }

    /// Pixel offset coordinates ``(n_px, 2)`` for local masks; ``None`` for global.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        match &self.inner {
            MaskInner::Local(lm) => Some(lm.coords.clone().into_pyarray_bound(py)),
            MaskInner::Global { .. } => None,
        }
    }

    /// Binary subset mask ``((2*size+1), (2*size+1))`` for local masks; ``None`` for global.
    #[getter]
    fn subset_mask<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<i32>>> {
        match &self.inner {
            MaskInner::Local(lm) => Some(lm.subset_mask.clone().into_pyarray_bound(py)),
            MaskInner::Global { .. } => None,
        }
    }

    /// Pixels remaining after the most recent :meth:`mask` call (local only).
    #[getter]
    fn m_n_px(&self) -> Option<usize> {
        match &self.inner {
            MaskInner::Local(lm) => lm.m_n_px,
            MaskInner::Global { .. } => None,
        }
    }

    // -----------------------------------------------------------------------
    // Global-mask getters (None for local masks)
    // -----------------------------------------------------------------------

    /// Binary image mask ``(H, W)`` for global masks; ``None`` for local.
    #[getter]
    fn binary<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<u8>>> {
        match &self.inner {
            MaskInner::Global { binary } => Some(binary.clone().into_pyarray_bound(py)),
            MaskInner::Local(_) => None,
        }
    }

    // -----------------------------------------------------------------------
    // Methods
    // -----------------------------------------------------------------------

    /// Apply a binary image mask to a local mask, updating ``coords`` and ``m_n_px``.
    ///
    /// Parameters
    /// ----------
    /// centre : array-like [x, y]
    ///     Centre coordinates (integer pixel).
    /// mask : np.ndarray (H, W), dtype uint8
    ///     Binary image mask: 0 = inactive, 1 = active.
    fn mask(&mut self, centre: [f64; 2], mask: PyReadonlyArray2<u8>) -> PyResult<()> {
        match &mut self.inner {
            MaskInner::Local(lm) => {
                lm.mask_update(centre, mask.as_array());
                Ok(())
            }
            MaskInner::Global { .. } => Err(PyTypeError::new_err(
                "mask() can only be called on local masks",
            )),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            MaskInner::Local(lm) => {
                let shape = match lm.shape {
                    MaskShape::Circle => "circle",
                    MaskShape::Square => "square",
                };
                format!("Mask(mask_type='local', shape='{}', size={})", shape, lm.size)
            }
            MaskInner::Global { binary } => {
                let (h, w) = binary.dim();
                format!("Mask(mask_type='global', shape=({}, {}))", h, w)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMask>()?;
    Ok(())
}
