import logging
import re
import glob

import numpy as np

import geopyv_dev._geopyv_dev as _core

log = logging.getLogger(__name__)


class CalibrationParams:
    """Thin Python wrapper around the Rust CalibrationParams.

    Parameters
    ----------
    intmat : array-like (3, 3)
        Camera intrinsic matrix.
    extmat : array-like (4, 4)
        Extrinsic matrix for the chosen object plane.
    dist : array-like (5,)
        Distortion coefficients [k1, k2, p1, p2, k3].
    """

    def __init__(self, intmat, extmat, dist):
        self._inner = _core.CalibrationParams(
            np.asarray(intmat, dtype=np.float64),
            np.asarray(extmat, dtype=np.float64),
            np.asarray(dist, dtype=np.float64),
        )

    def o2i(self, objpnts):
        return self._inner.o2i(np.asarray(objpnts, dtype=np.float64))

    def i2o(self, imgpnts):
        return self._inner.i2o(np.asarray(imgpnts, dtype=np.float64))

    def modify(self, dangles=(0.0, 0.0, 0.0), centre=(0.0, 0.0)):
        """Return a new CalibrationParams with the extrinsic matrix perturbed
        by an additional rotation (axis-angle, Rodrigues form) and/or
        translation. Does not mutate this object."""
        new = CalibrationParams.__new__(CalibrationParams)
        new._inner = self._inner.modify(
            [float(a) for a in dangles], [float(c) for c in centre]
        )
        return new

    @property
    def intmat(self):
        return self._inner.intmat

    @property
    def extmat(self):
        return self._inner.extmat

    @property
    def dist(self):
        return self._inner.dist

    def __repr__(self):
        return repr(self._inner)


class Calibration:
    """Camera calibration using a ChArUco board and OpenCV.

    Parameters
    ----------
    calibration_dir : str
        Directory containing calibration images.
    common_name : str, optional
        Filename prefix for the calibration images.
    file_format : str, optional
        Image extension (default '.jpg').
    dictionary : cv2.aruco.Dictionary, optional
        ArUco dictionary to use.
    board_parameters : tuple (columns, rows, square_length, marker_length)
        ChArUco board specification.
    show : bool, optional
        Display the board pattern at initialisation.
    """

    def __init__(
        self,
        *,
        calibration_dir=".",
        common_name="",
        file_format=".jpg",
        dictionary=None,
        board_parameters=None,
        show=False,
    ):
        try:
            import cv2
            from cv2 import aruco
        except ImportError as exc:
            raise ImportError("opencv-python is required for Calibration") from exc

        if not calibration_dir.endswith("/"):
            calibration_dir += "/"
        if not file_format.startswith("."):
            file_format = "." + file_format

        self._calibration_dir = calibration_dir
        self._common_name = common_name
        self._file_format = file_format
        self._dictionary = (
            dictionary
            if dictionary is not None
            else aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        )
        self.solved = False
        self.params = None

        columns, rows, square_length, marker_length = board_parameters
        self._board = aruco.CharucoBoard(
            (columns, rows), square_length, marker_length, self._dictionary
        )
        self._objpnts = self._board.getChessboardCorners()

        if show:
            import matplotlib.pyplot as plt
            imboard = self._board.generateImage((290, 180))
            fig, ax = plt.subplots()
            ax.imshow(imboard, interpolation="nearest")
            ax.axis("off")
            plt.show()

        pattern = calibration_dir + common_name + "*" + file_format
        self._calibration_images = sorted(glob.glob(pattern))
        if not self._calibration_images:
            raise FileNotFoundError(f"No images found matching {pattern!r}")
        if len(self._calibration_images) < 10:
            log.warning(
                "%d images found; at least 10 are recommended.",
                len(self._calibration_images),
            )

        img0 = cv2.imread(self._calibration_images[0])
        self._image_size = img0.shape[:2]

    def solve(self, *, ext_id=None, binary_threshold=None, acceptance_threshold=50):
        """Run the ArUco/ChArUco calibration pipeline.

        Parameters
        ----------
        ext_id : int, optional
            Numeric suffix of the calibration image to use as the object plane.
            Defaults to the last accepted image.
        binary_threshold : float, optional
            Pixel threshold for binarisation before corner detection.
        acceptance_threshold : int, optional
            Minimum ChArUco corners for a frame to be accepted (default 50).
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required for Calibration.solve") from exc

        self._binary_threshold = binary_threshold
        self._ext_id = ext_id

        allCorners, allIds, imsize = self._read_chessboards(acceptance_threshold)
        if not allCorners:
            raise RuntimeError("Calibration failed: no usable frames detected.")
        self._all_corners = allCorners
        self._all_ids = allIds
        self._imsize = imsize

        self._calibrate_camera(allCorners, allIds, imsize)
        self._extrinsic_matrix_generator()
        self._dist = self._dist.flatten()
        self._reprojection()
        self.solved = True

        self.params = CalibrationParams(self._intmat, self._extmat, self._dist)

    def _read_chessboards(self, acceptance_threshold):
        import cv2

        # cv2.aruco.detectMarkers/interpolateCornersCharuco (free functions) were
        # removed in modern OpenCV (still present as of the original geopyv's
        # OpenCV version, gone by 4.7+); CharucoDetector.detectBoard is the
        # current replacement and does its own corner refinement internally.
        detector = cv2.aruco.CharucoDetector(self._board)
        allCorners, allIds, accepted = [], [], []

        for path in self._calibration_images:
            frame = cv2.imread(path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if self._binary_threshold is not None:
                _, src = cv2.threshold(gray, self._binary_threshold, 255, cv2.THRESH_BINARY)
            else:
                src = gray

            chcnrs, chids, _, _ = detector.detectBoard(src)
            if chcnrs is not None and chids is not None and len(chcnrs) > acceptance_threshold:
                allCorners.append(chcnrs)
                allIds.append(chids)
                accepted.append(path)

        self._accepted_images = accepted
        # Determine extrinsic image index
        if self._ext_id is not None:
            indices = [int(re.findall(r"\d+", p)[-1]) for p in accepted]
            try:
                self._index = indices.index(self._ext_id)
            except ValueError:
                log.warning("ext_id %s not found in accepted images; using last.", self._ext_id)
                self._index = len(accepted) - 1
        else:
            self._index = len(accepted) - 1

        imsize = cv2.imread(accepted[0], cv2.IMREAD_GRAYSCALE).shape if accepted else (0, 0)
        return allCorners, allIds, imsize

    def _calibrate_camera(self, allCorners, allIds, imsize):
        import cv2

        # cv2.aruco.calibrateCameraCharuco (free function) was removed in
        # modern OpenCV alongside detectMarkers/interpolateCornersCharuco;
        # board.matchImagePoints + cv2.calibrateCamera is the replacement.
        h, w = imsize
        cam_init = np.array([[1000., 0., w / 2.], [0., 1000., h / 2.], [0., 0., 1.]])
        dist_init = np.zeros((5, 1))

        obj_points, img_points = [], []
        for corners, ids in zip(allCorners, allIds):
            objp, imgp = self._board.matchImagePoints(corners, ids)
            obj_points.append(objp)
            img_points.append(imgp)

        _, self._intmat, self._dist, self._rot, self._trans = cv2.calibrateCamera(
            obj_points, img_points, (w, h), cam_init, dist_init
        )
        self._rot = np.asarray(self._rot)
        self._trans = np.asarray(self._trans)

    def _extrinsic_matrix_generator(self):
        import cv2

        n = len(self._rot)
        extmats = np.zeros((n, 4, 4))
        extmats[:, 3, 3] = 1.0
        extmats[:, :3, 3] = self._trans[:, :3].reshape(-1, 3)
        for i in range(n):
            extmats[i, :3, :3], _ = cv2.Rodrigues(self._rot[i])
        self._extmats = extmats
        self._extmat = extmats[self._index]

    def _find_objpnts(self, index):
        """Board object-space corners for the ChArUco ids detected in image `index`."""
        ids = self._all_ids[index].flatten()
        objpnts = np.ones((len(ids), 4))
        objpnts[:, :3] = self._objpnts[ids].reshape(-1, 3)
        return objpnts

    def _project(self, extmat, objpnts):
        """Project (N, 4) homogeneous object points through `extmat` plus this
        calibration's intrinsic/distortion model — same formula as
        CalibrationParams.o2i, just batched per-image with that image's own
        extrinsic matrix rather than the single selected `ext_id` pose."""
        k1, k2, p1, p2, k3 = self._dist
        X_c = extmat @ objpnts.T
        X_c = X_c / X_c[2]
        r2 = X_c[0] ** 2 + X_c[1] ** 2
        f = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        X_pp = np.ones((objpnts.shape[0], 3))
        X_pp[:, 0] = X_c[0] * f + 2 * p1 * X_c[0] * X_c[1] + p2 * (r2 + 2 * X_c[0] ** 2)
        X_pp[:, 1] = X_c[1] * f + p1 * (r2 + 2 * X_c[1] ** 2) + 2 * p2 * X_c[0] * X_c[1]
        return (self._intmat @ X_pp.T).T[:, :2]

    def _reprojection(self):
        """Reproject each accepted image's own board corners through the
        solved camera model, for use by error()."""
        self._reimgpnts = [
            self._project(self._extmats[index], self._find_objpnts(index))
            for index in range(len(self._all_corners))
        ]

    def modify(self, dangles=(0.0, 0.0, 0.0), centre=(0.0, 0.0)):
        """Perturb the solved extrinsic matrix by an additional rotation
        (axis-angle, Rodrigues form) and/or translation, replacing
        ``self.params`` with the result.

        Parameters
        ----------
        dangles : array-like (3,), optional
            Axis-angle rotation vector added to the current pose's own.
        centre : array-like (2,), optional
            Image-space point; after rotating, the translation is shifted so
            this point's object-space projection (under the new rotation)
            is added to it.
        """
        if not self.solved or self.params is None:
            raise RuntimeError("Call solve() before modify().")
        self.params = self.params.modify(dangles=dangles, centre=centre)

    def calibrate(self, obj, override=False):
        """Calibrate a Region in-place by mapping its nodes to object space.

        Parameters
        ----------
        obj : CircleRegion or PathRegion
            The region to calibrate.
        override : bool, optional
            Re-calibrate even if already marked calibrated.
        """
        if not self.solved or self.params is None:
            raise RuntimeError("Call solve() before calibrate().")

        region_types = (_core.CircleRegion, _core.PathRegion)
        if not isinstance(obj, region_types):
            raise TypeError(
                f"calibrate() only supports CircleRegion / PathRegion; got {type(obj).__name__}"
            )

        if obj.calibrated and not override:
            return

        obj.current_nodes = self.params.i2o(obj.current_nodes)
        obj.history_nodes = [self.params.i2o(n) for n in obj.history_nodes]

        c = self.params.i2o(np.array([obj.current_centre]))
        obj.current_centre = tuple(c[0].tolist())

        obj.history_centres = [
            tuple(self.params.i2o(np.array([ctr]))[0].tolist())
            for ctr in obj.history_centres
        ]
        obj.calibrated = True

    def inspect(self, image_index=0, **kwargs):
        from . import plots
        return plots.inspect_calibration(self, image_index=image_index, **kwargs)

    def visualise(self, **kwargs):
        from . import plots
        return plots.visualise_calibration(self, **kwargs)

    def contour(self, quantity="R", **kwargs):
        from . import plots
        return plots.contour_calibration(self, quantity=quantity, **kwargs)

    def error(self, quantity="R", **kwargs):
        from . import plots
        return plots.error_calibration(self, quantity=quantity, **kwargs)

    def save(self, path):
        """Save the solved calibration (camera model + diagnostic data behind
        inspect/visualise/contour/error) to a .pyv file."""
        if not self.solved:
            raise RuntimeError("Calibration has not been solved; cannot save.")
        sol = _core.CalibrationSolution(
            np.asarray(self._intmat, dtype=np.float64),
            np.asarray(self._extmat, dtype=np.float64),
            np.asarray(self._dist, dtype=np.float64),
            [np.asarray(c).reshape(-1, 2).astype(np.float64) for c in self._all_corners],
            [np.asarray(i).flatten().astype(np.int32) for i in self._all_ids],
            list(self._accepted_images),
            tuple(self._imsize),
            [np.asarray(r).astype(np.float64) for r in self._reimgpnts],
        )
        sol.save(path)

    @classmethod
    def _from_solution(cls, raw):
        """Reconstruct a Calibration from a loaded CalibrationSolution — used
        by gp.load(). The ChArUco board/dictionary setup is not persisted
        (only its outputs), so this object is ready for inspect/visualise/
        contour/error/calibrate/modify, but not for a fresh solve()."""
        obj = cls.__new__(cls)
        obj.solved = True
        obj._intmat = np.asarray(raw.intmat)
        obj._extmat = np.asarray(raw.extmat)
        obj._dist = np.asarray(raw.dist)
        obj._all_corners = [np.asarray(c).reshape(-1, 1, 2) for c in raw.corners]
        obj._all_ids = [np.asarray(i).reshape(-1, 1) for i in raw.ids]
        obj._accepted_images = list(raw.accepted_images)
        obj._imsize = tuple(raw.image_size)
        obj._reimgpnts = [np.asarray(r) for r in raw.reimgpnts]
        obj.params = CalibrationParams(obj._intmat, obj._extmat, obj._dist)
        return obj
