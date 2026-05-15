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

        self._calibrate_camera(allCorners, allIds, imsize)
        self._extrinsic_matrix_generator()
        self.solved = True

        self._dist = self._dist.flatten()
        self.params = CalibrationParams(self._intmat, self._extmat, self._dist)

    def _read_chessboards(self, acceptance_threshold):
        import cv2

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        allCorners, allIds, accepted = [], [], []

        for path in self._calibration_images:
            frame = cv2.imread(path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if self._binary_threshold is not None:
                _, src = cv2.threshold(gray, self._binary_threshold, 255, cv2.THRESH_BINARY)
            else:
                src = gray

            arcnrs, arids, _ = cv2.aruco.detectMarkers(
                src, self._dictionary, parameters=cv2.aruco.DetectorParameters()
            )
            if not arcnrs:
                continue
            for c in arcnrs:
                cv2.cornerSubPix(gray, c, (3, 3), (-1, -1), criteria)
            ret, chcnrs, chids = cv2.aruco.interpolateCornersCharuco(
                arcnrs, arids, gray, self._board
            )
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

        h, w = imsize
        cam_init = np.array([[1000., 0., w / 2.], [0., 1000., h / 2.], [0., 0., 1.]])
        dist_init = np.zeros((5, 1))
        _, self._intmat, self._dist, self._rot, self._trans = (
            cv2.aruco.calibrateCameraCharuco(
                charucoCorners=allCorners,
                charucoIds=allIds,
                board=self._board,
                imageSize=(w, h),
                cameraMatrix=cam_init,
                distCoeffs=dist_init,
            )
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
