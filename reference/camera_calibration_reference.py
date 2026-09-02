"""Reference extracted from the supplied camera-calibration project.

Use as a behavioural example only. The implementing team owns validation,
error thresholds, calibration persistence, and all hardware acceptance tests.
"""

import glob

import cv2 as cv
import numpy as np


def collect_chessboard_points(horizontal_points_num: int, width_points_num: int, image_glob: str = "*.jpg"):
    """Return object/image points and the final grayscale frame for calibration."""
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((width_points_num * horizontal_points_num, 3), np.float32)
    objp[:, :2] = np.mgrid[0:horizontal_points_num, 0:width_points_num].T.reshape(-1, 2)

    objpoints, imgpoints, gray = [], [], None
    for filename in glob.glob(image_glob):
        image = cv.imread(filename)
        if image is None:
            continue
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, (horizontal_points_num, width_points_num), None)
        if found:
            objpoints.append(objp)
            imgpoints.append(cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))

    if gray is None or not objpoints:
        raise ValueError("No valid chessboard observations were found")
    return objpoints, imgpoints, gray


def calibration(objpoints, imgpoints, gray):
    """Reference function: returns RMS reprojection value, K, distortion, rvecs, tvecs."""
    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def undistortion_or_remapping(image_path, cam_matrix, distortion_coeff, use_remap: bool = False):
    """Apply the estimated intrinsics to one image and return its valid ROI crop."""
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"Image could not be read: {image_path}")
    height, width = image.shape[:2]
    new_matrix, roi = cv.getOptimalNewCameraMatrix(
        cam_matrix, distortion_coeff, (width, height), 1, (width, height)
    )
    if use_remap:
        map_x, map_y = cv.initUndistortRectifyMap(
            cam_matrix, distortion_coeff, None, new_matrix, (width, height), cv.CV_16SC2
        )
        undistorted = cv.remap(image, map_x, map_y, cv.INTER_LINEAR)
    else:
        undistorted = cv.undistort(image, cam_matrix, distortion_coeff, None, new_matrix)

    x, y, roi_width, roi_height = roi
    return undistorted[y:y + roi_height, x:x + roi_width]
