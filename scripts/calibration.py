import cv2 as cv
import numpy as np
import pickle
import yaml
import os

CHECKERBOARD = (10, 7)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def prepare_object_points():
    """Prepare 3D object points for a 10x7 chessboard."""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    return objp

def calibrate_camera(image_paths):
    """Calibrate a camera using a list of image paths."""
    objpoints = []
    imgpoints = []
    objp = prepare_object_points()

    for fname in image_paths:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(
            gray, CHECKERBOARD,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            objpoints.append(objp)
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)

    if not objpoints:
        raise ValueError("No valid chessboard detections found for calibration.")

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist

def save_calibration(filename, mtx, dist):
    """Save calibration parameters to a YAML file."""
    data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist()
    }
    with open(filename, 'w') as f:
        yaml.dump(data, f)

if __name__ == "__main__":
    # Load stereo pairs from Task 1
    with open("../outputs/stereo_pairs.pkl", "rb") as f:
        stereo_pairs = pickle.load(f)

    # Extract left (cam3) and right (cam4) images
    left_images = [pair[0] for pair in stereo_pairs]  # cam3
    right_images = [pair[1] for pair in stereo_pairs]  # cam4

    # Calibrate left camera (cam3)
    print("Calibrating Left Camera (cam3)...")
    ret_left, mtx_left, dist_left = calibrate_camera(left_images)
    print("Left Camera (cam3) Calibration:")
    print("Camera Matrix:\n", mtx_left)
    print("Distortion Coefficients:\n", dist_left)

    # Calibrate right camera (cam4)
    print("\nCalibrating Right Camera (cam4)...")
    ret_right, mtx_right, dist_right = calibrate_camera(right_images)
    print("Right Camera (cam4) Calibration:")
    print("Camera Matrix:\n", mtx_right)
    print("Distortion Coefficients:\n", dist_right)

    # Compare results
    print("\nComparison:")
    print("Focal Lengths (fx, fy): Left (cam3) =", mtx_left[0, 0], mtx_left[1, 1], 
          "vs Right (cam4) =", mtx_right[0, 0], mtx_right[1, 1])
    print("Principal Point (cx, cy): Left (cam3) =", mtx_left[0, 2], mtx_left[1, 2], 
          "vs Right (cam4) =", mtx_right[0, 2], mtx_right[1, 2])

    # Save calibrations
    save_calibration("../outputs/cam3_calibration.yaml", mtx_left, dist_left)
    save_calibration("../outputs/cam4_calibration.yaml", mtx_right, dist_right)
    print("\nSaved calibrations to 'cam3_calibration.yaml' and 'cam4_calibration.yaml'")