import cv2 as cv
import numpy as np
import yaml
import os
import glob

def load_calibration(filename):
    """Load calibration parameters from a YAML file."""
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["distortion_coefficients"])
    return mtx, dist

def undistort_image(image_path, mtx, dist):
    """Undistort an image using camera matrix and distortion coefficients."""
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    h, w = img.shape[:2]
    new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv.undistort(img, mtx, dist, None, new_mtx)
    
    # Ensure the undistorted image matches the original size
    if undistorted.shape[:2] != (h, w):
        undistorted = cv.resize(undistorted, (w, h), interpolation=cv.INTER_LINEAR)
    
    return img, undistorted

def display_side_by_side(original, undistorted, title, scale_factor=0.5):
    """Display original and undistorted images side by side, scaled individually."""
    # Scale both images to 50% of their original size
    h, w = original.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    original_scaled = cv.resize(original, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    undistorted_scaled = cv.resize(undistorted, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    
    # Combine scaled images horizontally
    combined = np.hstack((original_scaled, undistorted_scaled))
    cv.imshow(title, combined)

if __name__ == "__main__":
    # Relative paths from scripts/ directory
    outputs_dir = "../outputs"
    dataset_dir = "../dataset"

    # Load calibration data from outputs folder
    mtx_left, dist_left = load_calibration(os.path.join(outputs_dir, "cam3_calibration.yaml"))
    mtx_right, dist_right = load_calibration(os.path.join(outputs_dir, "cam4_calibration.yaml"))

    # Find the first available image for each camera
    cam3_images = glob.glob(os.path.join(dataset_dir, "cam3", "*__cam3.png"))
    cam4_images = glob.glob(os.path.join(dataset_dir, "cam4", "*__cam4.png"))

    if not cam3_images or not cam4_images:
        raise FileNotFoundError("No images found in cam3 or cam4 folders. Check the dataset path.")

    left_test_image = cam3_images[0]  # Use the first cam3 image
    right_test_image = cam4_images[0]  # Use the first cam4 image
    print(f"Using Left Image: {left_test_image}")
    print(f"Using Right Image: {right_test_image}")

    # Undistort left image (cam3)
    print("Undistorting Left Image (cam3)...")
    orig_left, undist_left = undistort_image(left_test_image, mtx_left, dist_left)
    display_side_by_side(orig_left, undist_left, "Cam3: Original (left) vs Undistorted (right)", scale_factor=0.5)

    # Undistort right image (cam4)
    print("Undistorting Right Image (cam4)...")
    orig_right, undist_right = undistort_image(right_test_image, mtx_right, dist_right)
    display_side_by_side(orig_right, undist_right, "Cam4: Original (left) vs Undistorted (right)", scale_factor=0.5)

    # Wait for key press after both windows are displayed
    cv.waitKey(0)
    #cv.destroyAllWindows()