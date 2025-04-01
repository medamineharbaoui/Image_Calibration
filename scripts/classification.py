import cv2 as cv
import numpy as np
import glob
import os
import pickle

# Chessboard size (inner corners) - confirmed as 10x7 for 11x8 squares
CHECKERBOARD = (10, 7)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def detect_chessboard(image_path):
    """Detect chessboard in an image."""
    img = cv.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False, image_path
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(
        gray, CHECKERBOARD,
        flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    )
    return ret, image_path

def select_images(cam3_path, cam4_path):
    """Select stereo pairs with cam3 as left and cam4 as right."""
    stereo_pairs = []  # List of (left, right) pairs

    cam3_images = sorted(glob.glob(os.path.join(cam3_path, "*_cam3.png")), key=lambda x: int(os.path.basename(x).split('_')[0]))
    cam4_images = sorted(glob.glob(os.path.join(cam4_path, "*_cam4.png")), key=lambda x: int(os.path.basename(x).split('_')[0]))

    if len(cam3_images) != len(cam4_images):
        print(f"Warning: Unequal number of images - cam3: {len(cam3_images)}, cam4: {len(cam4_images)}")

    for left_img, right_img in zip(cam3_images, cam4_images):
        left_detected, left_path = detect_chessboard(left_img)
        right_detected, right_path = detect_chessboard(right_img)

        # Only include pairs where chessboard is detected in both (per your dataset)
        if left_detected and right_detected:
            stereo_pairs.append((left_path, right_path))
        else:
            print(f"Warning: Chessboard not detected in pair: {os.path.basename(left_img)}, {os.path.basename(right_img)}")

    return stereo_pairs

if __name__ == "__main__":
    base_path = "../dataset/"
    cam3_path = os.path.join(base_path, "cam3")  # Left camera
    cam4_path = os.path.join(base_path, "cam4")  # Right camera

    stereo_pairs = select_images(cam3_path, cam4_path)

    # Print results (limited to first 10 for brevity)
    print("Stereo pairs (cam3 as left, cam4 as right):", [(os.path.basename(l), os.path.basename(r)) for l, r in stereo_pairs[:10]])
    print(f"Total stereo pairs: {len(stereo_pairs)}")

    # Save for Task 2
    with open("../outputs/stereo_pairs.pkl", "wb") as f:
        pickle.dump(stereo_pairs, f)
    print("Saved stereo pairs to 'stereo_pairs.pkl'")