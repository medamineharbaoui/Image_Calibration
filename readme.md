# Stereo Camera Calibration and Distortion Removal

## Overview
This project implements a stereo camera calibration pipeline using OpenCV in Python, as part of a computer vision lab. It processes a dataset of 100 stereo image pairs (cam3 as left, cam4 as right) with a chessboard pattern to:

### Tasks:
1. **Classify image pairs** based on chessboard visibility.
2. **Perform intrinsic calibration** for both cameras.
3. **Remove distortion** and display results.

The dataset consists of 100 pairs (0__cam3.png to 99__cam3.png and 0__cam4.png to 99__cam4.png), each containing an 11x8 chessboard (10x7 inner corners).

## Project Structure
```
Image_Calibration/
├── dataset/                  # Dataset folder (see Setup section)
├── instructions/             # Dataset folder (see Setup section)
│   └──Lab.pdf               # Instruction document
├── outputs/                  # Output files (calibration YAMLs)
│   ├── stereo_pairs.pkl      # Saved stereo pairs from Task 1
│   ├── cam3_calibration.yaml
│   └── cam4_calibration.yaml
├── scripts/                  # Python scripts
│   ├── classification.py     # Task 1: Classify image pairs
│   ├── calibration.py        # Task 2: Calibrate cameras
│   └── distortion_removal.py # Task 3: Remove distortion
├── screenshots/              # Screenshots for documentation
│   ├── selection.png         # Task 1 Results
│   ├── calibration.png       # Task 2 Results
│   ├── cam3-original-vs-undistorted.png and   # Task 3 Results
│   └── cam3-original-vs-undistorted.png       # Task 3 Results
├── requirements.txt          # Required dependencies
└── README.md                 # Project documentation
```

## Prerequisites
- **Python 3.6+**
- Dependencies listed in `requirements.txt`:
  - `opencv-python`
  - `numpy`
  - `pyyaml` (for YAML file handling)

To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Setup
### Clone the Repository:
```bash
git clone https://github.com/medamineharbaoui/Image_Calibration
cd Image_Calibration
```

### Dataset:
Download the dataset from [this link](https://drive.google.com/file/d/1vP_RkrCeSSd_fohlavI9GGCk5RTWuhTq/view) and extract it into the `dataset/` folder within the project directory. The structure should be:
```
dataset/
├── cam1/  
├── cam2/
├── cam3/  # Left camera images
├── cam4/  # Right camera images
```
If your dataset is elsewhere, update the paths in the scripts (`dataset_dir = "../dataset"`).

### Virtual Environment (Optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
### Task 1: Classify Image Pairs
Classifies stereo pairs based on chessboard visibility.
#### Script: `classification.py`
Run:
```bash
cd scripts
python classification.py
```
**Output:**
- Lists "left only," "right only," and "both visible" pairs.
- Saves "both visible" pairs to `../outputs/stereo_pairs.pkl`.

### Task 2: Calibrate Cameras
Performs intrinsic calibration for cam3 and cam4.
#### Script: `calibration.py`
Run:
```bash
python calibration.py
```
**Output:**
- Camera matrices and distortion coefficients for both cameras.
- Saves results to `../outputs/cam3_calibration.yaml` and `../outputs/cam4_calibration.yaml`.

### Task 3: Remove Distortion
Undistorts sample images and displays results.
#### Script: `distortion_removal.py`
Run:
```bash
python distortion_removal.py
```
**Output:**
- Displays original vs undistorted images for cam3 and cam4 at 50% scale.

## Results  
See screenshots folder :
- **Task 1:** All 100 pairs were "both visible," confirming the dataset’s suitability for stereo calibration.
- **Task 2:** Cam3 has a narrower field (focal lengths ~2250) and stronger distortion compared to cam4 (focal lengths ~1080).
- **Task 3:** Distortion correction was effective, especially for cam4’s fisheye effect.

## Notes
- The scripts use relative paths (`../dataset/`, `../outputs/`), assuming execution from `scripts/`.
- Adjust `scale_factor` in `distortion_removal.py` if needed.

## References
- **OpenCV Documentation:** `cv.findChessboardCorners`, `cv.calibrateCamera`, `cv.undistort`.
- **Lab Instructions:** See document under instructions folder.

## License
This project is for educational purposes.

