# Computer_Vision Examples

Executable example code for all 20 lessons in the Computer_Vision folder.

## Folder Structure

```
examples/
├── 01_environment_basics.py     # Environment setup and basics
├── 02_image_basics.py           # Image basic operations
├── 03_color_spaces.py           # Color spaces
├── 04_geometric_transforms.py   # Geometric transformations
├── 05_filtering.py              # Image filtering
├── 06_morphology.py             # Morphological operations
├── 07_thresholding.py           # Binarization and thresholding
├── 08_edge_detection.py         # Edge detection
├── 09_contours.py               # Contour detection
├── 10_shape_analysis.py         # Shape analysis
├── 11_hough_transform.py        # Hough transform
├── 12_histogram.py              # Histogram analysis
├── 13_feature_detection.py      # Feature point detection
├── 14_feature_matching.py       # Feature matching
├── 15_object_detection.py       # Object detection basics
├── 16_face_detection.py         # Face detection and recognition
├── 17_video_processing.py       # Video processing
├── 18_camera_calibration.py     # Camera calibration
├── 19_dnn_module.py             # Deep learning DNN module
└── 20_practical_project.py      # Practical projects
```

## Environment Setup

```bash
# Create virtual environment
python -m venv cv-env
source cv-env/bin/activate  # Windows: cv-env\Scripts\activate

# Install required packages
pip install opencv-python numpy matplotlib

# Extended packages (SIFT, SURF, etc.)
pip install opencv-contrib-python

# For face recognition (optional)
pip install dlib face_recognition
```

## How to Run

```bash
# Run individual examples
cd Computer_Vision/examples
python 01_environment_basics.py

# Prepare test images (if needed)
# Test images such as sample.jpg are required before running examples
# Webcam examples require a connected camera
```

## Lesson-by-Lesson Example Overview

| Lesson | Topic | Key Functions/Concepts |
|--------|-------|----------------------|
| 01 | Environment setup | `cv2.__version__`, installation check |
| 02 | Image basics | `imread`, `imshow`, `imwrite`, ROI |
| 03 | Color spaces | `cvtColor`, BGR/HSV/LAB, `split`/`merge` |
| 04 | Geometric transforms | `resize`, `rotate`, `warpAffine`, `warpPerspective` |
| 05 | Filtering | `GaussianBlur`, `medianBlur`, `bilateralFilter` |
| 06 | Morphology | `erode`, `dilate`, `morphologyEx` |
| 07 | Thresholding | `threshold`, OTSU, `adaptiveThreshold` |
| 08 | Edge detection | `Sobel`, `Laplacian`, `Canny` |
| 09 | Contours | `findContours`, `drawContours`, `approxPolyDP` |
| 10 | Shape analysis | `moments`, `boundingRect`, `convexHull` |
| 11 | Hough transform | `HoughLines`, `HoughLinesP`, `HoughCircles` |
| 12 | Histogram | `calcHist`, `equalizeHist`, CLAHE |
| 13 | Feature detection | Harris, SIFT, ORB |
| 14 | Feature matching | BFMatcher, FLANN, homography |
| 15 | Object detection | template matching, Haar cascade |
| 16 | Face detection | Haar face, dlib landmarks |
| 17 | Video processing | `VideoCapture`, background subtraction, optical flow |
| 18 | Calibration | chessboard, distortion correction |
| 19 | DNN module | `readNet`, `blobFromImage` |
| 20 | Practical projects | document scanner, lane detection |

## Preparing Test Images

The following test images are needed to run the examples:

```bash
# Generate simple test images (included in example 01)
python 01_environment_basics.py  # Auto-generates test images

# Or prepare images manually
# - sample.jpg: A general color image
# - face.jpg: An image containing a face (for example 16)
# - checkerboard.jpg: A chessboard image (for example 18)
```

## Learning Path

### Stage 1: Basics (01-04)
```
01 -> 02 -> 03 -> 04
```

### Stage 2: Image Processing (05-08)
```
05 -> 06 -> 07 -> 08
```

### Stage 3: Object Analysis (09-12)
```
09 -> 10 -> 11 -> 12
```

### Stage 4: Features/Detection (13-16)
```
13 -> 14 -> 15 -> 16
```

### Stage 5: Advanced (17-20)
```
17 -> 18 -> 19 -> 20
```

## References

- [OpenCV Official Documentation](https://docs.opencv.org/)
- [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [PyImageSearch](https://pyimagesearch.com/)
