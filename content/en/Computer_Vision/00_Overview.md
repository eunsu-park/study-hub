# OpenCV / Computer Vision Learning Guide

## Introduction

This folder contains systematic learning materials for computer vision using OpenCV. You can learn step-by-step from the basics of image processing to deep learning-based object detection.

**Target Audience**: Developers with Python basics, computer vision beginners, those preparing for video processing projects

---

## Learning Roadmap

```
[Basics]                 [Intermediate]             [Advanced]
  │                         │                         │
  ▼                         ▼                         ▼
Setup (01) ─────────▶ Filtering (05) ──────▶ Feature Detection (13)
  │                         │                         │
  ▼                         ▼                         ▼
Image Basics (02) ──▶ Morphology (06) ─────▶ Feature Matching (14)
  │                         │                         │
  ▼                         ▼                         ▼
Color Spaces (03) ──▶ Thresholding (07) ───▶ Object Detection (15)
  │                         │                         │
  ▼                         ▼                         ▼
Geometric Transforms (04) ─▶ Edge Detection (08) ──▶ Face Detection (16)
                            │                         │
                            ▼                         ▼
                     Contours (09) ─────────▶ Video Processing (17)
                            │                         │
                            ▼                         ▼
                     Shape Analysis (10) ───▶ Calibration (18)
                            │                         │
                     ┌──────┴──────┐                  ▼
                     ▼             ▼           DNN Module (19)
               Hough Transform (11)  Histogram (12)        │
                                                      ▼
                                              Practical Projects (20)
```

---

## Prerequisites

### Required

- Python basics (variables, control flow, functions, classes)
- NumPy basics (ndarray, indexing, slicing, broadcasting)
- File I/O, exception handling

### Recommended

- Linear algebra basics (matrix operations)
- Probability/statistics basics
- Machine learning concepts (classification, training)

---

## File List

### Basics (01-04)

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [01_Environment_Setup.md](./01_Environment_Setup.md) | ⭐ | OpenCV installation, opencv-python vs contrib, version check |
| [02_Image_Basics.md](./02_Image_Basics.md) | ⭐ | imread, imshow, imwrite, pixel access, ROI |
| [03_Color_Spaces.md](./03_Color_Spaces.md) | ⭐⭐ | BGR/RGB, HSV, LAB, cvtColor, channel separation |
| [04_Geometric_Transforms.md](./04_Geometric_Transforms.md) | ⭐⭐ | resize, rotate, flip, warpAffine, warpPerspective |

### Image Processing (05-08)

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [05_Image_Filtering.md](./05_Image_Filtering.md) | ⭐⭐ | blur, GaussianBlur, medianBlur, bilateralFilter |
| [06_Morphology.md](./06_Morphology.md) | ⭐⭐ | erode, dilate, opening, closing, gradient |
| [07_Thresholding.md](./07_Thresholding.md) | ⭐⭐ | threshold, OTSU, adaptiveThreshold |
| [08_Edge_Detection.md](./08_Edge_Detection.md) | ⭐⭐⭐ | Sobel, Scharr, Laplacian, Canny |

### Object Analysis (09-12)

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [09_Contours.md](./09_Contours.md) | ⭐⭐⭐ | findContours, drawContours, hierarchy, approxPolyDP |
| [10_Shape_Analysis.md](./10_Shape_Analysis.md) | ⭐⭐⭐ | moments, boundingRect, convexHull, matchShapes |
| [11_Hough_Transform.md](./11_Hough_Transform.md) | ⭐⭐⭐ | HoughLines, HoughLinesP, HoughCircles |
| [12_Histogram_Analysis.md](./12_Histogram_Analysis.md) | ⭐⭐ | calcHist, equalizeHist, CLAHE, backprojection |

### Features and Detection (13-15)

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [13_Feature_Detection.md](./13_Feature_Detection.md) | ⭐⭐⭐ | Harris, FAST, SIFT, ORB, keypoints/descriptors |
| [14_Feature_Matching.md](./14_Feature_Matching.md) | ⭐⭐⭐ | BFMatcher, FLANN, ratio test, homography |
| [15_Object_Detection_Basics.md](./15_Object_Detection_Basics.md) | ⭐⭐⭐ | template matching, Haar cascade, HOG+SVM |

### Advanced Topics (16-18)

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [16_Face_Detection.md](./16_Face_Detection.md) | ⭐⭐⭐⭐ | Haar/dlib face detection, landmarks, LBPH, face_recognition |
| [17_Video_Processing.md](./17_Video_Processing.md) | ⭐⭐⭐ | VideoCapture, VideoWriter, background subtraction, optical flow |
| [18_Camera_Calibration.md](./18_Camera_Calibration.md) | ⭐⭐⭐⭐ | camera matrix, distortion correction, chessboard calibration |

### DNN and Practice (19-20)

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [19_DNN_Module.md](./19_DNN_Module.md) | ⭐⭐⭐⭐ | cv2.dnn, readNet, blobFromImage, YOLO, SSD |
| [20_Practical_Projects.md](./20_Practical_Projects.md) | ⭐⭐⭐⭐ | document scanner, lane detection, AR marker, face filter |

### 3D Vision (21-23)

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [21_3D_Vision_Basics.md](./21_3D_Vision_Basics.md) | ⭐⭐⭐ | stereo vision, depth map, point cloud, 3D reconstruction |
| [22_Depth_Estimation.md](./22_Depth_Estimation.md) | ⭐⭐⭐⭐ | monocular depth estimation, MiDaS, DPT, Structure from Motion |
| [23_SLAM_Introduction.md](./23_SLAM_Introduction.md) | ⭐⭐⭐⭐ | Visual SLAM, ORB-SLAM, LiDAR SLAM, Loop Closure |

### Segmentation (25-27)

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [25_Semantic_Segmentation.md](./25_Semantic_Segmentation.md) | ⭐⭐⭐⭐ | FCN, U-Net, DeepLab, pixel-wise classification |
| [26_Instance_Segmentation.md](./26_Instance_Segmentation.md) | ⭐⭐⭐⭐ | Mask R-CNN, YOLACT, instance-level masks |
| [27_Panoptic_Segmentation.md](./27_Panoptic_Segmentation.md) | ⭐⭐⭐⭐ | Panoptic FPN, stuff + things, unified segmentation |

### Neural 3D and Novel View Synthesis (28-29)

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [28_Neural_Radiance_Fields.md](./28_Neural_Radiance_Fields.md) | ⭐⭐⭐⭐⭐ | NeRF, volumetric rendering, positional encoding |
| [29_3D_Gaussian_Splatting.md](./29_3D_Gaussian_Splatting.md) | ⭐⭐⭐⭐⭐ | 3DGS, differentiable rasterization, real-time rendering |

### Video and Data (30-32)

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [30_Video_Understanding.md](./30_Video_Understanding.md) | ⭐⭐⭐⭐ | C3D, I3D, SlowFast, Video Transformers, action recognition |
| [31_Optical_Flow.md](./31_Optical_Flow.md) | ⭐⭐⭐⭐ | Lucas-Kanade, Farneback, RAFT, FlowNet |
| [32_Synthetic_Data_Generation.md](./32_Synthetic_Data_Generation.md) | ⭐⭐⭐⭐ | Domain randomization, sim-to-real, data augmentation |

---

## Recommended Learning Sequence

### Quick Start (1 week)
```
01 → 02 → 03 → 05 → 07 → 08 → 09
```

### Basic Completion (2-3 weeks)
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12
```

### Intermediate Completion (4-5 weeks)
```
Basic Completion + 13 → 14 → 15 → 16 → 17
```

### Full Mastery (6-8 weeks)
```
Intermediate Completion + 18 → 19 → 20
```

### 3D Vision Deep Dive (2 weeks)
```
Full Mastery + 21 → 22 → 23
```

### Segmentation and Modern Topics (3-4 weeks)
```
3D Vision + 25 → 26 → 27 → 28 → 29 → 30 → 31 → 32
```

---

## Practice Environment

### Installation

```bash
# Basic installation (most features)
pip install opencv-python numpy matplotlib

# Extended installation (additional features like SIFT, SURF)
pip install opencv-contrib-python

# For face recognition
pip install dlib face_recognition

# Version check
python -c "import cv2; print(cv2.__version__)"
```

### Recommended Environment

```
- Python 3.8 or higher
- OpenCV 4.x
- IDE: VSCode, PyCharm, Jupyter Notebook
- OS: Windows, macOS, Linux all supported
```

### Example Project Structure

```
my_cv_project/
├── images/           # Input images
├── output/           # Output results
├── models/           # Trained models (Haar, DNN, etc.)
├── src/              # Source code
│   ├── preprocessing.py
│   ├── detection.py
│   └── utils.py
├── notebooks/        # Jupyter experiments
└── requirements.txt
```

---

## OpenCV Quick Function Reference

### Image I/O

| Function | Description | Example |
|----------|-------------|---------|
| `cv2.imread()` | Read image | `img = cv2.imread('image.jpg')` |
| `cv2.imshow()` | Display image | `cv2.imshow('Window', img)` |
| `cv2.imwrite()` | Save image | `cv2.imwrite('out.jpg', img)` |

### Color Conversion

| Function | Description | Example |
|----------|-------------|---------|
| `cv2.cvtColor()` | Color space conversion | `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |
| `cv2.split()` | Split channels | `b, g, r = cv2.split(img)` |
| `cv2.merge()` | Merge channels | `img = cv2.merge([b, g, r])` |

### Geometric Transforms

| Function | Description | Example |
|----------|-------------|---------|
| `cv2.resize()` | Resize | `resized = cv2.resize(img, (w, h))` |
| `cv2.rotate()` | Rotate | `rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` |
| `cv2.warpAffine()` | Affine transform | `dst = cv2.warpAffine(img, M, (w, h))` |

### Filtering

| Function | Description | Example |
|----------|-------------|---------|
| `cv2.GaussianBlur()` | Gaussian blur | `blur = cv2.GaussianBlur(img, (5, 5), 0)` |
| `cv2.Canny()` | Canny edge detection | `edges = cv2.Canny(img, 100, 200)` |
| `cv2.threshold()` | Thresholding | `_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)` |

### Contours/Shapes

| Function | Description | Example |
|----------|-------------|---------|
| `cv2.findContours()` | Find contours | `contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` |
| `cv2.drawContours()` | Draw contours | `cv2.drawContours(img, contours, -1, (0, 255, 0), 2)` |
| `cv2.boundingRect()` | Bounding rectangle | `x, y, w, h = cv2.boundingRect(contour)` |

---

## Related Resources

### Links to Other Folders

| Folder | Related Content |
|--------|-----------------|
| [Python/](../Python/00_Overview.md) | Advanced Python syntax, testing, packaging |
| [Algorithm/](../Algorithm/00_Overview.md) | Algorithms for image processing (graphs, DP) |
| [Linux/](../Linux/00_Overview.md) | Development environment, file handling |

### External Resources

- [OpenCV Official Documentation](https://docs.opencv.org/)
- [OpenCV-Python Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [PyImageSearch](https://pyimagesearch.com/) - Practical tutorials
- [Learn OpenCV](https://learnopencv.com/) - Advanced examples

---

## Learning Tips

1. **Practice-oriented**: Execute code directly and check results
2. **Visualization**: Visualize intermediate results with matplotlib
3. **Parameter tuning**: Real-time adjustment with trackbars (createTrackbar)
4. **Step-by-step processing**: Break down complex tasks into pipelines
5. **Debugging**: Check images at each step with imshow

---

## Interview Preparation Key Topics

| Topic | Key Questions |
|-------|---------------|
| Color spaces | RGB vs HSV - When to use HSV? |
| Filtering | Gaussian vs Bilateral - What's the difference? |
| Thresholding | What is the principle of Otsu's method? |
| Edge detection | What are the steps of Canny algorithm? |
| Features | SIFT vs ORB - Compare pros and cons |
| Object detection | Haar cascade vs HOG+SVM differences |
| DNN | YOLO vs SSD - Speed and accuracy trade-offs |
