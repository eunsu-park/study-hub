# Environment Setup and Basics

## Overview

OpenCV (Open Source Computer Vision Library) is an open-source library for real-time computer vision. This document covers OpenCV installation, running your first program, and understanding the basic structure of image data.

**Difficulty**: ⭐ (Beginner)

## Learning Objectives

After completing this lesson, you will be able to:

1. Install OpenCV and configure development environment
2. Verify version and write first program
3. Understand the relationship between OpenCV and NumPy
4. Understand how images are represented as ndarrays

---

## Table of Contents

1. [Introduction to OpenCV](#1-introduction-to-opencv)
2. [Installation Methods](#2-installation-methods)
3. [Development Environment Setup](#3-development-environment-setup)
4. [Version Check and First Program](#4-version-check-and-first-program)
5. [OpenCV and NumPy Relationship](#5-opencv-and-numpy-relationship)
6. [Images are ndarrays](#6-images-are-ndarrays)
7. [Practice Problems](#7-practice-problems)
8. [Next Steps](#8-next-steps)
9. [References](#9-references)

---

## 1. Introduction to OpenCV

### What is OpenCV?

OpenCV is a computer vision library originally started by Intel and now maintained as open source.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenCV Application Areas                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │  Image      │   │  Object     │   │  Face       │          │
│   │  Processing │   │  Detection  │   │  Recognition│          │
│   │  Filtering  │   │  YOLO/SSD   │   │  Auth Systems│         │
│   │  Transform  │   │  Tracking   │   │  Emotion    │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │  Medical    │   │  Autonomous │   │  AR/VR      │          │
│   │  Imaging    │   │  Driving    │   │  Marker     │          │
│   │  CT/MRI     │   │  Lane Detect│   │  Recognition│          │
│   │  Diagnosis  │   │  Obstacles  │   │  3D Recon   │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### OpenCV Features

| Feature | Description |
|---------|-------------|
| **Cross-platform** | Supports Windows, macOS, Linux, Android, iOS |
| **Multi-language** | Bindings for C++, Python, Java, and more |
| **Real-time Processing** | Optimized algorithms for real-time video processing |
| **Rich Features** | Over 2500 optimized algorithms |
| **Active Community** | Extensive documentation, examples, and active development |

---

## 2. Installation Methods

### opencv-python vs opencv-contrib-python

```
┌────────────────────────────────────────────────────────────────┐
│                     OpenCV Python Packages                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   opencv-python                opencv-contrib-python           │
│   ┌──────────────────┐        ┌──────────────────────────┐    │
│   │  Main modules    │        │  Main modules            │    │
│   │  - core          │        │  - core                  │    │
│   │  - imgproc       │        │  - imgproc               │    │
│   │  - video         │        │  - video                 │    │
│   │  - highgui       │   ⊂    │  + Extra modules         │    │
│   │  - calib3d       │        │    - SIFT, SURF          │    │
│   │  - features2d    │        │    - xfeatures2d         │    │
│   │  - objdetect     │        │    - tracking            │    │
│   │  - dnn           │        │    - aruco               │    │
│   │  - ml            │        │    - face                │    │
│   └──────────────────┘        └──────────────────────────┘    │
│                                                                │
│   → Covers most features        → For additional algorithms   │
│   → Quick installation           → Includes patented/research │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Installation via pip

```bash
# Basic installation: covers ~95% of use cases (filtering, detection, video I/O)
# Choose this when you don't need patented algorithms like SIFT/SURF
pip install opencv-python

# contrib adds extra modules including SIFT, SURF, ArUco, and face recognition
# Use this when you need research-grade features not yet in the main module
pip install opencv-contrib-python

# Install with NumPy and matplotlib (recommended for development)
# NumPy is required at runtime; matplotlib makes Jupyter-based visualization easy
pip install opencv-python numpy matplotlib

# Install specific version (useful when pinning for reproducible environments)
pip install opencv-python==4.8.0.76

# Upgrade
pip install --upgrade opencv-python
```

**Warning**: Do not install both `opencv-python` and `opencv-contrib-python` simultaneously. This can cause conflicts.

```bash
# Wrong (causes conflicts)
pip install opencv-python opencv-contrib-python  # ✗

# Correct (choose one)
pip install opencv-contrib-python  # ✓ (contrib includes basic features)
```

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv opencv_env

# Activate (Windows)
opencv_env\Scripts\activate

# Activate (macOS/Linux)
source opencv_env/bin/activate

# Install packages
pip install opencv-contrib-python numpy matplotlib

# Deactivate
deactivate
```

---

## 3. Development Environment Setup

### VSCode Setup

```
┌─────────────────────────────────────────────────────────────┐
│                   VSCode Recommended Settings                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Essential Extensions:                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  1. Python (Microsoft)        - Python support      │   │
│   │  2. Pylance                   - Code analysis       │   │
│   │  3. Jupyter                   - Notebook support    │   │
│   │  4. Python Image Preview      - Image preview       │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   Recommended Extensions:                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  5. Image Preview             - Image file preview  │   │
│   │  6. Rainbow CSV               - CSV readability     │   │
│   │  7. GitLens                   - Git history         │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Recommended settings.json**:

```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "[python]": {
        "editor.formatOnSave": true
    }
}
```

### PyCharm Setup

1. **Create Project**: File → New Project → Pure Python
2. **Configure Interpreter**: Settings → Project → Python Interpreter
3. **Install Packages**: + button → search opencv-contrib-python → Install

### Jupyter Notebook

```bash
# Install Jupyter
pip install jupyter

# Run
jupyter notebook

# Or JupyterLab
pip install jupyterlab
jupyter lab
```

Displaying images in Jupyter:

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
# BGR → RGB conversion (matplotlib uses RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis('off')
plt.show()
```

---

## 4. Version Check and First Program

### Installation Verification

```python
import cv2
import numpy as np

# Check OpenCV version — important because APIs change between major versions
print(f"OpenCV version: {cv2.__version__}")
# Output example: OpenCV version: 4.8.0

# NumPy version matters: OpenCV relies on NumPy's array layout (C-contiguous)
print(f"NumPy version: {np.__version__}")
# Output example: NumPy version: 1.24.3

# getBuildInformation() reveals whether GPU (CUDA/OpenCL) and optimized BLAS
# libraries are available — useful for diagnosing performance bottlenecks
print(cv2.getBuildInformation())
```

### First Program: Reading and Displaying an Image

```python
import cv2

# Read image
img = cv2.imread('sample.jpg')

# Check if image was read successfully
if img is None:
    print("Cannot read image!")
else:
    print(f"Image size: {img.shape}")

    # Display image in window
    cv2.imshow('My First OpenCV', img)

    # Wait for key press (0 = wait indefinitely)
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()
```

### Testing Without an Image

```python
import cv2
import numpy as np

# np.zeros creates a black canvas: dtype=uint8 gives the [0, 255] range
# that OpenCV functions expect — using float here would cause display issues
img = np.zeros((300, 400, 3), dtype=np.uint8)

# Add text — useful as a quick smoke test without needing an image file on disk
cv2.putText(img, 'Hello OpenCV!', (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Draw circle: thickness=2 (outline); use -1 for a filled circle
cv2.circle(img, (200, 200), 50, (0, 255, 0), 2)

# Display
cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 5. OpenCV and NumPy Relationship

### NumPy-based Structure

In OpenCV-Python, images are represented as NumPy arrays (ndarray).

```
┌─────────────────────────────────────────────────────────────────┐
│               Relationship between OpenCV and NumPy             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   cv2.imread()                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────────┐          │
│   │              numpy.ndarray                       │          │
│   │  ┌─────────────────────────────────────────┐    │          │
│   │  │  shape: (height, width, channels)       │    │          │
│   │  │  dtype: uint8 (0-255)                   │    │          │
│   │  │  data: actual pixel values              │    │          │
│   │  └─────────────────────────────────────────┘    │          │
│   └─────────────────────────────────────────────────┘          │
│        │                                                        │
│        ▼                                                        │
│   NumPy operations available:                                  │
│   - Slicing: img[100:200, 50:150]                              │
│   - Operations: img + 50, img * 1.5                            │
│   - Functions: np.mean(img), np.max(img)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Example Using NumPy Operations

```python
import cv2
import numpy as np

img = cv2.imread('sample.jpg')

# Use NumPy functions
print(f"Average brightness: {np.mean(img):.2f}")
print(f"Maximum: {np.max(img)}")
print(f"Minimum: {np.min(img)}")

# np.clip prevents uint8 overflow: without it, 255 + 1 wraps around to 0
# (silent integer overflow), which creates dark spots instead of bright ones
brighter = np.clip(img + 50, 0, 255).astype(np.uint8)
darker = np.clip(img - 50, 0, 255).astype(np.uint8)

# Boolean array: True/False mask for downstream masking or counting pixels
bright_pixels = img > 200

# Statistics
print(f"Standard deviation: {np.std(img):.2f}")
```

### OpenCV Functions vs NumPy Operations

```python
import cv2
import numpy as np

img = cv2.imread('sample.jpg')

# Method 1: Using OpenCV functions
mean_cv = cv2.mean(img)
print(f"OpenCV mean: {mean_cv}")  # (B_avg, G_avg, R_avg, 0)

# Method 2: Using NumPy
mean_np = np.mean(img, axis=(0, 1))
print(f"NumPy mean: {mean_np}")  # [B_avg, G_avg, R_avg]

# Performance comparison (OpenCV is usually faster)
import time

# Gaussian blur comparison
img_large = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)

start = time.time()
blur_cv = cv2.GaussianBlur(img_large, (5, 5), 0)
print(f"OpenCV: {time.time() - start:.4f}s")
```

---

## 6. Images are ndarrays

Understanding this connection is what separates OpenCV from older image libraries: because an image is just a NumPy array, you can apply the full power of NumPy — slicing, broadcasting, vectorized math — directly to pixel data without writing a single loop.

### Image Data Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Image = 3D Array                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   img.shape = (height, width, channels)                         │
│                                                                 │
│   e.g., (480, 640, 3) → 480 rows × 640 cols × 3 channels (BGR)  │
│                                                                 │
│         width (columns, x-axis)                                 │
│       ←───────────────→                                         │
│      ┌─────────────────┐  ↑                                     │
│      │ B G R │ B G R │ │  │                                     │
│      │ pixel │ pixel │ │  │ height                              │
│      ├───────┼───────┤ │  │ (rows, y-axis)                      │
│      │ B G R │ B G R │ │  │                                     │
│      │ pixel │ pixel │ │  │                                     │
│      └─────────────────┘  ↓                                     │
│                                                                 │
│   Access: img[y, x] or img[y, x, channel]                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Types (dtype)

```python
import cv2
import numpy as np

img = cv2.imread('sample.jpg')

# Basic data type
print(f"Data type: {img.dtype}")  # uint8

# Common data types
# uint8:  0 ~ 255 (most common)
# float32: 0.0 ~ 1.0 (deep learning, precision calculations)
# float64: 0.0 ~ 1.0 (scientific computing)

# Type conversion
img_float = img.astype(np.float32) / 255.0
print(f"After conversion: {img_float.dtype}, range: {img_float.min():.2f} ~ {img_float.max():.2f}")

# Back to uint8 (for saving/display)
img_back = (img_float * 255).astype(np.uint8)
```

### Various Image Forms

```python
import cv2
import numpy as np

# Color image (3 channels)
color_img = cv2.imread('sample.jpg', cv2.IMREAD_COLOR)
print(f"Color: {color_img.shape}")  # (H, W, 3)

# Grayscale (1 channel, 2D)
gray_img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
print(f"Gray: {gray_img.shape}")  # (H, W)

# With alpha channel (4 channels)
alpha_img = cv2.imread('sample.png', cv2.IMREAD_UNCHANGED)
if alpha_img is not None and alpha_img.shape[2] == 4:
    print(f"With alpha: {alpha_img.shape}")  # (H, W, 4)

# Create new images
blank_color = np.zeros((300, 400, 3), dtype=np.uint8)  # Black color
blank_gray = np.zeros((300, 400), dtype=np.uint8)       # Black gray
white_img = np.ones((300, 400, 3), dtype=np.uint8) * 255  # White
```

### Checking Image Properties

```python
import cv2

img = cv2.imread('sample.jpg')

if img is not None:
    # Basic properties
    print(f"Shape (H, W, C): {img.shape}")
    print(f"Height: {img.shape[0]}px")
    print(f"Width: {img.shape[1]}px")
    print(f"Channels: {img.shape[2]}")

    # Data properties
    print(f"Data type: {img.dtype}")
    print(f"Total pixels: {img.size}")  # H * W * C
    print(f"Memory size: {img.nbytes} bytes")

    # Dimensions
    print(f"Dimensions: {img.ndim}")  # Color=3, Gray=2
```

---

## 7. Practice Problems

### Exercise 1: Environment Check Script

Write a script that outputs the following information:
- OpenCV version
- NumPy version
- Python version
- GPU acceleration availability (`cv2.cuda.getCudaEnabledDeviceCount()`)

```python
# Hint
import cv2
import numpy as np
import sys

# Write your code here
```

### Exercise 2: Image Info Printer

Write a function that outputs all properties of a given image file:

```python
def print_image_info(filepath):
    """
    Prints detailed information about an image file.

    Output items:
    - File path
    - Load success status
    - Image size (width x height)
    - Number of channels
    - Data type
    - Memory usage
    - Pixel value range (min, max)
    - Average brightness
    """
    # Write your code here
    pass
```

### Exercise 3: Creating Blank Canvas

Create and save images with the following conditions:

1. 800x600 black image
2. 800x600 white image
3. 800x600 red image (what is red in BGR?)
4. 400x400 checkerboard pattern (50px units)

### Exercise 4: NumPy Operations Practice

After loading an image, perform the following operations:

```python
# 1. Increase brightness by 50 (apply clipping)
# 2. Decrease brightness by 50 (apply clipping)
# 3. Increase contrast by 1.5x
# 4. Invert image (255 - img)
```

### Exercise 5: Channel Separation Preview

Write code to separate a color image into BGR channels and display each as grayscale. Use NumPy indexing.

---

## 8. Next Steps

In [02_Image_Basics.md](./02_Image_Basics.md), you'll learn basic image operations such as reading/writing images, pixel access, and ROI settings!

**Topics to Learn Next**:
- Details of `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()`
- Pixel-level access and modification
- Region of Interest (ROI) settings
- Image copying vs referencing

---

## 9. References

### Official Documentation

- [OpenCV Official Documentation](https://docs.opencv.org/)
- [OpenCV-Python Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [NumPy Official Documentation](https://numpy.org/doc/)

### Useful Links

- [PyImageSearch](https://pyimagesearch.com/) - Many practical examples
- [Learn OpenCV](https://learnopencv.com/) - Advanced tutorials
- [OpenCV GitHub](https://github.com/opencv/opencv)

### Related Learning Materials

| Folder | Related Content |
|--------|-----------------|
| [Python/](../Python/) | NumPy array operations, type hints |
| [Linux/](../Linux/) | Development environment, terminal usage |
