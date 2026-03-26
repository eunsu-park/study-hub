# Feature Detection

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what feature points are and define the properties that make a good keypoint (repeatability, distinctiveness, locality)
2. Implement Harris corner detection and interpret the corner response function
3. Apply FAST, SIFT, and ORB detectors to find keypoints in images using OpenCV
4. Compare the speed, accuracy, and patent/licensing trade-offs among Harris, FAST, SIFT, and ORB
5. Extract and visualize keypoint descriptors and understand how they encode local image structure
6. Evaluate detector performance by analyzing keypoint repeatability across transformed versions of an image

---

## Overview

Features are unique and repeatable points that can be detected in an image. These include corners, blobs, edge intersections, and more. They are used in image matching, object recognition, 3D reconstruction, and other applications. In this lesson, we will learn about various feature detection algorithms including Harris, FAST, SIFT, and ORB.

---

## Table of Contents

1. [Feature Point Fundamentals](#1-feature-point-fundamentals)
2. [Corner Detection - Harris](#2-corner-detection---harris)
3. [Good Features to Track](#3-good-features-to-track)
4. [FAST Detector](#4-fast-detector)
5. [SIFT Detector](#5-sift-detector)
6. [ORB Detector](#6-orb-detector)
7. [Keypoints and Descriptors](#7-keypoints-and-descriptors)
8. [Practice Problems](#8-practice-problems)

---

## 1. Feature Point Fundamentals

### What are Feature Points?

```
Feature Point / Keypoint:
A uniquely identifiable point in an image

Requirements for Good Features:
1. Repeatability: Same object should produce same features
2. Distinctiveness: Different features should be distinguishable
3. Invariance: Robust to rotation, scale, and lighting changes
4. Accuracy: Precise location detection

Types of Features:
+-------------------------------------------------------------+
|  Corner                    Blob                              |
|                                                              |
|       +------              *****                             |
|       |                    *******                           |
|    ---+                    ********                          |
|                            *******                           |
|   Change in two            *****                             |
|   directions               Specific region size              |
+-------------------------------------------------------------+
```

### Feature Detection Pipeline

```
1. Feature Detection
   - Find keypoint locations in image
   - Harris, FAST, SIFT, ORB, etc.
         |
         v
2. Feature Description
   - Generate feature vector around each keypoint
   - SIFT descriptor, ORB descriptor, BRIEF, etc.
         |
         v
3. Feature Matching
   - Compare descriptors with other images
   - BFMatcher, FLANN, etc.
```

### Detector Comparison

```
+----------------+-----------+-----------+-----------+----------+
| Algorithm      | Speed     | Rotation  | Scale     | Patent   |
|                |           | Invariant | Invariant |          |
+----------------+-----------+-----------+-----------+----------+
| Harris         | Fast      | O         | X         | None     |
| FAST           | Very Fast | X         | X         | None     |
| SIFT           | Slow      | O         | O         | Expired  |
| SURF           | Medium    | O         | O         | Yes      |
| ORB            | Fast      | O         | O         | None     |
| AKAZE          | Medium    | O         | O         | None     |
+----------------+-----------+-----------+-----------+----------+
```

---

## 2. Corner Detection - Harris

### Concept

```
Harris Corner Detection:
Analyzes intensity changes when shifting an image patch

- Flat region: No change in any direction
- Edge: No change along edge direction, large change perpendicular
- Corner: Large change in all directions

Auto-correlation Matrix M:
M = sum [Ix^2    IxIy]
        [IxIy   Iy^2 ]

Ix, Iy: Derivatives in x, y directions

Corner Response Function:
R = det(M) - k * (trace(M))^2
R = lambda1*lambda2 - k(lambda1 + lambda2)^2

- R > threshold: Corner
- R ~ 0: Flat
- R < 0: Edge
```

**Geometric intuition**: The eigenvalues λ1 and λ2 of M represent the two principal directions of intensity change around a pixel. At a **corner**, the image changes strongly in all directions, so both eigenvalues are large — making det(M) = λ1·λ2 large and R positive. At an **edge**, intensity changes in only one direction, so one eigenvalue is large and the other small — making det(M) small but trace(M) still large, so R goes negative. At a **flat region**, neither eigenvalue is significant, so R is near zero. The parameter k (typically 0.04–0.06) controls sensitivity: smaller k detects more corners but also more false positives.

### cv2.cornerHarris()

```python
import cv2
import numpy as np

def harris_corner_detection(image_path):
    """Harris corner detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Harris corner detection
    dst = cv2.cornerHarris(
        gray,
        blockSize=2,     # Small patch (2px) captures fine structure; larger values
                         # smooth over detail and miss tight corners
        ksize=3,         # Sobel kernel size: 3 is the smallest that gives a
                         # good gradient estimate without excessive blurring
        k=0.04           # Harris sensitivity: 0.04–0.06 is the empirically safe
                         # range — smaller k detects more corners but adds false
                         # positives; larger k misses weak corners
    )

    # Dilate result to expand corner peaks so they are visible as distinct
    # bright regions — without dilation, corners appear as single-pixel dots
    dst = cv2.dilate(dst, None)

    # Threshold at 1% of peak response: scales automatically to image content
    # so the same code works on both high-contrast and low-contrast images
    result = img.copy()
    result[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('Harris Corners', result)
    cv2.waitKey(0)

    return dst

harris_corner_detection('chessboard.jpg')
```

### Sub-pixel Accuracy

```python
import cv2
import numpy as np

def harris_subpixel(image_path):
    """Harris corners with sub-pixel accuracy"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    # Harris corners — same parameters as above (blockSize=2, ksize=3, k=0.04)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)

    # Dilate then threshold: dilation merges nearby high-response pixels into
    # connected blobs, making connectedComponentsWithStats find cleaner centroids
    dst = cv2.dilate(dst, None)
    ret, dst_thresh = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst_thresh = np.uint8(dst_thresh)

    # Find corner centroids using connected components
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_thresh)

    # Refine to sub-pixel precision
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(
        gray,
        np.float32(centroids),
        (5, 5),      # Window size
        (-1, -1),    # Zero zone
        criteria
    )

    result = img.copy()
    for i, corner in enumerate(corners):
        x, y = corner.ravel()
        if i == 0:  # First one is background
            continue
        cv2.circle(result, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('SubPixel Corners', result)
    cv2.waitKey(0)

    return corners

harris_subpixel('chessboard.jpg')
```

---

## 3. Good Features to Track

### cv2.goodFeaturesToTrack()

```
Shi-Tomasi Corner Detection (Harris improvement):
R = min(lambda1, lambda2)

More stable corner detection than Harris
-> Suitable for optical flow tracking
```

```python
import cv2
import numpy as np

def good_features_demo(image_path):
    """Good features detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect good features
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,     # Cap output count — unlimited corners overwhelm
                            # downstream trackers (e.g., Lucas-Kanade)
        qualityLevel=0.01,  # Accept corners with response >= 1% of the strongest;
                            # low value keeps weaker but still useful corners
        minDistance=10,     # Enforce spatial spread: prevents clusters of
                            # redundant corners from the same region
        blockSize=3,        # Slightly larger patch than Harris (2) for more
                            # stable gradient estimates in noisy images
        useHarrisDetector=False,  # Use Shi-Tomasi (min eigenvalue) — it avoids
                                  # the k trade-off and is more numerically stable
        k=0.04              # Harris parameter (only active when useHarrisDetector=True)
    )

    result = img.copy()

    if corners is not None:
        corners = np.int_(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)

        print(f"Detected corners: {len(corners)}")

    cv2.imshow('Good Features', result)
    cv2.waitKey(0)

    return corners

good_features_demo('building.jpg')
```

### Using Mask to Restrict Region

```python
import cv2
import numpy as np

def features_with_mask(image_path):
    """Detect features only in specific region"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Create ROI mask (center region only)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)

    # Detect features only in masked region
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=10,
        mask=mask
    )

    result = img.copy()

    # Show mask region
    cv2.rectangle(result, (w//4, h//4), (3*w//4, 3*h//4), (128, 128, 128), 2)

    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('Features with Mask', result)
    cv2.waitKey(0)

features_with_mask('scene.jpg')
```

---

## 4. FAST Detector

### Concept

```
FAST (Features from Accelerated Segment Test):
Very fast corner detection algorithm

Principle:
Examine circular pattern (16 pixels) around center pixel P

        1  2  3
     16           4
   15               5
  14        P        6
   13               7
     12          8
        11 10 9

Decision criterion (N=12):
- If N consecutive pixels are brighter than P: Corner
- If N consecutive pixels are darker than P: Corner

Characteristics:
- Very fast (real-time processing)
- No rotation invariance
- No scale invariance
- Non-maximum suppression (NMS) prevents multiple detections
```

### cv2.FastFeatureDetector

```python
import cv2
import numpy as np

def fast_detection(image_path):
    """FAST feature detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create FAST detector
    fast = cv2.FastFeatureDetector_create(
        threshold=20,           # Intensity threshold
        nonmaxSuppression=True  # Non-maximum suppression
    )

    # Detect features
    keypoints = fast.detect(gray, None)

    # Draw results
    result = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"Detected features: {len(keypoints)}")

    cv2.imshow('FAST', result)
    cv2.waitKey(0)

    return keypoints

fast_detection('building.jpg')
```

### Comparing FAST Thresholds

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_fast_thresholds(image_path):
    """Compare FAST thresholds"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresholds = [10, 20, 30, 50]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, thresh in zip(axes, thresholds):
        fast = cv2.FastFeatureDetector_create(
            threshold=thresh,
            nonmaxSuppression=True
        )
        kps = fast.detect(gray, None)
        result = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0))

        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Threshold={thresh}, Points={len(kps)}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

compare_fast_thresholds('building.jpg')
```

---

## 5. SIFT Detector

### Concept

```
SIFT (Scale-Invariant Feature Transform):
Feature detection and description invariant to scale and rotation

Steps:
1. Scale-space extrema detection (DoG: Difference of Gaussians)
2. Keypoint localization (sub-pixel accuracy, edge removal)
3. Orientation assignment (gradient histogram)
4. Descriptor computation (4x4 grid x 8 directions = 128 dimensions)

Scale Space:
+-------------------------------------------------+
|  Octave 0    Octave 1    Octave 2               |
|  +-------+   +-----+    +---+                   |
|  | s=1.6|   | s=1.6|   |s=1.6|  -> Scale-wise   |
|  | s=2.0|   | s=2.0|   |s=2.0|     Gaussian     |
|  | s=2.5|   | s=2.5|   |s=2.5|     blur         |
|  | s=3.2|   | s=3.2|   |s=3.2|                  |
|  +-------+   +-----+    +---+                   |
|  Original    1/2 size   1/4 size                |
+-------------------------------------------------+

DoG (Difference of Gaussians):
D(x, y, sigma) = L(x, y, k*sigma) - L(x, y, sigma)
-> Blob detection via difference between adjacent scales
```

### cv2.SIFT_create()

```python
import cv2
import numpy as np

def sift_detection(image_path):
    """SIFT feature detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create(
        nfeatures=0,          # 0 = unlimited; set a cap (e.g. 500) to bound
                              # descriptor computation time at inference
        nOctaveLayers=3,      # Layers between octaves: 3 gives good coverage
                              # of the scale space without redundant computation
        contrastThreshold=0.04,  # Reject low-contrast keypoints (texture-less areas)
                                 # — lower values keep more but noisier keypoints
        edgeThreshold=10,     # Reject edge responses; higher values are more
                              # permissive and allow more edge-like keypoints
        sigma=1.6             # Initial Gaussian blur matches the assumed camera
                              # smoothing; Lowe's original paper validated 1.6
    )

    # Compute keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw results
    result = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"Detected features: {len(keypoints)}")
    if descriptors is not None:
        print(f"Descriptor size: {descriptors.shape}")

    cv2.imshow('SIFT', result)
    cv2.waitKey(0)

    return keypoints, descriptors

kps, descs = sift_detection('object.jpg')
```

### Analyzing SIFT Keypoints

```python
import cv2
import numpy as np

def analyze_sift_keypoints(image_path):
    """Detailed analysis of SIFT keypoints"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    print("SIFT Keypoint Analysis:")
    print("-" * 50)

    # Keypoint attributes
    for i, kp in enumerate(keypoints[:5]):
        print(f"Keypoint {i}:")
        print(f"  Location (x, y): ({kp.pt[0]:.1f}, {kp.pt[1]:.1f})")
        print(f"  Size (scale): {kp.size:.1f}")
        print(f"  Angle: {kp.angle:.1f} degrees")
        print(f"  Response: {kp.response:.4f}")
        print(f"  Octave: {kp.octave}")

    # Scale distribution
    scales = [kp.size for kp in keypoints]
    print(f"\nScale range: {min(scales):.1f} ~ {max(scales):.1f}")

    # Descriptor analysis
    if descriptors is not None:
        print(f"\nDescriptors:")
        print(f"  Dimensions: {descriptors.shape[1]}")
        print(f"  Value range: {descriptors.min():.1f} ~ {descriptors.max():.1f}")

analyze_sift_keypoints('object.jpg')
```

---

## 6. ORB Detector

### Concept

```
ORB (Oriented FAST and Rotated BRIEF):
Improved version of FAST + BRIEF, patent-free

Components:
1. oFAST: FAST with orientation information
   - Computes orientation for rotation invariance
   - Image pyramid for scale invariance

2. rBRIEF: Rotated BRIEF
   - BRIEF: Binary descriptor (256 bits)
   - Learned comparison patterns for better discrimination
   - Fast matching with Hamming distance

Characteristics:
- Much faster than SIFT/SURF
- Patent-free
- Suitable for real-time processing
- Binary descriptor -> Fast matching

BRIEF Descriptor:
Compare intensities of two points in a patch
tau(P; x, y) = { 1 if P(x) < P(y)
               { 0 otherwise
-> n comparisons yield n-bit binary string
```

### cv2.ORB_create()

```python
import cv2
import numpy as np

def orb_detection(image_path):
    """ORB feature detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(
        nfeatures=500,        # Practical limit for real-time use: 500 gives good
                              # coverage without overwhelming the matcher
        scaleFactor=1.2,      # Mild downscale per pyramid level; smaller values
                              # build finer-grained scale space but cost more memory
        nlevels=8,            # 8 levels gives ~3.6× scale range (1.2^8), enough
                              # to handle typical viewpoint scale changes
        edgeThreshold=31,     # Matches patchSize: keeps FAST away from borders
                              # so BRIEF can sample a full 31×31 patch
        firstLevel=0,         # Start detection at full resolution for small features
        WTA_K=2,              # Default binary comparison (pairs); use 3 or 4 for
                              # more discriminative but larger descriptors
        scoreType=cv2.ORB_HARRIS_SCORE,  # Harris score is more accurate than FAST
                                         # score for ranking keypoints by quality
        patchSize=31,         # BRIEF patch size: larger patches are more
                              # discriminative but slower to compute
        fastThreshold=20      # FAST intensity threshold: 20 balances sensitivity
                              # and false positive rate for typical images
    )

    # Compute keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw results
    result = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"Detected features: {len(keypoints)}")
    if descriptors is not None:
        print(f"Descriptor size: {descriptors.shape}")
        print(f"Descriptor type: {descriptors.dtype}")  # uint8 (binary)

    cv2.imshow('ORB', result)
    cv2.waitKey(0)

    return keypoints, descriptors

kps, descs = orb_detection('object.jpg')
```

### SIFT vs ORB Comparison

```python
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def compare_sift_orb(image_path):
    """Compare SIFT and ORB performance"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SIFT
    sift = cv2.SIFT_create()
    start = time.time()
    kps_sift, descs_sift = sift.detectAndCompute(gray, None)
    sift_time = time.time() - start

    # ORB
    orb = cv2.ORB_create(nfeatures=len(kps_sift))
    start = time.time()
    kps_orb, descs_orb = orb.detectAndCompute(gray, None)
    orb_time = time.time() - start

    print("Performance Comparison:")
    print("-" * 50)
    print(f"SIFT: {len(kps_sift)} points, {sift_time*1000:.1f}ms")
    print(f"ORB:  {len(kps_orb)} points, {orb_time*1000:.1f}ms")
    print(f"Speed ratio: ORB is {sift_time/orb_time:.1f}x faster")

    if descs_sift is not None and descs_orb is not None:
        print(f"\nSIFT descriptor: {descs_sift.shape}, {descs_sift.dtype}")
        print(f"ORB descriptor: {descs_orb.shape}, {descs_orb.dtype}")

    # Visualization
    result_sift = cv2.drawKeypoints(img, kps_sift, None, color=(0, 255, 0))
    result_orb = cv2.drawKeypoints(img, kps_orb, None, color=(0, 0, 255))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(cv2.cvtColor(result_sift, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'SIFT: {len(kps_sift)} points')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(result_orb, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'ORB: {len(kps_orb)} points')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

compare_sift_orb('object.jpg')
```

---

## 7. Keypoints and Descriptors

### KeyPoint Structure

```python
import cv2
import numpy as np

def keypoint_structure():
    """Understanding keypoint structure"""
    # Manually create keypoint
    kp = cv2.KeyPoint(
        x=100.5,        # x coordinate
        y=200.5,        # y coordinate
        size=20,        # Feature size (diameter)
        angle=45,       # Orientation (degrees)
        response=0.8,   # Response strength
        octave=0,       # Octave (scale)
        class_id=-1     # Class ID
    )

    print("KeyPoint Attributes:")
    print(f"  Location: ({kp.pt[0]}, {kp.pt[1]})")
    print(f"  Size: {kp.size}")
    print(f"  Angle: {kp.angle}")
    print(f"  Response: {kp.response}")
    print(f"  Octave: {kp.octave}")
    print(f"  Class ID: {kp.class_id}")

keypoint_structure()
```

### Understanding Descriptors

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_descriptors(image_path):
    """Visualize descriptors"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # SIFT descriptor (128-dimensional float)
    sift = cv2.SIFT_create()
    kps_sift, descs_sift = sift.detectAndCompute(img, None)

    # ORB descriptor (32 bytes = 256 bits)
    orb = cv2.ORB_create()
    kps_orb, descs_orb = orb.detectAndCompute(img, None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SIFT descriptor histogram
    if descs_sift is not None and len(descs_sift) > 0:
        axes[0, 0].bar(range(128), descs_sift[0])
        axes[0, 0].set_title('SIFT Descriptor (128D)')
        axes[0, 0].set_xlabel('Dimension')

        axes[0, 1].imshow(descs_sift[:50], aspect='auto', cmap='viridis')
        axes[0, 1].set_title('SIFT Descriptors (first 50)')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Keypoint')

    # ORB descriptor (binary)
    if descs_orb is not None and len(descs_orb) > 0:
        # Convert binary to bits
        bits = np.unpackbits(descs_orb[0])
        axes[1, 0].bar(range(256), bits)
        axes[1, 0].set_title('ORB Descriptor (256 bits)')
        axes[1, 0].set_xlabel('Bit')

        # Multiple descriptors
        bits_all = np.unpackbits(descs_orb[:50], axis=1)
        axes[1, 1].imshow(bits_all, aspect='auto', cmap='binary')
        axes[1, 1].set_title('ORB Descriptors (first 50)')
        axes[1, 1].set_xlabel('Bit')
        axes[1, 1].set_ylabel('Keypoint')

    plt.tight_layout()
    plt.show()

visualize_descriptors('object.jpg')
```

### Using Various Detectors

```python
import cv2
import numpy as np

def use_various_detectors(image_path):
    """Use various feature detectors"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detectors = {
        'SIFT': cv2.SIFT_create(),
        'ORB': cv2.ORB_create(),
        'BRISK': cv2.BRISK_create(),
        'AKAZE': cv2.AKAZE_create(),
        # 'KAZE': cv2.KAZE_create(),  # Slow
    }

    results = {}

    for name, detector in detectors.items():
        kps, descs = detector.detectAndCompute(gray, None)
        results[name] = {
            'keypoints': kps,
            'descriptors': descs,
            'count': len(kps),
            'desc_size': descs.shape[1] if descs is not None else 0
        }

        print(f"{name}:")
        print(f"  Feature count: {len(kps)}")
        if descs is not None:
            print(f"  Descriptor: {descs.shape}, {descs.dtype}")
        print()

    return results

results = use_various_detectors('object.jpg')
```

---

## 8. Practice Problems

### Problem 1: Select Best Keypoints

Select only the 50 strongest keypoints from an image.

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def select_best_keypoints(image_path, n=50):
    """Select N strongest keypoints"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect many keypoints with ORB
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Sort by response strength
    keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)

    # Select top N
    best_keypoints = keypoints_sorted[:n]

    # Select corresponding descriptors
    indices = [keypoints.index(kp) for kp in best_keypoints]
    best_descriptors = descriptors[indices] if descriptors is not None else None

    result = cv2.drawKeypoints(
        img, best_keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow(f'Best {n} Keypoints', result)
    cv2.waitKey(0)

    return best_keypoints, best_descriptors

kps, descs = select_best_keypoints('building.jpg', n=50)
```

</details>

### Problem 2: Uniformly Distributed Keypoints

Divide the image into a grid and select one keypoint from each cell.

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def uniform_keypoints(image_path, grid_size=(8, 8)):
    """Select keypoints uniformly per grid cell"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Calculate grid size
    cell_h = h // grid_size[0]
    cell_w = w // grid_size[1]

    # Select strongest keypoint per cell
    selected_kps = []
    selected_indices = []

    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            # Cell region
            x_min = col * cell_w
            x_max = (col + 1) * cell_w
            y_min = row * cell_h
            y_max = (row + 1) * cell_h

            # Filter keypoints in cell
            cell_kps = []
            for i, kp in enumerate(keypoints):
                if x_min <= kp.pt[0] < x_max and y_min <= kp.pt[1] < y_max:
                    cell_kps.append((i, kp))

            if cell_kps:
                # Select strongest keypoint
                best_idx, best_kp = max(cell_kps, key=lambda x: x[1].response)
                selected_kps.append(best_kp)
                selected_indices.append(best_idx)

    # Descriptors
    selected_descs = descriptors[selected_indices] if descriptors is not None else None

    result = cv2.drawKeypoints(
        img, selected_kps, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Draw grid
    for row in range(1, grid_size[0]):
        cv2.line(result, (0, row * cell_h), (w, row * cell_h), (128, 128, 128), 1)
    for col in range(1, grid_size[1]):
        cv2.line(result, (col * cell_w, 0), (col * cell_w, h), (128, 128, 128), 1)

    cv2.imshow('Uniform Keypoints', result)
    cv2.waitKey(0)

    return selected_kps, selected_descs

kps, descs = uniform_keypoints('building.jpg', grid_size=(6, 8))
```

</details>

### Problem 3: Rotation Invariance Test

Rotate an image and verify that the same features are detected.

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def test_rotation_invariance(image_path, angle=45):
    """Test rotation invariance"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Rotate image
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h))

    # SIFT (rotation invariant)
    sift = cv2.SIFT_create(nfeatures=100)

    kps1, descs1 = sift.detectAndCompute(gray, None)
    kps2, descs2 = sift.detectAndCompute(rotated, None)

    # Feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs1, descs2, k=2)

    # Filter good matches (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Original features: {len(kps1)}")
    print(f"Rotated image features: {len(kps2)}")
    print(f"Matched features: {len(good_matches)}")
    print(f"Match rate: {len(good_matches) / len(kps1) * 100:.1f}%")

    # Visualization
    result = cv2.drawMatches(
        gray, kps1, rotated, kps2,
        good_matches, None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow('Rotation Invariance Test', result)
    cv2.waitKey(0)

test_rotation_invariance('object.jpg', angle=30)
```

</details>

### Recommended Problems

| Difficulty | Topic | Description |
|------------|-------|-------------|
| * | Basic Detection | Compare Harris, FAST, ORB |
| ** | Performance Comparison | Compare detection speed and count |
| ** | Parameter Tuning | Find optimal parameters |
| *** | Scale Invariance | Test against size changes |
| *** | Real-time Detection | Display features in real-time from webcam |

---

## Next Steps

- [14_Feature_Matching.md](./14_Feature_Matching.md) - BFMatcher, FLANN, Homography

---

## References

- [OpenCV Feature Detection](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [SIFT Paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [ORB Paper](https://www.willowgarage.com/sites/default/files/orb_final.pdf)
