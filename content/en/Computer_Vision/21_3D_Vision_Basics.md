# 3D Vision Basics

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the principles of stereo vision, including epipolar geometry and the disparity-to-depth relationship.
2. Implement depth map generation using OpenCV's stereo matching algorithms (StereoBM, StereoSGBM).
3. Define point clouds and explain how to generate them from depth maps and camera intrinsics.
4. Apply Open3D to visualize, process, and analyze 3D point cloud data.
5. Describe the 3D reconstruction pipeline and implement a basic structure-from-motion workflow.

---

## Overview

3D vision is the technology for extracting and reconstructing three-dimensional information from 2D images. This covers the fundamentals of stereo vision, depth maps, point cloud processing, and 3D reconstruction.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: Camera calibration, feature detection/matching, linear algebra

---

## Table of Contents

1. [3D Vision Overview](#1-3d-vision-overview)
2. [Stereo Vision Principles](#2-stereo-vision-principles)
3. [Depth Map Generation](#3-depth-map-generation)
4. [Point Clouds](#4-point-clouds)
5. [Open3D Basics](#5-open3d-basics)
6. [3D Reconstruction](#6-3d-reconstruction)
7. [Exercises](#7-exercises)

---

## 1. 3D Vision Overview

### Theory: Why One Image Is Not Enough

Given a camera with known intrinsics (§18), each pixel `(u, v)` back-projects to a **ray** in 3D space emanating from the optical center. Every 3D point along that ray projects to the same pixel. There is no way to tell from a single image whether an object is small and close or large and far.

A second image from a different viewpoint breaks the ambiguity. The same 3D point projects to two pixels, one in each image. The two rays — one per image — intersect at the true 3D location (up to noise). This is the fundamental idea of stereo vision, structure-from-motion, SLAM, and multi-view reconstruction.

### Goals of 3D Vision

```
3D Vision Pipeline:

┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  2D Image ─────▶ Depth Estimation ─────▶ 3D Reconstruction       │
│      │                                                           │
│      │           ┌─────────────┐                                 │
│      └──────────▶│ Depth Info  │──────▶ Point Cloud              │
│                  └─────────────┘            │                    │
│                                             │                    │
│                                             ▼                    │
│                                      ┌─────────────┐             │
│                                      │  3D Mesh    │             │
│                                      │  3D Model   │             │
│                                      └─────────────┘             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Depth Extraction Methods:
┌─────────────────────┬──────────────────────────────────────────┐
│ Method              │ Description                              │
├─────────────────────┼──────────────────────────────────────────┤
│ Stereo Vision       │ Calculate depth from disparity of 2 cams │
│ Structured Light    │ Measure depth by projecting known pattern│
│ ToF (Time-of-Flight)│ Measure distance by light travel time    │
│ Monocular Depth Est.│ Predict depth with single cam + DL       │
│ LiDAR               │ Precise depth measurement by laser scan  │
└─────────────────────┴──────────────────────────────────────────┘
```

### Understanding Coordinate Systems

```
Camera Coordinate System:

        Y (up)
        │
        │
        │
        │_________ X (right)
       /
      /
     Z (camera forward direction)

World Coordinate System → Camera Coordinate System Transform:
P_cam = R * P_world + t

Image Coordinate System:
┌─────────────────────▶ u (horizontal, pixels)
│
│   ● (cx, cy) principal point
│
▼
v (vertical, pixels)

3D → 2D Projection:
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy
```

---

## 2. Stereo Vision Principles

### Theory: Epipolar Geometry

Given two calibrated cameras viewing the same scene, the relative position and orientation of the cameras induces a constraint on corresponding image points. This constraint is captured by the **fundamental matrix** `F` or its calibrated counterpart, the **essential matrix** `E`.

#### B.1 The epipolar constraint

For any 3D point observed in both images at pixels `x₁` and `x₂` (in homogeneous coordinates):

```
x₂ᵀ · F · x₁ = 0
```

This says: given a point `x₁` in image 1, the corresponding point in image 2 must lie on a specific line, called the **epipolar line** `F · x₁`. The epipolar constraint reduces the search for correspondences from a 2D search over the whole image to a 1D search along a line — a massive computational win.

#### B.2 The essential matrix and camera pose

When the intrinsic matrices are known, normalize pixel coordinates (`x̂ = K⁻¹ · x`) and use the **essential matrix** instead:

```
x̂₂ᵀ · E · x̂₁ = 0
```

`E = [t]× · R` factors into the **relative rotation** `R` and the **skew-symmetric matrix of translation** `[t]×`. Decomposing `E` recovers both (up to sign ambiguity, typically resolved by cheirality — both points should have positive depth in both cameras). This is how structure-from-motion recovers camera poses without priors.

#### B.3 Rectification

If you make the two image planes coplanar and align their horizontal axes, epipolar lines become **horizontal** — the corresponding point in image 2 lies on the same scanline as in image 1. This is **stereo rectification**, and it turns stereo matching into 1D row-wise search instead of search along arbitrary epipolar lines.

OpenCV's `stereoRectify` + `initUndistortRectifyMap` + `remap` precomputes the rectification warps for both cameras. After rectification, you can use a simple block-matching algorithm that scans horizontally.

### Epipolar Geometry

```
Epipolar Geometry:

             Epipole (e)
              │
   ┌──────────┼──────────┐
   │          │          │
   │    ●─────┼──────────┼─────● Epipolar line
   │   P      │          │   P'
   │          │          │
   └──────────┴──────────┘
       Left          Right
       Image         Image

If 3D point P projects to point p in the left image,
it projects to p' somewhere on the epipolar line in the right image.

Key Matrices:
┌───────────────────┬─────────────────────────────────────────┐
│ Matrix            │ Description                             │
├───────────────────┼─────────────────────────────────────────┤
│ Essential Matrix  │ Geometric relationship in normalized    │
│ (E)               │ coordinates. E = [t]x * R               │
├───────────────────┼─────────────────────────────────────────┤
│ Fundamental Matrix│ Geometric relationship in pixel         │
│ (F)               │ coordinates. F = K'^(-T) * E * K^(-1)   │
│                   │ p'^T * F * p = 0                        │
└───────────────────┴─────────────────────────────────────────┘
```

**Why is the epipolar constraint so important?** Without it, finding the corresponding point of pixel p in the right image would require searching the entire 2D image — an O(W×H) problem for every pixel. The epipolar constraint tells us that the match must lie on a specific line (the epipolar line), reducing the search to a 1D problem along that line. After stereo rectification (aligning both images so epipolar lines are horizontal), this further simplifies to scanning along the same row — which is why rectification is a standard preprocessing step before disparity computation.

### Disparity and Depth

```
Stereo Disparity:

Left Camera          Right Camera
    C_L ─────────────── C_R
     │                    │
     │    b (baseline)    │
     │    ◄─────────────► │
     │                    │
     │                    │
     ▼                    ▼
    p_L        d        p_R
    ●─────────────────────●
    │                     │
    │     Disparity (d)   │
    │     d = x_L - x_R   │

Depth Calculation:
Z = (f * b) / d

Where:
- Z: Depth (distance from camera)
- f: Focal length
- b: Baseline (distance between two cameras)
- d: Disparity (in pixels)

Disparity Range Example:
┌─────────────────────────────────────────┐
│ Distance │ Disparity (f=500, b=0.1m)    │
├──────────┼──────────────────────────────┤
│ 1m       │ 50 pixels                    │
│ 5m       │ 10 pixels                    │
│ 10m      │ 5 pixels                     │
│ Infinity │ 0 pixels                     │
└─────────────────────────────────────────┘
```

### Stereo Rectification

```python
import cv2
import numpy as np

def stereo_calibrate(obj_points, img_points_left, img_points_right,
                     K1, D1, K2, D2, img_size):
    """Stereo camera calibration"""

    flags = (cv2.CALIB_FIX_INTRINSIC +
             cv2.CALIB_RATIONAL_MODEL)

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_left,
        img_points_right,
        K1, D1,
        K2, D2,
        img_size,
        flags=flags
    )

    print(f"Stereo calibration RMS error: {ret:.4f}")
    print(f"\nRotation matrix R:\n{R}")
    print(f"\nTranslation vector T:\n{T.ravel()}")
    print(f"\nBaseline: {np.linalg.norm(T):.4f} units")

    return R, T, E, F

def stereo_rectify(K1, D1, K2, D2, img_size, R, T):
    """Stereo Rectification"""

    # Calculate rectification transform
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1,
        K2, D2,
        img_size,
        R, T,
        alpha=0,  # 0: valid pixels only, 1: all pixels
        newImageSize=img_size
    )

    # Q matrix: used for disparity → 3D conversion
    # [X Y Z W]^T = Q * [x y disparity 1]^T
    print("Q matrix (disparity → 3D transform):")
    print(Q)

    return R1, R2, P1, P2, Q, roi1, roi2

def create_rectification_maps(K, D, R, P, img_size):
    """Generate rectification maps"""

    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, R, P, img_size, cv2.CV_32FC1
    )

    return map1, map2

def rectify_stereo_pair(img_left, img_right, maps_left, maps_right):
    """Rectify stereo image pair"""

    rect_left = cv2.remap(img_left, maps_left[0], maps_left[1],
                          cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, maps_right[0], maps_right[1],
                           cv2.INTER_LINEAR)

    return rect_left, rect_right
```

---

## 3. Depth Map Generation

### Theory: Disparity and Depth

After rectification, the same 3D point appears at `(x_L, y)` in the left image and `(x_R, y)` in the right (same `y`, different `x`). The **disparity** is `d = x_L - x_R`. Geometrically:

```
Z = f · B / d
```

where:

- `Z` = depth of the 3D point (distance from camera, in meters).
- `f` = focal length in pixels (common for both rectified cameras after calibration).
- `B` = **baseline**, the distance between the two camera centers (in meters).
- `d` = disparity (in pixels).

**Key properties**:

- Near objects have **large disparity**; far objects have small disparity. A point at infinity has zero disparity.
- Depth precision **degrades quadratically with distance**: `ΔZ ≈ (Z²/fB) · Δd`. For a fixed disparity resolution, depth error at 10m is 100× the depth error at 1m. This is why stereo doesn't work well far from the camera.
- Larger baseline `B` gives better far-range precision but narrows the overlap between the two views.
- Sub-pixel disparity estimation (interpolating the matching cost function) improves depth precision significantly.

### StereoBM (Block Matching)

```python
import cv2
import numpy as np

def compute_disparity_bm(left, right, num_disparities=64, block_size=15):
    """Compute disparity map using StereoBM"""

    # Convert to grayscale
    if len(left.shape) == 3:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Create StereoBM
    stereo = cv2.StereoBM_create(
        numDisparities=num_disparities,  # Must be a multiple of 16; larger values
                                         # search a wider range of depths but cost more compute
        blockSize=block_size              # Odd number, 5~21; larger blocks give smoother
                                         # disparity but lose fine detail at depth boundaries
    )

    # Parameter tuning (optional)
    stereo.setMinDisparity(0)
    stereo.setSpeckleWindowSize(100)   # Remove isolated "speckle" blobs of bad disparities
    stereo.setSpeckleRange(32)
    stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
    stereo.setPreFilterSize(9)
    stereo.setPreFilterCap(31)
    stereo.setTextureThreshold(10)     # Skip textureless regions where matching is unreliable
    stereo.setUniquenessRatio(15)      # Reject ambiguous matches: best match must be at least
                                       # 15% better than the second-best candidate

    # Compute disparity
    disparity = stereo.compute(left, right)

    # Normalize disparity values (scaled by 16)
    disparity = disparity.astype(np.float32) / 16.0

    return disparity

def visualize_disparity(disparity):
    """Visualize disparity map"""

    # Use only valid disparity
    valid_mask = disparity > 0

    # Normalize
    disp_vis = np.zeros_like(disparity)
    if np.any(valid_mask):
        disp_min = np.min(disparity[valid_mask])
        disp_max = np.max(disparity[valid_mask])
        disp_vis = (disparity - disp_min) / (disp_max - disp_min) * 255

    disp_vis = disp_vis.astype(np.uint8)

    # Apply colormap
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # Black out invalid regions
    disp_color[~valid_mask] = [0, 0, 0]

    return disp_color
```

### StereoSGBM (Semi-Global Block Matching)

```python
def compute_disparity_sgbm(left, right, num_disparities=64, block_size=5):
    """Compute disparity map using StereoSGBM"""

    # Convert to grayscale
    if len(left.shape) == 3:
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    else:
        gray_left, gray_right = left, right

    # SGBM parameters
    # P1, P2: Penalty for disparity difference between adjacent pixels.
    # P1 penalizes small 1-pixel disparity changes (smooth surfaces),
    # P2 penalizes larger jumps (depth discontinuities). P2 > P1 enforces
    # piecewise-smooth disparity maps. The *3*block_size^2 scaling is the
    # OpenCV-recommended baseline that keeps penalties proportional to block area.
    P1 = 8 * 3 * block_size ** 2
    P2 = 32 * 3 * block_size ** 2

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,           # Left-right consistency check tolerance;
                                    # tight value (1) catches occlusion artifacts
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Aggregates cost along 3 directions
                                               # for better accuracy vs. the default 5-way
    )

    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right)
    disparity = disparity.astype(np.float32) / 16.0

    return disparity

def disparity_to_depth(disparity, Q):
    """Convert disparity map to depth map"""

    # 3D reprojection using Q matrix
    # points_3d[y, x] = [X, Y, Z, W]
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    # Extract Z value (depth)
    depth = points_3d[:, :, 2]

    # Filter invalid depth
    valid_mask = (disparity > 0) & (depth > 0) & (depth < 10000)
    depth[~valid_mask] = 0

    return depth, points_3d

def create_depth_colormap(depth, max_depth=10.0):
    """Visualize depth map"""

    # Clip depth
    depth_clipped = np.clip(depth, 0, max_depth)

    # Normalize (0-255)
    depth_norm = (depth_clipped / max_depth * 255).astype(np.uint8)

    # Apply colormap (close = red, far = blue)
    depth_color = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_JET)

    # Mask invalid regions
    depth_color[depth <= 0] = [0, 0, 0]

    return depth_color
```

### Improved Disparity with WLS Filter

```python
def compute_disparity_with_wls(left, right, num_disparities=64):
    """Compute improved disparity map with WLS filter"""

    # Grayscale
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Left matcher
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Right matcher (for left-right consistency check): computing disparity in
    # both directions lets the WLS filter identify occlusions and unreliable
    # regions where left and right estimates disagree
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Compute disparity
    left_disp = left_matcher.compute(gray_left, gray_right)
    right_disp = right_matcher.compute(gray_right, gray_left)

    # WLS (Weighted Least Squares) filter: smooths the disparity map while
    # preserving depth edges by weighting smoothness with image color gradients —
    # pixels with similar color get smoothed together; edges are kept sharp
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(80000)    # Higher lambda = stronger smoothness; trade-off
                                    # between noise suppression and edge preservation
    wls_filter.setSigmaColor(1.2)  # Color sensitivity: lower values preserve more
                                    # edges but allow more noise to pass through

    # Apply filter: uses the right disparity as a confidence guide to fill
    # occluded pixels that only the right matcher could observe
    filtered_disp = wls_filter.filter(left_disp, left, None, right_disp)
    filtered_disp = filtered_disp.astype(np.float32) / 16.0

    return filtered_disp
```

---

## 4. Point Clouds

### Theory: Point Clouds and Meshes

#### E.1 Point clouds

A **point cloud** is a discrete set of 3D points: `{(X_i, Y_i, Z_i)}`, optionally with colors, normals, or other per-point attributes. The natural output of stereo, depth sensors, and SFM. Compact (one tuple per sample), easy to visualize, easy to downsample. But no connectivity — adjacent points aren't explicitly linked.

Common operations:

- **Downsampling**: voxel grid (one point per small 3D cube) to reduce density uniformly.
- **Outlier removal**: statistical (point has too few neighbors) or radius-based.
- **Normal estimation**: fit a plane to each point's local neighborhood.
- **Registration**: align two point clouds using ICP (Iterative Closest Point).

#### E.2 Meshes

A **mesh** adds connectivity: vertices + triangles. It is the standard representation for rendering and physical simulation. Converting point clouds to meshes:

- **Poisson surface reconstruction**: solve a PDE to find an implicit surface that matches given normal directions, then extract an isosurface.
- **Ball-pivoting**: roll a virtual ball over the point cloud; wherever it touches three points, create a triangle.
- **Marching cubes**: extract isosurface from an occupancy grid.

Meshes are denser, less noisy, and easier to render than raw points — but they introduce topology choices and can hide localization error.

#### E.3 From depth map to point cloud

If you have a depth map `D(u, v)` and camera intrinsics `K`, each pixel converts to a 3D point:

```
Z = D(u, v)
X = (u - c_x) · Z / f_x
Y = (v - c_y) · Z / f_y
```

This is just the inverse of §18.A.1 projection. Every depth sensor and stereo algorithm produces a depth map that you convert to a point cloud this way.

### Point Cloud Generation

```python
import cv2
import numpy as np

def create_point_cloud(depth, rgb, K):
    """Create point cloud from depth map and RGB image"""

    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel coordinate grid
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

    # Valid depth mask
    valid = depth > 0

    # Calculate 3D coordinates
    Z = depth[valid]
    X = (u[valid] - cx) * Z / fx
    Y = (v[valid] - cy) * Z / fy

    # Point cloud (N x 3)
    points = np.stack([X, Y, Z], axis=-1)

    # Color information (N x 3)
    if len(rgb.shape) == 3:
        colors = rgb[valid]
    else:
        colors = np.stack([rgb[valid]] * 3, axis=-1)

    return points, colors

def subsample_point_cloud(points, colors, voxel_size=0.01):
    """Downsample point cloud using voxel grid"""

    # Calculate voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Select only unique voxels
    _, unique_indices = np.unique(
        voxel_indices, axis=0, return_index=True
    )

    return points[unique_indices], colors[unique_indices]

def save_point_cloud_ply(filename, points, colors):
    """Save point cloud in PLY format"""

    n_points = len(points)

    # PLY header
    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(filename, 'w') as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    print(f"Saved: {filename} ({n_points} points)")
```

### Point Cloud Processing

```python
def remove_outliers_statistical(points, colors, nb_neighbors=20, std_ratio=2.0):
    """Statistical outlier removal"""

    from scipy.spatial import KDTree

    # Build KD-Tree
    tree = KDTree(points)

    # Calculate k-NN distance for each point
    distances, _ = tree.query(points, k=nb_neighbors + 1)
    mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self

    # Global mean and standard deviation
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)

    # Outlier mask
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances < threshold

    print(f"Outlier removal: {len(points)} → {np.sum(inlier_mask)} points")

    return points[inlier_mask], colors[inlier_mask]

def estimate_normals(points, k=30):
    """Estimate point cloud normal vectors"""

    from scipy.spatial import KDTree
    from numpy.linalg import eig

    tree = KDTree(points)
    normals = np.zeros_like(points)

    for i, point in enumerate(points):
        # k-NN search
        _, indices = tree.query(point, k=k)
        neighbors = points[indices]

        # Covariance matrix
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = np.dot(centered.T, centered) / k

        # Eigenvector of smallest eigenvalue is the normal
        eigenvalues, eigenvectors = eig(cov)
        min_idx = np.argmin(eigenvalues)
        normals[i] = eigenvectors[:, min_idx]

    return normals
```

---

## 5. Open3D Basics

### Open3D Installation and Basic Usage

```python
# pip install open3d

import open3d as o3d
import numpy as np

def create_open3d_point_cloud(points, colors=None):
    """Create Open3D point cloud"""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        # Normalize colors to 0-1 range
        if colors.max() > 1:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def visualize_point_cloud(pcd):
    """Visualize point cloud"""

    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries(
        [pcd, coordinate_frame],
        window_name="Point Cloud",
        width=1280,
        height=720,
        point_show_normal=False
    )

def process_point_cloud_open3d(pcd):
    """Process point cloud with Open3D"""

    print(f"Original point count: {len(pcd.points)}")

    # 1. Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=0.02)
    print(f"After downsampling: {len(pcd_down.points)}")

    # 2. Outlier removal
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
    print(f"After outlier removal: {len(pcd_clean.points)}")

    # 3. Normal estimation
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )

    # 4. Orient normals consistently
    pcd_clean.orient_normals_consistent_tangent_plane(k=15)

    return pcd_clean
```

### Mesh Reconstruction

```python
def reconstruct_mesh_poisson(pcd, depth=9):
    """Poisson surface reconstruction"""

    # Normals required
    if not pcd.has_normals():
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Remove low-density regions
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"Mesh vertices: {len(mesh.vertices)}")
    print(f"Mesh triangles: {len(mesh.triangles)}")

    return mesh

def reconstruct_mesh_ball_pivoting(pcd):
    """Ball pivoting surface reconstruction"""

    if not pcd.has_normals():
        pcd.estimate_normals()

    # Estimate radii
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist, avg_dist * 2, avg_dist * 4]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    return mesh

def save_mesh(mesh, filename):
    """Save mesh"""
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"Mesh saved: {filename}")
```

### RGBD Image Processing

```python
def create_rgbd_from_opencv(color_img, depth_img, K):
    """Convert OpenCV images to Open3D RGBD"""

    # BGR → RGB
    color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    # Convert to Open3D images
    color_o3d = o3d.geometry.Image(color_rgb)
    depth_o3d = o3d.geometry.Image(depth_img.astype(np.float32))

    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,  # mm → m
        depth_trunc=3.0,     # Maximum depth
        convert_rgb_to_intensity=False
    )

    return rgbd

def rgbd_to_point_cloud(rgbd, K, width, height):
    """Create point cloud from RGBD image"""

    # Open3D camera parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        K[0, 0], K[1, 1],  # fx, fy
        K[0, 2], K[1, 2]   # cx, cy
    )

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic
    )

    return pcd
```

---

## 6. 3D Reconstruction

### Theory: Triangulation

Given calibrated camera matrices `P₁`, `P₂` and a correspondence `(x₁, x₂)`, triangulation recovers the 3D point `X`. The forward projection equations:

```
x₁ = P₁ · X       (projected to image 1)
x₂ = P₂ · X       (projected to image 2)
```

are two equations in four unknowns (`X` in homogeneous coordinates, up to scale). In the noise-free case these are consistent and have a unique solution. With noise, the system is overdetermined — solve by least-squares (`cv2.triangulatePoints` uses the DLT algorithm).

The practical workflow:

1. Detect and match features in both images (§13, §14).
2. Compute fundamental matrix with RANSAC (§14.D).
3. Extract relative pose (`R`, `t`) from the essential matrix (§B.2).
4. Triangulate each matched point pair to get a sparse point cloud.
5. (Optional) Bundle adjustment: jointly refine all camera poses and 3D points by minimizing reprojection error over the whole set.

This is exactly the pipeline used by structure-from-motion software (COLMAP, OpenSfM).

### Multi-View Stereo (MVS) Concept

```
Multi-View Stereo Pipeline:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. Image Acquisition                                           │
│     Capture subject from multiple angles                        │
│         📷 📷 📷 📷 📷                                          │
│                                                                 │
│  2. Feature Detection and Matching                              │
│     Find correspondences between images using SIFT, ORB, etc.   │
│         ● ─────────── ●                                         │
│                                                                 │
│  3. Structure from Motion (SfM)                                 │
│     Camera pose estimation + sparse point cloud                 │
│         📷────┐    ●                                            │
│         📷────┼────● ●                                          │
│         📷────┘    ●                                            │
│                                                                 │
│  4. Dense Reconstruction                                        │
│     Estimate depth for all pixels                               │
│         [:::::::::::]                                           │
│                                                                 │
│  5. Mesh Generation                                             │
│     Point cloud → Triangle mesh                                 │
│         ▲▲▲▲▲▲▲▲                                              │
│                                                                 │
│  6. Texture Mapping                                             │
│     Apply texture to mesh using original images                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Essential Matrix-based Pose Estimation

```python
import cv2
import numpy as np

def estimate_pose_from_essential(pts1, pts2, K):
    """Estimate relative pose from Essential Matrix"""

    # Use Essential (not Fundamental) Matrix because we have calibrated cameras:
    # E encodes only the relative rotation and translation (5 DOF), while F
    # would also absorb unknown intrinsics. Using K here normalizes the points
    # to metric coordinates, giving a more geometrically meaningful constraint.
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,   # RANSAC rejects mismatches; essential for noisy matches
        prob=0.999,           # High confidence: accept some extra iterations to avoid
                              # returning a matrix fit to outliers
        threshold=1.0         # Epipolar line distance tolerance in pixels
    )

    print(f"Inlier ratio: {np.sum(mask) / len(mask) * 100:.1f}%")

    # Recover R, t from Essential Matrix
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    print(f"\nRotation matrix R:\n{R}")
    print(f"\nTranslation vector t (unit vector):\n{t.ravel()}")

    return R, t

def triangulate_points(pts1, pts2, K, R, t):
    """Triangulate 3D points from two views"""

    # Construct projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    # Triangulation
    pts1_h = pts1.T  # (2, N)
    pts2_h = pts2.T

    points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)

    # Homogeneous coordinates → 3D coordinates
    points_3d = points_4d[:3] / points_4d[3]

    return points_3d.T  # (N, 3)

def incremental_sfm(images, K):
    """Incremental SfM (simple version)"""

    # SIFT detector
    sift = cv2.SIFT_create()

    # Initialize with first two images
    kp1, desc1 = sift.detectAndCompute(images[0], None)
    kp2, desc2 = sift.detectAndCompute(images[1], None)

    # Matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Initial pose and 3D points
    R, t = estimate_pose_from_essential(pts1, pts2, K)
    points_3d = triangulate_points(pts1, pts2, K, R, t)

    # Store camera poses
    camera_poses = [
        {'R': np.eye(3), 't': np.zeros((3, 1))},  # First camera
        {'R': R, 't': t}                           # Second camera
    ]

    print(f"Initial 3D points: {len(points_3d)}")

    # Add subsequent images (estimate pose using PnP)
    for i in range(2, len(images)):
        kp_new, desc_new = sift.detectAndCompute(images[i], None)

        # Match with previous image
        matches = bf.knnMatch(desc2, desc_new, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 3D-2D correspondences
        obj_points = points_3d[[m.queryIdx for m in good_matches]]
        img_points = np.float32([kp_new[m.trainIdx].pt for m in good_matches])

        # Estimate pose using PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points, img_points, K, None
        )

        if success:
            R_new, _ = cv2.Rodrigues(rvec)
            camera_poses.append({'R': R_new, 't': tvec})
            print(f"Image {i} registered (inliers: {len(inliers)})")

        # Update for next iteration
        desc2 = desc_new

    return points_3d, camera_poses
```

### Bundle Adjustment

```
Bundle Adjustment:
Simultaneously optimize camera parameters and 3D point positions

Minimization Objective:
E = Σ_i Σ_j || x_ij - π(K, R_i, t_i, X_j) ||²

Where:
- x_ij: 2D coordinates of point j observed in image i
- π(): 3D → 2D projection function
- K: Camera intrinsic parameters
- R_i, t_i: Camera i's pose
- X_j: 3D coordinates of point j

Optimization Tools:
- Ceres Solver
- g2o
- SciPy (for small problems)
```

---

## 7. Exercises

### Exercise 1: Stereo Depth Estimation

Generate a depth map from a stereo image pair.

**Requirements**:
- Compare StereoBM and StereoSGBM
- Visualize disparity map
- Convert to depth map
- Quality improvement (filtering)

<details>
<summary>Hint</summary>

```python
# Parameter tuning needed
stereo = cv2.StereoSGBM_create(
    numDisparities=128,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2
)

# Improve with WLS filter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
```

</details>

### Exercise 2: Point Cloud Filtering

Refine a noisy point cloud.

**Requirements**:
- Statistical outlier removal
- Voxel downsampling
- Extract planar regions
- Visualize results

<details>
<summary>Hint</summary>

```python
import open3d as o3d

# Outlier removal
pcd_clean, _ = pcd.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=2.0
)

# Downsampling
pcd_down = pcd_clean.voxel_down_sample(0.02)

# Plane extraction (RANSAC)
plane_model, inliers = pcd_down.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)
```

</details>

### Exercise 3: 3D Reconstruction from Two Views

Reconstruct 3D points from two images.

**Requirements**:
- Feature detection and matching
- Essential Matrix calculation
- Camera pose recovery
- Generate 3D points via triangulation

<details>
<summary>Hint</summary>

```python
# Essential Matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K)

# Pose recovery
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# Triangulation
points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3d = points_4d[:3] / points_4d[3]
```

</details>

### Exercise 4: Mesh Reconstruction

Generate a 3D mesh from a point cloud.

**Requirements**:
- Point cloud preprocessing
- Normal vector estimation
- Poisson or ball pivoting reconstruction
- Save and visualize results

<details>
<summary>Hint</summary>

```python
# Normal estimation
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(k=15)

# Poisson reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# Remove low-density regions
densities = np.asarray(densities)
mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.01))
```

</details>

### Exercise 5: Real-time Stereo Vision

Implement real-time depth estimation using webcam or stereo camera.

**Requirements**:
- Apply camera calibration
- Real-time disparity calculation
- Depth visualization
- FPS measurement

<details>
<summary>Hint</summary>

```python
# Pre-compute remapping maps
map1_left, map2_left = cv2.initUndistortRectifyMap(...)
map1_right, map2_right = cv2.initUndistortRectifyMap(...)

while True:
    # Rectification
    rect_left = cv2.remap(left, map1_left, map2_left, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, map1_right, map2_right, cv2.INTER_LINEAR)

    # Disparity calculation (SGBM)
    disparity = stereo.compute(rect_left, rect_right)
```

</details>

---

## Next Steps

- [22_Depth_Estimation.md](./22_Depth_Estimation.md) - Monocular depth estimation, MiDaS, DPT, Structure from Motion

---

## References

- [OpenCV Stereo Vision Tutorial](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- [Open3D Documentation](http://www.open3d.org/docs/)
- [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- [Structure from Motion Tutorial](https://github.com/colmap/colmap)
- [Stereo Vision: A Tutorial](https://people.cs.rutgers.edu/~elgammal/classes/cs534/lectures/Stereo_2.pdf)
