# 3D 비전 기초 (3D Vision Basics)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 스테레오 비전(Stereo Vision)의 원리, 에피폴라 기하학(Epipolar Geometry), 시차-깊이(Disparity-to-Depth) 관계를 설명할 수 있습니다.
2. OpenCV의 스테레오 매칭 알고리즘(StereoBM, StereoSGBM)을 사용하여 깊이 맵(Depth Map)을 생성할 수 있습니다.
3. 포인트 클라우드(Point Cloud)를 정의하고 깊이 맵과 카메라 내부 파라미터로부터 생성하는 방법을 설명할 수 있습니다.
4. Open3D를 적용하여 3D 포인트 클라우드 데이터를 시각화, 처리, 분석할 수 있습니다.
5. 3D 재구성(3D Reconstruction) 파이프라인을 기술하고 기본적인 Structure from Motion 워크플로우를 구현할 수 있습니다.

---

## 개요

3D 비전은 2D 이미지로부터 3차원 정보를 추출하고 복원하는 기술입니다. 스테레오 비전, 깊이 맵, 포인트 클라우드 처리, 3D 재구성의 기초를 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: 카메라 캘리브레이션, 특징점 검출/매칭, 선형대수

---

## 목차

1. [3D 비전 개요](#1-3d-비전-개요)
2. [스테레오 비전 원리](#2-스테레오-비전-원리)
3. [깊이 맵 생성](#3-깊이-맵-생성)
4. [포인트 클라우드](#4-포인트-클라우드)
5. [Open3D 기초](#5-open3d-기초)
6. [3D 재구성](#6-3d-재구성)
7. [연습 문제](#7-연습-문제)

---

## 1. 3D 비전 개요

### 3D 비전의 목표

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

### 좌표계 이해

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

## 2. 스테레오 비전 원리

### 에피폴라 기하학

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

**에피폴라 제약 조건이 왜 중요한가?** 이 제약 조건이 없으면 왼쪽 이미지의 픽셀 p에 대응하는 점을 오른쪽 이미지 전체에서 검색해야 합니다 — 모든 픽셀에 대해 O(W×H) 문제입니다. 에피폴라 제약 조건은 대응점이 특정 선(에피폴라 선) 위에 있어야 한다고 알려주어 검색을 1D 문제로 줄입니다. 스테레오 정렬(rectification, 두 이미지를 에피폴라 선이 수평이 되도록 정렬) 후에는 같은 행을 따라 스캔하는 것으로 더욱 단순화됩니다 — 이것이 정렬이 시차(disparity) 계산 전 표준 전처리 단계인 이유입니다.

### 시차와 깊이

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

### 스테레오 정합

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

## 3. 깊이 맵 생성

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

### WLS 필터를 이용한 시차 개선

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

## 4. 포인트 클라우드

### 포인트 클라우드 생성

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

### 포인트 클라우드 처리

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

## 5. Open3D 기초

### Open3D 설치 및 기본 사용

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

### 메쉬 재구성

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

### RGBD 이미지 처리

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

## 6. 3D 재구성

### 다중 뷰 스테레오 (MVS) 개념

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

### Essential Matrix 기반 포즈 추정

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

### 번들 조정 (Bundle Adjustment)

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

## 7. 연습 문제

### 문제 1: 스테레오 깊이 추정

스테레오 이미지 쌍에서 깊이 맵을 생성하세요.

**요구사항**:
- StereoBM과 StereoSGBM 비교
- 시차 맵 시각화
- 깊이 맵으로 변환
- 품질 개선 (필터링)

<details>
<summary>힌트</summary>

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

### 문제 2: 포인트 클라우드 필터링

노이즈가 있는 포인트 클라우드를 정제하세요.

**요구사항**:
- 통계적 이상치 제거
- 복셀 다운샘플링
- 평면 영역 추출
- 결과 시각화

<details>
<summary>힌트</summary>

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

### 문제 3: 두 뷰에서 3D 재구성

두 이미지에서 3D 포인트를 재구성하세요.

**요구사항**:
- 특징점 검출 및 매칭
- Essential Matrix 계산
- 카메라 포즈 복구
- 삼각측량으로 3D 점 생성

<details>
<summary>힌트</summary>

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

### 문제 4: 메쉬 재구성

포인트 클라우드에서 3D 메쉬를 생성하세요.

**요구사항**:
- 포인트 클라우드 전처리
- 법선 벡터 추정
- 포아송 또는 볼 피벗팅 재구성
- 결과 저장 및 시각화

<details>
<summary>힌트</summary>

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

### 문제 5: 실시간 스테레오 비전

웹캠 또는 스테레오 카메라로 실시간 깊이 추정을 구현하세요.

**요구사항**:
- 카메라 캘리브레이션 적용
- 실시간 시차 계산
- 깊이 시각화
- FPS 측정

<details>
<summary>힌트</summary>

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

## 다음 단계

- [단안 깊이 추정 (Monocular Depth Estimation)](./22_Depth_Estimation.md) - 단안 깊이 추정, MiDaS, DPT, Structure from Motion

---

## 참고 자료

- [OpenCV Stereo Vision Tutorial](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- [Open3D Documentation](http://www.open3d.org/docs/)
- [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- [Structure from Motion Tutorial](https://github.com/colmap/colmap)
- [Stereo Vision: A Tutorial](https://people.cs.rutgers.edu/~elgammal/classes/cs534/lectures/Stereo_2.pdf)
