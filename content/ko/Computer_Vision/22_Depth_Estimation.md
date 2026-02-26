# 단안 깊이 추정 (Monocular Depth Estimation)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단안 깊이 추정(Monocular Depth Estimation) 문제를 설명하고 스테레오 기반 깊이 추정 방법과 비교할 수 있습니다.
2. OpenCV의 DNN 모듈을 통해 MiDaS 모델을 사용한 깊이 추론을 구현할 수 있습니다.
3. 고해상도 깊이 맵 생성을 위해 밀집 예측 트랜스포머(DPT, Dense Prediction Transformer)를 적용할 수 있습니다.
4. Structure from Motion(SfM) 파이프라인을 기술하고 기본적인 다중 뷰 깊이 추정 워크플로우를 구현할 수 있습니다.
5. 깊이 맵 출력을 분석하고 장면 이해(Scene Understanding) 및 3D 효과 같은 후속 작업에 적용할 수 있습니다.

---

## 개요

단안 깊이 추정은 단일 2D 이미지에서 픽셀별 깊이 정보를 추정하는 기술입니다. MiDaS, DPT 같은 딥러닝 모델과 Structure from Motion (SfM)을 통한 기하학적 접근 방법을 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: DNN 모듈, 특징점 검출/매칭, 카메라 캘리브레이션

---

## 목차

1. [단안 깊이 추정 개요](#1-단안-깊이-추정-개요)
2. [MiDaS 모델](#2-midas-모델)
3. [DPT (Dense Prediction Transformer)](#3-dpt-dense-prediction-transformer)
4. [Structure from Motion (SfM)](#4-structure-from-motion-sfm)
5. [깊이 맵 응용](#5-깊이-맵-응용)
6. [연습 문제](#6-연습-문제)

---

## 1. 단안 깊이 추정 개요

깊이(Depth)는 2D 이미지에서 누락된 차원입니다. 모든 픽셀은 색상과 밝기는 담고 있지만, 카메라로부터 얼마나 떨어져 있는지는 담고 있지 않습니다. 픽셀별 깊이를 복원하면 평면 이미지를 장면에 대한 계측적 이해로 전환할 수 있으며, 스테레오 리그(Stereo Rig)나 LiDAR와 같은 하드웨어 없이도 3D 재구성, 장애물 회피, 증강 현실(Augmented Reality) 같은 후속 작업을 가능하게 합니다.

### 왜 단안 깊이 추정인가?

```
Stereo vs Monocular Depth Estimation:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Stereo Vision                                                  │
│  ┌───────────┐    ┌───────────┐                                 │
│  │   📷      │    │     📷    │                                 │
│  │   Left    │◄──►│   Right   │  Two cameras required           │
│  └───────────┘    └───────────┘                                 │
│                                                                 │
│  Pros: Geometrically accurate, absolute depth measurement       │
│  Cons: Two cameras required, calibration mandatory              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Monocular Depth Estimation                                     │
│  ┌───────────┐                                                  │
│  │    📷     │  Single camera sufficient                        │
│  │  Single   │  Suitable for smartphones, drones, robots        │
│  └───────────┘                                                  │
│                                                                 │
│  Pros: Single camera, simple setup, suitable for mobile devices │
│  Cons: Relative depth, scale ambiguity, depends on training data│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 깊이 추정의 어려움

```
Inherent Ambiguity in Monocular Depth Estimation:

Infinitely many 3D scenes can produce the same 2D image

                        │
                        │
         ●              │         🎾  Small ball, close
        /│\             │
         │              │
                        │
                        │         🏀  Large ball, far
    ───────────────────[📷]───────────────────

Appears the same size!

Solutions:
1. Learned Prior Knowledge (Deep Learning)
   - Typical object sizes
   - Perspective rules
   - Texture gradients

2. Multiple Images (SfM)
   - Using viewpoint changes
   - Geometric constraints

3. Additional Sensors
   - LiDAR assistance
   - Structured light assistance
```

### 깊이 추정 방법론

```
Depth Estimation Approaches:

┌─────────────────────────────────────────────────────────────────┐
│ 1. Supervised Learning                                          │
│    - Train with RGB-D datasets                                  │
│    - Requires ground truth depth                                │
│    - Datasets: NYU Depth V2, KITTI, ScanNet                    │
│                                                                 │
│ 2. Self-supervised Learning                                     │
│    - Train with stereo pairs or consecutive frames              │
│    - No ground truth required                                   │
│    - Monodepth2, PackNet-SfM                                   │
│                                                                 │
│ 3. Zero-shot Learning (Cross-domain)                            │
│    - Pre-trained on diverse datasets                            │
│    - Generalize to new domains                                  │
│    - MiDaS, DPT, ZoeDepth                                      │
│                                                                 │
│ 4. Geometric Methods                                            │
│    - Structure from Motion                                      │
│    - Multi-View Stereo                                          │
│    - Use explicit geometric constraints                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. MiDaS 모델

### MiDaS 개요

```
MiDaS (Mixing Datasets for Monocular Depth Estimation):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Key Idea: Improve generalization by mixing diverse datasets    │
│                                                                 │
│  Training Data:                                                 │
│  - ReDWeb (internet images)                                     │
│  - DIML (indoor)                                                │
│  - Movies (movie scenes)                                        │
│  - MegaDepth (outdoor)                                          │
│  - WSVD (video)                                                 │
│                                                                 │
│  Features:                                                      │
│  - Scale-invariant loss function                                │
│  - Relative depth prediction                                    │
│  - Various backbones (EfficientNet, ResNeXt, ViT)              │
│                                                                 │
│  Model Versions:                                                │
│  ┌──────────────────┬───────────┬─────────────────────────┐     │
│  │ Model            │ Input Size│ Features                │     │
│  ├──────────────────┼───────────┼─────────────────────────┤     │
│  │ MiDaS v2.1 Large │ 384x384   │ High quality, slow      │     │
│  │ MiDaS v2.1 Small │ 256x256   │ Lightweight, fast       │     │
│  │ MiDaS v3 (DPT)   │ 384x384   │ Transformer-based       │     │
│  │ MiDaS v3.1 (DPT) │ Various   │ Latest, various backbones│    │
│  └──────────────────┴───────────┴─────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MiDaS 사용하기

```python
import cv2
import numpy as np
import torch

def load_midas_model(model_type='DPT_Large'):
    """Load MiDaS model (PyTorch Hub)"""

    # Model types:
    # - 'DPT_Large': Most accurate
    # - 'DPT_Hybrid': Balanced
    # - 'MiDaS_small': Fastest

    model = torch.hub.load('intel-isl/MiDaS', model_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load preprocessing transforms
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

    if model_type in ['DPT_Large', 'DPT_Hybrid']:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return model, transform, device

def estimate_depth_midas(img, model, transform, device):
    """Estimate depth with MiDaS"""

    # BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocessing
    input_batch = transform(img_rgb).to(device)

    # Inference
    with torch.no_grad():
        prediction = model(input_batch)

        # Resize to original size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    return depth_map

def normalize_depth(depth_map):
    """Normalize depth map (for visualization)"""

    depth_min = depth_map.min()
    depth_max = depth_map.max()

    # MiDaS outputs *relative* inverse depth (larger value = closer object),
    # not metric meters. Min-max normalization maps the scene-specific range
    # to [0, 255] so the colormap spans the full visible range regardless of
    # the actual depth scale — suitable for visualization but not metric use.
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    depth_normalized = (depth_normalized * 255).astype(np.uint8)

    return depth_normalized

def colorize_depth(depth_map, colormap=cv2.COLORMAP_INFERNO):
    """Apply colormap to depth map"""

    depth_norm = normalize_depth(depth_map)
    depth_colored = cv2.applyColorMap(depth_norm, colormap)

    return depth_colored

# Usage example
def main():
    # Load model
    print("Loading model...")
    model, transform, device = load_midas_model('DPT_Large')

    # Load image
    img = cv2.imread('sample.jpg')

    # Estimate depth
    print("Estimating depth...")
    depth = estimate_depth_midas(img, model, transform, device)

    # Visualization
    depth_colored = colorize_depth(depth)

    cv2.imshow('Original', img)
    cv2.imshow('Depth', depth_colored)
    cv2.waitKey(0)
```

### OpenCV DNN으로 MiDaS 실행

```python
import cv2
import numpy as np

class MiDaSDepthEstimator:
    """Run MiDaS with OpenCV DNN"""

    def __init__(self, model_path):
        """
        model_path: ONNX model path
        Download: https://github.com/isl-org/MiDaS/releases
        """
        self.net = cv2.dnn.readNetFromONNX(model_path)

        # Use GPU (if available)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Input size (depends on model)
        self.input_size = (384, 384)  # DPT_Large
        # self.input_size = (256, 256)  # MiDaS_small

    def estimate(self, img):
        """Estimate depth"""

        h, w = img.shape[:2]

        # Preprocessing
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1/255.0,
            size=self.input_size,
            mean=(0.485, 0.456, 0.406),  # ImageNet mean
            swapRB=True,
            crop=False
        )

        # Standard deviation normalization (manual)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        blob = blob / std

        # Inference
        self.net.setInput(blob)
        output = self.net.forward()

        # Post-processing
        depth = output[0, 0]

        # Resize to original size
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

        return depth

    def visualize(self, depth, colormap=cv2.COLORMAP_MAGMA):
        """Visualize depth map"""

        # cv2.NORM_MINMAX stretches the full depth range to [0, 255];
        # this is purely for display — the original float depth values
        # are what you pass to any downstream 3D computation
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_norm, colormap)

        return depth_colored

# Usage example
estimator = MiDaSDepthEstimator('midas_v21_384.onnx')

img = cv2.imread('sample.jpg')
depth = estimator.estimate(img)
depth_vis = estimator.visualize(depth)

cv2.imshow('Depth', depth_vis)
cv2.waitKey(0)
```

---

## 3. DPT (Dense Prediction Transformer)

### DPT 아키텍처

```
DPT (Dense Prediction Transformer):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Vision Transformer (ViT)-based dense prediction model         │
│                                                                 │
│  Input: Image (H × W × 3)                                       │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Patch Embedding                                        │    │
│  │  Split image into patches and embed                     │    │
│  │  Patch size: 16×16                                      │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Transformer Encoder                                    │    │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐               │    │
│  │  │ Block │→│ Block │→│ Block │→│ Block │               │    │
│  │  └───────┘ └───────┘ └───────┘ └───────┘               │    │
│  │     │          │          │          │                  │    │
│  │     └──────────┼──────────┼──────────┘                  │    │
│  │                ▼          ▼          ▼                  │    │
│  │         Multi-scale feature extraction                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Reassemble + Fusion                                    │    │
│  │  Multi-scale feature fusion                             │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Head (Conv Layers)                                     │    │
│  │  Final depth map output                                 │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  Output: Depth Map (H × W)                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### DPT 구현

```python
import cv2
import numpy as np
import torch
from torchvision import transforms

class DPTDepthEstimator:
    """DPT Depth Estimator"""

    def __init__(self, model_type='DPT_Large'):
        """
        model_type: 'DPT_Large', 'DPT_Hybrid', 'DPT_SwinV2_L_384'
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Load model from PyTorch Hub
        self.model = torch.hub.load('intel-isl/MiDaS', model_type)
        self.model.to(self.device)
        self.model.eval()

        # Load preprocessing transforms
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = midas_transforms.dpt_transform

    def estimate(self, img):
        """Estimate depth"""

        h, w = img.shape[:2]

        # BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocessing and inference
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode='bicubic',
                align_corners=False
            ).squeeze()

        depth = prediction.cpu().numpy()

        return depth

    def get_metric_depth(self, depth, scale=10.0):
        """Relative depth → Metric depth conversion (approximation)"""

        # MiDaS/DPT outputs *inverse* relative depth: high values = close objects.
        # The stereo formula Z = f*b/d shows depth is inversely proportional to
        # disparity, so dividing a constant by the predicted value converts the
        # network's output back to a metric-like scale. The +1e-6 prevents
        # division by zero in far-background regions where predicted depth ≈ 0.
        # 'scale' is scene-dependent and must be calibrated against known distances
        # for truly metric output — this is an approximation for relative use.
        depth_metric = scale / (depth + 1e-6)

        return depth_metric

def estimate_depth_with_confidence(estimator, img, num_samples=5):
    """Estimate depth uncertainty with Monte Carlo dropout"""

    # Note: Actually requires a model with dropout
    # Here we substitute with data augmentation

    depths = []

    for _ in range(num_samples):
        # Slight image variation: perturbing brightness simulates real-world
        # exposure changes and gives a rough uncertainty estimate across plausible
        # inputs — a pragmatic substitute when the model lacks explicit dropout layers
        augmented = img.copy()

        # Brightness change
        factor = np.random.uniform(0.9, 1.1)
        augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)

        depth = estimator.estimate(augmented)
        depths.append(depth)

    depths = np.stack(depths, axis=0)

    # Mean and standard deviation
    mean_depth = np.mean(depths, axis=0)
    std_depth = np.std(depths, axis=0)

    return mean_depth, std_depth
```

### Depth Anything 모델

```python
# Depth Anything: More recent SOTA model

class DepthAnythingEstimator:
    """Depth Anything Model (2024)"""

    def __init__(self, model_size='small'):
        """
        model_size: 'small', 'base', 'large'
        """
        from transformers import pipeline

        model_name = f"LiheYoung/depth-anything-{model_size}-hf"
        self.pipe = pipeline(
            task='depth-estimation',
            model=model_name
        )

    def estimate(self, img):
        """Estimate depth"""

        # BGR → RGB, PIL conversion
        from PIL import Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Inference
        result = self.pipe(img_pil)

        # Extract depth map
        depth = np.array(result['depth'])

        # Resize to original size
        if depth.shape[:2] != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]))

        return depth
```

---

## 4. Structure from Motion (SfM)

### SfM 개요

```
Structure from Motion (SfM):
Recover 3D structure using camera motion

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Input: Consecutive images (video or multi-view images)        │
│                                                                 │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                   │
│  │ t=1 │  │ t=2 │  │ t=3 │  │ t=4 │  │ t=5 │                   │
│  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘                   │
│      │       │       │       │       │                          │
│      └───────┴───────┴───────┴───────┘                          │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────────┐                          │
│          │  1. Feature Detection     │                          │
│          │     and Matching          │                          │
│          │     SIFT, ORB, SuperPoint │                          │
│          └───────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────────┐                          │
│          │  2. Camera Pose Estimation│                          │
│          │     Essential Matrix      │                          │
│          │     PnP                   │                          │
│          └───────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────────┐                          │
│          │  3. Triangulation         │                          │
│          │     3D Point Recovery     │                          │
│          └───────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────────┐                          │
│          │  4. Bundle Adjustment     │                          │
│          │     Global Optimization   │                          │
│          └───────────────────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│  Output: 3D Point Cloud + Camera Trajectory                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SfM 구현 (간단한 버전)

```python
import cv2
import numpy as np

class SimpleSfM:
    """Simple 2-view SfM implementation"""

    def __init__(self, K):
        """
        K: Camera intrinsic parameter matrix
        """
        self.K = K
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def detect_and_match(self, img1, img2):
        """Feature detection and matching"""

        # Feature detection
        kp1, desc1 = self.sift.detectAndCompute(img1, None)
        kp2, desc2 = self.sift.detectAndCompute(img2, None)

        # Matching
        matches = self.bf.knnMatch(desc1, desc2, k=2)

        # Ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Match point coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return pts1, pts2, good_matches, kp1, kp2

    def estimate_pose(self, pts1, pts2):
        """Estimate pose from Essential Matrix"""

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        # Recover R, t
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        return R, t, mask.ravel().astype(bool)

    def triangulate(self, pts1, pts2, R, t):
        """Triangulate to recover 3D points"""

        # Projection matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])

        # Triangulation
        pts1_h = pts1.T  # (2, N)
        pts2_h = pts2.T

        points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)

        # Homogeneous → Euclidean coordinates
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T  # (N, 3)

    def filter_points(self, pts1, pts2, points_3d, R, t):
        """Filter valid 3D points"""

        # Calculate reprojection error
        P2 = self.K @ np.hstack([R, t])

        projected = P2 @ np.hstack([points_3d, np.ones((len(points_3d), 1))]).T
        projected = projected[:2] / projected[2]
        projected = projected.T

        errors = np.linalg.norm(pts2 - projected, axis=1)

        # Check if in front of camera
        # First camera reference
        valid_depth1 = points_3d[:, 2] > 0

        # Second camera reference
        points_cam2 = (R @ points_3d.T + t).T
        valid_depth2 = points_cam2[:, 2] > 0

        # Reprojection error threshold
        valid_reproj = errors < 2.0

        valid = valid_depth1 & valid_depth2 & valid_reproj

        return points_3d[valid], valid

    def run(self, img1, img2):
        """Run complete SfM pipeline"""

        # 1. Feature matching
        pts1, pts2, matches, kp1, kp2 = self.detect_and_match(img1, img2)
        print(f"Match points: {len(pts1)}")

        # 2. Pose estimation
        R, t, inlier_mask = self.estimate_pose(pts1, pts2)
        pts1 = pts1[inlier_mask]
        pts2 = pts2[inlier_mask]
        print(f"Inliers: {len(pts1)}")

        # 3. Triangulation
        points_3d = self.triangulate(pts1, pts2, R, t)

        # 4. Filtering
        points_3d, valid = self.filter_points(pts1, pts2, points_3d, R, t)
        print(f"Valid 3D points: {len(points_3d)}")

        return points_3d, R, t

# Usage example
K = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

sfm = SimpleSfM(K)
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
points_3d, R, t = sfm.run(img1, img2)
```

### 다중 뷰 SfM

```python
class IncrementalSfM:
    """Incremental SfM"""

    def __init__(self, K):
        self.K = K
        self.sift = cv2.SIFT_create(nfeatures=8000)
        self.bf = cv2.BFMatcher()

        # Global data
        self.points_3d = None
        self.point_colors = None
        self.camera_poses = []
        self.keypoints_all = []
        self.descriptors_all = []

    def add_image(self, img):
        """Add new image"""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)

        self.keypoints_all.append(kp)
        self.descriptors_all.append(desc)

        return len(self.keypoints_all) - 1

    def initialize(self, idx1, idx2):
        """Initialize with first two images"""

        # Matching
        matches = self.bf.knnMatch(
            self.descriptors_all[idx1],
            self.descriptors_all[idx2],
            k=2
        )

        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        pts1 = np.float32([self.keypoints_all[idx1][m.queryIdx].pt for m in good])
        pts2 = np.float32([self.keypoints_all[idx2][m.trainIdx].pt for m in good])

        # Essential Matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        mask = mask.ravel().astype(bool)
        pts1 = pts1[mask]
        pts2 = pts2[mask]

        # Triangulation
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])

        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        self.points_3d = (points_4d[:3] / points_4d[3]).T

        # Store camera poses
        self.camera_poses = [
            {'R': np.eye(3), 't': np.zeros((3, 1))},
            {'R': R, 't': t}
        ]

        print(f"Initialization complete: {len(self.points_3d)} 3D points")

    def register_image(self, idx):
        """Register new image (PnP)"""

        if self.points_3d is None or len(self.points_3d) == 0:
            print("Initialization required first.")
            return False

        # Match with last added image
        last_idx = len(self.camera_poses) - 1

        matches = self.bf.knnMatch(
            self.descriptors_all[last_idx],
            self.descriptors_all[idx],
            k=2
        )

        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) < 8:
            print("Insufficient matches")
            return False

        # 3D-2D correspondences (simplified: use previous image match indices)
        # In practice, track management is needed
        obj_points = []
        img_points = []

        for m in good[:len(self.points_3d)]:
            if m.queryIdx < len(self.points_3d):
                obj_points.append(self.points_3d[m.queryIdx])
                img_points.append(
                    self.keypoints_all[idx][m.trainIdx].pt
                )

        if len(obj_points) < 6:
            print("Insufficient correspondences")
            return False

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points, img_points, self.K, None
        )

        if not success:
            print("PnP failed")
            return False

        R, _ = cv2.Rodrigues(rvec)
        self.camera_poses.append({'R': R, 't': tvec})

        print(f"Image {idx} registered")
        return True

    def bundle_adjust(self):
        """Bundle adjustment (using scipy)"""

        from scipy.optimize import least_squares

        # Simple bundle adjustment implementation
        # In practice, recommend using g2o, Ceres, etc.

        print("Bundle adjustment: recommend specialized libraries (g2o, Ceres)")

    def get_point_cloud(self):
        """Return point cloud"""
        return self.points_3d

    def get_camera_trajectory(self):
        """Return camera trajectory"""
        positions = []
        for pose in self.camera_poses:
            R = pose['R']
            t = pose['t']
            # Camera position = -R^T * t
            pos = -R.T @ t
            positions.append(pos.ravel())

        return np.array(positions)
```

---

## 5. 깊이 맵 응용

### 깊이 기반 이미지 효과

```python
import cv2
import numpy as np

def apply_bokeh_effect(img, depth, focus_depth=0.5, aperture=0.1):
    """Depth-based bokeh effect (depth of field simulation)"""

    # Normalize depth to [0, 1] so focus_depth and aperture are scene-independent
    # parameters — 0.5 always means "mid-range" regardless of actual distance units
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # Calculate deviation from focus distance
    depth_diff = np.abs(depth_norm - focus_depth)

    # Blur strength proportional to distance from focus plane, capped at 31
    # so kernel sizes stay odd (2*level+1) and within OpenCV's supported range
    blur_strength = (depth_diff / aperture * 30).astype(int)
    blur_strength = np.clip(blur_strength, 0, 31)

    # Apply blur (different strength per pixel)
    result = np.zeros_like(img, dtype=np.float32)

    for blur_level in range(0, 32, 2):
        mask = (blur_strength >= blur_level) & (blur_strength < blur_level + 2)

        if blur_level == 0:
            blurred = img.astype(np.float32)
        else:
            ksize = blur_level * 2 + 1
            blurred = cv2.GaussianBlur(img, (ksize, ksize), 0).astype(np.float32)

        result += blurred * mask[:, :, np.newaxis]

    return result.astype(np.uint8)

def create_depth_fog(img, depth, fog_color=(200, 200, 200), max_fog=0.8):
    """Depth-based fog effect"""

    # Normalize depth
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # Fog strength (stronger farther away)
    fog_factor = depth_norm * max_fog

    # Apply fog
    fog = np.full_like(img, fog_color, dtype=np.float32)
    result = img.astype(np.float32) * (1 - fog_factor[:, :, np.newaxis])
    result += fog * fog_factor[:, :, np.newaxis]

    return result.astype(np.uint8)

def depth_based_segmentation(img, depth, num_layers=5):
    """Depth-based layer segmentation"""

    # Normalize depth
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # Segment by depth intervals
    layers = []
    for i in range(num_layers):
        lower = i / num_layers
        upper = (i + 1) / num_layers
        mask = (depth_norm >= lower) & (depth_norm < upper)

        layer = np.zeros_like(img)
        layer[mask] = img[mask]
        layers.append(layer)

    return layers

def remove_background_with_depth(img, depth, threshold=0.5):
    """Depth-based background removal"""

    # Normalize depth
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # Foreground mask (parts closer than threshold)
    foreground_mask = depth_norm < threshold

    # Refine mask
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(
        foreground_mask.astype(np.uint8),
        cv2.MORPH_CLOSE, kernel
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN, kernel
    )

    # Remove background
    result = np.zeros_like(img)
    result[foreground_mask == 1] = img[foreground_mask == 1]

    return result, foreground_mask
```

### 3D 효과 생성

```python
def create_3d_ken_burns(img, depth, num_frames=60, zoom=0.1):
    """Ken Burns effect (3D camera movement)"""

    h, w = img.shape[:2]
    frames = []

    for i in range(num_frames):
        t = i / (num_frames - 1)

        # Zoom factor
        scale = 1 + zoom * t

        # Parallax by depth
        parallax = (depth - depth.mean()) * 0.001 * t

        # Calculate new coordinates
        y_coords, x_coords = np.meshgrid(range(h), range(w), indexing='ij')

        # Center-based scaling
        new_x = (x_coords - w/2) / scale + w/2 + parallax
        new_y = (y_coords - h/2) / scale + h/2

        # Remapping
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)

        frame = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        frames.append(frame)

    return frames

def depth_aware_zoom(img, depth, zoom_center, zoom_factor=2.0):
    """Depth-aware zoom"""

    h, w = img.shape[:2]
    cx, cy = zoom_center

    # Normalize depth
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # Apply different zoom by depth (closer objects zoom more)
    depth_factor = 1 - depth_norm * 0.5  # 0.5 ~ 1.0

    # Coordinate grid
    y_coords, x_coords = np.meshgrid(range(h), range(w), indexing='ij')

    # Zoom transform (different scale per depth)
    effective_zoom = zoom_factor * depth_factor

    new_x = (x_coords - cx) / effective_zoom + cx
    new_y = (y_coords - cy) / effective_zoom + cy

    # Remapping
    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)

    result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    return result
```

---

## 6. 연습 문제

### 문제 1: MiDaS 깊이 추정

MiDaS를 사용하여 이미지의 깊이를 추정하세요.

**요구사항**:
- 모델 로드 및 추론
- 깊이 맵 시각화 (컬러맵)
- 여러 이미지에 대해 테스트

<details>
<summary>힌트</summary>

```python
import torch

model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = midas_transforms.dpt_transform
```

</details>

### 문제 2: 깊이 기반 배경 블러

인물 사진에서 배경만 블러 처리하세요.

**요구사항**:
- 깊이 추정
- 전경/배경 분리
- 배경에만 블러 적용
- 자연스러운 경계 처리

<details>
<summary>힌트</summary>

```python
# Depth-based mask generation
threshold = np.percentile(depth, 30)  # Treat closest 30% as foreground
foreground_mask = depth < threshold

# Blur mask (smooth boundaries)
mask_blur = cv2.GaussianBlur(
    foreground_mask.astype(np.float32), (21, 21), 0
)

# Background blur
background_blur = cv2.GaussianBlur(img, (25, 25), 0)

# Composite
result = img * mask_blur[..., None] + background_blur * (1 - mask_blur[..., None])
```

</details>

### 문제 3: SfM으로 3D 복원

두 이미지에서 3D 포인트 클라우드를 복원하세요.

**요구사항**:
- 특징점 매칭
- Essential Matrix 계산
- 삼각측량
- 포인트 클라우드 시각화

<details>
<summary>힌트</summary>

```python
# Essential Matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# Projection matrices
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = K @ np.hstack([R, t])

# Triangulation
points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3d = points_4d[:3] / points_4d[3]
```

</details>

### 문제 4: 실시간 깊이 추정

웹캠으로 실시간 깊이 추정을 구현하세요.

**요구사항**:
- 경량 모델 사용 (MiDaS small)
- FPS 측정 및 표시
- 깊이 시각화

<details>
<summary>힌트</summary>

```python
# Lightweight model
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')

while True:
    ret, frame = cap.read()

    start = time.time()
    depth = estimate_depth(frame, model, transform)
    fps = 1.0 / (time.time() - start)

    cv2.putText(depth_vis, f"FPS: {fps:.1f}", ...)
```

</details>

### 문제 5: 깊이 기반 3D 뷰어

깊이 맵을 이용해 간단한 3D 뷰어를 만드세요.

**요구사항**:
- 깊이 맵 → 포인트 클라우드 변환
- Open3D로 시각화
- 마우스로 회전/줌

<details>
<summary>힌트</summary>

```python
import open3d as o3d

# Create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

# Visualization
o3d.visualization.draw_geometries([pcd])
```

</details>

---

## 다음 단계

- [SLAM 입문 (Visual SLAM Introduction)](./23_SLAM_Introduction.md) - Visual SLAM, ORB-SLAM, LiDAR SLAM, Loop Closure

---

## 참고 자료

- [MiDaS GitHub](https://github.com/isl-org/MiDaS)
- [DPT Paper](https://arxiv.org/abs/2103.13413)
- [Depth Anything](https://github.com/LiheYoung/Depth-Anything)
- [Structure from Motion Tutorial](https://cmsc426.github.io/sfm/)
- [OpenCV SfM Tutorial](https://docs.opencv.org/4.x/d4/d18/tutorial_sfm_scene_reconstruction.html)
- [Monodepth2](https://github.com/nianticlabs/monodepth2)
