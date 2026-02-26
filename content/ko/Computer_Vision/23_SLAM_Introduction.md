# SLAM 입문 (Visual SLAM Introduction)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. SLAM(Simultaneous Localization and Mapping)을 정의하고 미지의 환경에서 동시 위치 추정과 지도 작성이 어떻게 이루어지는지 설명할 수 있습니다.
2. 특징점 매칭(Feature Matching)을 사용한 비주얼 오도메트리(Visual Odometry)를 구현하여 연속 프레임 간 카메라 자세를 추정할 수 있습니다.
3. ORB-SLAM 파이프라인의 특징점 추출, 추적(Tracking), 지역 매핑(Local Mapping), 루프 클로저(Loop Closure) 단계를 설명할 수 있습니다.
4. 센서 특성, 정확도, 계산 비용 측면에서 Visual SLAM과 LiDAR SLAM을 비교할 수 있습니다.
5. 루프 클로저(Loop Closure) 검출의 개념과 SLAM 시스템에서 누적 드리프트를 보정하는 역할을 설명할 수 있습니다.
6. 기본적인 SLAM 시뮬레이션을 구현하고 실측 데이터(Ground Truth)와 비교하여 궤적 정확도를 평가할 수 있습니다.

---

## 개요

SLAM (Simultaneous Localization and Mapping)은 로봇이나 자율주행 시스템이 미지의 환경에서 지도를 작성하면서 동시에 자신의 위치를 추정하는 기술입니다. Visual SLAM, LiDAR SLAM, Loop Closure의 기초를 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: 3D 비전, 특징점 검출/매칭, 카메라 캘리브레이션, 기본 확률론

---

## 목차

1. [SLAM 개요](#1-slam-개요)
2. [Visual Odometry](#2-visual-odometry)
3. [ORB-SLAM](#3-orb-slam)
4. [LiDAR SLAM](#4-lidar-slam)
5. [Loop Closure](#5-loop-closure)
6. [SLAM 구현 실습](#6-slam-구현-실습)
7. [연습 문제](#7-연습-문제)

---

## 1. SLAM 개요

### SLAM이란?

SLAM은 자율 주행에서 근본적인 부트스트랩 문제를 해결합니다. 위치를 파악하려면 지도가 필요하고, 지도를 만들려면 위치를 알아야 합니다. SLAM은 이 두 가지를 순차적으로 해결하는 대신, 로봇의 궤적과 지도 모두에 대한 결합 확률 분포(Joint Probability Distribution)를 유지하며 새로운 센서 데이터가 들어올 때마다 두 가지를 동시에 업데이트합니다.

SLAM 사후 확률(Posterior)은 이 결합 불확실성을 포착합니다:

**p(x_{1:t}, m | z_{1:t}, u_{1:t})**

- **x_{1:t}**: 로봇의 전체 궤적 (매 시간 단계의 자세)
- **m**: 지도 (랜드마크 위치 또는 점유 격자 셀)
- **z_{1:t}**: 지금까지의 모든 센서 관측 (픽셀 특징점, 레이저 반사값)
- **u_{1:t}**: 모든 제어 입력 (휠 오도메트리(Wheel Odometry), IMU)

핵심 통찰은 관측값 z가 자세와 지도를 *동시에* 제약한다는 것입니다. 로봇이 알려진 랜드마크를 다시 방문할 때, 두 값 모두에 대한 불확실성이 줄어듭니다. 이것이 루프 클로저(Loop Closure)가 그렇게 극적인 보정을 만들어내는 이유입니다.

```
SLAM (Simultaneous Localization and Mapping):
Simultaneous localization and mapping

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Key Questions:                                                 │
│  "How can you know your position without a map?"                │
│  "How can you build a map without knowing your position?"       │
│                                                                 │
│  → Solve both simultaneously! (Chicken and egg problem)         │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │                                                        │     │
│  │     Sensor Data                                        │     │
│  │     (Camera, LiDAR, IMU)                               │     │
│  │            │                                           │     │
│  │            ▼                                           │     │
│  │     ┌──────────────┐                                   │     │
│  │     │    SLAM      │                                   │     │
│  │     │  Algorithm   │                                   │     │
│  │     └──────┬───────┘                                   │     │
│  │            │                                           │     │
│  │     ┌──────┴───────┐                                   │     │
│  │     │              │                                   │     │
│  │     ▼              ▼                                   │     │
│  │  ┌─────────┐  ┌─────────┐                             │     │
│  │  │   Map   │  │  Pose   │                             │     │
│  │  │  (Map)  │  │ (Pose)  │                             │     │
│  │  └─────────┘  └─────────┘                             │     │
│  │                                                        │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Applications:
┌─────────────────┬─────────────────────────────────────────┐
│ Field           │ Examples                                │
├─────────────────┼─────────────────────────────────────────┤
│ Autonomous      │ Cars, drones, delivery robots           │
│ Driving         │                                         │
├─────────────────┼─────────────────────────────────────────┤
│ Augmented       │ ARKit, ARCore, HoloLens                 │
│ Reality         │                                         │
├─────────────────┼─────────────────────────────────────────┤
│ Robot Vacuum    │ Roomba, Roborock                        │
│ Cleaners        │                                         │
├─────────────────┼─────────────────────────────────────────┤
│ 3D Scanning     │ Architecture, cultural heritage         │
│                 │ restoration                             │
├─────────────────┼─────────────────────────────────────────┤
│ Navigation      │ Indoor localization                     │
└─────────────────┴─────────────────────────────────────────┘
```

### SLAM 분류

```
SLAM Method Classification:

1. Sensor-based Classification
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Visual SLAM (V-SLAM)                                           │
│  - Camera (monocular, stereo, RGB-D)                            │
│  - Feature-based or direct methods                              │
│  - Examples: ORB-SLAM, LSD-SLAM, DSO                            │
│                                                                 │
│  LiDAR SLAM                                                     │
│  - Laser scanner                                                │
│  - Point cloud matching                                         │
│  - Examples: Cartographer, LOAM, LeGO-LOAM                      │
│                                                                 │
│  Visual-Inertial SLAM                                           │
│  - Camera + IMU fusion                                          │
│  - Examples: VINS-Mono, OKVIS, MSCKF                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

2. Methodology-based Classification
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Filter-based                                                   │
│  - EKF-SLAM, UKF-SLAM                                           │
│  - Real-time updates                                            │
│  - Linearization error accumulation issues                      │
│                                                                 │
│  Graph-based                                                    │
│  - Pose graph optimization                                      │
│  - Bundle adjustment                                            │
│  - More accurate but computationally expensive                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

3. Front-end/Back-end
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Front-end                                                      │
│  - Sensor data processing                                       │
│  - Feature extraction and matching                              │
│  - Initial pose estimation                                      │
│  - Loop closure detection                                       │
│                                                                 │
│  Back-end                                                       │
│  - Global optimization                                          │
│  - Graph optimization                                           │
│  - Uncertainty estimation                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Visual Odometry

### Visual Odometry 개념

```
Visual Odometry (VO):
Estimating camera motion from consecutive images

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Frame t-1        Frame t          Frame t+1                    │
│  ┌───────┐        ┌───────┐        ┌───────┐                   │
│  │   📷  │──T₁───▶│   📷  │──T₂───▶│   📷  │                   │
│  └───────┘        └───────┘        └───────┘                   │
│                                                                 │
│  Accumulated Pose: P_t = T₁ * T₂ * ... * T_t                    │
│                                                                 │
│  Problems:                                                      │
│  - Accumulated drift                                            │
│  - Scale ambiguity (monocular camera)                           │
│  - Vulnerable to fast motion                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

VO Pipeline:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. Image Acquisition                                           │
│       ▼                                                         │
│  2. Feature Extraction (ORB, SIFT, Harris corners)              │
│       ▼                                                         │
│  3. Feature Matching/Tracking (BF Matcher, Optical Flow)        │
│       ▼                                                         │
│  4. Motion Estimation (Essential Matrix, PnP)                   │
│       ▼                                                         │
│  5. Local Optimization (Local BA)                               │
│       ▼                                                         │
│  6. Pose Update                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 단안 Visual Odometry 구현

```python
import cv2
import numpy as np

class MonocularVO:
    """Monocular Visual Odometry"""

    def __init__(self, K, detector='ORB'):
        """
        K: Camera intrinsic parameter matrix
        detector: Feature detector ('ORB', 'SIFT', 'FAST')
        """
        self.K = K
        self.focal = K[0, 0]
        self.pp = (K[0, 2], K[1, 2])  # principal point

        # Feature detector
        if detector == 'ORB':
            # ORB is preferred over SIFT for real-time VO: binary descriptors
            # enable Hamming-distance matching (~10x faster than SIFT's L2),
            # and ORB is patent-free with comparable repeatability outdoors.
            self.detector = cv2.ORB_create(3000)
        elif detector == 'SIFT':
            self.detector = cv2.SIFT_create(3000)
        else:
            self.detector = cv2.FastFeatureDetector_create(threshold=25)

        # Optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),   # 21×21 search window: large enough to handle ~10px/frame motion
                                # without losing the feature, but small enough to stay on one surface
            maxLevel=3,         # 3-level pyramid lets us track features displaced up to ~80px
                                # (21 * 2^3) — covers typical camera motion between frames
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            # Stop when displacement < 0.01px OR after 30 iterations — trades accuracy vs speed
        )

        # State
        self.prev_frame = None
        self.prev_pts = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.trajectory = []

    def detect_features(self, img):
        """Detect features"""
        if hasattr(self.detector, 'detectAndCompute'):
            kp, _ = self.detector.detectAndCompute(img, None)
        else:
            kp = self.detector.detect(img, None)

        pts = np.array([p.pt for p in kp], dtype=np.float32)
        return pts.reshape(-1, 1, 2)

    def track_features(self, prev_img, cur_img, prev_pts):
        """Track features using optical flow"""

        cur_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_img, cur_img, prev_pts, None, **self.lk_params
        )

        status = status.reshape(-1)
        prev_pts = prev_pts[status == 1]
        cur_pts = cur_pts[status == 1]

        return prev_pts, cur_pts

    def estimate_pose(self, pts1, pts2):
        """Estimate pose using Essential Matrix"""

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,  # RANSAC discards outlier matches from incorrect tracking
                                # or moving objects; without it, even a few bad matches
                                # corrupt the entire Essential Matrix estimate
            prob=0.999,         # 99.9% confidence that at least one sample is outlier-free;
                                # higher than typical (0.99) because VO errors compound over time
            threshold=1.0       # 1.0px Sampson distance tolerance — tight enough to reject
                                # mismatches yet forgiving of sub-pixel calibration residuals
        )

        # recoverPose selects the unique R,t from the 4 possible decompositions
        # by checking which solution has the most points in front of both cameras
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        return R, t

    def process_frame(self, frame):
        """Process frame"""

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if self.prev_frame is None:
            # First frame
            self.prev_frame = gray
            self.prev_pts = self.detect_features(gray)
            return self.cur_R, self.cur_t

        # Track features
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            prev_pts, cur_pts = self.track_features(
                self.prev_frame, gray, self.prev_pts
            )

            if len(prev_pts) >= 8:
                # 8 is the theoretical minimum for the 8-point algorithm that
                # underlies findEssentialMat; in practice more is better, but
                # we need at least this many to have a determined linear system
                R, t = self.estimate_pose(
                    prev_pts.reshape(-1, 2),
                    cur_pts.reshape(-1, 2)
                )

                # Accumulate pose in world frame: translation must be rotated
                # by the *current* world-to-camera rotation before adding,
                # so that all increments are expressed in the same reference frame
                self.cur_t = self.cur_t + self.cur_R @ t
                self.cur_R = R @ self.cur_R

                # Detect new features if needed
                if len(cur_pts) < 1000:
                    # Replenish when tracked count falls below threshold —
                    # too few features degrades pose estimation accuracy and
                    # increases drift; 1000 is a practical balance for HD video
                    new_pts = self.detect_features(gray)
                    if len(cur_pts) > 0:
                        self.prev_pts = np.vstack([
                            cur_pts.reshape(-1, 1, 2),
                            new_pts
                        ])
                    else:
                        self.prev_pts = new_pts
                else:
                    self.prev_pts = cur_pts.reshape(-1, 1, 2)
            else:
                self.prev_pts = self.detect_features(gray)
        else:
            self.prev_pts = self.detect_features(gray)

        self.prev_frame = gray

        # Save trajectory
        self.trajectory.append(self.cur_t.copy())

        return self.cur_R, self.cur_t

    def get_trajectory(self):
        """Return trajectory"""
        return np.array([t.ravel() for t in self.trajectory])

# Usage example
K = np.array([
    [718.856, 0, 607.1928],
    [0, 718.856, 185.2157],
    [0, 0, 1]
], dtype=np.float32)

vo = MonocularVO(K)

cap = cv2.VideoCapture('driving.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    R, t = vo.process_frame(frame)

    # Print current position
    x, y, z = t.ravel()
    print(f"Position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

cap.release()

# Visualize trajectory
trajectory = vo.get_trajectory()
```

### 스테레오 Visual Odometry

```python
class StereoVO:
    """Stereo Visual Odometry"""

    def __init__(self, K, baseline, detector='ORB'):
        self.K = K
        self.baseline = baseline
        self.focal = K[0, 0]

        self.detector = cv2.ORB_create(3000)
        # NORM_HAMMING matches binary ORB descriptors by XOR bit-count,
        # which is much faster than L2 for floating-point descriptors like SIFT
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Stereo matcher — SGBM (Semi-Global Block Matching) enforces
        # disparity smoothness along multiple scan-line directions,
        # giving more complete depth maps than local block matching alone
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,  # Search range 0–128px; covers ~2–50m depth at typical baselines
            blockSize=5,         # 5×5 matching window: small enough to preserve edges,
                                 # large enough for reliable texture matching
            P1=8 * 3 * 5 ** 2,  # Penalty for disparity change of 1 — standard heuristic
            P2=32 * 3 * 5 ** 2  # Penalty for larger jumps; P2 > P1 discourages discontinuities
        )

        self.prev_pts_3d = None
        self.prev_kp = None
        self.prev_desc = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

    def compute_depth(self, left, right):
        """Compute depth using stereo matching"""

        disparity = self.stereo.compute(left, right).astype(np.float32) / 16.0

        # Disparity → depth
        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = self.focal * self.baseline / disparity[valid]

        return depth

    def get_3d_points(self, kp, depth):
        """Convert 2D keypoints to 3D"""

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        pts_3d = []
        valid_indices = []

        for i, pt in enumerate(kp):
            x, y = int(pt.pt[0]), int(pt.pt[1])

            if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                z = depth[y, x]

                if z > 0 and z < 100:  # 0–100m: reject invalid (0) and implausibly far points;
                                       # stereo becomes unreliable beyond ~baseline×50 range
                    X = (pt.pt[0] - cx) * z / fx
                    Y = (pt.pt[1] - cy) * z / fy
                    pts_3d.append([X, Y, z])
                    valid_indices.append(i)

        return np.array(pts_3d), valid_indices

    def process_frame(self, left, right):
        """Process stereo frame"""

        # Convert to grayscale
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Compute depth
        depth = self.compute_depth(gray_left, gray_right)

        # Detect features
        kp, desc = self.detector.detectAndCompute(gray_left, None)

        # Compute 3D points
        pts_3d, valid_idx = self.get_3d_points(kp, depth)

        if self.prev_pts_3d is None:
            self.prev_pts_3d = pts_3d
            self.prev_kp = [kp[i] for i in valid_idx]
            self.prev_desc = desc[valid_idx]
            return self.cur_R, self.cur_t

        # Match with previous frame
        matches = self.bf.knnMatch(self.prev_desc, desc[valid_idx], k=2)

        good_matches = []
        for m, n in matches:
            # Lowe's ratio test: accept only matches where the best is
            # significantly better than the second-best (0.7 threshold).
            # This rejects ambiguous matches in repetitive textures.
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 6:
            # 3D-2D correspondences
            obj_points = np.array([
                self.prev_pts_3d[m.queryIdx] for m in good_matches
            ])
            img_points = np.array([
                kp[valid_idx[m.trainIdx]].pt for m in good_matches
            ])

            # PnP (Perspective-n-Point): given 3D map points and their 2D
            # projections, recover camera pose directly — more stable than
            # Essential Matrix decomposition because depth is already known.
            # RANSAC variant handles moving objects and descriptor mismatches.
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_points, img_points, self.K, None
            )

            if success and inliers is not None and len(inliers) > 10:
                R, _ = cv2.Rodrigues(rvec)

                # Accumulate pose
                self.cur_t = self.cur_t + self.cur_R @ tvec
                self.cur_R = R @ self.cur_R

        # Update state
        self.prev_pts_3d = pts_3d
        self.prev_kp = [kp[i] for i in valid_idx]
        self.prev_desc = desc[valid_idx]

        return self.cur_R, self.cur_t
```

---

## 3. ORB-SLAM

### ORB-SLAM 개요

```
ORB-SLAM Architecture:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ORB-SLAM: Most widely used Visual SLAM system                  │
│                                                                 │
│  Versions:                                                      │
│  - ORB-SLAM (2015): Monocular                                   │
│  - ORB-SLAM2 (2017): Monocular/Stereo/RGB-D                     │
│  - ORB-SLAM3 (2021): Visual-Inertial, multi-map                 │
│                                                                 │
│  Three parallel threads:                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │  Tracking   │  │Local Mapping│  │Loop Closing │     │    │
│  │  │   Thread    │  │   Thread    │  │   Thread    │     │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │    │
│  │         │                │                │            │    │
│  │         │    Keyframes   │                │            │    │
│  │         └───────────────▶│                │            │    │
│  │                          │    Keyframes   │            │    │
│  │                          └───────────────▶│            │    │
│  │                                           │            │    │
│  │  ┌───────────────────────────────────────┐│            │    │
│  │  │           Map (MapPoints)             ││            │    │
│  │  │         & Covisibility Graph          ││            │    │
│  │  └───────────────────────────────────────┘│            │    │
│  │                                           │            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Tracking Thread:
- Process every frame
- ORB feature extraction
- Match with previous frame or map
- Initial pose estimation
- Keyframe decision

Local Mapping Thread:
- Insert new keyframes
- Cull recent MapPoints
- Create new MapPoints
- Local Bundle Adjustment
- Remove redundant keyframes

Loop Closing Thread:
- Detect loop candidates (DBoW2)
- Verify and correct loops
- Essential Graph optimization
- Global Bundle Adjustment
```

### ORB 특징과 Bag of Words

```python
import cv2
import numpy as np

class ORBVocabulary:
    """ORB-based Bag of Words"""

    def __init__(self, num_words=1000):
        self.orb = cv2.ORB_create(1000)
        self.num_words = num_words
        self.vocabulary = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def train(self, images):
        """Train vocabulary from images"""

        all_descriptors = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, desc = self.orb.detectAndCompute(gray, None)
            if desc is not None:
                all_descriptors.append(desc)

        all_desc = np.vstack(all_descriptors)

        # K-means clusters descriptors into "visual words" — each cluster
        # center becomes a prototype that any descriptor can be quantized to.
        # This lets us compare images by word-frequency histogram rather than
        # exhaustive descriptor matching, reducing query cost from O(N·M) to O(N).
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                   100, 0.2)
        _, labels, centers = cv2.kmeans(
            all_desc.astype(np.float32),
            self.num_words,  # Vocabulary size: more words → finer discrimination,
                             # but slower lookup and larger database memory footprint
            None,
            criteria,
            10,              # 10 random restarts pick the best clustering (lowest inertia)
            cv2.KMEANS_RANDOM_CENTERS
        )

        self.vocabulary = centers.astype(np.uint8)
        print(f"Vocabulary created: {self.num_words} words")

    def compute_bow(self, img):
        """Compute BoW vector for image"""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, desc = self.orb.detectAndCompute(gray, None)

        if desc is None:
            return np.zeros(self.num_words)

        # Assign each descriptor to nearest vocabulary word
        matches = self.bf.match(desc, self.vocabulary)

        bow = np.zeros(self.num_words)
        for m in matches:
            bow[m.trainIdx] += 1  # Accumulate word frequencies (term frequency)

        # L2-normalize so cosine similarity (dot product) is scale-invariant —
        # images with more features shouldn't score higher just because of feature count
        bow = bow / (np.linalg.norm(bow) + 1e-6)  # 1e-6 avoids division by zero on blank frames

        return bow

    def compute_similarity(self, bow1, bow2):
        """Similarity between two BoW vectors"""
        return np.dot(bow1, bow2)


class SimpleSLAM:
    """Simple SLAM system (ORB-SLAM concept)"""

    def __init__(self, K):
        self.K = K
        self.orb = cv2.ORB_create(2000)
        # crossCheck=True enforces mutual consistency: match A→B is kept only if
        # B also maps back to A as its nearest neighbor, eliminating one-sided matches.
        # This is simpler than Lowe's ratio test and works well for keyframe matching
        # where both descriptor sets are from the same detector.
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Map
        self.keyframes = []      # Keyframe list
        self.map_points = []     # 3D points
        self.poses = []          # Keyframe poses

        # Current state
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None

        # Keyframe criteria: 30 matches is the practical lower bound before
        # Essential Matrix estimation becomes unreliable. Below this, RANSAC
        # may converge to a degenerate solution (e.g., pure rotation).
        self.kf_threshold = 30   # Minimum matches

    def is_keyframe(self, num_matches, motion):
        """Decide if keyframe"""

        # Simple criteria: keyframe if few matches or large motion
        translation = np.linalg.norm(motion)

        if num_matches < self.kf_threshold or translation > 0.5:
            return True
        return False

    def add_keyframe(self, frame, kp, desc, pose):
        """Add keyframe"""

        keyframe = {
            'frame': frame.copy(),
            'keypoints': kp,
            'descriptors': desc,
            'pose': pose.copy()
        }

        self.keyframes.append(keyframe)
        self.poses.append(pose)

        print(f"Keyframe added: total {len(self.keyframes)}")

    def process_frame(self, frame):
        """Process frame (Tracking)"""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)

        if self.prev_frame is None:
            # First frame → keyframe
            pose = {'R': np.eye(3), 't': np.zeros((3, 1))}
            self.add_keyframe(gray, kp, desc, pose)
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_desc = desc
            return self.cur_R, self.cur_t

        # Match with previous frame
        matches = self.bf.match(self.prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)[:500]

        if len(matches) >= 8:
            # Extract matched points
            pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

            # Estimate pose using Essential Matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, self.K)
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

            # Accumulate pose
            self.cur_t = self.cur_t + self.cur_R @ t
            self.cur_R = R @ self.cur_R

            # Check keyframe
            if self.is_keyframe(len(matches), t):
                pose = {'R': self.cur_R.copy(), 't': self.cur_t.copy()}
                self.add_keyframe(gray, kp, desc, pose)

        # Update state
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_desc = desc

        return self.cur_R, self.cur_t

    def get_camera_trajectory(self):
        """Return camera trajectory"""
        trajectory = []
        for pose in self.poses:
            R = pose['R']
            t = pose['t']
            # Camera position in world frame = -R^T * t
            # Because t is the world origin expressed in camera coordinates,
            # we invert the transform: world position = R^T * (-t)
            pos = -R.T @ t
            trajectory.append(pos.ravel())
        return np.array(trajectory)
```

---

## 4. LiDAR SLAM

### LiDAR SLAM 개요

```
LiDAR SLAM:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  LiDAR Sensor Characteristics:                                  │
│  - 360-degree scanning                                          │
│  - Accurate distance measurement                                │
│  - Robust to lighting conditions                                │
│  - Rich 3D point clouds                                         │
│                                                                 │
│  LiDAR Types:                                                   │
│  ┌──────────────────┬─────────────────────────────────────┐     │
│  │ 2D LiDAR         │ Planar scan, affordable, robot      │     │
│  │ (e.g., RPLiDAR)  │ vacuum cleaners                     │     │
│  ├──────────────────┼─────────────────────────────────────┤     │
│  │ 3D LiDAR         │ 3D point clouds, autonomous         │     │
│  │ (e.g., Velodyne) │ driving                             │     │
│  ├──────────────────┼─────────────────────────────────────┤     │
│  │ Solid-State      │ Non-rotating, compact, latest       │     │
│  │ (e.g., Livox)    │ trend                               │     │
│  └──────────────────┴─────────────────────────────────────┘     │
│                                                                 │
│  Key Algorithms:                                                │
│  - ICP (Iterative Closest Point)                                │
│  - NDT (Normal Distributions Transform)                         │
│  - LOAM (LiDAR Odometry and Mapping)                            │
│  - LeGO-LOAM (Lightweight Ground-Optimized)                     │
│  - Cartographer (Google)                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ICP (Iterative Closest Point)

ICP는 LiDAR SLAM의 핵심 하위 문제를 해결합니다. 서로 다른 시간에 캡처된 두 개의 겹치는 포인트 클라우드(Point Cloud)가 주어졌을 때, 두 클라우드를 가장 잘 정렬하는 강체 변환(Rigid-Body Transform, 회전 + 평행 이동)을 찾는 것입니다. 이 정렬을 통해 인공 랜드마크나 GPS 없이도 스캔 사이에 로봇이 얼마나 이동했는지를 알 수 있습니다.

```python
import numpy as np
from scipy.spatial import KDTree

def icp(source, target, max_iterations=50, tolerance=1e-6):
    """
    ICP algorithm for aligning two point clouds

    Parameters:
        source: Source point cloud (N x 3)
        target: Target point cloud (M x 3)

    Returns:
        R: Rotation matrix (3 x 3)
        t: Translation vector (3,)
        transformed: Transformed source points
    """

    src = source.copy()
    prev_error = float('inf')

    R_total = np.eye(3)
    t_total = np.zeros(3)

    # KD-Tree indexes the *target* cloud for O(log M) nearest-neighbor queries
    # instead of O(N·M) brute force — critical when clouds have 10k+ points
    tree = KDTree(target)

    for i in range(max_iterations):
        # 1. Find nearest correspondences (the "Closest Point" in ICP's name)
        #    Assumes the current alignment is good enough that the nearest
        #    target point is the correct correspondent — improves each iteration
        distances, indices = tree.query(src)
        correspondences = target[indices]

        # 2. Estimate transformation (SVD)
        #    The Orthogonal Procrustes Problem: given matched point pairs,
        #    find the rotation R that minimizes sum of squared distances.
        #    SVD of the cross-covariance matrix H gives the optimal solution.
        src_centroid = np.mean(src, axis=0)
        tgt_centroid = np.mean(correspondences, axis=0)

        src_centered = src - src_centroid
        tgt_centered = correspondences - tgt_centroid

        H = src_centered.T @ tgt_centered  # 3×3 cross-covariance matrix
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Correct reflection: det(R) = -1 means SVD returned an improper rotation
        # (reflection). Flipping the last row of Vt forces a proper rotation (det=+1).
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - R @ src_centroid

        # 3. Apply transformation
        src = (R @ src.T).T + t

        # Accumulate transformation
        R_total = R @ R_total
        t_total = R @ t_total + t

        # 4. Check convergence
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            print(f"ICP converged: {i+1} iterations, error: {mean_error:.6f}")
            break
        prev_error = mean_error

    return R_total, t_total, src

class LiDARSLAM:
    """Simple 2D LiDAR SLAM"""

    def __init__(self, map_resolution=0.05):
        self.resolution = map_resolution
        self.pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.trajectory = [self.pose.copy()]

        # Occupancy grid map — initialize all cells to 0.5 (maximum uncertainty)
        # so that unvisited areas are neither claimed free nor occupied.
        # 0.5 is the prior probability; updates push cells toward 0 (free) or 1 (occupied).
        self.map_size = 1000
        self.occupancy_map = np.ones((self.map_size, self.map_size)) * 0.5
        self.map_origin = np.array([self.map_size // 2, self.map_size // 2])
        # Center origin so the robot can map equally in all directions

    def scan_to_points(self, scan_ranges, scan_angles):
        """Convert scan data to 2D points"""

        valid = (scan_ranges > 0.1) & (scan_ranges < 30.0)
        # 0.1m minimum: filters out the robot's own body returns
        # 30.0m maximum: beyond this, 2D LiDAR returns become unreliable
        ranges = scan_ranges[valid]
        angles = scan_angles[valid]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        return np.column_stack([x, y])

    def transform_points(self, points, pose):
        """Transform points to world coordinates"""

        x, y, theta = pose
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        transformed = (R @ points.T).T + np.array([x, y])
        return transformed

    def point_to_grid(self, points):
        """Convert points to grid coordinates"""

        grid_x = (points[:, 0] / self.resolution + self.map_origin[0]).astype(int)
        grid_y = (points[:, 1] / self.resolution + self.map_origin[1]).astype(int)

        # Limit to map bounds
        valid = (grid_x >= 0) & (grid_x < self.map_size) & \
                (grid_y >= 0) & (grid_y < self.map_size)

        return grid_x[valid], grid_y[valid], valid

    def update_map(self, scan_points, pose):
        """Update occupancy grid map"""

        world_points = self.transform_points(scan_points, pose)
        gx, gy, valid = self.point_to_grid(world_points)

        # Update occupancy probability (log odds)
        # +0.1 per hit: laser endpoint = occupied. Log-odds representation
        # would be more principled, but this additive approximation is fast
        # and sufficient for simple mapping (see Problem 4 for full log-odds).
        self.occupancy_map[gy, gx] = np.clip(
            self.occupancy_map[gy, gx] + 0.1, 0, 1
        )

    def match_scan(self, current_points, previous_points):
        """Estimate relative motion using scan matching"""

        if len(previous_points) < 10 or len(current_points) < 10:
            return np.array([0, 0, 0])

        # Apply ICP
        R, t, _ = icp(current_points, previous_points)

        # Extract theta in 2D
        theta = np.arctan2(R[1, 0], R[0, 0])

        return np.array([t[0], t[1], theta])

    def process_scan(self, scan_ranges, scan_angles, prev_scan=None):
        """Process scan"""

        current_points = self.scan_to_points(scan_ranges, scan_angles)

        if prev_scan is not None:
            prev_points = self.scan_to_points(prev_scan[0], prev_scan[1])

            # Scan matching
            delta_pose = self.match_scan(current_points, prev_points)

            # Update pose
            self.pose[2] += delta_pose[2]
            R = np.array([
                [np.cos(self.pose[2]), -np.sin(self.pose[2])],
                [np.sin(self.pose[2]), np.cos(self.pose[2])]
            ])
            self.pose[:2] += R @ delta_pose[:2]

        # Update map
        self.update_map(current_points, self.pose)

        # Save trajectory
        self.trajectory.append(self.pose.copy())

        return self.pose

    def get_occupancy_map(self):
        """Return occupancy map"""
        return self.occupancy_map

    def get_trajectory(self):
        """Return trajectory"""
        return np.array(self.trajectory)
```

---

## 5. Loop Closure

### Loop Closure 개념

```
Loop Closure:
Recognizing previously visited places to correct accumulated drift

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Problem: Drift (accumulated error)                             │
│                                                                 │
│       Actual Path      Estimated Path (with drift)              │
│       ┌─────────┐      ┌─────────┐                              │
│       │         │      │         ╲                              │
│       │         │      │          ╲                             │
│       │         │      │           ╲                            │
│       └─────────┘      └────────────╲                           │
│       (closed loop)     (open curve)                            │
│                                                                 │
│  Solution: Loop Closure                                         │
│       1. Detect if current location was visited before          │
│       2. Add loop constraint                                    │
│       3. Pose graph optimization                                │
│                                                                 │
│       ┌─────────┐                                               │
│       │    ●────●  ← Loop detection                             │
│       │    │    │                                               │
│       │    │    │  ← Graph optimization                         │
│       │    ●────●                                               │
│       └─────────┘                                               │
│       (corrected path)                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Loop Closure 구현

```python
import cv2
import numpy as np
from collections import deque

class LoopClosureDetector:
    """Bag of Words-based loop closure detection"""

    def __init__(self, vocabulary_size=1000, min_score=0.3):
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.vocabulary = None
        self.vocabulary_size = vocabulary_size
        self.min_score = min_score  # 0.3: empirically chosen floor — below this,
                                    # BoW similarity is likely coincidental overlap

        # Keyframe database
        self.keyframe_bows = []
        self.keyframe_descs = []
        self.keyframe_kps = []

        # Exclude recent N keyframes from loop candidates — nearby frames are
        # always visually similar (temporal correlation), so excluding them
        # prevents false positives from consecutive keyframes being flagged as loops
        self.temporal_window = 30

    def build_vocabulary(self, training_images):
        """Build vocabulary"""

        all_descriptors = []

        for img in training_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, desc = self.orb.detectAndCompute(gray, None)
            if desc is not None:
                all_descriptors.append(desc)

        all_desc = np.vstack(all_descriptors).astype(np.float32)

        # K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                   100, 0.2)
        _, _, self.vocabulary = cv2.kmeans(
            all_desc, self.vocabulary_size, None,
            criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        self.vocabulary = self.vocabulary.astype(np.uint8)

    def compute_bow(self, descriptors):
        """Compute BoW vector"""

        if self.vocabulary is None or descriptors is None:
            return None

        matches = self.bf.match(descriptors, self.vocabulary)

        bow = np.zeros(self.vocabulary_size)
        for m in matches:
            bow[m.trainIdx] += 1

        # L2 normalization
        norm = np.linalg.norm(bow)
        if norm > 0:
            bow = bow / norm

        return bow

    def add_keyframe(self, frame):
        """Add keyframe"""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)

        if desc is None:
            return -1

        bow = self.compute_bow(desc)

        self.keyframe_bows.append(bow)
        self.keyframe_descs.append(desc)
        self.keyframe_kps.append(kp)

        return len(self.keyframe_bows) - 1

    def detect_loop(self, query_idx):
        """Detect loop candidates"""

        if query_idx < self.temporal_window + 1:
            return None, 0

        query_bow = self.keyframe_bows[query_idx]

        best_match = -1
        best_score = 0

        # Search only temporally distant keyframes
        for i in range(query_idx - self.temporal_window):
            score = np.dot(query_bow, self.keyframe_bows[i])

            if score > best_score and score > self.min_score:
                best_score = score
                best_match = i

        if best_match >= 0:
            return best_match, best_score

        return None, 0

    def verify_loop(self, query_idx, candidate_idx, min_inliers=50):
        """Verify loop using geometric verification"""

        desc1 = self.keyframe_descs[query_idx]
        desc2 = self.keyframe_descs[candidate_idx]
        kp1 = self.keyframe_kps[query_idx]
        kp2 = self.keyframe_kps[candidate_idx]

        # Feature matching
        matches = self.bf.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for m, n in matches:
            # 0.75 ratio test (slightly looser than typical 0.7 used in VO)
            # because loop frames can be viewed from significantly different
            # angles, making descriptors somewhat less similar even for true matches
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            return False, None

        # Geometric verification using Fundamental Matrix
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # We use the Fundamental Matrix (not Essential) here because we don't
        # need calibrated coordinates — just a geometric consistency check.
        # Inlier count under RANSAC is our confidence signal: many random
        # descriptor matches will fail the epipolar constraint, so a high
        # inlier count strongly suggests these two frames share real 3D structure.
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        if mask is None:
            return False, None

        num_inliers = np.sum(mask)

        if num_inliers >= min_inliers:
            return True, {
                'query_idx': query_idx,
                'match_idx': candidate_idx,
                'inliers': num_inliers,
                'pts1': pts1[mask.ravel() == 1],
                'pts2': pts2[mask.ravel() == 1]
            }

        return False, None


class PoseGraphOptimizer:
    """Simple pose graph optimization"""

    def __init__(self):
        self.poses = []         # Nodes (poses)
        self.edges = []         # Edges (relative transforms)
        self.loop_constraints = []  # Loop constraints

    def add_pose(self, pose):
        """Add pose node"""
        self.poses.append(pose.copy())
        return len(self.poses) - 1

    def add_odometry_edge(self, i, j, relative_pose, info_matrix=None):
        """Add odometry edge"""

        if info_matrix is None:
            info_matrix = np.eye(3)

        self.edges.append({
            'from': i,
            'to': j,
            'measurement': relative_pose,
            'info': info_matrix
        })

    def add_loop_constraint(self, i, j, relative_pose, info_matrix=None):
        """Add loop constraint"""

        if info_matrix is None:
            # Loop constraints get 100× higher weight than odometry edges because
            # they are geometrically verified (RANSAC inliers confirm real overlap),
            # while odometry accumulates noise continuously. The optimizer should
            # strongly prefer satisfying a verified loop over individual odometry steps.
            info_matrix = np.eye(3) * 100

        self.loop_constraints.append({
            'from': i,
            'to': j,
            'measurement': relative_pose,
            'info': info_matrix
        })

    def optimize(self, num_iterations=10):
        """Graph optimization (Gauss-Newton)"""

        # Simple implementation (in practice, use g2o, Ceres, etc.)
        print("Pose graph optimization recommended to use specialized libraries like g2o")

        # Simple correction using loop constraints
        for constraint in self.loop_constraints:
            i = constraint['from']
            j = constraint['to']

            # Calculate accumulated drift
            drift = self.poses[j][:2] - self.poses[i][:2]
            drift -= constraint['measurement'][:2]

            # Distribute drift using linear interpolation
            for k in range(i, j + 1):
                alpha = (k - i) / (j - i) if j > i else 0
                self.poses[k][:2] -= alpha * drift

        return self.poses
```

---

## 6. SLAM 구현 실습

### 간단한 SLAM 시스템

```python
import cv2
import numpy as np

class SimpleVSLAM:
    """Simple Visual SLAM system"""

    def __init__(self, K):
        self.K = K

        # Modules
        self.vo = MonocularVO(K)
        self.loop_detector = LoopClosureDetector()
        self.pose_graph = PoseGraphOptimizer()

        # State
        self.frame_count = 0
        # Sample one keyframe every 10 frames: frequent enough to capture
        # scene change, sparse enough that consecutive keyframes have meaningful
        # baseline for loop detection (adjacent frames are too similar to be useful)
        self.keyframe_interval = 10

    def process_frame(self, frame):
        """Process frame"""

        self.frame_count += 1

        # Visual Odometry
        R, t = self.vo.process_frame(frame)

        # Add keyframe
        if self.frame_count % self.keyframe_interval == 0:
            kf_idx = self.loop_detector.add_keyframe(frame)

            # Add node to pose graph
            pose = np.array([t[0, 0], t[1, 0], 0])  # 2D approximation
            node_idx = self.pose_graph.add_pose(pose)

            # Connect edge with previous keyframe
            if node_idx > 0:
                prev_pose = self.pose_graph.poses[node_idx - 1]
                relative = pose - prev_pose
                self.pose_graph.add_odometry_edge(
                    node_idx - 1, node_idx, relative
                )

            # Loop detection
            if kf_idx > 30:  # Need at least 30 keyframes before loop search:
                             # fewer than this and the BoW vocabulary hasn't seen
                             # enough variety to distinguish places reliably
                candidate, score = self.loop_detector.detect_loop(kf_idx)

                if candidate is not None:
                    verified, loop_info = self.loop_detector.verify_loop(
                        kf_idx, candidate
                    )

                    if verified:
                        print(f"Loop detected: {kf_idx} -> {candidate}")

                        # Add loop constraint
                        relative = pose - self.pose_graph.poses[candidate]
                        self.pose_graph.add_loop_constraint(
                            candidate, node_idx, relative
                        )

                        # Optimize
                        self.pose_graph.optimize()

        return R, t

    def get_map(self):
        """Return map"""
        return self.vo.get_trajectory()

    def get_optimized_trajectory(self):
        """Return optimized trajectory"""
        return np.array(self.pose_graph.poses)
```

### 시각화

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_slam_result(trajectory, loop_closures=None):
    """Visualize SLAM results"""

    fig = plt.figure(figsize=(12, 5))

    # 2D trajectory
    ax1 = fig.add_subplot(121)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1],
               c='green', s=100, marker='o', label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1],
               c='red', s=100, marker='x', label='End')

    if loop_closures:
        for lc in loop_closures:
            i, j = lc['from'], lc['to']
            ax1.plot([trajectory[i, 0], trajectory[j, 0]],
                    [trajectory[i, 1], trajectory[j, 1]],
                    'g--', linewidth=2, alpha=0.5)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('2D Trajectory')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True)

    # 3D trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            'b-', linewidth=1)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Trajectory')

    plt.tight_layout()
    plt.show()

def visualize_occupancy_map(occupancy_map, trajectory=None):
    """Visualize occupancy map"""

    plt.figure(figsize=(10, 10))

    # Display map
    plt.imshow(occupancy_map, cmap='gray', origin='lower')

    # Overlay trajectory
    if trajectory is not None:
        # Convert to map coordinates
        map_center = occupancy_map.shape[0] // 2
        resolution = 0.05
        traj_map = trajectory / resolution + map_center

        plt.plot(traj_map[:, 0], traj_map[:, 1], 'r-', linewidth=2)
        plt.scatter(traj_map[0, 0], traj_map[0, 1], c='green', s=100)
        plt.scatter(traj_map[-1, 0], traj_map[-1, 1], c='blue', s=100)

    plt.title('Occupancy Grid Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Occupancy Probability')
    plt.show()
```

---

## 7. 연습 문제

### 문제 1: Visual Odometry 구현

단안 Visual Odometry를 구현하세요.

**요구사항**:
- ORB 특징 검출
- 광학 흐름 또는 디스크립터 매칭
- Essential Matrix로 포즈 추정
- 궤적 시각화

<details>
<summary>힌트</summary>

```python
# Essential Matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# Accumulate pose
cur_t = cur_t + cur_R @ t
cur_R = R @ cur_R
```

</details>

### 문제 2: 루프 클로저 탐지

BoW 기반 루프 클로저를 구현하세요.

**요구사항**:
- ORB vocabulary 구축
- BoW 벡터 계산
- 유사도 기반 후보 탐지
- 기하학적 검증

<details>
<summary>힌트</summary>

```python
# BoW similarity
score = np.dot(bow1, bow2)

# Geometric verification
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
inliers = np.sum(mask)
```

</details>

### 문제 3: ICP 구현

ICP 알고리즘을 구현하세요.

**요구사항**:
- 최근접 대응점 검색
- SVD로 변환 추정
- 반복 최적화
- 수렴 조건

<details>
<summary>힌트</summary>

```python
# Calculate R, t using SVD
H = src_centered.T @ tgt_centered
U, _, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
t = tgt_centroid - R @ src_centroid
```

</details>

### 문제 4: 점유 격자 맵

LiDAR 데이터로 점유 격자 맵을 생성하세요.

**요구사항**:
- 스캔 데이터를 포인트로 변환
- 격자 좌표 변환
- 점유 확률 업데이트
- 맵 시각화

<details>
<summary>힌트</summary>

```python
# Log odds update
log_odds = np.log(p / (1 - p))
log_odds[occupied] += 0.5
log_odds[free] -= 0.2
p = 1 / (1 + np.exp(-log_odds))
```

</details>

### 문제 5: 완전한 SLAM 시스템

VO, 루프 클로저, 맵핑을 통합한 SLAM을 구현하세요.

**요구사항**:
- 키프레임 관리
- 루프 탐지 및 검증
- 포즈 그래프 최적화
- 3D 맵 생성

<details>
<summary>힌트</summary>

```python
# Integrated system
class SLAM:
    def process(self, frame):
        # 1. Tracking
        pose = self.track(frame)

        # 2. Update map if keyframe
        if self.is_keyframe():
            self.local_mapping()

            # 3. Loop detection
            if self.detect_loop():
                self.optimize_graph()
```

</details>

---

## 다음 단계

- 실제 SLAM 라이브러리 사용 (ORB-SLAM3, RTAB-Map)
- ROS 연동
- Visual-Inertial SLAM
- 딥러닝 기반 SLAM

---

## 참고 자료

- [ORB-SLAM3 GitHub](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [SLAM Tutorial - Cyrill Stachniss](https://www.youtube.com/playlist?list=PLgnQpQtFTOGQrZ4O5QzbIHgl3b1JHimN_)
- [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- [Probabilistic Robotics (Thrun et al.)](http://www.probabilistic-robotics.org/)
- [LOAM Paper](https://www.ri.cmu.edu/pub_files/2014/7/Ji_LidarMapping_RSS2014_v8.pdf)
- [Cartographer](https://google-cartographer.readthedocs.io/)
