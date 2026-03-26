[Previous: Video Understanding](./30_Video_Understanding.md) | [Next: Synthetic Data Generation](./32_Synthetic_Data_Generation.md)

---

# 31. Optical Flow

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain optical flow as pixel-level motion estimation between frames
2. Implement the Lucas-Kanade method for sparse optical flow
3. Describe deep learning approaches: FlowNet, PWC-Net, and RAFT
4. Visualize optical flow fields using standard color coding
5. Apply optical flow for motion estimation, video stabilization, and action recognition

---

## Table of Contents

1. [Optical Flow Fundamentals](#1-optical-flow-fundamentals)
2. [Lucas-Kanade Method](#2-lucas-kanade-method)
3. [Horn-Schunck Dense Flow](#3-horn-schunck-dense-flow)
4. [Deep Optical Flow (FlowNet)](#4-deep-optical-flow-flownet)
5. [RAFT: State of the Art](#5-raft-state-of-the-art)
6. [Flow Visualization and Evaluation](#6-flow-visualization-and-evaluation)
7. [Applications](#7-applications)
8. [Exercises](#8-exercises)

---

## 1. Optical Flow Fundamentals

### 1.1 What Is Optical Flow?

```
Optical flow: A dense vector field describing the apparent motion
of each pixel between two consecutive frames.

For each pixel (x, y) in frame I₁:
  Flow (u, v) = displacement to corresponding pixel in frame I₂

  Pixel at (x, y) in frame 1 → pixel at (x+u, y+v) in frame 2

Brightness Constancy Assumption:
  I(x, y, t) = I(x + u, y + v, t + 1)
  "A pixel's brightness doesn't change between frames"

  Taylor expansion → Optical Flow Equation:
  Ix·u + Iy·v + It = 0

  Where Ix, Iy = spatial gradients, It = temporal gradient
  One equation, two unknowns (u, v) → need additional constraints!
```

---

## 2. Lucas-Kanade Method

### 2.1 LK Sparse Flow

```python
import cv2
import numpy as np


def lucas_kanade_demo(video_path):
    """Lucas-Kanade sparse optical flow tracking."""
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Detect good features to track
    feature_params = dict(maxCorners=100, qualityLevel=0.3,
                         minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)

    # LK parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))

    colors = np.random.randint(0, 255, (100, 3))
    mask = np.zeros_like(old_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        # Select good points
        if p1 is not None:
            good_new = p1[status == 1]
            good_old = p0[status == 1]

        # Draw tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)

        output = cv2.add(frame, mask)
        cv2.imshow('Optical Flow - Lucas-Kanade', output)

        if cv2.waitKey(30) & 0xFF == 27:
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()
```

---

## 3. Horn-Schunck Dense Flow

### 3.1 Dense Flow with Farneback

```python
def farneback_dense_flow(video_path):
    """Dense optical flow using Farneback method."""
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Visualize using HSV color coding
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(old_frame)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
        hsv[..., 1] = 255                        # Saturation = max
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255,
                                     cv2.NORM_MINMAX)  # Value = magnitude

        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('Dense Optical Flow', flow_rgb)

        if cv2.waitKey(30) & 0xFF == 27:
            break

        old_gray = gray

    cap.release()
    cv2.destroyAllWindows()
```

---

## 4. Deep Optical Flow (FlowNet)

### 4.1 FlowNet Architecture

```
FlowNet (2015): First CNN-based optical flow estimation.

FlowNetS (Simple):
  Concatenate two frames → Encoder-Decoder → Flow prediction
  Input: [I₁, I₂] (6 channels) → Output: (u, v) per pixel

FlowNetC (Correlation):
  Separate encoders for each frame → Correlation layer → Decoder
  Correlation captures matching between frames

Architecture:
  I₁ ─┐
      ├─ Concat ─→ Encoder ─→ Decoder ─→ Flow (u, v)
  I₂ ─┘

Training: Supervised on synthetic data (Flying Chairs, FlyingThings3D)
  Loss: EPE (End-Point Error) = ||flow_pred - flow_gt||₂
```

### 4.2 Simplified FlowNet

```python
import torch
import torch.nn as nn


class FlowNetS(nn.Module):
    """Simplified FlowNet-Simple architecture."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
        )

        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1),  # 2 channels: (u, v)
        )

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=1)  # (B, 6, H, W)
        features = self.encoder(x)
        flow = self.decoder(features)
        return flow  # (B, 2, H, W)
```

---

## 5. RAFT: State of the Art

### 5.1 RAFT Architecture

```
RAFT (Recurrent All-Pairs Field Transforms, 2020):
  State-of-the-art optical flow estimation.

Key components:
  1. Feature encoder: Extract features from both frames
  2. Correlation volume: All-pairs correlation between features
  3. GRU-based iterative refinement: Repeatedly update flow estimate

  I₁ → Encoder → Features₁ ──┐
                                ├── Correlation Volume
  I₂ → Encoder → Features₂ ──┘       │
                                       ▼
  Initial flow (zero) → GRU → Update → GRU → Update → ... → Final flow
                          ↑              ↑
                     Correlation     Correlation
                     lookup          lookup

  Each GRU iteration refines the flow estimate.
  Typically 12-32 iterations during training, any number at test time.
```

### 5.2 Using RAFT

```python
# Using pretrained RAFT from torchvision
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def compute_raft_flow(img1, img2):
    """Compute optical flow using pretrained RAFT."""
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights)
    model.eval()

    # Preprocess
    transforms = weights.transforms()
    img1_t, img2_t = transforms(img1, img2)

    with torch.no_grad():
        flow_predictions = model(img1_t.unsqueeze(0), img2_t.unsqueeze(0))

    # Last prediction is the final flow
    flow = flow_predictions[-1][0]  # (2, H, W)
    return flow
```

---

## 6. Flow Visualization and Evaluation

### 6.1 Flow Color Coding

```python
def flow_to_color(flow, max_flow=None):
    """Convert optical flow to color image using Middlebury color coding."""
    u = flow[..., 0]
    v = flow[..., 1]

    if max_flow is None:
        max_flow = max(np.abs(u).max(), np.abs(v).max())

    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(-v, -u) / np.pi  # [-1, 1]

    # Map to HSV
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + 1) / 2 * 179).astype(np.uint8)  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = np.minimum(magnitude / max_flow * 255, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def end_point_error(pred_flow, gt_flow):
    """End-Point Error: L2 distance between predicted and GT flow."""
    epe = np.sqrt(((pred_flow - gt_flow) ** 2).sum(axis=-1))
    return epe.mean()
```

---

## 7. Applications

### 7.1 Video Stabilization

```python
def stabilize_video(frames, flows):
    """Simple video stabilization using optical flow."""
    H, W = frames[0].shape[:2]
    cumulative_dx = 0
    cumulative_dy = 0
    stabilized = [frames[0]]

    for i in range(1, len(frames)):
        # Average flow gives global motion
        dx = flows[i-1][..., 0].mean()
        dy = flows[i-1][..., 1].mean()

        cumulative_dx += dx
        cumulative_dy += dy

        # Create inverse transformation
        M = np.float32([[1, 0, -cumulative_dx],
                        [0, 1, -cumulative_dy]])

        stabilized_frame = cv2.warpAffine(frames[i], M, (W, H))
        stabilized.append(stabilized_frame)

    return stabilized
```

---

## 8. Exercises

### Exercise 1: Lucas-Kanade Feature Tracking

Implement and evaluate LK tracking:
1. Track 50 features across a video using LK optical flow
2. Implement forward-backward consistency check
3. Measure tracking accuracy on annotated dataset
4. Visualize trajectories of tracked points
5. Handle feature loss and re-detection

### Exercise 2: Dense Flow Visualization

Build a dense optical flow visualizer:
1. Compute Farneback flow on a video
2. Implement Middlebury color coding
3. Create flow magnitude heatmap
4. Overlay flow arrows on the original image
5. Compare: Farneback, LK interpolated, and RAFT

### Exercise 3: FlowNet Training

Train FlowNet from scratch:
1. Generate synthetic training data (Flying Chairs style)
2. Implement FlowNetS architecture
3. Train with EPE loss and multi-scale supervision
4. Evaluate on Sintel or KITTI benchmarks
5. Compare with traditional methods on same data

### Exercise 4: Motion Segmentation

Use optical flow for motion segmentation:
1. Compute dense flow between consecutive frames
2. Cluster flow vectors to separate moving objects
3. Generate binary masks for moving vs static regions
4. Track moving objects across video
5. Measure: segmentation quality vs simple background subtraction

### Exercise 5: Video Stabilization System

Build a complete video stabilization pipeline:
1. Compute global motion using optical flow averaging
2. Implement path smoothing (moving average on cumulative motion)
3. Apply inverse transformation for stabilization
4. Handle: zoom-in to avoid black borders
5. Compare quality with OpenCV's video stabilization module

---

*End of Lesson 31*
