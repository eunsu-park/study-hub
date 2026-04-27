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

### Theory: The Brightness Constancy Constraint

The fundamental assumption behind optical flow is that **a physical point's brightness does not change between consecutive frames**. If at time `t` a scene point projects to pixel `(x, y)` with intensity `I(x, y, t)`, and at time `t + dt` it has moved to `(x + dx, y + dy)`, then:

```
I(x + dx, y + dy, t + dt) = I(x, y, t)
```

A first-order Taylor expansion of the left side around `(x, y, t)`:

```
I(x, y, t) + (∂I/∂x)·dx + (∂I/∂y)·dy + (∂I/∂t)·dt  =  I(x, y, t)
```

Canceling `I(x, y, t)` and dividing by `dt`, with velocity `u = dx/dt`, `v = dy/dt`:

```
(∂I/∂x)·u + (∂I/∂y)·v + (∂I/∂t) = 0       ← brightness constancy constraint equation
```

Usually written as `I_x · u + I_y · v + I_t = 0` or `∇I · (u, v) + I_t = 0`. This is a **single linear equation in two unknowns** `u` and `v`. One equation per pixel, two unknowns per pixel, so without further information the system is underdetermined.

The brightness constancy assumption is violated in practice by shadows, specular highlights, and changes in ambient lighting. More modern formulations use gradient constancy or learned feature constancy to mitigate, but the basic framework still applies.

### Theory: The Aperture Problem

Geometrically, the single constraint equation says: **the component of motion along the image gradient** is determined, but the component perpendicular to the gradient is not. From the equation:

```
component of (u, v) along ∇I  =  -I_t / |∇I|       (determined)
component of (u, v) perpendicular to ∇I           (undetermined)
```

This is the **aperture problem**: looking through a small aperture at a moving edge, you can see only how the edge moves *perpendicular to itself*, not parallel. A horizontal edge sliding horizontally looks identical to one that isn't moving at all.

Any useful optical flow algorithm must add extra information to resolve the perpendicular component. The two classical solutions are two different choices of what to add.

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

### Theory: Lucas-Kanade: Local Constancy

**Assumption**: pixels in a small window all move the same way. Under this assumption, every pixel in the window contributes one brightness-constancy equation with the *same* `(u, v)`:

```
For pixel i in window W:
    I_x(i) · u + I_y(i) · v = -I_t(i)
```

Stacking all `N` equations from an `N`-pixel window:

```
⎡ I_x(1)  I_y(1) ⎤ ⎡ u ⎤     ⎡ -I_t(1) ⎤
⎢ I_x(2)  I_y(2) ⎥ ⎢   ⎥  =  ⎢ -I_t(2) ⎥
⎢   ...    ...   ⎥ ⎣ v ⎦     ⎢   ...   ⎥
⎣ I_x(N)  I_y(N) ⎦           ⎣ -I_t(N) ⎦

       A            (u,v)         b
```

This is an overdetermined linear system. The least-squares solution is `(u, v) = (AᵀA)⁻¹ · Aᵀ · b`. Expanding:

```
⎡ u ⎤     ⎡ Σ I_x²    Σ I_x·I_y ⎤⁻¹  ⎡ -Σ I_x·I_t ⎤
⎣ v ⎦  =  ⎣ Σ I_x·I_y  Σ I_y²   ⎦    ⎣ -Σ I_y·I_t ⎦
```

The matrix on the left is **exactly the structure tensor** from §13.B. Lucas-Kanade's local system is well-conditioned (invertible) exactly when the structure tensor has two large eigenvalues — i.e. at **corners**. At edges one eigenvalue is small and the solution becomes unstable along the edge direction (the aperture problem persists locally). In flat regions both eigenvalues are small and no flow can be recovered.

This explains why Lucas-Kanade is typically applied only to sparse **keypoints** (corners detected by Harris/Shi-Tomasi): it is numerically well-conditioned there, and it would fail elsewhere anyway.

### Theory: Coarse-to-Fine Pyramids: Handling Large Motion

The Taylor expansion in §A assumes `(dx, dy)` is small — on the order of one pixel. For motions larger than that, the first-order approximation is invalid, and both Lucas-Kanade and Horn-Schunck fail.

**Fix**: build a Gaussian pyramid of both frames, solve optical flow at the coarsest (most-reduced) level where motion is small, then **propagate** the result to the next finer level (scaling the flow by 2, warping the second frame by this estimate so residual motion becomes small again), and refine. Repeat down to full resolution.

This is why OpenCV's Lucas-Kanade variant is `calcOpticalFlowPyrLK` — the `Pyr` is the pyramid trick — and it is essential for all but the smallest motions.

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

### Theory: Horn-Schunck: Global Smoothness

**Assumption**: the flow field is globally smooth. Instead of assuming constant flow in each window, Horn-Schunck adds a **smoothness regularization** term to the objective function:

```
E(u, v) = ∫∫ [ (I_x u + I_y v + I_t)² + α²·( |∇u|² + |∇v|² ) ] dx dy
          └───────── data term ─────────┘  └── smoothness ──┘
```

The first term penalizes violations of brightness constancy (from §A); the second penalizes non-smooth flow fields. The weight `α` is a hyperparameter: large `α` forces very smooth flow (good for gentle motions); small `α` allows sharper changes (better at motion boundaries but more noise).

Minimizing `E` gives a coupled system of PDEs; the standard solver is Gauss-Seidel iteration over the discretized image, each iteration updating `(u, v)` at every pixel based on its neighbors. The result is a **dense flow field** — every pixel gets a flow estimate, unlike Lucas-Kanade's sparse corners.

Horn-Schunck produces dense output but over-smooths across motion discontinuities (two adjacent pixels moving differently get averaged toward the middle). Modern variational methods use robust non-quadratic penalties and total-variation regularization to handle discontinuities better.

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

### Theory: What Modern Learned Methods Change

Deep learning optical flow (FlowNet, PWC-Net, RAFT) replaces the hand-crafted data and smoothness terms with learned representations, but the underlying structure of the problem is the same:

- **Data term**: learned feature similarity between warped patches (robust to brightness changes, shadows, etc.) instead of raw intensity difference.
- **Smoothness prior**: implicit in the convolutional architecture (spatial smoothness is a strong inductive bias of CNNs) and in the refinement modules.
- **Pyramid**: explicit in PWC-Net (pyramid + warping + cost volume), implicit in RAFT (GRU-based iterative refinement).
- **Aperture problem**: resolved the same way as classical methods — by aggregating information over a neighborhood, just now learned instead of fixed.

The benchmark gap between RAFT and classical variational methods is large, but on domains without training data (scientific imagery, novel sensors) the classical Lucas-Kanade or Horn-Schunck with modern regularizers is still competitive.

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
