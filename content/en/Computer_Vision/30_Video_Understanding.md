[Previous: 3D Gaussian Splatting](./29_3D_Gaussian_Splatting.md) | [Next: Optical Flow](./31_Optical_Flow.md)

---

# 30. Video Understanding

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain action recognition and temporal modeling in video
2. Implement two-stream networks and 3D convolutions for video features
3. Build video transformers with spatiotemporal attention (TimeSformer, ViViT)
4. Describe the SlowFast architecture for multi-rate temporal modeling
5. Apply video understanding to action detection and temporal localization

---

## Table of Contents

1. [Video Understanding Overview](#1-video-understanding-overview)
2. [Temporal Modeling Approaches](#2-temporal-modeling-approaches)
3. [3D Convolutions (C3D, I3D)](#3-3d-convolutions-c3d-i3d)
4. [SlowFast Networks](#4-slowfast-networks)
5. [Video Transformers](#5-video-transformers)
6. [Temporal Action Detection](#6-temporal-action-detection)
7. [Practical Video Pipeline](#7-practical-video-pipeline)
8. [Exercises](#8-exercises)

---

## 1. Video Understanding Overview

### Theory: Why Per-Frame Classification Fails

The naïve approach to video classification: classify each frame independently with an image model, then average the predictions. This loses critical information:

- **Temporal order**: a hand moving up vs down looks identical when you average.
- **Speed**: same poses but executed at different speeds (fast vs slow) are different actions.
- **Inter-frame coherence**: motion smoothness vs jerkiness contains information.

A video model must aggregate information **across time** in a way that preserves these temporal cues.

### 1.1 Tasks in Video Understanding

```
Video Classification:
  Input: Video clip → Output: Action label
  "This video shows swimming"

Temporal Action Detection:
  Input: Untrimmed video → Output: (action, start_time, end_time)
  "Swimming from 5.2s to 12.8s, diving from 13.0s to 15.5s"

Action Recognition:
  Input: Short clip (3-10s) → Output: Action class
  Online setting for real-time applications

Video Captioning:
  Input: Video → Output: Natural language description
  "A person dives into a pool and swims across"

Video Question Answering:
  Input: Video + Question → Output: Answer
  "How many people are swimming?" → "Three"
```

---

## 2. Temporal Modeling Approaches

### Theory: Two-Stream Networks: Appearance + Motion

Two-stream networks (Simonyan & Zisserman, 2014) take an early approach: explicitly compute optical flow (§31) between adjacent frames, treat the flow field as a separate input, and run two parallel networks:

- **Spatial stream**: classifies a single (random) RGB frame from the clip — captures *what* objects/scene appears.
- **Temporal stream**: takes a stack of optical flow fields (typically 10 frames worth) — captures *how* things move.

Final prediction: average the two streams' softmax outputs (or learn a fusion). The flow field encodes motion explicitly, which the spatial stream can't capture from a single frame.

This was the dominant action recognition approach for several years. Its key weakness: optical flow must be computed in advance, which is slow and adds engineering complexity.

### 2.1 Approach Taxonomy

```
How to handle the temporal dimension in video?

1. Frame-level (2D CNN + aggregation):
   Process each frame independently, then aggregate
   Simple but loses motion information

2. Two-Stream (RGB + Optical Flow):
   Spatial stream: appearance from RGB frames
   Temporal stream: motion from optical flow
   Fuse predictions from both streams

3. 3D Convolutions:
   Extend 2D conv to 3D: convolve in space AND time
   Directly captures spatiotemporal patterns

4. Video Transformers:
   Self-attention across space and time
   Global receptive field from the start

5. Recurrent (LSTM/GRU):
   Process frames sequentially with memory
   Good for variable-length videos
```

### 2.2 Two-Stream Architecture

```python
import torch
import torch.nn as nn
import torchvision.models as models


class TwoStreamNetwork(nn.Module):
    """Two-stream architecture for action recognition."""

    def __init__(self, n_classes, n_flow_frames=10):
        super().__init__()
        # Spatial stream: single RGB frame
        self.spatial = models.resnet50(pretrained=True)
        self.spatial.fc = nn.Linear(2048, n_classes)

        # Temporal stream: stacked optical flow
        self.temporal = models.resnet50(pretrained=True)
        # Modify first conv for flow input (2*n_flow_frames channels)
        self.temporal.conv1 = nn.Conv2d(
            2 * n_flow_frames, 64, kernel_size=7, stride=2, padding=3
        )
        self.temporal.fc = nn.Linear(2048, n_classes)

    def forward(self, rgb_frame, flow_stack):
        """
        Args:
            rgb_frame: (B, 3, H, W) single RGB frame
            flow_stack: (B, 2*T, H, W) stacked optical flow
        Returns:
            logits: (B, n_classes) fused predictions
        """
        spatial_logits = self.spatial(rgb_frame)
        temporal_logits = self.temporal(flow_stack)

        # Late fusion: average logits
        return 0.5 * spatial_logits + 0.5 * temporal_logits
```

---

## 3. 3D Convolutions (C3D, I3D)

### Theory: 3D Convolutions: Convolving Across Space AND Time

A 2D conv slides a `k × k` filter over `H × W`. A **3D conv** slides a `k × k × k` filter over `T × H × W` — the time axis is treated as a third spatial dimension.

C3D (Tran et al., 2015) was the first deep network built entirely from 3D convs. I3D (Carreira & Zisserman, 2017) showed how to **inflate** a 2D ImageNet-pretrained backbone to 3D: replicate each 2D filter `k` times along the new time dimension and divide by `k`. This trick lets you initialize 3D networks from 2D pretraining, dramatically reducing data requirements.

Trade-offs:

- **Parameters scale**: a 3×3 conv has 9 weights per channel; a 3×3×3 conv has 27.
- **Compute scales linearly with `T`**.
- **Captures local space-time patterns** (a hand moving across a few frames) directly without separate optical flow.

Modern variants:

- **R(2+1)D**: factor a 3D conv into a spatial 2D conv followed by a temporal 1D conv. Same expressive power, fewer parameters, easier optimization.
- **CSN (Channel-Separated Networks)**: 3D depthwise convs for further efficiency.

### 3.1 3D Convolution Concept

```
2D Conv: kernel (k, k) slides over (H, W) → spatial features
3D Conv: kernel (k, k, k) slides over (T, H, W) → spatiotemporal features

  2D Conv on video (frame-by-frame):
  Cannot capture motion between frames.

  3D Conv on video:
  Kernel spans multiple frames → captures motion patterns!
  e.g., 3×3×3 kernel looks at 3 frames × 3×3 spatial window
```

### 3.2 I3D (Inflated 3D)

```python
class I3DBlock(nn.Module):
    """Inflated 3D convolution block (inflate 2D filters to 3D)."""

    def __init__(self, in_channels, out_channels, temporal_kernel=3):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(temporal_kernel, 3, 3),
            padding=(temporal_kernel // 2, 1, 1),
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T, H, W)
        return self.relu(self.bn(self.conv(x)))


class SimpleI3D(nn.Module):
    """Simplified I3D for action recognition."""

    def __init__(self, n_classes=400, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            I3DBlock(in_channels, 64, temporal_kernel=7),
            nn.MaxPool3d((1, 2, 2)),
            I3DBlock(64, 128),
            nn.MaxPool3d((2, 2, 2)),
            I3DBlock(128, 256),
            I3DBlock(256, 256),
            nn.MaxPool3d((2, 2, 2)),
            I3DBlock(256, 512),
            I3DBlock(512, 512),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        # x: (B, 3, T, H, W) - video clip
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
```

---

## 4. SlowFast Networks

### Theory: SlowFast: Multi-Rate Temporal Modeling

SlowFast (Feichtenhofer et al., 2019) observed that the **rate** at which different visual content matters varies:

- **Spatial content** (objects, scenes): consistent across frames; doesn't need many frames per second.
- **Temporal/motion content** (gestures, movements): needs high frame rate to capture fast actions.

SlowFast addresses this with **two pathways**:

- **Slow pathway**: low frame rate (e.g. 4 frames per clip), heavyweight network with many channels. Captures scene/objects.
- **Fast pathway**: high frame rate (e.g. 32 frames per clip), lightweight network with few channels. Captures motion.
- **Lateral connections**: at multiple stages, the fast pathway feeds into the slow pathway, fusing motion cues into the deeper representation.

The architecture trades temporal resolution against channel capacity per pathway. SlowFast achieved state-of-the-art on action recognition benchmarks for several years.

### 4.1 SlowFast Concept

```
SlowFast: Two pathways operating at different temporal rates.

Slow pathway:
  Low frame rate (e.g., 4 FPS)
  High channel capacity
  Captures spatial semantics and appearance

Fast pathway:
  High frame rate (e.g., 32 FPS)
  Low channel capacity (8x fewer channels)
  Captures fine temporal patterns and motion

  Slow: ████████████  (4 frames, 64 channels)
  Fast: ████████████████████████████████  (32 frames, 8 channels)

Lateral connections: Fast → Slow (fuse temporal info into spatial)
```

### 4.2 SlowFast Implementation

```python
class SlowFastNetwork(nn.Module):
    """Simplified SlowFast network."""

    def __init__(self, n_classes=400, alpha=8, beta=8):
        super().__init__()
        self.alpha = alpha  # Frame rate ratio (fast/slow)
        self.beta = beta    # Channel ratio (slow/fast)

        # Slow pathway
        self.slow_conv1 = nn.Conv3d(3, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.slow_pool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_res = self._make_layer(64 + 64 // beta, 256, n_blocks=3)

        # Fast pathway
        self.fast_conv1 = nn.Conv3d(3, 64 // beta, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.fast_pool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res = self._make_layer(64 // beta, 256 // beta, n_blocks=3)

        # Lateral connection (fast → slow)
        self.lateral = nn.Conv3d(64 // beta, 64 // beta, (5, 1, 1),
                                stride=(alpha, 1, 1), padding=(2, 0, 0))

        # Classifier
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256 + 256 // beta, n_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks):
        layers = [nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.ReLU()]
        for _ in range(n_blocks - 1):
            layers.extend([nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.ReLU()])
        return nn.Sequential(*layers)

    def forward(self, video):
        """
        Args:
            video: (B, 3, T, H, W) - full frame rate video
        Returns:
            logits: (B, n_classes)
        """
        # Subsample for slow pathway
        slow_input = video[:, :, ::self.alpha]  # Every alpha-th frame

        # Slow pathway
        slow = self.slow_conv1(slow_input)
        slow = self.slow_pool(slow)

        # Fast pathway
        fast = self.fast_conv1(video)
        fast = self.fast_pool(fast)

        # Lateral connection: fast → slow
        lateral = self.lateral(fast)
        slow = torch.cat([slow, lateral], dim=1)

        # Continue processing
        slow = self.slow_res(slow)
        fast = self.fast_res(fast)

        # Merge and classify
        slow = self.pool(slow).flatten(1)
        fast = self.pool(fast).flatten(1)
        x = torch.cat([slow, fast], dim=1)

        return self.fc(x)
```

---

## 5. Video Transformers

### Theory: Video Transformers

Vision transformers (ViT) replaced CNNs in image classification, and the same shift happened in video. **TimeSformer**, **ViViT**, and **Video Swin Transformer** apply attention over space and time:

- **Tokens**: split each frame into patches, treat all patches across all frames as a token sequence.
- **Spatial attention**: attend among patches in the same frame.
- **Temporal attention**: attend among patches at the same spatial location across frames.
- **Joint attention**: attend over all space-time tokens simultaneously (more expensive but more flexible).

The factored variants (separate spatial and temporal attention layers) are computationally cheaper than fully joint attention, with similar performance.

The state-of-the-art today: **Video MAE (Masked Autoencoder)** for self-supervised pretraining + transformer fine-tuning. Mask out 90%+ of patches and train to reconstruct them. The strong pretext task gives massive video models good representations even with limited labeled data.

### 5.1 TimeSformer

```python
class TimeSformerBlock(nn.Module):
    """TimeSformer: Divided Space-Time Attention."""

    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        # Temporal attention (attend across time at same spatial position)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)

        # Spatial attention (attend across space at same time step)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, T, S):
        """
        Args:
            x: (B, T*S, D) flattened video tokens
            T: number of time steps
            S: number of spatial tokens per frame
        """
        B, N, D = x.shape

        # 1. Temporal attention
        x_t = x.reshape(B * S, T, D)  # Group by spatial position
        attn_out, _ = self.temporal_attn(x_t, x_t, x_t)
        x = x + attn_out.reshape(B, N, D)
        x = self.temporal_norm(x)

        # 2. Spatial attention
        x_s = x.reshape(B * T, S, D)  # Group by time step
        attn_out, _ = self.spatial_attn(x_s, x_s, x_s)
        x = x + attn_out.reshape(B, N, D)
        x = self.spatial_norm(x)

        # 3. FFN
        x = x + self.ffn(x)
        x = self.ffn_norm(x)

        return x
```

---

## 6. Temporal Action Detection

### Theory: Temporal Action Detection

Beyond classification ("what action is happening?"), video understanding also includes **temporal localization** ("when does it start and end?"). This is the temporal analog of object detection — instead of bounding boxes in `(x, y)`, you output bounding intervals in time.

Approaches mirror image detection:

- **Two-stage**: propose temporal segments, then classify each (analogous to Faster R-CNN).
- **One-stage**: directly regress action class + start + end at each time step (analogous to YOLO).
- **Anchor-free**: predict per-frame action class plus offsets to action boundaries.

Evaluation: temporal IoU + AP, computed analogously to image-level detection but in 1D instead of 2D.

### 6.1 Action Detection Pipeline

```python
def temporal_action_detection(video_features, model, threshold=0.5):
    """
    Detect actions in an untrimmed video.

    Output: List of (action_class, start_time, end_time, confidence)
    """
    # 1. Generate temporal proposals (candidate segments)
    proposals = generate_proposals(video_features)

    # 2. Classify each proposal
    detections = []
    for start, end in proposals:
        segment_features = video_features[start:end]
        pooled = segment_features.mean(dim=0)

        class_scores = model.classify(pooled)
        action_class = class_scores.argmax().item()
        confidence = class_scores.max().item()

        if confidence > threshold:
            detections.append({
                'class': action_class,
                'start': start / fps,
                'end': end / fps,
                'confidence': confidence,
            })

    # 3. Non-maximum suppression
    detections = temporal_nms(detections, iou_threshold=0.5)

    return detections
```

---

## 7. Practical Video Pipeline

### 7.1 Video Dataset Loading

```python
import cv2
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """Video dataset with clip sampling."""

    def __init__(self, video_paths, labels, clip_length=16,
                 frame_rate=4, transform=None):
        self.videos = video_paths
        self.labels = labels
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.transform = transform

    def __getitem__(self, idx):
        # Load video
        cap = cv2.VideoCapture(self.videos[idx])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # Sample frames at target rate
        sample_interval = max(1, int(video_fps / self.frame_rate))
        frame_indices = list(range(0, total_frames, sample_interval))

        # Random temporal crop
        if len(frame_indices) > self.clip_length:
            start = np.random.randint(0, len(frame_indices) - self.clip_length)
            frame_indices = frame_indices[start:start + self.clip_length]

        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()

        # Pad if needed
        while len(frames) < self.clip_length:
            frames.append(frames[-1])

        clip = np.stack(frames[:self.clip_length])  # (T, H, W, 3)

        if self.transform:
            clip = self.transform(clip)

        # (T, H, W, 3) → (3, T, H, W)
        clip = torch.FloatTensor(clip).permute(3, 0, 1, 2) / 255.0

        return clip, self.labels[idx]

    def __len__(self):
        return len(self.videos)
```

---

## 8. Exercises

### Exercise 1: 3D CNN Action Recognition

Build an action recognition system:
1. Implement a 3D CNN (C3D-style) for video classification
2. Train on Kinetics-400 subset or UCF-101
3. Compare: 2D CNN per frame vs 3D CNN
4. Visualize: what temporal patterns does the 3D conv learn?
5. Report: top-1 and top-5 accuracy

### Exercise 2: Two-Stream Network

Implement the two-stream architecture:
1. Compute optical flow using OpenCV (Farneback or RAFT)
2. Build spatial stream (RGB frames) and temporal stream (flow stacks)
3. Implement early and late fusion strategies
4. Compare: spatial-only, temporal-only, and two-stream
5. Analyze: which actions rely more on appearance vs motion?

### Exercise 3: SlowFast from Scratch

Build a simplified SlowFast network:
1. Implement slow and fast pathways
2. Add lateral connections (fast → slow)
3. Train on a video dataset and compare with single-pathway
4. Vary alpha (frame rate ratio): {4, 8, 16} and measure impact
5. Visualize: what does each pathway focus on?

### Exercise 4: Video Transformer

Implement divided space-time attention:
1. Patch-embed video frames (16×16 patches)
2. Implement temporal attention and spatial attention separately
3. Train on video classification task
4. Compare with 3D CNN on same dataset
5. Visualize attention patterns: temporal and spatial

### Exercise 5: Real-Time Action Recognition

Build a real-time action recognition system:
1. Use webcam as input
2. Implement sliding window over last N frames
3. Run lightweight model (MobileNet-based) for classification
4. Display: current action, confidence, FPS
5. Handle: transition between actions, "no action" class

---

*End of Lesson 30*
