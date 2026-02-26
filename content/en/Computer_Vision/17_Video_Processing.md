# Video Processing

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how VideoCapture and VideoWriter work to read and write video files and camera streams in OpenCV.
2. Implement frame-by-frame video processing pipelines with accurate FPS measurement.
3. Apply background subtraction algorithms (MOG2, KNN) to detect moving objects in video.
4. Implement optical flow techniques (Lucas-Kanade, Farneback) to analyze motion between frames.
5. Compare and select appropriate object tracking algorithms for different video analysis tasks.

---

## Overview

Video is a sequence of continuous image frames. We will learn to process video files and camera streams using OpenCV, and explore motion analysis methods using background subtraction and optical flow.

Unlike single-image processing, video introduces a temporal dimension: each frame has a predecessor and successor, so algorithms can exploit motion cues unavailable in still images. This also adds real-time constraints — a processing pipeline that takes 50 ms per frame limits throughput to 20 FPS, making performance awareness essential for video work.

**Difficulty**: ***

**Prerequisites**: Basic image operations, filtering, object detection

---

## Table of Contents

1. [VideoCapture: Files and Cameras](#1-videocapture-files-and-cameras)
2. [VideoWriter: Saving Video](#2-videowriter-saving-video)
3. [Frame-by-frame Processing](#3-frame-by-frame-processing)
4. [FPS Calculation](#4-fps-calculation)
5. [Background Subtraction (MOG2, KNN)](#5-background-subtraction-mog2-knn)
6. [Optical Flow](#6-optical-flow)
7. [Object Tracking](#7-object-tracking)
8. [Practice Problems](#8-practice-problems)

---

## 1. VideoCapture: Files and Cameras

### Understanding Video Structure

```
Video = Sequence of continuous image frames

Time ------------------------------------------>
    +-----++-----++-----++-----++-----+
    |Frame||Frame||Frame||Frame||Frame| ...
    |  1  ||  2  ||  3  ||  4  ||  5  |
    +-----++-----++-----++-----++-----+

FPS (Frames Per Second): Number of frames per second
- 24 FPS: Movie standard
- 30 FPS: General video
- 60 FPS: Gaming, sports
- 120+ FPS: Slow motion

Resolution: Size of each frame
- 640x480: VGA
- 1280x720: HD (720p)
- 1920x1080: Full HD (1080p)
- 3840x2160: 4K
```

### Reading Video Files

```python
import cv2

# Open video file
cap = cv2.VideoCapture('video.mp4')

# Check if opened successfully
if not cap.isOpened():
    print("Cannot open video")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

# Frame reading loop
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error")
        break

    # Frame processing
    cv2.imshow('Video', frame)

    # Exit with 'q' key, wait 1ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

### Camera Input

```python
import cv2

# Open camera (device ID: 0=default camera)
cap = cv2.VideoCapture(0)

# If camera fails to open
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set camera properties — explicitly request resolution and FPS because
# cameras often default to a lower mode; requesting forces negotiation
# with the driver (actual values may still differ, always verify after setting)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# BUFFERSIZE=1 keeps only the most recent frame in the driver buffer,
# trading throughput for latency — critical for real-time applications
# where a stale frame is worse than a dropped one
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # Horizontal flip (mirror effect)
    frame = cv2.flip(frame, 1)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Key VideoCapture Properties

```python
import cv2

cap = cv2.VideoCapture('video.mp4')

# Read properties
properties = {
    'CAP_PROP_FRAME_WIDTH': cv2.CAP_PROP_FRAME_WIDTH,    # Frame width
    'CAP_PROP_FRAME_HEIGHT': cv2.CAP_PROP_FRAME_HEIGHT,  # Frame height
    'CAP_PROP_FPS': cv2.CAP_PROP_FPS,                    # FPS
    'CAP_PROP_FRAME_COUNT': cv2.CAP_PROP_FRAME_COUNT,    # Total frame count
    'CAP_PROP_POS_FRAMES': cv2.CAP_PROP_POS_FRAMES,      # Current frame position
    'CAP_PROP_POS_MSEC': cv2.CAP_PROP_POS_MSEC,          # Current position (ms)
    'CAP_PROP_FOURCC': cv2.CAP_PROP_FOURCC,              # Codec 4-char code
    'CAP_PROP_BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,      # Brightness (camera)
    'CAP_PROP_CONTRAST': cv2.CAP_PROP_CONTRAST,          # Contrast (camera)
}

for name, prop in properties.items():
    value = cap.get(prop)
    print(f"{name}: {value}")

# Seek to specific frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Go to frame 100

# Seek to specific time (milliseconds)
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)  # Go to 5 seconds

cap.release()
```

---

## 2. VideoWriter: Saving Video

### Basic Video Saving

```python
import cv2

# Video capture setup
cap = cv2.VideoCapture(0)

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0

# Codec setup (4-character code)
# 'XVID': for AVI container — widely compatible, moderate compression
# 'mp4v': for MP4 container — good balance of compatibility and file size
# 'MJPG': Motion JPEG — fast but large files (each frame independently compressed)
# 'avc1'/'X264': H.264 — highest compression ratio but requires codec install
# Choose mp4v when portability matters; use XVID when H.264 is unavailable
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create VideoWriter
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

print("Recording started... Press 'q' to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame
    out.write(frame)

    # Recording indicator
    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle
    cv2.putText(frame, 'REC', (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording complete: output.mp4")
```

### Major Codecs

```
+-----------+-------------+------------------------+
|   Codec   |  Container  |      Characteristics   |
+-----------+-------------+------------------------+
| 'XVID'    | .avi        | Widely supported,      |
|           |             | decent compression     |
| 'MJPG'    | .avi        | Motion JPEG, fast      |
| 'mp4v'    | .mp4        | MPEG-4, good compat    |
| 'avc1'    | .mp4        | H.264, high compression|
| 'X264'    | .mp4        | H.264 (requirements)   |
| 'VP80'    | .webm       | VP8, for web           |
| 'VP90'    | .webm       | VP9, high efficiency   |
+-----------+-------------+------------------------+

# Codec test
def test_codec(codec_str, extension):
    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    out = cv2.VideoWriter(f'test.{extension}', fourcc, 30, (640, 480))
    if out.isOpened():
        print(f"{codec_str}: Supported")
        out.release()
        return True
    else:
        print(f"{codec_str}: Not supported")
        return False
```

### Processing and Saving Video

```python
import cv2

def process_and_save_video(input_path, output_path, process_func):
    """Process video and save"""

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed = process_func(frame)

        # Save
        out.write(processed)

        # Progress display
        frame_num += 1
        progress = (frame_num / total_frames) * 100
        print(f"\rProcessing: {progress:.1f}%", end='')

    print("\nComplete!")

    cap.release()
    out.release()

# Usage example: Grayscale conversion and edge detection
def edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Convert to 3 channels (VideoWriter is set for color video)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

process_and_save_video('input.mp4', 'edges.mp4', edge_detection)
```

---

## 3. Frame-by-frame Processing

### Frame Processing Pipeline

```
Frame Processing Pipeline:

Input --> Preprocessing --> Analysis --> Postprocessing --> Output
              |              |              |
              v              v              v
          - Resize       - Detection    - Visualization
          - Color conv   - Tracking     - Filtering
          - Noise        - Recognition  - Compositing
            removal
```

### Multi-processing Example

```python
import cv2
import numpy as np

class VideoProcessor:
    """Video frame processor"""

    def __init__(self):
        self.processors = []

    def add_processor(self, name, func):
        """Add processing function"""
        self.processors.append((name, func))

    def process_frame(self, frame):
        """Apply all processing functions"""
        result = frame.copy()
        for name, func in self.processors:
            result = func(result)
        return result

    def process_video(self, input_source, output_path=None, display=True):
        """Process video"""
        cap = cv2.VideoCapture(input_source)

        out = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process
            processed = self.process_frame(frame)

            # Save
            if out:
                out.write(processed)

            # Display
            if display:
                cv2.imshow('Processed', processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

# Usage example
processor = VideoProcessor()

# Add processing functions
processor.add_processor('blur', lambda f: cv2.GaussianBlur(f, (5, 5), 0))
processor.add_processor('edge', lambda f: cv2.Canny(f, 50, 150))

def add_timestamp(frame):
    import datetime
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, now, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

processor.add_processor('timestamp', add_timestamp)

# Process webcam
processor.process_video(0, output_path='recorded.mp4')
```

### Frame Skipping and Buffering

```python
import cv2
import time

def skip_frames_processing(video_path, skip=5):
    """Frame skipping (speed improvement)"""

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every skip frames
        if frame_count % skip != 0:
            continue

        # Perform heavy processing
        processed = heavy_processing(frame)

        cv2.imshow('Skipped Processing', processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def buffered_reading(video_path, buffer_size=10):
    """Frame buffering (smooth playback)"""
    from collections import deque
    from threading import Thread

    cap = cv2.VideoCapture(video_path)
    buffer = deque(maxlen=buffer_size)
    stop_flag = False

    def read_frames():
        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
            if len(buffer) < buffer_size:
                buffer.append(frame)

    # Start reading thread
    thread = Thread(target=read_frames)
    thread.start()

    # Wait for initial buffer fill
    time.sleep(0.5)

    while True:
        if len(buffer) > 0:
            frame = buffer.popleft()
            cv2.imshow('Buffered', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    stop_flag = True
    thread.join()
    cap.release()
```

---

## 4. FPS Calculation

### FPS Measurement Method

```python
import cv2
import time

class FPSCounter:
    """FPS measurement class"""

    def __init__(self, avg_frames=30):
        self.frame_times = []
        # avg_frames=30 gives a ~1-second rolling window at 30 FPS —
        # large enough to smooth out single-frame spikes, small enough
        # to respond to genuine performance changes within seconds
        self.avg_frames = avg_frames
        self.last_time = time.time()

    def update(self):
        """Call after processing each frame"""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

        # Sliding window: discard the oldest sample so the average
        # reflects recent performance rather than startup conditions
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)

    def get_fps(self):
        """Return current FPS"""
        if len(self.frame_times) == 0:
            return 0
        # Averaging inter-frame intervals then inverting is more stable
        # than counting frames in a fixed time window, because it handles
        # irregular processing times without a separate timer thread
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0

# Usage example
cap = cv2.VideoCapture(0)
fps_counter = FPSCounter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame processing
    # ...

    fps_counter.update()
    fps = fps_counter.get_fps()

    # Display FPS
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('FPS', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### Processing Time Analysis

```python
import cv2
import time

class PerformanceMonitor:
    """Performance monitoring"""

    def __init__(self):
        self.timings = {}

    def start(self, name):
        """Start timing"""
        self.timings[name] = {'start': time.time()}

    def stop(self, name):
        """Stop timing"""
        if name in self.timings:
            elapsed = time.time() - self.timings[name]['start']
            self.timings[name]['elapsed'] = elapsed
            return elapsed
        return 0

    def get_report(self):
        """Performance report"""
        report = []
        for name, data in self.timings.items():
            if 'elapsed' in data:
                report.append(f"{name}: {data['elapsed']*1000:.2f}ms")
        return '\n'.join(report)

# Usage example
monitor = PerformanceMonitor()

cap = cv2.VideoCapture(0)

while True:
    # Measure total frame time
    monitor.start('total')

    ret, frame = cap.read()
    if not ret:
        break

    # Measure preprocessing time
    monitor.start('preprocess')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    monitor.stop('preprocess')

    # Measure detection time
    monitor.start('detection')
    edges = cv2.Canny(blur, 50, 150)
    monitor.stop('detection')

    monitor.stop('total')

    # Display performance
    y = 30
    for line in monitor.get_report().split('\n'):
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 20

    cv2.imshow('Performance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

---

## 5. Background Subtraction (MOG2, KNN)

### Background Subtraction Principle

```
Background Subtraction:
Separate moving foreground objects from stationary background

+-----------------+     +-----------------+     +-----------------+
| Current frame   |  -  | Background model|  =  | Foreground mask |
|                 |     |                 |     |                 |
|    +---+        |     |                 |     |    +---+        |
|    | * | (person)|    |   (empty room)  |     |    |###|        |
|    +---+        |     |                 |     |    +---+        |
|                 |     |                 |     |                 |
+-----------------+     +-----------------+     +-----------------+

Background model learning:
- Analyze multiple frames to learn background statistics
- Handle lighting changes, shadows, etc.
- Adapt to dynamic backgrounds (tree leaves, etc.)
```

### MOG2 (Mixture of Gaussians)

```python
import cv2
import numpy as np

# Create MOG2 background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,          # Frames used to build background model —
                          # larger = slower adaptation to scene changes
                          # (e.g., 500 frames at 30 FPS ≈ 16 seconds of memory)
    varThreshold=16,      # Mahalanobis distance threshold for classifying a pixel
                          # as foreground; lower = more sensitive but more noise
    detectShadows=True    # Marks shadows as 127 (gray) instead of 255 (white),
                          # letting you remove them separately to avoid false positives
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    # fgMask: foreground=255, background=0, shadow=127
    fgMask = backSub.apply(frame)

    # Remove shadows (127 -> 0)
    fgMask_no_shadow = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)[1]

    # Remove noise with morphological operations:
    # OPEN (erode then dilate) removes small speckles/salt noise
    # CLOSE (dilate then erode) fills holes inside detected objects
    # ELLIPSE kernel is rotationally symmetric — better for blob-shaped objects
    # than RECT, which leaves corner artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask_clean = cv2.morphologyEx(fgMask_no_shadow, cv2.MORPH_OPEN, kernel)
    fgMask_clean = cv2.morphologyEx(fgMask_clean, cv2.MORPH_CLOSE, kernel)

    # Extract foreground
    foreground = cv2.bitwise_and(frame, frame, mask=fgMask_clean)

    # Display results
    cv2.imshow('Original', frame)
    cv2.imshow('FG Mask', fgMask)
    cv2.imshow('Cleaned Mask', fgMask_clean)
    cv2.imshow('Foreground', foreground)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### KNN Background Subtraction

```python
import cv2

# Create KNN background subtractor
backSub = cv2.createBackgroundSubtractorKNN(
    history=500,          # Background learning frame count
    dist2Threshold=400.0, # Distance threshold
    detectShadows=True    # Shadow detection
)

cap = cv2.VideoCapture('traffic.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Background subtraction
    fgMask = backSub.apply(frame)

    # Remove noise
    fgMask = cv2.medianBlur(fgMask, 5)

    # Contour detection
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Mark moving objects
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area filter
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Motion Detection', frame)
    cv2.imshow('Mask', fgMask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
```

### MOG2 vs KNN Comparison

```
+----------------+----------------------+----------------------+
|     Item       |        MOG2          |        KNN           |
+----------------+----------------------+----------------------+
| Algorithm      | Gaussian Mixture Model| K-Nearest Neighbors |
| Speed          | Fast                 | Medium               |
| Memory         | Low                  | High                 |
| Dynamic BG     | Medium               | Good                 |
| Lighting Change| Medium               | Good                 |
| Noise          | Sensitive            | Robust               |
| Recommended    | Static scenes,       | Complex scenes       |
|                | real-time            |                      |
+----------------+----------------------+----------------------+
```

---

## 6. Optical Flow

### Optical Flow Concept

```
Optical Flow:
Estimate pixel movement between consecutive frames

Frame t                    Frame t+1
+-----------------+        +-----------------+
|                 |        |                 |
|    *            |   ->   |        *        |
|                 |        |                 |
+-----------------+        +-----------------+

Velocity vector (u, v):
- Pixel (x, y) moves to (x+u, y+v) in next frame
- I(x, y, t) = I(x+u, y+v, t+1) (brightness constancy assumption)

Types:
1. Sparse: Only compute movement for specific points (Lucas-Kanade)
2. Dense: Compute movement for all pixels (Farneback)
```

### Lucas-Kanade Optical Flow

```python
import cv2
import numpy as np

# Lucas-Kanade parameters
lk_params = dict(
    winSize=(15, 15),      # Search window: larger = handles bigger motion but slower
                           # and prone to aperture problem on textureless regions
    maxLevel=2,            # Image pyramid levels: pyramid lets LK handle fast motion
                           # by first estimating flow on a downsampled image, then
                           # refining on higher resolution (maxLevel=2 → 3 scales)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    # Stop iterating when error < 0.03 OR after 10 iterations — combining both
    # prevents wasting time on converged estimates and limits worst-case cost
)

# Feature detection parameters
feature_params = dict(
    maxCorners=100,        # Cap at 100 to keep tracking computationally feasible
    qualityLevel=0.3,      # Keep only corners scoring ≥ 30% of the strongest one,
                           # filtering weak features that would drift under noise
    minDistance=7,         # Enforce spatial spread so features cover the whole frame,
                           # not just one high-contrast region
    blockSize=7
)

cap = cv2.VideoCapture(0)

# Read first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect features
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# For trajectory visualization
mask = np.zeros_like(old_frame)

# Colors
colors = np.random.randint(0, 255, (100, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None and len(p0) > 0:
        # Compute optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is not None:
            # Select good points only
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Visualize movement
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)

                # Trajectory line
                mask = cv2.line(mask, (a, b), (c, d),
                               colors[i % 100].tolist(), 2)
                # Current position point
                frame = cv2.circle(frame, (a, b), 5,
                                   colors[i % 100].tolist(), -1)

            # Update for next frame
            p0 = good_new.reshape(-1, 1, 2)

    # Combine trajectory
    img = cv2.add(frame, mask)

    cv2.imshow('Lucas-Kanade', img)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Re-detect features with 'r' key
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)

    old_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()
```

### Farneback Dense Optical Flow

```python
import cv2
import numpy as np

def draw_flow(img, flow, step=16):
    """Visualize flow vectors"""
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    fx, fy = flow[y, x].T

    # Draw lines
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    vis = img.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 2, (0, 255, 0), -1)

    return vis

def flow_to_hsv(flow):
    """Convert flow to HSV color"""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Direction -> Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude -> Value

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prvs, next_gray,
        None,           # Initial flow (None = start from zero displacement)
        pyr_scale=0.5,  # Each pyramid level is half the previous resolution —
                        # 0.5 is the standard choice; lower values handle larger
                        # motions but increase computation
        levels=3,       # 3 pyramid levels cover displacements up to ~8× winsize
        winsize=15,     # Neighborhood for polynomial expansion; larger = smoother
                        # flow but blurs motion boundaries
        iterations=3,   # Refinement passes per pyramid level; 3 is enough for
                        # typical video, more iterations rarely improve quality
        poly_n=5,       # Pixel neighborhood size for polynomial fit (5 or 7)
        poly_sigma=1.2, # Gaussian weighting of the neighborhood; must match poly_n
                        # (use 1.1 for poly_n=5, 1.5 for poly_n=7)
        flags=0
    )

    # Visualization
    flow_vis = draw_flow(frame2, flow)
    hsv_vis = flow_to_hsv(flow)

    cv2.imshow('Flow Vectors', flow_vis)
    cv2.imshow('Flow HSV', hsv_vis)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    prvs = next_gray

cap.release()
cv2.destroyAllWindows()
```

---

## 7. Object Tracking

### OpenCV Built-in Trackers

```python
import cv2

# Tracker types
TRACKERS = {
    'BOOSTING': cv2.legacy.TrackerBoosting_create,
    'MIL': cv2.TrackerMIL_create,
    'KCF': cv2.TrackerKCF_create,
    'CSRT': cv2.TrackerCSRT_create,
    'MOSSE': cv2.legacy.TrackerMOSSE_create
}

def track_object(video_path, tracker_type='CSRT'):
    """Single object tracking"""

    # Create tracker
    tracker = TRACKERS[tracker_type]()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    # Select object to track (mouse drag)
    bbox = cv2.selectROI('Select Object', frame, False)
    cv2.destroyWindow('Select Object')

    # Initialize tracker
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracking
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, tracker_type, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Tracking Failed', (100, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Tracking', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Usage example
track_object('video.mp4', 'CSRT')
```

### Multi-object Tracking

```python
import cv2

class MultiObjectTracker:
    """Multi-object tracker"""

    def __init__(self, tracker_type='CSRT'):
        self.tracker_type = tracker_type
        self.trackers = []
        self.colors = []

    def add_tracker(self, frame, bbox):
        """Add new tracker"""
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        self.trackers.append(tracker)
        self.colors.append((
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        ))

    def update(self, frame):
        """Update all trackers"""
        results = []

        for i, tracker in enumerate(self.trackers):
            success, bbox = tracker.update(frame)
            if success:
                results.append({
                    'id': i,
                    'bbox': bbox,
                    'color': self.colors[i]
                })

        return results

    def draw(self, frame, results):
        """Visualize results"""
        for r in results:
            x, y, w, h = [int(v) for v in r['bbox']]
            cv2.rectangle(frame, (x, y), (x+w, y+h), r['color'], 2)
            cv2.putText(frame, f"ID: {r['id']}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, r['color'], 2)
        return frame

# Usage example
import numpy as np

cap = cv2.VideoCapture(0)
multi_tracker = MultiObjectTracker()

ret, frame = cap.read()

# Select multiple objects (ESC to finish)
while True:
    bbox = cv2.selectROI('Select Objects (Press ESC when done)', frame, False)
    if bbox == (0, 0, 0, 0):  # ESC pressed
        break
    multi_tracker.add_tracker(frame, bbox)

cv2.destroyWindow('Select Objects (Press ESC when done)')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = multi_tracker.update(frame)
    frame = multi_tracker.draw(frame, results)

    cv2.imshow('Multi Tracking', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
```

### Background Subtraction + Tracking Combined

```python
import cv2
import numpy as np

class MotionTracker:
    """Background subtraction-based motion tracking"""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.tracks = {}  # {id: {'centroid': (x,y), 'frames': count}}
        self.next_id = 0
        self.max_distance = 50  # Distance for same object judgment

    def process(self, frame):
        """Process frame"""
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        # Remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        # Contour detection
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Current frame's objects
        current_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                centroid = (x + w//2, y + h//2)
                current_objects.append({
                    'centroid': centroid,
                    'bbox': (x, y, w, h)
                })

        # Match with existing tracks
        self._match_tracks(current_objects)

        return fg_mask, current_objects

    def _match_tracks(self, current_objects):
        """Match current objects with existing tracks"""
        matched = set()

        for obj in current_objects:
            cx, cy = obj['centroid']
            best_match = None
            best_dist = float('inf')

            # Find closest existing track
            for track_id, track in self.tracks.items():
                tx, ty = track['centroid']
                dist = np.sqrt((cx-tx)**2 + (cy-ty)**2)

                if dist < self.max_distance and dist < best_dist:
                    best_dist = dist
                    best_match = track_id

            if best_match is not None:
                # Update existing track
                self.tracks[best_match]['centroid'] = obj['centroid']
                self.tracks[best_match]['bbox'] = obj['bbox']
                self.tracks[best_match]['frames'] += 1
                obj['id'] = best_match
                matched.add(best_match)
            else:
                # Create new track
                obj['id'] = self.next_id
                self.tracks[self.next_id] = {
                    'centroid': obj['centroid'],
                    'bbox': obj['bbox'],
                    'frames': 1
                }
                self.next_id += 1

        # Remove old tracks
        to_remove = [tid for tid in self.tracks if tid not in matched]
        for tid in to_remove:
            if self.tracks[tid]['frames'] < 10:  # Remove short tracks immediately
                del self.tracks[tid]

    def draw(self, frame, objects):
        """Visualize"""
        for obj in objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if 'id' in obj:
                cv2.putText(frame, f"ID: {obj['id']}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

# Usage example
cap = cv2.VideoCapture(0)
tracker = MotionTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask, objects = tracker.process(frame)
    output = tracker.draw(frame, objects)

    cv2.imshow('Motion Tracking', output)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
```

---

## 8. Practice Problems

### Problem 1: Video Player

Implement a basic video player.

**Requirements**:
- Play/pause toggle (spacebar)
- Forward/backward skip (arrow keys)
- Frame-by-frame navigation (./,)
- Display current time/total time
- Progress bar

<details>
<summary>Hint</summary>

```python
# Frame navigation
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

# Key handling
key = cv2.waitKey(delay) & 0xFF
if key == ord(' '):  # Spacebar
    paused = not paused
elif key == 83:  # Right arrow
    skip_forward()
```

</details>

### Problem 2: Motion Heatmap

Visualize areas with lots of motion as a heatmap.

**Requirements**:
- Detect motion with background subtraction
- Generate accumulated motion map
- Apply colormap (COLORMAP_JET)
- Blend original with heatmap

<details>
<summary>Hint</summary>

```python
# Initialize accumulation map
accumulator = np.zeros((height, width), dtype=np.float32)

# Accumulate per frame
accumulator += fg_mask.astype(np.float32) / 255.0

# Normalize and apply colormap
normalized = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)
heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
```

</details>

### Problem 3: Speed Measurement

Measure object movement speed using optical flow.

**Requirements**:
- Compute average flow within specific ROI
- Convert pixel speed to actual speed (calibration needed)
- Display real-time speed graph

<details>
<summary>Hint</summary>

```python
# Average flow in ROI
roi_flow = flow[y:y+h, x:x+w]
avg_flow = np.mean(roi_flow, axis=(0, 1))

# Speed calculation (pixels/frame)
speed = np.sqrt(avg_flow[0]**2 + avg_flow[1]**2)

# Convert to actual speed (e.g., 1 pixel = 1cm, 30fps)
real_speed = speed * pixels_to_cm * fps  # cm/s
```

</details>

### Problem 4: Vehicle Counter

Count vehicles passing through in road video.

**Requirements**:
- Detect vehicles with background subtraction
- Set virtual line (counting line)
- Count objects crossing the line
- Distinguish entry/exit direction

<details>
<summary>Hint</summary>

```python
# Define virtual line
line_y = height // 2

# Check if object crossed line
def crossed_line(prev_y, curr_y, line_y):
    # Top to bottom
    if prev_y < line_y and curr_y >= line_y:
        return 'down'
    # Bottom to top
    if prev_y > line_y and curr_y <= line_y:
        return 'up'
    return None
```

</details>

### Problem 5: Gesture Recognition

Analyze optical flow patterns to recognize simple gestures (hand waving, drawing circles).

**Requirements**:
- Detect hand region (skin color-based)
- Track movement patterns
- Classify patterns (rule-based or template matching)
- Display recognized gesture

<details>
<summary>Hint</summary>

```python
# Skin color detection (HSV)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_skin = np.array([0, 20, 70])
upper_skin = np.array([20, 255, 255])
mask = cv2.inRange(hsv, lower_skin, upper_skin)

# Store movement trajectory
trajectory = []
trajectory.append(centroid)

# Trajectory analysis
# Hand waving: oscillation in x direction
# Circle drawing: start and end points close + certain area
```

</details>

---

## Next Steps

- [18_Camera_Calibration.md](./18_Camera_Calibration.md) - Camera matrix, distortion correction

---

## References

- [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
- [Background Subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
- [Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [Object Tracking](https://docs.opencv.org/4.x/d9/df8/group__tracking.html)
- Horn, B. K., & Schunck, B. G. (1981). "Determining Optical Flow"
- Lucas, B. D., & Kanade, T. (1981). "An Iterative Image Registration Technique"
