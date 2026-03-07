# 14. 컴퓨터 비전을 위한 Edge AI

**이전**: [실시간 추론](./13_Real_Time_Inference.md) | **다음**: [NLP를 위한 Edge AI](./15_Edge_AI_for_NLP.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 엣지 디바이스에서 효율적인 객체 탐지 모델(YOLO-Nano, SSD-MobileNet)을 배포할 수 있다
2. 적절한 전처리/후처리와 함께 엣지 하드웨어에서 이미지 분류를 실행할 수 있다
3. 실시간 엣지 추론을 위한 시맨틱 세그멘테이션 모델을 최적화할 수 있다
4. 제한된 하드웨어에서 프레임 레이트를 유지하는 비디오 처리 파이프라인을 구축할 수 있다
5. 탐지, 분류, 추적을 연결하는 다중 모델 파이프라인을 설계할 수 있다
6. 엣지 배포를 위한 컴퓨터 비전 모델 패밀리를 벤치마킹하고 비교할 수 있다

---

컴퓨터 비전은 가장 일반적인 엣지 AI 워크로드입니다 -- 보안 카메라와 자율 주행 차량부터 제조 검사와 소매 분석까지 다양합니다. 그러나 엣지 디바이스에서 비전 모델을 실행하면 고유한 과제가 발생합니다: 이미지가 크고(1080p 프레임은 6 MB), 모델은 컴퓨팅 집약적이며(프레임당 수십억 번의 곱셈-누적 연산), 애플리케이션은 실시간 처리(30+ FPS)를 요구합니다. 이 레슨에서는 엣지에서 컴퓨터 비전을 실용적으로 만드는 모델 아키텍처, 최적화 기법, 파이프라인 패턴을 다룹니다.

---

## 1. 효율적인 객체 탐지

### 1.1 엣지 탐지 모델 현황

```
+-----------------------------------------------------------------+
|         Object Detection Models for Edge                         |
+-----------------------------------------------------------------+
|                                                                   |
|   Model Family      Params    mAP(COCO)  Latency*   Size        |
|   +---------------------------------------------------------+   |
|   | SSD-MobileNetV2   3.4M    22.0       12ms      14 MB    |   |
|   | YOLOv5-nano       1.9M    28.0       15ms       7 MB    |   |
|   | YOLOv8-nano       3.2M    37.3       18ms      12 MB    |   |
|   | EfficientDet-D0   3.9M    34.6       25ms      15 MB    |   |
|   | NanoDet-Plus      1.2M    30.4       10ms       5 MB    |   |
|   | YOLO-NAS-S        12.2M   47.5       35ms      48 MB    |   |
|   +---------------------------------------------------------+   |
|                                                                   |
|   * Latency measured on Snapdragon 888 (INT8)                    |
|   * mAP = mean Average Precision on COCO validation set          |
|                                                                   |
|   Trade-off: smaller models are faster but less accurate.        |
|   Choose based on your accuracy threshold + latency budget.      |
|                                                                   |
+-----------------------------------------------------------------+
```

### 1.2 SSD-MobileNet 배포

```python
#!/usr/bin/env python3
"""Deploy SSD-MobileNet for object detection on edge."""

import numpy as np
import time
from typing import List, Tuple

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


class EdgeObjectDetector:
    """SSD-MobileNet object detector optimized for edge inference."""

    def __init__(self, model_path: str, labels_path: str,
                 score_threshold: float = 0.5,
                 num_threads: int = 4):
        self.score_threshold = score_threshold

        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_height = self.input_details[0]["shape"][1]
        self.input_width = self.input_details[0]["shape"][2]
        self.is_quantized = self.input_details[0]["dtype"] != np.float32

        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalize image for SSD-MobileNet.

        SSD-MobileNet expects:
        - Shape: (1, 300, 300, 3) or (1, 320, 320, 3)
        - Range: [0, 255] for uint8 quantized, [-1, 1] for float
        """
        from PIL import Image

        img = Image.fromarray(image).resize(
            (self.input_width, self.input_height)
        )
        input_data = np.array(img)

        if self.is_quantized:
            input_data = input_data.astype(np.uint8)
        else:
            input_data = (input_data.astype(np.float32) - 127.5) / 127.5

        return np.expand_dims(input_data, axis=0)

    def detect(self, image: np.ndarray) -> List[dict]:
        """Run detection and return bounding boxes with class labels.

        SSD-MobileNet TFLite output tensors:
        [0] boxes:      (1, N, 4) - [ymin, xmin, ymax, xmax] normalized
        [1] classes:     (1, N) - class indices
        [2] scores:      (1, N) - confidence scores
        [3] num_detections: scalar
        """
        input_data = self.preprocess(image)
        h, w = image.shape[:2]

        self.interpreter.set_tensor(
            self.input_details[0]["index"], input_data
        )
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0]
        classes = self.interpreter.get_tensor(
            self.output_details[1]["index"]
        )[0]
        scores = self.interpreter.get_tensor(
            self.output_details[2]["index"]
        )[0]

        detections = []
        for i in range(len(scores)):
            if scores[i] < self.score_threshold:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            class_id = int(classes[i])

            detections.append({
                "class_id": class_id,
                "label": self.labels[class_id] if class_id < len(self.labels) else "unknown",
                "score": float(scores[i]),
                "bbox": {
                    "x1": int(xmin * w),
                    "y1": int(ymin * h),
                    "x2": int(xmax * w),
                    "y2": int(ymax * h),
                },
            })

        return detections


if __name__ == "__main__":
    detector = EdgeObjectDetector(
        model_path="ssd_mobilenet_v2.tflite",
        labels_path="coco_labels.txt",
        score_threshold=0.5
    )

    # Simulate a 640x480 image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    start = time.perf_counter()
    detections = detector.detect(image)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Detected {len(detections)} objects in {elapsed:.1f} ms")
    for det in detections:
        print(f"  {det['label']}: {det['score']:.2f} at {det['bbox']}")
```

### 1.3 YOLOv8-Nano 엣지 배포

```python
#!/usr/bin/env python3
"""YOLOv8-nano deployment with ONNX Runtime on edge."""

import numpy as np
import time
import onnxruntime as ort


class YOLOv8NanoDetector:
    """YOLOv8-nano optimized for edge deployment.

    Export workflow:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        model.export(format="onnx", imgsz=640, half=True, simplify=True)
    """

    def __init__(self, model_path: str, input_size: int = 640,
                 score_threshold: float = 0.25,
                 nms_iou_threshold: float = 0.45):
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

        # ONNX Runtime with optimizations
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(model_path, opts)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple]:
        """Letterbox resize preserving aspect ratio."""
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        from PIL import Image as PILImage
        resized = np.array(
            PILImage.fromarray(image).resize((new_w, new_h))
        )

        # Pad to square
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        padded = np.full(
            (self.input_size, self.input_size, 3), 114, dtype=np.uint8
        )
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # NCHW format, float32, normalized to [0, 1]
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        return blob, scale, (pad_h, pad_w)

    def postprocess(self, output: np.ndarray,
                    scale: float, padding: Tuple,
                    orig_shape: Tuple) -> List[dict]:
        """Decode YOLOv8 output: (1, 84, 8400) -> boxes + classes."""
        # output shape: (1, num_classes + 4, num_predictions)
        predictions = output[0].T  # (8400, 84)

        # Split boxes and class scores
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        # Filter by confidence
        max_scores = class_scores.max(axis=1)
        mask = max_scores > self.score_threshold
        boxes_xywh = boxes_xywh[mask]
        class_scores = class_scores[mask]
        max_scores = max_scores[mask]
        class_ids = class_scores.argmax(axis=1)

        # Convert xywh to xyxy
        boxes = np.zeros_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Remove padding and scale back to original image
        pad_h, pad_w = padding
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= scale

        # Clip to image bounds
        h, w = orig_shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        # NMS
        keep = self._nms(boxes, max_scores, self.nms_iou_threshold)

        detections = []
        for i in keep:
            detections.append({
                "class_id": int(class_ids[i]),
                "score": float(max_scores[i]),
                "bbox": {
                    "x1": int(boxes[i, 0]), "y1": int(boxes[i, 1]),
                    "x2": int(boxes[i, 2]), "y2": int(boxes[i, 3]),
                },
            })
        return detections

    def _nms(self, boxes: np.ndarray, scores: np.ndarray,
             iou_threshold: float) -> list:
        """Non-Maximum Suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, image: np.ndarray) -> List[dict]:
        """Full detection pipeline."""
        blob, scale, padding = self.preprocess(image)
        output = self.session.run(None, {self.input_name: blob})[0]
        return self.postprocess(output, scale, padding, image.shape)
```

---

## 2. 엣지에서의 이미지 분류

### 2.1 효율적인 분류 모델

| 모델 | Top-1 정확도 | 파라미터 | FLOPs | 지연 시간 (RPi4) | 크기 |
|-------|-----------|--------|-------|-----------------|------|
| MobileNetV3-Small | 67.4% | 2.5M | 56M | 18ms | 10 MB |
| MobileNetV3-Large | 75.2% | 5.4M | 219M | 45ms | 22 MB |
| EfficientNet-Lite0 | 75.1% | 4.7M | 407M | 55ms | 18 MB |
| ShuffleNetV2-0.5x | 60.6% | 1.4M | 41M | 12ms | 5 MB |
| MNASNet-0.5 | 67.7% | 2.2M | 102M | 22ms | 9 MB |

### 2.2 최적화된 분류 파이프라인

```python
#!/usr/bin/env python3
"""Optimized image classification pipeline for edge."""

import numpy as np
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


class EdgeClassifier:
    """Optimized classifier with pre-allocated buffers."""

    def __init__(self, model_path: str, labels_path: str,
                 num_threads: int = 4):
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        _, self.input_h, self.input_w, _ = self.input_details[0]["shape"]
        self.input_dtype = self.input_details[0]["dtype"]

        with open(labels_path) as f:
            self.labels = [l.strip() for l in f.readlines()]

        # Pre-allocate input buffer (avoids per-frame allocation)
        self._input_buffer = np.empty(
            self.input_details[0]["shape"],
            dtype=self.input_dtype
        )

    def classify(self, image: np.ndarray, top_k: int = 5) -> dict:
        """Classify an image and return top-k predictions."""
        # Resize (using numpy for speed, avoiding PIL import overhead)
        self._fast_resize(image, self._input_buffer[0])

        # Normalize in-place
        if self.input_dtype == np.float32:
            self._input_buffer[:] = (self._input_buffer - 127.5) / 127.5

        # Inference
        start = time.perf_counter()
        self.interpreter.set_tensor(
            self.input_details[0]["index"], self._input_buffer
        )
        self.interpreter.invoke()
        inference_ms = (time.perf_counter() - start) * 1000

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0]

        # Top-K
        top_indices = output.argsort()[-top_k:][::-1]
        predictions = [
            {
                "label": self.labels[i] if i < len(self.labels) else f"class_{i}",
                "score": float(output[i]),
            }
            for i in top_indices
        ]

        return {
            "predictions": predictions,
            "inference_ms": inference_ms,
        }

    def _fast_resize(self, src: np.ndarray, dst: np.ndarray):
        """Fast nearest-neighbor resize using numpy slicing.

        For edge inference, bilinear interpolation is often unnecessary.
        Nearest-neighbor is ~3x faster and has negligible accuracy impact
        on classification tasks.
        """
        h_src, w_src = src.shape[:2]
        h_dst, w_dst = dst.shape[:2]

        row_indices = (np.arange(h_dst) * h_src / h_dst).astype(int)
        col_indices = (np.arange(w_dst) * w_src / w_dst).astype(int)

        dst[:] = src[np.ix_(row_indices, col_indices)]

    def benchmark(self, num_frames: int = 200) -> dict:
        """Benchmark classification throughput."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(10):
            self.classify(image, top_k=1)

        times = []
        for _ in range(num_frames):
            start = time.perf_counter()
            self.classify(image, top_k=1)
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)
        return {
            "mean_ms": float(np.mean(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "fps": float(1000 / np.mean(times)),
        }
```

---

## 3. 시맨틱 세그멘테이션 최적화

### 3.1 엣지 세그멘테이션 모델

```
+-----------------------------------------------------------------+
|          Segmentation Models for Edge                             |
+-----------------------------------------------------------------+
|                                                                   |
|   Model                mIoU    Params   Latency*   Resolution    |
|   +----------------------------------------------------------+  |
|   | DeepLabV3-MNV2      72.4    2.1M     45ms      513x513   |  |
|   | BiSeNetV2            73.4    3.4M     20ms      1024x512  |  |
|   | PP-LiteSeg-T         73.1    0.6M     15ms      1024x512  |  |
|   | Fast-SCNN            68.0    1.1M     12ms      1024x2048 |  |
|   | TopFormer-T          71.5    1.4M     18ms      512x512   |  |
|   +----------------------------------------------------------+  |
|                                                                   |
|   * Latency on Qualcomm Snapdragon 888 (INT8)                   |
|                                                                   |
|   Key insight: segmentation resolution can often be reduced      |
|   (e.g., 256x256 instead of 512x512) with minimal quality       |
|   loss when the output is upscaled back. This gives ~4x speedup.|
|                                                                   |
+-----------------------------------------------------------------+
```

### 3.2 해상도 축소를 통한 세그멘테이션

```python
#!/usr/bin/env python3
"""Semantic segmentation optimized for edge with resolution tricks."""

import numpy as np
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


class EdgeSegmenter:
    """Semantic segmentation with configurable resolution.

    Segmentation at full resolution (e.g., 1024x512) is expensive.
    For many applications, running at half or quarter resolution
    and upscaling the output produces visually similar results at
    2-4x the speed.
    """

    def __init__(self, model_path: str, num_threads: int = 4):
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        _, self.model_h, self.model_w, _ = self.input_details[0]["shape"]

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Run segmentation and return class map.

        Returns:
            Class map (H, W) with integer class labels, same size as input
        """
        orig_h, orig_w = image.shape[:2]

        # Resize to model input
        from PIL import Image
        resized = np.array(
            Image.fromarray(image).resize((self.model_w, self.model_h))
        )
        input_data = resized.astype(np.float32) / 127.5 - 1.0
        input_data = np.expand_dims(input_data, 0)

        # Inference
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_data.astype(self.input_details[0]["dtype"])
        )
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        # Output: (1, H, W, num_classes) -> argmax -> (H, W)
        if output.ndim == 4:
            class_map = output[0].argmax(axis=-1).astype(np.uint8)
        else:
            class_map = output[0].astype(np.uint8)

        # Upscale to original resolution using nearest-neighbor
        class_map_resized = np.array(
            Image.fromarray(class_map).resize(
                (orig_w, orig_h), Image.NEAREST
            )
        )

        return class_map_resized

    def segment_with_overlay(self, image: np.ndarray,
                             alpha: float = 0.4) -> np.ndarray:
        """Segment and overlay colored masks on the original image."""
        class_map = self.segment(image)

        # Color palette (20 classes for common datasets)
        palette = np.array([
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        ], dtype=np.uint8)

        # Map classes to colors
        color_mask = palette[class_map % len(palette)]

        # Blend with original
        overlay = (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)
        return overlay
```

---

## 4. 엣지에서의 비디오 처리

### 4.1 프레임 스킵과 시간적 최적화

```python
#!/usr/bin/env python3
"""Efficient video processing with frame skipping and tracking."""

import numpy as np
import time
from typing import List, Optional


class VideoProcessor:
    """Edge video processor with intelligent frame skipping.

    Running detection on every frame at 30 FPS requires ~33ms per frame.
    Most models take 30-100ms. Solutions:
    1. Skip frames: run detection every Nth frame
    2. Track between detections: use lightweight tracking for interim frames
    3. Adaptive skipping: detect more often when scene is changing
    """

    def __init__(self, detector, skip_frames: int = 3,
                 motion_threshold: float = 0.05):
        self.detector = detector
        self.skip_frames = skip_frames
        self.motion_threshold = motion_threshold

        self.frame_count = 0
        self.last_detections = []
        self.prev_frame = None

    def process_frame(self, frame: np.ndarray) -> List[dict]:
        """Process a video frame with adaptive detection/tracking."""
        self.frame_count += 1

        # Check for significant motion
        motion = self._compute_motion(frame)

        # Decide: detect or reuse
        should_detect = (
            self.frame_count % self.skip_frames == 0  # Regular interval
            or motion > self.motion_threshold           # Scene change
            or len(self.last_detections) == 0           # No previous detections
        )

        if should_detect:
            self.last_detections = self.detector.detect(frame)
        else:
            # Optionally update bounding boxes with simple tracking
            self.last_detections = self._simple_track(
                self.last_detections, frame
            )

        self.prev_frame = frame.copy()
        return self.last_detections

    def _compute_motion(self, frame: np.ndarray) -> float:
        """Estimate motion between consecutive frames.

        Uses mean absolute difference of downsampled grayscale frames.
        Cheap to compute (~0.5ms) and sufficient for skip decisions.
        """
        if self.prev_frame is None:
            return 1.0

        # Downsample to 80x60 for speed
        small_curr = frame[::8, ::8, 0].astype(np.float32)
        small_prev = self.prev_frame[::8, ::8, 0].astype(np.float32)

        diff = np.abs(small_curr - small_prev).mean() / 255.0
        return diff

    def _simple_track(self, detections: List[dict],
                      frame: np.ndarray) -> List[dict]:
        """Simple centroid tracking (placeholder for full tracker).

        For production, consider:
        - SORT (Simple Online Realtime Tracking) - Kalman + Hungarian
        - ByteTrack - handles low-confidence detections
        - OC-SORT - observation-centric SORT (robust to occlusion)
        """
        # In a real system, predict new bbox positions from motion model
        return detections  # Reuse previous detections as approximation


class AdaptiveResolutionProcessor:
    """Dynamically adjust inference resolution based on scene complexity."""

    def __init__(self, detector, resolutions: list = None):
        self.detector = detector
        self.resolutions = resolutions or [
            (160, 160),  # Fast mode
            (320, 320),  # Balanced
            (640, 640),  # High quality
        ]
        self.current_res_idx = 1  # Start at balanced

    def process(self, frame: np.ndarray) -> List[dict]:
        """Process frame with adaptive resolution.

        Scale up resolution when many objects are detected.
        Scale down when scene is simple (few/no objects).
        """
        res = self.resolutions[self.current_res_idx]

        from PIL import Image
        resized = np.array(
            Image.fromarray(frame).resize(res)
        )

        detections = self.detector.detect(resized)

        # Adapt resolution for next frame
        if len(detections) > 10 and self.current_res_idx < len(self.resolutions) - 1:
            self.current_res_idx += 1  # More objects -> higher resolution
        elif len(detections) < 2 and self.current_res_idx > 0:
            self.current_res_idx -= 1  # Few objects -> lower resolution

        return detections
```

---

## 5. 다중 모델 파이프라인

### 5.1 파이프라인 아키텍처

```
+-----------------------------------------------------------------+
|          Multi-Model Vision Pipeline                             |
+-----------------------------------------------------------------+
|                                                                   |
|   Input Frame                                                    |
|       |                                                          |
|       v                                                          |
|   +------------------+                                           |
|   | Object Detector  |  Stage 1: Find regions of interest       |
|   | (SSD-MobileNet)  |  ~15ms                                   |
|   +--------+---------+                                           |
|            |                                                     |
|            v  [crop ROIs]                                        |
|   +--------+---------+                                           |
|   | Classifier       |  Stage 2: Fine-grained classification    |
|   | (MobileNetV3)    |  ~8ms per ROI                            |
|   +--------+---------+                                           |
|            |                                                     |
|            v  [matched IDs]                                      |
|   +--------+---------+                                           |
|   | Tracker          |  Stage 3: Temporal association            |
|   | (ByteTrack)      |  ~2ms                                    |
|   +--------+---------+                                           |
|            |                                                     |
|            v                                                     |
|   Final Output: tracked objects with fine-grained labels         |
|                                                                   |
+-----------------------------------------------------------------+
```

### 5.2 다중 모델 파이프라인 구현

```python
#!/usr/bin/env python3
"""Multi-model vision pipeline: detect -> classify -> track."""

import numpy as np
import time
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class TrackedObject:
    track_id: int
    class_label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    fine_class: str = ""
    age: int = 0  # Frames since last detection


class VisionPipeline:
    """Multi-stage vision pipeline for edge deployment.

    Design principles:
    1. Run the detector at low resolution to find ROIs
    2. Crop and classify ROIs only (avoids wasted compute)
    3. Track objects across frames to maintain identity
    4. Skip re-classification for tracked objects (cache class)
    """

    def __init__(self, detector, classifier, max_tracks: int = 50):
        self.detector = detector
        self.classifier = classifier
        self.max_tracks = max_tracks

        self.tracks = {}
        self.next_track_id = 0
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> List[TrackedObject]:
        """Process one video frame through the full pipeline."""
        self.frame_count += 1

        # Stage 1: Detect objects
        detections = self.detector.detect(frame)

        # Stage 2: Classify each detected ROI
        for det in detections:
            bbox = det["bbox"]
            roi = frame[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]

            if roi.size > 0:
                fine_class = self.classifier.classify(roi, top_k=1)
                det["fine_class"] = fine_class["predictions"][0]["label"]
            else:
                det["fine_class"] = det.get("label", "unknown")

        # Stage 3: Track (simple IoU matching)
        tracked = self._update_tracks(detections)

        return tracked

    def _update_tracks(self, detections: List[dict]) -> List[TrackedObject]:
        """Simple IoU-based tracker."""
        # Match detections to existing tracks
        matched = set()
        for det in detections:
            best_iou = 0
            best_track_id = None

            bbox = det["bbox"]
            det_box = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])

            for track_id, track in self.tracks.items():
                iou = self._compute_iou(det_box, track.bbox)
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                track = self.tracks[best_track_id]
                track.bbox = det_box
                track.confidence = det["score"]
                track.fine_class = det.get("fine_class", track.fine_class)
                track.age = 0
                matched.add(best_track_id)
            else:
                # Create new track
                self.tracks[self.next_track_id] = TrackedObject(
                    track_id=self.next_track_id,
                    class_label=det.get("label", "unknown"),
                    confidence=det["score"],
                    bbox=det_box,
                    fine_class=det.get("fine_class", ""),
                )
                matched.add(self.next_track_id)
                self.next_track_id += 1

        # Age unmatched tracks and remove old ones
        to_remove = []
        for track_id in self.tracks:
            if track_id not in matched:
                self.tracks[track_id].age += 1
                if self.tracks[track_id].age > 30:  # Lost for 30 frames
                    to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

        return list(self.tracks.values())

    @staticmethod
    def _compute_iou(box_a: tuple, box_b: tuple) -> float:
        """Compute Intersection over Union."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0
```

---

## 6. 비전 모델 벤치마킹

### 6.1 종합 비전 벤치마크

```python
#!/usr/bin/env python3
"""Benchmark and compare vision models for edge deployment."""

import numpy as np
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class VisionBenchmark:
    model_name: str
    task: str  # "classification", "detection", "segmentation"
    input_resolution: str
    model_size_mb: float
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_fps: float
    num_threads: int
    device: str  # "cpu", "gpu", "edgetpu"


def benchmark_vision_model(model_path: str,
                           task: str,
                           input_shape: tuple = (1, 3, 640, 640),
                           num_threads: int = 4,
                           num_runs: int = 200) -> VisionBenchmark:
    """Benchmark a vision model."""
    import os

    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

    model_size = os.path.getsize(model_path) / (1024 * 1024)

    interp = Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()

    inp = interp.get_input_details()
    out = interp.get_output_details()
    actual_shape = inp[0]["shape"]

    dummy = np.random.randint(0, 255, size=actual_shape).astype(inp[0]["dtype"])

    # Warmup
    for _ in range(10):
        interp.set_tensor(inp[0]["index"], dummy)
        interp.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interp.set_tensor(inp[0]["index"], dummy)
        interp.invoke()
        interp.get_tensor(out[0]["index"])
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    res = f"{actual_shape[1]}x{actual_shape[2]}"

    return VisionBenchmark(
        model_name=Path(model_path).stem,
        task=task,
        input_resolution=res,
        model_size_mb=round(model_size, 2),
        mean_latency_ms=round(float(np.mean(times)), 2),
        p95_latency_ms=round(float(np.percentile(times, 95)), 2),
        throughput_fps=round(1000.0 / float(np.mean(times)), 1),
        num_threads=num_threads,
        device="cpu",
    )


def print_comparison(benchmarks: list):
    """Print a formatted comparison table."""
    print(f"\n{'Model':<25} {'Task':<15} {'Size(MB)':>10} "
          f"{'Mean(ms)':>10} {'P95(ms)':>10} {'FPS':>8}")
    print("-" * 85)
    for b in benchmarks:
        print(f"{b.model_name:<25} {b.task:<15} {b.model_size_mb:>10.1f} "
              f"{b.mean_latency_ms:>10.2f} {b.p95_latency_ms:>10.2f} "
              f"{b.throughput_fps:>8.1f}")
```

---

## 연습 문제

### 연습 1: 객체 탐지
1. YOLOv8-nano를 TFLite(INT8 양자화)로 내보내고 Raspberry Pi에 배포하십시오
2. 입력 해상도 320x320, 480x480, 640x640에서 FPS를 측정하십시오
3. 동일한 하드웨어에서 SSD-MobileNetV2와 비교하십시오

### 연습 2: 비디오 파이프라인
1. 프레임 스킵을 사용하는 `VideoProcessor`로 비디오 처리 파이프라인을 구축하십시오
2. 스킵 값 1, 3, 5, 10으로 테스트하십시오
3. 유효 FPS와 탐지 품질(누락된 탐지)을 측정하십시오

### 연습 3: 다중 모델 파이프라인
1. 탐지 후 분류 파이프라인을 구현하십시오: 사람을 탐지한 후 행동을 분류
2. 각 단계를 프로파일링하여 병목 지점을 식별하십시오
3. 추적된 객체의 분류를 캐싱하여 최적화하십시오

---

**이전**: [실시간 추론](./13_Real_Time_Inference.md) | **다음**: [NLP를 위한 Edge AI](./15_Edge_AI_for_NLP.md)
