#!/usr/bin/env python3
"""
Image Analysis Project - Simulation Mode
Image capture, object detection, and motion detection using Pi Camera

Runs in simulation mode without actual hardware
"""

import time
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import io
import random

# numpy is optional in simulation mode
# Why: Making numpy optional with a fallback to plain lists means this script
# can demonstrate the full image-analysis pipeline on minimal environments
# (e.g., a fresh Raspberry Pi OS Lite with no pip packages installed yet).
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[Warning] numpy is not installed. Running in simulation mode.")
    print("         Install with 'pip install numpy' for actual use\n")


# ==============================================================================
# Data Classes
# ==============================================================================

class DetectionClass(Enum):
    """Detectable object classes"""
    PERSON = "person"
    CAR = "car"
    DOG = "dog"
    CAT = "cat"
    BICYCLE = "bicycle"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Detection:
    """Object detection result"""
    class_name: str
    confidence: float
    bbox: BoundingBox
    timestamp: datetime


@dataclass
class MotionEvent:
    """Motion detection event"""
    regions: List[BoundingBox]
    area: int
    timestamp: datetime
    frame_id: int


# ==============================================================================
# Simulated Camera
# ==============================================================================

class SimulatedCamera:
    """Simulated Camera (replaces actual Pi Camera)"""

    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self.resolution = resolution
        self.is_running = False
        self.frame_count = 0
        print(f"[Simulation] Camera initialized: {resolution[0]}x{resolution[1]}")

    def start(self):
        """Start camera"""
        self.is_running = True
        print("[Simulation] Camera started")

    def stop(self):
        """Stop camera"""
        self.is_running = False
        print("[Simulation] Camera stopped")

    def capture_frame(self):
        """Capture frame (simulation)"""
        if not self.is_running:
            raise RuntimeError("Camera not started")

        # Generate frame with random noise
        width, height = self.resolution

        if HAS_NUMPY:
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

            # Add a simple pattern (moving rectangle)
            x_offset = (self.frame_count * 5) % width
            y_offset = 100
            frame[y_offset:y_offset+50, x_offset:x_offset+50] = [255, 0, 0]
        else:
            # Simple list-based simulation when numpy is not available
            frame = [[[random.randint(0, 255) for _ in range(3)]
                      for _ in range(width)]
                     for _ in range(height)]

        self.frame_count += 1
        return frame

    def capture_image(self, filename: str):
        """Save image (simulation)"""
        frame = self.capture_frame()
        print(f"[Simulation] Image saved: {filename} ({frame.shape})")
        # In practice: PIL.Image.fromarray(frame).save(filename)


# ==============================================================================
# TFLite Object Detection (Simulation)
# ==============================================================================

class TFLiteObjectDetector:
    """TFLite Object Detector (Simulation)"""

    COCO_LABELS = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe'
    ]

    def __init__(self, model_path: str = None, threshold: float = 0.5):
        self.model_path = model_path or "simulated_model.tflite"
        self.threshold = threshold
        self.input_size = (300, 300)
        print(f"[Simulation] TFLite model loaded: {self.model_path}")
        print(f"  - Input size: {self.input_size}")
        print(f"  - Threshold: {self.threshold}")

    def preprocess(self, frame):
        """Preprocessing"""
        # Resize simulation
        target_h, target_w = self.input_size

        if HAS_NUMPY:
            # In practice: use cv2.resize
            resized = np.random.randint(0, 256, (target_h, target_w, 3), dtype=np.uint8)
            # Normalization
            normalized = resized.astype(np.float32) / 255.0
            return normalized
        else:
            # Simple processing when numpy is not available
            return frame

    def detect(self, frame) -> List[Detection]:
        """Object detection (simulation)"""
        # Preprocessing
        input_data = self.preprocess(frame)

        # Inference simulation (detect objects with some probability)
        detections = []

        # Randomly detect 0-3 objects
        num_objects = random.choice([0, 0, 0, 1, 1, 2])  # Usually 0, occasionally 1-2

        if HAS_NUMPY:
            h, w = frame.shape[:2]
        else:
            h, w = len(frame), len(frame[0]) if frame else 0

        for _ in range(num_objects):
            # Random class selection
            class_id = random.randint(0, min(len(self.COCO_LABELS), 10) - 1)
            class_name = self.COCO_LABELS[class_id]

            # Random confidence
            confidence = random.uniform(self.threshold, 1.0)

            # Random bounding box
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = x1 + random.randint(50, w // 3)
            y2 = y1 + random.randint(50, h // 3)

            bbox = BoundingBox(
                x1=min(x1, w-1),
                y1=min(y1, h-1),
                x2=min(x2, w-1),
                y2=min(y2, h-1)
            )

            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                timestamp=datetime.now()
            )
            detections.append(detection)

        return detections

    def draw_detections(self, frame, detections: List[Detection]):
        """Visualize detection results (simulation)"""
        if HAS_NUMPY:
            result = frame.copy()
        else:
            result = frame  # No copy in simulation mode

        for det in detections:
            # In practice: use cv2.rectangle, cv2.putText
            label = f"{det.class_name}: {det.confidence:.2f}"
            print(f"  [Detection] {label} at ({det.bbox.x1}, {det.bbox.y1})")

        return result


# ==============================================================================
# Motion Detection
# ==============================================================================

# Why: Frame-differencing is the simplest motion detection algorithm: subtract
# consecutive grayscale frames and threshold the absolute difference. It runs
# in O(pixels) with no model weights, making it ideal for edge devices where
# TFLite inference is reserved for object classification.
class MotionDetector:
    """Motion Detector"""

    def __init__(self, threshold: int = 30, min_area: int = 500):
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
        self.motion_count = 0
        print(f"[Simulation] Motion detector initialized")
        print(f"  - Threshold: {threshold}")
        print(f"  - Min area: {min_area} pixels")

    def detect_motion(self, frame) -> Tuple[bool, List[BoundingBox]]:
        """Detect motion"""
        if HAS_NUMPY:
            # Grayscale conversion simulation
            gray = np.mean(frame, axis=2).astype(np.uint8)

            # Gaussian blur simulation
            # In practice: use cv2.GaussianBlur

            if self.prev_frame is None:
                self.prev_frame = gray
                return False, []

            # Compute frame difference
            frame_delta = np.abs(gray.astype(np.int16) - self.prev_frame.astype(np.int16))

            # Apply threshold
            thresh = (frame_delta > self.threshold).astype(np.uint8) * 255

            # Calculate ratio of changed area
            motion_pixels = np.sum(thresh > 0)
            total_pixels = thresh.size
            motion_ratio = motion_pixels / total_pixels

            self.prev_frame = gray

            # Motion detected if change exceeds threshold ratio
            motion_detected = motion_ratio > 0.05  # More than 5% change

            h, w = frame.shape[:2]
        else:
            # Simple simulation when numpy is not available
            if self.prev_frame is None:
                self.prev_frame = frame
                return False, []

            # Random motion detection (10% probability)
            motion_detected = random.random() < 0.1
            h, w = len(frame), len(frame[0]) if frame else 0

        regions = []
        if motion_detected:
            # Simulation: generate random motion regions
            num_regions = random.randint(1, 3)

            for _ in range(num_regions):
                x1 = random.randint(0, w // 2)
                y1 = random.randint(0, h // 2)
                x2 = x1 + random.randint(50, 150)
                y2 = y1 + random.randint(50, 150)

                bbox = BoundingBox(
                    x1=min(x1, w-1),
                    y1=min(y1, h-1),
                    x2=min(x2, w-1),
                    y2=min(y2, h-1)
                )

                if bbox.area >= self.min_area:
                    regions.append(bbox)

            self.motion_count += 1

        return motion_detected, regions


# ==============================================================================
# Video Streaming (Concept)
# ==============================================================================

class VideoStreamer:
    """Video Streamer (MJPEG over HTTP concept)"""

    def __init__(self, camera: SimulatedCamera, port: int = 8080):
        self.camera = camera
        self.port = port
        self.is_streaming = False
        self.frame_rate = 30
        self.lock = threading.Lock()
        self.current_frame = None
        print(f"[Simulation] Video streamer initialized (port {port})")

    def _capture_loop(self):
        """Capture loop"""
        frame_interval = 1.0 / self.frame_rate

        while self.is_streaming:
            start_time = time.time()

            frame = self.camera.capture_frame()

            # JPEG encoding simulation
            # In practice: PIL.Image.fromarray(frame).save(buffer, 'JPEG')

            with self.lock:
                self.current_frame = frame

            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

    def start_streaming(self):
        """Start streaming"""
        if self.is_streaming:
            return

        self.is_streaming = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[Simulation] Streaming started: http://0.0.0.0:{self.port}/video_feed")

    def stop_streaming(self):
        """Stop streaming"""
        self.is_streaming = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        print("[Simulation] Streaming stopped")

    def get_frame(self):
        """Return current frame"""
        with self.lock:
            return self.current_frame


# ==============================================================================
# Result Logging
# ==============================================================================

# Why: JSONL (one JSON object per line) is chosen over a single JSON array
# because it is append-safe: a crash mid-write never corrupts earlier entries.
# This is the same pattern used in sensor_reading.py's logging.
class ResultLogger:
    """Detection and motion result logger"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        print(f"[Simulation] Logger initialized: {self.log_file}")

    def log_detection(self, detections: List[Detection], frame_id: int):
        """Log object detection"""
        log_entry = {
            "type": "detection",
            "timestamp": datetime.now().isoformat(),
            "frame_id": frame_id,
            "count": len(detections),
            "objects": [
                {
                    "class": det.class_name,
                    "confidence": det.confidence,
                    "bbox": [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
                }
                for det in detections
            ]
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_motion(self, event: MotionEvent):
        """Log motion event"""
        log_entry = {
            "type": "motion",
            "timestamp": event.timestamp.isoformat(),
            "frame_id": event.frame_id,
            "region_count": len(event.regions),
            "total_area": event.area,
            "regions": [
                [r.x1, r.y1, r.x2, r.y2] for r in event.regions
            ]
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def get_statistics(self) -> Dict:
        """Log statistics"""
        if not os.path.exists(self.log_file):
            return {}

        detection_count = 0
        motion_count = 0
        object_classes = {}

        with open(self.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry['type'] == 'detection':
                    detection_count += 1
                    for obj in entry['objects']:
                        cls = obj['class']
                        object_classes[cls] = object_classes.get(cls, 0) + 1
                elif entry['type'] == 'motion':
                    motion_count += 1

        return {
            "detection_events": detection_count,
            "motion_events": motion_count,
            "object_classes": object_classes
        }


# ==============================================================================
# Performance Monitoring
# ==============================================================================

class PerformanceMonitor:
    """Performance Monitor"""

    def __init__(self):
        self.metrics = {
            "fps": [],
            "detection_time": [],
            "frame_count": 0,
            "start_time": None
        }

    def start(self):
        """Start monitoring"""
        self.metrics["start_time"] = time.time()

    def record_frame(self, processing_time: float):
        """Record frame processing time"""
        self.metrics["frame_count"] += 1
        self.metrics["detection_time"].append(processing_time)

        if processing_time > 0:
            fps = 1.0 / processing_time
            self.metrics["fps"].append(fps)

    def get_report(self) -> Dict:
        """Performance report"""
        if self.metrics["start_time"] is None:
            return {}

        elapsed = time.time() - self.metrics["start_time"]

        if HAS_NUMPY:
            avg_fps = np.mean(self.metrics["fps"]) if self.metrics["fps"] else 0
            avg_detection_time = np.mean(self.metrics["detection_time"]) if self.metrics["detection_time"] else 0
        else:
            avg_fps = sum(self.metrics["fps"]) / len(self.metrics["fps"]) if self.metrics["fps"] else 0
            avg_detection_time = sum(self.metrics["detection_time"]) / len(self.metrics["detection_time"]) if self.metrics["detection_time"] else 0

        return {
            "total_frames": self.metrics["frame_count"],
            "elapsed_time": elapsed,
            "average_fps": avg_fps,
            "average_detection_time_ms": avg_detection_time * 1000,
            "frames_per_second_actual": self.metrics["frame_count"] / elapsed if elapsed > 0 else 0
        }


# ==============================================================================
# Integrated Image Analysis System
# ==============================================================================

class ImageAnalysisSystem:
    """Integrated Image Analysis System"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}

        # Camera
        resolution = config.get('resolution', (640, 480))
        self.camera = SimulatedCamera(resolution)

        # Object detection
        self.detector = TFLiteObjectDetector(
            threshold=config.get('detection_threshold', 0.5)
        )

        # Motion detection
        self.motion_detector = MotionDetector(
            threshold=config.get('motion_threshold', 30),
            min_area=config.get('min_motion_area', 500)
        )

        # Logger
        self.logger = ResultLogger(log_dir=config.get('log_dir', 'logs'))

        # Performance monitor
        self.perf_monitor = PerformanceMonitor()

        # Streaming (optional)
        self.enable_streaming = config.get('enable_streaming', False)
        if self.enable_streaming:
            self.streamer = VideoStreamer(self.camera, port=config.get('stream_port', 8080))

        self.is_running = False

        print("\n" + "="*60)
        print("Image analysis system initialization complete")
        print("="*60)

    def run(self, duration: float = 60, detect_objects: bool = True, detect_motion: bool = True):
        """Run the system"""
        print(f"\nSystem started (duration: {duration}s)")
        print(f"  - Object detection: {'ON' if detect_objects else 'OFF'}")
        print(f"  - Motion detection: {'ON' if detect_motion else 'OFF'}")
        print(f"  - Streaming: {'ON' if self.enable_streaming else 'OFF'}")
        print()

        self.camera.start()
        self.perf_monitor.start()

        if self.enable_streaming:
            self.streamer.start_streaming()

        self.is_running = True
        start_time = time.time()
        frame_id = 0

        try:
            while time.time() - start_time < duration and self.is_running:
                frame_start = time.time()

                # Capture frame
                frame = self.camera.capture_frame()
                frame_id += 1

                # Object detection
                if detect_objects:
                    detections = self.detector.detect(frame)
                    if detections:
                        print(f"[Frame {frame_id}] Objects detected: {len(detections)}")
                        for det in detections:
                            print(f"  - {det.class_name} (confidence: {det.confidence:.2f})")
                        self.logger.log_detection(detections, frame_id)

                # Motion detection
                if detect_motion:
                    motion_detected, regions = self.motion_detector.detect_motion(frame)
                    if motion_detected:
                        total_area = sum(r.area for r in regions)
                        event = MotionEvent(
                            regions=regions,
                            area=total_area,
                            timestamp=datetime.now(),
                            frame_id=frame_id
                        )
                        print(f"[Frame {frame_id}] Motion detected: {len(regions)} region(s) (area: {total_area})")
                        self.logger.log_motion(event)

                # Record performance
                processing_time = time.time() - frame_start
                self.perf_monitor.record_frame(processing_time)

                # Why: Subtracting the actual processing time from the target frame
                # interval gives a consistent ~10 FPS regardless of how fast inference
                # runs. This prevents unnecessary CPU spinning on fast hardware.
                sleep_time = max(0, 0.1 - processing_time)  # ~10 FPS
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            self._cleanup()

    def _cleanup(self):
        """Cleanup"""
        self.is_running = False
        self.camera.stop()

        if self.enable_streaming:
            self.streamer.stop_streaming()

        # Print statistics
        print("\n" + "="*60)
        print("Run complete - Statistics")
        print("="*60)

        # Performance statistics
        perf_report = self.perf_monitor.get_report()
        print("\n[Performance]")
        print(f"  Total frames: {perf_report.get('total_frames', 0)}")
        print(f"  Elapsed time: {perf_report.get('elapsed_time', 0):.1f}s")
        print(f"  Average FPS: {perf_report.get('average_fps', 0):.1f}")
        print(f"  Avg detection time: {perf_report.get('average_detection_time_ms', 0):.1f}ms")

        # Log statistics
        log_stats = self.logger.get_statistics()
        print("\n[Detection Statistics]")
        print(f"  Detection events: {log_stats.get('detection_events', 0)}")
        print(f"  Motion events: {log_stats.get('motion_events', 0)}")

        object_classes = log_stats.get('object_classes', {})
        if object_classes:
            print("\n[Detected Objects]")
            for cls, count in sorted(object_classes.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {count}")

        print("\n" + "="*60)


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main function"""
    print("Image Analysis Project - Simulation Mode")
    print("="*60)
    print("This program runs in simulation mode without an actual Pi Camera.")
    print()

    # Configuration
    config = {
        'resolution': (640, 480),
        'detection_threshold': 0.6,
        'motion_threshold': 30,
        'min_motion_area': 500,
        'log_dir': 'logs',
        'enable_streaming': False,  # Do not run actual Flask server
        'stream_port': 8080
    }

    # Create system
    system = ImageAnalysisSystem(config)

    # Run (short duration for testing; increase duration for actual use)
    import sys
    test_mode = '--test' in sys.argv
    duration = 5 if test_mode else 30

    system.run(
        duration=duration,  # Test: 5s, Normal: 30s
        detect_objects=True,
        detect_motion=True
    )


if __name__ == "__main__":
    main()
