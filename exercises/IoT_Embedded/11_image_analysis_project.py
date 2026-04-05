"""
Exercises for Lesson 11: Image Analysis Project
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates Pi Camera capture, MJPEG streaming, TFLite object detection,
and motion detection with MQTT alerts using Python data structures.

On a real Raspberry Pi:
    pip install picamera2      # Pi Camera control
    pip install opencv-python  # Image processing
    pip install tflite-runtime # TFLite inference
    pip install paho-mqtt      # MQTT alerts
    pip install flask          # Streaming server
"""

import time
import json
import random
import hashlib
from datetime import datetime, timedelta
from collections import deque


# ---------------------------------------------------------------------------
# Simulated Camera and Image Pipeline
# ---------------------------------------------------------------------------

class SimulatedPiCamera:
    """Simulate Raspberry Pi Camera Module for capture exercises.

    Real picamera2 usage:
        from picamera2 import Picamera2
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"})
        camera.configure(config)
        camera.start()
        frame = camera.capture_array()
    """

    def __init__(self, resolution=(640, 480), fps=30):
        self.resolution = resolution
        self.fps = fps
        self._running = False
        self._frame_count = 0

    def start(self):
        self._running = True
        print(f"    Camera started: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")

    def stop(self):
        self._running = False
        print(f"    Camera stopped after {self._frame_count} frames")

    def capture_array(self):
        """Simulate capturing a single frame.

        Returns a dict representing frame metadata (real code returns ndarray).
        """
        if not self._running:
            raise RuntimeError("Camera not started")
        self._frame_count += 1

        # Simulate frame with random pixel statistics
        mean_intensity = random.uniform(80, 180)
        return {
            "frame_id": self._frame_count,
            "width": self.resolution[0],
            "height": self.resolution[1],
            "mean_intensity": round(mean_intensity, 1),
            "timestamp": datetime.now().isoformat(),
            "size_bytes": self.resolution[0] * self.resolution[1] * 3,
        }

    def capture_still(self, filename):
        """Simulate saving a still image to file."""
        frame = self.capture_array()
        print(f"    Saved still image: {filename} "
              f"({frame['size_bytes'] / 1024:.0f} KB)")
        return frame


class SimulatedObjectDetector:
    """Simulate TFLite-based object detection on camera frames.

    Real TFLite detection:
        interpreter = tflite.Interpreter(model_path='detect.tflite')
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_index, preprocessed_frame)
        interpreter.invoke()
        boxes = interpreter.get_tensor(boxes_index)
        classes = interpreter.get_tensor(classes_index)
        scores = interpreter.get_tensor(scores_index)
    """

    CLASSES = ["person", "car", "cat", "dog", "bicycle", "chair",
               "bottle", "laptop", "phone", "book"]

    def __init__(self, model_path="detect.tflite", confidence_threshold=0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._inference_count = 0

    def detect(self, frame):
        """Run object detection on a frame (simulated).

        Returns list of detections with bounding boxes and confidence scores.
        """
        self._inference_count += 1

        # Simulate 0-3 detections per frame
        num_detections = random.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0]
        detections = []

        for _ in range(num_detections):
            cls = random.choice(self.CLASSES)
            confidence = random.uniform(self.confidence_threshold, 0.98)

            # Random bounding box (x, y, w, h) as fractions of image size
            x = random.uniform(0.05, 0.6)
            y = random.uniform(0.05, 0.6)
            w = random.uniform(0.1, 0.4)
            h = random.uniform(0.1, 0.4)

            detections.append({
                "class": cls,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "width": round(w, 3),
                    "height": round(h, 3),
                },
            })

        # Simulate inference latency (ms)
        latency_ms = random.uniform(25, 80)

        return {
            "detections": detections,
            "num_objects": len(detections),
            "latency_ms": round(latency_ms, 1),
            "frame_id": frame.get("frame_id", 0),
        }


class MotionDetector:
    """Frame-differencing motion detection system.

    Compares consecutive frames to detect significant pixel changes.
    When motion exceeds the threshold, triggers an alert.

    Real implementation uses cv2.absdiff on grayscale frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_area = cv2.countNonZero(thresh) / total_pixels
    """

    def __init__(self, threshold=0.05, cooldown_sec=3):
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self._prev_intensity = None
        self._last_alert_time = None
        self.alerts = []

    def process_frame(self, frame):
        """Process a frame and return motion detection result."""
        intensity = frame["mean_intensity"]

        if self._prev_intensity is None:
            self._prev_intensity = intensity
            return {"motion": False, "delta": 0.0}

        # Simulate motion as normalized intensity change
        delta = abs(intensity - self._prev_intensity) / 255.0
        motion_detected = delta > self.threshold

        # Cooldown to avoid duplicate alerts
        now = datetime.now()
        in_cooldown = (self._last_alert_time is not None and
                       (now - self._last_alert_time).total_seconds() < self.cooldown_sec)

        if motion_detected and not in_cooldown:
            alert = {
                "type": "MOTION_DETECTED",
                "frame_id": frame["frame_id"],
                "delta": round(delta, 4),
                "threshold": self.threshold,
                "timestamp": frame["timestamp"],
            }
            self.alerts.append(alert)
            self._last_alert_time = now

        self._prev_intensity = intensity
        return {
            "motion": motion_detected,
            "delta": round(delta, 4),
            "alert_triggered": motion_detected and not in_cooldown,
        }


# ---------------------------------------------------------------------------
# Exercise Solutions
# ---------------------------------------------------------------------------

# === Exercise 1: Camera Capture and Streaming ===
# Problem: Set up Pi Camera, capture stills, and implement MJPEG streaming.

def exercise_1():
    """Solution: Camera capture with still images and MJPEG streaming."""

    print("  Camera Capture and Streaming\n")

    camera = SimulatedPiCamera(resolution=(640, 480), fps=30)
    camera.start()

    # Part 1: Capture still images
    print("    --- Part 1: Still Image Capture ---\n")

    for i in range(3):
        frame = camera.capture_still(f"capture_{i:03d}.jpg")
        print(f"      Frame #{frame['frame_id']}: "
              f"mean_intensity={frame['mean_intensity']}")

    # Part 2: Simulated video stream
    print("\n    --- Part 2: Video Stream (MJPEG) ---\n")

    stream_frames = []
    stream_start = time.time()
    for _ in range(10):
        frame = camera.capture_array()
        stream_frames.append(frame)

    elapsed = 0.33  # Simulated time for 10 frames
    actual_fps = len(stream_frames) / elapsed

    print(f"    Streamed {len(stream_frames)} frames")
    print(f"    Effective FPS: {actual_fps:.1f}")
    print(f"    Frame size: {stream_frames[0]['size_bytes'] / 1024:.0f} KB (raw)")
    print(f"    Bandwidth (raw): "
          f"{stream_frames[0]['size_bytes'] * actual_fps / 1024 / 1024:.1f} MB/s")

    # Part 3: MJPEG streaming server reference
    print("""
    --- Reference: Flask MJPEG Server ---

    from flask import Flask, Response
    from picamera2 import Picamera2
    import cv2

    app = Flask(__name__)
    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(
        main={"size": (640, 480)}))
    camera.start()

    def generate_frames():
        while True:
            frame = camera.capture_array()
            _, jpeg = cv2.imencode('.jpg', frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\\r\\n'
                   b'Content-Type: image/jpeg\\r\\n\\r\\n'
                   + jpeg.tobytes() + b'\\r\\n')

    @app.route('/stream')
    def stream():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    """)

    camera.stop()


# === Exercise 2: Object Detection Pipeline ===
# Problem: Run TFLite object detection on camera frames in real time.
# Log detections and compute performance metrics.

def exercise_2():
    """Solution: Real-time object detection with TFLite on Pi Camera."""

    print("  Object Detection Pipeline\n")

    camera = SimulatedPiCamera(resolution=(320, 320), fps=15)
    camera.start()
    detector = SimulatedObjectDetector(confidence_threshold=0.5)

    # Part 1: Single-frame detection
    print("    --- Part 1: Single Frame Detection ---\n")

    frame = camera.capture_array()
    result = detector.detect(frame)

    print(f"    Frame #{frame['frame_id']}: {result['num_objects']} objects "
          f"({result['latency_ms']:.1f}ms)")
    for det in result["detections"]:
        bbox = det["bbox"]
        print(f"      [{det['class']}] confidence={det['confidence']:.1%} "
              f"bbox=({bbox['x']:.2f}, {bbox['y']:.2f}, "
              f"{bbox['width']:.2f}, {bbox['height']:.2f})")

    # Part 2: Continuous detection benchmark
    print("\n    --- Part 2: Detection Benchmark (20 frames) ---\n")

    total_detections = 0
    latencies = []
    class_counts = {}

    for _ in range(20):
        frame = camera.capture_array()
        result = detector.detect(frame)
        latencies.append(result["latency_ms"])
        total_detections += result["num_objects"]

        for det in result["detections"]:
            cls = det["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

    avg_latency = sum(latencies) / len(latencies)
    avg_fps = 1000.0 / avg_latency

    print(f"    Frames processed: {len(latencies)}")
    print(f"    Total detections: {total_detections}")
    print(f"    Avg latency: {avg_latency:.1f}ms (FPS: {avg_fps:.1f})")
    print(f"    Min/Max latency: {min(latencies):.1f}ms / {max(latencies):.1f}ms")

    if class_counts:
        print(f"\n    Detection breakdown:")
        for cls, count in sorted(class_counts.items(),
                                  key=lambda x: x[1], reverse=True):
            print(f"      {cls:<12}: {count}")

    camera.stop()


# === Exercise 3: Motion Detection with Alerts ===
# Problem: Implement frame-differencing motion detection.
# Send MQTT alerts and trigger video recording on motion events.

def exercise_3():
    """Solution: Motion detection with MQTT alerting and recording trigger."""

    print("  Motion Detection System\n")

    camera = SimulatedPiCamera(resolution=(640, 480), fps=15)
    camera.start()
    motion = MotionDetector(threshold=0.08, cooldown_sec=2)

    # Simulated MQTT publisher
    mqtt_alerts = []

    def publish_alert(alert):
        mqtt_alerts.append(alert)

    # Part 1: Process frames and detect motion
    print("    --- Part 1: Motion Detection ---\n")

    recording = False
    recording_frames = []

    print(f"    {'Frame':>6} {'Intensity':>10} {'Delta':>8} {'Motion':>8} {'Action'}")
    print(f"    {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*20}")

    for i in range(15):
        frame = camera.capture_array()

        # Inject a motion event around frame 6-8 by manipulating intensity
        if 6 <= i <= 8:
            frame["mean_intensity"] += random.uniform(30, 60)

        result = motion.process_frame(frame)
        action = ""

        if result.get("alert_triggered"):
            alert = {
                "type": "MOTION",
                "frame_id": frame["frame_id"],
                "delta": result["delta"],
                "timestamp": frame["timestamp"],
            }
            publish_alert(alert)
            action = "ALERT + REC START"
            recording = True
            recording_frames = []
        elif result["motion"] and recording:
            action = "RECORDING..."
        elif not result["motion"] and recording:
            action = "REC STOP"
            recording = False

        if recording:
            recording_frames.append(frame["frame_id"])

        motion_str = "YES" if result["motion"] else "no"
        print(f"    {frame['frame_id']:>6} {frame['mean_intensity']:>9.1f} "
              f"{result['delta']:>7.4f} {motion_str:>8} {action}")

    # Part 2: Alert summary
    print(f"\n    --- Part 2: Alert Summary ---\n")
    print(f"    Motion alerts triggered: {len(mqtt_alerts)}")
    print(f"    Motion detector alerts:  {len(motion.alerts)}")

    for alert in mqtt_alerts:
        print(f"      [{alert['type']}] frame={alert['frame_id']} "
              f"delta={alert['delta']:.4f}")

    if recording_frames:
        print(f"\n    Recorded frames: {recording_frames}")

    # Part 3: MQTT alert reference
    print("""
    --- Reference: MQTT Motion Alert ---

    import paho.mqtt.client as mqtt
    import json

    client = mqtt.Client()
    client.connect("localhost", 1883)

    def on_motion(frame_id, delta):
        payload = {
            "event": "motion_detected",
            "camera_id": "pi_cam_front",
            "frame_id": frame_id,
            "delta": delta,
            "timestamp": datetime.now().isoformat(),
        }
        client.publish(
            "security/pi_cam_front/motion",
            json.dumps(payload),
            qos=1,
        )

    # Triggered video recording with picamera2:
    # camera.start_recording(encoder, output)
    # time.sleep(recording_duration)
    # camera.stop_recording()
    """)

    camera.stop()


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 11: Image Analysis Project - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Exercise 1: Camera Capture and Streaming")
    print("-" * 50)
    exercise_1()

    print("\n\n>>> Exercise 2: Object Detection Pipeline")
    print("-" * 50)
    exercise_2()

    print("\n\n>>> Exercise 3: Motion Detection with Alerts")
    print("-" * 50)
    exercise_3()
