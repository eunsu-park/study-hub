# 11. Image Analysis Project

**Previous**: [Home Automation Project](./10_Home_Automation_Project.md) | **Next**: [Cloud IoT Integration](./12_Cloud_IoT_Integration.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up the Pi Camera hardware and configure picamera2 for image capture
2. Capture still images and record video programmatically
3. Implement MJPEG and WebSocket-based real-time video streaming servers
4. Perform real-time object detection using TFLite on camera frames
5. Build a frame-differencing motion detection system with MQTT alerts
6. Trigger video recording automatically when motion is detected

---

Adding a camera to your IoT system opens up an entirely new dimension of sensing. Instead of measuring a single variable like temperature, a camera can detect people, identify objects, read license plates, or monitor an entire room. This project combines the Pi Camera with the Edge AI skills from earlier lessons to build practical vision systems that run entirely on local hardware.

---

## 1. Pi Camera Setup

### 1.1 Hardware Connection

```
┌─────────────────────────────────────────────────────────────┐
│                    Pi Camera Connection                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. Power OFF Raspberry Pi                                 │
│   2. Open CSI port connector (pull gently)                  │
│   3. Insert ribbon cable (blue side faces Ethernet port)    │
│   4. Close connector                                         │
│   5. Power ON                                                │
│                                                              │
│   ┌─────────────────────────┐                               │
│   │      Raspberry Pi       │                               │
│   │   ┌─────────────────┐   │                               │
│   │   │    CSI Port     │   │                               │
│   │   │  [▓▓▓▓▓▓▓▓▓▓]  │←─ Camera ribbon cable            │
│   │   └─────────────────┘   │                               │
│   │                         │                               │
│   └─────────────────────────┘                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Software Setup

```bash
# Enable camera (raspi-config)
sudo raspi-config
# Interface Options > Legacy Camera > Disable
# (picamera2 requires legacy camera disabled)

# Install picamera2
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera

# Additional packages
pip install numpy pillow opencv-python

# Test camera
libcamera-hello --list-cameras
```

### 1.3 Camera Verification

```python
#!/usr/bin/env python3
"""Pi Camera verification"""

from picamera2 import Picamera2
import time

def test_camera():
    """Test camera"""
    picam2 = Picamera2()

    # Camera info
    camera_info = picam2.camera_properties
    print("=== Camera Info ===")
    print(f"Model: {camera_info.get('Model', 'Unknown')}")
    print(f"Pixel size: {camera_info.get('PixelArraySize', 'Unknown')}")

    # Configuration
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)

    # Capture image
    picam2.start()
    time.sleep(2)  # Sensor stabilization

    frame = picam2.capture_array()
    print(f"\nCaptured image: {frame.shape}")

    # Save
    from PIL import Image
    img = Image.fromarray(frame)
    img.save("test_capture.jpg")
    print("Image saved: test_capture.jpg")

    picam2.stop()

if __name__ == "__main__":
    test_camera()
```

---

## 2. picamera2 Library

### 2.1 Basic Usage

```python
#!/usr/bin/env python3
"""picamera2 basic usage"""

from picamera2 import Picamera2
import time

class CameraHandler:
    """Pi Camera handler"""

    def __init__(self, resolution: tuple = (640, 480)):
        self.picam2 = Picamera2()
        self.resolution = resolution

        # Preview configuration
        self.preview_config = self.picam2.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"}
        )

        # Still configuration (high resolution)
        self.still_config = self.picam2.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )

    def start(self, mode: str = "preview"):
        """Start camera"""
        if mode == "preview":
            self.picam2.configure(self.preview_config)
        else:
            self.picam2.configure(self.still_config)

        self.picam2.start()
        time.sleep(0.5)  # Stabilization

    def stop(self):
        """Stop camera"""
        self.picam2.stop()

    def capture_frame(self):
        """Capture frame"""
        return self.picam2.capture_array()

    def capture_image(self, filename: str):
        """Save image file"""
        self.picam2.capture_file(filename)
        print(f"Image saved: {filename}")

    def set_controls(self, **kwargs):
        """Set camera controls"""
        # e.g.: brightness, contrast, exposure_time, analogue_gain
        self.picam2.set_controls(kwargs)

# Usage example
if __name__ == "__main__":
    camera = CameraHandler((640, 480))
    camera.start()

    # Capture frame
    frame = camera.capture_frame()
    print(f"Frame size: {frame.shape}")

    # Save image
    camera.capture_image("snapshot.jpg")

    camera.stop()
```

### 2.2 Video Recording

```python
#!/usr/bin/env python3
"""Video recording"""

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import time

def record_video(filename: str, duration: int = 10):
    """Record video"""
    picam2 = Picamera2()

    # Video configuration
    video_config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(video_config)

    # Encoder and output setup
    encoder = H264Encoder(bitrate=10000000)
    output = FfmpegOutput(filename)

    # Start recording
    picam2.start_recording(encoder, output)
    print(f"Recording started: {filename}")

    time.sleep(duration)

    # Stop recording
    picam2.stop_recording()
    print(f"Recording complete: {duration}s")

if __name__ == "__main__":
    record_video("video.mp4", duration=10)
```

---

## 3. Real-time Video Streaming

### 3.1 MJPEG Streaming Server

```python
#!/usr/bin/env python3
"""MJPEG streaming server"""

from flask import Flask, Response
from picamera2 import Picamera2
import io
from PIL import Image
import threading
import time

app = Flask(__name__)

class StreamingServer:
    """MJPEG streaming server"""

    def __init__(self):
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(self.config)
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        """Start streaming"""
        self.picam2.start()
        self.running = True

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Capture loop"""
        while self.running:
            frame = self.picam2.capture_array()

            # Encode to JPEG
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)
            jpeg_data = buffer.getvalue()

            with self.lock:
                self.frame = jpeg_data

            time.sleep(0.033)  # ~30 FPS

    def get_frame(self):
        """Get current frame"""
        with self.lock:
            return self.frame

    def stop(self):
        """Stop streaming"""
        self.running = False
        self.picam2.stop()

# Global streaming server
streamer = StreamingServer()

def generate_frames():
    """Frame generator"""
    while True:
        frame = streamer.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

@app.route('/')
def index():
    return '''
    <html>
    <head><title>Pi Camera Stream</title></head>
    <body>
        <h1>Pi Camera Real-time Stream</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    streamer.start()
    try:
        app.run(host='0.0.0.0', port=8080, threaded=True)
    finally:
        streamer.stop()
```

### 3.2 WebSocket Streaming

```python
#!/usr/bin/env python3
"""WebSocket video streaming"""

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from picamera2 import Picamera2
import base64
import io
from PIL import Image
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class WebSocketStreamer:
    """WebSocket streamer"""

    def __init__(self):
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(self.config)
        self.running = False

    def start_streaming(self):
        """Start streaming"""
        self.picam2.start()
        self.running = True

        while self.running:
            frame = self.picam2.capture_array()

            # Base64 encoding
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Send to clients
            socketio.emit('frame', {'image': img_str})

            time.sleep(0.05)  # ~20 FPS

    def stop(self):
        self.running = False
        self.picam2.stop()

streamer = WebSocketStreamer()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Stream</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    </head>
    <body>
        <h1>WebSocket Video Stream</h1>
        <canvas id="canvas" width="640" height="480"></canvas>
        <script>
            const socket = io();
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            socket.on('frame', function(data) {
                const img = new Image();
                img.onload = function() {
                    ctx.drawImage(img, 0, 0);
                };
                img.src = 'data:image/jpeg;base64,' + data.image;
            });
        </script>
    </body>
    </html>
    '''

@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == "__main__":
    stream_thread = threading.Thread(target=streamer.start_streaming, daemon=True)
    stream_thread.start()

    try:
        socketio.run(app, host='0.0.0.0', port=8080)
    finally:
        streamer.stop()
```

---

## 4. Object Detection (TFLite)

### 4.1 Real-time Object Detection

```python
#!/usr/bin/env python3
"""Real-time object detection (TFLite + Pi Camera)"""

from picamera2 import Picamera2
import numpy as np
import cv2
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class RealtimeDetector:
    """Real-time object detector"""

    COCO_LABELS = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        # ... (complete COCO labels)
    ]

    def __init__(self, model_path: str, threshold: float = 0.5):
        # Load TFLite model
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        self.threshold = threshold

        # Initialize camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocessing"""
        # Resize
        resized = cv2.resize(frame, (self.input_width, self.input_height))

        # Normalize
        input_data = resized.astype(np.float32) / 255.0

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)

        return input_data

    def detect(self, frame: np.ndarray) -> list:
        """Object detection"""
        input_data = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Parse output (model-dependent)
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        detections = []
        h, w = frame.shape[:2]

        for i in range(len(scores)):
            if scores[i] > self.threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                detections.append({
                    'class_id': int(classes[i]),
                    'class_name': self.COCO_LABELS[int(classes[i])] if int(classes[i]) < len(self.COCO_LABELS) else 'unknown',
                    'score': float(scores[i]),
                    'box': [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)]
                })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Visualize detection results"""
        result = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class_name']}: {det['score']:.2f}"

            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result

    def run(self, duration: float = 60, display: bool = False):
        """Run real-time detection"""
        self.picam2.start()
        print(f"Real-time detection started ({duration}s)")

        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                frame = self.picam2.capture_array()

                # Detection
                detections = self.detect(frame)
                frame_count += 1

                if detections:
                    print(f"[{frame_count}] Detected: {len(detections)}")
                    for det in detections:
                        print(f"  - {det['class_name']}: {det['score']:.2f}")

                if display:
                    result = self.draw_detections(frame, detections)
                    cv2.imshow("Detection", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self.picam2.stop()
            if display:
                cv2.destroyAllWindows()

            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"\nFPS: {fps:.1f}")

if __name__ == "__main__":
    detector = RealtimeDetector("ssd_mobilenet.tflite", threshold=0.5)
    detector.run(duration=30, display=False)
```

### 4.2 Stream Detection Results

```python
#!/usr/bin/env python3
"""Stream object detection results"""

from flask import Flask, Response, jsonify
from picamera2 import Picamera2
import numpy as np
import cv2
import io
from PIL import Image
import threading
import time
import json

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

class DetectionStreamer:
    """Detection result streamer"""

    def __init__(self, model_path: str):
        # Load model
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)

        self.frame = None
        self.detections = []
        self.lock = threading.Lock()

    def start(self):
        self.picam2.start()
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _process_loop(self):
        while self.running:
            frame = self.picam2.capture_array()
            detections = self._detect(frame)

            # Draw results
            result = self._draw(frame, detections)

            # JPEG encoding
            img = Image.fromarray(result)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)

            with self.lock:
                self.frame = buffer.getvalue()
                self.detections = detections

            time.sleep(0.05)

    def _detect(self, frame: np.ndarray) -> list:
        # Simplified detection logic
        input_size = (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2])
        resized = cv2.resize(frame, input_size)
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Parse output (requires model-specific implementation)
        return []  # Actual implementation needed

    def _draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        result = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.get('box', [0, 0, 0, 0])
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return result

    def get_frame(self):
        with self.lock:
            return self.frame

    def get_detections(self):
        with self.lock:
            return self.detections

    def stop(self):
        self.running = False
        self.picam2.stop()

streamer = DetectionStreamer("model.tflite")

def generate():
    while True:
        frame = streamer.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def detections():
    return jsonify(streamer.get_detections())

if __name__ == "__main__":
    streamer.start()
    try:
        app.run(host='0.0.0.0', port=8080)
    finally:
        streamer.stop()
```

---

## 5. Motion Detection

### 5.1 Frame Difference-based Motion Detection

```python
#!/usr/bin/env python3
"""Motion detection system"""

from picamera2 import Picamera2
import numpy as np
import cv2
import time
from datetime import datetime
import paho.mqtt.client as mqtt
import json

class MotionDetector:
    """Motion detector"""

    def __init__(self, threshold: int = 30, min_area: int = 500):
        self.threshold = threshold
        self.min_area = min_area

        # Camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)

        self.prev_frame = None
        self.motion_detected = False

        # MQTT (optional)
        self.mqtt_client = mqtt.Client()
        try:
            self.mqtt_client.connect("localhost", 1883)
            self.mqtt_client.loop_start()
            self.mqtt_enabled = True
        except:
            self.mqtt_enabled = False

    def detect_motion(self, frame: np.ndarray) -> tuple:
        """Detect motion"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, frame, []

        # Calculate frame difference
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # Remove noise
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_regions = []
        result = frame.copy()

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            motion_regions.append({'x': x, 'y': y, 'w': w, 'h': h})
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.prev_frame = gray
        motion_detected = len(motion_regions) > 0

        return motion_detected, result, motion_regions

    def on_motion(self, regions: list):
        """Motion detected event"""
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] Motion detected! Regions: {len(regions)}")

        # Publish via MQTT
        if self.mqtt_enabled:
            data = {
                "event": "motion_detected",
                "regions": regions,
                "timestamp": timestamp
            }
            self.mqtt_client.publish("camera/motion", json.dumps(data))

        # Save snapshot
        # self.save_snapshot(f"motion_{timestamp}.jpg")

    def run(self, duration: float = 60, cooldown: float = 5):
        """Run motion detection"""
        self.picam2.start()
        print(f"Motion detection started ({duration}s)")

        start_time = time.time()
        last_motion_time = 0

        try:
            while time.time() - start_time < duration:
                frame = self.picam2.capture_array()

                detected, result, regions = self.detect_motion(frame)

                if detected:
                    current_time = time.time()
                    if current_time - last_motion_time > cooldown:
                        self.on_motion(regions)
                        last_motion_time = current_time

                time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            self.picam2.stop()
            if self.mqtt_enabled:
                self.mqtt_client.loop_stop()
            print("Motion detection stopped")

if __name__ == "__main__":
    detector = MotionDetector(threshold=30, min_area=500)
    detector.run(duration=120, cooldown=5)
```

### 5.2 Motion Detection + Recording

```python
#!/usr/bin/env python3
"""Record on motion detection"""

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import numpy as np
import cv2
import time
from datetime import datetime
import threading
import os

class MotionRecorder:
    """Motion detection recorder"""

    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.picam2 = Picamera2()
        self.video_config = self.picam2.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        self.picam2.configure(self.video_config)

        self.encoder = H264Encoder()
        self.is_recording = False
        self.prev_frame = None
        self.threshold = 30
        self.min_area = 500

    def detect_motion(self, frame: np.ndarray) -> bool:
        """Detect motion"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.prev_frame = gray

        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                return True

        return False

    def start_recording(self):
        """Start recording"""
        if self.is_recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"motion_{timestamp}.h264")

        output = FileOutput(filename)
        self.picam2.start_recording(self.encoder, output)
        self.is_recording = True
        self.recording_start = time.time()
        print(f"Recording started: {filename}")

    def stop_recording(self):
        """Stop recording"""
        if not self.is_recording:
            return

        self.picam2.stop_recording()
        self.is_recording = False
        duration = time.time() - self.recording_start
        print(f"Recording complete: {duration:.1f}s")

    def run(self, pre_record: float = 2, post_record: float = 5):
        """Run motion detection recording"""
        self.picam2.start()
        print("Motion detection recording system started")

        last_motion_time = 0

        try:
            while True:
                frame = self.picam2.capture_array()
                motion = self.detect_motion(frame)

                if motion:
                    last_motion_time = time.time()
                    if not self.is_recording:
                        self.start_recording()

                # Stop recording after post_record if no motion
                if self.is_recording:
                    if time.time() - last_motion_time > post_record:
                        self.stop_recording()

                time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            if self.is_recording:
                self.stop_recording()
            self.picam2.stop()

if __name__ == "__main__":
    recorder = MotionRecorder("recordings")
    recorder.run(post_record=5)
```

---

## Exercises

### Exercise 1: Timelapse
1. Implement a timelapse system that captures images at regular intervals.
2. Convert captured images to video.

### Exercise 2: Face Detection
1. Implement face detection using OpenCV or TFLite.
2. Send notifications when faces are detected.

### Exercise 3: Cloud Integration
1. Upload images to cloud on motion detection.
2. Save detection results to database.

---

**Previous**: [Home Automation Project](./10_Home_Automation_Project.md) | **Next**: [Cloud IoT Integration](./12_Cloud_IoT_Integration.md)
