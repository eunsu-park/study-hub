# Face Detection and Recognition

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement face and eye detection using OpenCV's Haar Cascade classifier
2. Apply dlib's HOG-based face detector and extract 68-point facial landmarks
3. Build a face recognition system using LBPH (Local Binary Patterns Histograms) with OpenCV
4. Use the face_recognition library to encode, compare, and identify faces from images
5. Compare the accuracy and performance trade-offs among Haar Cascade, dlib HOG, and deep-learning-based face detectors
6. Design a real-time face detection and recognition pipeline using a webcam feed

---

## Overview

Face detection and recognition is one of the most practical applications of computer vision. We will learn various face processing techniques using Haar Cascade, dlib, and the face_recognition library.

Face detection is the essential first step that unlocks a broad class of downstream tasks: face recognition, emotion analysis, gaze estimation, attendance systems, and portrait photography automation. Getting detection right — with accurate bounding boxes and few false positives — directly determines the quality of everything built on top of it.

**Difficulty**: ****

**Prerequisites**: Object detection basics, feature detection, image transformations

---

## Table of Contents

1. [Haar Cascade Face/Eye Detection](#1-haar-cascade-faceeye-detection)
2. [dlib Face Detector (HOG-based)](#2-dlib-face-detector-hog-based)
3. [dlib Face Landmarks (68 Points)](#3-dlib-face-landmarks-68-points)
4. [LBPH Face Recognition](#4-lbph-face-recognition)
5. [face_recognition Library](#5-face_recognition-library)
6. [Real-time Face Detection](#6-real-time-face-detection)
7. [Practice Problems](#7-practice-problems)

---

## 1. Haar Cascade Face/Eye Detection

### Face Detection Principle

The integral image (summed-area table) is what makes Haar Cascade practical. Without it, computing the sum of pixel values in any rectangle takes O(w×h) operations. With the integral image, any rectangular sum reduces to exactly 4 array lookups regardless of rectangle size — making it O(1). This constant-time property means thousands of Haar features can be evaluated at each window position fast enough for real-time video.

```
Haar Cascade Face Detection Process:

1. Compute integral image
   +-----------------+
   | Original image  | -> Integral image (fast feature computation)
   +-----------------+

2. Scan with windows of various sizes
   +-----------------------------+
   |  +--+                       |
   |  |  |  -> Small window      |
   |  +--+                       |
   |     +-----+                 |
   |     |     | -> Medium window|
   |     +-----+                 |
   |        +--------+           |
   |        |        | -> Large  |
   |        +--------+           |
   +-----------------------------+

3. Apply Cascade classifier at each window

   Window -> Stage 1 -> Stage 2 -> ... -> Stage N
            (face?)   (face?)         (face!)

4. Group detection results (remove duplicates)
```

### Basic Face Detection

```python
import cv2
import numpy as np

def detect_faces_haar(img, scale_factor=1.1, min_neighbors=5,
                      min_size=(30, 30)):
    """Face detection using Haar Cascade"""

    # Load Cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Grayscale conversion
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Histogram equalization spreads pixel intensities across the full range,
    # reducing the effect of uneven lighting that would otherwise cause the
    # Haar feature responses to vary with ambient brightness
    gray = cv2.equalizeHist(gray)

    # Face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,   # 1.1 = 10% size reduction per pyramid level;
                                    # smaller values (1.05) catch more scales but are slower
        minNeighbors=min_neighbors, # Requires this many overlapping detections before
                                    # accepting a hit — the primary knob to trade recall vs precision
        minSize=min_size,           # Skip windows smaller than this; set based on the
                                    # expected minimum face size in the scene
        flags=cv2.CASCADE_SCALE_IMAGE  # Scales the image rather than the detector window;
                                       # more numerically stable across the pyramid
    )

    return faces

# Usage example
img = cv2.imread('photo.jpg')
faces = detect_faces_haar(img)

# Draw results
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Face center point
    center = (x + w//2, y + h//2)
    cv2.circle(img, center, 3, (0, 0, 255), -1)

print(f"Faces detected: {len(faces)}")
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
```

### Face and Eye Detection Together

```python
import cv2

class HaarFaceEyeDetector:
    """Haar Cascade-based face/eye detector"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.eye_glasses_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )

    def detect(self, img, detect_eyes=True):
        """Detect face and eyes"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        results = []

        # Face detection
        faces = self.face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_data = {
                'face_rect': (x, y, w, h),
                'eyes': []
            }

            if detect_eyes:
                # Restricting to the top 50% of the face ROI is critical: the eye
                # cascade produces many false positives in the mouth/chin region
                # (nostrils, teeth gaps look like "eyes" to a trained Haar cascade)
                roi_gray = gray[y:y+h//2, x:x+w]

                # Try regular eye detection
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, 1.1, 3, minSize=(20, 20)
                )

                # Fall back to the glasses-aware cascade when fewer than 2 eyes
                # are found — glasses change the local contrast pattern that the
                # standard eye cascade was trained on, causing missed detections
                if len(eyes) < 2:
                    eyes = self.eye_glasses_cascade.detectMultiScale(
                        roi_gray, 1.1, 3, minSize=(20, 20)
                    )

                # Select best two eyes
                eyes = self._select_best_eyes(eyes, w)

                for (ex, ey, ew, eh) in eyes:
                    face_data['eyes'].append((x + ex, y + ey, ew, eh))

            results.append(face_data)

        return results

    def _select_best_eyes(self, eyes, face_width):
        """Select best two eyes"""
        if len(eyes) <= 2:
            return eyes

        # Filter by eye size and y-coordinate
        eyes = sorted(eyes, key=lambda e: e[1])  # Sort by y-coordinate

        # Select from top 4 candidates
        candidates = eyes[:4]

        # Separate left/right by x-coordinate
        mid_x = face_width // 2
        left_eyes = [e for e in candidates if e[0] + e[2]//2 < mid_x]
        right_eyes = [e for e in candidates if e[0] + e[2]//2 >= mid_x]

        result = []
        if left_eyes:
            result.append(left_eyes[0])
        if right_eyes:
            result.append(right_eyes[0])

        return result

    def draw_results(self, img, results):
        """Visualize results"""
        output = img.copy()

        for face_data in results:
            x, y, w, h = face_data['face_rect']

            # Face rectangle
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Eyes
            for (ex, ey, ew, eh) in face_data['eyes']:
                center = (ex + ew//2, ey + eh//2)
                radius = (ew + eh) // 4
                cv2.circle(output, center, radius, (255, 0, 0), 2)

        return output

# Usage example
detector = HaarFaceEyeDetector()
img = cv2.imread('portrait.jpg')
results = detector.detect(img)
output = detector.draw_results(img, results)
cv2.imshow('Detection', output)
```

---

## 2. dlib Face Detector (HOG-based)

### dlib Installation

```bash
# dlib installation (requires C++ compiler)
pip install dlib

# Or use conda (easier)
conda install -c conda-forge dlib
```

### HOG-based Detector

```python
import cv2
import dlib
import numpy as np

# HOG-based face detector
detector = dlib.get_frontal_face_detector()

# Load image
img = cv2.imread('photo.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Upsample by 1 level (2× each dimension) before detection: dlib's HOG
# detector has a minimum face size (~80px); upsampling lets it find smaller
# faces at the cost of ~4× more computation.  Use 0 for real-time scenarios.
faces = detector(rgb, 1)

print(f"Faces detected: {len(faces)}")

# Visualize results
for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Confidence score (dlib detector provides scores by default)
    # Use detector.run() to get scores

cv2.imshow('dlib HOG Detection', img)
cv2.waitKey(0)
```

### CNN-based Detector (More Accurate)

```python
import cv2
import dlib

# CNN face detector (requires model file)
# Download: http://dlib.net/files/mmod_human_face_detector.dat.bz2
cnn_detector = dlib.cnn_face_detection_model_v1(
    'mmod_human_face_detector.dat'
)

img = cv2.imread('photo.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# CNN detection
detections = cnn_detector(rgb, 1)

for d in detections:
    x1 = d.rect.left()
    y1 = d.rect.top()
    x2 = d.rect.right()
    y2 = d.rect.bottom()
    confidence = d.confidence

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'{confidence:.2f}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
```

### Haar vs dlib Comparison

```
+----------------+------------------+------------------+
|     Item       |   Haar Cascade   |   dlib HOG       |
+----------------+------------------+------------------+
| Speed          | Fast             | Medium           |
| Accuracy       | Medium           | High             |
| False Positives| Many             | Few              |
| Profile Face   | Separate model   | Not supported    |
| Small Faces    | Detects well     | Harder to detect |
| Installation   | Included w/OpenCV| Separate install |
| Memory Usage   | Low              | Medium           |
+----------------+------------------+------------------+

dlib CNN Detector:
- Most accurate but slow without GPU
- Detects profile faces well
- Excellent small face detection
```

---

## 3. dlib Face Landmarks (68 Points)

### 68-Point Landmark Structure

```
Face Landmarks 68 Points:

        17-21    22-26
         ____     ____
    0   /    \   /    \   16
    |  | 36-41| |42-47 |  |
    |   \____/   \____/   |
    |      48-67          |
    |      /    \         |
    |     /      \        |
   8     \________/

Point Groups:
- 0-16:   Jawline
- 17-21:  Left eyebrow
- 22-26:  Right eyebrow
- 27-35:  Nose
- 36-41:  Left eye
- 42-47:  Right eye
- 48-67:  Mouth
  - 48-59: Outer lip
  - 60-67: Inner lip
```

### Landmark Detection

```python
import cv2
import dlib
import numpy as np

# Load detector and predictor
detector = dlib.get_frontal_face_detector()
# Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(img):
    """Detect face landmarks"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face detection
    faces = detector(rgb, 1)

    all_landmarks = []

    for face in faces:
        # Predict landmarks
        shape = predictor(rgb, face)

        # Convert dlib shape to numpy array
        landmarks = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)

        all_landmarks.append({
            'face_rect': (face.left(), face.top(),
                         face.right(), face.bottom()),
            'landmarks': landmarks
        })

    return all_landmarks

def draw_landmarks(img, landmarks_data, draw_indices=False):
    """Visualize landmarks"""
    output = img.copy()

    for data in landmarks_data:
        landmarks = data['landmarks']

        # Draw all points
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(output, (x, y), 2, (0, 255, 0), -1)
            if draw_indices:
                cv2.putText(output, str(i), (x-5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        # Draw connecting lines
        # Jawline
        for i in range(16):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 255, 0), 1)

        # Eyebrows
        for i in range(17, 21):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 255, 0), 1)
        for i in range(22, 26):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 255, 0), 1)

        # Nose
        for i in range(27, 30):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 255, 0), 1)
        for i in range(31, 35):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 255, 0), 1)

        # Eyes
        eye_indices = [(36,41), (42,47)]
        for start, end in eye_indices:
            for i in range(start, end):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                        (0, 255, 255), 1)
            cv2.line(output, tuple(landmarks[end]), tuple(landmarks[start]),
                    (0, 255, 255), 1)

        # Mouth
        for i in range(48, 59):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (0, 0, 255), 1)
        cv2.line(output, tuple(landmarks[59]), tuple(landmarks[48]),
                (0, 0, 255), 1)

        for i in range(60, 67):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (0, 0, 255), 1)
        cv2.line(output, tuple(landmarks[67]), tuple(landmarks[60]),
                (0, 0, 255), 1)

    return output

# Usage example
img = cv2.imread('face.jpg')
landmarks_data = get_landmarks(img)
result = draw_landmarks(img, landmarks_data)
cv2.imshow('Landmarks', result)
```

### Landmark Applications

```python
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

class FaceLandmarkAnalyzer:
    """Face landmark analyzer"""

    # Landmark index definitions
    JAWLINE = list(range(0, 17))
    LEFT_EYEBROW = list(range(17, 22))
    RIGHT_EYEBROW = list(range(22, 27))
    NOSE_BRIDGE = list(range(27, 31))
    NOSE_TIP = list(range(31, 36))
    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42, 48))
    OUTER_LIP = list(range(48, 60))
    INNER_LIP = list(range(60, 68))

    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_landmarks(self, img):
        """Extract landmarks"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, 0)

        if len(faces) == 0:
            return None

        shape = self.predictor(rgb, faces[0])
        landmarks = np.array([[shape.part(i).x, shape.part(i).y]
                             for i in range(68)])
        return landmarks

    def eye_aspect_ratio(self, eye_points):
        """Compute Eye Aspect Ratio (EAR) - used for drowsiness detection"""
        # EAR = (vertical_height) / (horizontal_width) — a scale-invariant ratio.
        # Dividing by the horizontal distance normalizes for face size, so the
        # same threshold (~0.2) signals a closed eye whether the face is near or far.
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        # Average two vertical measurements to handle asymmetric eye shapes
        C = dist.euclidean(eye_points[0], eye_points[3])

        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth_points):
        """Compute Mouth Aspect Ratio (MAR) - used for yawn detection"""
        # Vertical distances
        A = dist.euclidean(mouth_points[2], mouth_points[10])  # 51, 59
        B = dist.euclidean(mouth_points[4], mouth_points[8])   # 53, 57
        # Horizontal distance
        C = dist.euclidean(mouth_points[0], mouth_points[6])   # 49, 55

        mar = (A + B) / (2.0 * C)
        return mar

    def get_face_angle(self, landmarks):
        """Compute face tilt angle"""
        # Use eye center points
        left_eye_center = landmarks[self.LEFT_EYE].mean(axis=0)
        right_eye_center = landmarks[self.RIGHT_EYE].mean(axis=0)

        # Compute angle
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        return angle

    def analyze_face(self, img):
        """Comprehensive face analysis"""
        landmarks = self.get_landmarks(img)
        if landmarks is None:
            return None

        # Eye analysis
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Mouth analysis
        outer_lip = landmarks[self.OUTER_LIP]
        mar = self.mouth_aspect_ratio(outer_lip)

        # Face angle
        angle = self.get_face_angle(landmarks)

        return {
            'landmarks': landmarks,
            'eye_aspect_ratio': avg_ear,
            'mouth_aspect_ratio': mar,
            'face_angle': angle,
            'eyes_closed': avg_ear < 0.2,  # Threshold-based
            'mouth_open': mar > 0.5
        }

# Usage example
analyzer = FaceLandmarkAnalyzer('shape_predictor_68_face_landmarks.dat')
img = cv2.imread('face.jpg')
analysis = analyzer.analyze_face(img)

if analysis:
    print(f"Eye Aspect Ratio (EAR): {analysis['eye_aspect_ratio']:.3f}")
    print(f"Mouth Aspect Ratio (MAR): {analysis['mouth_aspect_ratio']:.3f}")
    print(f"Face tilt: {analysis['face_angle']:.1f} degrees")
    print(f"Eyes closed: {analysis['eyes_closed']}")
    print(f"Mouth open: {analysis['mouth_open']}")
```

---

## 4. LBPH Face Recognition

### Understanding LBP (Local Binary Patterns)

```
LBP: Represents pattern around each pixel as binary code

   Surrounding pixels  Comparison (> center?)    Binary code
   +---+---+---+    +---+---+---+
   | 6 | 5 | 2 |    | 1 | 1 | 0 |    11000011
   +---+---+---+    +---+---+---+    = 195
   | 7 |[4]| 1 |    | 1 |   | 0 |
   +---+---+---+    +---+---+---+
   | 8 | 3 | 2 |    | 1 | 0 | 0 |
   +---+---+---+    +---+---+---+

   Compare center pixel (4) with neighbors:
   6>4=1, 5>4=1, 2<4=0, 1<4=0, 2<4=0, 3<4=0, 8>4=1, 7>4=1

LBPH (LBP Histogram):
- Divide image into multiple cells
- Compute LBP histogram per cell
- Concatenate all histograms for feature vector
```

### LBPH Face Recognizer

```python
import cv2
import numpy as np
import os

class LBPHFaceRecognizer:
    """LBPH-based face recognizer"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,        # Pixel spacing for neighbor comparison; radius=1 captures
                             # fine micro-texture detail; larger radius captures coarser patterns
            neighbors=8,     # 8 neighbors = 8-bit code, giving 256 possible LBP values;
                             # more neighbors increase discriminability but also noise sensitivity
            grid_x=8,        # Dividing the face into an 8×8 spatial grid preserves location
            grid_y=8         # information — "which region" matters as much as "what texture"
        )
        self.label_names = {}

    def prepare_training_data(self, data_dir):
        """Prepare training data"""
        faces = []
        labels = []

        for label_id, person_name in enumerate(os.listdir(data_dir)):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            self.label_names[label_id] = person_name

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                # Face detection
                detected_faces = self.face_cascade.detectMultiScale(
                    img, 1.1, 5, minSize=(50, 50)
                )

                for (x, y, w, h) in detected_faces:
                    face_roi = img[y:y+h, x:x+w]
                    # Size normalization
                    face_roi = cv2.resize(face_roi, (100, 100))
                    faces.append(face_roi)
                    labels.append(label_id)

        return faces, labels

    def train(self, faces, labels):
        """Train model"""
        self.recognizer.train(faces, np.array(labels))
        print(f"Training complete: {len(set(labels))} people, {len(faces)} images")

    def save_model(self, path):
        """Save model"""
        self.recognizer.save(path)
        # Also save label names
        np.save(path + '_labels.npy', self.label_names)

    def load_model(self, path):
        """Load model"""
        self.recognizer.read(path)
        self.label_names = np.load(path + '_labels.npy',
                                   allow_pickle=True).item()

    def predict(self, img):
        """Face recognition"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))

            label, confidence = self.recognizer.predict(face_roi)

            # LBPH confidence is a histogram distance (lower = better match),
            # unlike probability scores where higher is better.
            # Threshold at 100 is a practical heuristic: values below 50 indicate
            # a very strong match, 50-100 is uncertain, above 100 is likely a
            # different person or unrecognized face.
            name = self.label_names.get(label, "Unknown")
            if confidence > 100:
                name = "Unknown"

            results.append({
                'rect': (x, y, w, h),
                'name': name,
                'confidence': confidence
            })

        return results

    def draw_results(self, img, results):
        """Visualize results"""
        output = img.copy()

        for result in results:
            x, y, w, h = result['rect']
            name = result['name']
            conf = result['confidence']

            # Color: green for recognized, red for unknown
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

            label = f"{name} ({conf:.1f})"
            cv2.putText(output, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output

# Usage example
"""
Data directory structure:
faces/
    person1/
        img1.jpg
        img2.jpg
    person2/
        img1.jpg
        img2.jpg
"""

# Training
recognizer = LBPHFaceRecognizer()
faces, labels = recognizer.prepare_training_data('faces')
recognizer.train(faces, labels)
recognizer.save_model('face_model.yml')

# Recognition
recognizer.load_model('face_model.yml')
test_img = cv2.imread('test.jpg')
results = recognizer.predict(test_img)
output = recognizer.draw_results(test_img, results)
cv2.imshow('Recognition', output)
```

---

## 5. face_recognition Library

### Installation

```bash
pip install face_recognition
```

### Basic Usage

```python
import face_recognition
import cv2
import numpy as np

# Load image
img = face_recognition.load_image_file('photo.jpg')

# Face location detection
face_locations = face_recognition.face_locations(img)
# Or use CNN model (more accurate)
# face_locations = face_recognition.face_locations(img, model='cnn')

print(f"Faces detected: {len(face_locations)}")

# Face encoding (128-dimensional feature vector)
face_encodings = face_recognition.face_encodings(img, face_locations)

# Face landmarks
face_landmarks = face_recognition.face_landmarks(img, face_locations)

# Visualize results
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imshow('Detection', img_bgr)
```

### Face Comparison and Recognition

```python
import face_recognition
import cv2
import numpy as np
import os

class FaceRecognitionSystem:
    """face_recognition-based face recognition system"""

    def __init__(self):
        self.known_encodings = []
        self.known_names = []

    def add_face(self, img_path, name):
        """Add known face"""
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            self.known_encodings.append(encodings[0])
            self.known_names.append(name)
            print(f"'{name}' face registered")
            return True
        else:
            print(f"No face found in '{img_path}'")
            return False

    def load_faces_from_directory(self, data_dir):
        """Load faces from directory"""
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                self.add_face(img_path, person_name)

        print(f"Total {len(self.known_encodings)} faces loaded")

    def recognize(self, img, tolerance=0.6):
        """Face recognition"""
        # RGB conversion
        if isinstance(img, str):
            img = face_recognition.load_image_file(img)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face detection and encoding
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        results = []

        for (top, right, bottom, left), encoding in zip(face_locations,
                                                         face_encodings):
            # tolerance=0.6 is the recommended default for the 128-d dlib embedding:
            # it corresponds to an Euclidean distance threshold below which two faces
            # are considered the same person.  Lower (0.4-0.5) is stricter and reduces
            # false accepts; higher (0.7) improves recall in poor lighting or aging cases.
            matches = face_recognition.compare_faces(
                self.known_encodings, encoding, tolerance=tolerance
            )

            # face_distance returns Euclidean distances in 128-d embedding space.
            # Using argmin rather than just the first True match is important when
            # multiple known faces pass the tolerance threshold — we want the closest
            # one, not the one that appears first in the list.
            distances = face_recognition.face_distance(
                self.known_encodings, encoding
            )

            name = "Unknown"
            confidence = 0.0

            if True in matches:
                # Find closest match
                best_match_idx = np.argmin(distances)
                if matches[best_match_idx]:
                    name = self.known_names[best_match_idx]
                    confidence = 1 - distances[best_match_idx]

            results.append({
                'location': (top, right, bottom, left),
                'name': name,
                'confidence': confidence
            })

        return results

    def save_encodings(self, path):
        """Save encodings"""
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names
        }
        np.save(path, data)

    def load_encodings(self, path):
        """Load encodings"""
        data = np.load(path, allow_pickle=True).item()
        self.known_encodings = data['encodings']
        self.known_names = data['names']

# Usage example
system = FaceRecognitionSystem()

# Register known faces
system.add_face('known_faces/person1.jpg', 'Alice')
system.add_face('known_faces/person2.jpg', 'Bob')
# Or load from directory
# system.load_faces_from_directory('known_faces')

# Recognition
test_img = cv2.imread('test.jpg')
results = system.recognize(test_img)

# Visualization
for result in results:
    top, right, bottom, left = result['location']
    name = result['name']
    conf = result['confidence']

    cv2.rectangle(test_img, (left, top), (right, bottom), (0, 255, 0), 2)
    label = f"{name} ({conf:.2%})"
    cv2.putText(test_img, label, (left, top - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow('Recognition', test_img)
```

### Face Clustering

```python
import face_recognition
from sklearn.cluster import DBSCAN
import numpy as np
import os

def cluster_faces(image_dir, output_dir='clustered'):
    """Group similar faces together"""

    encodings = []
    image_paths = []
    face_locations_list = []

    # Extract face encodings from all images
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = face_recognition.load_image_file(img_path)

        locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, locations)

        for encoding, location in zip(face_encodings, locations):
            encodings.append(encoding)
            image_paths.append(img_path)
            face_locations_list.append(location)

    # DBSCAN clustering
    # eps: distance threshold for same cluster
    # min_samples: minimum samples to form a cluster
    clt = DBSCAN(metric='euclidean', eps=0.5, min_samples=2)
    clt.fit(encodings)

    # Organize results by cluster
    label_ids = np.unique(clt.labels_)
    num_unique = len(label_ids[label_ids > -1])  # -1 is noise

    print(f"Unique people found: {num_unique}")

    # Save images by cluster
    os.makedirs(output_dir, exist_ok=True)

    for label_id in label_ids:
        indices = np.where(clt.labels_ == label_id)[0]

        if label_id == -1:
            folder = os.path.join(output_dir, 'unknown')
        else:
            folder = os.path.join(output_dir, f'person_{label_id}')

        os.makedirs(folder, exist_ok=True)
        print(f"Cluster {label_id}: {len(indices)} faces")

    return clt.labels_, image_paths, face_locations_list
```

---

## 6. Real-time Face Detection

### Basic Real-time Detection

```python
import cv2
import time

def realtime_face_detection():
    """Real-time face detection (Haar Cascade)"""

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_counter = 0
    fps = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reduce frame size (speed up)
        small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,  # Increase for faster detection
            minNeighbors=4,
            minSize=(30, 30)
        )

        # Scale coordinates back
        for (x, y, w, h) in faces:
            x, y, w, h = x*2, y*2, w*2, h*2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Compute FPS
        fps_counter += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            start_time = time.time()

        # Display FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

realtime_face_detection()
```

### Real-time Face Recognition

```python
import cv2
import face_recognition
import numpy as np
import time

class RealtimeFaceRecognition:
    """Real-time face recognition system"""

    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        # Process every 3rd frame: face encoding is too slow for 30 FPS on CPU.
        # Skipping frames and reusing the previous detection result is nearly
        # invisible to viewers since faces don't move more than ~10px per frame.
        self.process_every_n_frames = 3

    def add_known_face(self, img_path, name):
        """Add known face"""
        img = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(img)

        if encoding:
            self.known_encodings.append(encoding[0])
            self.known_names.append(name)

    def run(self, camera_id=0):
        """Run real-time recognition"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        face_locations = []
        face_names = []

        fps_time = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Scale down to 1/4 size: face_locations runs on the small frame
            # (16× fewer pixels = ~16× faster), then we multiply coordinates by 4
            # to map detections back to the original frame for drawing
            small_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Process every nth frame
            if frame_count % self.process_every_n_frames == 0:
                # Face detection
                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(
                    rgb_small, face_locations
                )

                face_names = []
                for encoding in face_encodings:
                    name = "Unknown"

                    if self.known_encodings:
                        matches = face_recognition.compare_faces(
                            self.known_encodings, encoding, tolerance=0.6
                        )
                        distances = face_recognition.face_distance(
                            self.known_encodings, encoding
                        )

                        if len(distances) > 0:
                            best_idx = np.argmin(distances)
                            if matches[best_idx]:
                                name = self.known_names[best_idx]

                    face_names.append(name)

            # Display results (scale coordinates back)
            for (top, right, bottom, left), name in zip(face_locations,
                                                         face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Name background
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom),
                             color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Compute and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()

            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame with 's' key
                cv2.imwrite(f'capture_{frame_count}.jpg', frame)

        cap.release()
        cv2.destroyAllWindows()

# Usage example
system = RealtimeFaceRecognition()
system.add_known_face('alice.jpg', 'Alice')
system.add_known_face('bob.jpg', 'Bob')
system.run()
```

### Performance Optimization Tips

```
+-------------------------------------------------------------+
|                   Real-time Processing Optimization           |
+-------------------------------------------------------------+
|                                                              |
| 1. Frame size reduction                                      |
|    - Reducing to 1/4 size decreases processing 16x          |
|    - small = cv2.resize(frame, None, fx=0.25, fy=0.25)      |
|                                                              |
| 2. Frame skipping                                            |
|    - Not every frame needs processing                        |
|    - Detect every 2-5 frames                                 |
|    - Use previous results for intermediate frames            |
|                                                              |
| 3. ROI-based processing                                      |
|    - Only search around previous detection location          |
|    - Combine tracking with detection                         |
|                                                              |
| 4. Model selection                                           |
|    - Haar: Fastest, lower accuracy                          |
|    - dlib HOG: Medium                                        |
|    - dlib CNN: Slow, GPU recommended                         |
|    - face_recognition: dlib-based                            |
|                                                              |
| 5. Multi-threading                                           |
|    - Separate detection and display threads                  |
|    - Use Queue for frame passing                             |
|                                                              |
| 6. GPU acceleration                                          |
|    - dlib CUDA build                                         |
|    - OpenCV DNN (CUDA backend)                               |
+-------------------------------------------------------------+
```

---

## 7. Practice Problems

### Problem 1: Attendance Check System

Implement a face recognition-based attendance check system.

**Requirements**:
- Manage registered user face DB
- Real-time webcam recognition
- Record attendance time (CSV or DB)
- Prevent duplicate attendance (within time period)

<details>
<summary>Hint</summary>

```python
import datetime
import csv

class AttendanceSystem:
    def __init__(self):
        self.attendance_log = {}  # {name: last_check_time}
        self.cooldown = 3600  # 1 hour

    def mark_attendance(self, name):
        now = datetime.datetime.now()
        last_check = self.attendance_log.get(name)

        if last_check is None or (now - last_check).seconds > self.cooldown:
            self.attendance_log[name] = now
            self.save_to_csv(name, now)
            return True
        return False
```

</details>

### Problem 2: Drowsiness Detection System

Implement a drowsiness detection system using Eye Aspect Ratio (EAR).

**Requirements**:
- Real-time eye state monitoring
- Alert when EAR is below threshold for certain duration
- Sound or visual alert
- Display current EAR value

<details>
<summary>Hint</summary>

```python
import dlib
from scipy.spatial import distance as dist
import pygame  # For alert sound

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20  # Consecutive frame count

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

counter = 0  # Consecutive frame counter
# Alert when EAR < THRESHOLD persists for CONSEC_FRAMES or more
```

</details>

### Problem 3: Face Mosaic Processing

Write a program that applies mosaic to specific person's face.

**Requirements**:
- Mosaic all faces except designated person
- Or mosaic only designated person
- Adjustable mosaic intensity

<details>
<summary>Hint</summary>

```python
def mosaic_face(img, rect, scale=0.1):
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w]

    # Reduce then enlarge (mosaic effect)
    small = cv2.resize(roi, None, fx=scale, fy=scale)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    img[y:y+h, x:x+w] = mosaic
    return img
```

</details>

### Problem 4: Face Alignment

Implement a program that aligns faces based on eye positions.

**Requirements**:
- Detect both eye positions
- Rotate to make eyes horizontal
- Crop face region and normalize size

<details>
<summary>Hint</summary>

```python
import numpy as np

def align_face(img, left_eye, right_eye, desired_size=(256, 256)):
    # Compute angle between eyes
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Rotation center (midpoint between eyes)
    center = ((left_eye[0] + right_eye[0]) // 2,
              (left_eye[1] + right_eye[1]) // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return aligned
```

</details>

### Problem 5: Emotion Analysis System

Implement a simple emotion analysis system using face landmarks.

**Requirements**:
- Analyze eye, eyebrow, and mouth shapes
- Classify basic emotions (happy, sad, surprised, neutral)
- Display emotion in real-time

<details>
<summary>Hint</summary>

```python
# Example emotion judgment criteria:
# - Happy: Mouth corners raised (mouth ends y < mouth center y)
# - Surprised: Eyes and mouth wide open (high EAR, high MAR)
# - Sad: Eyebrows drooped, mouth corners down
# - Neutral: Little change

def analyze_emotion(landmarks):
    # Mouth analysis
    mouth = landmarks[48:68]
    mouth_height = mouth[14][1] - mouth[10][1]  # Mouth height

    # Eye analysis
    left_eye = landmarks[36:42]
    ear = eye_aspect_ratio(left_eye)

    # Rule-based judgment
    if mouth_height > threshold and ear > threshold:
        return "Surprised"
    # ...
```

</details>

---

## Next Steps

- [17_Video_Processing.md](./17_Video_Processing.md) - VideoCapture, background subtraction, optical flow

---

## References

- [dlib Documentation](http://dlib.net/python/index.html)
- [face_recognition GitHub](https://github.com/ageitgey/face_recognition)
- [OpenCV Face Recognition](https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html)
- [68 Face Landmarks](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
- Kazemi, V., & Sullivan, J. (2014). "One Millisecond Face Alignment with an Ensemble of Regression Trees"
