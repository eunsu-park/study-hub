"""
Exercise Solutions for Lesson 16: Face Detection and Recognition
Computer Vision - Haar Cascade, dlib, Face Recognition

Topics covered:
- Attendance check system (face DB simulation)
- Drowsiness detection (Eye Aspect Ratio)
- Face mosaic processing
- Face alignment based on eye positions
- Emotion analysis using landmark geometry
"""

import numpy as np


# =============================================================================
# Helper: Simulated face/eye detection
# =============================================================================

def simulate_face_detection(img):
    """Simulate face detection returning list of (x, y, w, h) regions."""
    h, w = img.shape[:2]
    # Detect bright oval-ish regions as "faces"
    faces = []
    visited = np.zeros(img.shape[:2], dtype=bool)

    for i in range(0, h - 30, 10):
        for j in range(0, w - 30, 10):
            region = img[i:i+30, j:j+30]
            if len(region.shape) == 3:
                brightness = np.mean(region)
            else:
                brightness = np.mean(region)

            if brightness > 140 and not visited[i, j]:
                faces.append((j, i, 30, 30))
                visited[i:i+30, j:j+30] = True

    return faces


# =============================================================================
# Exercise 1: Attendance Check System
# =============================================================================

def exercise_1_attendance_system():
    """
    Simulate a face recognition-based attendance system.

    Features:
    - Registered user face DB (simulated with feature vectors)
    - Recognition by comparing feature distances
    - Attendance log with timestamps
    - Duplicate prevention (cooldown period)

    Returns:
        attendance_log dict
    """
    # Simulated face database (each person = feature vector)
    np.random.seed(42)
    db = {
        'Alice':   np.random.randn(128) * 0.5 + np.array([1.0] * 64 + [0.0] * 64),
        'Bob':     np.random.randn(128) * 0.5 + np.array([0.0] * 64 + [1.0] * 64),
        'Charlie': np.random.randn(128) * 0.5 + np.array([-1.0] * 64 + [0.5] * 64),
    }

    def recognize_face(feature, db, threshold=15.0):
        """Match feature vector against database."""
        best_name = "Unknown"
        best_dist = float('inf')
        for name, ref in db.items():
            dist = np.sqrt(np.sum((feature - ref)**2))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return (best_name, best_dist) if best_dist < threshold else ("Unknown", best_dist)

    # Simulate attendance events
    attendance_log = {}
    cooldown = 3600  # 1 hour (simulated as seconds)

    events = [
        (0, 'Alice', np.random.randn(128) * 0.5 + np.array([1.0] * 64 + [0.0] * 64)),
        (100, 'Bob', np.random.randn(128) * 0.5 + np.array([0.0] * 64 + [1.0] * 64)),
        (200, 'Alice', np.random.randn(128) * 0.5 + np.array([1.0] * 64 + [0.0] * 64)),  # Duplicate
        (4000, 'Alice', np.random.randn(128) * 0.5 + np.array([1.0] * 64 + [0.0] * 64)),  # After cooldown
        (300, 'Unknown', np.random.randn(128) * 5),  # Unknown person
    ]

    print("Attendance System Simulation")
    print("=" * 60)

    for timestamp, expected, feature in events:
        name, dist = recognize_face(feature, db)

        # Check cooldown
        if name != "Unknown":
            last_time = attendance_log.get(name, {}).get('last_time', -cooldown - 1)
            if timestamp - last_time < cooldown:
                print(f"  t={timestamp:>5}s: {name} - DUPLICATE (cooldown active), "
                      f"dist={dist:.2f}")
                continue

            attendance_log[name] = {'last_time': timestamp, 'count': attendance_log.get(name, {}).get('count', 0) + 1}
            print(f"  t={timestamp:>5}s: {name} - CHECKED IN, dist={dist:.2f}")
        else:
            print(f"  t={timestamp:>5}s: Unknown person (dist={dist:.2f})")

    print(f"\nAttendance Summary:")
    for name, info in attendance_log.items():
        print(f"  {name}: {info['count']} check-in(s)")

    return attendance_log


# =============================================================================
# Exercise 2: Drowsiness Detection (EAR)
# =============================================================================

def exercise_2_drowsiness_detection():
    """
    Implement drowsiness detection using Eye Aspect Ratio (EAR).
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Simulates a sequence of EAR values and triggers alert.

    Returns:
        list of (frame, ear, is_drowsy) tuples
    """
    EAR_THRESHOLD = 0.25
    CONSEC_FRAMES = 5

    def eye_aspect_ratio(eye_points):
        """Calculate EAR from 6 eye landmark points."""
        # eye_points: [(x1,y1), (x2,y2), ..., (x6,y6)]
        p1, p2, p3, p4, p5, p6 = eye_points
        A = np.sqrt((p2[0]-p6[0])**2 + (p2[1]-p6[1])**2)
        B = np.sqrt((p3[0]-p5[0])**2 + (p3[1]-p5[1])**2)
        C = np.sqrt((p1[0]-p4[0])**2 + (p1[1]-p4[1])**2)
        return (A + B) / (2.0 * C) if C > 0 else 0

    # Simulate eye landmarks over time
    # Normal eye: EAR ~0.3, Closed eye: EAR ~0.15
    np.random.seed(42)
    n_frames = 30
    ear_values = np.ones(n_frames) * 0.30 + np.random.randn(n_frames) * 0.03
    # Simulate drowsy period (frames 15-25)
    ear_values[15:25] = 0.18 + np.random.randn(10) * 0.02

    counter = 0
    results = []

    print(f"Drowsiness Detection (threshold={EAR_THRESHOLD}, "
          f"consec_frames={CONSEC_FRAMES})")
    print(f"{'Frame':>6} | {'EAR':>6} | {'Counter':>8} | {'Status':>10}")
    print("-" * 40)

    for frame_idx, ear in enumerate(ear_values):
        ear = max(0.05, ear)  # Clamp

        if ear < EAR_THRESHOLD:
            counter += 1
        else:
            counter = 0

        is_drowsy = counter >= CONSEC_FRAMES
        status = "ALERT!" if is_drowsy else "Normal"

        results.append((frame_idx, ear, is_drowsy))
        print(f"{frame_idx:>6} | {ear:>6.3f} | {counter:>8} | {status:>10}")

    alert_frames = sum(1 for _, _, d in results if d)
    print(f"\nAlert frames: {alert_frames}/{n_frames}")

    return results


# =============================================================================
# Exercise 3: Face Mosaic Processing
# =============================================================================

def exercise_3_face_mosaic(img, face_rects, mosaic_scale=0.1):
    """
    Apply mosaic (pixelation) to detected face regions.

    Parameters:
        img: (H, W, C) or (H, W) image
        face_rects: list of (x, y, w, h) face regions
        mosaic_scale: downsample factor (smaller = more mosaic)

    Returns:
        mosaicked image
    """
    result = img.copy()
    h_img, w_img = img.shape[:2]

    for x, y, fw, fh in face_rects:
        # Clamp to image bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + fw), min(h_img, y + fh)
        roi = result[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        rh, rw = roi.shape[:2]
        small_w = max(1, int(rw * mosaic_scale))
        small_h = max(1, int(rh * mosaic_scale))

        # Downsample (average pooling)
        if len(roi.shape) == 3:
            small = np.zeros((small_h, small_w, roi.shape[2]), dtype=roi.dtype)
        else:
            small = np.zeros((small_h, small_w), dtype=roi.dtype)

        for si in range(small_h):
            for sj in range(small_w):
                sy1 = int(si * rh / small_h)
                sy2 = int((si + 1) * rh / small_h)
                sx1 = int(sj * rw / small_w)
                sx2 = int((sj + 1) * rw / small_w)
                if len(roi.shape) == 3:
                    small[si, sj] = np.mean(roi[sy1:sy2, sx1:sx2], axis=(0, 1))
                else:
                    small[si, sj] = np.mean(roi[sy1:sy2, sx1:sx2])

        # Upsample (nearest neighbor) to create mosaic effect
        for i in range(rh):
            for j in range(rw):
                si = min(int(i * small_h / rh), small_h - 1)
                sj = min(int(j * small_w / rw), small_w - 1)
                roi[i, j] = small[si, sj]

        result[y1:y2, x1:x2] = roi
        print(f"  Face ({x},{y}) {fw}x{fh}: mosaic {small_w}x{small_h}")

    return result


# =============================================================================
# Exercise 4: Face Alignment
# =============================================================================

def exercise_4_face_alignment(img, left_eye, right_eye, desired_size=(128, 128)):
    """
    Align a face based on eye positions: rotate to make eyes horizontal,
    then crop and resize.

    Parameters:
        img: source image
        left_eye: (x, y) of left eye center
        right_eye: (x, y) of right eye center
        desired_size: output face size

    Returns:
        aligned face image
    """
    # Compute angle between eyes
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Eye distance for scale reference
    eye_dist = np.sqrt(dX**2 + dY**2)

    # Rotation center: midpoint between eyes
    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = (left_eye[1] + right_eye[1]) / 2

    # Rotate image
    h, w = img.shape[:2]
    theta = np.radians(-angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)

    if len(img.shape) == 3:
        rotated = np.zeros_like(img)
    else:
        rotated = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            src_x = cos_a * (j - center_x) + sin_a * (i - center_y) + center_x
            src_y = -sin_a * (j - center_x) + cos_a * (i - center_y) + center_y
            si, sj = int(round(src_y)), int(round(src_x))
            if 0 <= si < h and 0 <= sj < w:
                rotated[i, j] = img[si, sj]

    # Crop face region around eyes (eyes at ~35% from top)
    face_w = int(eye_dist * 2.5)
    face_h = int(face_w * desired_size[1] / desired_size[0])

    crop_x1 = int(center_x - face_w // 2)
    crop_y1 = int(center_y - face_h * 0.35)
    crop_x2 = crop_x1 + face_w
    crop_y2 = crop_y1 + face_h

    # Clamp
    crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
    crop_x2, crop_y2 = min(w, crop_x2), min(h, crop_y2)

    cropped = rotated[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to desired size
    if cropped.size > 0:
        out_h, out_w = desired_size
        row_idx = (np.arange(out_h) * cropped.shape[0] / out_h).astype(int)
        col_idx = (np.arange(out_w) * cropped.shape[1] / out_w).astype(int)
        row_idx = np.clip(row_idx, 0, cropped.shape[0] - 1)
        col_idx = np.clip(col_idx, 0, cropped.shape[1] - 1)
        aligned = cropped[np.ix_(row_idx, col_idx)]
    else:
        aligned = np.zeros(desired_size, dtype=img.dtype)

    print(f"Eye positions: L={left_eye}, R={right_eye}")
    print(f"Rotation angle: {angle:.1f} deg")
    print(f"Eye distance: {eye_dist:.1f}px")
    print(f"Aligned face: {aligned.shape}")

    return aligned


# =============================================================================
# Exercise 5: Emotion Analysis
# =============================================================================

def exercise_5_emotion_analysis():
    """
    Simple rule-based emotion classification using facial landmark geometry.

    Analyzes:
    - Eye Aspect Ratio (EAR) for surprise
    - Mouth Aspect Ratio (MAR) for smile/surprise
    - Eyebrow position for sadness

    Returns:
        detected emotion string
    """
    def classify_emotion(ear, mar, brow_height):
        """
        Rule-based emotion classifier.
        ear: Eye Aspect Ratio
        mar: Mouth Aspect Ratio
        brow_height: relative eyebrow height (higher = raised)
        """
        if ear > 0.35 and mar > 0.6:
            return "Surprised"
        elif mar > 0.4 and brow_height > 0.5:
            return "Happy"
        elif brow_height < 0.3 and mar < 0.2:
            return "Sad"
        elif mar < 0.15 and ear < 0.22:
            return "Angry"
        else:
            return "Neutral"

    # Test cases with simulated landmark measurements
    test_cases = [
        {"name": "Normal", "ear": 0.28, "mar": 0.20, "brow": 0.45},
        {"name": "Smiling", "ear": 0.25, "mar": 0.50, "brow": 0.55},
        {"name": "Surprised", "ear": 0.40, "mar": 0.70, "brow": 0.65},
        {"name": "Sad", "ear": 0.26, "mar": 0.15, "brow": 0.25},
        {"name": "Drowsy", "ear": 0.18, "mar": 0.10, "brow": 0.35},
    ]

    print(f"{'Expression':>12} | {'EAR':>5} | {'MAR':>5} | {'Brow':>5} | {'Emotion':>10}")
    print("-" * 50)

    for tc in test_cases:
        emotion = classify_emotion(tc['ear'], tc['mar'], tc['brow'])
        print(f"{tc['name']:>12} | {tc['ear']:>5.2f} | {tc['mar']:>5.2f} | "
              f"{tc['brow']:>5.2f} | {emotion:>10}")

    return test_cases


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Attendance Check System")
    exercise_1_attendance_system()

    print("\n>>> Exercise 2: Drowsiness Detection")
    exercise_2_drowsiness_detection()

    print("\n>>> Exercise 3: Face Mosaic")
    face_img = np.random.randint(100, 200, (100, 150, 3), dtype=np.uint8)
    exercise_3_face_mosaic(face_img, [(20, 10, 40, 40), (80, 30, 35, 35)])

    print("\n>>> Exercise 4: Face Alignment")
    align_img = np.random.randint(80, 180, (200, 200), dtype=np.uint8)
    exercise_4_face_alignment(align_img, left_eye=(70, 90), right_eye=(130, 80))

    print("\n>>> Exercise 5: Emotion Analysis")
    exercise_5_emotion_analysis()

    print("\nAll exercises completed successfully.")
