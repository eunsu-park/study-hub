"""
Exercise Solutions for Lesson 19: DNN Module
Computer Vision - YOLO, SSD, Object Detection with Deep Learning

Topics covered:
- Detection performance comparison (YOLO vs SSD simulation)
- Custom class filtering
- Video object tracking + detection
- Model ensemble (Weighted NMS)
- Real-time object counting system
"""

import numpy as np


# =============================================================================
# Helper: Simulated object detector
# =============================================================================

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'
]


def simulate_detections(img, detector_type='yolo', noise=0.0):
    """
    Simulate object detections on a synthetic image.
    Returns list of dicts: {class_id, class_name, confidence, box: (x,y,w,h)}
    """
    h, w = img.shape[:2]
    np.random.seed(hash(detector_type) % 2**31)

    detections = []

    # Scan for bright regions as "objects"
    step = 20
    for i in range(0, h - step, step):
        for j in range(0, w - step, step):
            region = img[i:i+step, j:j+step]
            if len(region.shape) == 3:
                brightness = np.mean(region)
            else:
                brightness = np.mean(region)

            if brightness > 140:
                class_id = (i * w + j) % len(COCO_CLASSES)
                conf = 0.4 + (brightness / 255.0) * 0.5 + np.random.randn() * noise
                conf = np.clip(conf, 0.1, 0.99)

                detections.append({
                    'class_id': class_id,
                    'class_name': COCO_CLASSES[class_id],
                    'confidence': conf,
                    'box': (j, i, step, step),
                    'id': len(detections),
                })

    return detections


def compute_iou(box1, box2):
    """Compute IoU between two boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter

    return inter / union if union > 0 else 0


# =============================================================================
# Exercise 1: Object Detection Performance Comparison
# =============================================================================

def exercise_1_detection_comparison():
    """
    Compare performance of simulated YOLO and SSD detectors.

    Metrics:
    - Detection speed (simulated)
    - Detection count
    - Average confidence
    - IoU-based accuracy against ground truth

    Returns:
        comparison dict
    """
    import time

    # Create test images
    np.random.seed(42)
    test_images = []
    ground_truths = []

    for idx in range(5):
        img = np.random.randint(40, 80, (120, 160), dtype=np.uint8)
        gt = []

        # Place objects
        n_objects = np.random.randint(2, 5)
        for _ in range(n_objects):
            x = np.random.randint(10, 130)
            y = np.random.randint(10, 90)
            w, h = 20, 20
            img[y:y+h, x:x+w] = np.random.randint(160, 220)
            gt.append({
                'class_id': np.random.randint(0, len(COCO_CLASSES)),
                'box': (x, y, w, h),
            })

        test_images.append(img)
        ground_truths.append(gt)

    # Run both detectors
    results = {'yolo': [], 'ssd': []}
    timings = {'yolo': [], 'ssd': []}

    for dtype in ['yolo', 'ssd']:
        for img in test_images:
            start = time.perf_counter()
            dets = simulate_detections(img, dtype, noise=0.05)

            # Simulate speed difference (YOLO slightly faster)
            if dtype == 'ssd':
                # SSD processes slightly more
                _ = np.sum(img > 128)

            elapsed = time.perf_counter() - start
            results[dtype].append(dets)
            timings[dtype].append(elapsed)

    # Compute metrics
    print("Object Detection Performance Comparison")
    print("=" * 65)
    print(f"  Test images: {len(test_images)}")
    print(f"  Image size: 160x120")

    print(f"\n  {'Metric':>25} | {'YOLO':>12} | {'SSD':>12}")
    print(f"  {'-'*55}")

    for dtype in ['yolo', 'ssd']:
        all_dets = [d for dets in results[dtype] for d in dets]
        avg_count = np.mean([len(d) for d in results[dtype]])
        avg_conf = np.mean([d['confidence'] for d in all_dets]) if all_dets else 0
        avg_time = np.mean(timings[dtype]) * 1000  # ms

        if dtype == 'yolo':
            yolo_stats = {
                'count': avg_count, 'conf': avg_conf,
                'time': avg_time, 'total': len(all_dets)
            }
        else:
            ssd_stats = {
                'count': avg_count, 'conf': avg_conf,
                'time': avg_time, 'total': len(all_dets)
            }

    print(f"  {'Avg detections/image':>25} | "
          f"{yolo_stats['count']:>12.1f} | {ssd_stats['count']:>12.1f}")
    print(f"  {'Avg confidence':>25} | "
          f"{yolo_stats['conf']:>12.3f} | {ssd_stats['conf']:>12.3f}")
    print(f"  {'Avg time (ms)':>25} | "
          f"{yolo_stats['time']:>12.3f} | {ssd_stats['time']:>12.3f}")
    print(f"  {'Total detections':>25} | "
          f"{yolo_stats['total']:>12} | {ssd_stats['total']:>12}")

    # IoU matching with ground truth
    for dtype in ['yolo', 'ssd']:
        matched = 0
        total_gt = 0
        for v in range(len(test_images)):
            total_gt += len(ground_truths[v])
            for gt in ground_truths[v]:
                for det in results[dtype][v]:
                    if compute_iou(gt['box'], det['box']) > 0.3:
                        matched += 1
                        break
        recall = matched / total_gt if total_gt > 0 else 0
        print(f"  {'Recall (IoU>0.3) [' + dtype + ']':>25} | "
              f"{recall:>12.1%} |")

    return {'yolo': yolo_stats, 'ssd': ssd_stats}


# =============================================================================
# Exercise 2: Custom Class Filtering
# =============================================================================

def exercise_2_class_filtering():
    """
    Filter detections to keep only specific target classes.

    Features:
    - Configurable target class list
    - Per-class confidence thresholds
    - Per-class color assignment
    - Detection statistics

    Returns:
        filtered detection results
    """

    class FilteredDetector:
        def __init__(self, target_classes, class_thresholds=None):
            self.target_classes = target_classes
            self.class_thresholds = class_thresholds or {
                cls: 0.5 for cls in target_classes
            }
            # Assign colors (BGR) per class
            np.random.seed(123)
            self.class_colors = {
                cls: tuple(np.random.randint(50, 255, 3).tolist())
                for cls in target_classes
            }

        def filter(self, detections):
            """Filter detections to keep only target classes."""
            filtered = []
            rejected = {'wrong_class': 0, 'low_confidence': 0}

            for det in detections:
                name = det['class_name']
                if name not in self.target_classes:
                    rejected['wrong_class'] += 1
                    continue

                thresh = self.class_thresholds.get(name, 0.5)
                if det['confidence'] < thresh:
                    rejected['low_confidence'] += 1
                    continue

                det_copy = det.copy()
                det_copy['color'] = self.class_colors[name]
                filtered.append(det_copy)

            return filtered, rejected

    # Create test image with multiple object types
    np.random.seed(42)
    img = np.random.randint(40, 80, (100, 150), dtype=np.uint8)

    # Place various "objects"
    for y_pos in range(10, 80, 20):
        for x_pos in range(10, 130, 25):
            img[y_pos:y_pos+15, x_pos:x_pos+15] = np.random.randint(150, 220)

    # Get all detections
    all_dets = simulate_detections(img, 'yolo', noise=0.1)

    # Configure filter for vehicles only
    target = ['car', 'bus', 'truck', 'motorcycle']
    thresholds = {
        'car': 0.4,
        'bus': 0.5,
        'truck': 0.5,
        'motorcycle': 0.3,
    }

    detector = FilteredDetector(target, thresholds)
    filtered, rejected = detector.filter(all_dets)

    print("Custom Class Filtering")
    print("=" * 60)
    print(f"\n  Target classes: {target}")
    print(f"  Thresholds: {thresholds}")
    print(f"\n  Total detections: {len(all_dets)}")
    print(f"  After filtering:  {len(filtered)}")
    print(f"  Rejected (wrong class):     {rejected['wrong_class']}")
    print(f"  Rejected (low confidence):  {rejected['low_confidence']}")

    if filtered:
        print(f"\n  Filtered Detections:")
        for det in filtered[:10]:
            print(f"    {det['class_name']:>12} | conf={det['confidence']:.3f} | "
                  f"box={det['box']} | color={det['color']}")

    # Class distribution
    class_counts = {}
    for det in filtered:
        name = det['class_name']
        class_counts[name] = class_counts.get(name, 0) + 1

    print(f"\n  Class Distribution (filtered):")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:>12}: {count}")

    return filtered


# =============================================================================
# Exercise 3: Video Object Tracking + Detection
# =============================================================================

def exercise_3_tracking_detection():
    """
    Combine detection (every N frames) with simple tracking
    for intermediate frames.

    Features:
    - Periodic detection (every N frames)
    - Simple centroid tracking between detections
    - Object ID assignment and maintenance
    - Re-detection on tracking failure

    Returns:
        tracking results per frame
    """
    n_frames = 30
    detect_every_n = 5
    h, w = 100, 150

    # Generate synthetic video with moving objects
    np.random.seed(42)
    objects_gt = [
        {'id': 0, 'start': (20, 30), 'vel': (2, 1), 'class': 'person'},
        {'id': 1, 'start': (100, 60), 'vel': (-1, 0.5), 'class': 'car'},
        {'id': 2, 'start': (60, 10), 'vel': (0.5, 2), 'class': 'person'},
    ]

    class SimpleTracker:
        def __init__(self, max_dist=30):
            self.tracked = {}  # id -> {pos, class_name, age}
            self.next_id = 0
            self.max_dist = max_dist

        def update_with_detections(self, detections):
            """Match detections to existing tracks or create new ones."""
            new_tracked = {}
            used_dets = set()

            # Match existing tracks to nearest detection
            for tid, track in self.tracked.items():
                best_dist = float('inf')
                best_det_idx = -1

                for i, det in enumerate(detections):
                    if i in used_dets:
                        continue
                    dx = det['cx'] - track['pos'][0]
                    dy = det['cy'] - track['pos'][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_det_idx = i

                if best_det_idx >= 0 and best_dist < self.max_dist:
                    det = detections[best_det_idx]
                    new_tracked[tid] = {
                        'pos': (det['cx'], det['cy']),
                        'class_name': det['class_name'],
                        'age': 0,
                    }
                    used_dets.add(best_det_idx)

            # Create new tracks for unmatched detections
            for i, det in enumerate(detections):
                if i not in used_dets:
                    new_tracked[self.next_id] = {
                        'pos': (det['cx'], det['cy']),
                        'class_name': det['class_name'],
                        'age': 0,
                    }
                    self.next_id += 1

            self.tracked = new_tracked

        def predict(self):
            """Simple prediction: keep same position (no motion model)."""
            for tid in self.tracked:
                self.tracked[tid]['age'] += 1

        def get_positions(self):
            return {tid: t.copy() for tid, t in self.tracked.items()}

    tracker = SimpleTracker(max_dist=40)
    frame_results = []

    print("Video Object Tracking + Detection")
    print(f"  Frames: {n_frames}, Detect every {detect_every_n} frames")
    print(f"  Objects: {len(objects_gt)}")
    print("=" * 65)

    for f in range(n_frames):
        # Generate frame
        frame = np.random.randint(40, 70, (h, w), dtype=np.uint8)

        # Place objects at current positions
        current_dets = []
        for obj in objects_gt:
            x = int(obj['start'][0] + obj['vel'][0] * f)
            y = int(obj['start'][1] + obj['vel'][1] * f)
            if 0 <= x < w - 15 and 0 <= y < h - 15:
                frame[y:y+15, x:x+15] = 180
                current_dets.append({
                    'cx': x + 7, 'cy': y + 7,
                    'class_name': obj['class'],
                    'confidence': 0.85 + np.random.randn() * 0.05,
                })

        # Detection or tracking
        if f % detect_every_n == 0:
            tracker.update_with_detections(current_dets)
            mode = "DETECT"
        else:
            tracker.predict()
            mode = "TRACK"

        positions = tracker.get_positions()
        frame_results.append({'frame': f, 'mode': mode, 'tracks': positions})

        if f % 5 == 0 or mode == "DETECT":
            track_str = ", ".join(
                [f"ID{tid}=({t['pos'][0]:.0f},{t['pos'][1]:.0f})"
                 for tid, t in positions.items()]
            )
            print(f"  Frame {f:>3} [{mode:>6}]: {track_str}")

    print(f"\n  Total tracked IDs: {tracker.next_id}")
    print(f"  Active tracks at end: {len(tracker.tracked)}")

    return frame_results


# =============================================================================
# Exercise 4: Model Ensemble
# =============================================================================

def exercise_4_model_ensemble():
    """
    Combine results from multiple detectors using Weighted NMS.

    Steps:
    1. Run multiple detectors
    2. Collect all boxes with weighted confidence
    3. Apply NMS to merge overlapping boxes
    4. Output final results

    Returns:
        ensemble results
    """

    def weighted_nms(boxes, scores, classes, iou_threshold=0.5):
        """
        Weighted Non-Maximum Suppression.
        boxes: list of (x, y, w, h)
        scores: list of confidence scores
        classes: list of class IDs
        """
        if len(boxes) == 0:
            return [], [], []

        indices = np.argsort(scores)[::-1]
        keep_boxes = []
        keep_scores = []
        keep_classes = []

        while len(indices) > 0:
            best = indices[0]
            keep_boxes.append(boxes[best])
            keep_scores.append(scores[best])
            keep_classes.append(classes[best])

            # Find overlapping boxes of same class
            remaining = []
            for idx in indices[1:]:
                if classes[idx] != classes[best]:
                    remaining.append(idx)
                    continue

                iou = compute_iou(boxes[best], boxes[idx])
                if iou < iou_threshold:
                    remaining.append(idx)
                # else: suppress (merged into best)

            indices = remaining

        return keep_boxes, keep_scores, keep_classes

    # Create test image
    np.random.seed(42)
    img = np.random.randint(40, 70, (100, 150), dtype=np.uint8)

    # Place objects
    obj_positions = [(20, 20, 25, 25), (70, 40, 30, 30), (110, 15, 20, 20)]
    for x, y, ow, oh in obj_positions:
        img[y:y+oh, x:x+ow] = np.random.randint(160, 220)

    # Run 3 "detectors" with different characteristics
    detector_configs = [
        ('YOLO_v4', 0.4, 0.05),   # (name, weight, noise)
        ('SSD_300', 0.35, 0.08),
        ('EfficientDet', 0.25, 0.03),
    ]

    all_boxes = []
    all_scores = []
    all_classes = []
    detector_results = {}

    print("Model Ensemble (Weighted NMS)")
    print("=" * 60)

    for det_name, weight, noise in detector_configs:
        dets = simulate_detections(img, det_name, noise=noise)
        detector_results[det_name] = dets

        for det in dets:
            all_boxes.append(det['box'])
            all_scores.append(det['confidence'] * weight)
            all_classes.append(det['class_id'])

        print(f"  {det_name}: {len(dets)} detections, weight={weight}")

    # Apply Weighted NMS
    final_boxes, final_scores, final_classes = weighted_nms(
        all_boxes, all_scores, all_classes, iou_threshold=0.4
    )

    total_raw = len(all_boxes)
    total_final = len(final_boxes)

    print(f"\n  Raw detections (all models): {total_raw}")
    print(f"  After Weighted NMS:          {total_final}")
    print(f"  Reduction: {total_raw - total_final} boxes merged/suppressed")

    print(f"\n  Final Detections:")
    for i, (box, score, cls) in enumerate(
            zip(final_boxes, final_scores, final_classes)):
        cls_name = COCO_CLASSES[cls % len(COCO_CLASSES)]
        print(f"    [{i}] {cls_name:>12}: conf={score:.3f}, "
              f"box=({box[0]},{box[1]},{box[2]},{box[3]})")

    return {
        'raw_count': total_raw,
        'final_count': total_final,
        'final_boxes': final_boxes,
        'final_scores': final_scores,
    }


# =============================================================================
# Exercise 5: Real-time Object Counting System
# =============================================================================

def exercise_5_object_counting():
    """
    Count objects crossing a virtual counting line in video.

    Features:
    - Detection-based counting
    - Counting line with direction detection
    - Entry/exit tracking
    - Statistics display (per-class, cumulative)

    Returns:
        counting statistics dict
    """

    class ObjectCounter:
        def __init__(self, count_line_y, target_classes=None):
            self.count_line_y = count_line_y
            self.target_classes = target_classes
            self.tracked = {}  # id -> {'prev_y', 'class'}
            self.count_in = 0
            self.count_out = 0
            self.class_counts = {}  # class_name -> {'in': n, 'out': n}
            self.events = []
            self.next_id = 0

        def process_detections(self, detections, frame_idx):
            """Process detections and count line crossings."""
            new_tracked = {}

            for det in detections:
                cls = det['class_name']
                if self.target_classes and cls not in self.target_classes:
                    continue

                # Get center y
                _, y, _, h = det['box']
                center_y = y + h // 2

                # Try to match with existing track
                matched_id = None
                for tid, track in self.tracked.items():
                    if tid in new_tracked:
                        continue
                    # Simple nearest match
                    if abs(center_y - track['prev_y']) < 30:
                        matched_id = tid
                        break

                if matched_id is None:
                    matched_id = self.next_id
                    self.next_id += 1

                # Check line crossing
                if matched_id in self.tracked:
                    prev_y = self.tracked[matched_id]['prev_y']
                    if prev_y < self.count_line_y <= center_y:
                        self.count_out += 1
                        self._update_class_count(cls, 'out')
                        self.events.append(
                            (frame_idx, matched_id, cls, 'OUT'))
                    elif prev_y > self.count_line_y >= center_y:
                        self.count_in += 1
                        self._update_class_count(cls, 'in')
                        self.events.append(
                            (frame_idx, matched_id, cls, 'IN'))

                new_tracked[matched_id] = {
                    'prev_y': center_y, 'class': cls
                }

            self.tracked = new_tracked

        def _update_class_count(self, cls, direction):
            if cls not in self.class_counts:
                self.class_counts[cls] = {'in': 0, 'out': 0}
            self.class_counts[cls][direction] += 1

    # Simulation parameters
    h, w = 120, 160
    n_frames = 50
    line_y = h // 2

    # Generate moving objects
    np.random.seed(42)
    moving_objects = [
        # (start_x, start_y, vx, vy, class, start_frame)
        (30, 10, 0, 2, 'person', 0),
        (80, 100, 0, -2, 'person', 0),
        (120, 15, -1, 2, 'car', 5),
        (50, 95, 1, -1.5, 'car', 10),
        (100, 5, 0, 3, 'person', 15),
        (40, 110, 0, -2.5, 'truck', 8),
        (70, 20, 0, 1.5, 'bus', 12),
    ]

    counter = ObjectCounter(
        count_line_y=line_y,
        target_classes=['person', 'car', 'truck', 'bus']
    )

    print("Real-time Object Counting System")
    print(f"  Frame size: {w}x{h}")
    print(f"  Counting line: y={line_y}")
    print(f"  Target classes: {counter.target_classes}")
    print("=" * 65)

    for f in range(n_frames):
        detections = []
        for sx, sy, vx, vy, cls, sf in moving_objects:
            if f < sf:
                continue
            t = f - sf
            x = int(sx + vx * t)
            y = int(sy + vy * t)
            if 0 <= x < w - 15 and 0 <= y < h - 15:
                detections.append({
                    'class_name': cls,
                    'confidence': 0.8,
                    'box': (x, y, 15, 15),
                })

        counter.process_detections(detections, f)

    # Print events
    print(f"\n  Crossing Events:")
    for frame_idx, obj_id, cls, direction in counter.events:
        print(f"    Frame {frame_idx:>3}: ID{obj_id} ({cls}) -> {direction}")

    # Statistics
    print(f"\n  Summary:")
    print(f"    Total IN:  {counter.count_in}")
    print(f"    Total OUT: {counter.count_out}")
    print(f"    Net (IN-OUT): {counter.count_in - counter.count_out}")

    print(f"\n  Per-Class Counts:")
    print(f"    {'Class':>10} | {'IN':>5} | {'OUT':>5} | {'Net':>5}")
    print(f"    {'-'*35}")
    for cls, counts in sorted(counter.class_counts.items()):
        net = counts['in'] - counts['out']
        print(f"    {cls:>10} | {counts['in']:>5} | {counts['out']:>5} | "
              f"{net:>5}")

    return {
        'count_in': counter.count_in,
        'count_out': counter.count_out,
        'class_counts': counter.class_counts,
        'events': counter.events,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Detection Performance Comparison")
    exercise_1_detection_comparison()

    print("\n>>> Exercise 2: Custom Class Filtering")
    exercise_2_class_filtering()

    print("\n>>> Exercise 3: Video Object Tracking + Detection")
    exercise_3_tracking_detection()

    print("\n>>> Exercise 4: Model Ensemble")
    exercise_4_model_ensemble()

    print("\n>>> Exercise 5: Real-time Object Counting")
    exercise_5_object_counting()

    print("\nAll exercises completed successfully.")
