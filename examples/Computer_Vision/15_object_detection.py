"""
15. Object Detection Basics
- Template Matching
- Haar Cascade
- HOG + SVM (concept)
"""

import cv2
import numpy as np


def create_scene_with_objects():
    """Create a scene image with objects"""
    scene = np.zeros((400, 600, 3), dtype=np.uint8)
    scene[:] = [200, 200, 200]

    # Star-shaped object (target to find)
    def draw_star(img, center, size, color):
        pts = []
        for i in range(5):
            outer = np.radians(i * 72 - 90)
            inner = np.radians(i * 72 + 36 - 90)
            pts.append([int(center[0] + size * np.cos(outer)),
                       int(center[1] + size * np.sin(outer))])
            pts.append([int(center[0] + size * 0.4 * np.cos(inner)),
                       int(center[1] + size * 0.4 * np.sin(inner))])
        cv2.fillPoly(img, [np.array(pts, np.int32)], color)

    # Place multiple stars
    draw_star(scene, (100, 100), 30, (0, 0, 150))
    draw_star(scene, (300, 200), 40, (0, 0, 180))
    draw_star(scene, (500, 300), 35, (0, 0, 160))

    # Distractor objects
    cv2.circle(scene, (200, 300), 40, (150, 0, 0), -1)
    cv2.rectangle(scene, (400, 50), (480, 130), (0, 150, 0), -1)

    # Template (single star)
    template = np.zeros((80, 80, 3), dtype=np.uint8)
    template[:] = [200, 200, 200]
    draw_star(template, (40, 40), 30, (0, 0, 150))

    return scene, template


def template_matching_demo():
    """Template matching demo"""
    print("=" * 50)
    print("Template Matching")
    print("=" * 50)

    scene, template = create_scene_with_objects()

    # Convert to grayscale
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template_gray.shape

    # Template matching methods
    methods = [
        ('TM_CCOEFF', cv2.TM_CCOEFF),
        ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
        ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
        ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED),
    ]

    print("\nMatching method results:")

    for name, method in methods:
        result = cv2.matchTemplate(scene_gray, template_gray, method)

        # Find min/max location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # For SQDIFF, minimum value is optimal
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            score = min_val
        else:
            top_left = max_loc
            score = max_val

        print(f"  {name}: score={score:.4f}, loc={top_left}")

        # Visualize result
        scene_copy = scene.copy()
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(scene_copy, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imwrite(f'template_{name}.jpg', scene_copy)

    print("\nMatching method properties:")
    print("  TM_SQDIFF: Sum of squared differences (lower is better)")
    print("  TM_CCORR: Cross-correlation (higher is better)")
    print("  TM_CCOEFF: Correlation coefficient (higher is better)")
    print("  _NORMED: Normalized version (-1~1 or 0~1)")

    cv2.imwrite('template_scene.jpg', scene)
    cv2.imwrite('template_template.jpg', template)


def multi_scale_template_demo():
    """Multi-scale template matching"""
    print("\n" + "=" * 50)
    print("Multi-scale Template Matching")
    print("=" * 50)

    scene, template = create_scene_with_objects()
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_val = -1
    best_scale = 1.0
    best_loc = (0, 0)

    # Match at various scales
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    for scale in scales:
        # Resize template
        new_w = int(template_gray.shape[1] * scale)
        new_h = int(template_gray.shape[0] * scale)

        if new_w > scene_gray.shape[1] or new_h > scene_gray.shape[0]:
            continue

        resized = cv2.resize(template_gray, (new_w, new_h))

        # Matching
        result = cv2.matchTemplate(scene_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        print(f"  Scale {scale:.2f}: score={max_val:.4f}")

        if max_val > best_val:
            best_val = max_val
            best_scale = scale
            best_loc = max_loc
            best_match = resized.shape

    print(f"\nBest scale: {best_scale}, score={best_val:.4f}")

    # Draw result
    result_img = scene.copy()
    h, w = best_match
    cv2.rectangle(result_img, best_loc, (best_loc[0]+w, best_loc[1]+h), (0, 255, 0), 2)
    cv2.imwrite('multi_scale_result.jpg', result_img)


def find_all_matches_demo():
    """Find all matches"""
    print("\n" + "=" * 50)
    print("Find All Matches")
    print("=" * 50)

    scene, template = create_scene_with_objects()
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template_gray.shape

    # Template matching
    result = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find all locations above threshold
    threshold = 0.6
    locations = np.where(result >= threshold)

    scene_copy = scene.copy()
    match_count = 0

    # Apply NMS to remove duplicates
    boxes = []
    for pt in zip(*locations[::-1]):
        boxes.append([pt[0], pt[1], pt[0]+w, pt[1]+h])

    # Simple NMS
    boxes = np.array(boxes)
    if len(boxes) > 0:
        # Sort by score
        scores = [result[b[1], b[0]] for b in boxes]
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)

            # Calculate overlap with other boxes
            remaining = []
            for j in indices[1:]:
                # IoU calculation (simple version)
                x_overlap = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0]))
                y_overlap = max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
                overlap = x_overlap * y_overlap
                area = w * h
                iou = overlap / area

                if iou < 0.5:  # Keep if overlap < 50%
                    remaining.append(j)

            indices = np.array(remaining)

        # Draw results
        for i in keep:
            pt = (boxes[i][0], boxes[i][1])
            cv2.rectangle(scene_copy, pt, (pt[0]+w, pt[1]+h), (0, 255, 0), 2)
            score = result[pt[1], pt[0]]
            cv2.putText(scene_copy, f'{score:.2f}', (pt[0], pt[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            match_count += 1

    print(f"Threshold: {threshold}")
    print(f"Number of detected objects: {match_count}")

    cv2.imwrite('find_all_matches.jpg', scene_copy)


def haar_cascade_demo():
    """Haar Cascade demo"""
    print("\n" + "=" * 50)
    print("Haar Cascade Detector")
    print("=" * 50)

    # Test image (face simulation)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # Face shape simulation
    cv2.ellipse(img, (200, 150), (50, 60), 0, 0, 360, (180, 150, 130), -1)
    cv2.circle(img, (180, 130), 8, (50, 50, 50), -1)  # Eye
    cv2.circle(img, (220, 130), 8, (50, 50, 50), -1)
    cv2.ellipse(img, (200, 170), (15, 8), 0, 0, 180, (50, 50, 50), 2)  # Mouth

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Cannot find Haar Cascade file.")
    else:
        # Detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        result = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

        print(f"Detected faces: {len(faces)}")

        cv2.imwrite('haar_input.jpg', img)
        cv2.imwrite('haar_result.jpg', result)

    print("\nHaar Cascade parameters:")
    print("  scaleFactor: Image pyramid scale")
    print("  minNeighbors: Minimum neighbor count for confirmed detection")
    print("  minSize: Minimum object size")

    print("\nAvailable Haar Cascades:")
    print(f"  Path: {cv2.data.haarcascades}")
    print("  - haarcascade_frontalface_default.xml")
    print("  - haarcascade_eye.xml")
    print("  - haarcascade_smile.xml")
    print("  - haarcascade_fullbody.xml")


def hog_concept_demo():
    """HOG concept demo"""
    print("\n" + "=" * 50)
    print("HOG (Histogram of Oriented Gradients)")
    print("=" * 50)

    # Test image
    img = np.zeros((128, 64, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # Human shape simulation
    cv2.ellipse(img, (32, 25), (12, 15), 0, 0, 360, (100, 100, 100), -1)  # Head
    cv2.rectangle(img, (20, 40), (44, 90), (100, 100, 100), -1)  # Torso
    cv2.rectangle(img, (18, 90), (30, 125), (100, 100, 100), -1)  # Left leg
    cv2.rectangle(img, (34, 90), (46, 125), (100, 100, 100), -1)  # Right leg

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG descriptor computation
    # winSize: Detection window size
    # blockSize: Block size
    # blockStride: Block stride
    # cellSize: Cell size
    # nbins: Number of histogram bins

    hog = cv2.HOGDescriptor(
        _winSize=(64, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    # Compute HOG features
    features = hog.compute(gray)

    print(f"Input image: {gray.shape}")
    print(f"HOG feature vector size: {features.shape}")

    print("\nHOG properties:")
    print("  - Histogram of gradient orientations")
    print("  - Robust to illumination changes")
    print("  - Effective for pedestrian detection")
    print("  - Used together with SVM")

    # Default HOG pedestrian detector
    hog_detector = cv2.HOGDescriptor()
    hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    print("\nPre-trained HOG+SVM detector:")
    print("  - cv2.HOGDescriptor_getDefaultPeopleDetector()")
    print("  - SVM weights trained for pedestrian detection")

    cv2.imwrite('hog_input.jpg', img)


def detection_comparison():
    """Detection method comparison"""
    print("\n" + "=" * 50)
    print("Object Detection Method Comparison")
    print("=" * 50)

    print("""
    | Method | Advantages | Disadvantages | Use Case |
    |--------|-----------|---------------|----------|
    | Template Matching | Simple, fast | Not rotation/scale invariant | Fixed patterns |
    | Haar Cascade | Fast, face-specialized | Limited accuracy | Face detection |
    | HOG+SVM | Accurate, robust | Slow | Pedestrian detection |
    | Feature Matching | Rotation/scale invariant | Computationally heavy | Object recognition |
    | Deep Learning (YOLO, etc.) | Very accurate | GPU required | General detection |
    """)


def main():
    """Main function"""
    # Template matching
    template_matching_demo()

    # Multi-scale
    multi_scale_template_demo()

    # Find all matches
    find_all_matches_demo()

    # Haar Cascade
    haar_cascade_demo()

    # HOG concept
    hog_concept_demo()

    # Method comparison
    detection_comparison()

    print("\nObject detection basics demo complete!")


if __name__ == '__main__':
    main()
