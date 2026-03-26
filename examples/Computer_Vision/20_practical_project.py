"""
20. Practical Projects
- Document scanner
- License plate recognition
- Real-time object tracking
- Image panorama
"""

import cv2
import numpy as np


# ============================================================
# Project 1: Document Scanner
# ============================================================

def document_scanner():
    """Document scanner project"""
    print("=" * 60)
    print("Project 1: Document Scanner")
    print("=" * 60)

    # Create simulated document image
    # In practice, use an image captured by camera
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = [150, 150, 150]  # Gray background

    # Tilted document (trapezoid)
    doc_pts = np.array([[150, 100], [650, 80], [700, 520], [100, 550]], np.int32)
    cv2.fillPoly(img, [doc_pts], (255, 255, 255))

    # Simulate document content
    cv2.putText(img, 'DOCUMENT TITLE', (220, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.line(img, (200, 250), (600, 240), (100, 100, 100), 2)
    cv2.line(img, (200, 300), (600, 290), (100, 100, 100), 2)
    cv2.line(img, (200, 350), (550, 340), (100, 100, 100), 2)

    cv2.imwrite('scanner_input.jpg', img)

    # 1. Grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # 3. Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Find the largest rectangular contour
    doc_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:  # Minimum size
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > max_area:
                doc_contour = approx
                max_area = area

    if doc_contour is not None:
        # Sort corners
        pts = doc_contour.reshape(4, 2)
        rect = order_points(pts)

        # Mark result on image
        result_contour = img.copy()
        cv2.drawContours(result_contour, [doc_contour], -1, (0, 255, 0), 3)
        for pt in rect:
            cv2.circle(result_contour, tuple(pt.astype(int)), 10, (0, 0, 255), -1)

        cv2.imwrite('scanner_contour.jpg', result_contour)

        # 5. Perspective transform
        width, height = 500, 700  # Output size
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (width, height))

        # 6. Binarization (scan effect)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        scanned = cv2.adaptiveThreshold(
            warped_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        cv2.imwrite('scanner_warped.jpg', warped)
        cv2.imwrite('scanner_result.jpg', scanned)

        print("Document scanner complete!")
        print("  - scanner_input.jpg: Original")
        print("  - scanner_contour.jpg: Document detection")
        print("  - scanner_warped.jpg: Perspective correction")
        print("  - scanner_result.jpg: Final scan")
    else:
        print("Document not found.")

    print("\nProcessing pipeline:")
    print("  1. Grayscale + Blur")
    print("  2. Canny edge detection")
    print("  3. Contour detection and approximation")
    print("  4. Select quadrilateral document")
    print("  5. Perspective transform")
    print("  6. Binarization (optional)")


def order_points(pts):
    """Sort corner points in [top-left, top-right, bottom-right, bottom-left] order"""
    rect = np.zeros((4, 2), dtype=np.float32)

    # Smallest sum: top-left
    # Largest sum: bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Smallest difference: top-right
    # Largest difference: bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# ============================================================
# Project 2: License Plate Recognition (Concept)
# ============================================================

def license_plate_recognition():
    """License plate recognition project (concept)"""
    print("\n" + "=" * 60)
    print("Project 2: License Plate Recognition")
    print("=" * 60)

    # Simulated license plate image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # Vehicle shape
    cv2.rectangle(img, (100, 100), (500, 350), (80, 80, 80), -1)
    cv2.rectangle(img, (120, 120), (480, 250), (60, 60, 60), -1)

    # License plate
    plate_x, plate_y = 200, 280
    plate_w, plate_h = 200, 50
    cv2.rectangle(img, (plate_x, plate_y), (plate_x+plate_w, plate_y+plate_h),
                 (255, 255, 255), -1)
    cv2.rectangle(img, (plate_x, plate_y), (plate_x+plate_w, plate_y+plate_h),
                 (0, 0, 0), 2)
    cv2.putText(img, '12AB3456', (plate_x+20, plate_y+35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imwrite('lpr_input.jpg', img)

    print("\nLicense plate recognition pipeline:")
    print("""
1. Plate Detection
   - Haar Cascade (trained classifier)
   - DNN (YOLO, SSD)
   - Edge-based detection

2. Plate Region Extraction
   - Contour detection
   - Perspective correction

3. Character Segmentation
   - Binarization
   - Separate each character by contour
   - Connected component analysis

4. Character Recognition (OCR)
   - Tesseract OCR
   - DNN-based recognition
   - Template matching

5. Post-processing
   - Format validation
   - Noise removal
""")

    code = '''
# License plate recognition code example
import cv2
import pytesseract

# 1. Plate detection
plate_cascade = cv2.CascadeClassifier('haarcascade_plate.xml')
plates = plate_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in plates:
    # 2. Extract plate region
    plate_img = gray[y:y+h, x:x+w]

    # 3. Preprocessing
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2)
    _, thresh = cv2.threshold(plate_img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. OCR
    text = pytesseract.image_to_string(thresh, config='--psm 7')
    print(f"License plate: {text.strip()}")
'''
    print(code)

    print("\nRequired libraries:")
    print("  - pytesseract: pip install pytesseract")
    print("  - Tesseract-OCR: System installation required")


# ============================================================
# Project 3: Real-time Object Tracking
# ============================================================

def object_tracking_project():
    """Real-time object tracking project"""
    print("\n" + "=" * 60)
    print("Project 3: Real-time Object Tracking")
    print("=" * 60)

    # Generate simulation frame sequence
    frames = []
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = [50, 50, 50]

        # Moving object
        x = 100 + i * 15
        y = 240 + int(50 * np.sin(i * 0.3))
        cv2.circle(frame, (x, y), 40, (0, 200, 0), -1)

        # Static object
        cv2.rectangle(frame, (400, 100), (500, 200), (200, 0, 0), -1)

        frames.append(frame)

    # Select tracking target in first frame
    first_frame = frames[0].copy()
    bbox = (60, 200, 80, 80)  # x, y, w, h
    cv2.rectangle(first_frame, (bbox[0], bbox[1]),
                 (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
    cv2.imwrite('tracking_init.jpg', first_frame)

    # Tracking simulation
    print("\nTracking simulation (KCF Tracker concept)")

    # Tracking result visualization
    result_frame = frames[15].copy()
    new_x = 100 + 15 * 15
    new_y = 240 + int(50 * np.sin(15 * 0.3))
    cv2.rectangle(result_frame, (new_x-40, new_y-40),
                 (new_x+40, new_y+40), (0, 255, 0), 2)
    cv2.putText(result_frame, 'Tracking', (new_x-30, new_y-50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite('tracking_result.jpg', result_frame)

    print("\nComplete tracking code:")
    code = '''
import cv2

# Video capture
cap = cv2.VideoCapture(0)  # or 'video.mp4'

# Read first frame
ret, frame = cap.read()

# Select ROI (drag with mouse)
bbox = cv2.selectROI("Select Object", frame, fromCenter=False)
cv2.destroyAllWindows()

# Create tracker (multiple options)
# tracker = cv2.TrackerBoosting_create()
# tracker = cv2.TrackerMIL_create()
tracker = cv2.TrackerKCF_create()
# tracker = cv2.TrackerCSRT_create()  # More accurate

# Initialize
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
        cv2.putText(frame, "Tracking", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
    print(code)

    print("\nTracker comparison:")
    trackers = [
        ('KCF', 'Fast, general performance'),
        ('CSRT', 'Accurate, somewhat slow'),
        ('MOSSE', 'Very fast, lower accuracy'),
        ('MedianFlow', 'For predictable motion'),
    ]
    for name, desc in trackers:
        print(f"  {name}: {desc}")


# ============================================================
# Project 4: Image Panorama
# ============================================================

def panorama_stitching():
    """Image panorama project"""
    print("\n" + "=" * 60)
    print("Project 4: Image Panorama")
    print("=" * 60)

    # Create overlapping images for simulation
    # Background
    full_scene = np.zeros((300, 800, 3), dtype=np.uint8)
    full_scene[:] = [200, 200, 200]

    # Place objects in the scene
    cv2.circle(full_scene, (100, 150), 50, (0, 0, 150), -1)
    cv2.rectangle(full_scene, (250, 100), (350, 200), (0, 150, 0), -1)
    cv2.circle(full_scene, (500, 150), 60, (150, 0, 0), -1)
    cv2.rectangle(full_scene, (650, 80), (750, 220), (150, 150, 0), -1)

    # Two images with overlapping region
    img1 = full_scene[:, :450].copy()
    img2 = full_scene[:, 300:].copy()

    cv2.imwrite('panorama_img1.jpg', img1)
    cv2.imwrite('panorama_img2.jpg', img2)

    print("Images for stitching generated")

    # Feature detection and matching
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB features
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is not None and des2 is not None:
        # Matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        print(f"Good matches: {len(good)}")

        if len(good) >= 4:
            # Compute homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Generate panorama
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]

                # Calculate result image size
                corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, H)

                all_corners = np.concatenate([
                    np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
                    transformed
                ])

                x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
                x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

                translation = np.array([
                    [1, 0, -x_min],
                    [0, 1, -y_min],
                    [0, 0, 1]
                ])

                # Warping and compositing
                result_width = x_max - x_min
                result_height = y_max - y_min

                result = cv2.warpPerspective(img2, translation @ H,
                                            (result_width, result_height))
                result[-y_min:-y_min+h1, -x_min:-x_min+w1] = img1

                cv2.imwrite('panorama_result.jpg', result)
                print("Panorama generation complete: panorama_result.jpg")

    print("\nPanorama generation pipeline:")
    print("  1. Feature detection (SIFT/ORB)")
    print("  2. Feature matching")
    print("  3. Homography computation")
    print("  4. Image warping")
    print("  5. Blending (smooth boundaries)")

    # Using OpenCV Stitcher class
    print("\nUsing OpenCV Stitcher:")
    code = '''
# Simple method: cv2.Stitcher
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, panorama = stitcher.stitch([img1, img2, img3])

if status == cv2.Stitcher_OK:
    cv2.imwrite('panorama.jpg', panorama)
else:
    print(f"Stitching failed: {status}")
'''
    print(code)


# ============================================================
# Project 5: AR Marker
# ============================================================

def ar_marker_project():
    """AR marker project (concept)"""
    print("\n" + "=" * 60)
    print("Project 5: AR Marker-based Augmented Reality")
    print("=" * 60)

    # ArUco marker generation (simulation)
    marker_size = 200

    # Simulated marker image
    marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker[:] = 255

    # Simple pattern (actual ArUco markers are more complex)
    cv2.rectangle(marker, (10, 10), (190, 190), 0, 10)
    cv2.rectangle(marker, (40, 40), (80, 80), 0, -1)
    cv2.rectangle(marker, (120, 40), (160, 80), 0, -1)
    cv2.rectangle(marker, (40, 120), (80, 160), 0, -1)
    cv2.rectangle(marker, (120, 120), (160, 160), 0, -1)
    cv2.rectangle(marker, (80, 80), (120, 120), 0, -1)

    cv2.imwrite('ar_marker.jpg', marker)

    print("\nArUco markers:")
    print("  - Built-in marker system in OpenCV")
    print("  - Automatic detection and ID recognition")
    print("  - Pose estimation from 4 corners")

    code = '''
# ArUco marker generation
import cv2

# Select dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate marker (ID=42, size=200x200)
marker = cv2.aruco.generateImageMarker(aruco_dict, 42, 200)
cv2.imwrite('marker_42.png', marker)

# Marker detection
detector = cv2.aruco.ArucoDetector(aruco_dict)
corners, ids, rejected = detector.detectMarkers(gray)

# Draw markers
cv2.aruco.drawDetectedMarkers(image, corners, ids)

# Pose estimation (camera calibration required)
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
    corners, marker_length, camera_matrix, dist_coeffs
)

# Draw coordinate axes
for rvec, tvec in zip(rvecs, tvecs):
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
'''
    print(code)

    print("\nAR applications:")
    print("  - 3D object overlay")
    print("  - Virtual furniture placement")
    print("  - Games/Education")


# ============================================================
# Project Structure Guide
# ============================================================

def project_structure_guide():
    """Project structure guide"""
    print("\n" + "=" * 60)
    print("Computer Vision Project Structure Guide")
    print("=" * 60)

    print("""
Recommended project structure:

project/
├── main.py           # Main execution file
├── config.py         # Settings (paths, parameters)
├── requirements.txt  # Dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── detection.py      # Object detection
│   ├── preprocessing.py  # Preprocessing
│   ├── tracking.py       # Tracking
│   └── utils.py          # Utilities
│
├── models/           # Trained model files
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   └── ...
│
├── data/             # Input data
│   ├── images/
│   └── videos/
│
├── output/           # Result storage
│   ├── results/
│   └── logs/
│
└── tests/            # Test code
    └── test_detection.py
""")

    print("\nDevelopment tips:")
    print("  1. Modularize: Separate by functionality")
    print("  2. Config files: Avoid hardcoding")
    print("  3. Logging: Easier debugging")
    print("  4. Testing: Write unit tests")
    print("  5. Documentation: Function/class docstrings")


def main():
    """Main function"""
    # Project 1: Document scanner
    document_scanner()

    # Project 2: License plate recognition
    license_plate_recognition()

    # Project 3: Object tracking
    object_tracking_project()

    # Project 4: Panorama
    panorama_stitching()

    # Project 5: AR marker
    ar_marker_project()

    # Project structure guide
    project_structure_guide()

    print("\n" + "=" * 60)
    print("Practical project demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
