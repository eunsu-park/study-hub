# Practical Projects

## Learning Objectives

After completing this lesson, you will be able to:

1. Design and implement a document scanner that detects edges, applies perspective transformation, and produces a rectified scan image.
2. Implement a lane detection system using Canny edge detection, Hough transforms, and region-of-interest masking.
3. Build an AR marker detection pipeline using ArUco markers for pose estimation and overlay rendering.
4. Implement a real-time face filter application combining facial landmark detection with image overlay techniques.
5. Integrate multiple computer vision techniques into complete, end-to-end application projects.

---

## Overview

We will implement practical application projects by combining all the OpenCV techniques learned so far. Each project provides step-by-step guidance on creating complete applications by combining multiple technologies.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: All previous chapter content

---

## Table of Contents

1. [Project 1: Document Scanner](#project-1-document-scanner)
2. [Project 2: Lane Detection](#project-2-lane-detection)
3. [Project 3: AR Marker Detection](#project-3-ar-marker-detection)
4. [Project 4: Real-time Face Filter](#project-4-real-time-face-filter)
5. [Project 5: Object Tracking System](#project-5-object-tracking-system)
6. [Exercises and Extension Ideas](#exercises-and-extension-ideas)

---

## Project 1: Document Scanner

A mobile phone photo of a document is almost always skewed and perspective-distorted. The document scanner project addresses this by automatically detecting the document boundary and applying a perspective correction, producing a clean, axis-aligned image suitable for OCR or archiving — replacing a flatbed scanner with a smartphone camera.

### Project Overview

```
Document Scanner:
Transform photographed documents into aligned scan images

┌──────────────────┐        ┌──────────────────┐
│   Captured Doc   │        │  Scanned Result  │
│  /‾‾‾‾‾‾‾‾‾‾‾\   │        │ ┌──────────────┐ │
│ /             \  │  ──▶   │ │              │ │
│ \             /  │        │ │   Document   │ │
│  \___________/   │        │ │   Content    │ │
│                  │        │ └──────────────┘ │
└──────────────────┘        └──────────────────┘
    Tilted Original              Aligned Result

Technologies used:
- Edge detection (Canny)
- Contour detection (findContours)
- Polygon approximation (approxPolyDP)
- Perspective transform (warpPerspective)
- Binarization (adaptiveThreshold)
```

### Step-by-step Implementation

```python
import cv2
import numpy as np

class DocumentScanner:
    """Document Scanner"""

    def __init__(self):
        pass

    def order_points(self, pts):
        """Order 4 points in order (top-left, top-right, bottom-right, bottom-left)"""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Top-left: smallest x+y sum
        # Bottom-right: largest x+y sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right: smallest y-x difference
        # Bottom-left: largest y-x difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def four_point_transform(self, image, pts):
        """Align document using perspective transform"""
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Calculate new image size
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # Target coordinates
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        return warped

    def find_document_contour(self, image):
        """Find document contour"""
        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Blur first to suppress texture noise inside the document

        # Edge detection — thresholds 75/200 chosen for documents: high lower bound avoids texture,
        # high upper bound ensures only strong edges (document border) are marked as definite
        edged = cv2.Canny(blur, 75, 200)

        # Morphological operations to connect edges — dilate then erode closes small gaps in the border line
        # caused by shadows or worn edges, making the contour a closed loop
        kernel = np.ones((5, 5), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)
        edged = cv2.erode(edged, kernel, iterations=1)

        # Contour detection
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Sort by area descending — the document is almost always the largest object in the frame,
        # so inspecting only the top 5 avoids processing hundreds of small contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        document_contour = None
        for contour in contours[:5]:  # Check top 5 only
            peri = cv2.arcLength(contour, True)
            # 2% of perimeter as epsilon — tight enough to reject rounded shapes,
            # loose enough to handle slightly curved document edges
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                document_contour = approx
                break

        return document_contour, edged

    def enhance_document(self, image):
        """Enhance document image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding — chosen over global Otsu because documents often have uneven lighting
        # (e.g. a shadow across one corner); adaptive threshold computes a local threshold per 11×11 region,
        # making text readable even where the background is darker
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2   # blockSize=11 (local window), C=2 (subtract 2 from weighted mean to fine-tune threshold)
        )

        # Or OTSU thresholding
        # _, binary = cv2.threshold(gray, 0, 255,
        #                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def scan(self, image, enhance=True):
        """Complete document scanning process"""
        original = image.copy()
        height, width = image.shape[:2]

        # Resize for processing (maintain ratio)
        ratio = 500.0 / height
        resized = cv2.resize(image, None, fx=ratio, fy=ratio)

        # Find document contour
        contour, edged = self.find_document_contour(resized)

        if contour is None:
            print("Document contour not found")
            return None, None

        # Convert coordinates to original size
        contour = contour.reshape(4, 2) / ratio

        # Perspective transform
        scanned = self.four_point_transform(original, contour)

        # Document enhancement (optional)
        if enhance:
            scanned = self.enhance_document(scanned)

        return scanned, contour

    def visualize(self, image, contour):
        """Visualize results"""
        vis = image.copy()
        if contour is not None:
            cv2.drawContours(vis, [contour.astype(int)], -1, (0, 255, 0), 3)

            # Mark corner points
            for point in contour:
                cv2.circle(vis, tuple(point.astype(int)), 10, (0, 0, 255), -1)

        return vis

# Usage example
scanner = DocumentScanner()

# Load image
img = cv2.imread('document_photo.jpg')

# Scan
scanned, contour = scanner.scan(img, enhance=True)

if scanned is not None:
    # Visualize results
    vis = scanner.visualize(img, contour)

    cv2.imshow('Original with Contour', vis)
    cv2.imshow('Scanned', scanned)
    cv2.waitKey(0)

    # Save
    cv2.imwrite('scanned_document.jpg', scanned)
```

### Real-time Document Scanner

```python
import cv2
import numpy as np

def realtime_document_scanner():
    """Real-time document scanner"""

    scanner = DocumentScanner()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Document contour detection
        height = frame.shape[0]
        ratio = 500.0 / height
        resized = cv2.resize(frame, None, fx=ratio, fy=ratio)

        contour, _ = scanner.find_document_contour(resized)

        display = frame.copy()

        if contour is not None:
            # Convert to original size
            contour = (contour.reshape(4, 2) / ratio).astype(int)

            # Draw contour
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 3)

            # Guide text
            cv2.putText(display, "Press 's' to scan", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Document not detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Document Scanner', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and contour is not None:
            # Perform scan
            scanned, _ = scanner.scan(frame)
            if scanned is not None:
                cv2.imshow('Scanned', scanned)
                cv2.imwrite('scanned.jpg', scanned)

    cap.release()
    cv2.destroyAllWindows()

# Run
# realtime_document_scanner()
```

---

## Project 2: Lane Detection

Lane detection is one of the core building blocks of ADAS (Advanced Driver Assistance Systems) and autonomous driving. The challenge is that lanes are thin, often faded or partially occluded, and must be detected in real time. This project uses classical CV techniques — color filtering, edge detection, Hough transforms — because they are fast, interpretable, and do not require a labeled training dataset.

### Project Overview

```
Lane Detection:
Detect and visualize lanes in road images

┌────────────────────────────────────┐
│            Road Image              │
│                                    │
│     ╲                    ╱         │
│      ╲      Lane       ╱          │
│       ╲              ╱            │
│        ╲  Detection ╱              │
│         ╲        ╱                │
│          ╲      ╱                 │
│           ╲    ╱                  │
│            ╲  ╱                   │
└────────────────────────────────────┘

Processing Pipeline:
1. Region of Interest (ROI) setup
2. Color space conversion (HSV)
3. White/yellow mask generation
4. Canny edge detection
5. Hough transform for line detection
6. Lane synthesis
```

### Step-by-step Implementation

```python
import cv2
import numpy as np

class LaneDetector:
    """Lane Detector"""

    def __init__(self):
        pass

    def region_of_interest(self, img):
        """Region of interest masking (road area only)"""
        height, width = img.shape[:2]

        # Trapezoidal ROI
        vertices = np.array([[
            (int(width * 0.1), height),           # Bottom-left
            (int(width * 0.4), int(height * 0.6)), # Top-left
            (int(width * 0.6), int(height * 0.6)), # Top-right
            (int(width * 0.9), height)            # Bottom-right
        ]], dtype=np.int32)

        mask = np.zeros_like(img)

        if len(img.shape) == 3:
            cv2.fillPoly(mask, vertices, (255, 255, 255))
        else:
            cv2.fillPoly(mask, vertices, 255)

        masked = cv2.bitwise_and(img, mask)
        return masked

    def color_filter(self, img):
        """Color filter (white/yellow lanes)"""
        # HSV is used instead of BGR/RGB because hue is separable from brightness;
        # a white lane in shadow has the same hue/saturation but lower value — RGB thresholds fail here
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # White mask: any hue (0-255), very low saturation (near-gray), high brightness
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Yellow mask: hue 15-35° covers yellow, saturation >80 excludes washed-out colors,
        # value >100 excludes shadows
        lower_yellow = np.array([15, 80, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Apply mask
        filtered = cv2.bitwise_and(img, img, mask=combined_mask)

        return filtered, combined_mask

    def detect_edges(self, img):
        """Edge detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def detect_lines(self, edges):
        """Line detection using Hough transform"""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,              # 1-pixel distance resolution — finer is more accurate but slower
            theta=np.pi/180,    # 1-degree angular resolution — sufficient for lane angles
            threshold=50,       # Minimum votes: requires 50 edge pixels to agree on a line; filters noise
            minLineLength=50,   # Reject short segments that are likely noise or road debris
            maxLineGap=150      # Allow gaps up to 150px — handles dashed lane markings as one line
        )
        return lines

    def separate_lines(self, lines, img_width):
        """Separate left/right lanes"""
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        center = img_width / 2

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Reject near-horizontal lines (|slope| < 0.3): lane lines are always angled;
            # horizontal segments are typically road cracks, shadows, or distant markings
            if abs(slope) < 0.3:
                continue

            # Left lane: negative slope (goes up-left in image coords) AND both endpoints left of center;
            # the position check prevents misclassifying a right-side diagonal that crosses center
            if slope < 0 and x1 < center and x2 < center:
                left_lines.append(line[0])
            elif slope > 0 and x1 > center and x2 > center:
                right_lines.append(line[0])

        return left_lines, right_lines

    def average_line(self, lines, img_height):
        """Average multiple line segments into one line"""
        if len(lines) == 0:
            return None

        x_coords = []
        y_coords = []

        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        # Fit x as a function of y (not the usual y=f(x)) because lane lines are nearly vertical
        # and would cause numerical instability (infinite slope) in standard linear regression
        poly = np.polyfit(y_coords, x_coords, deg=1)

        # Set y range
        y1 = img_height
        y2 = int(img_height * 0.6)

        # Calculate x coordinates
        x1 = int(np.polyval(poly, y1))
        x2 = int(np.polyval(poly, y2))

        return [x1, y1, x2, y2]

    def draw_lanes(self, img, left_line, right_line):
        """Draw lanes"""
        overlay = np.zeros_like(img)

        # Draw lanes
        if left_line is not None:
            cv2.line(overlay, (left_line[0], left_line[1]),
                    (left_line[2], left_line[3]), (0, 0, 255), 10)

        if right_line is not None:
            cv2.line(overlay, (right_line[0], right_line[1]),
                    (right_line[2], right_line[3]), (0, 0, 255), 10)

        # Fill lane area
        if left_line is not None and right_line is not None:
            pts = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], np.int32)

            cv2.fillPoly(overlay, [pts], (0, 255, 0))

        # Blend with original
        result = cv2.addWeighted(img, 1, overlay, 0.3, 0)

        return result

    def detect(self, img):
        """Complete lane detection pipeline"""
        height, width = img.shape[:2]

        # 1. Color filtering
        filtered, color_mask = self.color_filter(img)

        # 2. Edge detection
        edges = self.detect_edges(filtered)

        # 3. Apply ROI
        roi_edges = self.region_of_interest(edges)

        # 4. Line detection
        lines = self.detect_lines(roi_edges)

        # 5. Separate left/right lanes
        left_lines, right_lines = self.separate_lines(lines, width)

        # 6. Calculate average lanes
        left_lane = self.average_line(left_lines, height)
        right_lane = self.average_line(right_lines, height)

        # 7. Visualize results
        result = self.draw_lanes(img, left_lane, right_lane)

        return result, {
            'edges': roi_edges,
            'color_mask': color_mask,
            'left_lane': left_lane,
            'right_lane': right_lane
        }

# Usage example
detector = LaneDetector()

# Lane detection from image
img = cv2.imread('road.jpg')
result, debug = detector.detect(img)

cv2.imshow('Lane Detection', result)
cv2.imshow('Edges', debug['edges'])
cv2.waitKey(0)
```

### Video Lane Detection

```python
import cv2
import numpy as np

def video_lane_detection(video_path):
    """Lane detection from video"""

    detector = LaneDetector()
    cap = cv2.VideoCapture(video_path)

    # Output video setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('lane_output.mp4', fourcc, fps, (width, height))

    # Previous frame lanes (for smoothing)
    prev_left = None
    prev_right = None
    # Exponential moving average: alpha=0.7 weights the previous frame heavily, so lane lines
    # move smoothly even when Hough detects a slightly different position each frame
    # (lower alpha = more smoothing, higher lag; higher alpha = more jitter, faster response)
    alpha = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, debug = detector.detect(frame)

        # Lane smoothing (prevent abrupt changes)
        left = debug['left_lane']
        right = debug['right_lane']

        if prev_left is not None and left is not None:
            left = [int(alpha * prev_left[i] + (1 - alpha) * left[i])
                    for i in range(4)]
        if prev_right is not None and right is not None:
            right = [int(alpha * prev_right[i] + (1 - alpha) * right[i])
                     for i in range(4)]

        prev_left = left
        prev_right = right

        # Redraw with smoothed lanes
        result = detector.draw_lanes(frame, left, right)

        out.write(result)
        cv2.imshow('Lane Detection', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Run
# video_lane_detection('driving.mp4')
```

---

## Project 3: AR Marker Detection

Augmented reality requires knowing exactly where a known reference point is in 3D space from a single 2D image. AR markers solve this by encoding a unique ID in a high-contrast square pattern that is easy to detect even under varying lighting. Knowing the four corners of a marker whose real-world size is known is enough to compute the camera pose and anchor 3D content precisely.

### Project Overview

```
AR Marker Detection:
Detect square markers in images and composite 3D objects

Marker structure:
┌────────────────────┐
│ ██████████████████ │
│ █                █ │
│ █  ████    ████  █ │
│ █  ████    ████  █ │
│ █                █ │
│ █  ████████████  █ │
│ █  ████████████  █ │
│ █                █ │
│ ██████████████████ │
└────────────────────┘

Processing steps:
1. Square contour detection
2. Normalize marker with perspective transform
3. Marker ID recognition
4. Project 3D object using homography
```

### Step-by-step Implementation

```python
import cv2
import numpy as np

class ARMarkerDetector:
    """AR Marker Detector"""

    def __init__(self, marker_size=100):
        self.marker_size = marker_size

    def order_points(self, pts):
        """Order 4 points in sequence"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect

    def find_markers(self, img):
        """Find marker candidates"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding — markers are black-on-white and must be detectable under varying
        # lighting; local thresholding handles shadows that would make global Otsu miss part of the border
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Invert so marker interior (black) becomes white foreground
            11, 2
        )

        # RETR_LIST retrieves all contours without hierarchy — faster than RETR_TREE when we
        # don't need parent/child relationships (markers can be nested but we check them individually)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)

        markers = []

        for contour in contours:
            # Area filter: too small (<1000px²) = noise; too large (>50% of image) = frame border
            area = cv2.contourArea(contour)
            if area < 1000 or area > img.shape[0] * img.shape[1] * 0.5:
                continue

            # 4% epsilon — more lenient than the document scanner (2%) because marker edges may
            # be blurry or perspective-distorted, making the polygon less regular
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            # Only quadrilaterals
            if len(approx) == 4:
                # Convexity check rejects non-square quads (e.g. an L-shape that approximates to 4 points)
                # Real markers are always convex squares/rectangles
                if cv2.isContourConvex(approx):
                    markers.append(approx.reshape(4, 2))

        return markers, binary

    def get_marker_transform(self, corners):
        """Transform matrix for marker normalization"""
        ordered = self.order_points(corners.astype(np.float32))

        dst = np.array([
            [0, 0],
            [self.marker_size - 1, 0],
            [self.marker_size - 1, self.marker_size - 1],
            [0, self.marker_size - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered, dst)
        return M, ordered

    def decode_marker(self, warped):
        """Decode marker ID (simple example)"""
        # Convert to grayscale
        if len(warped.shape) == 3:
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Binarization
        _, binary = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

        # Divide into 5x5 grid (edges are black border)
        grid_size = self.marker_size // 5
        grid = np.zeros((5, 5), dtype=np.uint8)

        for i in range(5):
            for j in range(5):
                cell = binary[i*grid_size:(i+1)*grid_size,
                             j*grid_size:(j+1)*grid_size]
                # Determine 0/1 based on cell average brightness
                grid[i, j] = 1 if np.mean(cell) > 127 else 0

        # Simple ID calculation (inner 3x3 region)
        inner = grid[1:4, 1:4]
        marker_id = 0
        for i in range(3):
            for j in range(3):
                marker_id = marker_id * 2 + inner[i, j]

        return marker_id, grid

    def draw_cube(self, img, corners, size=50):
        """Draw 3D cube on marker"""
        # 4 points of marker plane
        corners = self.order_points(corners.astype(np.float32))

        # Bottom face coordinates
        bottom = corners.astype(int)

        # Calculate top face coordinates (simple approximation using homography)
        center = np.mean(corners, axis=0)

        # Top face shrinks toward marker center + moves up
        scale = 0.7
        offset = np.array([0, -size])  # Move up

        top = []
        for pt in corners:
            vec = pt - center
            new_pt = center + vec * scale + offset
            top.append(new_pt.astype(int))
        top = np.array(top)

        # Draw faces (semi-transparent)
        overlay = img.copy()

        # Top face (red)
        cv2.fillPoly(overlay, [top], (0, 0, 200))

        # Side faces (green)
        for i in range(4):
            pts = np.array([bottom[i], bottom[(i+1)%4],
                           top[(i+1)%4], top[i]])
            cv2.fillPoly(overlay, [pts], (0, 200, 0))

        # Blend
        result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        # Draw edges
        for i in range(4):
            cv2.line(result, tuple(bottom[i]), tuple(bottom[(i+1)%4]),
                    (255, 255, 255), 2)
            cv2.line(result, tuple(top[i]), tuple(top[(i+1)%4]),
                    (255, 255, 255), 2)
            cv2.line(result, tuple(bottom[i]), tuple(top[i]),
                    (255, 255, 255), 2)

        return result

    def detect(self, img):
        """Marker detection and AR rendering"""
        result = img.copy()

        markers, binary = self.find_markers(img)

        detected_markers = []

        for corners in markers:
            # Normalize marker
            M, ordered = self.get_marker_transform(corners)
            warped = cv2.warpPerspective(img, M,
                                         (self.marker_size, self.marker_size))

            # Decode marker ID
            marker_id, grid = self.decode_marker(warped)

            # Border check (edges should be black)
            border_check = (grid[0, :].sum() + grid[4, :].sum() +
                           grid[:, 0].sum() + grid[:, 4].sum())

            if border_check < 5:  # Mostly black
                detected_markers.append({
                    'id': marker_id,
                    'corners': ordered
                })

                # Draw 3D cube
                result = self.draw_cube(result, ordered)

                # Display ID
                center = np.mean(ordered, axis=0).astype(int)
                cv2.putText(result, f"ID: {marker_id}", tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return result, detected_markers, binary

# Usage example
detector = ARMarkerDetector()

# Detect markers from image
img = cv2.imread('ar_marker.jpg')
result, markers, binary = detector.detect(img)

print(f"Detected markers: {len(markers)}")
for m in markers:
    print(f"  ID: {m['id']}")

cv2.imshow('AR Detection', result)
cv2.imshow('Binary', binary)
cv2.waitKey(0)
```

### Using ArUco Markers (OpenCV Built-in)

```python
import cv2
import numpy as np

def aruco_marker_detection():
    """OpenCV ArUco marker detection"""

    # Select ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

        # Visualize results
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, corner in enumerate(corners):
                # Draw cube or axis on each marker
                # (if camera calibration is available)
                pass

        cv2.imshow('ArUco Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def generate_aruco_marker(marker_id=0, size=200):
    """Generate ArUco marker"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)

    cv2.imwrite(f'aruco_marker_{marker_id}.png', marker_img)
    return marker_img

# Generate marker
# marker = generate_aruco_marker(0)
# cv2.imshow('Marker', marker)
```

---

## Project 4: Real-time Face Filter

Face filters (popularized by Snapchat and Instagram) work by anchoring virtual objects to specific facial features. The key challenge is robustly tracking landmark positions across frames so that overlays stay stable even when the head moves. This project uses a 68-point facial landmark model because it provides enough precision to anchor accessories like sunglasses and hats to the correct anatomical positions.

### Project Overview

```
Real-time Face Filter:
Apply filter effects based on facial landmarks

┌────────────────────────────────────┐
│                                    │
│        Sunglasses Filter           │
│       /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\           │
│      │  ●────────●   │           │
│      │   \      /    │           │
│       \    ▽       /            │
│        \   ∪     /              │
│                                    │
└────────────────────────────────────┘

Technologies used:
- dlib facial landmarks (68 points)
- Transparent image compositing
- Affine/perspective transform
- Real-time processing optimization
```

### Step-by-step Implementation

```python
import cv2
import numpy as np
import dlib

class FaceFilter:
    """Real-time Face Filter"""

    def __init__(self, predictor_path):
        # dlib's HOG-based detector is chosen over Haar cascades because it handles
        # slightly tilted faces better without the high false-positive rate of Haar
        self.detector = dlib.get_frontal_face_detector()
        # 68-point predictor gives anatomically precise landmarks (eye corners, lip edges, etc.)
        # needed to position accessories correctly; a 5-point model is faster but lacks the detail
        self.predictor = dlib.shape_predictor(predictor_path)

        # Filter images
        self.filters = {}

    def load_filter(self, name, image_path, alpha_path=None):
        """Load filter image (PNG with alpha recommended)"""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img.shape[2] == 4:
            # Already has alpha channel
            self.filters[name] = img
        else:
            # Add alpha channel (make white background transparent)
            if alpha_path:
                alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(gray, 250, 255,
                                         cv2.THRESH_BINARY_INV)

            b, g, r = cv2.split(img)
            self.filters[name] = cv2.merge([b, g, r, alpha])

    def get_landmarks(self, img, face):
        """Extract facial landmarks"""
        shape = self.predictor(img, face)
        landmarks = np.array([[shape.part(i).x, shape.part(i).y]
                              for i in range(68)])
        return landmarks

    def overlay_image(self, background, overlay, x, y):
        """Composite transparent image"""
        h, w = overlay.shape[:2]

        # Boundary check
        if x < 0:
            overlay = overlay[:, -x:]
            w = overlay.shape[1]
            x = 0
        if y < 0:
            overlay = overlay[-y:, :]
            h = overlay.shape[0]
            y = 0

        bh, bw = background.shape[:2]
        if x + w > bw:
            overlay = overlay[:, :bw - x]
            w = overlay.shape[1]
        if y + h > bh:
            overlay = overlay[:bh - y, :]
            h = overlay.shape[0]

        if w <= 0 or h <= 0:
            return background

        # Alpha blending: composite = alpha * foreground + (1 - alpha) * background
        # Using the PNG's alpha channel means partially transparent pixels blend smoothly
        # rather than showing a harsh edge around the filter image
        overlay_rgb = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0  # Normalize alpha to [0,1] for the blend formula

        roi = background[y:y+h, x:x+w]

        for c in range(3):
            roi[:, :, c] = (alpha * overlay_rgb[:, :, c] +
                           (1 - alpha) * roi[:, :, c])

        background[y:y+h, x:x+w] = roi

        return background

    def apply_sunglasses(self, img, landmarks, filter_img):
        """Apply sunglasses filter"""
        # Eye coordinates
        left_eye = landmarks[36:42].mean(axis=0).astype(int)
        right_eye = landmarks[42:48].mean(axis=0).astype(int)

        # Eye distance and angle
        eye_width = np.linalg.norm(right_eye - left_eye)
        eye_center = ((left_eye + right_eye) / 2).astype(int)
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                      right_eye[0] - left_eye[0]))

        # 2.5× eye width gives sunglasses that extend slightly past the face edges — matching
        # how real sunglasses are proportioned relative to inter-pupillary distance
        filter_width = int(eye_width * 2.5)
        filter_height = int(filter_width * filter_img.shape[0] /
                           filter_img.shape[1])  # Preserve filter's original aspect ratio

        resized_filter = cv2.resize(filter_img, (filter_width, filter_height))

        # Rotate
        M = cv2.getRotationMatrix2D((filter_width // 2, filter_height // 2),
                                    -angle, 1)
        rotated_filter = cv2.warpAffine(resized_filter, M,
                                        (filter_width, filter_height),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0, 0))

        # Calculate position
        x = eye_center[0] - filter_width // 2
        y = eye_center[1] - filter_height // 2

        # Composite
        result = self.overlay_image(img, rotated_filter, x, y)

        return result

    def apply_hat(self, img, landmarks, filter_img):
        """Apply hat filter"""
        # Forehead position (above eyebrows)
        left_brow = landmarks[17:22].mean(axis=0)
        right_brow = landmarks[22:27].mean(axis=0)

        brow_center = ((left_brow + right_brow) / 2).astype(int)
        brow_width = np.linalg.norm(right_brow - left_brow)

        # Hat size
        hat_width = int(brow_width * 3)
        hat_height = int(hat_width * filter_img.shape[0] /
                        filter_img.shape[1])

        resized_hat = cv2.resize(filter_img, (hat_width, hat_height))

        # Position (place above eyebrows)
        x = brow_center[0] - hat_width // 2
        y = brow_center[1] - hat_height

        # Composite
        result = self.overlay_image(img, resized_hat, x, y)

        return result

    def apply_mustache(self, img, landmarks, filter_img):
        """Apply mustache filter"""
        # Below nose, above mouth
        nose_tip = landmarks[33]
        upper_lip = landmarks[51]

        center = ((nose_tip + upper_lip) / 2).astype(int)

        # Mustache size (based on mouth width)
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        mustache_width = int(mouth_width * 1.5)
        mustache_height = int(mustache_width * filter_img.shape[0] /
                             filter_img.shape[1])

        resized = cv2.resize(filter_img, (mustache_width, mustache_height))

        x = center[0] - mustache_width // 2
        y = center[1] - mustache_height // 2

        result = self.overlay_image(img, resized, x, y)

        return result

    def process(self, img, filter_name='sunglasses'):
        """Apply filter"""
        if filter_name not in self.filters:
            return img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, 0)

        result = img.copy()

        for face in faces:
            landmarks = self.get_landmarks(rgb, face)

            if filter_name == 'sunglasses':
                result = self.apply_sunglasses(result, landmarks,
                                               self.filters[filter_name])
            elif filter_name == 'hat':
                result = self.apply_hat(result, landmarks,
                                        self.filters[filter_name])
            elif filter_name == 'mustache':
                result = self.apply_mustache(result, landmarks,
                                             self.filters[filter_name])

        return result

# Usage example
def realtime_face_filter():
    """Real-time face filter"""

    filter_app = FaceFilter('shape_predictor_68_face_landmarks.dat')

    # Load filters (transparent PNG recommended)
    filter_app.load_filter('sunglasses', 'sunglasses.png')
    # filter_app.load_filter('hat', 'hat.png')
    # filter_app.load_filter('mustache', 'mustache.png')

    cap = cv2.VideoCapture(0)

    current_filter = 'sunglasses'
    filters = list(filter_app.filters.keys())
    filter_idx = 0

    print("Press 'n' to change filter, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Apply filter
        result = filter_app.process(frame, current_filter)

        # Display current filter
        cv2.putText(result, f"Filter: {current_filter}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Filter', result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            filter_idx = (filter_idx + 1) % len(filters)
            current_filter = filters[filter_idx]

    cap.release()
    cv2.destroyAllWindows()

# Run
# realtime_face_filter()
```

---

## Project 5: Object Tracking System

Running a full object detector on every frame is computationally expensive and produces jittery IDs — the same physical object may get a different bounding box on consecutive frames. This project combines background subtraction (cheap, frame-by-frame motion detection) with Kalman filtering (predicts where each object will be next) and the Hungarian algorithm (globally optimal assignment of detections to tracks), which is the classical SORT (Simple Online and Realtime Tracking) architecture.

### Project Overview

```
Object Tracking System:
Multi-object tracking combining background subtraction and Kalman filter

Processing flow:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Frame   │ → │ Background│ → │ Contour │ → │ Kalman  │
│ Input   │    │ Subtract │    │ Detect  │    │ Filter  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                  │
                                                  ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Result  │ ← │ ID      │ ← │Hungarian│ ← │ Predict │
│ Output  │    │ Assign  │    │ Match   │    │ Position│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### Step-by-step Implementation

```python
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanTracker:
    """Kalman filter-based single object tracker"""

    def __init__(self, initial_pos):
        # Initialize Kalman filter
        # State vector: [x, y, vx, vy]
        self.kalman = cv2.KalmanFilter(4, 2)

        # Constant velocity model: next_pos = current_pos + velocity
        # Simple but effective for short prediction horizons (a few frames)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # We only observe (x, y) position, not velocity directly;
        # the filter infers velocity from successive position measurements
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Low process noise (0.03): we trust the motion model — use higher values for erratic objects
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise (1.0): background-subtracted centroids have ~1px localization error;
        # higher trust in measurements than in prediction when detections are available
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1

        # Initial state
        self.kalman.statePre = np.array([
            [initial_pos[0]],
            [initial_pos[1]],
            [0],
            [0]
        ], dtype=np.float32)

        self.kalman.statePost = self.kalman.statePre.copy()

        self.age = 0  # Number of tracked frames
        self.hits = 1  # Successful matches
        self.time_since_update = 0  # Frames since last update

    def predict(self):
        """Predict next position"""
        prediction = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return prediction[:2].flatten()

    def update(self, measurement):
        """Update state with measurement"""
        self.kalman.correct(np.array(measurement, dtype=np.float32))
        self.hits += 1
        self.time_since_update = 0

    def get_state(self):
        """Return current state"""
        return self.kalman.statePost[:2].flatten()


class MultiObjectTracker:
    """Multi-object tracking system"""

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.trackers = []
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        # MOG2 is used instead of simple frame differencing because it models each pixel
        # as a mixture of Gaussians, adapting to gradual lighting changes (clouds, day/night)
        # history=500: uses ~500 frames to learn the background model
        # varThreshold=16: pixels deviating more than √16 std-devs are foreground
        # detectShadows=True: marks shadow pixels separately (gray) rather than treating them as objects
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

    def detect_objects(self, frame):
        """Object detection using background subtraction"""
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Shadow removal
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Contour detection
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                detections.append({
                    'bbox': (x, y, w, h),
                    'center': center
                })

        return detections, fg_mask

    def iou(self, bbox1, bbox2):
        """Calculate IoU (Intersection over Union)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def associate_detections(self, detections):
        """Match detections with trackers (Hungarian algorithm)"""
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.trackers)))

        # Calculate cost matrix (distance-based)
        cost_matrix = np.zeros((len(detections), len(self.trackers)))

        for d, det in enumerate(detections):
            for t, tracker in enumerate(self.trackers):
                pred = tracker['kalman'].predict()
                dist = np.linalg.norm(np.array(det['center']) - pred)
                cost_matrix[d, t] = dist

        # Hungarian algorithm finds the globally optimal assignment in O(n³) — greedy nearest-neighbor
        # assignment can give wrong IDs when two objects cross paths; Hungarian prevents that
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))

        for row, col in zip(row_indices, col_indices):
            # Reject matches where distance > 100px — even the "best" assignment may be wrong
            # if an object disappeared and a new one appeared far away
            if cost_matrix[row, col] < 100:  # Distance threshold
                matched.append((row, col))
                unmatched_detections.remove(row)
                unmatched_trackers.remove(col)

        return matched, unmatched_detections, unmatched_trackers

    def update(self, frame):
        """Update tracking"""
        # Object detection
        detections, fg_mask = self.detect_objects(frame)

        # Prediction
        for tracker in self.trackers:
            tracker['kalman'].predict()

        # Matching
        matched, unmatched_dets, unmatched_trks = \
            self.associate_detections(detections)

        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx]['kalman'].update(
                np.array(detections[det_idx]['center'])
            )
            self.trackers[trk_idx]['bbox'] = detections[det_idx]['bbox']

        # Create new trackers
        for det_idx in unmatched_dets:
            tracker = {
                'id': self.next_id,
                'kalman': KalmanTracker(detections[det_idx]['center']),
                'bbox': detections[det_idx]['bbox'],
                'color': (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
            }
            self.trackers.append(tracker)
            self.next_id += 1

        # Remove old trackers
        self.trackers = [t for t in self.trackers
                        if t['kalman'].time_since_update < self.max_age]

        # Return results
        results = []
        for tracker in self.trackers:
            # min_hits guard: only report tracks that have been confirmed by at least min_hits detections;
            # prevents single-frame false positives from polluting the output with spurious IDs
            if tracker['kalman'].hits >= self.min_hits:
                results.append({
                    'id': tracker['id'],
                    'bbox': tracker['bbox'],
                    'color': tracker['color'],
                    'center': tracker['kalman'].get_state()
                })

        return results, fg_mask

    def draw(self, frame, results):
        """Visualize results"""
        for obj in results:
            x, y, w, h = obj['bbox']
            color = obj['color']

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID: {obj['id']}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display trajectory (center point)
            center = tuple(obj['center'].astype(int))
            cv2.circle(frame, center, 4, color, -1)

        return frame

# Usage example
def multi_object_tracking(video_path):
    """Run multi-object tracking"""

    tracker = MultiObjectTracker()
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracking
        results, fg_mask = tracker.update(frame)

        # Visualization
        output = tracker.draw(frame, results)

        # Display info
        cv2.putText(output, f"Objects: {len(results)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Multi-Object Tracking', output)
        cv2.imshow('Foreground Mask', fg_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run
# multi_object_tracking('traffic.mp4')
```

---

## Exercises and Extension Ideas

### Project 1: Document Scanner Extensions

1. **OCR Integration**: Integrate Tesseract OCR for text extraction
2. **Auto Color Correction**: Improve document readability with histogram equalization
3. **Multi-page Support**: Create PDF from continuous captures
4. **Handwriting Recognition**: Digitize handwritten documents
5. **Receipt Parsing**: Automatically extract amounts, dates, etc.

### Project 2: Lane Detection Extensions

1. **Curved Lane Detection**: 2nd/3rd degree polynomial fitting
2. **Lane Departure Warning**: Compare vehicle center with lane center
3. **Night Mode**: Adjust parameters based on lighting conditions
4. **Multi-lane Detection**: Detect adjacent lanes
5. **Vehicle Detection Integration**: Combine with YOLO for front car detection

### Project 3: AR Marker Extensions

1. **3D Model Rendering**: Display 3D objects with OpenGL integration
2. **Multi-marker Interaction**: Recognize relationships between markers
3. **Markerless AR**: Plane detection-based AR
4. **Game Development**: Simple AR game based on markers
5. **Furniture Placement Simulation**: Place virtual furniture in real space

### Project 4: Face Filter Extensions

1. **Expression Recognition**: Detect smiles, eye blinks to change filters
2. **3D Filters**: 3D transformation matching face pose
3. **Background Replacement**: Replace only background with segmentation
4. **Face Swap**: Exchange faces between two people
5. **Aging Filter**: Face aging/de-aging effects

### Project 5: Object Tracking Extensions

1. **Re-ID Feature**: Re-identify objects that left and re-entered the frame
2. **Speed Measurement**: Calculate actual speed after calibration
3. **Zone Intrusion Detection**: Alert when entering specific areas
4. **Trajectory Analysis**: Analyze movement patterns and detect anomalies
5. **Deep Learning Integration**: Improve accuracy with YOLO + DeepSORT

---

## Next Steps

You have mastered the basics of OpenCV and computer vision. For deeper learning, we recommend the following topics:

### Deep Learning Frameworks
- **PyTorch**: Powerful for research and prototyping
- **TensorFlow/Keras**: Suitable for production deployment
- **ONNX**: Standard for model compatibility

### Advanced Computer Vision
- **Image Segmentation**: U-Net, Mask R-CNN
- **Pose Estimation**: OpenPose, MediaPipe
- **GAN-based Image Generation**: StyleGAN, Pix2Pix
- **3D Vision**: Stereo vision, depth estimation

### Application Domains
- **Autonomous Driving**: SLAM, sensor fusion
- **Medical Imaging**: CT/MRI analysis, disease detection
- **Industrial Inspection**: Defect detection, quality control
- **Security/Surveillance**: Anomaly detection, face recognition

---

## References

- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [PyImageSearch](https://pyimagesearch.com/) - Practical project tutorials
- [Learn OpenCV](https://learnopencv.com/) - Advanced examples
- [Mediapipe](https://google.github.io/mediapipe/) - Google's ML solutions
- [Papers With Code](https://paperswithcode.com/) - Latest research and code
- Bradski, G., & Kaehler, A. (2008). "Learning OpenCV"
- Szeliski, R. (2010). "Computer Vision: Algorithms and Applications"
