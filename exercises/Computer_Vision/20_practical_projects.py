"""
Exercise Solutions for Lesson 20: Practical Projects
Computer Vision - Document Scanner, Lane Detection, AR, Face Filters, Tracking

Topics covered (Extension Ideas):
- Document scanner with OCR simulation and auto color correction
- Lane detection with polynomial fitting and departure warning
- AR marker detection with 3D overlay simulation
- Face filter with expression-based filter switching
- Object tracking with re-identification and zone intrusion
"""

import numpy as np


# =============================================================================
# Project 1: Document Scanner Extensions
# =============================================================================

def project_1_document_scanner():
    """
    Document scanner extensions:
    1. OCR simulation (character region detection)
    2. Auto color correction (histogram equalization)
    3. Multi-page document assembly

    Returns:
        dict with processed pages
    """
    np.random.seed(42)

    # Create synthetic document image (white paper with text lines)
    h, w = 200, 150
    doc = np.ones((h, w), dtype=np.uint8) * 220  # White paper

    # Add text-like lines (dark horizontal regions)
    text_lines = []
    for row in range(20, 180, 15):
        line_len = np.random.randint(60, 130)
        start_x = np.random.randint(10, 30)
        doc[row:row+3, start_x:start_x+line_len] = np.random.randint(30, 60)
        text_lines.append((start_x, row, line_len, 3))

    # Add uneven illumination (darker on one side)
    for j in range(w):
        factor = 0.6 + 0.4 * j / w
        doc[:, j] = np.clip(doc[:, j] * factor, 0, 255).astype(np.uint8)

    print("Project 1: Document Scanner Extensions")
    print("=" * 60)

    # Extension 1: OCR region detection (find text lines)
    print("\n  [1] OCR Region Detection:")
    # Horizontal projection to find text lines
    row_sums = np.sum(255 - doc, axis=1).astype(np.float64)
    threshold = np.mean(row_sums) + np.std(row_sums)

    detected_lines = []
    in_line = False
    line_start = 0

    for i, val in enumerate(row_sums):
        if val > threshold and not in_line:
            in_line = True
            line_start = i
        elif val <= threshold and in_line:
            in_line = False
            detected_lines.append((line_start, i))

    print(f"    Ground truth text lines: {len(text_lines)}")
    print(f"    Detected text regions:   {len(detected_lines)}")
    for i, (start, end) in enumerate(detected_lines[:5]):
        print(f"      Line {i+1}: rows [{start}, {end}]")

    # Extension 2: Auto color correction
    print("\n  [2] Auto Color Correction:")
    # Histogram equalization
    hist = np.zeros(256, dtype=np.int64)
    for val in doc.ravel():
        hist[val] += 1

    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    total_pixels = h * w

    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = int((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255)

    corrected = lut[doc]

    print(f"    Before: range=[{doc.min()}, {doc.max()}], "
          f"std={np.std(doc):.1f}")
    print(f"    After:  range=[{corrected.min()}, {corrected.max()}], "
          f"std={np.std(corrected):.1f}")
    print(f"    Contrast improvement: "
          f"{np.std(corrected)/max(np.std(doc),1):.2f}x")

    # Extension 3: Multi-page assembly
    print("\n  [3] Multi-page Assembly:")
    n_pages = 3
    pages = []
    for p in range(n_pages):
        page = corrected.copy()
        # Add slight variation per page
        page = np.clip(page.astype(np.int16) + np.random.randint(-10, 10),
                       0, 255).astype(np.uint8)
        pages.append(page)
        print(f"    Page {p+1}: {page.shape[1]}x{page.shape[0]}, "
              f"mean={np.mean(page):.1f}")

    # Simulated PDF assembly
    total_h = sum(p.shape[0] for p in pages)
    print(f"    Total document: {n_pages} pages, {total_h} rows")

    return {'pages': pages, 'detected_lines': detected_lines}


# =============================================================================
# Project 2: Lane Detection Extensions
# =============================================================================

def project_2_lane_detection():
    """
    Lane detection extensions:
    1. Curved lane detection (polynomial fitting)
    2. Lane departure warning
    3. Multi-lane detection

    Returns:
        lane analysis results
    """
    np.random.seed(42)
    h, w = 200, 300

    # Create synthetic road image
    road = np.ones((h, w), dtype=np.uint8) * 80  # Dark road

    # Generate curved lane markings using 2nd degree polynomial
    # Left lane: x = a*y^2 + b*y + c
    a_left, b_left, c_left = -0.002, 0.5, 80
    a_right, b_right, c_right = -0.002, 0.5, 220

    left_pts = []
    right_pts = []

    for y in range(50, h):
        xl = int(a_left * y**2 + b_left * y + c_left)
        xr = int(a_right * y**2 + b_right * y + c_right)
        if 0 <= xl < w:
            road[y, max(0, xl-1):min(w, xl+2)] = 220
            left_pts.append((xl, y))
        if 0 <= xr < w:
            road[y, max(0, xr-1):min(w, xr+2)] = 220
            right_pts.append((xr, y))

    print("Project 2: Lane Detection Extensions")
    print("=" * 60)

    # Extension 1: Polynomial fitting
    print("\n  [1] Curved Lane Detection (Polynomial Fitting):")

    def fit_polynomial(points, degree=2):
        """Fit polynomial to lane points."""
        if len(points) < degree + 1:
            return None
        pts = np.array(points, dtype=np.float64)
        coeffs = np.polyfit(pts[:, 1], pts[:, 0], degree)
        return coeffs

    left_coeffs = fit_polynomial(left_pts, degree=2)
    right_coeffs = fit_polynomial(right_pts, degree=2)

    if left_coeffs is not None:
        print(f"    Left lane:  {left_coeffs[0]:.6f}*y^2 + "
              f"{left_coeffs[1]:.4f}*y + {left_coeffs[2]:.1f}")
        print(f"    True:       {a_left}*y^2 + {b_left}*y + {c_left}")

    if right_coeffs is not None:
        print(f"    Right lane: {right_coeffs[0]:.6f}*y^2 + "
              f"{right_coeffs[1]:.4f}*y + {right_coeffs[2]:.1f}")

    # Fitting error
    if left_coeffs is not None:
        errors = []
        for x, y in left_pts:
            x_pred = np.polyval(left_coeffs, y)
            errors.append(abs(x - x_pred))
        print(f"    Left fitting error: {np.mean(errors):.2f} px")

    # Extension 2: Lane departure warning
    print("\n  [2] Lane Departure Warning:")

    # Vehicle assumed at bottom center of image
    vehicle_x = w // 2
    vehicle_y = h - 10

    # Lane center at vehicle position
    if left_coeffs is not None and right_coeffs is not None:
        left_x = np.polyval(left_coeffs, vehicle_y)
        right_x = np.polyval(right_coeffs, vehicle_y)
        lane_center = (left_x + right_x) / 2
        lane_width = right_x - left_x
        offset = vehicle_x - lane_center
        offset_pct = abs(offset) / (lane_width / 2) * 100

        print(f"    Lane center: {lane_center:.1f} px")
        print(f"    Vehicle pos: {vehicle_x} px")
        print(f"    Offset: {offset:.1f} px ({offset_pct:.1f}%)")
        print(f"    Lane width: {lane_width:.1f} px")

        if offset_pct > 70:
            warning = "DEPARTURE WARNING!"
        elif offset_pct > 40:
            warning = "Drifting - caution"
        else:
            warning = "Centered - OK"
        print(f"    Status: {warning}")

    # Extension 3: Multi-lane detection
    print("\n  [3] Multi-lane Detection:")

    # Add adjacent lanes
    adjacent_lanes = [
        (-0.002, 0.5, 40),   # Far left
        (-0.002, 0.5, 260),  # Far right
    ]

    all_lanes = [
        ('Left', left_coeffs),
        ('Right', right_coeffs),
    ]

    for i, (a, b, c) in enumerate(adjacent_lanes):
        pts = [(int(a * y**2 + b * y + c), y) for y in range(50, h)]
        pts = [(x, y) for x, y in pts if 0 <= x < w]
        if len(pts) > 10:
            coeffs = fit_polynomial(pts, 2)
            name = f"Adjacent_{i+1}"
            all_lanes.append((name, coeffs))

    print(f"    Detected lanes: {len(all_lanes)}")
    for name, coeffs in all_lanes:
        if coeffs is not None:
            x_bottom = np.polyval(coeffs, h - 1)
            print(f"      {name:>12}: x={x_bottom:.0f} at bottom")

    return {
        'left_coeffs': left_coeffs,
        'right_coeffs': right_coeffs,
        'n_lanes': len(all_lanes),
    }


# =============================================================================
# Project 3: AR Marker Extensions
# =============================================================================

def project_3_ar_markers():
    """
    AR marker extensions:
    1. Marker detection (binary pattern)
    2. Homography-based 3D overlay positioning
    3. Multi-marker interaction

    Returns:
        AR processing results
    """
    np.random.seed(42)

    # Create synthetic marker (5x5 binary grid)
    def create_marker(marker_id, size=50):
        """Create a binary AR marker image."""
        grid = np.zeros((5, 5), dtype=np.uint8)
        # Encode ID in binary pattern (center 3x3)
        bits = format(marker_id % 512, '09b')
        for i in range(3):
            for j in range(3):
                grid[i+1, j+1] = int(bits[i*3+j]) * 255

        # Scale up
        marker = np.zeros((size, size), dtype=np.uint8)
        cell = size // 5
        for i in range(5):
            for j in range(5):
                marker[i*cell:(i+1)*cell, j*cell:(j+1)*cell] = grid[i, j]

        # Add border (black)
        marker[:cell, :] = 0
        marker[-cell:, :] = 0
        marker[:, :cell] = 0
        marker[:, -cell:] = 0

        return marker, grid

    # Create scene with markers
    h, w = 200, 300
    scene = np.ones((h, w), dtype=np.uint8) * 128

    markers = {}
    positions = [(30, 30), (150, 50), (80, 120)]

    print("Project 3: AR Marker Extensions")
    print("=" * 60)

    print("\n  [1] Marker Detection:")
    for i, (px, py) in enumerate(positions):
        marker, grid = create_marker(i * 7 + 1, size=50)
        markers[i] = {'id': i * 7 + 1, 'pos': (px, py), 'grid': grid}

        # Place in scene
        y1, y2 = py, min(h, py + 50)
        x1, x2 = px, min(w, px + 50)
        mh = y2 - y1
        mw = x2 - x1
        scene[y1:y2, x1:x2] = marker[:mh, :mw]

        print(f"    Marker {i}: ID={i*7+1}, pos=({px},{py}), size=50x50")

    # Extension 2: Homography computation
    print("\n  [2] 3D Overlay Positioning:")

    for mid, minfo in markers.items():
        px, py = minfo['pos']
        # Marker corners in image (with slight perspective)
        src = np.array([
            [px, py],
            [px + 50, py + 2],
            [px + 48, py + 50],
            [px - 2, py + 48]
        ], dtype=np.float64)

        # Desired square corners
        dst = np.array([
            [0, 0], [50, 0], [50, 50], [0, 50]
        ], dtype=np.float64)

        # Compute homography using DLT (Direct Linear Transform)
        A = []
        for i in range(4):
            x, y = src[i]
            u, v = dst[i]
            A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H /= H[2, 2]

        # Project a virtual 3D point
        virtual_center = np.array([25, 25, 1])
        projected = H @ virtual_center
        projected /= projected[2]

        print(f"    Marker {mid}: overlay center = "
              f"({projected[0]:.1f}, {projected[1]:.1f})")

    # Extension 3: Multi-marker interaction
    print("\n  [3] Multi-marker Interaction:")
    if len(markers) >= 2:
        for i in range(len(markers)):
            for j in range(i + 1, len(markers)):
                p1 = np.array(markers[i]['pos'])
                p2 = np.array(markers[j]['pos'])
                dist = np.sqrt(np.sum((p1 - p2)**2))
                print(f"    Distance M{i}-M{j}: {dist:.1f}px")

        # Compute centroid of all markers
        all_pos = np.array([m['pos'] for m in markers.values()])
        centroid = np.mean(all_pos, axis=0)
        print(f"    Marker cluster centroid: ({centroid[0]:.1f}, "
              f"{centroid[1]:.1f})")

    return markers


# =============================================================================
# Project 4: Face Filter Extensions
# =============================================================================

def project_4_face_filters():
    """
    Face filter extensions:
    1. Expression detection (EAR, MAR based)
    2. Filter selection based on expression
    3. Background segmentation (depth-based simulation)

    Returns:
        filter application results
    """
    np.random.seed(42)

    # Simulated facial landmark measurements over time
    n_frames = 20
    expressions = []

    for f in range(n_frames):
        if f < 5:
            # Neutral
            ear = 0.28 + np.random.randn() * 0.02
            mar = 0.20 + np.random.randn() * 0.02
            brow = 0.45
        elif f < 10:
            # Smiling
            ear = 0.25 + np.random.randn() * 0.02
            mar = 0.50 + np.random.randn() * 0.03
            brow = 0.55
        elif f < 15:
            # Surprised
            ear = 0.38 + np.random.randn() * 0.02
            mar = 0.65 + np.random.randn() * 0.03
            brow = 0.65
        else:
            # Eye blink (wink)
            ear = 0.15 + np.random.randn() * 0.02
            mar = 0.20 + np.random.randn() * 0.02
            brow = 0.45

        expressions.append({'ear': ear, 'mar': mar, 'brow': brow})

    # Extension 1: Expression detection
    print("Project 4: Face Filter Extensions")
    print("=" * 60)
    print("\n  [1] Expression Detection:")

    def detect_expression(ear, mar, brow):
        if ear > 0.35 and mar > 0.5:
            return "Surprised"
        elif mar > 0.4 and brow > 0.5:
            return "Smiling"
        elif ear < 0.2:
            return "Blink/Wink"
        else:
            return "Neutral"

    # Extension 2: Filter selection
    filter_map = {
        'Neutral': 'Dog_ears',
        'Smiling': 'Sunglasses',
        'Surprised': 'Crown',
        'Blink/Wink': 'Hearts',
    }

    print(f"\n  [2] Expression-based Filter Switching:")
    print(f"  {'Frame':>6} | {'EAR':>5} | {'MAR':>5} | "
          f"{'Expression':>10} | {'Filter':>12}")
    print(f"  {'-'*55}")

    filter_usage = {}
    for f, expr in enumerate(expressions):
        detected = detect_expression(expr['ear'], expr['mar'], expr['brow'])
        active_filter = filter_map.get(detected, 'None')
        filter_usage[active_filter] = filter_usage.get(active_filter, 0) + 1

        if f % 3 == 0 or f < 3:
            print(f"  {f:>6} | {expr['ear']:>5.2f} | {expr['mar']:>5.2f} | "
                  f"{detected:>10} | {active_filter:>12}")

    print(f"\n  Filter Usage Summary:")
    for filt, count in sorted(filter_usage.items(), key=lambda x: -x[1]):
        pct = 100 * count / n_frames
        print(f"    {filt:>12}: {count} frames ({pct:.0f}%)")

    # Extension 3: Background segmentation
    print(f"\n  [3] Background Segmentation (depth-based):")
    h, w = 100, 120
    # Simulate depth map (face = close, background = far)
    depth = np.ones((h, w), dtype=np.float64) * 5.0  # Background at 5m

    # Face region (closer)
    yy, xx = np.ogrid[:h, :w]
    face_mask = ((xx - w//2)**2 / 25**2 + (yy - h//2)**2 / 35**2) <= 1
    depth[face_mask] = 0.5 + np.random.randn(np.sum(face_mask)) * 0.1

    # Foreground/background separation
    threshold = np.percentile(depth, 30)
    fg_mask = depth < threshold
    fg_pixels = np.sum(fg_mask)
    bg_pixels = np.sum(~fg_mask)

    print(f"    Depth range: [{depth.min():.2f}, {depth.max():.2f}] m")
    print(f"    Threshold: {threshold:.2f} m")
    print(f"    Foreground: {fg_pixels} px ({100*fg_pixels/(h*w):.1f}%)")
    print(f"    Background: {bg_pixels} px ({100*bg_pixels/(h*w):.1f}%)")

    # Simulate background blur
    # Blur only background pixels (replace with average)
    bg_mean = 128  # Simulated blurred background value
    print(f"    Background blur applied (simulated)")

    return {
        'expressions': expressions,
        'filter_usage': filter_usage,
        'fg_ratio': fg_pixels / (h * w),
    }


# =============================================================================
# Project 5: Object Tracking Extensions
# =============================================================================

def project_5_object_tracking():
    """
    Object tracking extensions:
    1. Re-identification (feature matching after disappearance)
    2. Speed measurement with calibration
    3. Zone intrusion detection
    4. Trajectory analysis

    Returns:
        tracking analysis results
    """
    np.random.seed(42)

    # Simulate tracked objects with trajectories
    n_frames = 60

    class TrackedObject:
        def __init__(self, obj_id, start_pos, velocity, feature, visible_range):
            self.id = obj_id
            self.start = np.array(start_pos, dtype=np.float64)
            self.vel = np.array(velocity, dtype=np.float64)
            self.feature = feature  # Feature vector for re-ID
            self.visible = visible_range  # (start_frame, end_frame)
            self.positions = []

        def get_position(self, frame):
            if self.visible[0] <= frame <= self.visible[1]:
                pos = self.start + self.vel * (frame - self.visible[0])
                self.positions.append(pos.copy())
                return pos
            return None

    obj0_feat = np.random.randn(32)
    obj1_feat = np.random.randn(32)
    obj2_feat = np.random.randn(32)
    # Object 3 shares similar features with object 2 (re-identification)
    obj3_feat = obj2_feat + np.random.randn(32) * 0.1

    objects = [
        TrackedObject(0, (20, 30), (2, 0.5), obj0_feat, (0, 55)),
        TrackedObject(1, (100, 80), (-1, -0.3), obj1_feat, (0, 45)),
        TrackedObject(2, (50, 10), (0.5, 1.5), obj2_feat, (5, 25)),
        # Object 2 reappears (same feature, different start)
        TrackedObject(3, (70, 60), (0.5, 1.0), obj3_feat, (35, 55)),
    ]

    print("Project 5: Object Tracking Extensions")
    print("=" * 60)

    # Extension 1: Re-identification
    print("\n  [1] Re-Identification:")

    def match_features(feat1, feat2, threshold=5.0):
        """Check if two feature vectors match."""
        dist = np.sqrt(np.sum((feat1 - feat2)**2))
        return dist < threshold, dist

    # Check if object 3 is a re-appearance of object 2
    matched, dist = match_features(objects[2].feature, objects[3].feature)
    print(f"    Object 2 vs Object 3: dist={dist:.2f}, "
          f"matched={matched}")
    if matched:
        print(f"    -> Re-identified: Object 3 is Object 2 (re-entered)")

    # Extension 2: Speed measurement
    print("\n  [2] Speed Measurement:")
    pixels_per_meter = 10.0  # Calibration
    fps = 30.0

    for obj in objects[:3]:
        speed_px = np.sqrt(np.sum(obj.vel**2))  # pixels/frame
        speed_m_s = speed_px / pixels_per_meter * fps
        speed_km_h = speed_m_s * 3.6

        print(f"    Object {obj.id}: {speed_px:.2f} px/frame = "
              f"{speed_m_s:.2f} m/s = {speed_km_h:.1f} km/h")

    # Extension 3: Zone intrusion detection
    print("\n  [3] Zone Intrusion Detection:")
    restricted_zone = (60, 40, 80, 60)  # (x, y, w, h)
    zx, zy, zw, zh = restricted_zone
    print(f"    Restricted zone: ({zx},{zy}) {zw}x{zh}")

    intrusions = {}
    for frame in range(n_frames):
        for obj in objects:
            pos = obj.get_position(frame)
            if pos is not None:
                x, y = pos
                if zx <= x <= zx + zw and zy <= y <= zy + zh:
                    if obj.id not in intrusions:
                        intrusions[obj.id] = []
                    intrusions[obj.id].append(frame)

    for obj_id, frames_list in intrusions.items():
        print(f"    Object {obj_id}: intrusion at frames "
              f"{frames_list[0]}-{frames_list[-1]} "
              f"({len(frames_list)} frames)")

    if not intrusions:
        print(f"    No intrusions detected")

    # Extension 4: Trajectory analysis
    print("\n  [4] Trajectory Analysis:")
    for obj in objects[:3]:
        if len(obj.positions) > 2:
            traj = np.array(obj.positions)
            # Path length
            diffs = np.diff(traj, axis=0)
            path_len = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

            # Displacement (start to end)
            displacement = np.sqrt(np.sum((traj[-1] - traj[0])**2))

            # Linearity ratio
            linearity = displacement / max(path_len, 1e-6)

            # Direction
            angle = np.degrees(np.arctan2(
                traj[-1][1] - traj[0][1],
                traj[-1][0] - traj[0][0]))

            print(f"    Object {obj.id}:")
            print(f"      Path length: {path_len:.1f} px")
            print(f"      Displacement: {displacement:.1f} px")
            print(f"      Linearity: {linearity:.3f} (1.0 = straight)")
            print(f"      Direction: {angle:.1f} deg")

            # Anomaly: sudden direction changes
            if len(diffs) > 2:
                angles = np.arctan2(diffs[:, 1], diffs[:, 0])
                angle_changes = np.abs(np.diff(angles))
                max_change = np.degrees(np.max(angle_changes))
                print(f"      Max direction change: {max_change:.1f} deg")
                if max_change > 90:
                    print(f"      ANOMALY: Sudden direction change!")

    return {
        'intrusions': intrusions,
        'n_objects': len(objects),
        're_id_match': matched,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Project 1: Document Scanner Extensions")
    project_1_document_scanner()

    print("\n>>> Project 2: Lane Detection Extensions")
    project_2_lane_detection()

    print("\n>>> Project 3: AR Marker Extensions")
    project_3_ar_markers()

    print("\n>>> Project 4: Face Filter Extensions")
    project_4_face_filters()

    print("\n>>> Project 5: Object Tracking Extensions")
    project_5_object_tracking()

    print("\nAll projects completed successfully.")
