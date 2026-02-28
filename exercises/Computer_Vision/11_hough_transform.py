"""
Exercise Solutions for Lesson 11: Hough Transform
Computer Vision - Line and Circle Detection

Topics covered:
- Chessboard grid detection
- Lane detection (from video frames)
- Pizza slice counter (radial lines)
- Analog clock reading
- Building window counter
"""

import numpy as np


# =============================================================================
# Helper: Hough Line Transform
# =============================================================================

def hough_lines(edge_img, rho_res=1, theta_res=np.pi/180, threshold=50):
    """
    Standard Hough Transform for line detection.
    Returns list of (rho, theta) pairs.
    """
    h, w = edge_img.shape
    diag = int(np.ceil(np.sqrt(h**2 + w**2)))
    n_rho = 2 * diag
    n_theta = int(np.pi / theta_res)

    # Accumulator
    accumulator = np.zeros((n_rho, n_theta), dtype=np.int32)
    thetas = np.arange(0, np.pi, theta_res)
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    # Vote
    edge_y, edge_x = np.where(edge_img > 0)
    for x, y in zip(edge_x, edge_y):
        for t_idx, (cos_t, sin_t) in enumerate(zip(cos_thetas, sin_thetas)):
            rho = int(x * cos_t + y * sin_t) + diag
            if 0 <= rho < n_rho:
                accumulator[rho, t_idx] += 1

    # Find peaks above threshold
    lines = []
    for rho_idx in range(n_rho):
        for t_idx in range(n_theta):
            if accumulator[rho_idx, t_idx] >= threshold:
                rho = rho_idx - diag
                theta = thetas[t_idx]
                lines.append((rho, theta))

    return lines, accumulator


def hough_circles(edge_img, r_min=10, r_max=50, threshold=30):
    """
    Hough Transform for circle detection.
    Returns list of (cx, cy, radius) tuples.
    """
    h, w = edge_img.shape
    accumulator = np.zeros((h, w, r_max - r_min + 1), dtype=np.int32)

    edge_y, edge_x = np.where(edge_img > 0)

    for x, y in zip(edge_x, edge_y):
        for r_idx, r in enumerate(range(r_min, r_max + 1)):
            for theta_deg in range(0, 360, 5):
                theta = np.radians(theta_deg)
                a = int(x - r * np.cos(theta))
                b = int(y - r * np.sin(theta))
                if 0 <= a < w and 0 <= b < h:
                    accumulator[b, a, r_idx] += 1

    # Find peaks
    circles = []
    for b in range(h):
        for a in range(w):
            for r_idx in range(r_max - r_min + 1):
                if accumulator[b, a, r_idx] >= threshold:
                    circles.append((a, b, r_min + r_idx))

    # Non-maximum suppression
    filtered = []
    for c in circles:
        too_close = False
        for f in filtered:
            if np.sqrt((c[0]-f[0])**2 + (c[1]-f[1])**2) < min(c[2], f[2]):
                too_close = True
                break
        if not too_close:
            filtered.append(c)

    return filtered


# =============================================================================
# Exercise 1: Chessboard Detection
# =============================================================================

def exercise_1_chessboard_detection():
    """
    Detect horizontal and vertical lines in a chessboard image
    and find their intersection points.

    Returns:
        list of intersection points
    """
    # Create synthetic chessboard
    cell = 20
    board_size = 8
    size = cell * board_size
    board = np.zeros((size, size), dtype=np.uint8)

    for r in range(board_size):
        for c in range(board_size):
            if (r + c) % 2 == 0:
                board[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = 255

    # Detect edges (grid lines)
    edges = np.zeros_like(board)
    # Horizontal edges
    for i in range(1, size-1):
        for j in range(size):
            if abs(int(board[i+1, j]) - int(board[i-1, j])) > 100:
                edges[i, j] = 255
    # Vertical edges
    for i in range(size):
        for j in range(1, size-1):
            if abs(int(board[i, j+1]) - int(board[i, j-1])) > 100:
                edges[i, j] = 255

    # Find lines using simplified approach: look for consistent rows/cols
    horizontal_lines = []
    vertical_lines = []

    for y in range(size):
        if np.sum(edges[y, :] > 0) > size * 0.5:
            # Check if not too close to an existing line
            if not horizontal_lines or y - horizontal_lines[-1] > cell // 2:
                horizontal_lines.append(y)

    for x in range(size):
        if np.sum(edges[:, x] > 0) > size * 0.5:
            if not vertical_lines or x - vertical_lines[-1] > cell // 2:
                vertical_lines.append(x)

    # Calculate intersections
    intersections = []
    for y in horizontal_lines:
        for x in vertical_lines:
            intersections.append((x, y))

    print(f"Chessboard: {board_size}x{board_size}, cell={cell}px")
    print(f"Horizontal lines: {len(horizontal_lines)} at y={horizontal_lines}")
    print(f"Vertical lines:   {len(vertical_lines)} at x={vertical_lines}")
    print(f"Intersections:    {len(intersections)}")
    print(f"Expected inner intersections: {(board_size-1)**2} = {(board_size-1)**2}")

    return intersections


# =============================================================================
# Exercise 2: Lane Detection
# =============================================================================

def exercise_2_lane_detection():
    """
    Detect lane lines in a synthetic road image using Hough transform.
    Applies ROI to focus on the lower half where lanes appear.

    Returns:
        list of detected lane line parameters
    """
    # Create synthetic road image
    h, w = 150, 200
    road = np.ones((h, w), dtype=np.uint8) * 100  # Gray road

    # Draw lane markings as angled white lines
    # Left lane line
    for y in range(h // 2, h):
        x = int(w * 0.3 + (y - h//2) * 0.3)
        if 0 <= x < w - 2:
            road[y, x:x+2] = 255

    # Right lane line
    for y in range(h // 2, h):
        x = int(w * 0.7 - (y - h//2) * 0.3)
        if 0 <= x - 1 < w:
            road[y, x-1:x+1] = 255

    # Edge detection in ROI (lower half)
    roi = road[h//2:, :]
    edges = np.zeros_like(roi)
    for i in range(1, roi.shape[0]-1):
        for j in range(1, roi.shape[1]-1):
            gx = int(roi[i, j+1]) - int(roi[i, j-1])
            gy = int(roi[i+1, j]) - int(roi[i-1, j])
            if np.sqrt(gx**2 + gy**2) > 50:
                edges[i, j] = 255

    # Hough line detection
    lines, _ = hough_lines(edges, threshold=20)

    # Filter lines by angle (lanes should be roughly 30-70 degrees)
    lane_lines = []
    for rho, theta in lines:
        angle_deg = np.degrees(theta)
        if (20 < angle_deg < 80) or (100 < angle_deg < 160):
            lane_lines.append((rho, theta))

    # Remove duplicate/near lines
    filtered = []
    for line in lane_lines:
        is_dup = False
        for existing in filtered:
            if abs(line[0] - existing[0]) < 10 and abs(line[1] - existing[1]) < 0.1:
                is_dup = True
                break
        if not is_dup:
            filtered.append(line)

    print(f"Road image: {w}x{h}")
    print(f"Edge pixels in ROI: {np.sum(edges > 0)}")
    print(f"Raw Hough lines: {len(lines)}")
    print(f"Lane lines (angle filtered): {len(lane_lines)}")
    print(f"Final lanes (deduplicated): {len(filtered)}")

    for i, (rho, theta) in enumerate(filtered):
        print(f"  Lane {i+1}: rho={rho:.1f}, theta={np.degrees(theta):.1f} deg")

    return filtered


# =============================================================================
# Exercise 3: Pizza Slice Counter
# =============================================================================

def exercise_3_pizza_slices():
    """
    Count pizza slices by detecting radial lines from center.

    Returns:
        number of slices
    """
    # Create synthetic pizza image (circle with radial divisions)
    size = 150
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = 60

    # Draw pizza circle
    yy, xx = np.ogrid[:size, :size]
    circle = ((xx - cx)**2 + (yy - cy)**2) <= radius**2
    img[circle] = 180

    # Draw 8 slice lines (every 45 degrees)
    n_slices = 8
    for i in range(n_slices):
        angle = 2 * np.pi * i / n_slices
        for r in range(radius):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = 40  # Dark line

    # Detect edges
    edges = np.zeros_like(img)
    for i in range(1, size-1):
        for j in range(1, size-1):
            if circle[i, j]:
                gx = int(img[i, j+1]) - int(img[i, j-1])
                gy = int(img[i+1, j]) - int(img[i-1, j])
                if np.sqrt(gx**2 + gy**2) > 40:
                    edges[i, j] = 255

    # Count lines through center
    # Check angles from center that have edge pixels
    angle_bins = np.zeros(360)
    for i in range(size):
        for j in range(size):
            if edges[i, j] > 0:
                dx = j - cx
                dy = i - cy
                dist = np.sqrt(dx**2 + dy**2)
                if 5 < dist < radius - 5:  # Not too close to center or edge
                    angle = int(np.degrees(np.arctan2(dy, dx))) % 360
                    angle_bins[angle] += 1

    # Find peaks in angle histogram (= slice lines)
    # Smooth the histogram
    smoothed = np.convolve(angle_bins, np.ones(5)/5, mode='same')

    peaks = []
    for i in range(len(smoothed)):
        prev_i = (i - 1) % 360
        next_i = (i + 1) % 360
        if smoothed[i] > smoothed[prev_i] and smoothed[i] > smoothed[next_i]:
            if smoothed[i] > 2:  # Minimum vote threshold
                # Check not too close to existing peak
                if not peaks or min(abs(i - p) for p in peaks) > 15:
                    peaks.append(i)

    detected_slices = len(peaks)
    print(f"Pizza image: {size}x{size}, radius={radius}")
    print(f"Edge pixels: {np.sum(edges > 0)}")
    print(f"Detected radial lines: {len(peaks)}")
    print(f"Detected slices: {detected_slices}")
    print(f"Expected slices: {n_slices}")
    print(f"Line angles: {peaks}")

    return detected_slices


# =============================================================================
# Exercise 4: Analog Clock Reading
# =============================================================================

def exercise_4_read_clock():
    """
    Detect clock circle and hands, calculate the time from hand angles.

    Returns:
        (hours, minutes)
    """
    # Create synthetic clock face
    size = 150
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = 60

    # Draw clock circle
    yy, xx = np.ogrid[:size, :size]
    circle_edge = (((xx - cx)**2 + (yy - cy)**2) >= (radius-2)**2) & \
                  (((xx - cx)**2 + (yy - cy)**2) <= (radius+2)**2)
    img[circle_edge] = 255

    # Draw hour hand (short, pointing to 3 o'clock = 0 degrees)
    hour_angle = np.radians(0)  # 3 o'clock
    hour_length = 30
    for r in range(5, hour_length):
        x = int(cx + r * np.cos(hour_angle))
        y = int(cy + r * np.sin(hour_angle))
        if 0 <= x < size and 0 <= y < size:
            img[y, x] = 200
            if y+1 < size:
                img[y+1, x] = 200  # Thicker

    # Draw minute hand (long, pointing to 12 o'clock = -90 degrees)
    minute_angle = np.radians(-90)  # 12 o'clock
    minute_length = 50
    for r in range(5, minute_length):
        x = int(cx + r * np.cos(minute_angle))
        y = int(cy + r * np.sin(minute_angle))
        if 0 <= x < size and 0 <= y < size:
            img[y, x] = 200

    # Detect hands by analyzing pixel angles from center
    hand_angles = []
    angle_strength = np.zeros(360)

    for i in range(size):
        for j in range(size):
            if img[i, j] > 100 and not circle_edge[i, j]:
                dx = j - cx
                dy = i - cy
                dist = np.sqrt(dx**2 + dy**2)
                if 8 < dist < radius - 5:
                    angle = int(np.degrees(np.arctan2(dy, dx))) % 360
                    angle_strength[angle] += dist  # Weight by distance

    # Find the two strongest angle peaks (hour and minute hands)
    # Smooth
    smoothed = np.convolve(angle_strength, np.ones(5)/5, mode='same')
    peaks = []
    for i in range(360):
        prev_i = (i - 1) % 360
        next_i = (i + 1) % 360
        if smoothed[i] > smoothed[prev_i] and smoothed[i] > smoothed[next_i]:
            if smoothed[i] > 10:
                peaks.append((i, smoothed[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)

    if len(peaks) >= 2:
        # Determine which is hour (shorter, lower weight) and minute (longer, higher weight)
        hand1_angle = peaks[0][0]
        hand2_angle = peaks[1][0]

        # Convert from mathematical angle to clock angle
        # Clock: 12=top, goes clockwise. Math: 0=right, goes counterclockwise
        def math_angle_to_clock(angle_deg):
            clock_deg = (angle_deg + 90) % 360
            return clock_deg

        clock1 = math_angle_to_clock(hand1_angle)
        clock2 = math_angle_to_clock(hand2_angle)

        # Longer hand = minute, shorter = hour
        # The peak with higher weight (further pixels) is the minute hand
        minute_clock = clock1
        hour_clock = clock2

        hours = int(hour_clock / 30) % 12
        minutes = int(minute_clock / 6) % 60

        print(f"Clock image: {size}x{size}")
        print(f"Detected hand angles: {hand1_angle} deg, {hand2_angle} deg")
        print(f"Clock angles: {clock1:.0f} deg, {clock2:.0f} deg")
        print(f"Time: {hours}:{minutes:02d}")

        return hours, minutes

    print("Could not detect clock hands")
    return None, None


# =============================================================================
# Exercise 5: Building Window Counter
# =============================================================================

def exercise_5_count_windows():
    """
    Count windows in a synthetic building image by detecting
    regular rectangular patterns.

    Returns:
        window count
    """
    # Create synthetic building facade
    h, w = 150, 120
    building = np.ones((h, w), dtype=np.uint8) * 160  # Wall color

    # Add windows (dark rectangles in a grid pattern)
    win_w, win_h = 15, 20
    gap_x, gap_y = 10, 15
    start_x, start_y = 15, 15
    rows, cols = 4, 3

    expected = 0
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * (win_w + gap_x)
            y = start_y + r * (win_h + gap_y)
            if x + win_w < w and y + win_h < h:
                building[y:y+win_h, x:x+win_w] = 60  # Dark windows
                expected += 1

    # Detect windows by finding dark rectangular regions
    # Threshold to find dark regions
    dark_mask = (building < 100).astype(np.uint8) * 255

    # Find connected components
    visited = np.zeros_like(dark_mask, dtype=bool)
    windows = []

    for i in range(h):
        for j in range(w):
            if dark_mask[i, j] > 0 and not visited[i, j]:
                # Flood fill
                component = []
                stack = [(i, j)]
                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w:
                        continue
                    if visited[cy, cx] or dark_mask[cy, cx] == 0:
                        continue
                    visited[cy, cx] = True
                    component.append((cx, cy))
                    stack.extend([(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)])

                if len(component) < 50:  # Too small
                    continue

                pts = np.array(component)
                x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
                y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
                bbox_w = x_max - x_min + 1
                bbox_h = y_max - y_min + 1
                fill_ratio = len(component) / (bbox_w * bbox_h)

                # Windows should be roughly rectangular
                if fill_ratio > 0.8 and 10 < bbox_w < 30 and 10 < bbox_h < 40:
                    windows.append({
                        'bbox': (x_min, y_min, bbox_w, bbox_h),
                        'area': len(component),
                        'fill': fill_ratio
                    })

    print(f"Building facade: {w}x{h}")
    print(f"Expected windows: {expected} ({rows} rows x {cols} cols)")
    print(f"Detected windows: {len(windows)}")

    for i, win in enumerate(windows):
        x, y, ww, wh = win['bbox']
        print(f"  Window {i+1}: ({x},{y}) {ww}x{wh}, fill={win['fill']:.2f}")

    return len(windows)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Chessboard Detection")
    exercise_1_chessboard_detection()

    print("\n>>> Exercise 2: Lane Detection")
    exercise_2_lane_detection()

    print("\n>>> Exercise 3: Pizza Slice Counter")
    exercise_3_pizza_slices()

    print("\n>>> Exercise 4: Analog Clock Reading")
    exercise_4_read_clock()

    print("\n>>> Exercise 5: Building Window Counter")
    exercise_5_count_windows()

    print("\nAll exercises completed successfully.")
