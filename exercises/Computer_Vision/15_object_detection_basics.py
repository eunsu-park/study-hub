"""
Exercise Solutions for Lesson 15: Object Detection Basics
Computer Vision - Template Matching, Haar Cascade, HOG+SVM

Topics covered:
- Multiple template matching with different colors
- Rotation-invariant template matching
- Real-time face detection optimization
- HOG feature visualization
- License plate detection (aspect ratio-based)
"""

import numpy as np


# =============================================================================
# Helper: Template Matching (Normalized Cross-Correlation)
# =============================================================================

def match_template_ncc(img, template):
    """
    Normalized cross-correlation template matching.
    Returns result map where higher = better match.
    """
    ih, iw = img.shape[:2]
    th, tw = template.shape[:2]
    rh, rw = ih - th + 1, iw - tw + 1

    img_f = img.astype(np.float64)
    tmpl_f = template.astype(np.float64)
    tmpl_mean = tmpl_f.mean()
    tmpl_std = tmpl_f.std()

    result = np.zeros((rh, rw), dtype=np.float64)

    if tmpl_std < 1e-6:
        return result

    tmpl_norm = tmpl_f - tmpl_mean

    for i in range(rh):
        for j in range(rw):
            region = img_f[i:i+th, j:j+tw]
            region_mean = region.mean()
            region_std = region.std()

            if region_std < 1e-6:
                continue

            ncc = np.sum((region - region_mean) * tmpl_norm) / (th * tw * region_std * tmpl_std)
            result[i, j] = ncc

    return result


# =============================================================================
# Exercise 1: Multiple Template Matching
# =============================================================================

def exercise_1_multiple_templates():
    """
    Match 3 different templates simultaneously in one scene.
    Each template result shown with different identification.

    Returns:
        list of (template_name, location, score) results
    """
    # Create scene with 3 different objects
    scene = np.ones((120, 200), dtype=np.uint8) * 128

    # Object 1: Small bright square
    scene[20:35, 30:45] = 220

    # Object 2: Dark rectangle
    scene[60:80, 100:130] = 50

    # Object 3: Circle-like bright blob
    yy, xx = np.ogrid[:120, :200]
    circle_mask = ((xx - 160)**2 + (yy - 40)**2) <= 12**2
    scene[circle_mask] = 200

    # Create templates
    templates = {
        'Square': scene[20:35, 30:45].copy(),
        'Rectangle': scene[60:80, 100:130].copy(),
        'Circle': scene[28:52, 148:172].copy(),  # Crop around circle
    }

    results = []
    print(f"Scene: {scene.shape[1]}x{scene.shape[0]}")
    print(f"Templates: {len(templates)}")
    print()

    for name, tmpl in templates.items():
        result_map = match_template_ncc(scene, tmpl)
        max_val = result_map.max()
        max_loc = np.unravel_index(result_map.argmax(), result_map.shape)
        y, x = max_loc

        results.append((name, (x, y), max_val))
        print(f"  {name:>12}: location=({x:>3}, {y:>3}), "
              f"score={max_val:.4f}, template={tmpl.shape[1]}x{tmpl.shape[0]}")

    return results


# =============================================================================
# Exercise 2: Rotation-Invariant Template Matching
# =============================================================================

def exercise_2_rotation_invariant_matching():
    """
    Rotate template at various angles and find the best match.
    Records the best score at each rotation angle.

    Returns:
        (best_angle, best_score, best_location)
    """
    # Create scene with a rotated object
    size = 100
    scene = np.ones((size, size), dtype=np.uint8) * 128

    # Place a small "L" shape rotated ~30 degrees
    target_angle = 30
    cx, cy = 50, 50
    theta = np.radians(target_angle)

    l_points = [(0, 0), (0, 15), (3, 15), (3, 5), (10, 5), (10, 0)]
    for y in range(size):
        for x in range(size):
            # Inverse rotate to check if point is in L shape
            dx = x - cx
            dy = y - cy
            ox = np.cos(-theta) * dx + np.sin(-theta) * dy
            oy = -np.sin(-theta) * dx + np.cos(-theta) * dy

            # Simple point-in-polygon for the L shape
            if 0 <= ox <= 10 and 0 <= oy <= 5:
                scene[y, x] = 220
            elif 0 <= ox <= 3 and 5 <= oy <= 15:
                scene[y, x] = 220

    # Create upright template
    template = np.ones((18, 13), dtype=np.uint8) * 128
    for i in range(6):
        template[i, :11] = 220  # Top bar
    for i in range(6, 16):
        template[i, :4] = 220   # Vertical bar

    def rotate_image(img, angle):
        h, w = img.shape
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        cx_r, cy_r = w / 2, h / 2

        rotated = np.ones_like(img) * 128
        for i in range(h):
            for j in range(w):
                src_x = cos_a * (j - cx_r) + sin_a * (i - cy_r) + cx_r
                src_y = -sin_a * (j - cx_r) + cos_a * (i - cy_r) + cy_r
                si, sj = int(round(src_y)), int(round(src_x))
                if 0 <= si < h and 0 <= sj < w:
                    rotated[i, j] = img[si, sj]
        return rotated

    best_angle = 0
    best_score = -1
    best_location = (0, 0)

    print(f"Testing rotation angles 0-350 (step=10):")
    print(f"{'Angle':>6} | {'Score':>8} | {'Location':>12}")
    print("-" * 35)

    for angle in range(0, 360, 10):
        rotated_tmpl = rotate_image(template, angle)
        result = match_template_ncc(scene, rotated_tmpl)

        max_val = result.max()
        if max_val > best_score:
            best_score = max_val
            best_angle = angle
            max_loc = np.unravel_index(result.argmax(), result.shape)
            best_location = (max_loc[1], max_loc[0])

        if max_val > 0.3:  # Only print notable scores
            loc = np.unravel_index(result.argmax(), result.shape)
            print(f"{angle:>6} | {max_val:>8.4f} | ({loc[1]:>3}, {loc[0]:>3})")

    print(f"\nBest match: angle={best_angle} deg, score={best_score:.4f}, "
          f"location={best_location}")
    print(f"Expected angle: ~{target_angle} deg")

    return best_angle, best_score, best_location


# =============================================================================
# Exercise 3: Face Detection Optimization (FPS Simulation)
# =============================================================================

def exercise_3_face_detection_optimization():
    """
    Simulate face detection at different processing resolutions
    and measure relative speed.

    Returns:
        dict of scale -> (detection_time_ratio, faces_detected)
    """
    import time

    # Create a synthetic face-like image (bright oval on dark background)
    h, w = 480, 640
    frame = np.random.randint(30, 70, (h, w), dtype=np.uint8)

    # Add "face" regions (bright ovals)
    face_centers = [(200, 150), (400, 200), (300, 350)]
    for fx, fy in face_centers:
        yy, xx = np.ogrid[:h, :w]
        face_mask = (((xx - fx) / 40.0)**2 + ((yy - fy) / 50.0)**2) <= 1
        frame[face_mask] = np.random.randint(160, 200)

    scales = [1.0, 0.75, 0.5, 0.25]
    results = {}

    print(f"Original frame: {w}x{h}")
    print(f"Face regions: {len(face_centers)}")
    print(f"\n{'Scale':>6} | {'Size':>12} | {'Time Ratio':>11} | {'Simulated FPS':>14}")
    print("-" * 55)

    for scale in scales:
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Simulate resize
        start = time.perf_counter()
        if scale < 1.0:
            # Downsample
            row_idx = (np.arange(new_h) * h / new_h).astype(int)
            col_idx = (np.arange(new_w) * w / new_w).astype(int)
            row_idx = np.clip(row_idx, 0, h - 1)
            col_idx = np.clip(col_idx, 0, w - 1)
            small = frame[np.ix_(row_idx, col_idx)]
        else:
            small = frame

        # Simulate detection (scanning with window)
        # Detection time is proportional to pixels processed
        pixels = new_h * new_w
        # Simulate some work
        _ = np.sum(small > 128)

        elapsed = time.perf_counter() - start

        time_ratio = (scale ** 2)  # Quadratic reduction
        simulated_fps = 30.0 / time_ratio

        results[scale] = {
            'size': (new_w, new_h),
            'time_ratio': time_ratio,
            'fps': simulated_fps,
        }

        print(f"{scale:>6.2f} | {new_w:>4}x{new_h:<4} | {time_ratio:>10.3f}x | "
              f"{simulated_fps:>13.1f}")

    print(f"\nRecommendation: scale=0.5 gives ~4x speedup while maintaining "
          f"decent detection for faces >= 60px")

    return results


# =============================================================================
# Exercise 4: HOG Feature Visualization
# =============================================================================

def exercise_4_hog_visualization(img, cell_size=8, n_bins=9):
    """
    Compute and visualize HOG (Histogram of Oriented Gradients) features.

    Parameters:
        img: grayscale image
        cell_size: size of each cell in pixels
        n_bins: number of orientation bins

    Returns:
        (hog_descriptor, cell_histograms)
    """
    h, w = img.shape
    img_f = img.astype(np.float64)

    # Compute gradients
    gx = np.zeros_like(img_f)
    gy = np.zeros_like(img_f)
    gx[:, 1:-1] = (img_f[:, 2:] - img_f[:, :-2]) / 2
    gy[1:-1, :] = (img_f[2:, :] - img_f[:-2, :]) / 2

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180  # Unsigned

    # Compute cell histograms
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    cell_hists = np.zeros((n_cells_y, n_cells_x, n_bins))

    bin_width = 180.0 / n_bins

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_size
            x0 = cx * cell_size

            for i in range(cell_size):
                for j in range(cell_size):
                    y, x = y0 + i, x0 + j
                    if y < h and x < w:
                        angle = orientation[y, x]
                        mag = magnitude[y, x]
                        bin_idx = int(angle / bin_width) % n_bins
                        cell_hists[cy, cx, bin_idx] += mag

    # Flatten to descriptor
    descriptor = cell_hists.flatten()

    print(f"HOG Features:")
    print(f"  Image: {w}x{h}")
    print(f"  Cell size: {cell_size}x{cell_size}")
    print(f"  Bins: {n_bins}")
    print(f"  Grid: {n_cells_x}x{n_cells_y} cells")
    print(f"  Descriptor dimension: {len(descriptor)}")
    print(f"  Descriptor range: [{descriptor.min():.1f}, {descriptor.max():.1f}]")

    # Show dominant orientation per cell
    print(f"\n  Dominant orientations (degrees per cell):")
    for cy in range(min(n_cells_y, 4)):
        row = ""
        for cx in range(min(n_cells_x, 6)):
            dom_bin = np.argmax(cell_hists[cy, cx])
            dom_angle = dom_bin * bin_width + bin_width / 2
            row += f"{dom_angle:>5.0f}"
        print(f"    {row}")

    return descriptor, cell_hists


# =============================================================================
# Exercise 5: License Plate Detection
# =============================================================================

def exercise_5_license_plate_detection(img):
    """
    Detect license plate regions based on:
    1. Edge detection
    2. Rectangular contour detection
    3. Aspect ratio filtering (plates are ~4:1 to 5:1)

    Parameters:
        img: grayscale image

    Returns:
        list of detected plate regions (x, y, w, h)
    """
    h, w = img.shape

    # Edge detection
    edges = np.zeros_like(img)
    for i in range(1, h-1):
        for j in range(1, w-1):
            gx = int(img[i, j+1]) - int(img[i, j-1])
            gy = int(img[i+1, j]) - int(img[i-1, j])
            if np.sqrt(gx**2 + gy**2) > 50:
                edges[i, j] = 255

    # Find connected components (potential plate regions)
    visited = np.zeros_like(edges, dtype=bool)
    candidates = []

    for i in range(h):
        for j in range(w):
            if edges[i, j] > 0 and not visited[i, j]:
                component = []
                stack = [(i, j)]
                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w:
                        continue
                    if visited[cy, cx] or edges[cy, cx] == 0:
                        continue
                    visited[cy, cx] = True
                    component.append((cx, cy))
                    stack.extend([(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)])

                if len(component) > 20:
                    pts = np.array(component)
                    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
                    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
                    bbox_w = x_max - x_min + 1
                    bbox_h = y_max - y_min + 1
                    candidates.append((x_min, y_min, bbox_w, bbox_h))

    # Filter by aspect ratio (license plates are typically 4:1 to 5:1)
    plates = []
    for x, y, bw, bh in candidates:
        if bh == 0:
            continue
        aspect = bw / bh
        # Accept aspect ratios between 3:1 and 6:1
        if 3.0 <= aspect <= 6.0 and bw >= 30 and bh >= 8:
            plates.append((x, y, bw, bh))

    print(f"Image: {w}x{h}")
    print(f"Edge pixels: {np.sum(edges > 0)}")
    print(f"Candidate regions: {len(candidates)}")
    print(f"Plate-like regions: {len(plates)}")
    for i, (x, y, bw, bh) in enumerate(plates):
        print(f"  Plate {i+1}: ({x},{y}) {bw}x{bh}, ratio={bw/bh:.1f}:1")

    return plates


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Multiple Template Matching")
    exercise_1_multiple_templates()

    print("\n>>> Exercise 2: Rotation-Invariant Template Matching")
    exercise_2_rotation_invariant_matching()

    print("\n>>> Exercise 3: Face Detection Optimization")
    exercise_3_face_detection_optimization()

    print("\n>>> Exercise 4: HOG Visualization")
    hog_img = np.zeros((64, 64), dtype=np.uint8)
    hog_img[:, 32:] = 200  # Vertical edge
    hog_img[32:, :] = 150  # Horizontal edge
    exercise_4_hog_visualization(hog_img, cell_size=8, n_bins=9)

    print("\n>>> Exercise 5: License Plate Detection")
    plate_img = np.ones((100, 200), dtype=np.uint8) * 128
    # Draw a plate-like rectangle (high contrast border)
    plate_img[40:55, 60:140] = 200   # Plate background
    plate_img[40, 60:140] = 30       # Top edge
    plate_img[54, 60:140] = 30       # Bottom edge
    plate_img[40:55, 60] = 30        # Left edge
    plate_img[40:55, 139] = 30       # Right edge
    exercise_5_license_plate_detection(plate_img)

    print("\nAll exercises completed successfully.")
