"""
Exercise Solutions for Lesson 13: Feature Detection
Computer Vision - Harris, FAST, SIFT, ORB Keypoints

Topics covered:
- Select best N keypoints by response strength
- Uniformly distributed keypoints via grid cells
- Rotation invariance test
"""

import numpy as np


# =============================================================================
# Helper: Simple corner detection (Harris-like)
# =============================================================================

def detect_corners_harris(img, block_size=3, k=0.04, threshold_ratio=0.01):
    """
    Simplified Harris corner detection using numpy.

    Returns list of (x, y, response) tuples.
    """
    h, w = img.shape
    img_f = img.astype(np.float64)

    # Sobel gradients
    # Horizontal: [-1, 0, 1]
    Ix = np.zeros_like(img_f)
    Iy = np.zeros_like(img_f)
    Ix[:, 1:-1] = (img_f[:, 2:] - img_f[:, :-2]) / 2
    Iy[1:-1, :] = (img_f[2:, :] - img_f[:-2, :]) / 2

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Sum over block (box filter)
    pad = block_size // 2
    response = np.zeros_like(img_f)

    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            sxx = np.sum(Ixx[i-pad:i+pad+1, j-pad:j+pad+1])
            syy = np.sum(Iyy[i-pad:i+pad+1, j-pad:j+pad+1])
            sxy = np.sum(Ixy[i-pad:i+pad+1, j-pad:j+pad+1])

            det = sxx * syy - sxy * sxy
            trace = sxx + syy
            response[i, j] = det - k * trace * trace

    # Threshold and non-maximum suppression
    thresh = threshold_ratio * response.max()
    keypoints = []

    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            if response[i, j] > thresh:
                # Local maximum check (3x3 window)
                local = response[max(0,i-1):i+2, max(0,j-1):j+2]
                if response[i, j] == local.max():
                    keypoints.append((j, i, response[i, j]))

    return keypoints


# =============================================================================
# Exercise 1: Select Best Keypoints
# =============================================================================

def exercise_1_select_best_keypoints(img, n=50):
    """
    Detect many keypoints and select only the N strongest by response.

    Parameters:
        img: grayscale image (H, W) uint8
        n: number of best keypoints to keep

    Returns:
        list of top-N (x, y, response) keypoints
    """
    # Detect keypoints
    all_kps = detect_corners_harris(img, block_size=3, k=0.04, threshold_ratio=0.005)

    print(f"Total keypoints detected: {len(all_kps)}")

    # Sort by response (descending)
    sorted_kps = sorted(all_kps, key=lambda kp: kp[2], reverse=True)

    # Select top N
    best = sorted_kps[:n]

    print(f"Selected top {min(n, len(best))} keypoints:")
    if best:
        print(f"  Strongest response: {best[0][2]:.2f} at ({best[0][0]}, {best[0][1]})")
        if len(best) > 1:
            print(f"  Weakest selected:   {best[-1][2]:.2f} at ({best[-1][0]}, {best[-1][1]})")
        print(f"  Response range: [{best[-1][2]:.2f}, {best[0][2]:.2f}]")

    return best


# =============================================================================
# Exercise 2: Uniformly Distributed Keypoints
# =============================================================================

def exercise_2_uniform_keypoints(img, grid_size=(8, 8)):
    """
    Divide image into grid cells and select one strongest keypoint per cell.

    Parameters:
        img: grayscale image (H, W)
        grid_size: (rows, cols) grid division

    Returns:
        list of selected keypoints, one per occupied cell
    """
    h, w = img.shape
    all_kps = detect_corners_harris(img, block_size=3, k=0.04, threshold_ratio=0.005)

    cell_h = h // grid_size[0]
    cell_w = w // grid_size[1]

    selected = []
    occupied_cells = 0

    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            x_min = col * cell_w
            x_max = (col + 1) * cell_w
            y_min = row * cell_h
            y_max = (row + 1) * cell_h

            # Filter keypoints in this cell
            cell_kps = [kp for kp in all_kps
                       if x_min <= kp[0] < x_max and y_min <= kp[1] < y_max]

            if cell_kps:
                # Select strongest
                best = max(cell_kps, key=lambda kp: kp[2])
                selected.append(best)
                occupied_cells += 1

    total_cells = grid_size[0] * grid_size[1]
    print(f"Grid: {grid_size[0]}x{grid_size[1]} = {total_cells} cells")
    print(f"Cell size: {cell_w}x{cell_h}")
    print(f"Occupied cells: {occupied_cells}/{total_cells}")
    print(f"Selected keypoints: {len(selected)}")

    return selected


# =============================================================================
# Exercise 3: Rotation Invariance Test
# =============================================================================

def exercise_3_rotation_invariance(img, angle=30):
    """
    Rotate an image and verify that similar features are detected.
    Compare keypoint locations between original and rotated versions.

    Parameters:
        img: grayscale image (H, W)
        angle: rotation angle in degrees

    Returns:
        (match_count, match_rate)
    """
    h, w = img.shape

    # Rotate the image
    theta = np.radians(angle)
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)
    cx, cy = w / 2.0, h / 2.0

    rotated = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            # Inverse map
            src_x = cos_a * (j - cx) + sin_a * (i - cy) + cx
            src_y = -sin_a * (j - cx) + cos_a * (i - cy) + cy
            si, sj = int(round(src_y)), int(round(src_x))
            if 0 <= si < h and 0 <= sj < w:
                rotated[i, j] = img[si, sj]

    # Detect keypoints in both
    kps_orig = detect_corners_harris(img, threshold_ratio=0.01)
    kps_rot = detect_corners_harris(rotated, threshold_ratio=0.01)

    print(f"Original keypoints: {len(kps_orig)}")
    print(f"Rotated keypoints ({angle} deg): {len(kps_rot)}")

    # Match: for each original keypoint, rotate it and check if a keypoint
    # exists nearby in the rotated image
    match_dist = 5.0  # pixels tolerance
    matches = 0

    for x, y, _ in kps_orig:
        # Rotate the point
        rx = cos_a * (x - cx) - sin_a * (y - cy) + cx
        ry = sin_a * (x - cx) + cos_a * (y - cy) + cy

        # Check if any rotated keypoint is nearby
        for rx2, ry2, _ in kps_rot:
            if np.sqrt((rx - rx2)**2 + (ry - ry2)**2) < match_dist:
                matches += 1
                break

    match_rate = matches / len(kps_orig) * 100 if kps_orig else 0

    print(f"Matched keypoints: {matches}")
    print(f"Match rate: {match_rate:.1f}%")
    print(f"(Higher rate = better rotation invariance)")

    return matches, match_rate


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Create a test image with corners and edges
    test_img = np.zeros((120, 160), dtype=np.uint8)
    # Checkerboard-like pattern for many corners
    for r in range(6):
        for c in range(8):
            if (r + c) % 2 == 0:
                test_img[r*20:(r+1)*20, c*20:(c+1)*20] = 200
    # Add some diagonal lines
    for i in range(80):
        if i < 120 and i < 160:
            test_img[i, i] = 255
    # Add bright spots
    test_img[30:35, 80:85] = 250
    test_img[70:75, 120:125] = 250

    print("\n>>> Exercise 1: Select Best Keypoints")
    exercise_1_select_best_keypoints(test_img, n=20)

    print("\n>>> Exercise 2: Uniformly Distributed Keypoints")
    exercise_2_uniform_keypoints(test_img, grid_size=(6, 8))

    print("\n>>> Exercise 3: Rotation Invariance Test")
    exercise_3_rotation_invariance(test_img, angle=30)

    print("\nAll exercises completed successfully.")
