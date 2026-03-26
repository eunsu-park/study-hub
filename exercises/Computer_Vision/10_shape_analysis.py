"""
Exercise Solutions for Lesson 10: Shape Analysis
Computer Vision - Moments, Contour Properties, Circularity

Topics covered:
- Sort shapes by area and display rankings
- Find rectangles with specific aspect ratio (2:1)
- Find the most circular shape
"""

import numpy as np


# =============================================================================
# Helper functions
# =============================================================================

def create_shapes_image():
    """Create a synthetic image with various shapes for analysis."""
    h, w = 200, 300
    img = np.zeros((h, w), dtype=np.uint8)

    # Triangle (small area)
    pts = np.array([(40, 60), (20, 90), (60, 90)])
    for y in range(h):
        for x in range(w):
            # Point in triangle using barycentric coordinates
            v0 = pts[2] - pts[0]
            v1 = pts[1] - pts[0]
            v2 = np.array([x, y]) - pts[0]
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d02 = np.dot(v0, v2)
            d11 = np.dot(v1, v1)
            d12 = np.dot(v1, v2)
            inv_denom = 1.0 / (d00 * d11 - d01 * d01) if (d00 * d11 - d01 * d01) != 0 else 0
            u = (d11 * d02 - d01 * d12) * inv_denom
            v = (d00 * d12 - d01 * d02) * inv_denom
            if u >= 0 and v >= 0 and u + v <= 1:
                img[y, x] = 255

    # Circle (medium area, high circularity)
    yy, xx = np.ogrid[:h, :w]
    circle_mask = ((xx - 130)**2 + (yy - 75)**2) <= 30**2
    img[circle_mask] = 255

    # Large rectangle (largest area)
    img[30:90, 190:280] = 255

    # Small square
    img[130:155, 30:55] = 255

    # Elongated rectangle (2:1 ratio)
    img[120:150, 100:160] = 255  # 60x30 = 2:1

    # Ellipse (medium circularity)
    ellipse_mask = (((xx - 230) / 25.0)**2 + ((yy - 150) / 15.0)**2) <= 1
    img[ellipse_mask] = 255

    return img


def find_contours_labeled(binary):
    """Find contours and return them with basic properties."""
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    contours = []

    for i in range(h):
        for j in range(w):
            if binary[i, j] > 0 and not visited[i, j]:
                # Flood fill to find connected component
                component = []
                boundary = []
                stack = [(i, j)]
                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w:
                        continue
                    if visited[cy, cx] or binary[cy, cx] == 0:
                        continue
                    visited[cy, cx] = True
                    component.append((cx, cy))

                    # Check if boundary pixel
                    is_boundary = False
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = cy+dy, cx+dx
                        if ny < 0 or ny >= h or nx < 0 or nx >= w or binary[ny, nx] == 0:
                            is_boundary = True
                            break
                    if is_boundary:
                        boundary.append((cx, cy))

                    stack.extend([(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)])

                if len(boundary) > 5:
                    contours.append({
                        'points': np.array(boundary),
                        'all_points': np.array(component),
                        'area': len(component),
                    })

    return contours


def compute_moments(contour_points):
    """Compute image moments from contour points."""
    x = contour_points[:, 0].astype(np.float64)
    y = contour_points[:, 1].astype(np.float64)
    n = len(x)

    m00 = n  # Area (number of pixels)
    m10 = np.sum(x)
    m01 = np.sum(y)
    m20 = np.sum(x * x)
    m02 = np.sum(y * y)
    m11 = np.sum(x * y)

    cx = m10 / m00 if m00 > 0 else 0
    cy = m01 / m00 if m00 > 0 else 0

    return {
        'm00': m00, 'm10': m10, 'm01': m01,
        'm20': m20, 'm02': m02, 'm11': m11,
        'cx': cx, 'cy': cy
    }


# =============================================================================
# Exercise 1: Sort Shapes by Area
# =============================================================================

def exercise_1_sort_shapes_by_area():
    """
    Sort detected shapes by area and display rankings.

    Returns:
        list of (rank, area, centroid) tuples
    """
    img = create_shapes_image()
    contours = find_contours_labeled(img)

    print(f"Detected {len(contours)} shapes")

    # Sort by area (descending)
    contours.sort(key=lambda c: c['area'], reverse=True)

    rankings = []
    print(f"\n{'Rank':>5} | {'Area':>8} | {'Centroid':>15} | {'BBox':>20}")
    print("-" * 60)

    for rank, contour in enumerate(contours, 1):
        area = contour['area']
        moments = compute_moments(contour['all_points'])
        cx, cy = moments['cx'], moments['cy']

        # Bounding box
        pts = contour['all_points']
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        bbox = f"({x_min},{y_min})-({x_max},{y_max})"

        rankings.append((rank, area, (cx, cy)))
        print(f"{rank:>5} | {area:>8} | ({cx:>5.1f}, {cy:>5.1f}) | {bbox:>20}")

    return rankings


# =============================================================================
# Exercise 2: Find Specific Aspect Ratio (2:1)
# =============================================================================

def exercise_2_find_2to1_rectangles(tolerance=0.3):
    """
    Detect rectangles with approximately 2:1 aspect ratio.

    Parameters:
        tolerance: allowed deviation from 2.0 ratio

    Returns:
        list of matching rectangles with properties
    """
    img = create_shapes_image()
    contours = find_contours_labeled(img)

    target_ratio = 2.0
    found = []

    print(f"Searching for rectangles with {target_ratio}:1 aspect ratio "
          f"(tolerance={tolerance})")
    print(f"\n{'Shape':>6} | {'W':>5} | {'H':>5} | {'Ratio':>8} | {'Match':>6}")
    print("-" * 45)

    for idx, contour in enumerate(contours):
        pts = contour['all_points']
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1

        if bbox_w == 0 or bbox_h == 0:
            continue

        # Calculate aspect ratio (always >= 1)
        aspect_ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)

        # Check rectangularity: area should be close to bbox area
        bbox_area = bbox_w * bbox_h
        fill_ratio = contour['area'] / bbox_area if bbox_area > 0 else 0

        is_rect = fill_ratio > 0.85  # Must be mostly filled (rectangular)
        ratio_match = abs(aspect_ratio - target_ratio) < tolerance

        match = is_rect and ratio_match
        status = "YES" if match else "no"

        print(f"{idx+1:>6} | {bbox_w:>5} | {bbox_h:>5} | {aspect_ratio:>7.2f} | {status:>6}")

        if match:
            found.append({
                'bbox': (x_min, y_min, bbox_w, bbox_h),
                'ratio': aspect_ratio,
                'area': contour['area'],
                'fill_ratio': fill_ratio
            })

    print(f"\nFound {len(found)} rectangles with ~2:1 ratio")
    return found


# =============================================================================
# Exercise 3: Find Most Circular Shape
# =============================================================================

def exercise_3_find_most_circular():
    """
    Find and display the shape with the highest circularity.
    Circularity = 4*pi*area / perimeter^2 (1.0 = perfect circle)

    Returns:
        (best_circularity, shape_properties)
    """
    img = create_shapes_image()
    contours = find_contours_labeled(img)

    best_circularity = 0
    best_idx = -1

    print(f"{'Shape':>6} | {'Area':>8} | {'Perimeter':>10} | {'Circularity':>12}")
    print("-" * 50)

    for idx, contour in enumerate(contours):
        area = contour['area']

        # Calculate perimeter from boundary points
        boundary = contour['points']
        n = len(boundary)
        if n < 3:
            continue

        # Sort boundary points by angle from centroid for proper perimeter
        cx = np.mean(boundary[:, 0])
        cy = np.mean(boundary[:, 1])
        angles = np.arctan2(boundary[:, 1] - cy, boundary[:, 0] - cx)
        sorted_idx = np.argsort(angles)
        sorted_boundary = boundary[sorted_idx]

        perimeter = 0
        for i in range(n):
            dx = sorted_boundary[(i+1) % n, 0] - sorted_boundary[i, 0]
            dy = sorted_boundary[(i+1) % n, 1] - sorted_boundary[i, 1]
            perimeter += np.sqrt(dx**2 + dy**2)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        marker = " <-- BEST" if circularity > best_circularity else ""
        print(f"{idx+1:>6} | {area:>8} | {perimeter:>10.1f} | {circularity:>11.4f}{marker}")

        if circularity > best_circularity:
            best_circularity = circularity
            best_idx = idx

    if best_idx >= 0:
        best = contours[best_idx]
        moments = compute_moments(best['all_points'])
        print(f"\nMost circular shape: #{best_idx+1}")
        print(f"  Circularity: {best_circularity:.4f} (1.0 = perfect circle)")
        print(f"  Area: {best['area']}")
        print(f"  Centroid: ({moments['cx']:.1f}, {moments['cy']:.1f})")

    return best_circularity, best_idx


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Sort Shapes by Area")
    exercise_1_sort_shapes_by_area()

    print("\n>>> Exercise 2: Find 2:1 Aspect Ratio Rectangles")
    exercise_2_find_2to1_rectangles(tolerance=0.3)

    print("\n>>> Exercise 3: Find Most Circular Shape")
    exercise_3_find_most_circular()

    print("\nAll exercises completed successfully.")
