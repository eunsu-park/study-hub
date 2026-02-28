"""
Exercise Solutions for Lesson 09: Contours
Computer Vision - findContours, drawContours, Hierarchy

Topics covered:
- Coin counter (classify by size)
- Document rectangle detection
- Detect empty spaces (holes) in binary image
"""

import numpy as np


# =============================================================================
# Helper: Simple contour finder for binary images
# =============================================================================

def find_contours_simple(binary):
    """
    Find external contours in a binary image using border following.
    Returns list of contours, each a list of (x, y) boundary points.
    """
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    contours = []

    # 8-connected neighbor offsets (clockwise from right)
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]

    for i in range(h):
        for j in range(w):
            if binary[i, j] > 0 and not visited[i, j]:
                # Check if it's a boundary pixel (has at least one bg neighbor)
                is_boundary = False
                for d in range(8):
                    ny, nx = i + dy[d], j + dx[d]
                    if ny < 0 or ny >= h or nx < 0 or nx >= w or binary[ny, nx] == 0:
                        is_boundary = True
                        break

                if not is_boundary:
                    visited[i, j] = True
                    continue

                # Trace contour
                contour = []
                cy, cx = i, j
                start_y, start_x = i, j
                contour.append((cx, cy))
                visited[cy, cx] = True

                # Simple flood-fill based boundary tracing
                stack = [(cy, cx)]
                while stack:
                    py, px = stack.pop()
                    for d in range(8):
                        ny, nx = py + dy[d], px + dx[d]
                        if (0 <= ny < h and 0 <= nx < w and
                            binary[ny, nx] > 0 and not visited[ny, nx]):
                            # Check if boundary
                            is_bound = False
                            for d2 in range(8):
                                nny, nnx = ny + dy[d2], nx + dx[d2]
                                if nny < 0 or nny >= h or nnx < 0 or nnx >= w or binary[nny, nnx] == 0:
                                    is_bound = True
                                    break
                            if is_bound:
                                contour.append((nx, ny))
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                if len(contour) > 2:
                    contours.append(np.array(contour))

    return contours


def contour_area(contour):
    """Calculate contour area using the Shoelace formula."""
    n = len(contour)
    if n < 3:
        return 0
    x = contour[:, 0].astype(np.float64)
    y = contour[:, 1].astype(np.float64)
    area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) +
                      x[-1] * y[0] - x[0] * y[-1])
    return area


def contour_perimeter(contour):
    """Calculate contour perimeter."""
    n = len(contour)
    if n < 2:
        return 0
    diffs = np.diff(contour.astype(np.float64), axis=0)
    lengths = np.sqrt(np.sum(diffs**2, axis=1))
    # Add closing segment
    closing = np.sqrt(np.sum((contour[-1].astype(np.float64) - contour[0].astype(np.float64))**2))
    return np.sum(lengths) + closing


def min_enclosing_circle(contour):
    """Approximate minimum enclosing circle using centroid and max distance."""
    cx = np.mean(contour[:, 0])
    cy = np.mean(contour[:, 1])
    dists = np.sqrt((contour[:, 0] - cx)**2 + (contour[:, 1] - cy)**2)
    radius = np.max(dists)
    return (cx, cy), radius


# =============================================================================
# Exercise 1: Coin Counter
# =============================================================================

def exercise_1_coin_counter():
    """
    Create a synthetic coin image, detect contours, classify coins by size,
    and calculate total value.

    Returns:
        (coin_counts, total_value)
    """
    # Create synthetic image with circles of different sizes (coins)
    h, w = 200, 300
    img = np.zeros((h, w), dtype=np.uint8)

    # Draw coins as filled circles
    coins_info = [
        # (cx, cy, radius, value)
        (50, 50, 18, 10),    # Small = 10 won
        (120, 50, 18, 10),   # Small
        (50, 120, 28, 50),   # Medium = 50 won
        (130, 130, 28, 50),  # Medium
        (220, 60, 35, 100),  # Large = 100 won
        (230, 150, 35, 100), # Large
        (180, 50, 18, 10),   # Small
    ]

    for cx, cy, r, _ in coins_info:
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx)**2 + (yy - cy)**2) <= r**2
        img[mask] = 255

    # Find contours
    contours = find_contours_simple(img)

    # Classify by size
    small_coins = []   # 10 won (radius < 22)
    medium_coins = []  # 50 won (22 <= radius < 32)
    large_coins = []   # 100 won (radius >= 32)

    print(f"Detected {len(contours)} contours")

    for contour in contours:
        area = contour_area(contour)
        if area < 200:  # Noise filter
            continue

        center, radius = min_enclosing_circle(contour)

        # Check circularity
        perimeter = contour_perimeter(contour)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.5:  # Not circular enough
                continue

        if radius < 22:
            small_coins.append((center, radius))
        elif radius < 32:
            medium_coins.append((center, radius))
        else:
            large_coins.append((center, radius))

    total = len(small_coins) * 10 + len(medium_coins) * 50 + len(large_coins) * 100

    print(f"\nCoin Classification:")
    print(f"  10 won (small):  {len(small_coins)}")
    print(f"  50 won (medium): {len(medium_coins)}")
    print(f"  100 won (large): {len(large_coins)}")
    print(f"  Total value:     {total} won")

    return {'small': len(small_coins), 'medium': len(medium_coins),
            'large': len(large_coins)}, total


# =============================================================================
# Exercise 2: Document Rectangle Detection
# =============================================================================

def exercise_2_document_detection():
    """
    Find the largest 4-sided contour in an image (simulating document detection).
    Use polygon approximation to find the 4 vertices.

    Returns:
        4 corner points or None
    """
    # Create synthetic scene with a tilted rectangle (document)
    h, w = 200, 300
    scene = np.zeros((h, w), dtype=np.uint8)

    # Draw a quadrilateral representing a document
    doc_corners = np.array([(50, 30), (250, 40), (240, 170), (60, 160)])

    # Fill the quadrilateral
    for y in range(h):
        for x in range(w):
            # Point-in-polygon test using ray casting
            n = len(doc_corners)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = doc_corners[i]
                xj, yj = doc_corners[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            if inside:
                scene[y, x] = 200

    # Add some noise/smaller objects
    scene[5:15, 5:15] = 150
    scene[180:195, 270:290] = 180

    # Find contours
    binary = np.where(scene > 100, 255, 0).astype(np.uint8)
    contours = find_contours_simple(binary)

    # Sort by area, largest first
    contour_areas = [(c, contour_area(c)) for c in contours]
    contour_areas.sort(key=lambda x: x[1], reverse=True)

    print(f"Found {len(contours)} contours")

    document_corners = None

    for contour, area in contour_areas[:5]:
        if area < 100:
            continue

        # Polygon approximation (Douglas-Peucker simplification)
        perimeter = contour_perimeter(contour)
        epsilon = 0.02 * perimeter

        # Simple approximation: find the 4 points farthest from centroid
        cx_c = np.mean(contour[:, 0])
        cy_c = np.mean(contour[:, 1])
        dists = np.sqrt((contour[:, 0] - cx_c)**2 + (contour[:, 1] - cy_c)**2)

        # Get indices of top N farthest points
        if len(contour) >= 4:
            # Group points by quadrant relative to centroid
            quadrants = {0: [], 1: [], 2: [], 3: []}
            for idx, (px, py) in enumerate(contour):
                q = (0 if px >= cx_c else 1) + (0 if py >= cy_c else 2)
                quadrants[q].append((idx, dists[idx]))

            corners = []
            for q in range(4):
                if quadrants[q]:
                    best_idx = max(quadrants[q], key=lambda x: x[1])[0]
                    corners.append(contour[best_idx])

            if len(corners) == 4:
                document_corners = np.array(corners)
                print(f"Document found! Area = {area:.0f}")
                print(f"Corners: {document_corners.tolist()}")
                break

    if document_corners is None:
        print("No document found.")

    return document_corners


# =============================================================================
# Exercise 3: Detect Empty Spaces (Holes)
# =============================================================================

def exercise_3_count_holes():
    """
    Count holes inside objects in a binary image.
    Uses connected component analysis with hierarchy-like logic.

    Returns:
        number of holes
    """
    # Create image with objects containing holes
    h, w = 150, 150
    img = np.zeros((h, w), dtype=np.uint8)

    # Object 1: Ring (1 hole)
    yy, xx = np.ogrid[:h, :w]
    outer1 = ((xx - 40)**2 + (yy - 40)**2) <= 30**2
    inner1 = ((xx - 40)**2 + (yy - 40)**2) <= 15**2
    img[outer1 & ~inner1] = 255

    # Object 2: Figure-8 shape (2 holes)
    outer2a = ((xx - 110)**2 + (yy - 35)**2) <= 25**2
    inner2a = ((xx - 110)**2 + (yy - 35)**2) <= 10**2
    outer2b = ((xx - 110)**2 + (yy - 75)**2) <= 25**2
    inner2b = ((xx - 110)**2 + (yy - 75)**2) <= 10**2
    img[(outer2a & ~inner2a) | (outer2b & ~inner2b)] = 255

    # Object 3: Solid rectangle (0 holes)
    img[100:140, 20:80] = 255

    print(f"Image with objects and holes: {w}x{h}")

    # Count holes using flood fill approach:
    # 1. Flood fill background from edges
    # 2. Remaining black pixels inside white objects are holes
    # 3. Count connected black regions that are NOT background

    # Step 1: Label background by flood filling from edges
    is_background = np.zeros((h, w), dtype=bool)
    stack = []

    # Add all edge pixels that are black
    for i in range(h):
        if img[i, 0] == 0:
            stack.append((i, 0))
        if img[i, w-1] == 0:
            stack.append((i, w-1))
    for j in range(w):
        if img[0, j] == 0:
            stack.append((0, j))
        if img[h-1, j] == 0:
            stack.append((h-1, j))

    while stack:
        cy, cx = stack.pop()
        if cy < 0 or cy >= h or cx < 0 or cx >= w:
            continue
        if is_background[cy, cx] or img[cy, cx] > 0:
            continue
        is_background[cy, cx] = True
        stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])

    # Step 2: Find black pixels that are NOT background (= holes)
    holes_mask = (img == 0) & (~is_background)

    # Step 3: Count connected components in holes_mask
    visited = np.zeros((h, w), dtype=bool)
    hole_count = 0

    for i in range(h):
        for j in range(w):
            if holes_mask[i, j] and not visited[i, j]:
                # Found a new hole, flood fill to mark it
                hole_count += 1
                hole_size = 0
                fill_stack = [(i, j)]
                while fill_stack:
                    cy, cx = fill_stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w:
                        continue
                    if visited[cy, cx] or not holes_mask[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    hole_size += 1
                    fill_stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])

                if hole_size < 5:  # Filter noise
                    hole_count -= 1
                else:
                    print(f"  Hole #{hole_count}: {hole_size} pixels")

    print(f"\nTotal holes detected: {hole_count}")
    print(f"Expected: 3 (ring=1, figure-8=2)")

    return hole_count


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Coin Counter")
    exercise_1_coin_counter()

    print("\n>>> Exercise 2: Document Rectangle Detection")
    exercise_2_document_detection()

    print("\n>>> Exercise 3: Detect Empty Spaces (Holes)")
    exercise_3_count_holes()

    print("\nAll exercises completed successfully.")
