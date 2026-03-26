"""
Exercise Solutions for Lesson 08: Edge Detection
Computer Vision - Sobel, Canny, Gradient Direction

Topics covered:
- Adaptive Canny with automatic threshold selection
- Separating edges by direction (horizontal vs vertical)
- Multi-scale edge detection combining multiple blur levels
"""

import numpy as np


# =============================================================================
# Helper: Convolution
# =============================================================================

def convolve2d(img, kernel):
    """2D convolution with reflect padding."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    h, w = img.shape
    padded = np.pad(img.astype(np.float64), ((ph, ph), (pw, pw)), mode='reflect')
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return result


# =============================================================================
# Exercise 1: Adaptive Canny
# =============================================================================

def exercise_1_adaptive_canny(img, sigma=0.33):
    """
    Implement adaptive Canny edge detection that automatically sets
    thresholds based on the median brightness of the image.

    Parameters:
        img: grayscale image (H, W) uint8
        sigma: controls threshold spread around median (default 0.33)

    Returns:
        (edges, low_thresh, high_thresh)
    """
    # Step 1: Gaussian blur
    ksize = 5
    ax = np.arange(-ksize//2 + 1, ksize//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    g_kernel = np.exp(-(xx**2 + yy**2) / (2 * 1.4**2))
    g_kernel /= g_kernel.sum()
    blurred = convolve2d(img, g_kernel)

    # Step 2: Calculate adaptive thresholds from median
    median_val = np.median(blurred)
    low = int(max(0, (1.0 - sigma) * median_val))
    high = int(min(255, (1.0 + sigma) * median_val))

    # Step 3: Sobel gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = convolve2d(blurred, sobel_x)
    gy = convolve2d(blurred, sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * 180 / np.pi
    direction = direction % 180  # Map to [0, 180)

    # Step 4: Non-maximum suppression
    h, w = img.shape
    nms = np.zeros((h, w), dtype=np.float64)

    for i in range(1, h-1):
        for j in range(1, w-1):
            angle = direction[i, j]
            mag = magnitude[i, j]

            # Determine neighbor pixels based on gradient direction
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                n1, n2 = magnitude[i, j-1], magnitude[i, j+1]
            elif 22.5 <= angle < 67.5:
                n1, n2 = magnitude[i-1, j+1], magnitude[i+1, j-1]
            elif 67.5 <= angle < 112.5:
                n1, n2 = magnitude[i-1, j], magnitude[i+1, j]
            else:
                n1, n2 = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if mag >= n1 and mag >= n2:
                nms[i, j] = mag

    # Step 5: Hysteresis thresholding
    strong = (nms >= high).astype(np.uint8) * 255
    weak = ((nms >= low) & (nms < high)).astype(np.uint8) * 255

    edges = strong.copy()
    changed = True
    while changed:
        changed = False
        padded = np.pad(edges, 1, mode='constant', constant_values=0)
        for i in range(h):
            for j in range(w):
                if weak[i, j] > 0 and edges[i, j] == 0:
                    if np.any(padded[i:i+3, j:j+3] == 255):
                        edges[i, j] = 255
                        changed = True

    print(f"Adaptive Canny (sigma={sigma}):")
    print(f"  Median brightness: {median_val:.1f}")
    print(f"  Thresholds: low={low}, high={high}")
    print(f"  Edge pixels: {np.sum(edges > 0)}")

    return edges, low, high


# =============================================================================
# Exercise 2: Separate Edges by Direction
# =============================================================================

def exercise_2_separate_edges_by_direction(img, angle_threshold=30):
    """
    Separate horizontal and vertical edges based on gradient direction.

    Parameters:
        img: grayscale image (H, W) uint8
        angle_threshold: allowed deviation from horizontal/vertical (degrees)

    Returns:
        (horizontal_edges, vertical_edges)
    """
    # Gaussian blur
    ksize = 5
    ax = np.arange(-ksize//2 + 1, ksize//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    g_kernel = np.exp(-(xx**2 + yy**2) / 2.0)
    g_kernel /= g_kernel.sum()
    blurred = convolve2d(img, g_kernel)

    # Sobel gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = convolve2d(blurred, sobel_x)
    gy = convolve2d(blurred, sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    # Gradient direction in degrees [0, 180)
    direction = np.degrees(np.arctan2(gy, gx)) % 180

    # Edge threshold
    edge_mask = magnitude > 50

    # Horizontal edges: gradient direction near 90 degrees (vertical gradient
    # means the edge itself is horizontal)
    horizontal_mask = (
        (direction > 90 - angle_threshold) &
        (direction < 90 + angle_threshold)
    )
    horizontal_edges = np.zeros_like(img)
    horizontal_edges[horizontal_mask & edge_mask] = 255

    # Vertical edges: gradient direction near 0 or 180 degrees
    vertical_mask = (
        (direction < angle_threshold) |
        (direction > 180 - angle_threshold)
    )
    vertical_edges = np.zeros_like(img)
    vertical_edges[vertical_mask & edge_mask] = 255

    h_count = np.sum(horizontal_edges > 0)
    v_count = np.sum(vertical_edges > 0)
    total_edges = np.sum(edge_mask)

    print(f"Edge Direction Separation (threshold={angle_threshold} deg):")
    print(f"  Total edge pixels:      {total_edges}")
    print(f"  Horizontal edge pixels: {h_count}")
    print(f"  Vertical edge pixels:   {v_count}")
    print(f"  Other direction:        {total_edges - h_count - v_count}")

    return horizontal_edges, vertical_edges


# =============================================================================
# Exercise 3: Multi-Scale Edge Detection
# =============================================================================

def exercise_3_multi_scale_canny(img, scales=(1.0, 2.0, 4.0), low=50, high=150):
    """
    Detect edges at multiple Gaussian blur scales and combine them.
    Larger sigma captures coarser edges while smaller sigma captures fine detail.

    Parameters:
        img: grayscale image (H, W) uint8
        scales: tuple of sigma values for Gaussian blur
        low: Canny low threshold
        high: Canny high threshold

    Returns:
        combined edge map
    """
    h, w = img.shape
    combined = np.zeros((h, w), dtype=np.uint8)

    for sigma in scales:
        # Create Gaussian kernel
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        ax = np.arange(-ksize//2 + 1, ksize//2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        g_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        g_kernel /= g_kernel.sum()

        blurred = convolve2d(img, g_kernel)

        # Sobel gradients
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        gx = convolve2d(blurred, sobel_x)
        gy = convolve2d(blurred, sobel_y)
        magnitude = np.sqrt(gx**2 + gy**2)

        # Simple threshold-based edge detection at this scale
        edges = np.where(magnitude > high, 255, 0).astype(np.uint8)

        # Add weak edges connected to strong
        weak = ((magnitude >= low) & (magnitude < high))
        padded = np.pad(edges, 1, mode='constant', constant_values=0)
        for i in range(h):
            for j in range(w):
                if weak[i, j] and np.any(padded[i:i+3, j:j+3] == 255):
                    edges[i, j] = 255

        edge_count = np.sum(edges > 0)
        print(f"  Scale sigma={sigma:.1f}: ksize={ksize}, edges={edge_count}")

        # Combine using OR
        combined = np.maximum(combined, edges)

    total_combined = np.sum(combined > 0)
    print(f"Combined edges: {total_combined}")

    return combined


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Adaptive Canny")
    # Create image with edges at various contrasts
    test_img = np.zeros((80, 80), dtype=np.uint8)
    test_img[:, 40:] = 180       # Strong vertical edge
    test_img[20:60, 20:60] = 120  # Moderate rectangle
    test_img[35:45, 35:45] = 60   # Darker center
    exercise_1_adaptive_canny(test_img, sigma=0.33)

    print("\n>>> Exercise 2: Separate Edges by Direction")
    # Create image with clear horizontal and vertical edges
    dir_img = np.zeros((80, 80), dtype=np.uint8)
    dir_img[20:22, 10:70] = 200   # Horizontal line
    dir_img[50:52, 10:70] = 200   # Horizontal line
    dir_img[10:70, 30:32] = 200   # Vertical line
    dir_img[10:70, 60:62] = 200   # Vertical line
    exercise_2_separate_edges_by_direction(dir_img, angle_threshold=30)

    print("\n>>> Exercise 3: Multi-Scale Edge Detection")
    # Create image with features at different scales
    multi_img = np.zeros((80, 80), dtype=np.uint8)
    multi_img[10:70, 10:70] = 150    # Large rectangle
    multi_img[25:55, 25:55] = 80     # Medium rectangle
    multi_img[35:45, 35:45] = 200    # Small bright square
    # Add fine texture
    multi_img[15:65:4, 15:65] = np.clip(
        multi_img[15:65:4, 15:65].astype(int) + 40, 0, 255
    ).astype(np.uint8)
    exercise_3_multi_scale_canny(multi_img, scales=(1.0, 2.0, 3.0), low=30, high=100)

    print("\nAll exercises completed successfully.")
