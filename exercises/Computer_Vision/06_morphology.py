"""
Exercise Solutions for Lesson 06: Morphological Operations
Computer Vision - Erosion, Dilation, Opening, Closing

Topics covered:
- Structuring element effects comparison
- Character thickness adjustment
- Boundary extraction comparison (3 methods)
- Braille recognition preprocessing
- Cell separation (Watershed preprocessing)
"""

import numpy as np


# =============================================================================
# Helper: Morphological Operations
# =============================================================================

def erode(img, kernel, iterations=1):
    """Morphological erosion using minimum filter."""
    result = img.copy()
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    for _ in range(iterations):
        padded = np.pad(result, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
        temp = np.zeros_like(result)
        h, w = result.shape
        for i in range(h):
            for j in range(w):
                region = padded[i:i + kh, j:j + kw]
                # Erosion: pixel is foreground only if ALL kernel positions are foreground
                temp[i, j] = 255 if np.all(region[kernel == 1] == 255) else 0
        result = temp
    return result


def dilate(img, kernel, iterations=1):
    """Morphological dilation using maximum filter."""
    result = img.copy()
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    for _ in range(iterations):
        padded = np.pad(result, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
        temp = np.zeros_like(result)
        h, w = result.shape
        for i in range(h):
            for j in range(w):
                region = padded[i:i + kh, j:j + kw]
                # Dilation: pixel is foreground if ANY kernel position is foreground
                temp[i, j] = 255 if np.any(region[kernel == 1] == 255) else 0
        result = temp
    return result


def morph_open(img, kernel):
    """Opening = erosion then dilation. Removes small foreground noise."""
    return dilate(erode(img, kernel), kernel)


def morph_close(img, kernel):
    """Closing = dilation then erosion. Fills small holes in foreground."""
    return erode(dilate(img, kernel), kernel)


def morph_gradient(img, kernel):
    """Morphological gradient = dilation - erosion. Extracts boundaries."""
    return (dilate(img, kernel).astype(np.int16) -
            erode(img, kernel).astype(np.int16)).clip(0, 255).astype(np.uint8)


# =============================================================================
# Helper: Create Structuring Elements
# =============================================================================

def get_structuring_element(shape, size):
    """Create a structuring element of given shape and size."""
    if shape == 'rect':
        return np.ones((size, size), dtype=np.uint8)
    elif shape == 'cross':
        se = np.zeros((size, size), dtype=np.uint8)
        mid = size // 2
        se[mid, :] = 1
        se[:, mid] = 1
        return se
    elif shape == 'ellipse':
        se = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if ((i - center) / (size / 2.0)) ** 2 + ((j - center) / (size / 2.0)) ** 2 <= 1:
                    se[i, j] = 1
        return se
    else:
        return np.ones((size, size), dtype=np.uint8)


# =============================================================================
# Exercise 1: Compare Structuring Element Effects
# =============================================================================

def exercise_1_structuring_element_effects():
    """
    Apply erosion and dilation to the same binary image using RECT, CROSS,
    and ELLIPSE structuring elements and compare the results.
    """
    # Create a binary test image with various shapes
    img = np.zeros((100, 100), dtype=np.uint8)
    # Circle
    yy, xx = np.ogrid[-30:30, -30:30]
    circle = (xx**2 + yy**2 <= 20**2).astype(np.uint8) * 255
    img[20:80, 20:80] = circle[:60, :60]
    # Add small noise dots
    np.random.seed(42)
    noise_y = np.random.randint(0, 100, 15)
    noise_x = np.random.randint(0, 100, 15)
    img[noise_y, noise_x] = 255

    shapes = ['rect', 'cross', 'ellipse']
    se_size = 5

    print(f"{'Operation':>12} | {'SE Shape':>8} | {'FG pixels':>10} | {'Change':>8}")
    print("-" * 50)

    original_fg = np.sum(img > 0)
    print(f"{'Original':>12} | {'---':>8} | {original_fg:>10} | {'---':>8}")

    for shape in shapes:
        se = get_structuring_element(shape, se_size)

        eroded = erode(img, se)
        dilated = dilate(img, se)

        er_fg = np.sum(eroded > 0)
        di_fg = np.sum(dilated > 0)

        print(f"{'Eroded':>12} | {shape:>8} | {er_fg:>10} | {er_fg - original_fg:>+8}")
        print(f"{'Dilated':>12} | {shape:>8} | {di_fg:>10} | {di_fg - original_fg:>+8}")

    print(f"\nStructuring element pixel counts:")
    for shape in shapes:
        se = get_structuring_element(shape, se_size)
        print(f"  {shape}: {np.sum(se)} active pixels out of {se_size*se_size}")


# =============================================================================
# Exercise 2: Adjust Character Thickness
# =============================================================================

def exercise_2_adjust_stroke_width(img, amount):
    """
    Adjust thickness of characters in a binary image.
    amount > 0: Thicken with dilation
    amount < 0: Thin with erosion

    Parameters:
        img: binary image (0 and 255 values)
        amount: positive for thicker, negative for thinner

    Returns:
        adjusted image
    """
    se = get_structuring_element('ellipse', 3)

    if amount > 0:
        result = dilate(img, se, iterations=abs(amount))
        operation = "dilation"
    elif amount < 0:
        result = erode(img, se, iterations=abs(amount))
        operation = "erosion"
    else:
        result = img.copy()
        operation = "none"

    original_fg = np.sum(img > 0)
    result_fg = np.sum(result > 0)
    print(f"Stroke adjustment: amount={amount:+d} ({operation})")
    print(f"  FG pixels: {original_fg} -> {result_fg} ({result_fg - original_fg:+d})")

    return result


# =============================================================================
# Exercise 3: Compare Boundary Extraction
# =============================================================================

def exercise_3_boundary_extraction(img):
    """
    Extract object boundaries using three methods and compare:
    1. Morphological gradient (dilation - erosion)
    2. Simple edge detection (Sobel-like gradient magnitude)
    3. Contour tracing (find transitions in binary image)

    Parameters:
        img: binary image (H, W) with 0 and 255

    Returns:
        dict with three boundary images
    """
    h, w = img.shape

    # Method 1: Morphological gradient
    se = get_structuring_element('rect', 3)
    morph_boundary = morph_gradient(img, se)

    # Method 2: Sobel-like gradient magnitude
    # Horizontal and vertical Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    padded = np.pad(img.astype(np.float64), 1, mode='reflect')
    gx = np.zeros((h, w), dtype=np.float64)
    gy = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(region * sobel_x)
            gy[i, j] = np.sum(region * sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    sobel_boundary = (magnitude > magnitude.max() * 0.3).astype(np.uint8) * 255

    # Method 3: Simple contour tracing (find pixels adjacent to background)
    padded_bin = np.pad(img, 1, mode='constant', constant_values=0)
    contour_boundary = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                # Check 4-connected neighbors
                neighbors = [
                    padded_bin[i, j+1], padded_bin[i+2, j+1],
                    padded_bin[i+1, j], padded_bin[i+1, j+2]
                ]
                if any(n == 0 for n in neighbors):
                    contour_boundary[i, j] = 255

    results = {
        'morphological': morph_boundary,
        'sobel': sobel_boundary,
        'contour': contour_boundary,
    }

    for name, boundary in results.items():
        edge_count = np.sum(boundary > 0)
        print(f"  {name:>15}: {edge_count} edge pixels")

    return results


# =============================================================================
# Exercise 4: Braille Recognition Preprocessing
# =============================================================================

def exercise_4_braille_preprocessing():
    """
    Design a preprocessing pipeline to detect individual dots in a
    synthetic braille-like pattern.

    Pipeline:
    1. Threshold to binary
    2. Erosion to separate touching dots
    3. Connected component counting

    Returns:
        (preprocessed_image, dot_count)
    """
    # Create synthetic braille-like image (dots on white background)
    img = np.ones((100, 150), dtype=np.uint8) * 240  # white background

    # Add dots (dark circles) at braille positions
    dot_positions = [
        (20, 20), (20, 40), (20, 60),  # Column 1
        (40, 20), (40, 60),             # Column 2
        (70, 30), (70, 50),             # Column 3
        (90, 20), (90, 40), (90, 60),   # Column 4
        (120, 20), (120, 50),           # Column 5
    ]

    for cx, cy in dot_positions:
        yy, xx = np.ogrid[-6:7, -6:7]
        mask = (xx**2 + yy**2 <= 5**2)
        y_start = max(0, cy - 6)
        x_start = max(0, cx - 6)
        y_end = min(100, cy + 7)
        x_end = min(150, cx + 7)
        my_start = max(0, 6 - cy)
        mx_start = max(0, 6 - cx)
        my_end = my_start + (y_end - y_start)
        mx_end = mx_start + (x_end - x_start)
        img[y_start:y_end, x_start:x_end][mask[my_start:my_end, mx_start:mx_end]] = 30

    print(f"Original image: {img.shape}")

    # Step 1: Threshold (invert so dots are white on black)
    binary = np.where(img < 128, 255, 0).astype(np.uint8)
    print(f"Step 1 (Threshold): FG pixels = {np.sum(binary > 0)}")

    # Step 2: Erosion to clean up and separate any touching dots
    se = get_structuring_element('ellipse', 3)
    cleaned = erode(binary, se, iterations=1)
    print(f"Step 2 (Erosion): FG pixels = {np.sum(cleaned > 0)}")

    # Step 3: Dilation to restore dot size
    restored = dilate(cleaned, se, iterations=1)
    print(f"Step 3 (Dilation): FG pixels = {np.sum(restored > 0)}")

    # Step 4: Count connected components (simple flood fill)
    labeled = np.zeros_like(restored, dtype=int)
    label_id = 0
    visited = np.zeros_like(restored, dtype=bool)

    def flood_fill(start_y, start_x, label):
        stack = [(start_y, start_x)]
        while stack:
            cy, cx = stack.pop()
            if (cy < 0 or cy >= restored.shape[0] or
                cx < 0 or cx >= restored.shape[1]):
                continue
            if visited[cy, cx] or restored[cy, cx] == 0:
                continue
            visited[cy, cx] = True
            labeled[cy, cx] = label
            stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])

    for i in range(restored.shape[0]):
        for j in range(restored.shape[1]):
            if restored[i, j] > 0 and not visited[i, j]:
                label_id += 1
                flood_fill(i, j, label_id)

    print(f"Step 4 (Connected components): {label_id} dots detected")
    print(f"Expected dots: {len(dot_positions)}")

    return restored, label_id


# =============================================================================
# Exercise 5: Cell Separation (Watershed Preprocessing)
# =============================================================================

def exercise_5_cell_separation():
    """
    Implement preprocessing to separate connected cells:
    1. Binarization
    2. Noise removal (opening/closing)
    3. Find sure background area (dilation)
    4. Find sure foreground area (distance transform + threshold)

    Returns:
        dict with intermediate results
    """
    # Create synthetic microscope cell image (overlapping circles)
    img = np.ones((120, 120), dtype=np.uint8) * 200  # Light background

    # Draw "cells" as dark circles, some overlapping
    cell_centers = [(30, 30), (50, 45), (75, 30), (35, 75), (65, 70), (90, 60)]
    cell_radii = [15, 18, 14, 16, 13, 15]

    for (cx, cy), r in zip(cell_centers, cell_radii):
        yy, xx = np.ogrid[:120, :120]
        mask = ((xx - cx)**2 + (yy - cy)**2) <= r**2
        img[mask] = np.clip(img[mask].astype(int) - 120, 30, 255).astype(np.uint8)

    print(f"Synthetic cell image: {img.shape}")

    # Step 1: Binarization
    binary = np.where(img < 128, 255, 0).astype(np.uint8)
    print(f"Step 1 (Binary): FG = {np.sum(binary > 0)} pixels")

    # Step 2: Opening to remove noise, then closing to fill holes
    se = get_structuring_element('ellipse', 3)
    opened = morph_open(binary, se)
    cleaned = morph_close(opened, se)
    print(f"Step 2 (Open/Close): FG = {np.sum(cleaned > 0)} pixels")

    # Step 3: Sure background = dilation of cleaned image
    se_large = get_structuring_element('ellipse', 5)
    sure_bg = dilate(cleaned, se_large, iterations=2)
    print(f"Step 3 (Sure BG via dilation): BG region = {np.sum(sure_bg == 0)} pixels")

    # Step 4: Distance transform for sure foreground
    # Simple Euclidean distance transform using iterative approximation
    dist = np.zeros_like(cleaned, dtype=np.float64)
    fg_mask = cleaned > 0

    # Compute distance to nearest background pixel for each foreground pixel
    bg_coords = np.argwhere(~fg_mask)
    if len(bg_coords) > 0:
        for i in range(cleaned.shape[0]):
            for j in range(cleaned.shape[1]):
                if fg_mask[i, j]:
                    dists = np.sqrt((bg_coords[:, 0] - i)**2 + (bg_coords[:, 1] - j)**2)
                    dist[i, j] = np.min(dists)

    # Threshold distance transform to get sure foreground
    dist_max = dist.max() if dist.max() > 0 else 1
    sure_fg = np.where(dist > 0.5 * dist_max, 255, 0).astype(np.uint8)

    # Unknown region = sure_bg - sure_fg
    unknown = np.where((sure_bg > 0) & (sure_fg == 0), 255, 0).astype(np.uint8)

    fg_count = np.sum(sure_fg > 0)
    unknown_count = np.sum(unknown > 0)
    print(f"Step 4 (Distance transform):")
    print(f"  Max distance: {dist_max:.1f}")
    print(f"  Sure FG pixels: {fg_count}")
    print(f"  Unknown region: {unknown_count}")

    results = {
        'original': img,
        'binary': binary,
        'cleaned': cleaned,
        'sure_bg': sure_bg,
        'distance': dist,
        'sure_fg': sure_fg,
        'unknown': unknown,
    }

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Structuring Element Effects")
    exercise_1_structuring_element_effects()

    print("\n>>> Exercise 2: Adjust Character Thickness")
    # Create a simple "T" character
    char_img = np.zeros((40, 30), dtype=np.uint8)
    char_img[5:8, 5:25] = 255   # Top bar
    char_img[8:35, 13:17] = 255  # Vertical bar
    exercise_2_adjust_stroke_width(char_img, amount=2)
    exercise_2_adjust_stroke_width(char_img, amount=-1)

    print("\n>>> Exercise 3: Boundary Extraction Comparison")
    shape_img = np.zeros((60, 60), dtype=np.uint8)
    shape_img[15:45, 15:45] = 255  # Square
    exercise_3_boundary_extraction(shape_img)

    print("\n>>> Exercise 4: Braille Preprocessing")
    exercise_4_braille_preprocessing()

    print("\n>>> Exercise 5: Cell Separation")
    exercise_5_cell_separation()

    print("\nAll exercises completed successfully.")
