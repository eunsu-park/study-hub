"""
Exercise Solutions for Lesson 12: Histogram Analysis
Computer Vision - Histogram Computation, Equalization, CLAHE

Topics covered:
- Automatic contrast adjustment (histogram stretching)
- Dominant color extraction (K-means)
- Illumination normalization for documents
"""

import numpy as np


# =============================================================================
# Exercise 1: Automatic Contrast Adjustment
# =============================================================================

def exercise_1_auto_contrast(img):
    """
    Analyze histogram and perform optimal contrast adjustment using
    histogram stretching. Handles both grayscale and color (via LAB).

    Parameters:
        img: (H, W) grayscale or (H, W, 3) BGR image, uint8

    Returns:
        contrast-adjusted image
    """
    if len(img.shape) == 3:
        # Color image: convert to LAB-like space, stretch L channel
        # Approximate L channel as weighted sum: 0.114*B + 0.587*G + 0.299*R
        b, g, r = img[:,:,0].astype(np.float64), img[:,:,1].astype(np.float64), img[:,:,2].astype(np.float64)
        luminance = 0.114 * b + 0.587 * g + 0.299 * r

        l_min = luminance.min()
        l_max = luminance.max()

        if l_max - l_min < 1:
            print("Warning: Image has nearly constant brightness")
            return img.copy()

        # Stretch luminance to [0, 255]
        scale = 255.0 / (l_max - l_min)
        offset = -l_min * scale

        # Apply same scaling to all channels
        result = np.clip(img.astype(np.float64) * scale + offset, 0, 255).astype(np.uint8)

        print(f"Color image contrast stretch:")
        print(f"  Luminance range: [{l_min:.1f}, {l_max:.1f}] -> [0, 255]")
        print(f"  Scale factor: {scale:.3f}")
    else:
        # Grayscale
        img_min = img.min()
        img_max = img.max()

        if img_max - img_min < 1:
            print("Warning: Image has nearly constant brightness")
            return img.copy()

        result = ((img.astype(np.float64) - img_min) * 255.0 / (img_max - img_min))
        result = result.astype(np.uint8)

        print(f"Grayscale contrast stretch:")
        print(f"  Range: [{img_min}, {img_max}] -> [0, 255]")

    # Report histogram statistics
    print(f"  Original std: {np.std(img):.1f}")
    print(f"  Result std:   {np.std(result):.1f}")

    return result


# =============================================================================
# Exercise 2: Dominant Color Extraction (K-means)
# =============================================================================

def exercise_2_dominant_colors(img, k=3, max_iter=20):
    """
    Extract the top-k dominant colors from a BGR image using K-means clustering.

    Parameters:
        img: (H, W, 3) BGR image
        k: number of color clusters
        max_iter: maximum iterations for K-means

    Returns:
        list of (color_bgr, percentage) tuples sorted by frequency
    """
    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3).astype(np.float64)
    n_pixels = len(pixels)

    # K-means clustering
    # Initialize centroids randomly from data points
    np.random.seed(42)
    indices = np.random.choice(n_pixels, k, replace=False)
    centroids = pixels[indices].copy()

    labels = np.zeros(n_pixels, dtype=np.int32)

    for iteration in range(max_iter):
        # Assignment step: assign each pixel to nearest centroid
        old_labels = labels.copy()

        for i in range(n_pixels):
            dists = np.sum((centroids - pixels[i])**2, axis=1)
            labels[i] = np.argmin(dists)

        # Update step: recalculate centroids
        for c in range(k):
            cluster_pixels = pixels[labels == c]
            if len(cluster_pixels) > 0:
                centroids[c] = np.mean(cluster_pixels, axis=0)

        # Check convergence
        if np.array_equal(labels, old_labels):
            print(f"  K-means converged at iteration {iteration + 1}")
            break

    # Count pixels per cluster and sort by frequency
    colors = []
    for c in range(k):
        count = np.sum(labels == c)
        pct = 100.0 * count / n_pixels
        bgr = centroids[c].astype(int)
        colors.append((bgr, pct))

    colors.sort(key=lambda x: x[1], reverse=True)

    # Display results
    print(f"\nDominant Colors (k={k}):")
    print(f"{'Rank':>5} | {'BGR':>20} | {'Percentage':>10}")
    print("-" * 42)

    for rank, (bgr, pct) in enumerate(colors, 1):
        print(f"{rank:>5} | ({bgr[0]:>3}, {bgr[1]:>3}, {bgr[2]:>3})     | {pct:>9.1f}%")

    # Create color palette visualization
    palette = np.zeros((50, 300, 3), dtype=np.uint8)
    x_pos = 0
    for bgr, pct in colors:
        width = int(pct * 3)
        palette[:, x_pos:x_pos + width] = bgr
        x_pos += width

    return colors


# =============================================================================
# Exercise 3: Illumination Normalization
# =============================================================================

def exercise_3_illumination_normalization(img):
    """
    Normalize a document image with uneven illumination.
    Pipeline: background estimation -> division -> CLAHE.

    Parameters:
        img: grayscale document image (H, W) uint8

    Returns:
        normalized image
    """
    h, w = img.shape
    img_f = img.astype(np.float64)

    # Step 1: Estimate background using large Gaussian blur
    blur_size = 51  # Must be odd
    sigma = blur_size / 3.0

    # Create Gaussian kernel
    ax = np.arange(-blur_size//2 + 1, blur_size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    # Apply convolution (simplified for small images)
    pad = blur_size // 2
    padded = np.pad(img_f, pad, mode='reflect')
    background = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            background[i, j] = np.sum(padded[i:i+blur_size, j:j+blur_size] * kernel)

    print(f"Step 1 - Background estimation:")
    print(f"  Background range: [{background.min():.1f}, {background.max():.1f}]")

    # Step 2: Divide original by background to remove illumination gradient
    # result = (original / background) * 255
    normalized = np.where(background > 1,
                          img_f / background * 200,
                          img_f)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    print(f"Step 2 - Normalization:")
    print(f"  Normalized range: [{normalized.min()}, {normalized.max()}]")

    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Simplified CLAHE: tile-based histogram equalization with clip limit
    tile_h, tile_w = 8, 8
    clip_limit = 4.0

    cell_h = h // tile_h
    cell_w = w // tile_w

    enhanced = normalized.copy()

    for tr in range(tile_h):
        for tc in range(tile_w):
            y0 = tr * cell_h
            y1 = min((tr + 1) * cell_h, h)
            x0 = tc * cell_w
            x1 = min((tc + 1) * cell_w, w)

            tile = normalized[y0:y1, x0:x1]

            # Compute histogram
            hist = np.zeros(256, dtype=np.float64)
            for val in tile.ravel():
                hist[val] += 1

            # Clip histogram
            excess = 0
            clip_val = clip_limit * tile.size / 256
            for i in range(256):
                if hist[i] > clip_val:
                    excess += hist[i] - clip_val
                    hist[i] = clip_val

            # Redistribute excess equally
            hist += excess / 256

            # Compute CDF
            cdf = np.cumsum(hist)
            cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
            cdf_range = cdf[-1] - cdf_min

            if cdf_range > 0:
                lut = ((cdf - cdf_min) / cdf_range * 255).astype(np.uint8)
                for i in range(y0, y1):
                    for j in range(x0, x1):
                        enhanced[i, j] = lut[normalized[i, j]]

    print(f"Step 3 - CLAHE enhancement:")
    print(f"  Enhanced range: [{enhanced.min()}, {enhanced.max()}]")
    print(f"  Std improvement: {np.std(img):.1f} -> {np.std(enhanced):.1f}")

    return enhanced


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Automatic Contrast Adjustment")
    # Create low-contrast test image
    low_contrast = np.random.randint(80, 150, (100, 100), dtype=np.uint8)
    exercise_1_auto_contrast(low_contrast)

    # Color version
    low_contrast_color = np.random.randint(80, 150, (100, 100, 3), dtype=np.uint8)
    exercise_1_auto_contrast(low_contrast_color)

    print("\n>>> Exercise 2: Dominant Color Extraction")
    # Create image with distinct color regions
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    color_img[:50, :50] = [200, 50, 30]     # Blue-ish
    color_img[:50, 50:] = [30, 200, 50]     # Green-ish
    color_img[50:, :] = [50, 50, 200]       # Red-ish
    exercise_2_dominant_colors(color_img, k=3)

    print("\n>>> Exercise 3: Illumination Normalization")
    # Create document with uneven lighting
    doc = np.ones((80, 100), dtype=np.uint8) * 200  # White paper
    # Add text lines
    for row in [15, 30, 45, 60]:
        doc[row:row+3, 10:90] = 40
    # Add uneven lighting (darker on right side)
    for j in range(100):
        doc[:, j] = np.clip(doc[:, j].astype(int) - j, 0, 255).astype(np.uint8)
    exercise_3_illumination_normalization(doc)

    print("\nAll exercises completed successfully.")
