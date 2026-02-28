"""
Exercise Solutions for Lesson 07: Thresholding
Computer Vision - Global, Otsu, Adaptive, Hysteresis Thresholding

Topics covered:
- Automatic optimal threshold search (histogram valley)
- Adaptive thresholding parameter tuning
- Business card scanner pipeline
- Color separation tool
- Hysteresis thresholding implementation
"""

import numpy as np


# =============================================================================
# Exercise 1: Automatic Optimal Threshold Search
# =============================================================================

def exercise_1_find_valley_threshold(img):
    """
    Find the valley between two peaks in the histogram and return it as
    the threshold. Compare with Otsu's method.

    Parameters:
        img: grayscale image (H, W) uint8

    Returns:
        (valley_threshold, otsu_threshold)
    """
    # Compute histogram
    hist = np.zeros(256, dtype=np.int64)
    for val in img.ravel():
        hist[val] += 1

    # Smooth histogram to reduce noise (moving average)
    window = 11
    smoothed = np.convolve(hist, np.ones(window) / window, mode='same')

    # Find peaks (local maxima)
    peaks = []
    for i in range(1, 254):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            if smoothed[i] > smoothed.max() * 0.05:  # Ignore tiny peaks
                peaks.append((i, smoothed[i]))

    # Find valley between the two tallest peaks
    valley_thresh = 128  # Default fallback
    if len(peaks) >= 2:
        peaks.sort(key=lambda x: x[1], reverse=True)
        peak1_idx = peaks[0][0]
        peak2_idx = peaks[1][0]
        lo = min(peak1_idx, peak2_idx)
        hi = max(peak1_idx, peak2_idx)

        # Find minimum between the two peaks
        valley_region = smoothed[lo:hi+1]
        valley_thresh = lo + np.argmin(valley_region)

    # Otsu's method implementation
    total = img.size
    current_max = 0
    otsu_thresh = 0

    for t in range(256):
        w0 = np.sum(hist[:t+1])
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue

        sum0 = np.sum(np.arange(t+1) * hist[:t+1])
        sum1 = np.sum(np.arange(t+1, 256) * hist[t+1:])
        mean0 = sum0 / w0
        mean1 = sum1 / w1

        # Between-class variance
        variance = w0 * w1 * (mean0 - mean1) ** 2
        if variance > current_max:
            current_max = variance
            otsu_thresh = t

    print(f"Valley threshold: {valley_thresh}")
    print(f"Otsu threshold:   {otsu_thresh}")
    print(f"Difference:       {abs(valley_thresh - otsu_thresh)}")

    # Apply both and compare
    valley_binary = np.where(img > valley_thresh, 255, 0).astype(np.uint8)
    otsu_binary = np.where(img > otsu_thresh, 255, 0).astype(np.uint8)

    agreement = np.sum(valley_binary == otsu_binary) / img.size * 100
    print(f"Pixel agreement:  {agreement:.1f}%")

    return valley_thresh, otsu_thresh


# =============================================================================
# Exercise 2: Adaptive Thresholding Parameter Tuning
# =============================================================================

def exercise_2_adaptive_threshold_tuning(img):
    """
    Test different blockSize and C values for adaptive thresholding
    and compare results.

    Parameters:
        img: grayscale image (H, W) uint8

    Returns:
        dict of results for different parameter combinations
    """
    def adaptive_threshold_mean(img, block_size, c_val):
        """Adaptive thresholding using mean of local neighborhood."""
        h, w = img.shape
        result = np.zeros_like(img)
        pad = block_size // 2
        padded = np.pad(img.astype(np.float64), pad, mode='reflect')

        for i in range(h):
            for j in range(w):
                region = padded[i:i + block_size, j:j + block_size]
                local_mean = np.mean(region)
                result[i, j] = 255 if img[i, j] > local_mean - c_val else 0

        return result

    block_sizes = [7, 11, 15, 21]
    c_values = [2, 5, 10, 15]

    results = {}
    print(f"{'blockSize':>10} | {'C':>5} | {'FG%':>8} | {'Transitions':>12}")
    print("-" * 45)

    for bs in block_sizes:
        for c in c_values:
            binary = adaptive_threshold_mean(img, bs, c)
            fg_pct = np.sum(binary > 0) / img.size * 100

            # Count transitions (edges) as quality metric
            transitions = (np.sum(np.abs(np.diff(binary.astype(int), axis=0)) > 0) +
                          np.sum(np.abs(np.diff(binary.astype(int), axis=1)) > 0))

            results[(bs, c)] = binary
            print(f"{bs:>10} | {c:>5} | {fg_pct:>7.1f}% | {transitions:>12}")

    return results


# =============================================================================
# Exercise 3: Business Card Scanner
# =============================================================================

def exercise_3_business_card_scanner(img):
    """
    Preprocess a business card image:
    1. Shadow/uneven lighting correction
    2. Binarization
    3. Noise removal (morphological operations)

    Parameters:
        img: grayscale card image (H, W) uint8

    Returns:
        cleaned binary image
    """
    h, w = img.shape
    print(f"Input: {w}x{h}")

    # Step 1: Shadow correction using local background estimation
    # Estimate background with large-window mean filter
    bg_size = 51
    pad = bg_size // 2
    padded = np.pad(img.astype(np.float64), pad, mode='reflect')
    background = np.zeros_like(img, dtype=np.float64)

    # Use strided approach for efficiency
    for i in range(h):
        for j in range(w):
            region = padded[i:i + bg_size, j:j + bg_size]
            background[i, j] = np.mean(region)

    # Normalize: original / background * 255
    corrected = np.where(background > 0,
                         img.astype(np.float64) / background * 200,
                         img.astype(np.float64))
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    print(f"Step 1 (Lighting correction): range [{corrected.min()}, {corrected.max()}]")

    # Step 2: Otsu binarization
    hist = np.zeros(256, dtype=np.int64)
    for val in corrected.ravel():
        hist[val] += 1

    total = corrected.size
    best_thresh = 0
    best_var = 0
    for t in range(256):
        w0 = np.sum(hist[:t+1])
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mean0 = np.sum(np.arange(t+1) * hist[:t+1]) / w0
        mean1 = np.sum(np.arange(t+1, 256) * hist[t+1:]) / w1
        var = w0 * w1 * (mean0 - mean1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = t

    binary = np.where(corrected > best_thresh, 255, 0).astype(np.uint8)
    print(f"Step 2 (Otsu threshold={best_thresh}): FG={np.sum(binary > 0)}")

    # Step 3: Morphological noise removal (close then open)
    se = np.ones((3, 3), dtype=np.uint8)

    # Close small holes in text
    closed = binary.copy()
    # Simple dilation then erosion
    pad = 1
    padded = np.pad(closed, pad, mode='constant', constant_values=0)
    dilated = np.zeros_like(closed)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            dilated[i, j] = 255 if np.any(region[se == 1] == 255) else 0

    padded = np.pad(dilated, pad, mode='constant', constant_values=0)
    cleaned = np.zeros_like(dilated)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            cleaned[i, j] = 255 if np.all(region[se == 1] == 255) else 0

    # Open to remove small noise
    padded = np.pad(cleaned, pad, mode='constant', constant_values=0)
    eroded = np.zeros_like(cleaned)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            eroded[i, j] = 255 if np.all(region[se == 1] == 255) else 0

    padded = np.pad(eroded, pad, mode='constant', constant_values=0)
    final = np.zeros_like(eroded)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            final[i, j] = 255 if np.any(region[se == 1] == 255) else 0

    print(f"Step 3 (Morph cleanup): FG={np.sum(final > 0)}")

    return final


# =============================================================================
# Exercise 4: Color Separation Tool
# =============================================================================

def exercise_4_color_separation(img_bgr):
    """
    Extract specific color regions from a BGR image and calculate
    the area percentage of each color.

    Parameters:
        img_bgr: (H, W, 3) BGR image

    Returns:
        dict of color masks and percentages
    """
    h, w = img_bgr.shape[:2]
    total = h * w

    # Convert to simplified HSV
    bgr_f = img_bgr.astype(np.float64) / 255.0
    r, g, b = bgr_f[:, :, 2], bgr_f[:, :, 1], bgr_f[:, :, 0]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    hue = np.zeros_like(diff)
    mask_r = (cmax == r) & (diff > 0)
    mask_g = (cmax == g) & (diff > 0)
    mask_b = (cmax == b) & (diff > 0)

    hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)
    hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2)
    hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4)

    sat = np.where(cmax > 0, diff / cmax, 0)
    val = cmax

    # Scale: H: 0-180, S: 0-255, V: 0-255
    h_s = (hue / 2).astype(np.uint8)
    s_s = (sat * 255).astype(np.uint8)
    v_s = (val * 255).astype(np.uint8)

    # Define color ranges in HSV (OpenCV scale)
    color_ranges = {
        'Red':    ((0, 50, 50), (10, 255, 255)),   # Low red
        'Red2':   ((170, 50, 50), (180, 255, 255)), # High red (wrap-around)
        'Orange': ((10, 50, 50), (25, 255, 255)),
        'Yellow': ((25, 50, 50), (35, 255, 255)),
        'Green':  ((35, 50, 50), (85, 255, 255)),
        'Blue':   ((85, 50, 50), (130, 255, 255)),
        'Purple': ((130, 50, 50), (170, 255, 255)),
    }

    results = {}
    print(f"{'Color':>10} | {'Pixels':>8} | {'Percentage':>10}")
    print("-" * 35)

    for color_name, ((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)) in color_ranges.items():
        mask = ((h_s >= h_lo) & (h_s <= h_hi) &
                (s_s >= s_lo) & (s_s <= s_hi) &
                (v_s >= v_lo) & (v_s <= v_hi))

        count = np.sum(mask)
        pct = 100 * count / total
        results[color_name] = {'mask': mask.astype(np.uint8) * 255, 'percentage': pct}
        print(f"{color_name:>10} | {count:>8} | {pct:>9.1f}%")

    # Combine Red and Red2
    if 'Red' in results and 'Red2' in results:
        combined = results['Red']['percentage'] + results['Red2']['percentage']
        print(f"{'Red total':>10} | {'':>8} | {combined:>9.1f}%")

    return results


# =============================================================================
# Exercise 5: Hysteresis Thresholding
# =============================================================================

def exercise_5_hysteresis_threshold(img, low_thresh, high_thresh):
    """
    Implement hysteresis thresholding (as used in Canny edge detection):
    - Above high threshold: Definite edge
    - Below low threshold: Definitely not an edge
    - Between: Edge only if connected to a definite edge

    Parameters:
        img: grayscale image (H, W) uint8
        low_thresh: lower threshold
        high_thresh: upper threshold

    Returns:
        binary edge map
    """
    h, w = img.shape

    # Classify pixels into three categories
    strong = (img >= high_thresh).astype(np.uint8) * 255
    weak = ((img >= low_thresh) & (img < high_thresh)).astype(np.uint8) * 255

    print(f"Strong edges: {np.sum(strong > 0)} pixels")
    print(f"Weak edges:   {np.sum(weak > 0)} pixels")
    print(f"Suppressed:   {np.sum(img < low_thresh)} pixels")

    # Connect weak edges to strong edges using flood fill
    result = strong.copy()
    changed = True

    while changed:
        changed = False
        padded = np.pad(result, 1, mode='constant', constant_values=0)

        for i in range(h):
            for j in range(w):
                if weak[i, j] > 0 and result[i, j] == 0:
                    # Check 8-connected neighbors for strong edges
                    neighbors = padded[i:i+3, j:j+3]
                    if np.any(neighbors == 255):
                        result[i, j] = 255
                        changed = True

    final_edges = np.sum(result > 0)
    print(f"Final edges:  {final_edges} pixels "
          f"(strong + connected weak)")

    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Automatic Optimal Threshold Search")
    # Create bimodal image (two populations of pixel values)
    bimodal = np.concatenate([
        np.random.normal(70, 15, 5000).clip(0, 255),
        np.random.normal(180, 20, 5000).clip(0, 255)
    ]).astype(np.uint8).reshape(100, 100)
    exercise_1_find_valley_threshold(bimodal)

    print("\n>>> Exercise 2: Adaptive Threshold Parameter Tuning")
    # Image with varying illumination
    adapt_img = np.zeros((40, 40), dtype=np.uint8)
    for i in range(40):
        for j in range(40):
            adapt_img[i, j] = int(50 + 150 * i / 40 + 30 * np.sin(j * 0.5))
    adapt_img = np.clip(adapt_img, 0, 255).astype(np.uint8)
    exercise_2_adaptive_threshold_tuning(adapt_img)

    print("\n>>> Exercise 3: Business Card Scanner")
    # Simulate card with uneven lighting
    card = np.ones((60, 80), dtype=np.uint8) * 200
    # Add "text" lines
    for row in [15, 25, 35, 45]:
        card[row:row+2, 10:70] = 40
    # Add uneven lighting gradient
    for i in range(60):
        card[i] = np.clip(card[i].astype(int) - i, 0, 255).astype(np.uint8)
    exercise_3_business_card_scanner(card)

    print("\n>>> Exercise 4: Color Separation Tool")
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    color_img[:33, :, :] = [0, 0, 200]    # Red region
    color_img[33:66, :, :] = [0, 200, 0]  # Green region
    color_img[66:, :, :] = [200, 0, 0]    # Blue region
    exercise_4_color_separation(color_img)

    print("\n>>> Exercise 5: Hysteresis Thresholding")
    # Create gradient image with some edges
    hyst_img = np.zeros((50, 50), dtype=np.uint8)
    hyst_img[20:30, 10:40] = 200  # Strong edge region
    hyst_img[18:20, 10:40] = 80   # Weak edge (should connect)
    hyst_img[30:32, 10:40] = 80   # Weak edge (should connect)
    hyst_img[5:8, 5:8] = 80       # Isolated weak edge (should be suppressed)
    exercise_5_hysteresis_threshold(hyst_img, low_thresh=50, high_thresh=150)

    print("\nAll exercises completed successfully.")
