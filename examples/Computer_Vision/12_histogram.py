"""
12. Histogram Analysis
- calcHist (histogram calculation)
- equalizeHist (histogram equalization)
- CLAHE (adaptive histogram equalization)
- Histogram back projection
"""

import cv2
import numpy as np


def create_low_contrast_image():
    """Create a low contrast image"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # Use only a narrow range of brightness values (100-150)
    img[:] = 120

    # Draw shapes
    cv2.rectangle(img, (50, 50), (150, 150), 140, -1)
    cv2.circle(img, (300, 150), 60, 130, -1)
    cv2.putText(img, 'LOW', (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 150, 2)

    return img


def create_color_image():
    """Color test image"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Various color regions
    img[0:150, 0:200] = [200, 50, 50]      # Blue
    img[0:150, 200:400] = [50, 200, 50]    # Green
    img[150:300, 0:200] = [50, 50, 200]    # Red
    img[150:300, 200:400] = [200, 200, 50] # Cyan

    return img


def calc_histogram_demo():
    """Histogram calculation demo"""
    print("=" * 50)
    print("Histogram Calculation (calcHist)")
    print("=" * 50)

    img = create_low_contrast_image()

    # Histogram calculation
    # images: Input image list
    # channels: Channel index (grayscale: [0], BGR: [0], [1], [2])
    # mask: Mask (None = entire image)
    # histSize: Number of bins (typically 256)
    # ranges: Value range

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    print(f"Histogram shape: {hist.shape}")
    print(f"Total pixels: {hist.sum()}")
    print(f"Maximum frequency value: {hist.max():.0f}")
    print(f"Maximum frequency position: {hist.argmax()}")

    # Histogram visualization (text)
    print("\nHistogram distribution (summary):")
    for i in range(0, 256, 32):
        count = hist[i:i+32].sum()
        bar = '#' * int(count / 1000)
        print(f"  {i:3d}-{i+31:3d}: {bar} ({count:.0f})")

    cv2.imwrite('histogram_input.jpg', img)

    return hist


def histogram_color_demo():
    """Color histogram demo"""
    print("\n" + "=" * 50)
    print("Color Histogram")
    print("=" * 50)

    img = create_color_image()

    # BGR channel histograms
    colors = ('b', 'g', 'r')

    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        peak = hist.argmax()
        print(f"{color.upper()} channel: Peak position={peak}, Value={hist[peak][0]:.0f}")

    cv2.imwrite('histogram_color.jpg', img)


def equalize_histogram_demo():
    """Histogram equalization demo"""
    print("\n" + "=" * 50)
    print("Histogram Equalization (equalizeHist)")
    print("=" * 50)

    img = create_low_contrast_image()

    # Histogram equalization
    equalized = cv2.equalizeHist(img)

    # Before/after statistics comparison
    print("Before equalization:")
    print(f"  Min={img.min()}, Max={img.max()}")
    print(f"  Mean={img.mean():.1f}, Std={img.std():.1f}")

    print("\nAfter equalization:")
    print(f"  Min={equalized.min()}, Max={equalized.max()}")
    print(f"  Mean={equalized.mean():.1f}, Std={equalized.std():.1f}")

    print("\nEqualization effects:")
    print("  - Enhanced contrast")
    print("  - Histogram uniformly distributed")
    print("  - Applied uniformly to entire image")

    cv2.imwrite('equalize_before.jpg', img)
    cv2.imwrite('equalize_after.jpg', equalized)


def clahe_demo():
    """CLAHE demo"""
    print("\n" + "=" * 50)
    print("CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    print("=" * 50)

    img = create_low_contrast_image()

    # Standard equalization
    equalized = cv2.equalizeHist(img)

    # CLAHE
    # clipLimit: Contrast limit (higher = stronger contrast)
    # tileGridSize: Tile size

    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe3 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    result1 = clahe1.apply(img)
    result2 = clahe2.apply(img)
    result3 = clahe3.apply(img)

    print("CLAHE vs Standard Equalization:")
    print("  - CLAHE: Applied locally (tile by tile)")
    print("  - Prevents noise amplification (clipLimit)")
    print("  - Effective for uneven lighting")

    print("\nCLAHE parameters:")
    print("  clipLimit: Contrast limit (2.0~4.0 recommended)")
    print("  tileGridSize: Tile size (8x8 recommended)")

    cv2.imwrite('clahe_equalized.jpg', equalized)
    cv2.imwrite('clahe_2_8.jpg', result1)
    cv2.imwrite('clahe_4_8.jpg', result2)
    cv2.imwrite('clahe_2_16.jpg', result3)


def clahe_color_demo():
    """Color image CLAHE demo"""
    print("\n" + "=" * 50)
    print("Color Image CLAHE")
    print("=" * 50)

    img = create_color_image()

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge back
    lab_clahe = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    print("Color image CLAHE method:")
    print("  1. Convert BGR -> LAB")
    print("  2. Apply CLAHE to L channel")
    print("  3. Convert LAB -> BGR")
    print("  (Color (a,b) preserved, only lightness (L) adjusted)")

    cv2.imwrite('clahe_color_before.jpg', img)
    cv2.imwrite('clahe_color_after.jpg', result)


def histogram_comparison_demo():
    """Histogram comparison demo"""
    print("\n" + "=" * 50)
    print("Histogram Comparison (compareHist)")
    print("=" * 50)

    # Create images to compare
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img1[:] = 100
    cv2.rectangle(img1, (20, 20), (80, 80), 150, -1)

    img2 = img1.copy()  # Identical

    img3 = np.zeros((100, 100), dtype=np.uint8)
    img3[:] = 50
    cv2.rectangle(img3, (20, 20), (80, 80), 200, -1)

    # Calculate histograms
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])

    # Normalize
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist3, hist3, 0, 1, cv2.NORM_MINMAX)

    # Comparison methods
    methods = [
        (cv2.HISTCMP_CORREL, 'Correlation'),
        (cv2.HISTCMP_CHISQR, 'Chi-Square'),
        (cv2.HISTCMP_INTERSECT, 'Intersection'),
        (cv2.HISTCMP_BHATTACHARYYA, 'Bhattacharyya'),
    ]

    print("img1 vs img2 (identical), img1 vs img3 (different):\n")

    for method, name in methods:
        score12 = cv2.compareHist(hist1, hist2, method)
        score13 = cv2.compareHist(hist1, hist3, method)
        print(f"  {name:15}: identical={score12:.4f}, different={score13:.4f}")

    print("\nComparison method interpretation:")
    print("  Correlation: Closer to 1 = more similar")
    print("  Chi-Square: Closer to 0 = more similar")
    print("  Intersection: Higher = more similar")
    print("  Bhattacharyya: Closer to 0 = more similar")


def back_projection_demo():
    """Histogram back projection demo"""
    print("\n" + "=" * 50)
    print("Histogram Back Projection")
    print("=" * 50)

    # Target image (various colors)
    target = np.zeros((300, 400, 3), dtype=np.uint8)
    target[:] = [100, 100, 100]  # Gray background

    # Red objects
    cv2.circle(target, (100, 100), 40, (50, 50, 200), -1)
    cv2.circle(target, (300, 200), 50, (30, 30, 180), -1)
    cv2.rectangle(target, (150, 200), (220, 280), (40, 40, 210), -1)

    # Blue object
    cv2.circle(target, (350, 100), 30, (200, 50, 50), -1)

    # ROI (red color sample)
    roi = target[60:140, 60:140]  # Red circle region

    # HSV conversion
    hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calculate ROI histogram (Hue and Saturation only)
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Back projection
    back_proj = cv2.calcBackProject([hsv_target], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    # Improve result (morphological operation)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(back_proj, -1, kernel, back_proj)
    _, thresh = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)

    print("Back projection process:")
    print("  1. Calculate histogram of ROI")
    print("  2. Compute how similar each pixel in target is to ROI")
    print("  3. Similar color regions appear bright")
    print("  4. Used for object tracking (MeanShift, CamShift)")

    cv2.imwrite('backproj_target.jpg', target)
    cv2.imwrite('backproj_roi.jpg', roi)
    cv2.imwrite('backproj_result.jpg', back_proj)
    cv2.imwrite('backproj_thresh.jpg', thresh)


def main():
    """Main function"""
    # Histogram calculation
    calc_histogram_demo()

    # Color histogram
    histogram_color_demo()

    # Histogram equalization
    equalize_histogram_demo()

    # CLAHE
    clahe_demo()

    # Color CLAHE
    clahe_color_demo()

    # Histogram comparison
    histogram_comparison_demo()

    # Back projection
    back_projection_demo()

    print("\nHistogram analysis demo complete!")


if __name__ == '__main__':
    main()
