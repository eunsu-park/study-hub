"""
08. Edge Detection
- Sobel, Scharr filters
- Laplacian
- Canny edge detection
"""

import cv2
import numpy as np


def create_test_image():
    """Create a test image"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 200

    # Rectangle
    cv2.rectangle(img, (50, 50), (150, 150), 50, -1)

    # Circle
    cv2.circle(img, (300, 150), 60, 80, -1)

    # Triangle
    pts = np.array([[200, 250], [150, 290], [250, 290]], np.int32)
    cv2.fillPoly(img, [pts], 100)

    # Text
    cv2.putText(img, 'EDGE', (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 30, 2)

    return img


def sobel_demo():
    """Sobel filter demo"""
    print("=" * 50)
    print("Sobel Filter")
    print("=" * 50)

    img = create_test_image()

    # Sobel filter (x direction)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    # Sobel filter (y direction)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Absolute value and 8-bit conversion
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)

    # Combine x and y
    sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

    # Magnitude calculation (exact method)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))

    print("Sobel filter:")
    print("  - First derivative-based edge detection")
    print("  - x direction: Detects vertical edges")
    print("  - y direction: Detects horizontal edges")
    print("  - ksize: Kernel size (3, 5, 7)")

    cv2.imwrite('edge_original.jpg', img)
    cv2.imwrite('sobel_x.jpg', sobel_x_abs)
    cv2.imwrite('sobel_y.jpg', sobel_y_abs)
    cv2.imwrite('sobel_combined.jpg', sobel_combined)
    cv2.imwrite('sobel_magnitude.jpg', sobel_magnitude)


def scharr_demo():
    """Scharr filter demo"""
    print("\n" + "=" * 50)
    print("Scharr Filter")
    print("=" * 50)

    img = create_test_image()

    # Scharr filter (more accurate than Sobel)
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)

    scharr_x_abs = cv2.convertScaleAbs(scharr_x)
    scharr_y_abs = cv2.convertScaleAbs(scharr_y)

    scharr_combined = cv2.addWeighted(scharr_x_abs, 0.5, scharr_y_abs, 0.5, 0)

    print("Scharr filter:")
    print("  - Improved version of Sobel")
    print("  - Supports only 3x3 kernel")
    print("  - More accurate gradient computation")
    print("  - Equivalent to Sobel(ksize=-1)")

    cv2.imwrite('scharr_x.jpg', scharr_x_abs)
    cv2.imwrite('scharr_y.jpg', scharr_y_abs)
    cv2.imwrite('scharr_combined.jpg', scharr_combined)


def laplacian_demo():
    """Laplacian filter demo"""
    print("\n" + "=" * 50)
    print("Laplacian Filter")
    print("=" * 50)

    img = create_test_image()

    # Apply blur since Laplacian is sensitive to noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Laplacian filter
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # Varying kernel size
    lap_k1 = cv2.Laplacian(blurred, cv2.CV_64F, ksize=1)
    lap_k3 = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    lap_k5 = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)

    print("Laplacian filter:")
    print("  - Second derivative-based edge detection")
    print("  - Detects edges in all directions at once")
    print("  - Sensitive to noise -> blur required")

    cv2.imwrite('laplacian.jpg', laplacian_abs)
    cv2.imwrite('laplacian_k1.jpg', cv2.convertScaleAbs(lap_k1))
    cv2.imwrite('laplacian_k3.jpg', cv2.convertScaleAbs(lap_k3))
    cv2.imwrite('laplacian_k5.jpg', cv2.convertScaleAbs(lap_k5))


def canny_demo():
    """Canny edge detection demo"""
    print("\n" + "=" * 50)
    print("Canny Edge Detection")
    print("=" * 50)

    img = create_test_image()

    # Canny edge detection
    # threshold1: Lower threshold
    # threshold2: Upper threshold
    canny_50_150 = cv2.Canny(img, 50, 150)
    canny_100_200 = cv2.Canny(img, 100, 200)
    canny_30_100 = cv2.Canny(img, 30, 100)

    print("Canny edge detection steps:")
    print("  1. Noise removal with Gaussian filter")
    print("  2. Gradient computation with Sobel")
    print("  3. Non-Maximum Suppression")
    print("  4. Edge determination with double threshold")
    print("     - Strong edge: > threshold2")
    print("     - Weak edge: threshold1 ~ threshold2")
    print("     - Non-edge: < threshold1")
    print("  5. Hysteresis edge tracking")

    cv2.imwrite('canny_50_150.jpg', canny_50_150)
    cv2.imwrite('canny_100_200.jpg', canny_100_200)
    cv2.imwrite('canny_30_100.jpg', canny_30_100)


def canny_with_blur():
    """Canny with blur preprocessing"""
    print("\n" + "=" * 50)
    print("Blur + Canny")
    print("=" * 50)

    img = create_test_image()

    # Add noise
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Without blur
    canny_no_blur = cv2.Canny(noisy, 50, 150)

    # After Gaussian blur
    blurred = cv2.GaussianBlur(noisy, (5, 5), 0)
    canny_with_blur = cv2.Canny(blurred, 50, 150)

    # Adjusting apertureSize within Canny
    canny_aperture3 = cv2.Canny(noisy, 50, 150, apertureSize=3)
    canny_aperture5 = cv2.Canny(noisy, 50, 150, apertureSize=5)

    print("Noise removal:")
    print("  - Blur preprocessing recommended")
    print("  - apertureSize adjustable (3, 5, 7)")

    cv2.imwrite('canny_noisy.jpg', noisy)
    cv2.imwrite('canny_no_blur.jpg', canny_no_blur)
    cv2.imwrite('canny_with_blur.jpg', canny_with_blur)


def auto_canny_threshold():
    """Automatic Canny threshold"""
    print("\n" + "=" * 50)
    print("Automatic Canny Threshold")
    print("=" * 50)

    img = create_test_image()

    # Median-based automatic threshold
    median = np.median(img)
    sigma = 0.33

    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    auto_canny = cv2.Canny(img, lower, upper)

    print(f"Image median: {median}")
    print(f"Automatic threshold: lower={lower}, upper={upper}")
    print(f"Formula: lower = (1-sigma)*median, upper = (1+sigma)*median")

    cv2.imwrite('canny_auto.jpg', auto_canny)

    return lower, upper


def log_edge_detection():
    """LoG (Laplacian of Gaussian) edge detection"""
    print("\n" + "=" * 50)
    print("LoG Edge Detection")
    print("=" * 50)

    img = create_test_image()

    # LoG = Gaussian blur + Laplacian
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
    log = cv2.Laplacian(blurred, cv2.CV_64F)

    # Find zero-crossings (simple method)
    log_abs = cv2.convertScaleAbs(log)

    print("LoG (Laplacian of Gaussian):")
    print("  - Noise removal with Gaussian")
    print("  - Second derivative with Laplacian")
    print("  - Zero-crossings are the edges")

    cv2.imwrite('log_edge.jpg', log_abs)


def compare_edge_methods():
    """Edge detection method comparison"""
    print("\n" + "=" * 50)
    print("Edge Detection Method Comparison")
    print("=" * 50)

    img = create_test_image()

    # Apply each method
    sobel_x = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    laplacian = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))

    canny = cv2.Canny(img, 50, 150)

    print("""
    | Method | Features | Use Case |
    |--------|----------|----------|
    | Sobel | 1st derivative, directional | When gradient direction needed |
    | Scharr | Improved Sobel | More accurate gradient |
    | Laplacian | 2nd derivative | All directions at once |
    | Canny | Multi-stage processing | Most commonly used |
    """)

    # Create comparison image
    compare = np.hstack([
        sobel,
        laplacian,
        canny
    ])
    cv2.imwrite('edge_compare.jpg', compare)


def practical_example():
    """Practical example: Contour extraction"""
    print("\n" + "=" * 50)
    print("Practical Example: Contour Extraction")
    print("=" * 50)

    # Create color image
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 200), -1)
    cv2.circle(img, (300, 150), 60, (200, 0, 0), -1)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny edge
    edges = cv2.Canny(gray, 50, 150)

    # Draw contours on original
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    cv2.imwrite('practical_input.jpg', img)
    cv2.imwrite('practical_edges.jpg', edges)
    cv2.imwrite('practical_contours.jpg', result)
    print("Contour extraction images saved successfully")


def main():
    """Main function"""
    # Sobel
    sobel_demo()

    # Scharr
    scharr_demo()

    # Laplacian
    laplacian_demo()

    # Canny
    canny_demo()

    # Blur + Canny
    canny_with_blur()

    # Automatic threshold
    auto_canny_threshold()

    # LoG
    log_edge_detection()

    # Method comparison
    compare_edge_methods()

    # Practical example
    practical_example()

    print("\nEdge detection demo complete!")


if __name__ == '__main__':
    main()
