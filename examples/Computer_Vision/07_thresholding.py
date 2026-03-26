"""
07. Binarization and Thresholding
- Basic thresholding (threshold)
- Otsu's method
- Adaptive thresholding (adaptiveThreshold)
- Multi-level thresholding
"""

import cv2
import numpy as np


def create_gradient_image():
    """Create a gradient test image"""
    img = np.zeros((200, 400), dtype=np.uint8)

    # Horizontal gradient
    for j in range(400):
        img[:, j] = int(j * 255 / 400)

    return img


def create_text_image():
    """Test image with text"""
    img = np.zeros((200, 400), dtype=np.uint8)
    img[:] = 200  # Bright background

    cv2.putText(img, 'Threshold', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 50, 3)
    cv2.putText(img, 'OpenCV', (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 80, 2)

    return img


def create_uneven_lighting():
    """Uneven lighting image"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # Uneven lighting background
    for i in range(300):
        for j in range(400):
            img[i, j] = int(150 + 80 * np.sin(i / 50) * np.cos(j / 50))

    # Add text
    cv2.putText(img, 'UNEVEN', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 30, 3)
    cv2.putText(img, 'LIGHTING', (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 50, 2)

    return img


def basic_threshold_demo():
    """Basic thresholding demo"""
    print("=" * 50)
    print("Basic Thresholding (threshold)")
    print("=" * 50)

    img = create_gradient_image()

    # Binarize with threshold value 127
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    print("THRESH_BINARY:")
    print("  - pixel > threshold -> max value (255)")
    print("  - pixel <= threshold -> 0")

    print("\nTHRESH_BINARY_INV:")
    print("  - pixel > threshold -> 0")
    print("  - pixel <= threshold -> max value (255)")

    cv2.imwrite('thresh_original.jpg', img)
    cv2.imwrite('thresh_binary.jpg', binary)
    cv2.imwrite('thresh_binary_inv.jpg', binary_inv)


def threshold_types_demo():
    """Various threshold types"""
    print("\n" + "=" * 50)
    print("Threshold Types")
    print("=" * 50)

    img = create_gradient_image()
    thresh_val = 127

    # Various threshold types
    _, binary = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    _, binary_inv = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY_INV)
    _, trunc = cv2.threshold(img, thresh_val, 255, cv2.THRESH_TRUNC)
    _, tozero = cv2.threshold(img, thresh_val, 255, cv2.THRESH_TOZERO)
    _, tozero_inv = cv2.threshold(img, thresh_val, 255, cv2.THRESH_TOZERO_INV)

    print("Threshold types:")
    print("  BINARY:     pixel > T -> 255, else 0")
    print("  BINARY_INV: pixel > T -> 0, else 255")
    print("  TRUNC:      pixel > T -> T, else original")
    print("  TOZERO:     pixel > T -> original, else 0")
    print("  TOZERO_INV: pixel > T -> 0, else original")

    cv2.imwrite('thresh_trunc.jpg', trunc)
    cv2.imwrite('thresh_tozero.jpg', tozero)
    cv2.imwrite('thresh_tozero_inv.jpg', tozero_inv)


def otsu_demo():
    """Otsu's method demo"""
    print("\n" + "=" * 50)
    print("Otsu's Method")
    print("=" * 50)

    img = create_text_image()

    # Manual thresholding
    _, binary_100 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    _, binary_150 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    # Otsu's method (automatic threshold)
    otsu_val, binary_otsu = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(f"Manual threshold: 100, 150")
    print(f"Otsu automatic threshold: {otsu_val}")

    print("\nOtsu's method properties:")
    print("  - Histogram-based automatic threshold determination")
    print("  - Effective for bimodal distributions")
    print("  - Maximizes inter-class variance")

    cv2.imwrite('otsu_input.jpg', img)
    cv2.imwrite('otsu_100.jpg', binary_100)
    cv2.imwrite('otsu_150.jpg', binary_150)
    cv2.imwrite('otsu_auto.jpg', binary_otsu)


def adaptive_threshold_demo():
    """Adaptive thresholding demo"""
    print("\n" + "=" * 50)
    print("Adaptive Threshold")
    print("=" * 50)

    img = create_uneven_lighting()

    # Global thresholding (fails with uneven lighting)
    _, binary_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Otsu is also limited with uneven lighting
    _, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive thresholding (Mean)
    binary_mean = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size (odd number)
        2    # C (constant)
    )

    # Adaptive thresholding (Gaussian)
    binary_gaussian = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    print("Adaptive thresholding:")
    print("  - Threshold determined by local mean around each pixel")
    print("  - MEAN: Simple average")
    print("  - GAUSSIAN: Gaussian weighted average")
    print("  - Effective for uneven lighting")

    cv2.imwrite('adaptive_input.jpg', img)
    cv2.imwrite('adaptive_global.jpg', binary_global)
    cv2.imwrite('adaptive_otsu.jpg', binary_otsu)
    cv2.imwrite('adaptive_mean.jpg', binary_mean)
    cv2.imwrite('adaptive_gaussian.jpg', binary_gaussian)


def adaptive_params_demo():
    """Adaptive thresholding parameters"""
    print("\n" + "=" * 50)
    print("Adaptive Thresholding Parameters")
    print("=" * 50)

    img = create_uneven_lighting()

    # Varying block size
    sizes = [5, 11, 31, 51]
    for size in sizes:
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            size, 2
        )
        cv2.imwrite(f'adaptive_size_{size}.jpg', binary)

    # Varying C value
    c_values = [0, 2, 5, 10]
    for c in c_values:
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, c
        )
        cv2.imwrite(f'adaptive_c_{c}.jpg', binary)

    print("Parameter effects:")
    print("  - Block size: Larger considers wider area")
    print("  - C value: Constant subtracted from threshold")
    print("    Larger C -> more area becomes foreground")


def triangle_threshold_demo():
    """Triangle thresholding demo"""
    print("\n" + "=" * 50)
    print("Triangle Thresholding")
    print("=" * 50)

    # Image with skewed histogram
    img = np.zeros((200, 400), dtype=np.uint8)
    img[:] = 200
    cv2.rectangle(img, (50, 50), (150, 150), 30, -1)
    cv2.circle(img, (300, 100), 40, 50, -1)

    # Triangle method
    tri_val, binary_tri = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )

    # Compare with Otsu
    otsu_val, binary_otsu = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(f"Triangle automatic threshold: {tri_val}")
    print(f"Otsu automatic threshold: {otsu_val}")

    print("\nTriangle method:")
    print("  - Effective for unimodal distributions")
    print("  - Finds the farthest point from histogram peak")

    cv2.imwrite('triangle_input.jpg', img)
    cv2.imwrite('triangle_result.jpg', binary_tri)


def multi_threshold_demo():
    """Multi-level thresholding"""
    print("\n" + "=" * 50)
    print("Multi-level Thresholding")
    print("=" * 50)

    img = create_gradient_image()

    # Multi-level quantization
    result = np.zeros_like(img)
    thresholds = [50, 100, 150, 200]
    values = [0, 64, 128, 192, 255]

    for i, (low, high, val) in enumerate(
        zip([0] + thresholds, thresholds + [256], values)
    ):
        mask = (img >= low) & (img < high)
        result[mask] = val

    print("Multi-level thresholding:")
    print(f"  Thresholds: {thresholds}")
    print(f"  Result values: {values}")

    cv2.imwrite('multi_thresh.jpg', result)


def practical_document_scan():
    """Practical example: Document scan binarization"""
    print("\n" + "=" * 50)
    print("Practical Example: Document Scan")
    print("=" * 50)

    # Simulate a document image
    img = np.zeros((300, 400), dtype=np.uint8)

    # Uneven background
    for i in range(300):
        for j in range(400):
            img[i, j] = int(200 + 30 * np.sin(i / 100) + 20 * np.cos(j / 100))

    # Text
    cv2.putText(img, 'Document', (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 30, 2)
    cv2.putText(img, 'Scanning', (90, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 50, 2)
    cv2.putText(img, 'Example', (110, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, 40, 2)

    # Reduce noise with Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10
    )

    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite('document_original.jpg', img)
    cv2.imwrite('document_binary.jpg', cleaned)
    print("Document scan images saved successfully")


def main():
    """Main function"""
    # Basic thresholding
    basic_threshold_demo()

    # Threshold types
    threshold_types_demo()

    # Otsu's method
    otsu_demo()

    # Adaptive thresholding
    adaptive_threshold_demo()

    # Adaptive parameters
    adaptive_params_demo()

    # Triangle method
    triangle_threshold_demo()

    # Multi-level thresholding
    multi_threshold_demo()

    # Practical example
    practical_document_scan()

    print("\nBinarization and thresholding demo complete!")


if __name__ == '__main__':
    main()
