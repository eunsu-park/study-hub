"""
05. Image Filtering
- blur, GaussianBlur, medianBlur
- bilateralFilter
- Custom filters (filter2D)
- Sharpening
"""

import cv2
import numpy as np


def create_noisy_image():
    """Create a test image with noise"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # Draw shapes
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)
    cv2.circle(img, (300, 150), 50, (255, 0, 0), -1)
    cv2.putText(img, 'Filter', (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Add Gaussian noise
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, noisy


def create_salt_pepper_noise(img):
    """Add salt-and-pepper noise"""
    noisy = img.copy()
    # Salt (white)
    salt = np.random.random(img.shape[:2]) < 0.02
    noisy[salt] = 255
    # Pepper (black)
    pepper = np.random.random(img.shape[:2]) < 0.02
    noisy[pepper] = 0
    return noisy


def blur_demo():
    """Blur filter demo"""
    print("=" * 50)
    print("Blur Filters")
    print("=" * 50)

    original, noisy = create_noisy_image()

    # Average blur (Box Filter)
    blur_3x3 = cv2.blur(noisy, (3, 3))
    blur_5x5 = cv2.blur(noisy, (5, 5))
    blur_7x7 = cv2.blur(noisy, (7, 7))

    print("Average Blur (Box Filter):")
    print("  - Equal weight for all pixels")
    print("  - Larger kernel = more blur")

    cv2.imwrite('original.jpg', original)
    cv2.imwrite('noisy.jpg', noisy)
    cv2.imwrite('blur_3x3.jpg', blur_3x3)
    cv2.imwrite('blur_5x5.jpg', blur_5x5)
    cv2.imwrite('blur_7x7.jpg', blur_7x7)


def gaussian_blur_demo():
    """Gaussian blur demo"""
    print("\n" + "=" * 50)
    print("Gaussian Blur")
    print("=" * 50)

    _, noisy = create_noisy_image()

    # Gaussian blur
    # GaussianBlur(src, ksize, sigmaX)
    gauss_3x3 = cv2.GaussianBlur(noisy, (3, 3), 0)
    gauss_5x5 = cv2.GaussianBlur(noisy, (5, 5), 0)
    gauss_7x7 = cv2.GaussianBlur(noisy, (7, 7), 0)

    # Difference depending on sigma value
    gauss_s1 = cv2.GaussianBlur(noisy, (5, 5), 1)
    gauss_s3 = cv2.GaussianBlur(noisy, (5, 5), 3)
    gauss_s5 = cv2.GaussianBlur(noisy, (5, 5), 5)

    print("Gaussian Blur:")
    print("  - Higher weight at center, lower at edges")
    print("  - Natural blur effect")
    print("  - Larger sigma = more blur")

    cv2.imwrite('gauss_3x3.jpg', gauss_3x3)
    cv2.imwrite('gauss_5x5.jpg', gauss_5x5)
    cv2.imwrite('gauss_sigma1.jpg', gauss_s1)
    cv2.imwrite('gauss_sigma5.jpg', gauss_s5)


def median_blur_demo():
    """Median blur demo"""
    print("\n" + "=" * 50)
    print("Median Blur")
    print("=" * 50)

    original, _ = create_noisy_image()
    sp_noisy = create_salt_pepper_noise(original)

    # Median blur
    median_3 = cv2.medianBlur(sp_noisy, 3)
    median_5 = cv2.medianBlur(sp_noisy, 5)
    median_7 = cv2.medianBlur(sp_noisy, 7)

    print("Median Blur:")
    print("  - Uses median value within kernel")
    print("  - Effective for salt-and-pepper noise removal")
    print("  - Edge preservation effect")

    cv2.imwrite('salt_pepper_noisy.jpg', sp_noisy)
    cv2.imwrite('median_3.jpg', median_3)
    cv2.imwrite('median_5.jpg', median_5)


def bilateral_filter_demo():
    """Bilateral filter demo"""
    print("\n" + "=" * 50)
    print("Bilateral Filter")
    print("=" * 50)

    _, noisy = create_noisy_image()

    # Bilateral filter
    # bilateralFilter(src, d, sigmaColor, sigmaSpace)
    # d: Filter size (-1 for auto-calculation from sigmaSpace)
    # sigmaColor: Sigma in color space
    # sigmaSpace: Sigma in coordinate space

    bilateral_1 = cv2.bilateralFilter(noisy, 9, 75, 75)
    bilateral_2 = cv2.bilateralFilter(noisy, 9, 150, 150)
    bilateral_3 = cv2.bilateralFilter(noisy, -1, 75, 75)

    print("Bilateral Filter:")
    print("  - Removes noise while preserving edges")
    print("  - sigmaColor: Color difference tolerance range")
    print("  - sigmaSpace: Spatial influence range")
    print("  - Slower than other filters")

    cv2.imwrite('bilateral_75.jpg', bilateral_1)
    cv2.imwrite('bilateral_150.jpg', bilateral_2)


def custom_filter_demo():
    """Custom filter demo"""
    print("\n" + "=" * 50)
    print("Custom Filters (filter2D)")
    print("=" * 50)

    original, _ = create_noisy_image()

    # Average filter kernel
    kernel_avg = np.ones((3, 3), dtype=np.float32) / 9
    avg_filtered = cv2.filter2D(original, -1, kernel_avg)

    # Sharpening kernel
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(original, -1, kernel_sharpen)

    # Strong sharpening
    kernel_sharpen_strong = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    sharpened_strong = cv2.filter2D(original, -1, kernel_sharpen_strong)

    # Embossing kernel
    kernel_emboss = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)
    embossed = cv2.filter2D(original, -1, kernel_emboss)

    print("Custom kernel examples:")
    print(f"  Average filter:\n{kernel_avg}")
    print(f"\n  Sharpening:\n{kernel_sharpen}")
    print(f"\n  Embossing:\n{kernel_emboss}")

    cv2.imwrite('custom_avg.jpg', avg_filtered)
    cv2.imwrite('custom_sharpen.jpg', sharpened)
    cv2.imwrite('custom_sharpen_strong.jpg', sharpened_strong)
    cv2.imwrite('custom_emboss.jpg', embossed)


def unsharp_masking_demo():
    """Unsharp masking demo"""
    print("\n" + "=" * 50)
    print("Unsharp Masking")
    print("=" * 50)

    original, _ = create_noisy_image()

    # Unsharp masking: original + (original - blur) * strength
    blurred = cv2.GaussianBlur(original, (5, 5), 0)

    # Method 1: Direct calculation
    unsharp = cv2.addWeighted(original, 1.5, blurred, -0.5, 0)

    # Method 2: Formula application
    alpha = 1.5  # Sharpening strength
    unsharp2 = cv2.addWeighted(original, 1 + alpha, blurred, -alpha, 0)

    print("Unsharp Masking:")
    print("  result = original + alpha * (original - blur)")
    print("  Larger alpha = stronger sharpening effect")

    cv2.imwrite('unsharp_mask.jpg', unsharp)
    cv2.imwrite('unsharp_mask2.jpg', unsharp2)


def filter_comparison():
    """Filter comparison"""
    print("\n" + "=" * 50)
    print("Filter Comparison Summary")
    print("=" * 50)

    print("""
    | Filter | Characteristics | Use Case |
    |--------|----------------|----------|
    | blur (Box) | Uniform weights | Simple averaging |
    | GaussianBlur | Higher center weight | Natural blur |
    | medianBlur | Uses median value | Salt-and-pepper noise |
    | bilateralFilter | Edge preserving | Skin smoothing, etc. |
    """)


def main():
    """Main function"""
    # Blur filter
    blur_demo()

    # Gaussian blur
    gaussian_blur_demo()

    # Median blur
    median_blur_demo()

    # Bilateral filter
    bilateral_filter_demo()

    # Custom filter
    custom_filter_demo()

    # Unsharp masking
    unsharp_masking_demo()

    # Comparison
    filter_comparison()

    print("\nImage filtering demo complete!")


if __name__ == '__main__':
    main()
