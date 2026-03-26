"""
06. Morphological Operations
- Erosion (erode), Dilation (dilate)
- Opening, Closing
- Gradient, Top Hat, Black Hat
- Structuring element
"""

import cv2
import numpy as np


def create_binary_image():
    """Create a binary image"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # Rectangle
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

    # Circle
    cv2.circle(img, (300, 150), 50, 255, -1)

    # Text
    cv2.putText(img, 'MORPH', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)

    return img


def create_noisy_binary():
    """Create a noisy binary image"""
    img = create_binary_image()

    # Add small noise dots (salt-and-pepper)
    noise_salt = np.random.random(img.shape) < 0.01
    noise_pepper = np.random.random(img.shape) < 0.01
    img[noise_salt] = 255
    img[noise_pepper] = 0

    return img


def structuring_element_demo():
    """Structuring element demo"""
    print("=" * 50)
    print("Structuring Element")
    print("=" * 50)

    # Rectangular structuring element
    rect_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    rect_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Cross-shaped structuring element
    cross_3x3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cross_5x5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    # Elliptical structuring element
    ellipse_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    print("Rectangle 3x3:")
    print(rect_3x3)
    print("\nCross 3x3:")
    print(cross_3x3)
    print("\nEllipse 5x5:")
    print(ellipse_5x5)

    return rect_5x5


def erosion_demo():
    """Erosion demo"""
    print("\n" + "=" * 50)
    print("Erosion")
    print("=" * 50)

    img = create_binary_image()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Apply erosion
    eroded_1 = cv2.erode(img, kernel, iterations=1)
    eroded_2 = cv2.erode(img, kernel, iterations=2)
    eroded_3 = cv2.erode(img, kernel, iterations=3)

    print("Erosion properties:")
    print("  - Shrinks foreground (white) regions")
    print("  - Removes small noise")
    print("  - More iterations = more shrinkage")

    cv2.imwrite('morph_original.jpg', img)
    cv2.imwrite('erode_1.jpg', eroded_1)
    cv2.imwrite('erode_2.jpg', eroded_2)
    cv2.imwrite('erode_3.jpg', eroded_3)


def dilation_demo():
    """Dilation demo"""
    print("\n" + "=" * 50)
    print("Dilation")
    print("=" * 50)

    img = create_binary_image()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Apply dilation
    dilated_1 = cv2.dilate(img, kernel, iterations=1)
    dilated_2 = cv2.dilate(img, kernel, iterations=2)
    dilated_3 = cv2.dilate(img, kernel, iterations=3)

    print("Dilation properties:")
    print("  - Expands foreground (white) regions")
    print("  - Fills holes")
    print("  - Connects objects")

    cv2.imwrite('dilate_1.jpg', dilated_1)
    cv2.imwrite('dilate_2.jpg', dilated_2)
    cv2.imwrite('dilate_3.jpg', dilated_3)


def opening_demo():
    """Opening demo"""
    print("\n" + "=" * 50)
    print("Opening = Erosion + Dilation")
    print("=" * 50)

    img = create_noisy_binary()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Opening = erosion followed by dilation
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Can also be done manually
    opened_manual = cv2.dilate(cv2.erode(img, kernel), kernel)

    print("Opening properties:")
    print("  - Erosion followed by dilation")
    print("  - Removes small noise (white dots)")
    print("  - Object size is mostly preserved")

    cv2.imwrite('noisy_binary.jpg', img)
    cv2.imwrite('opened.jpg', opened)


def closing_demo():
    """Closing demo"""
    print("\n" + "=" * 50)
    print("Closing = Dilation + Erosion")
    print("=" * 50)

    img = create_noisy_binary()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Closing = dilation followed by erosion
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    print("Closing properties:")
    print("  - Dilation followed by erosion")
    print("  - Fills small holes (black dots)")
    print("  - Object size is mostly preserved")

    cv2.imwrite('closed.jpg', closed)


def gradient_demo():
    """Morphological gradient demo"""
    print("\n" + "=" * 50)
    print("Morphological Gradient")
    print("=" * 50)

    img = create_binary_image()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Gradient = dilation - erosion
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # Manual calculation
    dilated = cv2.dilate(img, kernel)
    eroded = cv2.erode(img, kernel)
    gradient_manual = cv2.subtract(dilated, eroded)

    print("Gradient properties:")
    print("  - Dilation - erosion")
    print("  - Extracts object outlines")

    cv2.imwrite('gradient.jpg', gradient)


def tophat_blackhat_demo():
    """Top Hat and Black Hat demo"""
    print("\n" + "=" * 50)
    print("Top Hat & Black Hat")
    print("=" * 50)

    # Simulate an image with uneven lighting
    img = np.zeros((300, 400), dtype=np.uint8)

    # Gradient background (uneven lighting)
    for i in range(300):
        for j in range(400):
            img[i, j] = int(50 + 100 * (i / 300) + 50 * (j / 400))

    # Bright objects
    cv2.rectangle(img, (100, 100), (150, 150), 255, -1)
    cv2.circle(img, (300, 150), 30, 255, -1)

    # Dark objects
    cv2.rectangle(img, (50, 200), (100, 250), 30, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    # Top Hat = original - opening (highlights bright regions)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # Black Hat = closing - original (highlights dark regions)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    print("Top Hat:")
    print("  - Original - opening")
    print("  - Highlights regions brighter than background")
    print("  - Used for uneven lighting correction")

    print("\nBlack Hat:")
    print("  - Closing - original")
    print("  - Highlights regions darker than background")

    cv2.imwrite('uneven_lighting.jpg', img)
    cv2.imwrite('tophat.jpg', tophat)
    cv2.imwrite('blackhat.jpg', blackhat)


def hit_miss_demo():
    """Hit-Miss transform demo"""
    print("\n" + "=" * 50)
    print("Hit-Miss Transform")
    print("=" * 50)

    # Find specific patterns
    img = np.zeros((200, 200), dtype=np.uint8)
    # Create L-shaped pattern
    cv2.rectangle(img, (50, 50), (70, 100), 255, -1)
    cv2.rectangle(img, (50, 80), (100, 100), 255, -1)

    # At another location too
    cv2.rectangle(img, (120, 120), (140, 170), 255, -1)
    cv2.rectangle(img, (120, 150), (170, 170), 255, -1)

    # Hit-Miss kernel (find L-shaped corners)
    kernel = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ], dtype=np.int8)

    # Hit-Miss transform
    hitmiss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

    print("Hit-Miss Transform:")
    print("  - Find specific patterns")
    print("  - 1: foreground, 0: background, -1: don't care")

    cv2.imwrite('hitmiss_input.jpg', img)
    cv2.imwrite('hitmiss_result.jpg', hitmiss)


def practical_example():
    """Practical example: Text cleanup"""
    print("\n" + "=" * 50)
    print("Practical Example: Text Cleanup")
    print("=" * 50)

    # Create noisy text image
    img = np.zeros((100, 400), dtype=np.uint8)
    cv2.putText(img, 'OpenCV', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)

    # Add noise
    noise = np.random.random(img.shape) < 0.05
    img[noise] = 255

    # Remove noise with opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Enhance text with closing
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    enhanced = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel2)

    cv2.imwrite('text_noisy.jpg', img)
    cv2.imwrite('text_cleaned.jpg', cleaned)
    cv2.imwrite('text_enhanced.jpg', enhanced)
    print("Text cleanup images saved successfully")


def main():
    """Main function"""
    # Structuring element
    structuring_element_demo()

    # Erosion
    erosion_demo()

    # Dilation
    dilation_demo()

    # Opening
    opening_demo()

    # Closing
    closing_demo()

    # Gradient
    gradient_demo()

    # Top Hat / Black Hat
    tophat_blackhat_demo()

    # Hit-Miss
    hit_miss_demo()

    # Practical example
    practical_example()

    print("\nMorphological operations demo complete!")


if __name__ == '__main__':
    main()
