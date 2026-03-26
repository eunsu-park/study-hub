"""
03. Color Spaces
- BGR, RGB, HSV, LAB, YCrCb
- cvtColor conversion
- Channel splitting/merging
- Color-based object extraction
"""

import cv2
import numpy as np


def create_color_image():
    """Create a test image with various colors"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Rainbow colors
    colors = [
        [0, 0, 255],      # Red
        [0, 128, 255],    # Orange
        [0, 255, 255],    # Yellow
        [0, 255, 0],      # Green
        [255, 255, 0],    # Cyan
        [255, 0, 0],      # Blue
        [255, 0, 128],    # Purple
    ]

    width = 400 // len(colors)
    for i, color in enumerate(colors):
        img[:, i*width:(i+1)*width] = color

    return img


def bgr_rgb_demo():
    """BGR vs RGB demo"""
    print("=" * 50)
    print("BGR vs RGB")
    print("=" * 50)

    img = create_color_image()

    # OpenCV uses BGR order
    print("OpenCV uses BGR order")
    print(f"Red pixel BGR: {img[150, 25]}")  # [0, 0, 255]

    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"After conversion RGB: {rgb[150, 25]}")  # [255, 0, 0]

    # matplotlib uses RGB
    # plt.imshow(rgb)  # Correct colors
    # plt.imshow(img)  # Colors swapped

    cv2.imwrite('color_bgr.jpg', img)
    cv2.imwrite('color_rgb.jpg', rgb)


def hsv_demo():
    """HSV color space demo"""
    print("\n" + "=" * 50)
    print("HSV Color Space")
    print("=" * 50)

    img = create_color_image()

    # BGR -> HSV conversion
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Split HSV channels
    h, s, v = cv2.split(hsv)

    print("HSV ranges:")
    print("  H (Hue): 0-179")
    print("  S (Saturation): 0-255")
    print("  V (Value): 0-255")

    # HSV of the red region
    print(f"\nRed HSV: {hsv[150, 25]}")
    print(f"  H={hsv[150, 25, 0]}, S={hsv[150, 25, 1]}, V={hsv[150, 25, 2]}")

    # Save per-channel images
    cv2.imwrite('hsv_h.jpg', h)
    cv2.imwrite('hsv_s.jpg', s)
    cv2.imwrite('hsv_v.jpg', v)
    print("\nHSV channel images saved successfully")

    return hsv


def color_extraction_demo():
    """Color-based object extraction demo"""
    print("\n" + "=" * 50)
    print("Color-Based Object Extraction")
    print("=" * 50)

    # Create image with various colored objects
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(img, (100, 150), 50, (0, 0, 255), -1)   # Red circle
    cv2.circle(img, (200, 150), 50, (0, 255, 0), -1)   # Green circle
    cv2.circle(img, (300, 150), 50, (255, 0, 0), -1)   # Blue circle

    # HSV conversion
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red range (HSV)
    # Red is near H=0 or H=180
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Extract red
    red_only = cv2.bitwise_and(img, img, mask=red_mask)

    # Green range
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_only = cv2.bitwise_and(img, img, mask=green_mask)

    # Blue range
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_only = cv2.bitwise_and(img, img, mask=blue_mask)

    cv2.imwrite('original_circles.jpg', img)
    cv2.imwrite('red_extracted.jpg', red_only)
    cv2.imwrite('green_extracted.jpg', green_only)
    cv2.imwrite('blue_extracted.jpg', blue_only)
    print("Color extraction images saved successfully")


def lab_demo():
    """LAB color space demo"""
    print("\n" + "=" * 50)
    print("LAB Color Space")
    print("=" * 50)

    img = create_color_image()

    # BGR -> LAB conversion
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    print("LAB ranges:")
    print("  L (Lightness): 0-255")
    print("  a (Green-Red): 0-255 (128 is neutral)")
    print("  b (Blue-Yellow): 0-255 (128 is neutral)")

    cv2.imwrite('lab_l.jpg', l)
    cv2.imwrite('lab_a.jpg', a)
    cv2.imwrite('lab_b.jpg', b)
    print("LAB channel images saved successfully")


def ycrcb_demo():
    """YCrCb color space demo"""
    print("\n" + "=" * 50)
    print("YCrCb Color Space")
    print("=" * 50)

    img = create_color_image()

    # BGR -> YCrCb conversion
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Split YCrCb channels
    y, cr, cb = cv2.split(ycrcb)

    print("YCrCb ranges:")
    print("  Y (Luminance): 0-255")
    print("  Cr (Red-difference): 0-255")
    print("  Cb (Blue-difference): 0-255")

    cv2.imwrite('ycrcb_y.jpg', y)
    cv2.imwrite('ycrcb_cr.jpg', cr)
    cv2.imwrite('ycrcb_cb.jpg', cb)
    print("YCrCb channel images saved successfully")


def grayscale_methods():
    """Grayscale conversion methods"""
    print("\n" + "=" * 50)
    print("Grayscale Conversion")
    print("=" * 50)

    img = create_color_image()

    # Method 1: cvtColor
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Method 2: Weighted average (manual calculation)
    # Gray = 0.299*R + 0.587*G + 0.114*B
    b, g, r = cv2.split(img)
    gray2 = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    # Method 3: Simple average
    gray3 = np.mean(img, axis=2).astype(np.uint8)

    cv2.imwrite('gray_cvtcolor.jpg', gray1)
    cv2.imwrite('gray_weighted.jpg', gray2)
    cv2.imwrite('gray_average.jpg', gray3)
    print("Grayscale conversion images saved successfully")

    print(f"\nConversion result comparison (specific pixel):")
    print(f"  cvtColor: {gray1[150, 25]}")
    print(f"  Weighted: {gray2[150, 25]}")
    print(f"  Average: {gray3[150, 25]}")


def color_conversion_table():
    """Color conversion code table"""
    print("\n" + "=" * 50)
    print("Key Color Conversion Codes")
    print("=" * 50)

    conversions = [
        ("BGR -> Gray", "cv2.COLOR_BGR2GRAY"),
        ("BGR -> RGB", "cv2.COLOR_BGR2RGB"),
        ("BGR -> HSV", "cv2.COLOR_BGR2HSV"),
        ("BGR -> LAB", "cv2.COLOR_BGR2LAB"),
        ("BGR -> YCrCb", "cv2.COLOR_BGR2YCrCb"),
        ("HSV -> BGR", "cv2.COLOR_HSV2BGR"),
        ("Gray -> BGR", "cv2.COLOR_GRAY2BGR"),
    ]

    for desc, code in conversions:
        print(f"  {desc:15} -> {code}")


def main():
    """Main function"""
    # BGR vs RGB
    bgr_rgb_demo()

    # HSV
    hsv_demo()

    # Color extraction
    color_extraction_demo()

    # LAB
    lab_demo()

    # YCrCb
    ycrcb_demo()

    # Grayscale
    grayscale_methods()

    # Conversion code table
    color_conversion_table()

    print("\nColor spaces demo complete!")


if __name__ == '__main__':
    main()
