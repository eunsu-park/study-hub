"""
02. Image Basic Operations
- imread, imshow, imwrite
- Pixel access and modification
- ROI (Region of Interest)
- Image copying and channel manipulation
"""

import cv2
import numpy as np


def create_sample_image():
    """Create a sample image"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Color regions
    img[0:100, 0:200] = [255, 0, 0]      # Blue
    img[0:100, 200:400] = [0, 255, 0]    # Green
    img[100:200, 0:200] = [0, 0, 255]    # Red
    img[100:200, 200:400] = [255, 255, 0] # Cyan
    img[200:300, :] = [128, 128, 128]    # Gray

    return img


def image_read_write_demo():
    """Image read/write demo"""
    print("=" * 50)
    print("Image Read/Write")
    print("=" * 50)

    # Create and save image
    img = create_sample_image()
    cv2.imwrite('test_image.jpg', img)
    print("test_image.jpg saved successfully")

    # Read image
    # cv2.IMREAD_COLOR: Color (default)
    # cv2.IMREAD_GRAYSCALE: Grayscale
    # cv2.IMREAD_UNCHANGED: Original as-is (including alpha channel)

    img_color = cv2.imread('test_image.jpg', cv2.IMREAD_COLOR)
    img_gray = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)

    print(f"Color image shape: {img_color.shape}")
    print(f"Grayscale image shape: {img_gray.shape}")

    return img_color


def pixel_access_demo(img):
    """Pixel access demo"""
    print("\n" + "=" * 50)
    print("Pixel Access")
    print("=" * 50)

    # Single pixel access (note the y, x order!)
    pixel = img[50, 100]  # Position (y=50, x=100)
    print(f"Pixel (50, 100) BGR value: {pixel}")

    # Individual channel access
    b = img[50, 100, 0]
    g = img[50, 100, 1]
    r = img[50, 100, 2]
    print(f"B={b}, G={g}, R={r}")

    # Pixel modification
    img_copy = img.copy()
    img_copy[50, 100] = [0, 0, 0]  # Change to black

    # Region modification (more efficient)
    img_copy[0:50, 0:50] = [255, 255, 255]  # White rectangle

    return img_copy


def roi_demo(img):
    """ROI (Region of Interest) demo"""
    print("\n" + "=" * 50)
    print("ROI (Region of Interest)")
    print("=" * 50)

    # Extract ROI (slicing)
    roi = img[50:150, 100:250]  # y: 50~150, x: 100~250
    print(f"Original shape: {img.shape}")
    print(f"ROI shape: {roi.shape}")

    # Copy ROI (no effect on original)
    roi_copy = img[50:150, 100:250].copy()

    # Paste ROI
    img_with_roi = img.copy()
    img_with_roi[150:250, 200:350] = roi  # Paste at a different location

    cv2.imwrite('roi_demo.jpg', img_with_roi)
    print("roi_demo.jpg saved successfully")

    return roi


def channel_operations_demo(img):
    """Channel operations demo"""
    print("\n" + "=" * 50)
    print("Channel Operations")
    print("=" * 50)

    # Split channels
    b, g, r = cv2.split(img)
    print(f"B channel shape: {b.shape}")
    print(f"G channel shape: {g.shape}")
    print(f"R channel shape: {r.shape}")

    # Merge channels
    merged = cv2.merge([b, g, r])
    print(f"After merge shape: {merged.shape}")

    # Change channel order (BGR -> RGB)
    rgb = cv2.merge([r, g, b])

    # Create images using a single channel only
    zeros = np.zeros_like(b)
    only_blue = cv2.merge([b, zeros, zeros])
    only_green = cv2.merge([zeros, g, zeros])
    only_red = cv2.merge([zeros, zeros, r])

    cv2.imwrite('only_blue.jpg', only_blue)
    cv2.imwrite('only_green.jpg', only_green)
    cv2.imwrite('only_red.jpg', only_red)
    print("Channel-separated images saved successfully")


def image_properties_demo(img):
    """Image properties demo"""
    print("\n" + "=" * 50)
    print("Image Properties")
    print("=" * 50)

    # Basic properties
    print(f"Shape (H, W, C): {img.shape}")
    print(f"Height: {img.shape[0]}")
    print(f"Width: {img.shape[1]}")
    print(f"Channels: {img.shape[2] if len(img.shape) > 2 else 1}")

    # Data type
    print(f"Data type: {img.dtype}")

    # Total number of pixels
    print(f"Total pixels: {img.size}")

    # Memory usage
    print(f"Memory (bytes): {img.nbytes}")

    # Pixel value range
    print(f"Min value: {img.min()}")
    print(f"Max value: {img.max()}")
    print(f"Mean value: {img.mean():.2f}")


def image_arithmetic_demo():
    """Image arithmetic operations demo"""
    print("\n" + "=" * 50)
    print("Image Arithmetic Operations")
    print("=" * 50)

    # Create two images
    img1 = np.full((200, 200, 3), 100, dtype=np.uint8)
    img2 = np.full((200, 200, 3), 200, dtype=np.uint8)

    # NumPy addition (overflow occurs)
    result_np = img1 + img2
    print(f"NumPy addition result (100+200): {result_np[0, 0]}")  # 44 (overflow)

    # OpenCV addition (saturated operation)
    result_cv = cv2.add(img1, img2)
    print(f"OpenCV addition result (100+200): {result_cv[0, 0]}")  # 255 (saturated)

    # Weighted sum (blending)
    alpha = 0.7
    beta = 0.3
    blended = cv2.addWeighted(img1, alpha, img2, beta, 0)
    print(f"Blending result (0.7*100 + 0.3*200): {blended[0, 0]}")

    # Subtraction
    diff = cv2.subtract(img2, img1)
    print(f"Subtraction result (200-100): {diff[0, 0]}")

    # Bitwise operations
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255

    masked = cv2.bitwise_and(img1, img1, mask=mask)
    cv2.imwrite('masked_result.jpg', masked)
    print("masked_result.jpg saved successfully")


def main():
    """Main function"""
    # Image read/write
    img = image_read_write_demo()

    # Pixel access
    modified = pixel_access_demo(img)

    # ROI
    roi = roi_demo(img)

    # Channel operations
    channel_operations_demo(img)

    # Image properties
    image_properties_demo(img)

    # Arithmetic operations
    image_arithmetic_demo()

    print("\nImage basic operations complete!")


if __name__ == '__main__':
    main()
