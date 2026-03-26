"""
01. Environment Setup and Basics
- OpenCV installation check
- Version information
- Creating basic test images
"""

import cv2
import numpy as np
import sys


def check_opencv_installation():
    """Check OpenCV installation and version"""
    print("=" * 50)
    print("OpenCV Environment Check")
    print("=" * 50)

    # Version check
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")

    # Build information
    print("\n[Build Information]")
    build_info = cv2.getBuildInformation()
    # Print only key information
    for line in build_info.split('\n')[:20]:
        if line.strip():
            print(line)


def check_available_modules():
    """Check available modules"""
    print("\n" + "=" * 50)
    print("Available Key Features")
    print("=" * 50)

    # SIFT check (requires contrib)
    try:
        sift = cv2.SIFT_create()
        print("SIFT: Available")
    except AttributeError:
        print("SIFT: Not available (opencv-contrib-python required)")

    # ORB check
    try:
        orb = cv2.ORB_create()
        print("ORB: Available")
    except AttributeError:
        print("ORB: Not available")

    # DNN check
    try:
        net = cv2.dnn.readNet
        print("DNN module: Available")
    except AttributeError:
        print("DNN module: Not available")

    # Haar Cascade check
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"Haar Cascade path: {cv2.data.haarcascades}")


def create_test_image():
    """Create a test image"""
    print("\n" + "=" * 50)
    print("Test Image Generation")
    print("=" * 50)

    # Create color image (400x400, BGR)
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Background gradient
    for i in range(400):
        img[i, :] = [i * 255 // 400, 100, 255 - i * 255 // 400]

    # Draw shapes
    # Rectangle
    cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), 2)
    cv2.rectangle(img, (60, 60), (140, 140), (0, 255, 0), -1)  # Filled

    # Circle
    cv2.circle(img, (300, 100), 50, (255, 0, 0), 2)
    cv2.circle(img, (300, 100), 30, (255, 0, 0), -1)

    # Line
    cv2.line(img, (50, 250), (350, 250), (0, 0, 255), 3)
    cv2.line(img, (50, 280), (350, 320), (255, 255, 0), 2)

    # Ellipse
    cv2.ellipse(img, (200, 350), (100, 30), 0, 0, 360, (255, 0, 255), 2)

    # Text
    cv2.putText(img, 'OpenCV Test', (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save
    cv2.imwrite('sample.jpg', img)
    print("sample.jpg created successfully")

    # Also create a grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('sample_gray.jpg', gray)
    print("sample_gray.jpg created successfully")

    return img


def basic_operations_demo(img):
    """Basic operations demo"""
    print("\n" + "=" * 50)
    print("Basic Operations Demo")
    print("=" * 50)

    # Image properties
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image size: {img.size}")

    # Pixel access
    pixel = img[100, 100]
    print(f"Pixel (100, 100) BGR value: {pixel}")

    # ROI (Region of Interest)
    roi = img[50:150, 50:150]
    print(f"ROI shape: {roi.shape}")


def main():
    """Main function"""
    # Environment check
    check_opencv_installation()
    check_available_modules()

    # Create test image
    img = create_test_image()

    # Basic operations demo
    basic_operations_demo(img)

    # Display image (in GUI environment)
    print("\n[Image Display]")
    print("Press any key to close the image window...")

    try:
        cv2.imshow('Test Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"GUI display not available: {e}")
        print("cv2.imshow() cannot be used in headless environments")

    print("\nEnvironment setup check complete!")


if __name__ == '__main__':
    main()
