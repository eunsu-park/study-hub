"""
13. Feature Detection
- Harris corner detection
- FAST features
- SIFT, ORB
- Keypoints and descriptors
"""

import cv2
import numpy as np


def create_test_image():
    """Test image with corners"""
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # Rectangle (clear corners)
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 2)
    cv2.rectangle(img, (50, 50), (150, 150), (100, 100, 100), -1)

    # Another rectangle
    cv2.rectangle(img, (200, 80), (350, 180), (50, 50, 50), -1)

    # Checkerboard pattern (many corners)
    for i in range(4):
        for j in range(4):
            x = 50 + i * 40
            y = 220 + j * 40
            if (i + j) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + 40, y + 40), (0, 0, 0), -1)

    # Circle (no corners)
    cv2.circle(img, (400, 100), 50, (80, 80, 80), -1)

    # Text
    cv2.putText(img, 'FEATURES', (280, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img


def harris_corner_demo():
    """Harris corner detection demo"""
    print("=" * 50)
    print("Harris Corner Detection")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Harris corner detection
    # blockSize: Corner detection window size
    # ksize: Sobel kernel size
    # k: Harris parameter (0.04~0.06)
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Dilate result (for visualization)
    harris_dilated = cv2.dilate(harris, None)

    # Apply threshold
    threshold = 0.01 * harris_dilated.max()
    result = img.copy()
    result[harris_dilated > threshold] = [0, 0, 255]  # Mark in red

    # Precise corner location (SubPixel)
    _, harris_binary = cv2.threshold(harris_dilated, threshold, 255, cv2.THRESH_BINARY)
    harris_binary = np.uint8(harris_binary)

    # Count corners using connected components
    num_corners = cv2.connectedComponents(harris_binary)[0] - 1

    print(f"Number of corners detected: {num_corners}")
    print("\nHarris corner properties:")
    print("  - Rotation invariant")
    print("  - Sensitive to scale changes")
    print("  - Corner response function R = det(M) - k*trace(M)^2")

    cv2.imwrite('harris_input.jpg', img)
    cv2.imwrite('harris_result.jpg', result)


def shi_tomasi_demo():
    """Shi-Tomasi corner detection demo"""
    print("\n" + "=" * 50)
    print("Shi-Tomasi Corner Detection (goodFeaturesToTrack)")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi corner detection
    # maxCorners: Maximum number of corners
    # qualityLevel: Quality level (0~1)
    # minDistance: Minimum distance between corners
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=10
    )

    result = img.copy()

    if corners is not None:
        corners = np.int32(corners)
        print(f"Number of corners detected: {len(corners)}")

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)

    print("\nShi-Tomasi vs Harris:")
    print("  - Uses R = min(lambda1, lambda2)")
    print("  - More stable than Harris")
    print("  - Selects corners suitable for tracking")

    cv2.imwrite('shi_tomasi_result.jpg', result)


def fast_demo():
    """FAST feature detection demo"""
    print("\n" + "=" * 50)
    print("FAST Feature Detection")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # FAST detector creation
    # threshold: Brightness difference threshold
    # nonmaxSuppression: Non-maximum suppression
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

    # Keypoint detection
    keypoints = fast.detect(gray, None)

    # Draw results
    result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    print(f"Number of keypoints detected: {len(keypoints)}")
    print(f"Threshold: {fast.getThreshold()}")
    print(f"NonMax Suppression: {fast.getNonmaxSuppression()}")

    print("\nFAST properties:")
    print("  - Very fast speed")
    print("  - Corner detection using circular pattern")
    print("  - No descriptor (detection only)")

    cv2.imwrite('fast_result.jpg', result)


def orb_demo():
    """ORB feature detection demo"""
    print("\n" + "=" * 50)
    print("ORB (Oriented FAST and Rotated BRIEF)")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ORB detector creation
    orb = cv2.ORB_create(nfeatures=500)

    # Compute keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw results
    result = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"Number of keypoints detected: {len(keypoints)}")
    if descriptors is not None:
        print(f"Descriptor shape: {descriptors.shape}")
        print(f"Descriptor type: {descriptors.dtype}")

    print("\nORB properties:")
    print("  - FAST-based keypoint detection")
    print("  - BRIEF-based descriptor")
    print("  - Rotation invariance added")
    print("  - Patent-free, fast")

    cv2.imwrite('orb_result.jpg', result)

    return keypoints, descriptors


def sift_demo():
    """SIFT feature detection demo"""
    print("\n" + "=" * 50)
    print("SIFT (Scale-Invariant Feature Transform)")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        # SIFT detector creation
        sift = cv2.SIFT_create()

        # Compute keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Draw results
        result = cv2.drawKeypoints(
            img, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        print(f"Number of keypoints detected: {len(keypoints)}")
        if descriptors is not None:
            print(f"Descriptor shape: {descriptors.shape}")
            print(f"Descriptor type: {descriptors.dtype}")

        print("\nSIFT properties:")
        print("  - Scale invariant (DoG pyramid)")
        print("  - Rotation invariant (orientation assignment)")
        print("  - 128-dimensional descriptor")
        print("  - Requires opencv-contrib-python")

        cv2.imwrite('sift_result.jpg', result)

    except AttributeError:
        print("opencv-contrib-python is required to use SIFT.")
        print("pip install opencv-contrib-python")


def keypoint_info_demo():
    """Keypoint information demo"""
    print("\n" + "=" * 50)
    print("Keypoint Information")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=10)
    keypoints, _ = orb.detectAndCompute(gray, None)

    print("Keypoint attributes:")
    for i, kp in enumerate(keypoints[:5]):
        print(f"\n  Keypoint {i}:")
        print(f"    Position (pt): ({kp.pt[0]:.1f}, {kp.pt[1]:.1f})")
        print(f"    Size: {kp.size:.1f}")
        print(f"    Angle: {kp.angle:.1f}")
        print(f"    Response: {kp.response:.4f}")
        print(f"    Octave: {kp.octave}")


def compare_detectors():
    """Feature detector comparison"""
    print("\n" + "=" * 50)
    print("Feature Detector Comparison")
    print("=" * 50)

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detectors = []

    # FAST
    fast = cv2.FastFeatureDetector_create()
    kp_fast = fast.detect(gray, None)
    detectors.append(('FAST', len(kp_fast)))

    # ORB
    orb = cv2.ORB_create()
    kp_orb, _ = orb.detectAndCompute(gray, None)
    detectors.append(('ORB', len(kp_orb)))

    # SIFT (if available)
    try:
        sift = cv2.SIFT_create()
        kp_sift, _ = sift.detectAndCompute(gray, None)
        detectors.append(('SIFT', len(kp_sift)))
    except AttributeError:
        detectors.append(('SIFT', 'N/A'))

    # Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 10)
    detectors.append(('Shi-Tomasi', len(corners) if corners is not None else 0))

    print("\n| Detector | Keypoints | Features |")
    print("|----------|-----------|----------|")
    print(f"| FAST | {detectors[0][1]} | Fast, no descriptor |")
    print(f"| ORB | {detectors[1][1]} | Fast, patent-free |")
    print(f"| SIFT | {detectors[2][1]} | Accurate, slow |")
    print(f"| Shi-Tomasi | {detectors[3][1]} | For tracking |")


def main():
    """Main function"""
    # Harris corner
    harris_corner_demo()

    # Shi-Tomasi
    shi_tomasi_demo()

    # FAST
    fast_demo()

    # ORB
    orb_demo()

    # SIFT
    sift_demo()

    # Keypoint info
    keypoint_info_demo()

    # Comparison
    compare_detectors()

    print("\nFeature detection demo complete!")


if __name__ == '__main__':
    main()
