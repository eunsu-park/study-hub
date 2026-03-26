"""
04. Geometric Transformations
- resize, rotate, flip
- Affine transformation (warpAffine)
- Perspective transformation (warpPerspective)
- Translation, rotation, scaling
"""

import cv2
import numpy as np


def create_test_image():
    """Create a test image"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]  # Light gray background

    # Rectangle
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)

    # Text
    cv2.putText(img, 'OpenCV', (180, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Circle
    cv2.circle(img, (300, 200), 50, (255, 0, 0), -1)

    return img


def resize_demo():
    """Resize demo"""
    print("=" * 50)
    print("Resize")
    print("=" * 50)

    img = create_test_image()
    h, w = img.shape[:2]
    print(f"Original size: {w}x{h}")

    # Resize to absolute dimensions
    resized1 = cv2.resize(img, (200, 150))  # (width, height)
    print(f"Absolute resize: {resized1.shape[1]}x{resized1.shape[0]}")

    # Resize by ratio
    resized2 = cv2.resize(img, None, fx=0.5, fy=0.5)
    print(f"50% reduction: {resized2.shape[1]}x{resized2.shape[0]}")

    resized3 = cv2.resize(img, None, fx=2, fy=2)
    print(f"200% enlargement: {resized3.shape[1]}x{resized3.shape[0]}")

    # Interpolation method comparison
    print("\nInterpolation methods:")
    print("  cv2.INTER_NEAREST: Nearest neighbor (fast, low quality)")
    print("  cv2.INTER_LINEAR: Bilinear (default, balanced)")
    print("  cv2.INTER_CUBIC: Bicubic (good quality, slow)")
    print("  cv2.INTER_AREA: Area-based (suitable for downscaling)")

    # Apply each interpolation method
    enlarged_nearest = cv2.resize(img, (800, 600), interpolation=cv2.INTER_NEAREST)
    enlarged_linear = cv2.resize(img, (800, 600), interpolation=cv2.INTER_LINEAR)
    enlarged_cubic = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite('resize_half.jpg', resized2)
    cv2.imwrite('resize_double.jpg', resized3)
    cv2.imwrite('resize_nearest.jpg', enlarged_nearest)
    cv2.imwrite('resize_cubic.jpg', enlarged_cubic)
    print("\nResized images saved successfully")


def rotate_demo():
    """Rotation demo"""
    print("\n" + "=" * 50)
    print("Rotation")
    print("=" * 50)

    img = create_test_image()

    # Simple 90-degree rotation (built-in function)
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    print("Simple rotations:")
    print(f"  90 degrees clockwise: {rotated_90.shape[1]}x{rotated_90.shape[0]}")
    print(f"  180 degrees: {rotated_180.shape[1]}x{rotated_180.shape[0]}")
    print(f"  270 degrees (counter-clockwise): {rotated_270.shape[1]}x{rotated_270.shape[0]}")

    cv2.imwrite('rotate_90.jpg', rotated_90)
    cv2.imwrite('rotate_180.jpg', rotated_180)
    cv2.imwrite('rotate_270.jpg', rotated_270)

    # Arbitrary angle rotation (using getRotationMatrix2D)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 45-degree rotation, scale 1.0
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_45 = cv2.warpAffine(img, M, (w, h))

    # 30-degree rotation, scale 0.8
    M = cv2.getRotationMatrix2D(center, 30, 0.8)
    rotated_30_scaled = cv2.warpAffine(img, M, (w, h))

    cv2.imwrite('rotate_45.jpg', rotated_45)
    cv2.imwrite('rotate_30_scaled.jpg', rotated_30_scaled)
    print("\nRotated images saved successfully")


def flip_demo():
    """Flip demo"""
    print("\n" + "=" * 50)
    print("Flip")
    print("=" * 50)

    img = create_test_image()

    # Horizontal flip (left-right)
    flipped_h = cv2.flip(img, 1)

    # Vertical flip (top-bottom)
    flipped_v = cv2.flip(img, 0)

    # Both (top-bottom and left-right)
    flipped_both = cv2.flip(img, -1)

    print("Flip codes:")
    print("  flip(img, 1): Horizontal (left-right)")
    print("  flip(img, 0): Vertical (top-bottom)")
    print("  flip(img, -1): Both directions")

    cv2.imwrite('flip_horizontal.jpg', flipped_h)
    cv2.imwrite('flip_vertical.jpg', flipped_v)
    cv2.imwrite('flip_both.jpg', flipped_both)
    print("\nFlipped images saved successfully")


def translation_demo():
    """Translation demo"""
    print("\n" + "=" * 50)
    print("Translation")
    print("=" * 50)

    img = create_test_image()
    h, w = img.shape[:2]

    # Translation matrix: [[1, 0, tx], [0, 1, ty]]
    # tx: x-axis shift (positive: right)
    # ty: y-axis shift (positive: down)

    tx, ty = 50, 30
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, M, (w, h))

    print(f"Translation: x={tx}, y={ty}")
    print(f"Translation matrix:\n{M}")

    cv2.imwrite('translated.jpg', translated)
    print("\nTranslated image saved successfully")


def affine_transform_demo():
    """Affine transformation demo"""
    print("\n" + "=" * 50)
    print("Affine Transform")
    print("=" * 50)

    img = create_test_image()
    h, w = img.shape[:2]

    # Affine transform: 3-point correspondence
    # 3 points in the original image
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    # 3 points after transformation
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    # Compute transformation matrix
    M = cv2.getAffineTransform(pts1, pts2)

    # Apply transformation
    affine = cv2.warpAffine(img, M, (w, h))

    print("Affine transform properties:")
    print("  - Parallel lines remain parallel")
    print("  - Defined by 3-point correspondence")
    print("  - Combination of translation, rotation, scale, and shear")

    cv2.imwrite('affine.jpg', affine)
    print("\nAffine transformed image saved successfully")


def perspective_transform_demo():
    """Perspective transformation demo"""
    print("\n" + "=" * 50)
    print("Perspective Transform")
    print("=" * 50)

    img = create_test_image()
    h, w = img.shape[:2]

    # Perspective transform: 4-point correspondence
    # 4 points in the original image (rectangle corners)
    pts1 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])

    # 4 points after transformation (with perspective effect)
    pts2 = np.float32([[50, 50], [w-50, 20], [30, h-30], [w-30, h-50]])

    # Compute transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply transformation
    perspective = cv2.warpPerspective(img, M, (w, h))

    print("Perspective transform properties:")
    print("  - Parallel lines converge at a vanishing point")
    print("  - Defined by 4-point correspondence")
    print("  - Used for document scan correction")

    cv2.imwrite('perspective.jpg', perspective)

    # Inverse transformation (document correction simulation)
    pts_doc = np.float32([[50, 50], [350, 30], [60, 280], [340, 270]])
    pts_rect = np.float32([[0, 0], [300, 0], [0, 200], [300, 200]])

    M_rect = cv2.getPerspectiveTransform(pts_doc, pts_rect)
    rectified = cv2.warpPerspective(img, M_rect, (300, 200))

    cv2.imwrite('perspective_rectified.jpg', rectified)
    print("Perspective transformed images saved successfully")


def combined_transforms_demo():
    """Combined transformations demo"""
    print("\n" + "=" * 50)
    print("Combined Transformations")
    print("=" * 50)

    img = create_test_image()
    h, w = img.shape[:2]

    # Translation -> Rotation -> Scale all at once
    center = (w // 2, h // 2)
    angle = 30
    scale = 0.8

    # Rotation + scale matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Apply additional translation
    M[0, 2] += 50  # x shift
    M[1, 2] += 20  # y shift

    result = cv2.warpAffine(img, M, (w, h))

    print(f"Combined transform: rotation {angle} degrees, scale {scale}, translation (50, 20)")

    cv2.imwrite('combined_transform.jpg', result)
    print("Combined transform image saved successfully")


def main():
    """Main function"""
    # Resize
    resize_demo()

    # Rotation
    rotate_demo()

    # Flip
    flip_demo()

    # Translation
    translation_demo()

    # Affine transform
    affine_transform_demo()

    # Perspective transform
    perspective_transform_demo()

    # Combined transforms
    combined_transforms_demo()

    print("\nGeometric transformations demo complete!")


if __name__ == '__main__':
    main()
