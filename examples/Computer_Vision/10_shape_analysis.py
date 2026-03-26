"""
10. Shape Analysis
- moments
- boundingRect, minAreaRect, minEnclosingCircle
- convexHull
- matchShapes
"""

import cv2
import numpy as np


def create_shapes():
    """Create an image with various shapes"""
    img = np.zeros((400, 500), dtype=np.uint8)

    # Rectangle
    cv2.rectangle(img, (30, 30), (130, 100), 255, -1)

    # Rotated rectangle
    pts = np.array([[200, 30], [280, 60], [250, 140], [170, 110]], np.int32)
    cv2.fillPoly(img, [pts], 255)

    # Circle
    cv2.circle(img, (400, 80), 50, 255, -1)

    # Irregular shape
    pts2 = np.array([[50, 200], [100, 180], [150, 220], [130, 280],
                     [80, 300], [30, 260]], np.int32)
    cv2.fillPoly(img, [pts2], 255)

    # L-shape
    pts3 = np.array([[200, 180], [280, 180], [280, 220], [240, 220],
                     [240, 320], [200, 320]], np.int32)
    cv2.fillPoly(img, [pts3], 255)

    # Star shape
    pts_star = []
    for i in range(5):
        outer = np.radians(i * 72 - 90)
        inner = np.radians(i * 72 + 36 - 90)
        pts_star.append([int(400 + 50 * np.cos(outer)), int(250 + 50 * np.sin(outer))])
        pts_star.append([int(400 + 25 * np.cos(inner)), int(250 + 25 * np.sin(inner))])
    cv2.fillPoly(img, [np.array(pts_star, np.int32)], 255)

    return img


def moments_demo():
    """Moments demo"""
    print("=" * 50)
    print("Moments")
    print("=" * 50)

    img = create_shapes()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i, cnt in enumerate(contours):
        # Calculate moments
        M = cv2.moments(cnt)

        # Centroid
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Mark centroid
            cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(color_img, f'{i}', (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            print(f"\nShape {i}:")
            print(f"  Area (m00): {M['m00']:.1f}")
            print(f"  Centroid: ({cx}, {cy})")

            # Hu moments (invariant moments)
            hu = cv2.HuMoments(M)
            print(f"  Hu[0]: {hu[0][0]:.6f}")

    print("\nMoment types:")
    print("  m00: Area (0th moment)")
    print("  m10, m01: 1st moments (for centroid calculation)")
    print("  m20, m02, m11: 2nd moments")
    print("  Hu moments: Rotation and scale invariant")

    cv2.imwrite('moments_centroids.jpg', color_img)


def bounding_shapes_demo():
    """Bounding shapes demo"""
    print("\n" + "=" * 50)
    print("Bounding Shapes")
    print("=" * 50)

    img = create_shapes()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Images for each type of bounding shape
    bound_rect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    min_rect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    min_circle = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    fit_ellipse = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        # 1. Bounding rectangle (axis-aligned)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(bound_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 2. Minimum area rotated rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(min_rect, [box], 0, (0, 255, 0), 2)

        # 3. Minimum enclosing circle
        (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(min_circle, (int(x_c), int(y_c)), int(radius), (0, 255, 0), 2)

        # 4. Ellipse fitting (requires at least 5 points)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(fit_ellipse, ellipse, (0, 255, 0), 2)

    print("Bounding shape types:")
    print("  boundingRect: Axis-aligned rectangle")
    print("  minAreaRect: Minimum area rotated rectangle")
    print("  minEnclosingCircle: Minimum enclosing circle")
    print("  fitEllipse: Ellipse fitting")

    cv2.imwrite('bound_rect.jpg', bound_rect)
    cv2.imwrite('min_rect.jpg', min_rect)
    cv2.imwrite('min_circle.jpg', min_circle)
    cv2.imwrite('fit_ellipse.jpg', fit_ellipse)


def convex_hull_demo():
    """Convex hull demo"""
    print("\n" + "=" * 50)
    print("Convex Hull")
    print("=" * 50)

    img = create_shapes()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hull_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    defects_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        # Convex hull
        hull = cv2.convexHull(cnt)
        cv2.drawContours(hull_img, [hull], 0, (0, 255, 0), 2)

        # Convexity check
        is_convex = cv2.isContourConvex(cnt)

        # Convexity defects
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(cnt, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    far = tuple(cnt[f][0])
                    # Show only defects with sufficient depth
                    if d > 1000:
                        cv2.circle(defects_img, far, 5, (0, 0, 255), -1)

        # Area comparison
        contour_area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print(f"Shape at ({cx}, {cy}): Convex={is_convex}, Solidity={solidity:.2f}")

    print("\nConvex hull applications:")
    print("  - Shape simplification")
    print("  - Solidity = area / convex hull area (fill ratio)")
    print("  - Hand gesture recognition (defect points between fingers)")

    cv2.imwrite('convex_hull.jpg', hull_img)
    cv2.imwrite('convex_defects.jpg', defects_img)


def match_shapes_demo():
    """Shape matching demo"""
    print("\n" + "=" * 50)
    print("Shape Matching (matchShapes)")
    print("=" * 50)

    # Reference shape
    template = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(template, (100, 100), 50, 255, -1)

    # Comparison shapes
    shapes = []

    # Circle (similar)
    shape1 = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(shape1, (100, 100), 60, 255, -1)
    shapes.append(('Circle (larger)', shape1))

    # Ellipse (different)
    shape2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(shape2, (100, 100), (60, 40), 0, 0, 360, 255, -1)
    shapes.append(('Ellipse', shape2))

    # Square (very different)
    shape3 = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(shape3, (40, 40), (160, 160), 255, -1)
    shapes.append(('Square', shape3))

    # Triangle (very different)
    shape4 = np.zeros((200, 200), dtype=np.uint8)
    pts = np.array([[100, 30], [30, 170], [170, 170]], np.int32)
    cv2.fillPoly(shape4, [pts], 255)
    shapes.append(('Triangle', shape4))

    # Template contour
    cnt_template, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("Template: Circle")
    print("Matching results (lower = more similar):\n")

    for name, shape in shapes:
        cnt_shape, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Hu moment-based matching
        match1 = cv2.matchShapes(cnt_template[0], cnt_shape[0], cv2.CONTOURS_MATCH_I1, 0)
        match2 = cv2.matchShapes(cnt_template[0], cnt_shape[0], cv2.CONTOURS_MATCH_I2, 0)
        match3 = cv2.matchShapes(cnt_template[0], cnt_shape[0], cv2.CONTOURS_MATCH_I3, 0)

        print(f"  {name:15}: I1={match1:.4f}, I2={match2:.4f}, I3={match3:.4f}")

    print("\nMatching methods:")
    print("  CONTOURS_MATCH_I1: sum|1/huA - 1/huB|")
    print("  CONTOURS_MATCH_I2: sum|huA - huB|")
    print("  CONTOURS_MATCH_I3: max(|huA - huB|/|huA|)")

    cv2.imwrite('match_template.jpg', template)


def extreme_points_demo():
    """Extreme points demo"""
    print("\n" + "=" * 50)
    print("Extreme Points")
    print("=" * 50)

    img = create_shapes()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        # Find extreme points
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

        # Display
        cv2.circle(color_img, leftmost, 5, (255, 0, 0), -1)    # Blue: left
        cv2.circle(color_img, rightmost, 5, (0, 255, 0), -1)   # Green: right
        cv2.circle(color_img, topmost, 5, (0, 0, 255), -1)     # Red: top
        cv2.circle(color_img, bottommost, 5, (255, 255, 0), -1) # Cyan: bottom

    print("Extreme points:")
    print("  - Leftmost, rightmost, topmost, bottommost points")
    print("  - Used for finding fingertips in hand detection")

    cv2.imwrite('extreme_points.jpg', color_img)


def shape_descriptors_demo():
    """Shape descriptors demo"""
    print("\n" + "=" * 50)
    print("Shape Descriptors")
    print("=" * 50)

    img = create_shapes()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        # Calculate descriptors
        aspect_ratio = float(w) / h
        extent = area / (w * h)  # Area relative to bounding rectangle
        solidity = area / hull_area if hull_area > 0 else 0  # Area relative to convex hull
        equiv_diameter = np.sqrt(4 * area / np.pi)  # Equivalent diameter
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        print(f"\nShape {i}:")
        print(f"  Aspect Ratio (width/height): {aspect_ratio:.2f}")
        print(f"  Extent (area/bounding area): {extent:.2f}")
        print(f"  Solidity (area/convex area): {solidity:.2f}")
        print(f"  Equivalent Diameter: {equiv_diameter:.1f}")
        print(f"  Circularity: {circularity:.2f}")


def main():
    """Main function"""
    # Moments
    moments_demo()

    # Bounding shapes
    bounding_shapes_demo()

    # Convex hull
    convex_hull_demo()

    # Shape matching
    match_shapes_demo()

    # Extreme points
    extreme_points_demo()

    # Shape descriptors
    shape_descriptors_demo()

    print("\nShape analysis demo complete!")


if __name__ == '__main__':
    main()
