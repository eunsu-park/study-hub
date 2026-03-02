"""
09. Contour Detection
- findContours, drawContours
- Contour hierarchy
- Contour approximation (approxPolyDP)
"""

import cv2
import numpy as np


def create_shapes_image():
    """Binary image with shapes"""
    img = np.zeros((400, 500), dtype=np.uint8)

    # Rectangle
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

    # Circle
    cv2.circle(img, (300, 100), 50, 255, -1)

    # Triangle
    pts = np.array([[400, 50], [350, 150], [450, 150]], np.int32)
    cv2.fillPoly(img, [pts], 255)

    # Nested rectangles (hierarchy)
    cv2.rectangle(img, (50, 200), (200, 350), 255, -1)
    cv2.rectangle(img, (80, 230), (170, 320), 0, -1)  # Inner hole
    cv2.rectangle(img, (100, 250), (150, 290), 255, -1)  # Object inside hole

    # Star shape
    pts_star = np.array([
        [350, 200], [365, 250], [420, 250], [375, 280],
        [390, 330], [350, 300], [310, 330], [325, 280],
        [280, 250], [335, 250]
    ], np.int32)
    cv2.fillPoly(img, [pts_star], 255)

    return img


def find_contours_demo():
    """Find contours demo"""
    print("=" * 50)
    print("Finding Contours (findContours)")
    print("=" * 50)

    img = create_shapes_image()

    # Find contours
    # RETR_EXTERNAL: External contours only
    # RETR_LIST: All contours (no hierarchy)
    # RETR_TREE: All contours (with hierarchy)
    # RETR_CCOMP: Two-level hierarchy

    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Number of contours detected: {len(contours)}")
    print(f"Hierarchy shape: {hierarchy.shape if hierarchy is not None else None}")

    # Retrieval mode description
    print("\nRetrieval Mode:")
    print("  RETR_EXTERNAL: Outermost contours only")
    print("  RETR_LIST: All contours (flat list)")
    print("  RETR_TREE: Full hierarchy")
    print("  RETR_CCOMP: Two-level hierarchy")

    return img, contours, hierarchy


def draw_contours_demo():
    """Draw contours demo"""
    print("\n" + "=" * 50)
    print("Drawing Contours (drawContours)")
    print("=" * 50)

    img = create_shapes_image()
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to color image (for drawing)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw all contours
    all_contours = color_img.copy()
    cv2.drawContours(all_contours, contours, -1, (0, 255, 0), 2)

    # Draw specific contours only
    specific = color_img.copy()
    cv2.drawContours(specific, contours, 0, (255, 0, 0), 2)  # First contour
    cv2.drawContours(specific, contours, 1, (0, 255, 0), 2)  # Second contour

    # Fill
    filled = color_img.copy()
    cv2.drawContours(filled, contours, 0, (0, 0, 255), -1)  # thickness=-1 -> fill

    print("drawContours parameters:")
    print("  contourIdx=-1: All contours")
    print("  contourIdx=n: Only the n-th contour")
    print("  thickness=-1: Fill interior")

    cv2.imwrite('contours_original.jpg', img)
    cv2.imwrite('contours_all.jpg', all_contours)
    cv2.imwrite('contours_specific.jpg', specific)
    cv2.imwrite('contours_filled.jpg', filled)


def contour_hierarchy_demo():
    """Contour hierarchy demo"""
    print("\n" + "=" * 50)
    print("Contour Hierarchy")
    print("=" * 50)

    img = create_shapes_image()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # hierarchy[i] = [Next, Previous, First_Child, Parent]
    print("Hierarchy format: [Next, Previous, First_Child, Parent]")
    print("(-1 means no such relationship)")

    for i, h in enumerate(hierarchy[0]):
        print(f"Contour {i}: Next={h[0]}, Prev={h[1]}, Child={h[2]}, Parent={h[3]}")

    # External contours only (those without a parent)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    external_only = color_img.copy()

    for i, h in enumerate(hierarchy[0]):
        if h[3] == -1:  # No parent = external contour
            cv2.drawContours(external_only, contours, i, (0, 255, 0), 2)

    cv2.imwrite('contours_external.jpg', external_only)


def contour_approximation_demo():
    """Contour approximation demo"""
    print("\n" + "=" * 50)
    print("Contour Approximation (approxPolyDP)")
    print("=" * 50)

    # Complex curve image
    img = np.zeros((300, 400), dtype=np.uint8)
    pts = []
    for angle in range(0, 360, 5):
        r = 80 + 20 * np.sin(5 * np.radians(angle))
        x = int(200 + r * np.cos(np.radians(angle)))
        y = int(150 + r * np.sin(np.radians(angle)))
        pts.append([x, y])
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], 255)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Original contour
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Approximate with various epsilon values
    epsilons = [0.01, 0.02, 0.05, 0.1]

    for eps in epsilons:
        approx_img = color_img.copy()
        for cnt in contours:
            # epsilon = ratio of perimeter
            epsilon = eps * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            print(f"epsilon={eps}: {len(cnt)} points -> {len(approx)} points")

            cv2.drawContours(approx_img, [approx], -1, (0, 255, 0), 2)
            # Mark vertices
            for pt in approx:
                cv2.circle(approx_img, tuple(pt[0]), 3, (0, 0, 255), -1)

        cv2.imwrite(f'approx_eps_{eps}.jpg', approx_img)


def contour_properties_demo():
    """Contour properties demo"""
    print("\n" + "=" * 50)
    print("Contour Properties")
    print("=" * 50)

    img = create_shapes_image()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        # Area
        area = cv2.contourArea(cnt)

        # Perimeter (arc length)
        perimeter = cv2.arcLength(cnt, True)

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)

        # Centroid (moments)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        print(f"\nContour {i}:")
        print(f"  Area: {area:.1f}")
        print(f"  Perimeter: {perimeter:.1f}")
        print(f"  Bounding rectangle: ({x}, {y}, {w}, {h})")
        print(f"  Centroid: ({cx}, {cy})")


def detect_shapes():
    """Shape recognition"""
    print("\n" + "=" * 50)
    print("Shape Recognition")
    print("=" * 50)

    img = create_shapes_image()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        # Contour approximation
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Calculate centroid
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            continue

        # Determine shape by number of vertices
        vertices = len(approx)

        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            # Square vs Rectangle
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        elif vertices > 5:
            # Circle determination (area/perimeter ratio)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            shape = "Circle" if circularity > 0.8 else f"Polygon({vertices})"
        else:
            shape = f"Polygon({vertices})"

        # Display result
        cv2.drawContours(color_img, [approx], -1, (0, 255, 0), 2)
        cv2.putText(color_img, shape, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"  {shape}: {vertices} vertices")

    cv2.imwrite('shapes_detected.jpg', color_img)


def main():
    """Main function"""
    # Find contours
    find_contours_demo()

    # Draw contours
    draw_contours_demo()

    # Hierarchy
    contour_hierarchy_demo()

    # Contour approximation
    contour_approximation_demo()

    # Contour properties
    contour_properties_demo()

    # Shape recognition
    detect_shapes()

    print("\nContour detection demo complete!")


if __name__ == '__main__':
    main()
