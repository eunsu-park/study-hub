"""
11. Hough Transform
- HoughLines (standard Hough line)
- HoughLinesP (probabilistic Hough line)
- HoughCircles (Hough circle)
"""

import cv2
import numpy as np


def create_line_image():
    """Test image with lines"""
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # Various lines
    cv2.line(img, (50, 50), (450, 50), (0, 0, 0), 2)      # Horizontal line
    cv2.line(img, (50, 50), (50, 350), (0, 0, 0), 2)      # Vertical line
    cv2.line(img, (100, 100), (400, 300), (0, 0, 0), 2)   # Diagonal line
    cv2.line(img, (100, 300), (400, 100), (0, 0, 0), 2)   # Diagonal line
    cv2.line(img, (250, 150), (250, 350), (0, 0, 0), 2)   # Vertical line

    return img


def create_circle_image():
    """Test image with circles"""
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # Various circles
    cv2.circle(img, (100, 100), 40, (0, 0, 0), 2)
    cv2.circle(img, (300, 100), 50, (0, 0, 0), 2)
    cv2.circle(img, (150, 250), 60, (0, 0, 0), 2)
    cv2.circle(img, (350, 280), 45, (0, 0, 0), 2)

    # Filled circle
    cv2.circle(img, (450, 350), 30, (0, 0, 0), -1)

    return img


def hough_lines_demo():
    """Standard Hough line transform demo"""
    print("=" * 50)
    print("Standard Hough Line Transform (HoughLines)")
    print("=" * 50)

    img = create_line_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Hough line transform
    # rho: Distance resolution (pixels)
    # theta: Angle resolution (radians)
    # threshold: Minimum number of votes
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    result = img.copy()

    if lines is not None:
        print(f"Number of lines detected: {len(lines)}")

        for line in lines:
            rho, theta = line[0]

            # Polar to Cartesian conversion
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Draw line (sufficiently long segment)
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print("\nHoughLines parameters:")
    print("  rho: Distance resolution (typically 1 pixel)")
    print("  theta: Angle resolution (typically pi/180)")
    print("  threshold: Minimum votes to be considered a line")

    cv2.imwrite('hough_lines_input.jpg', img)
    cv2.imwrite('hough_lines_edges.jpg', edges)
    cv2.imwrite('hough_lines_result.jpg', result)


def hough_lines_p_demo():
    """Probabilistic Hough line transform demo"""
    print("\n" + "=" * 50)
    print("Probabilistic Hough Line Transform (HoughLinesP)")
    print("=" * 50)

    img = create_line_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Probabilistic Hough transform
    # minLineLength: Minimum line length
    # maxLineGap: Maximum gap to consider as a single line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                            minLineLength=50, maxLineGap=10)

    result = img.copy()

    if lines is not None:
        print(f"Number of segments detected: {len(lines)}")

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(result, (x1, y1), 5, (255, 0, 0), -1)
            cv2.circle(result, (x2, y2), 5, (0, 0, 255), -1)

    print("\nHoughLinesP advantages:")
    print("  - Returns start and end points of segments")
    print("  - Faster than standard method")
    print("  - Easier parameter tuning")

    cv2.imwrite('hough_linesp_result.jpg', result)


def hough_lines_params_demo():
    """Hough line parameter effects"""
    print("\n" + "=" * 50)
    print("Hough Line Parameter Effects")
    print("=" * 50)

    img = create_line_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Various threshold values
    thresholds = [30, 50, 100, 150]

    for thresh in thresholds:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, thresh,
                                minLineLength=30, maxLineGap=10)
        result = img.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            print(f"threshold={thresh}: {len(lines)} segments detected")

        cv2.imwrite(f'hough_thresh_{thresh}.jpg', result)


def hough_circles_demo():
    """Hough circle transform demo"""
    print("\n" + "=" * 50)
    print("Hough Circle Transform (HoughCircles)")
    print("=" * 50)

    img = create_circle_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply blur (noise reduction)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough circle transform
    # cv2.HOUGH_GRADIENT: Hough gradient method
    # dp: Accumulator resolution ratio (1 = same as input)
    # minDist: Minimum distance between circle centers
    # param1: Upper threshold for Canny edge detection
    # param2: Circle detection threshold (lower = more circles detected)
    # minRadius, maxRadius: Circle radius range

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    result = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Number of circles detected: {len(circles[0])}")

        for circle in circles[0, :]:
            cx, cy, r = circle

            # Draw circle
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
            # Mark center
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

            print(f"  Circle: center=({cx}, {cy}), radius={r}")

    print("\nHoughCircles parameters:")
    print("  dp: Accumulator resolution (1 recommended)")
    print("  minDist: Minimum distance between circle centers")
    print("  param1: Upper Canny threshold")
    print("  param2: Circle detection threshold (lower = more detections)")
    print("  minRadius/maxRadius: Radius range")

    cv2.imwrite('hough_circles_input.jpg', img)
    cv2.imwrite('hough_circles_result.jpg', result)


def hough_circles_params_demo():
    """Hough circle parameter effects"""
    print("\n" + "=" * 50)
    print("Hough Circle Parameter Effects")
    print("=" * 50)

    img = create_circle_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Varying param2 values
    param2_values = [20, 30, 40, 50]

    for p2 in param2_values:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 50,
            param1=100, param2=p2, minRadius=20, maxRadius=100
        )

        result = img.copy()
        count = 0

        if circles is not None:
            circles = np.uint16(np.around(circles))
            count = len(circles[0])

            for circle in circles[0, :]:
                cx, cy, r = circle
                cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

        print(f"param2={p2}: {count} circles detected")
        cv2.imwrite(f'hough_circles_p2_{p2}.jpg', result)


def practical_lane_detection():
    """Practical example: Lane detection simulation"""
    print("\n" + "=" * 50)
    print("Practical Example: Lane Detection")
    print("=" * 50)

    # Simulate road image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = [100, 100, 100]  # Gray road

    # Draw lanes
    cv2.line(img, (100, 400), (250, 200), (255, 255, 255), 5)  # Left lane
    cv2.line(img, (500, 400), (350, 200), (255, 255, 255), 5)  # Right lane
    cv2.line(img, (300, 400), (300, 250), (255, 255, 0), 3)    # Center line (dashed)

    # Grayscale and edge
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # ROI (Region of Interest) mask
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(50, 400), (550, 400), (350, 180), (250, 180)]], np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough line detection
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30,
                            minLineLength=50, maxLineGap=100)

    result = img.copy()

    if lines is not None:
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)

                # Classify left/right lane by slope
                if slope < -0.3:  # Left lane
                    left_lines.append(line[0])
                elif slope > 0.3:  # Right lane
                    right_lines.append(line[0])

        # Draw lanes
        for line in left_lines:
            cv2.line(result, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)
        for line in right_lines:
            cv2.line(result, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)

        print(f"Left lane: {len(left_lines)}, Right lane: {len(right_lines)}")

    cv2.imwrite('lane_input.jpg', img)
    cv2.imwrite('lane_edges.jpg', masked_edges)
    cv2.imwrite('lane_result.jpg', result)
    print("Lane detection images saved successfully")


def main():
    """Main function"""
    # Standard Hough line
    hough_lines_demo()

    # Probabilistic Hough line
    hough_lines_p_demo()

    # Line parameters
    hough_lines_params_demo()

    # Hough circle
    hough_circles_demo()

    # Circle parameters
    hough_circles_params_demo()

    # Practical example
    practical_lane_detection()

    print("\nHough transform demo complete!")


if __name__ == '__main__':
    main()
