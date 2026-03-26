"""
Exercises for Lesson 26: Computational Geometry
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: CCW, Line intersection, Convex hull, Polygon area, Closest pair.
"""

import math


# === Exercise 1: CCW (Counter-Clockwise) Test ===
# Problem: Determine the orientation of three points.
#   CCW > 0: counter-clockwise
#   CCW = 0: collinear
#   CCW < 0: clockwise

def exercise_1():
    """Solution: Cross product for orientation test."""
    def ccw(p1, p2, p3):
        """
        Returns the cross product of vectors (p1->p2) and (p1->p3).
        Positive = CCW, Negative = CW, Zero = collinear.
        """
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    tests = [
        ((0, 0), (1, 0), (0, 1), "CCW"),        # counter-clockwise
        ((0, 0), (0, 1), (1, 0), "CW"),          # clockwise
        ((0, 0), (1, 1), (2, 2), "collinear"),   # collinear
    ]

    for p1, p2, p3, expected_type in tests:
        result = ccw(p1, p2, p3)
        if result > 0:
            direction = "CCW"
        elif result < 0:
            direction = "CW"
        else:
            direction = "collinear"
        print(f"ccw({p1}, {p2}, {p3}) = {result} ({direction})")
        assert direction == expected_type

    print("All CCW tests passed!")


# === Exercise 2: Line Segment Intersection ===
# Problem: Determine if two line segments intersect.
# Approach: Use CCW to check if endpoints straddle each other.

def exercise_2():
    """Solution: Segment intersection using CCW."""
    def ccw(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    def on_segment(p, q, r):
        """Check if point q lies on segment p-r (assuming collinear)."""
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment (p1,p2) intersects segment (p3,p4)."""
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)

        # Standard case: segments straddle each other
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        # Collinear cases
        if d1 == 0 and on_segment(p3, p1, p4):
            return True
        if d2 == 0 and on_segment(p3, p2, p4):
            return True
        if d3 == 0 and on_segment(p1, p3, p2):
            return True
        if d4 == 0 and on_segment(p1, p4, p2):
            return True

        return False

    tests = [
        ((0, 0), (4, 4), (0, 4), (4, 0), True),   # X-shaped intersection
        ((0, 0), (1, 1), (2, 2), (3, 3), False),   # parallel, no overlap
        ((0, 0), (2, 2), (1, 1), (3, 3), True),    # collinear overlap
        ((0, 0), (1, 0), (0, 1), (1, 1), False),   # parallel, not touching
        ((0, 0), (1, 0), (1, 0), (2, 0), True),    # endpoint touching
    ]

    for p1, p2, p3, p4, expected in tests:
        result = segments_intersect(p1, p2, p3, p4)
        print(f"intersect({p1}-{p2}, {p3}-{p4}) = {result}")
        assert result == expected

    print("All Segment Intersection tests passed!")


# === Exercise 3: Convex Hull (Andrew's Monotone Chain) ===
# Problem: Find the convex hull of a set of points.
# Approach: Sort points, build lower and upper hulls.

def exercise_3():
    """Solution: Andrew's monotone chain algorithm for convex hull."""
    def convex_hull(points):
        """Return convex hull vertices in counter-clockwise order."""
        points = sorted(set(points))
        if len(points) <= 1:
            return points

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Remove last point of each half because it's repeated
        return lower[:-1] + upper[:-1]

    # Test case 1: square
    points = [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]
    hull = convex_hull(points)
    print(f"Points: {points}")
    print(f"Convex hull: {hull}")
    assert len(hull) == 4  # 4 corners of the square
    assert (1, 1) not in hull  # interior point excluded

    # Test case 2: triangle with points on edges
    points = [(0, 0), (4, 0), (2, 3), (1, 0), (2, 0), (3, 0)]
    hull = convex_hull(points)
    print(f"\nPoints: {points}")
    print(f"Convex hull: {hull}")
    assert len(hull) == 3  # only the 3 triangle vertices

    print("All Convex Hull tests passed!")


# === Exercise 4: Polygon Area (Shoelace Formula) ===
# Problem: Compute the area of a polygon given its vertices.

def exercise_4():
    """Solution: Shoelace formula for polygon area."""
    def polygon_area(vertices):
        """
        Compute the area of a polygon given vertices in order.
        Uses the Shoelace formula: 2*Area = |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
        """
        n = len(vertices)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2

    # Test case 1: unit square
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    area = polygon_area(square)
    print(f"Square area: {area}")
    assert abs(area - 1.0) < 1e-9

    # Test case 2: right triangle
    triangle = [(0, 0), (4, 0), (0, 3)]
    area = polygon_area(triangle)
    print(f"Triangle area: {area}")
    assert abs(area - 6.0) < 1e-9

    # Test case 3: irregular polygon
    polygon = [(0, 0), (4, 0), (4, 3), (2, 5), (0, 3)]
    area = polygon_area(polygon)
    print(f"Irregular polygon area: {area}")
    assert abs(area - 16.0) < 1e-9  # calculated by hand

    print("All Polygon Area tests passed!")


# === Exercise 5: Closest Pair of Points ===
# Problem: Find the minimum distance between any two points.
# Approach: Divide and conquer for O(n log n), but here we show both
#   brute force and D&C approaches.

def exercise_5():
    """Solution: Closest pair using divide and conquer."""
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def closest_pair(points):
        """Find minimum distance between any two points in O(n log n)."""
        def closest_rec(pts_x, pts_y):
            n = len(pts_x)
            if n <= 3:
                # Brute force for small sets
                min_d = float('inf')
                for i in range(n):
                    for j in range(i + 1, n):
                        d = dist(pts_x[i], pts_x[j])
                        if d < min_d:
                            min_d = d
                return min_d

            mid = n // 2
            mid_point = pts_x[mid]

            # Split points sorted by y into left and right halves
            pts_yl = [p for p in pts_y if p[0] < mid_point[0] or
                       (p[0] == mid_point[0] and p[1] <= mid_point[1])]
            pts_yr = [p for p in pts_y if p[0] > mid_point[0] or
                       (p[0] == mid_point[0] and p[1] > mid_point[1])]

            # Handle tie-breaking for exact mid_point split
            while len(pts_yl) > mid:
                pts_yr.insert(0, pts_yl.pop())
            while len(pts_yl) < mid:
                pts_yl.append(pts_yr.pop(0))

            dl = closest_rec(pts_x[:mid], pts_yl)
            dr = closest_rec(pts_x[mid:], pts_yr)
            d = min(dl, dr)

            # Check strip
            strip = [p for p in pts_y if abs(p[0] - mid_point[0]) < d]

            for i in range(len(strip)):
                j = i + 1
                while j < len(strip) and (strip[j][1] - strip[i][1]) < d:
                    d = min(d, dist(strip[i], strip[j]))
                    j += 1

            return d

        pts_x = sorted(points)
        pts_y = sorted(points, key=lambda p: p[1])
        return closest_rec(pts_x, pts_y)

    # Simpler brute force for verification
    def closest_brute(points):
        min_d = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                d = dist(points[i], points[j])
                if d < min_d:
                    min_d = d
        return min_d

    # Test case 1
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    result = closest_pair(points)
    expected = closest_brute(points)
    print(f"Closest pair distance: {result:.4f} (brute: {expected:.4f})")
    assert abs(result - expected) < 1e-9

    # Test case 2
    points = [(0, 0), (1, 0), (0, 1), (1, 1)]
    result = closest_pair(points)
    print(f"Square corners closest: {result:.4f}")
    assert abs(result - 1.0) < 1e-9

    print("All Closest Pair tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: CCW Test ===")
    exercise_1()
    print("\n=== Exercise 2: Line Segment Intersection ===")
    exercise_2()
    print("\n=== Exercise 3: Convex Hull ===")
    exercise_3()
    print("\n=== Exercise 4: Polygon Area ===")
    exercise_4()
    print("\n=== Exercise 5: Closest Pair of Points ===")
    exercise_5()
    print("\nAll exercises completed!")
