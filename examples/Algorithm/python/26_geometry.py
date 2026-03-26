"""
Computational Geometry
Computational Geometry Algorithms

Algorithms for handling geometric objects such as points, lines, and polygons.
"""

from typing import List, Tuple, Optional
from math import sqrt, atan2, pi, inf
from functools import cmp_to_key


# =============================================================================
# 1. Basic Geometric Operations
# =============================================================================

class Point:
    """2D Point"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)

    def __eq__(self, other: 'Point') -> bool:
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def dot(self, other: 'Point') -> float:
        """Dot product"""
        return self.x * other.x + self.y * other.y

    def cross(self, other: 'Point') -> float:
        """Cross product (z component)"""
        return self.x * other.y - self.y * other.x

    def norm(self) -> float:
        """Vector magnitude"""
        return sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> 'Point':
        """Unit vector"""
        n = self.norm()
        return Point(self.x / n, self.y / n) if n > 0 else Point(0, 0)


def distance(p1: Point, p2: Point) -> float:
    """Distance between two points"""
    return (p2 - p1).norm()


# =============================================================================
# 2. CCW (Counter-Clockwise)
# =============================================================================

def ccw(a: Point, b: Point, c: Point) -> int:
    """
    Orientation of three points
    Returns: 1 (counter-clockwise), -1 (clockwise), 0 (collinear)
    """
    cross = (b - a).cross(c - a)
    if cross > 1e-9:
        return 1   # Counter-clockwise
    elif cross < -1e-9:
        return -1  # Clockwise
    return 0       # Collinear


def ccw_tuple(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> int:
    """Tuple version of CCW"""
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if cross > 1e-9:
        return 1
    elif cross < -1e-9:
        return -1
    return 0


# =============================================================================
# 3. Segment Intersection
# =============================================================================

def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    Check if segments p1-p2 and p3-p4 intersect
    """
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    # General intersection
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    # Boundary case (point lies on segment)
    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return (min(p.x, r.x) <= q.x <= max(p.x, r.x) and
                min(p.y, r.y) <= q.y <= max(p.y, r.y))

    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True

    return False


def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """
    Compute intersection point of two lines
    Line p1-p2 and line p3-p4
    """
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1.cross(d2)

    if abs(cross) < 1e-9:
        return None  # Parallel

    t = (p3 - p1).cross(d2) / cross
    return p1 + d1 * t


# =============================================================================
# 4. Convex Hull
# =============================================================================

def convex_hull_graham(points: List[Point]) -> List[Point]:
    """
    Graham Scan Algorithm
    Time Complexity: O(n log n)
    """
    if len(points) < 3:
        return points[:]

    # Find bottom-most, left-most point
    start = min(points, key=lambda p: (p.y, p.x))

    # Sort by polar angle
    def polar_angle(p: Point) -> float:
        return atan2(p.y - start.y, p.x - start.x)

    def dist_sq(p: Point) -> float:
        return (p.x - start.x) ** 2 + (p.y - start.y) ** 2

    sorted_points = sorted(points, key=lambda p: (polar_angle(p), dist_sq(p)))

    # Build convex hull using stack
    hull = []
    for p in sorted_points:
        while len(hull) >= 2 and ccw(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull


def convex_hull_monotone_chain(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Monotone Chain Algorithm
    Time Complexity: O(n log n)
    """
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    # Lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and ccw_tuple(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and ccw_tuple(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# =============================================================================
# 5. Polygon Operations
# =============================================================================

def polygon_area(vertices: List[Point]) -> float:
    """
    Polygon area (Shoelace formula)
    Vertices must be ordered counter-clockwise
    """
    n = len(vertices)
    if n < 3:
        return 0

    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i].cross(vertices[j])

    return abs(area) / 2


def polygon_perimeter(vertices: List[Point]) -> float:
    """Polygon perimeter"""
    n = len(vertices)
    perimeter = 0
    for i in range(n):
        j = (i + 1) % n
        perimeter += distance(vertices[i], vertices[j])
    return perimeter


def point_in_polygon(point: Point, polygon: List[Point]) -> int:
    """
    Check if point is inside polygon
    Returns: 1 (inside), 0 (on boundary), -1 (outside)
    """
    n = len(polygon)
    winding = 0

    for i in range(n):
        j = (i + 1) % n
        p1, p2 = polygon[i], polygon[j]

        # Check if point is on boundary
        if ccw(p1, p2, point) == 0:
            if (min(p1.x, p2.x) <= point.x <= max(p1.x, p2.x) and
                min(p1.y, p2.y) <= point.y <= max(p1.y, p2.y)):
                return 0

        # Winding number calculation
        if p1.y <= point.y:
            if p2.y > point.y and ccw(p1, p2, point) > 0:
                winding += 1
        else:
            if p2.y <= point.y and ccw(p1, p2, point) < 0:
                winding -= 1

    return 1 if winding != 0 else -1


def is_convex(polygon: List[Point]) -> bool:
    """Check if polygon is convex"""
    n = len(polygon)
    if n < 3:
        return False

    sign = 0
    for i in range(n):
        d = ccw(polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n])
        if d != 0:
            if sign == 0:
                sign = d
            elif sign != d:
                return False

    return True


# =============================================================================
# 6. Point-to-Line/Segment Distance
# =============================================================================

def point_to_line_distance(point: Point, line_p1: Point, line_p2: Point) -> float:
    """Distance from point to line"""
    v = line_p2 - line_p1
    w = point - line_p1
    return abs(v.cross(w)) / v.norm()


def point_to_segment_distance(point: Point, seg_p1: Point, seg_p2: Point) -> float:
    """Distance from point to segment"""
    v = seg_p2 - seg_p1
    w = point - seg_p1

    c1 = w.dot(v)
    if c1 <= 0:
        return distance(point, seg_p1)

    c2 = v.dot(v)
    if c2 <= c1:
        return distance(point, seg_p2)

    t = c1 / c2
    proj = seg_p1 + v * t
    return distance(point, proj)


# =============================================================================
# 7. Closest Pair of Points
# =============================================================================

def closest_pair(points: List[Point]) -> Tuple[Point, Point, float]:
    """
    Find the closest pair of points
    Divide and conquer: O(n log n)
    """
    def brute_force(pts: List[Point]) -> Tuple[Point, Point, float]:
        min_dist = inf
        p1, p2 = None, None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = distance(pts[i], pts[j])
                if d < min_dist:
                    min_dist = d
                    p1, p2 = pts[i], pts[j]
        return p1, p2, min_dist

    def closest_split(pts_y: List[Point], mid_x: float, delta: float) -> Tuple[Point, Point, float]:
        # Only consider points near the dividing line
        strip = [p for p in pts_y if abs(p.x - mid_x) < delta]

        min_dist = delta
        p1, p2 = None, None

        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j].y - strip[i].y < min_dist:
                d = distance(strip[i], strip[j])
                if d < min_dist:
                    min_dist = d
                    p1, p2 = strip[i], strip[j]
                j += 1

        return p1, p2, min_dist

    def divide_conquer(pts_x: List[Point], pts_y: List[Point]) -> Tuple[Point, Point, float]:
        if len(pts_x) <= 3:
            return brute_force(pts_x)

        mid = len(pts_x) // 2
        mid_point = pts_x[mid]

        # Split into left/right
        left_x = pts_x[:mid]
        right_x = pts_x[mid:]

        left_set = set(id(p) for p in left_x)
        left_y = [p for p in pts_y if id(p) in left_set]
        right_y = [p for p in pts_y if id(p) not in left_set]

        # Recursive calls
        l1, l2, left_dist = divide_conquer(left_x, left_y)
        r1, r2, right_dist = divide_conquer(right_x, right_y)

        if left_dist < right_dist:
            best = (l1, l2, left_dist)
        else:
            best = (r1, r2, right_dist)

        # Check near the dividing line
        s1, s2, split_dist = closest_split(pts_y, mid_point.x, best[2])
        if split_dist < best[2]:
            return s1, s2, split_dist

        return best

    if len(points) < 2:
        return None, None, inf

    pts_x = sorted(points, key=lambda p: p.x)
    pts_y = sorted(points, key=lambda p: p.y)

    return divide_conquer(pts_x, pts_y)


# =============================================================================
# 8. Rotating Calipers
# =============================================================================

def rotating_calipers_diameter(hull: List[Point]) -> Tuple[Point, Point, float]:
    """
    Diameter of convex hull (farthest pair of points)
    Time Complexity: O(n)
    """
    n = len(hull)
    if n < 2:
        return None, None, 0
    if n == 2:
        return hull[0], hull[1], distance(hull[0], hull[1])

    # Find farthest pair
    max_dist = 0
    p1, p2 = None, None

    j = 1
    for i in range(n):
        # Find farthest point from edge hull[i]-hull[(i+1)%n]
        while True:
            next_j = (j + 1) % n
            # Compare triangle areas
            area1 = abs((hull[(i + 1) % n] - hull[i]).cross(hull[j] - hull[i]))
            area2 = abs((hull[(i + 1) % n] - hull[i]).cross(hull[next_j] - hull[i]))
            if area2 > area1:
                j = next_j
            else:
                break

        d1 = distance(hull[i], hull[j])
        d2 = distance(hull[(i + 1) % n], hull[j])

        if d1 > max_dist:
            max_dist = d1
            p1, p2 = hull[i], hull[j]
        if d2 > max_dist:
            max_dist = d2
            p1, p2 = hull[(i + 1) % n], hull[j]

    return p1, p2, max_dist


# =============================================================================
# 9. Half-Plane Intersection
# =============================================================================

class HalfPlane:
    """Half-plane: ax + by + c >= 0"""
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c
        self.angle = atan2(b, a)

    @classmethod
    def from_points(cls, p1: Point, p2: Point) -> 'HalfPlane':
        """Left half-plane of direction p1 -> p2"""
        a = p2.y - p1.y
        b = p1.x - p2.x
        c = -a * p1.x - b * p1.y
        return cls(a, b, c)

    def side(self, p: Point) -> float:
        """Which side the point is on (positive: inside, negative: outside)"""
        return self.a * p.x + self.b * p.y + self.c


def half_plane_intersection_point(h1: HalfPlane, h2: HalfPlane) -> Optional[Point]:
    """Intersection point of two half-plane boundaries"""
    det = h1.a * h2.b - h2.a * h1.b
    if abs(det) < 1e-9:
        return None
    x = (h1.b * h2.c - h2.b * h1.c) / det
    y = (h2.a * h1.c - h1.a * h2.c) / det
    return Point(x, y)


# =============================================================================
# 10. Practical Problem: Triangle Area
# =============================================================================

def triangle_area(p1: Point, p2: Point, p3: Point) -> float:
    """Triangle area"""
    return abs((p2 - p1).cross(p3 - p1)) / 2


def triangle_circumcircle(p1: Point, p2: Point, p3: Point) -> Tuple[Point, float]:
    """Circumscribed circle of a triangle (center, radius)"""
    ax, ay = p1.x, p1.y
    bx, by = p2.x, p2.y
    cx, cy = p3.x, p3.y

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None, 0

    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d

    center = Point(ux, uy)
    radius = distance(center, p1)

    return center, radius


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Computational Geometry Examples")
    print("=" * 60)

    # 1. CCW
    print("\n[1] CCW (Orientation)")
    a, b, c = Point(0, 0), Point(4, 0), Point(2, 2)
    d = Point(2, -2)
    print(f"    A={a}, B={b}, C={c}")
    print(f"    CCW(A,B,C) = {ccw(a, b, c)} (counter-clockwise)")
    print(f"    CCW(A,B,D) = {ccw(a, b, d)} (clockwise, D={d})")

    # 2. Segment Intersection
    print("\n[2] Segment Intersection")
    p1, p2 = Point(0, 0), Point(4, 4)
    p3, p4 = Point(0, 4), Point(4, 0)
    print(f"    Segment 1: {p1}-{p2}")
    print(f"    Segment 2: {p3}-{p4}")
    print(f"    Intersects: {segments_intersect(p1, p2, p3, p4)}")

    intersection = line_intersection(p1, p2, p3, p4)
    print(f"    Intersection point: {intersection}")

    # 3. Convex Hull
    print("\n[3] Convex Hull")
    points = [Point(0, 0), Point(1, 1), Point(2, 2), Point(0, 2),
              Point(2, 0), Point(1, 0), Point(0, 1), Point(2, 1)]
    hull = convex_hull_graham(points)
    print(f"    Points: {points}")
    print(f"    Convex hull: {hull}")

    # Tuple version
    pts_tuple = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0), (0, 1), (2, 1)]
    hull_tuple = convex_hull_monotone_chain(pts_tuple)
    print(f"    Monotone Chain: {hull_tuple}")

    # 4. Polygon Operations
    print("\n[4] Polygon Operations")
    polygon = [Point(0, 0), Point(4, 0), Point(4, 3), Point(0, 3)]
    print(f"    Polygon: {polygon}")
    print(f"    Area: {polygon_area(polygon)}")
    print(f"    Perimeter: {polygon_perimeter(polygon)}")
    print(f"    Convex: {is_convex(polygon)}")

    test_point = Point(2, 1)
    print(f"    Point {test_point} location: {point_in_polygon(test_point, polygon)} (1=inside)")

    # 5. Point-to-Segment Distance
    print("\n[5] Point-to-Segment Distance")
    point = Point(2, 3)
    seg_start = Point(0, 0)
    seg_end = Point(4, 0)
    dist = point_to_segment_distance(point, seg_start, seg_end)
    print(f"    Point: {point}")
    print(f"    Segment: {seg_start}-{seg_end}")
    print(f"    Distance: {dist}")

    # 6. Closest Pair
    print("\n[6] Closest Pair of Points")
    rand_points = [Point(2, 3), Point(12, 30), Point(40, 50),
                   Point(5, 1), Point(12, 10), Point(3, 4)]
    p1, p2, dist = closest_pair(rand_points)
    print(f"    Points: {rand_points}")
    print(f"    Closest pair: {p1}, {p2}")
    print(f"    Distance: {dist:.4f}")

    # 7. Rotating Calipers
    print("\n[7] Convex Hull Diameter (Rotating Calipers)")
    hull_for_diameter = [Point(0, 0), Point(4, 0), Point(5, 2), Point(3, 4), Point(1, 3)]
    p1, p2, diam = rotating_calipers_diameter(hull_for_diameter)
    print(f"    Convex hull: {hull_for_diameter}")
    print(f"    Farthest pair: {p1}, {p2}")
    print(f"    Diameter: {diam:.4f}")

    # 8. Triangle
    print("\n[8] Triangle Operations")
    t1, t2, t3 = Point(0, 0), Point(4, 0), Point(2, 3)
    print(f"    Triangle: {t1}, {t2}, {t3}")
    print(f"    Area: {triangle_area(t1, t2, t3)}")
    center, radius = triangle_circumcircle(t1, t2, t3)
    print(f"    Circumcircle center: {center}")
    print(f"    Circumcircle radius: {radius:.4f}")

    # 9. Algorithm Complexity
    print("\n[9] Algorithm Complexity")
    print("    | Algorithm          | Time Complexity |")
    print("    |--------------------|-----------------|")
    print("    | CCW                | O(1)            |")
    print("    | Segment Intersect  | O(1)            |")
    print("    | Convex Hull        | O(n log n)      |")
    print("    | Point in Polygon   | O(n)            |")
    print("    | Closest Pair       | O(n log n)      |")
    print("    | Rotating Calipers  | O(n)            |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
