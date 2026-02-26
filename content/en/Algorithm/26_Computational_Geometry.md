# 27. Computational Geometry

## Learning Objectives
- Understanding CCW (Counter Clockwise) algorithm
- Implementing line segment intersection detection
- Convex hull algorithms
- Computing polygon area
- Closest pair of points problem

## 1. Basic Concepts

### Points and Vectors

```python
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)

    def dot(self, other):
        """Dot Product"""
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        """Cross Product - scalar in 2D"""
        return self.x * other.y - self.y * other.x

    def norm(self):
        """Vector magnitude"""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        """Unit vector"""
        n = self.norm()
        return Point(self.x / n, self.y / n) if n > 0 else Point(0, 0)

    def __repr__(self):
        return f"({self.x}, {self.y})"
```

### Basic Operations

```
Dot Product:
A · B = |A||B|cos(θ)
      = Ax*Bx + Ay*By

- > 0: acute angle
- = 0: right angle
- < 0: obtuse angle

Cross Product:
A × B = |A||B|sin(θ)
      = Ax*By - Ay*Bx

- > 0: B is counterclockwise from A
- = 0: parallel (collinear)
- < 0: B is clockwise from A
```

---

## 2. CCW (Counter Clockwise)

### Concept

Determine the orientation of three points A, B, C

```
      C                    C
     /                      \
    /                        \
   A -----> B          A -----> B

  CCW > 0 (counter)    CCW < 0 (clockwise)

       C
       |
  A ---+--- B

  CCW = 0 (collinear)
```

### Implementation

```python
def ccw(a, b, c):
    """
    Determine orientation of three points
    Returns:
        > 0: counterclockwise
        = 0: collinear
        < 0: clockwise
    """
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

def ccw_sign(a, b, c):
    """Return only sign of CCW (-1, 0, 1)"""
    result = ccw(a, b, c)
    if result > 0:
        return 1
    elif result < 0:
        return -1
    return 0

# Tuple version
def ccw_tuple(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
```

### CCW Applications

```python
def is_left_turn(a, b, c):
    """Is A→B→C a left turn?"""
    return ccw(a, b, c) > 0

def is_collinear(a, b, c):
    """Are three points collinear?"""
    return ccw(a, b, c) == 0

def angle_direction(origin, p1, p2):
    """Direction from p1 to p2 viewed from origin"""
    return ccw_sign(origin, p1, p2)
```

---

## 3. Line Segment Intersection

### Algorithm

Determine if two line segments AB and CD intersect

```
Intersection conditions:
1. CCW(A,B,C) * CCW(A,B,D) < 0  (C and D on opposite sides of AB)
2. CCW(C,D,A) * CCW(C,D,B) < 0  (A and B on opposite sides of CD)

Exception: when collinear (CCW = 0)
```

### Implementation

```python
def segments_intersect(a, b, c, d):
    """
    Check if line segment AB and CD intersect
    """
    d1 = ccw_sign(a, b, c)
    d2 = ccw_sign(a, b, d)
    d3 = ccw_sign(c, d, a)
    d4 = ccw_sign(c, d, b)

    # General intersection
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    # Collinear cases
    if d1 == 0 and on_segment(a, b, c):
        return True
    if d2 == 0 and on_segment(a, b, d):
        return True
    if d3 == 0 and on_segment(c, d, a):
        return True
    if d4 == 0 and on_segment(c, d, b):
        return True

    return False

def on_segment(a, b, p):
    """Is point P on segment AB (assuming collinear)"""
    return (min(a.x, b.x) <= p.x <= max(a.x, b.x) and
            min(a.y, b.y) <= p.y <= max(a.y, b.y))
```

### Computing Intersection Point

```python
def intersection_point(a, b, c, d):
    """
    Compute intersection point of segments AB and CD
    Returns None if they don't intersect
    """
    denom = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x)

    if abs(denom) < 1e-10:  # Parallel
        return None

    t = ((c.x - a.x) * (d.y - c.y) - (c.y - a.y) * (d.x - c.x)) / denom
    s = ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)) / denom

    if 0 <= t <= 1 and 0 <= s <= 1:
        x = a.x + t * (b.x - a.x)
        y = a.y + t * (b.y - a.y)
        return Point(x, y)

    return None
```

---

## 4. Convex Hull

### Concept

The smallest convex polygon containing all given points

```
    *           *
      *       *
    *   HULL    *
      *       *
    *           *

Think of it as a rubber band wrapped around the points
```

### Graham Scan Algorithm

```python
def convex_hull_graham(points):
    """
    Graham Scan: O(n log n)
    Returns convex hull in counterclockwise order
    """
    n = len(points)
    if n < 3:
        return points[:]

    # Anchor point: bottommost (then leftmost) point is guaranteed on the hull
    # because no point lies to its left and below simultaneously
    start = min(range(n), key=lambda i: (points[i].y, points[i].x))

    # Sort remaining points by polar angle from anchor.
    # This ordering ensures we visit points in CCW sweep order,
    # which is the key invariant that makes the stack-based elimination correct.
    def polar_angle(p):
        return math.atan2(p.y - points[start].y, p.x - points[start].x)

    def dist_sq(p):
        return (p.x - points[start].x)**2 + (p.y - points[start].y)**2

    sorted_points = sorted(range(n), key=lambda i: (polar_angle(points[i]), dist_sq(points[i])))

    # Stack-based hull construction: maintain the invariant that the stack
    # always makes left turns (CCW). If adding a new point would create a right
    # turn or go straight, the previous point cannot be on the convex hull
    # (a concave vertex), so we pop it. Each point is pushed and popped at most
    # once, giving O(n) for this phase.
    hull = []
    for i in sorted_points:
        while len(hull) >= 2 and ccw(points[hull[-2]], points[hull[-1]], points[i]) <= 0:
            hull.pop()  # Remove concave vertex — it is inside the hull
        hull.append(i)

    return [points[i] for i in hull]
```

### Andrew's Monotone Chain

```python
def convex_hull_andrew(points):
    """
    Andrew's Monotone Chain: O(n log n)
    Simpler and more stable than Graham Scan because it uses coordinate sorting
    instead of polar angles, avoiding floating-point atan2 precision issues.
    """
    # Lexicographic sort: leftmost points first, then bottommost on ties.
    # This guarantees we sweep left-to-right for the lower hull and right-to-left for upper.
    points = sorted(points, key=lambda p: (p.x, p.y))
    n = len(points)

    if n < 3:
        return points[:]

    # Lower hull: scan left to right, keeping only left turns.
    # Result is the bottom boundary of the convex hull.
    lower = []
    for p in points:
        while len(lower) >= 2 and ccw(lower[-2], lower[-1], p) <= 0:
            lower.pop()  # Right turn or collinear — not on the lower hull
        lower.append(p)

    # Upper hull: scan right to left, same logic.
    # Together, lower + upper form a closed convex polygon.
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and ccw(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove duplicate endpoints (the leftmost and rightmost points appear in both)
    return lower[:-1] + upper[:-1]
```

### C++ Implementation

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

struct Point {
    ll x, y;
    bool operator<(const Point& o) const {
        return tie(x, y) < tie(o.x, o.y);
    }
};

ll ccw(Point a, Point b, Point c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

vector<Point> convexHull(vector<Point> pts) {
    sort(pts.begin(), pts.end());
    int n = pts.size();

    vector<Point> hull;

    // Lower hull
    for (int i = 0; i < n; i++) {
        while (hull.size() >= 2 && ccw(hull[hull.size()-2], hull[hull.size()-1], pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }

    // Upper hull
    int lower_size = hull.size();
    for (int i = n - 2; i >= 0; i--) {
        while (hull.size() > lower_size && ccw(hull[hull.size()-2], hull[hull.size()-1], pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }

    hull.pop_back();
    return hull;
}
```

---

## 5. Polygon Area

### Shoelace Formula

```python
def polygon_area(points):
    """
    Polygon area (Shoelace Formula)
    Points must be given in order (clockwise or counterclockwise)
    """
    n = len(points)
    area = 0

    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y

    return abs(area) / 2

def polygon_area_signed(points):
    """
    Signed area
    > 0: counterclockwise
    < 0: clockwise
    """
    n = len(points)
    area = 0

    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y

    return area / 2
```

### Triangle Area

```python
def triangle_area(a, b, c):
    """Triangle area using CCW"""
    return abs(ccw(a, b, c)) / 2
```

---

## 6. Point and Polygon

### Point Inside Polygon

```python
def point_in_polygon(point, polygon):
    """
    Ray Casting Algorithm
    O(n)
    """
    n = len(polygon)
    count = 0

    for i in range(n):
        j = (i + 1) % n
        pi, pj = polygon[i], polygon[j]

        # Does ray from point to right intersect edge
        if (pi.y <= point.y < pj.y) or (pj.y <= point.y < pi.y):
            # x-coordinate of intersection
            x_intersect = (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x
            if point.x < x_intersect:
                count += 1

    return count % 2 == 1  # Odd means inside

def point_in_convex_polygon(point, polygon):
    """
    Point in convex polygon (O(log n))
    polygon must be sorted counterclockwise
    """
    n = len(polygon)

    # All points must be in same direction relative to first point
    if ccw(polygon[0], polygon[1], point) < 0:
        return False
    if ccw(polygon[0], polygon[n-1], point) > 0:
        return False

    # Binary search
    lo, hi = 1, n - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ccw(polygon[0], polygon[mid], point) >= 0:
            lo = mid
        else:
            hi = mid

    return ccw(polygon[lo], polygon[hi], point) >= 0
```

---

## 7. Closest Pair of Points

### Divide and Conquer

```python
def closest_pair(points):
    """
    Find closest pair of points: O(n log n)
    Divide and conquer avoids the O(n²) brute-force by exploiting geometry:
    once we know the best distance d in each half, only points within a 2d-wide
    strip around the dividing line can possibly improve d.
    """
    points_sorted_x = sorted(points, key=lambda p: (p.x, p.y))
    # Pre-sort by y so the strip can be processed in y-order without re-sorting at each level
    points_sorted_y = sorted(points, key=lambda p: (p.y, p.x))

    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def closest_util(pts_x, pts_y):
        n = len(pts_x)

        # Base case: brute force is fine for tiny subproblems
        if n <= 3:
            min_dist = float('inf')
            pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    d = distance(pts_x[i], pts_x[j])
                    if d < min_dist:
                        min_dist = d
                        pair = (pts_x[i], pts_x[j])
            return min_dist, pair

        # Divide at the median x-coordinate
        mid = n // 2
        mid_point = pts_x[mid]

        # Split y-sorted list into left/right halves maintaining y-order —
        # this avoids re-sorting at each recursion level, keeping total work O(n log n)
        left_y = [p for p in pts_y if p.x < mid_point.x or (p.x == mid_point.x and p.y <= mid_point.y)]
        right_y = [p for p in pts_y if p.x > mid_point.x or (p.x == mid_point.x and p.y > mid_point.y)]

        dl, pair_l = closest_util(pts_x[:mid], left_y)
        dr, pair_r = closest_util(pts_x[mid:], right_y)

        if dl <= dr:
            d = dl
            pair = pair_l
        else:
            d = dr
            pair = pair_r

        # Check the strip of width 2d around the dividing line.
        # By geometry, each point in the strip has at most 7 other points within
        # distance d that could be closer — so the inner loop is O(1) per point.
        strip = [p for p in pts_y if abs(p.x - mid_point.x) < d]

        for i in range(len(strip)):
            for j in range(i + 1, min(i + 7, len(strip))):  # At most 7 candidates per point
                dist = distance(strip[i], strip[j])
                if dist < d:
                    d = dist
                    pair = (strip[i], strip[j])

        return d, pair

    return closest_util(points_sorted_x, points_sorted_y)

# Usage example
points = [Point(2, 3), Point(12, 30), Point(40, 50),
          Point(5, 1), Point(12, 10), Point(3, 4)]
dist, (p1, p2) = closest_pair(points)
print(f"Minimum distance: {dist:.4f}")
print(f"Points: {p1}, {p2}")
```

---

## 8. Rotating Calipers

### Farthest Pair

```python
def farthest_pair(points):
    """
    Farthest pair (only possible in convex hull): O(n log n)
    """
    hull = convex_hull_andrew(points)
    n = len(hull)

    if n == 1:
        return 0, (hull[0], hull[0])
    if n == 2:
        d = math.sqrt((hull[0].x - hull[1].x)**2 + (hull[0].y - hull[1].y)**2)
        return d, (hull[0], hull[1])

    def dist_sq(p1, p2):
        return (p1.x - p2.x)**2 + (p1.y - p2.y)**2

    # Rotate calipers
    max_dist = 0
    pair = (hull[0], hull[1])
    j = 1

    for i in range(n):
        # Rotate j counterclockwise
        while True:
            next_j = (j + 1) % n
            # Compare vectors to decide rotation direction
            edge = Point(hull[(i+1) % n].x - hull[i].x, hull[(i+1) % n].y - hull[i].y)
            diag = Point(hull[next_j].x - hull[j].x, hull[next_j].y - hull[j].y)

            if edge.cross(diag) <= 0:
                break
            j = next_j

        d = dist_sq(hull[i], hull[j])
        if d > max_dist:
            max_dist = d
            pair = (hull[i], hull[j])

        # Check next point too
        d = dist_sq(hull[(i+1) % n], hull[j])
        if d > max_dist:
            max_dist = d
            pair = (hull[(i+1) % n], hull[j])

    return math.sqrt(max_dist), pair
```

---

## 9. Practical Problem Patterns

### Pattern 1: Point on Line Detection

```python
def is_on_line(a, b, p):
    """Is point P on line AB"""
    return abs(ccw(a, b, p)) < 1e-9

def is_on_segment(a, b, p):
    """Is point P on segment AB"""
    return (is_on_line(a, b, p) and
            min(a.x, b.x) <= p.x <= max(a.x, b.x) and
            min(a.y, b.y) <= p.y <= max(a.y, b.y))
```

### Pattern 2: Point to Line Distance

```python
def point_to_line_dist(a, b, p):
    """Distance from point P to line AB"""
    # |AP × AB| / |AB|
    ap = Point(p.x - a.x, p.y - a.y)
    ab = Point(b.x - a.x, b.y - a.y)
    return abs(ap.cross(ab)) / ab.norm()

def point_to_segment_dist(a, b, p):
    """Distance from point P to segment AB"""
    ab = Point(b.x - a.x, b.y - a.y)
    ap = Point(p.x - a.x, p.y - a.y)
    bp = Point(p.x - b.x, p.y - b.y)

    # Projection outside segment
    if ab.dot(ap) < 0:
        return ap.norm()
    if ab.dot(bp) > 0:
        return bp.norm()

    # Projection on segment
    return abs(ap.cross(ab)) / ab.norm()
```

### Pattern 3: Polygon Centroid

```python
def polygon_centroid(points):
    """Polygon center of mass"""
    n = len(points)
    cx, cy = 0, 0
    area = 0

    for i in range(n):
        j = (i + 1) % n
        cross = points[i].x * points[j].y - points[j].x * points[i].y
        cx += (points[i].x + points[j].x) * cross
        cy += (points[i].y + points[j].y) * cross
        area += cross

    area /= 2
    cx /= (6 * area)
    cy /= (6 * area)

    return Point(cx, cy)
```

### Pattern 4: Convex Polygon Area

```python
def convex_hull_area(points):
    """Area of convex hull of points"""
    hull = convex_hull_andrew(points)
    return polygon_area(hull)
```

---

## 10. Time Complexity Summary

| Algorithm | Time Complexity |
|-----------|----------------|
| CCW | O(1) |
| Line intersection | O(1) |
| Graham Scan | O(n log n) |
| Andrew's Monotone Chain | O(n log n) |
| Polygon area | O(n) |
| Point in polygon | O(n) / O(log n) (convex) |
| Closest pair | O(n log n) |
| Farthest pair | O(n log n) |

---

## 11. Common Mistakes

### Mistake 1: Floating Point Comparison

```python
# Incorrect
if ccw(a, b, c) == 0:  # Floating point error!

# Correct
EPS = 1e-9
if abs(ccw(a, b, c)) < EPS:
```

### Mistake 2: Integer Overflow

```cpp
// When coordinates can be 10^6
long long ccw = (long long)(b.x - a.x) * (c.y - a.y)
              - (long long)(b.y - a.y) * (c.x - a.x);
```

### Mistake 3: Convex Hull Edge Cases

```python
# When there are 2 or fewer points
if len(points) < 3:
    return points[:]  # Can't form hull
```

---

## 12. Practice Problems

| Difficulty | Problem Type | Key Concept |
|-----------|--------------|-------------|
| ★★☆ | CCW basics | Three point orientation |
| ★★☆ | Line intersection | CCW application |
| ★★★ | Convex hull | Graham/Andrew |
| ★★★ | Polygon area | Shoelace |
| ★★★★ | Closest pair | Divide and conquer |

---

## Next Steps

- [27_Game_Theory.md](./27_Game_Theory.md) - Game theory

---

## Learning Checklist

1. What does CCW = 0 mean?
2. When to pop points from stack in convex hull?
3. When is CCW alone insufficient for line segment intersection?
4. How does the Shoelace formula work?
