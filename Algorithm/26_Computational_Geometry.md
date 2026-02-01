# 27. 기하 알고리즘 (Computational Geometry)

## 학습 목표
- CCW(Counter Clockwise) 알고리즘 이해
- 선분 교차 판정 구현
- 볼록 껍질(Convex Hull) 알고리즘
- 다각형 넓이 계산
- 가장 가까운 두 점 문제

## 1. 기본 개념

### 점과 벡터

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
        """내적 (Dot Product)"""
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        """외적 (Cross Product) - 2D에서는 스칼라"""
        return self.x * other.y - self.y * other.x

    def norm(self):
        """벡터의 크기"""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        """단위 벡터"""
        n = self.norm()
        return Point(self.x / n, self.y / n) if n > 0 else Point(0, 0)

    def __repr__(self):
        return f"({self.x}, {self.y})"
```

### 기본 연산

```
내적 (Dot Product):
A · B = |A||B|cos(θ)
      = Ax*Bx + Ay*By

- > 0: 예각
- = 0: 직각
- < 0: 둔각

외적 (Cross Product):
A × B = |A||B|sin(θ)
      = Ax*By - Ay*Bx

- > 0: B가 A의 반시계 방향
- = 0: 평행 (일직선)
- < 0: B가 A의 시계 방향
```

---

## 2. CCW (Counter Clockwise)

### 개념

세 점 A, B, C의 방향 관계 판별

```
      C                    C
     /                      \
    /                        \
   A -----> B          A -----> B

  CCW > 0 (반시계)    CCW < 0 (시계)

       C
       |
  A ---+--- B

  CCW = 0 (일직선)
```

### 구현

```python
def ccw(a, b, c):
    """
    세 점의 방향 판별
    Returns:
        > 0: 반시계 방향
        = 0: 일직선
        < 0: 시계 방향
    """
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

def ccw_sign(a, b, c):
    """CCW의 부호만 반환 (-1, 0, 1)"""
    result = ccw(a, b, c)
    if result > 0:
        return 1
    elif result < 0:
        return -1
    return 0

# 튜플 버전
def ccw_tuple(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
```

### CCW 활용

```python
def is_left_turn(a, b, c):
    """A→B→C가 좌회전인가?"""
    return ccw(a, b, c) > 0

def is_collinear(a, b, c):
    """세 점이 일직선상에 있는가?"""
    return ccw(a, b, c) == 0

def angle_direction(origin, p1, p2):
    """origin에서 볼 때 p1→p2 방향"""
    return ccw_sign(origin, p1, p2)
```

---

## 3. 선분 교차 판정

### 알고리즘

두 선분 AB와 CD가 교차하는지 판별

```
교차 조건:
1. CCW(A,B,C) * CCW(A,B,D) < 0  (C와 D가 AB 기준 반대편)
2. CCW(C,D,A) * CCW(C,D,B) < 0  (A와 B가 CD 기준 반대편)

예외: 일직선상일 때 (CCW = 0)
```

### 구현

```python
def segments_intersect(a, b, c, d):
    """
    선분 AB와 선분 CD가 교차하는지 판별
    """
    d1 = ccw_sign(a, b, c)
    d2 = ccw_sign(a, b, d)
    d3 = ccw_sign(c, d, a)
    d4 = ccw_sign(c, d, b)

    # 일반적인 교차
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    # 일직선상에 있는 경우
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
    """점 P가 선분 AB 위에 있는지 (일직선 가정)"""
    return (min(a.x, b.x) <= p.x <= max(a.x, b.x) and
            min(a.y, b.y) <= p.y <= max(a.y, b.y))
```

### 교차점 계산

```python
def intersection_point(a, b, c, d):
    """
    선분 AB와 CD의 교차점 계산
    교차하지 않으면 None 반환
    """
    denom = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x)

    if abs(denom) < 1e-10:  # 평행
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

## 4. 볼록 껍질 (Convex Hull)

### 개념

주어진 점들을 모두 포함하는 가장 작은 볼록 다각형

```
    *           *
      *       *
    *   HULL    *
      *       *
    *           *

점들을 감싸는 고무줄을 생각하면 됨
```

### Graham Scan 알고리즘

```python
def convex_hull_graham(points):
    """
    Graham Scan: O(n log n)
    점들의 볼록 껍질을 반시계 방향으로 반환
    """
    n = len(points)
    if n < 3:
        return points[:]

    # 가장 아래 왼쪽 점 찾기
    start = min(range(n), key=lambda i: (points[i].y, points[i].x))

    # 각도 기준 정렬
    def polar_angle(p):
        return math.atan2(p.y - points[start].y, p.x - points[start].x)

    def dist_sq(p):
        return (p.x - points[start].x)**2 + (p.y - points[start].y)**2

    sorted_points = sorted(range(n), key=lambda i: (polar_angle(points[i]), dist_sq(points[i])))

    # 스택으로 볼록 껍질 구성
    hull = []
    for i in sorted_points:
        while len(hull) >= 2 and ccw(points[hull[-2]], points[hull[-1]], points[i]) <= 0:
            hull.pop()
        hull.append(i)

    return [points[i] for i in hull]
```

### Andrew's Monotone Chain

```python
def convex_hull_andrew(points):
    """
    Andrew's Monotone Chain: O(n log n)
    더 간단하고 안정적
    """
    points = sorted(points, key=lambda p: (p.x, p.y))
    n = len(points)

    if n < 3:
        return points[:]

    # 아래 껍질 (Lower Hull)
    lower = []
    for p in points:
        while len(lower) >= 2 and ccw(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # 위 껍질 (Upper Hull)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and ccw(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # 합치기 (시작점과 끝점은 중복)
    return lower[:-1] + upper[:-1]
```

### C++ 구현

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

## 5. 다각형 넓이

### Shoelace 공식

```python
def polygon_area(points):
    """
    다각형 넓이 (Shoelace Formula)
    점들은 순서대로 (시계/반시계) 주어져야 함
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
    부호 있는 넓이
    > 0: 반시계 방향
    < 0: 시계 방향
    """
    n = len(points)
    area = 0

    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y

    return area / 2
```

### 삼각형 넓이

```python
def triangle_area(a, b, c):
    """CCW를 이용한 삼각형 넓이"""
    return abs(ccw(a, b, c)) / 2
```

---

## 6. 점과 다각형

### 점이 다각형 내부에 있는지

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

        # 점에서 오른쪽으로 쏜 반직선이 변과 교차하는지
        if (pi.y <= point.y < pj.y) or (pj.y <= point.y < pi.y):
            # 교차점의 x좌표
            x_intersect = (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x
            if point.x < x_intersect:
                count += 1

    return count % 2 == 1  # 홀수면 내부

def point_in_convex_polygon(point, polygon):
    """
    볼록 다각형에서 점 포함 여부 (O(log n))
    polygon은 반시계 방향으로 정렬되어 있어야 함
    """
    n = len(polygon)

    # 첫 번째 점 기준으로 모든 점이 같은 방향이어야 함
    if ccw(polygon[0], polygon[1], point) < 0:
        return False
    if ccw(polygon[0], polygon[n-1], point) > 0:
        return False

    # 이진 탐색
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

## 7. 가장 가까운 두 점

### 분할 정복

```python
def closest_pair(points):
    """
    가장 가까운 두 점 찾기: O(n log n)
    """
    points_sorted_x = sorted(points, key=lambda p: (p.x, p.y))
    points_sorted_y = sorted(points, key=lambda p: (p.y, p.x))

    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def closest_util(pts_x, pts_y):
        n = len(pts_x)

        # 기저 사례
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

        # 분할
        mid = n // 2
        mid_point = pts_x[mid]

        # y 기준 정렬된 점들도 분할
        left_y = [p for p in pts_y if p.x < mid_point.x or (p.x == mid_point.x and p.y <= mid_point.y)]
        right_y = [p for p in pts_y if p.x > mid_point.x or (p.x == mid_point.x and p.y > mid_point.y)]

        # 재귀
        dl, pair_l = closest_util(pts_x[:mid], left_y)
        dr, pair_r = closest_util(pts_x[mid:], right_y)

        if dl <= dr:
            d = dl
            pair = pair_l
        else:
            d = dr
            pair = pair_r

        # 중앙 띠에서 확인
        strip = [p for p in pts_y if abs(p.x - mid_point.x) < d]

        for i in range(len(strip)):
            for j in range(i + 1, min(i + 7, len(strip))):  # 최대 6개만 확인
                dist = distance(strip[i], strip[j])
                if dist < d:
                    d = dist
                    pair = (strip[i], strip[j])

        return d, pair

    return closest_util(points_sorted_x, points_sorted_y)

# 사용 예시
points = [Point(2, 3), Point(12, 30), Point(40, 50),
          Point(5, 1), Point(12, 10), Point(3, 4)]
dist, (p1, p2) = closest_pair(points)
print(f"최소 거리: {dist:.4f}")
print(f"점: {p1}, {p2}")
```

---

## 8. 회전하는 캘리퍼스

### 가장 먼 두 점 (Rotating Calipers)

```python
def farthest_pair(points):
    """
    가장 먼 두 점 (볼록 껍질에서만 가능): O(n log n)
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

    # 캘리퍼스 회전
    max_dist = 0
    pair = (hull[0], hull[1])
    j = 1

    for i in range(n):
        # j를 반시계 방향으로 회전
        while True:
            next_j = (j + 1) % n
            # 벡터 비교로 회전 방향 결정
            edge = Point(hull[(i+1) % n].x - hull[i].x, hull[(i+1) % n].y - hull[i].y)
            diag = Point(hull[next_j].x - hull[j].x, hull[next_j].y - hull[j].y)

            if edge.cross(diag) <= 0:
                break
            j = next_j

        d = dist_sq(hull[i], hull[j])
        if d > max_dist:
            max_dist = d
            pair = (hull[i], hull[j])

        # 다음 점도 확인
        d = dist_sq(hull[(i+1) % n], hull[j])
        if d > max_dist:
            max_dist = d
            pair = (hull[(i+1) % n], hull[j])

    return math.sqrt(max_dist), pair
```

---

## 9. 실전 문제 패턴

### 패턴 1: 직선 위 점 판별

```python
def is_on_line(a, b, p):
    """점 P가 직선 AB 위에 있는지"""
    return abs(ccw(a, b, p)) < 1e-9

def is_on_segment(a, b, p):
    """점 P가 선분 AB 위에 있는지"""
    return (is_on_line(a, b, p) and
            min(a.x, b.x) <= p.x <= max(a.x, b.x) and
            min(a.y, b.y) <= p.y <= max(a.y, b.y))
```

### 패턴 2: 점과 직선 사이 거리

```python
def point_to_line_dist(a, b, p):
    """점 P에서 직선 AB까지의 거리"""
    # |AP × AB| / |AB|
    ap = Point(p.x - a.x, p.y - a.y)
    ab = Point(b.x - a.x, b.y - a.y)
    return abs(ap.cross(ab)) / ab.norm()

def point_to_segment_dist(a, b, p):
    """점 P에서 선분 AB까지의 거리"""
    ab = Point(b.x - a.x, b.y - a.y)
    ap = Point(p.x - a.x, p.y - a.y)
    bp = Point(p.x - b.x, p.y - b.y)

    # 투영이 선분 밖에 있는 경우
    if ab.dot(ap) < 0:
        return ap.norm()
    if ab.dot(bp) > 0:
        return bp.norm()

    # 투영이 선분 위에 있는 경우
    return abs(ap.cross(ab)) / ab.norm()
```

### 패턴 3: 다각형 중심

```python
def polygon_centroid(points):
    """다각형 무게중심"""
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

### 패턴 4: 볼록 다각형 넓이

```python
def convex_hull_area(points):
    """점들의 볼록 껍질 넓이"""
    hull = convex_hull_andrew(points)
    return polygon_area(hull)
```

---

## 10. 시간 복잡도 정리

| 알고리즘 | 시간 복잡도 |
|---------|------------|
| CCW | O(1) |
| 선분 교차 | O(1) |
| Graham Scan | O(n log n) |
| Andrew's Monotone Chain | O(n log n) |
| 다각형 넓이 | O(n) |
| 점 in 다각형 | O(n) / O(log n) (볼록) |
| 가장 가까운 두 점 | O(n log n) |
| 가장 먼 두 점 | O(n log n) |

---

## 11. 자주 하는 실수

### 실수 1: 부동소수점 비교

```python
# 잘못됨
if ccw(a, b, c) == 0:  # 부동소수점 오차!

# 올바름
EPS = 1e-9
if abs(ccw(a, b, c)) < EPS:
```

### 실수 2: 정수 오버플로

```cpp
// 좌표가 10^6일 때
long long ccw = (long long)(b.x - a.x) * (c.y - a.y)
              - (long long)(b.y - a.y) * (c.x - a.x);
```

### 실수 3: 볼록 껍질 경계 케이스

```python
# 점이 2개 이하일 때
if len(points) < 3:
    return points[:]  # 껍질 불가
```

---

## 12. 연습 문제

| 난이도 | 문제 유형 | 핵심 개념 |
|--------|----------|-----------|
| ★★☆ | CCW 기본 | 세 점 방향 판별 |
| ★★☆ | 선분 교차 | CCW 활용 |
| ★★★ | 볼록 껍질 | Graham/Andrew |
| ★★★ | 다각형 넓이 | Shoelace |
| ★★★★ | 가장 가까운 두 점 | 분할 정복 |

---

## 다음 단계

- [27_Game_Theory.md](./27_Game_Theory.md) - 게임 이론

---

## 학습 점검

1. CCW가 0인 경우의 의미는?
2. 볼록 껍질에서 점을 스택에서 pop하는 조건은?
3. 선분 교차 판정에서 CCW만으로 부족한 경우는?
4. Shoelace 공식이 작동하는 원리는?
