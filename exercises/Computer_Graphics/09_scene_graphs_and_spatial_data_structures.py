"""
Exercises for Lesson 09: Scene Graphs and Spatial Data Structures
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_translation(tx, ty, tz):
    """Create a 4x4 translation matrix."""
    M = np.eye(4)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M


def make_rotation_y(angle_deg):
    """Create a 4x4 rotation matrix around the Y axis."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    M = np.eye(4)
    M[0, 0] = c;  M[0, 2] = s
    M[2, 0] = -s; M[2, 2] = c
    return M


# ---------------------------------------------------------------------------
# Exercise 1 -- Scene Graph Transform (manual + verification)
# ---------------------------------------------------------------------------

def exercise_1():
    """
    Given a scene graph where node A has translation (3,0,0) and rotation 45
    degrees around Y, and child node B has translation (2,0,0), compute B's
    world position by hand and verify programmatically.

    Hand derivation:
      M_A = R_y(45) * T(3,0,0)
      M_B_local = T(2,0,0)
      M_B_world = M_A * M_B_local

      R_y(45) rotates the X axis toward Z:
        cos45 ~ 0.7071, sin45 ~ 0.7071

      After R_y(45):
        x' = cos45*x + sin45*z
        z' = -sin45*x + cos45*z

      M_A applied to origin of A:
        T(3,0,0) moves to (3,0,0),
        R_y(45) rotates that: (3*cos45, 0, -3*sin45) = (2.121, 0, -2.121)

      B in A's local frame is at (2,0,0).
      B_world = M_A * [2,0,0,1]
        = R_y(45) * T(3,0,0) * [2,0,0,1]
        = R_y(45) * [5,0,0,1]
        = (5*cos45, 0, -5*sin45, 1)
        = (3.536, 0, -3.536)
    """
    # Build transforms
    T_A = make_translation(3, 0, 0)
    R_A = make_rotation_y(45)
    M_A = R_A @ T_A  # rotation applied after translation

    T_B_local = make_translation(2, 0, 0)
    M_B_world = M_A @ T_B_local

    # B's world position is the translation column of M_B_world
    B_world = M_B_world[:3, 3]
    print(f"  Node A world transform (rotation 45 deg Y + translate (3,0,0)):")
    print(f"    A origin in world: ({M_A[0,3]:.3f}, {M_A[1,3]:.3f}, {M_A[2,3]:.3f})")
    print(f"  Node B (child of A, translate (2,0,0)):")
    print(f"    B world position:  ({B_world[0]:.3f}, {B_world[1]:.3f}, {B_world[2]:.3f})")
    print(f"    Expected ~(3.536, 0.000, -3.536)")


# ---------------------------------------------------------------------------
# Exercise 2 -- AABB Construction
# ---------------------------------------------------------------------------

def exercise_2():
    """
    Given 5 triangles with specified vertices, compute the AABB for each
    triangle, then compute the AABB that encloses all five.
    """
    np.random.seed(42)
    # Generate 5 triangles as (3, 3) arrays -- 3 vertices, 3 coords each
    triangles = [np.random.uniform(-5, 5, (3, 3)) for _ in range(5)]

    per_tri_aabbs = []
    for i, tri in enumerate(triangles):
        aabb_min = tri.min(axis=0)
        aabb_max = tri.max(axis=0)
        per_tri_aabbs.append((aabb_min, aabb_max))
        print(f"  Triangle {i}: min=({aabb_min[0]:.2f}, {aabb_min[1]:.2f}, {aabb_min[2]:.2f})"
              f"  max=({aabb_max[0]:.2f}, {aabb_max[1]:.2f}, {aabb_max[2]:.2f})")

    # Enclosing AABB
    all_min = np.min([a[0] for a in per_tri_aabbs], axis=0)
    all_max = np.max([a[1] for a in per_tri_aabbs], axis=0)
    print(f"  Enclosing AABB: min=({all_min[0]:.2f}, {all_min[1]:.2f}, {all_min[2]:.2f})"
          f"  max=({all_max[0]:.2f}, {all_max[1]:.2f}, {all_max[2]:.2f})")


# ---------------------------------------------------------------------------
# Exercise 3 -- BVH Split Comparison (SAH vs Midpoint)
# ---------------------------------------------------------------------------

class AABB:
    """Axis-Aligned Bounding Box."""
    def __init__(self, min_pt, max_pt):
        self.min_pt = np.asarray(min_pt, dtype=float)
        self.max_pt = np.asarray(max_pt, dtype=float)

    def surface_area(self):
        d = self.max_pt - self.min_pt
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

    @staticmethod
    def union(a, b):
        return AABB(np.minimum(a.min_pt, b.min_pt),
                    np.maximum(a.max_pt, b.max_pt))

    def intersect_ray(self, origin, inv_dir):
        t1 = (self.min_pt - origin) * inv_dir
        t2 = (self.max_pt - origin) * inv_dir
        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)
        t_enter = np.max(t_min)
        t_exit = np.min(t_max)
        hit = (t_enter <= t_exit) and (t_exit >= 0.0)
        return hit, t_enter


class SimpleSphere:
    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def aabb(self):
        r = np.full(3, self.radius)
        return AABB(self.center - r, self.center + r)

    def intersect_ray(self, origin, direction):
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return False, float('inf')
        sq = np.sqrt(disc)
        t = (-b - sq) / (2 * a)
        if t < 1e-4:
            t = (-b + sq) / (2 * a)
        if t < 1e-4:
            return False, float('inf')
        return True, t


class BVHNode:
    def __init__(self, aabb, left=None, right=None, primitives=None):
        self.aabb = aabb
        self.left = left
        self.right = right
        self.primitives = primitives or []

    @property
    def is_leaf(self):
        return len(self.primitives) > 0


def build_bvh_sah(primitives, max_leaf=2):
    """Build BVH using Surface Area Heuristic."""
    aabbs = [p.aabb() for p in primitives]
    total = aabbs[0]
    for b in aabbs[1:]:
        total = AABB.union(total, b)

    if len(primitives) <= max_leaf:
        return BVHNode(total, primitives=primitives)

    best_cost = float('inf')
    best_axis = 0
    best_idx = len(primitives) // 2

    for axis in range(3):
        centers = [p.center[axis] for p in primitives]
        order = np.argsort(centers)
        sorted_p = [primitives[i] for i in order]
        sorted_a = [aabbs[i] for i in order]
        n = len(sorted_p)

        left_a = [None] * n
        right_a = [None] * n
        left_a[0] = sorted_a[0]
        for i in range(1, n):
            left_a[i] = AABB.union(left_a[i-1], sorted_a[i])
        right_a[-1] = sorted_a[-1]
        for i in range(n-2, -1, -1):
            right_a[i] = AABB.union(right_a[i+1], sorted_a[i])

        parent_sa = total.surface_area()
        for i in range(1, n):
            cost = 1.0 + (left_a[i-1].surface_area() / parent_sa) * i * 4.0 \
                       + (right_a[i].surface_area() / parent_sa) * (n - i) * 4.0
            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_idx = i

    centers = [p.center[best_axis] for p in primitives]
    order = np.argsort(centers)
    sorted_p = [primitives[i] for i in order]
    left_p = sorted_p[:best_idx]
    right_p = sorted_p[best_idx:]
    if not left_p or not right_p:
        mid = len(sorted_p) // 2
        left_p = sorted_p[:mid]
        right_p = sorted_p[mid:]

    return BVHNode(total,
                   left=build_bvh_sah(left_p, max_leaf),
                   right=build_bvh_sah(right_p, max_leaf))


def build_bvh_midpoint(primitives, max_leaf=2):
    """Build BVH using midpoint splitting on the longest axis."""
    aabbs = [p.aabb() for p in primitives]
    total = aabbs[0]
    for b in aabbs[1:]:
        total = AABB.union(total, b)

    if len(primitives) <= max_leaf:
        return BVHNode(total, primitives=primitives)

    d = total.max_pt - total.min_pt
    axis = int(np.argmax(d))
    mid = (total.min_pt[axis] + total.max_pt[axis]) / 2.0
    left_p = [p for p in primitives if p.center[axis] < mid]
    right_p = [p for p in primitives if p.center[axis] >= mid]

    if not left_p or not right_p:
        centers = [p.center[axis] for p in primitives]
        order = np.argsort(centers)
        sorted_p = [primitives[i] for i in order]
        half = len(sorted_p) // 2
        left_p = sorted_p[:half]
        right_p = sorted_p[half:]

    return BVHNode(total,
                   left=build_bvh_midpoint(left_p, max_leaf),
                   right=build_bvh_midpoint(right_p, max_leaf))


_aabb_test_count = 0


def bvh_intersect(node, origin, direction, inv_dir):
    """Traverse BVH; counts AABB tests for comparison."""
    global _aabb_test_count
    _aabb_test_count += 1
    hit_box, _ = node.aabb.intersect_ray(origin, inv_dir)
    if not hit_box:
        return False, float('inf'), None
    if node.is_leaf:
        best_t = float('inf')
        best_p = None
        for p in node.primitives:
            h, t = p.intersect_ray(origin, direction)
            if h and t < best_t:
                best_t = t
                best_p = p
        return best_p is not None, best_t, best_p
    h_l, t_l, p_l = bvh_intersect(node.left, origin, direction, inv_dir)
    h_r, t_r, p_r = bvh_intersect(node.right, origin, direction, inv_dir)
    if h_l and h_r:
        return (True, t_l, p_l) if t_l <= t_r else (True, t_r, p_r)
    if h_l:
        return True, t_l, p_l
    if h_r:
        return True, t_r, p_r
    return False, float('inf'), None


def exercise_3():
    """
    Modify BVH code to use midpoint splitting instead of SAH. Generate 1000
    random spheres and compare the average number of AABB intersection tests
    per ray between SAH and midpoint BVH.
    """
    global _aabb_test_count
    np.random.seed(123)
    spheres = [SimpleSphere(center=np.random.uniform(-10, 10, 3),
                            radius=np.random.uniform(0.2, 0.8))
               for _ in range(1000)]

    bvh_sah = build_bvh_sah(spheres, max_leaf=4)
    bvh_mid = build_bvh_midpoint(spheres, max_leaf=4)

    num_rays = 500
    total_tests_sah = 0
    total_tests_mid = 0

    for _ in range(num_rays):
        origin = np.array([0.0, 0.0, 15.0])
        target = np.random.uniform(-10, 10, 3)
        direction = target - origin
        direction /= np.linalg.norm(direction)
        inv_dir = np.where(np.abs(direction) > 1e-10, 1.0 / direction, 1e10)

        _aabb_test_count = 0
        bvh_intersect(bvh_sah, origin, direction, inv_dir)
        total_tests_sah += _aabb_test_count

        _aabb_test_count = 0
        bvh_intersect(bvh_mid, origin, direction, inv_dir)
        total_tests_mid += _aabb_test_count

    avg_sah = total_tests_sah / num_rays
    avg_mid = total_tests_mid / num_rays
    print(f"  1000 spheres, {num_rays} random rays:")
    print(f"    SAH BVH     -- avg AABB tests/ray: {avg_sah:.1f}")
    print(f"    Midpoint BVH -- avg AABB tests/ray: {avg_mid:.1f}")
    print(f"    SAH improvement: {(avg_mid - avg_sah) / avg_mid * 100:.1f}% fewer tests")


# ---------------------------------------------------------------------------
# Exercise 4 -- Octree Implementation
# ---------------------------------------------------------------------------

class OctreeNode:
    """3D Octree node for spatial partitioning."""
    MAX_OBJECTS = 4
    MAX_DEPTH = 6

    def __init__(self, cx, cy, cz, half, depth=0):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.half = half
        self.depth = depth
        self.objects = []
        self.children = None

    def _contains(self, x, y, z):
        h = self.half
        return (self.cx - h <= x < self.cx + h and
                self.cy - h <= y < self.cy + h and
                self.cz - h <= z < self.cz + h)

    def subdivide(self):
        h = self.half / 2
        d = self.depth + 1
        self.children = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    self.children.append(
                        OctreeNode(self.cx + sx * h,
                                   self.cy + sy * h,
                                   self.cz + sz * h, h, d))

    def insert(self, x, y, z, data=None):
        if not self._contains(x, y, z):
            return False
        if self.children is None:
            self.objects.append((x, y, z, data))
            if len(self.objects) > self.MAX_OBJECTS and self.depth < self.MAX_DEPTH:
                self.subdivide()
                old = self.objects
                self.objects = []
                for ox, oy, oz, od in old:
                    inserted = False
                    for child in self.children:
                        if child.insert(ox, oy, oz, od):
                            inserted = True
                            break
                    if not inserted:
                        self.objects.append((ox, oy, oz, od))
            return True
        for child in self.children:
            if child.insert(x, y, z, data):
                return True
        return False

    def nearest_neighbor(self, qx, qy, qz, best_dist=float('inf'), best_obj=None):
        """Find the nearest object to the query point."""
        # Check objects at this node
        for ox, oy, oz, od in self.objects:
            d = np.sqrt((ox - qx)**2 + (oy - qy)**2 + (oz - qz)**2)
            if d < best_dist:
                best_dist = d
                best_obj = (ox, oy, oz, od)
        # Recurse into children, pruning those too far away
        if self.children:
            # Sort children by distance to query for better pruning
            child_dists = []
            for child in self.children:
                dx = max(0, abs(qx - child.cx) - child.half)
                dy = max(0, abs(qy - child.cy) - child.half)
                dz = max(0, abs(qz - child.cz) - child.half)
                child_dists.append(np.sqrt(dx*dx + dy*dy + dz*dz))
            for idx in np.argsort(child_dists):
                if child_dists[idx] < best_dist:
                    best_dist, best_obj = self.children[idx].nearest_neighbor(
                        qx, qy, qz, best_dist, best_obj)
        return best_dist, best_obj


def exercise_4():
    """
    Extend the quadtree code to 3D (octree). Insert 100 random spheres and
    implement a nearest-neighbor query.
    """
    np.random.seed(7)
    octree = OctreeNode(0, 0, 0, half=50, depth=0)
    points = np.random.uniform(-40, 40, (100, 3))
    for i, (x, y, z) in enumerate(points):
        octree.insert(x, y, z, data=f"sphere_{i}")

    # Query nearest neighbor for a test point
    query = np.array([5.0, 5.0, 5.0])
    dist, obj = octree.nearest_neighbor(query[0], query[1], query[2])

    # Verify with brute force
    dists = np.linalg.norm(points - query, axis=1)
    bf_idx = np.argmin(dists)
    bf_dist = dists[bf_idx]

    print(f"  Inserted 100 points into octree (bounds: [-50, 50]^3)")
    print(f"  Query: ({query[0]}, {query[1]}, {query[2]})")
    print(f"  Octree nearest: {obj[3]}, dist={dist:.4f}")
    print(f"  Brute-force nearest: sphere_{bf_idx}, dist={bf_dist:.4f}")
    print(f"  Match: {abs(dist - bf_dist) < 1e-6}")


# ---------------------------------------------------------------------------
# Exercise 5 -- Frustum Culling
# ---------------------------------------------------------------------------

def exercise_5():
    """
    Implement a function that takes 6 frustum planes and an AABB, and returns
    whether the AABB is OUTSIDE, INSIDE, or INTERSECTING the frustum.

    Each plane is (nx, ny, nz, d) where nx*x + ny*y + nz*z + d >= 0 for
    points inside the frustum.
    """

    def classify_aabb_frustum(planes, aabb_min, aabb_max):
        """
        Test AABB against frustum planes.
        Returns: 'outside', 'inside', or 'intersecting'.
        """
        all_inside = True
        for nx, ny, nz, d in planes:
            # Find the "p-vertex" (most in the direction of the normal)
            px = aabb_max[0] if nx >= 0 else aabb_min[0]
            py = aabb_max[1] if ny >= 0 else aabb_min[1]
            pz = aabb_max[2] if nz >= 0 else aabb_min[2]

            # Find the "n-vertex" (most against the normal)
            nnx = aabb_min[0] if nx >= 0 else aabb_max[0]
            nny = aabb_min[1] if ny >= 0 else aabb_max[1]
            nnz = aabb_min[2] if nz >= 0 else aabb_max[2]

            # If p-vertex is outside, entire AABB is outside
            if nx * px + ny * py + nz * pz + d < 0:
                return 'outside'

            # If n-vertex is outside, AABB is not fully inside
            if nx * nnx + ny * nny + nz * nnz + d < 0:
                all_inside = False

        return 'inside' if all_inside else 'intersecting'

    # Define a simple frustum (box-like for clarity):
    # left, right, bottom, top, near, far
    planes = [
        ( 1,  0,  0, 5),   # left:   x >= -5
        (-1,  0,  0, 5),   # right:  x <= 5
        ( 0,  1,  0, 5),   # bottom: y >= -5
        ( 0, -1,  0, 5),   # top:    y <= 5
        ( 0,  0,  1, 0),   # near:   z >= 0
        ( 0,  0, -1, 10),  # far:    z <= 10
    ]

    test_cases = [
        (np.array([1, 1, 2]), np.array([3, 3, 4]),    "fully inside"),
        (np.array([10, 10, 20]), np.array([12, 12, 22]), "fully outside"),
        (np.array([-6, 0, 2]), np.array([0, 3, 5]),   "intersecting left"),
        (np.array([0, 0, 8]), np.array([2, 2, 12]),    "intersecting far"),
    ]

    for aabb_min, aabb_max, desc in test_cases:
        result = classify_aabb_frustum(planes, aabb_min, aabb_max)
        print(f"  AABB [{aabb_min} .. {aabb_max}] ({desc}): {result}")


# ---------------------------------------------------------------------------
# Exercise 6 -- BSP Ordering (2D)
# ---------------------------------------------------------------------------

def exercise_6():
    """
    Given 5 polygons (line segments) in 2D with positions and normals,
    construct a BSP tree. Show the front-to-back traversal order for two
    different viewpoints.
    """

    class BSPNode:
        def __init__(self, name, pos, normal, front=None, back=None):
            self.name = name
            self.pos = np.array(pos, dtype=float)
            self.normal = np.array(normal, dtype=float)
            self.front = front
            self.back = back

    def classify(point, plane_pos, plane_normal):
        """Positive = front, negative = back."""
        return np.dot(np.array(point) - np.array(plane_pos), np.array(plane_normal))

    def bsp_insert(root, name, pos, normal):
        d = classify(pos, root.pos, root.normal)
        if d >= 0:
            if root.front is None:
                root.front = BSPNode(name, pos, normal)
            else:
                bsp_insert(root.front, name, pos, normal)
        else:
            if root.back is None:
                root.back = BSPNode(name, pos, normal)
            else:
                bsp_insert(root.back, name, pos, normal)

    def traverse_front_to_back(node, eye, result):
        """Traverse BSP to produce front-to-back ordering from eye."""
        if node is None:
            return
        d = classify(eye, node.pos, node.normal)
        if d >= 0:
            # Eye is in front: visit front first, then self, then back
            traverse_front_to_back(node.front, eye, result)
            result.append(node.name)
            traverse_front_to_back(node.back, eye, result)
        else:
            # Eye is behind: visit back first, then self, then front
            traverse_front_to_back(node.back, eye, result)
            result.append(node.name)
            traverse_front_to_back(node.front, eye, result)

    # 5 polygons (simplified as named points with normals in 2D)
    polygons = [
        ("A", [0, 0], [0, 1]),
        ("B", [2, 3], [1, 0]),
        ("C", [-3, 1], [-1, 0]),
        ("D", [1, -2], [0, -1]),
        ("E", [-1, 4], [0, 1]),
    ]

    # Build BSP: use first polygon as root
    root = BSPNode(*polygons[0])
    for name, pos, normal in polygons[1:]:
        bsp_insert(root, name, pos, normal)

    # Traverse for two viewpoints
    for eye_label, eye_pos in [("Eye1 (0, 10)", [0, 10]),
                                ("Eye2 (5, -5)", [5, -5])]:
        order = []
        traverse_front_to_back(root, eye_pos, order)
        print(f"  {eye_label}: front-to-back order = {' -> '.join(order)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Exercise 1: Scene Graph Transform ===")
    exercise_1()

    print("\n=== Exercise 2: AABB Construction ===")
    exercise_2()

    print("\n=== Exercise 3: BVH Split Comparison (SAH vs Midpoint) ===")
    exercise_3()

    print("\n=== Exercise 4: Octree with Nearest-Neighbor ===")
    exercise_4()

    print("\n=== Exercise 5: Frustum Culling ===")
    exercise_5()

    print("\n=== Exercise 6: BSP Ordering (2D) ===")
    exercise_6()

    print("\nAll exercises completed!")
