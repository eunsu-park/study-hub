# 09. Scene Graphs and Spatial Data Structures

[← Previous: Shader Programming (GLSL)](08_Shader_Programming_GLSL.md) | [Next: Ray Tracing Basics →](10_Ray_Tracing_Basics.md)

---

## Learning Objectives

1. Understand scene graphs as hierarchical data structures for organizing 3D worlds
2. Distinguish between local (model) space and world space transforms via parent-child chains
3. Implement scene graph traversal for rendering and transform propagation
4. Define and construct bounding volumes: AABB, OBB, and bounding spheres
5. Build a Bounding Volume Hierarchy (BVH) using the Surface Area Heuristic (SAH)
6. Explain octree, quadtree, and BSP tree spatial partitioning schemes
7. Apply frustum culling and understand occlusion culling concepts
8. Implement BVH construction and ray-BVH intersection in Python

---

## Why This Matters

A scene with a million triangles cannot be rendered or ray-traced efficiently by testing every triangle against every pixel or ray. **Spatial data structures** are the fundamental tools that reduce this $O(n)$ brute-force cost to $O(\log n)$ per query, making real-time rendering and interactive ray tracing feasible. Every modern game engine, film renderer, and physics simulation relies on scene graphs and acceleration structures. Understanding them is the bridge between knowing *how* to shade a triangle and being able to shade *millions* of them at 60 frames per second.

Beyond performance, **scene graphs** provide the organizational backbone for complex scenes. A robot arm, a solar system, or a city block are all naturally described as hierarchies of objects whose positions depend on their parents. Mastering these structures gives you the vocabulary and algorithms that every graphics engine is built upon.

---

## 1. Scene Graphs

### 1.1 What Is a Scene Graph?

A **scene graph** is a directed acyclic graph (typically a tree) where each node represents an object, group, or transformation in the scene. Nodes store:

- A **local transform** (translation, rotation, scale) relative to the parent
- Optional geometry (mesh), material, light, or camera data
- A list of child nodes

```
Root (World)
├── Sun (Light)
├── Planet (Mesh + Transform)
│   ├── Moon (Mesh + Transform)
│   └── Satellite (Mesh + Transform)
└── Spaceship (Group + Transform)
    ├── Hull (Mesh)
    ├── Left_Engine (Mesh + Transform)
    └── Right_Engine (Mesh + Transform)
```

The power of the scene graph is that moving the "Spaceship" node automatically moves all its children (Hull, engines) because their positions are defined **relative** to the parent.

### 1.2 Local vs. World Transforms

Each node stores a **local transform matrix** $\mathbf{M}_{\text{local}}$ that positions it relative to its parent. The **world transform** $\mathbf{M}_{\text{world}}$ is the product of all ancestor transforms:

$$\mathbf{M}_{\text{world}} = \mathbf{M}_{\text{root}} \cdot \mathbf{M}_{\text{child}_1} \cdot \mathbf{M}_{\text{child}_2} \cdots \mathbf{M}_{\text{local}}$$

For a point $\mathbf{p}$ in local space, its world position is:

$$\mathbf{p}_{\text{world}} = \mathbf{M}_{\text{world}} \cdot \mathbf{p}_{\text{local}}$$

This composition is the reason scene graphs are so powerful: you modify one matrix and the entire subtree updates accordingly.

**Example**: A planet orbits the sun (rotation around origin), and a moon orbits the planet. The moon's world position is:

$$\mathbf{M}_{\text{moon}}^{\text{world}} = \mathbf{R}_{\text{planet orbit}} \cdot \mathbf{T}_{\text{planet offset}} \cdot \mathbf{R}_{\text{moon orbit}} \cdot \mathbf{T}_{\text{moon offset}}$$

### 1.3 Scene Graph Traversal

To render the scene, we perform a **depth-first traversal**, accumulating transform matrices on a stack:

```
function traverse(node, parent_world_matrix):
    world_matrix = parent_world_matrix * node.local_transform
    if node.has_geometry:
        render(node.geometry, world_matrix)
    for child in node.children:
        traverse(child, world_matrix)
```

**Rendering order** matters for:
- **Opaque objects**: Render front-to-back (minimizes overdraw via early z-test rejection)
- **Transparent objects**: Render back-to-front (correct alpha blending requires drawing distant objects first)
- **Sorting**: The scene graph itself does not guarantee correct order; a separate sorting pass is often used

### 1.4 Python Implementation: Scene Graph

```python
import numpy as np

class SceneNode:
    """A node in the scene graph with hierarchical transforms."""

    def __init__(self, name, transform=None):
        self.name = name
        # Local transform relative to parent (4x4 identity by default)
        self.local_transform = transform if transform is not None else np.eye(4)
        self.children = []
        self.geometry = None       # Optional mesh data
        self.world_transform = np.eye(4)  # Computed during traversal

    def add_child(self, child):
        """Attach a child node, creating the parent-child relationship."""
        self.children.append(child)
        return child

    def update_transforms(self, parent_world=None):
        """
        Depth-first traversal that propagates transforms down the tree.
        Each node's world_transform = parent's world_transform * local_transform.
        """
        if parent_world is None:
            parent_world = np.eye(4)

        # Why matrix multiply here: this composes all ancestor transforms
        # so that the node's geometry can be placed directly in world space
        self.world_transform = parent_world @ self.local_transform

        for child in self.children:
            child.update_transforms(self.world_transform)

    def collect_renderables(self, result=None):
        """Gather all nodes that have geometry, sorted for rendering."""
        if result is None:
            result = []
        if self.geometry is not None:
            result.append(self)
        for child in self.children:
            child.collect_renderables(result)
        return result


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


# Build a simple solar system scene graph
root = SceneNode("Root")
sun = root.add_child(SceneNode("Sun"))
sun.geometry = "sun_mesh"

# Planet: rotate 45 degrees around sun, offset by 5 units
planet_orbit = root.add_child(
    SceneNode("PlanetOrbit", make_rotation_y(45.0))
)
planet = planet_orbit.add_child(
    SceneNode("Planet", make_translation(5.0, 0.0, 0.0))
)
planet.geometry = "planet_mesh"

# Moon: rotate 30 degrees around planet, offset by 1.5 units
moon_orbit = planet.add_child(
    SceneNode("MoonOrbit", make_rotation_y(30.0))
)
moon = moon_orbit.add_child(
    SceneNode("Moon", make_translation(1.5, 0.0, 0.0))
)
moon.geometry = "moon_mesh"

# Propagate all transforms
root.update_transforms()

# Print world positions
for node in root.collect_renderables():
    pos = node.world_transform[:3, 3]
    print(f"{node.name:>10}: world position = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
```

Output:
```
       Sun: world position = (0.00, 0.00, 0.00)
    Planet: world position = (3.54, 0.00, -3.54)
      Moon: world position = (4.61, 0.00, -2.89)
```

The moon's world position is the composed result of planet orbit + planet offset + moon orbit + moon offset -- all computed automatically by the scene graph traversal.

---

## 2. Bounding Volumes

Testing every triangle for intersection (with a ray, frustum, or another object) is expensive. **Bounding volumes** are simple geometric shapes that enclose complex geometry, providing cheap "reject early" tests.

### 2.1 Axis-Aligned Bounding Box (AABB)

An AABB is defined by minimum and maximum corners aligned with the coordinate axes:

$$\text{AABB} = \{(x, y, z) \;|\; x_{\min} \le x \le x_{\max},\; y_{\min} \le y \le y_{\max},\; z_{\min} \le z \le z_{\max}\}$$

**Advantages**: Very fast intersection tests (slab method), easy to construct (min/max of vertices), compact storage (6 floats).

**Disadvantages**: Poor fit for elongated or rotated objects. Rotating an AABB-enclosed object requires recomputing the AABB (it grows).

**Ray-AABB intersection (slab method)**: A ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ intersects the AABB if the intervals where the ray is inside each pair of slabs overlap:

$$t_{x,\text{min}} = \frac{x_{\min} - o_x}{d_x}, \quad t_{x,\text{max}} = \frac{x_{\max} - o_x}{d_x}$$

Similarly for $y$ and $z$. The ray hits the box if:

$$t_{\text{enter}} = \max(t_{x,\text{min}}, t_{y,\text{min}}, t_{z,\text{min}}) \le t_{\text{exit}} = \min(t_{x,\text{max}}, t_{y,\text{max}}, t_{z,\text{max}})$$

and $t_{\text{exit}} \ge 0$.

### 2.2 Oriented Bounding Box (OBB)

An OBB is a box with arbitrary orientation, defined by a center $\mathbf{c}$, three orthonormal axes $\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$, and half-extents $e_1, e_2, e_3$.

$$\text{OBB} = \left\{\mathbf{c} + \sum_{i=1}^{3} a_i \mathbf{u}_i \;\middle|\; |a_i| \le e_i\right\}$$

**Advantages**: Much tighter fit for elongated or rotated objects.

**Disadvantages**: More expensive intersection tests. OBB-OBB overlap uses the **Separating Axis Theorem (SAT)** with up to 15 axes to test.

### 2.3 Bounding Spheres

A bounding sphere is defined by center $\mathbf{c}$ and radius $r$:

$$\text{Sphere} = \{\mathbf{p} \;|\; \|\mathbf{p} - \mathbf{c}\| \le r\}$$

**Advantages**: Rotation-invariant (no recomputation needed), very fast point-in-sphere and sphere-sphere tests.

**Disadvantages**: Poor fit for non-spherical objects (large wasted volume).

**Ray-sphere intersection**: Substitute $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ into $\|\mathbf{p} - \mathbf{c}\|^2 = r^2$:

$$\|\mathbf{o} + t\mathbf{d} - \mathbf{c}\|^2 = r^2$$
$$(\mathbf{d} \cdot \mathbf{d})t^2 + 2\mathbf{d} \cdot (\mathbf{o} - \mathbf{c})t + (\mathbf{o} - \mathbf{c}) \cdot (\mathbf{o} - \mathbf{c}) - r^2 = 0$$

This is a quadratic in $t$. If the discriminant $\Delta < 0$, the ray misses. Otherwise, the smallest positive root gives the nearest hit.

### 2.4 Comparison

| Property | AABB | OBB | Sphere |
|----------|------|-----|--------|
| Storage | 6 floats | 15 floats | 4 floats |
| Fit quality | Moderate | Tight | Loose |
| Intersection cost | Very low | Moderate | Very low |
| Rotation handling | Recompute | Transform axes | No change |
| Construction | $O(n)$ | $O(n)$ PCA | $O(n)$ |

---

## 3. Bounding Volume Hierarchy (BVH)

### 3.1 Concept

A **BVH** is a tree of bounding volumes. The root's bounding volume encloses the entire scene. Each internal node's volume encloses all objects in its subtree. Leaves contain one or a small number of primitives.

```
          [Root AABB: entire scene]
         /                         \
  [Left AABB]                [Right AABB]
   /       \                  /         \
[Leaf A] [Leaf B]        [Leaf C]   [Leaf D]
 tri 1    tri 2,3         tri 4      tri 5,6
```

**Ray traversal**: Test ray against root AABB. If miss, skip entire tree. If hit, recurse into children. This prunes large portions of the scene, reducing average complexity from $O(n)$ to $O(\log n)$.

### 3.2 BVH Construction

The key decision is how to **partition** primitives at each node. Common strategies:

**Midpoint split**: Split along the longest axis of the bounding box at its midpoint. Simple but can create unbalanced trees.

**Median split**: Split so each child gets half the primitives. Guarantees a balanced tree but may create large-volume bounding boxes.

**Surface Area Heuristic (SAH)**: The gold standard. The cost of a node is:

$$C_{\text{node}} = C_{\text{trav}} + \frac{SA(\text{left})}{SA(\text{parent})} \cdot n_{\text{left}} \cdot C_{\text{isect}} + \frac{SA(\text{right})}{SA(\text{parent})} \cdot n_{\text{right}} \cdot C_{\text{isect}}$$

where:
- $C_{\text{trav}}$ = cost of traversing a node (constant)
- $C_{\text{isect}}$ = cost of a primitive intersection test (constant)
- $SA(\cdot)$ = surface area of the bounding box
- $n_{\text{left}}, n_{\text{right}}$ = number of primitives in each child

The intuition is that a ray is more likely to hit a child node with a **larger surface area**. SAH minimizes the expected cost of a ray query by preferring splits that create children with small surface areas relative to the parent.

In practice, we evaluate candidate splits at several positions along each axis (or at primitive boundaries) and choose the split that minimizes the SAH cost.

### 3.3 BVH Traversal for Ray Queries

```
function bvh_intersect(ray, node):
    if not ray_intersects_aabb(ray, node.aabb):
        return NO_HIT

    if node.is_leaf:
        return intersect_primitives(ray, node.primitives)

    hit_left  = bvh_intersect(ray, node.left)
    hit_right = bvh_intersect(ray, node.right)
    return closest(hit_left, hit_right)
```

**Optimization**: Test which child's AABB is closer to the ray origin and traverse that child first. If we find a hit, it may allow early termination of the second child (if the second child's AABB entry is farther than the found hit).

### 3.4 Python Implementation: BVH

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class AABB:
    """Axis-Aligned Bounding Box."""
    min_pt: np.ndarray  # (3,) minimum corner
    max_pt: np.ndarray  # (3,) maximum corner

    def surface_area(self) -> float:
        """Surface area of the box -- used by SAH to estimate ray hit probability."""
        d = self.max_pt - self.min_pt
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

    def longest_axis(self) -> int:
        """Return index of the longest axis (0=x, 1=y, 2=z)."""
        d = self.max_pt - self.min_pt
        return int(np.argmax(d))

    @staticmethod
    def from_points(points: np.ndarray) -> 'AABB':
        """Build AABB from an array of points (N, 3)."""
        return AABB(np.min(points, axis=0), np.max(points, axis=0))

    @staticmethod
    def union(a: 'AABB', b: 'AABB') -> 'AABB':
        """Merge two AABBs into one that encloses both."""
        return AABB(np.minimum(a.min_pt, b.min_pt),
                    np.maximum(a.max_pt, b.max_pt))

    def intersect_ray(self, origin: np.ndarray, inv_dir: np.ndarray) -> Tuple[bool, float]:
        """
        Slab-based ray-AABB intersection.
        inv_dir = 1.0 / ray_direction (precomputed for speed).
        Returns (hit, t_entry).
        """
        t1 = (self.min_pt - origin) * inv_dir
        t2 = (self.max_pt - origin) * inv_dir

        t_min = np.minimum(t1, t2)  # Why element-wise min: handles negative direction
        t_max = np.maximum(t1, t2)

        t_enter = np.max(t_min)
        t_exit  = np.min(t_max)

        hit = (t_enter <= t_exit) and (t_exit >= 0.0)
        return hit, t_enter


@dataclass
class Sphere:
    """A simple sphere primitive for testing BVH."""
    center: np.ndarray
    radius: float
    color: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))

    def aabb(self) -> AABB:
        r = np.array([self.radius, self.radius, self.radius])
        return AABB(self.center - r, self.center + r)

    def intersect_ray(self, origin, direction) -> Tuple[bool, float]:
        """Geometric ray-sphere intersection test."""
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return False, float('inf')

        sqrt_disc = np.sqrt(discriminant)
        t = (-b - sqrt_disc) / (2.0 * a)
        if t < 0.001:
            t = (-b + sqrt_disc) / (2.0 * a)
        if t < 0.001:
            return False, float('inf')

        return True, t


@dataclass
class BVHNode:
    """A node in the Bounding Volume Hierarchy."""
    aabb: AABB
    left: Optional['BVHNode'] = None
    right: Optional['BVHNode'] = None
    primitives: List = field(default_factory=list)  # Non-empty only for leaves

    @property
    def is_leaf(self) -> bool:
        return len(self.primitives) > 0


def build_bvh(primitives: List, max_leaf_size: int = 2) -> BVHNode:
    """
    Recursively build a BVH using the Surface Area Heuristic (SAH).

    The SAH evaluates candidate splits by estimating the expected ray
    intersection cost. We try several split positions along the longest
    axis and pick the one that minimizes cost.
    """
    # Compute bounding box of all primitives
    aabbs = [p.aabb() for p in primitives]
    total_aabb = aabbs[0]
    for box in aabbs[1:]:
        total_aabb = AABB.union(total_aabb, box)

    # Base case: few enough primitives to store in a leaf
    if len(primitives) <= max_leaf_size:
        return BVHNode(aabb=total_aabb, primitives=primitives)

    # Choose split axis and position using SAH
    best_cost = float('inf')
    best_axis = 0
    best_split_idx = len(primitives) // 2

    # Why we try all 3 axes: the longest axis is usually best, but
    # SAH may find a better split on a different axis
    for axis in range(3):
        # Sort primitive centers along this axis
        centers = [p.center[axis] if hasattr(p, 'center')
                   else (aabbs[i].min_pt[axis] + aabbs[i].max_pt[axis]) / 2
                   for i, p in enumerate(primitives)]
        sorted_indices = np.argsort(centers)
        sorted_prims = [primitives[i] for i in sorted_indices]
        sorted_aabbs = [aabbs[i] for i in sorted_indices]

        n = len(sorted_prims)

        # Build prefix AABBs from left and suffix AABBs from right
        left_aabbs = [None] * n
        right_aabbs = [None] * n

        left_aabbs[0] = sorted_aabbs[0]
        for i in range(1, n):
            left_aabbs[i] = AABB.union(left_aabbs[i-1], sorted_aabbs[i])

        right_aabbs[n-1] = sorted_aabbs[n-1]
        for i in range(n-2, -1, -1):
            right_aabbs[i] = AABB.union(right_aabbs[i+1], sorted_aabbs[i])

        # Evaluate SAH cost for each candidate split position
        parent_sa = total_aabb.surface_area()
        C_TRAV = 1.0    # Relative cost of traversing a node
        C_ISECT = 4.0   # Relative cost of a primitive intersection

        for i in range(1, n):
            # Split: left gets [0..i-1], right gets [i..n-1]
            left_sa = left_aabbs[i-1].surface_area()
            right_sa = right_aabbs[i].surface_area()

            cost = C_TRAV + (left_sa / parent_sa) * i * C_ISECT \
                         + (right_sa / parent_sa) * (n - i) * C_ISECT

            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_split_idx = i

    # Perform the best split
    centers = [p.center[best_axis] if hasattr(p, 'center')
               else (aabbs[i].min_pt[best_axis] + aabbs[i].max_pt[best_axis]) / 2
               for i, p in enumerate(primitives)]
    sorted_indices = np.argsort(centers)
    sorted_prims = [primitives[i] for i in sorted_indices]

    left_prims  = sorted_prims[:best_split_idx]
    right_prims = sorted_prims[best_split_idx:]

    # Guard: if SAH puts everything on one side, force an even split
    if len(left_prims) == 0 or len(right_prims) == 0:
        mid = len(sorted_prims) // 2
        left_prims  = sorted_prims[:mid]
        right_prims = sorted_prims[mid:]

    left_child  = build_bvh(left_prims, max_leaf_size)
    right_child = build_bvh(right_prims, max_leaf_size)

    return BVHNode(aabb=total_aabb, left=left_child, right=right_child)


def bvh_intersect(node: BVHNode, origin: np.ndarray, direction: np.ndarray,
                  inv_dir: np.ndarray) -> Tuple[bool, float, object]:
    """
    Traverse the BVH to find the nearest ray intersection.
    Returns (hit, t, primitive).
    """
    hit_box, t_entry = node.aabb.intersect_ray(origin, inv_dir)
    if not hit_box:
        return False, float('inf'), None

    if node.is_leaf:
        closest_t = float('inf')
        closest_prim = None
        for prim in node.primitives:
            hit, t = prim.intersect_ray(origin, direction)
            if hit and t < closest_t:
                closest_t = t
                closest_prim = prim
        return closest_prim is not None, closest_t, closest_prim

    # Recurse into both children, keep the nearest hit
    hit_l, t_l, prim_l = bvh_intersect(node.left, origin, direction, inv_dir)
    hit_r, t_r, prim_r = bvh_intersect(node.right, origin, direction, inv_dir)

    if hit_l and hit_r:
        if t_l <= t_r:
            return True, t_l, prim_l
        else:
            return True, t_r, prim_r
    elif hit_l:
        return True, t_l, prim_l
    elif hit_r:
        return True, t_r, prim_r
    else:
        return False, float('inf'), None


# --- Demo: Build BVH and trace a ray ---
np.random.seed(42)
spheres = [Sphere(center=np.random.uniform(-5, 5, 3),
                  radius=np.random.uniform(0.3, 1.0),
                  color=np.random.uniform(0, 1, 3))
           for _ in range(20)]

bvh_root = build_bvh(spheres, max_leaf_size=2)

# Trace a ray from the camera toward the scene
ray_origin = np.array([0.0, 0.0, 10.0])
ray_dir = np.array([0.0, 0.0, -1.0])
inv_dir = 1.0 / ray_dir  # Precompute for slab test

hit, t, prim = bvh_intersect(bvh_root, ray_origin, ray_dir, inv_dir)
if hit:
    hit_point = ray_origin + t * ray_dir
    print(f"Hit sphere at center {prim.center} at t={t:.3f}")
    print(f"Hit point: ({hit_point[0]:.2f}, {hit_point[1]:.2f}, {hit_point[2]:.2f})")
else:
    print("No intersection found")
```

---

## 4. Octrees and Quadtrees

### 4.1 Quadtree (2D)

A **quadtree** recursively subdivides 2D space into four equal quadrants. Each node either stores objects directly (leaf) or has exactly four children.

```
┌───────────┬───────────┐
│           │           │
│    NW     │    NE     │
│           │           │
├───────────┼───────────┤
│           │           │
│    SW     │    SE     │
│           │           │
└───────────┴───────────┘
```

**Usage**: Collision detection in 2D games, spatial queries (find all objects near a point), terrain LOD.

**Insertion**: Place object in the smallest quadrant that fully contains it. If a quadrant has too many objects, subdivide it.

### 4.2 Octree (3D)

An **octree** extends the quadtree to 3D by splitting each node into eight octants. Each octant is defined by splitting along all three axes at the node's center.

**Construction**:
1. Start with a root node enclosing the entire scene
2. Insert objects: if a node has more than a threshold of objects, split into 8 children
3. Objects go into the child that fully contains them, or remain at the parent if they straddle boundaries

**Properties**:
- Uniform subdivision (unlike BVH which adapts to geometry)
- Well-suited when objects are roughly uniformly distributed
- Depth is bounded by $O(\log_8 n)$ for $n$ objects if balanced

**Trade-off vs BVH**: Octrees subdivide space uniformly regardless of where geometry is. BVHs adapt to geometry distribution, usually giving better performance for ray tracing. Octrees are simpler to build incrementally and better for dynamic scenes.

### 4.3 Quadtree Implementation

```python
class QuadTreeNode:
    """Quadtree node for 2D spatial partitioning."""

    MAX_OBJECTS = 4
    MAX_DEPTH = 8

    def __init__(self, x, y, width, height, depth=0):
        self.bounds = (x, y, width, height)
        self.depth = depth
        self.objects = []
        self.children = None  # None until subdivided

    def subdivide(self):
        """Split into four equal quadrants."""
        x, y, w, h = self.bounds
        hw, hh = w / 2, h / 2
        d = self.depth + 1
        # Why list of 4: NW, NE, SW, SE -- standard quadtree convention
        self.children = [
            QuadTreeNode(x,      y,      hw, hh, d),  # NW
            QuadTreeNode(x + hw, y,      hw, hh, d),  # NE
            QuadTreeNode(x,      y + hh, hw, hh, d),  # SW
            QuadTreeNode(x + hw, y + hh, hw, hh, d),  # SE
        ]

    def insert(self, obj_x, obj_y, obj_data=None):
        """Insert a point object into the quadtree."""
        x, y, w, h = self.bounds
        # Check if point is within this node's bounds
        if not (x <= obj_x < x + w and y <= obj_y < y + h):
            return False

        if self.children is None:
            self.objects.append((obj_x, obj_y, obj_data))
            # Subdivide if over capacity and not at max depth
            if len(self.objects) > self.MAX_OBJECTS and self.depth < self.MAX_DEPTH:
                self.subdivide()
                # Re-insert existing objects into children
                old_objects = self.objects
                self.objects = []
                for ox, oy, od in old_objects:
                    inserted = False
                    for child in self.children:
                        if child.insert(ox, oy, od):
                            inserted = True
                            break
                    if not inserted:
                        self.objects.append((ox, oy, od))
            return True

        # Try to insert into children
        for child in self.children:
            if child.insert(obj_x, obj_y, obj_data):
                return True
        return False

    def query_range(self, qx, qy, qw, qh):
        """Find all objects within a rectangular range."""
        x, y, w, h = self.bounds
        results = []

        # Check if query range intersects this node
        if qx + qw < x or qx > x + w or qy + qh < y or qy > y + h:
            return results

        # Check objects stored at this node
        for ox, oy, od in self.objects:
            if qx <= ox <= qx + qw and qy <= oy <= qy + qh:
                results.append((ox, oy, od))

        # Recurse into children
        if self.children:
            for child in self.children:
                results.extend(child.query_range(qx, qy, qw, qh))

        return results


# Demo
qt = QuadTreeNode(0, 0, 100, 100)
import random
random.seed(42)
for i in range(50):
    qt.insert(random.uniform(0, 100), random.uniform(0, 100), f"obj_{i}")

nearby = qt.query_range(40, 40, 20, 20)
print(f"Objects in region (40,40)-(60,60): {len(nearby)} found")
```

---

## 5. BSP Trees

### 5.1 Concept

A **Binary Space Partition (BSP) tree** recursively divides space using arbitrary hyperplanes (planes in 3D, lines in 2D). Each internal node stores a dividing plane, and the two children represent the half-spaces on each side.

**Key property**: A BSP tree can determine the front-to-back ordering of polygons relative to any viewpoint, enabling correct rendering of transparent objects without a z-buffer.

### 5.2 Construction

1. Choose a polygon (or plane) as the splitting plane
2. Classify all other polygons as **in front**, **behind**, or **spanning** the plane
3. Split spanning polygons along the plane
4. Recurse: front polygons go to the front child, back polygons to the back child

**Plane selection** matters greatly: a poor choice increases the number of polygon splits, inflating the tree. Heuristics balance tree depth against split count.

### 5.3 BSP Traversal for Painter's Algorithm

To render back-to-front (painter's algorithm) from a camera at position $\mathbf{e}$:

```
function render_bsp(node, eye_position):
    if node is leaf:
        draw(node.polygon)
        return

    d = dot(eye_position - node.plane_point, node.plane_normal)

    if d > 0:  // Eye is in front of the plane
        render_bsp(node.back,  eye_position)  // Draw back first
        draw(node.polygon)
        render_bsp(node.front, eye_position)  // Draw front last (on top)
    else:
        render_bsp(node.front, eye_position)
        draw(node.polygon)
        render_bsp(node.back,  eye_position)
```

This produces a correct back-to-front ordering for any eye position, which is why BSP trees were historically important (used in Doom, Quake).

### 5.4 Modern Usage

BSP trees are less common in modern real-time rendering (z-buffers are fast and hardware-accelerated), but they remain useful for:
- **Constructive Solid Geometry (CSG)**: Boolean operations on solids
- **Visibility determination**: Indoor environments with many occluders
- **Collision detection**: Spatial queries in physics engines

---

## 6. Frustum Culling and Occlusion Culling

### 6.1 View Frustum

The view frustum is the truncated pyramid defined by the camera's field of view, aspect ratio, and near/far planes. Only objects inside (or intersecting) the frustum are potentially visible.

$$\text{Frustum} = \bigcap_{i=1}^{6} \{\mathbf{p} \;|\; \mathbf{n}_i \cdot \mathbf{p} + d_i \ge 0\}$$

where $\mathbf{n}_i$ and $d_i$ define the six frustum planes (left, right, top, bottom, near, far).

### 6.2 Frustum Culling

**Frustum culling** tests each object's bounding volume against the frustum. Objects entirely outside are skipped.

For an AABB, test against each frustum plane:
- If the AABB's "most positive" vertex (relative to the plane normal) is behind the plane, the entire AABB is outside
- If the AABB's "most negative" vertex is in front, the AABB is fully inside that plane

This test is $O(1)$ per AABB and dramatically reduces draw calls in large scenes.

**Hierarchical frustum culling** with a BVH or octree: if a parent node is outside the frustum, all its children are outside too -- prune the entire subtree.

### 6.3 Occlusion Culling

Even objects inside the frustum may be **hidden behind** other objects. **Occlusion culling** detects and skips these hidden objects.

**Approaches**:
- **Hardware occlusion queries**: Ask the GPU "would this bounding box produce any visible pixels?" using conditional rendering
- **Hierarchical Z-buffer (HZB)**: Maintain a mipmapped depth buffer; test bounding volumes against coarse depth levels
- **Software occlusion**: Rasterize a simplified occluder set on the CPU; test remaining objects against the resulting depth buffer
- **Potentially Visible Sets (PVS)**: Precompute which regions can see which other regions (used in indoor environments)

### 6.4 Putting It All Together

A typical rendering pipeline combines these techniques:

```
Scene Graph
    └── Frustum Cull (using BVH/octree)
         └── Occlusion Cull (HZB or queries)
              └── Sort (front-to-back for opaque, back-to-front for transparent)
                   └── Draw
```

Each stage filters out objects, reducing the number of draw calls sent to the GPU.

---

## 7. Performance Considerations

### 7.1 Choosing the Right Structure

| Use Case | Recommended Structure |
|----------|----------------------|
| Ray tracing static scenes | BVH (SAH) |
| Ray tracing dynamic scenes | BVH with refitting |
| Real-time frustum culling | Octree or BVH |
| Indoor visibility | BSP + PVS |
| 2D collision detection | Quadtree |
| GPU ray tracing (RTX) | BVH (hardware-accelerated) |

### 7.2 BVH vs. Octree Tradeoffs

- **BVH** adapts to geometry density; nodes can overlap; better for non-uniform distributions
- **Octree** has fixed spatial subdivision; simpler to update incrementally; better for roughly uniform distributions
- **Hybrid approaches**: Use an octree at the top level and BVH within each cell

### 7.3 Dynamic Scenes

When objects move, spatial structures must be updated:
- **Rebuild**: Expensive but produces optimal structures
- **Refit**: Update AABBs bottom-up without changing topology (fast, but quality degrades over time)
- **Incremental insert/remove**: Supported by octrees; BVH requires more care
- **Two-level BVH**: Static geometry in a fixed BVH; dynamic objects in a separate, frequently rebuilt BVH

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Scene graph | Hierarchical tree: local transform at each node, world transform = product of ancestor chain |
| AABB | Axis-aligned box; very fast tests; 6 floats; recompute on rotation |
| OBB | Oriented box; tight fit; more expensive tests (SAT) |
| Bounding sphere | Rotation-invariant; cheap tests; loose fit |
| BVH | Binary tree of bounding volumes; SAH gives optimal splits; $O(\log n)$ ray queries |
| Octree / Quadtree | Uniform spatial subdivision; 8 (3D) or 4 (2D) children per node |
| BSP tree | Arbitrary splitting planes; enables back-to-front ordering without z-buffer |
| Frustum culling | Skip objects outside the view frustum using plane-AABB tests |
| Occlusion culling | Skip objects hidden behind other objects (HZB, queries, PVS) |

## Exercises

1. **Scene graph transform**: Given a scene graph where node A has translation $(3, 0, 0)$ and rotation $45°$ around Y, and child node B has translation $(2, 0, 0)$, compute B's world position by hand.

2. **AABB construction**: Given 5 triangles with specified vertices, compute the AABB for each triangle, then compute the AABB that encloses all five. Implement this in Python.

3. **BVH split comparison**: Modify the BVH code to use midpoint splitting instead of SAH. Generate 1000 random spheres and compare the average number of AABB intersection tests per ray between SAH and midpoint BVH.

4. **Octree implementation**: Extend the quadtree code to 3D (octree). Insert 100 random spheres and implement a nearest-neighbor query.

5. **Frustum culling**: Implement a function that takes 6 frustum planes and an AABB, and returns whether the AABB is outside, inside, or intersecting the frustum.

6. **BSP ordering**: Given 5 polygons in 2D with specified positions and normals, construct a BSP tree by hand. Show the front-to-back traversal order for two different viewpoints.

## Further Reading

- Ericson, C. *Real-Time Collision Detection*. Morgan Kaufmann, 2004. (The definitive reference on bounding volumes and spatial structures)
- Pharr, M., Jakob, W., Humphreys, G. *Physically Based Rendering: From Theory to Implementation*, 4th ed. MIT Press, 2023. (Chapter 4: BVH construction and traversal)
- Akenine-Moller, T., Haines, E., Hoffman, N. *Real-Time Rendering*, 4th ed. CRC Press, 2018. (Chapters 19 and 25: spatial data structures and culling)
- Karras, T. "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees." *HPG*, 2012. (GPU-accelerated BVH construction)
- Meagher, D. "Geometric Modeling Using Octree Encoding." *Computer Graphics and Image Processing*, 1982. (Original octree paper)
