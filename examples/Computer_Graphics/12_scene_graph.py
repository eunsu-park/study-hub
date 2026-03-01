"""
Scene Graph with BVH Acceleration
===================================

Implements a scene graph system with:
1. Node-based scene graph with parent-child relationships
2. Local/world transform propagation
3. AABB (Axis-Aligned Bounding Box) computation per node
4. BVH (Bounding Volume Hierarchy) construction (median split)
5. Ray-BVH intersection traversal
6. Visualization of the BVH bounding boxes

The scene graph is the standard way to organize objects in a 3D scene.
Parent-child relationships let you group objects (e.g., a car's wheels
rotate with the car body) and transform them hierarchically.

The BVH accelerates ray intersection queries from O(N) per ray to
O(log N) by organizing objects into a binary tree of bounding boxes.
This is essential for ray tracing performance -- without acceleration,
ray tracing a scene with thousands of objects is impractically slow.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time

# ---------------------------------------------------------------------------
# 1. AABB (Axis-Aligned Bounding Box)
# ---------------------------------------------------------------------------


@dataclass
class AABB:
    """Axis-Aligned Bounding Box in 3D.

    Why axis-aligned?  AABB intersection tests are trivially fast --
    just 6 comparisons.  Oriented Bounding Boxes (OBBs) fit objects
    more tightly but are much more expensive to intersect.  AABBs are
    the standard choice for BVH because the tighter fit of OBBs rarely
    compensates for the added intersection cost at scale.
    """
    min_pt: np.ndarray = field(default_factory=lambda: np.array([np.inf, np.inf, np.inf]))
    max_pt: np.ndarray = field(default_factory=lambda: np.array([-np.inf, -np.inf, -np.inf]))

    def expand_point(self, p: np.ndarray):
        """Grow the AABB to include point p."""
        self.min_pt = np.minimum(self.min_pt, p)
        self.max_pt = np.maximum(self.max_pt, p)

    def expand_aabb(self, other: 'AABB'):
        """Grow the AABB to include another AABB."""
        self.min_pt = np.minimum(self.min_pt, other.min_pt)
        self.max_pt = np.maximum(self.max_pt, other.max_pt)

    def centroid(self) -> np.ndarray:
        """Return the center of the AABB."""
        return (self.min_pt + self.max_pt) * 0.5

    def surface_area(self) -> float:
        """Surface area of the AABB.

        Why surface area?  The SAH (Surface Area Heuristic) for BVH
        construction uses surface area as a proxy for the probability
        that a random ray will hit the box.  Larger boxes are hit more
        often, so we want to minimize the total weighted surface area.
        """
        d = self.max_pt - self.min_pt
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[0]*d[2])

    def intersect_ray(self, origin: np.ndarray, inv_dir: np.ndarray,
                      t_min: float = 0, t_max: float = np.inf) -> bool:
        """Ray-AABB intersection test using the slab method.

        Why inv_dir (1/direction) instead of direction?  We'd divide by
        direction components 6 times.  Pre-computing the reciprocal and
        multiplying is faster and avoids repeated division.

        The slab method: an AABB is the intersection of 3 pairs of
        parallel planes (slabs).  For each axis, compute the entry and
        exit distances.  If the intersection of all 3 intervals is
        non-empty, the ray hits the box.
        """
        for axis in range(3):
            t1 = (self.min_pt[axis] - origin[axis]) * inv_dir[axis]
            t2 = (self.max_pt[axis] - origin[axis]) * inv_dir[axis]

            # Ensure t1 <= t2
            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return False

        return True

    def is_valid(self) -> bool:
        return np.all(self.min_pt <= self.max_pt)


# ---------------------------------------------------------------------------
# 2. Scene Graph Node
# ---------------------------------------------------------------------------

class SceneNode:
    """A node in the scene graph hierarchy.

    Each node has:
    - A local transform (relative to parent)
    - A list of children
    - Optional geometry (sphere primitives for this demo)

    The world transform is computed by multiplying the parent's world
    transform by this node's local transform.  This cascading is why
    scene graphs are hierarchical: move the parent, all children follow.
    """

    def __init__(self, name: str, local_transform: np.ndarray = None):
        self.name = name
        self.parent: Optional[SceneNode] = None
        self.children: List[SceneNode] = []

        # Local transform (4x4 matrix, identity by default)
        self.local_transform = local_transform if local_transform is not None else np.eye(4)

        # Cached world transform (computed during update)
        self.world_transform = np.eye(4)

        # Geometry (optional)
        self.geometry: Optional[SphereGeometry] = None

        # World-space AABB (computed from geometry + world transform)
        self.world_aabb = AABB()

    def add_child(self, child: 'SceneNode'):
        """Add a child node.

        Why track parent references?  When we need to traverse from
        leaf to root (e.g., for world transform calculation), having
        the parent pointer avoids searching the entire tree.
        """
        child.parent = self
        self.children.append(child)

    def update_transforms(self, parent_world: np.ndarray = None):
        """Recursively propagate transforms down the hierarchy.

        Why recursive DFS?  The world transform of each node depends on
        its parent's world transform, so we must process parents before
        children.  Depth-first traversal naturally ensures this ordering.

        In a production engine, this is done in a flat array (breadth-first
        ordered) for cache efficiency, but recursion is clearer for learning.
        """
        if parent_world is None:
            parent_world = np.eye(4)

        self.world_transform = parent_world @ self.local_transform

        # Update world AABB if this node has geometry
        if self.geometry:
            self.world_aabb = self.geometry.compute_world_aabb(self.world_transform)

        for child in self.children:
            child.update_transforms(self.world_transform)

    def get_all_leaf_nodes(self) -> List['SceneNode']:
        """Collect all nodes that have geometry (for BVH construction)."""
        result = []
        if self.geometry:
            result.append(self)
        for child in self.children:
            result.extend(child.get_all_leaf_nodes())
        return result

    def __repr__(self):
        return f"SceneNode('{self.name}')"


# ---------------------------------------------------------------------------
# 3. Sphere Geometry
# ---------------------------------------------------------------------------

@dataclass
class SphereGeometry:
    """A sphere primitive attached to a scene graph node.

    Why spheres?  They're the simplest 3D primitive with a closed-form
    ray intersection.  The AABB of a sphere is trivial to compute.
    In a real engine, this would be a mesh with thousands of triangles.
    """
    radius: float = 1.0
    color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))

    def compute_world_aabb(self, world_transform: np.ndarray) -> AABB:
        """Compute world-space AABB from the transform.

        Why not just transform the AABB?  For a sphere, the world-space
        AABB depends on the world position and the scale.  We extract
        the position from the transform matrix and use the radius
        (potentially scaled) to build the AABB.

        For simplicity, we assume uniform scaling.
        """
        # Extract world position from the 4x4 transform matrix
        world_pos = world_transform[:3, 3]

        # Extract scale (approximate: take the max column norm)
        # Why max?  For non-uniform scaling, the AABB must encompass the
        # largest dimension.  Using the max scale ensures correctness.
        scale = max(
            np.linalg.norm(world_transform[:3, 0]),
            np.linalg.norm(world_transform[:3, 1]),
            np.linalg.norm(world_transform[:3, 2]),
        )
        world_radius = self.radius * scale

        aabb = AABB(
            min_pt=world_pos - world_radius,
            max_pt=world_pos + world_radius,
        )
        return aabb

    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray,
                      world_transform: np.ndarray) -> Optional[float]:
        """Ray-sphere intersection in world space.

        Returns the hit distance t, or None if no intersection.
        """
        center = world_transform[:3, 3]
        scale = max(
            np.linalg.norm(world_transform[:3, 0]),
            np.linalg.norm(world_transform[:3, 1]),
            np.linalg.norm(world_transform[:3, 2]),
        )
        r = self.radius * scale

        oc = origin - center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - r * r
        disc = b * b - 4 * a * c

        if disc < 0:
            return None

        t = (-b - np.sqrt(disc)) / (2 * a)
        if t < 1e-4:
            t = (-b + np.sqrt(disc)) / (2 * a)
            if t < 1e-4:
                return None
        return t


# ---------------------------------------------------------------------------
# 4. BVH (Bounding Volume Hierarchy)
# ---------------------------------------------------------------------------

class BVHNode:
    """A node in the Bounding Volume Hierarchy.

    The BVH is a binary tree where:
    - Leaf nodes contain a list of scene primitives
    - Internal nodes have an AABB that encloses all children
    - Ray traversal starts at the root; if a node's AABB is missed,
      its entire subtree is skipped

    Why median split?  More sophisticated methods (SAH) produce better
    trees, but median split is simple and produces reasonable results.
    It's O(N log N) to build and O(log N) per query on average.
    """

    def __init__(self):
        self.aabb = AABB()
        self.left: Optional[BVHNode] = None
        self.right: Optional[BVHNode] = None
        self.primitives: List[SceneNode] = []  # Only for leaf nodes
        self.depth = 0


def build_bvh(nodes: List[SceneNode], depth: int = 0,
              max_leaf_size: int = 2) -> BVHNode:
    """Build a BVH using median split.

    The algorithm:
    1. Compute the AABB enclosing all primitives
    2. If few enough primitives, make a leaf
    3. Otherwise, choose the axis with the largest AABB extent
    4. Sort primitives by their centroid along that axis
    5. Split at the median
    6. Recursively build left and right subtrees

    Why split along the longest axis?  It tends to produce balanced
    trees with minimal overlap between children.  The longest axis has
    the most spatial spread, so splitting there separates primitives
    most effectively.
    """
    bvh_node = BVHNode()
    bvh_node.depth = depth

    if len(nodes) == 0:
        return bvh_node

    # Compute enclosing AABB
    for node in nodes:
        bvh_node.aabb.expand_aabb(node.world_aabb)

    # Leaf condition: few enough primitives
    if len(nodes) <= max_leaf_size:
        bvh_node.primitives = nodes
        return bvh_node

    # Choose split axis: longest extent
    extents = bvh_node.aabb.max_pt - bvh_node.aabb.min_pt
    split_axis = np.argmax(extents)

    # Sort by centroid along the split axis
    nodes_sorted = sorted(nodes, key=lambda n: n.world_aabb.centroid()[split_axis])

    # Split at the median
    mid = len(nodes_sorted) // 2

    # Ensure we don't create empty children
    # (can happen if all centroids are identical)
    if mid == 0:
        mid = 1
    elif mid == len(nodes_sorted):
        mid = len(nodes_sorted) - 1

    bvh_node.left = build_bvh(nodes_sorted[:mid], depth + 1, max_leaf_size)
    bvh_node.right = build_bvh(nodes_sorted[mid:], depth + 1, max_leaf_size)

    return bvh_node


def intersect_bvh(bvh_node: BVHNode, origin: np.ndarray,
                  direction: np.ndarray) -> Optional[Tuple[float, SceneNode]]:
    """Traverse the BVH to find the closest ray-object intersection.

    Why BVH traversal?  Testing every object is O(N).  BVH lets us
    skip entire groups of objects by testing their bounding box first.
    If the ray misses a bounding box, all objects inside are guaranteed
    to be missed -- no need to test them individually.

    Returns (distance, SceneNode) of the closest hit, or None.
    """
    inv_dir = np.where(np.abs(direction) > 1e-10,
                       1.0 / direction,
                       np.sign(direction) * 1e10)

    # Test this node's AABB
    if not bvh_node.aabb.intersect_ray(origin, inv_dir):
        return None  # Ray misses this entire subtree

    # Leaf node: test individual primitives
    if bvh_node.primitives:
        closest = None
        min_t = np.inf

        for node in bvh_node.primitives:
            if node.geometry:
                t = node.geometry.intersect_ray(origin, direction,
                                                 node.world_transform)
                if t is not None and t < min_t:
                    min_t = t
                    closest = (t, node)

        return closest

    # Internal node: recurse into children
    left_hit = intersect_bvh(bvh_node.left, origin, direction) if bvh_node.left else None
    right_hit = intersect_bvh(bvh_node.right, origin, direction) if bvh_node.right else None

    if left_hit is None:
        return right_hit
    if right_hit is None:
        return left_hit
    return left_hit if left_hit[0] < right_hit[0] else right_hit


# ---------------------------------------------------------------------------
# 5. Transform helpers
# ---------------------------------------------------------------------------

def translate_4x4(tx, ty, tz):
    M = np.eye(4)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M


def scale_4x4(sx, sy, sz):
    M = np.eye(4)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M


def rotate_y_4x4(angle_deg):
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    M = np.eye(4)
    M[0, 0] = c; M[0, 2] = s
    M[2, 0] = -s; M[2, 2] = c
    return M


# ---------------------------------------------------------------------------
# 6. Build demo scene
# ---------------------------------------------------------------------------

def build_demo_scene() -> Tuple[SceneNode, List[SceneNode]]:
    """Build a hierarchical scene graph for demonstration.

    Scene structure:
      Root
       +-- SolarSystem (rotated)
       |    +-- Sun (large yellow sphere)
       |    +-- EarthOrbit (translated, rotated)
       |    |    +-- Earth (blue sphere)
       |    |    +-- MoonOrbit (translated)
       |    |         +-- Moon (small gray sphere)
       |    +-- MarsOrbit (translated, rotated)
       |         +-- Mars (red sphere)
       +-- Asteroid1 (standalone)
       +-- Asteroid2 (standalone)
       +-- Asteroid3 (standalone)

    Why a solar system?  It's a natural example of hierarchical
    transforms: the Moon orbits Earth, which orbits the Sun.  Moving
    the Sun moves everything.  Rotating EarthOrbit rotates Earth and
    its Moon together.
    """
    root = SceneNode("Root")

    # Solar system group
    solar = SceneNode("SolarSystem", rotate_y_4x4(15))
    root.add_child(solar)

    # Sun
    sun = SceneNode("Sun", translate_4x4(0, 0, 0))
    sun.geometry = SphereGeometry(radius=1.5, color=np.array([1.0, 0.9, 0.2]))
    solar.add_child(sun)

    # Earth orbit
    earth_orbit = SceneNode("EarthOrbit",
                             translate_4x4(5, 0, 0) @ rotate_y_4x4(30))
    solar.add_child(earth_orbit)

    earth = SceneNode("Earth", translate_4x4(0, 0, 0))
    earth.geometry = SphereGeometry(radius=0.7, color=np.array([0.2, 0.4, 0.9]))
    earth_orbit.add_child(earth)

    # Moon orbit (relative to Earth)
    moon_orbit = SceneNode("MoonOrbit", translate_4x4(1.8, 0.3, 0.5))
    earth_orbit.add_child(moon_orbit)

    moon = SceneNode("Moon", translate_4x4(0, 0, 0))
    moon.geometry = SphereGeometry(radius=0.25, color=np.array([0.7, 0.7, 0.7]))
    moon_orbit.add_child(moon)

    # Mars orbit
    mars_orbit = SceneNode("MarsOrbit",
                            translate_4x4(-4, 0, -6) @ rotate_y_4x4(-20))
    solar.add_child(mars_orbit)

    mars = SceneNode("Mars", translate_4x4(0, 0, 0))
    mars.geometry = SphereGeometry(radius=0.5, color=np.array([0.9, 0.3, 0.2]))
    mars_orbit.add_child(mars)

    # Standalone asteroids (not part of solar system hierarchy)
    for i, (x, z, r) in enumerate([(-3, 5, 0.3), (7, -2, 0.35), (2, -7, 0.25)]):
        asteroid = SceneNode(f"Asteroid{i+1}", translate_4x4(x, 0.5, z))
        asteroid.geometry = SphereGeometry(radius=r, color=np.array([0.5, 0.4, 0.3]))
        root.add_child(asteroid)

    # Propagate transforms
    root.update_transforms()

    # Collect all geometry nodes
    leaf_nodes = root.get_all_leaf_nodes()

    return root, leaf_nodes


# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------

def visualize_scene_graph(root: SceneNode, leaf_nodes: List[SceneNode],
                           bvh_root: BVHNode):
    """Visualize the scene from top-down (XZ plane) with BVH boxes.

    Why top-down view?  Our solar system is roughly planar.  A top-down
    view shows the spatial relationships and BVH partitioning clearly.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Scene Graph with BVH Acceleration",
                 fontsize=14, fontweight='bold')

    # --- Panel 1: Scene graph structure (tree diagram) ---
    ax = axes[0]
    ax.set_title("Scene Graph Hierarchy", fontsize=11)
    ax.axis('off')

    def draw_tree(node, x, y, dx, depth=0):
        color = 'lightblue' if node.geometry else 'lightyellow'
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color,
                    edgecolor='gray', alpha=0.9)
        label = node.name
        if node.geometry:
            label += f"\n(r={node.geometry.radius})"
        ax.text(x, y, label, fontsize=7, ha='center', va='center',
                bbox=bbox)

        for i, child in enumerate(node.children):
            n = len(node.children)
            child_x = x - dx * (n - 1) / 2 + dx * i
            child_y = y - 1.2
            ax.plot([x, child_x], [y - 0.3, child_y + 0.3],
                    'k-', linewidth=0.8, alpha=0.5)
            draw_tree(child, child_x, child_y, dx * 0.5, depth + 1)

    draw_tree(root, 0, 5, 3.0)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-4, 6)

    # --- Panel 2: Top-down view with objects ---
    ax = axes[1]
    ax.set_title("Top-Down View (XZ Plane)", fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    for node in leaf_nodes:
        pos = node.world_transform[:3, 3]
        r = node.geometry.radius * max(
            np.linalg.norm(node.world_transform[:3, 0]),
            np.linalg.norm(node.world_transform[:3, 1]),
            np.linalg.norm(node.world_transform[:3, 2]),
        )
        circle = plt.Circle((pos[0], pos[2]), r,
                             color=node.geometry.color, alpha=0.7, zorder=5)
        ax.add_patch(circle)
        ax.text(pos[0], pos[2] + r + 0.3, node.name,
                fontsize=7, ha='center', color='gray')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    # --- Panel 3: Top-down view with BVH boxes ---
    ax = axes[2]
    ax.set_title("BVH Bounding Boxes", fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Draw objects
    for node in leaf_nodes:
        pos = node.world_transform[:3, 3]
        r = node.geometry.radius * max(
            np.linalg.norm(node.world_transform[:3, 0]),
            np.linalg.norm(node.world_transform[:3, 1]),
            np.linalg.norm(node.world_transform[:3, 2]),
        )
        circle = plt.Circle((pos[0], pos[2]), r,
                             color=node.geometry.color, alpha=0.5, zorder=5)
        ax.add_patch(circle)

    # Draw BVH boxes recursively
    depth_colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']

    def draw_bvh_boxes(bvh_node, depth=0):
        if not bvh_node.aabb.is_valid():
            return

        color = depth_colors[depth % len(depth_colors)]
        aabb = bvh_node.aabb
        # Project to XZ plane
        x = aabb.min_pt[0]
        z = aabb.min_pt[2]
        w = aabb.max_pt[0] - aabb.min_pt[0]
        h = aabb.max_pt[2] - aabb.min_pt[2]

        rect = patches.Rectangle((x, z), w, h,
                                  linewidth=max(2 - depth * 0.3, 0.5),
                                  edgecolor=color,
                                  facecolor='none',
                                  linestyle='--' if depth > 0 else '-',
                                  alpha=max(1 - depth * 0.15, 0.3),
                                  zorder=10 - depth)
        ax.add_patch(rect)

        if bvh_node.left:
            draw_bvh_boxes(bvh_node.left, depth + 1)
        if bvh_node.right:
            draw_bvh_boxes(bvh_node.right, depth + 1)

    draw_bvh_boxes(bvh_root)

    # Legend for BVH depths
    for d in range(min(4, len(depth_colors))):
        ax.plot([], [], color=depth_colors[d], linewidth=2,
                label=f'BVH depth {d}')
    ax.legend(fontsize=8, loc='upper right')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_12_scene_bvh.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 8. BVH traversal demo
# ---------------------------------------------------------------------------

def demo_bvh_traversal(leaf_nodes: List[SceneNode], bvh_root: BVHNode):
    """Demonstrate BVH ray traversal by shooting rays and counting tests.

    Compares the number of intersection tests with and without BVH to
    quantify the acceleration.
    """
    rng = np.random.RandomState(42)

    n_rays = 1000
    brute_force_tests = 0
    bvh_tests = 0

    # Count BVH node visits by wrapping the intersection function
    bvh_visit_count = [0]
    original_intersect = AABB.intersect_ray

    def counting_intersect(self, origin, inv_dir, t_min=0, t_max=np.inf):
        bvh_visit_count[0] += 1
        return original_intersect(self, origin, inv_dir, t_min, t_max)

    # Shoot random rays
    hits_brute = 0
    hits_bvh = 0

    for _ in range(n_rays):
        # Random ray origin and direction
        origin = rng.uniform(-15, 15, size=3)
        origin[1] = rng.uniform(-2, 2)
        direction = rng.randn(3)
        direction = direction / np.linalg.norm(direction)

        # Brute force: test every object
        for node in leaf_nodes:
            brute_force_tests += 1
            if node.geometry:
                t = node.geometry.intersect_ray(origin, direction,
                                                 node.world_transform)
                if t is not None:
                    hits_brute += 1
                    break  # Just counting hits, not finding closest

        # BVH traversal
        bvh_visit_count[0] = 0
        AABB.intersect_ray = counting_intersect
        result = intersect_bvh(bvh_root, origin, direction)
        AABB.intersect_ray = original_intersect
        bvh_tests += bvh_visit_count[0]

        if result is not None:
            hits_bvh += 1

    print(f"\n  Ray intersection comparison ({n_rays} rays, {len(leaf_nodes)} objects):")
    print(f"  Brute force: {brute_force_tests:,} intersection tests "
          f"({brute_force_tests/n_rays:.1f} per ray)")
    print(f"  BVH:         {bvh_tests:,} AABB tests "
          f"({bvh_tests/n_rays:.1f} per ray)")
    print(f"  Speedup:     {brute_force_tests/max(bvh_tests,1):.1f}x fewer tests")
    print(f"  Hits (brute): {hits_brute}, Hits (BVH): {hits_bvh}")

    # Visualize the comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['Brute Force', 'BVH']
    tests_per_ray = [brute_force_tests / n_rays, bvh_tests / n_rays]
    bars = ax.bar(categories, tests_per_ray,
                   color=['#e74c3c', '#2ecc71'], width=0.5)

    ax.set_ylabel('Average Tests per Ray')
    ax.set_title('BVH Acceleration: Tests per Ray Comparison',
                 fontsize=13, fontweight='bold')

    for bar, val in zip(bars, tests_per_ray):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

    ax.set_ylim(0, max(tests_per_ray) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_12_bvh_comparison.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 9. Scene graph transform propagation demo
# ---------------------------------------------------------------------------

def demo_transform_propagation():
    """Visualize how local transforms propagate through the hierarchy.

    Shows the same scene with different root rotations to demonstrate
    that all children move together when the parent transforms.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Transform Propagation: Rotating the Root Affects Everything",
                 fontsize=14, fontweight='bold')

    for ax, angle in zip(axes, [0, 45, 90]):
        root, leaf_nodes = build_demo_scene()

        # Modify the solar system rotation
        for child in root.children:
            if child.name == "SolarSystem":
                child.local_transform = rotate_y_4x4(angle)

        root.update_transforms()

        ax.set_title(f"SolarSystem rotation = {angle} deg", fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        for node in leaf_nodes:
            pos = node.world_transform[:3, 3]
            r = node.geometry.radius * max(
                np.linalg.norm(node.world_transform[:3, 0]),
                np.linalg.norm(node.world_transform[:3, 1]),
                np.linalg.norm(node.world_transform[:3, 2]),
            )
            circle = plt.Circle((pos[0], pos[2]), r,
                                 color=node.geometry.color, alpha=0.7)
            ax.add_patch(circle)
            ax.text(pos[0], pos[2] + r + 0.3, node.name,
                    fontsize=6, ha='center', color='gray')

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_12_transform_propagation.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Scene Graph with BVH Acceleration")
    print("=" * 60)

    print("\n[1/3] Building scene graph and BVH...")
    root, leaf_nodes = build_demo_scene()

    print(f"  Scene graph nodes with geometry: {len(leaf_nodes)}")
    for node in leaf_nodes:
        pos = node.world_transform[:3, 3]
        print(f"    {node.name}: world pos = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    bvh_root = build_bvh(leaf_nodes, max_leaf_size=1)
    print(f"  BVH built successfully")

    print("\n[2/3] Visualizing scene graph and BVH...")
    visualize_scene_graph(root, leaf_nodes, bvh_root)

    print("\n[3/3] BVH traversal performance comparison...")
    demo_bvh_traversal(leaf_nodes, bvh_root)

    print("\n[Bonus] Transform propagation demo...")
    demo_transform_propagation()

    print("\nDone!")


if __name__ == "__main__":
    main()
