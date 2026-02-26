# 04. Rasterization

[&larr; Previous: 3D Transformations and Projections](03_3D_Transformations_and_Projections.md) | [Next: Shading Models &rarr;](05_Shading_Models.md)

---

## Learning Objectives

1. Understand line rasterization using both DDA and Bresenham's algorithm
2. Rasterize triangles using edge functions and barycentric coordinates
3. Implement the Z-buffer (depth buffer) algorithm for hidden surface removal
4. Interpolate vertex attributes (color, UV, normals) across triangle surfaces
5. Explain perspective-correct interpolation and why it matters
6. Understand anti-aliasing techniques: MSAA and FXAA at a conceptual level
7. Build a complete software rasterizer from scratch in Python
8. Appreciate the difference between CPU rasterization (for learning) and GPU rasterization (for performance)

---

## Why This Matters

Rasterization is the bridge between the continuous, mathematical world of geometry and the discrete, pixel-based world of your screen. Every triangle that appears in a 3D game or visualization was rasterized -- converted from three vertex positions into a set of colored pixels. Understanding rasterization deeply reveals *why* graphics APIs work the way they do, *why* certain artifacts appear (aliasing, z-fighting), and *how* GPUs achieve their extraordinary throughput. Building a software rasterizer is one of the most illuminating exercises in computer graphics education.

---

## 1. From Triangles to Pixels

After the vertex processing stage (Lessons 02-03), we have triangles defined by screen-space vertex positions. Rasterization answers the question: **which pixels does each triangle cover, and what color should they be?**

```
Input:  Triangle with 3 screen-space vertices
        Each vertex has: position (x, y, z), color, UV, normal, ...

Process: For each pixel in the triangle's bounding box:
           Is this pixel inside the triangle?
           If yes: interpolate vertex attributes, generate a fragment

Output: Stream of fragments (candidate pixels with interpolated attributes)
```

We start with the simpler case -- lines -- then build up to triangles.

---

## 2. Line Rasterization

### 2.1 DDA (Digital Differential Analyzer)

The simplest line-drawing algorithm. Given endpoints $(x_0, y_0)$ and $(x_1, y_1)$:

1. Compute the slope: $m = \frac{y_1 - y_0}{x_1 - x_0}$
2. Step along $x$ (if $|m| \leq 1$) or $y$ (if $|m| > 1$)
3. At each step, round the other coordinate to the nearest integer

```python
import numpy as np

def draw_line_dda(x0, y0, x1, y1):
    """
    DDA line rasterization.

    Simple and intuitive, but uses floating-point arithmetic at every step.
    Bresenham's algorithm (below) avoids this with integer-only operations.
    """
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))

    if steps == 0:
        return [(round(x0), round(y0))]

    # How much to increment x and y at each step
    x_inc = dx / steps
    y_inc = dy / steps

    pixels = []
    x, y = x0, y0

    for _ in range(int(steps) + 1):
        pixels.append((round(x), round(y)))
        x += x_inc
        y += y_inc

    return pixels
```

### 2.2 Bresenham's Line Algorithm

Bresenham's algorithm is the classic integer-only line rasterization method. It avoids all floating-point operations by using an error accumulator.

**Key idea**: At each step, we choose between two candidate pixels. The decision is based on which pixel center is closer to the ideal line.

For a line with slope $0 \leq m \leq 1$ (other octants handled by symmetry):

```python
def draw_line_bresenham(x0, y0, x1, y1):
    """
    Bresenham's line algorithm -- integer arithmetic only.

    Why this matters: in the early days of graphics, floating-point
    operations were extremely expensive. Bresenham's insight was that
    the decision at each pixel can be made with only integer addition
    and comparison. Modern GPUs still use variants of this idea.
    """
    pixels = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1  # Step direction for x
    sy = 1 if y0 < y1 else -1  # Step direction for y

    # Determine which axis is "steep" (more pixels along that axis)
    steep = dy > dx
    if steep:
        dx, dy = dy, dx

    # Decision variable: starts at 2*dy - dx
    # Positive means step in the minor axis direction
    error = 2 * dy - dx

    x, y = x0, y0

    for _ in range(dx + 1):
        pixels.append((x, y))

        if error > 0:
            # Step in both major and minor directions
            if steep:
                x += sx
            else:
                y += sy
            error -= 2 * dx

        # Always step in the major direction
        if steep:
            y += sy
        else:
            x += sx
        error += 2 * dy

    return pixels


# Demonstration
print("DDA line from (0,0) to (8,3):")
print(draw_line_dda(0, 0, 8, 3))

print("\nBresenham line from (0,0) to (8,3):")
print(draw_line_bresenham(0, 0, 8, 3))
```

### 2.3 Comparison

| Aspect | DDA | Bresenham |
|--------|-----|-----------|
| Arithmetic | Floating-point | Integer only |
| Speed | Slower (fp division) | Faster (additions only) |
| Accuracy | Rounding errors accumulate | Exact |
| Implementation | Simpler | Slightly more complex |

---

## 3. Triangle Rasterization

Triangles are the fundamental primitive in 3D graphics. Almost all 3D meshes are composed of triangles because:
- Any polygon can be decomposed into triangles
- Triangles are always planar (3 points define a plane)
- Triangle rasterization is highly parallelizable on GPUs

### 3.1 Edge Functions

The **edge function** approach is the method used by modern GPUs. For a triangle with vertices $A$, $B$, $C$ (in screen space), we define three edge functions:

$$E_{AB}(P) = (P_x - A_x)(B_y - A_y) - (P_y - A_y)(B_x - A_x)$$

$$E_{BC}(P) = (P_x - B_x)(C_y - B_y) - (P_y - B_y)(C_x - B_x)$$

$$E_{CA}(P) = (P_x - C_x)(A_y - C_y) - (P_y - C_y)(A_x - C_x)$$

Each edge function computes the signed area of the parallelogram formed by the edge vector and the vector from the edge start to point $P$. The sign tells us which side of the edge the point is on.

A point $P$ is **inside the triangle** if all three edge functions have the same sign (all positive or all negative, depending on winding order).

> **Why edge functions?** They are trivially parallelizable: each pixel's test is independent of all others. GPUs evaluate edge functions for many pixels simultaneously.

```python
def edge_function(a, b, p):
    """
    Compute the edge function for edge A->B evaluated at point P.

    Returns:
        Positive if P is to the left of edge A->B (CCW winding)
        Zero if P is exactly on the edge
        Negative if P is to the right

    This is equivalent to the z-component of the cross product
    (B-A) x (P-A), which is twice the signed area of triangle ABP.
    """
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
```

### 3.2 Barycentric Coordinates

The three edge function values directly give us **barycentric coordinates**:

$$\alpha = \frac{E_{BC}(P)}{E_{BC}(A)}, \quad \beta = \frac{E_{CA}(P)}{E_{CA}(B)}, \quad \gamma = \frac{E_{AB}(P)}{E_{AB}(C)}$$

Since $E_{BC}(A)$ is twice the area of triangle $ABC$, we can normalize:

$$\text{area} = E_{BC}(A) = (A_x - B_x)(C_y - B_y) - (A_y - B_y)(C_x - B_x)$$

$$\alpha = \frac{E_{BC}(P)}{\text{area}}, \quad \beta = \frac{E_{CA}(P)}{\text{area}}, \quad \gamma = 1 - \alpha - \beta$$

Barycentric coordinates are the "weights" indicating how much each vertex influences a given point:
- At vertex $A$: $(\alpha, \beta, \gamma) = (1, 0, 0)$
- At the centroid: $(\alpha, \beta, \gamma) = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$
- Outside the triangle: at least one coordinate is negative

### 3.3 Basic Triangle Rasterization

```python
def rasterize_triangle_basic(v0, v1, v2, width, height):
    """
    Rasterize a triangle using edge functions and barycentric coordinates.

    Parameters:
        v0, v1, v2: vertex positions as (x, y) tuples
        width, height: framebuffer dimensions

    Returns:
        List of (x, y, alpha, beta, gamma) for each covered pixel
    """
    fragments = []

    # Compute bounding box (no need to test pixels outside it)
    min_x = max(0, int(min(v0[0], v1[0], v2[0])))
    max_x = min(width - 1, int(max(v0[0], v1[0], v2[0])) + 1)
    min_y = max(0, int(min(v0[1], v1[1], v2[1])))
    max_y = min(height - 1, int(max(v0[1], v1[1], v2[1])) + 1)

    # Twice the signed area of the triangle (used for normalization)
    area = edge_function(v0, v1, v2)

    if abs(area) < 1e-10:
        return fragments  # Degenerate triangle (zero area)

    # Test each pixel center in the bounding box
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = (x + 0.5, y + 0.5)  # Pixel center

            # Evaluate edge functions
            w0 = edge_function(v1, v2, p)  # Opposite vertex v0
            w1 = edge_function(v2, v0, p)  # Opposite vertex v1
            w2 = edge_function(v0, v1, p)  # Opposite vertex v2

            # Inside test: all edge functions must be non-negative (CCW winding)
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Compute barycentric coordinates
                alpha = w0 / area
                beta = w1 / area
                gamma = w2 / area  # = 1 - alpha - beta

                fragments.append((x, y, alpha, beta, gamma))

    return fragments
```

### 3.4 Fill Rules

When triangles share edges (common in meshes), pixels exactly on the edge could be claimed by both triangles. The **top-left fill rule** resolves this:
- A pixel on a **top edge** (horizontal edge at the top of the triangle) belongs to the triangle
- A pixel on a **left edge** (edge going upward) belongs to the triangle
- All other edge pixels are excluded

This ensures every shared edge pixel is drawn exactly once.

---

## 4. The Z-Buffer Algorithm

### 4.1 The Hidden Surface Problem

When multiple triangles overlap at the same pixel, we need to determine which one is visible. The **Z-buffer** (depth buffer) is an elegantly simple solution.

### 4.2 Algorithm

Maintain a 2D array `depth_buffer[x][y]` initialized to the maximum depth value (far plane). For each fragment:

$$\text{if } z_{\text{fragment}} < \text{depth\_buffer}[x][y]: \quad \text{update both color and depth buffers}$$

The fragment with the smallest $z$ (closest to the camera) wins.

```python
def rasterize_with_zbuffer(triangles, width, height):
    """
    Complete rasterizer with Z-buffer hidden surface removal.

    Parameters:
        triangles: list of (v0, v1, v2) where each vertex is
                   {'pos': (x, y, z), 'color': (r, g, b)}
        width, height: framebuffer dimensions

    Returns:
        color_buffer: HxW array of RGB colors
        depth_buffer: HxW array of depth values
    """
    # Initialize buffers
    color_buffer = np.zeros((height, width, 3), dtype=float)
    depth_buffer = np.full((height, width), float('inf'))

    for tri_idx, (v0, v1, v2) in enumerate(triangles):
        p0, p1, p2 = v0['pos'], v1['pos'], v2['pos']
        c0, c1, c2 = v0['color'], v1['color'], v2['color']

        # Bounding box
        min_x = max(0, int(min(p0[0], p1[0], p2[0])))
        max_x = min(width - 1, int(max(p0[0], p1[0], p2[0])) + 1)
        min_y = max(0, int(min(p0[1], p1[1], p2[1])))
        max_y = min(height - 1, int(max(p0[1], p1[1], p2[1])) + 1)

        area = edge_function(p0[:2], p1[:2], p2[:2])
        if abs(area) < 1e-10:
            continue

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = (x + 0.5, y + 0.5)

                w0 = edge_function((p1[0], p1[1]), (p2[0], p2[1]), p)
                w1 = edge_function((p2[0], p2[1]), (p0[0], p0[1]), p)
                w2 = edge_function((p0[0], p0[1]), (p1[0], p1[1]), p)

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Barycentric coordinates
                    alpha = w0 / area
                    beta = w1 / area
                    gamma = w2 / area

                    # Interpolate depth
                    z = alpha * p0[2] + beta * p1[2] + gamma * p2[2]

                    # Z-buffer test: keep the closest fragment
                    if z < depth_buffer[y, x]:
                        depth_buffer[y, x] = z

                        # Interpolate color
                        r = alpha * c0[0] + beta * c1[0] + gamma * c2[0]
                        g = alpha * c0[1] + beta * c1[1] + gamma * c2[1]
                        b = alpha * c0[2] + beta * c1[2] + gamma * c2[2]
                        color_buffer[y, x] = [r, g, b]

    return color_buffer, depth_buffer
```

### 4.3 Z-Buffer Properties

| Property | Description |
|----------|-------------|
| **Memory** | One depth value per pixel (typically 24 or 32 bits) |
| **Order-independent** | Triangles can be drawn in any order (for opaque objects) |
| **Simple** | Only comparison + write per fragment |
| **GPU-friendly** | Highly parallelizable |
| **Weakness** | Cannot handle transparency (requires sorting) |

### 4.4 Z-Fighting

When two surfaces are very close in depth, floating-point precision may cause them to alternate which one "wins" the depth test, producing a flickering, striped pattern called **z-fighting**.

Mitigation:
- Increase near plane distance (improves depth precision)
- Use polygon offset to slightly separate coincident surfaces
- Use logarithmic or reversed-Z depth buffer

---

## 5. Attribute Interpolation

### 5.1 Linear Interpolation

Given barycentric coordinates $(\alpha, \beta, \gamma)$ and per-vertex attribute values, any attribute can be interpolated:

$$\text{attr}(P) = \alpha \cdot \text{attr}_0 + \beta \cdot \text{attr}_1 + \gamma \cdot \text{attr}_2$$

This works for:
- Colors (smooth Gouraud shading)
- Texture coordinates (UV mapping)
- Normals (Phong interpolation for smooth lighting)
- Depth values

### 5.2 Perspective-Correct Interpolation

There is a subtle but critical problem: **linear interpolation in screen space is not correct for perspective projections**. After perspective projection, equal steps in screen space do not correspond to equal steps in world space (objects farther away are compressed more).

The correct formula divides each attribute by its vertex $w$ before interpolation, then divides the result:

$$\text{attr}_{\text{correct}}(P) = \frac{\alpha \cdot \frac{\text{attr}_0}{w_0} + \beta \cdot \frac{\text{attr}_1}{w_1} + \gamma \cdot \frac{\text{attr}_2}{w_2}}{\alpha \cdot \frac{1}{w_0} + \beta \cdot \frac{1}{w_1} + \gamma \cdot \frac{1}{w_2}}$$

Where $w_0, w_1, w_2$ are the $w$-components from the clip-space positions of the three vertices.

```python
def perspective_correct_interpolation(alpha, beta, gamma,
                                       attr0, attr1, attr2,
                                       w0, w1, w2):
    """
    Perspective-correct attribute interpolation.

    Without this correction, textures appear to "swim" on surfaces
    when viewed at oblique angles -- a very noticeable artifact.

    The key insight: in screen space, equal pixel distances do NOT
    correspond to equal world-space distances due to perspective.
    Dividing by w before interpolation accounts for this non-linearity.
    """
    # Interpolate attr/w and 1/w separately
    attr_over_w = (alpha * attr0 / w0 +
                   beta * attr1 / w1 +
                   gamma * attr2 / w2)

    one_over_w = (alpha * (1.0 / w0) +
                  beta * (1.0 / w1) +
                  gamma * (1.0 / w2))

    # Recover the correctly interpolated attribute
    return attr_over_w / one_over_w
```

**Visual comparison**: On a textured quad viewed in perspective:
- **Without correction**: Checkerboard texture appears bent/warped along the diagonal
- **With correction**: Checkerboard appears straight and evenly spaced, as expected

---

## 6. Complete Software Rasterizer

```python
"""
A complete software rasterizer demonstrating the concepts from this lesson.

This renders a colored triangle (or multiple triangles) to a pixel buffer
using edge functions, barycentric coordinates, Z-buffer, and
perspective-correct interpolation.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# Framebuffer class
# ═══════════════════════════════════════════════════════════════

class Framebuffer:
    """Manages color and depth buffers."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Color buffer: RGBA (4 channels), float [0, 1]
        self.color = np.zeros((height, width, 4), dtype=float)
        self.color[:, :, 3] = 1.0  # Alpha = 1 (opaque background)
        # Depth buffer: initialized to far plane (1.0 in NDC)
        self.depth = np.ones((height, width), dtype=float)

    def clear(self, color=(0.1, 0.1, 0.1)):
        """Clear buffers to default values."""
        self.color[:, :, 0] = color[0]
        self.color[:, :, 1] = color[1]
        self.color[:, :, 2] = color[2]
        self.depth[:] = 1.0

    def set_pixel(self, x, y, z, color):
        """
        Write a pixel if it passes the depth test.

        Returns True if the pixel was written (passed depth test).
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        if z < self.depth[y, x]:
            self.depth[y, x] = z
            self.color[y, x, :3] = np.clip(color, 0, 1)
            return True
        return False

    def to_image(self):
        """Convert to uint8 image for display/saving."""
        return (np.clip(self.color[:, :, :3], 0, 1) * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# Vertex structure
# ═══════════════════════════════════════════════════════════════

class Vertex:
    """A vertex with position, color, UV, and clip-space w."""
    def __init__(self, x, y, z, w=1.0, r=1.0, g=1.0, b=1.0, u=0.0, v=0.0):
        self.pos = np.array([x, y, z], dtype=float)
        self.w = w  # Clip-space w (needed for perspective-correct interpolation)
        self.color = np.array([r, g, b], dtype=float)
        self.uv = np.array([u, v], dtype=float)


# ═══════════════════════════════════════════════════════════════
# Rasterizer
# ═══════════════════════════════════════════════════════════════

def edge_function(a, b, p):
    """Signed area of parallelogram formed by edge A->B and point P."""
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def rasterize_triangle(fb, v0, v1, v2, perspective_correct=True):
    """
    Rasterize a single triangle with Z-buffer and attribute interpolation.

    This is the core of the software rasterizer. A GPU executes this logic
    for millions of triangles per frame using thousands of parallel cores.

    Parameters:
        fb: Framebuffer instance
        v0, v1, v2: Vertex instances (screen-space positions)
        perspective_correct: use perspective-correct interpolation
    """
    p0, p1, p2 = v0.pos[:2], v1.pos[:2], v2.pos[:2]

    # Bounding box (clamped to screen)
    min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
    max_x = min(fb.width - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
    min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
    max_y = min(fb.height - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

    # Total signed area (for barycentric normalization)
    area = edge_function(p0, p1, p2)
    if abs(area) < 1e-10:
        return  # Skip degenerate triangles

    # Precompute 1/w for perspective correction
    inv_w0 = 1.0 / v0.w if perspective_correct else 1.0
    inv_w1 = 1.0 / v1.w if perspective_correct else 1.0
    inv_w2 = 1.0 / v2.w if perspective_correct else 1.0

    fragment_count = 0

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # Test pixel center
            px, py = x + 0.5, y + 0.5

            # Edge function evaluations
            w0 = edge_function(p1, p2, (px, py))
            w1 = edge_function(p2, p0, (px, py))
            w2 = edge_function(p0, p1, (px, py))

            # Inside test (all non-negative for CCW winding)
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Barycentric coordinates
                alpha = w0 / area
                beta = w1 / area
                gamma = w2 / area

                if perspective_correct:
                    # Perspective-correct interpolation
                    one_over_w = alpha * inv_w0 + beta * inv_w1 + gamma * inv_w2

                    # Interpolate color with perspective correction
                    color = (alpha * v0.color * inv_w0 +
                             beta * v1.color * inv_w1 +
                             gamma * v2.color * inv_w2) / one_over_w

                    # Interpolate depth with perspective correction
                    z = (alpha * v0.pos[2] * inv_w0 +
                         beta * v1.pos[2] * inv_w1 +
                         gamma * v2.pos[2] * inv_w2) / one_over_w
                else:
                    # Simple linear interpolation (incorrect for perspective)
                    color = (alpha * v0.color +
                             beta * v1.color +
                             gamma * v2.color)
                    z = (alpha * v0.pos[2] +
                         beta * v1.pos[2] +
                         gamma * v2.pos[2])

                # Depth test and pixel write
                fb.set_pixel(x, y, z, color)
                fragment_count += 1

    return fragment_count


# ═══════════════════════════════════════════════════════════════
# Demo: render two overlapping colored triangles
# ═══════════════════════════════════════════════════════════════

def main():
    width, height = 320, 240
    fb = Framebuffer(width, height)
    fb.clear(color=(0.05, 0.05, 0.1))  # Dark blue background

    # Triangle 1: Red-Green-Blue, closer to camera (z=0.3)
    t1_v0 = Vertex(80,  30,  0.3, r=1, g=0, b=0)  # Red (top)
    t1_v1 = Vertex(30,  200, 0.3, r=0, g=1, b=0)  # Green (bottom-left)
    t1_v2 = Vertex(200, 180, 0.3, r=0, g=0, b=1)  # Blue (bottom-right)

    # Triangle 2: Yellow-Cyan-Magenta, farther from camera (z=0.6)
    t2_v0 = Vertex(160, 20,  0.6, r=1, g=1, b=0)  # Yellow (top)
    t2_v1 = Vertex(100, 220, 0.6, r=0, g=1, b=1)  # Cyan (bottom-left)
    t2_v2 = Vertex(290, 150, 0.6, r=1, g=0, b=1)  # Magenta (bottom-right)

    # Render: draw the far triangle first, then the near triangle
    # Z-buffer ensures correct occlusion regardless of draw order
    n1 = rasterize_triangle(fb, t2_v0, t2_v1, t2_v2, perspective_correct=False)
    n2 = rasterize_triangle(fb, t1_v0, t1_v1, t1_v2, perspective_correct=False)

    print(f"Triangle 1 (far): {n1} fragments")
    print(f"Triangle 2 (near): {n2} fragments")
    print(f"Total pixels shaded: {n1 + n2}")
    print(f"Pixels that passed depth test: {np.sum(fb.depth < 1.0)}")

    # Save result
    try:
        from PIL import Image
        img = Image.fromarray(fb.to_image())
        img.save('rasterizer_output.png')
        print("Saved rasterizer_output.png")
    except ImportError:
        print("Install Pillow to save the image: pip install Pillow")
        print("Framebuffer shape:", fb.color.shape)


if __name__ == "__main__":
    main()
```

---

## 7. Anti-Aliasing

### 7.1 The Aliasing Problem

Rasterization produces **aliasing** artifacts: jagged edges (the "staircase" effect) along triangle boundaries. This occurs because we are sampling a continuous shape at discrete pixel locations -- the pixel grid cannot perfectly represent diagonal or curved edges.

Mathematically, aliasing is a violation of the **Nyquist sampling theorem**: the geometric signal (sharp triangle edges) contains arbitrarily high frequencies, but our pixel grid samples at a fixed rate.

### 7.2 Supersampling (SSAA)

The brute-force solution: render at a higher resolution and downsample.

For 4x SSAA: render at $2W \times 2H$, then average every $2 \times 2$ block into one pixel. This is extremely effective but expensive (4x the fragment shader cost).

### 7.3 Multi-Sample Anti-Aliasing (MSAA)

**MSAA** is a smarter version of supersampling. Instead of running the fragment shader for every subsample, it:

1. Tests **coverage** at multiple sample points within each pixel (e.g., 4 points for 4x MSAA)
2. Runs the **fragment shader only once** per pixel (at the pixel center)
3. Writes the shader result to all covered subsamples
4. Resolves: averages the subsamples to produce the final pixel color

```
┌─────────────────────┐
│  Pixel with 4x MSAA │
│                      │
│   ●         ●       │   ● = sample point
│       ◆             │   ◆ = pixel center (shader runs here)
│                      │
│   ●         ●       │   Triangle covers 3 of 4 samples:
│                      │   Final color = 75% triangle + 25% background
└─────────────────────┘
```

**Why MSAA is efficient**: The fragment shader (the expensive part) runs only once per pixel, not once per sample. Coverage testing (the cheap part) is done per sample.

```python
def rasterize_triangle_msaa(fb, v0, v1, v2, samples=4):
    """
    Triangle rasterization with MSAA (Multi-Sample Anti-Aliasing).

    MSAA evaluates coverage at multiple sub-pixel positions but
    runs the shader only once. The final pixel color is the average
    weighted by coverage.

    This is a simplified demonstration -- real MSAA implementations
    store per-sample depth and color in a larger buffer that is
    "resolved" (averaged) at the end of the frame.
    """
    # 4x MSAA sample pattern (Rotated Grid pattern)
    # These offsets are within the pixel, relative to pixel center
    if samples == 4:
        sample_offsets = [
            (-0.25, -0.125),
            (0.25, -0.375),
            (-0.125, 0.375),
            (0.375, 0.125)
        ]
    else:
        # Fallback: regular grid
        n = int(np.sqrt(samples))
        step = 1.0 / (n + 1)
        sample_offsets = [(step * (i + 1) - 0.5, step * (j + 1) - 0.5)
                          for i in range(n) for j in range(n)]

    p0, p1, p2 = v0.pos[:2], v1.pos[:2], v2.pos[:2]

    min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
    max_x = min(fb.width - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
    min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
    max_y = min(fb.height - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

    area = edge_function(p0, p1, p2)
    if abs(area) < 1e-10:
        return

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # Test coverage at each sample point
            covered = 0
            for dx, dy in sample_offsets:
                sx, sy = x + 0.5 + dx, y + 0.5 + dy
                w0 = edge_function(p1, p2, (sx, sy))
                w1 = edge_function(p2, p0, (sx, sy))
                w2 = edge_function(p0, p1, (sx, sy))
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    covered += 1

            if covered > 0:
                # Shade at pixel center (fragment shader runs once)
                px, py = x + 0.5, y + 0.5
                w0 = edge_function(p1, p2, (px, py))
                w1 = edge_function(p2, p0, (px, py))
                w2 = edge_function(p0, p1, (px, py))

                alpha = w0 / area
                beta = w1 / area
                gamma = 1 - alpha - beta

                color = (alpha * v0.color + beta * v1.color + gamma * v2.color)
                z = alpha * v0.pos[2] + beta * v1.pos[2] + gamma * v2.pos[2]

                # Weight by coverage fraction
                coverage = covered / len(sample_offsets)
                blended = color * coverage + fb.color[y, x, :3] * (1 - coverage)

                fb.set_pixel(x, y, z, blended)
```

### 7.4 FXAA (Fast Approximate Anti-Aliasing)

**FXAA** is a **post-processing** technique -- it works on the final rendered image, not during rasterization:

1. Detect edges by comparing neighboring pixel luminances
2. Along detected edges, blend pixels to smooth the staircase
3. Fast (single full-screen pass) and trivial to implement
4. Works with any rendering technique (forward, deferred, ray tracing)

**Trade-off**: FXAA can blur fine details because it cannot distinguish between geometric edges (which should be smoothed) and texture edges (which should not).

### 7.5 Anti-Aliasing Comparison

| Method | Quality | Performance Cost | Memory Cost | Notes |
|--------|---------|-----------------|-------------|-------|
| None | Jagged edges | 1x | 1x | Baseline |
| SSAA 4x | Excellent | 4x | 4x | Brute force |
| MSAA 4x | Very good | ~1.2x shader, 4x coverage | 4x depth/color | GPU hardware support |
| FXAA | Good | ~1 ms post-pass | None extra | Blurs some detail |
| TAA | Very good | ~1 ms + history buffer | 2x color | Uses temporal data |

---

## 8. Performance Considerations

### 8.1 Why GPUs Are Fast at Rasterization

The rasterization algorithm is embarrassingly parallel:
- Each triangle can be rasterized independently
- Each pixel within a triangle can be tested independently
- Edge function evaluation is just multiplication and addition

Modern GPUs partition the screen into **tiles** and process many triangles per tile in parallel. A high-end GPU can rasterize billions of triangles per second.

### 8.2 Optimizations in Real Rasterizers

- **Hierarchical rasterization**: Test blocks of pixels before individual pixels
- **Early depth rejection**: Skip fragments that are behind already-rendered geometry
- **SIMD edge function evaluation**: Evaluate edge functions for 4, 8, or 16 pixels at once
- **Backface culling**: Skip triangles facing away from the camera (saves ~50% of work)
- **Tile-based rendering**: Process one screen tile at a time to maximize cache hits (common in mobile GPUs)

---

## Summary

| Concept | Description |
|---------|-------------|
| **DDA** | Simple line drawing using floating-point increments |
| **Bresenham** | Integer-only line drawing using error accumulation |
| **Edge functions** | Determine if a point is inside a triangle via signed area |
| **Barycentric coords** | Weights for interpolating vertex attributes across a triangle |
| **Z-buffer** | Per-pixel depth comparison for hidden surface removal |
| **Perspective-correct** | Divide attributes by $w$ before interpolation, then correct |
| **MSAA** | Multi-sample coverage test with single shader evaluation |
| **FXAA** | Post-processing edge blur based on luminance detection |

**Key takeaways**:
- Rasterization converts continuous triangles to discrete fragments using edge functions
- Barycentric coordinates enable smooth interpolation of any per-vertex attribute
- The Z-buffer elegantly solves hidden surface removal with $O(n)$ per-pixel cost
- Perspective-correct interpolation is essential for correct texture mapping
- Anti-aliasing trades performance for smoother edges; MSAA is the standard real-time choice
- GPUs exploit the massive parallelism inherent in rasterization to achieve real-time performance

---

## Exercises

1. **Bresenham Implementation**: Implement Bresenham's algorithm for all eight octants (the version above handles a restricted case). Test it by drawing lines at angles $0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°$ and verify they look correct.

2. **Edge Function Visualization**: For a triangle with vertices $(50, 20)$, $(200, 150)$, $(30, 180)$, compute and visualize the edge function values across a 256x256 grid. Use three different colors for the three edge functions and display regions where each is positive/negative.

3. **Z-Buffer Artifact**: Create a scene where two triangles are nearly coplanar (Z difference of $0.0001$) at different distances from the camera. Demonstrate z-fighting by rendering with 16-bit vs 32-bit depth precision.

4. **Perspective-Correct Interpolation**: Render a textured quad (two triangles) with and without perspective correction. Use a checkerboard pattern and view the quad at an oblique angle to clearly see the difference.

5. **Software Rasterizer Extension**: Extend the complete rasterizer to support: (a) wireframe rendering (draw only triangle edges), (b) multiple triangles forming a simple mesh (e.g., a cube), (c) flat shading using the triangle's face normal.

6. **MSAA Analysis**: Render a triangle edge at 1x, 2x, 4x, and 8x MSAA. For each, count the number of pixels that have partial coverage (not fully inside or fully outside the triangle). How does this count change with the MSAA level?

---

## Further Reading

1. Marschner, S. & Shirley, P. *Fundamentals of Computer Graphics* (5th ed.), Ch. 8 -- "Rasterization"
2. Pineda, J. "A Parallel Algorithm for Polygon Rasterization" (1988) -- The original edge function paper
3. [Scratchapixel - Rasterization](https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation) -- Detailed tutorial with code
4. [A Trip through the Graphics Pipeline (Fabian Giesen)](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/) -- Excellent deep dive into how GPUs actually rasterize
5. [Learn OpenGL - Anti Aliasing](https://learnopengl.com/Advanced-OpenGL/Anti-Aliasing) -- MSAA in practice with OpenGL
