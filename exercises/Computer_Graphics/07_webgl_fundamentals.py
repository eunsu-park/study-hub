"""
Lesson 07 - WebGL Fundamentals
Topic: Computer Graphics

Exercises covering WebGL concepts adapted to Python: vertex buffers, index buffers,
shader data flow (attributes, uniforms, varyings), MVP matrix pipeline, rendering
multiple objects, and performance analysis. Since WebGL is a JavaScript/browser API,
these exercises implement the equivalent concepts in Python using numpy and matplotlib
to demonstrate the same underlying graphics principles.
"""

import numpy as np

matplotlib_available = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    matplotlib_available = False


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════

def make_identity():
    """Return a 4x4 identity matrix."""
    return np.eye(4, dtype=np.float32)


def make_translation(tx, ty, tz):
    """Create a 4x4 translation matrix."""
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def make_rotation_x(angle_deg):
    """Create a 4x4 rotation matrix about the X axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def make_rotation_y(angle_deg):
    """Create a 4x4 rotation matrix about the Y axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def make_rotation_z(angle_deg):
    """Create a 4x4 rotation matrix about the Z axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def make_scale(sx, sy, sz):
    """Create a 4x4 scaling matrix."""
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def look_at(eye, target, up):
    """Build a view matrix (world -> camera) using the look-at construction."""
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, forward)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[0, 3] = -np.dot(right, eye)
    m[1, 3] = -np.dot(true_up, eye)
    m[2, 3] = np.dot(forward, eye)
    return m


def perspective(fov_deg, aspect, near, far):
    """Build a perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def project_vertices(vertices, mvp, width, height):
    """
    Apply MVP matrix to vertices and perform perspective divide + viewport transform.
    Returns screen-space coordinates (x, y) and clip-space w for each vertex.
    """
    n = len(vertices)
    screen = np.zeros((n, 2), dtype=np.float32)
    w_values = np.zeros(n, dtype=np.float32)

    for i, v in enumerate(vertices):
        clip = mvp @ np.array([v[0], v[1], v[2], 1.0], dtype=np.float32)
        w = clip[3]
        w_values[i] = w
        if abs(w) < 1e-6:
            w = 1e-6
        ndc = clip[:3] / w
        screen[i, 0] = (ndc[0] + 1.0) * 0.5 * width
        screen[i, 1] = (1.0 - ndc[1]) * 0.5 * height  # flip Y for screen space

    return screen, w_values


# ═══════════════════════════════════════════════════════════════
# Exercise functions
# ═══════════════════════════════════════════════════════════════

def exercise_1():
    """
    Exercise 1: Hello Triangle (Vertex Buffer Simulation)

    Simulate the WebGL vertex buffer and attribute setup in Python.
    Create an interleaved vertex buffer with position (x, y) and color (r, g, b),
    then extract and render the triangle using the same stride/offset logic
    that WebGL's vertexAttribPointer uses.
    """
    # Interleaved vertex data: [x, y, r, g, b] per vertex (same as WebGL lesson)
    # This mimics a Float32Array in WebGL
    vertex_buffer = np.array([
        # Position     Color
         0.0,  0.5,   1.0, 0.0, 0.0,  # Top vertex (red)
        -0.5, -0.5,   0.0, 1.0, 0.0,  # Bottom-left (green)
         0.5, -0.5,   0.0, 0.0, 1.0,  # Bottom-right (blue)
    ], dtype=np.float32)

    # WebGL attribute layout simulation:
    # stride = 5 floats (20 bytes), position offset = 0, color offset = 2
    stride = 5
    num_vertices = len(vertex_buffer) // stride

    print(f"  Vertex buffer: {len(vertex_buffer)} floats ({len(vertex_buffer) * 4} bytes)")
    print(f"  Stride: {stride} floats ({stride * 4} bytes per vertex)")
    print(f"  Number of vertices: {num_vertices}")
    print()

    # Extract attributes using stride/offset (like vertexAttribPointer)
    positions = np.zeros((num_vertices, 2), dtype=np.float32)
    colors = np.zeros((num_vertices, 3), dtype=np.float32)

    for i in range(num_vertices):
        base = i * stride
        # Position: size=2, offset=0
        positions[i] = vertex_buffer[base:base + 2]
        # Color: size=3, offset=2
        colors[i] = vertex_buffer[base + 2:base + 5]

    print("  Extracted attributes:")
    for i in range(num_vertices):
        print(f"    Vertex {i}: pos=({positions[i, 0]:.1f}, {positions[i, 1]:.1f}), "
              f"color=({colors[i, 0]:.1f}, {colors[i, 1]:.1f}, {colors[i, 2]:.1f})")

    # Rasterize with barycentric color interpolation
    width, height = 200, 200
    image = np.zeros((height, width, 3), dtype=np.float32)

    # Map NDC [-1,1] to pixel coordinates
    screen_pos = np.zeros_like(positions)
    screen_pos[:, 0] = (positions[:, 0] + 1.0) * 0.5 * width
    screen_pos[:, 1] = (1.0 - positions[:, 1]) * 0.5 * height

    p0, p1, p2 = screen_pos[0], screen_pos[1], screen_pos[2]
    c0, c1, c2 = colors[0], colors[1], colors[2]

    # Edge function for barycentric coordinates
    def edge_function(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    area = edge_function(p0, p1, p2)

    for y in range(height):
        for x in range(width):
            p = np.array([x + 0.5, y + 0.5])
            w0 = edge_function(p1, p2, p)
            w1 = edge_function(p2, p0, p)
            w2 = edge_function(p0, p1, p)

            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                w0 /= area
                w1 /= area
                w2 /= area
                # Interpolate color (this is what the fragment shader does in WebGL)
                color = w0 * c0 + w1 * c1 + w2 * c2
                image[y, x] = np.clip(color, 0.0, 1.0)

    if matplotlib_available:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(image)
        ax.set_title("Hello Triangle (Software Rasterized)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        plt.savefig("07_ex1_hello_triangle.png", dpi=100)
        plt.close()
        print("\n  Saved: 07_ex1_hello_triangle.png")

    print(f"\n  Non-zero pixels: {np.sum(np.any(image > 0, axis=2))}")
    print("  This demonstrates the WebGL pipeline: vertex buffer -> attribute extraction ->")
    print("  vertex processing -> rasterization -> fragment color interpolation.")


def exercise_2():
    """
    Exercise 2: Animated Rotation (Uniform Time Simulation)

    Simulate WebGL's requestAnimationFrame loop by computing multiple frames
    of a rotating triangle. Compare two approaches:
    (a) Rotating via a uniform matrix (model matrix rotation)
    (b) Rotating via vertex shader math (sin/cos in shader)
    Both produce the same result, but the matrix approach is more flexible.
    """
    # Triangle vertices in NDC
    vertices = np.array([
        [0.0, 0.5],
        [-0.5, -0.5],
        [0.5, -0.5],
    ], dtype=np.float32)

    # Simulate 8 frames at different time values
    frame_times = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    print("  Simulating 8 animation frames of a rotating triangle:")
    print()

    # Method A: Rotation via uniform matrix (like passing uModelMatrix)
    print("  Method A - Uniform rotation matrix:")
    method_a_results = []
    for i, t in enumerate(frame_times):
        angle = t  # radians
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ], dtype=np.float32)
        rotated = (rotation_matrix @ vertices.T).T
        method_a_results.append(rotated.copy())
        centroid = rotated.mean(axis=0)
        print(f"    Frame {i}: angle={np.degrees(angle):6.1f} deg, "
              f"centroid=({centroid[0]:+.3f}, {centroid[1]:+.3f})")

    # Method B: Rotation via vertex shader math (sin/cos per vertex)
    print("\n  Method B - Per-vertex sin/cos (in shader):")
    method_b_results = []
    for i, t in enumerate(frame_times):
        rotated = np.zeros_like(vertices)
        for j, v in enumerate(vertices):
            # This is what a vertex shader would do with uTime
            rotated[j, 0] = v[0] * np.cos(t) - v[1] * np.sin(t)
            rotated[j, 1] = v[0] * np.sin(t) + v[1] * np.cos(t)
        method_b_results.append(rotated.copy())

    # Verify both methods produce identical results
    max_diff = 0.0
    for a, b in zip(method_a_results, method_b_results):
        max_diff = max(max_diff, np.max(np.abs(a - b)))
    print(f"\n  Max difference between methods: {max_diff:.2e}")
    print("  Both methods are mathematically identical.")
    print()
    print("  Key insight: The matrix approach is preferred because:")
    print("  - Composable: M_rotation * M_translation * M_scale")
    print("  - A single mat4 uniform handles all transforms")
    print("  - The vertex shader stays simple: gl_Position = uMVP * vec4(pos, 1.0)")

    if matplotlib_available:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for i in range(8):
            ax = axes[i // 4, i % 4]
            tri = method_a_results[i]
            # Close the triangle for plotting
            tri_closed = np.vstack([tri, tri[0]])
            ax.fill(tri_closed[:, 0], tri_closed[:, 1], alpha=0.5, color='steelblue')
            ax.plot(tri_closed[:, 0], tri_closed[:, 1], 'k-', linewidth=1.5)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            ax.set_title(f"Frame {i} ({np.degrees(frame_times[i]):.0f} deg)")
            ax.grid(True, alpha=0.3)
        plt.suptitle("Animated Rotation: 8 Frames", fontsize=14)
        plt.tight_layout()
        plt.savefig("07_ex2_animated_rotation.png", dpi=100)
        plt.close()
        print("\n  Saved: 07_ex2_animated_rotation.png")


def exercise_3():
    """
    Exercise 3: Textured Quad (Index Buffer and UV Simulation)

    Simulate rendering a textured quad using an index buffer and UV coordinates.
    Demonstrates vertex reuse through indexed drawing (drawElements) and
    texture sampling in the fragment shader.
    """
    # Quad with interleaved vertex data: [x, y, u, v] per vertex
    # 4 unique vertices, 6 indices (2 triangles)
    vertex_buffer = np.array([
        # Position   UV
        -0.5, -0.5,  0.0, 0.0,  # Vertex 0: bottom-left
         0.5, -0.5,  1.0, 0.0,  # Vertex 1: bottom-right
         0.5,  0.5,  1.0, 1.0,  # Vertex 2: top-right
        -0.5,  0.5,  0.0, 1.0,  # Vertex 3: top-left
    ], dtype=np.float32)

    # Index buffer: 2 triangles forming a quad
    index_buffer = np.array([
        0, 1, 2,  # First triangle
        0, 2, 3,  # Second triangle (reuses vertices 0 and 2)
    ], dtype=np.uint16)

    stride = 4
    num_vertices = len(vertex_buffer) // stride
    num_indices = len(index_buffer)
    num_triangles = num_indices // 3

    print(f"  Vertex buffer: {num_vertices} vertices (4 unique)")
    print(f"  Index buffer: {num_indices} indices ({num_triangles} triangles)")
    print(f"  Memory savings: {num_vertices} vertices vs {num_indices} (non-indexed)")
    print(f"  Vertex reuse ratio: {num_indices / num_vertices:.1f}x")
    print()

    # Extract vertex attributes
    positions = np.zeros((num_vertices, 2))
    uvs = np.zeros((num_vertices, 2))
    for i in range(num_vertices):
        base = i * stride
        positions[i] = vertex_buffer[base:base + 2]
        uvs[i] = vertex_buffer[base + 2:base + 4]

    # Create a procedural texture (checkerboard)
    tex_size = 64
    texture = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    checker_size = 8
    for ty in range(tex_size):
        for tx in range(tex_size):
            if ((tx // checker_size) + (ty // checker_size)) % 2 == 0:
                texture[ty, tx] = [0.9, 0.7, 0.2]  # gold
            else:
                texture[ty, tx] = [0.2, 0.3, 0.6]  # blue

    def sample_texture(tex, u, v):
        """Sample texture at UV coordinates using nearest-neighbor."""
        h, w = tex.shape[:2]
        tx = int(np.clip(u * w, 0, w - 1))
        ty = int(np.clip((1.0 - v) * h, 0, h - 1))  # flip V
        return tex[ty, tx]

    # Rasterize the textured quad
    img_size = 256
    image = np.zeros((img_size, img_size, 3), dtype=np.float32)

    # Map NDC to screen
    screen_pos = np.zeros_like(positions)
    screen_pos[:, 0] = (positions[:, 0] + 1.0) * 0.5 * img_size
    screen_pos[:, 1] = (1.0 - positions[:, 1]) * 0.5 * img_size

    def edge_fn(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    # Process each triangle from index buffer (like drawElements)
    print("  Processing triangles from index buffer:")
    for tri_idx in range(num_triangles):
        i0 = index_buffer[tri_idx * 3]
        i1 = index_buffer[tri_idx * 3 + 1]
        i2 = index_buffer[tri_idx * 3 + 2]
        print(f"    Triangle {tri_idx}: indices [{i0}, {i1}, {i2}]")

        p0, p1, p2 = screen_pos[i0], screen_pos[i1], screen_pos[i2]
        uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]

        area = edge_fn(p0, p1, p2)
        if abs(area) < 1e-6:
            continue

        # Bounding box
        min_x = max(0, int(min(p0[0], p1[0], p2[0])))
        max_x = min(img_size - 1, int(max(p0[0], p1[0], p2[0])) + 1)
        min_y = max(0, int(min(p0[1], p1[1], p2[1])))
        max_y = min(img_size - 1, int(max(p0[1], p1[1], p2[1])) + 1)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x + 0.5, y + 0.5])
                w0 = edge_fn(p1, p2, p) / area
                w1 = edge_fn(p2, p0, p) / area
                w2 = edge_fn(p0, p1, p) / area

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Interpolate UV (varying) from vertex shader output
                    u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                    v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                    # Fragment shader: sample texture
                    image[y, x] = sample_texture(texture, u, v)

    if matplotlib_available:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(texture)
        axes[0].set_title("Source Texture (64x64)")
        axes[1].imshow(image)
        axes[1].set_title("Textured Quad (256x256)")
        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        plt.tight_layout()
        plt.savefig("07_ex3_textured_quad.png", dpi=100)
        plt.close()
        print("\n  Saved: 07_ex3_textured_quad.png")

    filled = np.sum(np.any(image > 0, axis=2))
    print(f"\n  Filled pixels: {filled} / {img_size * img_size}")


def exercise_4():
    """
    Exercise 4: Interactive Camera (View Matrix Construction)

    Simulate an interactive camera system with translation (WASD) and rotation
    (mouse drag). Demonstrates how the look-at view matrix changes as the camera
    moves, and how different camera positions affect the rendered view.
    """
    # Define a simple 3D scene: a cube at the origin
    cube_vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5],
        [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=np.float32)

    cube_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # front face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
    ]

    # Simulate camera states (as if controlled by WASD + mouse)
    camera_states = [
        {"name": "Front View",    "eye": [0, 0, 3],     "yaw": 0,   "pitch": 0},
        {"name": "Right Side",    "eye": [3, 0, 0],     "yaw": -90, "pitch": 0},
        {"name": "Top Down",      "eye": [0, 3, 0.01],  "yaw": 0,   "pitch": -89},
        {"name": "Orbit 45 deg",  "eye": [2.1, 1.5, 2.1], "yaw": -45, "pitch": -20},
        {"name": "Close Up",      "eye": [0, 0, 1.5],   "yaw": 0,   "pitch": 0},
        {"name": "Far Away",      "eye": [0, 0, 8],     "yaw": 0,   "pitch": 0},
    ]

    print("  Simulating 6 camera positions:")
    print()

    proj = perspective(60.0, 1.0, 0.1, 100.0)
    model = make_identity()
    img_size = 200

    if matplotlib_available:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, cam in enumerate(camera_states):
        eye = np.array(cam["eye"], dtype=np.float32)
        target = np.array([0, 0, 0], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)

        view = look_at(eye, target, up)
        mvp = proj @ view @ model

        # Project all cube vertices
        screen, w_vals = project_vertices(cube_vertices, mvp, img_size, img_size)

        # Check visibility (all w > 0 means in front of camera)
        visible = np.all(w_vals > 0)

        print(f"  [{idx + 1}] {cam['name']:15s} | eye=({eye[0]:+.1f}, {eye[1]:+.1f}, {eye[2]:+.1f}) | "
              f"visible={visible}")

        if matplotlib_available:
            ax = axes[idx // 3, idx % 3]

            # Draw edges of projected cube
            for e0, e1 in cube_edges:
                if w_vals[e0] > 0 and w_vals[e1] > 0:
                    ax.plot([screen[e0, 0], screen[e1, 0]],
                            [screen[e0, 1], screen[e1, 1]], 'b-', linewidth=1.5)

            # Draw vertices
            valid = w_vals > 0
            ax.scatter(screen[valid, 0], screen[valid, 1], c='red', s=20, zorder=5)

            ax.set_xlim(0, img_size)
            ax.set_ylim(img_size, 0)
            ax.set_aspect('equal')
            ax.set_title(cam['name'], fontsize=10)
            ax.grid(True, alpha=0.2)

    if matplotlib_available:
        plt.suptitle("Interactive Camera: 6 Viewpoints of a Cube", fontsize=13)
        plt.tight_layout()
        plt.savefig("07_ex4_interactive_camera.png", dpi=100)
        plt.close()
        print("\n  Saved: 07_ex4_interactive_camera.png")

    print()
    print("  Camera control mapping (WebGL -> Python equivalent):")
    print("    W/S keys  -> translate eye along forward vector")
    print("    A/D keys  -> translate eye along right vector")
    print("    Mouse X   -> update yaw (rotationY)")
    print("    Mouse Y   -> update pitch (rotationX, clamped to +-89 deg)")
    print("    Scroll    -> adjust zoom (camera distance)")


def exercise_5():
    """
    Exercise 5: Multiple Objects (Instanced vs Separate Draw Calls)

    Render three cubes at different positions. Compare two approaches:
    (a) Separate draw calls: different model matrix uniform per cube
    (b) Instanced drawing: single draw call with per-instance data
    Analyze memory usage and draw call overhead.
    """
    # Cube geometry: 8 vertices, 36 indices (12 triangles)
    cube_verts = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5],
        [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=np.float32)

    cube_indices = np.array([
        0, 1, 2, 0, 2, 3,  # back
        4, 5, 6, 4, 6, 7,  # front
        0, 4, 7, 0, 7, 3,  # left
        1, 5, 6, 1, 6, 2,  # right
        3, 2, 6, 3, 6, 7,  # top
        0, 1, 5, 0, 5, 4,  # bottom
    ], dtype=np.uint16)

    # Three cube instances with different transforms
    instances = [
        {"name": "Red Cube",   "pos": [-2.0, 0, 0], "angle_y": 30,  "color": [1, 0.2, 0.2]},
        {"name": "Green Cube", "pos": [ 0.0, 0, 0], "angle_y": 0,   "color": [0.2, 1, 0.2]},
        {"name": "Blue Cube",  "pos": [ 2.0, 0, 0], "angle_y": -30, "color": [0.2, 0.2, 1]},
    ]

    print("  Approach A: Separate Draw Calls")
    print("  ─────────────────────────────────")
    print("  For each object:")
    print("    1. gl.useProgram(shaderProgram)")
    print("    2. gl.uniformMatrix4fv(uModel, modelMatrix_i)")
    print("    3. gl.bindVertexArray(cubeVAO)")
    print("    4. gl.drawElements(gl.TRIANGLES, 36, ...)")
    print()

    num_draw_calls_a = len(instances)
    uniforms_set_a = len(instances)  # one model matrix per draw call
    verts_processed_a = len(cube_verts) * len(instances)  # each cube processed separately

    print(f"    Draw calls: {num_draw_calls_a}")
    print(f"    Uniform updates: {uniforms_set_a}")
    print(f"    Vertices processed: {verts_processed_a}")
    print()

    print("  Approach B: Instanced Rendering")
    print("  ────────────────────────────────")
    print("  Setup:")
    print("    1. Store per-instance data in a buffer (model matrices + colors)")
    print("    2. gl.vertexAttribDivisor(instanceAttrLoc, 1)")
    print("    3. gl.drawElementsInstanced(gl.TRIANGLES, 36, ..., 3)")
    print()

    num_draw_calls_b = 1
    # Per-instance data: 4x4 matrix (16 floats) + color (3 floats) = 19 floats
    instance_buffer_size = len(instances) * (16 + 3) * 4  # bytes
    verts_processed_b = len(cube_verts) * len(instances)  # same total, but batched

    print(f"    Draw calls: {num_draw_calls_b}")
    print(f"    Instance buffer: {instance_buffer_size} bytes")
    print(f"    Vertices processed: {verts_processed_b} (same, but one GPU dispatch)")
    print()

    # Memory comparison
    print("  Memory Comparison:")
    vertex_buffer_size = len(cube_verts) * 3 * 4  # 3 floats x 4 bytes
    index_buffer_size = len(cube_indices) * 2  # uint16

    print(f"    Shared vertex buffer: {vertex_buffer_size} bytes")
    print(f"    Shared index buffer: {index_buffer_size} bytes")
    print(f"    Approach A extra: {num_draw_calls_a} uniform calls (CPU overhead)")
    print(f"    Approach B extra: {instance_buffer_size} byte instance buffer")
    print()

    # Scaling analysis
    print("  Scaling Analysis (N objects):")
    print("  ┌──────────┬──────────────────┬───────────────────┐")
    print("  │   N      │  Draw Calls (A)  │  Draw Calls (B)   │")
    print("  ├──────────┼──────────────────┼───────────────────┤")
    for n in [10, 100, 1000, 10000]:
        print(f"  │ {n:>8} │ {n:>16} │ {1:>17} │")
    print("  └──────────┴──────────────────┴───────────────────┘")
    print()
    print("  At large N, instanced rendering is dramatically more efficient")
    print("  because draw calls have significant CPU overhead (~1-10 us each).")

    # Render the 3 cubes as wireframes
    if matplotlib_available:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')

        proj_mat = perspective(60.0, 2.0, 0.1, 100.0)
        view_mat = look_at([0, 2, 6], [0, 0, 0], [0, 1, 0])

        cube_faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 4, 7, 3], [1, 5, 6, 2],
            [3, 2, 6, 7], [0, 1, 5, 4],
        ]

        for inst in instances:
            model = make_translation(*inst["pos"]) @ make_rotation_y(inst["angle_y"])
            transformed = np.zeros_like(cube_verts)
            for i, v in enumerate(cube_verts):
                tv = model @ np.array([v[0], v[1], v[2], 1.0])
                transformed[i] = tv[:3]

            faces = [[transformed[idx] for idx in face] for face in cube_faces]
            poly = Poly3DCollection(faces, alpha=0.3,
                                    facecolors=[inst["color"]] * len(faces),
                                    edgecolors='black', linewidths=0.5)
            ax.add_collection3d(poly)

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Three Cubes: Separate Draw Calls vs Instanced Rendering")
        plt.tight_layout()
        plt.savefig("07_ex5_multiple_objects.png", dpi=100)
        plt.close()
        print("\n  Saved: 07_ex5_multiple_objects.png")


def exercise_6():
    """
    Exercise 6: Performance Measurement (FPS and Bottleneck Analysis)

    Simulate rendering workloads of increasing complexity and analyze
    where performance bottlenecks occur. Models the relationship between
    triangle count, draw calls, and frame time.
    """
    import time

    print("  GPU Rendering Performance Model")
    print("  ═══════════════════════════════")
    print()

    # Performance model parameters (typical mid-range GPU)
    vertex_rate = 2e9          # vertices/second (modern GPU)
    fragment_rate = 8e9        # fragments/second (fill rate)
    draw_call_overhead = 5e-6  # 5 microseconds per draw call (CPU side)
    target_fps = 60
    frame_budget = 1.0 / target_fps  # ~16.67 ms

    print(f"  GPU vertex throughput: {vertex_rate / 1e9:.1f} G vertices/sec")
    print(f"  GPU fill rate: {fragment_rate / 1e9:.1f} G fragments/sec")
    print(f"  Draw call overhead: {draw_call_overhead * 1e6:.1f} us/call")
    print(f"  Frame budget @ {target_fps} FPS: {frame_budget * 1000:.2f} ms")
    print()

    # Test scenarios: increasing triangle counts
    scenarios = [
        {"triangles": 100,    "draw_calls": 1,    "resolution": (800, 600)},
        {"triangles": 1000,   "draw_calls": 10,   "resolution": (800, 600)},
        {"triangles": 10000,  "draw_calls": 100,  "resolution": (1920, 1080)},
        {"triangles": 100000, "draw_calls": 1000, "resolution": (1920, 1080)},
        {"triangles": 1000000,"draw_calls": 5000, "resolution": (3840, 2160)},
    ]

    print("  Performance Analysis:")
    print("  ┌────────────┬────────────┬──────────────┬───────────┬──────────┬───────────────┐")
    print("  │ Triangles  │ Draw Calls │ Vert Time ms │ Frag ms   │ CPU ms   │ Est. FPS      │")
    print("  ├────────────┼────────────┼──────────────┼───────────┼──────────┼───────────────┤")

    results = []
    for s in scenarios:
        num_tris = s["triangles"]
        num_dc = s["draw_calls"]
        w, h = s["resolution"]

        # Estimate vertex processing time
        num_verts = num_tris * 3  # worst case: no vertex reuse
        vert_time = num_verts / vertex_rate

        # Estimate fragment processing time (assume 50% overdraw)
        avg_frag_per_tri = (w * h) / max(num_tris, 1) * 1.5  # with overdraw
        total_frags = min(num_tris * avg_frag_per_tri, w * h * 3)  # cap at 3x screen
        frag_time = total_frags / fragment_rate

        # CPU draw call overhead
        cpu_time = num_dc * draw_call_overhead

        # Total frame time (vertex and fragment run in parallel on GPU,
        # but CPU work is serial)
        gpu_time = max(vert_time, frag_time)
        total_time = gpu_time + cpu_time

        est_fps = min(1.0 / total_time, 1000) if total_time > 0 else 1000

        # Identify bottleneck
        if cpu_time > gpu_time:
            bottleneck = "CPU (draw calls)"
        elif frag_time > vert_time:
            bottleneck = "Fill rate"
        else:
            bottleneck = "Vertex"

        results.append({
            "tris": num_tris,
            "dc": num_dc,
            "vert_ms": vert_time * 1000,
            "frag_ms": frag_time * 1000,
            "cpu_ms": cpu_time * 1000,
            "fps": est_fps,
            "bottleneck": bottleneck,
        })

        print(f"  │ {num_tris:>10,} │ {num_dc:>10,} │ {vert_time * 1000:>12.4f} │ "
              f"{frag_time * 1000:>9.4f} │ {cpu_time * 1000:>8.4f} │ {est_fps:>10,.0f}    │")

    print("  └────────────┴────────────┴──────────────┴───────────┴──────────┴───────────────┘")
    print()

    # Bottleneck analysis
    print("  Bottleneck Identification:")
    for r in results:
        print(f"    {r['tris']:>10,} tris -> {r['bottleneck']}")
    print()

    # Simulate actual Python-side computation time for comparison
    print("  Software Rasterizer Comparison (Python):")
    print("  Measuring time to process vertices in Python (no GPU)...")
    triangle_counts = [100, 1000, 10000]
    for n in triangle_counts:
        verts = np.random.randn(n * 3, 3).astype(np.float32)
        mvp = perspective(60, 1.5, 0.1, 100) @ look_at([0, 0, 5], [0, 0, 0], [0, 1, 0])

        start = time.perf_counter()
        # Simulate vertex shader: MVP transform
        ones = np.ones((len(verts), 1), dtype=np.float32)
        homo = np.hstack([verts, ones])
        clip = (mvp @ homo.T).T
        elapsed = time.perf_counter() - start

        sw_fps = 1.0 / elapsed if elapsed > 0 else float('inf')
        print(f"    {n:>6,} tris: {elapsed * 1000:>8.2f} ms ({sw_fps:>8.0f} FPS) "
              f"- GPU would be {vertex_rate * elapsed / (n * 3):>6.0f}x faster")

    if matplotlib_available:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        tris = [r["tris"] for r in results]
        fps_vals = [r["fps"] for r in results]
        vert_ms = [r["vert_ms"] for r in results]
        frag_ms = [r["frag_ms"] for r in results]
        cpu_ms = [r["cpu_ms"] for r in results]

        # FPS vs triangle count
        axes[0].semilogx(tris, fps_vals, 'bo-', linewidth=2, markersize=8)
        axes[0].axhline(y=60, color='r', linestyle='--', label='60 FPS target')
        axes[0].set_xlabel("Triangle Count")
        axes[0].set_ylabel("Estimated FPS")
        axes[0].set_title("FPS vs Triangle Count")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Time breakdown
        x = np.arange(len(results))
        width = 0.25
        axes[1].bar(x - width, vert_ms, width, label='Vertex (GPU)', color='steelblue')
        axes[1].bar(x, frag_ms, width, label='Fragment (GPU)', color='coral')
        axes[1].bar(x + width, cpu_ms, width, label='Draw Calls (CPU)', color='green')
        axes[1].set_xlabel("Scenario")
        axes[1].set_ylabel("Time (ms)")
        axes[1].set_title("Frame Time Breakdown")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{r['tris'] // 1000}K" if r['tris'] >= 1000
                                 else str(r['tris']) for r in results], fontsize=8)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("07_ex6_performance_measurement.png", dpi=100)
        plt.close()
        print("\n  Saved: 07_ex6_performance_measurement.png")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Hello Triangle", exercise_1),
        ("Exercise 2: Animated Rotation", exercise_2),
        ("Exercise 3: Textured Quad", exercise_3),
        ("Exercise 4: Interactive Camera", exercise_4),
        ("Exercise 5: Multiple Objects", exercise_5),
        ("Exercise 6: Performance Measurement", exercise_6),
    ]

    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()

    print(f"\n{'=' * 60}")
    print(" All exercises completed!")
    print(f"{'=' * 60}")
