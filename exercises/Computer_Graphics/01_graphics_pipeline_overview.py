"""
Exercises for Lesson 01: Graphics Pipeline Overview
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Conceptual: Fragment estimation with overdraw.

    A scene contains 10,000 triangles but only 3,000 are visible after frustum
    culling. The screen resolution is 1920x1080. Estimate the maximum number of
    fragments generated (assuming each visible triangle covers an average of 100
    pixels). How does overdraw affect this number?
    """
    total_triangles = 10000
    visible_triangles = 3000
    avg_pixels_per_triangle = 100
    screen_width = 1920
    screen_height = 1080
    total_pixels = screen_width * screen_height

    # Maximum fragments generated: each visible triangle produces fragments
    max_fragments = visible_triangles * avg_pixels_per_triangle
    print(f"Total triangles in scene: {total_triangles}")
    print(f"Visible after frustum culling: {visible_triangles}")
    print(f"Average pixels per triangle: {avg_pixels_per_triangle}")
    print(f"Maximum fragments generated: {max_fragments:,}")
    print(f"Screen resolution: {screen_width}x{screen_height} = {total_pixels:,} pixels")
    print()

    # Overdraw analysis
    overdraw_ratio = max_fragments / total_pixels
    print(f"Overdraw ratio: {overdraw_ratio:.2f}x")
    print(f"  This means each pixel is shaded ~{overdraw_ratio:.2f} times on average.")
    print()

    # With higher overdraw (e.g., many overlapping triangles)
    for overdraw in [1.0, 1.5, 2.0, 3.0]:
        effective_fragments = int(total_pixels * overdraw)
        print(f"  At {overdraw:.1f}x overdraw: ~{effective_fragments:,} total fragment shader invocations")

    print()
    print("Key insight: Overdraw wastes GPU fragment shader cycles.")
    print("Front-to-back rendering with early-Z can reduce overdraw significantly.")


def exercise_2():
    """
    Pipeline Tracing: Trace a single vertex with position (1, 2, 3) through
    the pipeline stages. Describe what happens at each stage conceptually.
    """
    vertex = np.array([1.0, 2.0, 3.0])
    print(f"Starting vertex position (object space): {vertex}")
    print()

    # Stage 1: Application (CPU)
    print("=== Stage 1: Application (CPU) ===")
    print("  - CPU determines this vertex is part of a visible object")
    print("  - CPU issues a draw call containing this vertex's mesh")
    print("  - Vertex data is already in GPU memory (uploaded earlier)")
    print()

    # Stage 2: Vertex Shader (GPU) - Model Transform
    model_matrix = np.array([
        [1, 0, 0, 2],
        [0, 1, 0, 0],
        [0, 0, 1, -5],
        [0, 0, 0, 1]
    ], dtype=float)

    p_homogeneous = np.append(vertex, 1.0)
    p_world = model_matrix @ p_homogeneous
    print("=== Stage 2: Geometry Processing - Model Transform ===")
    print(f"  Model matrix translates by (2, 0, -5)")
    print(f"  World space: {p_world[:3]}")
    print()

    # View Transform
    # Simple view: camera at (0, 0, 0) looking along -Z
    view_matrix = np.eye(4)
    p_eye = view_matrix @ p_world
    print("=== Stage 2: Geometry Processing - View Transform ===")
    print(f"  Camera at origin, looking along -Z")
    print(f"  Eye space: {p_eye[:3]}")
    print()

    # Projection Transform (simplified perspective)
    fov = 60.0
    aspect = 16.0 / 9.0
    near, far = 0.1, 100.0
    fov_rad = np.radians(fov)
    t = near * np.tan(fov_rad / 2)
    r = t * aspect

    proj_matrix = np.array([
        [near / r, 0, 0, 0],
        [0, near / t, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]
    ], dtype=float)

    p_clip = proj_matrix @ p_eye
    print("=== Stage 2: Geometry Processing - Projection Transform ===")
    print(f"  Perspective projection (FOV=60, aspect=16:9)")
    print(f"  Clip space: ({p_clip[0]:.4f}, {p_clip[1]:.4f}, {p_clip[2]:.4f}, {p_clip[3]:.4f})")
    print()

    # Stage 3: Clipping
    w = p_clip[3]
    inside = all(-w <= p_clip[i] <= w for i in range(3))
    print("=== Stage 3: Clipping ===")
    print(f"  w = {w:.4f}")
    print(f"  Inside frustum (-w <= x,y,z <= w)? {inside}")
    print()

    # Stage 4: Perspective Division -> NDC
    p_ndc = p_clip[:3] / p_clip[3]
    print("=== Stage 4: Perspective Division -> NDC ===")
    print(f"  NDC: ({p_ndc[0]:.4f}, {p_ndc[1]:.4f}, {p_ndc[2]:.4f})")
    print(f"  All coordinates should be in [-1, 1] if inside frustum")
    print()

    # Stage 5: Viewport Transform -> Screen
    screen_w, screen_h = 1920, 1080
    sx = (p_ndc[0] + 1) * 0.5 * screen_w
    sy = (1 - p_ndc[1]) * 0.5 * screen_h
    print("=== Stage 5: Viewport Transform -> Screen ===")
    print(f"  Screen coordinates: ({sx:.1f}, {sy:.1f})")
    print(f"  This pixel position is where the vertex appears on a {screen_w}x{screen_h} display")
    print()

    # Stage 6: Rasterization & Fragment Processing
    print("=== Stage 6: Rasterization & Fragment Processing ===")
    print("  - This vertex, along with 2 others, forms a triangle")
    print("  - The rasterizer determines which pixels the triangle covers")
    print("  - For each covered pixel, the fragment shader computes the color")
    print("  - The depth test determines visibility")
    print("  - The final color is written to the framebuffer")


def exercise_3():
    """
    Bottleneck Analysis: A game renders at 30 FPS. When you halve the screen
    resolution, FPS jumps to 58. When you halve the number of triangles instead
    (at original resolution), FPS stays at 32. Where is the bottleneck? Why?
    """
    print("Given:")
    print("  - Original: 30 FPS")
    print("  - Half resolution: 58 FPS (nearly doubled)")
    print("  - Half triangles: 32 FPS (barely changed)")
    print()

    print("Analysis:")
    print("  Halving resolution reduces the number of fragments by 4x")
    print("  (half width * half height = 1/4 the pixels)")
    print("  This nearly doubled the FPS (30 -> 58), showing strong sensitivity")
    print("  to fragment count.")
    print()
    print("  Halving triangle count reduces vertex processing workload by 2x")
    print("  but FPS barely changed (30 -> 32), showing low sensitivity to")
    print("  geometry complexity.")
    print()

    print("Conclusion:")
    print("  The bottleneck is in FRAGMENT PROCESSING (fragment-bound).")
    print()
    print("  Possible causes:")
    print("  - Complex fragment shader (many texture lookups, heavy math)")
    print("  - High overdraw (many overlapping triangles at the same pixel)")
    print("  - Large triangles covering many pixels each")
    print("  - Memory bandwidth limits (high-res textures)")
    print()
    print("  Mitigation strategies:")
    print("  - Simplify the fragment shader")
    print("  - Use mipmaps to reduce texture bandwidth")
    print("  - Render opaque objects front-to-back for early-Z rejection")
    print("  - Consider deferred rendering to shade each pixel only once")
    print("  - Lower the rendering resolution and upscale")


def exercise_4():
    """
    Draw Call Reduction: You have a forest scene with 1,000 trees, each drawn
    with a separate draw call. Propose two strategies to reduce the number of
    draw calls while maintaining visual quality.
    """
    original_draw_calls = 1000
    print(f"Original: {original_draw_calls} draw calls (one per tree)")
    print()

    print("Strategy 1: INSTANCED RENDERING")
    print("-" * 40)
    print("  Idea: All trees share the same mesh geometry but differ in")
    print("  position, rotation, and scale. GPU instancing draws many copies")
    print("  of the same mesh with a single draw call.")
    print()
    print("  Implementation:")
    print("  - Store per-instance data (model matrix, color variation) in a buffer")
    print("  - Use gl.drawElementsInstanced() with instance count = 1000")
    print("  - In the vertex shader, use gl_InstanceID to index per-instance data")
    print()
    unique_tree_types = 5
    instanced_draw_calls = unique_tree_types
    print(f"  Result: {instanced_draw_calls} draw calls ({unique_tree_types} tree variants)")
    print(f"  Reduction: {(1 - instanced_draw_calls / original_draw_calls) * 100:.0f}%")
    print()

    print("Strategy 2: STATIC BATCHING (Mesh Merging)")
    print("-" * 40)
    print("  Idea: Combine all tree meshes into a single large mesh on the CPU")
    print("  during loading. Pre-transform each tree's vertices by its model matrix.")
    print()
    print("  Implementation:")
    print("  - For each tree: transform vertices by model matrix, append to combined buffer")
    print("  - Upload the combined mesh as a single vertex/index buffer")
    print("  - Issue a single draw call for the entire forest")
    print()
    batched_draw_calls = 1
    print(f"  Result: {batched_draw_calls} draw call")
    print(f"  Reduction: {(1 - batched_draw_calls / original_draw_calls) * 100:.0f}%")
    print()
    print("  Trade-offs:")
    print("  - Uses more memory (no vertex sharing between instances)")
    print("  - Trees cannot be individually animated (pre-transformed)")
    print("  - Good for static scenery, not for dynamic objects")
    print()

    print("Comparison:")
    print(f"  {'Strategy':<25} {'Draw Calls':<15} {'Dynamic?':<10} {'Memory':<10}")
    print(f"  {'Original':<25} {original_draw_calls:<15} {'Yes':<10} {'Low':<10}")
    print(f"  {'Instancing':<25} {instanced_draw_calls:<15} {'Yes':<10} {'Low':<10}")
    print(f"  {'Static Batching':<25} {batched_draw_calls:<15} {'No':<10} {'High':<10}")


def exercise_5():
    """
    Transparency Challenge: Explain why rendering transparent objects is more
    difficult than opaque objects. What happens if you render transparent objects
    in random order?
    """
    print("Why transparency is difficult:")
    print("=" * 50)
    print()

    print("1. The Z-buffer cannot handle transparency correctly.")
    print("   - For opaque objects, the closest fragment wins (simple depth test)")
    print("   - For transparent objects, we need to BLEND colors, not replace them")
    print("   - The blending equation: C_final = alpha * C_src + (1-alpha) * C_dst")
    print("   - This requires the destination color to already be correct")
    print()

    print("2. Blending is ORDER-DEPENDENT.")
    print("   - alpha * A + (1-alpha) * B  !=  alpha * B + (1-alpha) * A")
    print("   - We must render transparent objects BACK-TO-FRONT")
    print()

    # Demonstrate order-dependent blending
    background = np.array([0.0, 0.0, 0.0])  # Black background
    red_glass = np.array([1.0, 0.0, 0.0])   # Red, alpha=0.5
    blue_glass = np.array([0.0, 0.0, 1.0])  # Blue, alpha=0.5
    alpha = 0.5

    # Correct order: blue behind red (render blue first, then red on top)
    after_blue = alpha * blue_glass + (1 - alpha) * background
    correct_result = alpha * red_glass + (1 - alpha) * after_blue
    print("Numerical example:")
    print(f"  Background: {background}")
    print(f"  Blue glass (behind, alpha=0.5): {blue_glass}")
    print(f"  Red glass (in front, alpha=0.5): {red_glass}")
    print()

    print("  Correct order (blue first, then red on top):")
    print(f"    After blue:  {after_blue}")
    print(f"    After red:   {correct_result}")
    print()

    # Wrong order: red first, then blue
    after_red = alpha * red_glass + (1 - alpha) * background
    wrong_result = alpha * blue_glass + (1 - alpha) * after_red
    print("  Wrong order (red first, then blue on top):")
    print(f"    After red:   {after_red}")
    print(f"    After blue:  {wrong_result}")
    print()

    print(f"  Results differ: {not np.allclose(correct_result, wrong_result)}")
    print(f"  Correct: {np.round(correct_result, 3)}")
    print(f"  Wrong:   {np.round(wrong_result, 3)}")
    print()

    print("3. Random order rendering causes:")
    print("   - Incorrect color blending (wrong visual result)")
    print("   - Flickering as objects sort differently frame-to-frame")
    print("   - Some transparent surfaces may be invisible")
    print()

    print("4. Two overlapping transparent objects:")
    print("   - Even harder: both need to contribute to the final pixel")
    print("   - Per-object sorting may not be sufficient (need per-triangle sorting)")
    print("   - Advanced solutions: Order-Independent Transparency (OIT),")
    print("     weighted blended OIT, depth peeling")


def exercise_6():
    """
    Modern Extensions: Compare tessellation shaders and geometry shaders.
    For the task of generating grass blades on a terrain, which approach would
    you choose and why?
    """
    print("Tessellation Shaders vs Geometry Shaders")
    print("=" * 50)
    print()

    comparison = [
        ("Input", "Patches (e.g., 3 control points)", "Complete primitives (triangle)"),
        ("Output count", "Fixed by tessellation levels", "Variable (0 to max_vertices)"),
        ("Parallelism", "Excellent (fixed topology)", "Poor (variable output)"),
        ("Use case", "Subdivision, displacement", "Particle emit, shadow volumes"),
        ("Performance", "Very good on modern GPUs", "Often slow due to variable output"),
        ("Hardware support", "DX11+ / GL4.0+ / WebGL (no)", "DX10+ / GL3.2+ / WebGL (no)"),
    ]

    print(f"  {'Aspect':<20} {'Tessellation':<35} {'Geometry Shader':<35}")
    print("  " + "-" * 88)
    for aspect, tess, geom in comparison:
        print(f"  {aspect:<20} {tess:<35} {geom:<35}")
    print()

    print("For generating grass blades on a terrain:")
    print("-" * 50)
    print()
    print("RECOMMENDED: Tessellation Shaders (or Compute Shaders)")
    print()
    print("Reasoning:")
    print("  1. Grass generation benefits from PREDICTABLE output count.")
    print("     Each terrain patch produces a known number of grass blades.")
    print()
    print("  2. Tessellation shaders have BETTER performance:")
    print("     - The tessellator stage is fixed-function hardware, very fast")
    print("     - Geometry shaders have variable output, causing pipeline stalls")
    print("     - GPU architectures are optimized for tessellation, not geometry shaders")
    print()
    print("  3. Adaptive density is natural with tessellation:")
    print("     - TCS can set tessellation levels based on camera distance")
    print("     - Near grass: high density; far grass: fewer blades")
    print("     - This LOD is automatic and smooth")
    print()
    print("  4. Displacement mapping integrates naturally:")
    print("     - TES can displace grass blade positions using terrain heightmap")
    print("     - Wind animation can be applied in the TES")
    print()
    print("  Alternative: Modern engines often prefer COMPUTE SHADERS for grass:")
    print("     - Generate grass blade positions and transforms in a compute pass")
    print("     - Render with instanced draw calls")
    print("     - Most flexible and performant approach on modern GPUs")
    print()
    print("  Geometry shaders would work but are NOT recommended because:")
    print("     - Variable output size causes GPU pipeline bubbles")
    print("     - Performance is typically 2-5x worse than tessellation")
    print("     - Modern GPU architectures are not optimized for them")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Fragment Estimation with Overdraw ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Pipeline Tracing ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Bottleneck Analysis ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Draw Call Reduction ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Transparency Challenge ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Modern Extensions ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
