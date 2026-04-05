# 01. Graphics Pipeline Overview

| [Next: 2D Transformations &rarr;](02_2D_Transformations.md)

---

## Learning Objectives

1. Understand the purpose and architecture of the real-time graphics pipeline
2. Distinguish between CPU-side (application stage) and GPU-side processing
3. Trace the journey of a single triangle from vertex data to a colored pixel on screen
4. Explain vertex processing, including vertex shaders, transformations, and clipping
5. Describe primitive assembly and triangle setup
6. Understand fragment processing: fragment shaders, depth testing, and blending
7. Explain double/triple buffering and vsync as frame-presentation strategies
8. Recognize modern pipeline extensions: tessellation, geometry shaders, and compute shaders

---

## Why This Matters

Every real-time 3D application -- from AAA games rendering millions of triangles at 120 FPS, to CAD software visualizing precision engineering models, to medical imaging tools reconstructing volumetric data -- relies on the graphics pipeline. Understanding this pipeline is the single most important prerequisite for all of computer graphics because every subsequent topic (transformations, shading, textures, GPU programming) maps directly onto one or more pipeline stages.

Think of the graphics pipeline as an **assembly line in a factory**. Raw materials (vertex data) enter at one end, pass through a series of specialized workstations (pipeline stages), and emerge as finished products (colored pixels on your display). Each workstation is optimized for one task, and the factory's throughput depends on the slowest station -- the *bottleneck*. Modern GPUs exploit this assembly-line parallelism to process billions of operations per second.

---

## 1. The Big Picture

The graphics pipeline transforms 3D scene data into a 2D image displayed on your screen. At its core, this is a function:

$$f: \text{3D Scene Description} \rightarrow \text{2D Pixel Array (Framebuffer)}$$

The pipeline is divided into broad stages, each performing a distinct category of work:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GRAPHICS PIPELINE                             │
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────┐ │
│  │ APPLICATION  │──▶│  GEOMETRY   │──▶│ RASTERIZATION│──▶│FRAGMENT│ │
│  │   (CPU)      │   │ PROCESSING  │   │              │   │  OPS   │ │
│  │             │   │   (GPU)     │   │   (GPU)      │   │ (GPU)  │ │
│  └─────────────┘   └─────────────┘   └──────────────┘   └────────┘ │
│        │                  │                  │                │      │
│   Scene setup       Vertex shader       Triangle →       Per-pixel  │
│   Draw calls        Transform/Clip      Fragments        shading    │
│   State changes     Projection          Z-interpolation  Depth test │
│                     Prim. assembly      Attribute interp Blending   │
│                                                                      │
│  ────────────────────────────────────────────────────────▶           │
│                     Data flows left to right                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Application Stage (CPU)

The application stage runs entirely on the CPU. It is responsible for:

### 2.1 Scene Management

The CPU decides *what* to render each frame. This involves:

- **Scene graph traversal**: Walking a hierarchical data structure of objects
- **Frustum culling**: Discarding objects entirely outside the camera's view volume
- **Occlusion culling**: Skipping objects hidden behind other objects
- **Level-of-detail (LOD) selection**: Choosing simpler meshes for distant objects

### 2.2 Draw Calls

A **draw call** is a command from the CPU to the GPU: "Render this set of triangles with these settings." Each draw call carries overhead because the CPU must:

1. Set GPU state (which shader to use, which textures to bind)
2. Upload or reference vertex/index buffer data
3. Issue the actual draw command

```python
# Pseudocode: a simplified rendering loop
def render_frame(scene, camera):
    """Main render loop -- runs on CPU every frame."""
    # 1. Update game logic, physics, animations
    scene.update(delta_time)

    # 2. Determine which objects are visible
    visible_objects = frustum_cull(scene.objects, camera.frustum)

    # 3. Sort objects (opaque front-to-back, transparent back-to-front)
    opaque, transparent = partition_by_transparency(visible_objects)
    opaque.sort(key=lambda obj: distance_to_camera(obj, camera))
    transparent.sort(key=lambda obj: -distance_to_camera(obj, camera))

    # 4. Issue draw calls to GPU
    for obj in opaque:
        gpu.set_shader(obj.material.shader)       # State change
        gpu.set_textures(obj.material.textures)    # State change
        gpu.set_uniforms(camera.view_proj_matrix,  # Upload matrices
                         obj.model_matrix)
        gpu.draw(obj.vertex_buffer, obj.index_buffer)  # Draw call!

    # 5. Render transparent objects (after all opaque)
    gpu.enable_blending()
    for obj in transparent:
        gpu.set_shader(obj.material.shader)
        gpu.draw(obj.vertex_buffer, obj.index_buffer)

    # 6. Present the frame
    gpu.swap_buffers()
```

> **Performance insight**: Modern games may issue thousands of draw calls per frame. Reducing draw calls through *batching* (combining similar objects into one call) and *instancing* (drawing many copies with one call) is a critical optimization.

### 2.3 CPU vs GPU Roles

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Architecture** | Few powerful cores (4-16) | Thousands of simple cores |
| **Strength** | Sequential logic, branching | Massively parallel computation |
| **Pipeline role** | Scene setup, culling, draw calls | Vertex/fragment processing |
| **Memory** | System RAM (16-64 GB) | VRAM (8-24 GB), high bandwidth |
| **Bottleneck sign** | Low GPU utilization | Low CPU utilization |

The CPU prepares *what* to draw; the GPU executes *how* to draw it.

---

## 3. Geometry Processing (GPU)

Once the GPU receives a draw call, geometry processing begins. This stage operates on **vertices** -- the points that define 3D shapes.

### 3.1 Vertex Shaders

The **vertex shader** is a small program that runs once per vertex. Its primary job is to transform vertex positions from *object space* to *clip space*:

$$\mathbf{p}_{\text{clip}} = \mathbf{M}_{\text{projection}} \cdot \mathbf{M}_{\text{view}} \cdot \mathbf{M}_{\text{model}} \cdot \mathbf{p}_{\text{object}}$$

Where:
- $\mathbf{p}_{\text{object}}$: vertex position in the object's local coordinate system
- $\mathbf{M}_{\text{model}}$: transforms object space $\rightarrow$ world space
- $\mathbf{M}_{\text{view}}$: transforms world space $\rightarrow$ camera (eye) space
- $\mathbf{M}_{\text{projection}}$: transforms eye space $\rightarrow$ clip space (applies perspective)

The vertex shader can also:
- Transform normals for lighting: $\mathbf{n}_{\text{world}} = (\mathbf{M}_{\text{model}}^{-1})^T \cdot \mathbf{n}_{\text{object}}$
- Pass texture coordinates to later stages
- Compute per-vertex lighting
- Animate vertices (skeletal animation, vertex displacement)

```python
import numpy as np

def vertex_shader(position, normal, model_matrix, view_matrix, proj_matrix):
    """
    Mimics what a GPU vertex shader does.

    Each vertex is independently transformed -- this is why GPUs
    can process millions of vertices in parallel.
    """
    # Model-View-Projection combined matrix
    mvp = proj_matrix @ view_matrix @ model_matrix

    # Transform position to clip space (4D homogeneous coordinates)
    clip_pos = mvp @ np.append(position, 1.0)

    # Transform normal to world space for lighting
    # Why inverse-transpose? Non-uniform scaling would distort normals otherwise.
    normal_matrix = np.linalg.inv(model_matrix[:3, :3]).T
    world_normal = normal_matrix @ normal
    world_normal = world_normal / np.linalg.norm(world_normal)  # Re-normalize

    return clip_pos, world_normal
```

### 3.2 Transformations Through Coordinate Spaces

A vertex travels through several coordinate spaces:

```
Object Space ──(Model Matrix)──▶ World Space ──(View Matrix)──▶ Eye Space
     │                                                              │
     │                                                     (Projection Matrix)
     │                                                              │
     │                                                              ▼
     │                                                        Clip Space
     │                                                              │
     │                                                   (Perspective Division)
     │                                                              │
     │                                                              ▼
     │                                                          NDC Space
     │                                                         [-1,1]^3
     │                                                              │
     │                                                    (Viewport Transform)
     │                                                              │
     └──────────────────────────────────────────────────────▶ Screen Space
                                                              (pixels)
```

We will explore each transformation in detail in Lessons 02 and 03.

### 3.3 Clipping

After the projection transform, vertices are in **clip space**. Clipping removes geometry that falls outside the view volume (the *frustum*). In clip space, a point $(x, y, z, w)$ is inside the frustum if:

$$-w \leq x \leq w, \quad -w \leq y \leq w, \quad -w \leq z \leq w$$

Triangles partially outside the frustum are *clipped* -- cut along the frustum boundaries, potentially producing new vertices. A triangle clipped against one plane can produce up to two triangles.

### 3.4 Perspective Division

After clipping, the GPU performs **perspective division**: dividing clip coordinates by $w$:

$$\mathbf{p}_{\text{NDC}} = \left(\frac{x}{w}, \frac{y}{w}, \frac{z}{w}\right)$$

This maps the frustum to the **Normalized Device Coordinate (NDC)** cube $[-1, 1]^3$, where objects farther from the camera appear smaller (perspective foreshortening).

### 3.5 Primitive Assembly

Individual transformed vertices are grouped into **primitives** -- typically triangles. If you submitted vertex indices `[0, 1, 2, 3, 4, 5]` as a triangle list, the GPU assembles triangles $(v_0, v_1, v_2)$ and $(v_3, v_4, v_5)$.

Common primitive types:
- **Triangle list**: Every 3 vertices form one triangle
- **Triangle strip**: Each new vertex forms a triangle with the previous two
- **Triangle fan**: All triangles share the first vertex

### 3.6 Triangle Setup

Before rasterization, the GPU computes per-triangle data needed for efficient rasterization:
- **Edge equations**: Used to test whether a pixel falls inside the triangle
- **Attribute gradients**: How vertex attributes (color, UV, normal) change across the triangle surface
- **Face orientation**: Determines front/back face for potential back-face culling

---

## 4. Rasterization

Rasterization converts continuous geometric primitives (triangles) into discrete **fragments** -- candidate pixels that may contribute to the final image.

### 4.1 The Rasterization Process

```
┌─────────────────────────────────────────────────────────────┐
│                     RASTERIZATION                            │
│                                                              │
│   Triangle (3 vertices with screen positions)                │
│         │                                                    │
│         ▼                                                    │
│   ┌──────────────────┐                                       │
│   │ For each pixel in │   "Which pixels does this            │
│   │ bounding box:     │    triangle cover?"                   │
│   │                   │                                       │
│   │  Is pixel inside  │◄── Edge function test                │
│   │  triangle?        │    e(x,y) = (x-x0)(y1-y0)           │
│   │                   │           - (y-y0)(x1-x0)            │
│   │  If yes:          │                                       │
│   │   ● Generate      │                                       │
│   │     fragment       │                                       │
│   │   ● Interpolate   │◄── Barycentric coordinates            │
│   │     attributes    │    (depth, UV, normal, color)         │
│   └──────────────────┘                                       │
│         │                                                    │
│         ▼                                                    │
│   Stream of fragments (with interpolated attributes)         │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Fragments vs Pixels

A **fragment** is not the same as a pixel:
- A fragment is a *candidate* pixel produced by rasterizing one triangle
- Multiple fragments may compete for the same pixel position (from overlapping triangles)
- The depth test and blending stages resolve which fragment(s) contribute to the final pixel color

### 4.3 Barycentric Interpolation

Given a point $P$ inside a triangle with vertices $A$, $B$, $C$, we can express $P$ as:

$$P = \alpha A + \beta B + \gamma C, \quad \alpha + \beta + \gamma = 1$$

The barycentric coordinates $(\alpha, \beta, \gamma)$ tell us the "influence" of each vertex on the point. We use them to smoothly interpolate any vertex attribute across the triangle:

$$\text{attr}(P) = \alpha \cdot \text{attr}(A) + \beta \cdot \text{attr}(B) + \gamma \cdot \text{attr}(C)$$

This is how smooth color gradients, texture coordinates, and normals are computed across triangle surfaces.

> **Important**: For perspective-correct interpolation, attributes must be divided by $w$ before interpolation and multiplied back afterward. Without this correction, textures appear to "swim" on surfaces.

We will implement rasterization from scratch in Lesson 04.

---

## 5. Fragment Processing

Each fragment produced by rasterization enters the fragment processing stage.

### 5.1 Fragment Shaders

The **fragment shader** (also called *pixel shader*) runs once per fragment. It computes the fragment's color based on:

- Interpolated vertex attributes (normals, texture coordinates)
- Texture samples (looking up colors from images)
- Lighting calculations (using light positions, material properties)
- Any custom computation the programmer desires

```python
def fragment_shader(frag_position, frag_normal, frag_uv,
                    light_pos, light_color, camera_pos, texture):
    """
    Simplified fragment shader implementing Phong lighting.

    This runs for EVERY fragment -- potentially millions per frame.
    GPU parallelism makes this feasible.
    """
    # Sample the texture at the fragment's UV coordinates
    albedo = texture.sample(frag_uv)

    # Lighting vectors
    N = normalize(frag_normal)             # Surface normal
    L = normalize(light_pos - frag_position)  # Direction to light
    V = normalize(camera_pos - frag_position) # Direction to camera
    H = normalize(L + V)                   # Halfway vector

    # Ambient: constant low-level illumination
    ambient = 0.1 * albedo

    # Diffuse: Lambert's cosine law
    diff = max(np.dot(N, L), 0.0)
    diffuse = diff * albedo * light_color

    # Specular: shiny highlight
    spec = max(np.dot(N, H), 0.0) ** 64.0
    specular = spec * light_color

    # Combine components
    color = ambient + diffuse + specular
    return np.clip(color, 0.0, 1.0)  # Clamp to valid range
```

### 5.2 Depth Test (Z-Buffer)

When multiple fragments compete for the same pixel, the **depth test** determines which one is visible. The GPU maintains a **depth buffer** (Z-buffer) -- a 2D array storing the depth (distance from camera) of the closest fragment seen so far at each pixel.

Algorithm:
1. For each incoming fragment at position $(x, y)$ with depth $z$:
2. Compare $z$ with the current value in the depth buffer at $(x, y)$
3. If $z < \text{depth\_buffer}[x][y]$ (fragment is closer): update both the color buffer and depth buffer
4. Otherwise: discard the fragment

This elegantly solves the **hidden surface problem** regardless of the order triangles are drawn.

### 5.3 Stencil Test

The **stencil buffer** is an additional per-pixel integer buffer used for masking effects:
- Shadow volumes
- Portal rendering
- Outline/silhouette effects
- Reflections in mirrors

### 5.4 Blending

For transparent objects, fragments are not simply accepted or rejected. Instead, the incoming fragment color is **blended** with the existing color in the framebuffer:

$$C_{\text{final}} = \alpha_{\text{src}} \cdot C_{\text{src}} + (1 - \alpha_{\text{src}}) \cdot C_{\text{dst}}$$

Where $\alpha_{\text{src}}$ is the opacity of the incoming fragment (0 = fully transparent, 1 = fully opaque).

> **Ordering problem**: Blending is order-dependent. Transparent objects must be rendered back-to-front (painter's algorithm) for correct results. This is why the CPU sorts transparent objects in the application stage.

---

## 6. Frame Presentation

### 6.1 Double Buffering

Without double buffering, the display would show the framebuffer *while it is being drawn to*, causing visible tearing artifacts. Double buffering uses two framebuffers:

- **Front buffer**: Currently displayed on screen
- **Back buffer**: Being rendered to by the GPU

When rendering completes, the buffers are **swapped**. The viewer only ever sees completed frames.

```
Time ──────────────────────────────────────────────────────▶

Frame N:   [  GPU renders to Back   ] ◄── swap ──▶ [Display shows Front]
Frame N+1: [  GPU renders to Back   ] ◄── swap ──▶ [Display shows Front]
```

### 6.2 VSync

**Vertical Synchronization (VSync)** synchronizes buffer swaps with the display's refresh rate (typically 60 Hz or 144 Hz). Without vsync:
- If the GPU finishes a frame mid-refresh, swapping causes **screen tearing** (top half shows old frame, bottom half shows new frame)

With vsync:
- Buffer swap waits for the display's **vertical blank interval**
- No tearing, but potential input lag (frame must wait for next refresh cycle)
- If a frame takes longer than one refresh period, FPS drops to a fraction (60 FPS &rarr; 30 FPS)

### 6.3 Triple Buffering

Triple buffering adds a third buffer to mitigate vsync's latency penalty:

| Scheme | Buffers | Tearing | Latency | FPS behavior |
|--------|---------|---------|---------|--------------|
| No vsync | 2 | Yes | Low | Uncapped |
| VSync (double) | 2 | No | Higher | Drops to 30 if < 60 |
| VSync (triple) | 3 | No | Medium | Smoother degradation |

With triple buffering, the GPU always has a back buffer available to render to, even if the previously completed frame is still waiting for the vsync swap.

---

## 7. The Modern Pipeline

The classic pipeline described above has been extended with additional programmable and optional stages.

### 7.1 Tessellation

Tessellation dynamically subdivides geometry into finer triangles on the GPU, enabling:
- Adaptive level of detail (more triangles close to camera)
- Displacement mapping (sculpting surfaces with textures)
- Smooth curves and surfaces from coarse control meshes

The tessellation pipeline inserts two new stages:

```
Vertex Shader ──▶ Tessellation Control Shader ──▶ Tessellator (fixed)
                                                        │
                                                        ▼
                                        Tessellation Evaluation Shader ──▶ ...
```

- **Tessellation Control Shader (TCS)**: Determines *how much* to subdivide (tessellation levels)
- **Tessellator**: Fixed-function stage that generates new vertices
- **Tessellation Evaluation Shader (TES)**: Positions the new vertices

### 7.2 Geometry Shaders

The **geometry shader** sits between vertex processing and rasterization. It receives a complete primitive (e.g., a triangle) and can:
- Emit zero or more output primitives
- Change primitive type (input triangles, output points for particle effects)
- Create new geometry (shadow volume extrusion, fur/grass generation)

> **Performance note**: Geometry shaders are generally slower than expected due to their variable output size. For tasks like particle systems, **compute shaders** are often preferred.

### 7.3 Compute Shaders

**Compute shaders** break free from the fixed pipeline structure entirely. They are general-purpose GPU programs that can:
- Read and write arbitrary buffer data
- Synchronize threads within a workgroup
- Perform any parallel computation

Common uses:
- Particle simulation
- Physics calculations
- Image post-processing
- GPU-driven culling (the GPU decides what to draw, reducing CPU overhead)

### 7.4 Full Modern Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MODERN GRAPHICS PIPELINE                         │
│                                                                     │
│  Input Assembly                                                     │
│       │                                                             │
│       ▼                                                             │
│  Vertex Shader ◄──── Programmable                                   │
│       │                                                             │
│       ▼                                                             │
│  Tessellation Control Shader ◄──── Programmable (optional)          │
│       │                                                             │
│       ▼                                                             │
│  Tessellator ◄──── Fixed-function                                   │
│       │                                                             │
│       ▼                                                             │
│  Tessellation Evaluation Shader ◄──── Programmable (optional)       │
│       │                                                             │
│       ▼                                                             │
│  Geometry Shader ◄──── Programmable (optional)                      │
│       │                                                             │
│       ▼                                                             │
│  Clipping + Perspective Division ◄──── Fixed-function               │
│       │                                                             │
│       ▼                                                             │
│  Rasterization ◄──── Fixed-function                                 │
│       │                                                             │
│       ▼                                                             │
│  Fragment Shader ◄──── Programmable                                 │
│       │                                                             │
│       ▼                                                             │
│  Per-Fragment Operations ◄──── Configurable                         │
│  (depth test, stencil, blending)                                    │
│       │                                                             │
│       ▼                                                             │
│  Framebuffer                                                        │
│                                                                     │
│  ─── Compute Shader ◄──── Programmable (independent, any time)      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Pipeline Performance Considerations

Understanding the pipeline helps identify bottlenecks:

### 8.1 Where Bottlenecks Occur

| Stage | Bottleneck Indicator | Mitigation |
|-------|---------------------|------------|
| **Application (CPU)** | GPU idle, low GPU usage | Reduce draw calls, use instancing |
| **Vertex processing** | High vertex count, complex vertex shader | Simplify meshes, LOD |
| **Rasterization** | Large triangles, high resolution | Reduce overdraw |
| **Fragment processing** | Complex shaders, many texture samples | Optimize shaders, use mipmaps |
| **Bandwidth** | High-res textures, large framebuffers | Compress textures, lower resolution |

### 8.2 Early-Z Optimization

Modern GPUs can perform the depth test *before* the fragment shader runs (**early-Z**). If the depth test fails, the expensive fragment shader is skipped entirely. This is why rendering opaque objects **front-to-back** improves performance -- more fragments fail the early depth test and are skipped.

> **Caveat**: If a fragment shader writes to `gl_FragDepth` or uses `discard`, early-Z is disabled for that draw call, because the GPU cannot know the final depth value in advance.

### 8.3 Overdraw

**Overdraw** occurs when multiple fragments are shaded for the same pixel. An overdraw ratio of 2x means each pixel was shaded twice on average. Overdraw wastes fragment shader cycles.

Strategies to reduce overdraw:
1. Front-to-back rendering (with early-Z)
2. Occlusion culling (skip entirely hidden objects)
3. Deferred rendering (shade each pixel only once by decoupling geometry and lighting passes)

---

## 9. A Complete Frame: Putting It All Together

Let us trace a single frame rendering a textured, lit teapot:

```python
"""
Conceptual trace of one frame through the graphics pipeline.
This is pseudocode to illustrate the flow, not runnable code.
"""

# ═══════════════════════════════════════════════════════
# STAGE 1: APPLICATION (CPU)
# ═══════════════════════════════════════════════════════
# The teapot has 6,320 vertices and 3,752 triangles.
# The CPU has already loaded the mesh into GPU memory.

camera.update(user_input)           # Process mouse/keyboard
teapot.model_matrix = rotate_y(time * 30)  # Spin the teapot

# CPU issues draw call:
# "GPU, draw teapot.mesh with teapot.material"
gpu.draw_call(teapot)

# ═══════════════════════════════════════════════════════
# STAGE 2: GEOMETRY PROCESSING (GPU, per vertex)
# ═══════════════════════════════════════════════════════
# For each of the 6,320 vertices IN PARALLEL:
#   1. Vertex shader transforms position: object → clip space
#   2. Normal is transformed to world space
#   3. UV coordinates are passed through

# Clipping: triangles outside the frustum are discarded.
# Remaining triangles: ~2,800 (some culled, some clipped)

# Perspective division: clip → NDC
# Viewport transform: NDC → screen coordinates

# ═══════════════════════════════════════════════════════
# STAGE 3: RASTERIZATION (GPU, per triangle)
# ═══════════════════════════════════════════════════════
# For each of ~2,800 visible triangles IN PARALLEL:
#   Determine which pixels the triangle covers
#   Generate fragments with interpolated attributes
# Total fragments generated: ~350,000 (at 1080p resolution)

# ═══════════════════════════════════════════════════════
# STAGE 4: FRAGMENT PROCESSING (GPU, per fragment)
# ═══════════════════════════════════════════════════════
# For each of ~350,000 fragments IN PARALLEL:
#   1. Early-Z test: skip if behind existing geometry
#   2. Fragment shader:
#      - Sample albedo texture at interpolated UV
#      - Compute Phong lighting with interpolated normal
#      - Output final RGBA color
#   3. Final depth test and write to depth buffer
#   4. Write color to framebuffer

# ═══════════════════════════════════════════════════════
# STAGE 5: FRAME PRESENTATION
# ═══════════════════════════════════════════════════════
# Wait for vsync
# Swap front and back buffers
# The teapot appears on screen!
```

---

## 10. Historical Context and API Landscape

| API | Platform | Pipeline Model | Notes |
|-----|----------|----------------|-------|
| **OpenGL** (1992) | Cross-platform | Fixed &rarr; Programmable | Legacy but still widely taught |
| **Direct3D 11** (2009) | Windows | Programmable | Dominant in PC gaming |
| **OpenGL ES** (2003) | Mobile | Subset of OpenGL | iOS, Android |
| **WebGL** (2011) | Browsers | Based on OpenGL ES 2.0/3.0 | We use this in Lesson 07 |
| **Vulkan** (2016) | Cross-platform | Low-level, explicit | Maximum control and performance |
| **Direct3D 12** (2015) | Windows | Low-level, explicit | Similar philosophy to Vulkan |
| **Metal** (2014) | Apple | Low-level, explicit | macOS, iOS |
| **WebGPU** (2023) | Browsers | Modern, explicit | Successor to WebGL |

> **Trend**: The industry is moving toward **lower-level, explicit APIs** (Vulkan, D3D12, Metal, WebGPU) that give developers more control over GPU resources, at the cost of increased complexity. Higher-level APIs and engines (Unity, Unreal) abstract these differences.

---

## Summary

| Pipeline Stage | Location | Programmable? | Key Operation |
|---------------|----------|---------------|---------------|
| Application | CPU | Yes (your code) | Scene setup, culling, draw calls |
| Vertex Processing | GPU | Yes (vertex shader) | Transform vertices to clip space |
| Tessellation | GPU | Yes (TCS + TES) | Subdivide geometry (optional) |
| Geometry Shader | GPU | Yes | Emit/modify primitives (optional) |
| Clipping | GPU | No (fixed-function) | Remove geometry outside frustum |
| Rasterization | GPU | No (fixed-function) | Convert triangles to fragments |
| Fragment Processing | GPU | Yes (fragment shader) | Compute per-pixel color |
| Per-Fragment Ops | GPU | Configurable | Depth test, stencil, blending |
| Frame Presentation | GPU/Display | Configurable | Double/triple buffering, vsync |

**Key takeaways**:
- The pipeline is an assembly line optimized for throughput through parallelism
- Vertices flow through coordinate space transformations: Object &rarr; World &rarr; Eye &rarr; Clip &rarr; NDC &rarr; Screen
- Rasterization bridges the gap between continuous geometry and discrete pixels
- Fragment shaders are where most visual quality computation happens
- Understanding the pipeline is essential for optimizing rendering performance

---

## Exercises

1. **Conceptual**: A scene contains 10,000 triangles but only 3,000 are visible after frustum culling. The screen resolution is 1920x1080. Estimate the maximum number of fragments generated (assuming each visible triangle covers an average of 100 pixels). How does overdraw affect this number?

2. **Pipeline Tracing**: Trace a single vertex with position $(1, 2, 3)$ through the pipeline stages. At each stage, describe what happens to it conceptually (you don't need to compute exact values -- just describe the operations).

3. **Bottleneck Analysis**: A game renders at 30 FPS. When you halve the screen resolution, FPS jumps to 58. When you halve the number of triangles instead (at original resolution), FPS stays at 32. Where is the bottleneck? Why?

4. **Draw Call Reduction**: You have a forest scene with 1,000 trees, each drawn with a separate draw call. Propose two strategies to reduce the number of draw calls while maintaining visual quality.

5. **Transparency Challenge**: Explain why rendering transparent objects is more difficult than opaque objects. What happens if you render transparent objects in random order? What about the case where two transparent objects overlap each other?

6. **Modern Extensions**: Compare tessellation shaders and geometry shaders. For the task of generating grass blades on a terrain, which approach would you choose and why?

---

## Further Reading

1. Marschner, S. & Shirley, P. *Fundamentals of Computer Graphics* (5th ed.), Ch. 8 -- "The Graphics Pipeline"
2. Akenine-Moller, T., Haines, E., & Hoffman, N. *Real-Time Rendering* (4th ed.), Ch. 2 -- "The Graphics Rendering Pipeline"
3. [Learn OpenGL -- Getting Started](https://learnopengl.com/Getting-started/OpenGL) -- Practical introduction to the OpenGL pipeline
4. [Life of a Triangle (NVIDIA)](https://developer.nvidia.com/content/life-triangle-nvidias-logical-pipeline) -- How triangles traverse NVIDIA's GPU architecture
5. [Vulkan Tutorial](https://vulkan-tutorial.com/) -- Modern low-level pipeline programming
