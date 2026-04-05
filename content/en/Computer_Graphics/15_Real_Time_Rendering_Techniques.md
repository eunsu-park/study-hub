# 15. Real-Time Rendering Techniques

[← Previous: GPU Computing](14_GPU_Computing.md) | [Next: Modern Graphics APIs Overview →](16_Modern_Graphics_APIs_Overview.md)

---

## Learning Objectives

1. Compare forward and deferred rendering pipelines and their trade-offs
2. Understand the G-buffer and its role in decoupling geometry from lighting
3. Implement shadow mapping techniques: basic, PCF, and cascaded shadow maps
4. Explain Screen-Space Ambient Occlusion (SSAO) and its approximation of indirect shadows
5. Describe bloom, HDR rendering, and tone mapping for photorealistic output
6. Understand Level of Detail (LOD) strategies for managing scene complexity
7. Explain temporal anti-aliasing (TAA) and its advantages over spatial methods
8. Build a deferred shading G-buffer simulation in Python

---

## Why This Matters

Rendering a scene at 60+ frames per second with shadows, reflections, ambient occlusion, bloom, and anti-aliasing requires a carefully orchestrated pipeline. Each technique in this lesson represents a hard-won solution to a specific rendering challenge, and together they form the foundation of every modern game engine (Unreal, Unity, Godot) and real-time 3D application.

Understanding these techniques is essential not only for engine developers but for anyone who uses these engines effectively -- artists, technical artists, and gameplay programmers all benefit from knowing *why* shadows look the way they do, *how* post-processing affects performance, and *where* the GPU spends its time.

---

## 1. Forward vs. Deferred Rendering

### 1.1 Forward Rendering

In **forward rendering**, each object is shaded completely in a single pass. The vertex shader transforms geometry, and the fragment shader evaluates all lights:

```
For each object:
    For each pixel the object covers:
        For each light:
            Compute lighting contribution
        Output final color
```

**Pros**:
- Simple implementation
- Handles transparency naturally
- Low memory bandwidth (no G-buffer)
- MSAA works directly

**Cons**:
- $O(objects \times lights)$ -- expensive with many lights
- Overdraw: fragments may be shaded but then occluded by closer objects
- Each object must evaluate all lights (or use light culling)

### 1.2 Deferred Rendering

**Deferred rendering** splits the process into two passes:

**Pass 1 (Geometry pass)**: Render all objects, but instead of computing lighting, store surface attributes into a set of textures called the **G-buffer**.

**Pass 2 (Lighting pass)**: For each pixel, read the G-buffer and compute lighting. Each light only affects the pixels it reaches.

```
Pass 1 (Geometry):
    For each object:
        Store position, normal, albedo, material into G-buffer textures

Pass 2 (Lighting):
    For each pixel:
        Read G-buffer
        For each light affecting this pixel:
            Compute lighting contribution
        Output final color
```

**Pros**:
- Lighting cost is $O(pixels \times lights)$, independent of scene complexity
- No overdraw wasted on lighting (only the visible surface is lit)
- Easy to add post-processing effects (all data is in screen space)
- Scales well with many lights (hundreds or thousands)

**Cons**:
- High memory bandwidth (G-buffer can be 64-128 bytes per pixel)
- Transparency is difficult (G-buffer stores one surface per pixel)
- MSAA is expensive with G-buffer (alternative: TAA)

### 1.3 Forward+ (Tiled Forward)

A hybrid approach: divide the screen into tiles, determine which lights affect each tile, then shade each pixel with only the relevant lights. Combines forward rendering's transparency support with deferred's light scalability.

### 1.4 Comparison Table

| Feature | Forward | Deferred | Forward+ |
|---------|---------|----------|----------|
| Light scalability | Poor | Excellent | Good |
| Memory bandwidth | Low | High | Moderate |
| Transparency | Natural | Difficult | Natural |
| MSAA support | Native | Complex | Native |
| Overdraw cost | High | Low | High |
| Implementation | Simple | Moderate | Complex |

---

## 2. G-Buffer

### 2.1 Structure

The **G-buffer** (Geometry Buffer) stores per-pixel surface information in multiple render targets (MRT):

| Texture | Format | Content |
|---------|--------|---------|
| RT0: Albedo + AO | RGBA8 | Diffuse color (RGB) + ambient occlusion (A) |
| RT1: Normal | RGB16F or RGB10A2 | World-space or view-space normal |
| RT2: Material | RGBA8 | Metallic (R), Roughness (G), Emissive (B), Flags (A) |
| Depth | D24S8 or D32F | Hardware depth buffer |

**Total**: Approximately 12-16 bytes per pixel, or ~37 MB at 1920x1080.

### 2.2 Normal Encoding

Storing full 3D normals (3 floats = 12 bytes) is expensive. Common compressions:

**Octahedral encoding** (2 floats, 8 bytes): Map the unit sphere to a 2D octahedron, then unfold to a square. Excellent quality with compact storage.

**Spherical coordinates** (2 floats): $(\theta, \phi)$. Simple but non-uniform distribution near poles.

**View-space Z reconstruction**: Store only X, Y components; reconstruct $Z = \sqrt{1 - X^2 - Y^2}$ (only works for normals facing the camera).

### 2.3 Position Reconstruction

Rather than storing world position explicitly, reconstruct it from the depth buffer:

1. Read depth value $z_{\text{buffer}}$ at pixel $(x, y)$
2. Compute NDC: $(x_{\text{ndc}}, y_{\text{ndc}}, z_{\text{ndc}})$
3. Transform through inverse projection matrix: $\mathbf{p}_{\text{view}} = \mathbf{P}^{-1} \cdot (x_{\text{ndc}}, y_{\text{ndc}}, z_{\text{ndc}}, 1)$
4. Divide by $w$: $\mathbf{p}_{\text{view}} = \mathbf{p}_{\text{view}} / p_w$

This saves one full G-buffer texture (12 bytes/pixel).

### 2.4 Python Implementation: G-Buffer Simulation

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class GBufferPixel:
    """Contents of one G-buffer pixel."""
    albedo: np.ndarray       # (3,) RGB diffuse color
    normal: np.ndarray       # (3,) world-space unit normal
    depth: float             # Linear depth from camera
    metallic: float          # 0.0 = dielectric, 1.0 = metallic
    roughness: float         # 0.0 = mirror, 1.0 = rough
    emissive: float          # Emission intensity


class GBuffer:
    """
    Simulated G-buffer for deferred rendering.
    Stores per-pixel geometry and material attributes.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Why separate arrays: mirrors GPU MRT (Multiple Render Targets)
        self.albedo   = np.zeros((height, width, 3))     # RT0: RGB
        self.normal   = np.zeros((height, width, 3))     # RT1: XYZ
        self.depth    = np.full((height, width), np.inf)  # Depth buffer
        self.metallic = np.zeros((height, width))         # RT2: R channel
        self.roughness = np.ones((height, width)) * 0.5   # RT2: G channel
        self.emissive = np.zeros((height, width))         # RT2: B channel

    def write(self, x, y, pixel: GBufferPixel):
        """Write geometry pass output to G-buffer (with depth test)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Why depth test here: only store the nearest surface
            if pixel.depth < self.depth[y, x]:
                self.depth[y, x] = pixel.depth
                self.albedo[y, x] = pixel.albedo
                self.normal[y, x] = pixel.normal
                self.metallic[y, x] = pixel.metallic
                self.roughness[y, x] = pixel.roughness
                self.emissive[y, x] = pixel.emissive

    def memory_usage_mb(self):
        """Estimate GPU memory usage of this G-buffer."""
        bytes_per_pixel = (
            3 * 4 +  # albedo: 3 floats
            3 * 4 +  # normal: 3 floats (or 2 with encoding)
            4 +      # depth: 1 float
            4 +      # metallic: 1 float
            4 +      # roughness: 1 float
            4        # emissive: 1 float
        )
        total = self.width * self.height * bytes_per_pixel
        return total / (1024 * 1024)


@dataclass
class PointLight:
    position: np.ndarray
    color: np.ndarray
    intensity: float
    radius: float           # Attenuation radius


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def deferred_lighting(gbuffer: GBuffer, lights: List[PointLight],
                      camera_pos: np.ndarray) -> np.ndarray:
    """
    Lighting pass: read G-buffer, compute PBR lighting for each pixel.
    This is the core of deferred rendering.
    """
    output = np.zeros((gbuffer.height, gbuffer.width, 3))

    for y in range(gbuffer.height):
        for x in range(gbuffer.width):
            if gbuffer.depth[y, x] >= 1e8:
                continue  # No geometry at this pixel

            albedo = gbuffer.albedo[y, x]
            N = gbuffer.normal[y, x]
            depth = gbuffer.depth[y, x]
            metallic = gbuffer.metallic[y, x]
            roughness = gbuffer.roughness[y, x]
            emissive = gbuffer.emissive[y, x]

            # Reconstruct world position from depth (simplified)
            # In a real renderer, this uses the inverse projection matrix
            world_pos = np.array([
                (x / gbuffer.width - 0.5) * 2 * depth,
                (0.5 - y / gbuffer.height) * 2 * depth,
                -depth
            ])

            V = normalize(camera_pos - world_pos)

            color = np.zeros(3)

            # Ambient (very simple approximation)
            ambient = 0.03 * albedo
            color += ambient

            # Emission
            color += emissive * albedo

            # Per-light contribution
            for light in lights:
                L = light.position - world_pos
                dist = np.linalg.norm(L)

                if dist > light.radius:
                    continue  # Light is out of range

                L = L / dist

                # Attenuation
                attenuation = 1.0 / (1.0 + dist * dist)
                attenuation *= max(0, 1.0 - dist / light.radius)

                # Diffuse (Lambertian)
                NdotL = max(0, np.dot(N, L))

                # Specular (Blinn-Phong simplified)
                H = normalize(L + V)
                NdotH = max(0, np.dot(N, H))
                spec_power = max(1, (1.0 - roughness) * 256)
                specular = NdotH ** spec_power

                # Mix diffuse and specular based on metallic
                diffuse_contribution = (1.0 - metallic) * albedo * NdotL
                specular_contribution = (metallic * albedo + (1 - metallic) * 0.04) * specular

                color += (diffuse_contribution + specular_contribution) \
                         * light.color * light.intensity * attenuation

            output[y, x] = color

    return output


# --- Demo: Fill G-buffer and run lighting pass ---

width, height = 80, 60
gbuf = GBuffer(width, height)

# Write some simple geometry to the G-buffer
# A "floor" at depth 5, normal pointing up
for y in range(height // 2, height):
    for x in range(width):
        gbuf.write(x, y, GBufferPixel(
            albedo=np.array([0.6, 0.6, 0.6]),
            normal=np.array([0, 1, 0]),
            depth=5.0 + (y - height // 2) * 0.1,
            metallic=0.0,
            roughness=0.7,
            emissive=0.0,
        ))

# A "sphere" (simplified: just a circle in the center)
cx, cy, r = width // 2, height // 3, min(width, height) // 6
for y in range(max(0, cy - r), min(height, cy + r)):
    for x in range(max(0, cx - r), min(width, cx + r)):
        dx, dy = (x - cx) / r, (y - cy) / r
        if dx * dx + dy * dy <= 1.0:
            dz = np.sqrt(max(0, 1 - dx * dx - dy * dy))
            gbuf.write(x, y, GBufferPixel(
                albedo=np.array([0.8, 0.2, 0.2]),
                normal=normalize(np.array([dx, -dy, dz])),
                depth=3.0 - dz * 0.5,
                metallic=0.3,
                roughness=0.3,
                emissive=0.0,
            ))

lights = [
    PointLight(np.array([2, 3, 2]),  np.array([1, 1, 0.9]),   2.0, 15.0),
    PointLight(np.array([-3, 2, 1]), np.array([0.3, 0.5, 1]), 1.5, 12.0),
]

camera_pos = np.array([0, 0, 5])

print("Running deferred lighting pass...")
result = deferred_lighting(gbuf, lights, camera_pos)

# Tone map and gamma correct
result = result / (1.0 + result)            # Reinhard
result = np.power(np.clip(result, 0, 1), 1.0 / 2.2)  # Gamma

print(f"G-buffer memory: {gbuf.memory_usage_mb():.2f} MB")
print(f"Output shape: {result.shape}")
print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(gbuf.albedo); axes[0, 0].set_title('G-Buffer: Albedo')
    norm_vis = (gbuf.normal + 1) / 2
    axes[0, 1].imshow(norm_vis); axes[0, 1].set_title('G-Buffer: Normal')
    axes[0, 2].imshow(gbuf.depth, cmap='gray_r'); axes[0, 2].set_title('G-Buffer: Depth')
    axes[1, 0].imshow(gbuf.roughness, cmap='gray'); axes[1, 0].set_title('G-Buffer: Roughness')
    axes[1, 1].imshow(gbuf.metallic, cmap='gray'); axes[1, 1].set_title('G-Buffer: Metallic')
    axes[1, 2].imshow(result); axes[1, 2].set_title('Final: Deferred Lit')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('deferred_shading.png', dpi=150)
    plt.close()
    print("Saved deferred_shading.png")
except ImportError:
    print("Install matplotlib for visualization")
```

---

## 3. Shadow Mapping

### 3.1 Basic Shadow Mapping

**Shadow mapping** (Williams, 1978) is a two-pass technique:

**Pass 1 (Shadow map generation)**: Render the scene from the light's perspective. Store only depth values in a texture (the shadow map).

**Pass 2 (Scene rendering)**: For each pixel, project it into the light's coordinate space. Compare its depth to the shadow map value:
- If pixel depth > shadow map depth, the pixel is **in shadow**
- If pixel depth <= shadow map depth, the pixel is **lit**

$$\text{shadow} = \begin{cases} 0 & \text{if } z_{\text{pixel}} > z_{\text{shadow map}} + \text{bias} \\ 1 & \text{otherwise} \end{cases}$$

### 3.2 Shadow Bias

The **shadow bias** prevents **shadow acne** -- a pattern of self-shadowing artifacts caused by the limited resolution of the shadow map. Without bias, surfaces facing the light incorrectly shadow themselves.

$$z_{\text{test}} = z_{\text{pixel}} - \text{bias}$$

Too little bias: shadow acne. Too much bias: **Peter Panning** (shadows detach from objects).

**Slope-scale bias** adapts to surface orientation:

$$\text{bias} = \text{constant\_bias} + \text{slope\_factor} \cdot \tan(\theta)$$

where $\theta$ is the angle between the surface normal and the light direction.

### 3.3 Percentage-Closer Filtering (PCF)

Basic shadow mapping produces hard-edged (aliased) shadows. **PCF** softens shadows by sampling multiple shadow map texels and averaging:

$$\text{shadow} = \frac{1}{N} \sum_{i=1}^{N} \text{compare}(z_{\text{pixel}}, z_{\text{shadow map}}(\mathbf{s}_i))$$

where $\mathbf{s}_i$ are offset sample positions in the shadow map. A 3x3 or 5x5 kernel is typical.

**Important**: PCF tests each sample *before* filtering (compare, then average), not the reverse. Filtering depth values and then comparing produces incorrect results.

### 3.4 Cascaded Shadow Maps (CSM)

For large outdoor scenes, a single shadow map cannot cover the entire view with sufficient resolution. **CSM** splits the view frustum into cascades (typically 3-4), each with its own shadow map:

```
Camera                                          Far
  |----[Cascade 0]----[Cascade 1]--------[Cascade 2]---------|
        512x512         512x512            512x512
       (near, high     (mid, medium      (far, low
        resolution)     resolution)       resolution)
```

Near objects get high-resolution shadows; distant objects get lower resolution. The shader selects the appropriate cascade based on pixel depth.

### 3.5 Shadow Map Variants

| Technique | Pros | Cons |
|-----------|------|------|
| Basic shadow map | Simple, fast | Hard edges, aliasing |
| PCF | Softer edges | Uniform kernel, still aliased |
| Variance Shadow Maps (VSM) | Filterable, soft shadows | Light bleeding |
| Exponential Shadow Maps (ESM) | Fast filtering | Over-darkening at edges |
| Moment Shadow Maps | No light bleeding | Higher memory + computation |
| Ray-traced shadows | Perfect accuracy, soft shadows | GPU ray tracing required |

---

## 4. Screen-Space Ambient Occlusion (SSAO)

### 4.1 Concept

**Ambient occlusion (AO)** darkens areas where surfaces are close together (corners, crevices, under furniture). It approximates the integral of indirect light visibility:

$$AO(\mathbf{p}) = \frac{1}{\pi} \int_{\Omega} V(\mathbf{p}, \omega) (\omega \cdot \mathbf{n})\, d\omega$$

where $V(\mathbf{p}, \omega) = 0$ if an occluder is nearby in direction $\omega$.

Computing true AO is expensive. **SSAO** (Crytek, 2007) approximates it using only screen-space data (depth and normal buffers from the G-buffer).

### 4.2 Algorithm

1. For each pixel, sample $N$ random points in a hemisphere around the surface normal
2. Project each sample into screen space and read the depth buffer
3. If the sample is **behind** the depth buffer surface (occluded), it contributes to occlusion
4. Average the results and apply as a darkening factor

$$AO(\mathbf{p}) \approx 1 - \frac{1}{N}\sum_{k=1}^{N}\begin{cases} 1 & \text{if sample}_k \text{ is occluded} \\ 0 & \text{otherwise} \end{cases}$$

### 4.3 SSAO Improvements

**HBAO (Horizon-Based AO)**: Traces rays along the depth buffer in screen space, measuring the "horizon angle" above each surface point. More physically accurate than sample-based SSAO.

**GTAO (Ground Truth AO)**: Uses a spatiotemporal approach with analytical integration. Better quality per sample.

**RTAO (Ray-Traced AO)**: Traces actual rays using hardware ray tracing. Ground truth quality, increasingly used in modern engines.

### 4.4 Implementation Sketch

```python
def ssao_sample(gbuffer, x, y, samples=16, radius=0.5):
    """
    Simplified SSAO computation for a single pixel.
    In practice, this runs as a full-screen compute/fragment shader.
    """
    if gbuffer.depth[y, x] >= 1e8:
        return 1.0  # No geometry -> no occlusion

    center_depth = gbuffer.depth[y, x]
    normal = gbuffer.normal[y, x]

    occlusion = 0.0
    for _ in range(samples):
        # Random offset in hemisphere around the normal
        rand_dir = np.random.randn(3)
        rand_dir = normalize(rand_dir)
        if np.dot(rand_dir, normal) < 0:
            rand_dir = -rand_dir  # Ensure hemisphere above surface

        # Scale by random radius
        offset = rand_dir * radius * np.random.random()

        # Project to screen space (simplified)
        sample_x = int(x + offset[0] * gbuffer.width * 0.05)
        sample_y = int(y - offset[1] * gbuffer.height * 0.05)
        sample_depth = center_depth - offset[2]

        # Bounds check
        if not (0 <= sample_x < gbuffer.width and 0 <= sample_y < gbuffer.height):
            continue

        # Compare: is the sample behind the depth buffer?
        buffer_depth = gbuffer.depth[sample_y, sample_x]
        if buffer_depth < sample_depth and (sample_depth - buffer_depth) < radius:
            occlusion += 1.0

    # Why (1 - ao): 0 = fully occluded, 1 = fully visible
    ao = 1.0 - (occlusion / samples)
    return ao
```

---

## 5. Bloom and HDR Rendering

### 5.1 High Dynamic Range (HDR)

Real-world luminance varies enormously:
- Starlight: $10^{-3}$ cd/m$^2$
- Indoor room: $10^{1}$ cd/m$^2$
- Sunlit snow: $10^{4}$ cd/m$^2$
- Sun's surface: $10^{9}$ cd/m$^2$

Displays can show roughly $10^{0}$ to $10^{3}$ cd/m$^2$. **HDR rendering** computes lighting in full floating-point range (FP16 or FP32 render targets), then applies **tone mapping** to compress the range for display.

### 5.2 Tone Mapping Operators

**Reinhard**:
$$L_{\text{display}} = \frac{L}{1 + L}$$

Simple and effective; maps $[0, \infty)$ to $[0, 1)$.

**ACES (Academy Color Encoding System)**: The industry standard for film and games:

$$f(x) = \frac{x(ax + b)}{x(cx + d) + e}$$

with constants $a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14$.

**Uncharted 2 (Hable)**:
$$f(x) = \frac{(x(Ax + CB) + DE)}{(x(Ax + B) + DF)} - E/F$$

A filmic curve with a nice shoulder and toe.

### 5.3 Bloom

**Bloom** simulates the glow around bright objects caused by light scattering in camera lenses and the human eye:

1. **Bright pass**: Extract pixels above a brightness threshold from the HDR buffer
2. **Blur**: Apply a wide Gaussian blur (often using downsampled mip chain for efficiency)
3. **Composite**: Add the blurred bright pixels back to the original image

```
HDR Image → Bright Pass → Downsample → Blur → Upsample → Add → Final
                            ↓                    ↑
                         Downsample → Blur → Upsample
                            ↓                    ↑
                         Downsample → Blur → Upsample
```

The multi-scale approach (downsample, blur, upsample at each level) creates a natural-looking bloom with both tight and wide halos.

### 5.4 Bloom Implementation

```python
def extract_bright(image, threshold=1.0):
    """Extract pixels brighter than threshold (simulated bright pass)."""
    brightness = np.max(image, axis=2)  # Max of RGB
    mask = (brightness > threshold).astype(float)[:, :, np.newaxis]
    return image * mask


def box_downsample(image, factor=2):
    """Downsample by averaging blocks of pixels."""
    h, w = image.shape[:2]
    nh, nw = h // factor, w // factor
    return image[:nh*factor, :nw*factor].reshape(nh, factor, nw, factor, -1).mean(axis=(1, 3))


def box_upsample(image, target_shape):
    """Upsample using nearest-neighbor."""
    h, w = target_shape[:2]
    sh, sw = image.shape[:2]
    result = np.zeros((h, w, image.shape[2]))
    for y in range(h):
        for x in range(w):
            sy = min(y * sh // h, sh - 1)
            sx = min(x * sw // w, sw - 1)
            result[y, x] = image[sy, sx]
    return result


def simple_bloom(hdr_image, threshold=0.8, levels=3, strength=0.5):
    """
    Multi-level bloom: extract bright pixels, blur at multiple resolutions,
    then composite back onto the original image.
    """
    bright = extract_bright(hdr_image, threshold)

    # Build mip chain (downsampled + blurred versions)
    mips = [bright]
    current = bright
    for _ in range(levels):
        current = box_downsample(current)
        mips.append(current)

    # Upsample and accumulate bloom
    bloom = np.zeros_like(hdr_image)
    for mip in mips:
        upsampled = box_upsample(mip, hdr_image.shape)
        bloom += upsampled

    bloom /= len(mips)

    return hdr_image + strength * bloom
```

---

## 6. Level of Detail (LOD)

### 6.1 Concept

Objects far from the camera occupy fewer pixels and do not need full geometric detail. **LOD** systems switch between pre-built mesh versions of decreasing complexity:

| LOD Level | Distance | Triangles | Use Case |
|-----------|----------|-----------|----------|
| LOD 0 | Near (<10m) | 10,000 | Full detail |
| LOD 1 | Medium (10-50m) | 2,500 | Reduced detail |
| LOD 2 | Far (50-200m) | 500 | Low poly |
| LOD 3 | Very far (>200m) | 50 | Billboard / imposter |

### 6.2 LOD Selection

The LOD level is selected based on **screen-space size** (projected area in pixels):

$$\text{screen\_size} = \frac{r_{\text{bounding sphere}}}{\text{distance}} \cdot \text{projection\_scale}$$

If screen_size < threshold, switch to a lower LOD.

### 6.3 LOD Transitions

**Discrete LOD**: Pop between levels (visible pop). Mitigated with a **hysteresis band** (switch up at a different distance than switching down).

**Cross-fade (dithered LOD)**: Render both LODs with alpha dithering during the transition. No pop, slight rendering cost.

**Continuous LOD (CLOD)**: Progressively simplify the mesh (Hoppe's progressive meshes). Rarely used in games due to complexity.

**Nanite (Unreal 5)**: Virtualized geometry system that streams and renders the appropriate detail level per-cluster using GPU-driven rendering. Handles billions of triangles seamlessly.

### 6.4 Other LOD Targets

LOD applies beyond meshes:
- **Texture LOD**: Mipmaps (already covered in L06)
- **Shader LOD**: Simpler shading models for distant objects
- **Animation LOD**: Reduce bone count or update frequency for distant characters
- **Particle LOD**: Fewer particles, larger sizes for distant effects

---

## 7. Temporal Anti-Aliasing (TAA)

### 7.1 The Aliasing Problem

Aliasing occurs when continuous geometry is sampled discretely at pixel locations. Jagged edges, flickering specular highlights, and shimmering textures are all aliasing artifacts.

### 7.2 Spatial Anti-Aliasing Review

| Method | Samples/Pixel | Cost | Quality |
|--------|--------------|------|---------|
| SSAA (Supersampling) | 4-16x | Very high | Excellent |
| MSAA (Multisample) | 2-8x | Moderate | Good (edges only) |
| FXAA (Fast Approximate) | 1x + post-process | Low | Fair (blurry) |
| SMAA (Enhanced Subpixel) | 1x + post-process | Low | Good |

### 7.3 Temporal Anti-Aliasing (TAA)

TAA accumulates samples **over time** by jittering the camera position slightly each frame and blending with previous frames:

$$C_{\text{out}} = \alpha \cdot C_{\text{current}} + (1 - \alpha) \cdot C_{\text{history}}$$

where $\alpha \approx 0.05-0.1$ (favor history, which has accumulated many samples).

**Key steps**:
1. **Jitter**: Offset the projection matrix by a sub-pixel amount (from a Halton sequence or other low-discrepancy pattern)
2. **Render**: Render the current frame with the jittered camera
3. **Reproject**: Find where each pixel was in the previous frame using motion vectors ($\mathbf{v} = \mathbf{p}_{\text{current}} - \mathbf{p}_{\text{previous}}$)
4. **Sample history**: Read the history buffer at the reprojected position
5. **Clamp/Clip**: Clamp the history sample to the neighborhood of the current frame to reject stale data (prevents ghosting)
6. **Blend**: Weighted average of current and clamped history

### 7.4 TAA Challenges

- **Ghosting**: Moving objects leave trails if history rejection is too weak
- **Blurring**: Over-blending reduces sharpness (mitigated with sharpening post-process)
- **Disocclusion**: When new surfaces appear (were previously hidden), there is no valid history
- **Sub-pixel jitter artifacts**: High-frequency detail can flicker

### 7.5 Modern Alternatives

**DLSS (NVIDIA)**: Neural network upscales a low-resolution frame using temporal data. Better than TAA at the same performance cost.

**FSR (AMD)**: Spatial upscaling (FSR 1.0) or temporal (FSR 2.0+). Cross-vendor alternative to DLSS.

**XeSS (Intel)**: ML-based temporal upscaling, similar goals to DLSS/FSR.

---

## 8. Putting It All Together: A Real-Time Frame

A typical frame in a modern game engine:

```
1. Shadow Pass
   - Render shadow maps for each shadow-casting light
   - Cascaded shadow maps for the sun

2. Geometry Pass (Deferred)
   - Render all opaque objects → fill G-buffer
   - Depth pre-pass (optional: early Z rejection)

3. SSAO Pass
   - Read G-buffer normals + depth
   - Compute AO term, blur, store in texture

4. Lighting Pass
   - For each pixel: read G-buffer + shadow maps + SSAO
   - Compute PBR direct lighting (all lights)
   - Add environment lighting (IBL / reflection probes)
   - Output to HDR buffer

5. Transparent Pass (Forward)
   - Render transparent objects front-to-back (forward)
   - Read depth buffer for soft particles

6. Post-Processing
   - Bloom: bright pass → blur pyramid → composite
   - Motion blur (velocity buffer)
   - Depth of field
   - Tone mapping (ACES)
   - TAA: jitter + reproject + blend
   - Sharpening (CAS or similar)
   - Color grading (LUT)
   - Gamma correction → output to display

Total: ~5-15 ms per frame (60-200 FPS at 1080p)
```

---

## Summary

| Technique | Key Idea |
|-----------|----------|
| Forward rendering | Shade each object with all lights in one pass; $O(\text{objects} \times \text{lights})$ |
| Deferred rendering | Store surface data in G-buffer; shade per-pixel with per-light; $O(\text{pixels} \times \text{lights})$ |
| G-buffer | MRT storing albedo, normals, depth, material; typically 12-16 bytes/pixel |
| Shadow mapping | Render depth from light view; compare in main render to determine shadow |
| PCF | Average multiple shadow map samples for softer edges |
| Cascaded shadows | Multiple shadow maps at different resolutions for near/mid/far |
| SSAO | Sample hemisphere around each pixel; compare depths for occlusion |
| Bloom | Extract bright pixels, multi-scale blur, add back for glow effect |
| HDR + Tone mapping | Compute in linear HDR; compress to display range (Reinhard, ACES) |
| LOD | Switch to lower-detail meshes for distant objects |
| TAA | Jitter camera each frame; blend with reprojected history; sub-pixel AA over time |

## Exercises

1. **G-buffer visualization**: Extend the G-buffer simulation to render 5 spheres of different materials. Visualize each G-buffer channel as a separate image (albedo, normals, depth, roughness, metallic).

2. **Shadow mapping**: Implement a basic 2D shadow map for a top-down scene with a directional light. Cast rays from the light to build the shadow map, then shade the scene using shadow map lookups.

3. **PCF implementation**: Add PCF to the shadow mapping exercise. Compare 1-sample, 3x3, and 5x5 PCF kernels. Measure the visual quality improvement and performance cost.

4. **SSAO quality**: Implement the SSAO function for the full G-buffer. Compare results with 8, 16, 32, and 64 samples. Add a bilateral blur post-process to smooth the AO without blurring across edges.

5. **Bloom parameter exploration**: Implement the multi-level bloom pipeline. Render a scene with one very bright light and several dim lights. Experiment with threshold (0.5, 1.0, 2.0), bloom strength, and number of blur levels.

6. **Tone mapping comparison**: Implement Reinhard, ACES, and Hable (Uncharted 2) tone mapping. Plot the transfer curves. Apply each to the same HDR image and compare the visual results.

## Further Reading

- Akenine-Moller, T. et al. *Real-Time Rendering*, 4th ed. CRC Press, 2018. (The definitive reference for all topics in this lesson)
- Kaplanyan, A. "Cascaded Shadow Maps." *ShaderX6*, 2007. (CSM implementation details)
- Mittring, M. "Finding Next Gen -- CryEngine 2." *SIGGRAPH Course*, 2007. (Introduction of SSAO)
- Karis, B. "Real Shading in Unreal Engine 4." *SIGGRAPH Course*, 2013. (UE4's deferred PBR pipeline)
- Salvi, M. "An Excursion in Temporal Supersampling." *GDC*, 2016. (TAA deep dive from Intel)
