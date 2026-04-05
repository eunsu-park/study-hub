# 06. Texture Mapping

[&larr; Previous: Shading Models](05_Shading_Models.md) | [Next: WebGL Fundamentals &rarr;](07_WebGL_Fundamentals.md)

---

## Learning Objectives

1. Understand UV coordinates and how they map 2D images onto 3D surfaces
2. Implement texture sampling with nearest-neighbor and bilinear filtering
3. Explain mipmaps: why they exist, how they are generated, and how the mipmap level is selected
4. Understand anisotropic filtering and when it improves quality
5. Distinguish between normal mapping, bump mapping, and displacement mapping
6. Describe PBR texture maps: albedo, metallic, roughness, ambient occlusion
7. Explain environment mapping using cubemaps and equirectangular projections
8. Implement bilinear texture sampling from scratch in Python

---

## Why This Matters

Without textures, every surface in a 3D scene would be a flat, uniform color -- the shading models from Lesson 05 produce smooth gradients, but no detail. Textures are what make a wooden table look like wood, a brick wall look like brick, and a character's face look alive. Texture mapping is arguably the most impactful visual technique in real-time graphics: a low-polygon model with good textures can look far more convincing than a high-polygon model without them. Modern PBR workflows use multiple texture maps (albedo, normal, roughness, metallic, AO) to control every aspect of surface appearance, and understanding how these textures are sampled and filtered is essential for both artists and programmers.

---

## 1. UV Coordinates

### 1.1 The Concept

**UV coordinates** (also called **texture coordinates**) define a mapping from a 3D surface to a 2D texture image. Each vertex of a mesh stores a $(u, v)$ coordinate pair that specifies where on the texture that vertex "looks."

$$\text{UV}: \text{Surface point} \rightarrow (u, v) \in [0, 1]^2$$

Convention:
- $u$ corresponds to the horizontal axis of the texture (left = 0, right = 1)
- $v$ corresponds to the vertical axis (bottom = 0 in OpenGL; top = 0 in some other APIs)

```
Texture Image:          3D Mesh (UV mapped):
(0,1) ┌──────────┐ (1,1)        ┌──────┐
      │          │              /  UV   /\
      │  BRICK   │             / coords / \
      │  PATTERN │            / map to /   \
      │          │           / texture/     \
(0,0) └──────────┘ (1,0)    └──────/       │
                                    surface
```

### 1.2 UV Parameterization

The process of assigning UV coordinates to a mesh is called **UV unwrapping** -- imagine cutting the 3D surface along seams and laying it flat, like unfolding a cardboard box.

Common parameterization methods:
- **Planar projection**: Project from a direction (good for flat surfaces)
- **Cylindrical projection**: Wrap around a cylinder (good for bottles, trees)
- **Spherical projection**: Wrap around a sphere (good for planets, eyes)
- **Box/cube projection**: Six planar projections (good for architectural elements)
- **Automatic unwrapping**: Algorithms like LSCM, ABF++ minimize distortion

### 1.3 UV Outside [0,1]

What happens when UV coordinates exceed $[0, 1]$? The **wrapping mode** determines behavior:

| Mode | Effect | Use Case |
|------|--------|----------|
| **Repeat** | Tile the texture: $(u, v) \mod 1$ | Brick walls, floor tiles |
| **Clamp to Edge** | Extend edge pixels: $\text{clamp}(u, 0, 1)$ | Sky textures, UI elements |
| **Mirrored Repeat** | Alternate tile direction | Seamless patterns |

```python
import numpy as np

def wrap_uv(u, v, mode='repeat'):
    """
    Apply UV wrapping mode.

    The wrapping mode determines what happens when the texture
    is sampled outside the [0,1] range. 'Repeat' is most common
    because it allows a small texture to cover a large surface.
    """
    if mode == 'repeat':
        u = u % 1.0
        v = v % 1.0
    elif mode == 'clamp':
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
    elif mode == 'mirrored_repeat':
        # Flip direction on odd repetitions
        u = u % 2.0
        v = v % 2.0
        if u > 1.0:
            u = 2.0 - u
        if v > 1.0:
            v = 2.0 - v
    return u, v
```

---

## 2. Texture Sampling

Given a UV coordinate, we need to look up the corresponding color in the texture. This requires converting continuous $(u, v) \in [0, 1]$ to discrete pixel indices in the texture image.

### 2.1 Nearest-Neighbor Sampling

The simplest approach: snap to the nearest texel (texture pixel).

$$\text{texel\_x} = \lfloor u \cdot W \rfloor, \quad \text{texel\_y} = \lfloor v \cdot H \rfloor$$

Where $W \times H$ is the texture resolution.

**Pros**: Fast, preserves sharp edges
**Cons**: Produces blocky/pixelated results when magnified, shimmering when minified

### 2.2 Bilinear Filtering

**Bilinear filtering** interpolates between the four nearest texels, producing smooth results:

Given continuous texel coordinates $(x, y)$ where $x = u \cdot W - 0.5$ and $y = v \cdot H - 0.5$:

1. Find the four surrounding texels: $(x_0, y_0)$, $(x_0+1, y_0)$, $(x_0, y_0+1)$, $(x_0+1, y_0+1)$
2. Compute fractional positions: $f_x = x - x_0$, $f_y = y - y_0$
3. Interpolate:

$$c = (1-f_x)(1-f_y) \cdot c_{00} + f_x(1-f_y) \cdot c_{10} + (1-f_x)f_y \cdot c_{01} + f_x \cdot f_y \cdot c_{11}$$

```python
def sample_nearest(texture, u, v):
    """
    Nearest-neighbor texture sampling.

    Fastest method but produces blocky results when the texture
    is magnified (viewed up close). Good for pixel art or when
    you want crisp, unfiltered texels.
    """
    h, w = texture.shape[:2]
    u, v = wrap_uv(u, v, 'repeat')

    # Map [0,1] to pixel indices
    x = int(u * w) % w
    y = int(v * h) % h

    return texture[y, x].astype(float) / 255.0


def sample_bilinear(texture, u, v):
    """
    Bilinear texture sampling -- smooth interpolation between 4 texels.

    This is the default filtering mode in most graphics APIs because
    it provides a good balance of quality and performance. It eliminates
    the blocky appearance of nearest-neighbor at a modest cost
    (4 texel reads + 3 lerps instead of 1 texel read).
    """
    h, w = texture.shape[:2]
    u, v = wrap_uv(u, v, 'repeat')

    # Continuous texel coordinates (shifted by 0.5 so texel centers
    # are at integer coordinates)
    x = u * w - 0.5
    y = v * h - 0.5

    # Integer parts (floor)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    # Fractional parts (interpolation weights)
    fx = x - x0
    fy = y - y0

    # Four surrounding texels (with wrapping)
    x0w, x1w = x0 % w, (x0 + 1) % w
    y0w, y1w = y0 % h, (y0 + 1) % h

    c00 = texture[y0w, x0w].astype(float) / 255.0
    c10 = texture[y0w, x1w].astype(float) / 255.0
    c01 = texture[y1w, x0w].astype(float) / 255.0
    c11 = texture[y1w, x1w].astype(float) / 255.0

    # Bilinear interpolation: blend horizontally, then vertically
    top = c00 * (1 - fx) + c10 * fx
    bottom = c01 * (1 - fx) + c11 * fx
    result = top * (1 - fy) + bottom * fy

    return result
```

### 2.3 Visual Comparison

```
Original Texture (8x8):   Nearest (magnified):    Bilinear (magnified):
┌──────────┐              ┌──────────┐              ┌──────────┐
│ ██  ██   │              │████  ████│              │▓███  ▓███│
│   ██     │              │    ████  │              │  ▓▓██▓   │
│ ██  ██   │              │████  ████│              │▓███  ▓███│
└──────────┘              └──────────┘              └──────────┘
                           Blocky edges              Smooth blending
```

---

## 3. Mipmaps

### 3.1 The Minification Problem

When a textured surface is far from the camera, many texels map to a single pixel. Without special handling, this causes **texture aliasing** -- sparkling, moire patterns, and visual noise because the single pixel sample cannot represent the average of many texels.

### 3.2 What Are Mipmaps?

A **mipmap** (from the Latin "multum in parvo" -- "much in a small space") is a precomputed pyramid of progressively smaller versions of a texture:

```
Level 0: 256x256 (original)
Level 1: 128x128 (each texel = average of 2x2 from level 0)
Level 2: 64x64
Level 3: 32x32
Level 4: 16x16
Level 5: 8x8
Level 6: 4x4
Level 7: 2x2
Level 8: 1x1
```

Total memory cost: approximately $\frac{1}{3}$ extra (the series $1 + \frac{1}{4} + \frac{1}{16} + \ldots \rightarrow \frac{4}{3}$).

### 3.3 Mipmap Generation

Each level is produced by downsampling the previous level with a 2x2 box filter:

```python
def generate_mipmaps(texture):
    """
    Generate a mipmap pyramid from a base texture.

    Each level is half the resolution of the previous one.
    We use a simple box filter (average of 2x2 blocks),
    though higher-quality filters (Lanczos, Kaiser) can be used
    for better results.

    The mipmap chain continues until we reach a 1x1 texture.
    """
    mipmaps = [texture.astype(float)]
    h, w = texture.shape[:2]

    while h > 1 or w > 1:
        # Halve dimensions (minimum 1)
        new_h = max(1, h // 2)
        new_w = max(1, w // 2)

        prev = mipmaps[-1]
        new_level = np.zeros((new_h, new_w, prev.shape[2]), dtype=float)

        for y in range(new_h):
            for x in range(new_w):
                # Average 2x2 block from previous level
                y0, y1 = y * 2, min(y * 2 + 1, h - 1)
                x0, x1 = x * 2, min(x * 2 + 1, w - 1)

                new_level[y, x] = (prev[y0, x0] + prev[y0, x1] +
                                    prev[y1, x0] + prev[y1, x1]) / 4.0

        mipmaps.append(new_level)
        h, w = new_h, new_w

    return mipmaps
```

### 3.4 Mipmap Level Selection

Which mipmap level to use depends on the **screen-space footprint** of the texture: how many texels correspond to one pixel.

The mipmap level $\lambda$ is computed from the screen-space derivatives of the texture coordinates:

$$\lambda = \log_2\left(\max\left(\left\|\frac{\partial \mathbf{uv}}{\partial x}\right\|, \left\|\frac{\partial \mathbf{uv}}{\partial y}\right\|\right) \cdot \text{texture\_size}\right)$$

Where $\frac{\partial \mathbf{uv}}{\partial x}$ is how much the UV coordinate changes per pixel horizontally. GPUs compute these derivatives automatically by differencing neighboring pixels.

- $\lambda = 0$: Use the full-resolution texture (surface is close)
- $\lambda = 1$: Use the half-resolution texture
- $\lambda = 3$: Use the $\frac{1}{8}$-resolution texture (surface is far away)

### 3.5 Trilinear Filtering

**Trilinear filtering** combines bilinear sampling on two adjacent mipmap levels, then linearly interpolates between them:

1. Compute the fractional mipmap level $\lambda$ (e.g., 2.3)
2. Bilinear sample from mipmap level $\lfloor \lambda \rfloor = 2$
3. Bilinear sample from mipmap level $\lceil \lambda \rceil = 3$
4. Blend: $\text{color} = (1 - 0.3) \cdot \text{level2} + 0.3 \cdot \text{level3}$

This eliminates visible "bands" where mipmap levels change.

```python
def sample_trilinear(mipmaps, u, v, lod):
    """
    Trilinear texture sampling: bilinear on two mip levels + blend.

    LOD (Level of Detail) is the fractional mipmap level.
    Without trilinear filtering, transitions between mip levels
    produce visible seams/bands on the surface.
    """
    max_level = len(mipmaps) - 1
    lod = np.clip(lod, 0, max_level)

    # Integer and fractional parts of LOD
    level_low = int(np.floor(lod))
    level_high = min(level_low + 1, max_level)
    frac = lod - level_low

    # Bilinear sample from each level
    color_low = sample_bilinear_from_array(mipmaps[level_low], u, v)
    color_high = sample_bilinear_from_array(mipmaps[level_high], u, v)

    # Blend between levels
    return color_low * (1 - frac) + color_high * frac


def sample_bilinear_from_array(tex, u, v):
    """Bilinear sample from a float array texture."""
    h, w = tex.shape[:2]
    x = (u * w - 0.5) % w
    y = (v * h - 0.5) % h

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    fx = x - x0
    fy = y - y0

    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    x0 = x0 % w
    y0 = y0 % h

    c00 = tex[y0, x0]
    c10 = tex[y0, x1]
    c01 = tex[y1, x0]
    c11 = tex[y1, x1]

    top = c00 * (1 - fx) + c10 * fx
    bottom = c01 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bottom * fy
```

---

## 4. Anisotropic Filtering

### 4.1 The Problem with Mipmaps

Mipmaps assume the texture footprint is approximately square (isotropic). But when a surface is viewed at a grazing angle, the footprint becomes highly elongated (anisotropic) -- think of a road receding into the distance.

Standard trilinear filtering selects the mipmap level based on the *larger* footprint dimension, which blurs the texture excessively along the shorter dimension.

### 4.2 How Anisotropic Filtering Works

Anisotropic filtering (AF) takes multiple samples along the elongated direction and averages them:

1. Compute the anisotropy ratio: $\text{ratio} = \frac{\max(\text{footprint})}{\min(\text{footprint})}$
2. Take $N$ (up to the max anisotropy level, e.g., 16x) trilinear samples along the long axis
3. Average the results

```
Isotropic footprint:     Anisotropic footprint:
   ┌───┐                    ┌───────────────┐
   │   │  ~square            │ ● ● ● ● ● ● │  multiple samples
   └───┘  1 sample           └───────────────┘  along long axis
```

**Quality hierarchy**: Nearest < Bilinear < Trilinear < Anisotropic

| Filter | Texel Reads | Quality |
|--------|-------------|---------|
| Nearest | 1 | Blocky |
| Bilinear | 4 | Smooth (2D) |
| Trilinear | 8 | Smooth (2D + mip transition) |
| 16x Anisotropic | Up to 128 | Best (preserves detail at angles) |

---

## 5. Normal Mapping

### 5.1 Motivation

Adding geometric detail to a mesh is expensive: more triangles mean more vertex processing and more memory. **Normal mapping** fakes surface detail by perturbing the surface normal at each pixel using a texture, without adding any geometry.

### 5.2 How It Works

A **normal map** is a texture where each texel stores a surface normal direction encoded as an RGB color:

- R channel $\rightarrow$ X component of normal
- G channel $\rightarrow$ Y component of normal
- B channel $\rightarrow$ Z component of normal

The normal is stored in **tangent space** (relative to the surface), where:
- Z points outward (perpendicular to the surface)
- X and Y lie in the surface plane

A flat surface has all normals pointing straight up: $(0, 0, 1)$, which encodes as $(128, 128, 255)$ -- the characteristic purple/blue color of normal maps.

### 5.3 Tangent Space

To use a tangent-space normal map, we need a coordinate frame at each vertex:
- **Normal** $\mathbf{N}$: perpendicular to the surface (from vertex data)
- **Tangent** $\mathbf{T}$: along the U direction of the texture
- **Bitangent** $\mathbf{B}$: $\mathbf{B} = \mathbf{N} \times \mathbf{T}$

The **TBN matrix** transforms tangent-space normals to world space:

$$\mathbf{n}_{\text{world}} = \begin{bmatrix} T_x & B_x & N_x \\ T_y & B_y & N_y \\ T_z & B_z & N_z \end{bmatrix} \mathbf{n}_{\text{tangent}}$$

```python
def apply_normal_map(normal_map_color, tangent, bitangent, normal):
    """
    Convert a normal map sample from tangent space to world space.

    Why tangent space? Because the same normal map can be applied to
    any surface orientation. The TBN matrix adapts the stored normals
    to the actual surface direction at each point.

    Parameters:
        normal_map_color: RGB sample from normal map, in [0, 1]
        tangent: surface tangent vector (along U)
        bitangent: surface bitangent vector (along V)
        normal: surface normal vector
    """
    # Decode: [0,1] -> [-1, 1]
    n_tangent = normal_map_color * 2.0 - 1.0

    # Build TBN matrix (columns are T, B, N)
    T = normalize(tangent)
    N = normalize(normal)
    B = normalize(bitangent)
    TBN = np.column_stack([T, B, N])

    # Transform to world space
    n_world = TBN @ n_tangent
    return normalize(n_world)
```

### 5.4 Bump Mapping vs Normal Mapping

| Technique | Input | Stores | Notes |
|-----------|-------|--------|-------|
| **Bump mapping** | Grayscale heightmap | Height values | Normal derived from height differences |
| **Normal mapping** | RGB texture | Normal vectors directly | More precise, standard approach |

**Bump mapping** (the original technique by James Blinn, 1978) uses a grayscale heightmap. The perturbed normal is computed from the height gradient:

$$\mathbf{n}' = \mathbf{n} - \frac{\partial h}{\partial u} \mathbf{T} - \frac{\partial h}{\partial v} \mathbf{B}$$

Normal mapping is more commonly used today because it directly stores the desired normal, avoiding derivative computation at runtime.

---

## 6. Displacement Mapping

Unlike normal mapping (which only changes lighting), **displacement mapping** actually moves vertices based on a heightmap texture:

$$\mathbf{p}' = \mathbf{p} + h(u, v) \cdot \mathbf{n}$$

Where $h(u, v)$ is the height sampled from the displacement map and $\mathbf{n}$ is the surface normal.

**Comparison**:

| Technique | Modifies Geometry? | Silhouette Correct? | Cost |
|-----------|-------------------|--------------------|----|
| Normal mapping | No | No (flat silhouette) | Cheap (per-pixel) |
| Parallax mapping | No (fakes it) | Partially | Moderate |
| Displacement mapping | Yes (moves vertices) | Yes | Expensive (needs tessellation) |

Displacement mapping is typically used with **tessellation** (Lesson 01, Section 7.1): the GPU subdivides the mesh into fine triangles, then displaces each vertex according to the heightmap.

---

## 7. PBR Texture Maps

Modern PBR materials use multiple texture maps, each controlling a different aspect of surface appearance:

### 7.1 Albedo (Base Color) Map

The surface color without any lighting or shadow. For dielectrics, this is the diffuse color. For metals, this defines the specular color ($F_0$).

- **Do**: Store only color information
- **Do not**: Bake lighting, shadows, or ambient occlusion into the albedo

### 7.2 Metallic Map

Binary or grayscale map indicating which parts of the surface are metallic (1.0) vs dielectric (0.0). In practice, values are usually 0 or 1, with rare in-between values for transitions (e.g., rust on metal).

### 7.3 Roughness Map

Controls the microsurface roughness at each texel:
- 0.0 = mirror smooth (polished chrome)
- 0.5 = moderate roughness (worn plastic)
- 1.0 = completely matte (chalk)

### 7.4 Ambient Occlusion (AO) Map

Precomputed information about how occluded each point is from ambient light. Crevices and tight corners are darker. This is multiplied with the ambient/indirect lighting term:

$$L_{\text{ambient}} = \text{AO}(u,v) \cdot k_a \cdot L_a$$

### 7.5 Complete PBR Texture Lookup

```python
def sample_pbr_material(textures, u, v):
    """
    Sample all PBR texture maps at a given UV coordinate.

    A complete PBR material typically uses 4-5 texture maps.
    Each map controls a different parameter of the Cook-Torrance BRDF.

    Returns a dictionary of material properties at this UV coordinate.
    """
    material = {}

    # Albedo: the base color (in sRGB, must convert to linear for shading)
    albedo_srgb = sample_bilinear(textures['albedo'], u, v)
    material['albedo'] = np.power(albedo_srgb, 2.2)  # sRGB -> linear

    # Normal: perturbed surface normal (tangent space)
    if 'normal' in textures:
        normal_sample = sample_bilinear(textures['normal'], u, v)
        material['normal'] = normal_sample * 2.0 - 1.0  # [0,1] -> [-1,1]

    # Metallic: 0 = dielectric, 1 = metal
    if 'metallic' in textures:
        material['metallic'] = sample_bilinear(textures['metallic'], u, v)[0]
    else:
        material['metallic'] = 0.0

    # Roughness: 0 = smooth mirror, 1 = matte
    if 'roughness' in textures:
        material['roughness'] = sample_bilinear(textures['roughness'], u, v)[0]
    else:
        material['roughness'] = 0.5

    # Ambient Occlusion: 0 = fully occluded, 1 = fully exposed
    if 'ao' in textures:
        material['ao'] = sample_bilinear(textures['ao'], u, v)[0]
    else:
        material['ao'] = 1.0

    return material
```

### 7.6 Typical PBR Texture Pipeline

```
Artist creates:
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  Albedo  │  │  Normal  │  │ Metallic │  │Roughness │  │    AO    │
  │  (sRGB)  │  │ (Linear) │  │ (Linear) │  │ (Linear) │  │ (Linear) │
  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │              │              │              │
       ▼              ▼              ▼              ▼              ▼
  Fragment Shader reads all maps at the fragment's UV coordinate
       │              │              │              │              │
       ▼              ▼              ▼              ▼              ▼
  Cook-Torrance BRDF:  diffuse+specular  Fresnel    highlight    ambient
                       color             F0         sharpness    darkening
```

---

## 8. Environment Mapping

### 8.1 Concept

**Environment mapping** captures the surrounding environment as a texture, which can then be used for:
- **Reflections**: Mirror-like surfaces reflect the environment
- **Image-Based Lighting (IBL)**: Use the environment as a light source for PBR

### 8.2 Cubemaps

A **cubemap** stores the environment as six square images, one for each face of a cube:

```
        ┌───────┐
        │  +Y   │
        │ (top) │
┌───────┼───────┼───────┬───────┐
│  -X   │  +Z   │  +X   │  -Z   │
│(left) │(front)│(right)│(back) │
└───────┼───────┼───────┴───────┘
        │  -Y   │
        │(bottom)│
        └───────┘
```

To sample a cubemap, we use a **3D direction vector** (not UV coordinates). The hardware determines which face to sample and the 2D coordinates within that face:

```python
def sample_cubemap(cubemap_faces, direction):
    """
    Sample a cubemap given a 3D direction vector.

    The direction is used to determine which face of the cube
    to sample, and where on that face to sample.

    cubemap_faces: dict with keys '+x', '-x', '+y', '-y', '+z', '-z'
                   each a 2D texture array
    """
    dx, dy, dz = direction
    abs_x, abs_y, abs_z = abs(dx), abs(dy), abs(dz)

    # Determine dominant axis -> which face to sample
    if abs_x >= abs_y and abs_x >= abs_z:
        if dx > 0:
            face = '+x'
            u = (-dz / abs_x + 1) / 2
            v = (-dy / abs_x + 1) / 2
        else:
            face = '-x'
            u = (dz / abs_x + 1) / 2
            v = (-dy / abs_x + 1) / 2
    elif abs_y >= abs_x and abs_y >= abs_z:
        if dy > 0:
            face = '+y'
            u = (dx / abs_y + 1) / 2
            v = (dz / abs_y + 1) / 2
        else:
            face = '-y'
            u = (dx / abs_y + 1) / 2
            v = (-dz / abs_y + 1) / 2
    else:
        if dz > 0:
            face = '+z'
            u = (dx / abs_z + 1) / 2
            v = (-dy / abs_z + 1) / 2
        else:
            face = '-z'
            u = (-dx / abs_z + 1) / 2
            v = (-dy / abs_z + 1) / 2

    return sample_bilinear(cubemap_faces[face], u, v)
```

### 8.3 Equirectangular Mapping

An alternative to cubemaps is the **equirectangular** (lat-long) format, which stores the environment in a single 2:1 aspect ratio image:

$$u = \frac{1}{2} + \frac{\text{atan2}(d_z, d_x)}{2\pi}, \quad v = \frac{1}{2} - \frac{\arcsin(d_y)}{\pi}$$

```python
def direction_to_equirectangular(direction):
    """
    Convert a 3D direction to equirectangular UV coordinates.

    Equirectangular maps are easier to create and edit (single image)
    but have more distortion at the poles compared to cubemaps.
    They are commonly used for HDR environment probes.
    """
    dx, dy, dz = normalize(direction)

    u = 0.5 + np.arctan2(dz, dx) / (2 * np.pi)
    v = 0.5 - np.arcsin(np.clip(dy, -1, 1)) / np.pi

    return u, v
```

### 8.4 Reflection Mapping

For mirror-like surfaces, the environment is sampled using the **reflection vector**:

$$\mathbf{R} = 2(\mathbf{N} \cdot \mathbf{V})\mathbf{N} - \mathbf{V}$$

$$\text{reflection\_color} = \text{cubemap}(\mathbf{R})$$

For rough surfaces, prefiltered environment maps (blurred versions of the cubemap) are used, with rougher materials sampling from blurrier levels.

---

## 9. Complete Texture Mapping Example

```python
"""
Complete texture mapping demonstration:
Renders a textured quad with bilinear filtering and mipmap selection.
"""

import numpy as np

def create_checkerboard(size=256, squares=8):
    """
    Create a checkerboard texture for testing.

    Checkerboards are the go-to test pattern for texture sampling
    because aliasing artifacts (moire patterns, shimmering) are
    immediately visible.
    """
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    block = size // squares

    for y in range(size):
        for x in range(size):
            if ((x // block) + (y // block)) % 2 == 0:
                texture[y, x] = [200, 200, 200]  # Light
            else:
                texture[y, x] = [50, 50, 50]    # Dark

    return texture


def render_textured_quad(texture, width=400, height=300,
                          filter_mode='bilinear'):
    """
    Render a textured quad in perspective using chosen filter mode.

    The quad extends from the bottom of the screen into the distance,
    like a floor stretching toward the horizon. This view maximizes
    the importance of mipmap filtering -- the near part of the floor
    needs high-resolution texture, while the far part needs heavily
    downsampled texture.
    """
    image = np.zeros((height, width, 3), dtype=float)

    # Generate mipmaps
    mipmaps = generate_mipmaps(texture)

    for y in range(height):
        for x in range(width):
            # Map pixel to UV on a perspective-projected floor
            # Simulate a floor plane at y=-1 viewed from above
            screen_x = (x / width - 0.5) * 2
            screen_y = (y / height)

            if screen_y < 0.01:
                continue  # Skip horizon line

            # Perspective UV mapping: farther rows have more compressed UVs
            # This simulates viewing a tiled floor in perspective
            u = screen_x / screen_y * 2.0
            v = 1.0 / screen_y

            if filter_mode == 'nearest':
                color = sample_nearest(texture, u % 1, v % 1)
            elif filter_mode == 'bilinear':
                color = sample_bilinear(texture, u % 1, v % 1)
            elif filter_mode == 'trilinear':
                # Estimate LOD from how much UV changes per pixel
                # (simplified -- real GPUs compute this per-quad)
                lod = np.log2(max(abs(1.0 / screen_y) * texture.shape[0]
                                   / height, 1.0))
                lod = np.clip(lod, 0, len(mipmaps) - 1)
                color = sample_trilinear(mipmaps, u % 1, v % 1, lod)
            else:
                color = sample_bilinear(texture, u % 1, v % 1)

            image[y, x] = color

    return image


if __name__ == "__main__":
    print("Creating checkerboard texture...")
    checker = create_checkerboard(256, 8)

    print("Generating mipmaps...")
    mipmaps = generate_mipmaps(checker)
    print(f"Mipmap levels: {len(mipmaps)}")
    for i, m in enumerate(mipmaps):
        print(f"  Level {i}: {m.shape[1]}x{m.shape[0]}")

    print("\nRendering with nearest filtering...")
    img_nearest = render_textured_quad(checker, filter_mode='nearest')

    print("Rendering with bilinear filtering...")
    img_bilinear = render_textured_quad(checker, filter_mode='bilinear')

    print("Rendering with trilinear filtering...")
    img_trilinear = render_textured_quad(checker, filter_mode='trilinear')

    print("\nDone! Compare the three results to see aliasing differences.")
    print("Nearest: shimmering moire patterns in the distance")
    print("Bilinear: reduced shimmer but still aliased in distance")
    print("Trilinear: smooth transition, no moire patterns")
```

---

## Summary

| Concept | Description | Key Formula/Method |
|---------|-------------|--------------------|
| **UV coordinates** | 2D parameterization of 3D surface | Artist-defined or auto-unwrapped |
| **Nearest sampling** | Snap to closest texel | Fast but blocky |
| **Bilinear sampling** | Blend 4 nearest texels | $c = \text{lerp}(\text{lerp}(c_{00}, c_{10}), \text{lerp}(c_{01}, c_{11}))$ |
| **Mipmaps** | Precomputed resolution pyramid | Extra $\frac{1}{3}$ memory |
| **Trilinear** | Bilinear on 2 mip levels + blend | Eliminates mip transitions |
| **Anisotropic** | Multiple samples along elongated footprint | Up to 16x quality |
| **Normal mapping** | Perturb normals with RGB texture | TBN matrix to world space |
| **Displacement** | Actually move vertices with heightmap | Requires tessellation |
| **PBR textures** | Albedo + metallic + roughness + AO + normal | Full material definition |
| **Environment maps** | Cubemap or equirectangular for reflections | Sample with reflection vector |

**Key takeaways**:
- UV coordinates are the bridge between 3D geometry and 2D textures
- Bilinear filtering eliminates the blocky appearance of nearest-neighbor sampling
- Mipmaps solve texture aliasing for distant surfaces at minimal memory cost
- Normal mapping adds visual detail without geometric cost
- PBR uses multiple coordinated texture maps for physically accurate materials
- Environment maps enable realistic reflections and image-based lighting

---

## Exercises

1. **Bilinear vs Nearest**: Create a 4x4 checkerboard texture and magnify it 16x using both nearest-neighbor and bilinear sampling. Compare the results side by side. Which preserves the sharp checkerboard pattern? Which looks smoother?

2. **Mipmap Visualization**: Generate a mipmap chain for a 256x256 texture. Display all mipmap levels side-by-side. Color each level differently (tint level 0 red, level 1 green, level 2 blue, etc.) and render a textured floor to visualize which mipmap level is used at each distance.

3. **Normal Map Creation**: Given a heightmap (grayscale image), compute the normal map by taking the finite difference gradient in $u$ and $v$ directions. Convert the resulting normal vectors to the $[0, 1]$ RGB encoding. Test your result by rendering a flat surface with the computed normal map.

4. **UV Wrapping Modes**: Render a quad with UV coordinates from $(-1, -1)$ to $(2, 2)$ using all three wrapping modes (repeat, clamp, mirrored repeat). Explain when each mode is appropriate.

5. **Perspective Correction**: Render a textured quad (two triangles) viewed at an angle, with and without perspective-correct UV interpolation. Use a checkerboard texture and describe the visual difference.

6. **Environment Reflection**: Implement a simple environment-mapped sphere using an equirectangular HDR image. Show how the reflection changes as you vary the roughness parameter (hint: use blurred versions of the environment map for rougher surfaces).

---

## Further Reading

1. Marschner, S. & Shirley, P. *Fundamentals of Computer Graphics* (5th ed.), Ch. 11 -- "Texture Mapping"
2. Akenine-Moller, T. et al. *Real-Time Rendering* (4th ed.), Ch. 6 -- "Texturing"
3. [Learn OpenGL - Textures](https://learnopengl.com/Getting-started/Textures) -- Practical OpenGL texture tutorial
4. [Blinn, J. "Simulation of Wrinkled Surfaces" (1978)](https://www.microsoft.com/en-us/research/publication/simulation-of-wrinkled-surfaces/) -- The original bump mapping paper
5. [Filament: Image-Based Lighting](https://google.github.io/filament/Filament.html#toc4.4) -- PBR environment mapping in practice
