# 08. Shader Programming (GLSL)

[&larr; Previous: WebGL Fundamentals](07_WebGL_Fundamentals.md) | [Next: Ray Tracing Basics &rarr;](09_Ray_Tracing_Basics.md)

---

## Learning Objectives

1. Master GLSL data types: scalars, vectors (vec2/3/4), matrices (mat2/3/4), and swizzling
2. Write vertex shaders that perform transformations and pass data to fragment shaders
3. Write fragment shaders that compute per-pixel color, including texture sampling
4. Understand the roles of uniforms, attributes (in), and varyings (out/in)
5. Use essential built-in functions: mix, clamp, smoothstep, normalize, reflect, dot
6. Use preprocessor directives for conditional compilation and debugging
7. Understand multi-pass rendering concepts and how shaders participate in each pass
8. Implement practical shader effects: Phong lighting, texture mapping, and post-processing blur

---

## Why This Matters

Shaders are the programs that run on the GPU, and mastering them is what separates a graphics programmer from someone who merely uses a 3D engine. Every visual effect you see in modern games and applications -- from realistic lighting to water reflections, from cel-shading to screen-space ambient occlusion -- is implemented as a shader. GLSL (OpenGL Shading Language) is the language used for shaders in OpenGL and WebGL. Understanding GLSL deeply enables you to implement any visual effect you can imagine, optimize rendering performance, and debug visual artifacts that would otherwise be mysterious black boxes.

---

## 1. GLSL Basics

### 1.1 Program Structure

Every GLSL shader has a `main()` function as its entry point. There is no `return` value from `main`; instead, the shader writes to built-in or declared output variables.

```glsl
#version 300 es
// Version directive MUST be the first line
// 300 es = GLSL ES 3.00 (WebGL 2)
// 100    = GLSL ES 1.00 (WebGL 1)

precision mediump float;
// Precision qualifier: required in fragment shaders (GLSL ES)
// highp   = 32-bit float (more precise, slower on mobile)
// mediump = 16-bit float (good default)
// lowp    = 10-bit float (only for things like color)

void main() {
    // Shader code here
}
```

### 1.2 Scalar Types

| Type | Description | Example |
|------|-------------|---------|
| `bool` | Boolean | `bool flag = true;` |
| `int` | 32-bit signed integer | `int count = 42;` |
| `uint` | 32-bit unsigned integer | `uint mask = 0xFFu;` |
| `float` | 32-bit floating-point | `float pi = 3.14159;` |

### 1.3 Vector Types

GLSL's vector types are the workhorses of shader programming:

| Type | Components | Usage |
|------|-----------|-------|
| `vec2` | 2 floats | UV coordinates, 2D positions |
| `vec3` | 3 floats | 3D positions, RGB colors, normals |
| `vec4` | 4 floats | Homogeneous positions, RGBA colors |
| `ivec2/3/4` | 2/3/4 ints | Pixel indices, integer coordinates |
| `bvec2/3/4` | 2/3/4 bools | Component-wise boolean results |

**Construction**:

```glsl
vec3 color = vec3(1.0, 0.5, 0.0);      // Explicit components
vec3 white = vec3(1.0);                  // All components = 1.0
vec4 pos = vec4(color, 1.0);            // Extend vec3 with a 4th component
vec2 uv = vec2(0.5, 0.5);              // 2D coordinate
vec4 full = vec4(uv, 0.0, 1.0);        // Combine vec2 + scalars
```

### 1.4 Swizzling

**Swizzling** is GLSL's powerful syntax for rearranging, replicating, or extracting vector components:

```glsl
vec4 v = vec4(1.0, 2.0, 3.0, 4.0);

// Access individual components (two naming conventions):
// Position: x, y, z, w
// Color:    r, g, b, a
// Texture:  s, t, p, q  (less common)

float x = v.x;           // 1.0
vec2 xy = v.xy;           // vec2(1.0, 2.0)
vec3 rgb = v.rgb;         // vec3(1.0, 2.0, 3.0)

// Reorder components
vec3 bgr = v.bgr;         // vec3(3.0, 2.0, 1.0)
vec4 yyxx = v.yyxx;       // vec4(2.0, 2.0, 1.0, 1.0)

// Duplicate components
vec3 rrr = v.rrr;         // vec3(1.0, 1.0, 1.0)

// Write to specific components
v.xy = vec2(5.0, 6.0);    // v = vec4(5.0, 6.0, 3.0, 4.0)

// Common patterns:
vec3 pos3d = gl_Position.xyz;           // Extract position from vec4
vec3 normal = normalize(N.xyz);         // Normalize a direction
float luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114));
```

> **Why swizzling matters**: It replaces what would be verbose index-based access in other languages with concise, readable expressions. GPUs execute swizzle operations at zero cost -- they are just register read masks.

### 1.5 Matrix Types

| Type | Size | Usage |
|------|------|-------|
| `mat2` | 2x2 | 2D rotations |
| `mat3` | 3x3 | Normal transforms, 2D affine |
| `mat4` | 4x4 | MVP, model, view, projection |

```glsl
mat4 identity = mat4(1.0);  // Identity matrix (1s on diagonal)

// Column-major construction (each vec4 is a COLUMN):
mat4 m = mat4(
    vec4(1, 0, 0, 0),  // Column 0
    vec4(0, 1, 0, 0),  // Column 1
    vec4(0, 0, 1, 0),  // Column 2
    vec4(0, 0, 0, 1)   // Column 3
);

// Matrix-vector multiplication: transforms a position
vec4 worldPos = uModel * vec4(aPosition, 1.0);

// Matrix-matrix multiplication: compose transformations
mat4 mvp = uProjection * uView * uModel;

// Access columns and elements
vec4 col0 = m[0];         // First column
float m01 = m[0][1];      // Row 1, Column 0
```

> **Important**: GLSL matrices are **column-major**. `m[i]` accesses column `i`, not row `i`. This matches OpenGL's convention and the `Float32Array` layout expected by `gl.uniformMatrix4fv`.

---

## 2. Vertex Shaders

The vertex shader processes each vertex independently. Its primary job is to set `gl_Position` (the clip-space position) and pass data to the fragment shader.

### 2.1 Input/Output

```glsl
#version 300 es

// INPUTS: per-vertex attributes from vertex buffers
in vec3 aPosition;    // Vertex position (object space)
in vec3 aNormal;      // Vertex normal
in vec2 aTexCoord;    // Texture coordinates

// UNIFORMS: constant for all vertices in a draw call
uniform mat4 uModel;       // Object -> World
uniform mat4 uView;        // World -> Camera
uniform mat4 uProjection;  // Camera -> Clip
uniform mat3 uNormalMatrix; // For transforming normals

// OUTPUTS: sent to fragment shader (interpolated across the triangle)
out vec3 vWorldPos;    // Fragment position in world space
out vec3 vNormal;      // Fragment normal in world space
out vec2 vTexCoord;    // Fragment texture coordinates

void main() {
    // Transform position through the MVP pipeline
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    vWorldPos = worldPos.xyz;

    // Transform normal to world space using the normal matrix
    // (inverse-transpose of model matrix's upper 3x3)
    vNormal = normalize(uNormalMatrix * aNormal);

    // Pass through texture coordinates unchanged
    vTexCoord = aTexCoord;

    // Final clip-space position (REQUIRED output)
    gl_Position = uProjection * uView * worldPos;
}
```

### 2.2 Built-in Vertex Shader Variables

| Variable | Type | Description |
|----------|------|-------------|
| `gl_Position` | `vec4` | **Output** (required): clip-space position |
| `gl_PointSize` | `float` | Output: size of point primitives |
| `gl_VertexID` | `int` | Input: index of the current vertex |
| `gl_InstanceID` | `int` | Input: index of the current instance (instanced rendering) |

---

## 3. Fragment Shaders

The fragment shader runs once per fragment (pixel candidate) and computes the output color.

### 3.1 Input/Output

```glsl
#version 300 es
precision highp float;

// INPUTS: interpolated from vertex shader outputs
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;

// UNIFORMS
uniform vec3 uLightPos;
uniform vec3 uLightColor;
uniform vec3 uCameraPos;
uniform sampler2D uAlbedoMap;    // 2D texture sampler

// OUTPUT: the fragment's color
out vec4 fragColor;

void main() {
    // Sample albedo texture at the fragment's UV coordinate
    vec3 albedo = texture(uAlbedoMap, vTexCoord).rgb;

    // Normalize the interpolated normal
    // (interpolation of unit vectors does NOT produce a unit vector)
    vec3 N = normalize(vNormal);

    // Compute Blinn-Phong lighting
    vec3 L = normalize(uLightPos - vWorldPos);
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 H = normalize(L + V);

    // Ambient
    vec3 ambient = 0.1 * albedo;

    // Diffuse (Lambert)
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * albedo * uLightColor;

    // Specular (Blinn-Phong)
    float spec = pow(max(dot(N, H), 0.0), 64.0);
    vec3 specular = spec * uLightColor * 0.5;

    // Combine
    vec3 color = ambient + diffuse + specular;

    fragColor = vec4(color, 1.0);
}
```

### 3.2 Built-in Fragment Shader Variables

| Variable | Type | Description |
|----------|------|-------------|
| `gl_FragCoord` | `vec4` | Input: window-space position (x, y, z, 1/w) |
| `gl_FrontFacing` | `bool` | Input: true if fragment is front-facing |
| `gl_FragDepth` | `float` | Output (optional): override depth value |

```glsl
// Example: use gl_FragCoord for screen-space effects
void main() {
    // Create a checkerboard pattern in screen space
    float checker = mod(floor(gl_FragCoord.x / 10.0) +
                        floor(gl_FragCoord.y / 10.0), 2.0);
    fragColor = vec4(vec3(checker), 1.0);
}
```

---

## 4. Uniforms vs Attributes vs Varyings

### 4.1 Data Flow Summary

```
JavaScript          Vertex Shader         Rasterizer          Fragment Shader
───────────────    ─────────────────    ──────────────      ─────────────────
Attributes ──────▶ in vec3 aPosition
(per-vertex)        │
                    │ Process
                    │
                    out vec3 vNormal ──▶ Interpolate ──────▶ in vec3 vNormal
                    (per-vertex output)   (per-fragment)      (per-fragment)

Uniforms ─────────────────────────────────────────────────▶ uniform mat4 uMVP
(constant)         uniform mat4 uMVP                         (same value)
                   (same value for all)
```

### 4.2 Comparison Table

| Qualifier | GLSL 300 es | Set by | Frequency | Example |
|-----------|-------------|--------|-----------|---------|
| **Attribute** | `in` (vertex) | `vertexAttribPointer` | Per vertex | Position, normal, UV |
| **Varying** | `out` (vertex) / `in` (fragment) | Vertex shader | Interpolated per fragment | World position, normal |
| **Uniform** | `uniform` | `gl.uniform*()` | Per draw call | MVP matrix, light position |

### 4.3 Flat Interpolation

By default, outputs from the vertex shader are smoothly interpolated across the triangle. The `flat` qualifier disables interpolation -- the value from the **provoking vertex** (usually the first or last vertex of the triangle) is used for all fragments:

```glsl
// Vertex shader
flat out int vFaceID;    // No interpolation -- same value for entire triangle

// Fragment shader
flat in int vFaceID;     // Receives the uninterpolated value
```

---

## 5. Essential Built-in Functions

GLSL provides a rich set of built-in functions that are hardware-accelerated on the GPU.

### 5.1 Mathematical Functions

```glsl
// ── Clamping and interpolation ──

// clamp(x, minVal, maxVal): restrict x to [min, max]
float brightness = clamp(intensity, 0.0, 1.0);
// Why: prevents overflow/underflow in color calculations

// mix(a, b, t): linear interpolation, a*(1-t) + b*t
vec3 color = mix(colorA, colorB, 0.5);  // 50% blend
// Why: blending between materials, fog, day/night transitions

// smoothstep(edge0, edge1, x): smooth Hermite interpolation
//   Returns 0 if x <= edge0, 1 if x >= edge1
//   Smooth S-curve in between (no sharp transitions)
float alpha = smoothstep(0.0, 0.1, distanceFromEdge);
// Why: soft edges, fade effects, anti-aliased procedural shapes

// step(edge, x): hard threshold
//   Returns 0 if x < edge, 1 if x >= edge
float mask = step(0.5, value);  // Binary threshold at 0.5
```

### 5.2 Geometric Functions

```glsl
// ── Vectors and geometry ──

// normalize(v): return unit-length vector in same direction
vec3 N = normalize(vNormal);
// Why: interpolated normals are not unit length; lighting requires them to be

// dot(a, b): dot product
float lambert = dot(N, L);
// Why: measures alignment between vectors (cosine of angle)

// cross(a, b): cross product (vec3 only)
vec3 bitangent = cross(normal, tangent);
// Why: creates perpendicular vectors for TBN matrix

// reflect(incident, normal): reflect a vector about a normal
vec3 R = reflect(-L, N);
// Why: computing specular reflections, environment mapping

// refract(incident, normal, eta): Snell's law refraction
vec3 refracted = refract(viewDir, normal, 1.0 / 1.5);  // Glass IOR
// Why: simulating transparent materials (glass, water)

// length(v): vector magnitude
float dist = length(lightPos - fragPos);
// Why: computing attenuation, distance-based effects

// distance(a, b): shorthand for length(a - b)
float d = distance(lightPos, fragPos);

// faceforward(N, I, Nref): flip N if dot(Nref, I) >= 0
vec3 correctedNormal = faceforward(N, viewDir, N);
// Why: ensure normals point toward the camera on double-sided surfaces
```

### 5.3 Texture Functions

```glsl
// ── Texture sampling ──

// texture(sampler, uv): sample a 2D texture
vec4 texColor = texture(uAlbedoMap, vTexCoord);

// textureLod(sampler, uv, lod): sample at a specific mipmap level
vec4 blurred = textureLod(uEnvironment, uv, 4.0);  // Blur level 4
// Why: prefiltered environment maps for PBR, manual LOD control

// textureSize(sampler, lod): get texture dimensions
ivec2 size = textureSize(uTexture, 0);  // Level 0 dimensions
// Why: computing texel offsets for manual filtering

// texelFetch(sampler, ivec2(x, y), lod): fetch exact texel (no filtering)
vec4 exact = texelFetch(uTexture, ivec2(x, y), 0);
// Why: accessing specific pixels in lookup tables, data textures
```

### 5.4 Common Patterns

```glsl
// ── Frequently used shader patterns ──

// Remap value from one range to another
float remap(float value, float inLow, float inHigh,
            float outLow, float outHigh) {
    return outLow + (value - inLow) * (outHigh - outLow) / (inHigh - inLow);
}

// Smooth pulse (like a bell curve at center between edge0 and edge1)
float pulse(float edge0, float edge1, float x) {
    return smoothstep(edge0, (edge0 + edge1) * 0.5, x) -
           smoothstep((edge0 + edge1) * 0.5, edge1, x);
}

// Luminance from RGB (human perception-weighted)
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// sRGB to linear conversion
vec3 srgbToLinear(vec3 srgb) {
    return pow(srgb, vec3(2.2));
}

// Linear to sRGB conversion
vec3 linearToSrgb(vec3 linear) {
    return pow(linear, vec3(1.0 / 2.2));
}
```

---

## 6. Preprocessor Directives

GLSL includes a C-like preprocessor for conditional compilation:

```glsl
// ── Version and extension control ──
#version 300 es
#extension GL_OES_standard_derivatives : enable

// ── Conditional compilation ──
// Define a macro from JavaScript by prepending to shader source:
// shaderSource = "#define HAS_NORMAL_MAP\n" + originalSource;

#ifdef HAS_NORMAL_MAP
    vec3 normal = applyNormalMap(vTexCoord, vNormal, vTangent);
#else
    vec3 normal = normalize(vNormal);
#endif

// ── Feature flags for material variants ──
#define MAX_LIGHTS 4

#if MAX_LIGHTS > 8
    // High-light-count path
#elif MAX_LIGHTS > 0
    // Normal path
#else
    // No dynamic lights (only ambient)
#endif

// ── Debugging ──
// Uncomment to visualize normals as colors
// #define DEBUG_NORMALS

#ifdef DEBUG_NORMALS
    fragColor = vec4(normal * 0.5 + 0.5, 1.0);
    return;
#endif
```

### 6.1 Uber-Shaders

A common technique is to write a single "uber-shader" with many `#ifdef` blocks and generate variants at compile time:

```javascript
// JavaScript: create shader variants by prepending defines
function getShaderVariant(baseSource, options) {
    let defines = '';
    if (options.hasNormalMap)  defines += '#define HAS_NORMAL_MAP\n';
    if (options.hasShadows)   defines += '#define HAS_SHADOWS\n';
    if (options.hasEmission)  defines += '#define HAS_EMISSION\n';
    defines += `#define NUM_LIGHTS ${options.numLights}\n`;

    // Insert defines after the #version line
    const versionEnd = baseSource.indexOf('\n') + 1;
    return baseSource.slice(0, versionEnd) + defines + baseSource.slice(versionEnd);
}
```

---

## 7. Multi-Pass Rendering

### 7.1 What Is Multi-Pass Rendering?

Instead of rendering the entire scene in a single pass, **multi-pass rendering** draws the scene multiple times (or draws to intermediate textures), each pass computing a different aspect of the final image.

```
Pass 1: Render scene to G-buffer
        ┌──────────────────┐
        │ Position texture  │
        │ Normal texture    │  ← "Geometry pass"
        │ Albedo texture    │
        └──────────────────┘
                │
Pass 2: Compute lighting from G-buffer
        ┌──────────────────┐
        │ Lit scene texture │  ← "Lighting pass"
        └──────────────────┘
                │
Pass 3: Apply post-processing
        ┌──────────────────┐
        │ Final image       │  ← "Post-process pass"
        └──────────────────┘
```

### 7.2 Render to Texture (Framebuffer Objects)

```javascript
/**
 * Create a framebuffer that renders to a texture instead of the screen.
 *
 * This is the foundation of multi-pass rendering: render the scene
 * to an off-screen texture, then use that texture as input for the
 * next pass. Shadows, reflections, post-processing, and deferred
 * rendering all rely on this technique.
 */
function createRenderTarget(gl, width, height) {
    // Create framebuffer
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

    // Create color texture
    const colorTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, colorTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0,
                  gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, colorTexture, 0);

    // Create depth renderbuffer
    const depthBuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24,
                           width, height);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
                               gl.RENDERBUFFER, depthBuffer);

    // Verify completeness
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Framebuffer is not complete');
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, colorTexture, depthBuffer, width, height };
}
```

---

## 8. Practical Examples

### 8.1 Phong Lighting Shader

A complete Phong/Blinn-Phong lighting implementation in GLSL:

```glsl
// ═══════════════════════════════════════════
// VERTEX SHADER: phong_vert.glsl
// ═══════════════════════════════════════════
#version 300 es

in vec3 aPosition;
in vec3 aNormal;
in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;

void main() {
    // Compute world-space position for lighting calculations
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    vWorldPos = worldPos.xyz;

    // Transform normal to world space
    // Normal matrix = inverse transpose of model's upper-left 3x3
    vNormal = uNormalMatrix * aNormal;

    // Pass through UVs
    vTexCoord = aTexCoord;

    // Output clip-space position
    gl_Position = uProjection * uView * worldPos;
}
```

```glsl
// ═══════════════════════════════════════════
// FRAGMENT SHADER: phong_frag.glsl
// ═══════════════════════════════════════════
#version 300 es
precision highp float;

// Interpolated inputs from vertex shader
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;

// Material and lighting uniforms
uniform sampler2D uDiffuseMap;
uniform vec3 uLightPos;
uniform vec3 uLightColor;
uniform vec3 uCameraPos;
uniform float uShininess;
uniform float uAmbientStrength;
uniform float uSpecularStrength;

out vec4 fragColor;

void main() {
    // ── Prepare vectors ──
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightPos - vWorldPos);
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 H = normalize(L + V);  // Blinn-Phong halfway vector

    // ── Sample material texture ──
    vec3 albedo = texture(uDiffuseMap, vTexCoord).rgb;

    // Convert from sRGB to linear space for correct lighting
    albedo = pow(albedo, vec3(2.2));

    // ── Ambient: fake indirect lighting ──
    vec3 ambient = uAmbientStrength * albedo;

    // ── Diffuse: Lambert's law ──
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = NdotL * albedo * uLightColor;

    // ── Specular: Blinn-Phong ──
    float NdotH = max(dot(N, H), 0.0);
    float spec = pow(NdotH, uShininess);
    vec3 specular = uSpecularStrength * spec * uLightColor;

    // ── Attenuation: inverse square law ──
    float distance = length(uLightPos - vWorldPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);

    // ── Combine ──
    vec3 result = ambient + (diffuse + specular) * attenuation;

    // Tone mapping (Reinhard) and gamma correction
    result = result / (result + vec3(1.0));       // HDR -> LDR
    result = pow(result, vec3(1.0 / 2.2));        // Linear -> sRGB

    fragColor = vec4(result, 1.0);
}
```

### 8.2 Post-Processing Blur Shader

Post-processing shaders render a full-screen quad and read from a texture (the previous pass's output):

```glsl
// ═══════════════════════════════════════════
// VERTEX SHADER: fullscreen_vert.glsl
// ═══════════════════════════════════════════
#version 300 es

// Full-screen triangle trick: no vertex buffer needed!
// Uses gl_VertexID to generate 3 vertices covering the screen
// This is more efficient than a quad (2 triangles) because
// it avoids the diagonal edge and its associated overhead.

out vec2 vTexCoord;

void main() {
    // Generate a full-screen triangle from vertex index
    // Vertex 0: (-1, -1)  UV: (0, 0)
    // Vertex 1: ( 3, -1)  UV: (2, 0)
    // Vertex 2: (-1,  3)  UV: (0, 2)
    // The triangle extends beyond the screen; the GPU clips it.
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    vTexCoord = vec2((x + 1.0) * 0.5, (y + 1.0) * 0.5);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
```

```glsl
// ═══════════════════════════════════════════
// FRAGMENT SHADER: blur_frag.glsl
// ═══════════════════════════════════════════
#version 300 es
precision highp float;

in vec2 vTexCoord;

uniform sampler2D uInputTexture;
uniform vec2 uTexelSize;     // 1.0 / textureResolution
uniform vec2 uDirection;     // (1,0) for horizontal, (0,1) for vertical
uniform float uRadius;       // Blur radius in pixels

out vec4 fragColor;

void main() {
    // ── Gaussian blur (9-tap kernel) ──
    // Gaussian weights for sigma ≈ 2.5
    // Precomputed to avoid per-pixel exp() calls
    float weights[5];
    weights[0] = 0.227027;  // Center
    weights[1] = 0.194596;
    weights[2] = 0.121622;
    weights[3] = 0.054054;
    weights[4] = 0.016216;

    // Start with the center texel (weight[0])
    vec3 result = texture(uInputTexture, vTexCoord).rgb * weights[0];

    // Sample symmetric pairs around the center
    for (int i = 1; i < 5; i++) {
        vec2 offset = uDirection * uTexelSize * float(i) * uRadius;

        // Sample in both directions and weight equally
        result += texture(uInputTexture, vTexCoord + offset).rgb * weights[i];
        result += texture(uInputTexture, vTexCoord - offset).rgb * weights[i];
    }

    fragColor = vec4(result, 1.0);
}
```

### 8.3 Using the Blur in Multi-Pass

```javascript
/**
 * Apply Gaussian blur as a post-processing effect.
 *
 * Gaussian blur is separable: a 2D blur can be decomposed into
 * two 1D blurs (horizontal then vertical). This reduces the
 * complexity from O(r^2) to O(2r) samples per pixel.
 */
function applyBlur(gl, inputTexture, renderTargetA, renderTargetB,
                   blurProgram, fullscreenVAO, radius) {
    gl.useProgram(blurProgram);

    const texelSizeLoc = gl.getUniformLocation(blurProgram, 'uTexelSize');
    const dirLoc = gl.getUniformLocation(blurProgram, 'uDirection');
    const radiusLoc = gl.getUniformLocation(blurProgram, 'uRadius');
    const texLoc = gl.getUniformLocation(blurProgram, 'uInputTexture');

    gl.uniform1f(radiusLoc, radius);
    gl.uniform1i(texLoc, 0);  // Texture unit 0
    gl.activeTexture(gl.TEXTURE0);

    // Pass 1: Horizontal blur (input -> renderTargetA)
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderTargetA.fbo);
    gl.uniform2f(texelSizeLoc,
                 1.0 / renderTargetA.width, 1.0 / renderTargetA.height);
    gl.uniform2f(dirLoc, 1.0, 0.0);  // Horizontal
    gl.bindTexture(gl.TEXTURE_2D, inputTexture);
    gl.bindVertexArray(fullscreenVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pass 2: Vertical blur (renderTargetA -> renderTargetB)
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderTargetB.fbo);
    gl.uniform2f(dirLoc, 0.0, 1.0);  // Vertical
    gl.bindTexture(gl.TEXTURE_2D, renderTargetA.colorTexture);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // renderTargetB.colorTexture now contains the blurred result
}
```

---

## 9. Shader Debugging Techniques

### 9.1 Visual Debugging

The most effective way to debug shaders is to output intermediate values as colors:

```glsl
// ── Debug normals: encode [-1,1] as [0,1] RGB ──
fragColor = vec4(vNormal * 0.5 + 0.5, 1.0);
// Correct normals: smooth rainbow over the surface
// Wrong normals: black, pure white, or uniform color

// ── Debug texture coordinates ──
fragColor = vec4(vTexCoord, 0.0, 1.0);
// Correct: smooth gradient from black to red (U) and green (V)

// ── Debug depth ──
float depth = gl_FragCoord.z;
fragColor = vec4(vec3(depth), 1.0);
// Near objects: dark; far objects: bright

// ── Debug lighting vectors ──
fragColor = vec4(normalize(uLightPos - vWorldPos) * 0.5 + 0.5, 1.0);

// ── Heatmap for numerical values ──
vec3 heatmap(float t) {
    // Blue (0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Red (1.0)
    return vec3(
        smoothstep(0.5, 0.75, t),
        smoothstep(0.0, 0.25, t) - smoothstep(0.75, 1.0, t),
        1.0 - smoothstep(0.25, 0.5, t)
    );
}
```

### 9.2 Common Shader Bugs

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All black | Normals facing away, no light reaching surface | Check normal direction, light position |
| All white | Values > 1.0 not clamped | Add tone mapping or clamp |
| Faceted appearance | Flat normals, not interpolating | Use smooth vertex normals |
| Texture stretching | Wrong UV coordinates | Debug UVs with color output |
| Flickering | Z-fighting, NaN in calculation | Check depth range, add epsilon to divisions |
| Moving texture | Missing perspective-correct interpolation | Ensure proper `in/out` usage (automatic in GLSL) |

---

## 10. Shader Performance Tips

### 10.1 General Guidelines

```glsl
// ── AVOID: branching in fragment shaders ──
// GPUs process fragments in groups (warps/wavefronts).
// If fragments within a group take different branches, BOTH branches
// are executed and results masked. This is called "divergence."

// BAD: divergent branch
if (vWorldPos.y > 0.0) {
    color = expensiveCalculation1();
} else {
    color = expensiveCalculation2();
}

// BETTER: use mix/step to avoid branching
float mask = step(0.0, vWorldPos.y);
color = mix(cheapResult2, cheapResult1, mask);

// ── AVOID: unnecessary normalization ──
// normalize() costs sqrt + division. Only normalize when needed.
vec3 N = normalize(vNormal);        // NEEDED (interpolated, not unit length)
vec3 L = normalize(uLightDir);      // NOT NEEDED if uLightDir is already unit

// ── PREFER: built-in functions over manual computation ──
// Built-ins map to GPU hardware instructions and are always faster.
float d = length(v);                // GOOD: single hardware instruction
float d = sqrt(dot(v, v));          // EQUIVALENT but more verbose
float d = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);  // BAD: manual expansion

// ── PREFER: MAD (multiply-add) friendly expressions ──
// GPUs have fused multiply-add (FMA) that computes a*b+c in one cycle
float result = a * b + c;           // One FMA instruction
```

### 10.2 Texture Performance

```glsl
// ── Use mipmaps: always ──
// Without mipmaps, distant textures cause cache thrashing
// (neighboring pixels access vastly different texel addresses)

// ── Minimize texture samples per fragment ──
// Each sample has latency (cache miss ~400 cycles).
// Use texture atlases to reduce the number of unique textures.

// ── Use appropriate precision ──
precision mediump float;  // Sufficient for most fragment operations
precision highp float;    // Only when needed (position calculations, shadows)
```

---

## Summary

| GLSL Concept | Description | Example |
|-------------|-------------|---------|
| **Vector types** | vec2, vec3, vec4 | `vec3 color = vec3(1.0, 0.5, 0.0);` |
| **Swizzling** | Component rearrangement | `v.rgb`, `v.xzy`, `v.rrr` |
| **Matrix types** | mat2, mat3, mat4 (column-major) | `mat4 mvp = proj * view * model;` |
| **Vertex shader** | Per-vertex processing | Sets `gl_Position`, outputs varyings |
| **Fragment shader** | Per-pixel color computation | Reads varyings, outputs `fragColor` |
| **Uniforms** | Per-draw-call constants | Matrices, light positions, time |
| **Attributes/in** | Per-vertex data from buffers | Position, normal, UV |
| **Varyings out/in** | Interpolated vertex-to-fragment data | World position, normal |
| **Built-in functions** | Hardware-accelerated operations | `mix`, `smoothstep`, `normalize`, `reflect` |
| **Preprocessor** | Conditional compilation | `#ifdef`, `#define`, uber-shaders |
| **Multi-pass** | Render to texture, then process | Deferred shading, post-processing |

**Key takeaways**:
- GLSL vectors and swizzling provide concise, GPU-efficient notation for graphics math
- Vertex shaders transform geometry; fragment shaders compute per-pixel appearance
- Data flows: attributes (per-vertex) -> varyings (interpolated) -> fragment output
- Built-in functions like `mix`, `smoothstep`, `normalize`, and `reflect` are the building blocks of shader effects
- Multi-pass rendering enables complex effects by chaining shaders through framebuffer textures
- Shader debugging relies on visual output -- encoding intermediate values as colors
- Performance optimization: avoid divergent branches, minimize texture samples, prefer built-ins

---

## Exercises

1. **Swizzle Practice**: Given `vec4 v = vec4(1.0, 2.0, 3.0, 4.0)`, what are the values of: `v.wzyx`, `v.xxyy`, `v.rgb`, `v.stp`? Write a fragment shader that outputs different swizzle combinations as colors and verify your answers.

2. **Phong to PBR Upgrade**: Take the Phong lighting shader from Section 8.1 and upgrade it to PBR (Cook-Torrance) as described in Lesson 05. Add `uniform float uRoughness` and `uniform float uMetallic` and implement the GGX NDF, Schlick Fresnel, and Smith geometry function in GLSL.

3. **Procedural Texture**: Write a fragment shader that generates a procedural checkerboard pattern using only `gl_FragCoord` and `step()`/`mod()`. Add anti-aliasing to the pattern edges using `smoothstep()` instead of `step()`.

4. **Toon Shading**: Implement a cel-shading (toon) shader that quantizes the diffuse lighting into 3-4 discrete bands. Add a black outline by detecting edge fragments (where `dot(N, V)` is close to 0).

5. **Gaussian Blur**: Implement the two-pass Gaussian blur from Section 8.2 and 8.3 in a WebGL application. Render a scene to a texture, then apply the blur. Allow the user to adjust the blur radius with a slider.

6. **Heat Distortion**: Create a post-processing shader that distorts the scene by offsetting UV coordinates based on a scrolling noise texture. This creates a "heat haze" or "underwater" effect. Use `sin()` and `texture()` to compute the distortion offsets.

---

## Further Reading

1. [OpenGL Shading Language Specification (4.60)](https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.pdf) -- The official GLSL reference
2. [The Book of Shaders](https://thebookofshaders.com/) -- Beautiful, interactive introduction to fragment shaders
3. [Shadertoy](https://www.shadertoy.com/) -- Community of creative shader programming (thousands of examples)
4. [Learn OpenGL - Shaders](https://learnopengl.com/Getting-started/Shaders) -- Practical shader programming tutorial
5. Akenine-Moller, T. et al. *Real-Time Rendering* (4th ed.), Ch. 3 -- "The Graphics Processing Unit" and Ch. 5 -- "Shading Basics"
