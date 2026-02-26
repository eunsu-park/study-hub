# 08. 셰이더 프로그래밍 (GLSL)

[← 이전: 07. WebGL 기초](07_WebGL_Fundamentals.md) | [다음: 09. 씬 그래프와 공간 자료구조 →](09_Scene_Graphs_and_Spatial_Data_Structures.md)

---

## 학습 목표

1. GLSL 자료형을 완전히 익힌다: 스칼라(scalar), 벡터(vec2/3/4), 행렬(mat2/3/4), 스위즐링(swizzling)
2. 변환을 수행하고 프래그먼트 셰이더로 데이터를 전달하는 버텍스 셰이더를 작성한다
3. 텍스처 샘플링을 포함하여 픽셀당 색상을 계산하는 프래그먼트 셰이더를 작성한다
4. 유니폼(uniform), 어트리뷰트(in), 베어링(varying, out/in)의 역할을 이해한다
5. 필수 내장 함수를 사용한다: mix, clamp, smoothstep, normalize, reflect, dot
6. 조건부 컴파일과 디버깅을 위한 전처리기 지시문(preprocessor directive)을 사용한다
7. 멀티 패스 렌더링(multi-pass rendering) 개념과 셰이더가 각 패스에 참여하는 방식을 이해한다
8. 실용적인 셰이더 효과를 구현한다: Phong 조명, 텍스처 매핑, 후처리(post-processing) 블러

---

## 왜 중요한가

셰이더는 GPU에서 실행되는 프로그램으로, 이것을 마스터하는 것이 단순히 3D 엔진을 사용하는 사람과 그래픽 프로그래머를 구별하는 기준입니다. 현대 게임과 애플리케이션에서 볼 수 있는 모든 시각 효과 -- 사실적인 조명부터 수면 반사, 셀 셰이딩(cel-shading)부터 화면 공간 주변 차폐(screen-space ambient occlusion)까지 -- 는 셰이더로 구현됩니다. GLSL(OpenGL Shading Language)은 OpenGL과 WebGL에서 셰이더에 사용되는 언어입니다. GLSL을 깊이 이해하면 상상하는 모든 시각 효과를 구현하고, 렌더링 성능을 최적화하며, 그렇지 않았다면 불가사의한 블랙박스로 남았을 시각적 결함을 디버깅할 수 있습니다.

---

## 1. GLSL 기초

### 1.1 프로그램 구조

모든 GLSL 셰이더는 진입점(entry point)으로 `main()` 함수를 가집니다. `main`의 반환값은 없으며, 대신 셰이더는 내장 또는 선언된 출력 변수에 값을 씁니다.

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

### 1.2 스칼라 타입

| 타입 | 설명 | 예시 |
|------|------|------|
| `bool` | 불리언(Boolean) | `bool flag = true;` |
| `int` | 32비트 부호 있는 정수 | `int count = 42;` |
| `uint` | 32비트 부호 없는 정수 | `uint mask = 0xFFu;` |
| `float` | 32비트 부동소수점 | `float pi = 3.14159;` |

### 1.3 벡터 타입

GLSL의 벡터 타입은 셰이더 프로그래밍의 핵심 도구입니다:

| 타입 | 성분 | 사용 |
|------|------|------|
| `vec2` | float 2개 | UV 좌표, 2D 위치 |
| `vec3` | float 3개 | 3D 위치, RGB 색상, 법선(normal) |
| `vec4` | float 4개 | 동차 좌표(homogeneous position), RGBA 색상 |
| `ivec2/3/4` | int 2/3/4개 | 픽셀 인덱스, 정수 좌표 |
| `bvec2/3/4` | bool 2/3/4개 | 성분별 불리언 결과 |

**생성(Construction)**:

```glsl
vec3 color = vec3(1.0, 0.5, 0.0);      // Explicit components
vec3 white = vec3(1.0);                  // All components = 1.0
vec4 pos = vec4(color, 1.0);            // Extend vec3 with a 4th component
vec2 uv = vec2(0.5, 0.5);              // 2D coordinate
vec4 full = vec4(uv, 0.0, 1.0);        // Combine vec2 + scalars
```

### 1.4 스위즐링(Swizzling)

**스위즐링(Swizzling)**은 벡터 성분을 재배열, 복제, 추출하는 GLSL의 강력한 문법입니다:

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

> **스위즐링이 중요한 이유**: 다른 언어에서는 인덱스 기반으로 장황하게 접근해야 할 것을 간결하고 읽기 쉬운 표현으로 대체합니다. GPU는 스위즐 연산을 비용 없이 실행합니다 -- 단순한 레지스터 읽기 마스크에 불과합니다.

### 1.5 행렬 타입

| 타입 | 크기 | 사용 |
|------|------|------|
| `mat2` | 2x2 | 2D 회전 |
| `mat3` | 3x3 | 법선 변환, 2D 아핀(affine) |
| `mat4` | 4x4 | MVP, 모델, 뷰, 투영 |

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

> **중요**: GLSL 행렬은 **열 우선(column-major)**입니다. `m[i]`는 행 `i`가 아니라 열 `i`에 접근합니다. 이는 OpenGL의 관례와 `gl.uniformMatrix4fv`가 기대하는 `Float32Array` 레이아웃과 일치합니다.

---

## 2. 버텍스 셰이더

버텍스 셰이더는 각 버텍스를 독립적으로 처리합니다. 주된 역할은 `gl_Position`(클립 공간 위치)을 설정하고 프래그먼트 셰이더로 데이터를 전달하는 것입니다.

### 2.1 입력/출력

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

### 2.2 버텍스 셰이더 내장 변수

| 변수 | 타입 | 설명 |
|------|------|------|
| `gl_Position` | `vec4` | **출력** (필수): 클립 공간 위치 |
| `gl_PointSize` | `float` | 출력: 점 프리미티브의 크기 |
| `gl_VertexID` | `int` | 입력: 현재 버텍스의 인덱스 |
| `gl_InstanceID` | `int` | 입력: 현재 인스턴스의 인덱스 (인스턴스 렌더링) |

---

## 3. 프래그먼트 셰이더

프래그먼트 셰이더는 프래그먼트(픽셀 후보)마다 한 번씩 실행되며 출력 색상을 계산합니다.

### 3.1 입력/출력

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

### 3.2 프래그먼트 셰이더 내장 변수

| 변수 | 타입 | 설명 |
|------|------|------|
| `gl_FragCoord` | `vec4` | 입력: 윈도우 공간 위치 (x, y, z, 1/w) |
| `gl_FrontFacing` | `bool` | 입력: 프래그먼트가 앞면(front-facing)이면 true |
| `gl_FragDepth` | `float` | 출력 (선택): 깊이값 오버라이드 |

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

## 4. 유니폼 vs 어트리뷰트 vs 베어링

### 4.1 데이터 흐름 요약

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

### 4.2 비교표

| 한정자 | GLSL 300 es | 설정 주체 | 빈도 | 예시 |
|--------|-------------|-----------|------|------|
| **어트리뷰트(Attribute)** | `in` (버텍스) | `vertexAttribPointer` | 버텍스당 | 위치, 법선, UV |
| **베어링(Varying)** | `out` (버텍스) / `in` (프래그먼트) | 버텍스 셰이더 | 프래그먼트당 보간 | 월드 위치, 법선 |
| **유니폼(Uniform)** | `uniform` | `gl.uniform*()` | 드로우 콜당 | MVP 행렬, 광원 위치 |

### 4.3 플랫 보간(Flat Interpolation)

기본적으로 버텍스 셰이더의 출력은 삼각형 전체에 걸쳐 부드럽게 보간됩니다. `flat` 한정자는 보간을 비활성화합니다 -- **주도 버텍스(provoking vertex)**(보통 삼각형의 첫 번째 또는 마지막 버텍스)의 값이 모든 프래그먼트에 사용됩니다:

```glsl
// Vertex shader
flat out int vFaceID;    // No interpolation -- same value for entire triangle

// Fragment shader
flat in int vFaceID;     // Receives the uninterpolated value
```

---

## 5. 필수 내장 함수

GLSL은 GPU에서 하드웨어 가속되는 풍부한 내장 함수 세트를 제공합니다.

### 5.1 수학 함수

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

### 5.2 기하 함수

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

### 5.3 텍스처 함수

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

### 5.4 자주 쓰이는 패턴

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

## 6. 전처리기 지시문

GLSL은 조건부 컴파일을 위한 C 언어 스타일의 전처리기를 포함합니다:

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

### 6.1 우버 셰이더(Uber-Shader)

단일 "우버 셰이더(uber-shader)"에 많은 `#ifdef` 블록을 작성하고 컴파일 시 변형(variant)을 생성하는 기법은 널리 사용됩니다:

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

## 7. 멀티 패스 렌더링

### 7.1 멀티 패스 렌더링이란?

단일 패스로 전체 장면을 렌더링하는 대신, **멀티 패스 렌더링(multi-pass rendering)**은 장면을 여러 번 그리거나(또는 중간 텍스처에 그려), 각 패스가 최종 이미지의 다른 측면을 계산합니다.

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

### 7.2 텍스처로 렌더링(Framebuffer Object)

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

## 8. 실용 예제

### 8.1 Phong 조명 셰이더

GLSL로 구현한 완전한 Phong/Blinn-Phong 조명:

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

### 8.2 후처리 블러 셰이더

후처리 셰이더는 전체 화면 쿼드를 렌더링하고 텍스처(이전 패스의 출력)에서 읽습니다:

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

### 8.3 멀티 패스에서 블러 사용하기

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

## 9. 셰이더 디버깅 기법

### 9.1 시각적 디버깅

셰이더를 디버깅하는 가장 효과적인 방법은 중간 값을 색상으로 출력하는 것입니다:

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

### 9.2 흔한 셰이더 버그

| 증상 | 가능한 원인 | 해결책 |
|------|------------|--------|
| 전부 검정 | 법선이 반대 방향, 광원이 표면에 도달하지 않음 | 법선 방향, 광원 위치 확인 |
| 전부 흰색 | 1.0을 초과하는 값이 클램핑되지 않음 | 톤 매핑 추가 또는 clamp 사용 |
| 면 각진 외관 | 플랫 법선, 보간되지 않음 | 부드러운 버텍스 법선 사용 |
| 텍스처 늘어남 | 잘못된 UV 좌표 | 색상 출력으로 UV 디버깅 |
| 깜박임 | Z-파이팅(Z-fighting), 계산에서 NaN 발생 | 깊이 범위 확인, 나눗셈에 엡실론(epsilon) 추가 |
| 움직이는 텍스처 | 퍼스펙티브 보정 보간(perspective-correct interpolation) 누락 | 올바른 `in/out` 사용 확인 (GLSL에서 자동) |

---

## 10. 셰이더 성능 팁

### 10.1 일반 지침

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

### 10.2 텍스처 성능

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

## 요약

| GLSL 개념 | 설명 | 예시 |
|-----------|------|------|
| **벡터 타입(Vector types)** | vec2, vec3, vec4 | `vec3 color = vec3(1.0, 0.5, 0.0);` |
| **스위즐링(Swizzling)** | 성분 재배열 | `v.rgb`, `v.xzy`, `v.rrr` |
| **행렬 타입(Matrix types)** | mat2, mat3, mat4 (열 우선) | `mat4 mvp = proj * view * model;` |
| **버텍스 셰이더(Vertex shader)** | 버텍스당 처리 | `gl_Position` 설정, 베어링 출력 |
| **프래그먼트 셰이더(Fragment shader)** | 픽셀당 색상 계산 | 베어링 읽기, `fragColor` 출력 |
| **유니폼(Uniforms)** | 드로우 콜당 상수 | 행렬, 광원 위치, 시간 |
| **어트리뷰트/in** | 버퍼에서 오는 버텍스당 데이터 | 위치, 법선, UV |
| **베어링 out/in** | 버텍스에서 프래그먼트로 보간된 데이터 | 월드 위치, 법선 |
| **내장 함수(Built-in functions)** | 하드웨어 가속 연산 | `mix`, `smoothstep`, `normalize`, `reflect` |
| **전처리기(Preprocessor)** | 조건부 컴파일 | `#ifdef`, `#define`, 우버 셰이더 |
| **멀티 패스(Multi-pass)** | 텍스처로 렌더링 후 처리 | 디퍼드 셰이딩(deferred shading), 후처리 |

**핵심 정리**:
- GLSL 벡터와 스위즐링은 그래픽 수학을 위한 간결하고 GPU 효율적인 표기법을 제공한다
- 버텍스 셰이더는 기하체를 변환하고, 프래그먼트 셰이더는 픽셀당 외관을 계산한다
- 데이터 흐름: 어트리뷰트(버텍스당) → 베어링(보간) → 프래그먼트 출력
- `mix`, `smoothstep`, `normalize`, `reflect` 같은 내장 함수가 셰이더 효과의 구성 요소이다
- 멀티 패스 렌더링은 프레임버퍼 텍스처를 통해 셰이더를 체이닝하여 복잡한 효과를 구현한다
- 셰이더 디버깅은 시각적 출력에 의존한다 -- 중간 값을 색상으로 인코딩한다
- 성능 최적화: 분기(divergent branch)를 피하고, 텍스처 샘플을 최소화하며, 내장 함수를 선호한다

---

## 연습 문제

1. **스위즐 연습**: `vec4 v = vec4(1.0, 2.0, 3.0, 4.0)`가 주어졌을 때, `v.wzyx`, `v.xxyy`, `v.rgb`, `v.stp`의 값은 무엇인가요? 다양한 스위즐 조합을 색상으로 출력하는 프래그먼트 셰이더를 작성하고 답을 확인해보세요.

2. **Phong에서 PBR로 업그레이드**: 8.1절의 Phong 조명 셰이더를 5강에서 설명한 PBR(Cook-Torrance)로 업그레이드하세요. `uniform float uRoughness`와 `uniform float uMetallic`을 추가하고 GLSL로 GGX NDF, Schlick Fresnel, Smith 기하 함수를 구현해보세요.

3. **절차적 텍스처(Procedural Texture)**: `gl_FragCoord`와 `step()`/`mod()`만을 사용하여 절차적 체커보드 패턴을 생성하는 프래그먼트 셰이더를 작성하세요. `step()` 대신 `smoothstep()`을 사용하여 패턴 가장자리에 안티 앨리어싱을 추가해보세요.

4. **툰 셰이딩(Toon Shading)**: 디퓨즈 조명을 3~4개의 이산 밴드로 양자화하는 셀 셰이딩(cel-shading, 툰) 셰이더를 구현하세요. `dot(N, V)`가 0에 가까운 가장자리 프래그먼트를 감지하여 검정 외곽선을 추가해보세요.

5. **가우시안 블러(Gaussian Blur)**: 8.2절과 8.3절의 2패스 가우시안 블러를 WebGL 애플리케이션에 구현하세요. 장면을 텍스처로 렌더링한 후 블러를 적용하고, 슬라이더로 블러 반경을 조정할 수 있게 만들어보세요.

6. **열 왜곡(Heat Distortion)**: 스크롤되는 노이즈 텍스처를 기반으로 UV 좌표를 오프셋하여 장면을 왜곡하는 후처리 셰이더를 만드세요. 이를 통해 "아지랑이" 또는 "수중" 효과를 구현할 수 있습니다. 왜곡 오프셋 계산에는 `sin()`과 `texture()`를 사용하세요.

---

## 더 읽을거리

1. [OpenGL Shading Language Specification (4.60)](https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.pdf) -- 공식 GLSL 레퍼런스
2. [The Book of Shaders](https://thebookofshaders.com/) -- 프래그먼트 셰이더에 대한 아름답고 인터랙티브한 입문서
3. [Shadertoy](https://www.shadertoy.com/) -- 크리에이티브 셰이더 프로그래밍 커뮤니티 (수천 가지 예제)
4. [Learn OpenGL - Shaders](https://learnopengl.com/Getting-started/Shaders) -- 실용적인 셰이더 프로그래밍 튜토리얼
5. Akenine-Moller, T. et al. *Real-Time Rendering* (4th ed.), Ch. 3 -- "The Graphics Processing Unit" and Ch. 5 -- "Shading Basics"
