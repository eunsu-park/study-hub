# 07. WebGL 기초

[← 이전: 06. 텍스처 매핑](06_Texture_Mapping.md) | [다음: 08. 셰이더 프로그래밍 (GLSL) →](08_Shader_Programming_GLSL.md)

---

## 학습 목표

1. HTML 캔버스에서 WebGL 렌더링 컨텍스트(rendering context)를 설정한다
2. 버텍스 버퍼(vertex buffer)와 인덱스 버퍼(index buffer)를 생성하고 바인딩한다
3. GLSL ES로 기본 버텍스 셰이더와 프래그먼트 셰이더를 작성한다
4. 버퍼, 셰이더, 유니폼으로 구성되는 WebGL 드로우 콜 파이프라인을 이해한다
5. `drawArrays`와 `drawElements`를 사용해 기하체(geometry)를 렌더링한다
6. 유니폼(uniform)과 어트리뷰트(attribute)를 통해 셰이더에 데이터를 전달한다
7. 인터랙티브 렌더링을 위한 기본 사용자 입력(마우스, 키보드)을 처리한다
8. 처음부터 완성된 인터랙티브 WebGL 애플리케이션을 구축한다

---

## 왜 중요한가

WebGL은 GPU 가속 3D 그래픽의 기능을 모든 웹 브라우저에서 사용할 수 있게 해줍니다 -- 플러그인도, 다운로드도, 플랫폼별 코드도 필요 없습니다. 01~06강에서 배운 개념들(파이프라인, 변환, 래스터화, 셰이딩, 텍스처)이 WebGL에서 구현될 때 비로소 생생하게 살아납니다. Python 소프트웨어 래스터라이저와 달리 WebGL은 실제 GPU 하드웨어를 활용하므로, 복잡한 장면을 60+ FPS로 실시간 렌더링할 수 있습니다. WebGL은 3D 웹 경험의 토대입니다: 데이터 시각화(Three.js, Deck.gl), 게임(PlayCanvas, Babylon.js), 가상 피팅, 3D 제품 구성기, 크리에이티브 코딩 등에 활용됩니다. 라이브러리를 사용하기 전에 WebGL의 원시 API를 이해하면 모든 3D 렌더링이 실제로 어떻게 작동하는지 깊이 있게 파악할 수 있습니다.

> **참고**: 이 강의에서는 Python 대신 JavaScript와 HTML을 사용합니다. 개념은 우리가 학습한 파이프라인과 직접 매핑되지만, 언어와 API가 다릅니다.

---

## 1. WebGL 아키텍처

WebGL은 HTML5 `<canvas>` 요소를 통해 GPU에 접근하는 JavaScript API입니다. OpenGL ES 2.0(WebGL 1)과 OpenGL ES 3.0(WebGL 2)을 기반으로 합니다.

```
┌──────────────────────────────────────────────┐
│                 JavaScript                    │
│  (Scene setup, matrix math, draw calls)      │
│                     │                         │
│                     ▼                         │
│              WebGL API                        │
│  (gl.bindBuffer, gl.drawArrays, etc.)        │
│                     │                         │
│                     ▼                         │
│               GPU Driver                      │
│                     │                         │
│                     ▼                         │
│                GPU Hardware                   │
│  (Vertex shader → Rasterizer → Fragment)     │
└──────────────────────────────────────────────┘
```

---

## 2. WebGL 설정

### 2.1 HTML 구조

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WebGL Fundamentals</title>
    <style>
        /* Make the canvas fill the viewport */
        body { margin: 0; overflow: hidden; background: #000; }
        canvas { display: block; width: 100vw; height: 100vh; }
    </style>
</head>
<body>
    <canvas id="glCanvas"></canvas>
    <script src="main.js"></script>
</body>
</html>
```

### 2.2 WebGL 컨텍스트 가져오기

```javascript
/**
 * Initialize WebGL and return the rendering context.
 *
 * Why getContext('webgl2')? WebGL 2 (based on OpenGL ES 3.0) adds
 * features like instanced rendering, multiple render targets, and
 * transform feedback. We fall back to WebGL 1 if unavailable.
 */
function initWebGL(canvasId) {
    const canvas = document.getElementById(canvasId);

    // Match the canvas internal resolution to its display size
    // Without this, rendering at a lower resolution causes blurriness
    canvas.width = canvas.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.clientHeight * window.devicePixelRatio;

    // Try WebGL 2 first, fall back to WebGL 1
    let gl = canvas.getContext('webgl2');
    if (!gl) {
        gl = canvas.getContext('webgl');
        console.warn('WebGL 2 not available, falling back to WebGL 1');
    }
    if (!gl) {
        throw new Error('WebGL not supported by this browser');
    }

    console.log(`Canvas: ${canvas.width}x${canvas.height}`);
    console.log(`WebGL Version: ${gl.getParameter(gl.VERSION)}`);
    console.log(`GLSL Version: ${gl.getParameter(gl.SHADING_LANGUAGE_VERSION)}`);

    return { gl, canvas };
}
```

### 2.3 렌더링 루프

WebGL 렌더링은 매 프레임 동일한 구조를 따릅니다:

```javascript
/**
 * Main render loop using requestAnimationFrame.
 *
 * requestAnimationFrame synchronizes with the display's refresh rate
 * (typically 60 Hz), providing smooth animation and automatically
 * pausing when the tab is not visible (saving battery/CPU).
 */
function startRenderLoop(gl, renderFunction) {
    let lastTime = 0;

    function frame(currentTime) {
        // Convert to seconds and compute delta
        const time = currentTime * 0.001;
        const deltaTime = time - lastTime;
        lastTime = time;

        // Call the user's render function
        renderFunction(gl, time, deltaTime);

        // Request the next frame
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}
```

---

## 3. 버퍼: GPU로 데이터 전송하기

### 3.1 버텍스 버퍼

**버텍스 버퍼(VBO -- Vertex Buffer Object)**는 버텍스당 데이터(위치, 색상, 법선, UV)를 GPU 메모리에 저장합니다.

```javascript
/**
 * Create a vertex buffer and upload data to the GPU.
 *
 * Why Float32Array? The GPU operates on 32-bit floats natively.
 * JavaScript numbers are 64-bit doubles, so we must convert to
 * the GPU's expected format using typed arrays.
 */
function createVertexBuffer(gl, data) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
    return buffer;
}

// Example: a triangle with position (x, y) and color (r, g, b) interleaved
// Each vertex: [x, y, r, g, b]
const triangleData = [
    // Position      Color
     0.0,  0.5,    1.0, 0.0, 0.0,  // Top vertex (red)
    -0.5, -0.5,    0.0, 1.0, 0.0,  // Bottom-left (green)
     0.5, -0.5,    0.0, 0.0, 1.0,  // Bottom-right (blue)
];
```

### 3.2 인덱스 버퍼

**인덱스 버퍼(EBO/IBO)**는 각 삼각형을 구성하는 버텍스를 지정하여 버텍스를 재사용할 수 있게 합니다:

```javascript
/**
 * Create an index buffer for indexed drawing.
 *
 * Why use indices? A cube has 8 unique vertices but 12 triangles
 * (36 vertex references). Without indices, we'd need to duplicate
 * vertices, wasting memory and vertex shader computations.
 * With indices, we store 8 vertices + 36 indices instead of 36 vertices.
 */
function createIndexBuffer(gl, indices) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    return buffer;
}

// Example: a quad (two triangles sharing two vertices)
const quadVertices = [
    -0.5, -0.5,  // Vertex 0: bottom-left
     0.5, -0.5,  // Vertex 1: bottom-right
     0.5,  0.5,  // Vertex 2: top-right
    -0.5,  0.5,  // Vertex 3: top-left
];
const quadIndices = [
    0, 1, 2,  // First triangle
    0, 2, 3,  // Second triangle (reuses vertices 0 and 2)
];
```

### 3.3 버텍스 배열 객체(VAO)

**VAO(Vertex Array Object)**는 버퍼 바인딩과 어트리뷰트 구성 상태를 저장하므로, 매 프레임마다 재구성할 필요가 없습니다:

```javascript
/**
 * Create a VAO that remembers vertex attribute layout.
 *
 * Without VAOs, you'd need to rebind buffers and reconfigure
 * attributes before every draw call. VAOs save this state so
 * you just bind the VAO and draw.
 */
function createVAO(gl, vertexBuffer, indexBuffer, attributes) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    // Bind vertex buffer and configure attributes
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    for (const attr of attributes) {
        gl.enableVertexAttribArray(attr.location);
        gl.vertexAttribPointer(
            attr.location,  // Attribute index (from shader)
            attr.size,      // Number of components (2 for vec2, 3 for vec3)
            gl.FLOAT,       // Data type
            false,          // Normalize?
            attr.stride,    // Bytes between consecutive vertices
            attr.offset     // Byte offset of this attribute within a vertex
        );
    }

    // Bind index buffer (if present)
    if (indexBuffer) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    }

    gl.bindVertexArray(null);
    return vao;
}
```

---

## 4. 셰이더: GPU에서 실행되는 프로그램

### 4.1 셰이더 컴파일

WebGL 셰이더는 **GLSL ES(GL Shading Language for Embedded Systems)**로 작성되며 런타임에 컴파일됩니다:

```javascript
/**
 * Compile a shader from source code.
 *
 * Shaders are compiled by the GPU driver at runtime because
 * different GPUs have different instruction sets. This is similar
 * to JIT compilation in JavaScript engines.
 */
function compileShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error(`Shader compilation failed:\n${info}`);
    }

    return shader;
}

/**
 * Link vertex and fragment shaders into a program.
 *
 * A shader program is the complete GPU pipeline configuration:
 * vertex shader processes each vertex, fragment shader processes
 * each pixel. They must be linked together because they share
 * data (varyings pass from vertex to fragment shader).
 */
function createProgram(gl, vertexSource, fragmentSource) {
    const vertexShader = compileShader(gl, vertexSource, gl.VERTEX_SHADER);
    const fragmentShader = compileShader(gl, fragmentSource, gl.FRAGMENT_SHADER);

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(program);
        gl.deleteProgram(program);
        throw new Error(`Program linking failed:\n${info}`);
    }

    // Clean up individual shaders (they're now part of the program)
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    return program;
}
```

### 4.2 기본 버텍스 셰이더

```javascript
const vertexShaderSource = `#version 300 es
    // Vertex attributes (input from vertex buffer)
    // 'in' means data flows INTO the shader from the buffer
    in vec2 aPosition;   // Vertex position (x, y)
    in vec3 aColor;      // Vertex color (r, g, b)

    // Uniforms: constant for all vertices in a single draw call
    uniform mat4 uModelViewProjection;

    // Output to fragment shader (interpolated across the triangle)
    // 'out' means data flows OUT to the next stage
    out vec3 vColor;

    void main() {
        // gl_Position is the built-in output: clip-space position
        // It MUST be set in every vertex shader
        gl_Position = uModelViewProjection * vec4(aPosition, 0.0, 1.0);

        // Pass color to fragment shader (will be interpolated)
        vColor = aColor;
    }
`;
```

### 4.3 기본 프래그먼트 셰이더

```javascript
const fragmentShaderSource = `#version 300 es
    // Fragment shaders require precision qualifier in GLSL ES
    // mediump = medium precision (good balance of speed and quality)
    precision mediump float;

    // Input from vertex shader (interpolated per-fragment)
    in vec3 vColor;

    // Output: the fragment's color
    out vec4 fragColor;

    void main() {
        // Output the interpolated color with full opacity
        fragColor = vec4(vColor, 1.0);
    }
`;
```

---

## 5. 드로잉

### 5.1 drawArrays

`drawArrays`는 버퍼에서 연속된 버텍스를 사용해 프리미티브(primitive)를 그립니다:

```javascript
/**
 * Draw a triangle using drawArrays (non-indexed).
 *
 * gl.TRIANGLES: every 3 vertices form a triangle
 * Offset 0: start at the first vertex
 * Count 3: draw 3 vertices (one triangle)
 */
gl.drawArrays(gl.TRIANGLES, 0, 3);

// Other primitive modes:
// gl.POINTS         - each vertex is a point
// gl.LINES          - every 2 vertices form a line
// gl.LINE_STRIP     - connected lines
// gl.TRIANGLE_STRIP - each new vertex forms a triangle with prev 2
// gl.TRIANGLE_FAN   - all triangles share the first vertex
```

### 5.2 drawElements

`drawElements`는 인덱스 버퍼를 사용하여 버텍스 재사용이 가능한 방식으로 그립니다:

```javascript
/**
 * Draw indexed geometry.
 *
 * This is more efficient for meshes with shared vertices (most meshes).
 * The index buffer specifies which vertices form each triangle,
 * so vertices shared by multiple triangles are processed only once
 * by the vertex shader.
 */
gl.drawElements(
    gl.TRIANGLES,       // Primitive type
    6,                  // Number of indices
    gl.UNSIGNED_SHORT,  // Index data type (Uint16Array)
    0                   // Byte offset into index buffer
);
```

---

## 6. 유니폼과 어트리뷰트

### 6.1 어트리뷰트

**어트리뷰트(Attribute)**는 버텍스당 데이터로, 각 버텍스가 고유한 값을 갖습니다. `vertexAttribPointer`를 사용해 구성합니다:

```javascript
/**
 * Set up vertex attributes for a shader program.
 *
 * Attributes map vertex buffer data to shader input variables.
 * The stride and offset tell WebGL how to navigate the interleaved
 * data in the vertex buffer.
 */
function setupAttributes(gl, program, vertexBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);

    // Get attribute locations (the shader compiler assigns indices)
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    const colorLoc = gl.getAttribLocation(program, 'aColor');

    // Stride: total bytes per vertex (2 floats pos + 3 floats color = 5 * 4 = 20)
    const stride = 5 * Float32Array.BYTES_PER_ELEMENT;

    // Position: 2 floats starting at byte 0
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);

    // Color: 3 floats starting at byte 8 (after 2 position floats)
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, stride, 2 * 4);
}
```

### 6.2 유니폼

**유니폼(Uniform)**은 하나의 드로우 콜 내 모든 버텍스/프래그먼트에 대해 상수값을 가집니다. 행렬, 광원 위치, 시간, 텍스처 등에 사용됩니다.

```javascript
/**
 * Set uniform values for a shader program.
 *
 * Uniforms are "global" to the shader -- every vertex and every
 * fragment sees the same value. Use them for data that doesn't
 * change per-vertex: transformation matrices, light positions,
 * material properties, time, etc.
 */
function setUniforms(gl, program, uniforms) {
    gl.useProgram(program);

    for (const [name, value] of Object.entries(uniforms)) {
        const location = gl.getUniformLocation(program, name);
        if (location === null) continue;  // Uniform not used in shader

        if (value instanceof Float32Array && value.length === 16) {
            // 4x4 matrix
            gl.uniformMatrix4fv(location, false, value);
        } else if (value instanceof Float32Array && value.length === 4) {
            gl.uniform4fv(location, value);
        } else if (value instanceof Float32Array && value.length === 3) {
            gl.uniform3fv(location, value);
        } else if (typeof value === 'number') {
            gl.uniform1f(location, value);
        }
    }
}
```

---

## 7. 사용자 입력 처리

### 7.1 마우스 입력

```javascript
/**
 * Set up mouse interaction for rotating the scene.
 *
 * Common 3D interaction: drag to rotate, scroll to zoom.
 * We track mouse state and compute rotation angles from
 * cumulative mouse movement while the button is held.
 */
function setupMouseInput(canvas) {
    const state = {
        isDragging: false,
        rotationX: 0,      // Pitch (up/down)
        rotationY: 0,      // Yaw (left/right)
        lastX: 0,
        lastY: 0,
        zoom: 3.0,         // Camera distance
    };

    canvas.addEventListener('mousedown', (e) => {
        state.isDragging = true;
        state.lastX = e.clientX;
        state.lastY = e.clientY;
    });

    canvas.addEventListener('mouseup', () => {
        state.isDragging = false;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!state.isDragging) return;

        const dx = e.clientX - state.lastX;
        const dy = e.clientY - state.lastY;

        // Sensitivity: pixels of mouse movement per degree of rotation
        state.rotationY += dx * 0.5;
        state.rotationX += dy * 0.5;

        // Clamp pitch to prevent flipping
        state.rotationX = Math.max(-89, Math.min(89, state.rotationX));

        state.lastX = e.clientX;
        state.lastY = e.clientY;
    });

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        state.zoom *= (1 + e.deltaY * 0.001);
        state.zoom = Math.max(0.5, Math.min(20.0, state.zoom));
    });

    return state;
}
```

### 7.2 키보드 입력

```javascript
/**
 * Track which keys are currently pressed.
 *
 * Using a Set of pressed keys allows checking multiple simultaneous
 * keys (e.g., W + A for diagonal movement).
 */
function setupKeyboardInput() {
    const keys = new Set();

    window.addEventListener('keydown', (e) => {
        keys.add(e.code);
    });

    window.addEventListener('keyup', (e) => {
        keys.delete(e.code);
    });

    return keys;
}
```

---

## 8. JavaScript에서의 행렬 연산

JavaScript에는 내장 행렬 라이브러리가 없으므로 유틸리티 함수가 필요합니다:

```javascript
/**
 * Minimal matrix math library for WebGL.
 *
 * WebGL expects matrices in COLUMN-MAJOR order (OpenGL convention).
 * This means a matrix stored as [m0, m1, m2, m3, m4, ...m15]
 * represents:
 *   | m0  m4  m8  m12 |
 *   | m1  m5  m9  m13 |
 *   | m2  m6  m10 m14 |
 *   | m3  m7  m11 m15 |
 *
 * Note: in production, use gl-matrix (https://glmatrix.net/) --
 * it's fast, well-tested, and the industry standard for WebGL math.
 */
const Mat4 = {
    identity() {
        return new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]);
    },

    perspective(fovDeg, aspect, near, far) {
        const fov = fovDeg * Math.PI / 180;
        const f = 1.0 / Math.tan(fov / 2);
        const rangeInv = 1.0 / (near - far);

        // Column-major layout
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (near + far) * rangeInv, -1,
            0, 0, near * far * rangeInv * 2, 0,
        ]);
    },

    lookAt(eye, target, up) {
        // Forward (camera looks along -z)
        let fx = target[0] - eye[0];
        let fy = target[1] - eye[1];
        let fz = target[2] - eye[2];
        let len = Math.sqrt(fx * fx + fy * fy + fz * fz);
        fx /= len; fy /= len; fz /= len;

        // Right = forward x up
        let rx = fy * up[2] - fz * up[1];
        let ry = fz * up[0] - fx * up[2];
        let rz = fx * up[1] - fy * up[0];
        len = Math.sqrt(rx * rx + ry * ry + rz * rz);
        rx /= len; ry /= len; rz /= len;

        // True up = right x forward
        const ux = ry * fz - rz * fy;
        const uy = rz * fx - rx * fz;
        const uz = rx * fy - ry * fx;

        // Column-major, negated forward for OpenGL convention
        return new Float32Array([
            rx, ux, -fx, 0,
            ry, uy, -fy, 0,
            rz, uz, -fz, 0,
            -(rx * eye[0] + ry * eye[1] + rz * eye[2]),
            -(ux * eye[0] + uy * eye[1] + uz * eye[2]),
            (fx * eye[0] + fy * eye[1] + fz * eye[2]),
            1,
        ]);
    },

    rotateY(mat, angleDeg) {
        const a = angleDeg * Math.PI / 180;
        const c = Math.cos(a);
        const s = Math.sin(a);

        const result = new Float32Array(mat);
        // Multiply mat by rotation-Y matrix
        for (let i = 0; i < 4; i++) {
            const x = mat[i];
            const z = mat[i + 8];
            result[i] = x * c + z * s;
            result[i + 8] = x * -s + z * c;
        }
        return result;
    },

    multiply(a, b) {
        const out = new Float32Array(16);
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                out[j * 4 + i] =
                    a[i] * b[j * 4] +
                    a[i + 4] * b[j * 4 + 1] +
                    a[i + 8] * b[j * 4 + 2] +
                    a[i + 12] * b[j * 4 + 3];
            }
        }
        return out;
    },
};
```

---

## 9. 완성된 WebGL 애플리케이션

```javascript
/**
 * Complete WebGL application: a rotating, color-interpolated triangle.
 *
 * This combines all concepts from this lesson:
 * - WebGL context setup
 * - Vertex buffer creation
 * - Shader compilation and linking
 * - Uniform and attribute setup
 * - Render loop with animation
 * - Mouse interaction
 */

// ═══════════════════════════════════════════════════════
// Shader sources
// ═══════════════════════════════════════════════════════

const VS_SOURCE = `#version 300 es
    in vec3 aPosition;
    in vec3 aColor;

    uniform mat4 uMVP;

    out vec3 vColor;

    void main() {
        gl_Position = uMVP * vec4(aPosition, 1.0);
        vColor = aColor;
    }
`;

const FS_SOURCE = `#version 300 es
    precision mediump float;

    in vec3 vColor;
    out vec4 fragColor;

    void main() {
        fragColor = vec4(vColor, 1.0);
    }
`;

// ═══════════════════════════════════════════════════════
// Geometry: a colored cube
// ═══════════════════════════════════════════════════════

function createCube(gl) {
    // Each face has 4 vertices with position (x,y,z) and color (r,g,b)
    // Interleaved: [x, y, z, r, g, b, ...]
    const vertices = new Float32Array([
        // Front face (red)
        -0.5, -0.5,  0.5,  1, 0, 0,
         0.5, -0.5,  0.5,  1, 0, 0,
         0.5,  0.5,  0.5,  1, 0, 0,
        -0.5,  0.5,  0.5,  1, 0, 0,
        // Back face (green)
        -0.5, -0.5, -0.5,  0, 1, 0,
        -0.5,  0.5, -0.5,  0, 1, 0,
         0.5,  0.5, -0.5,  0, 1, 0,
         0.5, -0.5, -0.5,  0, 1, 0,
        // Top face (blue)
        -0.5,  0.5, -0.5,  0, 0, 1,
        -0.5,  0.5,  0.5,  0, 0, 1,
         0.5,  0.5,  0.5,  0, 0, 1,
         0.5,  0.5, -0.5,  0, 0, 1,
        // Bottom face (yellow)
        -0.5, -0.5, -0.5,  1, 1, 0,
         0.5, -0.5, -0.5,  1, 1, 0,
         0.5, -0.5,  0.5,  1, 1, 0,
        -0.5, -0.5,  0.5,  1, 1, 0,
        // Right face (magenta)
         0.5, -0.5, -0.5,  1, 0, 1,
         0.5,  0.5, -0.5,  1, 0, 1,
         0.5,  0.5,  0.5,  1, 0, 1,
         0.5, -0.5,  0.5,  1, 0, 1,
        // Left face (cyan)
        -0.5, -0.5, -0.5,  0, 1, 1,
        -0.5, -0.5,  0.5,  0, 1, 1,
        -0.5,  0.5,  0.5,  0, 1, 1,
        -0.5,  0.5, -0.5,  0, 1, 1,
    ]);

    // Index buffer: 6 faces x 2 triangles x 3 indices
    const indices = new Uint16Array([
         0,  1,  2,   0,  2,  3,  // Front
         4,  5,  6,   4,  6,  7,  // Back
         8,  9, 10,   8, 10, 11,  // Top
        12, 13, 14,  12, 14, 15,  // Bottom
        16, 17, 18,  16, 18, 19,  // Right
        20, 21, 22,  20, 22, 23,  // Left
    ]);

    // Create buffers
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const ebo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    return { vbo, ebo, indexCount: indices.length };
}

// ═══════════════════════════════════════════════════════
// Main application
// ═══════════════════════════════════════════════════════

function main() {
    // 1. Initialize WebGL
    const { gl, canvas } = initWebGL('glCanvas');

    // 2. Create shader program
    const program = createProgram(gl, VS_SOURCE, FS_SOURCE);

    // 3. Create geometry
    const cube = createCube(gl);

    // 4. Set up VAO
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    gl.bindBuffer(gl.ARRAY_BUFFER, cube.vbo);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cube.ebo);

    const posLoc = gl.getAttribLocation(program, 'aPosition');
    const colorLoc = gl.getAttribLocation(program, 'aColor');
    const stride = 6 * 4;  // 6 floats per vertex * 4 bytes per float

    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, stride, 0);

    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, stride, 3 * 4);

    gl.bindVertexArray(null);

    // 5. Get uniform location
    const mvpLoc = gl.getUniformLocation(program, 'uMVP');

    // 6. Set up mouse interaction
    const mouse = setupMouseInput(canvas);

    // 7. Enable depth testing (so closer faces occlude farther ones)
    gl.enable(gl.DEPTH_TEST);

    // 8. Render loop
    startRenderLoop(gl, (gl, time, dt) => {
        // Handle canvas resize
        if (canvas.width !== canvas.clientWidth * devicePixelRatio ||
            canvas.height !== canvas.clientHeight * devicePixelRatio) {
            canvas.width = canvas.clientWidth * devicePixelRatio;
            canvas.height = canvas.clientHeight * devicePixelRatio;
            gl.viewport(0, 0, canvas.width, canvas.height);
        }

        // Clear screen
        gl.clearColor(0.1, 0.1, 0.15, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        // Build MVP matrix
        const aspect = canvas.width / canvas.height;
        const proj = Mat4.perspective(60, aspect, 0.1, 100);

        // Camera orbits based on mouse input
        const camDist = mouse.zoom;
        const pitchRad = mouse.rotationX * Math.PI / 180;
        const yawRad = mouse.rotationY * Math.PI / 180;
        const camX = camDist * Math.cos(pitchRad) * Math.sin(yawRad);
        const camY = camDist * Math.sin(pitchRad);
        const camZ = camDist * Math.cos(pitchRad) * Math.cos(yawRad);

        const view = Mat4.lookAt([camX, camY, camZ], [0, 0, 0], [0, 1, 0]);

        // Slowly rotate the cube
        const model = Mat4.rotateY(Mat4.identity(), time * 30);

        const mvp = Mat4.multiply(proj, Mat4.multiply(view, model));

        // Draw
        gl.useProgram(program);
        gl.uniformMatrix4fv(mvpLoc, false, mvp);

        gl.bindVertexArray(vao);
        gl.drawElements(gl.TRIANGLES, cube.indexCount, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);
    });
}

// Start when the page loads
window.addEventListener('load', main);
```

---

## 10. WebGL 상태와 모범 사례

### 10.1 상태 머신(State Machine)

WebGL은 **상태 머신**입니다: 전역 상태(어떤 버퍼가 바인딩되어 있는지, 어떤 프로그램이 활성화되어 있는지, 깊이 테스트가 켜져 있는지)를 구성하고, 현재 상태를 사용하는 드로우 명령을 실행합니다.

```javascript
// Common state settings
gl.enable(gl.DEPTH_TEST);           // Enable depth testing
gl.enable(gl.CULL_FACE);            // Skip back-facing triangles
gl.cullFace(gl.BACK);               // Define "back" as clockwise
gl.frontFace(gl.CCW);               // Counter-clockwise = front

gl.enable(gl.BLEND);                // Enable alpha blending
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

gl.viewport(0, 0, canvas.width, canvas.height);  // Drawing area
```

### 10.2 흔한 함정

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 아무것도 렌더링되지 않음 | `gl.viewport` 누락 또는 캔버스 크기 = 0 | 뷰포트를 캔버스 크기로 설정 |
| 전부 검정 | 프래그먼트 셰이더가 검정을 출력하거나 유니폼이 설정되지 않음 | 유니폼 위치와 값 확인 |
| 깜박임 | 깊이 테스트 없음 또는 잘못된 깊이 함수 | `gl.DEPTH_TEST` 활성화 |
| 뒷면이 보임 | 컬링(culling) 비활성화 또는 잘못된 와인딩(winding) 순서 | `gl.enable(gl.CULL_FACE)` |
| 흐릿한 렌더링 | 캔버스 해상도 < 디스플레이 해상도 | `devicePixelRatio` 곱하기 |
| 어트리뷰트 누락 | 어트리뷰트 위치 = -1 (최적화로 제거됨) | 셰이더 컴파일러가 사용되지 않는 어트리뷰트를 제거함 |

### 10.3 디버깅

```javascript
// Check for WebGL errors after critical operations
function checkError(gl, label) {
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
        const errors = {
            [gl.INVALID_ENUM]: 'INVALID_ENUM',
            [gl.INVALID_VALUE]: 'INVALID_VALUE',
            [gl.INVALID_OPERATION]: 'INVALID_OPERATION',
            [gl.OUT_OF_MEMORY]: 'OUT_OF_MEMORY',
        };
        console.error(`WebGL error at ${label}: ${errors[error] || error}`);
    }
}
```

---

## 요약

| 개념 | WebGL API | 목적 |
|------|-----------|------|
| **컨텍스트(Context)** | `canvas.getContext('webgl2')` | GPU 접근 |
| **버텍스 버퍼(Vertex Buffer)** | `gl.createBuffer()` + `gl.bufferData()` | GPU에 버텍스 데이터 저장 |
| **인덱스 버퍼(Index Buffer)** | 동일, `ELEMENT_ARRAY_BUFFER` 사용 | 인덱스를 통한 버텍스 재사용 |
| **VAO** | `gl.createVertexArray()` | 어트리뷰트 구성 저장 |
| **셰이더(Shader)** | `gl.createShader()` + `gl.compileShader()` | GPU 프로그램 (GLSL ES) |
| **프로그램(Program)** | `gl.createProgram()` + `gl.linkProgram()` | 버텍스 + 프래그먼트 셰이더 링크 |
| **어트리뷰트(Attribute)** | `gl.vertexAttribPointer()` | 버텍스당 데이터 입력 |
| **유니폼(Uniform)** | `gl.uniformMatrix4fv()` 등 | 상수 데이터 입력 |
| **드로우(Draw)** | `gl.drawArrays()` / `gl.drawElements()` | 파이프라인 실행 |

**핵심 정리**:
- WebGL은 `<canvas>` 요소를 통해 웹 브라우저에서 GPU 가속 렌더링을 제공한다
- 1강의 렌더링 파이프라인이 직접 구현된다: 버퍼는 버텍스 데이터를 담고, 셰이더가 처리하며, 드로우 콜이 래스터화를 트리거한다
- 셰이더는 GLSL ES 소스 코드로부터 런타임에 컴파일되고 링크된다
- 유니폼은 드로우 콜당 상수(행렬, 광원 데이터)를 제공하고, 어트리뷰트는 버텍스당 데이터를 제공한다
- WebGL은 상태 머신이다: 상태를 설정하고 드로우; VAO는 상태를 효율적으로 관리하는 데 도움을 준다
- 행렬 연산은 직접 구현하거나 가져와야 한다(WebGL을 위한 열 우선 순서(column-major order))

---

## 연습 문제

1. **Hello Triangle**: 이 강의의 코드를 사용해 단색 삼각형을 렌더링하세요. 버텍스 위치와 색상을 수정하여 다양한 도형을 만들어보세요.

2. **애니메이션 회전**: 삼각형을 z축 주위로 계속 회전시키세요. 버텍스 셰이더에서 `uniform float uTime`을 사용해 회전을 계산해보세요. 버텍스 셰이더에서 회전하는 것과 유니폼 행렬을 통해 회전하는 것의 차이는 무엇인가요?

3. **텍스처 쿼드(Textured Quad)**: 텍스처가 입혀진 쿼드(quad)를 렌더링하도록 애플리케이션을 확장하세요. 이미지를 WebGL 텍스처로 로드하고 UV 좌표를 사용해 프래그먼트 셰이더에서 샘플링하세요.

4. **인터랙티브 카메라**: WASD 키(이동)와 마우스 드래그(회전)로 제어할 수 있는 카메라를 구현하세요. 행렬 라이브러리의 lookAt 행렬을 사용하세요.

5. **다중 객체**: 각기 다른 위치에 세 개의 큐브를 렌더링하되, 각각 자신의 모델 행렬을 갖게 하세요. 인스턴싱(instancing)을 사용한 단일 드로우 콜과 서로 다른 유니폼 행렬을 사용한 세 번의 개별 드로우 콜을 비교해보세요.

6. **성능 측정**: 애플리케이션에 FPS 카운터를 추가하세요. 렌더링하는 삼각형 수를 늘려가며(100, 1000, 10000, 100000) FPS 변화를 관찰하고 병목 지점을 찾아보세요.

---

## 더 읽을거리

1. [WebGL2 Fundamentals](https://webgl2fundamentals.org/) -- 포괄적이고 초보자 친화적인 튜토리얼 시리즈
2. [Learn OpenGL](https://learnopengl.com/) -- WebGL에 직접 적용할 수 있는 OpenGL 개념
3. [MDN WebGL Tutorial](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial) -- Mozilla 공식 WebGL 가이드
4. [gl-matrix](https://glmatrix.net/) -- WebGL을 위한 표준 JavaScript 행렬 연산 라이브러리
5. [Khronos WebGL2 Specification](https://www.khronos.org/registry/webgl/specs/latest/2.0/) -- 공식 레퍼런스
