# 07. WebGL Fundamentals

[&larr; Previous: Texture Mapping](06_Texture_Mapping.md) | [Next: Shader Programming (GLSL) &rarr;](08_Shader_Programming_GLSL.md)

---

## Learning Objectives

1. Set up a WebGL rendering context in an HTML canvas
2. Create and bind vertex buffers and index buffers
3. Write basic vertex and fragment shaders in GLSL ES
4. Understand the WebGL draw call pipeline: buffers, shaders, and uniforms
5. Use `drawArrays` and `drawElements` to render geometry
6. Pass data to shaders via uniforms and attributes
7. Handle basic user input (mouse and keyboard) for interactive rendering
8. Build a complete, interactive WebGL application from scratch

---

## Why This Matters

WebGL brings the power of GPU-accelerated 3D graphics to every web browser -- no plugins, no downloads, no platform-specific code. The concepts from Lessons 01-06 (pipeline, transformations, rasterization, shading, textures) all come alive when you implement them in WebGL. Unlike our Python software rasterizer, WebGL leverages actual GPU hardware, enabling real-time rendering of complex scenes at 60+ FPS. WebGL is the foundation for 3D web experiences: data visualizations (Three.js, Deck.gl), games (PlayCanvas, Babylon.js), virtual try-ons, 3D product configurators, and creative coding. Understanding WebGL's raw API -- before reaching for a library -- gives you deep insight into how all 3D rendering works in practice.

> **Note**: This lesson uses JavaScript and HTML instead of Python. The concepts map directly to the pipeline we have studied, but the language and API are different.

---

## 1. WebGL Architecture

WebGL is a JavaScript API that provides access to the GPU through the HTML5 `<canvas>` element. It is based on OpenGL ES 2.0 (WebGL 1) and OpenGL ES 3.0 (WebGL 2).

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

## 2. Setting Up WebGL

### 2.1 HTML Structure

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

### 2.2 Getting the WebGL Context

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

### 2.3 The Rendering Loop

WebGL rendering follows the same structure every frame:

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

## 3. Buffers: Sending Data to the GPU

### 3.1 Vertex Buffers

A **vertex buffer** (VBO -- Vertex Buffer Object) stores per-vertex data (positions, colors, normals, UVs) in GPU memory.

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

### 3.2 Index Buffers

An **index buffer** (EBO/IBO) specifies which vertices form each triangle, enabling vertex reuse:

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

### 3.3 Vertex Array Objects (VAO)

A **VAO** captures the buffer binding and attribute configuration state, so you do not have to reconfigure it every frame:

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

## 4. Shaders: Programs Running on the GPU

### 4.1 Shader Compilation

WebGL shaders are written in **GLSL ES** (GL Shading Language for Embedded Systems) and compiled at runtime:

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

### 4.2 A Basic Vertex Shader

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

### 4.3 A Basic Fragment Shader

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

## 5. Drawing

### 5.1 drawArrays

`drawArrays` draws primitives using consecutive vertices from the buffer:

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

`drawElements` draws using an index buffer, enabling vertex reuse:

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

## 6. Uniforms and Attributes

### 6.1 Attributes

**Attributes** are per-vertex data: each vertex gets its own value. They are configured using `vertexAttribPointer`:

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

### 6.2 Uniforms

**Uniforms** are constant for all vertices/fragments in a single draw call. They are used for matrices, light positions, time, textures, etc.

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

## 7. Handling User Input

### 7.1 Mouse Input

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

### 7.2 Keyboard Input

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

## 8. Matrix Math in JavaScript

Since JavaScript does not have a built-in matrix library, we need utility functions:

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

## 9. Complete WebGL Application

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

## 10. WebGL State and Best Practices

### 10.1 The State Machine

WebGL is a **state machine**: you configure global state (which buffer is bound, which program is active, is depth testing on) and then issue draw commands that use the current state.

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

### 10.2 Common Pitfalls

| Problem | Cause | Solution |
|---------|-------|----------|
| Nothing renders | Missing `gl.viewport` or canvas size = 0 | Set viewport to canvas dimensions |
| All black | Fragment shader outputs black or uniforms not set | Check uniform locations and values |
| Flickering | No depth test or wrong depth function | Enable `gl.DEPTH_TEST` |
| Back faces visible | Culling not enabled or wrong winding | `gl.enable(gl.CULL_FACE)` |
| Blurry rendering | Canvas resolution < display resolution | Multiply by `devicePixelRatio` |
| Missing attributes | Attribute location = -1 (optimized out) | Shader compiler removes unused attributes |

### 10.3 Debugging

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

## Summary

| Concept | WebGL API | Purpose |
|---------|-----------|---------|
| **Context** | `canvas.getContext('webgl2')` | Access to GPU |
| **Vertex Buffer** | `gl.createBuffer()` + `gl.bufferData()` | Store vertex data on GPU |
| **Index Buffer** | Same, with `ELEMENT_ARRAY_BUFFER` | Vertex reuse via indices |
| **VAO** | `gl.createVertexArray()` | Save attribute configuration |
| **Shader** | `gl.createShader()` + `gl.compileShader()` | GPU programs (GLSL ES) |
| **Program** | `gl.createProgram()` + `gl.linkProgram()` | Link vertex + fragment shaders |
| **Attribute** | `gl.vertexAttribPointer()` | Per-vertex data input |
| **Uniform** | `gl.uniformMatrix4fv()` etc. | Constant data input |
| **Draw** | `gl.drawArrays()` / `gl.drawElements()` | Execute the pipeline |

**Key takeaways**:
- WebGL brings GPU-accelerated rendering to web browsers via the `<canvas>` element
- The rendering pipeline from Lesson 01 is directly implemented: buffers hold vertex data, shaders process it, and draw calls trigger rasterization
- Shaders are compiled and linked at runtime from GLSL ES source code
- Uniforms provide per-draw-call constants (matrices, light data); attributes provide per-vertex data
- WebGL is a state machine: set state, then draw; VAOs help manage state efficiently
- Matrix math must be implemented or imported (column-major order for WebGL)

---

## Exercises

1. **Hello Triangle**: Using the code from this lesson, render a single colored triangle. Modify the vertex positions and colors to create different shapes.

2. **Animated Rotation**: Make the triangle rotate continuously around the z-axis. Use a `uniform float uTime` in the vertex shader to compute the rotation. What happens if you rotate in the vertex shader vs. via a uniform matrix?

3. **Textured Quad**: Extend the application to render a textured quad. Load an image as a WebGL texture and sample it in the fragment shader using UV coordinates.

4. **Interactive Camera**: Implement a camera that can be controlled with WASD keys (translation) and mouse drag (rotation). Use the lookAt matrix from the matrix library.

5. **Multiple Objects**: Render three cubes at different positions, each with its own model matrix. Use a single draw call with instancing, or three separate draw calls with different uniform matrices. Compare the approaches.

6. **Performance Measurement**: Add an FPS counter to the application. Experiment with rendering increasing numbers of triangles (100, 1000, 10000, 100000) and observe how FPS changes. Identify the bottleneck.

---

## Further Reading

1. [WebGL2 Fundamentals](https://webgl2fundamentals.org/) -- Comprehensive, beginner-friendly tutorial series
2. [Learn OpenGL](https://learnopengl.com/) -- OpenGL concepts that apply directly to WebGL
3. [MDN WebGL Tutorial](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial) -- Mozilla's official WebGL guide
4. [gl-matrix](https://glmatrix.net/) -- The standard JavaScript matrix math library for WebGL
5. [Khronos WebGL2 Specification](https://www.khronos.org/registry/webgl/specs/latest/2.0/) -- Official reference
