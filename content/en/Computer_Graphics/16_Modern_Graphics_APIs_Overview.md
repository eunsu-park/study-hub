# 16. Modern Graphics APIs Overview

[← Previous: Real-Time Rendering Techniques](15_Real_Time_Rendering_Techniques.md) | [Next: Overview →](00_Overview.md)

---

## Learning Objectives

1. Trace the evolution of graphics APIs from immediate mode to explicit low-level control
2. Understand Vulkan's architecture: instances, devices, command buffers, render passes, and pipelines
3. Describe Metal's design philosophy and its similarities/differences with Vulkan
4. Recognize DirectX 12's position in the ecosystem and its key concepts
5. Explain WebGPU as the web-native explicit graphics API
6. Understand GPU synchronization primitives: fences, semaphores, and barriers
7. Describe render graphs as a frame-level abstraction for resource management
8. Recognize the trade-offs of explicit resource management and performance considerations

---

## Why This Matters

The transition from OpenGL and DirectX 11 to Vulkan, Metal, and DirectX 12 is the most significant shift in graphics programming in two decades. The old APIs made the GPU driver do enormous amounts of work behind the scenes -- managing memory, tracking resource states, recompiling shaders, and synchronizing CPU-GPU work. This "helpful" driver behavior was unpredictable, hard to optimize, and fundamentally limited multi-threaded rendering.

Modern APIs hand this control to the application. The result is dramatically better performance and predictability, but at the cost of vastly more complex code. Understanding these APIs is essential for anyone building high-performance graphics engines, and even if you use a higher-level engine (Unreal, Unity), knowing what happens under the hood helps you write faster, more efficient rendering code.

---

## 1. Evolution of Graphics APIs

### 1.1 Immediate Mode (OpenGL 1.x, 1992-2003)

The earliest approach: draw commands are executed immediately.

```c
// OpenGL 1.x: immediate mode (deprecated)
glBegin(GL_TRIANGLES);
    glColor3f(1, 0, 0);    glVertex3f(0, 1, 0);
    glColor3f(0, 1, 0);    glVertex3f(-1, -1, 0);
    glColor3f(0, 0, 1);    glVertex3f(1, -1, 0);
glEnd();
```

**Characteristics**: Simple to use, but the driver must validate state on every call. No batching, terrible for performance at scale. One vertex at a time crosses the CPU-GPU boundary.

### 1.2 Retained Mode / Stateful (OpenGL 2.x-4.x, DirectX 9-11, 2003-2015)

Introduced vertex buffers, shader objects, and state management:

```c
// OpenGL 3.3+: vertex buffer + shader pipeline
glBindVertexArray(vao);
glUseProgram(shader);
glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &mvp[0][0]);
glDrawArrays(GL_TRIANGLES, 0, vertex_count);
```

**Characteristics**: GPU work is batched (draw calls process many vertices). Driver manages memory, state transitions, and synchronization. Multi-threaded rendering is limited because the GL context is tied to one thread.

**The driver problem**: The driver does implicit state tracking, resource hazard detection, and lazy compilation. This is convenient but unpredictable -- a single state change can trigger a full shader recompile, causing frame hitches.

### 1.3 Explicit / Low-Level (Vulkan, Metal, DX12, 2014-present)

The application takes full control:

- **Memory management**: Application allocates GPU memory explicitly
- **Synchronization**: Application inserts barriers and fences
- **Command recording**: Application builds command buffers, potentially on multiple threads
- **Pipeline state**: Pre-compiled, immutable pipeline state objects

**Result**: 10-100x reduction in driver overhead, fully multi-threaded command recording, predictable performance. But the application must handle what the driver used to do -- a significant increase in complexity.

### 1.4 Timeline

```
1992  OpenGL 1.0          Immediate mode
1997  DirectX 6            Fixed-function pipeline
2004  OpenGL 2.0 / DX9    Programmable shaders
2009  OpenGL 3.3 / DX11   Modern shader model
2014  Metal                Apple's explicit API
2015  DirectX 12           Microsoft's explicit API
2016  Vulkan               Cross-platform explicit API (Khronos)
2023  WebGPU               Web-native explicit API (W3C)
```

---

## 2. Vulkan

### 2.1 Overview

**Vulkan** is a cross-platform, low-overhead graphics and compute API developed by the Khronos Group (the same organization behind OpenGL). It runs on Windows, Linux, Android, macOS/iOS (via MoltenVK translation layer), and Nintendo Switch.

### 2.2 Core Architecture

Vulkan has a layered architecture with explicit object management:

```
Application
    │
    ▼
┌─────────┐
│ Instance │  ← Entry point; enumerates physical devices
└────┬────┘
     ▼
┌──────────────┐
│Physical Device│  ← GPU hardware (query capabilities, memory)
└──────┬───────┘
       ▼
┌──────────────┐
│Logical Device │  ← Application's interface to the GPU
│  + Queues     │     Multiple queues: graphics, compute, transfer
└──────┬───────┘
       │
    ┌──┴──────────────────────────┐
    │                              │
    ▼                              ▼
┌──────────┐               ┌───────────┐
│ Command   │               │ Pipeline   │
│ Buffers   │               │ (shaders + │
│           │               │  state)    │
└──────────┘               └───────────┘
```

### 2.3 Key Vulkan Objects

**VkInstance**: The entry point. Created once per application. Used to enumerate GPUs and create the logical device.

**VkPhysicalDevice**: Represents a GPU. Query for memory types, queue families, format support, and limits.

**VkDevice**: The logical device -- the application's handle to the GPU. All resource creation goes through it.

**VkQueue**: A submission endpoint. GPUs expose multiple queue families:
- **Graphics queue**: Draw commands + compute
- **Compute queue**: Compute-only (can run async with graphics)
- **Transfer queue**: Memory copies (DMA engine)

**VkCommandBuffer**: A recorded list of GPU commands. Created from a **VkCommandPool** (one pool per thread). Key design:
- Commands are recorded on the CPU (can be multi-threaded)
- Submitted to a queue for execution
- Can be reused across frames

```
// Pseudocode: Vulkan command recording
vkBeginCommandBuffer(cmd);
    vkCmdBeginRenderPass(cmd, ...);
        vkCmdBindPipeline(cmd, graphicsPipeline);
        vkCmdBindDescriptorSets(cmd, ...);
        vkCmdBindVertexBuffers(cmd, ...);
        vkCmdDraw(cmd, vertexCount, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
vkEndCommandBuffer(cmd);

// Submit to GPU
vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence);
```

**VkRenderPass**: Describes the sequence of rendering operations (attachments, subpasses, dependencies). Allows the driver to optimize memory layout for tile-based GPUs (mobile).

**VkPipeline**: An immutable object containing compiled shaders, vertex input layout, rasterization state, blending state, and more. Pre-compiling pipelines eliminates the runtime shader compilation hitches that plagued OpenGL.

**VkDescriptorSet**: Binds resources (buffers, textures, samplers) to shader bindings. Think of it as a "resource table" that the shader reads from.

### 2.4 Vulkan Rendering Flow

```
1. Create Instance, Device, Queues
2. Create Swapchain (display surface)
3. Create Render Pass (attachment descriptions)
4. Create Pipeline (shaders + state)
5. Allocate Command Buffers

Per frame:
6. Acquire swapchain image (vkAcquireNextImageKHR)
7. Record command buffer:
   - Begin render pass
   - Bind pipeline
   - Bind descriptor sets (textures, uniforms)
   - Bind vertex/index buffers
   - Draw
   - End render pass
8. Submit command buffer to queue
9. Present swapchain image (vkQueuePresentKHR)
10. Synchronize (fences, semaphores)
```

### 2.5 Vulkan's Verbosity

A minimal Vulkan "hello triangle" is 800-1200 lines of code (compared to ~50 lines in OpenGL). This verbosity is intentional -- every decision that was implicit in OpenGL is now explicit. However, libraries like **vk-bootstrap** and **VMA (Vulkan Memory Allocator)** reduce boilerplate significantly.

---

## 3. Metal

### 3.1 Overview

**Metal** is Apple's graphics and compute API, introduced in 2014. It runs on iOS, macOS, iPadOS, and Apple TV. Metal replaced OpenGL ES (deprecated on Apple platforms since 2018).

### 3.2 Key Concepts

Metal's design is similar to Vulkan but with Apple-specific simplifications:

| Vulkan | Metal | Notes |
|--------|-------|-------|
| VkInstance | — | Not needed; Metal devices are queried directly |
| VkPhysicalDevice | MTLDevice | `MTLCreateSystemDefaultDevice()` |
| VkDevice | MTLDevice | Physical and logical device are merged |
| VkQueue | MTLCommandQueue | One per device (or multiple for async) |
| VkCommandBuffer | MTLCommandBuffer | Created from MTLCommandQueue |
| VkRenderPass | MTLRenderPassDescriptor | Configured per draw |
| VkPipeline | MTLRenderPipelineState | Compiled pipeline state |
| VkDescriptorSet | Argument Buffers | Or direct buffer/texture binds |

### 3.3 Metal Shading Language (MSL)

Metal uses its own C++-based shading language:

```metal
// Metal vertex shader
vertex VertexOut vertex_main(
    const device VertexIn* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]])
{
    VertexOut out;
    float4 pos = float4(vertices[vid].position, 1.0);
    out.position = uniforms.mvp * pos;
    out.color = vertices[vid].color;
    return out;
}

// Metal fragment shader
fragment float4 fragment_main(VertexOut in [[stage_in]])
{
    return in.color;
}
```

### 3.4 Metal Advantages

- **Unified memory** on Apple Silicon: CPU and GPU share the same physical memory, eliminating explicit copies
- **Tile shading**: On Apple GPUs (tile-based), Metal exposes tile memory directly for efficient deferred rendering
- **Mesh shaders**: Object and mesh shaders for GPU-driven geometry processing
- **MetalFX**: Temporal upscaling (Apple's equivalent of DLSS/FSR)
- **Ray tracing**: Hardware-accelerated ray tracing on Apple Silicon (M3+)

### 3.5 Metal vs. Vulkan

| Aspect | Metal | Vulkan |
|--------|-------|--------|
| Platform | Apple only | Cross-platform |
| Verbosity | Moderate | Very verbose |
| Render pass | Per-draw descriptor | Pre-defined structure |
| Memory | Unified (Apple Silicon) | Explicit allocation |
| Shader language | MSL (C++-based) | SPIR-V (compiled from GLSL/HLSL) |
| Validation | Metal validation layer | Vulkan validation layers |
| Maturity | 10+ years | 10+ years |

---

## 4. DirectX 12

### 4.1 Overview

**DirectX 12** (DX12) is Microsoft's low-level graphics API for Windows 10+ and Xbox. It is the successor to DirectX 11 and shares the same explicit design philosophy as Vulkan and Metal.

### 4.2 Key Concepts

| Vulkan | DirectX 12 | Notes |
|--------|-----------|-------|
| VkDevice | ID3D12Device | GPU interface |
| VkQueue | ID3D12CommandQueue | Graphics, compute, copy queues |
| VkCommandBuffer | ID3D12GraphicsCommandList | Command recording |
| VkPipeline | ID3D12PipelineState | Pipeline State Object (PSO) |
| VkDescriptorSet | Descriptor Heap/Table | Root signature defines layout |
| VkRenderPass | — | DX12 uses render targets directly |
| SPIR-V | DXIL | Shader intermediate representation |

### 4.3 DX12 Unique Features

- **Root Signature**: Defines the layout of shader resources (CBVs, SRVs, UAVs, samplers). More explicit than Vulkan descriptor set layouts.
- **Descriptor Heaps**: GPU-visible tables of resource descriptors. "Bindless" rendering puts all textures in one heap.
- **Work Graphs** (DX12 Ultimate): GPU-driven work dispatch -- the GPU creates and schedules its own work items.
- **DirectX Raytracing (DXR)**: The first hardware ray tracing API (2018), using BLAS/TLAS acceleration structures.
- **DirectStorage**: Bypass the CPU for loading assets directly from NVMe SSD to GPU memory.

### 4.4 HLSL Shading Language

DX12 uses HLSL (High-Level Shading Language), compiled to DXIL (DirectX Intermediate Language):

```hlsl
// HLSL vertex shader
struct VSInput {
    float3 position : POSITION;
    float3 color : COLOR;
};

struct VSOutput {
    float4 position : SV_POSITION;
    float3 color : COLOR;
};

cbuffer Constants : register(b0) {
    float4x4 mvp;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = mul(mvp, float4(input.position, 1.0));
    output.color = input.color;
    return output;
}
```

---

## 5. WebGPU

### 5.1 Overview

**WebGPU** is a W3C standard for GPU access in web browsers. It replaces WebGL with a modern, explicit API that maps to Vulkan, Metal, or DX12 under the hood.

### 5.2 Design Goals

- **Cross-platform**: Runs on any browser (Chrome, Firefox, Safari)
- **Safe**: Validates all operations to prevent GPU crashes/hangs
- **Modern**: Explicit resource binding, compute shaders, storage buffers
- **Performant**: Much less driver overhead than WebGL

### 5.3 WGSL (WebGPU Shading Language)

```wgsl
// WGSL vertex shader
struct Uniforms {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>
) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.mvp * vec4<f32>(position, 1.0);
    output.color = color;
    return output;
}

@fragment
fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}
```

### 5.4 WebGPU Architecture

```javascript
// WebGPU initialization (JavaScript)
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const context = canvas.getContext('webgpu');

// Create render pipeline
const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
        module: device.createShaderModule({ code: vertexShaderWGSL }),
        entryPoint: 'vs_main',
        buffers: [vertexBufferLayout],
    },
    fragment: {
        module: device.createShaderModule({ code: fragmentShaderWGSL }),
        entryPoint: 'fs_main',
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
    },
});

// Render loop
function render() {
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(3);
    pass.end();
    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(render);
}
```

### 5.5 WebGPU vs. WebGL

| Feature | WebGL 2 | WebGPU |
|---------|---------|--------|
| API model | Stateful (OpenGL ES 3.0) | Explicit (Vulkan-like) |
| Compute shaders | No | Yes |
| Storage buffers | Limited | Yes |
| Multi-threaded | No | Command encoding on workers |
| Shader language | GLSL ES | WGSL |
| Error handling | Silent failures | Validation errors |
| Performance | Driver overhead | Lower overhead |

---

## 6. Synchronization

### 6.1 Why Synchronization Matters

Modern APIs give no implicit ordering guarantees. The application must explicitly specify:
- When the GPU can start reading a resource the CPU wrote
- When one GPU operation must complete before another begins
- When the CPU can read back results from the GPU

### 6.2 Synchronization Primitives

**Fences** (CPU-GPU synchronization):
- The CPU submits work to the GPU and receives a fence
- The CPU can wait on the fence to know when the GPU is done
- Used for frame pacing: wait until frame N-2 is complete before reusing its resources

```
CPU: Submit frame N, get fence_N
     ...
CPU: Wait(fence_N-2)  // Ensure GPU finished frame N-2
CPU: Reuse frame N-2's command buffers and staging memory
```

**Semaphores** (GPU-GPU synchronization):
- Signal a semaphore when one queue operation completes
- Wait on the semaphore before starting another queue operation
- Used between render and present: render signals, present waits

```
GPU: Render pass signals semaphore_render_done
GPU: Present waits on semaphore_render_done
```

**Pipeline Barriers** (within a command buffer):
- Specify memory and execution dependencies between commands
- "Resource X must finish being written before resource X is read"
- Required for image layout transitions (e.g., from render target to shader input)

```
// Vulkan pipeline barrier example (pseudocode)
vkCmdPipelineBarrier(cmd,
    srcStage = COLOR_ATTACHMENT_OUTPUT,  // Wait for writes
    dstStage = FRAGMENT_SHADER,          // Before reads
    imageBarrier = {
        image = colorAttachment,
        oldLayout = COLOR_ATTACHMENT_OPTIMAL,
        newLayout = SHADER_READ_ONLY_OPTIMAL
    }
);
```

### 6.3 The Triple Buffering Pattern

Most applications use 2-3 frames of buffering to keep the GPU fed:

```
Frame N:   CPU record  │  GPU execute
Frame N+1:              │  CPU record   │  GPU execute
Frame N+2:                              │  CPU record   │  GPU execute

Buffers: [A] [B] [C]  -- rotate through three sets of resources
```

Each frame has its own command buffers, uniform buffers, and descriptor sets. Fences ensure we do not overwrite a buffer the GPU is still reading.

---

## 7. Render Graphs

### 7.1 The Problem

A modern frame involves dozens of render passes (shadow maps, G-buffer, SSAO, lighting, bloom, TAA, ...) with complex resource dependencies. Manually managing resource lifetimes, barriers, and memory aliasing is error-prone and hard to optimize.

### 7.2 Frame Graph Abstraction

A **render graph** (also called frame graph) is a declarative description of the rendering frame:

1. **Declare passes**: Each pass specifies which resources it reads, writes, and creates
2. **Build dependency graph**: The system determines execution order from resource dependencies
3. **Cull unused passes**: If a pass's output is never read, it is removed
4. **Schedule**: Determine optimal execution order, insert barriers, alias memory
5. **Execute**: Run passes in order

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Shadow   │───▶│ G-Buffer │───▶│ Lighting │───▶│  Post    │
│   Map     │    │  Fill    │    │   Pass   │    │ Process  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │              ▲   ▲              ▲
                     │              │   │              │
                     └── depth ────┘   │              │
                                       │              │
                ┌──────────┐           │              │
                │   SSAO   │───────────┘              │
                └──────────┘                          │
                                                      │
                ┌──────────┐                          │
                │  Bloom   │──────────────────────────┘
                └──────────┘
```

### 7.3 Benefits

- **Automatic barrier insertion**: The system knows exactly when each resource transitions and inserts minimal barriers
- **Memory aliasing**: Transient resources that do not overlap in time can share the same GPU memory (e.g., the shadow map and bloom buffer)
- **Pass culling**: Debug visualization passes that are not displayed are automatically removed
- **Multi-queue scheduling**: Async compute passes can be overlapped with graphics work

### 7.4 Implementation Sketch

```python
from collections import defaultdict
from typing import Dict, Set, List

class RenderPass:
    """A render pass declaration in the frame graph."""

    def __init__(self, name):
        self.name = name
        self.reads: Set[str] = set()     # Resources this pass reads
        self.writes: Set[str] = set()    # Resources this pass writes/creates
        self.execute_fn = None           # Callback to execute the pass

    def read(self, resource_name: str):
        self.reads.add(resource_name)
        return self

    def write(self, resource_name: str):
        self.writes.add(resource_name)
        return self

    def set_execute(self, fn):
        self.execute_fn = fn
        return self


class RenderGraph:
    """
    Frame graph system: declare passes and resources,
    then compile to an optimal execution order.
    """

    def __init__(self):
        self.passes: List[RenderPass] = []
        self.final_outputs: Set[str] = set()  # Resources that must be produced

    def add_pass(self, name: str) -> RenderPass:
        p = RenderPass(name)
        self.passes.append(p)
        return p

    def set_output(self, resource_name: str):
        """Mark a resource as a final output (prevents culling)."""
        self.final_outputs.add(resource_name)

    def compile(self) -> List[RenderPass]:
        """
        Build dependency graph, cull unused passes, and topological sort.
        Returns passes in execution order.
        """
        # Build resource -> writer mapping
        writer_of: Dict[str, RenderPass] = {}
        for p in self.passes:
            for w in p.writes:
                writer_of[w] = p

        # Mark passes that contribute to final output (reverse DFS)
        needed: Set[str] = set()
        stack = []

        # Start from passes that produce final outputs
        for p in self.passes:
            if p.writes & self.final_outputs:
                stack.append(p)

        while stack:
            p = stack.pop()
            if p.name in needed:
                continue
            needed.add(p.name)
            # This pass's reads require other passes' writes
            for r in p.reads:
                if r in writer_of and writer_of[r].name not in needed:
                    stack.append(writer_of[r])

        # Cull unnecessary passes
        active_passes = [p for p in self.passes if p.name in needed]
        culled = len(self.passes) - len(active_passes)
        if culled > 0:
            print(f"  Culled {culled} unused passes")

        # Topological sort based on read/write dependencies
        # Why topological sort: ensures each pass runs after its dependencies
        in_degree: Dict[str, int] = {p.name: 0 for p in active_passes}
        edges: Dict[str, List[str]] = defaultdict(list)

        for p in active_passes:
            for r in p.reads:
                if r in writer_of and writer_of[r].name in needed:
                    dep = writer_of[r].name
                    if dep != p.name:
                        edges[dep].append(p.name)
                        in_degree[p.name] += 1

        # Kahn's algorithm
        queue = [p for p in active_passes if in_degree[p.name] == 0]
        sorted_passes = []

        while queue:
            p = queue.pop(0)
            sorted_passes.append(p)
            for neighbor_name in edges[p.name]:
                in_degree[neighbor_name] -= 1
                if in_degree[neighbor_name] == 0:
                    neighbor = next(x for x in active_passes if x.name == neighbor_name)
                    queue.append(neighbor)

        return sorted_passes

    def execute(self):
        """Compile and execute all needed passes in order."""
        ordered = self.compile()
        print(f"\n  Execution order ({len(ordered)} passes):")

        # Determine barriers needed between passes
        last_writer: Dict[str, str] = {}

        for p in ordered:
            # Check if any reads require a barrier (resource was written by a previous pass)
            barriers = []
            for r in p.reads:
                if r in last_writer:
                    barriers.append(f"{r} (written by {last_writer[r]})")

            if barriers:
                print(f"    BARRIER: {', '.join(barriers)}")

            print(f"    Execute: {p.name}")
            print(f"      reads:  {p.reads or '{}'}")
            print(f"      writes: {p.writes or '{}'}")

            if p.execute_fn:
                p.execute_fn()

            for w in p.writes:
                last_writer[w] = p.name


# --- Demo: Build a frame graph ---

graph = RenderGraph()

# Shadow pass: writes shadow_map
shadow = graph.add_pass("ShadowMap")
shadow.write("shadow_map")

# G-buffer pass: writes albedo, normals, depth
gbuffer = graph.add_pass("GBuffer")
gbuffer.write("gbedo_tex").write("normal_tex").write("depth_tex")

# SSAO: reads normals + depth, writes ao_tex
ssao = graph.add_pass("SSAO")
ssao.read("normal_tex").read("depth_tex").write("ao_tex")

# Lighting: reads everything, writes hdr_color
lighting = graph.add_pass("Lighting")
lighting.read("gbedo_tex").read("normal_tex").read("depth_tex")
lighting.read("shadow_map").read("ao_tex").write("hdr_color")

# Bloom: reads hdr_color, writes bloom_tex
bloom = graph.add_pass("Bloom")
bloom.read("hdr_color").write("bloom_tex")

# Tone mapping: reads hdr_color + bloom, writes ldr_color (final output)
tonemap = graph.add_pass("ToneMap")
tonemap.read("hdr_color").read("bloom_tex").write("ldr_color")

# Debug pass (not connected to output -- should be culled)
debug = graph.add_pass("DebugVis")
debug.read("normal_tex").write("debug_output")

# Mark final output
graph.set_output("ldr_color")

# Compile and execute
graph.execute()
```

Output:
```
  Culled 1 unused passes
  Execution order (6 passes):
    Execute: ShadowMap
      reads:  {}
      writes: {'shadow_map'}
    Execute: GBuffer
      reads:  {}
      writes: {'gbedo_tex', 'normal_tex', 'depth_tex'}
    BARRIER: normal_tex (written by GBuffer), depth_tex (written by GBuffer)
    Execute: SSAO
      reads:  {'normal_tex', 'depth_tex'}
      writes: {'ao_tex'}
    BARRIER: gbedo_tex (written by GBuffer), normal_tex (written by GBuffer), ...
    Execute: Lighting
      reads:  {'gbedo_tex', 'normal_tex', 'depth_tex', 'shadow_map', 'ao_tex'}
      writes: {'hdr_color'}
    BARRIER: hdr_color (written by Lighting)
    Execute: Bloom
      reads:  {'hdr_color'}
      writes: {'bloom_tex'}
    BARRIER: hdr_color (written by Lighting), bloom_tex (written by Bloom)
    Execute: ToneMap
      reads:  {'hdr_color', 'bloom_tex'}
      writes: {'ldr_color'}
```

Note that DebugVis was culled because its output (`debug_output`) is not required by any pass that leads to the final output.

---

## 8. Explicit Resource Management

### 8.1 Memory Types

Modern APIs expose multiple GPU memory types:

| Memory Type | Vulkan Name | Characteristics |
|-------------|-------------|-----------------|
| Device-local | DEVICE_LOCAL | Fastest GPU access; not CPU-visible |
| Host-visible | HOST_VISIBLE | CPU can map and write; slower GPU access |
| Host-coherent | HOST_COHERENT | CPU writes are immediately visible to GPU |
| Host-cached | HOST_CACHED | CPU reads are fast (cached); useful for readback |

**Typical pattern**:
1. Create a staging buffer in HOST_VISIBLE memory
2. Copy data from CPU to staging buffer (memory map)
3. Issue a GPU transfer from staging buffer to DEVICE_LOCAL buffer
4. Use the DEVICE_LOCAL buffer for rendering

On unified memory architectures (Apple Silicon, integrated GPUs), device-local and host-visible may be the same physical memory.

### 8.2 Descriptors and Binding

Shaders access resources (buffers, textures) through **descriptors** -- metadata that tells the GPU where to find the resource and how to interpret it.

**Descriptor Sets** (Vulkan) / **Root Signatures + Descriptor Heaps** (DX12):

```
Descriptor Set Layout:
  Binding 0: Uniform buffer (per-frame data)
  Binding 1: Storage buffer (per-object transforms)
  Binding 2: Combined image sampler (diffuse texture)
  Binding 3: Combined image sampler (normal map)
```

**Bindless rendering**: Put all textures in a single large descriptor array. Shaders index into the array using a material ID. This eliminates descriptor set switches between draw calls, dramatically reducing CPU overhead.

### 8.3 Pipeline State Objects (PSOs)

In modern APIs, all rendering state is baked into immutable **Pipeline State Objects**:

```
PSO = {
    Vertex shader
    Fragment shader
    Vertex input layout
    Rasterization state (cull mode, polygon mode)
    Depth/stencil state
    Blend state
    Render target formats
    MSAA state
}
```

Changing **any** of these requires a different PSO. Applications pre-compile all needed PSOs at load time (or cache them on disk) to avoid runtime compilation.

**Pipeline caches**: Store compiled pipeline bytecode to avoid recompiling shaders on subsequent application launches.

---

## 9. Performance Considerations

### 9.1 CPU-Side Optimization

| Technique | Benefit |
|-----------|---------|
| Multi-threaded command recording | Scale CPU work across cores |
| Indirect drawing | GPU fills draw parameters; CPU issues one call |
| Bindless resources | Eliminate descriptor set changes |
| Pipeline state sorting | Minimize PSO switches |
| Persistent mapping | Avoid repeated map/unmap of staging buffers |

### 9.2 GPU-Side Optimization

| Technique | Benefit |
|-----------|---------|
| Async compute | Overlap compute with graphics |
| Render graph optimization | Minimal barriers, memory aliasing |
| Mesh shaders | GPU-driven geometry processing |
| GPU culling | Frustum/occlusion culling on GPU |
| Indirect dispatch | GPU decides how much work to do |

### 9.3 Choosing an API

| Need | Recommended API |
|------|-----------------|
| Cross-platform desktop | Vulkan (or abstraction layer) |
| Apple platforms | Metal |
| Windows games | DirectX 12 (or Vulkan) |
| Web | WebGPU |
| Simplicity/prototyping | OpenGL 4.6 or WebGL 2 |
| Maximum control | Vulkan or DX12 |

### 9.4 Abstraction Layers

Most production engines use an abstraction layer over multiple APIs:

- **bgfx**: C/C++ cross-platform rendering library
- **wgpu** (Rust): WebGPU implementation that runs natively (Vulkan/Metal/DX12)
- **Sokol**: Single-file C libraries for cross-platform graphics
- **SDL_GPU**: SDL's new GPU abstraction (2024)
- **Unreal RHI**: Unreal Engine's Rendering Hardware Interface

These abstractions sacrifice some API-specific optimizations but provide portability across all platforms.

---

## 10. The Future

### 10.1 GPU-Driven Rendering

Traditional: CPU decides what to draw; GPU executes.
Future: GPU scans the scene, culls, sorts, and generates draw calls itself. The CPU submits a single indirect dispatch.

**Nanite** (Unreal 5), **mesh shaders**, and **work graphs** (DX12) are steps toward fully GPU-driven pipelines.

### 10.2 Neural Rendering

ML models running on the GPU replace parts of the traditional pipeline:
- **DLSS/FSR**: Temporal upscaling via neural networks
- **Neural radiance caching**: Replace probe-based GI with neural networks
- **Gaussian splatting**: Novel view synthesis from point clouds

### 10.3 Convergence

The major APIs (Vulkan, Metal, DX12, WebGPU) are converging on similar concepts: explicit memory, command buffers, pipeline states, compute shaders, and ray tracing. Learning one deeply makes the others much easier to understand.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| API evolution | Immediate mode → stateful → explicit; each generation trades ease for control |
| Vulkan | Cross-platform explicit API; verbose but maximum control and portability |
| Metal | Apple's explicit API; unified memory on Apple Silicon; MSL shading language |
| DirectX 12 | Microsoft's explicit API; DXR ray tracing; work graphs |
| WebGPU | Web-native explicit API; WGSL shading language; safe and cross-platform |
| Fences | CPU-GPU sync: CPU waits for GPU to finish a submission |
| Semaphores | GPU-GPU sync: one queue signals, another waits |
| Barriers | Within a command buffer: resource layout transitions and hazard prevention |
| Render graph | Declarative frame description; automatic barrier insertion, culling, aliasing |
| PSO | Immutable compiled pipeline state; eliminates runtime shader compilation |
| Bindless | All resources in one descriptor array; index by material ID; minimal CPU overhead |
| Triple buffering | 3 sets of per-frame resources; fences prevent overwriting in-flight data |

## Exercises

1. **API comparison**: Choose a simple rendering task (draw a textured, lit cube). Write pseudocode for OpenGL 3.3, Vulkan, and WebGPU. Count the number of API calls and objects created in each.

2. **Render graph**: Extend the render graph implementation to support **memory aliasing**: resources that do not overlap in the execution timeline should share the same "memory allocation" (tracked as a simple counter). Print the total memory with and without aliasing.

3. **Synchronization design**: Design the fence and semaphore setup for a triple-buffered Vulkan application with async compute. Draw a timeline showing which fences and semaphores are signaled/waited on each frame.

4. **Pipeline state explosion**: A material system supports 3 vertex formats, 2 blend modes, 4 shader variants, and 3 render pass configurations. How many PSOs are needed? How would you use pipeline caches and lazy creation to manage this?

5. **Bindless vs. bound**: Compare the CPU overhead of drawing 1000 objects with (a) per-object descriptor set binds and (b) bindless rendering. Estimate the number of API calls for each approach.

6. **WebGPU compute**: Write a WebGPU compute shader (in WGSL) that doubles every element in a storage buffer. Write the JavaScript code to create the device, buffer, pipeline, and dispatch the shader.

## Further Reading

- Sellers, G. et al. *Vulkan Programming Guide*. Addison-Wesley, 2016. (Official Vulkan tutorial-style book)
- Vulkan Tutorial. https://vulkan-tutorial.com/ (The most popular step-by-step Vulkan guide)
- Apple Metal Documentation. https://developer.apple.com/metal/ (Official Metal programming guide)
- Microsoft DirectX 12 Documentation. https://learn.microsoft.com/en-us/windows/win32/direct3d12/ (DX12 reference)
- WebGPU Specification. W3C, 2024. https://www.w3.org/TR/webgpu/ (Official WebGPU spec)
- O'Donnell, Y. "FrameGraph: Extensible Rendering Architecture in Frostbite." *GDC*, 2017. (Frame graph design from EA's Frostbite engine)
