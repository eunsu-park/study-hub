# 16. 현대 그래픽스 API 개요

[← 이전: 15. 실시간 렌더링 기법](15_Real_Time_Rendering_Techniques.md) | [개요 →](00_Overview.md)

---

## 학습 목표

1. 즉시 모드(Immediate Mode)에서 명시적 저수준(Explicit Low-Level) 제어까지 그래픽스 API의 발전 과정을 추적한다
2. Vulkan의 아키텍처(인스턴스, 디바이스, 커맨드 버퍼, 렌더 패스, 파이프라인)를 이해한다
3. Metal의 설계 철학과 Vulkan과의 유사점/차이점을 설명한다
4. 생태계에서 DirectX 12의 위치와 핵심 개념을 인식한다
5. WebGPU를 웹 네이티브(Web-Native) 명시적 그래픽스 API로서 설명한다
6. GPU 동기화 프리미티브(Synchronization Primitives): 펜스(Fence), 세마포어(Semaphore), 배리어(Barrier)를 이해한다
7. 리소스 관리를 위한 프레임 수준 추상화인 렌더 그래프(Render Graph)를 설명한다
8. 명시적 리소스 관리의 트레이드오프와 성능 고려사항을 인식한다

---

## 왜 중요한가

OpenGL과 DirectX 11에서 Vulkan, Metal, DirectX 12로의 전환은 20년 만에 가장 중요한 그래픽스 프로그래밍의 변화다. 구형 API는 GPU 드라이버가 메모리 관리, 리소스 상태 추적, 셰이더 재컴파일, CPU-GPU 작업 동기화 등 막대한 작업을 내부적으로 처리하게 했다. 이 "도움이 되는" 드라이버 동작은 예측하기 어렵고 최적화가 힘들었으며, 멀티스레드 렌더링을 근본적으로 제한했다.

현대 API는 이 제어권을 애플리케이션에 넘긴다. 그 결과 성능과 예측 가능성이 극적으로 향상되었지만, 코드 복잡성이 대폭 증가했다. 이 API들을 이해하는 것은 고성능 그래픽스 엔진을 구축하는 모든 사람에게 필수적이며, 상위 수준 엔진(Unreal, Unity)을 사용하더라도 내부에서 무슨 일이 일어나는지 알면 더 빠르고 효율적인 렌더링 코드를 작성하는 데 도움이 된다.

---

## 1. 그래픽스 API의 발전

### 1.1 즉시 모드(Immediate Mode) (OpenGL 1.x, 1992-2003)

가장 초기의 방식: 그리기 명령이 즉시 실행된다.

```c
// OpenGL 1.x: immediate mode (deprecated)
glBegin(GL_TRIANGLES);
    glColor3f(1, 0, 0);    glVertex3f(0, 1, 0);
    glColor3f(0, 1, 0);    glVertex3f(-1, -1, 0);
    glColor3f(0, 0, 1);    glVertex3f(1, -1, 0);
glEnd();
```

**특성**: 사용하기 단순하지만, 드라이버가 매 호출마다 상태를 검증해야 한다. 배칭(Batching)이 없어 규모가 커지면 성능이 극히 나쁘다. 한 번에 하나의 버텍스(Vertex)가 CPU-GPU 경계를 넘는다.

### 1.2 보존 모드 / 상태 기반(Retained Mode / Stateful) (OpenGL 2.x-4.x, DirectX 9-11, 2003-2015)

버텍스 버퍼(Vertex Buffer), 셰이더 오브젝트(Shader Object), 상태 관리를 도입했다:

```c
// OpenGL 3.3+: vertex buffer + shader pipeline
glBindVertexArray(vao);
glUseProgram(shader);
glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &mvp[0][0]);
glDrawArrays(GL_TRIANGLES, 0, vertex_count);
```

**특성**: GPU 작업이 배칭된다 (드로우 콜이 많은 버텍스를 처리). 드라이버가 메모리, 상태 전환, 동기화를 관리한다. GL 컨텍스트(Context)가 하나의 스레드에 묶여 있어 멀티스레드 렌더링이 제한된다.

**드라이버 문제**: 드라이버가 암묵적 상태 추적, 리소스 해저드(Hazard) 감지, 지연 컴파일을 수행한다. 편리하지만 예측 불가능 -- 단일 상태 변경이 전체 셰이더 재컴파일을 유발하여 프레임 끊김을 일으킬 수 있다.

### 1.3 명시적 / 저수준(Explicit / Low-Level) (Vulkan, Metal, DX12, 2014-현재)

애플리케이션이 완전한 제어권을 갖는다:

- **메모리 관리**: 애플리케이션이 GPU 메모리를 명시적으로 할당한다
- **동기화**: 애플리케이션이 배리어(Barrier)와 펜스(Fence)를 삽입한다
- **커맨드 레코딩(Command Recording)**: 애플리케이션이 커맨드 버퍼를 구성하며, 잠재적으로 여러 스레드에서 가능하다
- **파이프라인 상태**: 사전 컴파일된 불변(Immutable) 파이프라인 스테이트 오브젝트(Pipeline State Object)

**결과**: 드라이버 오버헤드가 10-100배 감소하고, 완전히 멀티스레드 커맨드 레코딩이 가능하며, 예측 가능한 성능을 제공한다. 하지만 드라이버가 하던 작업을 애플리케이션이 처리해야 한다 -- 복잡성이 크게 증가한다.

### 1.4 타임라인

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

### 2.1 개요

**Vulkan**은 OpenGL을 개발한 단체인 Khronos Group이 개발한 크로스 플랫폼(Cross-Platform), 저오버헤드(Low-Overhead) 그래픽스 및 컴퓨트 API다. Windows, Linux, Android, macOS/iOS (MoltenVK 번역 레이어 경유), Nintendo Switch에서 실행된다.

### 2.2 핵심 아키텍처

Vulkan은 명시적 오브젝트 관리를 갖춘 계층적 아키텍처를 가진다:

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

### 2.3 핵심 Vulkan 오브젝트

**VkInstance**: 진입점이다. 애플리케이션당 한 번 생성된다. GPU를 열거하고 논리 디바이스를 생성하는 데 사용된다.

**VkPhysicalDevice**: GPU를 나타낸다. 메모리 타입, 큐 패밀리(Queue Family), 포맷 지원, 한계치를 조회한다.

**VkDevice**: 논리 디바이스(Logical Device) -- GPU에 대한 애플리케이션의 핸들이다. 모든 리소스 생성이 이를 통해 이루어진다.

**VkQueue**: 제출 엔드포인트(Submission Endpoint)다. GPU는 여러 큐 패밀리를 노출한다:
- **그래픽스 큐(Graphics Queue)**: 드로우 명령 + 컴퓨트
- **컴퓨트 큐(Compute Queue)**: 컴퓨트 전용 (그래픽스와 비동기로 실행 가능)
- **전송 큐(Transfer Queue)**: 메모리 복사 (DMA 엔진)

**VkCommandBuffer**: GPU 명령의 레코딩된 목록이다. **VkCommandPool** (스레드당 풀 하나)에서 생성된다. 주요 설계 원칙:
- 명령은 CPU에서 레코딩된다 (멀티스레드 가능)
- 실행을 위해 큐에 제출된다
- 프레임 간에 재사용 가능하다

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

**VkRenderPass**: 렌더링 작업의 순서(어태치먼트, 서브패스, 의존성)를 기술한다. 타일 기반 GPU(모바일)에서 메모리 레이아웃을 최적화할 수 있도록 드라이버를 지원한다.

**VkPipeline**: 컴파일된 셰이더, 버텍스 입력 레이아웃, 래스터화(Rasterization) 상태, 블렌딩(Blending) 상태 등을 포함하는 불변 오브젝트다. 파이프라인 사전 컴파일로 OpenGL에서 만연했던 런타임 셰이더 컴파일 끊김을 제거한다.

**VkDescriptorSet**: 리소스(버퍼, 텍스처, 샘플러)를 셰이더 바인딩에 연결한다. 셰이더가 읽는 "리소스 테이블"로 생각할 수 있다.

### 2.4 Vulkan 렌더링 플로우

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

### 2.5 Vulkan의 장황함

최소한의 Vulkan "헬로 트라이앵글(Hello Triangle)"은 약 800-1200 줄의 코드다 (OpenGL의 약 50줄과 비교). 이 장황함은 의도적인 것 -- OpenGL에서 암묵적이었던 모든 결정이 이제 명시적이다. 하지만 **vk-bootstrap**이나 **VMA (Vulkan Memory Allocator)** 같은 라이브러리가 보일러플레이트(Boilerplate)를 크게 줄여준다.

---

## 3. Metal

### 3.1 개요

**Metal**은 2014년에 도입된 Apple의 그래픽스 및 컴퓨트 API다. iOS, macOS, iPadOS, Apple TV에서 실행된다. Metal은 2018년부터 Apple 플랫폼에서 deprecated된 OpenGL ES를 대체했다.

### 3.2 핵심 개념

Metal의 설계는 Vulkan과 유사하지만 Apple 특유의 단순화가 적용되어 있다:

| Vulkan | Metal | 비고 |
|--------|-------|------|
| VkInstance | — | 불필요; Metal 디바이스를 직접 조회 |
| VkPhysicalDevice | MTLDevice | `MTLCreateSystemDefaultDevice()` |
| VkDevice | MTLDevice | 물리/논리 디바이스가 통합됨 |
| VkQueue | MTLCommandQueue | 디바이스당 하나 (또는 비동기용 여러 개) |
| VkCommandBuffer | MTLCommandBuffer | MTLCommandQueue에서 생성 |
| VkRenderPass | MTLRenderPassDescriptor | 드로우별로 구성 |
| VkPipeline | MTLRenderPipelineState | 컴파일된 파이프라인 상태 |
| VkDescriptorSet | Argument Buffers | 또는 직접 버퍼/텍스처 바인드 |

### 3.3 Metal 셰이딩 언어(MSL, Metal Shading Language)

Metal은 C++ 기반의 자체 셰이딩 언어를 사용한다:

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

### 3.4 Metal의 장점

- **통합 메모리(Unified Memory)**: Apple Silicon에서 CPU와 GPU가 같은 물리 메모리를 공유하여 명시적 복사가 불필요하다
- **타일 셰이딩(Tile Shading)**: Apple GPU(타일 기반)에서 Metal이 타일 메모리를 직접 노출하여 효율적인 디퍼드 렌더링이 가능하다
- **메시 셰이더(Mesh Shaders)**: GPU 주도 지오메트리 처리를 위한 오브젝트 및 메시 셰이더
- **MetalFX**: 시간적 업스케일링 (Apple의 DLSS/FSR 동등 기능)
- **레이 트레이싱(Ray Tracing)**: Apple Silicon (M3+)에서 하드웨어 가속 레이 트레이싱

### 3.5 Metal vs. Vulkan

| 측면 | Metal | Vulkan |
|------|-------|--------|
| 플랫폼 | Apple 전용 | 크로스 플랫폼 |
| 장황함 | 보통 | 매우 장황함 |
| 렌더 패스 | 드로우별 디스크립터 | 사전 정의된 구조 |
| 메모리 | 통합 (Apple Silicon) | 명시적 할당 |
| 셰이딩 언어 | MSL (C++ 기반) | SPIR-V (GLSL/HLSL에서 컴파일) |
| 검증 | Metal 검증 레이어 | Vulkan 검증 레이어 |
| 성숙도 | 10년 이상 | 10년 이상 |

---

## 4. DirectX 12

### 4.1 개요

**DirectX 12** (DX12)는 Windows 10+ 및 Xbox용 Microsoft의 저수준 그래픽스 API다. DirectX 11의 후속 버전이며 Vulkan, Metal과 동일한 명시적 설계 철학을 공유한다.

### 4.2 핵심 개념

| Vulkan | DirectX 12 | 비고 |
|--------|-----------|------|
| VkDevice | ID3D12Device | GPU 인터페이스 |
| VkQueue | ID3D12CommandQueue | 그래픽스, 컴퓨트, 복사 큐 |
| VkCommandBuffer | ID3D12GraphicsCommandList | 커맨드 레코딩 |
| VkPipeline | ID3D12PipelineState | 파이프라인 스테이트 오브젝트 (PSO) |
| VkDescriptorSet | Descriptor Heap/Table | 루트 시그니처(Root Signature)가 레이아웃 정의 |
| VkRenderPass | — | DX12는 렌더 타겟을 직접 사용 |
| SPIR-V | DXIL | 셰이더 중간 표현(Intermediate Representation) |

### 4.3 DX12 고유 기능

- **루트 시그니처(Root Signature)**: 셰이더 리소스(CBV, SRV, UAV, 샘플러)의 레이아웃을 정의한다. Vulkan 디스크립터 셋 레이아웃보다 더 명시적이다.
- **디스크립터 힙(Descriptor Heaps)**: GPU 가시(GPU-Visible) 리소스 디스크립터 테이블이다. "바인드리스(Bindless)" 렌더링은 모든 텍스처를 하나의 힙에 넣는다.
- **워크 그래프(Work Graphs)** (DX12 Ultimate): GPU 주도 작업 디스패치(Dispatch) -- GPU가 자체적으로 작업 항목을 생성하고 스케줄링한다.
- **DirectX 레이트레이싱(DXR, DirectX Raytracing)**: 최초의 하드웨어 레이 트레이싱 API (2018), BLAS/TLAS 가속 구조 사용.
- **DirectStorage**: NVMe SSD에서 GPU 메모리로 에셋(Asset)을 직접 로드하여 CPU를 우회한다.

### 4.4 HLSL 셰이딩 언어

DX12는 HLSL(High-Level Shading Language)을 사용하며, DXIL(DirectX Intermediate Language)로 컴파일된다:

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

### 5.1 개요

**WebGPU**는 웹 브라우저에서 GPU 접근을 위한 W3C 표준이다. WebGL을 내부적으로 Vulkan, Metal, DX12에 매핑되는 현대적이고 명시적인 API로 대체한다.

### 5.2 설계 목표

- **크로스 플랫폼(Cross-Platform)**: 모든 브라우저(Chrome, Firefox, Safari)에서 실행된다
- **안전성(Safe)**: GPU 충돌/멈춤을 방지하기 위해 모든 작업을 검증한다
- **현대적(Modern)**: 명시적 리소스 바인딩(Resource Binding), 컴퓨트 셰이더, 스토리지 버퍼(Storage Buffer)
- **고성능(Performant)**: WebGL보다 훨씬 낮은 드라이버 오버헤드

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

### 5.4 WebGPU 아키텍처

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

| 기능 | WebGL 2 | WebGPU |
|------|---------|--------|
| API 모델 | 상태 기반 (OpenGL ES 3.0) | 명시적 (Vulkan 유사) |
| 컴퓨트 셰이더 | 없음 | 있음 |
| 스토리지 버퍼 | 제한적 | 있음 |
| 멀티스레드 | 없음 | 워커(Worker)에서 커맨드 인코딩 |
| 셰이딩 언어 | GLSL ES | WGSL |
| 오류 처리 | 묵시적 실패 | 검증 오류 |
| 성능 | 드라이버 오버헤드 높음 | 오버헤드 낮음 |

---

## 6. 동기화(Synchronization)

### 6.1 동기화가 중요한 이유

현대 API는 암묵적 순서 보장을 제공하지 않는다. 애플리케이션이 명시적으로 지정해야 한다:
- GPU가 CPU가 쓴 리소스를 언제 읽기 시작할 수 있는지
- 한 GPU 작업이 다른 작업이 시작되기 전에 완료되어야 하는 시점
- CPU가 GPU 결과를 언제 읽어올 수 있는지

### 6.2 동기화 프리미티브(Synchronization Primitives)

**펜스(Fence)** (CPU-GPU 동기화):
- CPU가 GPU에 작업을 제출하고 펜스를 받는다
- CPU는 펜스를 대기하여 GPU 작업 완료를 알 수 있다
- 프레임 페이싱(Frame Pacing)에 사용: 프레임 N-2가 완료될 때까지 기다린 후 해당 리소스 재사용

```
CPU: Submit frame N, get fence_N
     ...
CPU: Wait(fence_N-2)  // Ensure GPU finished frame N-2
CPU: Reuse frame N-2's command buffers and staging memory
```

**세마포어(Semaphore)** (GPU-GPU 동기화):
- 한 큐 작업이 완료되면 세마포어를 시그널(Signal)한다
- 다른 큐 작업 시작 전에 세마포어를 대기한다
- 렌더링과 프레젠테이션(Presentation) 사이에 사용: 렌더링이 시그널, 프레젠테이션이 대기

```
GPU: Render pass signals semaphore_render_done
GPU: Present waits on semaphore_render_done
```

**파이프라인 배리어(Pipeline Barrier)** (커맨드 버퍼 내부):
- 명령 간 메모리 및 실행 의존성을 지정한다
- "리소스 X의 쓰기가 완료되어야 리소스 X를 읽을 수 있다"
- 이미지 레이아웃 전환(Image Layout Transition)에 필수적이다 (예: 렌더 타겟에서 셰이더 입력으로)

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

### 6.3 트리플 버퍼링(Triple Buffering) 패턴

대부분의 애플리케이션은 GPU가 계속 작업할 수 있도록 2-3 프레임의 버퍼링을 사용한다:

```
Frame N:   CPU record  │  GPU execute
Frame N+1:              │  CPU record   │  GPU execute
Frame N+2:                              │  CPU record   │  GPU execute

Buffers: [A] [B] [C]  -- rotate through three sets of resources
```

각 프레임은 고유한 커맨드 버퍼, 유니폼 버퍼(Uniform Buffer), 디스크립터 셋을 갖는다. 펜스는 GPU가 아직 읽고 있는 버퍼를 덮어쓰지 않도록 보장한다.

---

## 7. 렌더 그래프(Render Graph)

### 7.1 문제

현대적 프레임은 수십 개의 렌더 패스(그림자 맵, G-버퍼, SSAO, 조명, 블룸, TAA 등)와 복잡한 리소스 의존성을 포함한다. 리소스 수명, 배리어, 메모리 앨리어싱(Memory Aliasing)을 수동으로 관리하는 것은 오류가 발생하기 쉽고 최적화하기 어렵다.

### 7.2 프레임 그래프(Frame Graph) 추상화

**렌더 그래프**(프레임 그래프라고도 함)는 렌더링 프레임의 선언적 기술이다:

1. **패스 선언(Declare Passes)**: 각 패스가 읽고, 쓰고, 생성하는 리소스를 지정한다
2. **의존성 그래프 구축(Build Dependency Graph)**: 시스템이 리소스 의존성으로부터 실행 순서를 결정한다
3. **미사용 패스 제거(Cull Unused Passes)**: 패스의 출력이 사용되지 않으면 제거한다
4. **스케줄링(Schedule)**: 최적 실행 순서를 결정하고, 배리어를 삽입하며, 메모리를 앨리어싱한다
5. **실행(Execute)**: 순서대로 패스를 실행한다

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

### 7.3 장점

- **자동 배리어 삽입(Automatic Barrier Insertion)**: 시스템이 각 리소스 전환 시점을 정확히 알고 최소한의 배리어를 삽입한다
- **메모리 앨리어싱(Memory Aliasing)**: 시간적으로 겹치지 않는 일시적 리소스는 동일한 GPU 메모리를 공유할 수 있다 (예: 그림자 맵과 블룸 버퍼)
- **패스 컬링(Pass Culling)**: 표시되지 않는 디버그 시각화 패스가 자동으로 제거된다
- **멀티 큐 스케줄링(Multi-Queue Scheduling)**: 비동기 컴퓨트 패스를 그래픽스 작업과 겹쳐 실행할 수 있다

### 7.4 구현 스케치

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

출력:
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

DebugVis가 컬링(Culling)된 것은 그 출력(`debug_output`)이 최종 출력으로 이어지는 어떤 패스에서도 필요하지 않기 때문이다.

---

## 8. 명시적 리소스 관리(Explicit Resource Management)

### 8.1 메모리 타입(Memory Types)

현대 API는 여러 GPU 메모리 타입을 노출한다:

| 메모리 타입 | Vulkan 명칭 | 특성 |
|------------|------------|------|
| 디바이스 로컬(Device-Local) | DEVICE_LOCAL | 가장 빠른 GPU 접근; CPU에서 접근 불가 |
| 호스트 가시(Host-Visible) | HOST_VISIBLE | CPU가 매핑하고 쓸 수 있음; 느린 GPU 접근 |
| 호스트 일관(Host-Coherent) | HOST_COHERENT | CPU 쓰기가 즉시 GPU에 반영됨 |
| 호스트 캐시(Host-Cached) | HOST_CACHED | CPU 읽기가 빠름 (캐시됨); 리드백(Readback)에 유용 |

**일반적인 패턴**:
1. HOST_VISIBLE 메모리에 스테이징 버퍼(Staging Buffer)를 생성한다
2. 메모리 매핑(Memory Map)으로 CPU에서 스테이징 버퍼로 데이터를 복사한다
3. GPU 전송(Transfer)으로 스테이징 버퍼에서 DEVICE_LOCAL 버퍼로 복사한다
4. 렌더링에는 DEVICE_LOCAL 버퍼를 사용한다

통합 메모리 아키텍처(Apple Silicon, 통합 GPU)에서는 디바이스 로컬과 호스트 가시가 동일한 물리 메모리일 수 있다.

### 8.2 디스크립터(Descriptor)와 바인딩(Binding)

셰이더는 **디스크립터** -- GPU에 리소스의 위치와 해석 방법을 알려주는 메타데이터 -- 를 통해 리소스(버퍼, 텍스처)에 접근한다.

**디스크립터 셋(Descriptor Sets)** (Vulkan) / **루트 시그니처 + 디스크립터 힙(Root Signatures + Descriptor Heaps)** (DX12):

```
Descriptor Set Layout:
  Binding 0: Uniform buffer (per-frame data)
  Binding 1: Storage buffer (per-object transforms)
  Binding 2: Combined image sampler (diffuse texture)
  Binding 3: Combined image sampler (normal map)
```

**바인드리스 렌더링(Bindless Rendering)**: 모든 텍스처를 하나의 큰 디스크립터 배열에 넣는다. 셰이더는 머티리얼(Material) ID를 사용하여 배열을 인덱싱한다. 이렇게 하면 드로우 콜 사이의 디스크립터 셋 전환이 없어져 CPU 오버헤드가 크게 줄어든다.

### 8.3 파이프라인 스테이트 오브젝트(PSO, Pipeline State Objects)

현대 API에서 모든 렌더링 상태는 불변(Immutable) **파이프라인 스테이트 오브젝트**에 구워진다:

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

이 중 **어느 것이든** 변경하려면 다른 PSO가 필요하다. 애플리케이션은 런타임 컴파일을 피하기 위해 로드 시간에 필요한 모든 PSO를 사전 컴파일한다 (또는 디스크에 캐시).

**파이프라인 캐시(Pipeline Caches)**: 이후 애플리케이션 실행 시 셰이더를 재컴파일하지 않도록 컴파일된 파이프라인 바이트코드를 저장한다.

---

## 9. 성능 고려사항(Performance Considerations)

### 9.1 CPU 측 최적화

| 기법 | 이점 |
|------|------|
| 멀티스레드 커맨드 레코딩 | 코어 간에 CPU 작업 분산 |
| 간접 드로잉(Indirect Drawing) | GPU가 드로우 파라미터를 채움; CPU는 하나의 호출만 발행 |
| 바인드리스 리소스(Bindless Resources) | 디스크립터 셋 변경 제거 |
| 파이프라인 상태 정렬(Pipeline State Sorting) | PSO 전환 최소화 |
| 영속 매핑(Persistent Mapping) | 스테이징 버퍼의 반복적인 맵/언맵 방지 |

### 9.2 GPU 측 최적화

| 기법 | 이점 |
|------|------|
| 비동기 컴퓨트(Async Compute) | 컴퓨트와 그래픽스 작업 겹치기 |
| 렌더 그래프 최적화 | 최소한의 배리어, 메모리 앨리어싱 |
| 메시 셰이더(Mesh Shaders) | GPU 주도 지오메트리 처리 |
| GPU 컬링(GPU Culling) | GPU에서 절두체/오클루전 컬링(Frustum/Occlusion Culling) |
| 간접 디스패치(Indirect Dispatch) | GPU가 수행할 작업량을 결정 |

### 9.3 API 선택

| 필요 | 권장 API |
|------|---------|
| 크로스 플랫폼 데스크톱 | Vulkan (또는 추상화 레이어) |
| Apple 플랫폼 | Metal |
| Windows 게임 | DirectX 12 (또는 Vulkan) |
| 웹 | WebGPU |
| 단순성/프로토타이핑 | OpenGL 4.6 또는 WebGL 2 |
| 최대 제어 | Vulkan 또는 DX12 |

### 9.4 추상화 레이어(Abstraction Layers)

대부분의 프로덕션 엔진은 여러 API 위에 추상화 레이어를 사용한다:

- **bgfx**: C/C++ 크로스 플랫폼 렌더링 라이브러리
- **wgpu** (Rust): 네이티브에서도 실행되는 WebGPU 구현 (Vulkan/Metal/DX12)
- **Sokol**: 크로스 플랫폼 그래픽스를 위한 단일 파일 C 라이브러리
- **SDL_GPU**: SDL의 새로운 GPU 추상화 (2024)
- **Unreal RHI**: Unreal Engine의 렌더링 하드웨어 인터페이스(Rendering Hardware Interface)

이 추상화들은 일부 API 특유의 최적화를 희생하지만 모든 플랫폼에서의 이식성을 제공한다.

---

## 10. 미래 방향

### 10.1 GPU 주도 렌더링(GPU-Driven Rendering)

전통적: CPU가 무엇을 그릴지 결정하고 GPU가 실행한다.
미래: GPU가 씬을 스캔하고, 컬링하고, 정렬하고, 드로우 콜을 직접 생성한다. CPU는 단일 간접 디스패치를 제출한다.

**Nanite** (Unreal 5), **메시 셰이더(Mesh Shaders)**, **워크 그래프(Work Graphs)** (DX12)는 완전한 GPU 주도 파이프라인을 향한 단계들이다.

### 10.2 뉴럴 렌더링(Neural Rendering)

GPU에서 실행되는 ML 모델이 전통적 파이프라인의 일부를 대체한다:
- **DLSS/FSR**: 신경망을 통한 시간적 업스케일링
- **뉴럴 레디언스 캐싱(Neural Radiance Caching)**: 프로브(Probe) 기반 전역 조명(GI)을 신경망으로 대체
- **가우시안 스플래팅(Gaussian Splatting)**: 포인트 클라우드(Point Cloud)에서 새로운 시점 합성

### 10.3 수렴(Convergence)

주요 API들(Vulkan, Metal, DX12, WebGPU)은 명시적 메모리, 커맨드 버퍼, 파이프라인 상태, 컴퓨트 셰이더, 레이 트레이싱이라는 유사한 개념으로 수렴하고 있다. 하나를 깊이 배우면 나머지를 이해하는 것이 훨씬 쉬워진다.

---

## 요약

| 개념 | 핵심 아이디어 |
|------|--------------|
| API 발전 | 즉시 모드 → 상태 기반 → 명시적; 각 세대는 편의성과 제어를 교환 |
| Vulkan | 크로스 플랫폼 명시적 API; 장황하지만 최대 제어와 이식성 |
| Metal | Apple의 명시적 API; Apple Silicon 통합 메모리; MSL 셰이딩 언어 |
| DirectX 12 | Microsoft의 명시적 API; DXR 레이 트레이싱; 워크 그래프 |
| WebGPU | 웹 네이티브 명시적 API; WGSL 셰이딩 언어; 안전하고 크로스 플랫폼 |
| 펜스(Fence) | CPU-GPU 동기화: CPU가 GPU 제출 완료를 대기 |
| 세마포어(Semaphore) | GPU-GPU 동기화: 한 큐가 시그널, 다른 큐가 대기 |
| 배리어(Barrier) | 커맨드 버퍼 내부: 리소스 레이아웃 전환 및 해저드(Hazard) 방지 |
| 렌더 그래프(Render Graph) | 선언적 프레임 기술; 자동 배리어 삽입, 컬링, 앨리어싱 |
| PSO | 불변 컴파일된 파이프라인 상태; 런타임 셰이더 컴파일 제거 |
| 바인드리스(Bindless) | 하나의 디스크립터 배열에 모든 리소스; 머티리얼 ID로 인덱싱; 최소 CPU 오버헤드 |
| 트리플 버퍼링(Triple Buffering) | 프레임당 3세트의 리소스; 펜스가 진행 중인 데이터 덮어쓰기 방지 |

## 연습 문제

1. **API 비교**: 단순한 렌더링 작업(텍스처가 적용된 조명 큐브 그리기)을 선택한다. OpenGL 3.3, Vulkan, WebGPU에 대한 의사 코드(Pseudocode)를 작성한다. 각각에서 API 호출 수와 생성된 오브젝트 수를 세어 비교한다.

2. **렌더 그래프**: 렌더 그래프 구현에 **메모리 앨리어싱(Memory Aliasing)** 기능을 추가한다: 실행 타임라인에서 겹치지 않는 리소스는 동일한 "메모리 할당"을 공유해야 한다 (간단한 카운터로 추적). 앨리어싱 적용 전후의 총 메모리를 출력한다.

3. **동기화 설계**: 비동기 컴퓨트가 있는 트리플 버퍼링 Vulkan 애플리케이션의 펜스와 세마포어 설정을 설계한다. 각 프레임마다 시그널/대기되는 펜스와 세마포어를 보여주는 타임라인을 그린다.

4. **파이프라인 상태 폭발(Pipeline State Explosion)**: 머티리얼 시스템이 3가지 버텍스 포맷, 2가지 블렌드 모드, 4가지 셰이더 변형, 3가지 렌더 패스 구성을 지원한다고 하자. 필요한 PSO 수는 얼마인가? 파이프라인 캐시와 지연 생성(Lazy Creation)을 사용하여 이를 어떻게 관리할 것인가?

5. **바인드리스 vs. 바운드(Bindless vs. Bound)**: 1000개의 오브젝트를 (a) 오브젝트별 디스크립터 셋 바인드와 (b) 바인드리스 렌더링으로 그릴 때의 CPU 오버헤드를 비교한다. 각 접근 방식의 API 호출 수를 추정한다.

6. **WebGPU 컴퓨트**: 스토리지 버퍼(Storage Buffer)의 모든 원소를 두 배로 만드는 WebGPU 컴퓨트 셰이더(WGSL)를 작성한다. 디바이스, 버퍼, 파이프라인을 생성하고 셰이더를 디스패치하는 JavaScript 코드를 작성한다.

## 더 읽을거리

- Sellers, G. et al. *Vulkan Programming Guide*. Addison-Wesley, 2016. (공식 Vulkan 튜토리얼 스타일 도서)
- Vulkan Tutorial. https://vulkan-tutorial.com/ (가장 인기 있는 단계별 Vulkan 가이드)
- Apple Metal Documentation. https://developer.apple.com/metal/ (공식 Metal 프로그래밍 가이드)
- Microsoft DirectX 12 Documentation. https://learn.microsoft.com/en-us/windows/win32/direct3d12/ (DX12 레퍼런스)
- WebGPU Specification. W3C, 2024. https://www.w3.org/TR/webgpu/ (공식 WebGPU 사양)
- O'Donnell, Y. "FrameGraph: Extensible Rendering Architecture in Frostbite." *GDC*, 2017. (EA Frostbite 엔진의 프레임 그래프 설계)
