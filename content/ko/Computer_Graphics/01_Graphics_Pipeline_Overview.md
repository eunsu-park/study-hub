# 01. 그래픽스 파이프라인 개요

| [다음: 02. 2D 변환 →](02_2D_Transformations.md)

---

## 학습 목표

1. 실시간 그래픽스 파이프라인의 목적과 구조 이해
2. CPU 측(애플리케이션 단계)과 GPU 측 처리 구분
3. 하나의 삼각형이 정점 데이터에서 화면의 색상 픽셀로 변환되는 과정 추적
4. 정점 쉐이더(vertex shader), 변환(transformation), 클리핑(clipping)을 포함한 정점 처리 설명
5. 프리미티브 어셈블리(primitive assembly)와 삼각형 셋업(triangle setup) 기술
6. 프래그먼트 쉐이더(fragment shader), 깊이 테스트(depth testing), 블렌딩(blending)을 포함한 프래그먼트 처리 이해
7. 프레임 표시 전략으로서의 이중/삼중 버퍼링(double/triple buffering)과 수직 동기화(vsync) 설명
8. 테셀레이션(tessellation), 지오메트리 쉐이더(geometry shader), 컴퓨트 쉐이더(compute shader)와 같은 현대 파이프라인 확장 기능 인식

---

## 왜 이것이 중요한가

분당 120 프레임으로 수백만 개의 삼각형을 렌더링하는 AAA 게임부터, 정밀 공학 모델을 시각화하는 CAD 소프트웨어, 체적 데이터를 재구성하는 의료 영상 도구에 이르기까지 — 모든 실시간 3D 애플리케이션은 그래픽스 파이프라인에 의존한다. 이 파이프라인을 이해하는 것은 컴퓨터 그래픽스의 모든 분야에서 가장 중요한 전제 조건이다. 왜냐하면 이후에 다루는 모든 주제(변환, 쉐이딩, 텍스처, GPU 프로그래밍)가 하나 이상의 파이프라인 단계에 직접 대응되기 때문이다.

그래픽스 파이프라인을 **공장의 조립 라인(assembly line)**으로 생각하라. 원재료(정점 데이터)가 한쪽 끝으로 들어와 일련의 전문화된 작업 스테이션(파이프라인 단계)을 통과하여 완성품(디스플레이의 색상 픽셀)으로 나온다. 각 스테이션은 하나의 작업에 최적화되어 있으며, 공장의 처리량은 가장 느린 스테이션, 즉 *병목(bottleneck)*에 달려 있다. 현대 GPU는 이 조립 라인의 병렬성을 활용하여 초당 수십억 번의 연산을 처리한다.

---

## 1. 전체 그림

그래픽스 파이프라인은 3D 장면 데이터를 화면에 표시되는 2D 이미지로 변환한다. 핵심적으로 이것은 다음과 같은 함수다:

$$f: \text{3D Scene Description} \rightarrow \text{2D Pixel Array (Framebuffer)}$$

파이프라인은 각각 서로 다른 종류의 작업을 수행하는 큰 단계들로 나뉜다:

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

## 2. 애플리케이션 단계 (CPU)

애플리케이션 단계는 CPU에서 전적으로 실행된다. 이 단계는 다음을 담당한다:

### 2.1 장면 관리(Scene Management)

CPU는 각 프레임에서 *무엇을* 렌더링할지 결정한다. 이 과정에는 다음이 포함된다:

- **씬 그래프 순회(Scene graph traversal)**: 객체들의 계층적 데이터 구조를 탐색
- **뷰 절두체 컬링(Frustum culling)**: 카메라의 뷰 볼륨 밖에 완전히 있는 객체 제거
- **오클루전 컬링(Occlusion culling)**: 다른 객체 뒤에 숨겨진 객체 건너뛰기
- **레벨 오브 디테일(LOD, Level-of-Detail) 선택**: 먼 거리의 객체에 더 단순한 메시 선택

### 2.2 드로우 콜(Draw Call)

**드로우 콜(draw call)**은 CPU에서 GPU로 보내는 명령이다: "이 설정으로 이 삼각형 집합을 렌더링하라." CPU가 다음을 수행해야 하므로 각 드로우 콜에는 오버헤드가 발생한다:

1. GPU 상태 설정 (어떤 쉐이더 사용, 어떤 텍스처 바인딩)
2. 정점/인덱스 버퍼 데이터 업로드 또는 참조
3. 실제 드로우 명령 발행

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

> **성능 인사이트**: 현대 게임은 프레임당 수천 개의 드로우 콜을 발행할 수 있다. 유사한 객체를 하나의 콜로 합치는 *배칭(batching)*과 하나의 콜로 여러 복사본을 그리는 *인스턴싱(instancing)*을 통해 드로우 콜을 줄이는 것이 핵심 최적화다.

### 2.3 CPU vs GPU 역할

| 측면 | CPU | GPU |
|--------|-----|-----|
| **아키텍처** | 소수의 강력한 코어 (4-16개) | 수천 개의 단순한 코어 |
| **강점** | 순차적 논리, 분기 처리 | 대규모 병렬 연산 |
| **파이프라인 역할** | 씬 셋업, 컬링, 드로우 콜 | 정점/프래그먼트 처리 |
| **메모리** | 시스템 RAM (16-64 GB) | VRAM (8-24 GB), 고대역폭 |
| **병목 징후** | 낮은 GPU 사용률 | 낮은 CPU 사용률 |

CPU는 *무엇을* 그릴지 준비하고, GPU는 *어떻게* 그릴지 실행한다.

---

## 3. 지오메트리 처리(Geometry Processing) (GPU)

GPU가 드로우 콜을 받으면 지오메트리 처리가 시작된다. 이 단계는 3D 형태를 정의하는 점인 **정점(vertex)**에 대해 동작한다.

### 3.1 정점 쉐이더(Vertex Shader)

**정점 쉐이더(vertex shader)**는 정점당 한 번씩 실행되는 작은 프로그램이다. 주요 역할은 정점 위치를 *오브젝트 공간(object space)*에서 *클립 공간(clip space)*으로 변환하는 것이다:

$$\mathbf{p}_{\text{clip}} = \mathbf{M}_{\text{projection}} \cdot \mathbf{M}_{\text{view}} \cdot \mathbf{M}_{\text{model}} \cdot \mathbf{p}_{\text{object}}$$

여기서:
- $\mathbf{p}_{\text{object}}$: 오브젝트의 로컬 좌표계에서의 정점 위치
- $\mathbf{M}_{\text{model}}$: 오브젝트 공간 $\rightarrow$ 월드 공간(world space) 변환
- $\mathbf{M}_{\text{view}}$: 월드 공간 $\rightarrow$ 카메라(눈) 공간(eye space) 변환
- $\mathbf{M}_{\text{projection}}$: 눈 공간 $\rightarrow$ 클립 공간 변환 (원근감 적용)

정점 쉐이더는 다음도 수행할 수 있다:
- 조명을 위한 법선(normal) 변환: $\mathbf{n}_{\text{world}} = (\mathbf{M}_{\text{model}}^{-1})^T \cdot \mathbf{n}_{\text{object}}$
- 이후 단계로 텍스처 좌표 전달
- 정점당 조명(per-vertex lighting) 계산
- 정점 애니메이션 (스켈레탈 애니메이션, 정점 변위)

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

### 3.2 좌표 공간을 통한 변환(Transformations Through Coordinate Spaces)

정점은 여러 좌표 공간을 거쳐 이동한다:

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

각 변환은 레슨 02와 03에서 자세히 살펴볼 것이다.

### 3.3 클리핑(Clipping)

투영 변환 후, 정점들은 **클립 공간(clip space)**에 위치한다. 클리핑은 뷰 볼륨(절두체, *frustum*) 밖에 있는 지오메트리를 제거한다. 클립 공간에서 점 $(x, y, z, w)$가 절두체 안에 있으려면:

$$-w \leq x \leq w, \quad -w \leq y \leq w, \quad -w \leq z \leq w$$

절두체에 부분적으로 걸친 삼각형은 *클리핑(clipped)*된다 — 절두체 경계를 따라 잘리며, 새로운 정점이 생성될 수 있다. 한 평면에 대해 클리핑된 삼각형은 최대 두 개의 삼각형을 생성할 수 있다.

### 3.4 원근 분할(Perspective Division)

클리핑 후, GPU는 **원근 분할(perspective division)**을 수행한다: 클립 좌표를 $w$로 나눈다:

$$\mathbf{p}_{\text{NDC}} = \left(\frac{x}{w}, \frac{y}{w}, \frac{z}{w}\right)$$

이것은 절두체를 **정규화 장치 좌표(NDC, Normalized Device Coordinate)** 큐브 $[-1, 1]^3$에 매핑하며, 카메라에서 멀리 있는 객체는 더 작게 보인다(원근 단축, perspective foreshortening).

### 3.5 프리미티브 어셈블리(Primitive Assembly)

변환된 개별 정점들은 **프리미티브(primitive)** — 주로 삼각형 — 로 그룹화된다. 정점 인덱스 `[0, 1, 2, 3, 4, 5]`를 삼각형 목록으로 제출하면, GPU는 삼각형 $(v_0, v_1, v_2)$와 $(v_3, v_4, v_5)$를 어셈블한다.

일반적인 프리미티브 유형:
- **삼각형 목록(Triangle list)**: 3개 정점마다 하나의 삼각형 형성
- **삼각형 스트립(Triangle strip)**: 새 정점마다 이전 두 정점과 삼각형 형성
- **삼각형 팬(Triangle fan)**: 모든 삼각형이 첫 번째 정점을 공유

### 3.6 삼각형 셋업(Triangle Setup)

래스터화 이전에, GPU는 효율적인 래스터화에 필요한 삼각형당 데이터를 계산한다:
- **에지 방정식(Edge equations)**: 픽셀이 삼각형 내부에 있는지 테스트하는 데 사용
- **속성 기울기(Attribute gradients)**: 색상, UV, 법선 등 정점 속성이 삼각형 표면에서 어떻게 변하는지
- **면 방향(Face orientation)**: 잠재적인 백 페이스 컬링(back-face culling)을 위한 앞/뒷면 결정

---

## 4. 래스터화(Rasterization)

래스터화는 연속적인 기하 프리미티브(삼각형)를 이산적인 **프래그먼트(fragment)** — 최종 이미지에 기여할 수 있는 픽셀 후보 — 로 변환한다.

### 4.1 래스터화 과정

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

### 4.2 프래그먼트(Fragment)와 픽셀의 차이

**프래그먼트(fragment)**는 픽셀과 같지 않다:
- 프래그먼트는 하나의 삼각형을 래스터화하여 생성된 *후보* 픽셀이다
- 여러 프래그먼트가 같은 픽셀 위치를 두고 경쟁할 수 있다 (겹치는 삼각형들로 인해)
- 깊이 테스트와 블렌딩 단계에서 어떤 프래그먼트가 최종 픽셀 색상에 기여하는지 결정된다

### 4.3 무게중심 보간(Barycentric Interpolation)

정점 $A$, $B$, $C$로 구성된 삼각형 내부의 점 $P$를 다음과 같이 표현할 수 있다:

$$P = \alpha A + \beta B + \gamma C, \quad \alpha + \beta + \gamma = 1$$

무게중심 좌표(barycentric coordinates) $(\alpha, \beta, \gamma)$는 각 정점이 해당 점에 미치는 "영향력"을 나타낸다. 이를 사용하여 삼각형 전체에 걸쳐 정점 속성을 매끄럽게 보간한다:

$$\text{attr}(P) = \alpha \cdot \text{attr}(A) + \beta \cdot \text{attr}(B) + \gamma \cdot \text{attr}(C)$$

이것이 삼각형 표면에서 부드러운 색상 그라데이션, 텍스처 좌표, 법선이 계산되는 방식이다.

> **중요**: 원근 보정 보간(perspective-correct interpolation)을 위해서는 보간 전에 속성을 $w$로 나누고, 이후 다시 곱해야 한다. 이 보정 없이는 텍스처가 표면에서 "흘러다니는" 것처럼 보인다.

래스터화 구현은 레슨 04에서 처음부터 직접 해볼 것이다.

---

## 5. 프래그먼트 처리(Fragment Processing)

래스터화로 생성된 각 프래그먼트는 프래그먼트 처리 단계에 들어간다.

### 5.1 프래그먼트 쉐이더(Fragment Shader)

**프래그먼트 쉐이더(fragment shader)**(픽셀 쉐이더(pixel shader)라고도 함)는 프래그먼트당 한 번씩 실행된다. 다음을 기반으로 프래그먼트의 색상을 계산한다:

- 보간된 정점 속성 (법선, 텍스처 좌표)
- 텍스처 샘플 (이미지에서 색상 조회)
- 조명 계산 (광원 위치, 재질 속성 사용)
- 프로그래머가 원하는 모든 커스텀 연산

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

### 5.2 깊이 테스트(Depth Test) (Z-Buffer)

여러 프래그먼트가 같은 픽셀을 두고 경쟁할 때, **깊이 테스트(depth test)**가 어느 것이 보이는지 결정한다. GPU는 **깊이 버퍼(depth buffer)**(Z-버퍼)를 유지한다 — 각 픽셀에서 지금까지 본 가장 가까운 프래그먼트의 깊이(카메라로부터의 거리)를 저장하는 2D 배열이다.

알고리즘:
1. 위치 $(x, y)$에 깊이 $z$를 가진 각 프래그먼트에 대해:
2. $(x, y)$의 깊이 버퍼 현재 값과 $z$를 비교
3. $z < \text{depth\_buffer}[x][y]$ (프래그먼트가 더 가깝다면): 색상 버퍼와 깊이 버퍼 모두 업데이트
4. 그렇지 않으면: 프래그먼트 폐기

이것은 삼각형을 그리는 순서와 관계없이 **은면 제거(hidden surface problem)**를 우아하게 해결한다.

### 5.3 스텐실 테스트(Stencil Test)

**스텐실 버퍼(stencil buffer)**는 마스킹 효과를 위한 추가적인 픽셀당 정수 버퍼다:
- 그림자 볼륨(Shadow volumes)
- 포털 렌더링(Portal rendering)
- 외곽선/실루엣 효과
- 거울 반사

### 5.4 블렌딩(Blending)

투명한 객체의 경우, 프래그먼트는 단순히 수락되거나 거부되지 않는다. 대신, 들어오는 프래그먼트 색상이 프레임버퍼(framebuffer)의 기존 색상과 **블렌딩(blended)**된다:

$$C_{\text{final}} = \alpha_{\text{src}} \cdot C_{\text{src}} + (1 - \alpha_{\text{src}}) \cdot C_{\text{dst}}$$

여기서 $\alpha_{\text{src}}$는 들어오는 프래그먼트의 불투명도다 (0 = 완전 투명, 1 = 완전 불투명).

> **순서 의존성 문제**: 블렌딩은 순서에 의존한다. 올바른 결과를 위해 투명한 객체는 뒤에서 앞으로(화가 알고리즘, painter's algorithm) 렌더링되어야 한다. 이것이 CPU가 애플리케이션 단계에서 투명한 객체를 정렬하는 이유다.

---

## 6. 프레임 표시(Frame Presentation)

### 6.1 이중 버퍼링(Double Buffering)

이중 버퍼링이 없으면, 디스플레이는 프레임버퍼가 *그려지는 동안* 표시하게 되어 눈에 보이는 테어링(tearing) 아티팩트가 발생한다. 이중 버퍼링은 두 개의 프레임버퍼를 사용한다:

- **프론트 버퍼(Front buffer)**: 현재 화면에 표시되는 것
- **백 버퍼(Back buffer)**: GPU가 현재 렌더링하는 것

렌더링이 완료되면 버퍼가 **스왑(swap)**된다. 사용자는 항상 완성된 프레임만 보게 된다.

```
Time ──────────────────────────────────────────────────────▶

Frame N:   [  GPU renders to Back   ] ◄── swap ──▶ [Display shows Front]
Frame N+1: [  GPU renders to Back   ] ◄── swap ──▶ [Display shows Front]
```

### 6.2 수직 동기화(VSync)

**수직 동기화(VSync, Vertical Synchronization)**는 버퍼 스왑을 디스플레이의 주사율(일반적으로 60Hz 또는 144Hz)과 동기화한다. VSync 없이는:
- GPU가 화면 갱신 중에 프레임을 완료하면 스왑이 **화면 찢어짐(screen tearing)**을 유발한다 (위쪽 절반은 이전 프레임, 아래쪽 절반은 새 프레임)

VSync 사용 시:
- 버퍼 스왑이 디스플레이의 **수직 블랭킹 구간(vertical blank interval)**을 기다린다
- 찢어짐 없음, 그러나 잠재적인 입력 지연 (프레임이 다음 갱신 주기를 기다려야 함)
- 프레임이 하나의 갱신 주기보다 오래 걸리면 FPS가 절반으로 감소 (60 FPS → 30 FPS)

### 6.3 삼중 버퍼링(Triple Buffering)

삼중 버퍼링은 세 번째 버퍼를 추가하여 VSync의 지연 패널티를 완화한다:

| 방식 | 버퍼 수 | 테어링 | 지연 | FPS 동작 |
|--------|---------|---------|---------|--------------|
| VSync 없음 | 2 | 있음 | 낮음 | 제한 없음 |
| VSync (이중) | 2 | 없음 | 높음 | 60 미만 시 30으로 하락 |
| VSync (삼중) | 3 | 없음 | 중간 | 더 부드러운 성능 저하 |

삼중 버퍼링에서 GPU는 이전에 완료된 프레임이 VSync 스왑을 기다리는 중에도 항상 렌더링할 백 버퍼를 확보한다.

---

## 7. 현대 파이프라인

위에서 설명한 고전적인 파이프라인은 추가적인 프로그래밍 가능한 선택적 단계들로 확장되었다.

### 7.1 테셀레이션(Tessellation)

테셀레이션(tessellation)은 GPU에서 지오메트리를 동적으로 더 세밀한 삼각형으로 세분하여 다음을 가능하게 한다:
- 적응형 레벨 오브 디테일(카메라 가까이에 더 많은 삼각형)
- 변위 매핑(displacement mapping) (텍스처로 표면 조각)
- 거친 제어 메시에서 부드러운 곡선과 표면

테셀레이션 파이프라인은 두 개의 새로운 단계를 삽입한다:

```
Vertex Shader ──▶ Tessellation Control Shader ──▶ Tessellator (fixed)
                                                        │
                                                        ▼
                                        Tessellation Evaluation Shader ──▶ ...
```

- **테셀레이션 제어 쉐이더(TCS, Tessellation Control Shader)**: 얼마나 세분할지 결정 (테셀레이션 레벨)
- **테셀레이터(Tessellator)**: 새로운 정점을 생성하는 고정 기능 단계
- **테셀레이션 평가 쉐이더(TES, Tessellation Evaluation Shader)**: 새로운 정점의 위치 결정

### 7.2 지오메트리 쉐이더(Geometry Shader)

**지오메트리 쉐이더(geometry shader)**는 정점 처리와 래스터화 사이에 위치한다. 완전한 프리미티브(예: 삼각형)를 받아 다음을 할 수 있다:
- 0개 이상의 출력 프리미티브 방출
- 프리미티브 유형 변경 (입력 삼각형, 파티클 효과를 위한 출력 점)
- 새로운 지오메트리 생성 (그림자 볼륨 압출, 털/풀 생성)

> **성능 참고**: 지오메트리 쉐이더는 가변적인 출력 크기로 인해 일반적으로 기대보다 느리다. 파티클 시스템과 같은 작업에는 **컴퓨트 쉐이더(compute shader)**가 종종 선호된다.

### 7.3 컴퓨트 쉐이더(Compute Shader)

**컴퓨트 쉐이더(compute shader)**는 고정 파이프라인 구조에서 완전히 벗어난다. 범용 GPU 프로그램으로 다음을 할 수 있다:
- 임의의 버퍼 데이터 읽기 및 쓰기
- 워크그룹(workgroup) 내 스레드 동기화
- 모든 병렬 연산 수행

주요 용도:
- 파티클 시뮬레이션
- 물리 계산
- 이미지 후처리(post-processing)
- GPU 구동 컬링 (GPU가 무엇을 그릴지 결정하여 CPU 오버헤드 감소)

### 7.4 현대 파이프라인 전체 구조

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

## 8. 파이프라인 성능 고려사항

파이프라인을 이해하면 병목을 식별하는 데 도움이 된다:

### 8.1 병목이 발생하는 곳

| 단계 | 병목 지표 | 완화 방법 |
|-------|---------------------|------------|
| **애플리케이션 (CPU)** | GPU 유휴, 낮은 GPU 사용률 | 드로우 콜 감소, 인스턴싱 사용 |
| **정점 처리** | 높은 정점 수, 복잡한 정점 쉐이더 | 메시 단순화, LOD |
| **래스터화** | 큰 삼각형, 높은 해상도 | 오버드로우 감소 |
| **프래그먼트 처리** | 복잡한 쉐이더, 많은 텍스처 샘플 | 쉐이더 최적화, 밉맵 사용 |
| **대역폭** | 고해상도 텍스처, 큰 프레임버퍼 | 텍스처 압축, 해상도 낮추기 |

### 8.2 Early-Z 최적화

현대 GPU는 프래그먼트 쉐이더가 실행되기 *전에* 깊이 테스트를 수행할 수 있다 (**Early-Z**). 깊이 테스트에 실패하면 비용이 많이 드는 프래그먼트 쉐이더가 완전히 건너뛰어진다. 이것이 불투명한 객체를 **앞에서 뒤로(front-to-back)** 렌더링하면 성능이 향상되는 이유다 — 더 많은 프래그먼트가 이른 깊이 테스트에 실패하여 건너뛰어진다.

> **주의**: 프래그먼트 쉐이더가 `gl_FragDepth`에 쓰거나 `discard`를 사용하면, GPU가 최종 깊이 값을 미리 알 수 없으므로 해당 드로우 콜에서 Early-Z가 비활성화된다.

### 8.3 오버드로우(Overdraw)

**오버드로우(overdraw)**는 같은 픽셀에 대해 여러 프래그먼트가 쉐이딩될 때 발생한다. 오버드로우 비율 2x는 평균적으로 각 픽셀이 두 번 쉐이딩되었음을 의미한다. 오버드로우는 프래그먼트 쉐이더 사이클을 낭비한다.

오버드로우를 줄이는 전략:
1. 앞에서 뒤로 렌더링 (Early-Z와 함께)
2. 오클루전 컬링 (완전히 숨겨진 객체 건너뛰기)
3. 지연 렌더링(Deferred rendering) (지오메트리와 조명 패스를 분리하여 각 픽셀을 한 번만 쉐이딩)

---

## 9. 완전한 프레임: 모든 것을 합쳐서

텍스처와 조명이 적용된 주전자를 렌더링하는 하나의 프레임을 추적해 보자:

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

## 10. 역사적 맥락과 API 현황

| API | 플랫폼 | 파이프라인 모델 | 비고 |
|-----|----------|----------------|-------|
| **OpenGL** (1992) | 크로스 플랫폼 | 고정 기능 → 프로그래밍 가능 | 레거시이나 여전히 널리 교육됨 |
| **Direct3D 11** (2009) | Windows | 프로그래밍 가능 | PC 게임에서 주류 |
| **OpenGL ES** (2003) | 모바일 | OpenGL의 부분 집합 | iOS, Android |
| **WebGL** (2011) | 브라우저 | OpenGL ES 2.0/3.0 기반 | 레슨 07에서 사용 |
| **Vulkan** (2016) | 크로스 플랫폼 | 저수준, 명시적 | 최대 제어와 성능 |
| **Direct3D 12** (2015) | Windows | 저수준, 명시적 | Vulkan과 유사한 철학 |
| **Metal** (2014) | Apple | 저수준, 명시적 | macOS, iOS |
| **WebGPU** (2023) | 브라우저 | 현대적, 명시적 | WebGL의 후계자 |

> **트렌드**: 업계는 개발자에게 GPU 리소스에 대한 더 많은 제어권을 제공하는 **저수준, 명시적 API** (Vulkan, D3D12, Metal, WebGPU)로 이동하고 있으며, 복잡성이 증가하는 대신이다. 고수준 API와 엔진(Unity, Unreal)은 이러한 차이를 추상화한다.

---

## 요약

| 파이프라인 단계 | 위치 | 프로그래밍 가능? | 핵심 작업 |
|---------------|----------|---------------|---------------|
| 애플리케이션 | CPU | 가능 (사용자 코드) | 씬 셋업, 컬링, 드로우 콜 |
| 정점 처리 | GPU | 가능 (정점 쉐이더) | 정점을 클립 공간으로 변환 |
| 테셀레이션 | GPU | 가능 (TCS + TES) | 지오메트리 세분 (선택적) |
| 지오메트리 쉐이더 | GPU | 가능 | 프리미티브 방출/수정 (선택적) |
| 클리핑 | GPU | 불가능 (고정 기능) | 절두체 밖의 지오메트리 제거 |
| 래스터화 | GPU | 불가능 (고정 기능) | 삼각형을 프래그먼트로 변환 |
| 프래그먼트 처리 | GPU | 가능 (프래그먼트 쉐이더) | 픽셀당 색상 계산 |
| 프래그먼트별 연산 | GPU | 설정 가능 | 깊이 테스트, 스텐실, 블렌딩 |
| 프레임 표시 | GPU/디스플레이 | 설정 가능 | 이중/삼중 버퍼링, 수직 동기화 |

**핵심 요점**:
- 파이프라인은 병렬성을 통한 처리량을 위해 최적화된 조립 라인이다
- 정점은 좌표 공간 변환을 거쳐 흐른다: 오브젝트 → 월드 → 눈 → 클립 → NDC → 화면
- 래스터화는 연속적인 지오메트리와 이산적인 픽셀 사이의 간극을 메운다
- 프래그먼트 쉐이더는 대부분의 시각적 품질 연산이 이루어지는 곳이다
- 파이프라인을 이해하는 것은 렌더링 성능 최적화에 필수적이다

---

## 연습 문제

1. **개념적 이해**: 씬에 10,000개의 삼각형이 있지만 뷰 절두체 컬링 후 3,000개만 보인다. 화면 해상도는 1920x1080이다. 생성되는 최대 프래그먼트 수를 추정하라 (가시적인 삼각형 하나가 평균 100픽셀을 덮는다고 가정). 오버드로우는 이 숫자에 어떤 영향을 미치는가?

2. **파이프라인 추적**: 위치 $(1, 2, 3)$을 가진 하나의 정점을 파이프라인 단계를 통해 추적하라. 각 단계에서 개념적으로 무슨 일이 일어나는지 설명하라 (정확한 값을 계산할 필요는 없다 — 단지 연산을 설명하라).

3. **병목 분석**: 게임이 30 FPS로 실행된다. 화면 해상도를 절반으로 줄이면 FPS가 58로 올라간다. 대신 삼각형 수를 절반으로 줄이면 (원래 해상도에서) FPS가 32로 유지된다. 병목은 어디인가? 왜인가?

4. **드로우 콜 감소**: 1,000개의 나무가 있는 숲 씬에서 각 나무가 별도의 드로우 콜로 그려진다. 시각적 품질을 유지하면서 드로우 콜 수를 줄이는 두 가지 전략을 제안하라.

5. **투명도 과제**: 투명한 객체를 렌더링하는 것이 불투명한 객체보다 왜 더 어려운지 설명하라. 투명한 객체를 무작위 순서로 렌더링하면 어떻게 되는가? 두 개의 투명한 객체가 서로 겹치는 경우는 어떻게 되는가?

6. **현대적 확장**: 테셀레이션 쉐이더와 지오메트리 쉐이더를 비교하라. 지형에 풀 날을 생성하는 작업에서 어떤 접근 방식을 선택하겠는가, 그리고 왜인가?

---

## 더 읽을거리

1. Marschner, S. & Shirley, P. *Fundamentals of Computer Graphics* (5th ed.), Ch. 8 -- "The Graphics Pipeline"
2. Akenine-Moller, T., Haines, E., & Hoffman, N. *Real-Time Rendering* (4th ed.), Ch. 2 -- "The Graphics Rendering Pipeline"
3. [Learn OpenGL -- Getting Started](https://learnopengl.com/Getting-started/OpenGL) -- OpenGL 파이프라인에 대한 실용적 입문
4. [Life of a Triangle (NVIDIA)](https://developer.nvidia.com/content/life-triangle-nvidias-logical-pipeline) -- 삼각형이 NVIDIA GPU 아키텍처를 통과하는 방법
5. [Vulkan Tutorial](https://vulkan-tutorial.com/) -- 현대 저수준 파이프라인 프로그래밍
