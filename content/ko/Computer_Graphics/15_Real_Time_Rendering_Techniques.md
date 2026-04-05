# 15. 실시간 렌더링 기법

[← 이전: 14. GPU 컴퓨팅](14_GPU_Computing.md) | [다음: 16. 현대 그래픽스 API 개요 →](16_Modern_Graphics_APIs_Overview.md)

---

## 학습 목표

1. 포워드 렌더링(Forward Rendering)과 디퍼드 렌더링(Deferred Rendering) 파이프라인의 차이점과 트레이드오프를 비교한다
2. G-버퍼(G-buffer)의 구조와 지오메트리(Geometry)와 조명(Lighting) 계산을 분리하는 역할을 이해한다
3. 기본 그림자 매핑(Shadow Mapping), PCF, 캐스케이드 그림자 맵(Cascaded Shadow Maps) 기법을 구현한다
4. 화면 공간 주변광 차폐(SSAO, Screen-Space Ambient Occlusion)와 간접 그림자 근사 원리를 설명한다
5. 포토리얼리스틱(Photorealistic) 출력을 위한 블룸(Bloom), HDR 렌더링(HDR Rendering), 톤 매핑(Tone Mapping)을 설명한다
6. 씬(Scene) 복잡도 관리를 위한 레벨 오브 디테일(LOD, Level of Detail) 전략을 이해한다
7. 시간적 안티앨리어싱(TAA, Temporal Anti-Aliasing)과 공간적 방법 대비 이점을 설명한다
8. Python으로 디퍼드 셰이딩(Deferred Shading) G-버퍼 시뮬레이션을 구축한다

---

## 왜 중요한가

초당 60프레임 이상으로 그림자, 반사, 주변광 차폐, 블룸, 안티앨리어싱이 포함된 씬을 렌더링하려면 정밀하게 조율된 파이프라인이 필요하다. 이 레슨의 각 기법은 특정 렌더링 과제에 대한 치열한 해결책을 나타내며, 이것들이 모여 모든 현대 게임 엔진(Unreal, Unity, Godot)과 실시간 3D 애플리케이션의 기반을 이룬다.

이 기법들을 이해하는 것은 엔진 개발자뿐만 아니라 엔진을 효과적으로 활용하는 모든 사람에게 필수적이다 -- 아티스트, 테크니컬 아티스트, 게임플레이 프로그래머 모두 그림자가 왜 그렇게 보이는지, 포스트 프로세싱(Post-Processing)이 성능에 어떤 영향을 미치는지, GPU가 어디에 시간을 쓰는지 알면 크게 도움이 된다.

---

## 1. 포워드 렌더링 vs. 디퍼드 렌더링

### 1.1 포워드 렌더링(Forward Rendering)

**포워드 렌더링**에서는 각 오브젝트가 단일 패스(Pass)에서 완전히 셰이딩된다. 버텍스 셰이더(Vertex Shader)가 지오메트리를 변환하고, 프래그먼트 셰이더(Fragment Shader)가 모든 광원을 평가한다:

```
For each object:
    For each pixel the object covers:
        For each light:
            Compute lighting contribution
        Output final color
```

**장점**:
- 구현이 단순하다
- 투명도(Transparency) 처리가 자연스럽다
- 메모리 대역폭이 낮다 (G-버퍼 불필요)
- MSAA가 직접 작동한다

**단점**:
- $O(objects \times lights)$ -- 광원이 많을수록 비용이 급증한다
- 오버드로우(Overdraw): 프래그먼트가 셰이딩되었다가 가까운 오브젝트에 가려질 수 있다
- 각 오브젝트가 모든 광원을 평가해야 한다 (또는 라이트 컬링 사용)

### 1.2 디퍼드 렌더링(Deferred Rendering)

**디퍼드 렌더링**은 처리 과정을 두 패스로 분리한다:

**패스 1 (지오메트리 패스)**: 모든 오브젝트를 렌더링하되, 조명을 계산하는 대신 표면 속성을 **G-버퍼**라 불리는 텍스처 세트에 저장한다.

**패스 2 (조명 패스)**: 각 픽셀마다 G-버퍼를 읽고 조명을 계산한다. 각 광원은 자신이 영향을 미치는 픽셀에만 작용한다.

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

**장점**:
- 조명 비용이 $O(pixels \times lights)$로, 씬 복잡도와 무관하다
- 오버드로우로 인한 조명 낭비가 없다 (보이는 표면만 조명 처리)
- 포스트 프로세싱 효과를 쉽게 추가할 수 있다 (모든 데이터가 화면 공간에 있음)
- 수백~수천 개의 많은 광원에도 잘 확장된다

**단점**:
- 메모리 대역폭이 높다 (G-버퍼는 픽셀당 64-128 바이트)
- 투명도 처리가 어렵다 (G-버퍼는 픽셀당 하나의 표면만 저장)
- G-버퍼와 함께 MSAA를 사용하면 비용이 크다 (대안: TAA)

### 1.3 Forward+ (타일드 포워드)

하이브리드 방식: 화면을 타일로 분할하고 각 타일에 영향을 주는 광원을 결정한 다음, 각 픽셀을 관련 광원만으로 셰이딩한다. 포워드 렌더링의 투명도 지원과 디퍼드 렌더링의 광원 확장성을 결합한다.

### 1.4 비교표

| 특성 | 포워드 | 디퍼드 | Forward+ |
|------|--------|--------|----------|
| 광원 확장성 | 낮음 | 우수 | 양호 |
| 메모리 대역폭 | 낮음 | 높음 | 중간 |
| 투명도 | 자연스러움 | 어려움 | 자연스러움 |
| MSAA 지원 | 기본 | 복잡 | 기본 |
| 오버드로우 비용 | 높음 | 낮음 | 높음 |
| 구현 난이도 | 단순 | 보통 | 복잡 |

---

## 2. G-버퍼(G-Buffer)

### 2.1 구조

**G-버퍼**(지오메트리 버퍼, Geometry Buffer)는 다중 렌더 타겟(MRT, Multiple Render Targets)에 픽셀별 표면 정보를 저장한다:

| 텍스처 | 포맷 | 내용 |
|--------|------|------|
| RT0: 알베도(Albedo) + AO | RGBA8 | 확산 색상(Diffuse Color, RGB) + 주변광 차폐(Ambient Occlusion, A) |
| RT1: 법선(Normal) | RGB16F 또는 RGB10A2 | 월드 공간 또는 뷰 공간 법선 |
| RT2: 재질(Material) | RGBA8 | 금속성(Metallic, R), 거칠기(Roughness, G), 발광(Emissive, B), 플래그(A) |
| 깊이(Depth) | D24S8 또는 D32F | 하드웨어 깊이 버퍼 |

**합계**: 픽셀당 약 12-16 바이트, 1920x1080 해상도에서 약 37 MB.

### 2.2 법선 인코딩(Normal Encoding)

완전한 3D 법선(3개의 float = 12 바이트)을 저장하는 것은 비용이 크다. 일반적인 압축 방식:

**팔면체 인코딩(Octahedral Encoding)** (2개의 float, 8 바이트): 단위 구를 2D 팔면체에 매핑한 후 정사각형으로 펼친다. 작은 저장 공간으로 우수한 품질을 제공한다.

**구면 좌표계(Spherical Coordinates)** (2개의 float): $(\theta, \phi)$. 단순하지만 극 근처에서 분포가 불균일하다.

**뷰 공간 Z 재구성**: X, Y 성분만 저장하고 $Z = \sqrt{1 - X^2 - Y^2}$로 재구성한다 (카메라를 향한 법선에서만 작동).

### 2.3 위치 재구성(Position Reconstruction)

월드 위치를 명시적으로 저장하는 대신 깊이 버퍼에서 재구성한다:

1. 픽셀 $(x, y)$에서 깊이 값 $z_{\text{buffer}}$를 읽는다
2. NDC(Normalized Device Coordinates)를 계산한다: $(x_{\text{ndc}}, y_{\text{ndc}}, z_{\text{ndc}})$
3. 역 투영 행렬(Inverse Projection Matrix)로 변환한다: $\mathbf{p}_{\text{view}} = \mathbf{P}^{-1} \cdot (x_{\text{ndc}}, y_{\text{ndc}}, z_{\text{ndc}}, 1)$
4. $w$로 나눈다: $\mathbf{p}_{\text{view}} = \mathbf{p}_{\text{view}} / p_w$

이를 통해 G-버퍼 텍스처 하나(12 바이트/픽셀)를 절약할 수 있다.

### 2.4 Python 구현: G-버퍼 시뮬레이션

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

## 3. 그림자 매핑(Shadow Mapping)

### 3.1 기본 그림자 매핑(Basic Shadow Mapping)

**그림자 매핑**(Williams, 1978)은 두 패스로 구성된 기법이다:

**패스 1 (그림자 맵 생성)**: 광원의 시점에서 씬을 렌더링한다. 깊이 값만 텍스처(그림자 맵)에 저장한다.

**패스 2 (씬 렌더링)**: 각 픽셀을 광원의 좌표 공간으로 투영한다. 픽셀의 깊이를 그림자 맵 값과 비교한다:
- 픽셀 깊이 > 그림자 맵 깊이이면 픽셀은 **그림자 안**에 있다
- 픽셀 깊이 <= 그림자 맵 깊이이면 픽셀은 **조명을 받는** 상태다

$$\text{shadow} = \begin{cases} 0 & \text{if } z_{\text{pixel}} > z_{\text{shadow map}} + \text{bias} \\ 1 & \text{otherwise} \end{cases}$$

### 3.2 그림자 바이어스(Shadow Bias)

**그림자 바이어스**는 **섀도우 에크네(Shadow Acne)** -- 그림자 맵의 제한된 해상도로 인한 자기 그림자(Self-Shadowing) 아티팩트 패턴 -- 를 방지한다. 바이어스 없이는 광원을 향한 표면이 잘못 자기 자신을 그림자로 처리한다.

$$z_{\text{test}} = z_{\text{pixel}} - \text{bias}$$

바이어스가 너무 작으면 섀도우 에크네가 발생하고, 너무 크면 **피터 패닝(Peter Panning)** (그림자가 오브젝트에서 분리됨)이 발생한다.

**슬로프-스케일 바이어스(Slope-Scale Bias)**는 표면 방향에 적응한다:

$$\text{bias} = \text{constant\_bias} + \text{slope\_factor} \cdot \tan(\theta)$$

여기서 $\theta$는 표면 법선과 광원 방향 사이의 각도다.

### 3.3 퍼센테이지-클로저 필터링(PCF, Percentage-Closer Filtering)

기본 그림자 매핑은 경계가 선명한(앨리어싱된) 그림자를 생성한다. **PCF**는 여러 그림자 맵 텍셀(Texel)을 샘플링하고 평균을 내어 그림자를 부드럽게 만든다:

$$\text{shadow} = \frac{1}{N} \sum_{i=1}^{N} \text{compare}(z_{\text{pixel}}, z_{\text{shadow map}}(\mathbf{s}_i))$$

여기서 $\mathbf{s}_i$는 그림자 맵 내 오프셋 샘플 위치다. 3x3 또는 5x5 커널이 일반적이다.

**중요**: PCF는 각 샘플을 필터링 전에 *먼저* 테스트한다 (비교 후 평균). 깊이 값을 먼저 필터링하고 비교하면 잘못된 결과가 나온다.

### 3.4 캐스케이드 그림자 맵(CSM, Cascaded Shadow Maps)

대규모 야외 씬에서는 단일 그림자 맵이 전체 뷰를 충분한 해상도로 커버할 수 없다. **CSM**은 시야 절두체(View Frustum)를 캐스케이드(보통 3-4개)로 분할하고 각각 고유한 그림자 맵을 갖는다:

```
Camera                                          Far
  |----[Cascade 0]----[Cascade 1]--------[Cascade 2]---------|
        512x512         512x512            512x512
       (near, high     (mid, medium      (far, low
        resolution)     resolution)       resolution)
```

가까운 오브젝트는 고해상도 그림자를, 먼 오브젝트는 낮은 해상도 그림자를 받는다. 셰이더는 픽셀 깊이에 따라 적절한 캐스케이드를 선택한다.

### 3.5 그림자 맵 변형 기법

| 기법 | 장점 | 단점 |
|------|------|------|
| 기본 그림자 맵 | 단순하고 빠름 | 경계가 딱딱함, 앨리어싱 |
| PCF | 부드러운 경계 | 균일한 커널, 여전히 앨리어싱 |
| 분산 그림자 맵(VSM, Variance Shadow Maps) | 필터링 가능, 부드러운 그림자 | 라이트 블리딩(Light Bleeding) |
| 지수 그림자 맵(ESM, Exponential Shadow Maps) | 빠른 필터링 | 경계의 과도한 어두움 |
| 모멘트 그림자 맵(Moment Shadow Maps) | 라이트 블리딩 없음 | 더 높은 메모리 + 연산 |
| 레이 트레이싱 그림자(Ray-Traced Shadows) | 완벽한 정확도, 부드러운 그림자 | GPU 레이 트레이싱 필요 |

---

## 4. 화면 공간 주변광 차폐(SSAO, Screen-Space Ambient Occlusion)

### 4.1 개념

**주변광 차폐(Ambient Occlusion, AO)**는 표면이 서로 가까운 곳(모서리, 틈새, 가구 아래)을 어둡게 처리한다. 간접광 가시성의 적분을 근사한다:

$$AO(\mathbf{p}) = \frac{1}{\pi} \int_{\Omega} V(\mathbf{p}, \omega) (\omega \cdot \mathbf{n})\, d\omega$$

여기서 $V(\mathbf{p}, \omega) = 0$은 방향 $\omega$에 가까운 차폐물이 있는 경우다.

진정한 AO를 계산하는 것은 비용이 크다. **SSAO**(Crytek, 2007)는 화면 공간 데이터(G-버퍼의 깊이 및 법선 버퍼)만을 사용하여 근사한다.

### 4.2 알고리즘

1. 각 픽셀에 대해 표면 법선 주변 반구에서 $N$개의 임의 점을 샘플링한다
2. 각 샘플을 화면 공간으로 투영하고 깊이 버퍼를 읽는다
3. 샘플이 깊이 버퍼 표면 **뒤에** 있으면(차폐됨) 차폐에 기여한다
4. 결과를 평균 내어 어둡게 만드는 계수로 적용한다

$$AO(\mathbf{p}) \approx 1 - \frac{1}{N}\sum_{k=1}^{N}\begin{cases} 1 & \text{if sample}_k \text{ is occluded} \\ 0 & \text{otherwise} \end{cases}$$

### 4.3 SSAO 개선 기법

**HBAO (Horizon-Based AO)**: 화면 공간에서 깊이 버퍼를 따라 광선을 추적하고 각 표면 점 위의 "수평선 각도"를 측정한다. 샘플 기반 SSAO보다 물리적으로 더 정확하다.

**GTAO (Ground Truth AO)**: 분석적 적분을 사용하는 시공간적 접근 방식이다. 샘플당 더 높은 품질을 제공한다.

**RTAO (Ray-Traced AO)**: 하드웨어 레이 트레이싱을 사용하여 실제 광선을 추적한다. 지상 진실(Ground Truth) 수준의 품질로, 현대 엔진에서 점점 더 많이 사용된다.

### 4.4 구현 스케치

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

## 5. 블룸(Bloom)과 HDR 렌더링

### 5.1 고동적 범위(HDR, High Dynamic Range)

실제 세계의 휘도(Luminance)는 매우 다양하다:
- 별빛: $10^{-3}$ cd/m$^2$
- 실내 조명: $10^{1}$ cd/m$^2$
- 햇빛 받는 설원: $10^{4}$ cd/m$^2$
- 태양 표면: $10^{9}$ cd/m$^2$

디스플레이는 대략 $10^{0}$에서 $10^{3}$ cd/m$^2$ 범위를 표시할 수 있다. **HDR 렌더링**은 부동 소수점 전체 범위(FP16 또는 FP32 렌더 타겟)에서 조명을 계산하고, **톤 매핑(Tone Mapping)**을 적용하여 디스플레이용으로 범위를 압축한다.

### 5.2 톤 매핑 연산자(Tone Mapping Operators)

**Reinhard**:
$$L_{\text{display}} = \frac{L}{1 + L}$$

단순하고 효과적이며 $[0, \infty)$을 $[0, 1)$로 매핑한다.

**ACES (Academy Color Encoding System)**: 영화 및 게임 업계 표준:

$$f(x) = \frac{x(ax + b)}{x(cx + d) + e}$$

상수: $a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14$.

**Uncharted 2 (Hable)**:
$$f(x) = \frac{(x(Ax + CB) + DE)}{(x(Ax + B) + DF)} - E/F$$

어깨(Shoulder)와 발끝(Toe)이 자연스러운 필름 곡선이다.

### 5.3 블룸(Bloom)

**블룸**은 카메라 렌즈와 인간의 눈에서 빛이 산란하여 밝은 오브젝트 주변에 발광 효과를 시뮬레이션한다:

1. **밝기 패스(Bright Pass)**: HDR 버퍼에서 밝기 임계값 이상의 픽셀을 추출한다
2. **블러(Blur)**: 넓은 가우시안 블러 적용 (효율을 위해 다운샘플된 밉 체인 사용)
3. **합성(Composite)**: 블러 처리된 밝은 픽셀을 원본 이미지에 다시 추가한다

```
HDR Image → Bright Pass → Downsample → Blur → Upsample → Add → Final
                            ↓                    ↑
                         Downsample → Blur → Upsample
                            ↓                    ↑
                         Downsample → Blur → Upsample
```

멀티 스케일 방식(각 레벨에서 다운샘플, 블러, 업샘플)은 조밀하고 넓은 헤일로(Halo)를 모두 갖춘 자연스러운 블룸 효과를 만든다.

### 5.4 블룸 구현

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

## 6. 레벨 오브 디테일(LOD, Level of Detail)

### 6.1 개념

카메라에서 멀리 있는 오브젝트는 더 적은 픽셀을 차지하므로 완전한 기하학적 디테일이 필요하지 않다. **LOD** 시스템은 점점 복잡도가 낮아지는 사전 구축된 메시(Mesh) 버전 사이를 전환한다:

| LOD 레벨 | 거리 | 삼각형 수 | 용도 |
|----------|------|-----------|------|
| LOD 0 | 근거리 (<10m) | 10,000 | 완전한 디테일 |
| LOD 1 | 중거리 (10-50m) | 2,500 | 축소된 디테일 |
| LOD 2 | 원거리 (50-200m) | 500 | 로우 폴리(Low Poly) |
| LOD 3 | 매우 원거리 (>200m) | 50 | 빌보드(Billboard) / 임포스터(Imposter) |

### 6.2 LOD 선택

LOD 레벨은 **화면 공간 크기**(픽셀 단위 투영 면적)에 따라 선택된다:

$$\text{screen\_size} = \frac{r_{\text{bounding sphere}}}{\text{distance}} \cdot \text{projection\_scale}$$

screen_size가 임계값보다 작으면 더 낮은 LOD로 전환한다.

### 6.3 LOD 전환

**이산 LOD(Discrete LOD)**: 레벨 간에 팝(Pop) 전환이 발생한다. **히스테리시스 밴드(Hysteresis Band)** (전환 올라가는 거리와 내려가는 거리를 다르게 설정)로 완화할 수 있다.

**크로스페이드(Cross-Fade, 디더링 LOD)**: 전환 중에 두 LOD를 알파 디더링(Alpha Dithering)으로 렌더링한다. 팝이 없고 약간의 렌더링 비용이 추가된다.

**연속 LOD(CLOD, Continuous LOD)**: 메시를 점진적으로 단순화한다 (Hoppe의 프로그레시브 메시). 복잡성으로 인해 게임에서는 거의 사용되지 않는다.

**Nanite (Unreal 5)**: GPU 주도 렌더링(GPU-Driven Rendering)을 사용하여 클러스터 단위로 적절한 디테일 레벨을 스트리밍하고 렌더링하는 가상화 지오메트리(Virtualized Geometry) 시스템. 수십억 개의 삼각형을 원활하게 처리한다.

### 6.4 기타 LOD 대상

LOD는 메시 이외에도 적용된다:
- **텍스처 LOD(Texture LOD)**: 밉맵(Mipmaps) (L06에서 다룸)
- **셰이더 LOD(Shader LOD)**: 원거리 오브젝트에 더 단순한 셰이딩 모델 사용
- **애니메이션 LOD(Animation LOD)**: 원거리 캐릭터의 본 수 줄이거나 업데이트 빈도 감소
- **파티클 LOD(Particle LOD)**: 원거리 이펙트에 더 적은 수의 크고 단순한 파티클 사용

---

## 7. 시간적 안티앨리어싱(TAA, Temporal Anti-Aliasing)

### 7.1 앨리어싱 문제

앨리어싱(Aliasing)은 연속적인 지오메트리가 픽셀 위치에서 이산적으로 샘플링될 때 발생한다. 들쭉날쭉한 경계, 깜박이는 스펙큘러 하이라이트(Specular Highlights), 흔들리는 텍스처가 모두 앨리어싱 아티팩트다.

### 7.2 공간적 안티앨리어싱 복습

| 방법 | 픽셀당 샘플 | 비용 | 품질 |
|------|-----------|------|------|
| SSAA (Supersampling) | 4-16x | 매우 높음 | 우수 |
| MSAA (Multisample) | 2-8x | 보통 | 양호 (경계만) |
| FXAA (Fast Approximate) | 1x + 포스트 프로세싱 | 낮음 | 보통 (흐릿함) |
| SMAA (Enhanced Subpixel) | 1x + 포스트 프로세싱 | 낮음 | 양호 |

### 7.3 시간적 안티앨리어싱(TAA)

TAA는 매 프레임마다 카메라 위치를 약간씩 지터링(Jittering)하고 이전 프레임과 블렌딩하여 **시간 축으로** 샘플을 누적한다:

$$C_{\text{out}} = \alpha \cdot C_{\text{current}} + (1 - \alpha) \cdot C_{\text{history}}$$

여기서 $\alpha \approx 0.05-0.1$ (많은 샘플이 누적된 히스토리를 선호).

**핵심 단계**:
1. **지터링(Jitter)**: 투영 행렬을 서브픽셀(Sub-Pixel) 양만큼 오프셋한다 (Halton 수열 또는 기타 저불일치 패턴 사용)
2. **렌더링(Render)**: 지터링된 카메라로 현재 프레임을 렌더링한다
3. **재투영(Reproject)**: 모션 벡터($\mathbf{v} = \mathbf{p}_{\text{current}} - \mathbf{p}_{\text{previous}}$)를 사용하여 각 픽셀이 이전 프레임에서 어디에 있었는지 찾는다
4. **히스토리 샘플링(Sample History)**: 재투영된 위치에서 히스토리 버퍼를 읽는다
5. **클램프/클립(Clamp/Clip)**: 오래된 데이터를 거부하기 위해 히스토리 샘플을 현재 프레임 이웃 영역으로 클램프한다 (고스팅 방지)
6. **블렌딩(Blend)**: 현재 프레임과 클램프된 히스토리의 가중 평균

### 7.4 TAA 과제

- **고스팅(Ghosting)**: 히스토리 거부가 너무 약하면 움직이는 오브젝트가 잔상을 남긴다
- **흐릿함(Blurring)**: 과도한 블렌딩은 선명도를 낮춘다 (샤프닝 포스트 프로세스로 완화)
- **비차폐(Disocclusion)**: 새로운 표면이 나타날 때 (이전에 숨겨져 있던 경우) 유효한 히스토리가 없다
- **서브픽셀 지터 아티팩트**: 고주파 디테일이 깜박일 수 있다

### 7.5 현대적 대안

**DLSS (NVIDIA)**: 신경망이 시간적 데이터를 사용하여 저해상도 프레임을 업스케일한다. 동일한 성능 비용에서 TAA보다 우수하다.

**FSR (AMD)**: 공간적 업스케일링(FSR 1.0) 또는 시간적 업스케일링(FSR 2.0+). DLSS의 크로스 벤더(Cross-Vendor) 대안.

**XeSS (Intel)**: ML 기반 시간적 업스케일링으로, DLSS/FSR과 유사한 목표를 갖는다.

---

## 8. 종합: 실시간 프레임의 구성

현대 게임 엔진에서의 일반적인 프레임 구성:

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

## 요약

| 기법 | 핵심 아이디어 |
|------|--------------|
| 포워드 렌더링(Forward Rendering) | 각 오브젝트를 모든 광원으로 한 패스에 셰이딩; $O(\text{objects} \times \text{lights})$ |
| 디퍼드 렌더링(Deferred Rendering) | 표면 데이터를 G-버퍼에 저장; 픽셀당 광원별로 셰이딩; $O(\text{pixels} \times \text{lights})$ |
| G-버퍼(G-Buffer) | 알베도, 법선, 깊이, 재질을 저장하는 MRT; 일반적으로 12-16 바이트/픽셀 |
| 그림자 매핑(Shadow Mapping) | 광원 시점에서 깊이 렌더링; 메인 렌더에서 비교하여 그림자 결정 |
| PCF | 더 부드러운 경계를 위해 여러 그림자 맵 샘플을 평균화 |
| 캐스케이드 그림자(Cascaded Shadows) | 근거리/중거리/원거리용 다른 해상도의 여러 그림자 맵 |
| SSAO | 각 픽셀 주변 반구 샘플링; 차폐를 위한 깊이 비교 |
| 블룸(Bloom) | 밝은 픽셀 추출, 멀티 스케일 블러, 글로우 효과를 위해 다시 추가 |
| HDR + 톤 매핑(Tone Mapping) | 선형 HDR에서 계산; 디스플레이 범위로 압축 (Reinhard, ACES) |
| LOD | 원거리 오브젝트에 더 낮은 디테일 메시로 전환 |
| TAA | 매 프레임 카메라 지터링; 재투영된 히스토리와 블렌딩; 시간 축 서브픽셀 AA |

## 연습 문제

1. **G-버퍼 시각화**: G-버퍼 시뮬레이션을 확장하여 5개의 서로 다른 재질을 가진 구를 렌더링한다. 각 G-버퍼 채널(알베도, 법선, 깊이, 거칠기, 금속성)을 별도 이미지로 시각화한다.

2. **그림자 매핑**: 방향광(Directional Light)이 있는 탑다운(Top-Down) 씬에 대한 기본 2D 그림자 맵을 구현한다. 광원에서 광선을 투사하여 그림자 맵을 구성하고, 그림자 맵 룩업(Lookup)을 사용하여 씬을 셰이딩한다.

3. **PCF 구현**: 그림자 매핑 연습에 PCF를 추가한다. 1-샘플, 3x3, 5x5 PCF 커널을 비교한다. 시각적 품질 향상과 성능 비용을 측정한다.

4. **SSAO 품질**: 전체 G-버퍼에 대한 SSAO 함수를 구현한다. 8, 16, 32, 64 샘플 결과를 비교한다. 경계를 흐리지 않고 AO를 스무딩하는 양방향 블러(Bilateral Blur) 포스트 프로세스를 추가한다.

5. **블룸 파라미터 탐색**: 멀티 레벨 블룸 파이프라인을 구현한다. 매우 밝은 광원 하나와 여러 개의 어두운 광원이 있는 씬을 렌더링한다. 임계값(0.5, 1.0, 2.0), 블룸 강도, 블러 레벨 수를 실험한다.

6. **톤 매핑 비교**: Reinhard, ACES, Hable (Uncharted 2) 톤 매핑을 구현한다. 전달 곡선(Transfer Curves)을 그린다. 동일한 HDR 이미지에 각각을 적용하고 시각적 결과를 비교한다.

## 더 읽을거리

- Akenine-Moller, T. et al. *Real-Time Rendering*, 4th ed. CRC Press, 2018. (이 레슨의 모든 주제에 대한 최종 참고서)
- Kaplanyan, A. "Cascaded Shadow Maps." *ShaderX6*, 2007. (CSM 구현 세부 사항)
- Mittring, M. "Finding Next Gen -- CryEngine 2." *SIGGRAPH Course*, 2007. (SSAO 소개)
- Karis, B. "Real Shading in Unreal Engine 4." *SIGGRAPH Course*, 2013. (UE4의 디퍼드 PBR 파이프라인)
- Salvi, M. "An Excursion in Temporal Supersampling." *GDC*, 2016. (Intel의 TAA 심층 분석)
