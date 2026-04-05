# 06. 텍스처 매핑

[← 이전: 05. 셰이딩 모델](05_Shading_Models.md) | [다음: 07. WebGL 기초 →](07_WebGL_Fundamentals.md)

---

## 학습 목표

1. UV 좌표(UV Coordinates)와 2D 이미지를 3D 표면에 매핑하는 방법을 이해한다
2. 최근접(Nearest-Neighbor) 및 쌍선형(Bilinear) 필터링으로 텍스처 샘플링을 구현한다
3. 밉맵(Mipmap)이 존재하는 이유, 생성 방법, 밉맵 레벨 선택 방법을 설명한다
4. 이방성 필터링(Anisotropic Filtering)과 그것이 품질을 향상시키는 시기를 이해한다
5. 노멀 매핑(Normal Mapping), 범프 매핑(Bump Mapping), 변위 매핑(Displacement Mapping)을 구분한다
6. PBR 텍스처 맵(알베도, 메탈릭, 러프니스, 주변광 차폐)을 설명한다
7. 큐브맵(Cubemap)과 등직사각형(Equirectangular) 투영을 사용한 환경 매핑(Environment Mapping)을 설명한다
8. Python으로 쌍선형 텍스처 샘플링을 처음부터 구현한다

---

## 왜 중요한가

텍스처 없이는 3D 장면의 모든 표면이 균일한 단색이 된다 — 레슨 05의 셰이딩 모델은 부드러운 그라디언트를 생성하지만 디테일이 없다. 텍스처는 나무 탁자를 나무처럼, 벽돌 담을 벽돌처럼, 캐릭터의 얼굴을 생생하게 보이게 만드는 요소다. 텍스처 매핑(Texture Mapping)은 실시간 그래픽에서 가장 영향력 있는 시각적 기법이라 할 수 있다: 좋은 텍스처를 가진 저폴리곤 모델은 텍스처 없는 고폴리곤 모델보다 훨씬 설득력 있어 보일 수 있다. 현대 PBR 워크플로는 여러 텍스처 맵(알베도, 노멀, 러프니스, 메탈릭, AO)을 사용하여 표면 외관의 모든 측면을 제어하며, 이러한 텍스처가 어떻게 샘플링되고 필터링되는지 이해하는 것은 아티스트와 프로그래머 모두에게 필수적이다.

---

## 1. UV 좌표

### 1.1 개념

**UV 좌표(UV Coordinates)**(텍스처 좌표(Texture Coordinates)라고도 불림)는 3D 표면에서 2D 텍스처 이미지로의 매핑을 정의한다. 메시의 각 꼭짓점은 해당 꼭짓점이 텍스처의 어디를 "바라보는지"를 지정하는 $(u, v)$ 좌표 쌍을 저장한다.

$$\text{UV}: \text{표면 점} \rightarrow (u, v) \in [0, 1]^2$$

규약:
- $u$는 텍스처의 수평축에 해당한다 (왼쪽 = 0, 오른쪽 = 1)
- $v$는 수직축에 해당한다 (OpenGL에서 아래 = 0; 일부 다른 API에서 위 = 0)

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

### 1.2 UV 파라미터화(UV Parameterization)

메시에 UV 좌표를 할당하는 과정을 **UV 언래핑(UV Unwrapping)**이라고 한다 — 판지 상자를 펼치듯, 3D 표면을 이음새를 따라 잘라 평면으로 펼치는 것을 상상하면 된다.

일반적인 파라미터화 방법:
- **평면 투영(Planar Projection)**: 한 방향에서 투영 (평면 표면에 적합)
- **원통형 투영(Cylindrical Projection)**: 원기둥을 감싸듯 (병, 나무에 적합)
- **구형 투영(Spherical Projection)**: 구를 감싸듯 (행성, 눈에 적합)
- **박스/큐브 투영(Box/Cube Projection)**: 여섯 개의 평면 투영 (건축 요소에 적합)
- **자동 언래핑(Automatic Unwrapping)**: LSCM, ABF++ 같은 알고리즘으로 왜곡 최소화

### 1.3 [0,1] 범위를 벗어난 UV

UV 좌표가 $[0, 1]$을 초과하면 어떻게 될까? **래핑 모드(Wrapping Mode)**가 동작을 결정한다:

| 모드 | 효과 | 사용처 |
|------|------|--------|
| **반복(Repeat)** | 텍스처를 타일링: $(u, v) \mod 1$ | 벽돌 담, 바닥 타일 |
| **경계로 클램프(Clamp to Edge)** | 경계 픽셀 연장: $\text{clamp}(u, 0, 1)$ | 하늘 텍스처, UI 요소 |
| **미러 반복(Mirrored Repeat)** | 교대로 방향을 뒤집으며 타일링 | 이음매 없는 패턴 |

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

## 2. 텍스처 샘플링

UV 좌표가 주어지면 텍스처에서 해당 색상을 조회해야 한다. 이는 연속 $(u, v) \in [0, 1]$을 텍스처 이미지의 이산 픽셀 인덱스로 변환하는 과정이다.

### 2.1 최근접 샘플링(Nearest-Neighbor Sampling)

가장 단순한 방법: 가장 가까운 텍셀(Texel, 텍스처 픽셀)로 스냅한다.

$$\text{texel\_x} = \lfloor u \cdot W \rfloor, \quad \text{texel\_y} = \lfloor v \cdot H \rfloor$$

여기서 $W \times H$는 텍스처 해상도다.

**장점**: 빠르고 날카로운 경계를 보존한다
**단점**: 확대 시 블록/픽셀화된 결과, 축소 시 깜박임 발생

### 2.2 쌍선형 필터링(Bilinear Filtering)

**쌍선형 필터링(Bilinear Filtering)**은 가장 가까운 네 텍셀 사이를 보간하여 부드러운 결과를 생성한다:

$x = u \cdot W - 0.5$, $y = v \cdot H - 0.5$인 연속 텍셀 좌표 $(x, y)$가 주어졌을 때:

1. 네 개의 주변 텍셀 찾기: $(x_0, y_0)$, $(x_0+1, y_0)$, $(x_0, y_0+1)$, $(x_0+1, y_0+1)$
2. 소수 위치 계산: $f_x = x - x_0$, $f_y = y - y_0$
3. 보간:

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

### 2.3 시각적 비교

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

## 3. 밉맵(Mipmap)

### 3.1 축소 문제(The Minification Problem)

텍스처가 입혀진 표면이 카메라에서 멀리 있으면, 많은 텍셀이 하나의 픽셀에 매핑된다. 특별한 처리 없이는 **텍스처 앨리어싱(Texture Aliasing)** — 단일 픽셀 샘플이 많은 텍셀의 평균을 표현할 수 없어 반짝임, 무아레(Moire) 패턴, 시각적 노이즈가 발생한다.

### 3.2 밉맵이란?

**밉맵(Mipmap)**(라틴어 "multum in parvo" — "작은 공간에 많은 것"에서 유래)은 텍스처의 점진적으로 작아지는 버전을 미리 계산한 피라미드다:

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

총 메모리 비용: 약 $\frac{1}{3}$ 추가 (급수 $1 + \frac{1}{4} + \frac{1}{16} + \ldots \rightarrow \frac{4}{3}$).

### 3.3 밉맵 생성

각 레벨은 이전 레벨을 2×2 박스 필터로 다운샘플링하여 생성된다:

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

### 3.4 밉맵 레벨 선택

어떤 밉맵 레벨을 사용할지는 텍스처의 **화면 공간 발자국(Screen-Space Footprint)** — 하나의 픽셀에 몇 개의 텍셀이 해당하는지 — 에 따라 결정된다.

밉맵 레벨 $\lambda$는 텍스처 좌표의 화면 공간 도함수(Derivative)로부터 계산된다:

$$\lambda = \log_2\left(\max\left(\left\|\frac{\partial \mathbf{uv}}{\partial x}\right\|, \left\|\frac{\partial \mathbf{uv}}{\partial y}\right\|\right) \cdot \text{texture\_size}\right)$$

여기서 $\frac{\partial \mathbf{uv}}{\partial x}$는 UV 좌표가 픽셀 단위 수평으로 얼마나 변하는지다. GPU는 인접 픽셀의 차이를 계산하여 이 도함수를 자동으로 구한다.

- $\lambda = 0$: 전체 해상도 텍스처 사용 (표면이 가깝다)
- $\lambda = 1$: 절반 해상도 텍스처 사용
- $\lambda = 3$: $\frac{1}{8}$ 해상도 텍스처 사용 (표면이 멀다)

### 3.5 삼선형 필터링(Trilinear Filtering)

**삼선형 필터링(Trilinear Filtering)**은 인접한 두 밉맵 레벨에서 쌍선형 샘플링을 수행한 후 그 사이를 선형 보간한다:

1. 소수 밉맵 레벨 $\lambda$ 계산 (예: 2.3)
2. 밉맵 레벨 $\lfloor \lambda \rfloor = 2$에서 쌍선형 샘플링
3. 밉맵 레벨 $\lceil \lambda \rceil = 3$에서 쌍선형 샘플링
4. 블렌드: $\text{color} = (1 - 0.3) \cdot \text{level2} + 0.3 \cdot \text{level3}$

이를 통해 밉맵 레벨이 변경되는 지점에서 보이는 "밴드(Band)"를 제거한다.

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

## 4. 이방성 필터링(Anisotropic Filtering)

### 4.1 밉맵의 문제점

밉맵은 텍스처 발자국이 대략 정사각형(등방성, Isotropic)이라고 가정한다. 하지만 표면을 스침각에서 보면 발자국이 매우 길게 늘어난다(이방성, Anisotropic) — 멀리 뻗어 있는 도로를 생각해 보자.

표준 삼선형 필터링은 *더 큰* 발자국 치수를 기준으로 밉맵 레벨을 선택하여 더 짧은 치수 방향으로 텍스처를 과도하게 흐리게 만든다.

### 4.2 이방성 필터링의 동작 방식

이방성 필터링(AF, Anisotropic Filtering)은 길게 늘어난 방향을 따라 여러 샘플을 취하여 평균을 낸다:

1. 이방성 비율 계산: $\text{ratio} = \frac{\max(\text{footprint})}{\min(\text{footprint})}$
2. 긴 축을 따라 최대 이방성 레벨(예: 16x)까지 $N$번의 삼선형 샘플 취하기
3. 결과 평균 내기

```
Isotropic footprint:     Anisotropic footprint:
   ┌───┐                    ┌───────────────┐
   │   │  ~square            │ ● ● ● ● ● ● │  multiple samples
   └───┘  1 sample           └───────────────┘  along long axis
```

**품질 계층**: 최근접 < 쌍선형 < 삼선형 < 이방성

| 필터 | 텍셀 읽기 수 | 품질 |
|------|------------|------|
| 최근접(Nearest) | 1 | 블록 형태 |
| 쌍선형(Bilinear) | 4 | 부드러움 (2D) |
| 삼선형(Trilinear) | 8 | 부드러움 (2D + 밉 전환) |
| 16x 이방성(Anisotropic) | 최대 128 | 최상 (각도에서 디테일 보존) |

---

## 5. 노멀 매핑(Normal Mapping)

### 5.1 동기

메시에 기하학적 디테일을 추가하는 것은 비용이 많이 든다: 삼각형이 많을수록 꼭짓점 처리와 메모리가 더 필요하다. **노멀 매핑(Normal Mapping)**은 텍스처를 사용하여 각 픽셀에서 표면 법선을 변형함으로써 어떤 기하학도 추가하지 않고 표면 디테일을 가짜로 만들어낸다.

### 5.2 동작 방식

**노멀 맵(Normal Map)**은 각 텍셀이 RGB 색상으로 인코딩된 표면 법선 방향을 저장하는 텍스처다:

- R 채널 $\rightarrow$ 법선의 X 성분
- G 채널 $\rightarrow$ 법선의 Y 성분
- B 채널 $\rightarrow$ 법선의 Z 성분

법선은 **탄젠트 공간(Tangent Space)** (표면에 상대적)으로 저장된다:
- Z는 바깥쪽을 향한다 (표면에 수직)
- X와 Y는 표면 평면에 놓인다

평면 표면은 모든 법선이 위를 향한다: $(0, 0, 1)$, 이는 $(128, 128, 255)$로 인코딩된다 — 노멀 맵의 특징적인 보라/파란색이다.

### 5.3 탄젠트 공간(Tangent Space)

탄젠트 공간 노멀 맵을 사용하려면 각 꼭짓점에서 좌표 프레임이 필요하다:
- **법선(Normal)** $\mathbf{N}$: 표면에 수직 (꼭짓점 데이터에서)
- **탄젠트(Tangent)** $\mathbf{T}$: 텍스처의 U 방향을 따름
- **바이탄젠트(Bitangent)** $\mathbf{B}$: $\mathbf{B} = \mathbf{N} \times \mathbf{T}$

**TBN 행렬(TBN Matrix)**은 탄젠트 공간 법선을 월드 공간으로 변환한다:

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

### 5.4 범프 매핑(Bump Mapping) vs 노멀 매핑(Normal Mapping)

| 기법 | 입력 | 저장 내용 | 비고 |
|------|------|---------|------|
| **범프 매핑(Bump Mapping)** | 회색조 높이맵 | 높이 값 | 높이 차이에서 법선 유도 |
| **노멀 매핑(Normal Mapping)** | RGB 텍스처 | 법선 벡터 직접 저장 | 더 정밀하며 표준적인 접근법 |

**범프 매핑(Bump Mapping)**(James Blinn, 1978의 원래 기법)은 회색조 높이맵을 사용한다. 변형된 법선은 높이 그라디언트(Gradient)에서 계산된다:

$$\mathbf{n}' = \mathbf{n} - \frac{\partial h}{\partial u} \mathbf{T} - \frac{\partial h}{\partial v} \mathbf{B}$$

노멀 매핑은 원하는 법선을 직접 저장하여 런타임에 도함수 계산을 피할 수 있기 때문에 오늘날 더 일반적으로 사용된다.

---

## 6. 변위 매핑(Displacement Mapping)

노멀 매핑(조명만 변경)과 달리, **변위 매핑(Displacement Mapping)**은 높이맵 텍스처를 기반으로 실제로 꼭짓점을 이동시킨다:

$$\mathbf{p}' = \mathbf{p} + h(u, v) \cdot \mathbf{n}$$

여기서 $h(u, v)$는 변위 맵에서 샘플링된 높이이고, $\mathbf{n}$은 표면 법선이다.

**비교**:

| 기법 | 기하 변형? | 실루엣 정확성? | 비용 |
|------|----------|------------|------|
| 노멀 매핑(Normal Mapping) | 아니오 | 아니오 (납작한 실루엣) | 저렴 (픽셀당) |
| 패럴랙스 매핑(Parallax Mapping) | 아니오 (가짜) | 부분적 | 보통 |
| 변위 매핑(Displacement Mapping) | 예 (꼭짓점 이동) | 예 | 비쌈 (테셀레이션 필요) |

변위 매핑은 일반적으로 **테셀레이션(Tessellation)**(레슨 01, 섹션 7.1)과 함께 사용된다: GPU가 메시를 미세한 삼각형으로 세분화한 다음, 높이맵에 따라 각 꼭짓점을 변위시킨다.

---

## 7. PBR 텍스처 맵

현대 PBR 재질은 표면 외관의 서로 다른 측면을 제어하는 여러 텍스처 맵을 사용한다:

### 7.1 알베도(Albedo) (기본 색상) 맵

조명이나 그림자 없이 표면 색상. 유전체의 경우 확산 색상이다. 금속의 경우 반사 색상($F_0$)을 정의한다.

- **해야 할 것**: 색상 정보만 저장
- **하지 말아야 할 것**: 알베도에 조명, 그림자, 주변광 차폐(AO)를 굽지 않기

### 7.2 메탈릭(Metallic) 맵

표면의 어느 부분이 금속성(1.0) vs 유전성(0.0)인지 나타내는 이진 또는 회색조 맵. 실제로 값은 주로 0 또는 1이며, 전환을 위한 중간 값(예: 금속 위의 녹)은 드물다.

### 7.3 러프니스(Roughness) 맵

각 텍셀에서 마이크로표면 거칠기를 제어한다:
- 0.0 = 거울처럼 매끄러움 (광택 크롬)
- 0.5 = 보통 거칠기 (닳은 플라스틱)
- 1.0 = 완전한 무광 (분필)

### 7.4 주변광 차폐(AO, Ambient Occlusion) 맵

각 지점이 주변광에 얼마나 차단되는지에 대한 미리 계산된 정보. 틈새와 좁은 모서리는 더 어둡다. 이는 주변광/간접 조명 항에 곱해진다:

$$L_{\text{ambient}} = \text{AO}(u,v) \cdot k_a \cdot L_a$$

### 7.5 완전한 PBR 텍스처 조회

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

### 7.6 일반적인 PBR 텍스처 파이프라인

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

## 8. 환경 매핑(Environment Mapping)

### 8.1 개념

**환경 매핑(Environment Mapping)**은 주변 환경을 텍스처로 캡처하여 다음 용도로 활용한다:
- **반사(Reflections)**: 거울 같은 표면이 환경을 반사한다
- **이미지 기반 조명(IBL, Image-Based Lighting)**: 환경을 PBR 광원으로 사용한다

### 8.2 큐브맵(Cubemap)

**큐브맵(Cubemap)**은 정육면체의 각 면에 하나씩 총 여섯 개의 정사각형 이미지로 환경을 저장한다:

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

큐브맵을 샘플링하기 위해 **3D 방향 벡터**(UV 좌표가 아님)를 사용한다. 하드웨어가 어떤 면을 샘플링할지와 그 면 내의 2D 좌표를 결정한다:

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

### 8.3 등직사각형 매핑(Equirectangular Mapping)

큐브맵의 대안으로 **등직사각형(Equirectangular)** (위경도, Lat-Long) 형식이 있다. 이는 2:1 가로세로 비율의 단일 이미지에 환경을 저장한다:

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

### 8.4 반사 매핑(Reflection Mapping)

거울 같은 표면의 경우, **반사 벡터(Reflection Vector)**를 사용하여 환경을 샘플링한다:

$$\mathbf{R} = 2(\mathbf{N} \cdot \mathbf{V})\mathbf{N} - \mathbf{V}$$

$$\text{reflection\_color} = \text{cubemap}(\mathbf{R})$$

거친 표면의 경우, 미리 필터링된 환경 맵(큐브맵의 흐려진 버전)을 사용하며, 더 거친 재질일수록 더 흐려진 레벨에서 샘플링한다.

---

## 9. 완전한 텍스처 매핑 예제

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

## 요약

| 개념 | 설명 | 핵심 공식/방법 |
|------|------|--------------|
| **UV 좌표(UV Coordinates)** | 3D 표면의 2D 파라미터화 | 아티스트 정의 또는 자동 언래핑 |
| **최근접 샘플링(Nearest Sampling)** | 가장 가까운 텍셀로 스냅 | 빠르지만 블록 형태 |
| **쌍선형 샘플링(Bilinear Sampling)** | 주변 4개 텍셀 블렌드 | $c = \text{lerp}(\text{lerp}(c_{00}, c_{10}), \text{lerp}(c_{01}, c_{11}))$ |
| **밉맵(Mipmaps)** | 미리 계산된 해상도 피라미드 | $\frac{1}{3}$ 추가 메모리 |
| **삼선형(Trilinear)** | 2개 밉 레벨에서 쌍선형 + 블렌드 | 밉 전환 제거 |
| **이방성(Anisotropic)** | 길게 늘어난 발자국을 따라 여러 샘플 | 최대 16x 품질 |
| **노멀 매핑(Normal Mapping)** | RGB 텍스처로 법선 변형 | TBN 행렬로 월드 공간 변환 |
| **변위 매핑(Displacement)** | 높이맵으로 실제로 꼭짓점 이동 | 테셀레이션 필요 |
| **PBR 텍스처(PBR Textures)** | 알베도 + 메탈릭 + 러프니스 + AO + 노멀 | 완전한 재질 정의 |
| **환경 맵(Environment Maps)** | 반사를 위한 큐브맵 또는 등직사각형 | 반사 벡터로 샘플링 |

**핵심 정리**:
- UV 좌표는 3D 기하와 2D 텍스처 사이의 다리다
- 쌍선형 필터링은 최근접 샘플링의 블록 형태 외관을 제거한다
- 밉맵은 최소한의 메모리 비용으로 먼 표면의 텍스처 앨리어싱을 해결한다
- 노멀 매핑은 기하학적 비용 없이 시각적 디테일을 추가한다
- PBR은 물리적으로 정확한 재질을 위해 여러 조율된 텍스처 맵을 사용한다
- 환경 맵은 현실적인 반사와 이미지 기반 조명을 가능하게 한다

---

## 연습 문제

1. **쌍선형 vs 최근접**: 4×4 체커보드 텍스처를 만들고 최근접과 쌍선형 샘플링으로 16배 확대하라. 두 결과를 나란히 비교하라. 어느 것이 날카로운 체커보드 패턴을 보존하는가? 어느 것이 더 부드럽게 보이는가?

2. **밉맵 시각화**: 256×256 텍스처에 대한 밉맵 체인을 생성하라. 모든 밉맵 레벨을 나란히 표시하라. 각 레벨을 다른 색으로 색칠하고 (레벨 0은 빨강, 레벨 1은 초록, 레벨 2는 파랑 등) 텍스처가 입혀진 바닥을 렌더링하여 각 거리에서 어떤 밉맵 레벨이 사용되는지 시각화하라.

3. **노멀 맵 생성**: 높이맵(회색조 이미지)이 주어졌을 때, $u$ 및 $v$ 방향의 유한 차분 그라디언트(Finite Difference Gradient)를 취하여 노멀 맵을 계산하라. 결과 법선 벡터를 $[0, 1]$ RGB 인코딩으로 변환하라. 계산된 노멀 맵을 사용하여 평면 표면을 렌더링해 결과를 테스트하라.

4. **UV 래핑 모드**: UV 좌표가 $(-1, -1)$에서 $(2, 2)$까지인 쿼드를 세 가지 래핑 모드(반복, 클램프, 미러 반복) 모두로 렌더링하라. 각 모드가 언제 적합한지 설명하라.

5. **원근 보정(Perspective Correction)**: 체커보드 텍스처를 사용하여 각도에서 보는 텍스처가 입혀진 쿼드(두 삼각형)를 원근 보정 UV 보간 유무 모두로 렌더링하라. 시각적 차이를 설명하라.

6. **환경 반사**: 등직사각형 HDR 이미지를 사용하여 간단한 환경 매핑 구를 구현하라. 러프니스 파라미터를 변화시킬 때 반사가 어떻게 변하는지 보여라 (힌트: 거친 표면에는 환경 맵의 흐려진 버전을 사용하라).

---

## 더 읽을거리

1. Marschner, S. & Shirley, P. *Fundamentals of Computer Graphics* (5th ed.), Ch. 11 — "Texture Mapping"
2. Akenine-Moller, T. et al. *Real-Time Rendering* (4th ed.), Ch. 6 — "Texturing"
3. [Learn OpenGL - Textures](https://learnopengl.com/Getting-started/Textures) — 실용적인 OpenGL 텍스처 튜토리얼
4. [Blinn, J. "Simulation of Wrinkled Surfaces" (1978)](https://www.microsoft.com/en-us/research/publication/simulation-of-wrinkled-surfaces/) — 원본 범프 매핑 논문
5. [Filament: Image-Based Lighting](https://google.github.io/filament/Filament.html#toc4.4) — 실제 PBR 환경 매핑
