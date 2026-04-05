# 04. 래스터화

[&larr; 이전: 03. 3D 변환과 투영](03_3D_Transformations_and_Projections.md) | [다음: 05. 셰이딩 모델 →](05_Shading_Models.md)

---

## 학습 목표

1. DDA와 브레젠험(Bresenham) 알고리즘을 이용한 선 래스터화(Line rasterization)를 이해한다
2. 에지 함수(Edge function)와 무게중심 좌표(Barycentric coordinates)로 삼각형을 래스터화한다
3. 은면 제거(Hidden surface removal)를 위한 Z-버퍼(Depth buffer) 알고리즘을 구현한다
4. 삼각형 표면 전체에 걸쳐 정점 속성(색상, UV, 법선)을 보간(Interpolation)한다
5. 원근 보정 보간(Perspective-correct interpolation)과 그 필요성을 설명한다
6. 안티에일리어싱(Anti-aliasing) 기법인 MSAA와 FXAA를 개념적 수준에서 이해한다
7. Python으로 완전한 소프트웨어 래스터라이저를 처음부터 직접 만든다
8. CPU 래스터화(학습용)와 GPU 래스터화(성능용)의 차이를 이해한다

---

## 왜 이것이 중요한가

래스터화(Rasterization)는 연속적이고 수학적인 기하 세계와 여러분의 화면이 사용하는 이산적이고 픽셀 기반의 세계를 잇는 다리다. 3D 게임이나 시각화에서 보이는 모든 삼각형은 래스터화 과정을 거쳤다. 즉, 세 개의 정점 위치에서 색칠된 픽셀 집합으로 변환된 것이다. 래스터화를 깊이 이해하면 그래픽 API가 왜 그런 방식으로 동작하는지, 특정 아티팩트(에일리어싱(Aliasing), Z-파이팅(Z-fighting))가 왜 나타나는지, GPU가 어떻게 엄청난 처리량을 달성하는지를 알 수 있다. 소프트웨어 래스터라이저를 직접 만들어보는 것은 컴퓨터 그래픽스 교육에서 가장 인상적인 연습 중 하나다.

---

## 1. 삼각형에서 픽셀로

정점 처리 단계(레슨 02~03) 이후, 스크린 공간(Screen space) 정점 위치로 정의된 삼각형이 준비된다. 래스터화는 다음 질문에 답한다. **각 삼각형이 어떤 픽셀을 덮는가, 그리고 그 픽셀의 색은 무엇인가?**

```
Input:  Triangle with 3 screen-space vertices
        Each vertex has: position (x, y, z), color, UV, normal, ...

Process: For each pixel in the triangle's bounding box:
           Is this pixel inside the triangle?
           If yes: interpolate vertex attributes, generate a fragment

Output: Stream of fragments (candidate pixels with interpolated attributes)
```

더 단순한 경우인 선(Line)부터 시작해, 삼각형으로 점차 확장한다.

---

## 2. 선 래스터화(Line Rasterization)

### 2.1 DDA(Digital Differential Analyzer)

가장 단순한 선 그리기 알고리즘이다. 끝점 $(x_0, y_0)$와 $(x_1, y_1)$이 주어질 때:

1. 기울기 계산: $m = \frac{y_1 - y_0}{x_1 - x_0}$
2. $|m| \leq 1$이면 $x$ 방향으로, $|m| > 1$이면 $y$ 방향으로 이동
3. 각 단계마다 나머지 좌표를 가장 가까운 정수로 반올림

```python
import numpy as np

def draw_line_dda(x0, y0, x1, y1):
    """
    DDA line rasterization.

    Simple and intuitive, but uses floating-point arithmetic at every step.
    Bresenham's algorithm (below) avoids this with integer-only operations.
    """
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))

    if steps == 0:
        return [(round(x0), round(y0))]

    # How much to increment x and y at each step
    x_inc = dx / steps
    y_inc = dy / steps

    pixels = []
    x, y = x0, y0

    for _ in range(int(steps) + 1):
        pixels.append((round(x), round(y)))
        x += x_inc
        y += y_inc

    return pixels
```

### 2.2 브레젠험 선 알고리즘(Bresenham's Line Algorithm)

브레젠험 알고리즘은 오차 누산기(Error accumulator)를 사용해 부동소수점 연산을 전혀 사용하지 않는, 정수 전용 선 래스터화의 고전적 방법이다.

**핵심 아이디어**: 각 단계에서 두 후보 픽셀 중 하나를 선택한다. 어떤 픽셀 중심이 이상적인 선에 더 가까운지를 기준으로 결정한다.

기울기 $0 \leq m \leq 1$인 선에 대한 알고리즘 (나머지 8개 옥턴트(Octant)는 대칭으로 처리):

```python
def draw_line_bresenham(x0, y0, x1, y1):
    """
    Bresenham's line algorithm -- integer arithmetic only.

    Why this matters: in the early days of graphics, floating-point
    operations were extremely expensive. Bresenham's insight was that
    the decision at each pixel can be made with only integer addition
    and comparison. Modern GPUs still use variants of this idea.
    """
    pixels = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1  # Step direction for x
    sy = 1 if y0 < y1 else -1  # Step direction for y

    # Determine which axis is "steep" (more pixels along that axis)
    steep = dy > dx
    if steep:
        dx, dy = dy, dx

    # Decision variable: starts at 2*dy - dx
    # Positive means step in the minor axis direction
    error = 2 * dy - dx

    x, y = x0, y0

    for _ in range(dx + 1):
        pixels.append((x, y))

        if error > 0:
            # Step in both major and minor directions
            if steep:
                x += sx
            else:
                y += sy
            error -= 2 * dx

        # Always step in the major direction
        if steep:
            y += sy
        else:
            x += sx
        error += 2 * dy

    return pixels


# Demonstration
print("DDA line from (0,0) to (8,3):")
print(draw_line_dda(0, 0, 8, 3))

print("\nBresenham line from (0,0) to (8,3):")
print(draw_line_bresenham(0, 0, 8, 3))
```

### 2.3 비교

| 항목 | DDA | 브레젠험(Bresenham) |
|------|-----|---------------------|
| 연산 방식 | 부동소수점 | 정수 전용 |
| 속도 | 느림 (부동소수점 나눗셈) | 빠름 (덧셈만 사용) |
| 정확도 | 반올림 오차 누적 | 정확 |
| 구현 복잡도 | 더 단순 | 약간 복잡 |

---

## 3. 삼각형 래스터화(Triangle Rasterization)

삼각형은 3D 그래픽스의 기본 프리미티브(Primitive)다. 거의 모든 3D 메시(Mesh)가 삼각형으로 구성되는 이유:
- 어떤 다각형도 삼각형으로 분해(Decompose)할 수 있다
- 삼각형은 항상 평면적이다 (3개의 점은 하나의 평면을 정의한다)
- 삼각형 래스터화는 GPU에서 고도로 병렬화할 수 있다

### 3.1 에지 함수(Edge Functions)

**에지 함수** 방식은 현대 GPU가 사용하는 방법이다. 스크린 공간에서 정점 $A$, $B$, $C$로 이루어진 삼각형에 대해 세 개의 에지 함수를 정의한다.

$$E_{AB}(P) = (P_x - A_x)(B_y - A_y) - (P_y - A_y)(B_x - A_x)$$

$$E_{BC}(P) = (P_x - B_x)(C_y - B_y) - (P_y - B_y)(C_x - B_x)$$

$$E_{CA}(P) = (P_x - C_x)(A_y - C_y) - (P_y - C_y)(A_x - C_x)$$

각 에지 함수는 에지 벡터와 에지 시작점에서 점 $P$로의 벡터가 만드는 평행사변형의 부호 있는 면적(Signed area)을 계산한다. 부호는 점이 에지의 어느 쪽에 있는지를 알려준다.

세 에지 함수 값이 모두 같은 부호(권선 순서(Winding order)에 따라 모두 양수 또는 모두 음수)이면, 점 $P$는 **삼각형 내부**에 있다.

> **왜 에지 함수를 사용하는가?** 각 픽셀의 테스트가 다른 픽셀과 독립적이므로 자연스럽게 병렬화된다. GPU는 수많은 픽셀의 에지 함수를 동시에 평가한다.

```python
def edge_function(a, b, p):
    """
    Compute the edge function for edge A->B evaluated at point P.

    Returns:
        Positive if P is to the left of edge A->B (CCW winding)
        Zero if P is exactly on the edge
        Negative if P is to the right

    This is equivalent to the z-component of the cross product
    (B-A) x (P-A), which is twice the signed area of triangle ABP.
    """
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
```

### 3.2 무게중심 좌표(Barycentric Coordinates)

세 에지 함수 값은 **무게중심 좌표**를 직접 제공한다.

$$\alpha = \frac{E_{BC}(P)}{E_{BC}(A)}, \quad \beta = \frac{E_{CA}(P)}{E_{CA}(B)}, \quad \gamma = \frac{E_{AB}(P)}{E_{AB}(C)}$$

$E_{BC}(A)$가 삼각형 $ABC$ 넓이의 두 배이므로 정규화(Normalization)할 수 있다.

$$\text{area} = E_{BC}(A) = (A_x - B_x)(C_y - B_y) - (A_y - B_y)(C_x - B_x)$$

$$\alpha = \frac{E_{BC}(P)}{\text{area}}, \quad \beta = \frac{E_{CA}(P)}{\text{area}}, \quad \gamma = 1 - \alpha - \beta$$

무게중심 좌표는 주어진 점에서 각 정점이 얼마나 영향을 미치는지를 나타내는 "가중치"다.
- 정점 $A$에서: $(\alpha, \beta, \gamma) = (1, 0, 0)$
- 무게중심(Centroid)에서: $(\alpha, \beta, \gamma) = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$
- 삼각형 외부: 하나 이상의 좌표가 음수

### 3.3 기본 삼각형 래스터화

```python
def rasterize_triangle_basic(v0, v1, v2, width, height):
    """
    Rasterize a triangle using edge functions and barycentric coordinates.

    Parameters:
        v0, v1, v2: vertex positions as (x, y) tuples
        width, height: framebuffer dimensions

    Returns:
        List of (x, y, alpha, beta, gamma) for each covered pixel
    """
    fragments = []

    # Compute bounding box (no need to test pixels outside it)
    min_x = max(0, int(min(v0[0], v1[0], v2[0])))
    max_x = min(width - 1, int(max(v0[0], v1[0], v2[0])) + 1)
    min_y = max(0, int(min(v0[1], v1[1], v2[1])))
    max_y = min(height - 1, int(max(v0[1], v1[1], v2[1])) + 1)

    # Twice the signed area of the triangle (used for normalization)
    area = edge_function(v0, v1, v2)

    if abs(area) < 1e-10:
        return fragments  # Degenerate triangle (zero area)

    # Test each pixel center in the bounding box
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = (x + 0.5, y + 0.5)  # Pixel center

            # Evaluate edge functions
            w0 = edge_function(v1, v2, p)  # Opposite vertex v0
            w1 = edge_function(v2, v0, p)  # Opposite vertex v1
            w2 = edge_function(v0, v1, p)  # Opposite vertex v2

            # Inside test: all edge functions must be non-negative (CCW winding)
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Compute barycentric coordinates
                alpha = w0 / area
                beta = w1 / area
                gamma = w2 / area  # = 1 - alpha - beta

                fragments.append((x, y, alpha, beta, gamma))

    return fragments
```

### 3.4 채우기 규칙(Fill Rules)

삼각형이 에지를 공유할 때(메시에서 흔한 경우), 정확히 에지 위의 픽셀을 두 삼각형이 모두 요청할 수 있다. **상단-좌단 채우기 규칙(Top-left fill rule)**으로 이를 해결한다.
- **상단 에지(Top edge)** 위의 픽셀 (삼각형 상단의 수평 에지): 해당 삼각형에 속함
- **좌단 에지(Left edge)** 위의 픽셀 (위쪽으로 향하는 에지): 해당 삼각형에 속함
- 그 외 에지의 픽셀은 제외

이를 통해 공유 에지의 모든 픽셀이 정확히 한 번만 그려진다.

---

## 4. Z-버퍼 알고리즘(Z-Buffer Algorithm)

### 4.1 은면 문제(Hidden Surface Problem)

같은 픽셀에 여러 삼각형이 겹칠 때, 어떤 것이 보여야 하는지 결정해야 한다. **Z-버퍼**(깊이 버퍼, Depth buffer)는 우아하리만큼 단순한 해결책이다.

### 4.2 알고리즘

최대 깊이 값(원평면)으로 초기화된 2D 배열 `depth_buffer[x][y]`를 유지한다. 각 프래그먼트(Fragment)에 대해:

$$\text{if } z_{\text{fragment}} < \text{depth\_buffer}[x][y]: \quad \text{색상 버퍼와 깊이 버퍼를 모두 업데이트}$$

$z$가 가장 작은(카메라에 가장 가까운) 프래그먼트가 최종 색상이 된다.

```python
def rasterize_with_zbuffer(triangles, width, height):
    """
    Complete rasterizer with Z-buffer hidden surface removal.

    Parameters:
        triangles: list of (v0, v1, v2) where each vertex is
                   {'pos': (x, y, z), 'color': (r, g, b)}
        width, height: framebuffer dimensions

    Returns:
        color_buffer: HxW array of RGB colors
        depth_buffer: HxW array of depth values
    """
    # Initialize buffers
    color_buffer = np.zeros((height, width, 3), dtype=float)
    depth_buffer = np.full((height, width), float('inf'))

    for tri_idx, (v0, v1, v2) in enumerate(triangles):
        p0, p1, p2 = v0['pos'], v1['pos'], v2['pos']
        c0, c1, c2 = v0['color'], v1['color'], v2['color']

        # Bounding box
        min_x = max(0, int(min(p0[0], p1[0], p2[0])))
        max_x = min(width - 1, int(max(p0[0], p1[0], p2[0])) + 1)
        min_y = max(0, int(min(p0[1], p1[1], p2[1])))
        max_y = min(height - 1, int(max(p0[1], p1[1], p2[1])) + 1)

        area = edge_function(p0[:2], p1[:2], p2[:2])
        if abs(area) < 1e-10:
            continue

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = (x + 0.5, y + 0.5)

                w0 = edge_function((p1[0], p1[1]), (p2[0], p2[1]), p)
                w1 = edge_function((p2[0], p2[1]), (p0[0], p0[1]), p)
                w2 = edge_function((p0[0], p0[1]), (p1[0], p1[1]), p)

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Barycentric coordinates
                    alpha = w0 / area
                    beta = w1 / area
                    gamma = w2 / area

                    # Interpolate depth
                    z = alpha * p0[2] + beta * p1[2] + gamma * p2[2]

                    # Z-buffer test: keep the closest fragment
                    if z < depth_buffer[y, x]:
                        depth_buffer[y, x] = z

                        # Interpolate color
                        r = alpha * c0[0] + beta * c1[0] + gamma * c2[0]
                        g = alpha * c0[1] + beta * c1[1] + gamma * c2[1]
                        b = alpha * c0[2] + beta * c1[2] + gamma * c2[2]
                        color_buffer[y, x] = [r, g, b]

    return color_buffer, depth_buffer
```

### 4.3 Z-버퍼 특성

| 특성 | 설명 |
|------|------|
| **메모리** | 픽셀당 하나의 깊이 값 (보통 24비트 또는 32비트) |
| **순서 독립적** | 불투명(Opaque) 오브젝트의 경우 어떤 순서로 그려도 무방 |
| **단순함** | 프래그먼트당 비교와 쓰기만 수행 |
| **GPU 친화적** | 고도로 병렬화 가능 |
| **약점** | 투명도(Transparency) 처리 불가 (정렬 필요) |

### 4.4 Z-파이팅(Z-Fighting)

두 표면의 깊이가 매우 가까울 때, 부동소수점 정밀도 문제로 어떤 표면이 깊이 테스트를 "이기는지"가 번갈아 바뀌면서 깜박이는 줄무늬 패턴이 나타난다. 이를 **Z-파이팅**이라 한다.

완화 방법:
- 근평면 거리를 늘림 (깊이 정밀도 향상)
- 폴리곤 오프셋(Polygon offset)으로 겹치는 표면을 약간 분리
- 로그 깊이 버퍼 또는 역 Z 깊이 버퍼 사용

---

## 5. 속성 보간(Attribute Interpolation)

### 5.1 선형 보간(Linear Interpolation)

무게중심 좌표 $(\alpha, \beta, \gamma)$와 정점별 속성 값이 주어지면, 어떤 속성이든 보간할 수 있다.

$$\text{attr}(P) = \alpha \cdot \text{attr}_0 + \beta \cdot \text{attr}_1 + \gamma \cdot \text{attr}_2$$

다음에 적용 가능하다:
- 색상 (부드러운 구로 셰이딩(Gouraud shading))
- 텍스처 좌표 (UV 매핑)
- 법선 (부드러운 조명을 위한 퐁 보간(Phong interpolation))
- 깊이 값

### 5.2 원근 보정 보간(Perspective-Correct Interpolation)

여기에는 미묘하지만 중요한 문제가 있다. **스크린 공간에서의 선형 보간은 원근 투영에서 올바르지 않다.** 원근 투영 후에는 스크린 공간에서의 동일한 보폭이 월드 공간에서의 동일한 보폭에 대응하지 않는다 (멀리 있는 오브젝트는 더 많이 압축된다).

올바른 수식은 각 속성을 정점의 $w$로 나눈 후 보간하고, 결과를 다시 나누는 것이다.

$$\text{attr}_{\text{correct}}(P) = \frac{\alpha \cdot \frac{\text{attr}_0}{w_0} + \beta \cdot \frac{\text{attr}_1}{w_1} + \gamma \cdot \frac{\text{attr}_2}{w_2}}{\alpha \cdot \frac{1}{w_0} + \beta \cdot \frac{1}{w_1} + \gamma \cdot \frac{1}{w_2}}$$

여기서 $w_0, w_1, w_2$는 세 정점의 클립 공간 위치에서의 $w$ 성분이다.

```python
def perspective_correct_interpolation(alpha, beta, gamma,
                                       attr0, attr1, attr2,
                                       w0, w1, w2):
    """
    Perspective-correct attribute interpolation.

    Without this correction, textures appear to "swim" on surfaces
    when viewed at oblique angles -- a very noticeable artifact.

    The key insight: in screen space, equal pixel distances do NOT
    correspond to equal world-space distances due to perspective.
    Dividing by w before interpolation accounts for this non-linearity.
    """
    # Interpolate attr/w and 1/w separately
    attr_over_w = (alpha * attr0 / w0 +
                   beta * attr1 / w1 +
                   gamma * attr2 / w2)

    one_over_w = (alpha * (1.0 / w0) +
                  beta * (1.0 / w1) +
                  gamma * (1.0 / w2))

    # Recover the correctly interpolated attribute
    return attr_over_w / one_over_w
```

**시각적 비교**: 원근 투영으로 바라본 텍스처가 입혀진 사각형(Quad)에서:
- **보정 없이**: 체커보드 텍스처가 대각선을 따라 구부러지거나 왜곡되어 보임
- **보정 후**: 체커보드가 곧고 균등하게 간격을 두고 보임 (기대대로)

---

## 6. 완전한 소프트웨어 래스터라이저

```python
"""
A complete software rasterizer demonstrating the concepts from this lesson.

This renders a colored triangle (or multiple triangles) to a pixel buffer
using edge functions, barycentric coordinates, Z-buffer, and
perspective-correct interpolation.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# Framebuffer class
# ═══════════════════════════════════════════════════════════════

class Framebuffer:
    """Manages color and depth buffers."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Color buffer: RGBA (4 channels), float [0, 1]
        self.color = np.zeros((height, width, 4), dtype=float)
        self.color[:, :, 3] = 1.0  # Alpha = 1 (opaque background)
        # Depth buffer: initialized to far plane (1.0 in NDC)
        self.depth = np.ones((height, width), dtype=float)

    def clear(self, color=(0.1, 0.1, 0.1)):
        """Clear buffers to default values."""
        self.color[:, :, 0] = color[0]
        self.color[:, :, 1] = color[1]
        self.color[:, :, 2] = color[2]
        self.depth[:] = 1.0

    def set_pixel(self, x, y, z, color):
        """
        Write a pixel if it passes the depth test.

        Returns True if the pixel was written (passed depth test).
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        if z < self.depth[y, x]:
            self.depth[y, x] = z
            self.color[y, x, :3] = np.clip(color, 0, 1)
            return True
        return False

    def to_image(self):
        """Convert to uint8 image for display/saving."""
        return (np.clip(self.color[:, :, :3], 0, 1) * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# Vertex structure
# ═══════════════════════════════════════════════════════════════

class Vertex:
    """A vertex with position, color, UV, and clip-space w."""
    def __init__(self, x, y, z, w=1.0, r=1.0, g=1.0, b=1.0, u=0.0, v=0.0):
        self.pos = np.array([x, y, z], dtype=float)
        self.w = w  # Clip-space w (needed for perspective-correct interpolation)
        self.color = np.array([r, g, b], dtype=float)
        self.uv = np.array([u, v], dtype=float)


# ═══════════════════════════════════════════════════════════════
# Rasterizer
# ═══════════════════════════════════════════════════════════════

def edge_function(a, b, p):
    """Signed area of parallelogram formed by edge A->B and point P."""
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def rasterize_triangle(fb, v0, v1, v2, perspective_correct=True):
    """
    Rasterize a single triangle with Z-buffer and attribute interpolation.

    This is the core of the software rasterizer. A GPU executes this logic
    for millions of triangles per frame using thousands of parallel cores.

    Parameters:
        fb: Framebuffer instance
        v0, v1, v2: Vertex instances (screen-space positions)
        perspective_correct: use perspective-correct interpolation
    """
    p0, p1, p2 = v0.pos[:2], v1.pos[:2], v2.pos[:2]

    # Bounding box (clamped to screen)
    min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
    max_x = min(fb.width - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
    min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
    max_y = min(fb.height - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

    # Total signed area (for barycentric normalization)
    area = edge_function(p0, p1, p2)
    if abs(area) < 1e-10:
        return  # Skip degenerate triangles

    # Precompute 1/w for perspective correction
    inv_w0 = 1.0 / v0.w if perspective_correct else 1.0
    inv_w1 = 1.0 / v1.w if perspective_correct else 1.0
    inv_w2 = 1.0 / v2.w if perspective_correct else 1.0

    fragment_count = 0

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # Test pixel center
            px, py = x + 0.5, y + 0.5

            # Edge function evaluations
            w0 = edge_function(p1, p2, (px, py))
            w1 = edge_function(p2, p0, (px, py))
            w2 = edge_function(p0, p1, (px, py))

            # Inside test (all non-negative for CCW winding)
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Barycentric coordinates
                alpha = w0 / area
                beta = w1 / area
                gamma = w2 / area

                if perspective_correct:
                    # Perspective-correct interpolation
                    one_over_w = alpha * inv_w0 + beta * inv_w1 + gamma * inv_w2

                    # Interpolate color with perspective correction
                    color = (alpha * v0.color * inv_w0 +
                             beta * v1.color * inv_w1 +
                             gamma * v2.color * inv_w2) / one_over_w

                    # Interpolate depth with perspective correction
                    z = (alpha * v0.pos[2] * inv_w0 +
                         beta * v1.pos[2] * inv_w1 +
                         gamma * v2.pos[2] * inv_w2) / one_over_w
                else:
                    # Simple linear interpolation (incorrect for perspective)
                    color = (alpha * v0.color +
                             beta * v1.color +
                             gamma * v2.color)
                    z = (alpha * v0.pos[2] +
                         beta * v1.pos[2] +
                         gamma * v2.pos[2])

                # Depth test and pixel write
                fb.set_pixel(x, y, z, color)
                fragment_count += 1

    return fragment_count


# ═══════════════════════════════════════════════════════════════
# Demo: render two overlapping colored triangles
# ═══════════════════════════════════════════════════════════════

def main():
    width, height = 320, 240
    fb = Framebuffer(width, height)
    fb.clear(color=(0.05, 0.05, 0.1))  # Dark blue background

    # Triangle 1: Red-Green-Blue, closer to camera (z=0.3)
    t1_v0 = Vertex(80,  30,  0.3, r=1, g=0, b=0)  # Red (top)
    t1_v1 = Vertex(30,  200, 0.3, r=0, g=1, b=0)  # Green (bottom-left)
    t1_v2 = Vertex(200, 180, 0.3, r=0, g=0, b=1)  # Blue (bottom-right)

    # Triangle 2: Yellow-Cyan-Magenta, farther from camera (z=0.6)
    t2_v0 = Vertex(160, 20,  0.6, r=1, g=1, b=0)  # Yellow (top)
    t2_v1 = Vertex(100, 220, 0.6, r=0, g=1, b=1)  # Cyan (bottom-left)
    t2_v2 = Vertex(290, 150, 0.6, r=1, g=0, b=1)  # Magenta (bottom-right)

    # Render: draw the far triangle first, then the near triangle
    # Z-buffer ensures correct occlusion regardless of draw order
    n1 = rasterize_triangle(fb, t2_v0, t2_v1, t2_v2, perspective_correct=False)
    n2 = rasterize_triangle(fb, t1_v0, t1_v1, t1_v2, perspective_correct=False)

    print(f"Triangle 1 (far): {n1} fragments")
    print(f"Triangle 2 (near): {n2} fragments")
    print(f"Total pixels shaded: {n1 + n2}")
    print(f"Pixels that passed depth test: {np.sum(fb.depth < 1.0)}")

    # Save result
    try:
        from PIL import Image
        img = Image.fromarray(fb.to_image())
        img.save('rasterizer_output.png')
        print("Saved rasterizer_output.png")
    except ImportError:
        print("Install Pillow to save the image: pip install Pillow")
        print("Framebuffer shape:", fb.color.shape)


if __name__ == "__main__":
    main()
```

---

## 7. 안티에일리어싱(Anti-Aliasing)

### 7.1 에일리어싱 문제(Aliasing Problem)

래스터화는 **에일리어싱** 아티팩트를 만든다. 삼각형 경계를 따라 나타나는 들쭉날쭉한 가장자리("계단(Staircase)" 효과)가 그것이다. 이는 이산적인 픽셀 위치에서 연속적인 형태를 샘플링하기 때문에 발생한다. 픽셀 그리드는 대각선이나 곡선 가장자리를 완벽하게 표현하지 못한다.

수학적으로, 에일리어싱은 **나이퀴스트 표본화 정리(Nyquist sampling theorem)**를 위반한 것이다. 기하 신호(삼각형의 날카로운 가장자리)는 임의로 높은 주파수를 포함하지만, 픽셀 그리드는 고정된 속도로 샘플링한다.

### 7.2 슈퍼샘플링(SSAA, Supersampling)

무차별 대입 방식의 해결책: 더 높은 해상도로 렌더링한 다음 다운샘플링한다.

4x SSAA의 경우: $2W \times 2H$로 렌더링한 후, 모든 $2 \times 2$ 블록을 평균 내어 하나의 픽셀로 만든다. 매우 효과적이지만 비용이 크다 (프래그먼트 셰이더(Fragment shader) 비용의 4배).

### 7.3 다중 샘플 안티에일리어싱(MSAA, Multi-Sample Anti-Aliasing)

**MSAA**는 슈퍼샘플링의 더 스마트한 버전이다. 모든 서브샘플에서 프래그먼트 셰이더를 실행하는 대신:

1. 각 픽셀 내의 여러 샘플 위치에서 **커버리지(Coverage)**를 테스트한다 (예: 4x MSAA의 경우 4개 위치)
2. 픽셀당 **프래그먼트 셰이더를 단 한 번만** 실행한다 (픽셀 중심에서)
3. 셰이더 결과를 커버된 모든 서브샘플에 기록한다
4. 해결(Resolve): 서브샘플을 평균 내어 최종 픽셀 색상을 만든다

```
┌─────────────────────┐
│  Pixel with 4x MSAA │
│                      │
│   ●         ●       │   ● = sample point
│       ◆             │   ◆ = pixel center (shader runs here)
│                      │
│   ●         ●       │   Triangle covers 3 of 4 samples:
│                      │   Final color = 75% triangle + 25% background
└─────────────────────┘
```

**MSAA가 효율적인 이유**: 비용이 많이 드는 프래그먼트 셰이더는 픽셀당 한 번만 실행되고, 샘플당 한 번씩 실행되지 않는다. 저렴한 커버리지 테스트만 샘플별로 수행된다.

```python
def rasterize_triangle_msaa(fb, v0, v1, v2, samples=4):
    """
    Triangle rasterization with MSAA (Multi-Sample Anti-Aliasing).

    MSAA evaluates coverage at multiple sub-pixel positions but
    runs the shader only once. The final pixel color is the average
    weighted by coverage.

    This is a simplified demonstration -- real MSAA implementations
    store per-sample depth and color in a larger buffer that is
    "resolved" (averaged) at the end of the frame.
    """
    # 4x MSAA sample pattern (Rotated Grid pattern)
    # These offsets are within the pixel, relative to pixel center
    if samples == 4:
        sample_offsets = [
            (-0.25, -0.125),
            (0.25, -0.375),
            (-0.125, 0.375),
            (0.375, 0.125)
        ]
    else:
        # Fallback: regular grid
        n = int(np.sqrt(samples))
        step = 1.0 / (n + 1)
        sample_offsets = [(step * (i + 1) - 0.5, step * (j + 1) - 0.5)
                          for i in range(n) for j in range(n)]

    p0, p1, p2 = v0.pos[:2], v1.pos[:2], v2.pos[:2]

    min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
    max_x = min(fb.width - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
    min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
    max_y = min(fb.height - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

    area = edge_function(p0, p1, p2)
    if abs(area) < 1e-10:
        return

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # Test coverage at each sample point
            covered = 0
            for dx, dy in sample_offsets:
                sx, sy = x + 0.5 + dx, y + 0.5 + dy
                w0 = edge_function(p1, p2, (sx, sy))
                w1 = edge_function(p2, p0, (sx, sy))
                w2 = edge_function(p0, p1, (sx, sy))
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    covered += 1

            if covered > 0:
                # Shade at pixel center (fragment shader runs once)
                px, py = x + 0.5, y + 0.5
                w0 = edge_function(p1, p2, (px, py))
                w1 = edge_function(p2, p0, (px, py))
                w2 = edge_function(p0, p1, (px, py))

                alpha = w0 / area
                beta = w1 / area
                gamma = 1 - alpha - beta

                color = (alpha * v0.color + beta * v1.color + gamma * v2.color)
                z = alpha * v0.pos[2] + beta * v1.pos[2] + gamma * v2.pos[2]

                # Weight by coverage fraction
                coverage = covered / len(sample_offsets)
                blended = color * coverage + fb.color[y, x, :3] * (1 - coverage)

                fb.set_pixel(x, y, z, blended)
```

### 7.4 FXAA(Fast Approximate Anti-Aliasing)

**FXAA**는 래스터화 도중이 아니라 최종 렌더링 이미지에 적용되는 **후처리(Post-processing)** 기법이다.

1. 인접 픽셀의 밝기(Luminance)를 비교해 가장자리를 감지
2. 감지된 가장자리를 따라 픽셀을 블렌딩하여 계단 현상을 완화
3. 빠름 (화면 전체를 한 번만 처리)하고 구현이 매우 간단
4. 어떤 렌더링 기법(Forward, Deferred, 레이 트레이싱(Ray tracing))과도 호환

**트레이드오프**: FXAA는 기하 가장자리(부드럽게 해야 하는 것)와 텍스처 가장자리(부드럽게 하면 안 되는 것)를 구분하지 못하기 때문에 세밀한 디테일이 흐려질 수 있다.

### 7.5 안티에일리어싱 비교

| 방식 | 품질 | 성능 비용 | 메모리 비용 | 비고 |
|------|------|-----------|-------------|------|
| 없음 | 들쭉날쭉한 가장자리 | 1x | 1x | 기준 |
| SSAA 4x | 탁월 | 4x | 4x | 무차별 방식 |
| MSAA 4x | 매우 좋음 | 셰이더 ~1.2x, 커버리지 4x | 깊이/색상 4x | GPU 하드웨어 지원 |
| FXAA | 양호 | ~1ms 후처리 | 추가 없음 | 일부 디테일 흐림 |
| TAA | 매우 좋음 | ~1ms + 히스토리 버퍼 | 색상 2x | 시간적 데이터 활용 |

---

## 8. 성능 고려 사항

### 8.1 GPU가 래스터화에서 빠른 이유

래스터화 알고리즘은 병렬화하기 매우 쉬운 구조(Embarrassingly parallel)를 갖는다.
- 각 삼각형은 독립적으로 래스터화할 수 있다
- 삼각형 내의 각 픽셀은 독립적으로 테스트할 수 있다
- 에지 함수 평가는 단순히 곱셈과 덧셈이다

현대 GPU는 화면을 **타일(Tile)**로 분할하고, 타일당 많은 삼각형을 병렬로 처리한다. 고사양 GPU는 초당 수십억 개의 삼각형을 래스터화할 수 있다.

### 8.2 실제 래스터라이저의 최적화 기법

- **계층적 래스터화(Hierarchical rasterization)**: 개별 픽셀 전에 픽셀 블록을 먼저 테스트
- **조기 깊이 거부(Early depth rejection)**: 이미 렌더링된 기하 뒤에 있는 프래그먼트를 스킵
- **SIMD 에지 함수 평가**: 에지 함수를 4, 8, 또는 16개 픽셀에 대해 동시에 평가
- **후면 컬링(Backface culling)**: 카메라에서 멀어지는 방향을 향하는 삼각형 건너뜀 (작업량 약 50% 절감)
- **타일 기반 렌더링(Tile-based rendering)**: 캐시 히트(Cache hit)를 최대화하기 위해 화면 타일 단위로 처리 (모바일 GPU에서 일반적)

---

## 요약

| 개념 | 설명 |
|------|------|
| **DDA** | 부동소수점 증분을 이용한 단순 선 그리기 |
| **브레젠험(Bresenham)** | 오차 누산을 이용한 정수 전용 선 그리기 |
| **에지 함수(Edge functions)** | 부호 있는 면적을 통해 점이 삼각형 내부에 있는지 판별 |
| **무게중심 좌표(Barycentric coords)** | 삼각형 전체에 걸쳐 정점 속성을 보간하기 위한 가중치 |
| **Z-버퍼(Z-buffer)** | 은면 제거를 위한 픽셀당 깊이 비교 |
| **원근 보정(Perspective-correct)** | 보간 전 속성을 $w$로 나누고, 이후 보정 |
| **MSAA** | 단일 셰이더 평가로 다중 샘플 커버리지 테스트 |
| **FXAA** | 밝기 감지를 기반으로 한 후처리 에지 블러 |

**핵심 정리**:
- 래스터화는 에지 함수를 사용해 연속적인 삼각형을 이산적인 프래그먼트로 변환한다
- 무게중심 좌표는 정점별 속성을 부드럽게 보간할 수 있게 해준다
- Z-버퍼는 픽셀당 $O(n)$ 비용으로 은면 제거를 우아하게 해결한다
- 원근 보정 보간은 올바른 텍스처 매핑에 필수적이다
- 안티에일리어싱은 성능과 부드러운 가장자리를 맞바꾼다. MSAA가 실시간 렌더링의 표준이다
- GPU는 래스터화에 내재된 대규모 병렬성을 활용해 실시간 성능을 달성한다

---

## 연습 문제

1. **브레젠험 구현**: 8개 옥턴트(Octant) 모두에서 동작하도록 브레젠험 알고리즘을 구현하라 (위의 버전은 제한된 경우만 처리한다). $0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°$ 방향의 선을 그려 올바르게 보이는지 검증하라.

2. **에지 함수 시각화**: 정점 $(50, 20)$, $(200, 150)$, $(30, 180)$으로 이루어진 삼각형에 대해 256x256 그리드 전체에서 에지 함수 값을 계산하고 시각화하라. 세 에지 함수에 서로 다른 색상을 사용하고, 각 함수가 양수/음수인 영역을 표시하라.

3. **Z-버퍼 아티팩트**: 두 삼각형이 서로 거의 동일 평면(Coplanar)에 있지만 Z 차이가 $0.0001$인 장면을 만들라. 16비트 vs 32비트 깊이 정밀도로 렌더링하여 Z-파이팅을 시연하라.

4. **원근 보정 보간**: 원근 보정 유무에 따라 텍스처가 입혀진 사각형(두 삼각형)을 렌더링하라. 체커보드 패턴을 사용하고 비스듬한 각도에서 사각형을 바라봐 차이를 명확히 확인하라.

5. **소프트웨어 래스터라이저 확장**: 완전한 래스터라이저를 다음과 같이 확장하라: (a) 와이어프레임(Wireframe) 렌더링 (삼각형 가장자리만 그리기), (b) 단순한 메시(예: 큐브)를 구성하는 여러 삼각형, (c) 삼각형 면 법선을 이용한 플랫 셰이딩(Flat shading).

6. **MSAA 분석**: 1x, 2x, 4x, 8x MSAA로 삼각형 가장자리를 렌더링하라. 각 경우에서 부분적인 커버리지(삼각형 내부도, 외부도 완전히 아닌)를 갖는 픽셀 수를 세어라. MSAA 배율에 따라 이 수가 어떻게 변하는지 논의하라.

---

## 추가 자료

1. Marschner, S. & Shirley, P. *Fundamentals of Computer Graphics* (5th ed.), Ch. 8 -- "Rasterization"
2. Pineda, J. "A Parallel Algorithm for Polygon Rasterization" (1988) -- 에지 함수 원본 논문
3. [Scratchapixel - Rasterization](https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation) -- 코드가 포함된 상세 튜토리얼
4. [A Trip through the Graphics Pipeline (Fabian Giesen)](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/) -- GPU 래스터화 방식에 대한 훌륭한 심층 분석
5. [Learn OpenGL - Anti Aliasing](https://learnopengl.com/Advanced-OpenGL/Anti-Aliasing) -- OpenGL로 실습하는 MSAA
