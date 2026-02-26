# 6. 재결합 응용

## 학습 목표

이 레슨을 마치면 다음을 할 수 있어야 합니다:

1. 태양 플레어의 CSHKP 모델을 설명하고 핵심 재결합 신호를 식별하기
2. 코로나 질량 방출(CME)과 자속 로프 분출에서 재결합의 역할 설명하기
3. Dungey 순환과 자기권 서브스톰 이해하기
4. 토카막 톱니파 충돌과 Kadomtsev 재결합 모델 분석하기
5. 자기 섬 병합과 그 동역학 설명하기
6. 천체물리학 제트와 기타 고에너지 현상에서의 재결합 설명하기
7. 이러한 재결합 응용의 간단한 모델을 Python으로 구현하기

## 1. 태양 플레어

### 1.1 관측 개요

태양 플레어는 태양계에서 가장 강력한 폭발로, 수 분에서 수 시간 동안 최대 $10^{32}$ erg ($10^{25}$ J)의 에너지를 방출합니다. 이것은 수십억 메가톤의 TNT에 해당하거나, 천만 번의 화산 폭발이 동시에 발생하는 것과 같습니다.

**주요 관측 특징:**

- **전자기 방출**: 전파에서 감마선까지
  - 연X선 방출: 10–30 MK의 열 플라즈마
  - 경X선 방출: 비열적 전자(제동복사)
  - H-alpha 리본: 족점에서의 채층 밝아짐
  - 백색광 방출: 드물며, 가장 에너지가 큰 플레어에서만 관측

- **입자 가속**: 전자는 수십 MeV까지, 이온은 GeV까지
  - 상대론적 전자: 전파 폭발(자기동기복사, 플라즈마 방출)
  - 에너지 양성자: 핵 감마선 선(탈여기)

- **질량 방출**: 종종 CME와 관련됨(항상은 아님)

- **시간 척도**:
  - 플레어 전 단계: 수 분에서 수 시간(에너지 저장)
  - 충격 단계: 수 초에서 수 분(에너지 방출)
  - 점진 단계: 수 분에서 수 시간(냉각, 점진적 입자 가속)

**에너지 수지:**

방출되는 총 에너지는 응력을 받은 코로나 자기장에 저장된 자기 에너지에서 나옵니다:

$$E_{mag} = \frac{B^2}{2\mu_0} \cdot V$$

$B \sim 0.01$ T, 부피 $V \sim (10^8 \text{ m})^3$인 플레어 활동 영역의 경우:

$$E_{mag} \sim \frac{(0.01)^2}{2 \times 4\pi \times 10^{-7}} \times 10^{24} \sim 10^{26} \text{ J}$$

이 자기 에너지의 상당한 부분(10–50%)이 플레어 동안 방출됩니다.

### 1.2 CSHKP 표준 모델

분출하는 태양 플레어의 표준 모델은 Carmichael, Sturrock, Hirayama, Kopp, Pneuman(CSHKP 모델, 1964–1976년에 개발)의 이름을 따서 명명되었습니다. 이 모델은 자기 재결합을 주요 에너지 방출 메커니즘으로 사용합니다.

**만화 구조:**

```
                        CME
                         ║
                    ╔════╩════╗
                    ║         ║  분출하는 자속 로프
                    ║    ☀    ║
                    ╚════╦════╝
                         ║
    ═══════════════════════════════════  코로나
                    X    ║  재결합점
                    ↕    ║  전류 시트
    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                         ║  플레어 후 루프
                    ┌────╨────┐
                    │         │  플레어 아케이드
                    │ ░░░░░░░ │  고온 플라즈마
                    └─────────┘
    ═══════════════════════════════════  채층
                    ↓         ↓  하향 흐름
               리본 1    리본 2  H-alpha 밝아짐
```

**주요 구성 요소:**

1. **분출 전**: 자속 로프(필라멘트/홍염)가 코로나에 저장되며, 위에 덮는 아케이드 자기장에 의해 고정됩니다.

2. **평형 상실**: 자속 로프가 불안정해지고(예: 토러스 불안정성, 평형 상실) 상승하기 시작합니다.

3. **전류 시트 형성**: 자속 로프가 상승함에 따라 아래 자기장이 늘어나며 수직 전류 시트를 형성합니다.

4. **재결합 시작**: 전류 시트에서 재결합이 시작되어 자기 에너지를 방출합니다.

5. **플레어 후 루프**: 재결합된 자기장 선은 순차적으로 상승하는 고온 플레어 후 루프를 형성합니다.

6. **족점 가열**: 에너지 입자와 열 전도 전선이 자기장 선을 따라 내려가며 채층을 가열합니다.

7. **플레어 리본**: 가열된 채층 족점은 재결합이 진행됨에 따라 분리되는 밝은 H-alpha 리본으로 나타납니다.

8. **상향 제트**: 재결합 유출이 CME를 위쪽으로 발사합니다.

**에너지 방출:**

자기 에너지는 다음으로 변환됩니다:
- **운동 에너지**: 대량 플라즈마 흐름(유출 ~500–1000 km/s), CME 운동 에너지
- **열 에너지**: 플레어 루프를 10–30 MK로 가열
- **비열적 입자**: 가속된 전자와 이온
- **복사**: X선, UV, 광학, 전파

재결합 전기장이 입자를 가속하고, 재결합 영역의 난류가 확률적 가속에 기여합니다.

### 1.3 플레어에서의 재결합률

재결합률은 **리본 분리 속도**로부터 추정할 수 있습니다. 재결합이 진행됨에 따라 새로 재결합된 루프의 족점이 멀어집니다. 리본 분리 속도 $v_{sep}$는 재결합 유입 속도 $v_{in}$와 관련이 있습니다:

$$v_{sep} \approx v_{in} \frac{L_{corona}}{L_{ribbon}}$$

여기서 $L_{corona}$는 코로나 높이이고 $L_{ribbon}$는 채층 족점 분리 거리입니다.

관측된 리본 속도는 일반적으로:

$$v_{sep} \sim 10\text{–}100 \text{ km/s}$$

$L_{corona}/L_{ribbon} \sim 0.1$–1인 경우, 다음을 얻습니다:

$$v_{in} \sim 10\text{–}100 \text{ km/s}$$

코로나의 Alfvén 속도는 $v_A \sim 1000$ km/s이므로:

$$M_A = \frac{v_{in}}{v_A} \sim 0.01\text{–}0.1$$

이것은 빠른 재결합(Petschek 또는 Hall 영역)과 일치하며, Sweet-Parker가 아닙니다!

### 1.4 SDO 및 기타 임무의 관측

2010년에 발사된 Solar Dynamics Observatory(SDO)는 고케이던스(12 s), 고해상도(0.6 arcsec) 극자외선 이미지로 플레어 관측을 혁신했습니다.

**주요 발견:**

- **Supra-arcade downflows (SADs)**: 플레어 아케이드로 ~100 km/s로 떨어지는 어둡고 올챙이 모양의 구조. 재결합 지점에서 하향으로 방출된 밀도 공백(플라스모이드)으로 해석됩니다.

- **Above-the-loop-top (ALT) 경X선 소스**: 플레어 루프 위의 비열적 X선 방출로, 재결합 지점에서의 입자 가속을 나타냅니다.

- **재결합 유입/유출**: Doppler 측정(Hinode/EIS)은 ~10 km/s의 유입과 ~500 km/s의 유출을 보여주며, 재결합 모델과 일치합니다.

- **준주기적 맥동(QPPs)**: 수 밀리초에서 수 분의 주기를 가진 플레어 방출의 진동. 플라스모이드 방출과 관련이 있을 수 있습니다(레슨 7 참조).

- **자기 자속 로프 구조**: 분출 전 시그모이드(S자 모양 구조) 및 자속 로프의 관측은 CSHKP 모델을 지원합니다.

### 1.5 Python 예제: CSHKP 모델 다이어그램

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge, Ellipse
from matplotlib import patches

fig, ax = plt.subplots(figsize=(12, 14))

# Coordinate system: x horizontal, y vertical
# Chromosphere at y=0, corona y>0

# Chromosphere
ax.axhline(0, color='brown', linewidth=4, label='Chromosphere')
ax.fill_between([-6, 6], -0.5, 0, color='wheat', alpha=0.5)

# Flare ribbons
ribbon1 = Wedge((-2, 0), 0.5, 0, 180, color='red', alpha=0.7, label='Flare ribbons')
ribbon2 = Wedge((2, 0), 0.5, 0, 180, color='red', alpha=0.7)
ax.add_patch(ribbon1)
ax.add_patch(ribbon2)

# Post-flare loops
n_loops = 5
for i in range(n_loops):
    y_top = 1 + i * 0.5
    x_width = 1.5 + i * 0.3
    theta = np.linspace(0, np.pi, 50)
    x_loop = x_width * np.cos(theta)
    y_loop = y_top * np.sin(theta)
    ax.plot(x_loop, y_loop, color='orange', linewidth=2.5, alpha=0.8)

# Label one loop
ax.text(0, 1.8, 'Post-flare loops', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# Current sheet
sheet_x = [0, 0]
sheet_y = [3.5, 7]
ax.plot(sheet_x, sheet_y, color='blue', linewidth=4, linestyle='--', label='Current sheet')

# X-point
ax.plot(0, 4.5, 'kx', markersize=25, markeredgewidth=4, label='Reconnection X-point')

# Inflow arrows
ax.annotate('', xy=(-0.3, 4.5), xytext=(-1.5, 4.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
ax.annotate('', xy=(0.3, 4.5), xytext=(1.5, 4.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
ax.text(-2.2, 4.5, 'Inflow', fontsize=11, color='green', weight='bold')

# Downward outflow
ax.annotate('', xy=(0, 3.8), xytext=(0, 3.2),
            arrowprops=dict(arrowstyle='->', lw=3, color='red'))
ax.text(0.3, 3.5, 'Outflow\n(downward)', fontsize=11, color='red', weight='bold')

# Upward outflow
ax.annotate('', xy=(0, 5.2), xytext=(0, 5.8),
            arrowprops=dict(arrowstyle='->', lw=3, color='red'))
ax.text(0.3, 5.5, 'Outflow\n(upward)', fontsize=11, color='red', weight='bold')

# Erupting flux rope (CME)
flux_rope = Ellipse((0, 8.5), 3, 1.5, color='purple', alpha=0.4, label='Erupting flux rope (CME)')
ax.add_patch(flux_rope)
ax.plot(0, 8.5, 'o', color='purple', markersize=10)

# CME upward arrow
ax.annotate('', xy=(0, 10), xytext=(0, 9.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
ax.text(0.5, 9.8, 'CME', fontsize=13, color='purple', weight='bold')

# Particle beams to chromosphere
for x_foot in [-2, 2]:
    ax.annotate('', xy=(x_foot, 0.1), xytext=(0, 4.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='magenta', linestyle='dotted'))
ax.text(-3.5, 2, 'Energetic\nparticles', fontsize=11, color='magenta', weight='bold')

# Labels
ax.text(0, -1, 'Solar Flare: CSHKP Standard Model', fontsize=18, ha='center', weight='bold')
ax.text(-5, 7, 'Corona', fontsize=13, style='italic')
ax.text(-5, -0.3, 'Chromosphere', fontsize=13, style='italic', color='brown')

# Annotations
ax.text(-5, 9, 'Energy release: ~$10^{32}$ erg', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(-5, 8.2, 'Duration: minutes to hours', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(-5, 7.4, 'Reconnection rate: $M_A \\sim 0.01$–$0.1$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

ax.set_xlim(-6, 6)
ax.set_ylim(-1.5, 11)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=11)
ax.axis('off')

plt.tight_layout()
plt.savefig('solar_flare_cshkp_model.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 2. 코로나 질량 방출(CME)

### 2.1 CME란 무엇인가?

**코로나 질량 방출**은 태양 플라즈마와 자기장의 대규모 폭발이 행성 간 공간으로 방출되는 것입니다. CME는 종종(항상은 아니지만) 태양 플레어와 관련이 있습니다.

**전형적인 특성:**

- **질량**: $10^{15}$–$10^{16}$ g(10억~100억 톤)
- **운동 에너지**: $10^{30}$–$10^{32}$ erg
- **속도**: 200–3000 km/s(평균 ~500 km/s, 빠른 이벤트 >1000 km/s)
- **발생률**: 태양 극대기에 ~하루 1회, 태양 극소기에 ~5일당 1회

**구조:**

CME는 종종 세 부분 구조를 가집니다:
1. **밝은 전면 루프**: 압축된 덮개
2. **어두운 공동**: 저밀도 자속 로프 코어
3. **밝은 코어**: 홍염 물질

### 2.2 CME 시작과 재결합

CME는 코로나 자기 자속 로프의 분출로부터 발생하는 것으로 믿어집니다. 시작 메커니즘은 다음을 포함합니다:

**1. 토러스 불안정성(Torus Instability):**

자속 로프는 위에 덮는 자기장이 높이에 따라 충분히 빠르게 감소할 때 불안정해집니다. 임계 조건은:

$$\frac{d \ln B_{external}}{d \ln h} < -\frac{3}{2}$$

여기서 $h$는 높이입니다. 이것을 **토러스 불안정성** 기준이라고 합니다. 이 조건이 충족되면, 후프 힘(자체 인덕턴스)이 구속력을 극복하고 자속 로프가 분출합니다.

**2. 자속 소멸(Flux Cancellation):**

광구에서 반대 극성 자기 자속이 소멸(재결합)하여, 위에 덮는 자기장의 "닻"을 제거하고 자속 로프가 분출할 수 있게 합니다.

**3. 돌파 모델(Breakout Model):**

4극 구성이 자속 로프 위의 영점에서 재결합을 겪어 구속 자기장을 제거하고 분출을 유발합니다.

**4. 킹크 불안정성(Kink Instability):**

자속 로프가 임계 임계값을 넘어 꼬이면(일반적으로 길이당 비틀림 $\sim 2\pi$ 라디안), 킹크 불안정해지고 분출합니다.

**재결합의 역할:**

분출 동안, CME 아래 전류 시트에서의 재결합(CSHKP 모델처럼)은 두 가지 기능을 수행합니다:
1. **자기 에너지 방출**: 분출과 플레어에 동력 제공
2. **자기장 위상 구조 변화 허용**: 자속 로프가 태양으로부터 분리될 수 있게 함

### 2.3 우주 날씨 영향

빠른 CME가 지구에 충돌하면 상당한 우주 날씨 영향을 일으킬 수 있습니다:

- **지자기 폭풍**: 압축된 자기권, 강화된 고리 전류, 오로라 활동
  - 가장 강한 폭풍: Carrington 이벤트(1859), 할로윈 폭풍(2003), St. Patrick's Day 폭풍(2015)

- **복사 위험**: 에너지 입자가 우주 비행사와 위성을 위험에 빠뜨림

- **기술 중단**:
  - 전력망 장애(Quebec 정전, 1989)
  - 위성 손상 및 손실
  - GPS 및 통신 중단
  - 항공 복사 노출(극지 항로)

태양에서 지구까지의 **이동 시간**은 일반적인 CME의 경우 1–3일이며, 보호 조치를 위한 일부 경고 시간을 제공합니다.

### 2.4 CME 재결합의 관측

백색광 코로나그래프(SOHO/LASCO, STEREO/COR)는 코로나를 통해 전파하는 CME를 관측합니다. 주요 재결합 신호는 다음을 포함합니다:

- **스트리머 폭발(Streamer blowout)**: 헬멧 스트리머 구조의 분출
- **CME 후 전류 시트**: CME를 따라가는 희미한 광선 모양 구조
- **유입/유출**: UV/EUV 분광선의 Doppler 이동

STEREO의 Heliospheric Imagers는 CME를 지구까지 추적하여 복잡한 상호작용과 편향을 드러냈습니다.

## 3. 자기권 서브스톰

### 3.1 Dungey 순환

Dungey 순환(1961)은 태양풍-자기권 결합의 기본 모델로, **주간 및 야간 재결합**에 의해 구동됩니다.

**주간 재결합:**

행성 간 자기장(IMF)이 남향 성분($B_z < 0$)을 가질 때, **주간 자기권계면**에서 지구의 북향 자기장과 재결합할 수 있습니다:

```
    태양풍                  자기권

    IMF(남향)             지자기장
         ↓                            ↑
    ─ ─ ─X─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  자기권계면
         ↓                            ↓
       재결합된 자기장 선
```

새로 재결합된 자기장 선은 태양풍에 의해 꼬리 방향으로 당겨져 에너지와 자속을 자기권으로 전달합니다.

**야간 재결합:**

자기꼬리에서 늘어난 자기장은 **근지구 중성선**에서 재결합합니다:

```
         꼬리 로브(북쪽)
              ↑   ↑
    ─ ─ ─ ─ ─ ─X─ ─ ─ ─ ─ ─  중성선
              ↓   ↓
         꼬리 로브(남쪽)
```

재결합은 다음을 생성합니다:
- **지구 방향 흐름**: 지구를 향해 가속된 플라즈마(서브스톰 주입)
- **꼬리 방향 흐름**: 꼬리 하향으로 방출된 플라스모이드

순환은 지구 방향으로 이동하는 자속이 주간으로 돌아가면서 닫힙니다.

**에너지 수지:**

태양풍은 다음 속도로 에너지를 전달합니다:

$$P \sim v B^2 / \mu_0 \times A_{cross}$$

여기서 $A_{cross} \sim \pi R_M^2$는 자기권 단면적이고 $R_M \sim 10 R_E$(지구 반경)입니다. 일반적인 태양풍($v = 400$ km/s, $B = 5$ nT)의 경우:

$$P \sim 10^{11}\text{–}10^{12} \text{ W}$$

서브스톰 동안, 저장된 에너지(~$10^{15}$ J)가 ~1시간에 방출되어 전력 ~$10^{11}$ W를 제공합니다.

### 3.2 서브스톰 단계

자기권 서브스톰은 자기권의 전역 재구성으로, 일반적으로 2–3시간 지속됩니다.

**성장 단계(Growth phase)** (30–60분):

- 주간 재결합이 자속을 꼬리로 전달
- 자기꼬리가 늘어나고 얇아짐
- 꼬리 로브에 에너지 저장
- 오로라 타원이 적도 방향으로 확장
- 꼬리 횡단 전류 강화

**확장 단계(Expansion phase)** (30–60분):

- 야간 재결합 시작
- 오로라의 갑작스러운 밝아짐(서브스톰 시작)
- 서향 이동 급증
- 쌍극자화(Dipolarization): 꼬리 자기장이 더 쌍극자가 됨
- 내부 자기권으로의 에너지 입자 주입
- 플라즈마 시트의 폭발적 대량 흐름(BBFs)

**회복 단계(Recovery phase)** (~1시간):

- 꼬리 자기장이 조용한 시간 구성으로 이완
- 오로라 활동 감소
- 플라즈마 시트 두꺼워짐

**관측 신호:**

- **지상 자력계**: H-성분의 음의 만(북향 자기장 감소)
- **오로라 이미지**: 밝아짐과 극쪽 확장
- **현장 우주선**: 흐름 폭발, 쌍극자화 전선, 입자 주입
- **오로라 킬로미터 복사(AKR)**: 강렬한 전파 방출

### 3.3 근지구 중성선 모델

**근지구 중성선(Near-Earth Neutral Line, NENL) 모델**은 서브스톰 시작을 자기꼬리의 $X \sim -20$에서 $-30 R_E$에서 형성되는 재결합에 기인합니다.

**순서:**

1. 성장 단계: 전류 시트 얇아짐, 압력 축적
2. 시작: NENL에서 재결합 시작
3. 확장: 재결합 영역이 $X$ 및 $Y$(새벽-황혼) 방향으로 확장
4. NENL에서 지구 방향 및 꼬리 방향 제트 발사
5. 플라스모이드(자속 로프)가 꼬리 하향으로 방출
6. 쌍극자화 전선이 지구 방향으로 전파하여 에너지 입자 전달

**증거:**

- 꼬리 방향으로 이동하는 플라스모이드의 우주선 관측
- $v_x \sim 400$ km/s의 지구 방향 폭발적 대량 흐름
- 이동 압축 영역(TCRs)
- Hall 자기장 사중극(Cluster 관측)

서브스톰 동안의 재결합률은 $M_A \sim 0.1$로, 빠른(무충돌) 재결합을 나타냅니다.

### 3.4 Python 예제: Dungey 순환 만화

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge, Arc

fig, ax = plt.subplots(figsize=(14, 10))

# Earth
earth = Circle((0, 0), 0.3, color='blue', alpha=0.7, label='Earth')
ax.add_patch(earth)

# Magnetopause (dayside and nightside)
theta_day = np.linspace(-np.pi/2, np.pi/2, 50)
x_day = 1.5 * np.cos(theta_day)
y_day = 1.5 * np.sin(theta_day)
ax.plot(x_day, y_day, 'k-', linewidth=3, label='Magnetopause')

# Tail magnetopause
tail_y_top = np.linspace(1.5, 1.2, 30)
tail_x_top = -np.linspace(0, 5, 30)
tail_y_bot = np.linspace(-1.5, -1.2, 30)
tail_x_bot = -np.linspace(0, 5, 30)
ax.plot(tail_x_top, tail_y_top, 'k-', linewidth=3)
ax.plot(tail_x_bot, tail_y_bot, 'k-', linewidth=3)

# Dayside X-line
ax.plot(1.5, 0, 'rx', markersize=20, markeredgewidth=4, label='Reconnection X-line')
ax.text(1.5, -0.5, 'Dayside\nreconnection', fontsize=11, ha='center', color='red', weight='bold')

# Nightside X-line
ax.plot(-3, 0, 'rx', markersize=20, markeredgewidth=4)
ax.text(-3, -0.5, 'Nightside\nreconnection', fontsize=11, ha='center', color='red', weight='bold')

# Solar wind
for y_sw in np.linspace(-2, 2, 5):
    ax.annotate('', xy=(2, y_sw), xytext=(4, y_sw),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))
ax.text(4.5, 2.5, 'Solar Wind', fontsize=13, color='orange', weight='bold')

# Open field lines (dayside to tail)
# Freshly reconnected line
x_open1 = np.concatenate([np.linspace(1.5, 0, 20), np.linspace(0, -5, 30)])
y_open1 = np.concatenate([np.linspace(0, 1.8, 20), np.linspace(1.8, 1.2, 30)])
ax.plot(x_open1, y_open1, 'g-', linewidth=2, alpha=0.7)
ax.plot(x_open1, -y_open1, 'g-', linewidth=2, alpha=0.7)

# Add arrows to show tailward motion
ax.annotate('', xy=(-2, 1.5), xytext=(-1, 1.6),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.text(-0.5, 2.2, 'Tailward\nconvection', fontsize=10, color='green', weight='bold')

# Closed field lines (dipolar)
for r in [0.6, 0.9, 1.2]:
    theta_closed = np.linspace(-np.pi/3, np.pi/3, 40)
    x_closed = r * np.cos(theta_closed)
    y_closed = r * np.sin(theta_closed)
    ax.plot(x_closed, y_closed, 'b--', linewidth=1.5, alpha=0.5)
    ax.plot(x_closed, -y_closed, 'b--', linewidth=1.5, alpha=0.5)

# Sunward return flow
ax.annotate('', xy=(0.8, 0.6), xytext=(-1, 0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
ax.text(-0.5, 1.0, 'Sunward\nreturn', fontsize=10, color='purple', weight='bold')

# Plasmoid ejection
plasmoid = Circle((-4.5, 0), 0.4, color='red', alpha=0.4)
ax.add_patch(plasmoid)
ax.annotate('', xy=(-5.5, 0), xytext=(-4.5, 0),
            arrowprops=dict(arrowstyle='->', lw=3, color='red'))
ax.text(-5.5, -0.6, 'Plasmoid', fontsize=11, color='red', weight='bold')

# Title and labels
ax.text(0, -3.5, 'Dungey Cycle: Solar Wind-Magnetosphere Coupling', fontsize=16, ha='center', weight='bold')
ax.text(3, -2.8, 'IMF $B_z < 0$ (southward)', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# Axes and formatting
ax.set_xlim(-6, 5)
ax.set_ylim(-4, 3)
ax.set_aspect('equal')
ax.legend(loc='lower left', fontsize=11)
ax.axis('off')

plt.tight_layout()
plt.savefig('dungey_cycle.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 4. 토카막 톱니파 충돌

### 4.1 톱니파 진동

토카막 플라즈마에서 코어 전자 온도는 종종 **톱니파 진동**을 나타냅니다: 느린 상승과 갑작스러운 충돌.

```
T_e
 |     /|     /|     /|
 |    / |    / |    / |
 |   /  |   /  |   /  |
 |  /   |  /   |  /   |
 | /    | /    | /    |
 |/_____|/_____|/_____|_____  시간
   상승  충돌  상승  충돌
```

**특성:**

- **상승 단계**: 온도가 10–100 ms 동안 꾸준히 증가
- **충돌 단계**: 온도가 <100 μs 안에 ~50% 하락
- **반전 반경**: $T_e$가 하락하는 내부 반경, 외부는 $T_e$가 상승(재분배)
- **q-프로파일**: 안전 인자 $q(r) = r B_\phi / (R B_\theta)$가 코어에서 1 아래로 하락

**물리적 그림:**

1. 상승 단계: 중심 가열, 첨두 전류 프로파일, $q_0$가 1 아래로 하락
2. 트리거: 내부 킹크 불안정성($m/n = 1/1$ 모드) 발달
3. 재결합: 나선형 자기 표면이 재결합
4. 충돌: 열과 전류의 빠른 재분배
5. 재설정: $q_0$가 1 위로 다시 상승, 순환 반복

### 4.2 Kadomtsev 재결합 모델

**Kadomtsev 모델**(1975)은 충돌을 $q = 1$ 표면의 완전한 재결합으로 설명합니다.

**재결합 전:**

$q = 1$ 표면은 반경 $r_1$의 중첩된 자속 표면입니다. 내부에서 자기장 선은 각 폴로이달 회전당 토로이달로 한 번 감깁니다. 코어는 자기적으로 고립되어 있습니다.

**재결합 후:**

나선형 섭동은 $q = 1$ 표면이 **나선형 섬**이 되게 합니다. 완전한 재결합은 섬 O-점을 원래 코어와 병합하여 새로운 평평한 $q \approx 1$ 프로파일을 생성합니다.

**위상 구조 변화:**

```
이전:                  이후:

    중첩된                  평평한
    표면                  프로파일
      ○                      ─────
     ╱ ╲                    ╱     ╲
    ○ ○ ○        ────>     ─────────
     ╲ ╱                    ╲     ╱
      ○                      ─────
     q=1                     q≈1
```

**열 재분배:**

재결합은 고온 코어 플라즈마를 빠르게 외부로 혼합하고 더 차가운 가장자리 플라즈마를 내부로 혼합하여 다음을 유발합니다:
- 코어 온도 하락
- 가장자리 온도 상승(반전 반경 내)
- 온도 프로파일 평탄화

**재결합률:**

충돌 시간은 ~10–100 μs로, 저항 확산(수 초 걸림)보다 훨씬 빠릅니다. 이것은 다음을 시사합니다:

$$M_A \sim 0.01\text{–}0.1$$

전자 운동학적 척도에서의 무충돌 재결합과 일치합니다.

### 4.3 관측 및 시뮬레이션

**실험 증거:**

- **연X선 단층촬영**: $m=1$ 전구체 진동을 보여주고, 그 다음 빠른 충돌
- **ECE(Electron Cyclotron Emission)**: 고해상도 $T_e$ 프로파일 측정
- **자기 진단**: Mirnov 코일이 $m/n = 1/1$ 모드 성장 감지
- **부분 vs 완전 재결합**: 모든 충돌이 완전하지는 않으며, 일부는 불완전한 재결합을 보임

**수치 시뮬레이션:**

- 저항 MHD: 톱니파 순환을 재현하지만 충돌이 너무 느림
- 이유체/운동학적: 더 빠른 충돌, 관측에 더 가까움
- 확장 MHD: Hall, 전자 압력 포함, 빠른 충돌 포착

현대 시뮬레이션(예: M3D, NIMROD)은 이유체 효과가 저항 MHD에 비해 충돌을 상당히 가속한다는 것을 보여줍니다.

### 4.4 Python 예제: 톱니파 충돌 시뮬레이션(1D 모델)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple 1D model of sawtooth temperature evolution
# (Not a real reconnection simulation, just illustrative)

r = np.linspace(0, 1, 100)  # Normalized radius
# t_rise >> t_crash: 톱니파는 비대칭입니다. 상승(ohmic/RF 가열)은 느린
# 확산 과정인 반면, 충돌은 Alfvén 시간 척도(~μs 대 상승의 ~ms)에서
# 빠른 내부 kink에 의해 구동되기 때문입니다.
t_rise = 100  # Number of time steps in rise phase
t_crash = 5   # Number of time steps in crash
n_cycles = 3

# 반전 반경 r_inv ~ 0.3은 충돌에 대한 온도 응답이 부호가 바뀌는 경계를
# 표시합니다: r_inv 내부에서는 고온 코어 플라즈마가 바깥으로 퍼져 코어를
# 냉각하고, r_inv 외부에서는 도착한 고온 플라즈마가 온도를 높입니다 —
# 이것이 연X선 단층촬영으로 측정되는 톱니파 재분배의 관측 신호입니다.
r_inv = 0.3

# Initial profile
T0 = 1 - r**2

# Storage
T_history = []
time_history = []

T = T0.copy()
time = 0

for cycle in range(n_cycles):
    # Rise phase: gradual central heating
    for i in range(t_rise):
        # 가우시안 열 증착은 중성 빔 또는 ECRH에 의한 축상 가열을 모사합니다;
        # 좁은 폭 0.2는 r_inv 훨씬 안쪽에 집중된 입사 전력 프로파일을 나타냅니다.
        heat_source = 0.01 * np.exp(-(r / 0.2)**2)
        # 확산 냉각은 자속 표면을 따른 열 수송을 모델링합니다:
        # 2차 도함수 ∂²T/∂r²가 열을 고온에서 저온 영역으로 구동하며,
        # 여기서는 자체 일관적으로 모델링되지 않지만 정성적인 상승을 재현합니다.
        dTdr = np.gradient(T, r)
        d2Tdr2 = np.gradient(dTdr, r)
        cooling = 0.001 * d2Tdr2

        T += heat_source + cooling
        T_history.append(T.copy())
        time_history.append(time)
        time += 1

    # Crash phase: rapid flattening inside inversion radius
    T_before_crash = T.copy()
    for i in range(t_crash):
        # r_inv 내부에서 T를 평균으로 교체하는 것은 Kadomtsev 완전 재결합
        # 시나리오를 모델링합니다: m=1 kink가 고온 코어를 q=1 표면까지
        # 더 차가운 플라즈마와 완전히 혼합하여, 단 몇 Alfvén 시간 만에
        # 첨두 온도 구조를 파괴합니다.
        inside = r < r_inv
        T_avg_inside = np.mean(T[inside])
        T[inside] = T_avg_inside

        # 에너지 보존: 코어에서 제거된 열은 r_inv 바로 외부에 증착되어,
        # 톱니파 충돌 중 연X선 진단으로 관측되는 작은 온도 범프를 생성합니다.
        outside = r >= r_inv
        T[outside] += 0.05 * (T_before_crash[inside].mean() - T_avg_inside)

        T_history.append(T.copy())
        time_history.append(time)
        time += 1

# Convert to array
T_history = np.array(T_history)
time_history = np.array(time_history)

# Plot core temperature vs time
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel 1: Core temperature (sawtooth pattern)
ax = axes[0]
T_core = T_history[:, 0]
ax.plot(time_history, T_core, linewidth=2, color='darkblue')
ax.set_xlabel('Time (arbitrary units)', fontsize=13)
ax.set_ylabel('Core Temperature $T_e(r=0)$', fontsize=13)
ax.set_title('Sawtooth Oscillations: Core Temperature', fontsize=15)
ax.grid(True, alpha=0.3)

# Mark crashes
crash_indices = []
for i in range(1, len(T_core)):
    if T_core[i] < T_core[i-1] - 0.1:
        crash_indices.append(i)
for idx in crash_indices:
    ax.axvline(time_history[idx], color='red', linestyle='--', alpha=0.6)

# Panel 2: Radial profiles at different times
ax = axes[1]

# Plot profiles at selected times
times_to_plot = [50, 99, 102, 150, 199, 202]  # Before/after crashes
colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))

for i, t_idx in enumerate(times_to_plot):
    label = f't = {time_history[t_idx]}'
    if t_idx in [99, 199]:
        label += ' (before crash)'
        linestyle = '-'
        linewidth = 2.5
    elif t_idx in [102, 202]:
        label += ' (after crash)'
        linestyle = '--'
        linewidth = 2.5
    else:
        linestyle = '-'
        linewidth = 1.5

    ax.plot(r, T_history[t_idx], color=colors[i], linestyle=linestyle,
            linewidth=linewidth, label=label)

ax.axvline(r_inv, color='black', linestyle=':', linewidth=2, label=f'Inversion radius ($r={r_inv}$)')
ax.set_xlabel('Normalized radius $r/a$', fontsize=13)
ax.set_ylabel('Temperature $T_e$', fontsize=13)
ax.set_title('Radial Temperature Profiles', fontsize=15)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sawtooth_crash_simulation.png', dpi=150)
plt.show()
```

## 5. 자기 섬 병합

### 5.1 섬 병합의 물리학

같은 나선성을 가진 두 자기 섬(O-점)이 함께 모이면, 그들 사이의 X-점에서 재결합을 통해 병합할 수 있습니다. 이 과정을 **자기 섬 병합**이라고 합니다.

**초기 구성:**

```
    O─────X─────O

  섬 1  X  섬 2
```

**병합 중:**

중심 X-점에서의 재결합이 섬들을 병합할 수 있게 합니다:

```
    O───────────O   (함께 이동)
          X

    재결합 가속
```

**최종 상태:**

```
       ○○○
      ○   ○
      ○ O ○  단일 큰 섬
      ○   ○
       ○○○
```

### 5.2 동역학 및 재결합률

섬들은 자기 장력에 의해 함께 구동됩니다. 그들이 접근함에 따라 X-점 전류 시트가 강화되고 재결합이 가속됩니다.

**에너지 변환:**

- **초기 상태**: 두 섬에 저장된 자기 에너지
- **병합**: 재결합이 자기 에너지 방출
- **최종 상태**: 하나의 더 큰 섬 + 운동/열 에너지

**재결합률:**

시뮬레이션은 병합 재결합이 빠르며, 높은 Lundquist 수에서도 저항 MHD에서 $M_A \sim 0.1$임을 보여줍니다. 이것은 섬 성장의 Rutherford 영역이 병합 중에 폭발적이 되기 때문입니다.

**응용:**

- **토카막 붕괴**: 다중 찢김 모드가 병합되어 붕괴를 유발할 수 있음
- **태양 코로나**: 상호작용하는 자속 튜브/루프
- **자기꼬리**: 병합하는 플라스모이드

### 5.3 Python 예제: 2D 섬 병합

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple model: two magnetic islands approaching and merging

x = np.linspace(-4, 4, 100)
y = np.linspace(-2, 2, 80)
X, Y = np.meshgrid(x, y)

# 가우시안 자속 함수 -exp(-r²/size²)는 O-점(섬)을 모델링합니다:
# 음의 부호는 ψ의 최대값을 섬 중심에 놓고, 가우시안 포락선은
# 부드럽고 고립된 자속 구조를 만듭니다. 이것은 MHD 시뮬레이션에서
# 보이는 더 복잡한 전류 시트 유래 섬을 단순화한 해석적 대체물이지만,
# 위상적으로는 올바른 구조를 포착합니다.
def island_flux(X, Y, x0, y0, size):
    return -np.exp(-((X - x0)**2 + (Y - y0)**2) / size**2)

# Time snapshots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

times = [0, 1, 2, 3]
# 감소하는 간격은 인력을 나타냅니다: 같은 나선성을 가진 두 섬은
# 두 섬 사이 X-점의 자기장 선을 통과하는 자기 장력에 의해 끌어당겨집니다 —
# 서로 끌어당기는 두 평행한 전류 도선과 유사합니다.
separations = [2.5, 1.5, 0.8, 0]  # Island separation decreases

for ax, t, sep in zip(axes.flat, times, separations):
    # Two islands approaching
    psi1 = island_flux(X, Y, -sep/2, 0, 0.6)
    psi2 = island_flux(X, Y, sep/2, 0, 0.6)

    if sep > 0:
        # Before full merger
        psi = psi1 + psi2
        # X=0의 좁은 가우시안 전류 시트는 섬이 접근할수록 강해지는
        # X-점 전류 층을 나타냅니다: 재결합이 발생하여 섬이 병합할 수
        # 있게 하는 것이 바로 이 층입니다.
        sheet_contrib = 0.2 * np.exp(-X**2 / 0.1**2) * np.exp(-(Y)**2 / 2)
        psi += sheet_contrib
    else:
        # 병합된 섬은 더 큰 크기(0.6 대신 1.0)를 가집니다. 두 섬에서
        # 재결합된 자속이 이제 단일 O-점에 포함되어 있기 때문입니다;
        # 총 자기 에너지는 두 개의 분리된 섬의 합보다 낮으며, 그 차이가
        # 열과 흐름으로 방출됩니다.
        psi = island_flux(X, Y, 0, 0, 1.0)

    # 0.05·X·Y 배경 쌍곡선 자기장은 두 섬 사이에 X-점 기하학을 생성합니다:
    # 이것이 없으면 두 가우시안 섬의 중첩은 올바른 안장점 위상을 보이지 않습니다.
    psi += 0.05 * X * Y

    # Compute magnetic field
    By = np.gradient(psi, x, axis=1)
    Bx = -np.gradient(psi, y, axis=0)

    # Plot
    contour_levels = np.linspace(psi.min(), psi.max(), 20)
    ax.contour(X, Y, psi, levels=contour_levels, colors='blue', linewidths=0.8)

    # Streamplot for field lines
    ax.streamplot(X, Y, Bx, By, color='black', linewidth=0.6, density=1.2, arrowsize=0.8)

    # Mark O-points
    if sep > 0:
        ax.plot(-sep/2, 0, 'go', markersize=12, label='O-point (island)')
        ax.plot(sep/2, 0, 'go', markersize=12)
        if sep > 0.5:
            ax.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label='X-point')
    else:
        ax.plot(0, 0, 'go', markersize=12, label='Merged island')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_title(f'Time $t = {t}$ (separation = {sep:.1f})', fontsize=13)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Magnetic Island Coalescence', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('island_coalescence.png', dpi=150)
plt.show()
```

## 6. 천체물리학 제트와 재결합

### 6.1 천체물리학의 제트

**천체물리학 제트**는 많은 시스템에서 관측되는 고도로 시준된 유출입니다:

- **활동 은하핵(AGN)**: 초대질량 블랙홀로부터의 제트, Mpc(수백만 파섹) 확장
- **마이크로퀘이사**: X선 쌍성의 항성 질량 블랙홀 제트
- **젊은 항성 천체(YSOs)**: 원시별 제트(HH 천체)
- **펄서 바람 성운**: 펄서로부터의 상대론적 제트(Crab Nebula)
- **감마선 폭발(GRBs)**: 초상대론적 제트, Lorentz 인자 $\Gamma \sim 100$–1000

**공통 특징:**

- 높은 시준: 열림 각도 ~1°–10°
- 상대론적 속도: $v \sim 0.1c$에서 $>0.99c$
- 높은 전력: AGN의 경우 최대 $10^{47}$ erg/s
- 자기장: 강함(AGN 제트에서 $B \sim 0.1$–10 G)

### 6.2 재결합 구동 가속

자기 재결합은 다음에 대한 주요 후보입니다:

1. **제트 발사**: 자기 에너지를 운동 에너지로 변환
2. **제트 가속**: 제트를 따라 추가 가속
3. **입자 가속**: 비열적 입자 생성(동기복사 방출)

**발사 메커니즘:**

블랙홀 또는 중성자별의 자기권에서 회전하는 자기화된 강착원반은 대규모 폴로이달 자기장을 생성할 수 있습니다. 전류 시트에서의 재결합은:
- 자기 장력 방출
- Alfvén 속도 이상에서 유출 구동
- 흐름 시준

**제트에서의 재결합:**

제트의 불안정성(예: 킹크)은 재결합을 유발할 수 있습니다:
- 자기 에너지 소산
- 제트 방출의 플레어와 블롭(블레이저 변동성에서 관측)
- 비열적 에너지로 입자 가속(멱법칙 분포)

### 6.3 펄서 자기권

**펄서**는 초강한 자기장($B \sim 10^{12}$ G)을 가진 회전하는 중성자별입니다. 자기권 구조는 다음을 포함합니다:

- **닫힌 영역**: 광원통 내에서 닫히는 쌍극자 자기장 선
- **열린 영역**: 무한대로 확장되는 자기장 선
- **전류 시트**: 광원통 너머 적도면에 형성

**줄무늬 바람에서의 재결합:**

펄서 바람은 교대로 자기 극성을 가집니다(줄무늬 바람). 전류 시트에서의 재결합은:
- 자기 에너지(Poynting 자속)를 입자 에너지로 변환
- 관측된 비열적 방출 생성(전파, 광학, X선, 감마선)
- Crab Nebula의 고에너지 플레어 설명

**시그마 문제:**

펄서 근처에서 자기화 매개변수 $\sigma = B^2/(\mu_0 \rho c^2) \gg 1$(자기적으로 지배). 하지만 관측은 종결 충격에서 $\sigma \sim 0.01$–0.1을 요구합니다. 줄무늬 바람에서의 재결합이 이 **시그마 문제**에 대한 주요 해결책입니다.

### 6.4 감마선 폭발

**GRBs**는 우주에서 가장 밝은 폭발로, 수 초에서 수 분 동안 감마선으로 $\sim 10^{51}$–$10^{54}$ erg를 방출합니다.

**화구 모델:**

- **중심 엔진**: 붕괴별(거대 항성 붕괴) 또는 병합(중성자별-중성자별/블랙홀)
- **상대론적 유출**: Lorentz 인자 $\Gamma \sim 100$–1000
- **내부 충격**: 제트의 재결합과 충격이 즉발 감마선 방출 생성
- **외부 충격**: 제트가 ISM과 상호작용하여 잔광 생성

**재결합의 역할:**

- **에너지 소산**: 제트의 재결합이 자기 에너지를 복사로 변환
- **입자 가속**: 비열적 전자가 감마선 방사(동기복사, 역 콤프턴)
- **시간 변동성**: 재결합 플라스모이드가 빠른 변동성 생성(ms 시간 척도)

최근 시뮬레이션(Uzdensky, Werner, Sironi 등)은 상대론적 재결합이 요구되는 비열적 분포로 입자를 효율적으로 가속할 수 있음을 보여줍니다.

## 7. 요약

자기 재결합은 광범위한 현상에서 중심 역할을 합니다:

1. **태양 플레어**: CSHKP 모델은 전류 시트에서의 재결합을 통한 에너지 방출(~$10^{32}$ erg)을 설명합니다. 관측된 재결합률은 $M_A \sim 0.01$–$0.1$로, 빠른 재결합과 일치합니다. SDO 및 기타 임무의 관측은 supra-arcade downflows, above-the-loop-top 소스, 리본 동역학을 드러냅니다.

2. **코로나 질량 방출**: CME 시작은 종종 토러스 불안정성에 의해 유발되는 자속 로프 분출을 포함합니다. 분출하는 자속 로프 아래의 재결합은 에너지를 방출하고 위상 구조 변화를 허용합니다. CME는 상당한 우주 날씨 위험을 제기합니다.

3. **자기권 서브스톰**: Dungey 순환은 주간 및 야간 재결합을 통한 태양풍-자기권 결합을 설명합니다. 서브스톰은 성장 단계 동안 꼬리 로브에 에너지를 저장한 후 확장 단계 동안 폭발적 방출을 포함합니다. 근지구 중성선 모델은 시작을 $X \sim -20$에서 $-30 R_E$에서의 재결합에 기인합니다.

4. **토카막 톱니파 충돌**: 톱니파 진동은 내부 킹크 불안정성과 $q = 1$ 표면에서의 재결합으로부터 발생합니다. Kadomtsev 모델은 충돌을 빠른 열 재분배를 일으키는 완전한 재결합으로 설명합니다. 빠른 충돌 시간은 무충돌 재결합을 나타냅니다.

5. **자기 섬 병합**: 두 섬이 병합할 때, 중간 X-점에서의 재결합은 빠르며($M_A \sim 0.1$), 저항 MHD에서도 마찬가지입니다. 이 과정은 토카막 붕괴와 태양/자기권 동역학과 관련이 있습니다.

6. **천체물리학 제트**: 재결합은 AGN, 펄서, GRBs, YSOs에서 제트 발사, 가속, 입자 가속에 연루되어 있습니다. 상대론적 재결합은 펄서 바람과 GRB 제트에서 자기 에너지를 입자 에너지로 효율적으로 변환합니다.

이 모든 응용에서 재결합률은 빠른($M_A \sim 0.01$–$0.1$)것으로 관측되거나 추론되어, 무충돌(Hall, 운동학적) 재결합 물리학의 중요성을 지원합니다.

## 연습 문제

1. **태양 플레어 에너지:**
   a) $B = 0.02$ T, 부피 $V = (10^8 \text{ m})^3$인 플레어 활동 영역의 자기 에너지를 추정하세요.
   b) 이 에너지의 20%가 1000 s 지속되는 플레어에서 방출되면 평균 전력은 얼마입니까?
   c) 이것을 총 태양 광도($L_\odot = 3.8 \times 10^{26}$ W)와 비교하세요.

2. **플레어 리본 운동:**
   a) 관측된 플레어 리본이 $v_{sep} = 50$ km/s로 분리됩니다. 코로나 높이가 $h = 10^7$ m이고 족점 분리가 $d = 10^8$ m이면, 재결합 유입 속도 $v_{in}$을 추정하세요.
   b) Alfvén 속도 $v_A = 1000$ km/s로 $M_A$를 계산하세요.
   c) 이것이 Sweet-Parker, Petschek 또는 Hall 재결합과 일치합니까?

3. **CME 운동 에너지:**
   a) CME가 질량 $M = 10^{15}$ g이고 속도 $v = 1000$ km/s입니다. 운동 에너지를 erg로 계산하세요.
   b) CME가 태양풍 항력에 의해 $v = 500$ km/s로 감속되면 얼마나 많은 에너지가 소산됩니까?
   c) 이 에너지는 어디로 갑니까?

4. **토러스 불안정성:**
   a) 토러스 불안정성 기준을 설명하세요: $d \ln B_{ext} / d \ln h < -3/2$.
   b) 쌍극자 자기장 $B \propto r^{-3}$에 대해 $d \ln B / d \ln r$을 계산하세요($h \sim r$로 취급).
   c) 쌍극자 자기장은 토러스 불안정성에 대해 안정합니까, 불안정합니까?

5. **Dungey 순환 시간 척도:**
   a) 주간 재결합이 $d\Phi/dt = E_{rec} \cdot L_y$의 속도로 자기 자속을 전달하면, 여기서 $E_{rec} = v_{in} B_{sw}$이고 $L_y \sim 20 R_E$일 때, $v_{in} = 100$ km/s, $B_{sw} = 5$ nT, $R_E = 6.4 \times 10^6$ m에 대해 $d\Phi/dt$를 추정하세요.
   b) 꼬리 로브의 총 자속은 $\Phi_{tail} \sim B_{lobe} \cdot A_{lobe} \sim 1$ GWb입니다. 이 자속을 로드하는 데 얼마나 걸립니까?
   c) 관측된 서브스톰 성장 단계 지속 시간(~30–60분)과 비교하세요.

6. **서브스톰 에너지 방출:**
   a) 자기꼬리는 에너지 $\sim B^2 V / (2\mu_0)$를 저장합니다. $B = 20$ nT, $V \sim (10 R_E)^3$에 대해 이것을 추정하세요.
   b) 이 에너지가 서브스톰 동안 1시간에 걸쳐 방출되면 평균 전력은 얼마입니까?
   c) 태양풍 입력 전력(~$10^{11}$–$10^{12}$ W)과 비교하세요.

7. **톱니파 충돌 시간:**
   a) 토카막에서 톱니파 충돌 시간은 $\tau_{crash} \sim 50$ μs입니다. 소반경은 $a = 0.5$ m, $B = 3$ T, $n = 10^{20}$ m⁻³입니다.
   b) Alfvén 시간 $\tau_A = a/v_A$를 계산하세요.
   c) 재결합률 $M_A \sim \tau_A / \tau_{crash}$를 추정하세요.

8. **섬 병합:**
   a) 토카막에서 폭 $w = 5$ cm의 두 자기 섬이 거리 $d = 10$ cm로 분리되어 있습니다. 국소 Alfvén 속도는 $v_A = 10^6$ m/s입니다.
   b) 그들이 속도 $v \sim 0.1 v_A$로 접근하면 병합까지 얼마나 걸립니까?
   c) 병합 중에 자기 에너지의 몇 퍼센트가 방출됩니까(섬이 에너지 $\propto w^2 B^2$를 가지며, 최종 섬이 $w_{final} = \sqrt{2} w$를 가진다고 가정)?

9. **AGN 제트 전력:**
   a) AGN 제트가 반경 $R = 10^{16}$ m, 유출 속도 $v = 0.5c$이고, $B = 1$ G로 Poynting 자속 $S = B^2 v / \mu_0$를 운반합니다.
   b) 제트 전력 $P = S \cdot \pi R^2$를 계산하세요.
   c) $10^9 M_\odot$ 블랙홀의 Eddington 광도($L_{Edd} \sim 10^{47}$ erg/s)와 비교하세요.

10. **상대론적 재결합:**
    a) 펄서 바람에서 자기화 매개변수는 광원통 근처에서 $\sigma = B^2/(\mu_0 \rho c^2) = 10^3$입니다.
    b) 재결합이 자기 에너지의 50%를 입자 운동 에너지로 변환하면 최종 $\sigma$는 얼마입니까?
    c) 이것이 관측($\sigma_{obs} \sim 0.01$–0.1)을 설명하기에 충분합니까? 그렇지 않다면, 무엇이 더 필요합니까?

## 네비게이션

이전: [재결합 이론](./05_Reconnection_Theory.md) | 다음: [고급 재결합](./07_Advanced_Reconnection.md)
