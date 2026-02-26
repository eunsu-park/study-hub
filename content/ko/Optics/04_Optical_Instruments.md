# 04. 광학 기기

[← 이전: 03. 거울과 렌즈](03_Mirrors_and_Lenses.md) | [다음: 05. 파동광학 — 간섭 →](05_Wave_Optics_Interference.md)

---

## 학습 목표

1. 박막 렌즈 방정식으로 돋보기, 현미경, 망원경의 광학을 분석한다
2. 시각 기기의 각 배율(angular magnification)을 계산하고, 횡 배율(lateral magnification)과의 차이를 이해한다
3. 레일리 기준(Rayleigh criterion)을 적용하여 광학계의 분해능 한계를 구한다
4. 카메라의 광학 설계 — F수(f-number), 피사계 심도(depth of field), 노출 — 를 설명한다
5. 인간의 눈의 광학, 흔한 시력 결함, 교정 렌즈를 서술한다
6. 굴절 망원경과 반사 망원경 설계를 비교하고 트레이드오프를 이해한다
7. 기기 성능을 제한하는 실질적 한계 — 회절, 수차, 검출기 잡음 — 를 평가한다

---

## 왜 중요한가

광학 기기는 인간의 시야를 원자 수준에서 우주 수준까지 확장한다. 현미경은 세균, 세포, 바이러스를 드러내어 의학을 혁신했다. 망원경은 은하, 우주 마이크로파 배경 복사, 외계 행성을 밝혀내어 우주에 대한 우리의 이해를 바꾸어 놓았다. 카메라 — 이제는 모든 스마트폰에 내장된 — 는 일상에서 가장 큰 영향을 미치는 광학 기기라 할 수 있다. 이 기기들이 어떻게 작동하고, 성능을 무엇이 제한하는지 이해하는 것은 영상, 센싱, 광자공학(photonics)을 다루는 모든 이에게 필수다.

> **비유**: 광학 기기는 빛을 위한 번역기와 같다. 인간의 눈은 제한된 각 크기 범위(약 1분각의 분해능)와 밝기만 처리할 수 있다. 현미경은 세균의 각 크기를 0.001°(보이지 않는)에서 10°(쉽게 볼 수 있는)로 "번역"한다. 망원경은 먼 은하의 희미한 빛을 측정 불가능한 속삭임에서 측정 가능한 신호로 "번역"한다. 각 기기는 인간 지각의 좁은 대역폭에 맞게 빛을 재형성한다.

---

## 1. 인간의 눈

### 1.1 광학 설계

인간의 눈은 놀라운 광학 기기다:

| 구성 요소 | 기능 |
|-----------|------|
| 각막(Cornea) | 주요 굴절면 ($n \approx 1.376$, 파워 $\approx$ 43 D) |
| 방수(Aqueous humor) | 투명한 액체 ($n \approx 1.336$) |
| 홍채/동공(Iris / Pupil) | 조리개(aperture stop), 직경 2–8 mm |
| 수정체(Crystalline lens) | 가변 초점 렌즈 ($n \approx 1.39$–$1.41$, 파워 $\approx$ 15–30 D) |
| 유리체(Vitreous humor) | 투명한 겔 ($n \approx 1.337$) |
| 망막(Retina) | 검출기 (어두운 곳에서는 간상세포(rod), 색상에는 원추세포(cone)) |

총 광학 파워는 약 60 D로, 초점 거리는 약 17 mm다.

눈은 수정체의 형태를 변화시켜 조절(accommodation)한다:
- **원점(far point, 이완된 눈)**: 무한대에 초점 맞춤 (정상 눈). 선명하게 볼 수 있는 가장 먼 거리.
- **근점(near point, 최대 조절)**: 젊은 성인 기준 약 25 cm. 초점을 맞출 수 있는 가장 가까운 거리.

### 1.2 시력과 각 분해능

인간 눈의 최소 분해 각도는 약:

$$\theta_{\min} \approx 1' = \frac{1}{60}° \approx 2.9 \times 10^{-4} \text{ rad}$$

이는 망막에서 약 5 $\mu$m 간격에 해당하며, 중심와(fovea) 원추세포 직경 정도다. 직경 3 mm의 동공에서 550 nm 빛의 회절 한계는:

$$\theta_{\text{diff}} = 1.22 \frac{\lambda}{D} = 1.22 \times \frac{550 \times 10^{-9}}{3 \times 10^{-3}} \approx 2.2 \times 10^{-4} \text{ rad} \approx 0.77'$$

눈의 분해능은 회절 한계에 인상적으로 가깝다 — 진화가 망막 모자이크를 최적화하여 회절 한계에 맞춘 것이다.

### 1.3 흔한 시력 결함

| 결함 | 원인 | 상 위치 | 교정 |
|------|------|---------|------|
| **근시(Myopia)** | 안구가 너무 길거나 각막이 너무 구부러짐 | 망막 앞 | 발산 렌즈 (음의 파워) |
| **원시(Hyperopia)** | 안구가 너무 짧거나 각막이 너무 평평함 | 망막 뒤 | 수렴 렌즈 (양의 파워) |
| **노안(Presbyopia)** | 나이가 들면서 수정체 유연성 저하 | 근점이 멀어짐 | 돋보기 (수렴 렌즈) |
| **난시(Astigmatism)** | 각막이 구형이 아님 (방향별 곡률 차이) | 서로 다른 거리에서 선 상 형성 | 원주 렌즈(cylindrical lens) 또는 토릭 렌즈(toric lens) |

```python
import numpy as np
import matplotlib.pyplot as plt

# Corrective lens prescriptions: relationship between
# vision defect and lens power needed

# For myopia: far point is closer than infinity
# Lens power needed: P = -1/far_point (in meters)
far_points_m = np.linspace(0.2, 5.0, 100)  # far point from 20 cm to 5 m
P_myopia = -1 / far_points_m

# For hyperopia/presbyopia: near point is farther than 25 cm
# Lens power needed to bring near point back to 25 cm:
# P = 1/0.25 - 1/near_point (near point > 0.25 m)
near_points_m = np.linspace(0.3, 3.0, 100)
P_hyperopia = 1/0.25 - 1/near_points_m

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Myopia correction
ax1.plot(far_points_m * 100, P_myopia, 'b-', linewidth=2)
ax1.set_xlabel('Uncorrected Far Point (cm)', fontsize=12)
ax1.set_ylabel('Lens Power (Diopters)', fontsize=12)
ax1.set_title('Myopia Correction', fontsize=13)
ax1.axhline(0, color='gray', linewidth=0.5)
ax1.grid(True, alpha=0.3)
# Typical prescription range
ax1.axhspan(-6, -1, alpha=0.1, color='blue', label='Typical range (-1 to -6 D)')
ax1.legend(fontsize=10)

# Right: Hyperopia/Presbyopia correction
ax2.plot(near_points_m * 100, P_hyperopia, 'r-', linewidth=2)
ax2.set_xlabel('Uncorrected Near Point (cm)', fontsize=12)
ax2.set_ylabel('Lens Power (Diopters)', fontsize=12)
ax2.set_title('Hyperopia / Presbyopia Correction\n(bring near point to 25 cm)', fontsize=13)
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.grid(True, alpha=0.3)
ax2.axhspan(0.5, 3.0, alpha=0.1, color='red', label='Typical range (+0.5 to +3 D)')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('vision_correction.png', dpi=150)
plt.show()
```

---

## 2. 돋보기(단순 확대경)

### 2.1 각 배율

수렴 렌즈를 돋보기로 사용하면 물체를 "가까이" 당기는 것이 아니라, 눈에 보이는 물체의 **각 크기(angular size)**를 확대한다. 각 배율은 다음과 같이 정의된다:

$$M = \frac{\theta_{\text{렌즈 사용 시}}}{\theta_{\text{렌즈 없을 때}}}$$

렌즈 없이 최대 각 크기는 근점($D = 25$ cm)에서 얻어진다:

$$\theta_0 = \frac{h}{D}$$

초점 거리 $f$의 렌즈를 사용하면, 물체를 초점 혹은 그 안쪽에 놓는다:

**상이 무한대에 맺힐 때** (눈의 긴장이 없는 상태, 물체를 $f$에 놓음):

$$M_\infty = \frac{D}{f} = \frac{25 \text{ cm}}{f}$$

**상이 근점에 맺힐 때** (최대 배율, 물체를 $f$ 약간 안쪽에 놓음):

$$M_{\max} = \frac{D}{f} + 1 = \frac{25}{f} + 1$$

$f = 5$ cm인 일반적인 돋보기는 $M_\infty = 5\times$를 제공한다.

### 2.2 실용적 한계

돋보기는 약 $M \leq 10\times$로 제한된다. 초점 거리가 더 짧아지면 눈을 렌즈에 매우 가까이 가져가야 하여 심한 수차가 발생하기 때문이다. 더 높은 배율을 위해서는 복합 현미경이 필요하다.

---

## 3. 복합 현미경

### 3.1 광학 설계

복합 현미경은 두 렌즈 그룹을 사용한다:

1. **대물렌즈(Objective lens)** (짧은 초점 거리 $f_o$): 경통 내부에 표본의 확대된 실상을 형성
2. **접안렌즈(Eyepiece, 오큘라)** (초점 거리 $f_e$): 중간 상을 돋보기처럼 확대하여 관찰

중간 상은 대물렌즈 후방 초점 뒤 **경통 길이(tube length)** $L$(표준 160 mm, 또는 현대 현미경에서는 무한대 경통)만큼 떨어진 위치에 형성된다.

### 3.2 총 배율

총 각 배율은 대물렌즈의 횡 배율과 접안렌즈의 각 배율의 곱이다:

$$M_{\text{total}} = m_o \times M_e = -\frac{L}{f_o} \times \frac{D}{f_e}$$

$40\times$ 대물렌즈($f_o = 4$ mm), $10\times$ 접안렌즈($f_e = 25$ mm), $L = 160$ mm의 경우:

$$M_{\text{total}} = -\frac{160}{4} \times \frac{25}{25} = -40 \times 10 = -400\times$$

음의 부호는 최종 상이 도립임을 의미하며, 실제로는 프리즘이나 중계 렌즈로 보정한다.

### 3.3 분해능: 진짜 한계

현미경의 유용한 배율 한계는 렌즈가 아니라 **회절(diffraction)**이다. 최소 분해 가능한 특징 크기(아베 회절 한계(Abbe diffraction limit))는:

$$d_{\min} = \frac{0.61\lambda}{\text{NA}} = \frac{0.61\lambda}{n\sin\alpha}$$

여기서 **NA(개구수(Numerical Aperture))** = $n \sin\alpha$이며, $n$은 표본과 대물렌즈 사이 매질의 굴절률, $\alpha$는 집광 광선 원뿔의 반각(half-angle)이다.

| 대물렌즈 | NA | 분해능 (550 nm) |
|---------|-----|----------------|
| 4$\times$ (건식) | 0.10 | 3.4 $\mu$m |
| 10$\times$ (건식) | 0.25 | 1.3 $\mu$m |
| 40$\times$ (건식) | 0.65 | 0.52 $\mu$m |
| 100$\times$ (오일, $n=1.52$) | 1.25 | 0.27 $\mu$m |

**공배율(Empty magnification)**: $500 \times \text{NA}$에서 $1000 \times \text{NA}$를 초과하는 배율은 새로운 세부 정보를 보여주지 못한다 — 그저 흐릿함을 더 크게 확대할 뿐이다. 이를 공배율(또는 헛배율)이라 한다.

> **비유**: 분해능은 디지털 이미지의 픽셀 수와 같다. 배율은 화면의 줌과 같다. 원하는 만큼 확대할 수 있지만, 기본 분해능을 초과하면 더 큰 픽셀만 볼 수 있을 뿐 새로운 세부 정보는 없다. 현미경 대물렌즈의 NA가 광학 이미지의 "픽셀 수"를 결정한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Microscope resolution: Airy disk size as a function of numerical aperture
# Shows why oil immersion objectives are essential for high resolution

wavelength = 550e-9  # green light (m)

NA_values = np.linspace(0.05, 1.4, 200)

# Abbe resolution limit: d_min = 0.61 * lambda / NA
d_min = 0.61 * wavelength / NA_values * 1e6  # convert to micrometers

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: resolution vs NA
ax1.plot(NA_values, d_min, 'b-', linewidth=2)
ax1.set_xlabel('Numerical Aperture (NA)', fontsize=12)
ax1.set_ylabel('Minimum Resolvable Feature ($\\mu$m)', fontsize=12)
ax1.set_title('Microscope Resolution Limit (Abbe Criterion)', fontsize=13)
ax1.set_ylim(0, 8)
ax1.grid(True, alpha=0.3)

# Mark common objectives
objectives = [
    (0.10, '4x dry'),
    (0.25, '10x dry'),
    (0.65, '40x dry'),
    (0.95, '100x dry'),
    (1.25, '100x oil'),
    (1.40, '100x oil max'),
]
for na, label in objectives:
    d = 0.61 * wavelength / na * 1e6
    ax1.plot(na, d, 'ro', markersize=6)
    ax1.annotate(f'  {label}\n  d={d:.2f}μm', xy=(na, d), fontsize=8)

# Indicate the air limit (NA max = 1.0 for dry objectives)
ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='Air limit (NA=1)')
ax1.legend(fontsize=10)

# Right: useful magnification range
ax2.fill_between([0, 1.5], [0, 0], [500*0, 500*1.5], alpha=0.15, color='red',
                 label='Below useful (< 500·NA)')
ax2.fill_between([0, 1.5], [500*0, 500*1.5], [1000*0, 1000*1.5], alpha=0.15, color='green',
                 label='Useful range (500-1000·NA)')
ax2.fill_between([0, 1.5], [1000*0, 1000*1.5], [2000, 2000], alpha=0.15, color='orange',
                 label='Empty magnification (> 1000·NA)')

NA_range = np.linspace(0, 1.5, 100)
ax2.plot(NA_range, 500 * NA_range, 'r--', linewidth=1.5)
ax2.plot(NA_range, 1000 * NA_range, 'g--', linewidth=1.5)

ax2.set_xlabel('Numerical Aperture (NA)', fontsize=12)
ax2.set_ylabel('Total Magnification', fontsize=12)
ax2.set_title('Useful vs. Empty Magnification', fontsize=13)
ax2.set_xlim(0, 1.5)
ax2.set_ylim(0, 2000)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('microscope_resolution.png', dpi=150)
plt.show()
```

---

## 4. 망원경

### 4.1 굴절 망원경(케플러식(Keplerian))

굴절 망원경은 대물렌즈(큰 $f_o$)로 초점면에 실상을 형성하고, 접안렌즈(작은 $f_e$)로 그 상을 확대한다:

$$M = -\frac{f_o}{f_e}$$

망원경의 총 길이는 약 $f_o + f_e$이다.

**예**: $f_o = 1000$ mm, $f_e = 25$ mm의 망원경은 $M = -40\times$를 제공한다.

### 4.2 반사 망원경(뉴턴식(Newtonian))

아이작 뉴턴은 초기 굴절 망원경을 괴롭혔던 색수차를 피하기 위해 반사 망원경을 발명했다. 오목 주경(primary mirror)이 빛을 집광하여 초점을 맺고, 작은 평면 부경(secondary mirror)이 광선을 측면에 장착된 접안렌즈 쪽으로 방향을 바꾼다.

**굴절 망원경에 비한 장점**:
- 색수차 없음 (거울은 모든 파장을 동일하게 반사)
- 훨씬 크게 제작 가능 (렌즈는 가장자리에서만 지지 가능; 거울은 뒤에서 지지 가능)
- 유리가 적게 필요 (굴절면 4개 대신 반사면 1개)

### 4.3 카세그레인 망원경(Cassegrain Telescope)

카세그레인 설계는 오목 주경과 볼록 부경을 사용한다. 빛이 경통에 들어가 주경에 반사되고, 부경에 다시 반사되어 주경의 구멍을 통해 주경 뒤쪽의 초점에 도달한다.

**장점**: 주어진 초점 거리에 비해 경통 길이가 매우 짧다. 유효 초점 거리는 $f_{\text{eff}} = f_{\text{주경}} \times |m_{\text{부경}}|$이며, 여기서 $m_{\text{부경}}$은 부경의 배율이다.

| 망원경 | 주경 | 부경 | 장점 |
|--------|------|------|------|
| 뉴턴식(Newtonian) | 오목(포물면) | 평면(대각) | 단순, 저렴 |
| 카세그레인(Cassegrain) | 오목(포물면) | 볼록(쌍곡면) | 소형, 긴 $f_{\text{eff}}$ |
| 리치-크레티앵(Ritchey-Chretien) | 오목(쌍곡면) | 볼록(쌍곡면) | 광시야, 코마 없음 (HST, JWST에서 사용) |
| 슈미트-카세그레인(Schmidt-Cassegrain) | 구면 + 보정판 | 볼록 | 매우 소형, 대량 생산 |

### 4.4 각 분해능: 레일리 기준

원형 조리개 $D$를 가진 모든 망원경(또는 카메라)의 회절 한계 각 분해능은:

$$\theta_R = 1.22 \frac{\lambda}{D}$$

이것이 **레일리 기준(Rayleigh criterion)**이다 — 한 점 광원의 중심 극대가 다른 점 광원의 에어리 패턴(Airy pattern) 첫 번째 극소와 겹칠 때, 두 점 광원이 겨우 분리(resolved)된다.

| 기기 | 구경 | 분해능 (550 nm) |
|------|------|----------------|
| 인간의 눈 | 5 mm | 27.5" (각초) |
| 쌍안경 (50 mm) | 50 mm | 2.8" |
| 아마추어 망원경 (200 mm) | 200 mm | 0.69" |
| 허블 우주 망원경 | 2.4 m | 0.056" |
| 제임스 웹 우주 망원경 | 6.5 m | 0.07" (2 $\mu$m IR에서) |

> **참고**: 지상 망원경은 대체로 회절보다 대기 난류(시상(seeing), 보통 1"–2")에 의해 분해능이 제한된다. **적응 광학(adaptive optics, AO)**은 변형 가능 거울(deformable mirror)로 대기 왜곡을 실시간으로 보정하여 회절 한계에 근접하게 한다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function of the first kind, order 1

# Airy pattern: the diffraction-limited point spread function
# of a circular aperture (telescope, camera, etc.)

def airy_intensity(theta, D, wavelength):
    """
    Calculate the normalized Airy pattern intensity.
    theta: angle from axis (radians)
    D: aperture diameter (meters)
    wavelength: wavelength (meters)
    Returns: I/I_max
    """
    # x = pi * D * sin(theta) / wavelength ≈ pi * D * theta / lambda for small theta
    x = np.pi * D * theta / wavelength
    # I(x) = [2 * J1(x) / x]^2, with I(0) = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        pattern = np.where(np.abs(x) < 1e-10, 1.0, (2 * j1(x) / x)**2)
    return pattern

# Parameters
wavelength = 550e-9  # green light
D_telescope = 0.2    # 200 mm amateur telescope (8-inch)

# Angular range (in arcseconds)
theta_arcsec = np.linspace(-3, 3, 1000)
theta_rad = theta_arcsec * np.pi / (180 * 3600)  # convert arcsec to radians

# Single star Airy pattern
I_single = airy_intensity(theta_rad, D_telescope, wavelength)

# Two stars separated by exactly the Rayleigh criterion
theta_R = 1.22 * wavelength / D_telescope  # Rayleigh limit in radians
sep_arcsec = theta_R * 180 * 3600 / np.pi
print(f"Rayleigh limit: {sep_arcsec:.2f} arcseconds for D={D_telescope*1000:.0f}mm at λ={wavelength*1e9:.0f}nm")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

separations = [0.5 * sep_arcsec, sep_arcsec, 2.0 * sep_arcsec]
labels = ['Unresolved\n(sep < Rayleigh)', 'Just Resolved\n(sep = Rayleigh)', 'Well Resolved\n(sep > Rayleigh)']

for ax, sep, label in zip(axes, separations, labels):
    # Two star Airy patterns, separated by 'sep' arcseconds
    sep_rad = sep * np.pi / (180 * 3600)
    I_star1 = airy_intensity(theta_rad - sep_rad/2, D_telescope, wavelength)
    I_star2 = airy_intensity(theta_rad + sep_rad/2, D_telescope, wavelength)
    I_combined = I_star1 + I_star2

    ax.plot(theta_arcsec, I_star1, 'b--', alpha=0.5, linewidth=1, label='Star 1')
    ax.plot(theta_arcsec, I_star2, 'r--', alpha=0.5, linewidth=1, label='Star 2')
    ax.plot(theta_arcsec, I_combined, 'k-', linewidth=2, label='Combined')
    ax.fill_between(theta_arcsec, 0, I_combined, alpha=0.1, color='gray')

    ax.set_xlabel('Angle (arcseconds)', fontsize=11)
    ax.set_ylabel('Intensity', fontsize=11)
    ax.set_title(f'{label}\nsep = {sep:.2f}"', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.2)

plt.suptitle(f'Rayleigh Criterion (D={D_telescope*1000:.0f}mm, λ={wavelength*1e9:.0f}nm)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('rayleigh_criterion.png', dpi=150)
plt.show()
```

---

## 5. 카메라

### 5.1 기본 카메라 광학

카메라는 렌즈 시스템으로 피사체의 빛을 검출기(필름 또는 디지털 센서)에 집광한다. 주요 매개변수는 다음과 같다:

**초점 거리** $f$: 화각(field of view)과 상 배율을 결정한다.
- 광각: $f < 35$ mm (35mm 환산) — 넓은 화각
- 표준: $f \approx 50$ mm — 인간 눈의 원근감과 유사
- 망원: $f > 70$ mm — 좁은 화각, 높은 배율

**F수(f-stop)**:

$$N = \frac{f}{D}$$

여기서 $D$는 조리개 직경이다. 흔히 사용되는 F수: $f$/1.4, $f$/2, $f$/2.8, $f$/4, $f$/5.6, $f$/8, $f$/11, $f$/16.

한 스톱씩 변하면 집광 면적이 두 배 또는 절반이 된다:

$$\text{면적} \propto D^2 \propto \frac{f^2}{N^2}$$

$f$/2.8에서 $f$/4로 가면 빛이 절반이 된다(한 스톱 어두워짐).

### 5.2 피사계 심도(Depth of Field)

서로 다른 거리의 물체 모두를 동시에 완벽하게 초점 맞출 수는 없다. **피사계 심도(DOF)**는 허용 가능한 선명도를 유지하는 거리 범위다:

$$\text{DOF} \approx \frac{2 N c s^2}{f^2}$$

여기서 $N$은 F수, $c$는 착란원(circle of confusion) 직경(풀프레임에서 일반적으로 $\sim 30$ $\mu$m), $s$는 초점 거리다.

**핵심 관계**:
- $N$ 클수록 (조리개 좁을수록) → DOF 커짐 → 더 넓은 범위 선명
- $f$ 짧을수록 (광각) → DOF 커짐
- $s$ 멀수록 (초점 거리) → DOF 커짐

그래서 인물 사진가는 큰 조리개($f$/1.4–$f$/2.8)를 사용하여 배경을 흐릿하게(보케(bokeh)) 만들고, 풍경 사진가는 작은 조리개($f$/8–$f$/16)를 사용하여 전체를 선명하게 유지한다.

### 5.3 노출과 노출 삼각형(Exposure Triangle)

올바른 노출은 세 가지 설정에 달려 있다:

1. **조리개(Aperture)** ($N$): 빛의 양과 피사계 심도 제어
2. **셔터 속도(Shutter speed)** ($t$): 모션 블러와 총 빛의 양 제어
3. **ISO 감도**: 센서 증폭 및 잡음 제어

노출값(EV)은:

$$\text{EV} = \log_2\left(\frac{N^2}{t}\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Depth of field calculation for different camera settings
# Shows the relationship between f-number, focal length, and DOF

def depth_of_field(f_mm, N, s_m, c_mm=0.030):
    """
    Calculate depth of field.
    f_mm: focal length in mm
    N: f-number
    s_m: subject distance in meters
    c_mm: circle of confusion diameter in mm (0.030 for full-frame)
    Returns: (near_limit, far_limit, DOF) in meters
    """
    f_m = f_mm / 1000
    c_m = c_mm / 1000

    # Hyperfocal distance: H = f^2 / (N * c) + f
    H = f_m**2 / (N * c_m) + f_m

    # Near and far limits of acceptable sharpness
    near = s_m * (H - f_m) / (H + s_m - 2*f_m)
    if H > s_m:
        far = s_m * (H - f_m) / (H - s_m)
    else:
        far = float('inf')  # everything to infinity is in focus

    DOF = far - near if np.isfinite(far) else float('inf')
    return near, far, DOF

# Compare DOF for different scenarios
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: DOF vs f-number for 50mm lens at 3m focus distance
f_numbers = np.array([1.4, 2, 2.8, 4, 5.6, 8, 11, 16, 22])
near_limits = []
far_limits = []
for N in f_numbers:
    near, far, dof = depth_of_field(50, N, 3.0)
    near_limits.append(near)
    far_limits.append(min(far, 20))  # cap for plotting

ax1.fill_between(f_numbers, near_limits, far_limits, alpha=0.3, color='green', label='In-focus range')
ax1.plot(f_numbers, near_limits, 'b-o', linewidth=1.5, markersize=4, label='Near limit')
ax1.plot(f_numbers, far_limits, 'r-o', linewidth=1.5, markersize=4, label='Far limit')
ax1.axhline(3.0, color='gray', linestyle='--', alpha=0.5, label='Focus distance (3m)')
ax1.set_xlabel('F-number', fontsize=12)
ax1.set_ylabel('Distance (m)', fontsize=12)
ax1.set_title('Depth of Field vs F-number\n(50mm lens, subject at 3m)', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 15)

# Right: DOF vs focal length at f/4, subject at 3m
focal_lengths = np.array([24, 35, 50, 85, 100, 135, 200])
near_limits2 = []
far_limits2 = []
for fl in focal_lengths:
    near, far, dof = depth_of_field(fl, 4.0, 3.0)
    near_limits2.append(near)
    far_limits2.append(min(far, 30))

ax2.fill_between(focal_lengths, near_limits2, far_limits2, alpha=0.3, color='green')
ax2.plot(focal_lengths, near_limits2, 'b-o', linewidth=1.5, markersize=4, label='Near limit')
ax2.plot(focal_lengths, far_limits2, 'r-o', linewidth=1.5, markersize=4, label='Far limit')
ax2.axhline(3.0, color='gray', linestyle='--', alpha=0.5, label='Focus distance (3m)')
ax2.set_xlabel('Focal Length (mm)', fontsize=12)
ax2.set_ylabel('Distance (m)', fontsize=12)
ax2.set_title('Depth of Field vs Focal Length\n(f/4, subject at 3m)', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 15)

plt.tight_layout()
plt.savefig('depth_of_field.png', dpi=150)
plt.show()
```

---

## 6. 광학계의 분해능 한계

### 6.1 레일리 기준 (재검토)

원형 조리개 직경 $D$를 가진 모든 회절 한계 영상 시스템에서:

$$\theta_R = 1.22\frac{\lambda}{D}$$

상면(image plane)에서의 공간 분해능은:

$$\Delta x = 1.22\frac{\lambda f}{D} = 1.22 \lambda N$$

여기서 $N = f/D$는 F수다. 즉:
- $f$/2인 카메라는 $1.22 \times 0.55 \times 2 \approx 1.3$ $\mu$m 크기까지 분해 가능
- $f$/11인 카메라는 $\approx 7.4$ $\mu$m 크기까지 분해 가능

픽셀 피치가 $\sim 1$ $\mu$m인 현대 스마트폰 카메라는 작은 조리개에서 실제로 회절의 영향을 받기 시작한다.

### 6.2 스패로 기준(Sparrow Criterion)

레일리 기준은 다소 임의적이다. **스패로 기준**은 두 점의 합성 강도 프로파일에서 두 극대 사이의 딥(dip)이 사라지는 분리 거리를 분해능 한계로 정의한다:

$$\theta_S = 0.95 \frac{\lambda}{D} \approx 0.78 \, \theta_R$$

스패로 한계는 레일리 한계보다 약 22% 더 엄격하다.

### 6.3 도스 한계(Dawes Limit, 망원경 경험적 기준)

구경 $D$ (mm 단위)의 망원경으로 이중성(double star)을 육안으로 관측할 때의 경험적 한계:

$$\theta_{\text{Dawes}} \approx \frac{116}{D} \text{ (각초, } D \text{는 mm 단위)}$$

이 경험적 한계는 회절과 인간 눈의 대비 감도를 모두 반영한다.

---

## 7. 광학 기기 비교

| 기기 | 핵심 지표 | 전형적인 값 | 제한 요인 |
|------|-----------|------------|----------|
| 돋보기(Magnifying glass) | 각 배율 | 2–10$\times$ | 높은 $M$에서의 수차 |
| 복합 현미경(Compound microscope) | 분해능 (NA) | 0.2–1.4 NA | 회절 ($d \geq 0.61\lambda$/NA) |
| 굴절 망원경(Refracting telescope) | 각 배율 | $f_o/f_e$ | 색수차, 크기 |
| 반사 망원경(Reflecting telescope) | 집광력 ($\propto D^2$) | $D$ 최대 10 m | 거울 제작, 시상 |
| 카메라(Camera) | 공간 분해능 | $1.22\lambda N$ | 회절, 픽셀 크기, 잡음 |
| 인간의 눈(Human eye) | 각 분해능 | $\sim 1'$ | 원추세포 간격, 수차 |

---

## 8. 심화 주제: 적응 광학(Adaptive Optics)

지상 망원경은 대기 난류에 의해 분해능이 저하된다 — 온도가 다른 공기 덩어리들이 약한 렌즈의 무작위 집합처럼 작용하여 상을 흐리게 한다. **프리드 매개변수(Fried parameter)** $r_0$가 난류를 특성화한다:

$$\theta_{\text{seeing}} \sim \frac{\lambda}{r_0}$$

가시광선 파장에서 일반적으로 $r_0 \sim 10$–$20$ cm이므로, 시상 한계는 약 0.5"–1.5" — 대형 망원경의 회절 한계보다 훨씬 나쁘다.

**적응 광학(AO)**은 이를 실시간으로 보정한다:
1. **파면 센서(wavefront sensor)**가 왜곡을 측정한다 (자연 별 또는 레이저 가이드 별 사용)
2. **변형 가능 거울(deformable mirror)** (수백~수천 개의 구동기(actuator))이 형태를 조정하여 왜곡을 상쇄
3. 보정된 상이 회절 한계에 근접

8–10 m급 망원경의 현대 AO 시스템은 근적외선에서 일상적으로 0.05"–0.1" 분해능을 달성한다.

---

## 연습 문제

### 연습 1: 눈 교정

어느 사람이 원점이 50 cm(근시)이고 근점이 15 cm이다.

(a) 근시를 교정하려면 (원점을 무한대로 이동) 몇 디옵터의 렌즈가 필요한가?

(b) 교정 렌즈를 착용하면 새로운 근점은 얼마인가?

(c) 55세가 되어 조절 능력이 감소해 근점이 100 cm가 되었다. 25 cm에서 독서를 위한 추가 렌즈 파워(이중 초점 추가분)는 얼마인가?

### 연습 2: 현미경 설계

현미경에 $40\times$ 대물렌즈(NA = 0.65, $f_o = 4.5$ mm)와 $10\times$ 접안렌즈($f_e = 25$ mm)가 장착되어 있다.

(a) 총 배율은 얼마인가?

(b) $\lambda = 550$ nm에서 이론적 분해능은 얼마인가?

(c) 총 배율이 유용한 범위 안에 있는가, 아니면 일부가 "공배율"인가?

(d) 오일 침지($n = 1.52$)를 사용하면 분해능은 어떻게 변하는가?

### 연습 3: 망원경 비교

목성을 관측하기 위해 두 망원경을 비교한다:
- 망원경 A: $D = 100$ mm 굴절 망원경, $f = 900$ mm, 접안렌즈 $f_e = 9$ mm
- 망원경 B: $D = 200$ mm 뉴턴식 반사 망원경, $f = 1200$ mm, 접안렌즈 $f_e = 12$ mm

(a) 각각의 배율을 계산하라.

(b) 각각의 회절 한계 각 분해능을 계산하라 (550 nm에서).

(c) 어느 망원경이 더 세밀하게 보이며, 그 이유는? (시상이 2"라고 가정)

(d) 각각의 사출 동공(exit pupil)을 계산하라. 야간 관측에는 어느 것이 더 적합한가?

### 연습 4: 카메라 설정

35 mm 렌즈로 초점 거리 5 m의 풍경을 촬영한다.

(a) 피사계 심도가 2 m에서 무한대까지 이어지려면 어떤 F수가 필요한가? (�힌트: 원거리 한계는 과초점 거리(hyperfocal distance)에서 무한대에 도달한다.)

(b) 이 F수에서 회절 한계 분해능은 얼마인가?

(c) 센서의 픽셀이 5 $\mu$m라면, 이 시스템은 회절 한계인가 픽셀 한계인가?

### 연습 5: 위성 영상

정찰 위성이 고도 200 km 궤도를 돌며 $D = 2.4$ m의 망원경을 탑재하고 있다.

(a) $\lambda = 550$ nm에서 이론적 각 분해능은 얼마인가?

(b) 고도 200 km에서 이 값은 지상의 몇 m에 해당하는가?

(c) 이 위성이 자동차 번호판(문자 높이 $\sim 8$ cm)을 읽을 수 있는가? 그 이유는?

---

## 요약

| 개념 | 핵심 공식/사실 |
|------|---------------|
| 눈의 광학 | 총 파워 $\approx$ 60 D; 근점 25 cm; 분해능 $\sim 1'$ |
| 돋보기 | $M = D/f$ (이완 상태); $M = D/f + 1$ (최대); $D = 25$ cm |
| 복합 현미경 | $M = -(L/f_o)(D/f_e)$; 분해능 $d = 0.61\lambda$/NA |
| 개구수 | NA = $n\sin\alpha$; 오일 침지로 NA 1.0 초과 가능 |
| 유효 배율 | $500\cdot$NA ~ $1000\cdot$NA; 이를 초과하면 공배율 |
| 망원경 배율 | $M = -f_o/f_e$ (케플러식) |
| 레일리 기준 | $\theta_R = 1.22\lambda/D$ — 원형 조리개의 각 분해능 |
| 카메라 F수 | $N = f/D$; 한 스톱당 빛 두 배; 작은 $N$ = 밝음 |
| 피사계 심도 | DOF $\approx 2Ncs^2/f^2$; 큰 $N$ 또는 짧은 $f$ → 넓은 DOF |
| 노출값 | EV = $\log_2(N^2/t)$ |
| 적응 광학 | 변형 가능 거울로 대기 난류 보정 → 회절 한계에 근접 |

---

[← 이전: 03. 거울과 렌즈](03_Mirrors_and_Lenses.md) | [다음: 05. 파동광학 — 간섭 →](05_Wave_Optics_Interference.md)
