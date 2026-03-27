# 05. 파동광학 — 간섭

[← 이전: 04. 광학 기기](04_Optical_Instruments.md) | [다음: 06. 회절 →](06_Diffraction.md)

---

## 학습 목표

1. 중첩 원리(superposition principle)를 적용하여 둘 이상의 가간섭성(coherent) 광원에서 발생하는 간섭 무늬를 계산한다
2. 영(Young)의 이중 슬릿 실험의 세기 분포를 유도하고 무늬 간격을 예측한다
3. 반사 시 위상 변화 및 보강·상쇄 간섭 조건을 포함하여 박막 간섭(thin-film interference)을 분석한다
4. 마이컬슨 간섭계(Michelson interferometer)의 작동 원리와 계측학(metrology)에서의 응용을 설명한다
5. 시간적 가간섭성(temporal coherence)과 공간적 가간섭성(spatial coherence)을 구분하고, 이를 광원의 특성과 연관짓는다
6. 광원의 스펙트럼 폭으로부터 가간섭 길이(coherence length)와 가간섭 시간(coherence time)을 계산한다
7. 실용 시스템에 간섭 개념을 적용한다: 무반사 코팅(anti-reflection coatings), 뉴턴의 고리(Newton's rings), 파브리-페로 에탈론(Fabry-Perot etalons)

---

## 왜 중요한가

간섭(Interference)은 파동 거동의 결정적인 특징이다. 1801년 영(Young)의 이중 슬릿 실험은 빛의 본성에 관한 수 세기의 논쟁을 파동 이론의 승리로 결론지었다. 오늘날 간섭은 인류의 가장 정밀한 측정 중 일부를 뒷받침한다: LIGO는 레이저 간섭계를 이용해 $10^{-18}$ m — 양성자 지름의 만 분의 일 — 의 길이 변화를 측정하여 중력파를 검출한다. 모든 카메라 렌즈와 안경에 적용된 무반사 코팅도 박막 간섭에 기반한다. 광섬유 센서, 의료용 광 간섭 단층 촬영(OCT, optical coherence tomography), 미터 원기의 교정 모두 간섭 이해에 달려 있다.

> **비유**: 간섭은 같은 음을 연주하는 두 스피커가 방 안에 있는 것과 같다. 어떤 위치에서는 음파가 더해져 크게 들리고(밝은 지점), 다른 위치에서는 상쇄되어 조용해진다(어두운 지점). 이제 "소리"를 "빛"으로, "크고 조용한"을 "밝고 어두운"으로 바꾸면 간섭 무늬가 된다. 핵심 조건은 동일하다: 두 광원은 가간섭성이어야 한다(일정한 위상 관계를 유지해야 한다).

---

## 1. 중첩 원리(Superposition Principle)

### 1.1 원리

둘 이상의 파동이 공간에서 겹칠 때, 합성 전기장(electric field)은 개별 전기장의 벡터 합이다:

$$\mathbf{E}_{\text{total}}(\mathbf{r}, t) = \mathbf{E}_1(\mathbf{r}, t) + \mathbf{E}_2(\mathbf{r}, t) + \cdots$$

이 선형성(linearity)은 진공 및 선형 매질에서 맥스웰 방정식(Maxwell's equations)이 선형이라는 사실에서 직접 나온다.

### 1.2 중첩된 파동의 세기

같은 각진동수 $\omega$와 편광 방향을 갖는 두 단색광(monochromatic wave)에 대해:

$$E_1 = E_{01} \cos(k r_1 - \omega t + \phi_1)$$
$$E_2 = E_{02} \cos(k r_2 - \omega t + \phi_2)$$

두 파동이 모두 존재하는 지점에서의 시간 평균 세기(time-averaged intensity):

$$I = I_1 + I_2 + 2\sqrt{I_1 I_2}\cos\delta$$

여기서 $\delta$는 **위상차(phase difference)**:

$$\delta = k(r_2 - r_1) + (\phi_2 - \phi_1) = \frac{2\pi}{\lambda}\Delta r + \Delta\phi$$

이고 $\Delta r = r_2 - r_1$은 **경로차(path difference)**이다.

**핵심 경우**:
- $\delta = 0, \pm 2\pi, \pm 4\pi, \ldots$ → **보강 간섭(Constructive interference)**: $I = I_1 + I_2 + 2\sqrt{I_1 I_2} = (\sqrt{I_1} + \sqrt{I_2})^2$
- $\delta = \pm\pi, \pm 3\pi, \ldots$ → **상쇄 간섭(Destructive interference)**: $I = (\sqrt{I_1} - \sqrt{I_2})^2$

등세기 $I_1 = I_2 = I_0$인 경우:

$$I = 2I_0(1 + \cos\delta) = 4I_0 \cos^2\left(\frac{\delta}{2}\right)$$

최대 세기는 $4I_0$ ($2I_0$가 아님) — 에너지는 생성되거나 소멸되지 않고 어두운 무늬에서 밝은 무늬로 *재분배*될 뿐이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Interference of two plane waves: intensity as a function of phase difference
# Shows how intensity oscillates between 0 and 4*I_0

delta = np.linspace(-4*np.pi, 4*np.pi, 500)

# Equal intensity case: I = 4*I_0 * cos^2(delta/2)
I_equal = 4 * np.cos(delta / 2)**2

# Unequal intensity case: I1 = 1, I2 = 0.25
I1, I2 = 1.0, 0.25
I_unequal = I1 + I2 + 2*np.sqrt(I1*I2) * np.cos(delta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: equal intensities
ax1.plot(delta/np.pi, I_equal, 'b-', linewidth=2)
ax1.set_xlabel('Phase difference $\\delta / \\pi$', fontsize=12)
ax1.set_ylabel('Intensity $I / I_0$', fontsize=12)
ax1.set_title('Interference: Equal Intensities ($I_1 = I_2 = I_0$)', fontsize=13)
ax1.axhline(2, color='gray', linestyle='--', alpha=0.5, label='$2I_0$ (no interference)')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.2, 4.5)

# Right: unequal intensities
ax2.plot(delta/np.pi, I_unequal, 'r-', linewidth=2)
ax2.set_xlabel('Phase difference $\\delta / \\pi$', fontsize=12)
ax2.set_ylabel('Intensity', fontsize=12)
ax2.set_title('Interference: Unequal Intensities ($I_1 = 1, I_2 = 0.25$)', fontsize=13)
ax2.axhline(I1 + I2, color='gray', linestyle='--', alpha=0.5,
            label=f'$I_1 + I_2$ = {I1+I2} (no interference)')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Mark the visibility/contrast
I_max = (np.sqrt(I1) + np.sqrt(I2))**2
I_min = (np.sqrt(I1) - np.sqrt(I2))**2
V = (I_max - I_min) / (I_max + I_min)
ax2.annotate(f'$I_{{max}}$ = {I_max:.2f}\n$I_{{min}}$ = {I_min:.2f}\nVisibility V = {V:.2f}',
             xy=(0, I_max), xytext=(1.5, I_max-0.3), fontsize=10,
             arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.savefig('superposition_intensity.png', dpi=150)
plt.show()
```

### 1.3 무늬 가시도(Fringe Visibility, 대비)

간섭 무늬의 **가시도(visibility)** (또는 **대비(contrast)**)는 무늬가 얼마나 선명한지를 정량화한다:

$$V = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}}$$

등세기 빔의 경우: $V = 1$ (완전한 대비).
세기가 다른 경우: $V = \frac{2\sqrt{I_1 I_2}}{I_1 + I_2} < 1$.

가시도는 부분 가간섭성(partial coherence)에 의해서도 감소한다(6절 참조).

---

## 2. 영의 이중 슬릿 실험(Young's Double-Slit Experiment)

### 2.1 장치와 분석

토머스 영(Thomas Young)의 실험(1801)은 물리학에서 가장 중요한 실험 중 하나이다. 단색광이 간격 $d$만큼 떨어진 두 개의 좁은 슬릿을 비춘다. 거리 $L \gg d$에 위치한 스크린 위에 간섭 무늬가 나타난다.

스크린 위 각도 $\theta$ 방향의 점까지의 경로차:

$$\Delta r = d\sin\theta$$

**밝은 무늬(보강 간섭)**:

$$d\sin\theta = m\lambda, \qquad m = 0, \pm 1, \pm 2, \ldots$$

**어두운 무늬(상쇄 간섭)**:

$$d\sin\theta = \left(m + \frac{1}{2}\right)\lambda, \qquad m = 0, \pm 1, \pm 2, \ldots$$

### 2.2 무늬 간격

소각 근사($\sin\theta \approx \tan\theta = y/L$)에서, 스크린 위 $m$번째 밝은 무늬의 위치:

$$y_m = \frac{m\lambda L}{d}$$

**무늬 간격(fringe spacing)** (인접한 밝은 무늬 사이의 거리):

$$\Delta y = \frac{\lambda L}{d}$$

이것은 광학에서 가장 우아한 결과 중 하나이다: 무늬 간격은 파장에 정비례하고 슬릿 간격에 반비례한다.

### 2.3 세기 분포

동일한 두 개의 좁은 슬릿에 대한 세기 무늬:

$$I(\theta) = 4I_0 \cos^2\left(\frac{\pi d \sin\theta}{\lambda}\right)$$

여기서 $I_0$는 단일 슬릿에서의 세기이다. $\cos^2$ 무늬는 $d\sin\theta = m\lambda$에서 최댓값을 갖고 $d\sin\theta = (m+\frac{1}{2})\lambda$에서 영(0)이 된다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Young's double-slit experiment: intensity pattern on the screen

wavelength = 550e-9      # green light (m)
d = 0.2e-3               # slit separation: 0.2 mm
L = 1.0                  # screen distance: 1 m

# Position on screen
y = np.linspace(-0.01, 0.01, 1000)  # ±10 mm
theta = np.arctan(y / L)             # angle (exact)

# Two-slit interference pattern (ignoring single-slit diffraction envelope)
delta = 2 * np.pi * d * np.sin(theta) / wavelength
I_two_slit = 4 * np.cos(delta / 2)**2

# Fringe spacing
fringe_spacing = wavelength * L / d * 1000  # in mm
print(f"Fringe spacing: {fringe_spacing:.2f} mm")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top: intensity vs position on screen
ax1.plot(y * 1000, I_two_slit, 'b-', linewidth=1.5)
ax1.set_xlabel('Position on screen (mm)', fontsize=12)
ax1.set_ylabel('Intensity $I / I_0$', fontsize=12)
ax1.set_title(f"Young's Double-Slit: d = {d*1e3:.1f} mm, L = {L:.1f} m, "
              f"λ = {wavelength*1e9:.0f} nm\nFringe spacing = {fringe_spacing:.2f} mm",
              fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 4.5)

# Mark the central maximum and first-order fringes
for m in range(-3, 4):
    y_m = m * wavelength * L / d * 1000  # in mm
    if abs(y_m) < 10:
        ax1.axvline(y_m, color='red', linestyle=':', alpha=0.4)
        ax1.text(y_m, 4.2, f'm={m}', ha='center', fontsize=8, color='red')

# Bottom: 2D visualization of the interference pattern
Y, X = np.meshgrid(np.linspace(-8, 8, 400), np.linspace(-3, 3, 100))
theta_2d = np.arctan(Y / (L * 1000))
I_2d = 4 * np.cos(np.pi * d * np.sin(theta_2d) / wavelength)**2
ax2.imshow(I_2d, extent=[-8, 8, -3, 3], aspect='auto', cmap='inferno',
           vmin=0, vmax=4)
ax2.set_xlabel('Position on screen (mm)', fontsize=12)
ax2.set_ylabel('Vertical position (mm)', fontsize=12)
ax2.set_title('Fringe Pattern (2D View)', fontsize=13)

plt.tight_layout()
plt.savefig('youngs_double_slit.png', dpi=150)
plt.show()
```

---

## 3. 박막 간섭(Thin-Film Interference)

### 3.1 반사 시 위상 변화

빛이 경계면에서 반사될 때, 위상 변화는 굴절률에 따라 달라진다:

- **밀한 매질에서의 반사** ($n_1 < n_2$): $\pi$ (반파장)의 위상 변화
- **소한 매질에서의 반사** ($n_1 > n_2$): 위상 변화 없음

이는 줄 위의 파동과 유사하다: 고정단(밀한 매질)에서의 반사는 펄스를 반전시키고, 자유단(소한 매질)에서의 반사는 반전시키지 않는다.

### 3.2 박막 간섭의 조건

위쪽은 굴절률 $n_1$, 아래쪽은 굴절률 $n_2$인 매질 사이에 두께 $t$, 굴절률 $n_f$인 박막이 끼어 있는 경우를 생각하자. 빛은 박막의 위·아래 표면 모두에서 반사된다.

두 반사 빔 사이의 총 위상차:

$$\delta = \frac{2\pi}{\lambda} \cdot 2n_f t\cos\theta_t + \delta_{\text{reflection}}$$

여기서 $\theta_t$는 박막 내부에서의 굴절각이고 $\delta_{\text{reflection}}$은 두 경계면에서의 위상 변화를 합산한 것이다.

**일반적인 경우: 공기-박막-유리** ($n_1 < n_f < n_2$):
- 위 표면: $\pi$ 위상 변화 (공기 → 박막, 밀한 매질에서 반사)
- 아래 표면: $\pi$ 위상 변화 (박막 → 유리, 밀한 매질에서 반사)
- 알짜 반사 위상 변화: $\pi - \pi = 0$ (두 $\pi$ 변화가 상쇄!)

**보강 반사**: $2n_f t\cos\theta_t = m\lambda$ ($m = 0, 1, 2, \ldots$)

**일반적인 경우: 공기-박막-공기** ($n_1 = n_2 < n_f$, 예: 비눗방울 막):
- 위 표면: $\pi$ 위상 변화
- 아래 표면: 위상 변화 없음
- 알짜 반사 위상 변화: $\pi$

**보강 반사**: $2n_f t\cos\theta_t = (m + \frac{1}{2})\lambda$

> **비유**: 두 반사 빔을 트랙을 달리는 두 주자라고 생각하자. 한 주자(위에서 반사)는 즉시 출발하고, 다른 주자(아래에서 반사)는 박막을 통과해 내려갔다가 올라온 후 출발한다. 출발선으로 돌아올 때 보조가 맞으면 보강 반사(밝음)가 된다. 한 주자가 반 바퀴 앞서 있으면 상쇄(어둠)가 된다. 반사 시 위상 변화에 의한 "반 바퀴 선행"이 보조가 맞는 조건을 바꾼다.

### 3.3 무반사 코팅(Anti-Reflection Coatings)

유리 표면($n_g \approx 1.52$)의 반사를 최소화하려면 두께 $t$, 굴절률 $n_c$의 코팅을 아래 조건으로 적용한다:

1. **진폭 매칭**: $n_c = \sqrt{n_{\text{air}} \cdot n_g} = \sqrt{1.52} \approx 1.23$ (두 반사 빔의 진폭이 같을 때 반사 최소)
2. **상쇄 간섭**: $2n_c t = \frac{\lambda}{2}$ → $t = \frac{\lambda}{4n_c}$ (1/4 파장 두께)

$\lambda = 550$ nm, $n_c = 1.23$이면: $t = 112$ nm.

MgF$_2$ ($n = 1.38$)는 가장 흔한 단층 코팅 재료이다. 이상적인 $n_c = 1.23$은 구현하기 어려우므로 MgF$_2$가 실용적인 절충안이다.

다층 코팅(고굴절률과 저굴절률 층을 교대로 쌓은 것)은 넓은 파장 범위에 걸쳐 반사율 0.1% 이하를 달성할 수 있다 — 이것이 **광대역 무반사(BBAR, broadband anti-reflection)** 코팅이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Thin-film interference: reflectance of a single-layer anti-reflection coating
# as a function of wavelength

def single_layer_reflectance(wavelength, t, n_coat, n_glass, n_air=1.0):
    """
    Calculate reflectance of a single-layer coating on glass.
    Uses the exact formula from thin-film optics (normal incidence).

    wavelength: in nm
    t: coating thickness in nm
    n_coat: coating refractive index
    n_glass: glass refractive index
    n_air: ambient refractive index (usually 1.0)
    """
    # Fresnel reflection coefficients at each interface (normal incidence)
    r1 = (n_air - n_coat) / (n_air + n_coat)   # air → coating
    r2 = (n_coat - n_glass) / (n_coat + n_glass)  # coating → glass

    # Phase accumulated in the film (round trip)
    delta = 4 * np.pi * n_coat * t / wavelength

    # Total reflectance (Airy formula for a single film)
    numerator = r1**2 + r2**2 + 2 * r1 * r2 * np.cos(delta)
    denominator = 1 + r1**2 * r2**2 + 2 * r1 * r2 * np.cos(delta)
    R = numerator / denominator
    return R

# Wavelength range: 350 to 800 nm (covering visible spectrum)
wavelengths = np.linspace(350, 800, 500)
n_glass = 1.52

# Design wavelength: 550 nm (green, center of visible spectrum)
lambda_design = 550.0

# Case 1: MgF2 coating (n = 1.38), quarter-wave at 550 nm
n_mgf2 = 1.38
t_mgf2 = lambda_design / (4 * n_mgf2)

# Case 2: Ideal coating (n = sqrt(1.52) ≈ 1.233)
n_ideal = np.sqrt(n_glass)
t_ideal = lambda_design / (4 * n_ideal)

# Case 3: No coating (bare glass)
R_bare = ((1 - n_glass) / (1 + n_glass))**2

R_mgf2 = single_layer_reflectance(wavelengths, t_mgf2, n_mgf2, n_glass)
R_ideal = single_layer_reflectance(wavelengths, t_ideal, n_ideal, n_glass)

fig, ax = plt.subplots(figsize=(12, 6))

ax.axhline(R_bare * 100, color='gray', linestyle='--', linewidth=1.5,
           label=f'Bare glass ({R_bare*100:.1f}%)')
ax.plot(wavelengths, R_mgf2 * 100, 'b-', linewidth=2,
        label=f'MgF$_2$ (n={n_mgf2}, t={t_mgf2:.0f} nm)')
ax.plot(wavelengths, R_ideal * 100, 'r-', linewidth=2,
        label=f'Ideal (n={n_ideal:.3f}, t={t_ideal:.0f} nm)')

# Shade the visible spectrum for reference
ax.axvspan(380, 700, alpha=0.08, color='yellow', label='Visible range')

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Reflectance (%)', fontsize=12)
ax.set_title('Single-Layer Anti-Reflection Coating Performance', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 5)
ax.set_xlim(350, 800)

# Annotate the minimum
idx_min_mgf2 = np.argmin(R_mgf2)
ax.annotate(f'Min: {R_mgf2[idx_min_mgf2]*100:.2f}% at {wavelengths[idx_min_mgf2]:.0f} nm',
            xy=(wavelengths[idx_min_mgf2], R_mgf2[idx_min_mgf2]*100),
            xytext=(wavelengths[idx_min_mgf2]+80, 1.5),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10, color='blue')

plt.tight_layout()
plt.savefig('anti_reflection_coating.png', dpi=150)
plt.show()
```

### 3.4 뉴턴의 고리(Newton's Rings)

평볼록 렌즈를 평면 유리 위에 놓으면, 렌즈와 유리 사이의 공기층 두께가 반지름 방향으로 변한다. 이로 인해 **뉴턴의 고리(Newton's rings)**라 불리는 원형 간섭 무늬가 생긴다.

곡률 반지름 $R$인 렌즈에서 접촉점으로부터 반지름 $r$에서의 공기층 두께:

$$t(r) = \frac{r^2}{2R}$$

반사광에서의 밝은 고리(아래 경계면의 $\pi$ 위상 변화를 고려):

$$r_m = \sqrt{\left(m + \frac{1}{2}\right)\lambda R}, \qquad m = 0, 1, 2, \ldots$$

어두운 고리:

$$r_m = \sqrt{m \lambda R}, \qquad m = 0, 1, 2, \ldots$$

중앙 점은 반사에서 **어둡고**(유리-공기-유리 경계에서의 $\pi$ 위상 변화 때문), 투과에서 **밝다**.

### 3.5 비눗방울과 기름막

비눗방울과 물 위의 기름막에서 나타나는 영롱한 색상은 박막 간섭에서 비롯된다. 막 두께가 표면에 걸쳐 변하기 때문에, 서로 다른 영역이 서로 다른 파장에 대해 보강 간섭 조건을 만족하여 무지개 색상이 나타난다.

공기 중($n_1 = n_2 = 1$) 비눗방울 막($n \approx 1.33$)의 경우:
- 위 표면 반사: $\pi$ 위상 변화
- 아래 표면 반사: 위상 변화 없음
- 보강 반사: $2nt = (m + \frac{1}{2})\lambda$

두께 300 nm의 막은 청색($\lambda \approx 400$ nm, $m = 0$)을 강하게 반사하여 파랗게 보인다. 막이 중력으로 인해 얇아지면 색상이 변하고, 결국 가시광선을 반사하지 않을 만큼 얇아져 터지기 직전에 검게 보인다.

---

## 4. 마이컬슨 간섭계(Michelson Interferometer)

### 4.1 설계와 작동

마이컬슨 간섭계는 빛의 빔을 두 경로로 분리하여 각각 반사시킨 후 재결합하여 간섭 무늬를 만든다. 이것은 지금까지 고안된 가장 다용도적이고 정밀한 광학 기기 중 하나이다.

**구성 요소**:
1. **빔 분할기(beam splitter)**: 빛의 절반을 투과하고 절반을 반사하는 반은도금 거울
2. **거울 1** (고정): 투과된 빔을 반사
3. **거울 2** (이동 가능): 반사된 빔을 반사
4. **검출기/스크린**: 재결합된 빔이 간섭하는 곳

두 팔 사이의 광 경로차:

$$\Delta = 2(d_1 - d_2)$$

여기서 $d_1$과 $d_2$는 빔 분할기에서 각 거울까지의 거리이다.

### 4.2 무늬 형태

**원형 무늬(equal inclination fringes, 등경사 무늬)**: 거울이 정확히 수직일 때, 광 경로차가 같은 고리들이 광축을 중심으로 나타난다. 밝은 고리의 조건:

$$2d\cos\theta = m\lambda$$

여기서 $d = |d_1 - d_2|$는 법선 입사 시의 경로차이다.

**직선 무늬(equal thickness fringes, 등두께 무늬)**: 거울 하나가 약간 기울어져 있으면 유효 공기 쐐기(air wedge)가 형성되어 평행한 무늬가 생긴다.

거울 2를 $\lambda/2$만큼 이동하면 각 무늬가 한 무늬 간격만큼 이동한다. 무늬를 세어 파장 이하의 정밀도로 변위를 측정할 수 있다.

### 4.3 응용

**계측학(Metrology)**: $\lambda/10 \approx 50$ nm 이하의 정밀도로 길이 측정.

**분광학 (FTIR)**: 푸리에 변환 적외선 분광법(Fourier-transform infrared spectroscopy)은 스캔하는 마이컬슨 간섭계를 사용한다. 간섭도(interferogram, 세기 vs. 거울 변위)는 스펙트럼의 푸리에 변환이다.

**중력파 검출**: LIGO는 각 팔에 파브리-페로 공동(Fabry-Perot cavity)이 있는 킬로미터 규모의 마이컬슨 간섭계를 사용하여 $10^{-18}$ m의 변위 감도를 달성한다.

**마이컬슨-몰리 실험(1887)**: 물리학에서 가장 유명한 영 결과(null result). 이 실험은 빛의 속도의 방향 의존성을 관찰하여 지구가 "발광 에테르(luminiferous aether)"를 통해 움직이는 것을 검출하려 했다. 영 결과는 아인슈타인의 특수 상대성 이론으로 직결되었다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Michelson interferometer: simulate circular fringe patterns
# for different mirror separations

def michelson_fringes(d_um, wavelength_nm, max_angle_deg=5, N_points=500):
    """
    Calculate the Michelson interferometer fringe pattern.

    d_um: path difference in micrometers
    wavelength_nm: wavelength in nanometers
    max_angle_deg: maximum viewing angle in degrees
    N_points: number of points in each dimension

    Returns: 2D intensity array and extent
    """
    wavelength_um = wavelength_nm / 1000  # convert to micrometers

    # Create 2D grid of angles
    theta = np.linspace(-np.deg2rad(max_angle_deg),
                         np.deg2rad(max_angle_deg), N_points)
    TX, TY = np.meshgrid(theta, theta)
    angle = np.sqrt(TX**2 + TY**2)  # radial angle

    # Phase difference: delta = 4*pi*d*cos(theta) / lambda
    delta = 4 * np.pi * d_um * np.cos(angle) / wavelength_um

    # Intensity: I = I_0 * (1 + cos(delta)) / 2 (for 50/50 beam splitter)
    I = (1 + np.cos(delta)) / 2

    extent = [-max_angle_deg, max_angle_deg, -max_angle_deg, max_angle_deg]
    return I, extent

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Show fringe patterns for increasing path differences
# Larger d means more fringes (higher order at center)
path_diffs = [5, 20, 50, 100]  # micrometers

for ax, d in zip(axes, path_diffs):
    I, extent = michelson_fringes(d, 550)
    ax.imshow(I, extent=extent, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'd = {d} μm\n({2*d/0.55:.0f} fringes)', fontsize=11)
    ax.set_xlabel('Angle (deg)', fontsize=9)
    ax.set_ylabel('Angle (deg)', fontsize=9)

plt.suptitle('Michelson Interferometer: Circular Fringes (λ = 550 nm)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('michelson_fringes.png', dpi=150)
plt.show()
```

---

## 5. 다중 빔 간섭: 파브리-페로 에탈론(Fabry-Perot Etalon)

### 5.1 두 빔에서 다중 빔으로

박막 간섭에서는 두 개의 반사 빔만 고려했다. 실제로는 빛이 박막 내부에서 여러 번 반사된다. 표면 반사율이 높을 때($R \to 1$), 이러한 다중 반사가 중요해지며 두 빔 무늬보다 훨씬 날카로운 무늬를 만든다.

### 5.2 에어리 함수(Airy Function)

파브리-페로 에탈론(거리 $d$만큼 떨어진 두 개의 평행한 고반사율 거울)에서 투과 세기:

$$I_t = \frac{I_0}{1 + F\sin^2(\delta/2)}$$

여기서:
- $\delta = \frac{4\pi n d \cos\theta}{\lambda}$는 왕복 위상(round-trip phase)
- $F = \frac{4R}{(1-R)^2}$는 **피네스 계수(coefficient of finesse)**
- $R$은 각 거울의 반사율

이것이 **에어리 함수(Airy function)**이다. $\delta = 2m\pi$에서 날카로운 투과 최대(밝은 무늬)를 갖고, 그 사이에 넓고 어두운 구간이 있다.

### 5.3 피네스(Finesse)와 분해능

**피네스** $\mathcal{F}$는 투과 봉우리의 날카로움을 나타낸다:

$$\mathcal{F} = \frac{\pi\sqrt{F}}{2} = \frac{\pi\sqrt{R}}{1-R}$$

| 거울 반사율 $R$ | 피네스 $\mathcal{F}$ | 계수 $F$ |
|:-----------------------:|:---------------------:|:---------------:|
| 0.04 (무코팅 유리) | 0.64 | 0.17 |
| 0.50 | 4.4 | 8.0 |
| 0.90 | 30 | 360 |
| 0.99 | 313 | 39,600 |

**자유 스펙트럼 범위(FSR, free spectral range)**는 인접한 투과 봉우리 사이의 주파수 간격이다:

$$\Delta\nu_{\text{FSR}} = \frac{c}{2nd}$$

**스펙트럼 분해능**:

$$\delta\nu = \frac{\Delta\nu_{\text{FSR}}}{\mathcal{F}} = \frac{c}{2nd\mathcal{F}}$$

**분해 능력(resolving power)**:

$$\mathcal{R} = \frac{\nu}{\delta\nu} = m \cdot \mathcal{F}$$

여기서 $m$은 간섭 차수이다. $\mathcal{F} = 30$, $m = 10^4$ 차수에서 동작하는 파브리-페로 에탈론은 $\mathcal{R} = 3 \times 10^5$로 스펙트럼 특징을 분해할 수 있다 — 회절 격자를 훨씬 능가한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Fabry-Perot etalon: transmission (Airy function) for different reflectivities
# Shows how higher reflectivity produces sharper transmission peaks

delta = np.linspace(0, 6 * np.pi, 1000)  # phase in radians

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

reflectivities = [0.04, 0.30, 0.70, 0.90, 0.97]

for R in reflectivities:
    F = 4 * R / (1 - R)**2  # coefficient of finesse
    finesse = np.pi * np.sqrt(R) / (1 - R)
    # Airy function: transmitted intensity
    I_t = 1 / (1 + F * np.sin(delta / 2)**2)

    ax1.plot(delta / np.pi, I_t, linewidth=1.5 + R,
             label=f'R = {R:.2f} ($\\mathcal{{F}}$ = {finesse:.1f})')

ax1.set_xlabel('Phase $\\delta / \\pi$', fontsize=12)
ax1.set_ylabel('Transmitted Intensity $I_t / I_0$', fontsize=12)
ax1.set_title('Fabry-Perot Transmission (Airy Function)', fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.1)

# Right: resolving two closely spaced wavelengths with a Fabry-Perot
# Two wavelengths differ by delta_lambda, which maps to a phase shift
R = 0.90
F = 4 * R / (1 - R)**2
finesse = np.pi * np.sqrt(R) / (1 - R)

# Fine phase grid around a single order
phase_fine = np.linspace(1.8 * np.pi, 2.2 * np.pi, 1000)

# Two wavelengths separated by different amounts
separations = [0.02*np.pi, 0.05*np.pi, 0.10*np.pi]  # phase separations

for sep in separations:
    I1 = 1 / (1 + F * np.sin(phase_fine / 2)**2)
    I2 = 1 / (1 + F * np.sin((phase_fine - sep) / 2)**2)
    I_total = I1 + I2

    # Normalize for display
    ax2.plot(phase_fine / np.pi, I_total / I_total.max(), linewidth=2,
             label=f'$\\Delta\\delta = {sep/np.pi:.2f}\\pi$')

ax2.set_xlabel('Phase $\\delta / \\pi$', fontsize=12)
ax2.set_ylabel('Combined Intensity (normalized)', fontsize=12)
ax2.set_title(f'Resolving Two Wavelengths (R={R}, $\\mathcal{{F}}$={finesse:.0f})', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fabry_perot.png', dpi=150)
plt.show()
```

---

## 6. 가간섭성(Coherence)

### 6.1 가간섭성이 중요한 이유

실제 광원은 완벽하게 단색이거나 완벽한 점 광원이 아니다. 광원이 안정된 간섭 무늬를 만들 수 있는 정도를 **가간섭성(coherence)**으로 정량화한다.

### 6.2 시간적 가간섭성(Temporal Coherence)

**시간적 가간섭성**은 파동이 지연된 자신과 얼마나 잘 상관되는지를 측정한다. 이는 광원의 스펙트럼 폭 $\Delta\nu$와 관련된다:

$$\tau_c = \frac{1}{\Delta\nu} \qquad (\text{가간섭 시간, coherence time})$$

$$\ell_c = c\tau_c = \frac{c}{\Delta\nu} = \frac{\lambda^2}{\Delta\lambda} \qquad (\text{가간섭 길이, coherence length})$$

| 광원 | $\Delta\lambda$ | 가간섭 길이 |
|--------|-----------------|-----------------|
| 백색광 | $\sim 300$ nm | $\sim 1$ $\mu$m |
| LED | $\sim 30$ nm | $\sim 10$ $\mu$m |
| 나트륨 램프 (D선) | $\sim 0.02$ nm | $\sim 15$ mm |
| HeNe 레이저 | $\sim 0.001$ nm | $\sim 30$ cm |
| 안정화된 레이저 | $\sim 10^{-9}$ nm | $\sim 300$ km |

마이컬슨 간섭계에서 무늬는 경로차 $\Delta < \ell_c$일 때만 보인다. 이 때문에 백색광은 영 경로차 근처에서 몇 개의 색 무늬만 만들고, 레이저는 수백만 개의 무늬를 만든다.

### 6.3 공간적 가간섭성(Spatial Coherence)

**공간적 가간섭성**은 파동의 진행 방향에 수직인 서로 다른 두 공간 점에서 파동이 얼마나 잘 상관되는지를 측정한다. 이는 관측점에서 본 광원의 각크기 $\Delta\theta$에 의존한다:

$$d_c = \frac{\lambda}{\Delta\theta} \qquad (\text{가간섭 폭, coherence width})$$

영의 이중 슬릿 실험에서 선명한 무늬를 얻으려면 슬릿 간격이 가간섭 폭보다 작아야 한다: $d < d_c$.

태양($\Delta\theta \approx 0.5° = 0.0087$ rad)의 공간적 가간섭 폭은 약:

$$d_c = \frac{550 \text{ nm}}{0.0087} \approx 63 \text{ μm}$$

이 때문에 대부분의 장치에서 햇빛은 약한 간섭을 만든다 — 슬릿이 매우 가까워야 한다.

### 6.4 반 치테르트-체르니케 정리(Van Cittert-Zernike Theorem)

이 정리는 광원의 공간 세기 분포와 광원이 만드는 장의 가간섭성 사이의 정량적 관계를 제공한다. 공간 가간섭성 함수(복소 가간섭도)는 광원 세기 분포의 정규화된 푸리에 변환이다.

---

## 7. 간섭의 실용 응용

### 7.1 광학 검사(Optical Testing)

간섭계는 광학 표면 검사의 표준 도구이다. 트와이만-그린 간섭계(Twyman-Green interferometer, 마이컬슨의 변형)는 검사 표면에서 반사된 파면(wavefront)을 기준 평면과 비교한다. 무늬의 직선성에서의 편차는 나노미터 정밀도로 표면 결함을 드러낸다.

### 7.2 광 간섭 단층 촬영(OCT, Optical Coherence Tomography)

OCT는 저가간섭성(넓은 대역폭) 광원과 마이컬슨 간섭계를 이용하여 생체 조직의 단면 영상을 만든다. 짧은 가간섭 길이($\sim 10$ $\mu$m)가 깊이 게이트(depth gate) 역할을 한다: 조직의 특정 깊이에서 오는 빛만 기준 빔과 간섭한다. 기준 거울을 스캔하면 깊이 프로파일을 구축할 수 있다.

### 7.3 중력파 검출(LIGO)

LIGO의 간섭계는 4 km 팔에 파브리-페로 공동(유효 경로 길이 $\sim 1200$ km)을 갖는다. 지나가는 중력파는 한 팔을 늘이고 다른 팔을 압축하여 $\sim 10^{-18}$ m — 양성자 지름의 약 천 분의 일 — 의 차등 경로 변화를 만든다. 이것이 간섭 무늬의 변화로 검출된다.

---

## 연습 문제

### 연습 문제 1: 이중 슬릿 매개변수

HeNe 레이저 빛(632.8 nm)을 사용한 영의 이중 슬릿 실험에서 슬릿 간격은 0.15 mm, 스크린까지의 거리는 2.0 m이다.

(a) 스크린 위의 무늬 간격을 계산하라.

(b) 중앙에서 5번째 밝은 무늬의 각도 위치는?

(c) 한 슬릿을 중성 밀도 필터로 가려 세기를 다른 슬릿의 25%로 줄이면 무늬 가시도는 얼마인가?

### 연습 문제 2: 무반사 코팅 설계

고굴절률 유리 렌즈($n_g = 1.72$)에 $\lambda = 550$ nm에 최적화된 단층 무반사 코팅을 설계하라.

(a) 이상적인 코팅 굴절률은?

(b) 코팅 두께는?

(c) 이상 재료 대신 MgF$_2$ ($n = 1.38$)를 사용할 경우 550 nm에서의 잔류 반사율을 계산하라.

(d) 이 유리에 대해 MgF$_2$ 코팅이 반사율 영(0)을 달성하는 파장은?

### 연습 문제 3: 마이컬슨 간섭계

나트륨 램프($\lambda = 589.0$ nm와 $\lambda = 589.6$ nm — 나트륨 이중선)로 조명된 마이컬슨 간섭계는 거울 하나를 이동할 때 무늬가 주기적으로 사라졌다 나타났다를 반복한다.

(a) 무늬가 사라지는 이유를 설명하라. (힌트: 두 파장이 겹치는 무늬 패턴을 만든다.)

(b) 무늬가 처음 사라지는 경로차는?

(c) 나트륨 이중선의 가간섭 길이를 계산하라.

### 연습 문제 4: 파브리-페로 에탈론

파브리-페로 에탈론의 거울 반사율 $R = 0.95$, 간격 $d = 5$ mm이고, 녹색광($\lambda = 546$ nm)으로 사용한다.

(a) 피네스를 계산하라.

(b) 자유 스펙트럼 범위를 (주파수와 파장으로) 계산하라.

(c) 분해 가능한 최소 파장 차이는?

(d) 분해 능력은?

### 연습 문제 5: 가간섭성과 무늬 가시도

$\lambda_0 = 800$ nm에 중심을 두고 반치전폭(FWHM) 스펙트럼 폭 $\Delta\lambda = 40$ nm인 가우시안 스펙트럼 프로파일 광원으로 마이컬슨 간섭계를 조명한다.

(a) 가간섭 길이를 계산하라.

(b) 무늬 가시도가 최댓값의 $1/e$로 떨어지는 경로차는?

(c) 대비가 50% 이하로 떨어지기 전에 몇 개의 무늬가 보이는가?

---

## 요약

| 개념 | 핵심 공식 / 사실 |
|---------|-------------------|
| 중첩 원리 | $\mathbf{E}_{\text{total}} = \mathbf{E}_1 + \mathbf{E}_2$; 세기에 교차항 포함 |
| 두 빔 간섭 | $I = I_1 + I_2 + 2\sqrt{I_1 I_2}\cos\delta$ |
| 무늬 가시도 | $V = (I_{\max} - I_{\min})/(I_{\max} + I_{\min})$ |
| 이중 슬릿 밝은 무늬 | $d\sin\theta = m\lambda$; 무늬 간격 $\Delta y = \lambda L/d$ |
| 박막 반사 위상 | 밀한 매질에서 $\pi$ 변화; 소한 매질에서 변화 없음 |
| 무반사 코팅 | 1/4 파장 두께: $t = \lambda/(4n_c)$; 이상적 $n_c = \sqrt{n_g}$ |
| 뉴턴의 고리 (어두운) | $r_m = \sqrt{m\lambda R}$ |
| 마이컬슨 간섭계 | 광 경로차 = $2(d_1 - d_2)$; 거울 $\lambda/2$ 이동 시 무늬 한 간격 이동 |
| 파브리-페로 (에어리 함수) | $I_t = I_0/[1 + F\sin^2(\delta/2)]$; $\mathcal{F} = \pi\sqrt{R}/(1-R)$ |
| 자유 스펙트럼 범위 | $\Delta\nu_{\text{FSR}} = c/(2nd)$ |
| 가간섭 길이 | $\ell_c = c/\Delta\nu = \lambda^2/\Delta\lambda$ |
| 공간적 가간섭 폭 | $d_c = \lambda/\Delta\theta$ |

---

[← 이전: 04. 광학 기기](04_Optical_Instruments.md) | [다음: 06. 회절 →](06_Diffraction.md)
