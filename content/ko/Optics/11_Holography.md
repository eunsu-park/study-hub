# 11. 홀로그래피

[← 이전: 10. 푸리에 광학](10_Fourier_Optics.md) | [다음: 12. 비선형 광학 →](12_Nonlinear_Optics.md)

---

사진은 빛의 세기(intensity)를 기록합니다 — 장면의 각 점이 얼마나 밝은지를. 그러나 깊이와 3차원 구조를 인코딩하는 광파의 타이밍, 즉 위상(phase)은 버려집니다. 1948년 Dennis Gabor가 발명한 홀로그래피(Holography)(1971년 노벨상)는 진폭과 위상을 **모두** 기록하여, 완전한 3차원 광 필드를 충실하게 재현하는 기술입니다.

"홀로그래피(holography)"라는 단어는 그리스어 *holos*(전체)와 *graphe*(쓰기)에서 유래했습니다 — 온전한 기록, 즉 완전한 기록을 뜻합니다. 홀로그램(hologram)은 단순히 3D 사진을 표시하는 것이 아니라, 원래의 광파를 너무나 완벽하게 재현하여 홀로그램을 바라보는 것과 실제 물체를 바라보는 것의 차이를 눈이 구별하지 못합니다. 시점을 이동하면 물체의 모퉁이 너머를 볼 수 있습니다. 홀로그램의 서로 다른 부분이 서로 다른 시점을 보여줍니다. 홀로그램을 반으로 잘라도 각 절반에 전체 장면이 여전히 담겨 있습니다(해상도는 낮아지지만).

이 레슨에서는 홀로그래픽 기록 및 재현의 물리학, 주요 홀로그램 유형, 그리고 현대 디지털 홀로그래피 기술을 다룹니다.

**난이도**: ⭐⭐⭐

## 학습 목표

1. 홀로그램이 참조 빔과의 간섭을 통해 진폭과 위상 정보를 모두 인코딩하는 방법을 설명한다
2. 재현 과정을 유도하고, 실상(real image), 허상(virtual image), 켤레 빔(conjugate beam) 항을 식별한다
3. Gabor 인라인(inline) 홀로그래피와 Leith-Upatnieks 오프축(off-axis) 홀로그래피 구성을 비교하고 각각의 장점을 설명한다
4. 얇은 홀로그램과 두꺼운(부피형, volume) 홀로그램을 구분하고, 부피형 홀로그램에서의 Bragg 선택성을 설명한다
5. 디지털 홀로그래피의 원리를 설명한다: CCD/CMOS 센서에 기록하고 수치적으로 재현하기
6. 3D 디스플레이, 데이터 저장, 간섭계측, 보안 분야에서 홀로그래피의 주요 응용을 식별한다
7. Python을 사용하여 수치적 홀로그래픽 기록 및 재현을 수행한다

---

## 목차

1. [홀로그래픽 원리](#1-홀로그래픽-원리)
2. [홀로그램 기록](#2-홀로그램-기록)
3. [파면 재현](#3-파면-재현)
4. [Gabor 인라인 홀로그래피](#4-gabor-인라인-홀로그래피)
5. [Leith-Upatnieks 오프축 홀로그래피](#5-leith-upatnieks-오프축-홀로그래피)
6. [얇은 홀로그램 대 두꺼운 홀로그램](#6-얇은-홀로그램-대-두꺼운-홀로그램)
7. [부피형 홀로그램과 Bragg 회절](#7-부피형-홀로그램과-bragg-회절)
8. [디지털 홀로그래피](#8-디지털-홀로그래피)
9. [응용](#9-응용)
10. [Python 예제](#10-python-예제)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)
13. [참고 문헌](#13-참고-문헌)

---

## 1. 홀로그래픽 원리

### 1.1 위상 문제

광학 검출기(필름, CCD, 눈)는 **세기(intensity)** $I = |U|^2$를 측정합니다. 광 필드의 위상은 소실됩니다. 만약 $U = |U|e^{i\phi}$를 직접 기록할 수 있다면, 완전한 3D 파면을 재현할 수 있습니다. 그러나 위상을 직접 측정할 수는 없습니다.

### 1.2 Gabor의 통찰

Gabor는 위상 정보가 **간섭(interference)**을 통해 세기 변화로 인코딩될 수 있다는 것을 깨달았습니다. 알려진 **참조파(reference wave)** $U_R$을 미지의 **물체파(object wave)** $U_O$에 더하면, 총 세기 패턴에 위상이 담기게 됩니다:

$$I = |U_R + U_O|^2 = |U_R|^2 + |U_O|^2 + U_R^*U_O + U_R U_O^*$$

교차항(cross term) $U_R^*U_O$와 $U_R U_O^*$는 참조파와 물체파 사이의 상대 위상을 인코딩합니다. 이 간섭 패턴 — **홀로그램** — 이 감광 매체에 기록됩니다.

> **비유**: 복잡한 멜로디(물체파)를 녹음하고 싶은데, 녹음기가 음량만 잡아낼 수 있고 음정(위상)은 담지 못한다고 상상해 보세요. Gabor의 트릭은 동시에 단순하고 알려진 음(참조파)을 연주하는 것입니다. 멜로디와 음 사이의 맥놀이(beat)와 간섭 패턴이 음정 정보를 음량 변화로 인코딩합니다. 원래 멜로디를 다시 듣고 싶을 때는 동일한 참조음을 동시에 연주하면서 녹음을 재생합니다 — 간섭이 사라진 음정 정보를 되살려 줍니다.

### 1.3 홀로그래피의 요건

1. **가간섭성(coherent) 광원**: 두 빔은 상호 가간섭성이 있어야 합니다. 가간섭 길이(coherence length)는 최대 경로 차이를 초과해야 합니다. 이것이 레이저(1960년 발명) 이후에야 홀로그래피가 실용화된 이유입니다.

2. **안정적인 기록 기하학**: 노출 중 파장 이하 수준의 안정성 ($< \lambda/4 \approx 150\,\text{nm}$). 진동이 있으면 간섭 줄무늬가 흐려집니다.

3. **고해상도 기록 매체**: 줄무늬 간격은 $\lambda/(2\sin\theta) \approx 0.3\,\mu\text{m}$까지 미세해질 수 있으므로, 3000 줄/mm 이상의 해상도가 필요합니다(~200 줄/mm에 불과한 일반 사진 훨씬 이상).

---

## 2. 홀로그램 기록

### 2.1 설치

레이저 빔을 두 경로로 분리합니다:
- **참조 빔(reference beam)** $U_R$: 기록 매체로 직접 전파
- **물체 빔(object beam)** $U_O$: 물체를 조명하고, 산란/반사된 빛이 기록 매체에 도달

```
                    빔
    레이저 ─────────분할기──────────→ 참조 빔 (U_R)
                      │                     │
                      ↓                     │
                   물체                    │
                      │                     │
                   산란된                  │
                   빛 (U_O)               │
                      │                     │
                      ↓                     ↓
                    ┌─────────────────────────┐
                    │       기록 매체          │
                    │  (필름 / 광중합체)       │
                    └─────────────────────────┘
```

### 2.2 간섭 패턴

기록 평면에서 세기는:

$$I(x, y) = |U_R + U_O|^2 = I_R + I_O + U_R^*U_O + U_R U_O^*$$

여기서:
- $I_R = |U_R|^2$: 참조 빔 세기 (평면파의 경우 균일)
- $I_O = |U_O|^2$: 물체 빔 세기 (진폭 정보만 포함)
- $U_R^*U_O$: 홀로그래픽 항 (참조에 대한 물체 위상 인코딩)
- $U_R U_O^*$: 켤레 홀로그래픽 항

### 2.3 기록 매체 응답

기록 매체(사진 필름, 광중합체, 광굴절 결정)는 세기 패턴에 반응합니다. 선형 기록 매체의 경우, 현상 후 **진폭 투과율(amplitude transmittance)**은 노출에 비례합니다:

$$t(x, y) = t_0 + \beta I(x, y)$$

여기서 $t_0$는 바이어스 투과율이고, $\beta$는 감도입니다(필름의 경우 음수). 투과율은:

$$t = t_0 + \beta(I_R + I_O) + \beta U_R^*U_O + \beta U_R U_O^*$$

세 번째와 네 번째 항이 홀로그래픽 내용입니다.

---

## 3. 파면 재현

### 3.1 참조 빔으로 조명

재현하려면, 홀로그램을 원래 참조 빔 $U_R$으로 조명합니다. 투과 필드는:

$$U_{\text{trans}} = t \cdot U_R = (t_0 + \beta I_R + \beta I_O)U_R + \beta|U_R|^2 U_O + \beta U_R^2 U_O^*$$

세 항은:

1. **0차 항** $(t_0 + \beta I_R + \beta I_O)U_R$: 참조 빔의 수정된 버전 (DC 항, 감쇠 및 변조됨). 곧장 투과합니다.

2. **허상(virtual image)** $\beta|U_R|^2 U_O$: $U_O$에 비례합니다 (평면파 참조에 대해 상수 인수 $\beta I_R$). 원래 물체파의 정확한 복사본 — 홀로그램을 통해 보는 관찰자는 홀로그램 **뒤에** 있는 원래 3D 위치의 물체를 봅니다.

3. **켤레(실상, real image)** $\beta U_R^2 U_O^*$: 물체파의 복소 켤레에 비례합니다. 홀로그램 **앞에** 실상이 수렴합니다(의사 스코픽(pseudoscopic) — 깊이가 뒤집힘).

### 3.2 핵심 결과

두 번째 항은 모든 진폭과 위상 정보를 포함하여 원래 물체 파면을 충실하게 재현합니다. 홀로그램을 통해 보는 관찰자는 원래 장면과 구별할 수 없는 3차원 상을 봅니다.

### 3.3 작동 원리

마법은 참조 빔이 신호 처리 용어로는 **국소 발진기(local oscillator)**, 또는 **헤테로다인 검출기(heterodyne detector)**처럼 작동한다는 것입니다. 교차항 $U_R^*U_O$는 $U_O$의 위상을 세기 변조로 기록합니다. $U_R$으로 재조명하면 과정이 역전됩니다: $U_R \cdot U_R^*U_O = |U_R|^2 U_O \propto U_O$.

---

## 4. Gabor 인라인 홀로그래피

### 4.1 구성

Gabor의 원래 방식(1948)에서는 참조 빔과 물체 빔이 같은 축을 따라 진행합니다. 물체는 부분 투명하며, 조명 빔의 산란되지 않은 부분이 참조 빔 역할을 합니다:

```
   레이저 ────→ 물체 (부분 투명) ────→ 기록 매체
              │                        │
              └── 산란됨 (U_O) ────────→│
              └── 비산란됨 (U_R) ───────→│
```

### 4.2 장점

- 별도의 참조 빔 경로 없이 간단한 설치
- 하나의 가간섭성 빔만 필요
- 소형 기하학

### 4.3 단점

세 가지 재현 항(0차, 허상, 켤레상)이 **같은 축을 따라** 전파됩니다. 공간적으로 겹쳐:
- DC 항이 원치 않는 빛으로 상을 가득 채움
- 켤레상이 허상 위에 겹쳐지는 초점이 벗어난 "쌍둥이 상(twin image)"을 생성

이 쌍둥이 상 문제가 Gabor의 원래 홀로그래피를 심각하게 제한했습니다. 전자 홀로그래피(Gabor의 원래 동기는 전자 현미경 해상도 향상이었음)와 같은 특수 응용에만 실용적이었습니다.

---

## 5. Leith-Upatnieks 오프축 홀로그래피

### 5.1 오프축 해법 (1962)

Emmett Leith와 Juris Upatnieks는 **오프축 참조 빔(off-axis reference beam)**을 도입했습니다 — 참조파가 물체 빔에 각도 $\theta$로 입사합니다. 이 단순하지만 심오한 변형이 쌍둥이 상 문제를 해결했습니다.

각도 $\theta$의 평면파 참조에 대해:

$$U_R = A_R\,e^{ikx\sin\theta}$$

간섭 패턴은 이제 공간 반송파 주파수(spatial carrier frequency) $f_c = \sin\theta/\lambda$의 줄무늬를 포함합니다.

### 5.2 항의 공간적 분리

재현 시, 세 항은 서로 다른 방향으로 전파됩니다:

- **0차 항**: 참조 빔 방향으로 계속 진행
- **허상 (+1차)**: 원래 물체 빔 방향으로 회절
- **켤레상 (-1차)**: 반대쪽으로 회절

```
                     허상 (U_O)
                   ╱
                 ╱ 각도 θ
   참조 빔 ──╳────────→ 0차 항 (DC)
              ╲
                 ╲ 각도 -θ
                   ╲
                     켤레상 (U_O*)
```

각도 $\theta$가 충분히 크면 세 항이 **공간적으로 분리**됩니다:

$$\sin\theta > \frac{3}{2}\frac{\lambda}{d_{\min}}$$

여기서 $d_{\min}$은 물체의 가장 작은 특성 크기입니다. 이 조건은 세 회절 빔이 출력 평면에서 겹치지 않음을 보장합니다.

### 5.3 실용적 의의

Leith-Upatnieks 오프축 기하학은 고품질 홀로그래피를 실용화했습니다. 레이저(1960년 발명)와 결합하여, 모든 현대 홀로그래픽 응용의 문을 열었습니다. 실제 물체의 3D 홀로그램을 제작한 그들의 1964년 논문은 광학의 이정표입니다.

> **비유**: Gabor 홀로그래피와 Leith-Upatnieks 홀로그래피의 차이는 AM 라디오와 FM 라디오의 차이와 같습니다. AM(Gabor)에서는 신호가 반송파에 직접 중첩되어 있어, 잡음이나 겹치는 방송국이 간섭을 일으킵니다. FM(Leith-Upatnieks)에서는 신호가 특정 주파수 오프셋의 반송파에 인코딩되어, 조정된 필터로 다른 신호와 깔끔하게 분리할 수 있습니다. 오프축 각도는 유용한 신호를 잡음에서 분리하는 "반송파 주파수"입니다.

---

## 6. 얇은 홀로그램 대 두꺼운 홀로그램

### 6.1 Q 매개변수

얇은 홀로그램과 두꺼운 홀로그램의 구분은 **Klein-Cook Q 매개변수**에 의해 결정됩니다:

$$Q = \frac{2\pi\lambda d}{n\Lambda^2}$$

여기서 $d$는 홀로그램 두께, $\Lambda$는 줄무늬 간격, $n$은 굴절률입니다.

- **얇은 홀로그램** ($Q < 1$): 2D 격자처럼 작동. 여러 회절 차수 생성. Raman-Nath 영역.
- **두꺼운(부피형) 홀로그램** ($Q > 10$): 3D 격자처럼 작동. 하나의 회절 차수만 효율적으로 생성 (Bragg 회절). Bragg 영역.

### 6.2 얇은 홀로그램

- 얇은 사진 필름에 기록 (~5-15 $\mu$m)
- 각도 $\sin\theta_m = m\lambda/\Lambda$에서 여러 회절 차수
- 낮은 회절 효율 (진폭 홀로그램의 경우 일반적으로 < 6%)
- 진폭형 또는 위상형

### 6.3 위상 홀로그램 대 진폭 홀로그램

**진폭 홀로그램(amplitude hologram)**: 매체의 흡수를 변조. 최대 회절 효율: $\eta_{\max} = 6.25\%$ (얇은 경우) — 대부분의 빛이 0차 항으로 가거나 흡수됩니다.

**위상 홀로그램(phase hologram)**: 흡수 없이 굴절률 또는 두께를 변조. 최대 회절 효율: $\eta_{\max} = 33.9\%$ (얇은, Raman-Nath). 위상 홀로그램이 더 효율적이고 밝기 때문에 선호됩니다.

---

## 7. 부피형 홀로그램과 Bragg 회절

### 7.1 Bragg 조건

두꺼운 홀로그램에서 간섭 줄무늬는 3D 격자 — 기록 매체 부피 전체에 걸친 주기적 구조 — 를 형성합니다. **Bragg 조건(Bragg condition)**이 충족될 때만 효율적인 회절이 발생합니다:

$$\boxed{2n\Lambda\sin\theta_B = \lambda}$$

여기서 $\Lambda$는 줄무늬 간격, $\theta_B$는 Bragg 각도(줄무늬 평면에서 측정), $n$은 굴절률입니다.

### 7.2 각도 및 파장 선택성

Bragg 조건은 엄격한 선택성을 부여합니다:

**각도 선택성(angular selectivity)**: 홀로그램은 좁은 재현 각도 범위 내에서만 효율적으로 회절합니다:

$$\Delta\theta \approx \frac{\Lambda}{d}$$

1 mm 두꺼운 홀로그램에서 $\Lambda = 0.5\,\mu\text{m}$이면: $\Delta\theta \approx 0.5\,\text{mrad} \approx 0.03°$.

**파장 선택성(wavelength selectivity)**: 좁은 파장 대역만 회절됩니다:

$$\Delta\lambda \approx \frac{\lambda\Lambda}{d}$$

이 선택성이 부피형 홀로그램을 고밀도 데이터 저장과 WDM(파장분할다중화) 필터에 유용하게 만듭니다.

### 7.3 회절 효율

두꺼운 위상 홀로그램의 경우 (Kogelnik의 결합파 이론):

$$\eta = \sin^2\!\left(\frac{\pi\Delta n\, d}{\lambda\cos\theta_B}\right)$$

여기서 $\Delta n$은 굴절률 변조량입니다. $\frac{\pi\Delta n\, d}{\lambda\cos\theta_B} = \frac{\pi}{2}$일 때, 효율은 **100%**에 도달합니다 — 모든 입사광이 단일 차수로 회절됩니다. 이는 얇은 홀로그램보다 극적으로 더 좋습니다.

### 7.4 반사형 대 투과형 부피 홀로그램

**투과형 홀로그램(transmission hologram)**: 참조 빔과 물체 빔이 같은 면에서 입사합니다. 줄무늬는 표면에 대략 수직입니다. 뒤에서 레이저광으로 재현됩니다.

**반사형 홀로그램(reflection hologram)** (Denisyuk 홀로그램): 참조 빔과 물체 빔이 반대편에서 입사합니다. 줄무늬는 표면에 대략 평행합니다. 백색광으로도 관찰 가능합니다 — Bragg 파장 선택성이 컬러 필터 역할을 하여 광대역 스펙트럼에서 올바른 파장을 선택합니다. 박물관 전시 홀로그램과 보안 홀로그램이 이런 방식으로 동작합니다.

---

## 8. 디지털 홀로그래피

### 8.1 필름에서 센서로

디지털 홀로그래피(digital holography)에서는 사진 필름을 CCD 또는 CMOS 센서로 대체합니다. 홀로그램(간섭 패턴)은 디지털 이미지로 기록되고, 재현은 컴퓨터에서 수치적으로 수행됩니다.

### 8.2 기록

설치는 고전적 홀로그래피와 유사하지만, 기록 매체가 디지털 센서입니다:

```
   레이저 ────→ 빔분할기 ────→ 참조 빔
                │
                ↓
             물체 ────→ CCD/CMOS 센서
                          (I(x,y) 기록)
```

**샘플링 요건**: 센서 픽셀 피치 $\Delta p$는 최고 줄무늬 주파수에 대해 Nyquist 조건을 만족해야 합니다:

$$\Delta p < \frac{\lambda}{2\sin\theta}$$

$\lambda = 633\,\text{nm}$, $\theta = 5°$이면: $\Delta p < 3.6\,\mu\text{m}$. 1-5 $\mu\text{m}$ 픽셀의 현대 센서는 적당한 오프축 각도를 처리할 수 있습니다.

### 8.3 수치 재현

기록된 홀로그램 $I(x, y)$에 수치 참조파 $U_R(x, y)$를 곱한 후, Fresnel 전파 적분(FFT로 계산)을 사용하여 재현 평면으로 전파합니다:

$$U_{\text{recon}}(x', y') = \mathcal{F}^{-1}\left\{\mathcal{F}\left\{I(x,y) \cdot U_R(x,y)\right\} \cdot H(f_x, f_y; d)\right\}$$

여기서 $H$는 Fresnel 전달 함수이고, $d$는 재현 거리입니다.

### 8.4 디지털 홀로그래피의 장점

1. **정량적 위상 이미징(quantitative phase imaging)**: 수치적으로 재현된 복소 필드는 진폭과 위상을 **모두** 제공합니다 — 표면 프로파일, 굴절률 분포 등의 정밀 측정이 가능합니다.

2. **수치 초점 재조정(numerical refocusing)**: 단 하나의 기록된 홀로그램을 임의의 거리 $d$로 수치적으로 전파할 수 있어, 기계적 조정 없이 후처리 초점 맞추기가 가능합니다.

3. **수차 보정(aberration correction)**: 광학 수차를 후처리에서 수치적으로 보상할 수 있습니다.

4. **화학적 처리 불필요**: 암실 없이 즉시 결과를 얻습니다.

5. **시간 분해(time-resolved)**: 고속 센서가 동적 홀로그래피를 가능하게 합니다 (살아있는 세포의 디지털 홀로그래픽 현미경, 진동 분석).

---

## 9. 응용

### 9.1 3D 디스플레이

- **예술 및 박물관 전시**: 백색광으로 볼 수 있는 대형 반사 홀로그램
- **헤드업 디스플레이(HUD)**: 자동차 및 항공 분야의 홀로그래픽 광학 소자
- **홀로그래픽 비디오**: MIT 미디어 랩, Looking Glass Factory 등 실시간 동적 홀로그램을 생성하는 디스플레이

### 9.2 홀로그래픽 데이터 저장

부피형 홀로그램은 2D 표면만이 아니라 매체 두께 전체에 데이터를 저장할 수 있습니다. 각 데이터 페이지는 다른 각도(각도 다중화) 또는 파장으로 홀로그램으로 저장됩니다. 이론적 용량: 디스크당 ~1 TB (블루레이의 25-100 GB와 비교).

### 9.3 홀로그래픽 간섭계측

서로 다른 시점에 기록된 두 홀로그램(또는 이중 노출)이 간섭하여 물체의 미세한 변화를 드러냅니다 — 변형, 진동, 또는 굴절률 변화. 민감도: 파장 이하 ($< \lambda/10 \approx 50\,\text{nm}$).

응용: 항공기 부품 비파괴 검사, 타이어 검사, 악기 및 스피커의 진동 모드 분석.

### 9.4 보안 홀로그램

신용카드, 지폐, 여권의 무지갯빛 홀로그램은 엠보싱 표면 부조 홀로그램입니다. 위조가 극히 어렵습니다:
- 가간섭성 기록 장비가 필요
- 각 홀로그램은 복잡한 3D 미세 구조
- 대량 생산에는 비싼 전기 성형 마스터가 필요

### 9.5 홀로그래픽 광학 소자(HOE)

렌즈, 거울, 격자, 빔 분할기 등의 광학 소자 역할을 하는 홀로그램. 장점: 경량, 평면형, 하나의 소자에 여러 기능 결합 가능. 헤드 마운티드 디스플레이, 바코드 스캐너, 태양광 집광기에 사용됩니다.

### 9.6 디지털 홀로그래픽 현미경(DHM)

염색 없이 생물학적 세포의 정량적 위상 이미징. 세포 두께, 건조 질량, 굴절률 분포 측정. 세포 역학, 성장, 자극에 대한 반응의 실시간 모니터링.

---

## 10. Python 예제

### 10.1 디지털 홀로그래픽 기록 및 재현

```python
import numpy as np
import matplotlib.pyplot as plt

def create_object_wave(N, pixel_size, wavelength, object_distance):
    """
    Create a simple object wave: two point sources at different depths.

    We model each point source as a spherical wave emanating from
    a specific (x, y, z) position. The interference of multiple
    point sources at different depths creates a complex wavefront
    that encodes genuine 3D information — exactly what holography
    is designed to capture.
    """
    x = np.arange(-N//2, N//2) * pixel_size
    X, Y = np.meshgrid(x, x)
    k = 2 * np.pi / wavelength

    # Point source 1 — centered, distance d1
    d1 = object_distance
    r1 = np.sqrt(X**2 + Y**2 + d1**2)
    U1 = np.exp(1j * k * r1) / r1

    # Point source 2 — offset, distance d2 (different depth)
    d2 = object_distance * 1.2
    x_off, y_off = 50 * pixel_size, 30 * pixel_size
    r2 = np.sqrt((X - x_off)**2 + (Y - y_off)**2 + d2**2)
    U2 = 0.7 * np.exp(1j * k * r2) / r2  # Slightly dimmer

    return U1 + U2

def create_reference_wave(N, pixel_size, wavelength, angle_deg):
    """
    Create an off-axis plane wave reference beam.

    The reference angle must be large enough to spatially separate
    the three diffraction orders (DC, virtual image, conjugate image)
    upon reconstruction — this is the Leith-Upatnieks geometry.
    """
    x = np.arange(-N//2, N//2) * pixel_size
    X, Y = np.meshgrid(x, x)
    k = 2 * np.pi / wavelength
    angle = np.radians(angle_deg)

    # Plane wave tilted in x-direction
    U_R = np.exp(1j * k * X * np.sin(angle))
    return U_R

def fresnel_propagate(field, wavelength, pixel_size, distance):
    """
    Propagate a complex field using the angular spectrum method.

    This is the core numerical engine of digital holography.
    We decompose the field into plane waves (FFT), propagate
    each one by multiplying with the free-space transfer function,
    then recombine (inverse FFT). Exact within the scalar approximation.
    """
    N = field.shape[0]
    k = 2 * np.pi / wavelength

    # Spatial frequency grid
    df = 1.0 / (N * pixel_size)
    fx = np.arange(-N//2, N//2) * df
    FX, FY = np.meshgrid(fx, fx)

    # Transfer function of free space (angular spectrum)
    # Evanescent waves (fr > 1/lambda) are automatically handled
    # by the square root becoming imaginary → exponential decay
    arg = (1.0 / wavelength)**2 - FX**2 - FY**2
    H = np.exp(1j * 2 * np.pi * distance * np.sqrt(np.maximum(arg, 0).astype(complex)))
    H[arg < 0] = 0  # Kill evanescent waves explicitly

    # Propagate: FFT → multiply by H → inverse FFT
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    propagated = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum * H)))

    return propagated

# --- Parameters ---
N = 1024          # Grid size (pixels)
pixel_size = 5e-6 # 5 µm pixel pitch (typical CCD)
wavelength = 633e-9  # He-Ne laser, 633 nm
object_distance = 0.05  # 5 cm from object to hologram plane
ref_angle = 1.5   # Reference beam angle (degrees)

# --- Recording ---
U_obj = create_object_wave(N, pixel_size, wavelength, object_distance)
U_ref = create_reference_wave(N, pixel_size, wavelength, ref_angle)

# Hologram = interference pattern (intensity at the sensor)
hologram = np.abs(U_ref + U_obj)**2

# --- Reconstruction ---
# Multiply hologram by (numerical) reference wave
U_recon_field = hologram * U_ref

# Propagate back to object plane
U_reconstructed = fresnel_propagate(U_recon_field, wavelength, pixel_size,
                                     -object_distance)

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Original object intensity
axes[0, 0].imshow(np.abs(U_obj)**2, cmap='hot')
axes[0, 0].set_title('Object Wave Intensity', fontsize=12)
axes[0, 0].axis('off')

# Recorded hologram
axes[0, 1].imshow(hologram, cmap='gray')
axes[0, 1].set_title('Recorded Hologram (Interference Pattern)', fontsize=12)
axes[0, 1].axis('off')

# Zoomed view of hologram fringes
axes[1, 0].imshow(hologram[N//2-50:N//2+50, N//2-50:N//2+50], cmap='gray')
axes[1, 0].set_title('Hologram Fringes (Zoomed)', fontsize=12)
axes[1, 0].axis('off')

# Reconstructed image
I_recon = np.abs(U_reconstructed)**2
axes[1, 1].imshow(I_recon, cmap='hot')
axes[1, 1].set_title('Reconstructed Image (Amplitude²)', fontsize=12)
axes[1, 1].axis('off')

plt.suptitle('Digital Holography: Recording and Reconstruction', fontsize=14)
plt.tight_layout()
plt.savefig('digital_holography.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Numerical refocusing: propagate to different distances ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
distances = [0.8, 0.9, 1.0, 1.1]  # Relative to object_distance

for i, rel_d in enumerate(distances):
    d = -object_distance * rel_d
    U_refocus = fresnel_propagate(U_recon_field, wavelength, pixel_size, d)
    axes[i].imshow(np.abs(U_refocus)**2, cmap='hot')
    axes[i].set_title(f'd = {rel_d:.1f} × d_obj', fontsize=10)
    axes[i].axis('off')

plt.suptitle('Numerical Refocusing from a Single Hologram', fontsize=13)
plt.tight_layout()
plt.savefig('numerical_refocusing.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 10.2 홀로그래픽 간섭 줄무늬 분석

```python
import numpy as np
import matplotlib.pyplot as plt

def hologram_fringe_spacing(wavelength, angle_deg):
    """
    Calculate the fringe spacing for an off-axis hologram.

    The fringe period Lambda = lambda / sin(theta) determines
    the spatial frequency of the carrier. The sensor pixel pitch
    must be smaller than Lambda/2 to satisfy Nyquist sampling.
    """
    angle_rad = np.radians(angle_deg)
    if np.sin(angle_rad) == 0:
        return np.inf
    return wavelength / np.sin(angle_rad)

# Analyze fringe spacing vs. reference angle
angles = np.linspace(0.5, 30, 100)
wavelengths = [633e-9, 532e-9, 405e-9]  # Red, Green, Blue
colors = ['red', 'green', 'blue']
labels = ['633 nm (He-Ne)', '532 nm (Nd:YAG 2ω)', '405 nm (diode)']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for wl, color, label in zip(wavelengths, colors, labels):
    spacings = [hologram_fringe_spacing(wl, a) * 1e6 for a in angles]  # µm
    ax1.semilogy(angles, spacings, color=color, linewidth=2, label=label)

# Mark typical sensor pixel pitches
for pitch, name in [(1.67, '1.67 µm'), (3.45, '3.45 µm'), (5.0, '5.0 µm')]:
    ax1.axhline(y=2*pitch, color='gray', linestyle='--', alpha=0.5)
    ax1.text(25, 2*pitch*1.1, f'Nyquist for {name} pixel',
             fontsize=8, color='gray')

ax1.set_xlabel('Reference beam angle (degrees)', fontsize=11)
ax1.set_ylabel('Fringe spacing (µm)', fontsize=11)
ax1.set_title('Hologram Fringe Spacing vs. Reference Angle', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(1, 100)

# Maximum recordable angle for different pixel sizes
pixel_pitches = np.linspace(1, 10, 100)  # µm
for wl, color, label in zip(wavelengths, colors, labels):
    # Nyquist: pixel_pitch < lambda / (2*sin(theta))
    # => sin(theta_max) = lambda / (2*pixel_pitch)
    sin_theta_max = (wl * 1e6) / (2 * pixel_pitches)
    theta_max = np.degrees(np.arcsin(np.clip(sin_theta_max, 0, 1)))
    ax2.plot(pixel_pitches, theta_max, color=color, linewidth=2, label=label)

ax2.set_xlabel('Sensor pixel pitch (µm)', fontsize=11)
ax2.set_ylabel('Maximum reference angle (degrees)', fontsize=11)
ax2.set_title('Maximum Off-Axis Angle vs. Pixel Pitch', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hologram_fringe_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 10.3 부피형 홀로그램 Bragg 선택성

```python
import numpy as np
import matplotlib.pyplot as plt

def bragg_efficiency(delta_theta, thickness, wavelength, n, fringe_spacing):
    """
    Approximate angular selectivity of a volume hologram.

    Kogelnik's coupled wave theory predicts that the diffraction
    efficiency drops as the reconstruction angle deviates from the
    Bragg angle. The sinc-like profile has a width inversely
    proportional to the hologram thickness — thicker holograms
    are more selective, enabling denser multiplexing.
    """
    # Bragg angle
    theta_B = np.arcsin(wavelength / (2 * n * fringe_spacing))

    # Detuning parameter (simplified)
    xi = np.pi * n * thickness * delta_theta / wavelength

    # Efficiency (sinc² envelope, simplified model)
    eta = np.sinc(xi / np.pi)**2
    return eta

# Parameters
wavelength = 532e-9  # Green laser
n = 1.5              # Typical photopolymer
fringe_spacing = 0.5e-6  # 0.5 µm

# Angular selectivity for different thicknesses
delta_theta = np.linspace(-5e-3, 5e-3, 1000)  # radians

fig, ax = plt.subplots(figsize=(10, 6))

for thickness_mm in [0.01, 0.1, 1.0]:
    thickness = thickness_mm * 1e-3  # Convert to meters
    eta = bragg_efficiency(delta_theta, thickness, wavelength, n, fringe_spacing)
    ax.plot(np.degrees(delta_theta) * 1000, eta, linewidth=2,
            label=f'd = {thickness_mm} mm')

ax.set_xlabel('Angular detuning from Bragg angle (millidegrees)', fontsize=11)
ax.set_ylabel('Relative diffraction efficiency', fontsize=11)
ax.set_title('Volume Hologram Angular Selectivity', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-300, 300)

plt.tight_layout()
plt.savefig('bragg_selectivity.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. 요약

| 개념 | 핵심 공식 / 아이디어 |
|------|----------------------|
| 홀로그래픽 기록 | $I = \|U_R + U_O\|^2 = I_R + I_O + U_R^*U_O + U_RU_O^*$ |
| 재현 | $U_R$으로 조명: $\|U_R\|^2 U_O$ (허상) 획득 |
| Gabor 인라인 | 간단한 설치; 쌍둥이 상 문제 (모든 차수 겹침) |
| Leith-Upatnieks 오프축 | 오프축 참조; 세 차수를 공간적으로 분리 |
| 얇은 홀로그램 | $Q < 1$; 여러 회절 차수; $\eta_{\max} = 33.9\%$ (위상형) |
| 부피형 홀로그램 | $Q > 10$; 단일 Bragg 차수; $\eta$ 최대 100% |
| Bragg 조건 | $2n\Lambda\sin\theta_B = \lambda$ |
| 각도 선택성 | $\Delta\theta \approx \Lambda/d$ (두께 증가에 따라 좁아짐) |
| 디지털 홀로그래피 | CCD에 기록 → FFT를 통한 수치 재현 |
| 수치 초점 재조정 | 단일 홀로그램 → 임의 깊이 $d$에서 재현 |
| 위상 이미징 | 재현된 복소 필드가 정량적 위상 제공 |

---

## 12. 연습 문제

### 연습 문제 1: 홀로그램 해상도 요건

$\lambda = 532\,\text{nm}$, 참조 빔 각도 $\theta = 10°$로 오프축 홀로그램을 기록하려 합니다.

(a) 줄무늬 간격을 계산하세요.
(b) 기록 매체에 필요한 최소 해상도(줄/mm)는 얼마입니까?
(c) CCD 센서의 픽셀 피치가 3.45 $\mu$m입니다. 이 홀로그램을 기록할 수 있습니까?
(d) 532 nm에서 이 센서가 처리할 수 있는 최대 오프축 각도는 얼마입니까?

### 연습 문제 2: 부피형 홀로그램 설계

$\lambda = 633\,\text{nm}$ ($n = 1.5$)에서 표시용 반사 홀로그램을 설계하려 합니다.

(a) 수직 입사($\theta_B = 90°$, 표면과 평행한 줄무늬)에서의 줄무늬 간격을 계산하세요.
(b) 20 $\mu$m 두꺼운 유제(emulsion)의 경우, 파장 선택성 $\Delta\lambda$를 계산하세요.
(c) 이 홀로그램이 백색광에서 잘 재현됩니까? 설명하세요.
(d) 100% 회절 효율에 필요한 굴절률 변조량 $\Delta n$은 얼마입니까?

### 연습 문제 3: 디지털 홀로그래피 시뮬레이션

10.1절의 Python 코드를 수정하여:
(a) 더 복잡한 물체를 생성하세요: 여러 점 광원으로 이루어진 "H" 글자.
(b) 홀로그램을 기록하고 재현하세요.
(c) 일부 점 광원을 서로 다른 깊이에 배치하여 수치 초점 재조정을 시연하세요.
(d) 기록된 홀로그램에 잡음을 추가하여(카메라 잡음 시뮬레이션) 재현 품질에 미치는 영향을 관찰하세요.

### 연습 문제 4: Gabor 대 오프축 비교

Python을 사용하여 동일한 물체에 대해 Gabor 인라인 방식과 Leith-Upatnieks 오프축 방식 홀로그래피를 모두 시뮬레이션하세요.

(a) Gabor의 경우, 쌍둥이 상 아티팩트를 보여주세요.
(b) 오프축의 경우, 푸리에 영역에서 세 항의 공간적 분리를 보여주세요.
(c) 오프축의 경우, 푸리에 영역에서 공간 필터를 적용하여 허상 항을 분리하세요.
(d) 두 방법의 재현 상 품질을 비교하세요.

---

## 13. 참고 문헌

1. Gabor, D. (1948). "A new microscopic principle." *Nature*, 161, 777-778.
2. Leith, E. N., & Upatnieks, J. (1962). "Reconstructed wavefronts and communication theory." *JOSA*, 52(10), 1123-1130.
3. Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.). W. H. Freeman. — Chapter 9.
4. Hariharan, P. (2002). *Basics of Holography*. Cambridge University Press.
5. Schnars, U., & Jueptner, W. (2005). *Digital Holography*. Springer.
6. Kogelnik, H. (1969). "Coupled wave theory for thick hologram gratings." *Bell System Technical Journal*, 48(9), 2909-2947.

---

[← 이전: 10. 푸리에 광학](10_Fourier_Optics.md) | [다음: 12. 비선형 광학 →](12_Nonlinear_Optics.md)
