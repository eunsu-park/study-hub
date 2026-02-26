# 12. 비선형 광학

[← 이전: 11. 홀로그래피](11_Holography.md) | [다음: 13. 양자광학 입문 →](13_Quantum_Optics_Primer.md)

---

일상적인 경험에서 빛은 선형적으로 행동합니다. 유리잔에 두 개의 손전등을 비추어도 새로운 색이 생기지 않습니다. 두 빔은 서로 상호작용 없이 통과합니다. 그러나 레이저가 생성하는 극단적인 세기에서 — 원자의 결합 전기장($\sim 10^{11}\,\text{V/m}$)에 근접하는 전기장에서 — 물질의 광학적 반응은 비선형이 되고, 놀라운 현상들이 발생합니다. 빛이 색을 바꾸고, 빔들이 서로 상호작용하며, 물질은 빛의 세기에 따라 투명하거나 불투명해질 수 있습니다.

비선형 광학(Nonlinear Optics)은 레이저 발명 직후인 1961년 Franken과 동료들이 제2고조파 생성을 관찰하면서 탄생하여, 이제는 녹색 레이저 포인터(적외선의 주파수 배가)에서부터 초고속 펄스 압축, 광 파라메트릭 증폭기, 양자 얽힘 광자 광원에 이르는 응용을 가진 광대한 분야로 성장했습니다. 이 레슨에서는 비선형 분극(polarization)으로부터 이론을 전개하고, 주요 2차 및 3차 효과를 유도하며, 그 응용을 탐구합니다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. 비조화 진동자 모델을 통해 광학적 비선형성의 기원을 설명하고, 비선형 분극을 거듭제곱 급수로 표현한다
2. 2차 효과($\chi^{(2)}$) 과정을 식별하고 설명한다: 제2고조파 생성(SHG), 합주파수/차주파수 생성(SFG/DFG)
3. 위상 정합 조건을 유도하고, 복굴절 및 준위상 정합 기술을 설명한다
4. 3차 효과($\chi^{(3)}$) 과정을 식별하고 설명한다: Kerr 효과, 자기 위상 변조(SPM), 사파 혼합(FWM)
5. 광 파라메트릭 발진기(OPO) 및 증폭기(OPA)의 동작 원리를 설명한다
6. 주파수 변환, 초고속 펄스 생성, 얽힘 광자 광원 등의 응용을 분석한다
7. 위상 정합을 포함한 제2고조파 생성의 수치 시뮬레이션을 수행한다

---

## 목차

1. [광학적 비선형성의 기원](#1-광학적-비선형성의-기원)
2. [비선형 분극과 감수율](#2-비선형-분극과-감수율)
3. [제2고조파 생성 (SHG)](#3-제2고조파-생성-shg)
4. [합주파수 및 차주파수 생성](#4-합주파수-및-차주파수-생성)
5. [위상 정합](#5-위상-정합)
6. [3차 비선형 효과](#6-3차-비선형-효과)
7. [자기 위상 변조와 솔리톤](#7-자기-위상-변조와-솔리톤)
8. [광 파라메트릭 과정](#8-광-파라메트릭-과정)
9. [응용](#9-응용)
10. [Python 예제](#10-python-예제)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)
13. [참고 문헌](#13-참고-문헌)

---

## 1. 광학적 비선형성의 기원

### 1.1 선형 영역

선형 광학에서는 물질의 분극(polarization) $\mathbf{P}$ (단위 부피당 전기 쌍극자 모멘트)가 인가된 전기장에 비례합니다:

$$\mathbf{P} = \epsilon_0 \chi^{(1)} \mathbf{E}$$

여기서 $\chi^{(1)}$은 선형 감수율(linear susceptibility)입니다. 이로부터 굴절률 $n = \sqrt{1 + \chi^{(1)}}$과 선형 흡수가 생깁니다. 이 영역에서는 중첩 원리가 완벽하게 성립합니다. 빛의 빔들은 서로 상호작용 없이 통과합니다.

### 1.2 비조화 진동자 모델

핵에 결합된 전자를 생각해 봅시다. 선형 영역에서는 복원력이 조화적(harmonic)입니다: $F = -\kappa x$. 그러나 큰 변위(강한 전기장)에서는 복원력이 비조화적이 됩니다:

$$F = -\kappa x - \kappa_2 x^2 - \kappa_3 x^3 - \cdots$$

$x^2$ 항(비반전 대칭 물질에서)은 $\chi^{(2)}$ 효과를 유발하고, $x^3$ 항은 $\chi^{(3)}$ 효과를 유발합니다. 비선형 항은 작은 보정량입니다 — 중간 세기에서 일반적으로 $x_{\text{NL}}/x_{\text{L}} \sim E/E_{\text{atomic}} \sim 10^{-8}$ 수준 — 하지만 가간섭성 레이저 광원으로 검출이 가능합니다.

> **비유**: 그네를 미는 아이를 상상해 보세요 (조화 진동자). 작은 진폭에서는 복원력이 변위에 완벽하게 비례합니다 — 살짝 밀면 살짝 돌아옵니다. 하지만 세게 밀면 그네가 극단적인 각도에 이르러 운동이 더 이상 단순 조화 운동이 아닙니다 — 주기가 바뀌고 운동이 왜곡됩니다. 물질에서는 "그네"가 평형 위치에서의 전자 변위이고, "비선형 응답"은 구동 전기장이 원자 내부 전기장에 필적할 때 발생하는 왜곡입니다.

### 1.3 전기장은 얼마나 강해야 하는가?

특성적인 원자 전기장은:

$$E_{\text{atom}} = \frac{e}{4\pi\epsilon_0 a_0^2} \approx 5 \times 10^{11}\,\text{V/m}$$

10 $\mu$m 스팟에 집속된 1 W 레이저는 세기 $I \approx 3 \times 10^9\,\text{W/m}^2$와 전기장 세기 $E \approx 5 \times 10^7\,\text{V/m}$를 가집니다 — $E_{\text{atom}}$의 약 $10^{-4}$ 수준입니다. 이것은 매우 작아 보이지만, $\chi^{(2)}$ 과정은 두 전기장의 곱을 포함하고, 수 센티미터의 결정에 걸쳐 세심한 위상 정합을 통해 50-80%의 변환 효율이 일상적으로 달성됩니다.

---

## 2. 비선형 분극과 감수율

### 2.1 거듭제곱 급수 전개

일반적인 비선형 분극은 다음과 같이 전개됩니다:

$$P_i = \epsilon_0\left[\chi^{(1)}_{ij}E_j + \chi^{(2)}_{ijk}E_jE_k + \chi^{(3)}_{ijkl}E_jE_kE_l + \cdots\right]$$

여기서:
- $\chi^{(1)}_{ij}$: 선형 감수율 (2계 텐서) — 굴절률, 흡수
- $\chi^{(2)}_{ijk}$: 2차 감수율(second-order susceptibility) (3계 텐서) — SHG, SFG, DFG, Pockels 효과
- $\chi^{(3)}_{ijkl}$: 3차 감수율(third-order susceptibility) (4계 텐서) — Kerr 효과, THG, FWM

### 2.2 $\chi^{(2)}$에 대한 대칭 제약

중요한 대칭 원리: **$\chi^{(2)}$는 반전 대칭 매체에서 0이 됩니다.** 결정에 반전 대칭이 있으면, 모든 좌표를 반전하면 ($\mathbf{E} \to -\mathbf{E}$, $\mathbf{P} \to -\mathbf{P}$):

$$-P = \chi^{(2)}(-E)(-E) = \chi^{(2)}E^2 = P$$

이는 $P = 0$을 의미합니다. 따라서:
- **$\chi^{(2)} \neq 0$**: 비반전 대칭 결정 (KDP, BBO, LiNbO$_3$, KTP), 표면/계면
- **$\chi^{(2)} = 0$**: 반전 대칭 물질 (유리, 액체, 기체, Si, 대부분의 금속)
- **$\chi^{(3)} \neq 0$**: 모든 물질 (대칭 제약 없음)

### 2.3 대표적인 값

| 물질 | $\chi^{(2)}$ (pm/V) | $\chi^{(3)}$ (m$^2$/V$^2$) | 주요 용도 |
|------|---------------------|----------------------------|-----------|
| BBO ($\beta$-BaB$_2$O$_4$) | 2.2 | — | UV SHG, OPO |
| LiNbO$_3$ | 27 | — | SHG, EO 변조, QPM |
| KTP (KTiOPO$_4$) | 16 | — | SHG (녹색 레이저 포인터) |
| 용융 실리카 | 0 (반전 대칭) | $2.5 \times 10^{-22}$ | 광섬유 Kerr 효과 |
| CS$_2$ | 0 (액체) | $3 \times 10^{-20}$ | Kerr 게이팅 |

---

## 3. 제2고조파 생성 (SHG)

### 3.1 과정

주파수 $\omega$의 강한 빔이 $\chi^{(2)}$ 결정을 통과할 때, 비선형 분극은 $2\omega$에서의 항을 포함합니다:

$$P^{(2)}(2\omega) = \epsilon_0 \chi^{(2)} E(\omega)^2$$

$2\omega$에서 진동하는 이 분극은 주파수가 두 배(파장이 절반)인 새로운 전자기파를 방출합니다. 1064 nm 적외선 레이저가 532 nm 녹색 빛을 생성합니다.

### 3.2 역사적 맥락

미시간 대학교의 Peter Franken 그룹은 1961년 — 첫 번째 레이저 발명 불과 1년 후 — 에 SHG를 시연했습니다. 그들은 루비 레이저(694.3 nm, 빨간색)를 석영 결정에 집속하여 347.15 nm (UV)에서 희미한 신호를 검출했습니다. 신호가 너무 약해서, 저널 편집자가 필름 플레이트의 희미한 점을 흠집으로 생각하고 게재된 사진에서 제거했다는 이야기가 있습니다.

### 3.3 결합파 방정식

기본파($\omega$)와 제2고조파($2\omega$) 필드는 다음에 따라 발전합니다:

$$\frac{dA_{2\omega}}{dz} = -i\kappa_1 A_\omega^2 e^{i\Delta kz}$$

$$\frac{dA_\omega}{dz} = -i\kappa_2 A_{2\omega}A_\omega^* e^{-i\Delta kz}$$

여기서:
- $\kappa_1, \kappa_2$는 $\chi^{(2)}$에 비례하는 결합 계수
- $\Delta k = k_{2\omega} - 2k_\omega$는 **위상 불일치(phase mismatch)**

### 3.4 펌프 비고갈 근사

변환 효율이 낮으면 $A_\omega \approx$ const이고:

$$A_{2\omega}(L) = -i\kappa_1 A_\omega^2 \frac{e^{i\Delta kL} - 1}{i\Delta k}$$

SHG 세기는:

$$I_{2\omega} \propto \chi^{(2)2} I_\omega^2 L^2 \text{sinc}^2\!\left(\frac{\Delta kL}{2}\right)$$

$\text{sinc}^2$ 인수는 효율적인 SHG에 $\Delta k \approx 0$ — **위상 정합(phase matching)** — 이 필요함을 보여줍니다.

### 3.5 가간섭 길이

위상 정합 없이는, SH 신호가 결정 길이에 따라 주기적으로 진동합니다:

$$L_c = \frac{\pi}{|\Delta k|} = \frac{\lambda}{4(n_{2\omega} - n_\omega)}$$

일반적으로 $L_c \sim 10\text{-}100\,\mu\text{m}$ — 실용적인 SHG에는 너무 짧습니다. 위상 정합이 이 한계를 극복합니다.

---

## 4. 합주파수 및 차주파수 생성

### 4.1 합주파수 생성 (SFG)

주파수 $\omega_1$과 $\omega_2$의 두 빔이 $\chi^{(2)}$ 결정에서 혼합되어 $\omega_3 = \omega_1 + \omega_2$의 빔을 생성합니다:

$$\omega_1 + \omega_2 \to \omega_3$$

위상 정합 조건: $\mathbf{k}_1 + \mathbf{k}_2 = \mathbf{k}_3$ (에너지 및 운동량 보존).

SHG는 $\omega_1 = \omega_2$인 특수한 경우입니다.

### 4.2 차주파수 생성 (DFG)

강한 펌프 $\omega_3$와 약한 신호 $\omega_1$이 아이들러(idler) $\omega_2 = \omega_3 - \omega_1$을 생성합니다:

$$\omega_3 - \omega_1 \to \omega_2$$

$\omega_3$에서 소멸된 각 광자가 $\omega_1$에서 광자 하나와 $\omega_2$에서 광자 하나를 생성합니다. $\omega_1$에서의 신호가 증폭됩니다 (광 파라메트릭 증폭) — 이것이 OPO(광 파라메트릭 발진기)와 OPA(광 파라메트릭 증폭기)의 기반입니다.

### 4.3 에너지 및 운동량 보존

모든 $\chi^{(2)}$ 과정은 다음을 따릅니다:

**에너지 보존** (광자 관점):
$$\hbar\omega_3 = \hbar\omega_1 + \hbar\omega_2$$

**운동량 보존** (위상 정합):
$$\hbar\mathbf{k}_3 = \hbar\mathbf{k}_1 + \hbar\mathbf{k}_2$$

이는 정확한 양자역학적 보존 법칙입니다. 위상 정합 조건 $\Delta\mathbf{k} = 0$은 단순히 참여 광자들에 대한 운동량 보존입니다.

---

## 5. 위상 정합

### 5.1 문제

분산성 매체에서는 $n(\omega)$이 주파수와 함께 증가합니다(정상 분산). 따라서:

$$k_{2\omega} = \frac{2\omega n(2\omega)}{c} \neq 2k_\omega = \frac{2\omega n(\omega)}{c}$$

$n(2\omega) > n(\omega)$이기 때문입니다. 위상 불일치 $\Delta k = k_{2\omega} - 2k_\omega > 0$이 건설적 축적을 방해합니다.

### 5.2 복굴절 위상 정합

해결책은 **복굴절(birefringence)**을 이용합니다. 이방성 결정에서는 굴절률이 편광에 따라 다릅니다 (정상(ordinary) $n_o$와 비정상(extraordinary) $n_e$). 결정 방향(광축에 대한 각도 $\theta$)을 선택하면:

$$n_e(2\omega, \theta) = n_o(\omega)$$

이것이 **Type I 위상 정합**: 두 기본파 광자는 같은 편광(정상)을 가지고, SH 광자는 수직 편광(비정상)을 가집니다.

각도 $\theta$에서의 비정상 굴절률:

$$\frac{1}{n_e^2(\theta)} = \frac{\cos^2\theta}{n_o^2} + \frac{\sin^2\theta}{n_e^2}$$

**Type II 위상 정합**: 두 기본파 광자가 수직 편광을 가집니다 (하나는 정상, 하나는 비정상):

$$n_e(\omega, \theta) + n_o(\omega) = 2n_e(2\omega, \theta)$$

### 5.3 준위상 정합 (QPM)

대안적 접근: $\chi^{(2)}$의 부호를 주기 $\Lambda$로 주기적으로 반전시켜 **주기적 분극 역전(periodically poled)** 결정을 생성합니다:

$$\chi^{(2)}(z) = d_{\text{eff}}\,\text{sign}\!\left[\cos\!\left(\frac{2\pi z}{\Lambda}\right)\right]$$

주기적 반전은 위상 불일치를 보상하는 격자 벡터 $K_G = 2\pi/\Lambda$를 추가합니다:

$$\Delta k = k_{2\omega} - 2k_\omega - K_G = 0$$

$$\boxed{\Lambda = \frac{2\pi}{\Delta k} = \frac{2L_c}{\pi}}$$

> **비유**: 위상 정합은 그네를 정확한 타이밍에 미는 것과 같습니다. 그네가 당신에게 올 때마다 밀면 (위상이 맞음), 그네는 점점 더 높이 올라갑니다 — 이것이 위상 정합된 SHG로, SH 신호가 건설적으로 축적됩니다. 밀기 타이밍이 무작위라면, 때로는 앞으로, 때로는 뒤로 밀어 그네가 거의 움직이지 않습니다 — 이것이 위상 불일치 경우입니다. 준위상 정합은 한 주기의 절반 동안 앞으로 밀고, 나머지 절반 동안 옆으로 비켜서고, 다시 미는 것과 같습니다 — 완벽한 타이밍만큼 효율적이지는 않지만 그네는 여전히 점점 높아집니다.

**QPM의 장점**:
- 가장 **큰** $\chi^{(2)}$ 성분 사용 ($d_{33}$ in LiNbO$_3$, 복굴절 위상 정합에서 사용되는 성분보다 일반적으로 5배 큼)
- $\Lambda$ 선택으로 임의의 위상 정합 파장 설계 가능
- 워크오프(walk-off) 없음 (빔이 결정 축을 따라 전파)
- 주기적 분극 역전 리튬 니오베이트(PPLN)가 주력 재료

---

## 6. 3차 비선형 효과

### 6.1 Kerr 효과 (광학 Kerr 효과)

주파수 $\omega$에서의 3차 분극은 다음 항을 포함합니다:

$$P^{(3)}(\omega) = 3\epsilon_0\chi^{(3)}|E(\omega)|^2 E(\omega)$$

이로 인해 세기 의존 굴절률이 생깁니다:

$$\boxed{n = n_0 + n_2 I}$$

여기서 $n_2 = \frac{3\chi^{(3)}}{4n_0^2\epsilon_0 c}$는 비선형 굴절률입니다. 실리카 유리의 경우: $n_2 \approx 2.6 \times 10^{-20}\,\text{m}^2/\text{W}$.

### 6.2 자기 수렴

가우시안 세기 프로파일을 가진 빔은 중심(높은 세기)에서 가장자리(낮은 세기)보다 더 강한 굴절률 증가를 경험하여 양의 렌즈를 형성합니다. 빔 출력이 **임계 출력**을 초과하면:

$$P_{\text{cr}} = \frac{3.77\lambda^2}{8\pi n_0 n_2} \approx 3\,\text{MW} \quad (\text{800 nm에서 실리카})$$

빔이 초점으로 붕괴됩니다 — **자기 수렴(self-focusing)**. 이로 인해 고출력 레이저 시스템에서 치명적인 손상이 발생할 수 있으며, 대기 중 필라멘테이션(filamentation)의 원인이 됩니다.

### 6.3 사파 혼합 (FWM)

$\omega_1, \omega_2, \omega_3$에서 세 파동이 $\chi^{(3)}$을 통해 상호작용하여 다음 주파수에서 네 번째 파동을 생성합니다:

$$\omega_4 = \omega_1 + \omega_2 - \omega_3$$

위상 정합: $\mathbf{k}_4 = \mathbf{k}_1 + \mathbf{k}_2 - \mathbf{k}_3$.

**축퇴 FWM(degenerate FWM)** ($\omega_1 = \omega_2 = \omega_3 = \omega$): $\omega_4 = \omega$를 생성 — 위상 켤레(phase conjugation) (광학 위상 켤레 거울). 이는 파면 왜곡을 되돌릴 수 있습니다.

**비축퇴 FWM(non-degenerate FWM)**: 새로운 주파수 성분을 생성합니다. 광섬유 광학에서, WDM 채널 사이의 FWM은 혼선을 일으킵니다 — 이로 인해 비영 분산 이동 광섬유(non-zero dispersion-shifted fiber)가 개발되었습니다.

---

## 7. 자기 위상 변조와 솔리톤

### 7.1 자기 위상 변조 (SPM)

Kerr 매체를 통해 전파하는 펄스는 세기 의존 위상을 얻습니다:

$$\phi_{\text{NL}}(t) = -n_2 I(t) \frac{\omega_0 L}{c}$$

가우시안 펄스에 대해, 순간 주파수 이동:

$$\Delta\omega(t) = -\frac{d\phi_{\text{NL}}}{dt} = n_2\frac{\omega_0 L}{c}\frac{dI}{dt}$$

- **선단(leading edge)** ($dI/dt > 0$): 주파수 감소 (적색 이동)
- **후단(trailing edge)** ($dI/dt < 0$): 주파수 증가 (청색 이동)

SPM은 새로운 주파수를 생성하여 시간 프로파일을 변화시키지 않고 (분산이 없을 때) 펄스 스펙트럼을 넓힙니다. 이 스펙트럼 광대역화가 **초연속체 생성(supercontinuum generation)**의 기반입니다 — 단 하나의 초단 펄스가 전체 가시 영역에 걸친 스펙트럼을 생성할 수 있습니다.

### 7.2 광학 솔리톤

1550 nm에서 이상 분산($D > 0$)이 있는 광섬유에서는, SPM(스펙트럼을 넓힘)과 분산(적색 이동된 선단과 청색 이동된 후단을 압축)의 상호작용이 완벽하게 균형을 이룰 수 있습니다. 결과는 **솔리톤(soliton)** — 형태를 바꾸지 않고 전파하는 펄스:

$$A(z, t) = A_0\,\text{sech}\!\left(\frac{t}{\tau_0}\right)e^{i\gamma P_0 z/2}$$

솔리톤 조건은 최대 출력 $P_0$와 펄스 폭 $\tau_0$ 사이의 특정 관계를 요구합니다:

$$P_0 = \frac{|{\beta_2}|}{\gamma \tau_0^2}$$

여기서 $\beta_2$는 군속도 분산이고, $\gamma = n_2\omega/(cA_{\text{eff}})$는 비선형 매개변수입니다.

솔리톤은 장거리 광섬유 통신(솔리톤 전송)에 제안되었지만, 디지털 신호 처리가 포함된 코히어런트 검출로 대부분 대체되었습니다.

---

## 8. 광 파라메트릭 과정

### 8.1 광 파라메트릭 증폭 (OPA)

DFG에서 $\omega_1$에서의 신호파가 증폭되는 동시에 $\omega_2 = \omega_3 - \omega_1$에서 새로운 아이들러파가 생성됩니다. 신호 이득:

$$G = 1 + \left(\frac{\Gamma}{\kappa}\right)^2\sinh^2(\kappa L)$$

여기서 $\Gamma \propto \sqrt{I_{\text{pump}}}\,\chi^{(2)}$는 파라메트릭 이득 계수이고, $\kappa = \sqrt{\Gamma^2 - (\Delta k/2)^2}$입니다.

완벽한 위상 정합에서: $G \approx \frac{1}{4}e^{2\Gamma L}$ — 레이저 증폭기와 유사한 지수적 이득.

### 8.2 광 파라메트릭 발진기 (OPO)

파라메트릭 이득 매체를 광학 공동(cavity) 안에 배치합니다:

```
거울 ──── χ⁽²⁾ 결정 ──── 거울
   ↑ 펌프 (ω₃)
   ← 신호 (ω₁) — 공동에서 공진
   → 아이들러 (ω₂) — 공진할 수도 있음
```

파라메트릭 이득이 공동 손실을 초과하면, OPO가 발진하여 단일 펌프에서 신호 및 아이들러 빔을 생성합니다. 출력 주파수는 위상 정합과 공동 공진에 의해 결정됩니다.

### 8.3 파장 가변성

OPO의 큰 장점: **넓은 파장 가변성**. 결정 각도, 온도, 또는 분극 역전 주기를 변경함으로써 신호 및 아이들러 파장을 매우 넓은 범위에 걸쳐 연속적으로 조정할 수 있습니다.

예시: 1064 nm에서 펌프된 PPLN OPO는 1.4-4.5 $\mu$m의 신호 출력을 생성할 수 있으며, 분자 분광학, 가스 감지, 방위 응용에 중요한 전체 중적외선 영역을 커버합니다.

### 8.4 자발적 파라메트릭 하향 변환 (SPDC)

신호파가 없을 때, $\omega_1$에서의 양자 진공 요동이 파라메트릭 과정을 시작하여 펌프 광자가 자발적으로 분리될 수 있습니다:

$$\omega_{\text{pump}} \to \omega_{\text{signal}} + \omega_{\text{idler}}$$

신호 및 아이들러 광자는 동시에 생성되며, 에너지, 운동량, 편광, 시간 측면에서 **양자역학적으로 얽혀 있습니다(quantum-mechanically entangled)**. SPDC는 양자 광학 실험에서 가장 널리 사용되는 얽힘 광자 쌍 광원입니다(13강 참조).

---

## 9. 응용

### 9.1 주파수 변환

- **녹색 레이저 포인터**: Nd:YVO$_4$ (1064 nm) → KTP SHG → 532 nm 녹색
- **UV 레이저**: 다중 SHG 단계: 1064 → 532 → 266 nm (4고조파)
- **심자외선**: 리소그래피, 분광학을 위한 BBO 결정에서 SFG 및 SHG
- **중적외선 생성**: 분자 분광학용 DFG 및 OPO ($3\text{-}20\,\mu\text{m}$)

### 9.2 초고속 펄스 기술

- **SPM 스펙트럼 광대역화** + 처프 거울 압축: 수 펨토초(few-femtosecond) 펄스
- **광학 파라메트릭 처프 펄스 증폭(OPCPA)**: 극히 높은 최대 출력 (PW급 레이저)
- **주파수 빗(frequency comb)**: 모드 잠금 레이저 + SPM 광대역화 → 위상 안정 빗 (2005년 노벨상, Hall & Hansch)

### 9.3 양자 광학

- SPDC를 통한 **얽힘 광자 쌍**: 양자 통신 및 양자 컴퓨팅 실험의 기반
- 파라메트릭 과정을 통한 **압착광(squeezed light)**: 중력파 검출기(LIGO)에서 사용
- **헤럴디드 단일 광자 광원(heralded single photon sources)**: SPDC 쌍의 한 광자를 검출함으로써 짝의 존재를 "예고"

### 9.4 광통신

- FWM 또는 DFG를 사용한 광섬유 네트워크에서의 **파장 변환**
- Kerr 효과를 이용한 **전광 스위칭(all-optical switching)**
- WDM 채널 테스트 및 분광학을 위한 **초연속체 광원**

---

## 10. Python 예제

### 10.1 위상 정합이 있는 제2고조파 생성

```python
import numpy as np
import matplotlib.pyplot as plt

def shg_coupled_equations(L, N, d_eff, wavelength, n_omega, n_2omega,
                           I_pump, delta_k=0):
    """
    Solve the coupled wave equations for SHG.

    We integrate the pair of ODEs that describe energy exchange between
    the fundamental (ω) and second harmonic (2ω) fields. The phase
    mismatch Δk controls whether the exchange is constructive (Δk=0)
    or oscillatory (Δk≠0). This is a direct numerical demonstration
    of why phase matching is so critical.
    """
    c = 3e8
    eps_0 = 8.854e-12
    omega = 2 * np.pi * c / wavelength

    # Convert intensity to field amplitude: I = n*eps_0*c*|E|^2 / 2
    E_pump = np.sqrt(2 * I_pump / (n_omega * eps_0 * c))

    # Coupling coefficients
    kappa_1 = d_eff * omega / (n_2omega * c)  # for SH growth
    kappa_2 = d_eff * omega / (n_omega * c)   # for fundamental depletion

    # Initialize fields
    dz = L / N
    z = np.linspace(0, L, N + 1)
    A_omega = np.zeros(N + 1, dtype=complex)
    A_2omega = np.zeros(N + 1, dtype=complex)
    A_omega[0] = E_pump
    A_2omega[0] = 0.0

    # Runge-Kutta 4th order integration
    for i in range(N):
        zi = z[i]
        Aw = A_omega[i]
        A2w = A_2omega[i]

        def dA2w_dz(Aw_val, zi_val):
            return -1j * kappa_1 * Aw_val**2 * np.exp(1j * delta_k * zi_val)

        def dAw_dz(Aw_val, A2w_val, zi_val):
            return -1j * kappa_2 * A2w_val * np.conj(Aw_val) * np.exp(-1j * delta_k * zi_val)

        # RK4 for coupled system
        k1_2w = dz * dA2w_dz(Aw, zi)
        k1_w = dz * dAw_dz(Aw, A2w, zi)

        k2_2w = dz * dA2w_dz(Aw + k1_w/2, zi + dz/2)
        k2_w = dz * dAw_dz(Aw + k1_w/2, A2w + k1_2w/2, zi + dz/2)

        k3_2w = dz * dA2w_dz(Aw + k2_w/2, zi + dz/2)
        k3_w = dz * dAw_dz(Aw + k2_w/2, A2w + k2_2w/2, zi + dz/2)

        k4_2w = dz * dA2w_dz(Aw + k3_w, zi + dz)
        k4_w = dz * dAw_dz(Aw + k3_w, A2w + k3_2w, zi + dz)

        A_2omega[i+1] = A2w + (k1_2w + 2*k2_2w + 2*k3_2w + k4_2w) / 6
        A_omega[i+1] = Aw + (k1_w + 2*k2_w + 2*k3_w + k4_w) / 6

    # Convert to intensities
    I_omega = 0.5 * n_omega * eps_0 * c * np.abs(A_omega)**2
    I_2omega = 0.5 * n_2omega * eps_0 * c * np.abs(A_2omega)**2

    return z, I_omega, I_2omega

# Parameters for KTP crystal, SHG of 1064 nm → 532 nm
wavelength = 1064e-9
d_eff = 3.18e-12  # Effective nonlinear coefficient for KTP (m/V)
n_omega = 1.740    # Refractive index at 1064 nm
n_2omega = 1.779   # Refractive index at 532 nm
L = 0.02           # 20 mm crystal length
I_pump = 1e12      # 1 GW/m² (moderately focused pulsed laser)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: Phase-matched vs. mismatched ---
for dk_label, dk_val in [('Δk = 0 (perfect PM)', 0),
                          ('Δk = 100 /m', 100),
                          ('Δk = 1000 /m', 1000)]:
    z, I_w, I_2w = shg_coupled_equations(L, 5000, d_eff, wavelength,
                                          n_omega, n_2omega, I_pump, dk_val)
    efficiency = I_2w / I_pump
    axes[0].plot(z * 1e3, efficiency * 100, linewidth=2, label=dk_label)

axes[0].set_xlabel('Crystal length (mm)', fontsize=11)
axes[0].set_ylabel('SHG conversion efficiency (%)', fontsize=11)
axes[0].set_title('SHG Efficiency: Phase Matching Matters', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# --- Right: Quasi-phase matching ---
# Demonstrate the sinc² dependence on Δk*L
dk_range = np.linspace(-2000, 2000, 1000)
L_crystal = 0.01  # 10 mm

# Undepleted pump approximation: η ∝ sinc²(ΔkL/2)
eta_approx = np.sinc(dk_range * L_crystal / (2 * np.pi))**2

axes[1].plot(dk_range, eta_approx, 'b-', linewidth=2)
axes[1].set_xlabel('Phase mismatch Δk (1/m)', fontsize=11)
axes[1].set_ylabel('Normalized efficiency (sinc²)', fontsize=11)
axes[1].set_title(f'SHG Efficiency vs. Phase Mismatch (L = {L_crystal*1e3:.0f} mm)',
                   fontsize=13)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Perfect PM')
axes[1].legend()

plt.tight_layout()
plt.savefig('shg_phase_matching.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 10.2 자기 위상 변조

```python
import numpy as np
import matplotlib.pyplot as plt

def self_phase_modulation(t, pulse, n2, omega0, L, I_peak):
    """
    Compute self-phase modulation of an optical pulse.

    SPM arises because the Kerr effect makes the refractive index
    intensity-dependent: n = n0 + n2*I. A pulse with time-varying
    intensity acquires a time-varying phase, which generates new
    frequency components and broadens the spectrum — without
    changing the pulse shape in time (when dispersion is negligible).
    """
    c = 3e8
    # Normalize pulse intensity
    I_t = I_peak * np.abs(pulse)**2 / np.max(np.abs(pulse)**2)

    # Nonlinear phase
    phi_nl = n2 * I_t * omega0 * L / c

    # Apply SPM
    pulse_spm = pulse * np.exp(1j * phi_nl)

    # Instantaneous frequency shift
    dt = t[1] - t[0]
    delta_omega = -np.gradient(phi_nl, dt)

    return pulse_spm, phi_nl, delta_omega

# Gaussian pulse parameters
c = 3e8
wavelength = 800e-9  # Ti:Sapphire
omega0 = 2 * np.pi * c / wavelength
tau_fwhm = 50e-15  # 50 fs pulse
tau = tau_fwhm / (2 * np.sqrt(np.log(2)))  # Gaussian 1/e width

# Time grid
t = np.linspace(-200e-15, 200e-15, 4096)
pulse_in = np.exp(-t**2 / (2 * tau**2))

# Material parameters
n2 = 2.6e-20  # Silica glass
I_peak = 5e13  # W/m² (moderately intense)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Different propagation lengths to show SPM evolution
lengths = [0.001, 0.005, 0.01, 0.02]  # meters
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(lengths)))

for L, color in zip(lengths, colors):
    pulse_out, phi_nl, delta_omega = self_phase_modulation(
        t, pulse_in, n2, omega0, L, I_peak
    )

    max_phi = np.max(phi_nl)
    label = f'L = {L*1e3:.0f} mm (φ_max = {max_phi:.1f} rad)'

    # Temporal intensity (unchanged by SPM alone)
    axes[0, 0].plot(t * 1e15, np.abs(pulse_out)**2, color=color,
                     linewidth=1.5, label=label)

    # Nonlinear phase
    axes[0, 1].plot(t * 1e15, phi_nl, color=color, linewidth=1.5)

    # Spectrum (broadened by SPM)
    spectrum_out = np.fft.fftshift(np.abs(np.fft.fft(pulse_out))**2)
    freq = np.fft.fftshift(np.fft.fftfreq(len(t), t[1] - t[0]))
    # Convert to wavelength-like relative frequency
    axes[1, 0].plot(freq * 1e-12, spectrum_out / spectrum_out.max(),
                     color=color, linewidth=1.5, label=label)

    # Chirp (instantaneous frequency)
    axes[1, 1].plot(t * 1e15, delta_omega * 1e-12, color=color, linewidth=1.5)

axes[0, 0].set_xlabel('Time (fs)')
axes[0, 0].set_ylabel('|E|² (normalized)')
axes[0, 0].set_title('Temporal Intensity')
axes[0, 0].legend(fontsize=8)

axes[0, 1].set_xlabel('Time (fs)')
axes[0, 1].set_ylabel('Nonlinear phase (rad)')
axes[0, 1].set_title('SPM Phase φ_NL(t)')

axes[1, 0].set_xlabel('Frequency (THz)')
axes[1, 0].set_ylabel('Spectral intensity')
axes[1, 0].set_title('Spectrum (broadened by SPM)')
axes[1, 0].set_xlim(-30, 30)

axes[1, 1].set_xlabel('Time (fs)')
axes[1, 1].set_ylabel('Δω (THz)')
axes[1, 1].set_title('Chirp (Instantaneous Frequency Shift)')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.suptitle('Self-Phase Modulation of a 50 fs Pulse', fontsize=14)
plt.tight_layout()
plt.savefig('self_phase_modulation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 10.3 준위상 정합 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def qpm_comparison(L, coherence_length, N_points=10000):
    """
    Compare SHG growth for perfect PM, no PM, and quasi-PM.

    Perfect phase matching: SH grows quadratically with z.
    No phase matching: SH oscillates between 0 and a small maximum
    with period 2*L_c — energy sloshes back and forth.
    QPM: periodically flipping χ⁽²⁾ with period 2*L_c prevents
    the back-conversion, giving steady (though slower) growth.
    """
    z = np.linspace(0, L, N_points)
    dz = z[1] - z[0]

    # Perfect phase matching: Δk = 0, SH grows linearly with z
    # (undepleted pump: A_2w ∝ z, I_2w ∝ z²)
    shg_perfect = (z / L)**2

    # Phase mismatched: oscillates with coherence length
    dk = np.pi / coherence_length  # Δk = π/L_c
    shg_mismatched = np.sin(dk * z / 2)**2 / (dk * L / 2)**2

    # Quasi-phase matched: flip χ⁽²⁾ every L_c
    # The SH field grows in a staircase pattern
    period = 2 * coherence_length
    # Effective Δk after QPM: reduced by factor (2/π)
    # SH grows as (2z/(πL))² compared to (z/L)² for perfect PM
    shg_qpm = (2 * z / (np.pi * L))**2

    # For detailed visualization: actual staircase growth
    shg_qpm_exact = np.zeros_like(z)
    A_2w = 0.0  # SH field amplitude
    for i in range(1, len(z)):
        # Determine sign of χ⁽²⁾ based on QPM period
        domain = int(z[i] / coherence_length) % 2
        sign = 1.0 if domain == 0 else -1.0

        # Growth with phase mismatch, but χ⁽²⁾ flips to compensate
        A_2w += sign * np.exp(1j * dk * z[i]) * dz
        shg_qpm_exact[i] = np.abs(A_2w)**2

    # Normalize
    shg_qpm_exact /= shg_qpm_exact[-1] if shg_qpm_exact[-1] > 0 else 1

    return z, shg_perfect, shg_mismatched, shg_qpm, shg_qpm_exact

L = 10e-3  # 10 mm crystal
L_c = 0.5e-3  # 0.5 mm coherence length

z, perfect, mismatched, qpm_approx, qpm_exact = qpm_comparison(L, L_c, 50000)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(z * 1e3, perfect, 'g-', linewidth=2, label='Perfect phase matching')
ax.plot(z * 1e3, qpm_exact * (2/np.pi)**2, 'b-', linewidth=1.5,
        label=f'Quasi-phase matching (Λ = {2*L_c*1e3:.1f} mm)')
ax.plot(z * 1e3, mismatched * (L_c/L)**2 * 4, 'r-', linewidth=1.5,
        label='No phase matching (oscillating)')

# Mark coherence length
for i in range(int(L / L_c)):
    if i < 3:  # Only mark first few
        ax.axvline(x=(i + 0.5) * 2 * L_c * 1e3, color='gray',
                   linestyle=':', alpha=0.3)

ax.set_xlabel('Crystal length (mm)', fontsize=12)
ax.set_ylabel('SHG intensity (normalized)', fontsize=12)
ax.set_title('Comparison: Perfect PM vs. QPM vs. No PM', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qpm_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. 요약

| 개념 | 핵심 공식 / 아이디어 |
|------|----------------------|
| 비선형 분극 | $P = \epsilon_0(\chi^{(1)}E + \chi^{(2)}E^2 + \chi^{(3)}E^3 + \cdots)$ |
| $\chi^{(2)}$ 대칭 | 반전 대칭 매체에서 0 |
| SHG | $\omega + \omega \to 2\omega$; $I_{2\omega} \propto \chi^{(2)2}I_\omega^2 L^2\text{sinc}^2(\Delta kL/2)$ |
| 위상 정합 | $\Delta k = k_{2\omega} - 2k_\omega = 0$; 운동량 보존 |
| 복굴절 위상 정합 | $n_e(2\omega, \theta) = n_o(\omega)$ |
| 준위상 정합 | 주기적 분극 역전 결정; $\Lambda = 2L_c$ |
| 가간섭 길이 | $L_c = \pi/\|\Delta k\|$ |
| Kerr 효과 | $n = n_0 + n_2 I$ |
| 자기 위상 변조 | 세기 의존 위상 → 스펙트럼 광대역화 |
| 자기 수렴 | $P > P_{\text{cr}} = 3.77\lambda^2/(8\pi n_0 n_2)$ → 빔 붕괴 |
| FWM | $\omega_4 = \omega_1 + \omega_2 - \omega_3$ (사파 혼합) |
| OPO | $\chi^{(2)}$ 파라메트릭 발진기; 넓은 파장 가변성 |
| SPDC | 자발적 광자 쌍 생성; 얽힘 광자 |

---

## 12. 연습 문제

### 연습 문제 1: SHG 효율

KTP 결정 ($d_{\text{eff}} = 3.18\,\text{pm/V}$, $n_\omega = 1.740$, $n_{2\omega} = 1.779$)이 1064 nm 빛의 SHG에 대해 위상 정합되어 있습니다.

(a) 위상 정합 없이 가간섭 길이를 계산하세요.
(b) 결정 길이 10 mm, 펌프 세기 $I = 100\,\text{MW/cm}^2$에서, 펌프 비고갈 근사로 SHG 효율을 추정하세요.
(c) 펌프 비고갈 근사가 붕괴되는 펌프 세기는 얼마입니까 (예: 효율이 10%를 초과할 때)?

### 연습 문제 2: QPM 설계

상온에서 1550 nm → 775 nm의 SHG를 위한 PPLN(주기적 분극 역전 리튬 니오베이트) 결정을 설계하세요. 주어진 값: $n(1550\,\text{nm}) = 2.211$, $n(775\,\text{nm}) = 2.259$.

(a) QPM 없이 위상 불일치 $\Delta k$를 계산하세요.
(b) 필요한 QPM 주기 $\Lambda$를 계산하세요.
(c) $d_{33} = 27\,\text{pm/V}$이면, 1차 QPM에 대한 유효 $d_{\text{eff}}$는 얼마입니까?
(d) $I = 1\,\text{GW/m}^2$에서 50% 변환 효율을 위해 결정은 얼마나 길어야 합니까?

### 연습 문제 3: 자기 위상 변조

800 nm에서 최대 출력 1 MW의 100 fs 펄스가 용융 실리카 1 cm을 통과합니다 ($n_2 = 2.6 \times 10^{-20}\,\text{m}^2/\text{W}$, 빔 면적 = 100 $\mu$m$^2$).

(a) 최대 세기와 최대 비선형 위상 $\phi_{\text{NL,max}}$를 계산하세요.
(b) 스펙트럼 광대역화 인수를 추정하세요.
(c) 자기 수렴이 우려됩니까? $P/P_{\text{cr}}$를 계산하세요.
(d) Python SPM 코드를 수정하여 이 경우를 시각화하세요.

### 연습 문제 4: 위상 정합 각도

BBO 결정에서 800 nm의 Type I SHG를 위한 위상 정합 각도 ($n_o(\text{800}) = 1.6609$, $n_e(\text{800}) = 1.5426$, $n_o(\text{400}) = 1.6924$, $n_e(\text{400}) = 1.5667$):

(a) $n_e(400, \theta) = n_o(800)$인 위상 정합 각도 $\theta_{\text{PM}}$을 계산하세요.
(b) 각도 허용 대역폭 $\Delta\theta$를 계산하세요.
(c) 초단 펄스 SHG에서 BBO가 KTP보다 선호되는 이유는 무엇입니까?

---

## 13. 참고 문헌

1. Boyd, R. W. (2020). *Nonlinear Optics* (4th ed.). Academic Press. — The standard reference.
2. Saleh, B. E. A., & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.). Wiley. — Chapter 21.
3. Shen, Y. R. (2002). *The Principles of Nonlinear Optics*. Wiley Classics.
4. Agrawal, G. P. (2019). *Nonlinear Fiber Optics* (6th ed.). Academic Press.
5. Franken, P. A., et al. (1961). "Generation of optical harmonics." *Physical Review Letters*, 7, 118.
6. Armstrong, J. A., et al. (1962). "Interactions between light waves in a nonlinear dielectric." *Physical Review*, 127, 1918.

---

[← 이전: 11. 홀로그래피](11_Holography.md) | [다음: 13. 양자광학 입문 →](13_Quantum_Optics_Primer.md)
