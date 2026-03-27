# 6. 자기 거울과 단열 불변량

## 학습 목표

- 자기 거울 힘과 입자 구속에서의 역할 이해
- 첫 번째 단열 불변량(자기 모멘트) 유도 및 보존 증명
- 자기 거울 기하학, 손실 원뿔, 포획된 입자 대 통과 입자 분석
- 단열 불변량의 계층과 관련 시간 규모 탐구
- 토카막 궤도(바나나 궤도와 통과 입자) 연구
- Python을 사용하여 자기 거울 구속과 손실 원뿔 동역학 시뮬레이션

## 1. 자기 거울 힘

### 1.1 수렴하는 자기력선

자기력선이 수렴하는(자기장 세기가 증가하는) 영역에서, 자이로하는 입자는 더 높은 자기장 영역으로의 운동을 반대하는 힘을 경험합니다. 이것이 **자기 거울 힘**입니다.

호장 $s$에 따라 $B = B(s)$가 변하는 자기력선을 따라 움직이는 입자를 고려하세요:

```
    약한 장              강한 장
    B_low                   B_high
      ↓                       ↓
      |                       |||
      |        입자            |||
      |  ←──── 움직임 ────→   |||
      |          →            |||
      |                       |||

    큰 r_L              작은 r_L
    (자이로반경)          (자이로반경)
```

입자의 수직 속도 성분은 반경 $r_L = mv_\perp/(|q|B)$로 자이로합니다. $B$가 증가하면 $r_L$이 감소하지만, 수직 운동 에너지는 Lorentz 힘이 한 일로 인해 변합니다.

### 1.2 거울 힘의 유도

자기 모멘트는 다음과 같이 정의됩니다:

$$
\mu = \frac{mv_\perp^2}{2B} = \frac{W_\perp}{B}
$$

여기서 $W_\perp = \frac{1}{2}mv_\perp^2$는 수직 운동 에너지입니다.

천천히 변하는 장(단열 한계)에 대해, $\mu$가 보존됩니다 (섹션 2에서 엄밀하게 증명). $\mu$ = 상수로 가정:

$$
W_\perp = \mu B = \text{상수} \times B
$$

총 에너지는 보존됩니다:

$$
W = W_\perp + W_\parallel = \frac{1}{2}m(v_\perp^2 + v_\parallel^2) = \text{상수}
$$

따라서:

$$
\frac{dW_\parallel}{ds} = -\frac{dW_\perp}{ds} = -\mu\frac{dB}{ds}
$$

평행 힘은:

$$
F_\parallel = \frac{dW_\parallel}{ds} = -\mu\frac{dB}{ds} = -\mu\nabla_\parallel B
$$

**이것이 자기 거울 힘입니다**: 더 높은 $B$ 영역으로의 운동을 반대합니다.

### 1.3 물리적 해석

거울 힘은 $\mathbf{B}$가 불균일할 때 Lorentz 힘 $\mathbf{F} = q\mathbf{v}\times\mathbf{B}$가 자기력선을 따라 성분을 가지기 때문에 발생합니다. 수렴하는 장에서 자이로하는 입자를 고려하세요:

```
    자이로궤도 상단 (높은 B로 움직임)
         ↑ v_perp
         |
    ────┼──── B 자기력선
         |
         ↓ Lorentz 힘이 운동을 반대하는 성분을 가짐

    자이로궤도 하단 (낮은 B로 움직임)
         |
    ────┼──── B 자기력선
         ↓ v_perp
         ↑ Lorentz 힘이 운동을 돕는 성분을 가짐

    알짜 효과: 높은 B로의 운동을 반대하는 힘
```

자이로주기에 걸쳐 평균하면 거울 힘 $F_\parallel = -\mu\nabla_\parallel B$를 얻습니다.

## 2. 첫 번째 단열 불변량: 자기 모멘트 μ

### 2.1 작용-각 변수를 통한 보존 증명

자기 모멘트 $\mu$는 첫 번째 단열 불변량입니다. 보존을 증명하기 위해, 고전 역학의 작용-각 형식을 사용합니다.

자이로운동과 관련된 작용은:

$$
J_\perp = \oint p_\perp \, dq_\perp
$$

여기서 적분은 한 자이로주기에 걸쳐입니다. 자기장에서의 원운동에 대해:

$$
J_\perp = m v_\perp \cdot 2\pi r_L = m v_\perp \cdot 2\pi \frac{mv_\perp}{|q|B} = \frac{2\pi m^2 v_\perp^2}{|q|B}
$$

단열 불변량은:

$$
I = \frac{J_\perp}{2\pi} = \frac{m^2 v_\perp^2}{|q|B} = \frac{2m}{|q|}\frac{mv_\perp^2}{2B} = \frac{2m}{|q|}\mu
$$

따라서, 장이 자이로주기에 비해 천천히 변할 때 $\mu$가 보존됩니다:

$$
\frac{1}{\omega_c}\frac{dB}{dt} \ll B
$$

이것이 **단열 조건** 또는 **느린 변화 조건**입니다.

### 2.2 물리적 의미: 자속 보존

자기 모멘트는 또한 자이로궤도를 통과하는 자기 자속으로 해석될 수 있습니다:

$$
\Phi_\text{gyro} = \pi r_L^2 B = \pi \left(\frac{mv_\perp}{|q|B}\right)^2 B = \frac{\pi m^2 v_\perp^2}{q^2 B}
$$

이것은 $\mu$에 비례합니다:

$$
\Phi_\text{gyro} = \frac{\pi m^2 v_\perp^2}{q^2 B} = \frac{\pi m}{q^2} \cdot \frac{mv_\perp^2}{B} = \frac{2\pi m}{q^2}\mu
$$

**$\mu$의 보존은 자이로궤도를 통과하는 자기 자속이 보존됨을 의미합니다** (천천히 변하는 장에 대해).

### 2.3 단열성의 붕괴

단열 불변성은 다음의 경우 붕괴합니다:

1. **빠른 장 변화**: $\frac{1}{\omega_c}\frac{dB}{dt} \sim B$ (자이로주기와 비슷함)
2. **강한 기울기**: $r_L |\nabla B| / B \sim 1$ (기울기 척도가 자이로반경과 비슷함)
3. **공명**: 주파수 $\omega \approx n\omega_c$에서의 외부 섭동 (자이로공명)

예: 자기 재결합 영역에서, $B$가 빠르게 변할 수 있어, 단열성을 위반하고 입자가 $\mu$를 변경할 수 있게 합니다.

## 3. 자기 거울과 보틀

### 3.1 거울 기하학

자기 거울은 약한 장 영역으로 분리된 두 강한 장 영역(거울)으로 구성됩니다:

```
    거울 1              중앙면           거울 2
    B_max                   B_min             B_max
      |||                     |                 |||
      |||                     |                 |||
      ||| ←─── 입자 ───→  |  ←────────────→ |||
      |||      포획됨        |                 |||
      |||                     |                 |||
       ↑                      ↑                  ↑
    반사점                  적도            반사점
```

거울 비율은:

$$
R = \frac{B_{\text{max}}}{B_{\text{min}}}
$$

### 3.2 피치각과 손실 원뿔

피치각 $\alpha$는 속도와 자기장 사이의 각도입니다:

$$
\alpha = \arctan\left(\frac{v_\perp}{v_\parallel}\right)
$$

중앙면($B = B_0 = B_{\text{min}}$)에서, 피치각은 $\alpha_0$입니다:

$$
\tan\alpha_0 = \frac{v_{\perp,0}}{v_{\parallel,0}}
$$

$\mu$와 총 에너지 보존 사용:

$$
\mu = \frac{mv_\perp^2}{2B} = \text{상수}
$$

$$
v^2 = v_\perp^2 + v_\parallel^2 = \text{상수}
$$

거울점($v_\parallel = 0$)에서, $v_\perp = v$이고 $B = B_{\text{mirror}}$:

$$
\frac{mv^2}{2B_{\text{mirror}}} = \frac{mv_{\perp,0}^2}{2B_0}
$$

따라서:

$$
\frac{v_\perp^2}{v^2}\bigg|_{\text{mirror}} = 1 = \frac{v_{\perp,0}^2}{v^2} \cdot \frac{B_{\text{mirror}}}{B_0}
$$

$v_{\perp,0}^2/v^2 = \sin^2\alpha_0$이므로:

$$
\sin^2\alpha_0 = \frac{B_0}{B_{\text{mirror}}}
$$

입자가 **포획**되려면 ($B_{\text{max}}$에 도달하기 전에 반사):

$$
\sin^2\alpha_0 > \frac{B_0}{B_{\text{max}}} = \frac{1}{R}
$$

또는 동등하게:

$$
\boxed{\alpha_0 > \alpha_c = \arcsin\left(\frac{1}{\sqrt{R}}\right)}
$$

이것이 **손실 원뿔**을 정의합니다: $\alpha_0 < \alpha_c$인 입자는 포획되지 않고 탈출합니다.

### 3.3 손실 원뿔 입체각

속도 공간에서 손실 원뿔은 $v_\parallel$ 축 주위의 원뿔입니다:

```
    속도 공간

         v_parallel
            ↑
            |       통과 입자
            |      (탈출)
            |     /
            |    / α_c (손실 원뿔 각도)
            |   /
            |  /_______________
            | /               /
            |/_______________/ ← 손실 원뿔
           /|
          / |
         /  |
        /   |
    v_perp  | 포획된 입자
            | (구속됨)
```

손실 원뿔의 입체각은:

$$
\Delta\Omega = 2\pi(1 - \cos\alpha_c) = 2\pi\left(1 - \sqrt{1 - \frac{1}{R}}\right)
$$

큰 거울 비율 $R \gg 1$에 대해:

$$
\Delta\Omega \approx \frac{\pi}{R}
$$

손실 원뿔에 있는 입자의 비율 (등방성 분포 가정):

$$
f_{\text{loss}} = \frac{\Delta\Omega}{4\pi} \approx \frac{1}{4R}
$$

### 3.4 자기 보틀

**자기 보틀**은 닫힌 자기력선을 가진 거울 구성으로, 장에 수직인 모든 방향으로 구속을 제공합니다. 예:
- 단순 거울 장치
- 쌍원뿔 컵
- Minimum-B 구성 (사중극 장)

주요 손실 메커니즘은:
1. **손실 원뿔로의 산란**: 충돌이 피치각을 변경하여, 포획된 입자를 손실 원뿔로 산란시킵니다.
2. **끝단 손실**: 손실 원뿔에 있는 입자가 거울 목을 통해 탈출합니다.

구속 시간은:

$$
\tau_{\text{conf}} \sim \frac{R}{\nu_c}\frac{1}{f_{\text{loss}}}
$$

여기서 $\nu_c$는 충돌 주파수입니다.

## 4. 바운스 운동

### 4.1 바운스 궤적

포획된 입자는 두 거울점 사이를 바운스하며, $\mathbf{B}$에 수직으로 드리프트하면서 (grad-B와 곡률 드리프트로부터) 평행 방향으로 진동합니다.

평행 속도는:

$$
v_\parallel = \pm\sqrt{v^2 - v_\perp^2} = \pm v\sqrt{1 - \frac{\mu B}{W}}
$$

여기서 $\pm$는 운동 방향에 의존합니다. 거울점에서, $v_\parallel = 0$이므로:

$$
B_{\text{mirror}} = \frac{W}{\mu} = \frac{mv^2}{2\mu}
$$

### 4.2 바운스 주파수

바운스 주기는:

$$
\tau_b = 2\int_0^{s_{\text{mirror}}} \frac{ds}{v_\parallel(s)}
$$

여기서 $s$는 자기력선을 따른 호장이고 $s_{\text{mirror}}$는 거울점까지의 거리입니다.

$B(z) = B_0(1 + z^2/L^2)$인 포물선 거울에 대해:

$$
\tau_b \approx \frac{4L}{v_\parallel}
$$

바운스 주파수는:

$$
\omega_b = \frac{2\pi}{\tau_b} \approx \frac{\pi v_\parallel}{2L}
$$

전형적인 매개변수에 대해:
- $v_\parallel \sim 10^5$ m/s
- $L \sim 10$ m (거울 길이)
- $\omega_b \sim 10^4$ rad/s

이것은 자이로주파수 $\omega_c \sim 10^8$ rad/s보다 훨씬 느립니다.

### 4.3 바운스 평균 드리프트

grad-B와 곡률 드리프트는 자기력선을 따라 변합니다. 장기적 동작에 대해, 바운스 주기에 걸쳐 평균합니다:

$$
\langle\mathbf{v}_D\rangle_b = \frac{1}{\tau_b}\int_0^{\tau_b} \mathbf{v}_D(s(t)) \, dt
$$

이 바운스 평균 드리프트가 입자 궤도의 느린 진화를 결정합니다.

## 5. 두 번째와 세 번째 단열 불변량

### 5.1 시간 규모의 계층

자기화된 플라즈마에는 세 가지 자연 시간 규모가 있습니다:

$$
\omega_c \gg \omega_b \gg \omega_d
$$

여기서:
- $\omega_c = |q|B/m$: 자이로주파수
- $\omega_b \sim v_\parallel/L$: 바운스 주파수
- $\omega_d \sim v_D/R$: 드리프트 주파수

각 시간 규모는 단열 불변량과 연관됩니다.

### 5.2 두 번째 단열 불변량: J

두 번째 불변량은 바운스 운동과 연관됩니다:

$$
\boxed{J = \oint m v_\parallel \, ds}
$$

여기서 적분은 두 거울점 사이의 자기력선을 따릅니다. 이것은 바운스 운동에 대한 작용입니다.

**보존**: $J$는 자기장이 바운스 주기에 비해 천천히 변할 때 보존됩니다:

$$
\frac{1}{\omega_b}\frac{\partial B}{\partial t} \ll B
$$

### 5.3 J의 물리적 의미

$J$는 $(s, p_\parallel)$ 위상 공간에서 둘러싸인 면적과 관련됩니다:

```
    p_parallel = m v_parallel
         ↑
         |
         |    /\
         |   /  \    입자가 바운스
         |  /    \   거울점 사이
         | /      \
         |/        \___
        ─┼──────────────→ s (자기력선을 따른 위치)
         0       s_mirror

    둘러싸인 면적 = J
```

$J$의 보존은 장이 천천히 변하면 입자가 같은 바운스 궤도에 남아 있음을 의미합니다.

### 5.4 세 번째 단열 불변량: Φ

세 번째 불변량은 축 주위의 드리프트 운동과 연관됩니다:

$$
\boxed{\Phi = \oint \mathbf{A}\cdot d\mathbf{l}}
$$

여기서 적분은 드리프트 궤도 주위이고 $\mathbf{A}$는 벡터 포텐셜입니다 ($\mathbf{B} = \nabla\times\mathbf{A}$).

Stokes 정리 사용:

$$
\Phi = \int_S (\nabla\times\mathbf{A})\cdot d\mathbf{S} = \int_S \mathbf{B}\cdot d\mathbf{S}
$$

**$\Phi$는 드리프트 궤도에 의해 둘러싸인 자기 자속입니다.**

**보존**: $\Phi$는 자기장이 드리프트 주기에 비해 천천히 변할 때 보존됩니다:

$$
\frac{1}{\omega_d}\frac{\partial B}{\partial t} \ll B
$$

### 5.5 단열 불변량 요약

| 불변량 | 공식 | 관련 운동 | 시간 규모 | 보존 조건 |
|-----------|---------|-------------------|-----------|----------------------|
| **μ** | $\frac{mv_\perp^2}{2B}$ | 자이로운동 | $\omega_c^{-1}$ | $\tau_c \ll \tau_{\text{var}}$ |
| **J** | $\oint mv_\parallel ds$ | 바운스 | $\omega_b^{-1}$ | $\tau_b \ll \tau_{\text{var}}$ |
| **Φ** | $\oint \mathbf{A}\cdot d\mathbf{l}$ | 드리프트 | $\omega_d^{-1}$ | $\tau_d \ll \tau_{\text{var}}$ |

여기서 $\tau_{\text{var}}$는 장 변화의 특성 시간입니다.

**핵심 원리**: 각 불변량은 장이 관련 운동 주기에 비해 천천히 변할 때 보존됩니다.

## 6. 토카막 궤도

### 6.1 토카막 기하학 복습

토카막은 다음을 가집니다:
- **환형 장** $B_\phi \propto 1/R$ (주반경이 증가하면서 감소)
- **폴로이달 장** $B_\theta$ (플라즈마 전류로부터)
- **총 장**: $\mathbf{B} = B_\phi\hat{\phi} + B_\theta\hat{\theta}$

장은 내부측(작은 $R$)에서 외부측(큰 $R$)보다 강합니다:

```
    토카막 위에서 본 모습

         약한 장
         (외부측)
             |
        ─────┼─────
       /     |     \
      /      o      \   ← 플라즈마
     │  (자기       │
     │    축)       │
      \            /
       \          /
        ──────────
             |
         강한 장
         (내부측)
```

이것은 환형 방향으로 자기 거울 효과를 만듭니다.

### 6.2 포획된 입자와 통과 입자

$B_\phi$의 $1/R$ 변화로 인해, 입자는 다음이 될 수 있습니다:

1. **통과 입자**: 토러스 주위로 완전한 폴로이달 회로를 완성합니다.
2. **포획된 입자**: 내부측(높은 $B$)에서 반사되어 회로를 완성할 수 없습니다.

포획 조건은 자기 거울과 유사합니다. 외부 중앙면($\theta = 0$, $R = R_0 + a$)에서:

$$
\sin^2\alpha_0 < \frac{B_{\text{out}}}{B_{\text{in}}} \approx \frac{R_{\text{in}}}{R_{\text{out}}} = \frac{R_0 - a}{R_0 + a} \approx 1 - \frac{2a}{R_0} = 1 - 2\epsilon
$$

여기서 $\epsilon = a/R_0$는 역 종횡비입니다.

작은 $\epsilon$에 대해:

$$
\alpha_c \approx \sqrt{2\epsilon}
$$

포획된 입자의 비율은:

$$
f_{\text{trap}} \approx \sqrt{2\epsilon} \approx \sqrt{\frac{a}{R_0}}
$$

$\epsilon \sim 0.3$인 전형적인 토카막에 대해:

$$
f_{\text{trap}} \approx 0.77 \approx 77\%
$$

### 6.3 바나나 궤도

포획된 입자는 수직으로 드리프트하고 (grad-B + 곡률 드리프트) 폴로이달 평면에서 "바나나 모양" 궤도를 그립니다:

```
    폴로이달 단면

         상단
          |
      ────┼────
     /    |    \
    │  /──┴──\  │  ← 바나나 궤도
    │ │       │ │     (포획된 입자)
    │  \─────/  │
     \         /
      ─────────
          |
        하단

    통과 궤도: 주위로 완전한 원
    바나나 궤도: 반사되고, 드리프트가 바나나 모양 생성
```

바나나 폭(방사상 이탈)은:

$$
\Delta r_b \sim q\rho_i\sqrt{\epsilon}
$$

여기서:
- $q = rB_\phi/(R_0 B_\theta)$는 안전 인자
- $\rho_i = m_i v_{th,i}/(eB)$는 이온 Larmor 반경
- $\epsilon = a/R_0$

**중요성**: 바나나 궤도는 다음에 의해 구속을 감소시킵니다:
1. 방사상 수송에 대한 유효 단계 크기 증가
2. 신고전적 수송 생성 (충돌 탈포획)

### 6.4 통과 입자

통과 입자는 큰 $v_\parallel$를 가지고 완전한 폴로이달 회로를 완성합니다. 그들도 수직으로 드리프트하지만 닫힌 자속 표면에 남습니다 (이상적인 축대칭 기하학에서).

자속 표면으로부터의 궤도 이동은:

$$
\Delta r_{\text{pass}} \sim q\rho_i
$$

바나나 폭보다 훨씬 작습니다.

### 6.5 신고전적 수송

포획된 입자 궤도와 충돌의 조합은 고전적 (충돌 기반) 수송을 초과하는 **신고전적 수송**을 유도합니다:

$$
D_{\text{neo}} = \frac{q^2\rho_i^2}{\tau_e}\epsilon^{3/2}
$$

여기서 $\tau_e$는 전자 충돌 시간입니다.

이것은 고전적 확산 $D_{\text{class}} \sim \rho_i^2/\tau_e$보다 $\sim q^2\epsilon^{3/2}$ 배 더 큽니다.

## 7. Van Allen 복사대

### 7.1 자기 거울로서의 지구 자기권

지구의 쌍극자 자기장은 자연 자기 보틀을 형성합니다:

```
    태양풍
    ────────→

         자기권 경계
           ____
          /    \
         /      \
    ────┤ 지구  ├────  적도면
         \      /
          \____/

    자기력선:
    - 주간측에서 압축됨 (태양풍 압력)
    - 야간측에서 연장됨 (자기꼬리)
    - 닫힌 자기력선에 포획된 입자
```

전하를 띤 입자(태양풍과 우주선으로부터의 전자와 양성자)가 쌍극자 장에 포획되어 Van Allen 복사대를 형성합니다.

### 7.2 이중대 구조

두 개의 주요 대가 있습니다:

1. **내부대** ($1.2 - 2$ 지구 반경):
   - 대부분 고에너지 양성자 (10-100 MeV)
   - 출처: 우주선 알베도 중성자 붕괴 (CRAND)
   - 상대적으로 안정적

2. **외부대** ($4 - 6$ 지구 반경):
   - 대부분 전자 (0.1-10 MeV)
   - 출처: 지자기 폭풍 동안 태양풍 주입
   - 매우 가변적

### 7.3 쌍극자 장에서의 입자 운동

입자는 세 가지 운동을 겪습니다:
1. **자이로운동** 자기력선 주위 (양성자에 대해 $\omega_c \sim 10^4$ rad/s)
2. **바운스** 극 근처 거울점 사이 ($\omega_b \sim 1$ rad/s)
3. **드리프트** 지구 주위 ($\omega_d \sim 10^{-3}$ rad/s):
   - 양성자는 서쪽으로 드리프트
   - 전자는 동쪽으로 드리프트

드리프트는 지자기 폭풍 동안 **고리 전류**를 만듭니다.

### 7.4 손실 메커니즘

입자는 다음을 통해 손실됩니다:
1. **대기 산란**: 낮은 고도에서 중성 원자와의 충돌
2. **전하 교환**: 양성자가 전자를 포획하여, 중성이 되어, 탈출
3. **파동-입자 상호작용**: VLF/ELF 파동이 입자를 손실 원뿔로 산란
4. **자기권 경계 그림자**: 자기력선이 자기권 경계와 교차하여, 입자가 탈출

수명은 며칠(외부대 전자)에서 몇 년(내부대 양성자) 범위입니다.

## 8. Python 구현

### 8.1 자기 거울 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
m_p = 1.67e-27  # kg
q_p = 1.6e-19   # C

def mirror_field(z, B0=1e-3, L=10.0, R=5.0):
    """
    Parabolic mirror field: B(z) = B0 * (1 + (z/L)^2 * (R-1))

    B0: midplane field (T)
    L: mirror length scale (m)
    R: mirror ratio B_max/B_min
    """
    # 포물선 프로파일은 해석적으로 단순하면서도 z=0에서 B_min부터 |z|=L에서 B_max까지
    # 매끄럽고 단조롭게 증가하는 것을 제공합니다. 이는 실제 Helmholtz 코일
    # 거울 장치의 축상 자기장을 잘 근사합니다.
    return B0 * (1 + (z / L)**2 * (R - 1))

def mirror_grad(z, B0=1e-3, L=10.0, R=5.0):
    """Gradient of mirror field dB/dz"""
    return B0 * 2 * (R - 1) * z / L**2

def equations_of_motion_mirror(t, state, q, m, B0, L, R):
    """
    Equations of motion in mirror field
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state

    # Magnetic field (z-component only, aligned with z-axis)
    B = mirror_field(z, B0, L, R)
    Bx, By, Bz = 0, 0, B

    # Lorentz force
    v = np.array([vx, vy, vz])
    B_vec = np.array([Bx, By, Bz])
    F_lorentz = q * np.cross(v, B_vec)

    # Mirror force (adiabatic approximation)
    # 저장된 상수를 사용하지 않고 현재 vx,vy로부터 매 스텝에서 μ를 재계산합니다.
    # 이를 통해 μ가 수치적으로 얼마나 잘 보존되는지 모니터링할 수 있으며,
    # 평행 힘에 대한 유도 중심 근사는 여전히 사용합니다.
    # mu = m * (vx^2 + vy^2) / (2 * B)
    v_perp_sq = vx**2 + vy**2
    mu = m * v_perp_sq / (2 * B)
    # F_mirror = -μ ∂B/∂z: 이는 단열 불변량(adiabatic invariant)으로부터 유도된
    # 기울기 힘입니다. B가 증가하는 방향을 반대하기 때문에 음수입니다.
    # 포획된 입자는 평행 운동 에너지가 모두 수직 에너지로 변환되기 전에 반사됩니다.
    F_mirror = -mu * mirror_grad(z, B0, L, R)

    # Total force (Lorentz + mirror force in z-direction)
    Fx, Fy, Fz_lorentz = F_lorentz / m
    Fz = Fz_lorentz + F_mirror / m

    return np.array([vx, vy, vz, Fx, Fy, Fz])

def rk4_step(f, t, y, dt, *args):
    """4th-order Runge-Kutta"""
    k1 = f(t, y, *args)
    k2 = f(t + dt/2, y + dt*k1/2, *args)
    k3 = f(t + dt/2, y + dt*k2/2, *args)
    k4 = f(t + dt, y + dt*k3, *args)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def simulate_mirror(v_perp, v_para, B0=1e-3, L=10.0, R=5.0,
                   duration=1.0, dt=1e-4):
    """
    Simulate particle in magnetic mirror

    v_perp: initial perpendicular velocity (m/s)
    v_para: initial parallel velocity (m/s)
    """
    # Initial conditions
    x0, y0, z0 = 0.0, 0.0, 0.0
    vx0, vy0, vz0 = v_perp, 0.0, v_para
    state = np.array([x0, y0, z0, vx0, vy0, vz0])

    # Time array
    num_steps = int(duration / dt)
    times = np.linspace(0, duration, num_steps)

    # Storage
    trajectory = np.zeros((num_steps, 6))
    trajectory[0] = state

    # Integration
    for i in range(1, num_steps):
        state = rk4_step(equations_of_motion_mirror, times[i-1], state,
                        dt, q_p, m_p, B0, L, R)
        trajectory[i] = state

    return times, trajectory

# Calculate loss cone angle
B0 = 1e-3  # Tesla
L = 10.0   # meters
R = 5.0    # mirror ratio

# 손실 원뿔 공식: sin^2(alpha_c) = B_min/B_max = 1/R.
# 이는 μ 보존으로부터 옵니다: 거울점에서 v_perp = v_total이므로
# (v_perp,0)^2 / v^2 = B_min/B_max. 피치각이 더 작은 입자는
# 반사될 만큼 v_perp가 충분하지 않아 탈출합니다 — "손실 원뿔 내에 있습니다".
alpha_c = np.arcsin(1/np.sqrt(R)) * 180/np.pi  # degrees

print(f"Mirror ratio R = {R}")
print(f"Loss cone angle: {alpha_c:.2f}°")
print(f"Trapping condition: α > {alpha_c:.2f}°\n")

# Simulate trapped and lost particles
v_total = 1e5  # m/s

# Trapped particle (α = 60° > α_c)
alpha_trapped = 60 * np.pi/180
v_perp_trapped = v_total * np.sin(alpha_trapped)
v_para_trapped = v_total * np.cos(alpha_trapped)

print(f"Trapped particle: α = 60°")
print(f"  v_perp = {v_perp_trapped:.2e} m/s")
print(f"  v_para = {v_para_trapped:.2e} m/s")

t_trap, traj_trap = simulate_mirror(v_perp_trapped, v_para_trapped,
                                     B0, L, R, duration=2.0, dt=1e-4)

# Lost particle (α = 20° < α_c)
alpha_lost = 20 * np.pi/180
v_perp_lost = v_total * np.sin(alpha_lost)
v_para_lost = v_total * np.cos(alpha_lost)

print(f"\nLost particle: α = 20°")
print(f"  v_perp = {v_perp_lost:.2e} m/s")
print(f"  v_para = {v_para_lost:.2e} m/s")

t_lost, traj_lost = simulate_mirror(v_perp_lost, v_para_lost,
                                     B0, L, R, duration=1.0, dt=1e-4)

# Plotting
fig = plt.figure(figsize=(16, 10))

# 3D trajectories
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(traj_trap[:, 0], traj_trap[:, 1], traj_trap[:, 2],
        'b-', linewidth=0.5, label='Trapped')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title(f'Trapped Particle (α = 60° > α_c = {alpha_c:.1f}°)')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot(traj_lost[:, 0], traj_lost[:, 1], traj_lost[:, 2],
        'r-', linewidth=0.5, label='Lost')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_zlabel('z (m)')
ax2.set_title(f'Lost Particle (α = 20° < α_c = {alpha_c:.1f}°)')
ax2.legend()
ax2.grid(True)

# Z vs time (bounce motion)
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(t_trap, traj_trap[:, 2], 'b-', linewidth=1, label='Trapped')
ax3.axhline(y=L, color='k', linestyle='--', label=f'Mirror at ±{L} m')
ax3.axhline(y=-L, color='k', linestyle='--')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('z (m)')
ax3.set_title('Bounce Motion')
ax3.legend()
ax3.grid(True)

# Magnetic field along z
z_array = np.linspace(-L*1.5, L*1.5, 1000)
B_array = mirror_field(z_array, B0, L, R)

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(z_array, B_array * 1e3, 'k-', linewidth=2)
ax4.set_xlabel('z (m)')
ax4.set_ylabel('B (mT)')
ax4.set_title('Magnetic Field Profile')
ax4.grid(True)

# Velocity components vs time
ax5 = fig.add_subplot(2, 3, 5)
v_perp_trap = np.sqrt(traj_trap[:, 3]**2 + traj_trap[:, 4]**2)
v_para_trap = traj_trap[:, 5]
ax5.plot(t_trap, v_perp_trap/1e3, 'b-', linewidth=1, label='v_perp')
ax5.plot(t_trap, v_para_trap/1e3, 'r-', linewidth=1, label='v_para')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Velocity (km/s)')
ax5.set_title('Velocity Components (Trapped)')
ax5.legend()
ax5.grid(True)

# Adiabatic invariant μ
# 단열 불변량(adiabatic invariant) μ — μ(0)으로 정규화하여 플롯함으로써
# 1로부터의 편차가 직접 분수적 수치 오차를 보여줍니다. 잘 해상된 시뮬레이션은
# 바운스 내내 μ/μ(0)을 ~1% 이내로 유지합니다; 더 큰 드리프트는 dt가 너무 크거나
# 기울기 척도가 단열 근사가 성립하기에 너무 가파름을 나타냅니다.
B_traj_trap = mirror_field(traj_trap[:, 2], B0, L, R)
mu_trap = m_p * v_perp_trap**2 / (2 * B_traj_trap)

ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(t_trap, mu_trap / mu_trap[0], 'b-', linewidth=1)
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('μ(t) / μ(0)')
ax6.set_title('Conservation of Magnetic Moment μ')
ax6.grid(True)
ax6.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='μ = const')
ax6.legend()

plt.tight_layout()
plt.savefig('magnetic_mirror.png', dpi=150)
print("\nSaved: magnetic_mirror.png")

# Calculate bounce period
# v_z (평행 속도)의 부호 변화를 감지하여 바운스 주기를 계산합니다.
# 부호 변화는 v_z가 0을 통과했음을 의미합니다 — 입자가 거울점에서 방향을 바꿨습니다.
# 연속적인 반전 사이의 시간이 반(half) 바운스 주기입니다.
z_positions = traj_trap[:, 2]
# Find turning points (where v_z changes sign)
v_z = traj_trap[:, 5]
sign_changes = np.diff(np.sign(v_z))
bounce_indices = np.where(sign_changes != 0)[0]

if len(bounce_indices) >= 2:
    tau_bounce = t_trap[bounce_indices[1]] - t_trap[bounce_indices[0]]
    omega_bounce = 2 * np.pi / tau_bounce
    print(f"\nBounce period: {tau_bounce:.4f} s")
    print(f"Bounce frequency: {omega_bounce:.2f} rad/s")

    # Compare with gyrofrequency
    omega_c = q_p * B0 / m_p
    print(f"Gyrofrequency: {omega_c:.2e} rad/s")
    print(f"Ratio ω_c/ω_b: {omega_c/omega_bounce:.2e}")
```

### 8.2 손실 원뿔 시각화

```python
# Loss cone in velocity space
fig = plt.figure(figsize=(12, 5))

# 3D velocity space
ax1 = fig.add_subplot(121, projection='3d')

# Generate loss cone
theta = np.linspace(0, 2*np.pi, 50)
alpha_cone = np.linspace(0, alpha_c * np.pi/180, 20)
theta_grid, alpha_grid = np.meshgrid(theta, alpha_cone)

v = v_total
vx_cone = v * np.sin(alpha_grid) * np.cos(theta_grid)
vy_cone = v * np.sin(alpha_grid) * np.sin(theta_grid)
vz_cone = v * np.cos(alpha_grid)

ax1.plot_surface(vx_cone/1e3, vy_cone/1e3, vz_cone/1e3,
                alpha=0.3, color='red', label='Loss cone')

# Generate trapped region (example particles)
np.random.seed(42)
n_particles = 500
alpha_samples = np.arccos(np.random.uniform(-1, 1, n_particles))
theta_samples = np.random.uniform(0, 2*np.pi, n_particles)

# Separate trapped and lost
trapped_mask = alpha_samples > alpha_c * np.pi/180
lost_mask = ~trapped_mask

vx_trapped = v * np.sin(alpha_samples[trapped_mask]) * np.cos(theta_samples[trapped_mask])
vy_trapped = v * np.sin(alpha_samples[trapped_mask]) * np.sin(theta_samples[trapped_mask])
vz_trapped = v * np.cos(alpha_samples[trapped_mask])

vx_lost = v * np.sin(alpha_samples[lost_mask]) * np.cos(theta_samples[lost_mask])
vy_lost = v * np.sin(alpha_samples[lost_mask]) * np.sin(theta_samples[lost_mask])
vz_lost = v * np.cos(alpha_samples[lost_mask])

ax1.scatter(vx_trapped/1e3, vy_trapped/1e3, vz_trapped/1e3,
           c='blue', s=1, alpha=0.5, label='Trapped')
ax1.scatter(vx_lost/1e3, vy_lost/1e3, vz_lost/1e3,
           c='red', s=2, alpha=0.8, label='Lost')

ax1.set_xlabel('vx (km/s)')
ax1.set_ylabel('vy (km/s)')
ax1.set_zlabel('vz (km/s)')
ax1.set_title(f'Loss Cone in Velocity Space\nα_c = {alpha_c:.1f}°, R = {R}')
ax1.legend()

# 2D pitch angle distribution
ax2 = fig.add_subplot(122)
alpha_deg = np.linspace(0, 180, 1000)
alpha_rad = alpha_deg * np.pi / 180

# Distribution function (isotropic) proportional to sin(α)
f_alpha = np.sin(alpha_rad)

ax2.fill_between(alpha_deg, 0, f_alpha, where=(alpha_deg < alpha_c),
                color='red', alpha=0.5, label='Loss cone (escaping)')
ax2.fill_between(alpha_deg, 0, f_alpha, where=(alpha_deg >= alpha_c),
                color='blue', alpha=0.5, label='Trapped')
ax2.plot(alpha_deg, f_alpha, 'k-', linewidth=2)
ax2.axvline(x=alpha_c, color='red', linestyle='--', linewidth=2,
           label=f'α_c = {alpha_c:.1f}°')

ax2.set_xlabel('Pitch Angle α (degrees)', fontsize=12)
ax2.set_ylabel('f(α) ∝ sin(α)', fontsize=12)
ax2.set_title('Pitch Angle Distribution', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_cone.png', dpi=150)
print("Saved: loss_cone.png")

# Calculate fraction in loss cone
solid_angle_loss = 2 * np.pi * (1 - np.cos(alpha_c * np.pi/180))
solid_angle_total = 4 * np.pi
frac_loss = solid_angle_loss / solid_angle_total

print(f"\nLoss cone solid angle: {solid_angle_loss:.4f} sr")
print(f"Fraction in loss cone: {frac_loss:.4f} = {frac_loss*100:.2f}%")
print(f"Fraction trapped: {1-frac_loss:.4f} = {(1-frac_loss)*100:.2f}%")
```

### 8.3 바나나 궤도 시뮬레이션 (단순화된 토카막)

```python
def tokamak_field(R, Z, R0=2.0, a=0.5, B0=3.0, q=2.0):
    """
    Simplified tokamak field

    R, Z: cylindrical coordinates (m)
    R0: major radius (m)
    a: minor radius (m)
    B0: field at magnetic axis (T)
    q: safety factor
    """
    # 환형 장(toroidal field)은 직선 중심 솔레노이드 코일에 의해 생성되기 때문에
    # 1/R로 감소합니다; Ampere의 법칙에 의해 B_phi * 2πR = const → B_phi ∝ 1/R.
    # 이 1/R 변화는 바나나 궤도 드리프트를 일으키는 기울기와 곡률을 모두 만듭니다.
    B_phi = B0 * R0 / R

    # 폴로이달 장(poloidal field) 모델: 플라즈마 내부에서 선형(균일 전류 밀도),
    # 외부에서는 쌍극자 형태. 이를 통해 안전 인자(safety factor) q = rB_phi/(R0 B_theta)가
    # 내부에서 일정하게 됩니다 — 완전한 Grad-Shafranov를 피한 단순화입니다.
    r = np.sqrt((R - R0)**2 + Z**2)
    if r < a:
        B_theta = B0 * r / (q * R0)
    else:
        B_theta = B0 * a**2 / (q * R0 * r)

    # 폴로이달 장을 국소 폴로이달 각도를 사용하여 원통 좌표(R,Z) 성분으로 변환합니다.
    # 부호 규약은 B_theta가 자기 축(magnetic axis) 주위를 순환하도록 합니다.
    B_R = -B_theta * Z / r if r > 1e-6 else 0
    B_Z = B_theta * (R - R0) / r if r > 1e-6 else 0

    return B_R, B_Z, B_phi

def tokamak_magnitude(R, Z, R0=2.0, a=0.5, B0=3.0, q=2.0):
    """Total field magnitude"""
    B_R, B_Z, B_phi = tokamak_field(R, Z, R0, a, B0, q)
    return np.sqrt(B_R**2 + B_Z**2 + B_phi**2)

# Plot tokamak field magnitude
R_grid = np.linspace(1.0, 3.0, 100)
Z_grid = np.linspace(-1.0, 1.0, 100)
R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid)

B_mag = np.zeros_like(R_mesh)
for i in range(len(Z_grid)):
    for j in range(len(R_grid)):
        B_mag[i, j] = tokamak_magnitude(R_mesh[i, j], Z_mesh[i, j])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Field contours
contour = ax1.contourf(R_mesh, Z_mesh, B_mag, levels=20, cmap='viridis')
ax1.contour(R_mesh, Z_mesh, B_mag, levels=10, colors='white',
           linewidths=0.5, alpha=0.5)
plt.colorbar(contour, ax=ax1, label='B (T)')

# Mark regions
R0, a = 2.0, 0.5
circle = plt.Circle((R0, 0), a, fill=False, color='red',
                    linewidth=2, label='Last closed flux surface')
ax1.add_patch(circle)
ax1.plot([R0], [0], 'r*', markersize=15, label='Magnetic axis')

ax1.set_xlabel('R (m)', fontsize=12)
ax1.set_ylabel('Z (m)', fontsize=12)
ax1.set_title('Tokamak Magnetic Field |B|', fontsize=14, fontweight='bold')
ax1.set_aspect('equal')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Field along midplane
R_midplane = np.linspace(1.0, 3.0, 200)
B_midplane = [tokamak_magnitude(R, 0) for R in R_midplane]

ax2.plot(R_midplane, B_midplane, 'b-', linewidth=2)
ax2.axvline(x=R0-a, color='r', linestyle='--', label='Inboard edge')
ax2.axvline(x=R0+a, color='r', linestyle='--', label='Outboard edge')
ax2.axvline(x=R0, color='g', linestyle='--', label='Magnetic axis')

ax2.set_xlabel('R (m)', fontsize=12)
ax2.set_ylabel('B (T)', fontsize=12)
ax2.set_title('Field Along Midplane (Z=0)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('tokamak_field.png', dpi=150)
print("Saved: tokamak_field.png")

# 소 ε 근사 f_trap ≈ sqrt(2ε)를 사용하여 포획 비율을 계산합니다.
# 이 추정은 토카막 거울에 대한 손실 원뿔 조건에서 옵니다: sin^2(alpha) < 1 - B_out/B_in ≈ 2ε인
# 입자가 포획됩니다. ε ~ 0.3(전형적인 종횡비)에 대해, 약 77%의 입자가 바나나 궤도에
# 있습니다 — 신고전적 수송(neoclassical transport)에서 지배적인 효과입니다.
epsilon = a / R0
f_trapped = np.sqrt(2 * epsilon)
print(f"\nTokamak parameters:")
print(f"  Major radius R0 = {R0} m")
print(f"  Minor radius a = {a} m")
print(f"  Inverse aspect ratio ε = {epsilon:.3f}")
print(f"  Trapped fraction ≈ sqrt(2ε) = {f_trapped:.3f} = {f_trapped*100:.1f}%")

# Banana width estimate
T_keV = 10  # keV
T_J = T_keV * 1e3 * q_p
v_th_i = np.sqrt(2 * T_J / m_p)
rho_i = m_p * v_th_i / (q_p * B0)
q_safety = 2.0
banana_width = q_safety * rho_i * np.sqrt(epsilon)

print(f"\nBanana orbit:")
print(f"  Ion temperature T_i = {T_keV} keV")
print(f"  Thermal velocity v_th = {v_th_i:.2e} m/s")
print(f"  Ion Larmor radius ρ_i = {rho_i*1e3:.2f} mm")
print(f"  Banana width Δr_b ≈ {banana_width*1e3:.2f} mm")
```

## 요약

이 레슨에서, 우리는 자기 거울과 단열 불변량을 탐구했습니다:

1. **자기 거울 힘**: $F_\parallel = -\mu\nabla_\parallel B$는 입자를 고장 영역으로부터 반사시켜, 자기 보틀에서 구속을 가능하게 합니다.

2. **첫 번째 단열 불변량 μ**: 자기 모멘트 $\mu = mv_\perp^2/(2B)$는 장이 자이로주기에 비해 천천히 변할 때 보존됩니다. 이것은 자이로궤도를 통과하는 자속의 보존에 해당합니다.

3. **손실 원뿔**: 피치각 $\alpha < \alpha_c = \arcsin(1/\sqrt{R})$인 입자는 거울 목을 통해 탈출합니다. 큰 $R$에 대해 손실 원뿔에 있는 비율은 $\sim 1/(4R)$입니다.

4. **바운스 운동**: 포획된 입자는 주파수 $\omega_b \sim v_\parallel/L$로 거울점 사이를 진동하며, 자이로주파수보다 훨씬 느립니다.

5. **불변량의 계층**:
   - $\mu$: 자이로운동 (가장 빠름)
   - $J = \oint mv_\parallel ds$: 바운스 (중간)
   - $\Phi = \oint \mathbf{A}\cdot d\mathbf{l}$: 드리프트 (가장 느림)

6. **토카막 궤도**: $1/R$ 변화는 포획된 (바나나) 입자와 통과 입자를 만듭니다. 포획된 비율 $\sim\sqrt{\epsilon}$, 여기서 $\epsilon = a/R_0$.

7. **Van Allen 복사대**: 지구의 쌍극자 장은 자연 자기 보틀을 형성하여, 우주로부터 전하를 띤 입자를 포획합니다.

단열 불변량을 이해하는 것은 다음에 중요합니다:
- 자기 구속 장치 설계
- 입자 수송 예측
- 우주 플라즈마 동역학 분석

## 연습 문제

### 문제 1: 거울 구속 시간

단순 자기 거울이 $B_{\text{min}} = 0.2$ T, $B_{\text{max}} = 1.0$ T, 길이 $L = 5$ m를 가집니다. 플라즈마 밀도는 $n = 10^{18}$ m$^{-3}$이고 온도는 $T = 100$ eV입니다. 충돌 주파수는 $\nu_c = 10^4$ s$^{-1}$입니다.

(a) 거울 비율 $R$과 손실 원뿔 각도 $\alpha_c$를 계산하세요.

(b) 손실 원뿔에 있는 입자의 비율을 추정하세요 (등방성 분포 가정).

(c) 구속 시간 $\tau_{\text{conf}} \sim (R/\nu_c)(1/f_{\text{loss}})$를 계산하세요.

(d) 충돌 시간 $\tau_c = 1/\nu_c$와 비교하세요. 이것이 잘 구속된 플라즈마인가요?

---

### 문제 2: 천천히 변하는 장에서 μ의 보존

초기 에너지 $W = 1$ keV이고 피치각 $\alpha_0 = 45°$인 양성자가 거리 $L = 10$ m에 걸쳐 $B_0 = 0.1$ T에서 $B_1 = 0.5$ T로 증가하는 자기장에서 움직입니다.

(a) 초기 자기 모멘트 $\mu$를 계산하세요.

(b) $\mu$가 보존된다고 가정하고, 최종 수직 및 평행 속도를 구하세요.

(c) 최종 피치각 $\alpha_1$은 무엇인가요?

(d) 총 에너지가 보존됨을 확인하세요.

(e) 단열 조건을 확인하세요: $r_L |\nabla B|/B \ll 1$인가요?

---

### 문제 3: 포물선 거울에서의 바운스 주파수

$B(z) = B_0(1 + z^2/L^2)$인 포물선 거울에서, 여기서 $B_0 = 0.5$ T이고 $L = 10$ m, 중수소 핵이 중앙면($z = 0$)에서 에너지 $W = 10$ keV이고 피치각 $\alpha_0 = 60°$를 가집니다.

(a) $v_\parallel = 0$인 거울점 $z_{\text{mirror}}$를 구하세요.

(b) $\tau_b \approx 4z_{\text{mirror}}/\langle v_\parallel\rangle$를 사용하여 바운스 주기 $\tau_b$를 추정하세요, 여기서 $\langle v_\parallel\rangle$는 평균 평행 속도입니다.

(c) 바운스 주파수 $\omega_b = 2\pi/\tau_b$를 계산하세요.

(d) 중앙면에서 자이로주파수 $\omega_c$와 비교하세요. $\omega_c \gg \omega_b$임을 확인하세요.

**힌트**: 에너지 보존 사용: $\frac{1}{2}mv^2\sin^2\alpha_0 = \frac{1}{2}mv^2\frac{B(z_m)}{B_0}$.

---

### 문제 4: 토카막 바나나 폭

$R_0 = 3$ m, $a = 1$ m, $B_0 = 5$ T이고 안전 인자 $q = 2$인 토카막에서, 온도 $T_i = 20$ keV인 중수소 이온에 대한 바나나 폭을 계산하세요.

(a) 역 종횡비 $\epsilon = a/R_0$를 계산하세요.

(b) 이온 Larmor 반경 $\rho_i = m_i v_{th,i}/(eB_0)$를 계산하세요, 여기서 $v_{th,i} = \sqrt{2k_BT_i/m_i}$.

(c) 바나나 폭 $\Delta r_b \approx q\rho_i\sqrt{\epsilon}$를 추정하세요.

(d) 소반경의 어느 비율이 바나나 폭인가요? 이것이 구속에 중요한가요?

(e) 포획된 입자 비율 $f_{\text{trap}} \approx \sqrt{2\epsilon}$를 계산하세요.

---

### 문제 5: 쌍극자 장에서의 두 번째 단열 불변량

전자가 $L = 4$ 지구 반경 ($L$은 $R_E = 6.37\times 10^6$ m 단위의 적도 교차 거리)에서 쌍극자 자기력선에 포획되어 있습니다. 이 선의 자기장은 다음과 같이 변합니다:

$$
B(s) = B_{\text{eq}}\sqrt{1 + 3\sin^2\lambda}/\cos^6\lambda
$$

여기서 $\lambda$는 자기 위도이고 $B_{\text{eq}} = 10^{-6}$ T는 적도 장입니다.

(a) 전자가 에너지 $W = 100$ keV이고 적도 피치각 $\alpha_{\text{eq}} = 45°$를 가집니다. 입자가 반사되는 거울 위도 $\lambda_m$을 구하세요.

(b) 적도에서 거울점까지의 호장 $s_m$을 추정하세요 (작은 $\lambda$에 대해 $s \approx LR_E\lambda$ 사용).

(c) 두 번째 불변량 $J = \oint mv_\parallel ds \approx 4m\langle v_\parallel\rangle s_m$ (근사)를 계산하세요.

(d) 장이 천천히 감소하면 (예: 지자기 폭풍 동안), $J$는 일정하게 유지됩니다. 거울 위도가 어떻게 변하는지 정성적으로 설명하세요.

**힌트**: $\lambda_m$에 대해, $\sin^2\alpha_{\text{eq}} = B_{\text{eq}}/B(\lambda_m)$를 사용하세요.

---

## 내비게이션

- **이전**: [단일 입자 운동 II](./05_Single_Particle_Motion_II.md)
- **다음**: [Vlasov 방정식](./07_Vlasov_Equation.md)
