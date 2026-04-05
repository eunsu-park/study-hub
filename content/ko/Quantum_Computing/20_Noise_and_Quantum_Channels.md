# 레슨 20: 잡음과 양자 채널

[← 이전: 양자 걷기](19_Quantum_Walks.md) | [다음: 양자 화학 →](21_Quantum_Chemistry.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 일반적인 잡음 모델을 설명할 수 있다: 탈분극, 진폭 감쇠, 위상 감쇠, 비트 반전 채널
2. 양자 채널의 Kraus 연산자 표현을 도출하고 적용할 수 있다
3. Choi-Jamiolkowski 동형과 채널 특성화에서의 역할을 설명할 수 있다
4. 미지의 양자 채널을 재구성하기 위한 양자 과정 단층촬영을 수행할 수 있다
5. 랜덤화 벤치마킹과 게이트 세트 단층촬영을 통해 잡음을 특성화할 수 있다
6. 양자 알고리즘에 대한 잡음의 영향과 오류 임계값을 분석할 수 있다
7. Python으로 잡음 모델, Kraus 연산자, 과정 단층촬영을 구현할 수 있다

---

모든 실제 양자 컴퓨터는 잡음이 있습니다. 큐비트는 환경과 상호작용하고, 게이트는 불완전하며, 측정은 오류를 도입합니다. 잡음을 이해하는 것은 양자 알고리즘이 실제 하드웨어에서 의미 있는 결과를 생성할 수 있는지를 결정합니다.

**양자 채널**의 수학적 프레임워크는 잡음 과정에 대한 엄밀하고 완전한 기술을 제공합니다. 양자 채널은 양자 상태를 변환하는 모든 물리적 과정입니다. Kraus 연산자 형식은 컴팩트한 표현을 제공합니다: 모든 양자 채널은 $\rho \mapsto \sum_k E_k \rho E_k^\dagger$로 쓸 수 있으며, Kraus 연산자 $\{E_k\}$는 $\sum_k E_k^\dagger E_k = I$를 만족합니다.

> **비유:** 양자 채널은 양자 정보를 위한 잡음 있는 전화선과 같습니다. 완벽한 회선은 신호를 변경 없이 전송하고(유니터리 채널), 잡음 있는 회선은 신호를 무작위로 뒤섞거나(탈분극), 진폭을 점진적으로 줄이거나(진폭 감쇠), 무작위로 비트를 반전시킬 수 있습니다(비트 반전).

## 목차

1. [양자 잡음 모델](#1-양자-잡음-모델)
2. [Kraus 연산자 표현](#2-kraus-연산자-표현)
3. [일반적인 양자 채널](#3-일반적인-양자-채널)
4. [Choi-Jamiolkowski 동형](#4-choi-jamiolkowski-동형)
5. [양자 과정 단층촬영](#5-양자-과정-단층촬영)
6. [잡음 특성화 방법](#6-잡음-특성화-방법)
7. [양자 알고리즘에서의 잡음](#7-양자-알고리즘에서의-잡음)
8. [오류 임계값과 내결함성](#8-오류-임계값과-내결함성)
9. [Python 구현](#9-python-구현)
10. [연습 문제](#10-연습-문제)

---

## 1. 양자 잡음 모델

### 1.1 개방 양자 시스템

닫힌 양자 시스템은 유니터리하게 진화합니다: $\rho(t) = U(t)\rho(0)U(t)^\dagger$. 그러나 실제 큐비트는 환경과 상호작용하는 **개방 시스템**입니다:

$$|\Psi_{SE}\rangle = |\psi_S\rangle \otimes |0_E\rangle \xrightarrow{U_{SE}} |\Phi_{SE}\rangle$$

시스템 상태는 환경을 추적 제거하여 얻습니다:

$$\rho_S(t) = \text{Tr}_E[U_{SE}(|\psi_S\rangle\langle\psi_S| \otimes |0_E\rangle\langle 0_E|)U_{SE}^\dagger]$$

이 추적 연산이 비유니터리(잡음이 있는) 동역학을 도입하는 것입니다.

### 1.2 잡음 유형

| 잡음 유형 | 물리적 기원 | 큐비트에 대한 영향 | 시간 척도 |
|-----------|-----------|-------------------|----------|
| **이완** ($T_1$) | 환경과의 에너지 교환 | $\|1\rangle \to \|0\rangle$ 붕괴 | $T_1 \sim 50$-$500$ $\mu$s |
| **결맞음깨짐** ($T_2$) | 무작위 위상 요동 | 결맞음 손실 | $T_2 \leq 2T_1$ |
| **게이트 오류** | 불완전한 제어 펄스 | 잘못된 회전 각도/축 | 게이트당: $10^{-4}$-$10^{-2}$ |
| **측정 오류** | 판독 누화, 열적 여기 | 잘못된 비트 값 | $10^{-3}$-$10^{-1}$ |
| **누화(Crosstalk)** | 원하지 않는 큐비트-큐비트 결합 | 상관 오류 | 연결 구조에 따라 다름 |
| **누설(Leakage)** | 계산 부분공간 밖으로 점유 이탈 | 상태가 $\{|0\rangle, |1\rangle\}$를 벗어남 | 다양함 |

### 1.3 결맞은 오류 대 비결맞은 오류

**결맞은 오류**: 게이트의 체계적 과소/과대 회전. 유니터리이며 재교정으로 수정 가능합니다.

**비결맞은 오류**: 양자 채널로 기술되는 무작위, 비가역적 과정. 수정이 더 어렵고 오류 정정의 주요 관심사입니다.

---

## 2. Kraus 연산자 표현

### 2.1 정의

양자 채널 $\mathcal{E}$는 밀도 행렬에 대한 완전 양성, 대각합 보존(CPTP) 맵입니다:

$$\mathcal{E}(\rho) = \sum_{k=0}^{r-1} E_k \rho E_k^\dagger$$

**Kraus 연산자** $\{E_k\}$는 완전성 관계를 만족합니다: $\sum_k E_k^\dagger E_k = I$

### 2.2 물리적 해석

각 Kraus 연산자 $E_k$는 잡음 과정의 가능한 "결과"를 나타냅니다. 채널은 모든 결과에 대해 평균합니다.

---

## 3. 일반적인 양자 채널

### 3.1 비트 반전 채널

확률 $p$로 큐비트가 반전됩니다: $E_0 = \sqrt{1-p}\,I, \quad E_1 = \sqrt{p}\,X$

### 3.2 탈분극 채널

큐비트가 확률 $p$로 최대 혼합 상태로 대체됩니다:
$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

블로흐 구에 대한 효과: 블로흐 벡터를 $(1 - 4p/3)$로 균일하게 수축합니다.

### 3.3 진폭 감쇠 채널

에너지 이완($T_1$ 감쇠)을 모델링합니다:
$$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

블로흐 구가 수축하고 $|0\rangle$ 방향으로 이동합니다.

### 3.4 위상 감쇠 채널

순수 결맞음깨짐을 모델링합니다. $x$, $y$ 블로흐 성분을 $\sqrt{1-\lambda}$로 수축시키고, $z$를 보존합니다.

### 3.5 채널 비교

| 채널 | Kraus 랭크 | 블로흐 구 기하학 | 수축 인자 | 물리적 모델 |
|---------|-----------|----------------------|----------------|---------------|
| 비트 반전 | 2 | 구 $\to$ 타원체 (x 보존) | $r_x \to r_x$, $r_y \to (1{-}2p)r_y$, $r_z \to (1{-}2p)r_z$ | 무작위 X 오류 |
| 위상 반전 | 2 | 구 $\to$ 타원체 (z 보존) | $r_x \to (1{-}2p)r_x$, $r_y \to (1{-}2p)r_y$, $r_z \to r_z$ | 무작위 Z 오류 |
| 탈분극 | 4 | 구 $\to$ 더 작은 구 | 모든 $r_i \to (1{-}4p/3)r_i$ | 무작위 파울리 오류 |
| 진폭 감쇠 | 2 | 구 $\to$ $\|0\rangle$으로 이동한 달걀 모양 | $r_{x,y} \to \sqrt{1{-}\gamma}\,r_{x,y}$, $r_z \to (1{-}\gamma)r_z + \gamma$ | $T_1$ 이완 |
| 위상 감쇠 | 2 | 구 $\to$ 타원체 (z 보존) | $r_{x,y} \to \sqrt{1{-}\lambda}\,r_{x,y}$, $r_z \to r_z$ | $T_2$ 결맞음깨짐 |

---

## 4. Choi-Jamiolkowski 동형

채널 $\mathcal{E}$의 Choi 행렬: $\Lambda_{\mathcal{E}} = (\mathcal{E} \otimes \mathcal{I})(|\Omega\rangle\langle\Omega|)$

Choi 행렬은 채널에 대한 모든 것을 인코딩합니다:
- 완전 양성: $\Lambda \geq 0$
- 대각합 보존: $\text{Tr}_1[\Lambda] = I/d$
- Kraus 연산자: $\Lambda$의 고유벡터에서 추출

---

## 5. 양자 과정 단층촬영

### 5.1 표준 과정 단층촬영

1. 밀도 행렬 공간의 기저를 형성하는 입력 상태 집합을 준비
2. 각 입력 상태에 채널 적용
3. 각 출력 상태에 대해 양자 상태 단층촬영 수행
4. 입출력 데이터로부터 채널 재구성

---

## 6. 잡음 특성화 방법

### 6.1 랜덤화 벤치마킹 (RB)

전체 단층촬영 없이 게이트의 평균 오류율 측정:
1. 증가하는 길이의 무작위 Clifford 시퀀스 생성
2. 각 시퀀스에 역 Clifford 추가
3. 생존 확률 측정 및 피팅: $p(L) = A \cdot r^L + B$
4. 오류율 추출: $\epsilon = (1-r)(1 - 1/d)$

### 6.2 파울리 트월링(Pauli Twirling)

**파울리 트월링(Pauli twirling)**은 임의의 잡음 채널을 파울리 채널(파울리 오류의 확률적 혼합)로 변환하여, 잡음 분석과 오류 정정을 단순화하는 기법입니다.

잡음이 있는 연산 $\mathcal{E}$를 무작위 파울리 게이트로 감쌉니다. 게이트 전에 균일 무작위 파울리 $P_i$를 적용하고, 게이트 후에 $P_i^\dagger$를 적용합니다. 모든 $4^n$개 파울리 연산자에 대해 평균하면:

$$\mathcal{E}_{\text{twirled}}(\rho) = \frac{1}{4^n}\sum_{i} P_i \mathcal{E}(P_i \rho P_i^\dagger) P_i^\dagger$$

트월링된 채널 $\mathcal{E}_{\text{twirled}}$는 항상 **파울리 채널**입니다: $\mathcal{E}_{\text{twirled}}(\rho) = \sum_j p_j P_j \rho P_j$ (여기서 $p_j \geq 0$, $\sum_j p_j = 1$). 중요한 점은 평균 충실도가 보존된다는 것입니다 — 트월링은 평균적으로 잡음을 악화시키지 않고 그 구조만 단순화합니다.

파울리 트월링은 양자 오류 정정 분석(파울리 채널이 디코딩하기 훨씬 쉬움)과 확률적 오류 취소 같은 오류 완화 기법에 널리 사용됩니다. 실제로 무작위 파울리 게이트는 무시할 수 있는 오버헤드로 기존 회로에 컴파일할 수 있습니다.

---

## 7. 양자 알고리즘에서의 잡음

$G$개 게이트, 각 오류율 $\epsilon$인 회로의 전체 충실도: $F \approx (1 - \epsilon)^G \approx e^{-\epsilon G}$

현재 오류율($\epsilon \sim 10^{-3}$)로는 약 1000개 게이트로 제한됩니다.

---

## 8. 오류 임계값과 내결함성

물리적 오류율 $p$가 임계값 $p_{\text{th}}$ 이하이면, 임의로 긴 양자 계산이 가능합니다. 표면 코드 임계값은 약 1%입니다.

---

## 9. Python 구현

```python
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def apply_channel(rho, kraus_ops):
    """Kraus 연산자로 정의된 양자 채널을 밀도 행렬에 적용합니다."""
    rho_out = np.zeros_like(rho)
    for E in kraus_ops:
        rho_out += E @ rho @ E.conj().T
    return rho_out

def depolarizing_channel(p):
    """탈분극 채널."""
    return [np.sqrt(1-p)*I, np.sqrt(p/3)*X, np.sqrt(p/3)*Y, np.sqrt(p/3)*Z]

def amplitude_damping_channel(gamma):
    """진폭 감쇠 채널."""
    E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [E0, E1]

def density_to_bloch(rho):
    """밀도 행렬을 블로흐 벡터로 변환합니다."""
    rx = 2 * np.real(rho[0, 1])
    ry = 2 * np.imag(rho[1, 0])
    rz = np.real(rho[0, 0] - rho[1, 1])
    return np.array([rx, ry, rz])

print("=" * 60)
print("양자 채널이 블로흐 벡터에 미치는 영향")
print("=" * 60)

rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
bloch_in = density_to_bloch(rho_plus)
print(f"\n입력: |+>, 블로흐 = ({bloch_in[0]:.2f}, {bloch_in[1]:.2f}, {bloch_in[2]:.2f})")

for name, kraus in [('탈분극 p=0.1', depolarizing_channel(0.1)),
                     ('진폭감쇠 g=0.3', amplitude_damping_channel(0.3))]:
    rho_out = apply_channel(rho_plus, kraus)
    bloch_out = density_to_bloch(rho_out)
    print(f"  {name}: ({bloch_out[0]:.4f}, {bloch_out[1]:.4f}, {bloch_out[2]:.4f})")
```

---

## 10. 연습 문제

### 연습 1: 채널 합성
두 탈분극 채널의 합성에 대한 Kraus 연산자를 계산하고, 결과가 단일 탈분극 채널과 동등한지 확인하세요.

### 연습 2: Choi 행렬 분석
탈분극 채널의 Choi 행렬 고유값을 $p$의 함수로 그리세요.

### 연습 3: 잡음이 있는 과정 단층촬영
측정 잡음을 추가하고 최대 우도 추정을 구현하세요.

### 연습 4: 랜덤화 벤치마킹 확장
인터리브드 RB를 구현하여 특정 게이트의 오류율을 측정하세요.

### 연습 5: 잡음 인식 알고리즘 설계
3-큐비트 VQE에서 영잡음 외삽법의 효과를 분석하세요.

---

[← 이전: 양자 걷기](19_Quantum_Walks.md) | [다음: 양자 화학 →](21_Quantum_Chemistry.md)
