# 레슨 18: 양자 시뮬레이션

[← 이전: 양자 컴퓨팅 현황과 미래](16_Landscape_and_Future.md) | [다음: 양자 걷기 →](19_Quantum_Walks.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 해밀토니안 시뮬레이션의 개념과 양자 컴퓨터가 자연스러운 시뮬레이터인 이유를 설명할 수 있다
2. 시간 진화를 위한 Trotter-Suzuki 분해를 도출하고 구현할 수 있다
3. 다양한 차수의 곱 공식에 대한 오차 한계를 분석할 수 있다
4. 기본 수준을 넘어선 변분 양자 고유값 풀이(VQE) 발전을 설명할 수 있다
5. 에너지 고유값 추출을 위한 양자 위상 추정(QPE)을 적용할 수 있다
6. 곱 공식, LCU, 큐비트화 등 다양한 시뮬레이션 접근법을 비교할 수 있다
7. Python/NumPy로 해밀토니안 시뮬레이션 알고리즘을 구현할 수 있다

---

양자 시뮬레이션은 양자 컴퓨팅의 가장 유망한 근기 응용 분야입니다. Richard Feynman의 1982년 원래 통찰이 바로 이것이었습니다: 고전 컴퓨터로 양자 시스템을 시뮬레이션하는 것은 지수적으로 어렵지만, 양자 컴퓨터는 이를 자연스럽게 수행할 수 있습니다. $n$개의 상호작용하는 양자 입자 시스템은 $2^n$ 차원의 상태 공간을 가지며, 이는 적당한 크기(예: $n \geq 40$)에서도 고전적 시뮬레이션을 불가능하게 만듭니다. 그러나 $n$ 큐비트의 양자 컴퓨터는 이 상태를 직접 표현할 수 있습니다.

양자 시뮬레이션의 핵심 문제는: 양자 시스템을 기술하는 해밀토니안 $H$가 주어졌을 때, 시간 진화 연산자 $U(t) = e^{-iHt}$를 계산하여 초기 상태 $|\psi_0\rangle$에 적용하는 것입니다. 이 레슨은 Trotter-Suzuki 곱 공식부터 유니터리의 선형 결합(LCU) 및 양자 신호 처리와 같은 현대 기법에 이르기까지 핵심 알고리즘을 다룹니다.

> **비유:** 양자 시스템을 고전적으로 시뮬레이션하는 것은 체스 게임에서 가능한 모든 수를 일일이 기록하려는 것과 같습니다 — 조합 폭발이 압도적입니다. 양자 시뮬레이터는 반대로 게임을 직접 플레이하며, 자신의 양자적 본성을 사용하여 연구 대상 시스템을 반영합니다.

## 목차

1. [왜 양자 시뮬레이션인가?](#1-왜-양자-시뮬레이션인가)
2. [해밀토니안 시뮬레이션 문제](#2-해밀토니안-시뮬레이션-문제)
3. [Trotter-Suzuki 분해](#3-trotter-suzuki-분해)
4. [고차 곱 공식](#4-고차-곱-공식)
5. [시뮬레이션을 위한 양자 위상 추정](#5-시뮬레이션을-위한-양자-위상-추정)
6. [VQE 발전](#6-vqe-발전)
7. [곱 공식을 넘어서](#7-곱-공식을-넘어서)
8. [응용](#8-응용)
9. [Python 구현](#9-python-구현)
10. [연습 문제](#10-연습-문제)

---

## 1. 왜 양자 시뮬레이션인가?

### 1.1 Feynman의 비전

1982년, Richard Feynman은 $n$개 입자의 양자 시스템을 고전 컴퓨터에서 시뮬레이션하려면 $2^n$개의 복소 진폭을 저장하고 조작해야 한다고 관찰했습니다. $n = 50$ 입자의 경우, $2^{50} \approx 10^{15}$개의 복소수가 필요합니다 — 약 16 페타바이트의 메모리입니다. 50 큐비트의 양자 컴퓨터는 이 상태를 양자 레지스터에서 자연스럽게 표현합니다.

### 1.2 고전적 난해성

$n$개의 스핀-1/2 입자가 쌍별 상호작용하는 시스템을 고려합니다:

$$H = \sum_{i<j} J_{ij} \vec{\sigma}_i \cdot \vec{\sigma}_j + \sum_i h_i \sigma_i^z$$

힐베르트 공간 차원은 $2^n$입니다. 고전적 정확 대각화는 $O(2^{3n})$ 시간과 $O(2^{2n})$ 메모리가 필요합니다. 근사적 고전 방법(DMRG, 텐서 네트워크)도 특정 시스템 — 특히 강한 얽힘, 좌절된 상호작용, 고차원 — 에서는 실패합니다.

### 1.3 양자 시뮬레이션 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| **디지털** | $e^{-iHt}$를 근사하는 게이트 기반 양자 회로 | Trotter-Suzuki 시뮬레이션 |
| **아날로그** | 대상을 모방하는 해밀토니안 직접 공학 | 냉각 원자 격자 |
| **변분** | 바닥 상태를 근사하도록 최적화된 매개변수 회로 | VQE, QAOA |
| **하이브리드** | 양자-고전 자원 결합 | 양자-고전 피드백 루프 |

---

## 2. 해밀토니안 시뮬레이션 문제

### 2.1 문제 정의

**주어진 것**: $n$ 큐비트에 작용하는 해밀토니안 $H$와 시간 $t$

**목표**: 양자 컴퓨터에서 유니터리 $U(t) = e^{-iHt}$ 구현

**도전**: $H$는 일반적으로 비가환 항의 합입니다: $H = \sum_{k=1}^{L} H_k$, 그리고 $[A, B] \neq 0$일 때 $e^{-i(A+B)t} \neq e^{-iAt}e^{-iBt}$입니다.

### 2.2 해밀토니안 분해

대부분의 물리적 해밀토니안은 국소적 항의 합으로 자연스럽게 분해됩니다:

$$H = \sum_{k=1}^{L} \alpha_k P_k$$

여기서 각 $P_k$는 파울리 연산자의 텐서 곱(파울리 문자열)이고 $\alpha_k$는 실수 계수입니다.

### 2.3 시뮬레이션 복잡도

| 방법 | 게이트 복잡도 |
|------|-------------|
| 1차 Trotter | $O(L^2 t^2 / \epsilon)$ |
| 2차 Trotter | $O(L^{5/2} t^{3/2} / \epsilon^{1/2})$ |
| 유니터리의 선형 결합 | $O(L\alpha t \cdot \text{polylog}(1/\epsilon))$ |
| 양자 신호 처리 | $O(\alpha t + \log(1/\epsilon))$ |

---

## 3. Trotter-Suzuki 분해

### 3.1 1차 Trotter 공식

Lie-Trotter 곱 공식은 합의 지수를 근사합니다:

$$e^{-i(H_1 + H_2 + \cdots + H_L)t} \approx \left(\prod_{k=1}^{L} e^{-iH_k t/r}\right)^r$$

여기서 $r$은 Trotter 단계 수입니다. 단계당 오차:

$$\left\|e^{-i(A+B)\delta t} - e^{-iA\delta t}e^{-iB\delta t}\right\| \leq \frac{(\delta t)^2}{2}\|[A, B]\|$$

### 3.2 Trotterization이 작동하는 이유

핵심 통찰은 각 Trotter 단계가 $O(\delta t^2)$의 오차를 도입하지만, 오차가 단순히 선형으로 누적되지 않는다는 것입니다. 많은 물리적 시스템에서 해밀토니안의 구조 때문에 오차 상쇄가 발생하여 최악의 경우 한계보다 훨씬 나은 성능을 보입니다.

### 3.3 회로 구성

파울리 문자열 $P = \sigma_{i_1} \otimes \sigma_{i_2} \otimes \cdots \otimes \sigma_{i_m}$에 대해 $e^{-i\alpha P t}$를 구현하려면:

1. **기저 회전**: 각 큐비트를 Z 기저로 회전하는 단일 큐비트 게이트 적용
2. **패리티 계산**: CNOT 사다리를 적용하여 패리티를 보조 큐비트에 계산
3. **위상 회전**: 대상 큐비트에 $R_z(2\alpha t)$ 적용
4. **역계산**: CNOT 사다리와 기저 회전을 역순으로 적용

파울리 문자열당 $O(m)$개의 CNOT 게이트와 $O(m)$개의 단일 큐비트 게이트가 필요합니다.

### 3.4 $e^{-i\alpha Z_i Z_j t}$ 회로

다음 회로는 2-큐비트 ZZ 상호작용 항 $e^{-i\alpha Z_i Z_j t}$를 구현합니다:

```
  q_i ──●────────────────●──
        |                |
  q_j ──⊕── Rz(2αt) ────⊕──

  Step-by-step:
  1. CNOT(q_i → q_j): computes parity of q_i, q_j into q_j
  2. Rz(2αt) on q_j:   applies phase based on parity
  3. CNOT(q_i → q_j): uncomputes the parity (restores q_j)

  Net effect: |ab⟩ → e^{-iαt(-1)^(a⊕b)} |ab⟩
            = e^{-iαt·Z_i·Z_j} |ab⟩
```

이 패턴은 더 긴 파울리 문자열 $Z_i Z_j Z_k \cdots$로 일반화됩니다: CNOT 사다리를 확장하여 다중 큐비트 패리티를 계산하고, $R_z$를 적용한 다음, 사다리를 역순으로 적용합니다.

### 3.5 예제: 이징 모형

가로 자기장 이징 모형(transverse-field Ising model):

$$H = -J\sum_{i=1}^{n-1} Z_i Z_{i+1} - h\sum_{i=1}^{n} X_i$$

각 Trotter 단계는 다음을 구현합니다:

$$\prod_{i=1}^{n-1} e^{iJ\delta t Z_i Z_{i+1}} \cdot \prod_{i=1}^{n} e^{ih\delta t X_i}$$

$ZZ$ 항은 CNOT + $R_z$ + CNOT 패턴이 필요하고, $X$ 항은 단순한 $R_x$ 회전입니다.

---

## 4. 고차 곱 공식

### 4.1 2차(대칭화) Trotter

Suzuki-Trotter 2차 공식은 곱을 대칭화합니다:

$$S_2(\delta t) = \prod_{k=1}^{L} e^{-iH_k \delta t/2} \cdot \prod_{k=L}^{1} e^{-iH_k \delta t/2}$$

단계당 오차가 $O(\delta t^3)$로 개선됩니다.

### 4.2 랜덤화된 곱 공식

최근 개발된 **랜덤화 Trotter** (qDRIFT)은 개별 해밀토니안 항을 계수 크기에 비례한 확률로 무작위 추출합니다. 오차는 $O(\lambda^2 t^2/N)$으로, 항의 수 $L$과 무관합니다.

---

## 5. 시뮬레이션을 위한 양자 위상 추정

### 5.1 QPE를 통한 에너지 추정

해밀토니안 시뮬레이션을 위해, $U = e^{-iHt}$로 설정합니다. $|\psi\rangle$가 에너지 $E$인 $H$의 고유 상태라면:

$$e^{-iHt}|\psi\rangle = e^{-iEt}|\psi\rangle$$

QPE는 $\phi = Et/(2\pi)$를 추출하여 $E = 2\pi\phi/t$를 제공합니다.

### 5.2 QPE vs. VQE

| 측면 | QPE | VQE |
|------|-----|-----|
| 회로 깊이 | 깊음 | 얕음 |
| 정밀도 | 체계적 ($\Delta E \sim 2^{-m}$) | 최적화에 의해 제한 |
| 잡음 허용도 | 오류 정정 필요 | NISQ 호환 |
| 초기 상태 | 좋은 중첩 필요 | 처음부터 최적화 |

---

## 6. VQE 발전

### 6.1 적응적 VQE (ADAPT-VQE)

최적화 전에 안자츠 구조를 고정하는 대신, ADAPT-VQE는 회로를 반복적으로 확장합니다:

1. 참조 상태 $|\psi_0\rangle$에서 시작
2. 연산자 풀의 각 연산자에 대해 기울기 계산
3. 가장 큰 기울기를 가진 연산자를 안자츠에 추가
4. 모든 매개변수를 재최적화
5. 기울기 노름이 임계값 이하로 떨어질 때까지 반복

### 6.2 오류 완화 기법

- **영잡음 외삽법 (ZNE)**: 여러 잡음 수준에서 실행하고 영잡음으로 외삽
- **확률적 오류 상쇄 (PEC)**: 잡음 채널의 역을 구현 가능한 연산의 준확률 분포로 분해
- **대칭성 검증**: 대칭 연산자를 측정하고 올바른 대칭 섹터에서 후선택

---

## 7. 곱 공식을 넘어서

### 7.1 유니터리의 선형 결합 (LCU)

$e^{-iHt}$를 선형 결합으로 표현: $e^{-iHt} \approx \sum_{j} c_j U_j$

게이트 복잡도: $O(\lambda t \cdot \text{polylog}(1/\epsilon))$

### 7.2 양자 신호 처리 (QSP)

QSP는 블록 인코딩된 해밀토니안의 다항식 변환을 구현하여 최적의 해밀토니안 시뮬레이션을 달성합니다. 시간 진화를 위한 복잡도: $O(\alpha t + \log(1/\epsilon))$

---

## 8. 응용

### 8.1 응축 물질 물리학

- **허바드 모델**: 고온 초전도와 관련된 강상관 전자 시스템 시뮬레이션
- **스핀 체인**: 양자 상전이, 다체 국소화 연구
- **위상적 상**: 위상적 질서와 에이니온 여기 검출

### 8.2 양자 화학

- **분자 바닥 상태**: 고전적 난해성을 넘어선 바닥 상태 에너지 계산
- **반응 동역학**: 실시간 화학 반응 시뮬레이션

### 8.3 재료 과학

- **배터리 재료**: 양자 수준의 전기화학 과정 시뮬레이션
- **촉매**: 효소 및 산업 촉매의 메커니즘 이해

---

## 9. Python 구현

### 9.1 Trotter-Suzuki 시뮬레이션

```python
import numpy as np
from scipy.linalg import expm

# 파울리 행렬
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(ops):
    """연산자 리스트의 텐서 곱을 계산합니다."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def build_ising_hamiltonian(n_qubits, J=1.0, h=0.5):
    """횡자기장 이징 모델 해밀토니안을 구축합니다.

    H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i

    ZZ 상호작용(정렬된 스핀 선호)과 횡자기장 X(중첩 선호) 사이의
    경쟁이 J = h에서 양자 상전이를 생성합니다.
    """
    N = 2 ** n_qubits
    H = np.zeros((N, N), dtype=complex)
    terms = []

    # ZZ 상호작용 항
    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = Z
        ops[i + 1] = Z
        term = -J * kron_list(ops)
        H += term
        terms.append((-J, kron_list(ops) / (-J)))

    # 횡자기장 항
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = X
        term = -h * kron_list(ops)
        H += term
        terms.append((-h, kron_list(ops) / (-h)))

    return H, terms


def trotter_step_second_order(terms, dt):
    """2차(대칭화) Trotter 단계를 계산합니다."""
    N = terms[0][1].shape[0]
    U = np.eye(N, dtype=complex)

    for coeff, op in terms:
        U = expm(-1j * coeff * op * dt / 2) @ U

    for coeff, op in reversed(terms):
        U = expm(-1j * coeff * op * dt / 2) @ U

    return U


def trotter_simulation(H, terms, t_total, n_steps, order=2):
    """Trotter-Suzuki 분해를 사용하여 시간 진화를 시뮬레이션합니다."""
    dt = t_total / n_steps

    if order == 1:
        N = terms[0][1].shape[0]
        U_step = np.eye(N, dtype=complex)
        for coeff, op in terms:
            U_step = expm(-1j * coeff * op * dt) @ U_step
    else:
        U_step = trotter_step_second_order(terms, dt)

    U_trotter = np.linalg.matrix_power(U_step, n_steps)
    U_exact = expm(-1j * H * t_total)
    error = np.linalg.norm(U_trotter - U_exact, ord=2)

    return U_trotter, U_exact, error


# 시연: 이징 모델의 Trotter 시뮬레이션
print("=" * 65)
print("횡자기장 이징 모델의 Trotter-Suzuki 시뮬레이션")
print("=" * 65)

n_qubits = 3
J, h = 1.0, 0.5
H, terms = build_ising_hamiltonian(n_qubits, J, h)

print(f"\n시스템: {n_qubits}-큐비트 이징 체인, J={J}, h={h}")
print(f"해밀토니안 차원: {2**n_qubits} x {2**n_qubits}")
print(f"항의 수: {len(terms)}")

t_total = 2.0
print(f"\n시뮬레이션 시간: t = {t_total}")
print(f"\n{'단계':>8} {'차수':>6} {'오차':>14}")
print("-" * 32)

for order in [1, 2]:
    for n_steps in [5, 10, 20, 50, 100]:
        _, _, error = trotter_simulation(H, terms, t_total, n_steps, order)
        print(f"{n_steps:8d} {order:6d} {error:14.2e}")
    print()
```

### 9.2 양자 위상 추정 시연

```python
import numpy as np
from scipy.linalg import expm

def qpe_energy_estimation(H, psi0, t, n_ancilla):
    """에너지 고유값 추출을 위한 양자 위상 추정을 시뮬레이션합니다."""
    N = H.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    overlaps = np.abs(eigenvectors.T @ psi0) ** 2

    n_phases = 2 ** n_ancilla
    phase_distribution = np.zeros(n_phases)

    for k in range(len(eigenvalues)):
        if overlaps[k] < 1e-10:
            continue
        E = eigenvalues[k]
        phi = (E * t) / (2 * np.pi)

        for j in range(n_phases):
            phase_diff = phi - j / n_phases
            if abs(phase_diff * n_phases) < 1e-10:
                prob = 1.0
            else:
                prob = abs(np.sin(np.pi * phase_diff * n_phases) /
                          (n_phases * np.sin(np.pi * phase_diff))) ** 2
            phase_distribution[j] += overlaps[k] * prob

    estimated_energies = 2 * np.pi * np.arange(n_phases) / (n_phases * t)

    return estimated_energies, phase_distribution, eigenvalues


print("=" * 65)
print("에너지 고유값을 위한 양자 위상 추정")
print("=" * 65)

n_qubits = 3
H, _ = build_ising_hamiltonian(n_qubits, J=1.0, h=0.5)
exact_E = np.sort(np.linalg.eigvalsh(H))
print(f"\n정확한 고유값: {exact_E}")

N = 2 ** n_qubits
psi0 = np.ones(N, dtype=complex) / np.sqrt(N)

for n_ancilla in [4, 6, 8]:
    t = 2 * np.pi / (max(abs(exact_E)) + 1)
    energies, probs, _ = qpe_energy_estimation(H, psi0, t, n_ancilla)

    peak_indices = np.where(probs > 0.01)[0]
    peak_energies = energies[peak_indices]
    peak_probs = probs[peak_indices]

    sort_idx = np.argsort(-peak_probs)
    print(f"\nQPE ({n_ancilla}개 보조 큐비트):")
    for e, p in zip(peak_energies[sort_idx][:4], peak_probs[sort_idx][:4]):
        nearest = exact_E[np.argmin(np.abs(exact_E - e))]
        print(f"  추정 E = {e:.4f}, 확률 = {p:.4f}, 가장 가까운 정확값 = {nearest:.4f}")
```

---

## 10. 연습 문제

### 연습 1: Trotter 오차 분석

4-큐비트 횡자기장 이징 모델($J = 1$, $h = 0.5$)에 대해:
(a) $t = 1$, $r = 1, 2, 5, 10, 20, 50$ 단계로 Trotter 오차를 계산하세요.
(b) 로그-로그 스케일로 오차 대 $r$을 그리세요. 1차의 $O(1/r)$ 스케일링과 2차의 $O(1/r^2)$을 확인하세요.
(c) 목표 오차 $10^{-3}$에서 2차 Trotter가 1차보다 효율적인 단계 수를 찾으세요.

### 연습 2: 관측량 동역학

5-큐비트 이징 모델을 $|11111\rangle$에서 시작하여 시뮬레이션:
(a) $t \in [0, 10]$에서 자화 $\langle M_z \rangle$를 추적하세요.
(b) 왼쪽 2 큐비트와 오른쪽 3 큐비트의 얽힘 엔트로피를 추적하세요.

### 연습 3: qDRIFT 대 Trotter

하이젠베르크 모델 $H = \sum_{i} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})$, $n = 4$ 큐비트에서 비교:
(a) 총 게이트 예산 100으로, 어떤 방법이 더 낮은 오차를 달성합니까?
(b) 50번의 다른 난수 시드로 qDRIFT를 실행하세요. 오차의 분산은 얼마입니까?

### 연습 4: QPE 에너지 분해능

3-큐비트 이징 모델에 대해:
(a) 모든 고유값이 별개의 위상에 매핑되도록 $t$를 선택하세요.
(b) 두 개의 가장 낮은 에너지 준위를 분해하는 데 필요한 최소 보조 큐비트 수를 결정하세요.

### 연습 5: ADAPT-VQE 수렴

4-큐비트 이징 모델의 임계점($J = h = 1$)에서:
(a) ADAPT-VQE를 실행하고 각 연산자 추가 후 에너지를 기록하세요.
(b) 같은 깊이의 고정 하드웨어 효율적 안자츠와 수렴 속도를 비교하세요.

---

[← 이전: 양자 컴퓨팅 현황과 미래](16_Landscape_and_Future.md) | [다음: 양자 걷기 →](19_Quantum_Walks.md)
