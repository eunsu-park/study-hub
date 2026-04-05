# 레슨 25: 캡스톤 양자 응용

[← 이전: Qiskit 심층 분석](24_Qiskit_Deep_Dive.md) | [개요로 돌아가기](00_Overview.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 문제 정의부터 결과 분석까지 종단간 양자 컴퓨팅 프로젝트 설계
2. 실세계 최적화 문제를 양자 알고리즘(QAOA 또는 VQE)으로 매핑
3. 잡음이 있는 완전한 양자 회로를 구성, 트랜스파일, 시뮬레이션
4. 오류 완화 기법을 적용하여 잡음 시뮬레이션 결과 개선
5. 고전적 기준선과 양자 솔루션 벤치마크
6. 양자 접근법이 진정한 우위를 제공하는 시점을 비판적으로 평가
7. 적절한 주의사항과 함께 양자 컴퓨팅 결과를 문서화하고 발표

---

이 캡스톤 레슨은 앞선 24개 레슨의 모든 것을 하나의 완전한 프로젝트로 통합합니다. **분자 바닥 상태 에너지** 문제(VQE)와 **최대 절단** 문제(QAOA)에 대한 양자 알고리즘을 설계, 구현, 시뮬레이션, 분석합니다.

목표는 단순히 알고리즘을 실행하는 것이 아니라 전체 엔지니어링 워크플로우에 참여하는 것입니다: 올바른 문제 인코딩 선택, 효율적인 안자츠 설계, 잡음 처리, 오류 완화 적용, 고전적 대안과의 솔직한 결과 평가.

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [프로젝트 A: VQE를 통한 분자 바닥 상태](#2-프로젝트-a-vqe를-통한-분자-바닥-상태)
3. [프로젝트 B: QAOA를 통한 최대 절단](#3-프로젝트-b-qaoa를-통한-최대-절단)
4. [공통 파이프라인: 잡음과 완화](#4-공통-파이프라인-잡음과-완화)
5. [고전적 기준선](#5-고전적-기준선)
6. [분석과 평가](#6-분석과-평가)
7. [Python 구현: VQE 캡스톤](#7-python-구현-vqe-캡스톤)
8. [Python 구현: QAOA 캡스톤](#8-python-구현-qaoa-캡스톤)
9. [결과 논의](#9-결과-논의)
10. [연습 문제](#10-연습-문제)

---

## 1. 프로젝트 개요

### 1.1 종단간 워크플로우

```
1. 문제 정의 → 무엇을 계산하는가? 정답은 무엇인가?
2. 양자 인코딩 → 문제를 해밀토니안/회로로 매핑
3. 안자츠 설계 → 매개변수화 회로 선택
4. 이상적 시뮬레이션 → 잡음 없이 알고리즘 동작 확인
5. 잡음 시뮬레이션 → 현실적 잡음 모델 추가, 성능 저하 관찰
6. 오류 완화 → ZNE/판독 교정 적용, 정확도 회복
7. 고전적 비교 → 정확 및 근사 고전 방법과 비교
8. 분석 → 정확도, 자원 비용, 스케일링 평가
```

---

## 2. 프로젝트 A: VQE를 통한 분자 바닥 상태

**목표**: STO-3G 기저 세트에서 결합 길이 $R = 0.74$ A의 H$_2$ 분자 바닥 상태 에너지 계산.

**기대 답**: $E_0 \approx -1.137$ Ha
**화학적 정확도 임계값**: 오차 $< 1.6$ mHa

---

## 3. 프로젝트 B: QAOA를 통한 최대 절단

**목표**: $|V| = 6$인 무작위 그래프의 근사 최대 절단 찾기.

QAOA 깊이 $p$가 증가하면 근사비가 개선됩니다.

---

## 4. 공통 파이프라인: 잡음과 완화

판독 교정, 영잡음 외삽법(ZNE), 대칭성 검증을 포함한 오류 완화 전략.

---

## 5. 고전적 기준선

VQE: Hartree-Fock, Full CI, CCSD(T)
QAOA: 전수 탐색, Goemans-Williamson SDP

---

## 6. 분석과 평가

**솔직한 평가**: 이 캡스톤의 문제 크기에서 양자 컴퓨터는 우위를 제공하지 않습니다. 이 연습의 가치는 알고리즘을 검증하고 양자 자원이 충분해질 때를 위한 직관을 구축하는 데 있습니다.

---

## 7. Python 구현: VQE 캡스톤

```python
import numpy as np
from scipy.optimize import minimize
from functools import reduce

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(ops):
    return reduce(np.kron, ops)

def build_h2_hamiltonian():
    """H2/STO-3G 큐비트 해밀토니안."""
    g = {'I': -0.8105, 'Z0': 0.1721, 'Z1': 0.1721, 'Z2': -0.2232,
         'Z3': -0.2232, 'Z0Z1': 0.1686, 'Z0Z2': 0.1205, 'Z0Z3': 0.1659,
         'Z1Z2': 0.1659, 'Z1Z3': 0.1205, 'Z2Z3': 0.1743, 'XXYY': -0.0453}

    H = g['I'] * kron_list([I2]*4)
    H += g['Z0'] * kron_list([Z,I2,I2,I2]) + g['Z1'] * kron_list([I2,Z,I2,I2])
    H += g['Z2'] * kron_list([I2,I2,Z,I2]) + g['Z3'] * kron_list([I2,I2,I2,Z])
    H += g['Z0Z1'] * kron_list([Z,Z,I2,I2]) + g['Z0Z2'] * kron_list([Z,I2,Z,I2])
    H += g['Z0Z3'] * kron_list([Z,I2,I2,Z]) + g['Z1Z2'] * kron_list([I2,Z,Z,I2])
    H += g['Z1Z3'] * kron_list([I2,Z,I2,Z]) + g['Z2Z3'] * kron_list([I2,I2,Z,Z])
    c = g['XXYY']
    H += c * (kron_list([X,X,Y,Y]) - kron_list([X,Y,Y,X])
            + kron_list([Y,X,X,Y]) - kron_list([Y,Y,X,X]))
    return H

H = build_h2_hamiltonian()
exact_energy = np.min(np.linalg.eigvalsh(H))

print("=" * 60)
print("캡스톤 프로젝트 A: H2 바닥 상태 에너지를 위한 VQE")
print("=" * 60)
print(f"\n정확한 바닥 상태 에너지: {exact_energy:.6f} Ha")

# 간단한 VQE 실행
N = 16
def vqe_cost(params):
    state = np.zeros(N, dtype=complex)
    state[0b1100] = 1.0
    # 이중 여기 매개변수
    theta = params[0]
    a, b = state[12], state[3]
    state[12] = np.cos(theta) * a - np.sin(theta) * b
    state[3] = np.sin(theta) * a + np.cos(theta) * b
    return np.real(state.conj() @ H @ state)

result = minimize(vqe_cost, [0.0], method='COBYLA')
print(f"VQE 에너지: {result.fun:.6f} Ha")
print(f"오차: {abs(result.fun - exact_energy)*1000:.2f} mHa")
```

---

## 8. Python 구현: QAOA 캡스톤

```python
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

def maxcut_qaoa(adjacency, p=2):
    """최대 절단 문제에 대한 QAOA."""
    n = adjacency.shape[0]
    N = 2 ** n

    # 비용 해밀토니안
    H_C = np.zeros((N, N), dtype=complex)
    for i in range(n):
        for j in range(i+1, n):
            if adjacency[i,j]:
                ops = [I2]*n; ops[i] = Z; ops[j] = Z
                H_C += (np.eye(N) - kron_list(ops)) / 2

    # 혼합 해밀토니안
    H_M = np.zeros((N, N), dtype=complex)
    for i in range(n):
        ops = [I2]*n; ops[i] = X
        H_M += kron_list(ops)

    def cost(params):
        gamma, beta = params[:p], params[p:]
        state = np.ones(N, dtype=complex) / np.sqrt(N)
        for l in range(p):
            state = expm(-1j * gamma[l] * H_C) @ state
            state = expm(-1j * beta[l] * H_M) @ state
        return -np.real(state.conj() @ H_C @ state)

    best = float('inf')
    for _ in range(10):
        r = minimize(cost, np.random.uniform(0, np.pi, 2*p), method='COBYLA')
        best = min(best, r.fun)
    return -best

np.random.seed(42)
n = 6
adj = np.zeros((n,n), dtype=int)
for i in range(n):
    for j in range(i+1, n):
        if np.random.random() < 0.5:
            adj[i,j] = adj[j,i] = 1

print("\n" + "=" * 60)
print("캡스톤 프로젝트 B: 최대 절단을 위한 QAOA")
print("=" * 60)

for p in [1, 2, 3]:
    val = maxcut_qaoa(adj, p)
    print(f"  QAOA p={p}: <C> = {val:.4f}")
```

---

## 9. 결과 논의

이 캡스톤에서의 문제 크기에서 고전 방법이 양자 방법보다 훨씬 빠릅니다. 양자 우위는 VQE의 경우 ~50 큐비트 이상, QAOA의 경우 ~100 꼭짓점 이상에서 기대됩니다.

---

## 10. 연습 문제

### 연습 1: 확장된 VQE 프로젝트
LiH (12 큐비트)까지 VQE를 확장하세요.

### 연습 2: QAOA 스케일링
$n = 4, 6, 8, 10, 12$에 대한 QAOA 스케일링을 연구하세요.

### 연습 3: 완전한 잡음 파이프라인
열이완, 상관 오류를 포함한 포괄적 잡음 시뮬레이션을 구축하세요.

### 연습 4: 오류 완화 비교
판독 교정, ZNE, 대칭성 검증을 비교하세요.

### 연습 5: 나만의 캡스톤
자신만의 양자 컴퓨팅 캡스톤 프로젝트를 설계하고 구현하세요.

---

[← 이전: Qiskit 심층 분석](24_Qiskit_Deep_Dive.md) | [개요로 돌아가기](00_Overview.md)
