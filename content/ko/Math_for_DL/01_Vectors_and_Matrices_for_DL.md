# 레슨 1: 딥러닝을 위한 벡터와 행렬

## 학습 목표

- 일관된 인덱스 표기법을 사용하여 벡터, 행렬, 고차 텐서를 구별한다
- 아인슈타인 합산 규칙을 적용하여 행렬 연산을 간결하게 표현한다
- 배치 연산을 사용하여 행렬-벡터 및 행렬-행렬 곱을 계산한다
- 행렬 미적분의 레이아웃 규칙(분자 vs 분모 레이아웃)을 이해한다
- 벡터와 행렬에 대한 스칼라 함수의 미분을 수행한다
- 가중치 행렬, 편향 벡터, 활성화 텐서가 신경망 층을 통해 어떻게 흐르는지 인식한다
- 수학적 표기법과 NumPy/einsum 코드 사이의 변환을 수행한다

---

## 1. 텐서: 딥러닝의 언어

딥러닝은 **텐서** -- 다차원 숫자 배열 -- 위에서 작동합니다. 미적분에 들어가기 전에 이러한 객체에 대한 정확한 어휘가 필요합니다.

### 1.1 스칼라, 벡터, 행렬, 그 이상

| 객체 | 차수 | 표기법 | DL에서의 예 |
|------|------|--------|------------|
| 스칼라 | 0 | $x$, $\alpha$ | 학습률, 손실 값 |
| 벡터 | 1 | $\mathbf{x} \in \mathbb{R}^n$ | 특성 벡터, 편향 |
| 행렬 | 2 | $\mathbf{W} \in \mathbb{R}^{m \times n}$ | 가중치 행렬 |
| 3-텐서 | 3 | $\mathcal{X} \in \mathbb{R}^{B \times T \times d}$ | 시퀀스 배치 |

**$k$차 텐서**(또는 $k$-텐서, CS 의미에서 랭크-$k$ 텐서)는 $\mathbb{R}^{n_1 \times n_2 \times \cdots \times n_k}$의 원소입니다. 각 차원 $n_i$를 **축(axis)**이라 합니다.

```python
import numpy as np

scalar = 0.001          # shape: ()
vector = np.array([1.0, 2.0, 3.0])  # shape: (3,)
matrix = np.random.randn(4, 3)       # shape: (4, 3)
tensor3 = np.random.randn(8, 10, 64) # shape: (8, 10, 64) -- 시퀀스 배치
```

### 1.2 인덱스 표기법

인덱스 표기법은 텐서의 원소를 인덱스를 나열하여 표현합니다. 행렬 $\mathbf{A} \in \mathbb{R}^{m \times n}$의 $i$행 $j$열 원소는 $A_{ij}$ (또는 $a_{ij}$)입니다.

행렬-벡터 곱 $\mathbf{y} = \mathbf{A}\mathbf{x}$:

$$y_i = \sum_{j=1}^{n} A_{ij} x_j, \quad i = 1, \ldots, m$$

행렬-행렬 곱 $\mathbf{C} = \mathbf{A}\mathbf{B}$ ($\mathbf{A} \in \mathbb{R}^{m \times k}$, $\mathbf{B} \in \mathbb{R}^{k \times n}$):

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

### 1.3 아인슈타인 합산 규칙

**아인슈타인 규칙**은 $\sum$ 기호를 생략합니다: 곱 항에서 정확히 두 번 나타나는 인덱스는 암묵적으로 합산됩니다.

$$y_i = A_{ij} x_j \quad \text{(}j\text{에 대해 합산)}$$
$$C_{ij} = A_{ip} B_{pj} \quad \text{(}p\text{에 대해 합산)}$$

이는 NumPy의 `np.einsum`에 직접 대응됩니다:

```python
A = np.random.randn(4, 3)
x = np.random.randn(3)
B = np.random.randn(3, 5)

# 행렬-벡터 곱
y = np.einsum('ij,j->i', A, x)
assert np.allclose(y, A @ x)

# 행렬-행렬 곱
C = np.einsum('ip,pj->ij', A, B)
assert np.allclose(C, A @ B)

# 대각합 (trace): 대각 원소의 합
M = np.random.randn(4, 4)
tr = np.einsum('ii->', M)
assert np.isclose(tr, np.trace(M))

# 외적
u = np.array([1, 2, 3])
v = np.array([4, 5])
outer = np.einsum('i,j->ij', u, v)
assert np.allclose(outer, np.outer(u, v))
```

---

## 2. 배치 연산

딥러닝에서는 단일 샘플을 처리하는 경우가 드뭅니다. $B$개 입력 벡터 $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(B)} \in \mathbb{R}^n$의 **배치**는 행렬 $\mathbf{X} \in \mathbb{R}^{B \times n}$으로 쌓입니다.

### 2.1 배치 선형 변환

단일 선형 층은 하나의 샘플에 대해 $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$를 계산합니다. 배치의 경우:

$$\mathbf{Y} = \mathbf{X} \mathbf{W}^\top + \mathbf{1}_B \mathbf{b}^\top$$

여기서 $\mathbf{X} \in \mathbb{R}^{B \times n_{\text{in}}}$, $\mathbf{W} \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$, $\mathbf{b} \in \mathbb{R}^{n_{\text{out}}}$, $\mathbf{Y} \in \mathbb{R}^{B \times n_{\text{out}}}$.

실제로 NumPy 브로드캐스팅이 편향 덧셈을 처리합니다:

```python
B, n_in, n_out = 32, 784, 256
X = np.random.randn(B, n_in)
W = np.random.randn(n_out, n_in)
b = np.random.randn(n_out)

Y = X @ W.T + b  # 브로드캐스팅으로 b가 각 행에 더해짐
print(f"입력: {X.shape}, 출력: {Y.shape}")  # (32, 784), (32, 256)
```

### 2.2 einsum을 이용한 배치 행렬 곱셈

어텐션 메커니즘에서는 배치 행렬 곱셈이 필요합니다:

```python
# 쿼리와 키 행렬의 배치
B, T, d = 4, 10, 64  # 배치, 시퀀스 길이, 차원
Q = np.random.randn(B, T, d)
K = np.random.randn(B, T, d)

# 배치의 각 샘플에 대해 Q @ K^T 계산
scores = np.einsum('btd,bsd->bts', Q, K)
print(f"어텐션 스코어: {scores.shape}")  # (4, 10, 10)
```

---

## 3. 행렬 미적분 규칙

벡터와 행렬과 관련된 미분을 할 때 **레이아웃 규칙**을 선택해야 합니다. 두 가지 경쟁하는 표준이 있으며, 이들 사이의 혼동이 끝없는 버그를 일으킵니다.

### 3.1 분자 레이아웃 vs 분모 레이아웃

$\mathbf{y} \in \mathbb{R}^m$이 $\mathbf{x} \in \mathbb{R}^n$의 함수인 경우를 생각합시다.

**분자 레이아웃** (야코비안 규칙):

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}, \quad \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)_{ij} = \frac{\partial y_i}{\partial x_j}$$

**분모 레이아웃** (그래디언트 규칙):

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{n \times m}, \quad \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)_{ij} = \frac{\partial y_j}{\partial x_i}$$

> **이 과정의 규칙**: 딥러닝과 *Matrix Cookbook*에서 표준인 **분자 레이아웃** (야코비안 규칙)을 사용합니다. $\mathbf{y}$의 $\mathbf{x}$에 대한 야코비안은 $m \times n$ 크기입니다 -- 행은 출력, 열은 입력으로 인덱싱됩니다.

### 3.2 스칼라의 그래디언트

출력이 스칼라 $L \in \mathbb{R}$ (예: 손실 함수)인 경우, 벡터 $\mathbf{x} \in \mathbb{R}^n$에 대한 미분은 **그래디언트**입니다:

$$\nabla_{\mathbf{x}} L = \frac{\partial L}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial L}{\partial x_1} \\ \frac{\partial L}{\partial x_2} \\ \vdots \\ \frac{\partial L}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n$$

### 3.3 행렬에 대한 스칼라의 미분

손실 $L$이 가중치 행렬 $\mathbf{W} \in \mathbb{R}^{m \times n}$에 의존하는 경우:

$$\frac{\partial L}{\partial \mathbf{W}} \in \mathbb{R}^{m \times n}, \quad \left(\frac{\partial L}{\partial \mathbf{W}}\right)_{ij} = \frac{\partial L}{\partial W_{ij}}$$

그래디언트는 매개변수와 **같은 크기**입니다. 이것이 SGD 업데이트 $\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial L}{\partial \mathbf{W}}$를 차원적으로 일관되게 만드는 기본 규칙입니다.

---

## 4. 핵심 행렬 미적분 항등식

딥러닝에서 반복적으로 사용할 행렬 미적분 항등식입니다.

### 4.1 벡터 대 벡터 미분

$\mathbf{x}, \mathbf{a} \in \mathbb{R}^n$이고 $\mathbf{A} \in \mathbb{R}^{m \times n}$일 때.

| 함수 | $\mathbf{x}$에 대한 미분 | 크기 |
|------|------------------------|------|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}^\top$ | $1 \times n$ |
| $\mathbf{x}^\top \mathbf{A} \mathbf{x}$ | $\mathbf{x}^\top (\mathbf{A} + \mathbf{A}^\top)$ | $1 \times n$ |
| $\|\mathbf{x}\|^2 = \mathbf{x}^\top \mathbf{x}$ | $2\mathbf{x}^\top$ | $1 \times n$ |
| $\mathbf{A}\mathbf{x}$ | $\mathbf{A}$ (야코비안) | $m \times n$ |

### 4.2 유도 예제: 선형 층 그래디언트

한 샘플이 선형 층을 통과하는 경우:

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

하류 스칼라 손실 $L$과 그래디언트 $\frac{\partial L}{\partial \mathbf{y}} \in \mathbb{R}^m$이 주어졌을 때, $\frac{\partial L}{\partial \mathbf{W}}$와 $\frac{\partial L}{\partial \mathbf{x}}$를 구합니다.

연쇄 법칙에 의해:

$$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_i} x_j$$

행렬 형태로:

$$\boxed{\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{y}} \mathbf{x}^\top} \in \mathbb{R}^{m \times n}$$

입력 그래디언트:

$$\boxed{\frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^\top \frac{\partial L}{\partial \mathbf{y}}} \in \mathbb{R}^n$$

```python
# 수치적 검증
n_in, n_out = 3, 2
W = np.random.randn(n_out, n_in)
x = np.random.randn(n_in)
b = np.random.randn(n_out)

# 순전파
y = W @ x + b

# dL/dy가 하류에서 주어졌다고 가정
dL_dy = np.random.randn(n_out)

# 해석적 그래디언트
dL_dW = np.outer(dL_dy, x)     # (n_out, n_in)
dL_dx = W.T @ dL_dy            # (n_in,)
dL_db = dL_dy                  # (n_out,)

# 유한 차분을 통한 수치적 검증
eps = 1e-5
dL_dW_num = np.zeros_like(W)
for i in range(n_out):
    for j in range(n_in):
        W_plus = W.copy(); W_plus[i, j] += eps
        W_minus = W.copy(); W_minus[i, j] -= eps
        y_plus = W_plus @ x + b
        y_minus = W_minus @ x + b
        dL_dW_num[i, j] = dL_dy @ (y_plus - y_minus) / (2 * eps)

print(f"dL/dW 해석적:\n{dL_dW}")
print(f"dL/dW 수치적:\n{dL_dW_num}")
print(f"일치: {np.allclose(dL_dW, dL_dW_num, atol=1e-4)}")
```

---

## 5. 딥러닝에서의 특수 행렬

### 5.1 대각 행렬

대각 행렬 $\mathbf{D} = \text{diag}(d_1, \ldots, d_n)$은 $D_{ij} = d_i \delta_{ij}$. 곱셈 $\mathbf{D}\mathbf{x}$는 각 원소를 스케일링합니다.

**DL 용도**: 배치 정규화에서의 원소별 스케일링, 헤시안의 대각 근사 (Adam 옵티마이저).

### 5.2 직교 행렬

$\mathbf{Q} \in \mathbb{R}^{n \times n}$이 $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$이면 직교합니다. 핵심 성질: 직교 행렬은 노름을 보존합니다, $\|\mathbf{Q}\mathbf{x}\| = \|\mathbf{x}\|$.

**DL 용도**: 직교 가중치 초기화는 그래디언트 소실/폭발을 방지합니다.

### 5.3 대칭 양정부호 행렬

대칭 행렬은 $\mathbf{A} = \mathbf{A}^\top$. 모든 $\mathbf{x} \neq \mathbf{0}$에 대해 $\mathbf{x}^\top \mathbf{A} \mathbf{x} > 0$이면 **양정부호** (PD)입니다.

**DL 용도**: 손실의 헤시안은 대칭입니다. 양정부호 헤시안은 국소 볼록성(극소점)을 나타냅니다.

### 5.4 저랭크 행렬

랭크 $r \ll \min(m, n)$인 행렬 $\mathbf{W}$는 $\mathbf{W} = \mathbf{U}\mathbf{V}^\top$로 분해됩니다.

**DL 용도**: LoRA (Low-Rank Adaptation)는 저랭크 업데이트 $\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$ ($r \ll d$)를 학습하여 대규모 언어 모델을 미세 조정합니다.

---

## 6. 텐서 재형성과 전치

재형성은 메모리 레이아웃을 재해석하는 비용 없는 연산입니다. 재형성의 이해는 다중 헤드 어텐션, 합성곱 등의 DL 연산 구현에 필수적입니다.

```python
# 이미지 배치: (배치, 채널, 높이, 너비)
images = np.random.randn(8, 3, 32, 32)

# 완전연결 층을 위해 공간 차원을 평탄화
flat = images.reshape(8, -1)  # (8, 3072)

# 다중 헤드 어텐션 재형성
B, T, d_model, n_heads = 4, 10, 512, 8
d_head = d_model // n_heads  # 64

x = np.random.randn(B, T, d_model)
x_heads = x.reshape(B, T, n_heads, d_head)  # (4, 10, 8, 64)
x_heads = x_heads.transpose(0, 2, 1, 3)     # (4, 8, 10, 64) -- 헤드가 축 1이 됨
```

---

## 7. 딥러닝에서의 노름

### 7.1 벡터 노름

$\mathbf{x} \in \mathbb{R}^n$의 $L^p$ 노름:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}$$

| 노름 | 공식 | DL 용도 |
|------|------|---------|
| $L^1$ | $\sum \|x_i\|$ | 희소성 정규화 (Lasso) |
| $L^2$ | $\sqrt{\sum x_i^2}$ | 가중치 감쇠, 그래디언트 클리핑 |
| $L^\infty$ | $\max \|x_i\|$ | 적대적 견고성 |

### 7.2 행렬 노름

**프로베니우스 노름**은 행렬을 긴 벡터로 취급합니다:

$$\|\mathbf{W}\|_F = \sqrt{\sum_{i,j} W_{ij}^2} = \sqrt{\text{tr}(\mathbf{W}^\top \mathbf{W})}$$

**스펙트럼 노름**은 최대 특이값입니다:

$$\|\mathbf{W}\|_2 = \sigma_{\max}(\mathbf{W})$$

**DL 용도**: 스펙트럼 정규화는 GAN에서 판별기 네트워크의 립시츠 상수를 제한합니다.

---

## 8. DL 맥락에서 보기: 완전한 순전파

ReLU 활성화를 가진 2층 네트워크의 수학을 추적해봅시다:

$$\mathbf{h} = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\hat{y} = \mathbf{w}_2^\top \mathbf{h} + b_2$$

```python
# 네트워크 차원
n_in, n_hidden, n_out = 4, 8, 1

# 가중치 초기화
W1 = np.random.randn(n_hidden, n_in) * 0.01
b1 = np.zeros(n_hidden)
w2 = np.random.randn(n_hidden) * 0.01
b2 = 0.0

# 입력
x = np.random.randn(n_in)

# 명시적 수학으로 순전파
z1 = W1 @ x + b1         # 사전 활성화
h = np.maximum(z1, 0)     # ReLU
y_hat = w2 @ h + b2       # 출력

print(f"입력 x: {x.shape}")
print(f"은닉 z1: {z1.shape}")
print(f"은닉 h (ReLU 후): {h.shape}")
print(f"출력 y_hat: {y_hat:.4f}")
```

---

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 텐서 표기법 | 인덱스가 축을 표시; 아인슈타인 규칙은 반복 인덱스의 $\sum$을 생략 |
| 배치 연산 | 샘플을 배치 차원으로 쌓기; 배치 행렬곱에 `@` 또는 `einsum` 사용 |
| 레이아웃 규칙 | 분자 레이아웃: 그래디언트 크기가 매개변수 크기와 일치 |
| 행렬 미적분 | 선형 층에서 $\partial L / \partial \mathbf{W} = (\partial L / \partial \mathbf{y})\mathbf{x}^\top$ |
| 특수 행렬 | 대각, 직교, 양정부호, 저랭크 -- 각각 DL 응용이 있음 |
| 노름 | 프로베니우스는 정규화에, 스펙트럼은 립시츠 제약에 사용 |

---

## 연습문제

1. 곱 행렬을 형성하지 않고 `np.einsum`을 사용하여 $\mathbf{A}\mathbf{B}$의 대각합을 계산하세요.
2. 선형 층 $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$에 대한 $\frac{\partial L}{\partial \mathbf{b}}$를 유도하세요.
3. 임의의 직교 행렬 $\mathbf{Q}$에 대해 $\|\mathbf{Q}\mathbf{x}\|_2 = \|\mathbf{x}\|_2$를 수치적으로 검증하세요.
4. `einsum`을 사용하여 배치 선형 층 순전파를 구현하세요.
5. 그래디언트 $\frac{\partial L}{\partial \mathbf{W}}$의 프로베니우스 노름을 계산하고, 이 양이 학습 모니터링에 유용한 경우를 설명하세요.

---

**다음**: [02. 편미분과 그래디언트](02_Partial_Derivatives_and_Gradients.md)
