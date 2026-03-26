# 17. 어텐션과 트랜스포머의 수학

## 학습 목표

- 소프트맥스 함수의 수학적 성질과 미분 가능한 argmax로서의 해석을 이해할 수 있다
- 스케일드 닷-프로덕트 어텐션의 수학적 원리와 스케일링의 필요성을 설명할 수 있다
- 멀티헤드 어텐션의 부분공간 투영 해석과 파라미터 효율성을 이해할 수 있다
- 위치 인코딩(사인/코사인, RoPE, ALiBi)의 수학적 기초를 이해할 수 있다
- 어텐션의 계산 복잡도를 분석하고 효율적인 구현 방법을 이해할 수 있다
- 트랜스포머의 주요 응용(BERT, GPT, 크로스 어텐션)에서의 수학적 차이를 설명할 수 있다

---

## 1. 소프트맥스 함수의 수학

### 1.1 정의와 기본 성질

**소프트맥스 함수**:

$$\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^{n} \exp(z_j)}$$

**성질**:
- **확률 분포**: $\sum_i \text{softmax}(\mathbf{z})_i = 1$, 모든 원소 $\geq 0$
- **순서 보존**: $z_i > z_j \Rightarrow \text{softmax}(\mathbf{z})_i > \text{softmax}(\mathbf{z})_j$
- **평행 이동 불변성**: $\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})$

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    """
    소프트맥스 함수 (수치 안정성 포함)

    Parameters:
    -----------
    z : ndarray
        입력 벡터

    Returns:
    --------
    probs : ndarray
        확률 분포
    """
    # 수치 안정성: 최댓값 빼기
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    probs = exp_z / np.sum(exp_z)
    return probs

# 예제
z = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
probs = softmax(z)

print("입력 z:", z)
print("소프트맥스(z):", probs)
print("합:", np.sum(probs))
```

### 1.2 매끄러운 argmax로서의 해석

**Hard max**: $\text{argmax}_i z_i$ (미분 불가능)

**Soft max**: $\sum_i i \cdot \text{softmax}(\mathbf{z})_i$ (미분 가능한 기댓값)

큰 $z_i$에 높은 확률을 부여하지만, 다른 후보도 고려합니다.

```python
def visualize_softmax_vs_argmax():
    """소프트맥스와 argmax 비교"""
    z = np.linspace(-3, 3, 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1D 케이스: 두 개 원소
    for i, z1 in enumerate([-2, 0, 2]):
        z_vec = np.array([z1, 0])
        soft_probs = softmax(z_vec)

        axes[0].bar([i*3, i*3+1], soft_probs, width=0.8,
                    label=f'z=[{z1}, 0]')

    axes[0].set_title('Softmax probabilities')
    axes[0].set_ylabel('Probability')
    axes[0].legend()
    axes[0].grid(True, axis='y')

    # 온도 효과
    temperatures = [0.1, 1.0, 10.0]
    z_vec = np.array([1.0, 2.0, 3.0, 4.0])

    for tau in temperatures:
        soft_probs = softmax(z_vec / tau)
        axes[1].plot(soft_probs, 'o-', label=f'τ={tau}')

    axes[1].set_title('Temperature effect')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Probability')
    axes[1].legend()
    axes[1].grid(True)

    # 분포 변화
    max_val = 3.0
    z_other = np.linspace(-2, 2, 100)

    for temp in [0.5, 1.0, 2.0]:
        probs = []
        for z_o in z_other:
            z_vec = np.array([max_val, z_o])
            p = softmax(z_vec / temp)
            probs.append(p[0])  # 최댓값의 확률

        axes[2].plot(z_other, probs, label=f'τ={temp}')

    axes[2].set_title('Probability of max value')
    axes[2].set_xlabel('Other value')
    axes[2].set_ylabel('P(max)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('softmax_properties.png', dpi=150, bbox_inches='tight')
    print("소프트맥스 특성 시각화 저장 완료")

visualize_softmax_vs_argmax()
```

### 1.3 온도 매개변수 $\tau$

**온도 조절 소프트맥스**:

$$\text{softmax}(\mathbf{z}/\tau)_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$$

- **$\tau \to 0$**: 원-핫 분포 (hard max)
- **$\tau \to \infty$**: 균등 분포
- **$\tau = 1$**: 표준 소프트맥스

**응용**: 지식 증류(knowledge distillation), Gumbel-Softmax

### 1.4 야코비안 계산

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \begin{cases}
\text{softmax}(\mathbf{z})_i (1 - \text{softmax}(\mathbf{z})_i) & \text{if } i = j \\
-\text{softmax}(\mathbf{z})_i \cdot \text{softmax}(\mathbf{z})_j & \text{if } i \neq j
\end{cases}$$

행렬 형태: $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$, 여기서 $\mathbf{p} = \text{softmax}(\mathbf{z})$

```python
def softmax_jacobian(z):
    """소프트맥스의 야코비안"""
    p = softmax(z)
    n = len(p)
    J = np.diag(p) - np.outer(p, p)
    return J

z = np.array([1.0, 2.0, 3.0])
J = softmax_jacobian(z)
print("소프트맥스 야코비안:")
print(J)
print("\n행의 합 (0이어야 함):", np.sum(J, axis=1))
```

## 2. 스케일드 닷-프로덕트 어텐션

### 2.1 어텐션 메커니즘의 정의

**쿼리(Query)**, **키(Key)**, **값(Value)** 행렬:
- $Q \in \mathbb{R}^{n \times d_k}$
- $K \in \mathbb{R}^{m \times d_k}$
- $V \in \mathbb{R}^{m \times d_v}$

**스케일드 닷-프로덕트 어텐션**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**단계별 해석**:
1. **유사도 계산**: $S = QK^T \in \mathbb{R}^{n \times m}$
2. **스케일링**: $S' = S / \sqrt{d_k}$
3. **소프트맥스**: $A = \text{softmax}(S')$ (행 단위)
4. **가중 합**: $\text{Output} = A V$

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    스케일드 닷-프로덕트 어텐션

    Parameters:
    -----------
    Q : ndarray, shape (n, d_k)
        쿼리 행렬
    K : ndarray, shape (m, d_k)
        키 행렬
    V : ndarray, shape (m, d_v)
        값 행렬
    mask : ndarray, shape (n, m), optional
        어텐션 마스크 (0: 차단, 1: 허용)

    Returns:
    --------
    output : ndarray, shape (n, d_v)
        어텐션 출력
    attention_weights : ndarray, shape (n, m)
        어텐션 가중치
    """
    d_k = Q.shape[-1]

    # 유사도 점수
    scores = Q @ K.T / np.sqrt(d_k)

    # 마스킹 (옵션)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # 소프트맥스
    attention_weights = np.apply_along_axis(softmax, axis=1, arr=scores)

    # 가중 합
    output = attention_weights @ V

    return output, attention_weights

# 예제
np.random.seed(42)
n, m, d_k, d_v = 4, 5, 8, 8

Q = np.random.randn(n, d_k)
K = np.random.randn(m, d_k)
V = np.random.randn(m, d_v)

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print("쿼리 shape:", Q.shape)
print("키 shape:", K.shape)
print("값 shape:", V.shape)
print("\n어텐션 출력 shape:", output.shape)
print("어텐션 가중치 shape:", attn_weights.shape)
print("\n어텐션 가중치 (각 행의 합=1):")
print(attn_weights)
print("행의 합:", np.sum(attn_weights, axis=1))
```

### 2.2 스케일링의 필요성: 분산 분석

쿼리와 키의 각 차원이 독립적이고 $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$, $\text{Var}(q_i) = \text{Var}(k_i) = 1$이라면:

$$\text{Var}(q \cdot k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$

**문제**: $d_k$가 클 때, $q \cdot k$의 분산이 커져 소프트맥스가 극단값으로 포화될 수 있습니다.

**해결**: $\sqrt{d_k}$로 나누면 $\text{Var}(q \cdot k / \sqrt{d_k}) = 1$

```python
def demonstrate_scaling_effect():
    """스케일링 효과 시각화"""
    np.random.seed(42)
    d_k_values = [8, 32, 128, 512]
    n_samples = 1000

    fig, axes = plt.subplots(2, len(d_k_values), figsize=(16, 8))

    for idx, d_k in enumerate(d_k_values):
        # 랜덤 쿼리와 키 생성
        Q = np.random.randn(n_samples, d_k)
        K = np.random.randn(1, d_k)

        # 닷 프로덕트
        scores_unscaled = Q @ K.T
        scores_scaled = scores_unscaled / np.sqrt(d_k)

        # 히스토그램
        axes[0, idx].hist(scores_unscaled.flatten(), bins=50, alpha=0.7)
        axes[0, idx].set_title(f'd_k={d_k}, Unscaled')
        axes[0, idx].set_xlabel('Dot product value')
        axes[0, idx].axvline(0, color='r', linestyle='--')

        axes[1, idx].hist(scores_scaled.flatten(), bins=50, alpha=0.7)
        axes[1, idx].set_title(f'd_k={d_k}, Scaled')
        axes[1, idx].set_xlabel('Scaled value')
        axes[1, idx].axvline(0, color='r', linestyle='--')

        # 분산 출력
        var_unscaled = np.var(scores_unscaled)
        var_scaled = np.var(scores_scaled)
        print(f"d_k={d_k}: Var(unscaled)={var_unscaled:.2f}, Var(scaled)={var_scaled:.2f}")

    plt.tight_layout()
    plt.savefig('scaling_effect.png', dpi=150, bbox_inches='tight')
    print("\n스케일링 효과 시각화 저장 완료")

demonstrate_scaling_effect()
```

### 2.3 자기 어텐션 (Self-Attention)

**자기 어텐션**: $Q, K, V$가 모두 같은 입력에서 유래

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

여기서 $X \in \mathbb{R}^{n \times d_{\text{model}}}$은 입력 시퀀스입니다.

**효과**: 각 토큰이 다른 모든 토큰과 상호작용

## 3. 멀티헤드 어텐션 (Multi-Head Attention)

### 3.1 동기와 정의

**아이디어**: 여러 표현 부분공간(representation subspace)에서 병렬로 어텐션 수행

**단일 헤드**:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

여기서:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$

보통 $d_k = d_v = d_{\text{model}} / h$ (헤드 수 $h$)

**멀티헤드 어텐션**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

여기서 $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

```python
def multi_head_attention(Q, K, V, num_heads, W_Q, W_K, W_V, W_O, mask=None):
    """
    멀티헤드 어텐션

    Parameters:
    -----------
    Q, K, V : ndarray, shape (n, d_model)
        쿼리, 키, 값
    num_heads : int
        헤드 수
    W_Q, W_K, W_V : list of ndarray
        각 헤드의 투영 행렬
    W_O : ndarray, shape (h*d_v, d_model)
        출력 투영 행렬
    mask : ndarray, optional
        어텐션 마스크

    Returns:
    --------
    output : ndarray, shape (n, d_model)
        멀티헤드 어텐션 출력
    all_attention_weights : list
        각 헤드의 어텐션 가중치
    """
    head_outputs = []
    all_attention_weights = []

    for i in range(num_heads):
        # 각 헤드에 대한 투영
        Q_i = Q @ W_Q[i]
        K_i = K @ W_K[i]
        V_i = V @ W_V[i]

        # 스케일드 닷-프로덕트 어텐션
        head_out, attn_weights = scaled_dot_product_attention(Q_i, K_i, V_i, mask)

        head_outputs.append(head_out)
        all_attention_weights.append(attn_weights)

    # 헤드 연결
    concatenated = np.concatenate(head_outputs, axis=-1)

    # 출력 투영
    output = concatenated @ W_O

    return output, all_attention_weights

# 예제
d_model = 64
num_heads = 8
d_k = d_v = d_model // num_heads  # 8

n = 10
X = np.random.randn(n, d_model)

# 가중치 초기화
W_Q = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_K = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_V = [np.random.randn(d_model, d_v) * 0.1 for _ in range(num_heads)]
W_O = np.random.randn(num_heads * d_v, d_model) * 0.1

mha_output, attn_weights_list = multi_head_attention(
    X, X, X, num_heads, W_Q, W_K, W_V, W_O
)

print(f"입력 shape: {X.shape}")
print(f"멀티헤드 어텐션 출력 shape: {mha_output.shape}")
print(f"헤드 수: {num_heads}")
print(f"각 헤드의 어텐션 가중치 shape: {attn_weights_list[0].shape}")
```

### 3.2 부분공간 투영의 의미

각 헤드는 서로 다른 부분공간에서 작동:
- **구문 헤드** (syntactic head): 문법 구조 포착
- **의미 헤드** (semantic head): 의미 관계 포착
- **위치 헤드** (positional head): 상대 위치 학습

### 3.3 파라미터 수 분석

**단일 헤드 어텐션**:
- $W^Q, W^K, W^V$: $3 \times d_{\text{model}}^2$
- 합계: $3d_{\text{model}}^2$

**멀티헤드 어텐션** ($h$개 헤드, $d_k = d_v = d_{\text{model}}/h$):
- 각 헤드의 $W_i^Q, W_i^K, W_i^V$: $3h \times d_{\text{model}} \times \frac{d_{\text{model}}}{h} = 3d_{\text{model}}^2$
- $W^O$: $d_{\text{model}}^2$
- 합계: $4d_{\text{model}}^2$

**차이**: 멀티헤드는 단일 헤드와 비슷한 파라미터로 더 다양한 표현 학습

## 4. 위치 인코딩 (Positional Encoding)

### 4.1 문제: 위치 정보 부재

어텐션 메커니즘은 **순서 불변**(permutation-invariant)입니다:
- 입력 순서를 바꿔도 (마스크 제외) 출력은 같은 방식으로 바뀜
- 시퀀스의 위치 정보가 필요함

### 4.2 사인/코사인 위치 인코딩

**원래 Transformer 논문** (Vaswani et al., 2017):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

**주파수**: 차원 $i$마다 다른 주파수 $\omega_i = 1 / 10000^{2i/d_{\text{model}}}$

```python
def sinusoidal_positional_encoding(max_len, d_model):
    """
    사인/코사인 위치 인코딩

    Parameters:
    -----------
    max_len : int
        최대 시퀀스 길이
    d_model : int
        모델 차원

    Returns:
    --------
    PE : ndarray, shape (max_len, d_model)
        위치 인코딩 행렬
    """
    PE = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE

# 시각화
max_len = 100
d_model = 128

PE = sinusoidal_positional_encoding(max_len, d_model)

plt.figure(figsize=(12, 6))
plt.imshow(PE.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Value')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Sinusoidal Positional Encoding')
plt.tight_layout()
plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
print("위치 인코딩 시각화 저장 완료")
```

### 4.3 위치 인코딩의 수학적 특성

**상대 위치의 선형 변환**:

$$PE_{pos+k} = \mathbf{T}_k \cdot PE_{pos}$$

여기서 $\mathbf{T}_k$는 회전 행렬입니다 (각 주파수마다).

**증명**: 삼각함수 덧셈 공식

$$\sin(\omega(pos + k)) = \sin(\omega \cdot pos)\cos(\omega k) + \cos(\omega \cdot pos)\sin(\omega k)$$

이는 2D 회전 행렬로 표현 가능합니다.

### 4.4 RoPE (Rotary Position Embedding)

**아이디어**: 쿼리와 키에 위치 정보를 회전으로 인코딩

2차원 부분공간 $(q_{2i}, q_{2i+1})$을 각도 $m\theta_i$만큼 회전 ($m$은 위치):

$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

**장점**:
- 쿼리와 키의 내적이 상대 위치에만 의존
- 긴 시퀀스 외삽 성능 향상

```python
def rotary_position_embedding(q, k, positions, d_model):
    """
    RoPE (간단한 버전)

    Parameters:
    -----------
    q, k : ndarray, shape (n, d_model)
        쿼리와 키
    positions : ndarray, shape (n,)
        각 토큰의 위치
    d_model : int
        모델 차원

    Returns:
    --------
    q_rot, k_rot : ndarray
        회전된 쿼리와 키
    """
    assert d_model % 2 == 0

    # 주파수
    inv_freq = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))

    # 각도
    angles = positions[:, np.newaxis] * inv_freq[np.newaxis, :]  # (n, d_model/2)

    # 사인/코사인
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # 회전 적용
    def apply_rotation(x, cos, sin):
        x_rot = np.zeros_like(x)
        x_rot[:, 0::2] = x[:, 0::2] * cos - x[:, 1::2] * sin
        x_rot[:, 1::2] = x[:, 0::2] * sin + x[:, 1::2] * cos
        return x_rot

    q_rot = apply_rotation(q, cos_angles, sin_angles)
    k_rot = apply_rotation(k, cos_angles, sin_angles)

    return q_rot, k_rot

# 예제
n = 10
d_model = 64
positions = np.arange(n)

q = np.random.randn(n, d_model)
k = np.random.randn(n, d_model)

q_rot, k_rot = rotary_position_embedding(q, k, positions, d_model)

print("원본 쿼리 shape:", q.shape)
print("회전된 쿼리 shape:", q_rot.shape)

# 상대 위치 의존성 확인
attn_original = q @ k.T
attn_rope = q_rot @ k_rot.T

print("\n원본 어텐션 점수 (pos=0 기준):", attn_original[0, :5])
print("RoPE 어텐션 점수 (pos=0 기준):", attn_rope[0, :5])
```

### 4.5 ALiBi (Attention with Linear Biases)

**아이디어**: 어텐션 점수에 거리에 비례하는 바이어스 추가

$$\text{softmax}(q_i K^T / \sqrt{d_k} + m \cdot [0, -1, -2, \ldots, -(i-1)])$$

여기서 $m$은 헤드별 기울기입니다.

**장점**:
- 추가 파라미터 없음
- 외삽 성능 우수 (학습 길이보다 긴 시퀀스)

## 5. 계산 복잡도 분석

### 5.1 자기 어텐션의 복잡도

시퀀스 길이 $n$, 모델 차원 $d$:

**시간 복잡도**:
- $QK^T$: $O(n^2 d)$
- 소프트맥스: $O(n^2)$
- $(AV)$: $O(n^2 d)$
- **합계**: $O(n^2 d)$

**공간 복잡도**: $O(n^2)$ (어텐션 가중치 행렬)

**병목**: 긴 시퀀스 ($n$이 클 때)

### 5.2 Flash Attention

**아이디어**: 타일링(tiling)과 재계산으로 메모리 효율성 향상

**핵심**:
1. 소프트맥스를 온라인 방식으로 계산 (전체 $QK^T$ 저장 불필요)
2. GPU의 SRAM(빠른 메모리)에 타일을 로드하여 계산
3. 역전파 시 어텐션 가중치를 저장하지 않고 재계산

**효과**: 메모리 $O(n^2) \to O(n)$, 속도 향상 (I/O 감소)

### 5.3 선형 어텐션

**아이디어**: $\text{softmax}(QK^T)$를 **커널 근사**로 대체

$$\text{softmax}(q_i^T k_j) \approx \phi(q_i)^T \phi(k_j)$$

**선형 어텐션**:

$$\text{Attention}(Q, K, V) = \phi(Q) (\phi(K)^T V)$$

**복잡도**: 먼저 $\phi(K)^T V \in \mathbb{R}^{d_\phi \times d_v}$를 계산 ($O(nd_\phi d_v)$), 그 다음 $\phi(Q)$와 곱셈 ($O(nd_\phi d_v)$) → **$O(nd)$**

**예**: $\phi(x) = \text{elu}(x) + 1$ (Performer)

```python
def linear_attention(Q, K, V, phi=lambda x: np.maximum(0, x) + 1):
    """
    선형 어텐션 (간단한 근사)

    Parameters:
    -----------
    Q, K : ndarray, shape (n, d_k)
        쿼리와 키
    V : ndarray, shape (n, d_v)
        값
    phi : function
        특징 맵 함수

    Returns:
    --------
    output : ndarray, shape (n, d_v)
        어텐션 출력
    """
    # 특징 맵 적용
    Q_phi = phi(Q)
    K_phi = phi(K)

    # K^T V 먼저 계산 (n x d_v)
    KV = K_phi.T @ V  # (d_k, d_v)

    # Q와 곱셈
    output = Q_phi @ KV  # (n, d_v)

    # 정규화
    normalizer = Q_phi @ np.sum(K_phi, axis=0, keepdims=True).T
    output = output / (normalizer + 1e-6)

    return output

# 비교
n, d_k, d_v = 1000, 64, 64
Q = np.random.randn(n, d_k)
K = np.random.randn(n, d_k)
V = np.random.randn(n, d_v)

import time

# 표준 어텐션
start = time.time()
output_standard, _ = scaled_dot_product_attention(Q, K, V)
time_standard = time.time() - start

# 선형 어텐션
start = time.time()
output_linear = linear_attention(Q, K, V)
time_linear = time.time() - start

print(f"표준 어텐션 시간: {time_standard:.4f}s")
print(f"선형 어텐션 시간: {time_linear:.4f}s")
print(f"속도 향상: {time_standard / time_linear:.2f}x")
```

### 5.4 KV-캐시: 자기회귀 디코딩

**문제**: 자기회귀 생성 시 매 스텝 전체 시퀀스 재계산

**해결**: 이전 스텝의 키와 값을 캐시

$$K_{\text{new}} = [K_{\text{cached}}; k_{t+1}]$$
$$V_{\text{new}} = [V_{\text{cached}}; v_{t+1}]$$

**복잡도**: 스텝당 $O(td)$ ($t$는 현재 길이)

**메모리**: $O(t \cdot d \cdot \text{layers})$

## 6. ML 응용: BERT, GPT, 크로스 어텐션

### 6.1 BERT: 양방향 마스크

**마스크드 언어 모델** (MLM): 일부 토큰을 마스크하고 예측

**어텐션 마스크**: 모든 위치 간 어텐션 허용 (양방향)

$$\text{Mask}_{ij} = 1 \quad \forall i, j$$

### 6.2 GPT: 인과적 마스크

**자기회귀 언어 모델**: 이전 토큰만 보고 다음 토큰 예측

**어텐션 마스크**: 하삼각 행렬 (causal mask)

$$\text{Mask}_{ij} = \begin{cases}
1 & \text{if } j \leq i \\
0 & \text{if } j > i
\end{cases}$$

```python
def create_causal_mask(seq_len):
    """인과적 마스크 생성"""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask

seq_len = 5
causal_mask = create_causal_mask(seq_len)
print("인과적 마스크:")
print(causal_mask)

# 어텐션에 적용
Q = np.random.randn(seq_len, 8)
K = np.random.randn(seq_len, 8)
V = np.random.randn(seq_len, 8)

output_causal, attn_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print("\n인과적 어텐션 가중치:")
print(attn_causal)
```

### 6.3 크로스 어텐션 (Cross-Attention)

**인코더-디코더 아키텍처**: 디코더가 인코더의 출력에 어텐션

- **쿼리**: 디코더 상태
- **키/값**: 인코더 출력

$$\text{CrossAttention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})$$

**응용**: 번역, 요약, 이미지 캡셔닝

### 6.4 Mixture of Experts (MoE)의 라우팅 수학

**아이디어**: 여러 전문가 네트워크 중 일부만 활성화

**게이팅 네트워크**: 소프트맥스로 확률 계산

$$G(\mathbf{x}) = \text{softmax}(\mathbf{x}^T W_g)$$

**Top-K 라우팅**: 상위 $k$개 전문가만 선택

$$\text{Output} = \sum_{i \in \text{TopK}(G(\mathbf{x}))} G(\mathbf{x})_i \cdot E_i(\mathbf{x})$$

**수학적 도전**: 이산 선택의 미분 가능성 (Straight-Through Estimator 사용)

## 연습 문제

### 문제 1: 소프트맥스 야코비안 증명
소프트맥스 함수의 야코비안이 $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$임을 증명하시오. 이를 이용하여 $\sum_j J_{ij} = 0$임을 보이시오 (확률의 합 제약 조건).

### 문제 2: 어텐션 메커니즘 구현
NumPy만 사용하여 완전한 멀티헤드 어텐션 레이어를 구현하시오. 다음을 포함:
1. 선형 투영 ($W^Q, W^K, W^V, W^O$)
2. 스케일드 닷-프로덕트 어텐션
3. 인과적 마스크 지원
4. 그래디언트 체크 (수치 미분과 비교)

### 문제 3: 위치 인코딩 분석
사인/코사인 위치 인코딩에 대해:
1. 서로 다른 차원의 주파수가 어떻게 다른지 시각화
2. 두 위치 $pos_1, pos_2$의 내적 $PE_{pos_1}^T PE_{pos_2}$를 위치 차이의 함수로 플로팅
3. RoPE와의 비교: 상대 위치 정보를 얼마나 잘 보존하는지 정량화

### 문제 4: 계산 복잡도 실험
시퀀스 길이를 [100, 500, 1000, 2000, 5000]으로 변화시키며:
1. 표준 어텐션과 선형 어텐션의 실행 시간 측정
2. 메모리 사용량 추정 (어텐션 행렬 크기)
3. 두 방법의 출력 차이를 L2 norm으로 측정
4. 복잡도 그래프 플로팅 ($O(n^2)$ vs $O(n)$ 이론 곡선과 비교)

### 문제 5: 간단한 Transformer 블록
다음을 포함하는 Transformer 인코더 블록을 구현하시오:
1. 멀티헤드 자기 어텐션
2. Layer Normalization
3. 피드포워드 네트워크 (두 개의 선형 레이어 + ReLU)
4. 잔차 연결 (residual connection)

작은 시퀀스 분류 작업에 적용하고 학습시키시오.

## 참고 자료

### 논문
- Vaswani, A., et al. (2017). "Attention is All You Need." *NeurIPS*.
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*.
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*. [GPT-3]
- Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
- Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." *ICLR*. [ALiBi]
- Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*.

### 온라인 자료
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [Attention? Attention! (Lilian Weng)](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [Flash Attention Explained](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
- [Transformers from Scratch (Peter Bloem)](https://peterbloem.nl/blog/transformers)

### 라이브러리
- `torch.nn.MultiheadAttention`: PyTorch 구현
- `transformers` (Hugging Face): 사전 학습된 모델
- `einops`: 텐서 연산 단순화
